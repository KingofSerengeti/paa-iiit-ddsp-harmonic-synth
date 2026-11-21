"""
fused_precision.py - number-format variants of the fused harmonic kernel (same algorithm as fused_ddsp.py, real CUDA code,
so each variant has both a speed and an accuracy). Only the arithmetic of the phase / Nyquist setup and of the harmonic
phase changes; amplitudes, the sine and the accumulation are fp32 in every mode.

mode  name       per-sample phase setup                      Nyquist cutoff k_max              harmonic phase k*x
0     f64        fp64                                        fp64, reference formula          fp32  frac(k*x)
1     f32        fp32                                        fp32                             fp32  frac(k*x)
2     f32tb      fp32                                        fp32, fp64 only near a tie       fp32  frac(k*x)
3     df         double-float (pair of fp32, ~48-bit)        fp32, fp64 only near a tie       fp32  frac(k*x)
4     f64_u32    fp64                                        fp64, reference formula          uint32 fixed point, P_k = P_{k-1} + X
5     i40_u32    integer, 40 fractional bits (int64)         fp32, fp64 only near a tie       uint32 fixed point, P_k = P_{k-1} + X
6     f64_u64    fp64                                        fp64, reference formula          uint64 fixed point, P_k = P_{k-1} + X

Fixed-point phase: a phase in cycles is stored as an unsigned integer scaled by 2^32 (or 2^64). Integer overflow is exactly
"mod 1 cycle", so no floor() is needed and adding X to itself k times gives k*x mod 1 with no rounding at all.
"near a tie": k_max is first found in fp32 from r = nyq / f(t); only when r is within 1e-3 of an integer (a harmonic within
about 0.001 harmonic numbers of Nyquist) is k_max recomputed in fp64 with the reference formula.
Mode 0 is identical to fused_ddsp.fused_harmonic_synthesis (default flags).
"""
import math, os
import torch
from torch.utils.cpp_extension import load_inline
import fused_ddsp                                   # sets TORCH_CUDA_ARCH_LIST the same way
from fused_ddsp import frame_phase_offsets

MODES = {"f64": 0, "f32": 1, "f32tb": 2, "df": 3, "f64_u32": 4, "i40_u32": 5, "f64_u64": 6}

_CPP = ("torch::Tensor fused_precision(torch::Tensor f0, torch::Tensor amp, torch::Tensor dist, torch::Tensor phi, "
        "torch::Tensor auxf, torch::Tensor auxi, int64_t T, double sr, int64_t mode, bool fast_sin);")

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// ---- Nyquist cutoff -------------------------------------------------------------------------------------------
// Reference rule: harmonic k is silent iff (k*a)*(1-fr) + (k*a1)*fr >= sr/2, fr = t*(F/T) - floor(.)
template <typename R>
__device__ __forceinline__ int kmax_exact(R a, R a1, int K, int T, int F, int t, R nyq)
{
    const R src = (R)t * ((R)F / (R)T);
    const R fr  = src - floor(src);
    const R fd  = a * ((R)1 - fr) + a1 * fr;
    int kmax = K;
    if (fd > (R)0) {
        kmax = (int)min((R)K, floor(nyq / fd) + (R)1);
        while (kmax > 0 && ((R)kmax * a) * ((R)1 - fr) + ((R)kmax * a1) * fr >= nyq) --kmax;
        while (kmax < K && ((R)(kmax + 1) * a) * ((R)1 - fr) + ((R)(kmax + 1) * a1) * fr < nyq) ++kmax;
    }
    return kmax;
}

// fp32 cutoff; falls back to the fp64 rule only when a harmonic is within 1e-3 harmonic numbers of Nyquist.
__device__ __forceinline__ int kmax_tiebreak(float a, float a1, int K, int T, int F, int t, double nyq)
{
    const float src = (float)t * ((float)F / (float)T);
    const float fr  = src - floorf(src);
    const float fd  = a * (1.f - fr) + a1 * fr;
    if (!(fd > 0.f)) return K;
    const float r = (float)nyq / fd;                    // harmonic k is active iff k < r
    if (fabsf(r - rintf(r)) < 1e-3f) return kmax_exact<double>((double)a, (double)a1, K, T, F, t, nyq);
    const float km = ceilf(r) - 1.f;
    return (int)fminf(fmaxf(km, 0.f), (float)K);
}

// ---- double-float helpers (value = hi + lo, both fp32) -----------------------------------------------------------
struct df { float hi, lo; };
__device__ __forceinline__ df quick_two_sum(float a, float b) { float s = a + b; return {s, b - (s - a)}; }
__device__ __forceinline__ df two_sum(float a, float b) {
    float s = a + b; float bb = s - a; return {s, (a - (s - bb)) + (b - bb)};
}
__device__ __forceinline__ df two_prod(float a, float b) { float p = a * b; return {p, fmaf(a, b, -p)}; }
__device__ __forceinline__ df df_add(df x, df y) { df s = two_sum(x.hi, y.hi); return quick_two_sum(s.hi, s.lo + x.lo + y.lo); }
__device__ __forceinline__ df df_mul(df x, df y) {
    df p = two_prod(x.hi, y.hi); return quick_two_sum(p.hi, p.lo + (x.hi * y.lo + x.lo * y.hi));
}

template <int MODE, bool FAST>
__global__ void precision_kernel(const float* __restrict__ f0, const float* __restrict__ amp, const float* __restrict__ dist,
                                 const double* __restrict__ phi,           // [B,F] rad, mod 2pi (modes 0,1,2,4,6)
                                 const float* __restrict__ auxf,           // [B,F,2] phase offset in cycles, hi/lo (mode 3)
                                 const long long* __restrict__ auxi,       // [B,F,3] PHI40, A40, D40 (mode 5)
                                 float* __restrict__ out, int F, int K, int T, int hop, double nyq, double c,
                                 float isr_hi, float isr_lo, float ihop_hi, float ihop_lo)
{
    const int b = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const int j  = t / hop;
    const int n  = t - j * hop;
    const int j1 = min(j + 1, F - 1);
    const size_t fj = (size_t)b * F + j, fj1 = (size_t)b * F + j1;
    const float af = f0[fj], a1f = f0[fj1];

    float xf = 0.f; double xd = 0.0; uint32_t X32 = 0; int kmax = K;
    if (MODE == 0 || MODE == 4 || MODE == 6) {                       // fp64 setup
        const double a = af, a1 = a1f, d = a1 - a;
        double x = (phi[fj] + c * ((double)(n + 1) * a + d * (double)n * (double)(n + 1) / (2.0 * hop)))
                   * 0.15915494309189533576888376337251;
        x -= floor(x); xd = x; xf = (float)x;
        kmax = kmax_exact<double>(a, a1, K, T, F, t, nyq);
    } else if (MODE == 1) {                                          // fp32 setup (fused_ddsp fp32_setup ablation)
        const float d = a1f - af;
        float x = ((float)phi[fj] + (float)c * ((float)(n + 1) * af + d * (float)n * (float)(n + 1) / (2.f * (float)hop)))
                  * 0.15915494309189533576888376337251f;
        xf = x - floorf(x);
        kmax = kmax_exact<float>(af, a1f, K, T, F, t, (float)nyq);
    } else if (MODE == 2) {                                          // fp32 setup, fp64 only near a Nyquist tie
        const float d = a1f - af;
        float x = ((float)phi[fj] + (float)c * ((float)(n + 1) * af + d * (float)n * (float)(n + 1) / (2.f * (float)hop)))
                  * 0.15915494309189533576888376337251f;
        xf = x - floorf(x);
        kmax = kmax_tiebreak(af, a1f, K, T, F, t, nyq);
    } else if (MODE == 3) {                                          // double-float phase, in cycles
        const df d  = two_sum(a1f, -af);
        const float m = 0.5f * (float)n * (float)(n + 1);            // exact integer <= 2^24
        const df u  = df_mul(df_mul(d, df{m, 0.f}), df{ihop_hi, ihop_lo});
        const df v  = two_prod(af, (float)(n + 1));
        const df y  = df_mul(df_add(v, u), df{isr_hi, isr_lo});
        df x = df_add(df{auxf[2 * fj], auxf[2 * fj + 1]}, y);
        const float fl = floorf(x.hi);
        x = quick_two_sum(x.hi - fl, x.lo);
        float r = x.hi + x.lo; xf = r - floorf(r);
        kmax = kmax_tiebreak(af, a1f, K, T, F, t, nyq);
    } else if (MODE == 5) {                                          // integer setup, 40 fractional bits
        const unsigned long long PHI = (unsigned long long)auxi[3 * fj], A = (unsigned long long)auxi[3 * fj + 1],
                                 D = (unsigned long long)auxi[3 * fj + 2];
        const unsigned long long m = (unsigned long long)n * (unsigned long long)(n + 1) / 2ull;
        const unsigned long long S = PHI + (unsigned long long)(n + 1) * A + D * m;   // mod 2^64, low 40 bits are the phase
        X32 = (uint32_t)(S >> 8);
        kmax = kmax_tiebreak(af, a1f, K, T, F, t, nyq);
    }
    if (MODE == 4) X32 = (uint32_t)(xd * 4294967296.0);
    unsigned long long X64 = 0;
    if (MODE == 6) X64 = ((unsigned long long)(xd * 9007199254740992.0)) << 11;   // 53 significant bits -> 2^64 scale

    const float w  = 0.5f - 0.5f * cospif((float)n / (float)hop);
    const float g0 = amp[fj], g1 = amp[fj1];
    const float* d0 = dist + fj * K;
    const float* d1 = dist + fj1 * K;

    float acc = 0.f;
    uint32_t P32 = 0; unsigned long long P64 = 0;
    for (int k = 1; k <= kmax; ++k) {
        const float ak = (g0 * d0[k - 1]) * (1.f - w) + (g1 * d1[k - 1]) * w;
        float p;
        if (MODE == 4 || MODE == 5)      { P32 += X32; p = (float)(P32 >> 8) * 5.9604644775390625e-8f; }   // top 24 bits / 2^24
        else if (MODE == 6)              { P64 += X64; p = (float)(P64 >> 40) * 5.9604644775390625e-8f; }
        else                             { p = k * xf; p -= floorf(p); }
        float s;
        if (FAST) { const float r = p - rintf(p); s = __sinf(6.2831853071795864f * r); }
        else      { s = sinpif(2.f * p); }
        acc += ak * s;
    }
    out[(size_t)b * T + t] = acc;
}

torch::Tensor fused_precision(torch::Tensor f0, torch::Tensor amp, torch::Tensor dist, torch::Tensor phi,
                              torch::Tensor auxf, torch::Tensor auxi, int64_t T, double sr, int64_t mode, bool fast_sin)
{
    TORCH_CHECK(f0.is_cuda() && dist.is_cuda() && phi.is_cuda() && auxf.is_cuda() && auxi.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(phi.scalar_type() == at::kDouble && auxf.scalar_type() == at::kFloat && auxi.scalar_type() == at::kLong, "dtypes");
    const int B = dist.size(0), F = dist.size(1), K = dist.size(2);
    TORCH_CHECK(T % F == 0, "T must be divisible by the number of frames");
    const int hop = (int)(T / F);
    auto out = torch::empty({B, T}, dist.options());
    dim3 grid((unsigned)((T + 255) / 256), (unsigned)B);
    const double nyq = sr / 2.0, c = 2.0 * 3.14159265358979323846 / sr;
    const float isr_hi = (float)(1.0 / sr), isr_lo = (float)(1.0 / sr - (double)isr_hi);
    const float ihop_hi = (float)(1.0 / hop), ihop_lo = (float)(1.0 / hop - (double)ihop_hi);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define L(M, FS) precision_kernel<M, FS><<<grid, 256, 0, stream>>>(f0.data_ptr<float>(), amp.data_ptr<float>(), \
        dist.data_ptr<float>(), phi.data_ptr<double>(), auxf.data_ptr<float>(), (const long long*)auxi.data_ptr<int64_t>(), \
        out.data_ptr<float>(), F, K, (int)T, hop, nyq, c, isr_hi, isr_lo, ihop_hi, ihop_lo)
#define M2(M) if (fast_sin) L(M, true); else L(M, false)
    switch (mode) {
        case 0: M2(0); break; case 1: M2(1); break; case 2: M2(2); break; case 3: M2(3); break;
        case 4: M2(4); break; case 5: M2(5); break; case 6: M2(6); break;
        default: TORCH_CHECK(false, "unknown mode");
    }
#undef M2
#undef L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_ext = load_inline(name="fused_precision_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["fused_precision"], extra_cuda_cflags=["-O3"], verbose=False)


def precision_inputs(f0, amp, dist, T, sr, mode):
    """Host-side (PyTorch) per-frame tables for one mode. Returns the argument tuple of _ext.fused_precision."""
    dev = f0.device
    phi = frame_phase_offsets(f0, T, sr)                                 # [B,F] fp64 rad
    auxf = torch.zeros(1, device=dev, dtype=torch.float32)
    auxi = torch.zeros(1, device=dev, dtype=torch.int64)
    if mode == 3:
        x = phi / (2 * math.pi)
        hi = x.float(); lo = (x - hi.double()).float()
        auxf = torch.stack([hi, lo], dim=-1).contiguous()
    elif mode == 5:
        a = f0[:, :, 0].double(); F = a.shape[1]; hop = T // F
        d = torch.cat([a[:, 1:], a[:, -1:]], dim=1) - a
        s40 = 2.0 ** 40
        PHI = torch.floor(phi / (2 * math.pi) * s40).long()
        A = torch.round(a / sr * s40).long()
        D = torch.round(d / (sr * hop) * s40).long()
        auxi = torch.stack([PHI, A, D], dim=-1).contiguous()
    return (f0[:, :, 0].contiguous(), amp[:, :, 0].contiguous(), dist.contiguous(), phi, auxf, auxi, int(T), float(sr))


def fused_precision_synthesis(f0, amp, dist, T, sr=16000, mode="f64", fast_sin=False):
    m = MODES[mode] if isinstance(mode, str) else int(mode)
    return _ext.fused_precision(*precision_inputs(f0, amp, dist, T, sr, m), m, fast_sin)
