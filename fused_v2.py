"""
fused_v2.py - second version of the fused harmonic kernel: integer phase and a single kernel launch.

Changes from fused_ddsp.py (v1), each motivated by a measurement in precision_kernels_results.csv / ablate_fp64_results.csv:
  1. Phase in fixed point. A phase in cycles is an unsigned 64-bit integer with 48 fractional bits; overflow is "mod 1 cycle".
     Per frame j:  A_j = round(a_j/sr * 2^48)  (phase step at the frame start),
                   D_j = round((a_{j+1}-a_j)/(sr*hop) * 2^48)  (change of the step per sample),
                   S_j = hop*A_j + D_j*hop*(hop-1)/2  (phase advance over the whole frame, exact in integers).
     Sample n of frame j:  phase = PHI_j + (n+1)*A_j + D_j*n*(n+1)/2,  PHI_j = S_0 + ... + S_{j-1}.
     Integer addition is exact and associative, so the frame scan has no rounding error and no order dependence.
     fp64 is used once per frame (to round a_j/sr to an integer), not per sample.
  2. Frame offsets without PyTorch ops, two options (frame_scan):
     "scan" (default): a small kernel (one block per batch item) scans the F frames once and stores {PHI_j, A_j, D_j};
                       the synthesis kernel then uses integers only per sample (no fp64). Two launches per call.
     "block": one launch; each block sums S_i over the frames before its first frame (block reduction). The redundant work
              grows with F^2 / block; it was the default in the first v2 benchmark and was slow for the 40 s clip.
  3. Harmonic phase in uint32 (top 32 of the 48 fractional bits): P_k = P_{k-1} + X, exact mod 1, no floor().
  4. Nyquist cutoff in fp32; recomputed with the fp64 reference formula only when a harmonic is within 1e-3 harmonic numbers
     of Nyquist (the fused_ddsp v1 bug case f0 = 8000/22 Hz is such a tie).
Flags: fast_sin (__sinf on a range-reduced angle instead of sinpif), block (threads per block).
"""
import torch
from torch.utils.cpp_extension import load_inline
import fused_ddsp                                   # sets TORCH_CUDA_ARCH_LIST the same way as v1

_CPP = "torch::Tensor fused_v2(torch::Tensor f0, torch::Tensor amp, torch::Tensor dist, int64_t T, double sr, bool fast_sin, int64_t block, bool scan);"

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
typedef unsigned long long u64;

__device__ __forceinline__ int kmax_exact(double a, double a1, int K, int T, int F, int t, double nyq)
{
    const double src = (double)t * ((double)F / (double)T);
    const double fr  = src - floor(src);
    const double fd  = a * (1.0 - fr) + a1 * fr;
    int kmax = K;
    if (fd > 0.0) {
        kmax = (int)fmin((double)K, floor(nyq / fd) + 1.0);
        while (kmax > 0 && ((double)kmax * a) * (1.0 - fr) + ((double)kmax * a1) * fr >= nyq) --kmax;
        while (kmax < K && ((double)(kmax + 1) * a) * (1.0 - fr) + ((double)(kmax + 1) * a1) * fr < nyq) ++kmax;
    }
    return kmax;
}

__device__ __forceinline__ int kmax_tiebreak(float a, float a1, int K, int T, int F, int t, double nyq)
{
    const float src = (float)t * ((float)F / (float)T);
    const float fr  = src - floorf(src);
    const float fd  = a * (1.f - fr) + a1 * fr;
    if (!(fd > 0.f)) return K;
    const float r = (float)nyq / fd;                    // harmonic k is active iff k < r
    if (fabsf(r - rintf(r)) < 1e-3f) return kmax_exact((double)a, (double)a1, K, T, F, t, nyq);
    return (int)fminf(fmaxf(ceilf(r) - 1.f, 0.f), (float)K);
}

// Per-frame integer phase step A and step change D (48 fractional bits).
__device__ __forceinline__ void frame_AD(const float* __restrict__ f, int i, int F, int hop, double scale, u64& A, u64& D)
{
    const double a  = (double)f[i];
    const double a1 = (double)f[min(i + 1, F - 1)];
    A = (u64)llrint(a * scale);
    D = (u64)llrint((a1 - a) * (scale / (double)hop));
}
__device__ __forceinline__ u64 frame_total(const float* __restrict__ f, int i, int F, int hop, double scale)
{
    u64 A, D; frame_AD(f, i, F, hop, scale, A, D);
    return (u64)hop * A + D * ((u64)hop * (u64)(hop - 1) / 2ull);
}

// Frame scan (frame_scan="scan"): one block per batch item writes {PHI_j, A_j, D_j} for every frame, exact integer scan.
__global__ void frame_scan_kernel(const float* __restrict__ f0, u64* __restrict__ tab, int F, int hop, double scale)
{
    const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const float* fb = f0 + (size_t)b * F;
    const int chunk = (F + nt - 1) / nt, start = min(F, tid * chunk), end = min(F, start + chunk);
    u64 local = 0;
    for (int i = start; i < end; ++i) local += frame_total(fb, i, F, hop, scale);
    // block-wide exclusive scan of the per-thread sums
    const int lane = tid & 31, wid = tid >> 5;
    u64 incl = local;
    for (int o = 1; o < 32; o <<= 1) { u64 v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
    __shared__ u64 wsum[32];
    if (lane == 31) wsum[wid] = incl;
    __syncthreads();
    if (wid == 0) {
        u64 v = (lane < (nt + 31) / 32) ? wsum[lane] : 0ull, x = v;
        for (int o = 1; o < 32; o <<= 1) { u64 y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        wsum[lane] = x - v;                                              // exclusive prefix of warp sums
    }
    __syncthreads();
    u64 run = wsum[wid] + incl - local;
    u64* tb = tab + (size_t)b * F * 3;
    for (int i = start; i < end; ++i) {
        u64 A, D; frame_AD(fb, i, F, hop, scale, A, D);
        tb[3 * i] = run; tb[3 * i + 1] = A; tb[3 * i + 2] = D;
        run += (u64)hop * A + D * ((u64)hop * (u64)(hop - 1) / 2ull);
    }
}

template <bool FAST, bool SCAN>
__global__ void v2_kernel(const float* __restrict__ f0, const float* __restrict__ amp, const float* __restrict__ dist,
                          const u64* __restrict__ tab, float* __restrict__ out, int F, int K, int T, int hop, double nyq, double scale)
{
    const int b = blockIdx.y, tid = threadIdx.x;
    const int t0 = blockIdx.x * blockDim.x, t = t0 + tid;
    const float* fb = f0 + (size_t)b * F;
    const int j0 = t0 / hop;
    u64 S;
    if (SCAN) {
        if (t >= T) return;
        const int j = t / hop, n = t - j * hop;
        const u64* e = tab + ((size_t)b * F + j) * 3;
        S = e[0] + (u64)(n + 1) * e[1] + e[2] * ((u64)n * (u64)(n + 1) / 2ull);
    } else {
        // PHI_{j0} = sum of frame totals before the block's first frame (block reduction, integer, exact)
        u64 part = 0;
        for (int i = tid; i < j0; i += blockDim.x) part += frame_total(fb, i, F, hop, scale);
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        __shared__ u64 ws[32];
        if ((tid & 31) == 0) ws[tid >> 5] = part;
        __syncthreads();
        if (tid < 32) {
            u64 v = (tid < (int)((blockDim.x + 31) / 32)) ? ws[tid] : 0ull;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
            if (tid == 0) ws[0] = v;
        }
        __syncthreads();
        if (t >= T) return;

        const int j = t / hop, n = t - j * hop;
        u64 PHI = ws[0];
        for (int i = j0; i < j; ++i) PHI += frame_total(fb, i, F, hop, scale);
        u64 A, D; frame_AD(fb, j, F, hop, scale, A, D);
        S = PHI + (u64)(n + 1) * A + D * ((u64)n * (u64)(n + 1) / 2ull);
    }
    const int j = t / hop, n = t - j * hop, j1 = min(j + 1, F - 1);
    const uint32_t X = (uint32_t)(S >> 16);                          // top 32 of the 48 fractional bits

    const int kmax = kmax_tiebreak(fb[j], fb[j1], K, T, F, t, nyq);
    const float w  = 0.5f - 0.5f * cospif((float)n / (float)hop);
    const float g0 = amp[(size_t)b * F + j], g1 = amp[(size_t)b * F + j1];
    const float* d0 = dist + ((size_t)b * F + j) * K;
    const float* d1 = dist + ((size_t)b * F + j1) * K;

    float acc = 0.f;
    uint32_t P = 0;
    for (int k = 1; k <= kmax; ++k) {
        const float ak = (g0 * d0[k - 1]) * (1.f - w) + (g1 * d1[k - 1]) * w;
        P += X;
        const float p = (float)(P >> 8) * 5.9604644775390625e-8f;     // top 24 bits / 2^24, in [0,1)
        float s;
        if (FAST) { const float r = p - rintf(p); s = __sinf(6.2831853071795864f * r); }
        else      { s = sinpif(2.f * p); }
        acc += ak * s;
    }
    out[(size_t)b * T + t] = acc;
}

torch::Tensor fused_v2(torch::Tensor f0, torch::Tensor amp, torch::Tensor dist, int64_t T, double sr, bool fast_sin, int64_t block, bool scan)
{
    TORCH_CHECK(f0.is_cuda() && amp.is_cuda() && dist.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(f0.scalar_type() == at::kFloat && amp.scalar_type() == at::kFloat && dist.scalar_type() == at::kFloat, "float32 only");
    TORCH_CHECK(f0.is_contiguous() && amp.is_contiguous() && dist.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(block % 32 == 0 && block <= 1024, "block must be a multiple of 32, <= 1024");
    const int B = dist.size(0), F = dist.size(1), K = dist.size(2);
    TORCH_CHECK(T % F == 0, "T must be divisible by the number of frames");
    const int hop = (int)(T / F);
    auto out = torch::empty({B, T}, dist.options());
    dim3 grid((unsigned)((T + block - 1) / block), (unsigned)B);
    const double scale = 281474976710656.0 / sr;                     // 2^48 / sr
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor tab = scan ? torch::empty({B, F, 3}, f0.options().dtype(at::kLong)) : torch::empty({1}, f0.options().dtype(at::kLong));
    const u64* pt = (const u64*)tab.data_ptr<int64_t>();
    if (scan) frame_scan_kernel<<<B, 1024, 0, stream>>>(f0.data_ptr<float>(), (u64*)tab.data_ptr<int64_t>(), F, hop, scale);
#define L(FS, SC) v2_kernel<FS, SC><<<grid, (int)block, 0, stream>>>(f0.data_ptr<float>(), amp.data_ptr<float>(), \
        dist.data_ptr<float>(), pt, out.data_ptr<float>(), F, K, (int)T, hop, sr / 2.0, scale)
    if (scan) { if (fast_sin) L(true, true);  else L(false, true); }
    else      { if (fast_sin) L(true, false); else L(false, false); }
#undef L
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_ext = load_inline(name="fused_v2_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["fused_v2"], extra_cuda_cflags=["-O3"], verbose=False)


def fused_v2_synthesis(f0, amp, dist, T, sr=16000, fast_sin=False, block=256, frame_scan="scan"):
    """Same inputs/outputs as ddsp_port.harmonic_synthesis: f0 [B,F,1], amp [B,F,1], dist [B,F,K] -> [B,T].
    frame_scan="scan": two launches (frame scan kernel, then the synthesis kernel); "block": one launch, each block sums the
    frames before it (work grows with F^2 / block, see bench_v2 results for the 40 s clip)."""
    return _ext.fused_v2(f0[:, :, 0].contiguous(), amp[:, :, 0].contiguous(), dist.contiguous(), int(T), float(sr), fast_sin,
                         int(block), frame_scan == "scan")
