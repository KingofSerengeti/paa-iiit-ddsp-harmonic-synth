"""
fused_ddsp.py - fused CUDA kernel for the DDSP harmonic synthesizer (one thread per output sample).

Algorithm (same maths as ddsp_port.harmonic_synthesis, 'window' amplitude upsampling):
  1. Frame phase offsets: inside control frame j the pitch is linear, f(n) = a_j + d_j*n/hop, so the
     phase of the base pitch is  P_j(n) = c*((n+1)*a_j + d_j*n*(n+1)/(2*hop)),  c = 2*pi/sr.
     Whole-frame total S_j = c*(hop*a_j + d_j*(hop-1)/2). Offsets Phi_j = exclusive cumsum of S_j, done in
     fp64 over only F frames (not T samples) and reduced mod 2*pi.
  2. Kernel: x = frac((Phi_j + P_j(n)) / 2pi) in fp64 -> fp32.  Harmonic k has phase k*x (fractional cycles).
     amp_k = Hann-weighted blend of frames j and j+1 (last frame held).  Harmonics with k*f >= Nyquist (decided in fp64, as in the reference) are skipped.
     y[t] = sum_k amp_k * sin(2*pi*frac(k*x)).  No [B,T,K] table is ever written.
Flags (for the ablation study): skip_inactive (stop the harmonic loop at Nyquist), fast_sin (__sinf on a
range-reduced angle instead of sinpif), block (threads per block),
fp32_setup (ablation only: per-sample phase and Nyquist cutoff in fp32 instead of fp64).
"""
import math, os
import torch
from torch.utils.cpp_extension import load_inline

import re, shutil, subprocess


def _nvcc_version():
    from torch.utils.cpp_extension import CUDA_HOME
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc") if CUDA_HOME else shutil.which("nvcc")
    m = re.search(r"release (\d+)\.(\d+)", subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# Arch selection. sm_120 (Blackwell, RTX 50xx) needs nvcc >= 12.8. With an older nvcc (this machine has 12.0) we emit
# compute_90 PTX and the driver JIT-compiles it to the GPU's native code at load time. Override with FUSED_ARCH="x.y[+PTX]".
_cc = torch.cuda.get_device_capability()
_NVCC = _nvcc_version()
if "FUSED_ARCH" in os.environ:
    ARCH = os.environ["FUSED_ARCH"]
elif _cc >= (12, 0) and _NVCC < (12, 8):
    ARCH = "9.0+PTX"
else:
    ARCH = f"{_cc[0]}.{_cc[1]}"
os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH

_CPP = ("torch::Tensor fused_harmonic(torch::Tensor f0, torch::Tensor amp, torch::Tensor dist, "
        "torch::Tensor phi, int64_t T, double sr, bool skip_inactive, bool fast_sin, int64_t block, bool fp32_setup);")

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Per-sample setup: phase (fractional cycles) and the Nyquist cutoff k_max.
// R = double is the default (exact frame phase, cutoff identical to the fp64 reference).
// R = float is an ablation only (fp32_setup=True): it measures what the fp64 arithmetic costs on this GPU.
template <typename R>
__device__ __forceinline__ void sample_setup(const float* __restrict__ f0, const double* __restrict__ phi,
                                             int b, int F, int K, int T, int hop, int t, int j, int j1, int n,
                                             double nyq_d, double c_d, float& xf, int& kmax)
{
    const R a  = (R)f0[(size_t)b * F + j];
    const R a1 = (R)f0[(size_t)b * F + j1];
    const R d  = a1 - a;
    const R inv_two_pi = (R)0.15915494309189533576888376337251;
    const R c = (R)c_d, nyq = (R)nyq_d;

    R x = ((R)phi[(size_t)b * F + j] + c * ((R)(n + 1) * a + d * (R)n * (R)(n + 1) / ((R)2 * (R)hop))) * inv_two_pi;
    x -= floor(x);
    xf = (float)x;

    // Nyquist cutoff, decided once per sample with the reference's own formula:
    // harmonic k is silent iff (k*a)*(1-fr) + (k*a1)*fr >= sr/2, fr = t*(F/T) - j  (TF1 bilinear resize).
    // An fp32 test (k*freq < nyq) disagrees with it when k*f0 lands on Nyquist, e.g. f0 = 8000/22 Hz.
    const R src = (R)t * ((R)F / (R)T);
    const R fr  = src - floor(src);
    const R fd  = a * ((R)1 - fr) + a1 * fr;
    kmax = K;
    if (fd > (R)0) {
        kmax = (int)min((R)K, floor(nyq / fd) + (R)1);
        while (kmax > 0 && ((R)kmax * a) * ((R)1 - fr) + ((R)kmax * a1) * fr >= nyq) --kmax;
        while (kmax < K && ((R)(kmax + 1) * a) * ((R)1 - fr) + ((R)(kmax + 1) * a1) * fr < nyq) ++kmax;
    }
}

template <bool SKIP, bool FAST, bool F32>
__global__ void harmonic_kernel(const float* __restrict__ f0,     // [B,F]
                                const float* __restrict__ amp,    // [B,F]
                                const float* __restrict__ dist,   // [B,F,K]
                                const double* __restrict__ phi,   // [B,F] frame phase offsets (rad, mod 2pi)
                                float* __restrict__ out,          // [B,T]
                                int F, int K, int T, int hop, double nyq, double c)
{
    const int b = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const int j  = t / hop;
    const int n  = t - j * hop;
    const int j1 = min(j + 1, F - 1);

    float xf; int kmax;
    sample_setup<typename std::conditional<F32, float, double>::type>(f0, phi, b, F, K, T, hop, t, j, j1, n, nyq, c, xf, kmax);
    const float w    = 0.5f - 0.5f * cospif((float)n / (float)hop);

    const float g0 = amp[(size_t)b * F + j];
    const float g1 = amp[(size_t)b * F + j1];
    const float* d0 = dist + ((size_t)b * F + j)  * K;
    const float* d1 = dist + ((size_t)b * F + j1) * K;

    float acc = 0.f;
    const int kend = SKIP ? kmax : K;
    for (int k = 1; k <= kend; ++k) {
        const bool active = k <= kmax;
        const float ak = (g0 * d0[k - 1]) * (1.f - w) + (g1 * d1[k - 1]) * w;
        float p = k * xf;
        p -= floorf(p);                       // fractional cycles in [0,1)
        float s;
        if (FAST) { const float r = p - rintf(p); s = __sinf(6.2831853071795864f * r); }
        else      { s = sinpif(2.f * p); }
        acc += active ? ak * s : 0.f;
    }
    out[(size_t)b * T + t] = acc;
}

torch::Tensor fused_harmonic(torch::Tensor f0, torch::Tensor amp, torch::Tensor dist,
                             torch::Tensor phi, int64_t T, double sr,
                             bool skip_inactive, bool fast_sin, int64_t block, bool fp32_setup)
{
    TORCH_CHECK(f0.is_cuda() && amp.is_cuda() && dist.is_cuda() && phi.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(f0.scalar_type() == at::kFloat && amp.scalar_type() == at::kFloat && dist.scalar_type() == at::kFloat, "f0/amp/dist must be float32");
    TORCH_CHECK(phi.scalar_type() == at::kDouble, "phi must be float64");
    TORCH_CHECK(f0.is_contiguous() && amp.is_contiguous() && dist.is_contiguous() && phi.is_contiguous(), "tensors must be contiguous");
    const int B = dist.size(0), F = dist.size(1), K = dist.size(2);
    TORCH_CHECK(T % F == 0, "T must be divisible by the number of frames");
    const int hop = (int)(T / F);
    auto out = torch::empty({B, T}, dist.options());
    dim3 grid((unsigned)((T + block - 1) / block), (unsigned)B);
    const double nyq = sr / 2.0;
    const double c = 2.0 * 3.14159265358979323846 / sr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* pf = f0.data_ptr<float>(); const float* pa = amp.data_ptr<float>();
    const float* pd = dist.data_ptr<float>(); const double* pp = phi.data_ptr<double>();
    float* po = out.data_ptr<float>();
#define LAUNCH(S, FS, F32) harmonic_kernel<S, FS, F32><<<grid, (int)block, 0, stream>>>(pf, pa, pd, pp, po, F, K, (int)T, hop, nyq, c)
    if (fp32_setup) {
        if (skip_inactive) { if (fast_sin) LAUNCH(true, true, true);  else LAUNCH(true, false, true); }
        else               { if (fast_sin) LAUNCH(false, true, true); else LAUNCH(false, false, true); }
    } else {
        if (skip_inactive) { if (fast_sin) LAUNCH(true, true, false);  else LAUNCH(true, false, false); }
        else               { if (fast_sin) LAUNCH(false, true, false); else LAUNCH(false, false, false); }
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

_ext = load_inline(name="fused_ddsp_ext", cpp_sources=_CPP, cuda_sources=_CUDA,
                   functions=["fused_harmonic"], extra_cuda_cflags=["-O3"], verbose=False)


def frame_phase_offsets(f0, T, sr=16000):
    """f0 [B,F,1] float32 (cuda) -> exclusive frame phase offsets, float64 [B,F], radians mod 2*pi."""
    a = f0[:, :, 0].double()
    F = a.shape[1]
    hop = T // F
    d = torch.cat([a[:, 1:], a[:, -1:]], dim=1) - a
    c = 2.0 * math.pi / sr
    S = c * (hop * a + d * (hop - 1) / 2.0)
    return torch.remainder(torch.cumsum(S, dim=1) - S, 2.0 * math.pi).contiguous()


def fused_harmonic_synthesis(f0, amp, dist, T, sr=16000, skip_inactive=True, fast_sin=False, block=256, fp32_setup=False):
    """Same inputs/outputs as ddsp_port.harmonic_synthesis: f0 [B,F,1], amp [B,F,1], dist [B,F,K] -> [B,T]."""
    f0c = f0[:, :, 0].contiguous()
    ampc = amp[:, :, 0].contiguous()
    phi = frame_phase_offsets(f0, T, sr)
    return _ext.fused_harmonic(f0c, ampc, dist.contiguous(), phi, int(T), float(sr), skip_inactive, fast_sin, int(block), fp32_setup)
