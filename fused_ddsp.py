"""
fused_ddsp.py - fused CUDA kernel for the DDSP harmonic synthesizer (one thread per output sample).

Algorithm (same maths as ddsp_port.harmonic_synthesis, 'window' amplitude upsampling):
  1. Frame phase offsets: inside control frame j the pitch is linear, f(n) = a_j + d_j*n/hop, so the
     phase of the base pitch is  P_j(n) = c*((n+1)*a_j + d_j*n*(n+1)/(2*hop)),  c = 2*pi/sr.
     Whole-frame total S_j = c*(hop*a_j + d_j*(hop-1)/2). Offsets Phi_j = exclusive cumsum of S_j, done in
     fp64 over only F frames (not T samples) and reduced mod 2*pi.
  2. Kernel: x = frac((Phi_j + P_j(n)) / 2pi) in fp64 -> fp32.  Harmonic k has phase k*x (fractional cycles).
     amp_k = Hann-weighted blend of frames j and j+1 (last frame held).  Harmonics with k*f >= Nyquist are skipped.
     y[t] = sum_k amp_k * sin(2*pi*frac(k*x)).  No [B,T,K] table is ever written.
Flags (for the ablation study): skip_inactive (stop the harmonic loop at Nyquist), fast_sin (__sinf on a
range-reduced angle instead of sinpif), block (threads per block).
"""
import math, os
import torch
from torch.utils.cpp_extension import load_inline

_cc = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{_cc[0]}.{_cc[1]}"

_CPP = ("torch::Tensor fused_harmonic(torch::Tensor f0, torch::Tensor amp, torch::Tensor dist, "
        "torch::Tensor phi, int64_t T, double sr, bool skip_inactive, bool fast_sin, int64_t block);")

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>

template <bool SKIP, bool FAST>
__global__ void harmonic_kernel(const float* __restrict__ f0,     // [B,F]
                                const float* __restrict__ amp,    // [B,F]
                                const float* __restrict__ dist,   // [B,F,K]
                                const double* __restrict__ phi,   // [B,F] frame phase offsets (rad, mod 2pi)
                                float* __restrict__ out,          // [B,T]
                                int F, int K, int T, int hop, float nyq, double c)
{
    const int b = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const int j  = t / hop;
    const int n  = t - j * hop;
    const int j1 = min(j + 1, F - 1);

    const double a  = (double)f0[(size_t)b * F + j];
    const double a1 = (double)f0[(size_t)b * F + j1];
    const double d  = a1 - a;
    const double inv_two_pi = 0.15915494309189533576888376337251;

    double x = (phi[(size_t)b * F + j] + c * ((n + 1) * a + d * (double)n * (double)(n + 1) / (2.0 * hop))) * inv_two_pi;
    x -= floor(x);
    const float xf   = (float)x;
    const float freq = (float)(a + d * (double)n / (double)hop);
    const float w    = 0.5f - 0.5f * cospif((float)n / (float)hop);

    const float g0 = amp[(size_t)b * F + j];
    const float g1 = amp[(size_t)b * F + j1];
    const float* d0 = dist + ((size_t)b * F + j)  * K;
    const float* d1 = dist + ((size_t)b * F + j1) * K;

    float acc = 0.f;
    for (int k = 1; k <= K; ++k) {
        const bool active = (k * freq) < nyq;
        if (SKIP && !active) break;
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
                             bool skip_inactive, bool fast_sin, int64_t block)
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
    const float nyq = (float)(sr / 2.0);
    const double c = 2.0 * 3.14159265358979323846 / sr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* pf = f0.data_ptr<float>(); const float* pa = amp.data_ptr<float>();
    const float* pd = dist.data_ptr<float>(); const double* pp = phi.data_ptr<double>();
    float* po = out.data_ptr<float>();
    if (skip_inactive && !fast_sin)      harmonic_kernel<true,  false><<<grid, (int)block, 0, stream>>>(pf, pa, pd, pp, po, F, K, (int)T, hop, nyq, c);
    else if (skip_inactive && fast_sin)  harmonic_kernel<true,  true ><<<grid, (int)block, 0, stream>>>(pf, pa, pd, pp, po, F, K, (int)T, hop, nyq, c);
    else if (!skip_inactive && !fast_sin) harmonic_kernel<false, false><<<grid, (int)block, 0, stream>>>(pf, pa, pd, pp, po, F, K, (int)T, hop, nyq, c);
    else                                  harmonic_kernel<false, true ><<<grid, (int)block, 0, stream>>>(pf, pa, pd, pp, po, F, K, (int)T, hop, nyq, c);
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


def fused_harmonic_synthesis(f0, amp, dist, T, sr=16000, skip_inactive=True, fast_sin=False, block=256):
    """Same inputs/outputs as ddsp_port.harmonic_synthesis: f0 [B,F,1], amp [B,F,1], dist [B,F,K] -> [B,T]."""
    f0c = f0[:, :, 0].contiguous()
    ampc = amp[:, :, 0].contiguous()
    phi = frame_phase_offsets(f0, T, sr)
    return _ext.fused_harmonic(f0c, ampc, dist.contiguous(), phi, int(T), float(sr), skip_inactive, fast_sin, int(block))
