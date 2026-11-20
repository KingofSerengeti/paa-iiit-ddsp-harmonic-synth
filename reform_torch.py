"""Control experiment: the fused kernel's maths (closed-form frame phase, one phase per sample, fp64 Nyquist cutoff)
written in plain PyTorch, run eager and with torch.compile. Separates the gain from the reformulated maths from the gain
of the hand-written CUDA kernel. Same configs and harness as bench_fused.py; fused rows are re-measured in the same session.
Run:  python reform_torch.py 2>&1 | tee reform_torch_log.txt        (writes reform_torch_results.csv)"""
import csv, math, sys
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from ddsp_port import harmonic_synthesis, make_controls, upsample_with_windows
from fused_ddsp import fused_harmonic_synthesis, frame_phase_offsets
from bench_fused import measure, angular, SR, HOP


def reform(f0, amp, dist, T, sr=SR):
    """Same maths as fused_ddsp, as PyTorch tensor ops. f0/amp [B,F,1], dist [B,F,K] fp32 -> [B,T] fp32."""
    B, F, K = dist.shape
    hop = T // F
    a = f0[:, :, 0].double()
    a1 = torch.cat([a[:, 1:], a[:, -1:]], dim=1)
    d = a1 - a
    phi = frame_phase_offsets(f0, T, sr)                                     # [B,F] fp64, exclusive, mod 2pi
    t = torch.arange(T, device=f0.device)
    j, n = t // hop, (t % hop).double()
    c = 2 * math.pi / sr
    x = (phi[:, j] + c * ((n + 1) * a[:, j] + d[:, j] * n * (n + 1) / (2.0 * hop))) / (2 * math.pi)
    x = (x - torch.floor(x)).float()                                         # [B,T] base phase, cycles
    # Nyquist cutoff k_max per sample, fp64, reference formula (k*a)*(1-fr) + (k*a1)*fr < sr/2
    src = t.double() * (F / T)
    fr = src - torch.floor(src)
    aj, a1j = a[:, j], a1[:, j]
    fd = aj * (1 - fr) + a1j * fr
    nyq = sr / 2.0
    c0 = torch.floor(nyq / fd) - 1
    kmax = c0 - 1
    for m in range(3):
        km = c0 + m
        kmax = kmax + ((km * aj) * (1 - fr) + (km * a1j) * fr < nyq).double()
    kmax = kmax.clamp(0, K)                                                  # [B,T]
    k = torch.arange(1, K + 1, device=f0.device, dtype=torch.float32)
    amp_env = upsample_with_windows(amp * dist, T)                           # [B,T,K] fp32
    p = x[:, :, None] * k
    p = p - torch.floor(p)
    s = torch.sin(2 * math.pi * p)
    mask = (k[None, None, :] <= kmax.float()[:, :, None])
    return torch.where(mask, amp_env * s, torch.zeros_like(s)).sum(-1)


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    compiled_reform = torch.compile(reform, dynamic=False)
    compiled_angular = torch.compile(angular, dynamic=False)
    configs = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000), (8, 128, 64000), (16, 100, 64000)]
    rows = []
    for B, K, T in configs:
        f0, amp, dist = make_controls(B, K, T // HOP, SR)
        with torch.no_grad():
            ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                                T, SR, "window", False) for i in range(B)])
            torch.cuda.empty_cache()
        active = (f0 * torch.arange(1, K + 1, device=f0.device) < SR / 2).float().mean().item()
        variants = {
            "fused": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR),
            "reform_eager": lambda: reform(f0, amp, dist, T),
            "reform_compile": lambda: compiled_reform(f0, amp, dist, T),
            "faithful_angular_compile": lambda: compiled_angular(f0, amp, dist, T),
        }
        for name, fn in variants.items():
            try:
                r = measure(name, fn, B, K, T, ref, active); rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:300])
            torch.cuda.empty_cache()
    with open("reform_torch_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote reform_torch_results.csv")
