"""Correctness test: fused CUDA kernel vs fp64 truth (ddsp_port). Run from the project folder:
    python test_fused.py 2>&1 | tee test_fused_log.txt
Expect fused rel_l2 around 1e-6 to 1e-5 (far below angular eager fp32). Do not benchmark until this passes."""
import torch
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis

SR, HOP = 16000, 64
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
for B, K, T in [(2, 100, 1024), (2, 100, 16000), (1, 128, 64000), (8, 100, 64000)]:
    f0, amp, dist = make_controls(B, K, T // HOP, SR)
    with torch.no_grad():
        ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                            T, SR, "window", False) for i in range(B)])
        outs = {
            "fused (skip, sinpif)": fused_harmonic_synthesis(f0, amp, dist, T, SR),
            "fused no-skip": fused_harmonic_synthesis(f0, amp, dist, T, SR, skip_inactive=False),
            "fused fast-sin": fused_harmonic_synthesis(f0, amp, dist, T, SR, fast_sin=True),
            "angular eager fp32": harmonic_synthesis(f0, amp, dist, T, SR, "window", True),
        }
        print(f"--- B={B} K={K} T={T}")
        for name, o in outs.items():
            o = o.double()
            print(f"  {name:22s} max_abs={(o - ref).abs().max().item():.3e}  rel_l2={((o - ref).norm() / ref.norm()).item():.3e}")
