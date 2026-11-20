"""Profile the FAITHFUL port (ddsp_port.py): eager, angular eager, angular compile at (B=8, K=100, T=64000).
Run from the project folder (needs ddsp_port.py next to it):
    python profile_faithful.py > torch_profile_faithful.txt 2>&1
"""
import torch
from torch.profiler import profile, ProfilerActivity
from ddsp_port import harmonic_synthesis, make_controls

SR, HOP = 16000, 64
B, K, T = 8, 100, 64000
f0, amp, dist = make_controls(B, K, T // HOP, SR)


def eager(f0, amp, dist, T):
    return harmonic_synthesis(f0, amp, dist, T, SR, "window", False)


def eager_angular(f0, amp, dist, T):
    return harmonic_synthesis(f0, amp, dist, T, SR, "window", True)


compiled_angular = torch.compile(eager_angular, dynamic=False)
N = 20
for name, fn in [("faithful_eager", eager),
                 ("faithful_angular_eager", eager_angular),
                 ("faithful_angular_compile", compiled_angular)]:
    with torch.no_grad():
        for _ in range(10):
            fn(f0, amp, dist, T)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(N):
                fn(f0, amp, dist, T)
            torch.cuda.synchronize()
    print("=====", name, f"(B={B}, K={K}, T={T})")
    for key in ("device_time_total", "cuda_time_total"):
        try:
            print(prof.key_averages().table(sort_by=key, row_limit=12, max_name_column_width=60))
            break
        except Exception:
            continue
    ev = prof.events()
    kernels = [e for e in ev if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
    print("CUDA events per call (kernels + memcpy/memset):", len(kernels) / N)
