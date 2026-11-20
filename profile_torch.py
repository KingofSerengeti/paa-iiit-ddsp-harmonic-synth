"""Fallback profiler (works without Nsight): kernel count and time split for eager vs torch.compile.
Run from the same folder as baseline_ddsp.py:  python profile_torch.py > torch_profile.txt 2>&1"""
import torch
from torch.profiler import profile, ProfilerActivity
from baseline_ddsp import harmonic_synth, make_inputs

B, K, T = 8, 100, 64000
f0, amp = make_inputs(B, K, T)
compiled = torch.compile(harmonic_synth, dynamic=False)
N = 20
for name, fn in [("eager", harmonic_synth), ("compile", compiled)]:
    with torch.no_grad():
        for _ in range(10):
            fn(f0, amp, T)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(N):
                fn(f0, amp, T)
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
