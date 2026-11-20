"""torch.profiler kernel table for the fused kernel and the torch.compile baseline at (8,100,64000).
Stand-in for Nsight Compute, which is not available for sm_120 on this machine.
Run:  python profile_fused.py > torch_profile_fused.txt 2>&1"""
import torch
from torch.profiler import profile, ProfilerActivity
from ddsp_port import make_controls
from fused_ddsp import fused_harmonic_synthesis
from bench_fused import angular, SR, HOP

B, K, T = 8, 100, 64000
N = 20
f0, amp, dist = make_controls(B, K, T // HOP, SR)
compiled = torch.compile(angular, dynamic=False)
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
for name, fn in [("fused", lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR)),
                 ("faithful_angular_compile", lambda: compiled(f0, amp, dist, T))]:
    with torch.no_grad():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(N):
                fn()
            torch.cuda.synchronize()
    print("=====", name, f"(B={B}, K={K}, T={T}), {N} calls")
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=15, max_name_column_width=70))
    kernels = [e for e in prof.events() if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA]
    print("CUDA events per call (kernels + memcpy/memset):", len(kernels) / N)
