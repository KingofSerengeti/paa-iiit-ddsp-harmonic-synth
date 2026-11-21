"""Scaling sweeps (time only; no energy or error, to keep the sweep short). Base config (B,K,T) = (8,100,64000), f0 100-400 Hz.
  sweep B:  batch size 1..64
  sweep T:  clip length 1024..1024000 samples (B = 8)
  sweep K:  number of harmonics 16..256 at f0 100-400 Hz (many harmonics above Nyquist)
  sweep K_low: same K values at f0 20-25 Hz (all harmonics below Nyquist)
Implementations: faithful_angular_compile, v1 (fused_ddsp default), v2, v2_fastsin, v2_blockreduce. Per row: CUDA-event median of 50 calls
(ms_median, as in bench_fused.py) and 200 back-to-back calls / 200 (ms_loop). Out-of-memory is recorded as a failure.
Run:  python scaling_v2.py 2>&1 | tee scaling_v2_log.txt        (writes scaling_v2_results.csv)"""
import csv, sys
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from baseline_ddsp import time_ms
from ddsp_port import make_controls
from fused_ddsp import fused_harmonic_synthesis
from fused_v2 import fused_v2_synthesis
from bench_fused import angular, SR, HOP
from graphs_bench import loop_ms

SWEEPS = {
    "B": [(b, 100, 64000, (100.0, 400.0)) for b in (1, 2, 4, 8, 16, 32, 64)],
    "T": [(8, 100, t, (100.0, 400.0)) for t in (1024, 4096, 16000, 64000, 256000, 1024000)],
    "K": [(8, k, 64000, (100.0, 400.0)) for k in (16, 32, 64, 100, 128, 256)],
    "K_low": [(8, k, 64000, (20.0, 25.0)) for k in (16, 32, 64, 100, 128, 256)],
}

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    rows = []
    for sweep, cfgs in SWEEPS.items():
        for B, K, T, rng in cfgs:
            f0, amp, dist = make_controls(B, K, T // HOP, SR, f0_range=rng)
            active = (f0 * torch.arange(1, K + 1, device=f0.device) < SR / 2).float().mean().item()
            impls = {
                "faithful_angular_compile": lambda: (lambda c=torch.compile(angular, dynamic=False): c(f0, amp, dist, T)),
                "v1": lambda: (lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR)),
                "v2": lambda: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR)),
                "v2_fastsin": lambda: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR, fast_sin=True)),
                "v2_blockreduce": lambda: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR, frame_scan="block")),
            }
            for name, build in impls.items():
                torch._dynamo.reset(); torch.cuda.empty_cache()
                try:
                    fn = build()
                    with torch.no_grad():
                        ms, sd = time_ms(fn, warmup=10, iters=50)
                    r = dict(sweep=sweep, impl=name, B=B, K=K, T=T, f0_range=f"{rng[0]:g}-{rng[1]:g}",
                             active_harmonic_frac=round(active, 3), ms_median=round(ms, 4), ms_std=round(sd, 4),
                             ms_loop=round(loop_ms(fn), 4), samples_per_s=round(B * T / (ms / 1000)))
                    rows.append(r); print(r); sys.stdout.flush()
                except Exception as e:
                    print("FAILED", sweep, name, B, K, T, "->", repr(e)[:200]); sys.stdout.flush()
                fn = None
    with open("scaling_v2_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote scaling_v2_results.csv")
