"""Pitch sweep at (B,K,T) = (8,100,64000): how much does skipping harmonics above Nyquist save, and how does it
depend on f0? Compares fused (skip), fused_noskip and faithful_angular_compile for several f0 ranges.
Run:  python sweep_f0.py 2>&1 | tee sweep_f0_log.txt        (writes sweep_f0_results.csv)"""
import csv, sys
import torch
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis
from bench_fused import measure, angular, SR, HOP

B, K, T = 8, 100, 64000
RANGES = [(50, 60), (100, 120), (200, 240), (400, 480), (800, 960), (1600, 1920)]

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    compiled = torch.compile(angular, dynamic=False)
    rows = []
    for lo, hi in RANGES:
        f0, amp, dist = make_controls(B, K, T // HOP, SR, f0_range=(float(lo), float(hi)))
        with torch.no_grad():
            ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                                T, SR, "window", False) for i in range(B)])
            torch.cuda.empty_cache()
        active = (f0 * torch.arange(1, K + 1, device=f0.device) < SR / 2).float().mean().item()
        variants = {
            "fused": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR),
            "fused_noskip": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR, skip_inactive=False),
            "faithful_angular_compile": lambda: compiled(f0, amp, dist, T),
        }
        for name, fn in variants.items():
            try:
                r = {"f0_range": f"{lo}-{hi}", **measure(name, fn, B, K, T, ref, active)}
                rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, lo, hi, "->", repr(e)[:300])
    with open("sweep_f0_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote sweep_f0_results.csv")
