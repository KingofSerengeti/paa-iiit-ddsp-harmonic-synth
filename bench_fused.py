"""Benchmark + ablations for the fused kernel, in the same session and with the same harness as baseline_faithful.py.
Run:  python bench_fused.py 2>&1 | tee bench_fused_log.txt        (writes fused_results.csv)
      python bench_fused.py --no-baseline                          (skip the torch.compile baseline, faster)
      python bench_fused.py --out fused_results_run2.csv           (write to another CSV)
Rows: fused (default), fused_noskip, fused_fastsin, fused_kernel_only (phase offsets precomputed),
      block-size sweep at the main config, and faithful_angular_compile as the opponent."""
import csv, sys
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from baseline_ddsp import time_ms, energy_mj
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis, frame_phase_offsets, _ext

SR, HOP = 16000, 64
WITH_BASELINE = "--no-baseline" not in sys.argv
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "fused_results.csv"


def angular(f0, amp, dist, T):
    return harmonic_synthesis(f0, amp, dist, T, SR, "window", True)


def measure(name, fn, B, K, T, ref, active):
    with torch.no_grad():
        out = fn(); torch.cuda.synchronize()
        err = (out.double() - ref).abs().max().item()
        rel = ((out.double() - ref).norm() / ref.norm()).item()
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        ms, sd = time_ms(fn, warmup=10, iters=50)
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        emj, esd = energy_mj(fn, ms)
    return dict(impl=name, B=B, K=K, T=T, ms_median=round(ms, 4), ms_std=round(sd, 4),
                samples_per_s=round(B * T / (ms / 1000)), peak_extra_MB=round(peak, 1),
                energy_mJ=round(emj, 3), energy_std=round(esd, 3),
                max_abs_err_vs_fp64=f"{err:.3e}", rel_l2_err=f"{rel:.3e}", active_harmonic_frac=round(active, 3))


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    compiled = torch.compile(angular, dynamic=False) if WITH_BASELINE else None
    configs = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000), (8, 128, 64000), (16, 100, 64000)]
    rows = []
    for B, K, T in configs:
        f0, amp, dist = make_controls(B, K, T // HOP, SR)
        with torch.no_grad():
            ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                                T, SR, "window", False) for i in range(B)])
            torch.cuda.empty_cache()
        active = (f0 * torch.arange(1, K + 1, device=f0.device) < SR / 2).float().mean().item()
        f0c, ampc, distc = f0[:, :, 0].contiguous(), amp[:, :, 0].contiguous(), dist.contiguous()
        phi = frame_phase_offsets(f0, T, SR)
        variants = {
            "fused": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR),
            "fused_noskip": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR, skip_inactive=False),
            "fused_fastsin": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR, fast_sin=True),
            "fused_kernel_only": lambda: _ext.fused_harmonic(f0c, ampc, distc, phi, T, float(SR), True, False, 256, False),
        }
        if (B, K, T) == (8, 100, 64000):
            for blk in (64, 128, 512, 1024):
                variants[f"fused_block{blk}"] = (lambda blk=blk: fused_harmonic_synthesis(f0, amp, dist, T, SR, block=blk))
        if WITH_BASELINE:
            variants["faithful_angular_compile"] = lambda: compiled(f0, amp, dist, T)
        for name, fn in variants.items():
            try:
                r = measure(name, fn, B, K, T, ref, active); rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:300])
    if rows:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print("wrote", OUT)
