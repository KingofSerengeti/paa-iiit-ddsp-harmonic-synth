"""
Baseline benchmark v2: faithful port of DDSP harmonic_synthesis.
Variants: faithful_eager, faithful_compile, faithful_angular_eager (DDSP's own accurate-phase option), faithful_angular_compile
Run:  python baseline_faithful.py          (needs baseline_ddsp.py, ddsp_port.py in the same folder or on PYTHONPATH)
Writes baseline_faithful_results.csv
"""
import csv, sys
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from baseline_ddsp import time_ms, energy_mj        # timing + NVML energy helpers from the first script
from ddsp_port import harmonic_synthesis, make_controls

SR, HOP = 16000, 64


def eager(f0, amp, dist, T):
    return harmonic_synthesis(f0, amp, dist, T, SR, "window", False)


def eager_angular(f0, amp, dist, T):
    return harmonic_synthesis(f0, amp, dist, T, SR, "window", True)


def run_config(name, impl, B, K, T):
    nf = T // HOP
    f0, amp, dist = make_controls(B, K, nf, SR)
    with torch.no_grad():
        ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                            T, SR, "window", False) for i in range(B)])      # fp64 truth, batch by batch
        torch.cuda.empty_cache()
        out = impl(f0, amp, dist, T); torch.cuda.synchronize()
        err = (out.double() - ref).abs().max().item()
        rel = ((out.double() - ref).norm() / ref.norm()).item()
        freqs = f0 * torch.arange(1, K + 1, device=f0.device)
        active = (freqs < SR / 2).float().mean().item()
        torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        fn = lambda: impl(f0, amp, dist, T)
        ms, sd = time_ms(fn, warmup=10, iters=50)
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
        emj, esd = energy_mj(fn, ms)
    return dict(impl=name, B=B, K=K, T=T, ms_median=round(ms, 4), ms_std=round(sd, 4),
                samples_per_s=round(B * T / (ms / 1000)), peak_extra_MB=round(peak, 1),
                energy_mJ=round(emj, 3), energy_std=round(esd, 3),
                max_abs_err_vs_fp64=f"{err:.3e}", rel_l2_err=f"{rel:.3e}", active_harmonic_frac=round(active, 3))


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    impls = {"faithful_eager": eager, "faithful_angular_eager": eager_angular}
    for nm, fn in (("faithful_compile", eager), ("faithful_angular_compile", eager_angular)):
        try:
            impls[nm] = torch.compile(fn, dynamic=False)
        except Exception as e:
            print("compile unavailable", nm, e)
    configs = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000), (8, 128, 64000), (16, 100, 64000)]
    rows = []
    for B, K, T in configs:
        for name, impl in impls.items():
            try:
                r = run_config(name, impl, B, K, T); rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:300])
    if rows:
        with open("baseline_faithful_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print("wrote baseline_faithful_results.csv")
