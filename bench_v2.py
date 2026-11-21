"""Benchmark of the v2 kernel (fused_v2.py) against v1 (fused_ddsp.py) and torch.compile, with and without CUDA graphs.
Same harness as bench_fused.py / graphs_bench.py (CUDA-event median of 50, NVML energy, fp64 reference), plus ms_loop.
v2 = default frame_scan="scan" (two launches); v2_blockreduce = frame_scan="block" (one launch, F^2 redundant work).
Configs: the six standard ones plus a 40 s clip (1,100,640000) and a larger batch (32,100,64000).
Run:  python bench_v2.py 2>&1 | tee bench_v2_log.txt                 (writes v2_results.csv)
      python bench_v2.py --out v2_results_run2.csv"""
import csv, math, sys
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis
from fused_v2 import fused_v2_synthesis
from bench_fused import measure, angular, SR, HOP
from graphs_bench import graphed, loop_ms, stepped

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "v2_results.csv"

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    configs = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000), (8, 128, 64000), (16, 100, 64000),
               (1, 100, 640000), (32, 100, 64000)]
    rows = []
    for B, K, T in configs:
        f0, amp, dist = make_controls(B, K, T // HOP, SR)
        with torch.no_grad():
            ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                                T, SR, "window", False) for i in range(B)])
        active = (f0 * torch.arange(1, K + 1, device=f0.device) < SR / 2).float().mean().item()
        builders = {
            "faithful_angular_compile": lambda: (lambda c=torch.compile(angular, dynamic=False): c(f0, amp, dist, T)),
            "faithful_angular_compile_graphs": lambda: stepped(
                lambda c=torch.compile(angular, dynamic=False, mode="reduce-overhead"): c(f0, amp, dist, T)),
            "v1": lambda: (lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR)),
            "v1_graph": lambda: graphed(lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR)),
            "v2": lambda: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR)),
            "v2_graph": lambda: graphed(lambda: fused_v2_synthesis(f0, amp, dist, T, SR)),
            "v2_fastsin": lambda: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR, fast_sin=True)),
            "v2_fastsin_graph": lambda: graphed(lambda: fused_v2_synthesis(f0, amp, dist, T, SR, fast_sin=True)),
            "v2_blockreduce": lambda: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR, frame_scan="block")),
            "v2_blockreduce_graph": lambda: graphed(lambda: fused_v2_synthesis(f0, amp, dist, T, SR, frame_scan="block")),
        }
        if (B, K, T) == (8, 100, 64000):
            for blk in (64, 128, 512, 1024):
                builders[f"v2_block{blk}"] = (lambda blk=blk: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR, block=blk)))
                builders[f"v2_blockreduce_block{blk}"] = (lambda blk=blk: (lambda: fused_v2_synthesis(f0, amp, dist, T, SR, block=blk,
                                                                                                   frame_scan="block")))
        for name, build in builders.items():
            torch._dynamo.reset(); torch.cuda.empty_cache()
            try:
                fn = build()
                r = measure(name, fn, B, K, T, ref, active)
                r["ms_loop"] = round(loop_ms(fn), 4)
                r["snr_db"] = round(-20 * math.log10(float(r["rel_l2_err"])), 1)
                rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:300]); sys.stdout.flush()
            fn = None
        del ref; torch.cuda.empty_cache()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", OUT)
