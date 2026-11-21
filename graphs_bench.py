"""CUDA graph baselines. Question: does removing CPU launch overhead (CUDA graphs) make kernel fusion unnecessary?
Variants per config (same harness, inputs and fp64 reference as bench_fused.py):
  faithful_eager / faithful_eager_graph          DDSP port, angular path, eager / captured once in a CUDA graph and replayed
  faithful_angular_compile / _compile_graphs     torch.compile default / mode="reduce-overhead" (Inductor + CUDA graphs)
  reform_compile / reform_compile_graphs         kernel's maths in PyTorch (reform_torch.py), compile / reduce-overhead
  fused / fused_graph                            fused CUDA kernel incl. frame-offset ops / same captured in a CUDA graph
  fused_kernel_only                              kernel launch only (frame offsets precomputed)
Columns added to bench_fused.measure():
  ms_loop      time of 200 back-to-back calls / 200 (CPU may run ahead of the GPU; per-call ms_median does not allow that)
  reserved_MB  CUDA memory reserved by the variant (build + graph capture + timing), from an empty cache; includes graph pools
Run:  python graphs_bench.py 2>&1 | tee graphs_bench_log.txt        (writes graphs_results.csv)"""
import csv, sys
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis, frame_phase_offsets, _ext
from bench_fused import measure, angular, SR, HOP
from reform_torch import reform

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "graphs_results.csv"


def graphed(fn):
    """Capture fn() once into a CUDA graph; return a callable that replays it and returns the static output."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        out = fn()
    def run():
        g.replay()
        return out
    return run


def loop_ms(fn, n=200):
    with torch.no_grad():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); e.synchronize()
    return s.elapsed_time(e) / n


def stepped(fn):
    def run():
        torch.compiler.cudagraph_mark_step_begin()
        return fn()
    return run


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    configs = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000), (8, 128, 64000), (16, 100, 64000)]
    rows = []
    for B, K, T in configs:
        f0, amp, dist = make_controls(B, K, T // HOP, SR)
        with torch.no_grad():
            ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                                T, SR, "window", False) for i in range(B)])
        active = (f0 * torch.arange(1, K + 1, device=f0.device) < SR / 2).float().mean().item()
        f0c, ampc, distc = f0[:, :, 0].contiguous(), amp[:, :, 0].contiguous(), dist.contiguous()
        phi = frame_phase_offsets(f0, T, SR)
        # builders: each returns the callable to measure (graph capture / compilation happen inside the build)
        builders = {
            "faithful_eager": lambda: (lambda: angular(f0, amp, dist, T)),
            "faithful_eager_graph": lambda: graphed(lambda: angular(f0, amp, dist, T)),
            "faithful_angular_compile": lambda: (lambda c=torch.compile(angular, dynamic=False): c(f0, amp, dist, T)),
            "faithful_angular_compile_graphs": lambda: stepped(
                lambda c=torch.compile(angular, dynamic=False, mode="reduce-overhead"): c(f0, amp, dist, T)),
            "reform_compile": lambda: (lambda c=torch.compile(reform, dynamic=False): c(f0, amp, dist, T)),
            "reform_compile_graphs": lambda: stepped(
                lambda c=torch.compile(reform, dynamic=False, mode="reduce-overhead"): c(f0, amp, dist, T)),
            "fused": lambda: (lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR)),
            "fused_graph": lambda: graphed(lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR)),
            "fused_kernel_only": lambda: (lambda: _ext.fused_harmonic(f0c, ampc, distc, phi, T, float(SR), True, False, 256, False)),
        }
        for name, build in builders.items():
            torch._dynamo.reset()
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            r0 = torch.cuda.memory_reserved(); torch.cuda.reset_peak_memory_stats()
            try:
                fn = build()
                r = measure(name, fn, B, K, T, ref, active)
                r["ms_loop"] = round(loop_ms(fn), 4)
                r["reserved_MB"] = round((torch.cuda.memory_reserved() - r0) / 2**20, 1)
                rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:300]); sys.stdout.flush()
            fn = None
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print("wrote", OUT)
