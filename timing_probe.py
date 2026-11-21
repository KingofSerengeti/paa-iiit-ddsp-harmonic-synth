"""Timing probe for small configs. The per-call CUDA-event time of very short calls varied between scripts (e.g. v1 kernel-only
at (1,100,1024): 0.035-0.043 ms in bench_fused.py runs, 0.0116 ms in graphs_bench.py). This probe measures one call three ways:
  event_ms  CUDA events around one call, median of 200 (as bench_fused.py; includes CPU launch time when the GPU waits)
  loop_ms   200 back-to-back calls / 200 (CPU can queue ahead)
  gpu_ms    100 calls captured in one CUDA graph, replayed, / 100 (GPU time without CPU launch cost)
and repeats each measurement after 1 s of GPU idle ("cold") and directly after 1 s of heavy GPU work ("warm"),
5 rounds each, to test whether the GPU clock state explains the variation.
Run:  python timing_probe.py 2>&1 | tee timing_probe_log.txt        (writes timing_probe_results.csv)"""
import csv, statistics, sys, time
import torch
from ddsp_port import make_controls
from fused_ddsp import fused_harmonic_synthesis, frame_phase_offsets, _ext
from fused_v2 import fused_v2_synthesis
from bench_fused import SR, HOP


def event_ms(fn, n=200):
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def loop_ms(fn, n=200):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / n


def gpu_ms(fn, n=100):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); g.replay(); e.record(); e.synchronize()
    return s.elapsed_time(e) / n


def heat(seconds=1.0):
    x = torch.randn(4096, 4096, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        x = x @ x; x = x / x.norm()
    torch.cuda.synchronize()


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    rows = []
    for B, K, T in [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000)]:
        f0, amp, dist = make_controls(B, K, T // HOP, SR)
        f0c, ampc, distc = f0[:, :, 0].contiguous(), amp[:, :, 0].contiguous(), dist.contiguous()
        phi = frame_phase_offsets(f0, T, SR)
        impls = {
            "v1": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR),
            "v1_kernel_only": lambda: _ext.fused_harmonic(f0c, ampc, distc, phi, T, float(SR), True, False, 256, False),
            "v2": lambda: fused_v2_synthesis(f0, amp, dist, T, SR),
        }
        with torch.no_grad():
            for state in ("cold", "warm"):
                for rnd in range(5):
                    for name, fn in impls.items():
                        if state == "cold":
                            torch.cuda.synchronize(); time.sleep(1.0)
                        else:
                            heat(1.0)
                        r = dict(impl=name, B=B, K=K, T=T, state=state, round=rnd, event_ms=round(event_ms(fn), 4),
                                 loop_ms=round(loop_ms(fn), 4), gpu_ms=round(gpu_ms(fn), 4))
                        rows.append(r); print(r); sys.stdout.flush()
    with open("timing_probe_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote timing_probe_results.csv")
