"""Speed + accuracy of the number-format variants in fused_precision.py (real CUDA kernels, not emulation).
Rows: <mode> = full call (per-frame tables built in PyTorch + kernel); <mode>_kernel = kernel launch only (tables precomputed).
Also fast-sine (__sinf) versions of f64 and i40_u32. Same harness and fp64 reference as bench_fused.py; snr_db = -20*log10(rel_l2).
Run:  python precision_bench.py 2>&1 | tee precision_bench_log.txt        (writes precision_kernels_results.csv)"""
import csv, math, sys
import torch
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis
from fused_precision import fused_precision_synthesis, precision_inputs, MODES, _ext
from bench_fused import measure, SR, HOP

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "precision_kernels_results.csv"

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    configs = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000), (8, 128, 64000), (16, 100, 64000)]
    rows = []
    for B, K, T in configs:
        f0, amp, dist = make_controls(B, K, T // HOP, SR)
        with torch.no_grad():
            ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                                T, SR, "window", False) for i in range(B)])
            same = torch.equal(fused_precision_synthesis(f0, amp, dist, T, SR, "f64"), fused_harmonic_synthesis(f0, amp, dist, T, SR))
        print(f"config {(B, K, T)}: mode f64 bit-identical to fused_ddsp default: {same}")
        active = (f0 * torch.arange(1, K + 1, device=f0.device) < SR / 2).float().mean().item()
        variants = {}
        for name, m in MODES.items():
            args = precision_inputs(f0, amp, dist, T, SR, m)
            variants[name] = (lambda name=name: fused_precision_synthesis(f0, amp, dist, T, SR, name))
            variants[name + "_kernel"] = (lambda args=args, m=m: _ext.fused_precision(*args, m, False))
        for name in ("f64", "i40_u32"):
            args = precision_inputs(f0, amp, dist, T, SR, MODES[name])
            variants[name + "_fastsin_kernel"] = (lambda args=args, m=MODES[name]: _ext.fused_precision(*args, m, True))
        for name, fn in variants.items():
            try:
                r = measure(name, fn, B, K, T, ref, active)
                r["snr_db"] = round(-20 * math.log10(float(r["rel_l2_err"])), 1)
                rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:300])
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote", OUT)
