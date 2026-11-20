"""Ablation: cost of the fp64 per-sample setup (phase + Nyquist cutoff) on this GPU.
fused (fp64 setup, default) vs fused_fp32setup (same kernel, setup in fp32), full call and kernel-only, all 6 configs.
Run:  python ablate_fp64.py 2>&1 | tee ablate_fp64_log.txt        (writes ablate_fp64_results.csv)"""
import csv, sys
import torch
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis, frame_phase_offsets, _ext
from bench_fused import measure, SR, HOP

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
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
            "fused_fp32setup": lambda: fused_harmonic_synthesis(f0, amp, dist, T, SR, fp32_setup=True),
            "fused_kernel_only": lambda: _ext.fused_harmonic(f0c, ampc, distc, phi, T, float(SR), True, False, 256, False),
            "fused_kernel_only_fp32setup": lambda: _ext.fused_harmonic(f0c, ampc, distc, phi, T, float(SR), True, False, 256, True),
        }
        for name, fn in variants.items():
            try:
                r = measure(name, fn, B, K, T, ref, active); rows.append(r); print(r); sys.stdout.flush()
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:300])
    with open("ablate_fp64_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote ablate_fp64_results.csv")
