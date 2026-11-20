"""Precision study (software emulation in PyTorch, not a CUDA kernel): how many bits does the harmonic synthesizer need?
The phase is stored as fractional cycles x in [0,1), so it can be held in fixed point (b bits) or in a float format.
Each row changes only the stated part; everything else stays fp64. Error is measured against the fp64 reference
(ddsp_port.harmonic_synthesis), and SNR_dB = -20*log10(rel_l2).
Config: (B,K,T) = (8,100,64000), make_controls seed 0, f0 100-400 Hz.
Run:  python precision_sweep.py 2>&1 | tee precision_sweep_log.txt        (writes precision_sweep_results.csv)"""
import csv, math
import torch
from ddsp_port import harmonic_synthesis, make_controls, upsample_with_windows, resample_linear
from fused_ddsp import frame_phase_offsets

SR, HOP = 16000, 64
B, K, T = 8, 100, 64000
DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def base_phase_cycles(f0_item):
    """Exact base phase in cycles, fp64, [T] (same closed form as the fused kernel)."""
    a = f0_item[0, :, 0].double()
    F = a.shape[0]
    a1 = torch.cat([a[1:], a[-1:]])
    d = a1 - a
    phi = frame_phase_offsets(f0_item, T, SR)[0]
    t = torch.arange(T, device=a.device)
    j, n = t // HOP, (t % HOP).double()
    c = 2 * math.pi / SR
    x = (phi[j] + c * ((n + 1) * a[j] + d[j] * n * (n + 1) / (2.0 * HOP))) / (2 * math.pi)
    return x - torch.floor(x)


def synth(x, amp_env, kk, mode, arg):
    """x [T] fp64 cycles, amp_env [T,K] fp64 (Nyquist-masked), kk [K]. Returns [T] fp64."""
    if mode == "fixed":                      # b-bit fixed-point phase accumulator; k*x mod 1 is exact in integer arithmetic
        s = 2.0 ** arg
        xq = torch.floor(x * s)
        p = torch.remainder(xq[:, None] * kk[None, :], s) / s
        return (amp_env * torch.sin(2 * math.pi * p)).sum(-1)
    if mode == "phase":                      # phase and k*x mod 1 in the given float format, rest fp64
        xd = x.to(DT[arg])
        kd = kk.to(DT[arg])
        p = xd[:, None] * kd[None, :]
        p = (p - torch.floor(p)).double()
        return (amp_env * torch.sin(2 * math.pi * p)).sum(-1)
    if mode == "all":                        # phase, sin, amplitudes and accumulation all in the given format
        dt = DT[arg]
        xd, kd, ad = x.to(dt), kk.to(dt), amp_env.to(dt)
        p = xd[:, None] * kd[None, :]
        p = p - torch.floor(p)
        y = torch.sin(p * torch.tensor(2 * math.pi, dtype=dt, device=p.device)) * ad
        return y.sum(-1, dtype=dt).double()
    if mode == "fp64":
        return (amp_env * torch.sin(2 * math.pi * torch.remainder(x[:, None] * kk[None, :], 1.0))).sum(-1)
    raise ValueError(mode)


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    f0, amp, dist = make_controls(B, K, T // HOP, SR)
    kk = torch.arange(1, K + 1, device=f0.device, dtype=torch.float64)
    variants = [("fp64 closed form (sanity)", "fp64", None)]
    variants += [(f"fixed-point phase, {b} bits", "fixed", b) for b in (8, 10, 12, 16, 20, 24, 32)]
    variants += [(f"{f} phase, rest fp64", "phase", f) for f in ("fp32", "bf16", "fp16")]
    variants += [(f"{f} everywhere", "all", f) for f in ("fp32", "bf16", "fp16")]
    err = {v[0]: [0.0, 0.0, 0.0] for v in variants}           # sum sq diff, sum sq ref, max abs
    with torch.no_grad():
        for i in range(B):
            fi, ai, di = f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double()
            ref = harmonic_synthesis(fi, ai, di, T, SR, "window", False)[0]
            freq_env = resample_linear(fi * kk, T)[0]
            amp_env = upsample_with_windows(ai * di, T)[0]
            amp_env = torch.where(freq_env >= SR / 2.0, torch.zeros_like(amp_env), amp_env)
            x = base_phase_cycles(f0[i:i+1])
            for name, mode, arg in variants:
                y = synth(x, amp_env, kk, mode, arg)
                e = err[name]
                e[0] += (y - ref).pow(2).sum().item(); e[1] += ref.pow(2).sum().item()
                e[2] = max(e[2], (y - ref).abs().max().item())
    rows = []
    for name, mode, arg in variants:
        s, r, m = err[name]
        rel = math.sqrt(s / r)
        rows.append(dict(variant=name, max_abs_err_vs_fp64=f"{m:.3e}", rel_l2_err=f"{rel:.3e}",
                         snr_dB=f"{-20 * math.log10(rel):.1f}" if rel > 0 else "inf"))
        print(rows[-1])
    with open("precision_sweep_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote precision_sweep_results.csv")
