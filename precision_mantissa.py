"""Bit-width sweep (software emulation, accuracy only): how many significant bits does each part of the synthesizer need?
A float format with p significant bits (hidden bit included) is emulated by rounding an fp64 value to p bits
(round half to even); the exponent range is assumed large enough. p: bf16 = 8, fp16 = 11, AMD fp24 = 17, fp32 = 24,
a 40-bit float with 8 exponent bits = 32, fp64 = 53. Fixed point with b fractional bits is also swept for the phase.
Each row rounds only the named part; everything else stays fp64:
  phase   base phase x and the product k*x (frac() of a p-bit number is exact)
  amp     the upsampled amplitude a_k(t)
  sin     the sine output
  accum   the running sum over harmonics (rounded after every addition)
  all     all four together
Config (8,100,64000), make_controls seed 0. SNR_dB = -20*log10(rel_l2) vs the fp64 reference (ddsp_port).
Run:  python precision_mantissa.py 2>&1 | tee precision_mantissa_log.txt        (writes precision_mantissa_results.csv)"""
import csv, math
import torch
from ddsp_port import harmonic_synthesis, make_controls, upsample_with_windows, resample_linear
from precision_sweep import base_phase_cycles, SR, HOP, B, K, T

P_BITS = [8, 11, 12, 14, 16, 17, 18, 20, 22, 24, 26, 28, 32, 40, 48, 53]
FIX_BITS = list(range(8, 34, 2))
PARTS = ["phase", "amp", "sin", "accum", "all"]


def rnd(x, p):
    """Round fp64 tensor x to p significant bits."""
    if p >= 53:
        return x
    m, e = torch.frexp(x)                         # x = m * 2^e, |m| in [0.5, 1)
    return torch.ldexp(torch.round(m * 2.0 ** p) / 2.0 ** p, e)


def synth(x, amp_env, kk, part, p):
    r = (lambda v: rnd(v, p))
    q = (lambda v, name: r(v) if part in (name, "all") else v)
    xx = q(x, "phase")
    ph = q(xx[:, None] * kk[None, :], "phase")
    ph = ph - torch.floor(ph)
    s = q(torch.sin(2 * math.pi * ph), "sin")
    a = q(amp_env, "amp")
    terms = a * s
    if part in ("accum", "all"):
        acc = torch.zeros_like(x)
        for k in range(terms.shape[1]):
            acc = r(acc + terms[:, k])
        return acc
    return terms.sum(-1)


def synth_fixed(x, amp_env, kk, b):
    s = 2.0 ** b
    p = torch.remainder(torch.floor(x * s)[:, None] * kk[None, :], s) / s
    return (amp_env * torch.sin(2 * math.pi * p)).sum(-1)


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    f0, amp, dist = make_controls(B, K, T // HOP, SR)
    kk = torch.arange(1, K + 1, device=f0.device, dtype=torch.float64)
    variants = [("float", part, p) for part in PARTS for p in P_BITS] + [("fixed", "phase", b) for b in FIX_BITS]
    err = {v: [0.0, 0.0, 0.0] for v in variants}
    with torch.no_grad():
        for i in range(B):
            fi, ai, di = f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double()
            ref = harmonic_synthesis(fi, ai, di, T, SR, "window", False)[0]
            freq_env = resample_linear(fi * kk, T)[0]
            amp_env = upsample_with_windows(ai * di, T)[0]
            amp_env = torch.where(freq_env >= SR / 2.0, torch.zeros_like(amp_env), amp_env)
            x = base_phase_cycles(f0[i:i+1])
            for v in variants:
                kind, part, p = v
                y = synth(x, amp_env, kk, part, p) if kind == "float" else synth_fixed(x, amp_env, kk, p)
                e = err[v]
                e[0] += (y - ref).pow(2).sum().item(); e[1] += ref.pow(2).sum().item()
                e[2] = max(e[2], (y - ref).abs().max().item())
            print("item", i, "done", flush=True)
    rows = []
    for kind, part, p in variants:
        s, r, m = err[(kind, part, p)]
        rel = math.sqrt(s / r)
        rows.append(dict(format=kind, part=part, bits=p, max_abs_err_vs_fp64=f"{m:.3e}", rel_l2_err=f"{rel:.3e}",
                         snr_dB=f"{-20 * math.log10(rel):.1f}" if rel > 0 else "inf"))
        print(rows[-1])
    with open("precision_mantissa_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote precision_mantissa_results.csv")
