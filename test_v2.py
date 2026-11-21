"""Correctness and stress test for the v2 kernel (fused_v2.py), with v1 (fused_ddsp.py) and the fp32 angular port alongside.
Reference: ddsp_port.harmonic_synthesis in fp64. Pass criterion (chosen before running): max_abs < 1e-5 and rel_l2 < 1e-5.
Cases: the four test_fused.py configs; harmonics exactly at Nyquist (f0 = 8000/m); very low / very high / above-Nyquist pitch;
random pitch jumps between frames; hop sizes that are not powers of two; another sample rate; block sizes; a 40 s clip.
Run:  python test_v2.py 2>&1 | tee test_v2_log.txt"""
import math
import torch
from ddsp_port import harmonic_synthesis, make_controls
from fused_ddsp import fused_harmonic_synthesis
from fused_v2 import fused_v2_synthesis

TOL = 1e-5


def controls_from_f0(f0, K, sr, seed=1):
    """f0 [B,F,1] -> (f0, amp, dist) with a 1/k spectrum, frame-level Nyquist removal and normalisation (as make_controls)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, F, _ = f0.shape
    amp = 0.2 + 0.8 * torch.rand(B, F, 1, generator=g)
    k = torch.arange(1, K + 1, dtype=torch.float32)[None, None, :]
    dist = (1.0 / k) * (0.5 + torch.rand(B, F, K, generator=g))
    dist = torch.where(f0 * k >= sr / 2.0, torch.zeros_like(dist), dist)
    dist = dist / dist.sum(-1, keepdim=True).clamp_min(1e-7)
    return f0.cuda().float(), amp.cuda(), dist.cuda()


def check(label, f0, amp, dist, T, sr, impls):
    B = f0.shape[0]
    with torch.no_grad():
        ref = torch.cat([harmonic_synthesis(f0[i:i+1].double(), amp[i:i+1].double(), dist[i:i+1].double(),
                                            T, sr, "window", False) for i in range(B)])
        norm = ref.norm().item()
        res = []
        for name, fn in impls.items():
            o = fn(f0, amp, dist, T, sr).double()
            mx = (o - ref).abs().max().item()
            rel = (o - ref).norm().item() / norm if norm > 0 else float("nan")
            ok = mx < TOL and (rel < TOL or norm == 0)
            res.append(ok)
            print(f"  {label:44s} {name:16s} max_abs={mx:.3e} rel_l2={rel:.3e} {'PASS' if ok else 'FAIL'}")
    return res


IMPLS = {
    "v2": lambda f0, a, d, T, sr: fused_v2_synthesis(f0, a, d, T, sr),
    "v2 block-reduce": lambda f0, a, d, T, sr: fused_v2_synthesis(f0, a, d, T, sr, frame_scan="block"),
    "v2 fast-sin": lambda f0, a, d, T, sr: fused_v2_synthesis(f0, a, d, T, sr, fast_sin=True),
    "v1": lambda f0, a, d, T, sr: fused_harmonic_synthesis(f0, a, d, T, sr),
    "angular fp32": lambda f0, a, d, T, sr: harmonic_synthesis(f0, a, d, T, sr, "window", True),
}

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    summary = {n: [] for n in IMPLS}

    def run(label, f0, amp, dist, T, sr, impls=IMPLS):
        for n, ok in zip(impls, check(label, f0, amp, dist, T, sr, impls)):
            summary[n].append((label, ok))

    print("--- test_fused.py configs (make_controls, seed 0)")
    for B, K, T in [(2, 100, 1024), (2, 100, 16000), (1, 128, 64000), (8, 100, 64000)]:
        run(f"make_controls {(B, K, T)}", *make_controls(B, K, T // 64, 16000), T, 16000)

    print("--- harmonic exactly at Nyquist: constant f0 = 8000/m, m = 10..40")
    m = torch.arange(10, 41, dtype=torch.float32)
    run("f0 = 8000/m (31 items)", *controls_from_f0((8000.0 / m)[:, None, None].expand(-1, 250, 1).contiguous(), 100, 16000),
        16000, 16000)

    print("--- pitch extremes")
    for f in (20.0, 55.0, 3999.0, 4000.0, 7999.0, 8000.0, 9000.0):
        run(f"constant f0 = {f} Hz", *controls_from_f0(torch.full((1, 250, 1), f), 100, 16000), 16000, 16000)

    print("--- random pitch jumps every frame, 50-2000 Hz")
    g = torch.Generator().manual_seed(2)
    run("random jumps (4 items)", *controls_from_f0(50 + 1950 * torch.rand(4, 1000, 1, generator=g), 100, 16000), 64000, 16000)

    print("--- hop sizes that are not powers of two (make_controls pitch)")
    for hop in (100, 160, 250):
        F = 64000 // hop if 64000 % hop == 0 else 100
        T = F * hop
        run(f"hop {hop}, T {T}", *make_controls(2, 100, F, 16000), T, 16000)

    print("--- other sample rate: 44.1 kHz, hop 441")
    run("sr 44100, hop 441", *make_controls(2, 100, 100, 44100), 44100, 44100)

    print("--- block sizes (v2 only)")
    f0, amp, dist = make_controls(8, 100, 1000, 16000)
    blk = {f"v2 block {b}": (lambda f0, a, d, T, sr, b=b: fused_v2_synthesis(f0, a, d, T, sr, block=b)) for b in (32, 96, 512, 1024)}
    blk.update({f"v2 blockred {b}": (lambda f0, a, d, T, sr, b=b: fused_v2_synthesis(f0, a, d, T, sr, block=b, frame_scan="block"))
                for b in (32, 96, 512, 1024)})
    summary.update({n: [] for n in blk})
    run("make_controls (8,100,64000)", f0, amp, dist, 64000, 16000, blk)

    print("--- 40 s clip (1,100,640000)")
    run("40 s clip", *make_controls(1, 100, 10000, 16000), 640000, 16000)

    print("--- summary")
    for n, r in summary.items():
        fails = [l for l, ok in r if not ok]
        print(f"  {n:16s} {len(r) - len(fails)}/{len(r)} passed" + (f"; failed: {fails}" if fails else ""))
