"""
Baseline benchmark for the DDSP harmonic synthesizer (Evaluation 1).
Measures: latency (CUDA events), peak GPU memory, throughput, energy/call
(NVML total-energy counter + CLEAR-style repetition amplification), and
accuracy against an fp64 reference.

Run:   python baseline_ddsp.py                # full sweep -> baseline_results.csv
       python baseline_ddsp.py --profile      # one config, few iters (for nsys/ncu)
Needs: torch (CUDA build with sm_120 support), pynvml (pip install nvidia-ml-py)
"""
import argparse, csv, math, statistics, time
import torch
import torch.nn.functional as F

try:
    import pynvml
    pynvml.nvmlInit()
    NVML = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:  # energy is optional
    NVML = None
    print("NVML energy unavailable:", e)

SR = 16000          # sample rate (DDSP default). Try 48000 later as an extra sweep.
HOP = 64            # samples per control frame (250 frames/s at 16 kHz)


def harmonic_synth(f0_frames, amp_frames, T, sr=SR):
    """Reference DDSP-style harmonic synth, written the way DDSP does it:
    separate tensor ops, each materialising a full [B,K,T] or [B,T] tensor.
    f0_frames: [B,F] Hz   amp_frames: [B,F,K]   returns [B,T]"""
    K = amp_frames.shape[-1]
    f0 = F.interpolate(f0_frames[:, None, :], size=T, mode="linear",
                       align_corners=False)[:, 0]                      # [B,T]
    amps = F.interpolate(amp_frames.permute(0, 2, 1), size=T, mode="linear",
                         align_corners=False)                          # [B,K,T]
    omega = 2 * math.pi * f0 / sr
    phase = torch.cumsum(omega, dim=1)                                 # [B,T]
    k = torch.arange(1, K + 1, device=f0.device, dtype=f0.dtype)[None, :, None]
    mask = (f0[:, None, :] * k < sr / 2).to(f0.dtype)                  # anti-alias
    return (amps * mask * torch.sin(phase[:, None, :] * k)).sum(dim=1)


def make_inputs(B, K, T, dtype=torch.float32, device="cuda"):
    g = torch.Generator(device=device).manual_seed(0)
    Fr = T // HOP + 1
    f0 = 100 + 300 * torch.rand(B, Fr, device=device, generator=g)    # 100-400 Hz
    amp = torch.rand(B, Fr, K, device=device, generator=g) / K
    return f0.to(dtype), amp.to(dtype)


def time_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), statistics.pstdev(ts)


def energy_mj(fn, ms_per_call, trials=5):
    """Energy per call in mJ. NVML updates ~every 20-50 ms, so run the call
    back-to-back N times so the total lasts >= ~1 s (the CLEAR idea)."""
    if NVML is None:
        return float("nan"), float("nan")
    n = max(10, int(1000 / max(ms_per_call, 1e-3)))
    vals = []
    for _ in range(trials):
        torch.cuda.synchronize()
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(NVML)
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(NVML)
        vals.append((e1 - e0) / n)
        time.sleep(0.5)
    return statistics.mean(vals), statistics.pstdev(vals)


def run_config(name, impl, B, K, T):
    f0, amp = make_inputs(B, K, T)
    ref = harmonic_synth(f0.double(), amp.double(), T)                 # fp64 truth
    with torch.no_grad():
        out = impl(f0, amp, T)
        torch.cuda.synchronize()
        err = (out.double() - ref).abs().max().item()
        rel = ((out.double() - ref).norm() / ref.norm()).item()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        fn = lambda: impl(f0, amp, T)
        ms, sd = time_ms(fn)
        peak_mb = (torch.cuda.max_memory_allocated() - base_mem) / 2**20
        emj, esd = energy_mj(fn, ms)
    return dict(impl=name, B=B, K=K, T=T, ms_median=round(ms, 4), ms_std=round(sd, 4),
                samples_per_s=round(B * T / (ms / 1000)), peak_extra_MB=round(peak_mb, 1),
                energy_mJ=round(emj, 3), energy_std=round(esd, 3),
                max_abs_err_vs_fp64=f"{err:.3e}", rel_l2_err=f"{rel:.3e}",
                uJ_per_1k_samples=round(1000 * emj / (B * T) * 1000, 4) if emj == emj else "nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__,
          "| cuda", torch.version.cuda, "| cc", torch.cuda.get_device_capability(0))

    impls = {"eager": harmonic_synth}
    try:
        impls["torch_compile"] = torch.compile(harmonic_synth, dynamic=False)
    except Exception as e:
        print("torch.compile unavailable:", e)

    if a.profile:   # one config, few iterations -> for nsys / ncu
        f0, amp = make_inputs(8, 100, 64000)
        with torch.no_grad():
            for _ in range(5):
                harmonic_synth(f0, amp, 64000)
        torch.cuda.synchronize()
        raise SystemExit

    rows = []
    # batch x harmonics x samples(=0.5 s, 1 s, 4 s at 16 kHz); B=1,T=1024 ~ streaming buffer
    configs = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000),
               (8, 128, 64000), (32, 100, 64000)]
    for B, K, T in configs:
        for name, impl in impls.items():
            try:
                r = run_config(name, impl, B, K, T)
                rows.append(r); print(r)
            except Exception as e:
                print("FAILED", name, B, K, T, "->", repr(e)[:200])
    if rows:
        with open("baseline_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print("wrote baseline_results.csv")
