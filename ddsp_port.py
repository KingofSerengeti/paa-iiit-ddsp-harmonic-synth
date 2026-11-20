"""
Faithful PyTorch port of magenta/ddsp  core.harmonic_synthesis  (the operator behind synths.Harmonic.get_signal).
Mirrors the original structure on purpose (per-harmonic [B,T,K] envelopes, per-harmonic cumsum), so it is a fair baseline:

  harmonic_frequencies = f0 * k                      [B,F,K]     (frame rate)
  harmonic_amplitudes  = amplitude * distribution    [B,F,K]
  freq_env = resample(harmonic_frequencies, T)       linear (TF1 bilinear, align_corners=False)
  amp_env  = resample(harmonic_amplitudes,  T)       'window' = Hann overlap-add upsampling
  oscillator_bank: zero amps with freq >= Nyquist; omega = f*2pi/sr; phase = cumsum(omega, time) (or angular_cumsum);
                   audio = sum_k amp_env * sin(phase)

Deliberate difference: upsample_with_windows is written in its closed form (verified numerically equal to the
overlap-add, max diff ~4e-16 in fp64). It is cheaper than TF's windowed tensor, so this baseline is slightly
*kinder* to DDSP, not unfairly slow.
Validate against the real TensorFlow code with validate_vs_tf.py.
"""
import math
import torch
import torch.nn.functional as F


def get_harmonic_frequencies(f0, n_harmonics):             # f0 [B,F,1] -> [B,F,K]
    k = torch.arange(1, n_harmonics + 1, device=f0.device, dtype=f0.dtype)
    return f0 * k


def resample_linear(x, n_samples):
    """tf.compat.v1.image.resize BILINEAR, align_corners=False: src = t * (n_frames/n_samples), last frame held."""
    B, Fr, C = x.shape
    src = torch.arange(n_samples, device=x.device, dtype=torch.float64) * (Fr / n_samples)
    i0 = src.floor().long().clamp(max=Fr - 1)
    i1 = (i0 + 1).clamp(max=Fr - 1)
    frac = (src - i0.to(src.dtype)).to(x.dtype)[None, :, None]
    return x[:, i0] * (1 - frac) + x[:, i1] * frac           # [B,T,C]


def upsample_with_windows(x, n_samples):
    """Hann overlap-add upsampling (ddsp 'window'), closed form. Requires n_samples % n_frames == 0."""
    B, Fr, C = x.shape
    assert n_samples % Fr == 0, "n_samples must be divisible by n_frames for 'window' resampling"
    hop = n_samples // Fr
    xp = torch.cat([x, x[:, -1:]], dim=1)                    # hold last frame (add_endpoint=True)
    n = torch.arange(hop, device=x.device, dtype=x.dtype)
    w = 0.5 - 0.5 * torch.cos(math.pi * n / hop)             # rising half of periodic Hann(2*hop)
    a = xp[:, :-1, None, :]
    b = xp[:, 1:, None, :]
    out = a * (1 - w)[None, None, :, None] + b * w[None, None, :, None]   # [B,F,hop,C]
    return out.reshape(B, n_samples, C)


def angular_cumsum(omega, chunk_size=1000):
    """Chunked cumsum with mod-2pi offsets (ddsp.core.angular_cumsum). Accurate but slower."""
    B, T, K = omega.shape
    rem = T % chunk_size
    if rem:
        omega = F.pad(omega, (0, 0, 0, chunk_size - rem))
    L = omega.shape[1]
    n_chunks = L // chunk_size
    chunks = omega.reshape(B, n_chunks, chunk_size, K)
    phase = torch.cumsum(chunks, dim=2)
    two_pi = 2.0 * math.pi
    offsets = torch.remainder(phase[:, :, -1:, :], two_pi)             # [B,n_chunks,1,K]
    offsets = F.pad(offsets, (0, 0, 0, 0, 1, 0))[:, :-1]
    offsets = torch.remainder(torch.cumsum(offsets, dim=1), two_pi)
    phase = torch.remainder(phase + offsets, two_pi)
    return phase.reshape(B, L, K)[:, :T]


def oscillator_bank(freq_env, amp_env, sample_rate=16000, use_angular_cumsum=False):
    amp_env = torch.where(freq_env >= sample_rate / 2.0, torch.zeros_like(amp_env), amp_env)
    omegas = freq_env * (2.0 * math.pi)
    omegas = omegas / float(sample_rate)
    phases = angular_cumsum(omegas) if use_angular_cumsum else torch.cumsum(omegas, dim=1)
    return (amp_env * torch.sin(phases)).sum(dim=-1)                   # [B,T]


def harmonic_synthesis(f0, amplitudes, harmonic_distribution, n_samples=64000, sample_rate=16000,
                       amp_resample_method="window", use_angular_cumsum=False):
    """f0 [B,F,1] Hz, amplitudes [B,F,1], harmonic_distribution [B,F,K] (already scaled/normalised, as after get_controls)."""
    K = harmonic_distribution.shape[-1]
    harm_freqs = get_harmonic_frequencies(f0, K)
    harm_amps = amplitudes * harmonic_distribution
    freq_env = resample_linear(harm_freqs, n_samples)
    if amp_resample_method == "window":
        amp_env = upsample_with_windows(harm_amps, n_samples)
    else:
        amp_env = resample_linear(harm_amps, n_samples)
    return oscillator_bank(freq_env, amp_env, sample_rate, use_angular_cumsum)


def make_controls(B, K, n_frames, sample_rate=16000, device="cuda", dtype=torch.float32,
                  f0_range=(100.0, 400.0), seed=0):
    """Smooth, instrument-like controls (vibrato on a steady pitch, 1/k spectrum), then the same
    frame-level Nyquist removal + normalisation that synths.Harmonic.get_controls applies."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = f0_range[0] + (f0_range[1] - f0_range[0]) * torch.rand(B, 1, 1, generator=g)
    fr = torch.arange(n_frames, dtype=torch.float32)[None, :, None]
    phi = 2 * math.pi * torch.rand(B, 1, 1, generator=g)
    f0 = base * (1.0 + 0.02 * torch.sin(2 * math.pi * 5.0 * fr / 250.0 + phi))          # 5 Hz vibrato
    amp = 0.6 + 0.4 * torch.sin(2 * math.pi * 1.3 * fr / 250.0 + phi)
    k = torch.arange(1, K + 1, dtype=torch.float32)[None, None, :]
    dist = (1.0 / k) * (1.0 + 0.3 * torch.cos(2 * math.pi * (0.7 * fr / 250.0 + 0.13 * k)))
    freqs = f0 * k
    dist = torch.where(freqs >= sample_rate / 2.0, torch.zeros_like(dist), dist)       # remove above Nyquist
    dist = dist / dist.sum(-1, keepdim=True).clamp_min(1e-7)                            # normalise
    return f0.to(device, dtype), amp.to(device, dtype), dist.to(device, dtype)
