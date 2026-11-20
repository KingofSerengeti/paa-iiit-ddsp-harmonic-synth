import math, numpy as np
SR, HOP = 16000, 64

def controls(B, K, nf, seed=0):
    rng = np.random.default_rng(seed)
    base = 100 + 300 * rng.random((B, 1, 1)); phi = 2*np.pi*rng.random((B, 1, 1))
    fr = np.arange(nf, dtype=np.float32)[None, :, None]
    f0 = base * (1 + 0.02*np.sin(2*np.pi*5*fr/250 + phi))
    amp = 0.6 + 0.4*np.sin(2*np.pi*1.3*fr/250 + phi)
    k = np.arange(1, K+1, dtype=np.float32)[None, None, :]
    dist = (1/k) * (1 + 0.3*np.cos(2*np.pi*(0.7*fr/250 + 0.13*k)))
    dist = np.where(f0*k >= SR/2, 0, dist); dist = dist/np.maximum(dist.sum(-1, keepdims=True), 1e-7)
    return f0, amp, dist

# ---- numpy copy of ddsp_port (fp64 reference): per-harmonic cumsum, linear f0 resample, Hann amp upsample
def resample_linear(x, T):
    B, Fr, C = x.shape
    src = np.arange(T, dtype=np.float64) * (Fr / T)
    i0 = np.minimum(np.floor(src).astype(int), Fr-1); i1 = np.minimum(i0+1, Fr-1)
    fr = (src - i0)[None, :, None]
    return x[:, i0]*(1-fr) + x[:, i1]*fr

def upsample_windows(x, T):
    B, Fr, C = x.shape; hop = T // Fr
    xp = np.concatenate([x, x[:, -1:]], 1)
    n = np.arange(hop); w = 0.5 - 0.5*np.cos(np.pi*n/hop)
    a = xp[:, :-1, None, :]; b = xp[:, 1:, None, :]
    return (a*(1-w)[None, None, :, None] + b*w[None, None, :, None]).reshape(B, T, C)

def reference(f0, amp, dist, T):
    f0, amp, dist = f0.astype(np.float64), amp.astype(np.float64), dist.astype(np.float64)
    K = dist.shape[-1]
    hf = f0 * np.arange(1, K+1); ha = amp * dist
    fe = resample_linear(hf, T); ae = upsample_windows(ha, T)
    ae = np.where(fe >= SR/2, 0, ae)
    ph = np.cumsum(fe * 2*np.pi / SR, axis=1)
    return (ae*np.sin(ph)).sum(-1)

# ---- the kernel algorithm (fp32 per-harmonic part, fp64 frame scan)
def fused(f0, amp, dist, T, fp32_naive_cumsum=False):
    B, F, K = dist.shape; hop = T // F
    a = f0[:, :, 0].astype(np.float64)
    a1 = np.concatenate([a[:, 1:], a[:, -1:]], 1); d = a1 - a
    c = 2*np.pi/SR
    S = c*(hop*a + d*(hop-1)/2)
    phi = np.mod(np.cumsum(S, 1) - S, 2*np.pi)
    t = np.arange(T); j = t//hop; n = t % hop; j1 = np.minimum(j+1, F-1)
    A, A1 = a[:, j], a[:, j1]; dd = A1 - A
    x = (phi[:, j] + c*((n+1)*A + dd*n*(n+1)/(2*hop)))/(2*np.pi)
    xf = (x - np.floor(x)).astype(np.float32)
    freq = (A + dd*n/hop).astype(np.float32)
    w = (0.5 - 0.5*np.cos(np.pi*n/hop)).astype(np.float32)[None, :, None]
    ha = (amp*dist).astype(np.float32)
    ak = ha[:, j]*(1-w) + ha[:, j1]*w
    k = np.arange(1, K+1, dtype=np.float32)[None, None, :]
    p = k*xf[:, :, None]; p = p - np.floor(p)
    s = np.sin((2*np.pi*p).astype(np.float32)).astype(np.float32)
    ak = np.where(k*freq[:, :, None] >= SR/2, np.float32(0), ak)
    return (ak*s).sum(-1)

def naive_fp32(f0, amp, dist, T):      # plain fp32 per-harmonic cumsum (what eager does)
    f0, amp, dist = f0.astype(np.float32), amp.astype(np.float32), dist.astype(np.float32)
    K = dist.shape[-1]
    fe = resample_linear(f0*np.arange(1, K+1, dtype=np.float32), T).astype(np.float32)
    ae = upsample_windows(amp*dist, T).astype(np.float32)
    ae = np.where(fe >= SR/2, np.float32(0), ae)
    ph = np.cumsum(fe*np.float32(2*np.pi/SR), axis=1, dtype=np.float32)
    return (ae*np.sin(ph)).sum(-1)

for (B, K, T) in [(2, 100, 16000), (1, 128, 64000), (2, 64, 1024)]:
    f0, amp, dist = controls(B, K, T//HOP)
    ref = reference(f0, amp, dist, T)
    rel = lambda o: np.linalg.norm(o.astype(np.float64)-ref)/np.linalg.norm(ref)
    mx = lambda o: np.abs(o.astype(np.float64)-ref).max()
    o = fused(f0, amp, dist, T); nv = naive_fp32(f0, amp, dist, T)
    print(f"B={B} K={K} T={T}: fused rel_l2={rel(o):.2e} max={mx(o):.2e} | naive fp32 rel_l2={rel(nv):.2e}")
