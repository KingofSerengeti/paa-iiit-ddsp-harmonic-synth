"""Figures for RESULTS.md and the slides, drawn only from the CSVs in this folder. Writes figures/*.png.
Run:  python make_figures.py"""
import csv, os, statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

os.makedirs("figures", exist_ok=True)
# fixed colour per implementation (categorical order, never cycled)
C = {"v2": "#2a78d6", "v1": "#eb6834", "compile": "#1baf7a", "eager": "#eda100", "v2_fastsin": "#e87ba4",
     "v2_blockreduce": "#008300", "ink": "#2b2b2b", "muted": "#6b6b6b", "grid": "#e4e4e4"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"], "xtick.color": C["muted"],
                     "ytick.color": C["muted"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.frameon": False})


def rows(p):
    return list(csv.DictReader(open(p)))


def v2med(impl, cfg, key="ms_median"):
    vals = [float(r[key]) for f in ("v2_results.csv", "v2_results_run2.csv", "v2_results_run3.csv") for r in rows(f)
            if r["impl"] == impl and (r["B"], r["K"], r["T"]) == cfg]
    return statistics.median(vals)


def gr(impl, cfg, key="ms_median"):
    for r in rows("graphs_results.csv"):
        if r["impl"] == impl and (r["B"], r["K"], r["T"]) == cfg:
            return float(r[key])


# 1. CUDA graphs: with / without, two configs
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, cfg, title in [(axes[0], ("1", "100", "1024"), "(B,K,T) = (1,100,1024)"), (axes[1], ("8", "100", "64000"), "(B,K,T) = (8,100,64000)")]:
    items = [("DDSP port, eager", gr("faithful_eager", cfg), gr("faithful_eager_graph", cfg), "eager"),
             ("torch.compile", v2med("faithful_angular_compile", cfg), v2med("faithful_angular_compile_graphs", cfg), "compile"),
             ("fused v1", v2med("v1", cfg), v2med("v1_graph", cfg), "v1"),
             ("fused v2", v2med("v2", cfg), v2med("v2_graph", cfg), "v2")]
    y = range(len(items))
    for i, (lab, a, g, key) in enumerate(items):
        ax.barh(i + 0.2, a, height=0.36, color=C[key])
        ax.barh(i - 0.2, g, height=0.36, color=C[key], alpha=0.45, hatch="//", edgecolor="white")
        ax.text(a * 1.08, i + 0.2, f"{a:.4f}", va="center", fontsize=9, color=C["ink"])
        ax.text(g * 1.08, i - 0.2, f"{g:.4f} (graph)", va="center", fontsize=9, color=C["ink"])
    ax.set_yticks(list(y)); ax.set_yticklabels([i[0] for i in items])
    if ax is axes[0]: ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlabel("ms per call (log scale)"); ax.set_title(title, fontsize=11)
    ax.set_xlim(right=ax.get_xlim()[1] * 4)
fig.suptitle("CUDA graphs: solid = normal launch, hatched = captured in a CUDA graph", fontsize=12)
fig.savefig("figures/fig_cuda_graphs.png"); plt.close(fig)

# 2. Bit-width sweep (emulation)
pm = rows("precision_mantissa_results.csv")
fig, ax = plt.subplots(figsize=(8.5, 4.6))
styles = {"phase": ("phase (float)", C["v2"]), "accum": ("accumulation", C["v1"]), "amp": ("amplitude", C["compile"]),
          "sin": ("sine output", C["eager"])}
for part, (lab, col) in styles.items():
    pts = sorted((int(r["bits"]), float(r["snr_dB"])) for r in pm if r["format"] == "float" and r["part"] == part)
    ax.plot(*zip(*pts), color=col, lw=2, marker="o", ms=4, label=lab)
pts = sorted((int(r["bits"]), float(r["snr_dB"])) for r in pm if r["format"] == "fixed")
ax.plot(*zip(*pts), color=C["v2_fastsin"], lw=2, ls="--", marker="s", ms=4, label="phase (fixed point, fractional bits)")
for b, name in [(8, "bf16"), (11, "fp16"), (24, "fp32"), (32, "40-bit float"), (53, "fp64")]:
    ax.axvline(b, color=C["muted"], lw=0.8, ls=":")
    ax.text(b, 203, name, fontsize=8, color=C["muted"], ha="center", va="bottom")
ax.set_xlabel("significant bits (float) / fractional bits (fixed point)"); ax.set_ylabel("SNR vs fp64 reference (dB)")
ax.set_title("Bits needed per stage (software emulation, (8,100,64000))", fontsize=11)
ax.set_ylim(15, 212)
ax.legend(loc="lower right", fontsize=9)
fig.savefig("figures/fig_bit_sweep.png"); plt.close(fig)

# 3. Number-format kernels: speed vs accuracy (kernel only, main config)
pk = [r for r in rows("precision_kernels_results.csv") if (r["B"], r["K"], r["T"]) == ("8", "100", "64000") and r["impl"].endswith("_kernel")]
names = {"f64_kernel": "fp64 setup (v1)", "f32_kernel": "fp32 setup", "f32tb_kernel": "fp32 + fp64 tie-break",
         "df_kernel": "double-float setup", "f64_u32_kernel": "fp64 setup + uint32 phase", "i40_u32_kernel": "int40 setup + uint32 phase",
         "f64_u64_kernel": "fp64 setup + uint64 phase", "f64_fastsin_kernel": "fp64 setup, fast sine",
         "i40_u32_fastsin_kernel": "int40 + uint32, fast sine"}
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for r in pk:
    x, y = float(r["ms_median"]), float(r["snr_db"])
    integer = "u32" in r["impl"] or "u64" in r["impl"]
    ax.scatter(x, y, s=60, color=C["v2"] if integer else C["v1"], edgecolor="white", linewidth=1.5, zorder=3)
    off = (6, -12) if r["impl"] == "f64_fastsin_kernel" else (6, 4)
    ax.annotate(names[r["impl"]], (x, y), xytext=off, textcoords="offset points", fontsize=8, color=C["ink"])
ax.scatter([], [], color=C["v2"], label="integer (fixed-point) harmonic phase")
ax.scatter([], [], color=C["v1"], label="floating-point harmonic phase")
ax.set_xlabel("kernel time, ms (lower is better)"); ax.set_ylabel("SNR vs fp64 reference, dB (higher is better)")
ax.set_title("Number formats in the real kernel, (8,100,64000)", fontsize=11); ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0.03, 0.30)
fig.savefig("figures/fig_formats.png"); plt.close(fig)

# 4. Scaling
sc = rows("scaling_v2_results.csv")
impls = [("faithful_angular_compile", "torch.compile", "compile"), ("v1", "fused v1", "v1"), ("v2_blockreduce", "v2, block reduce", "v2_blockreduce"),
         ("v2", "fused v2", "v2"), ("v2_fastsin", "v2, fast sine", "v2_fastsin")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (sweep, key, xl) in zip(axes, [("B", "B", "batch size B (T=64000, K=100)"), ("T", "T", "clip length T, samples (B=8, K=100)"),
                                       ("K_low", "K", "harmonics K, all below Nyquist (B=8, T=64000)")]):
    for impl, lab, col in impls:
        pts = sorted((int(r[key]), float(r["ms_median"])) for r in sc if r["sweep"] == sweep and r["impl"] == impl)
        ax.plot(*zip(*pts), color=C[col], lw=2, marker="o", ms=4, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl, fontsize=10); ax.set_ylabel("ms per call")
    xs = sorted({int(r[key]) for r in sc if r["sweep"] == sweep})
    ax.set_xticks(xs); ax.set_xticklabels([f"{v:,}" if v < 10000 else f"{v // 1000}k" for v in xs], fontsize=8, rotation=45)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[0].legend(fontsize=8)
fig.suptitle("Scaling (time only, one run)", fontsize=12)
fig.savefig("figures/fig_scaling.png"); plt.close(fig)

# 5. Pitch sweep
sw = rows("sweep_v2_results.csv")
fig, ax = plt.subplots(figsize=(8, 4.4))
for impl, lab, col in [("fused", "fused v1", "v1"), ("v2", "fused v2", "v2"), ("v2_fastsin", "v2, fast sine", "v2_fastsin")]:
    pts = sorted((float(r["active_harmonic_frac"]), float(r["ms_median"])) for r in sw if r["impl"] == impl)
    ax.plot(*zip(*pts), color=C[col], lw=2, marker="o", ms=5, label=lab)
comp = [float(r["ms_median"]) for r in sw if r["impl"] == "faithful_angular_compile"]
ax.set_xlabel("fraction of harmonics below Nyquist (pitch 1600-1920 Hz on the left, 50-60 Hz on the right)")
ax.set_ylabel("ms per call"); ax.set_ylim(bottom=0)
ax.set_title(f"Pitch sweep, (8,100,64000); torch.compile: {min(comp):.2f}-{max(comp):.2f} ms at every pitch", fontsize=11)
ax.legend(fontsize=9)
fig.savefig("figures/fig_pitch.png"); plt.close(fig)
print("wrote", sorted(os.listdir("figures")))
