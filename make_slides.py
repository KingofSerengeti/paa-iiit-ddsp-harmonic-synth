"""Build the Evaluation 2 deck (slides/PAA_Eval2_DDSP_v5.pptx) from the v4 deck's theme and layout.
The v4 file is only read, never modified. All numbers are read from the CSV / log files in this folder.
Needs python-pptx.  Run:  python make_slides.py"""
import copy, csv, math, re
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SRC, OUT = "slides/PAA_Eval1_DDSP_v4.pptx", "slides/PAA_Eval2_DDSP_v5.pptx"
DK, TEAL, ORANGE, LIGHT, BORDER, GREY, TXT = (RGBColor(0x0B, 0x3C, 0x49), RGBColor(0x0E, 0x7C, 0x86), RGBColor(0xE0, 0x7A, 0x1F),
                                             RGBColor(0xF1, 0xF5, 0xF7), RGBColor(0xC9, 0xD3, 0xDA), RGBColor(0x55, 0x60, 0x6E),
                                             RGBColor(0x1F, 0x29, 0x33))
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# ---------- data ----------
def rows(path):
    return list(csv.DictReader(open(path)))


def pick(rs, impl, cfg, **extra):
    for r in rs:
        if r["impl"] == impl and (int(r["B"]), int(r["K"]), int(r["T"])) == cfg and all(r[k] == v for k, v in extra.items()):
            return r
    raise KeyError((impl, cfg))


SUM = rows("runs_summary.csv")
REF = rows("reform_torch_results.csv")
ABL = rows("ablate_fp64_results.csv")
SWP = rows("sweep_f0_results.csv")
PRC = rows("precision_sweep_results.csv")
OLD = rows("baseline_faithful_results.csv")
CFGS = [(1, 100, 1024), (1, 100, 16000), (8, 100, 64000), (8, 64, 64000), (8, 128, 64000), (16, 100, 64000)]
MAIN = (8, 100, 64000)
S = lambda impl, cfg=MAIN: pick(SUM, impl, cfg)
f = float
pct = lambda x: f"{100 * f(x):.3g}%"


def sci(x):
    m, e = f"{f(x):.1e}".split("e")
    return f"{m}e{int(e)}"


# ---------- deck plumbing ----------
prs = Presentation(SRC)
layout = prs.slides[3].slide_layout
for sp in layout.shapes:                                  # footer text lives in the layout
    if sp.has_text_frame and "Evaluation 1" in sp.text_frame.text:
        for p in sp.text_frame.paragraphs:
            for r in p.runs:
                r.text = r.text.replace("Evaluation 1", "Evaluation 2")
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):                              # drop the v4 slides from the copy
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)


def text(slide, x, y, w, h, paras, size=12, color=TXT, anchor=MSO_ANCHOR.TOP):
    """paras: list of str or (str, dict(bold, color, size, bullet, italic))."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, item in enumerate(paras):
        s, o = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(o.get("after", 3))
        if o.get("align") == "c": p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = ("•  " if o.get("bullet") else "") + s
        r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", False); r.font.italic = o.get("italic", False)
        r.font.color.rgb = o.get("color", color)
    return tb


def title(slide, s, n):
    text(slide, 0.6, 0.28, 12.2, 0.75, [(s, dict(size=26, bold=True, color=DK))])
    text(slide, 12.4, 7.05, 0.6, 0.3, [(str(n), dict(size=10, color=GREY, align="c"))])


def card(slide, x, y, w, h, head, lines, size=12):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT; box.line.fill.background(); box.shadow.inherit = False
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.08), Inches(h))
    bar.fill.solid(); bar.fill.fore_color.rgb = TEAL; bar.line.fill.background(); bar.shadow.inherit = False
    text(slide, x + 0.22, y + 0.1, w - 0.35, 0.4, [(head, dict(size=14, bold=True, color=DK))])
    text(slide, x + 0.22, y + 0.5, w - 0.35, h - 0.6, [(l, dict(bullet=True, after=4)) for l in lines], size=size)


def table(slide, x, y, w, data, col_w=None, size=11, row_h=0.3, hl_rows=()):
    nr, nc = len(data), len(data[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gt.table
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    for a in ("bandRow", "firstRow"): tblPr.set(a, "0")
    if col_w:
        tot = sum(col_w)
        for i, cw in enumerate(col_w): t.columns[i].width = Emu(int(Inches(w) * cw / tot))
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = t.cell(i, j); c.text = str(data[i][j])
            c.margin_left = c.margin_right = Inches(0.05); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = TEAL if i == 0 else (RGBColor(0xFD, 0xEB, 0xD8) if i in hl_rows else (WHITE if i % 2 else LIGHT))
            for p in c.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for r in p.runs:
                    r.font.size = Pt(size); r.font.bold = (i == 0)
                    r.font.color.rgb = WHITE if i == 0 else TXT
    return gt


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def new(n, s):
    sl = prs.slides.add_slide(layout)
    for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
    title(sl, s, n)
    return sl


cfg_s = lambda c: f"({c[0]},{c[1]},{c[2]})"

# ---------- 1. title ----------
sl = prs.slides.add_slide(layout)
for ph in list(sl.placeholders): ph._element.getparent().remove(ph._element)
text(sl, 0.6, 0.5, 12.2, 0.8, [("Benchmarking and Optimizing the DDSP Harmonic Synthesizer on a GPU", dict(size=28, bold=True, color=DK))])
text(sl, 0.6, 1.3, 12.0, 0.5, [("Evaluation 2: fused CUDA kernel, results and ablations", dict(size=18, color=TEAL))])
text(sl, 0.6, 2.1, 11.8, 1.2, [("A fused CUDA kernel for the harmonic synthesizer of DDSP (Engel et al., ICLR 2020) is compared with a "
                                "PyTorch port of DDSP and with torch.compile on time, extra GPU memory, energy per call and error against "
                                "an fp64 reference, on an RTX 5070 (Blackwell, sm_120).", dict(size=16))])
text(sl, 0.6, 3.7, 11.8, 1.4, [("Student: Krishna Kartheek (Roll No. 2025701032), solo project", dict(size=14)),
                               ("Repository: github.com/KingofSerengeti/paa-iiit-ddsp-harmonic-synth", dict(size=14, color=ORANGE, bold=True)),
                               ("Course: Programming AI Accelerators, IIIT Hyderabad", dict(size=14))])
text(sl, 12.4, 7.05, 0.6, 0.3, [("1", dict(size=10, color=GREY, align="c"))])
notes(sl, "Introduce the topic in one sentence: the harmonic synthesizer of DDSP turns slow control signals into audio as a sum of sine waves. "
          "Eval 1 measured the baseline; Eval 2 adds a working fused CUDA kernel, the comparison and the ablations.")

# ---------- 2. problem + baseline recap ----------
sl = new(2, "Problem and baseline (recap from Evaluation 1)")
card(sl, 0.6, 1.15, 6.3, 5.6, "The operator", [
    "Output sample t:  y[t] = sum over k = 1..K of  a_k(t) * sin(k * phi(t))",
    "Controls (pitch f0, loudness, harmonic mix) arrive at 250 frames/s; audio is 16 kHz, so one frame = 64 samples (hop).",
    "The PyTorch port of DDSP's harmonic_synthesis builds tensors of shape [B, T, K] for frequency, amplitude and phase. "
    "One such fp32 tensor at B=8, K=100, T=64000 is 204.8 MB; the useful output is 2 MB.",
    "The phase is a running sum (cumsum) over up to 64000 samples per harmonic. In fp32 the rounding error of this sum grows with clip length.",
    "Question for this phase: can one kernel avoid the [B,T,K] tensors and the long running sum, and what does it gain?"])
e1, a1, c1 = [pick(OLD, i, MAIN) for i in ("faithful_eager", "faithful_angular_eager", "faithful_angular_compile")]
table(sl, 7.15, 1.2, 5.6, [["Baseline (8,100,64000)", "ms", "extra MB", "rel. error"],
                           ["Eager (DDSP default)", e1["ms_median"], e1["peak_extra_MB"], pct(e1["rel_l2_err"])],
                           ["Eager + angular cumsum", a1["ms_median"], a1["peak_extra_MB"], pct(a1["rel_l2_err"])],
                           ["torch.compile, angular", c1["ms_median"], c1["peak_extra_MB"], pct(c1["rel_l2_err"])]],
      col_w=[2.6, 0.9, 1.1, 1.1], size=11, row_h=0.36)
errs = [pct(pick(OLD, "faithful_eager", c)["rel_l2_err"]) for c in [(1, 100, 1024), (1, 100, 16000), MAIN]]
text(sl, 7.15, 2.8, 5.6, 3.9, [
    (f"Plain fp32 eager error vs fp64 grows with clip length: {errs[0]} (T=1024), {errs[1]} (T=16000), {errs[2]} (T=64000).", dict(bullet=True, after=6)),
    ("torch.compile of DDSP's default path fails with an Inductor error, so the compiled baseline uses the angular-cumsum path (DDSP's accurate option).", dict(bullet=True, after=6)),
    ("Source: baseline_faithful_results.csv (Evaluation 1 run). All later slides use same-session measurements.", dict(bullet=True, italic=True, color=GREY, size=11))], size=12)
notes(sl, "Recap only, about one minute. Point at the formula, then at the 205 MB per tensor versus 2 MB of output. "
          "The compiled angular path (about 2 ms, 0.08% error) is the baseline to beat.")

# ---------- 3. implementation ----------
sl = new(3, "Implementation: one fused CUDA kernel")
def flowbox(x, y, w, h, head, body, fill):
    b = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.color.rgb = TEAL; b.shadow.inherit = False
    b.adjustments[0] = 0.08
    text(sl, x + 0.12, y + 0.08, w - 0.24, h - 0.16, [(head, dict(size=13, bold=True, color=DK))] + [(l, dict(size=11.5, after=2)) for l in body])
flowbox(0.6, 1.2, 3.6, 2.2, "Step A: PyTorch, fp64 (tiny)", [
    "Frame phase offsets: exclusive prefix sum over F = T/64 frames (1000 values per 4 s clip), reduced mod 2*pi.",
    "Replaces the 64000-step running sum per harmonic."], WHITE)
arr = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.3), Inches(2.1), Inches(0.5), Inches(0.4))
arr.fill.solid(); arr.fill.fore_color.rgb = TEAL; arr.line.fill.background()
flowbox(4.9, 1.2, 7.85, 2.2, "Step B: CUDA kernel, one thread per output sample t (grid: T/256 x B blocks)", [
    "1. Frame j = t / 64, position n = t mod 64.",
    "2. Base phase x in cycles, closed form inside the frame (pitch is linear there, so the phase is quadratic in n). fp64, then fp32.",
    "3. Nyquist limit k_max: largest k with k * f0(t) < 8 kHz, decided in fp64 with the reference formula.",
    "4. For k = 1..k_max:  acc += a_k(t) * sin(2*pi*frac(k*x)); a_k = Hann-weighted blend of frames j and j+1.",
    "5. Write y[t]. Output [B, T] only; no [B, T, K] tensor exists."], LIGHT)
card(sl, 0.6, 3.65, 6.3, 3.1, "Ideas that remove work", [
    "Exact phase: only 1000 frame totals need a prefix sum; the rest is a formula.",
    "One phase per sample: harmonic k has phase k*x (mod 1 cycle) instead of its own running sum.",
    "Skip harmonics above Nyquist: DDSP sets their amplitude to 0 but still computes them.",
    "Everything stays in registers; memory traffic is the inputs (3.3 MB) and the output (2 MB)."], size=11.5)
card(sl, 7.15, 3.65, 5.6, 3.1, "Build and variants", [
    "PyTorch C++/CUDA extension (torch.utils.cpp_extension.load_inline), file fused_ddsp.py.",
    "System nvcc 12.0 cannot target sm_120, so compute_90 PTX is compiled and the driver JIT-compiles it for the RTX 5070.",
    "Flags for ablations: skip_inactive, fast_sin (__sinf), block size, fp32_setup.",
    "Error reference: the PyTorch port in fp64."], size=11.5)
notes(sl, "Explain the two steps. Step A is a running total over 1000 frames instead of 64000 samples per harmonic. "
          "Step B: each GPU thread makes one audio sample and loops over harmonics, keeping the sum in a register. "
          "Nothing large is written to memory.")

# ---------- 4. correctness ----------
sl = new(4, "Correctness against the fp64 reference")
log = open("test_fused_log.txt").read()
blocks = re.findall(r"--- B=(\d+) K=(\d+) T=(\d+)\n(.*?)(?=\n---|\Z)", log, re.S)
data = [["(B, K, T)", "fused: max abs", "fused: rel. L2", "fast sine: rel. L2", "angular eager fp32: rel. L2"]]
for B, K, T, body in blocks:
    g = lambda name: re.search(re.escape(name) + r"\s+max_abs=(\S+)\s+rel_l2=(\S+)", body).groups()
    fz, fs, ae = g("fused (skip, sinpif)"), g("fused fast-sin"), g("angular eager fp32")
    data.append([f"({B},{K},{T})", sci(fz[0]), sci(fz[1]), sci(fs[1]), sci(ae[1])])
table(sl, 0.6, 1.2, 12.15, data, col_w=[1.6, 1.6, 1.6, 1.6, 2.2], size=12, row_h=0.36)
card(sl, 0.6, 3.25, 6.3, 3.5, "Test setup", [
    "test_fused.py compares every variant with the PyTorch port run in fp64, batch item by batch item.",
    "Fused error stays near 5e-7 for all lengths; the fp32 angular-cumsum path reaches 1.9e-3 at T = 64000.",
    "The test prints errors and has no pass/fail threshold; its stated expectation is 1e-6 to 1e-5 relative L2."], size=12)
card(sl, 7.15, 3.25, 5.6, 3.5, "Defect found by the test and fixed", [
    "First version: max abs error 2.8e-3 at (8,100,64000).",
    "Cause: one clip has f0 = 363.6364 Hz = 8000/22, so harmonic 22 lies exactly at Nyquist. The fp32 check kept it; the fp64 reference removed it.",
    "Fix in the kernel: the cutoff is decided in fp64 with the reference formula. The test was not changed.",
    "After the fix: max abs error 3.5e-6 for that config."], size=12)
notes(sl, "The kernel matches the fp64 reference to about 5e-7, which is the fp32 limit. "
          "Tell the Nyquist story briefly: a harmonic exactly on the 8 kHz limit was treated differently. Fixed in the kernel, not in the test.")

# ---------- 5. main results ----------
sl = new(5, "Results: fused kernel vs torch.compile")
data = [["(B, K, T)", "fused ms", "compile ms", "speed-up", "same maths + compile ms", "extra MB fused / compile", "mJ fused / compile", "rel. L2 fused / compile"]]
hl = ()
for i, c in enumerate(CFGS, 1):
    fu, co, rc = S("fused", c), S("faithful_angular_compile", c), pick(REF, "reform_compile", c)
    data.append([cfg_s(c), fu["ms_med"], co["ms_med"], f"{f(co['ms_med']) / f(fu['ms_med']):.1f}x", rc["ms_median"],
                 f"{fu['peak_extra_MB']} / {co['peak_extra_MB']}", f"{f(fu['mJ_med']):.1f} / {f(co['mJ_med']):.1f}",
                 f"{sci(fu['rel_l2_err'])} / {sci(co['rel_l2_err'])}"])
    if c == MAIN: hl = (i,)
table(sl, 0.6, 1.15, 12.15, data, col_w=[1.25, 0.9, 0.95, 0.85, 1.35, 1.6, 1.55, 1.9], size=11, row_h=0.36, hl_rows=hl)
text(sl, 0.6, 3.72, 12.1, 0.3, [("Medians of 4 runs (runs_summary.csv). 'Same maths + compile' = the kernel's algorithm written in PyTorch and "
                                 "compiled, one run (reform_torch_results.csv). compile = torch.compile of DDSP's angular path. Energy: NVML board energy.",
                                 dict(size=10, italic=True, color=GREY))])
fu, co, rc = S("fused"), S("faithful_angular_compile"), pick(REF, "reform_compile", MAIN)
sm = S("fused", (1, 100, 1024)); sc = S("faithful_angular_compile", (1, 100, 1024)); sk = S("fused_kernel_only", (1, 100, 1024))
card(sl, 0.6, 4.15, 6.3, 2.6, "At (8,100,64000)", [
    f"{f(co['ms_med']) / f(fu['ms_med']):.1f}x faster, {f(co['peak_extra_MB']) / f(fu['peak_extra_MB']):.0f}x less extra memory, "
    f"{f(co['mJ_med']) / f(fu['mJ_med']):.1f}x less energy per call.",
    f"Error {sci(fu['rel_l2_err'])} vs {sci(co['rel_l2_err'])} (about {f(co['rel_l2_err']) / f(fu['rel_l2_err']):.0f}x lower).",
    f"Smallest case (1,100,1024): the full fused call is slower ({sm['ms_med']} vs {sc['ms_med']} ms); the kernel alone takes {sk['ms_med']} ms."], size=12)
card(sl, 7.15, 4.15, 5.6, 2.6, "Where the gain comes from", [
    f"Same maths in PyTorch + compile: error {sci(rc['rel_l2_err'])}, time {rc['ms_median']} ms.",
    "So the accuracy comes from the maths (exact frame phase), and the speed comes from the hand-written kernel.",
    "Only one PyTorch formulation was tried."], size=12)
notes(sl, "Main slide. Read the highlighted row. Then the control column: the same algorithm written in PyTorch and compiled is accurate but slow. "
          "Say openly that compile wins the smallest case, because the full call has 11 small extra PyTorch operations.")

# ---------- 6. ablations ----------
sl = new(6, "Ablations at (8,100,64000)")
d, ns, fs, ko = S("fused"), S("fused_noskip"), S("fused_fastsin"), S("fused_kernel_only")
bl = {b: S(f"fused_block{b}") for b in (64, 128, 512, 1024)}
ak, ak32 = pick(ABL, "fused_kernel_only", MAIN), pick(ABL, "fused_kernel_only_fp32setup", MAIN)
data = [["Variant", "What changes", "ms", "mJ", "rel. L2", "Source"],
        ["fused (default)", "skip above Nyquist, sinpif, block 256", d["ms_med"], f"{f(d['mJ_med']):.1f}", sci(d["rel_l2_err"]), "4 runs"],
        ["no skip", "loop over all K harmonics", ns["ms_med"], f"{f(ns['mJ_med']):.1f}", sci(ns["rel_l2_err"]), "4 runs"],
        ["fast sine", "__sinf on reduced angle", fs["ms_med"], f"{f(fs['mJ_med']):.1f}", sci(fs["rel_l2_err"]), "4 runs"],
        ["block 64 / 128 / 512 / 1024", "threads per block", " / ".join(bl[b]["ms_med"] for b in bl), " / ".join(f"{f(bl[b]['mJ_med']):.1f}" for b in bl), sci(d["rel_l2_err"]), "4 runs"],
        ["kernel only", "frame offsets precomputed", ko["ms_med"], f"{f(ko['mJ_med']):.1f}", sci(ko["rel_l2_err"]), "4 runs"],
        ["kernel only, fp64 setup", "same run as next row", ak["ms_median"], f"{f(ak['energy_mJ']):.1f}", sci(ak["rel_l2_err"]), "1 run"],
        ["kernel only, fp32 setup", "phase and k_max in fp32", ak32["ms_median"], f"{f(ak32['energy_mJ']):.1f}", sci(ak32["rel_l2_err"]), "1 run"]]
table(sl, 0.6, 1.15, 12.15, data, col_w=[2.3, 2.9, 2.1, 1.7, 1.1, 0.8], size=11, row_h=0.34)
card(sl, 0.6, 4.05, 12.15, 2.7, "Findings", [
    f"Skipping harmonics above Nyquist: {100 * (1 - f(d['ms_med']) / f(ns['ms_med'])):.0f}% less time, {100 * (1 - f(d['mJ_med']) / f(ns['mJ_med'])):.0f}% less energy.",
    f"Fast sine: same time, {100 * (1 - f(fs['mJ_med']) / f(d['mJ_med'])):.0f}% less energy, slightly higher error.",
    "Block size 64 to 512: no measurable difference; 1024 is slower.",
    f"About {f(d['ms_med']) - f(ko['ms_med']):.3f} ms per call is spent outside the kernel (frame offsets in PyTorch: 11 small GPU operations).",
    f"fp32 setup: kernel {f(ak['ms_median']) / f(ak32['ms_median']):.1f}x faster, but the Nyquist error returns (max abs {sci(ak32['max_abs_err_vs_fp64'])})."], size=12)
notes(sl, "Each row changes one thing. Skipping saves energy more than time. The fp32-setup row matters most: "
          "it halves kernel time, which points to the fp64 setup as the main cost, at the price of accuracy.")

# ---------- 7. analysis ----------
sl = new(7, "Analysis: what limits the fused kernel")
data = [["f0 range (Hz)", "active harmonics", "fused ms", "fused mJ", "no-skip mJ", "compile ms", "compile rel. L2"]]
for rng in dict.fromkeys(r["f0_range"] for r in SWP):
    g = lambda impl: next(r for r in SWP if r["f0_range"] == rng and r["impl"] == impl)
    a, b, c = g("fused"), g("fused_noskip"), g("faithful_angular_compile")
    data.append([rng, pct(a["active_harmonic_frac"]), a["ms_median"], f"{f(a['energy_mJ']):.1f}", f"{f(b['energy_mJ']):.1f}", c["ms_median"], sci(c["rel_l2_err"])])
table(sl, 0.6, 1.15, 7.0, data, col_w=[1.2, 1.2, 0.9, 0.9, 1.0, 1.0, 1.2], size=10.5, row_h=0.32)
text(sl, 0.6, 3.45, 7.0, 0.3, [("Pitch sweep, (8,100,64000), one run (sweep_f0_results.csv).", dict(size=10, italic=True, color=GREY))])
# derived arithmetic
B, K, T, F = 8, 100, 64000, 1000
fb = B * F * K * 4 + B * F * 16 + B * T * 4
gbs = fb / (f(ko["ms_med"]) * 1e-3) / 1e9
cgbs = 2 * B * T * K * 4 / (f(co["ms_med"]) * 1e-3) / 1e9
prof = open("torch_profile_fused.txt").read()
kp = re.findall(r"(\d+\.\d+)%", next(l for l in prof.splitlines() if "harmonic_kernel" in l))[-1]
ev = re.findall(r"CUDA events per call.*?: (\S+)", prof)
card(sl, 7.85, 1.15, 4.9, 5.6, "Observations", [
    "Kernel time stays flat while active harmonics fall from 100% to 4%; energy falls. Kernel-only time is also the same for K = 64, 100, 128.",
    "So the sine loop is not what sets the time. Making the per-sample setup fp32 halves kernel time: the fp64 setup is the largest measured cost (GeForce GPUs have low fp64 throughput).",
    f"Memory: minimum traffic {fb / 1e6:.2f} MB per call, i.e. {gbs:.0f} GB/s = {100 * gbs / 672:.1f}% of the 672 GB/s peak (derived, not measured). Not memory-bound.",
    f"torch.profiler: the kernel is {kp}% of fused GPU time ({ev[0][:-2]} GPU operations per call); compile runs {ev[1][:-2]} kernels and needs at least {cgbs:.0f} GB/s.",
    "Not confirmed with Nsight Compute (no sm_120 support installed)."], size=11)
card(sl, 0.6, 3.85, 7.0, 2.9, "Reading the sweep", [
    "fused: time constant (0.214 to 0.228 ms); energy follows the number of active harmonics.",
    "no skip: energy constant, since all 100 harmonics are computed.",
    "compile: time constant (about 2.1 ms); its error grows with pitch (cause not tested)."], size=11)
notes(sl, "Key analysis. If the sine loop were the bottleneck, fewer active harmonics would mean less time. It does not. "
          "The fp32-setup ablation shows the fp64 per-sample setup costs about half of the kernel time. Memory traffic is tiny. "
          "This is supported by ablation, not yet by Nsight Compute.")

# ---------- 8. precision ----------
sl = new(8, "Precision study: how many bits does the phase need?")
keep = ["fixed-point phase, 8 bits", "fixed-point phase, 12 bits", "fixed-point phase, 16 bits", "fixed-point phase, 20 bits",
        "fixed-point phase, 24 bits", "fixed-point phase, 32 bits", "bf16 everywhere", "fp16 everywhere", "fp32 everywhere"]
data = [["Format (rest fp64 unless 'everywhere')", "rel. L2 error", "SNR (dB)"]]
for r in PRC:
    if r["variant"] in keep: data.append([r["variant"], sci(r["rel_l2_err"]), r["snr_dB"]])
table(sl, 0.6, 1.15, 6.4, data, col_w=[3.4, 1.5, 1.2], size=11, row_h=0.33)
text(sl, 0.6, 4.55, 6.4, 0.5, [("Software emulation in PyTorch at (8,100,64000); accuracy only, speed not measured (precision_sweep_results.csv). "
                                "SNR = -20 log10(rel. L2).", dict(size=10, italic=True, color=GREY))])
f32 = next(r for r in PRC if r["variant"] == "fp32 everywhere")
card(sl, 7.25, 1.15, 5.5, 5.6, "Findings", [
    "The phase is stored as a fraction of a cycle in [0, 1), so fixed point fits it naturally.",
    "Each extra bit of phase adds about 6 dB. About 20 bits give about 96 dB, the level of 16-bit audio.",
    "fp16 (11-bit significand) and bf16 (8-bit) are too coarse for the phase: the phase error is multiplied by k for harmonic k.",
    f"fp32 everywhere gives {f32['snr_dB']} dB, the same error as the fused kernel ({sci(f32['rel_l2_err'])}). The torch.compile baseline is at about "
    f"{-20 * math.log10(f(co['rel_l2_err'])):.0f} dB.",
    "fp64 is used only for the reference, the frame offsets and the per-sample setup; the harmonic loop is fp32.",
    "Candidate for the next version: a 32-bit integer phase."], size=12)
notes(sl, "This answers the Evaluation 1 question about lower precision. The domain tolerates fixed point well, but the phase needs about 20 bits. "
          "fp16 and bf16 are not enough for the phase. This is a simulation, not a kernel.")

# ---------- 9. limitations and next steps ----------
sl = new(9, "Summary, limitations and next steps")
card(sl, 0.6, 1.15, 12.15, 1.55, "Summary", [
    f"A fused CUDA kernel computes the DDSP harmonic synthesizer at (8,100,64000) in {fu['ms_med']} ms vs {co['ms_med']} ms for torch.compile, "
    f"with {fu['peak_extra_MB']} MB vs {co['peak_extra_MB']} MB extra memory and error {sci(fu['rel_l2_err'])} vs {sci(co['rel_l2_err'])}.",
    "The accuracy comes from the exact frame phase; the speed comes from the kernel. The fp64 per-sample setup is now the largest measured cost."], size=12)
card(sl, 0.6, 2.9, 6.0, 3.85, "Limitations", [
    "One GPU (RTX 5070); PTX JIT build, not a native sm_120 build.",
    "No Nsight Compute for sm_120 yet: DRAM and SM throughput not measured.",
    "Baseline is a PyTorch port of DDSP; validation against the TensorFlow original is pending.",
    "Forward pass only; one input generator (seed 0); energy is board-level NVML.",
    "Small inputs: the full call is slower than compile because of the PyTorch-side operations."], size=11.5)
card(sl, 6.75, 2.9, 6.0, 3.85, "Next steps (Evaluation 3)", [
    "fp64 once per frame, fp32 per sample: aim for fp32-setup speed with fp64 accuracy.",
    "Move the frame offsets into the kernel to remove the 11 small operations.",
    "Nsight Compute 2025 and a native sm_120 build (CUDA 12.8+).",
    "Runs on the RTX 5080 and GTX 1650; backward pass for training.",
    "Integer (fixed-point) phase, guided by the precision study."], size=11.5)
notes(sl, "Close with the summary sentence, name the main limitation (no Nsight Compute yet) and the first next step (fp64 once per frame).")

prs.save(OUT)
print("wrote", OUT, "with", len(prs.slides), "slides")
