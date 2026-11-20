"""Combine repeated benchmark runs (fused_results.csv + fused_results_run*.csv) into runs_summary.csv:
per (impl, B, K, T): number of runs, median / min / max of ms_median and energy_mJ, and peak memory and rel_l2 of run 1.
Run:  python summarize_runs.py"""
import csv, glob, statistics
files = ["fused_results.csv"] + sorted(glob.glob("fused_results_run*.csv"))
groups = {}
for fn in files:
    for r in csv.DictReader(open(fn)):
        groups.setdefault((r["impl"], r["B"], r["K"], r["T"]), []).append(r)
out = []
for (impl, B, K, T), rs in groups.items():
    ms = [float(r["ms_median"]) for r in rs]; mj = [float(r["energy_mJ"]) for r in rs]
    out.append(dict(impl=impl, B=B, K=K, T=T, runs=len(rs),
                    ms_med=round(statistics.median(ms), 4), ms_min=min(ms), ms_max=max(ms),
                    mJ_med=round(statistics.median(mj), 3), mJ_min=min(mj), mJ_max=max(mj),
                    peak_extra_MB=rs[0]["peak_extra_MB"], rel_l2_err=rs[0]["rel_l2_err"]))
with open("runs_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
print("files:", files); print("wrote runs_summary.csv,", len(out), "rows")
