"""Median of the three bench_v2.py runs -> v2_summary.csv (min / max show the spread; error and memory from run 1).
Run:  python summarize_v2.py"""
import csv, statistics
from collections import OrderedDict

FILES = ["v2_results.csv", "v2_results_run2.csv", "v2_results_run3.csv"]
groups = OrderedDict()
for f in FILES:
    for r in csv.DictReader(open(f)):
        groups.setdefault((r["impl"], r["B"], r["K"], r["T"]), []).append(r)
out = []
for (impl, B, K, T), rs in groups.items():
    ms = [float(r["ms_median"]) for r in rs]; lp = [float(r["ms_loop"]) for r in rs]; mj = [float(r["energy_mJ"]) for r in rs]
    out.append(dict(impl=impl, B=B, K=K, T=T, runs=len(rs), ms_med=round(statistics.median(ms), 4), ms_min=min(ms), ms_max=max(ms),
                    ms_loop_med=round(statistics.median(lp), 4), mJ_med=round(statistics.median(mj), 3),
                    peak_extra_MB=rs[0]["peak_extra_MB"], max_abs_err_vs_fp64=rs[0]["max_abs_err_vs_fp64"],
                    rel_l2_err=rs[0]["rel_l2_err"], snr_db=rs[0]["snr_db"]))
with open("v2_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys()); w.writeheader(); w.writerows(out)
print("wrote v2_summary.csv,", len(out), "rows")
