"""Collect benchmark, conformal, ablation and hidden-exceedance outputs into summary tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables"
pd.set_option("display.width", 260, "display.max_rows", 200)

bench = pd.concat([pd.read_csv(f) for f in RES.glob("bench_*_*.csv")])
bench.to_csv(TAB / "T_benchmark_all.csv", index=False)
cols = ["cindex", "cnll", "exc_auc", "exc_brier", "r2_detected"]
for (t, s), g in bench.groupby(["target", "setting"]):
    extra = ["r2_truth", "rmse_hidden", "bias_hidden", "prev_true", "prev_pred"] if s == "synth" else ["exc_p_exceed_undetermined"]
    g = g.sort_values("cnll")[["model", "family"] + cols + extra]
    g.insert(2, "rank_cnll", range(1, len(g) + 1))
    print(f"\n=== {t} / {s}")
    print(g.round(3).to_string(index=False))

fam = bench.groupby(["target", "setting", "family"])[["cindex", "cnll", "exc_auc"]].median().round(3)
print("\nFamily medians\n", fam.to_string())
fam.to_csv(TAB / "T_benchmark_family_medians.csv")

conf = [pd.read_csv(f) for f in RES.glob("conformal_*_*.csv")]
if conf:
    conf = pd.concat(conf)
    conf.to_csv(TAB / "T_conformal_all.csv", index=False)
    print("\nConformal\n", conf.round(3).to_string(index=False))

hid = RES / "hidden_exceedance_summary.csv"
if hid.exists():
    print("\nHidden exceedance\n", pd.read_csv(hid).round(4).to_string(index=False))
