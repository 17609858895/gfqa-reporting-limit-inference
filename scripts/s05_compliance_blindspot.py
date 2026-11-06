"""Compliance blind spot: non-detects whose reporting limit exceeds the quality standard.

A '<DL' result with DL > EQS cannot establish compliance or exceedance. This
script quantifies that undetermined share per contaminant, country and 5-year
block. Thresholds are screening values (EU EQS Directive 2013/39/EU AA-EQS for
inland waters where available; otherwise drinking-water or common status
boundaries) and are used only to define a decision boundary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DP, TAB = ROOT / "data_processed", ROOT / "tables"

THRESH = {  # target: (threshold in target unit, basis)
    "Cd": (0.08, "EU AA-EQS (class 1-2 hardness)"),
    "Pb": (1.2, "EU AA-EQS (bioavailable)"),
    "Ni": (4.0, "EU AA-EQS (bioavailable)"),
    "Hg": (0.07, "EU MAC-EQS"),
    "As": (10.0, "WHO drinking-water guideline"),
    "Atrazine": (0.6, "EU AA-EQS"),
    "PFOS": (0.00065, "EU AA-EQS"),
    "PFOA": (0.1, "Illustrative PFOA screening comparator; not an individual-PFOA legal limit"),
    "Glyphosate": (0.1, "EU drinking water (single pesticide)"),
    "NH4N": (0.2, "common good-status boundary (mg N/L)"),
    "TP": (0.1, "common good-status boundary (mg P/L)"),
    "NO2N": (0.03, "common good-status boundary (mg N/L)"),
}

st = pd.read_parquet(DP / "stations_clean.parquet")[["station", "country"]]
rows, by_c, by_t = [], [], []
for t, (thr, basis) in THRESH.items():
    d = pd.read_parquet(DP / f"target_{t}.parquet").merge(st, on="station", how="left")
    lt = d["cens"] == 1
    dl = 10 ** d["y_hi"]
    und = lt & (d["y_hi"] > np.log10(thr))
    exc = (d["cens"] == 0) & (d["y_lo"] > np.log10(thr))
    rows.append({"target": t, "threshold": thr, "basis": basis, "events": len(d),
                 "nd_share": lt.mean(), "undetermined_share": und.mean(),
                 "undetermined_of_nd": und.sum() / max(lt.sum(), 1),
                 "known_exceed_share": exc.mean(),
                 "stations_any_undetermined": d.loc[und, "station"].nunique() / d["station"].nunique()})
    g = d.assign(und=und, nd=lt).groupby("country").agg(events=("und", "size"), und=("und", "mean"), nd=("nd", "mean"))
    by_c.append(g.assign(target=t).reset_index())
    d["block"] = (d["date"].dt.year // 5) * 5
    g = d.assign(und=und, nd=lt, dl=np.where(lt, dl, np.nan)).groupby("block").agg(
        events=("und", "size"), und=("und", "mean"), nd=("nd", "mean"), median_dl=("dl", "median"))
    by_t.append(g.assign(target=t).reset_index())

res = pd.DataFrame(rows)
res.to_csv(TAB / "T_compliance_blindspot.csv", index=False)
pd.concat(by_c).to_csv(TAB / "T_blindspot_by_country.csv", index=False)
pd.concat(by_t).to_csv(TAB / "T_blindspot_by_block.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(3).to_string())
bt = pd.concat(by_t)
print(bt[bt.events > 500].pivot(index="block", columns="target", values="median_dl").round(4).to_string())
