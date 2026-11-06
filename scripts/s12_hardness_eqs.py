"""Hardness-dependent cadmium standard (EU Directive 2013/39/EU, AA-EQS by hardness class).

Hardness per sampling event: same-day total hardness (H-T, mg CaCO3/L); otherwise
2.497 Ca + 4.118 Mg from same-day calcium and magnesium; otherwise the station median
of the event hardness. Events without any hardness information get class 1 (0.08 ug/L).
Class limits (mg CaCO3/L): <40 -> 0.08, 40-<50 -> 0.08, 50-<100 -> 0.09, 100-<200 -> 0.15, >=200 -> 0.25 ug/L.
Uses cross-fitted latent predictions from s09 (no refitting): only the threshold changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
DP, RES, TAB = ROOT / "data_processed", ROOT / "results", ROOT / "tables"
MEAS = ds.dataset(DP / "measurements", format="parquet")
BAD_Q = {"Suspect", "Contamination", "Pending review"}


def event_hardness():
    f = DP / "events_hardness.parquet"
    if f.exists():
        return pd.read_parquet(f)
    t = MEAS.to_table(filter=ds.field("param").isin(["H-T", "Ca-Tot", "Ca-Dis", "Mg-Tot", "Mg-Dis"]),
                      columns=["station", "date", "param", "flag", "value", "unit", "quality"]).to_pandas()
    t = t[(t["unit"] == "mg/l") & ~t["quality"].isin(BAD_Q) & (t["value"] > 0) & (t["flag"].fillna("") != "<")]
    t["p"] = t["param"].str.split("-").str[0]
    w = t.groupby(["station", "date", "p"])["value"].median().unstack()
    h = w.get("H")
    ca_mg = 2.497 * w.get("Ca") + 4.118 * w.get("Mg")
    hard = h.fillna(ca_mg).rename("hardness").reset_index()
    hard = hard[hard["hardness"].notna()]
    hard.to_parquet(f, index=False)
    return hard


def cd_eqs(h):
    return np.select([h < 50, h < 100, h < 200, h >= 200], [0.08, 0.09, 0.15, 0.25], 0.08)


def run(t):
    d = pd.read_parquet(RES / f"crossfit_{t}.parquet")
    hard = event_hardness()
    d = d.merge(hard, on=["station", "date"], how="left")
    st_med = d.groupby("station")["hardness"].transform("median")
    src = np.where(d["hardness"].notna(), "same-day", np.where(st_med.notna(), "station median", "none"))
    d["hardness"] = d["hardness"].fillna(st_med)
    tau = np.log10(cd_eqs(d["hardness"].to_numpy()))
    lo, hi, mu, s, nd = d["y_lo"].to_numpy(), d["y_hi"].to_numpy(), d["mu"].to_numpy(), d["sigma"].to_numpy(), d["nd"].to_numpy()
    und = nd & (hi > tau)
    known = (~nd) & (lo > tau)
    pdl = ndtr((hi - mu) / s)
    post = np.where(und, (pdl - ndtr((tau - mu) / s)) / np.clip(pdl, 1e-12, None), 0.0)
    res = {"target": t, "standard": "hardness-class", "events": len(d),
           "hardness_same_day": float(np.mean(src == "same-day")), "hardness_station_median": float(np.mean(src == "station median")),
           "hardness_none": float(np.mean(src == "none")),
           "class_shares": pd.Series(cd_eqs(d["hardness"].to_numpy())).value_counts(normalize=True).round(3).to_dict(),
           "undetermined": float(und.mean()), "prev_nd_compliant": float(known.mean()),
           "prev_model": float((known + post).mean()),
           "prev_half": float((known | (nd & (hi - np.log10(2) > tau))).mean()),
           "prev_dl": float((known | und).mean()), "mean_post_undetermined": float(post[und].mean())}
    return res


if __name__ == "__main__":
    rows = [run(t) for t in (sys.argv[1:] or ["Cd"])]
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T_cd_hardness_sensitivity.csv", index=False)
    print(out.T.to_string())
