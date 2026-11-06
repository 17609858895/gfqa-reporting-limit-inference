"""Illusory compliance trends: within-station exceedance trends under different reporting rules.

For each contaminant x country, the per-event exceedance indicator of each rule is regressed on
time with station fixed effects (within-station demeaning; linear probability model), so changes
in which stations are monitored cannot create the trend. Rules:
  half  : detected > tau, or non-detect with DL/2 > tau (EU Directive 2009/90/EC practice)
  dl    : detected > tau, or non-detect with DL > tau
  ndc   : detected > tau only (non-detects compliant)
  model : detected > tau, plus posterior P(Y > tau | Y < DL, x) for blind-spot non-detects
Also returned: within-station trend of the blind-spot indicator and of log10 DL among non-detects.
Station-clustered standard errors by a station bootstrap (100 draws).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07, "Ni": 4.0, "PFOS": 0.00065, "Glyphosate": 0.1, "CdDis": 0.08}


def within_slope(y, t, g, w=None):
    df = pd.DataFrame({"y": y, "t": t, "g": g})
    df = df[df.groupby("g")["t"].transform("nunique") > 1]
    yd = df["y"] - df.groupby("g")["y"].transform("mean")
    td = df["t"] - df.groupby("g")["t"].transform("mean")
    return float((yd * td).sum() / (td ** 2).sum()) if (td ** 2).sum() > 0 else np.nan, df


def run(t, min_events=2000, min_years=8, n_boot=100, seed=0, tag=""):
    d = pd.read_parquet(RES / f"crossfit_{t}{tag}.parquet")
    tau = np.log10(THRESH[t])
    d["year"] = d["date"].dt.year + d["date"].dt.dayofyear / 365.25
    nd = d["nd"].to_numpy()
    lo, hi = d["y_lo"].to_numpy(), d["y_hi"].to_numpy()
    known = (~nd) & (lo > tau)
    d["r_half"] = (known | (nd & (hi - np.log10(2) > tau))).astype(float)
    d["r_dl"] = (known | (nd & (hi > tau))).astype(float)
    d["r_ndc"] = known.astype(float)
    d["r_model"] = known + d["post_exceed"].to_numpy()
    d["r_und"] = d["undetermined"].astype(float)
    rows = []
    rng = np.random.default_rng(seed)
    for c, g in d.groupby("country"):
        if len(g) < min_events or g["year"].max() - g["year"].min() < min_years:
            continue
        out = {"target": t, "variant": tag or "main", "country": c, "events": len(g), "stations": g["station"].nunique(),
               "years": f"{int(g['year'].min())}-{int(g['year'].max())}"}
        stations = g["station"].unique()
        for col in ["r_half", "r_dl", "r_ndc", "r_model", "r_und"]:
            est, _ = within_slope(g[col].to_numpy(), g["year"].to_numpy(), g["station"].to_numpy())
            out[col] = est * 1000  # percentage points per decade
        gnd = g[g["nd"]]
        if len(gnd) > 200:
            out["log_dl_trend"] = within_slope(gnd["y_hi"].to_numpy(), gnd["year"].to_numpy(), gnd["station"].to_numpy())[0] * 10
        # station bootstrap for the artefact (half - model)
        idx = {s: np.flatnonzero(g["station"].to_numpy() == s) for s in stations}
        art = []
        for _ in range(n_boot):
            pick = np.concatenate([idx[s] for s in rng.choice(stations, len(stations))])
            gb = g.iloc[pick]
            a1, _ = within_slope(gb["r_half"].to_numpy(), gb["year"].to_numpy(), gb["station"].to_numpy())
            a2, _ = within_slope(gb["r_model"].to_numpy(), gb["year"].to_numpy(), gb["station"].to_numpy())
            art.append((a1 - a2) * 1000)
        out["artefact"] = out["r_half"] - out["r_model"]
        out["artefact_lo"], out["artefact_hi"] = np.percentile(art, [2.5, 97.5])
        rows.append(out)
        print(t, c, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=["Cd", "Pb", "Hg"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    res = pd.concat([run(t, tag=a.tag) for t in a.targets])
    f = TAB / "T_trend_artefacts.csv"
    if f.exists():
        old = pd.read_csv(f)
        if "variant" not in old:
            old["variant"] = "main"
        key = set(zip(res["target"], res["variant"]))
        res = pd.concat([old[[k not in key for k in zip(old["target"], old["variant"])]], res])
    res.to_csv(f, index=False)
    pd.set_option("display.width", 250)
    print(res.round(2).to_string(index=False))
    r = res.dropna(subset=["r_und"])
    print("corr(artefact, blind-spot trend) =", np.corrcoef(r["artefact"], r["r_und"])[0, 1].round(3), "n =", len(r))
