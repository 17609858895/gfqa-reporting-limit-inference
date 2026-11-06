"""Exceedance prevalence under four reporting rules with station-bootstrap 95% intervals.

Also returns the assumption-free identification band [known exceedances, known + all blind-spot
non-detects]. Uses cross-fitted outputs of s09 (no refitting).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07, "Ni": 4.0, "PFOS": 0.00065, "Glyphosate": 0.1, "CdDis": 0.08,
          "CdLQ": 0.08, "PbLQ": 1.2, "HgLQ": 0.07}


def run(t, tag="", n_boot=300, seed=0):
    d = pd.read_parquet(RES / f"crossfit_{t}{tag}.parquet", columns=["station", "nd", "y_lo", "y_hi", "post_exceed", "undetermined"])
    tau = np.log10(THRESH[t])
    nd, lo, hi = d["nd"].to_numpy(), d["y_lo"].to_numpy(), d["y_hi"].to_numpy()
    known = (~nd) & (lo > tau)
    rules = pd.DataFrame({"ndc": known, "model": known + d["post_exceed"].to_numpy(),
                          "half": known | (nd & (hi - np.log10(2) > tau)), "dl": known | (nd & (hi > tau)),
                          "und": d["undetermined"].to_numpy()}).astype(float)
    rules["station"] = d["station"].to_numpy()
    st = rules.groupby("station").agg(["sum", "count"])
    sums = {c: st[(c, "sum")].to_numpy() for c in ["ndc", "model", "half", "dl", "und"]}
    cnt = st[("ndc", "count")].to_numpy()
    rng = np.random.default_rng(seed)
    boot = {c: [] for c in sums}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(cnt), len(cnt)), minlength=len(cnt))
        for c in sums:
            boot[c].append((w * sums[c]).sum() / (w * cnt).sum())
    out = {"target": t, "variant": tag or "main", "events": int(cnt.sum()), "stations": len(cnt)}
    for c in sums:
        out[c] = float(sums[c].sum() / cnt.sum())
        out[c + "_lo"], out[c + "_hi"] = np.percentile(boot[c], [2.5, 97.5])
    out["inflation_half_vs_model"] = out["half"] / out["model"]
    return out


if __name__ == "__main__":
    jobs = [(t, "") for t in THRESH] + [(t, "_noMX") for t in ["Cd", "Pb", "Hg"]]
    rows = []
    for t, tag in jobs:
        if (RES / f"crossfit_{t}{tag}.parquet").exists():
            rows.append(run(t, tag))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "T_prevalence_rules_ci.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["target", "variant", "events", "stations", "und", "ndc", "model", "model_lo", "model_hi", "half", "half_lo", "half_hi", "dl", "inflation_half_vs_model"]
    print((out[cols].set_index(["target", "variant"]) * 1).round(3).to_string())
