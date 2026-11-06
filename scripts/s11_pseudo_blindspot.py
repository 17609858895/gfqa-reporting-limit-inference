"""Real-data check of blind-spot triage: re-censor detected values at real blind-spot limits.

Detected events (truth known) are assigned a reporting limit drawn from the real
blind-spot limits (DL > threshold) of the same country; events with y < assigned DL
become emulated blind-spot non-detects. Cross-fitted LGB-Tobit predictions (s09; each
event predicted by a model that never saw its station) give the posterior
    P(Y > tau | Y < DL, x).
Calibration of this posterior is compared with the reporting rules, which classify every
blind-spot non-detect as exceeding (DL/2 or DL rule) or compliant (ND-as-compliant rule).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07, "Ni": 4.0, "NH4N": 0.2, "PFOS": 0.00065, "PFOA": 0.1,
          "Glyphosate": 0.1, "Atrazine": 0.6, "As": 10.0,
          "CdDis": 0.08, "PbDis": 1.2, "NiDis": 4.0, "HgDis": 0.07,
          "CdLQ": 0.08, "PbLQ": 1.2, "HgLQ": 0.07}


def run(t, seed=0, tag=""):
    rng = np.random.default_rng(seed)
    d = pd.read_parquet(RES / f"crossfit_{t}{tag}.parquet")
    tau = np.log10(THRESH[t])
    bs = d[d["undetermined"]]
    det = d[~d["nd"]].copy()
    pools = {c: g["y_hi"].to_numpy() for c, g in bs.groupby("country")}
    det = det[det["country"].isin(pools)]
    det["dl_em"] = [pools[c][rng.integers(len(pools[c]))] for c in det["country"]]
    em = det[det["y_lo"] < det["dl_em"]].copy()       # would have been reported as '<DL'
    y, mu, s, dl = em["y_lo"].to_numpy(), em["mu"].to_numpy(), em["sigma"].to_numpy(), em["dl_em"].to_numpy()
    pdl = ndtr((dl - mu) / s)
    post = (pdl - ndtr((tau - mu) / s)) / np.clip(pdl, 1e-12, None)
    truth = (y > tau).astype(float)
    bins = np.array([0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0001])
    b = np.digitize(post, bins) - 1
    rel = pd.DataFrame({"bin": b, "post": post, "truth": truth}).groupby("bin").agg(
        n=("truth", "size"), mean_post=("post", "mean"), obs=("truth", "mean")).reset_index()
    rel["target"], rel["variant"] = t, tag or "main"
    summ = {"target": t, "variant": tag or "main", "n_emulated": len(em), "true_exceed_share": float(truth.mean()),
            "mean_posterior": float(post.mean()), "brier_posterior": float(np.mean((post - truth) ** 2)),
            "brier_rule_exceed": float(np.mean((1 - truth) ** 2)), "brier_rule_compliant": float(np.mean(truth ** 2)),
            "likely_compliant_share(<0.05)": float(np.mean(post < 0.05)),
            "false_compliance_in_likely(<0.05)": float(truth[post < 0.05].mean()) if (post < 0.05).any() else np.nan,
            "likely_exceed_share(>0.5)": float(np.mean(post > 0.5)),
            "precision_likely_exceed": float(truth[post > 0.5].mean()) if (post > 0.5).any() else np.nan}
    return summ, rel


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=["Cd", "Pb", "Hg"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out, rels = [], []
    for t in a.targets:
        s, r = run(t, tag=a.tag)
        out.append(s)
        rels.append(r)
    pd.set_option("display.width", 250)
    out, rels = pd.DataFrame(out), pd.concat(rels)
    # merge with earlier runs (keyed by target x variant)
    for f, new in [(TAB / "T_pseudo_blindspot.csv", out), (TAB / "T_pseudo_blindspot_reliability.csv", rels)]:
        if f.exists():
            prev = pd.read_csv(f)
            if "variant" not in prev:
                prev["variant"] = "main"
            key = set(zip(new["target"], new["variant"]))
            new = pd.concat([prev[[k not in key for k in zip(prev["target"], prev["variant"])]], new])
        new.to_csv(f, index=False)
    print(out.round(4).T.to_string())
    print(rels.round(3).to_string(index=False))
