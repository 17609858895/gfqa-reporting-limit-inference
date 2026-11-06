"""LOD/LOQ inconsistency sensitivity: re-label detected values that fall below their own reporting limit.

Within a station x year x analysis-method group, non-detects report one limit in 93% of groups. A
detected value below the largest limit reported as '<' in its group was most likely quantified
between the detection and quantification limits. These values are re-labelled as left-censored at
that group limit and written as target_<t>LQ.parquet for refitting with s09.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DP, RES = ROOT / "data_processed", ROOT / "results"

if __name__ == "__main__":
    log = []
    for t in sys.argv[1:] or ["Cd", "Pb", "Hg"]:
        d = pd.read_parquet(DP / f"target_{t}.parquet")
        d["yr"] = d["date"].dt.year
        key = ["station", "yr", "method"]
        gdl = d[d["cens"] == 1].groupby(key)["y_hi"].max().rename("gdl")
        d = d.merge(gdl, left_on=key, right_index=True, how="left")
        flip = (d["cens"] == 0) & d["gdl"].notna() & (d["y_lo"] < d["gdl"])
        d.loc[flip, "y_lo"] = -np.inf
        d.loc[flip, "y_hi"] = d.loc[flip, "gdl"]
        d.loc[flip, "cens"] = 1
        d.drop(columns=["yr", "gdl"]).to_parquet(DP / f"target_{t}LQ.parquet", index=False)
        log.append({"target": t, "events": len(d), "relabelled": int(flip.sum()), "relabelled_share": float(flip.mean()),
                    "nd_share_before": float(((d["cens"] == 1) & ~flip).mean()), "nd_share_after": float((d["cens"] == 1).mean())})
        print(log[-1], flush=True)
    (RES / "s17_lodloq_log.json").write_text(json.dumps(log, indent=1))
