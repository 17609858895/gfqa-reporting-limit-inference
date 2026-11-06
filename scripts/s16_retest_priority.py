"""Where would more sensitive analysis pay off? Follow-up measurement priorities from the blind-spot triage.

Blind-spot non-detects (DL > standard) are triaged with the cross-fitted posterior
P(Y > standard | Y < DL, x):  likely compliant (< 0.05), likely exceeding (> 0.5), uncertain otherwise.
Per country (full modelling period, 1980 onward): blind-spot events, share resolved by triage, uncertain share, expected hidden exceedances
(sum of posteriors) and the fold reduction of the median DL needed to reach 0.3 x standard
(the EU 2009/90/EC performance criterion). Per station (events since 2015): expected hidden
exceedances among uncertain events, used as a follow-up measurement priority score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DP, RES, TAB = ROOT / "data_processed", ROOT / "results", ROOT / "tables"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07, "CdDis": 0.08}

st = pd.read_parquet(DP / "stations_clean.parquet")[["station", "wtype", "basin"]]
crows, srows = [], []
for t, thr in THRESH.items():
    d = pd.read_parquet(RES / f"crossfit_{t}.parquet")
    u = d[d["undetermined"]].copy()
    u["cls"] = np.select([u["post_exceed"] < 0.05, u["post_exceed"] > 0.5], ["compliant", "exceed"], "uncertain")
    u["dl_ratio"] = 10 ** u["y_hi"] / thr
    for c, g in u.groupby("country"):
        n_all = int((d["country"] == c).sum())
        if len(g) < 200:
            continue
        unc = g[g["cls"] == "uncertain"]
        crows.append({"target": t, "country": c, "events": n_all, "blind_spot": len(g),
                      "blind_spot_share": len(g) / n_all,
                      "resolved_share": float((g["cls"] != "uncertain").mean()),
                      "likely_compliant_share": float((g["cls"] == "compliant").mean()),
                      "likely_exceed_share": float((g["cls"] == "exceed").mean()),
                      "expected_hidden_exceedances": float(g["post_exceed"].sum()),
                      "median_dl_over_std_uncertain": float(unc["dl_ratio"].median()) if len(unc) else np.nan,
                      "dl_reduction_fold_to_0.3std": float(unc["dl_ratio"].median() / 0.3) if len(unc) else np.nan})
    recent = u[(u["date"].dt.year >= 2015) & (u["cls"] == "uncertain")]
    s = recent.groupby(["station", "country"]).agg(uncertain_events=("post_exceed", "size"),
                                                   expected_hidden_exceedances=("post_exceed", "sum"),
                                                   mean_posterior=("post_exceed", "mean"),
                                                   median_dl_over_std=("dl_ratio", "median"),
                                                   lat=("lat", "first"), lon=("lon", "first")).reset_index()
    s = s.merge(st, on="station", how="left").sort_values("expected_hidden_exceedances", ascending=False)
    s["rank"] = np.arange(1, len(s) + 1)
    s["target"] = t
    srows.append(s)

C = pd.DataFrame(crows).sort_values(["target", "expected_hidden_exceedances"], ascending=[True, False])
S = pd.concat(srows)
C.to_csv(TAB / "T_retest_priority_country.csv", index=False)
S.to_csv(TAB / "T_retest_priority_stations.csv", index=False)
pd.set_option("display.width", 250)
print(C.round(3).to_string(index=False))
for t in THRESH:
    s = S[S["target"] == t]
    top = s.head(50)
    print(f"\n{t}: {len(s)} stations with uncertain blind-spot events since 2015; top-50 carry "
          f"{top['expected_hidden_exceedances'].sum() / max(s['expected_hidden_exceedances'].sum(), 1e-9):.1%} of expected hidden exceedances")
    print(top.head(8)[["rank", "station", "country", "wtype", "uncertain_events", "expected_hidden_exceedances", "median_dl_over_std"]].round(2).to_string(index=False))
