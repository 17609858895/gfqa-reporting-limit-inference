"""Leave-one-country-out transfer and conformal coverage under shift.

For each of the largest countries of a target: train on all other countries (station-sampled),
calibrate the censoring-robust conformal bound on held-out stations of the training countries,
and evaluate on the unseen country. The country identifier is unseen at test time by design,
so this also probes reliance on provenance.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from censml import exceed_metrics, fit_sigma, interval_cindex, interval_nll, load_target
from ensembles import CensoredSuperLearner, conformal_upper, coverage_bounds
from learners import lgb_l2, lgb_tobit, xgb_aft

ROOT = Path(__file__).resolve().parents[1]
DP, RES = ROOT / "data_processed", ROOT / "results"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07, "Ni": 4.0, "NH4N": 0.2, "PFOS": 0.00065, "CdDis": 0.08}


def sample_stations(d, n, seed=1):
    if len(d) <= n:
        return d
    st = d["station"].unique()
    keep = set(np.random.default_rng(seed).choice(st, int(len(st) * n / len(d)), replace=False))
    return d[d["station"].isin(keep)]


def run(t, n_train=50_000, n_countries=8):
    d, feats = load_target(DP, t)
    tau = np.log10(THRESH[t])
    cnt = d.groupby("country", observed=True).agg(n=("cens", "size"), nd=("cens", lambda s: (s == 1).mean()))
    cnt = cnt[(cnt.n >= 3000) & (cnt.nd > 0.05) & (cnt.nd < 0.95)].sort_values("n", ascending=False).head(n_countries)
    rows = []
    for c in cnt.index:
        te = d[d["country"] == c]
        te = sample_stations(te, 20_000)
        tr = sample_stations(d[d["country"] != c], n_train).reset_index(drop=True)
        g = tr["station"].to_numpy()
        fit_idx, cal_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(tr, groups=g))
        a_idx, v_idx = next(GroupShuffleSplit(1, test_size=0.15, random_state=1).split(fit_idx, groups=g[fit_idx]))
        a, v = fit_idx[a_idx], fit_idx[v_idx]
        X, lo, hi = tr[feats], tr["y_lo"].to_numpy(), tr["y_hi"].to_numpy()
        Xt, lot, hit = te[feats], te["y_lo"].to_numpy(), te["y_hi"].to_numpy()
        fits = {"Sub-DL/2": lgb_l2, "XGB-AFT-N": xgb_aft("normal"), "LGB-Tobit": lgb_tobit}
        P_cal, P_te, P_v = {}, {}, {}
        for nm, f in fits.items():
            p = f(X.iloc[a], lo[a], hi[a], X.iloc[v], lo[v], hi[v])
            P_v[nm], P_cal[nm], P_te[nm] = p(X.iloc[v]), p(X.iloc[cal_idx]), p(Xt)
        Zv = np.c_[[P_v[k] for k in fits]].T
        csl = CensoredSuperLearner().fit(Zv, lo[v], hi[v])
        out = {nm: (P_cal[nm], P_te[nm], fit_sigma(P_v[nm], lo[v], hi[v])) for nm in fits}
        mc, sc = csl.predict(np.c_[[P_cal[k] for k in fits]].T)
        mt, st_ = csl.predict(np.c_[[P_te[k] for k in fits]].T)
        out["CSL"] = (mc, mt, float(sc[0]))
        for nm, (mc, mt, s) in out.items():
            up, q, _ = conformal_upper(mc, np.full(len(mc), s), hi[cal_idx], 0.1)
            U = np.minimum(up(mt, np.full(len(mt), s)), hit)
            cl, ch = coverage_bounds(U, lot, hit)
            und = (~np.isfinite(lot)) & (hit > tau)
            r = {"target": t, "country": c, "n_test": len(te), "nd_share": float(np.mean(~np.isfinite(lot))),
                 "model": nm, "cindex": interval_cindex(mt, lot, hit), "cnll": float(interval_nll(mt, lot, hit, s).mean()),
                 "cov_lower": cl, "cov_upper": ch, "certified_share": float((und & (U <= tau)).sum() / max(und.sum(), 1)),
                 "n_undetermined": int(und.sum())}
            r.update({f"exc_{k}": v_ for k, v_ in exceed_metrics(mt, s, lot, hit, tau).items()})
            rows.append(r)
        print(t, c, flush=True)
        pd.DataFrame(rows).to_csv(RES / f"transfer_{t}.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["Pb"]:
        r = run(t)
        pd.set_option("display.width", 220)
        print(r.pivot_table(index="country", columns="model", values=["cindex", "cov_lower"]).round(3).to_string())
