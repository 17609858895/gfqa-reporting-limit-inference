"""Semi-synthetic ablation: how the substitution penalty scales with censoring.

Detected-only events of one contaminant are re-censored at artificial limits.
Scenarios
  level-q   : group (country x 5-yr) limit at quantile q ~ U(q0-0.1, q0+0.1), q0 in {0.2,...,0.8}
  informative: limit quantile rises with the group's median concentration
              (less sensitive methods where concentrations are high)
Compared estimators (matched features, splits and early-stopping budget):
  Sub-DL/2, Sub-DL, Drop-ND (LightGBM L2), Tobit (linear), XGB-AFT-N, LGB-Tobit, CSL over those six.
Truth-based metrics on held-out stations: RMSE of all values, bias of hidden values,
error of exceedance prevalence, and coverage/certification of the conformal upper bound.
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
from scipy.special import ndtr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from censml import fit_sigma, load_target
from ensembles import CensoredSuperLearner, conformal_upper, p_exceed
from learners import lgb_l2, lgb_tobit, subst, tobit_lin, xgb_aft

ROOT = Path(__file__).resolve().parents[1]
DP, RES = ROOT / "data_processed", ROOT / "results"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07, "Ni": 4.0, "NH4N": 0.2, "PFOS": 0.00065}


def censor(d, y, q0, informative, seed):
    rng = np.random.default_rng(seed)
    grp = (d["country"].astype(str) + "_" + (d["date"].dt.year // 5).astype(str)).to_numpy()
    med = pd.Series(y).groupby(grp).transform("median").to_numpy()
    rank = pd.Series(med).rank(pct=True).to_numpy()
    dl = np.empty(len(y))
    for g in np.unique(grp):
        m = grp == g
        q = q0 + (0.3 * (rank[m][0] - 0.5) if informative else rng.uniform(-0.1, 0.1))
        dl[m] = np.quantile(y[m], np.clip(q, 0.02, 0.98))
    cen = y < dl
    return np.where(cen, -np.inf, y), np.where(cen, dl, y)


def drop_nd(Xtr, lotr, hitr, Xva, lova, hiva):
    a, b = np.isfinite(lotr), np.isfinite(lova)
    return lgb_l2(Xtr[a], lotr[a], hitr[a], Xva[b], lova[b], hiva[b])


def sub_dl(Xtr, lotr, hitr, Xva, lova, hiva):
    return lgb_l2(Xtr, lotr, np.where(np.isfinite(lotr), hitr, hitr + np.log10(2)), Xva, lova,
                  np.where(np.isfinite(lova), hiva, hiva + np.log10(2)))


def run(t, n_max=50_000, seeds=(0,)):
    d, feats = load_target(DP, t)
    d = d[np.isfinite(d["y_lo"])].reset_index(drop=True)
    if len(d) > n_max:
        st = d["station"].unique()
        keep = set(np.random.default_rng(1).choice(st, int(len(st) * n_max / len(d)), replace=False))
        d = d[d["station"].isin(keep)].reset_index(drop=True)
    y, g, X = d["y_lo"].to_numpy(), d["station"].to_numpy(), d[feats]
    tau = np.log10(THRESH[t])
    methods = {"Sub-DL/2": lgb_l2, "Sub-DL": sub_dl, "Drop-ND": drop_nd, "Tobit": tobit_lin(feats),
               "XGB-AFT-N": xgb_aft("normal"), "LGB-Tobit": lgb_tobit}
    rows = []
    scen = [(f"level-{q:.1f}", q, False) for q in (0.2, 0.35, 0.5, 0.65, 0.8)] + [("informative-0.5", 0.5, True)]
    for name, q0, inf in scen:
        for seed in seeds:
            lo, hi = censor(d, y, q0, inf, seed)
            dev, test = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(d, groups=g))
            oof = np.zeros((len(dev), len(methods)))
            tst = np.zeros((len(test), len(methods)))
            for k, (tr, va) in enumerate(GroupKFold(3).split(dev, groups=g[dev])):
                tr_abs, va_abs = dev[tr], dev[va]
                itr, ival = next(GroupShuffleSplit(1, test_size=0.12, random_state=k).split(tr_abs, groups=g[tr_abs]))
                a, b = tr_abs[itr], tr_abs[ival]
                for j, f in enumerate(methods.values()):
                    p = f(X.iloc[a], lo[a], hi[a], X.iloc[b], lo[b], hi[b])
                    oof[va, j] = p(X.iloc[va_abs])
                    tst[:, j] += p(X.iloc[test]) / 3
            csl = CensoredSuperLearner().fit(oof, lo[dev], hi[dev])
            preds = {nm: (tst[:, j], fit_sigma(oof[:, j], lo[dev], hi[dev])) for j, nm in enumerate(methods)}
            mu, s = csl.predict(tst)
            preds["CSL"] = (mu, s)
            yt, lt, ht = y[test], lo[test], hi[test]
            hid = ~np.isfinite(lt)
            cal, ev = next(GroupShuffleSplit(1, test_size=0.5, random_state=7).split(test, groups=g[test]))
            for nm, (mu, s) in preds.items():
                s = np.broadcast_to(s, mu.shape)
                up, _, _ = conformal_upper(mu[cal], s[cal], ht[cal], 0.1)
                U = np.minimum(up(mu[ev], s[ev]), ht[ev])
                und = (~np.isfinite(lt[ev])) & (ht[ev] > tau)
                cert = und & (U <= tau)
                # posterior exceedance of undetermined samples given Y < DL (truth known here)
                lt_e, ht_e, yt_e, mu_e, s_e = lt[ev], ht[ev], yt[ev], mu[ev], s[ev]
                pdl = ndtr((ht_e - mu_e) / s_e)
                post = (pdl - ndtr((tau - mu_e) / s_e)) / np.clip(pdl, 1e-12, None)
                post_u, true_u = post[und], (yt_e[und] > tau).astype(float)
                post_metrics = {"post_mean_und": float(post_u.mean()) if und.any() else np.nan,
                                "true_exc_und": float(true_u.mean()) if und.any() else np.nan,
                                "post_brier_und": float(np.mean((post_u - true_u) ** 2)) if und.any() else np.nan,
                                "likely_compliant_share": float(np.mean(post_u < 0.05)) if und.any() else np.nan,
                                "likely_compliant_false": float(true_u[post_u < 0.05].mean()) if (post_u < 0.05).any() else np.nan}
                rows.append({**post_metrics,"target": t, "scenario": name, "seed": seed, "model": nm,
                             "cens_share": float(np.mean(~np.isfinite(lo))),
                             "rmse_truth": float(np.sqrt(np.mean((mu - yt) ** 2))),
                             "rmse_hidden": float(np.sqrt(np.mean((mu[hid] - yt[hid]) ** 2))),
                             "bias_hidden": float(np.mean(mu[hid] - yt[hid])),
                             "prev_true": float(np.mean(yt > tau)),
                             "prev_pred": float(np.mean(p_exceed(mu, s, tau))),
                             "cov_true": float(np.mean(yt[ev] <= U)),
                             "undetermined_share": float(und.mean()),
                             "certified_share": float(cert.sum() / max(und.sum(), 1)),
                             "false_cert": float(np.mean(yt[ev][cert] > tau)) if cert.any() else np.nan})
            print(t, name, seed, flush=True)
            pd.DataFrame(rows).to_csv(RES / f"ablation_{t}.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=["Pb"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    a = ap.parse_args()
    for t in a.targets:
        r = run(t, seeds=tuple(a.seeds))
        pd.set_option("display.width", 220)
        print(r.groupby(["scenario", "model"])[["rmse_truth", "bias_hidden", "prev_pred", "prev_true"]].agg(["mean", "std"]).round(3).to_string())
