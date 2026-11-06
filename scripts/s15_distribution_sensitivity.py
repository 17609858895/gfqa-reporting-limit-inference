"""Error-distribution sensitivity of the blind-spot posterior.

Cross-fitted (5-fold, station-grouped) XGBoost-AFT with normal, logistic and extreme-value
(Gumbel-minimum) errors on log10 concentration. The scale of each distribution is profiled by
interval-censored maximum likelihood on held-out stations of each training fold. For every model:
  * exceedance prevalence (known exceedances + posterior P(Y > tau | Y < DL, x) of blind-spot non-detects)
  * real-data re-censoring check (same protocol and seed as s11).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_ndtr, ndtr
from scipy.stats import norm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from censml import LN10, load_target  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DP, RES, TAB = ROOT / "data_processed", ROOT / "results", ROOT / "tables"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07}


# ---------------------------------------------------------------- distributions (standardised)
def cdf(z, dist):
    if dist == "normal":
        return ndtr(z)
    if dist == "logistic":
        return expit(z)
    return -np.expm1(-np.exp(np.clip(z, -50, 50)))          # extreme value (Gumbel minimum)


def logcdf(z, dist):
    if dist == "normal":
        return log_ndtr(z)
    if dist == "logistic":
        return -np.logaddexp(0, -z)
    e = np.exp(np.clip(z, -50, 50))
    return np.log(np.clip(-np.expm1(-e), 1e-300, None))


def logsf(z, dist):
    if dist == "normal":
        return log_ndtr(-z)
    if dist == "logistic":
        return -np.logaddexp(0, z)
    return -np.exp(np.clip(z, -50, 50))


def logpdf(z, dist):
    if dist == "normal":
        return norm.logpdf(z)
    if dist == "logistic":
        return -z - 2 * np.logaddexp(0, -z)
    return z - np.exp(np.clip(z, -50, 50))


def nll(mu, lo, hi, s, dist):
    s = np.broadcast_to(np.asarray(s, float), np.shape(mu))
    exact = np.isfinite(lo) & (lo == hi)
    left = ~np.isfinite(lo) & np.isfinite(hi)
    right = np.isfinite(lo) & ~np.isfinite(hi)
    out = np.zeros(len(mu))
    out[exact] = -(logpdf((lo[exact] - mu[exact]) / s[exact], dist) - np.log(s[exact]))
    out[left] = -logcdf((hi[left] - mu[left]) / s[left], dist)
    out[right] = -logsf((lo[right] - mu[right]) / s[right], dist)
    return out.mean()


def fit_scale(mu, lo, hi, dist):
    r = minimize_scalar(lambda ls: nll(mu, lo, hi, np.exp(ls), dist), bounds=(np.log(0.02), np.log(5)), method="bounded")
    return float(np.exp(r.x))


def xgb_aft_fit(X, lo, hi, Xv, lov, hiv, dist):
    import xgboost as xgb

    def dm(Xm, l, h):
        d = xgb.DMatrix(Xm, enable_categorical=True)
        d.set_float_info("label_lower_bound", np.where(np.isfinite(l), 10.0 ** l, 0.0))
        d.set_float_info("label_upper_bound", np.where(np.isfinite(h), 10.0 ** h, np.inf))
        return d
    p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution=dist, aft_loss_distribution_scale=1.0,
             tree_method="hist", eta=0.08, max_depth=8, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
             nthread=8, max_cat_to_onehot=1)
    m = xgb.train(p, dm(X, lo, hi), 2000, evals=[(dm(Xv, lov, hiv), "va")], early_stopping_rounds=60, verbose_eval=False)
    return lambda Z: m.predict(xgb.DMatrix(Z, enable_categorical=True), output_margin=True,
                               iteration_range=(0, m.best_iteration + 1)) / LN10


def posterior(mu, s, dl, tau, dist):
    pdl = cdf((dl - mu) / s, dist)
    return (pdl - cdf((tau - mu) / s, dist)) / np.clip(pdl, 1e-12, None)


def run(t, seed=0):
    d, feats = load_target(DP, t)
    tau = np.log10(THRESH[t])
    X, lo, hi, g = d[feats], d["y_lo"].to_numpy(), d["y_hi"].to_numpy(), d["station"].to_numpy()
    country = d["country"].astype(str).to_numpy()
    rows = []
    for dist in ["normal", "logistic", "extreme"]:
        mu, sig = np.zeros(len(d)), np.zeros(len(d))
        for k, (tr, te) in enumerate(GroupKFold(5).split(d, groups=g)):
            a_, v_ = next(GroupShuffleSplit(1, test_size=0.1, random_state=k).split(tr, groups=g[tr]))
            a, v = tr[a_], tr[v_]
            f = xgb_aft_fit(X.iloc[a], lo[a], hi[a], X.iloc[v], lo[v], hi[v], dist)
            sig[te] = fit_scale(f(X.iloc[v]), lo[v], hi[v], dist)
            mu[te] = f(X.iloc[te])
        nd = ~np.isfinite(lo)
        und = nd & (hi > tau)
        known = (~nd) & (lo > tau)
        post = np.where(und, posterior(mu, sig, hi, tau, dist), 0.0)
        # real-data re-censoring at real blind-spot limits (protocol of s11)
        rng = np.random.default_rng(seed)
        det_idx = np.flatnonzero(~nd)
        pools = {c: hi[und & (country == c)] for c in np.unique(country[und])}
        det_idx = det_idx[np.isin(country[det_idx], list(pools))]
        dl_em = np.array([pools[c][rng.integers(len(pools[c]))] for c in country[det_idx]])
        keep = lo[det_idx] < dl_em
        idx, dl_em = det_idx[keep], dl_em[keep]
        pe = posterior(mu[idx], sig[idx], dl_em, tau, dist)
        truth = (lo[idx] > tau).astype(float)
        rows.append({"target": t, "dist": dist, "scale_median": float(np.median(sig)),
                     "cv_censored_nll": float(nll(mu, lo, hi, sig, dist)),
                     "prev_model": float((known + post).mean()), "mean_post_undetermined": float(post[und].mean()),
                     "pseudo_true": float(truth.mean()), "pseudo_mean_post": float(pe.mean()),
                     "pseudo_brier": float(np.mean((pe - truth) ** 2)),
                     "likely_compliant_share": float(np.mean(pe < 0.05)),
                     "false_compliance": float(truth[pe < 0.05].mean()) if (pe < 0.05).any() else np.nan,
                     "likely_exceed_share": float(np.mean(pe > 0.5)),
                     "precision_exceed": float(truth[pe > 0.5].mean()) if (pe > 0.5).any() else np.nan})
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    out = []
    for t in sys.argv[1:] or ["Cd", "Pb", "Hg"]:
        out += run(t)
        pd.DataFrame(out).to_csv(TAB / "T_distribution_sensitivity.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out).round(4).to_string(index=False))
