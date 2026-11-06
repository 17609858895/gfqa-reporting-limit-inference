"""Interval-censored learning utilities for detection-limited concentrations.

Labels are log10 intervals [lo, hi]; lo = -inf for '<DL', hi = +inf for '>'.
All likelihoods use a Gaussian error on log10 scale (a Tobit / AFT-normal model).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm

LN10 = np.log(10.0)

ROUTINE = ["TEMP", "pH", "EC", "DO", "NO3N", "NH4N", "TP", "TSS", "DOC", "TOC", "Cl", "Ca",
           "Alk", "HCO3", "BOD", "COD", "TURB", "Fe", "Mn"]
LOG_FEATS = ["EC", "DO", "NO3N", "NH4N", "TP", "TSS", "DOC", "TOC", "Cl", "Ca", "Alk", "HCO3",
             "BOD", "COD", "TURB", "Fe", "Mn", "area", "discharge"]
STATION = ["lat", "lon", "elev", "area", "discharge"]
CATS = ["country", "wtype", "mtype"]


# ----------------------------------------------------------------------------- data
def load_target(dp, name, exclude=()):
    t = pd.read_parquet(dp / f"target_{name}.parquet")
    ev = pd.read_parquet(dp / "events_routine.parquet")
    st = pd.read_parquet(dp / "stations_clean.parquet")
    d = t.merge(ev, on=["station", "date"], how="left").merge(st, on="station", how="left")
    d = d[d["date"].notna() & (d["date"].dt.year >= 1980)].reset_index(drop=True)
    d["year"] = d["date"].dt.year + (d["date"].dt.dayofyear - 1) / 365.25
    doy = d["date"].dt.dayofyear.to_numpy()
    d["msin"], d["mcos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    feats = [f for f in ROUTINE if f not in exclude and f != name] + STATION + ["depth", "year", "msin", "mcos"] + CATS
    for f in LOG_FEATS:
        if f in d:
            d[f] = np.log10(d[f].clip(lower=1e-6))
    for c in CATS:
        d[c] = d[c].astype("category")
    return d, feats


def linear_matrix(d_tr, d_te, feats):
    """Median-impute + missing indicators + standardise; one-hot categoricals (train-fitted)."""
    num = [f for f in feats if f not in CATS]
    med = d_tr[num].median().fillna(0.0)
    Xs = []
    for d in (d_tr, d_te):
        x = d[num].fillna(med)
        ind = d[num].isna().astype(float).add_suffix("_na")
        cat = pd.get_dummies(d[CATS].astype(str), dtype=float)
        Xs.append(pd.concat([x, ind, cat], axis=1))
    Xtr, Xte = Xs[0], Xs[1].reindex(columns=Xs[0].columns, fill_value=0.0)
    mu, sd = Xtr.mean(), Xtr.std().fillna(1.0)
    sd = sd.where(sd > 1e-6, 1.0)
    # winsorise standardised values so linear/neural learners cannot extrapolate wildly
    return (((Xtr - mu) / sd).clip(-6, 6).to_numpy(np.float32),
            ((Xte - mu) / sd).clip(-6, 6).to_numpy(np.float32))


# ------------------------------------------------------------------- likelihood pieces
def interval_nll(mu, lo, hi, sigma):
    """Per-sample negative log-likelihood of an interval-censored Gaussian (log10 scale)."""
    mu, sigma = np.asarray(mu, float), np.broadcast_to(np.asarray(sigma, float), np.shape(mu))
    exact = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    out = np.empty_like(mu)
    z = (lo[exact] - mu[exact]) / sigma[exact]
    out[exact] = 0.5 * z**2 + np.log(sigma[exact]) + 0.5 * np.log(2 * np.pi) + np.log(LN10)  # density on log10 scale
    left = ~np.isfinite(lo) & np.isfinite(hi)
    out[left] = -log_ndtr((hi[left] - mu[left]) / sigma[left])
    right = np.isfinite(lo) & ~np.isfinite(hi)
    out[right] = -log_ndtr(-(lo[right] - mu[right]) / sigma[right])
    inter = ~(exact | left | right)
    if inter.any():
        a = ndtr((hi[inter] - mu[inter]) / sigma[inter]) - ndtr((lo[inter] - mu[inter]) / sigma[inter])
        out[inter] = -np.log(np.clip(a, 1e-300, None))
    return out


def fit_sigma(mu, lo, hi):
    """Profile MLE of a common sigma given fixed means."""
    r = minimize_scalar(lambda s: interval_nll(mu, lo, hi, np.exp(s)).mean(), bounds=(np.log(0.02), np.log(5)), method="bounded")
    return float(np.exp(r.x))


def tobit_grad_hess(mu, lo, hi, sigma):
    """Gradient/Hessian of the interval-censored Gaussian NLL w.r.t. mu."""
    g = np.zeros_like(mu)
    h = np.zeros_like(mu)
    s2 = sigma * sigma
    exact = np.isfinite(lo) & (lo == hi)
    g[exact] = (mu[exact] - lo[exact]) / s2
    h[exact] = 1.0 / s2
    left = ~np.isfinite(lo) & np.isfinite(hi)
    z = (hi[left] - mu[left]) / sigma
    lam = np.exp(norm.logpdf(z) - log_ndtr(z))  # inverse Mills ratio phi/Phi
    g[left] = lam / sigma
    h[left] = lam * (z + lam) / s2
    right = np.isfinite(lo) & ~np.isfinite(hi)
    z = (lo[right] - mu[right]) / sigma
    hz = np.exp(norm.logpdf(z) - log_ndtr(-z))  # hazard phi/(1-Phi)
    g[right] = -hz / sigma
    h[right] = hz * (hz - z) / s2
    return g, np.clip(h, 1e-6 / s2, None)


def tobit_linear(X, lo, hi, l2=1e-3, iters=60):
    """Censored Gaussian linear regression by Newton steps on (beta, log sigma) via L-BFGS."""
    from scipy.optimize import minimize
    Xb = np.c_[np.ones(len(X)), X].astype(float)
    det = np.isfinite(lo) & (lo == hi)
    y0 = np.where(np.isfinite(hi), hi, lo)
    beta0 = np.linalg.lstsq(Xb[det], y0[det], rcond=None)[0] if det.sum() > Xb.shape[1] else np.zeros(Xb.shape[1])

    def f(p):
        b, ls = p[:-1], p[-1]
        s = np.exp(ls)
        mu = Xb @ b
        nll = interval_nll(mu, lo, hi, s)
        g, _ = tobit_grad_hess(mu, lo, hi, s)
        # d nll / d log sigma by finite difference (cheap, 1-D)
        eps = 1e-4
        dls = (interval_nll(mu, lo, hi, s * np.exp(eps)).mean() - nll.mean()) / eps
        grad_b = Xb.T @ g / len(mu) + l2 * np.r_[0, b[1:]]
        return nll.mean() + 0.5 * l2 * (b[1:] ** 2).sum(), np.r_[grad_b, dls]

    res = minimize(f, np.r_[beta0, np.log(0.5)], jac=True, method="L-BFGS-B", options={"maxiter": iters * 10})
    return res.x[:-1], float(np.exp(res.x[-1]))


# ---------------------------------------------------------------------------- metrics
def interval_cindex(mu, lo, hi, n_pairs=400_000, seed=0):
    """Concordance over pairs whose ordering is identified by the interval labels."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    a_less = hi[i] < lo[j]   # y_i < y_j certainly
    b_less = hi[j] < lo[i]
    comp = a_less | b_less
    conc = (a_less & (mu[i] < mu[j])) | (b_less & (mu[j] < mu[i]))
    ties = comp & (mu[i] == mu[j])
    return float((conc.sum() + 0.5 * ties.sum()) / max(comp.sum(), 1))


def exceed_labels(lo, hi, tau):
    """1 if certainly > tau, 0 if certainly <= tau, nan if undetermined (non-detect with DL > tau)."""
    y = np.full(len(lo), np.nan)
    y[lo > tau] = 1.0
    y[hi <= tau] = 0.0
    return y


def exceed_metrics(mu, sigma, lo, hi, tau):
    from sklearn.metrics import roc_auc_score
    y = exceed_labels(lo, hi, tau)
    k = ~np.isnan(y)
    p = 1 - ndtr((tau - mu) / sigma)
    out = {"n_known": int(k.sum()), "n_undetermined": int((~k).sum()), "prev_known": float(np.nanmean(y))}
    if k.sum() > 50 and 0 < y[k].mean() < 1:
        out["auc"] = float(roc_auc_score(y[k], mu[k]))
        out["brier"] = float(np.mean((p[k] - y[k]) ** 2))
    out["p_exceed_undetermined"] = float(p[~k].mean()) if (~k).any() else np.nan
    return out
