"""Stacking under left-censoring and censoring-robust conformal upper bounds.

Censored super learner (CSL):
    mu_i = b + sum_k w_k z_ik,  w_k >= 0
    sigma_i = exp(c0)                                   (CSL)
    sigma_i = exp(c0 + c1 * log sd_i)                   (D-CSL, sd_i = spread of base predictions)
    parameters maximise the interval-censored Gaussian likelihood of out-of-fold predictions.

Censoring-robust conformal upper bound:
    scores use the upper end of each label interval, s_i = (hi_i - mu_i) / sigma_i.
    For a non-detect hi_i = DL_i >= y_i, so s_i is never smaller than the true score and
    the (1-alpha) conformal quantile can only move up: marginal coverage >= 1-alpha.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from censml import interval_nll, tobit_grad_hess


def _spread(Z):
    return np.log(np.std(Z, axis=1) + 1e-3)


class CensoredSuperLearner:
    def __init__(self, hetero=False, l2=1e-4):
        self.hetero, self.l2 = hetero, l2

    def _unpack(self, p, K):
        a, b = p[:K], p[K]
        c = p[K + 1:]
        return np.exp(a), b, c

    def _sigma(self, c, Z):
        if self.hetero:
            return np.exp(c[0] + c[1] * _spread(Z))
        return np.full(len(Z), np.exp(c[0]))

    def fit(self, Z, lo, hi):
        n, K = Z.shape
        nc = 2 if self.hetero else 1
        y0 = np.where(np.isfinite(lo), lo, hi - np.log10(2))
        p0 = np.r_[np.full(K, np.log(1.0 / K)), 0.0, np.log(np.std(y0)), np.zeros(nc - 1)]

        def f(p):
            w, b, c = self._unpack(p, K)
            mu = b + Z @ w
            s = self._sigma(c, Z)
            nll = interval_nll(mu, lo, hi, s).mean() + self.l2 * (w ** 2).sum()
            g = np.zeros_like(p)
            # analytic gradient for weights/intercept; finite differences for the 1-2 scale parameters
            gm = _grad_mu(mu, lo, hi, s)
            g[:K] = (Z.T @ gm / n + 2 * self.l2 * w) * w
            g[K] = gm.mean()
            eps = 1e-5
            for j in range(nc):
                pj = p.copy(); pj[K + 1 + j] += eps
                _, _, cj = self._unpack(pj, K)
                g[K + 1 + j] = (interval_nll(mu, lo, hi, self._sigma(cj, Z)).mean() - (nll - self.l2 * (w ** 2).sum())) / eps
            return nll, g

        r = minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        self.w, self.b, self.c = self._unpack(r.x, K)
        return self

    def predict(self, Z):
        return self.b + Z @ self.w, self._sigma(self.c, Z)


def _grad_mu(mu, lo, hi, s):
    """d NLL / d mu for per-sample sigma (vectorised over unique sigma via broadcasting)."""
    g = np.zeros_like(mu)
    exact = np.isfinite(lo) & (lo == hi)
    g[exact] = (mu[exact] - lo[exact]) / s[exact] ** 2
    from scipy.special import log_ndtr
    from scipy.stats import norm
    left = ~np.isfinite(lo) & np.isfinite(hi)
    z = (hi[left] - mu[left]) / s[left]
    g[left] = np.exp(norm.logpdf(z) - log_ndtr(z)) / s[left]
    right = np.isfinite(lo) & ~np.isfinite(hi)
    z = (lo[right] - mu[right]) / s[right]
    g[right] = -np.exp(norm.logpdf(z) - log_ndtr(-z)) / s[right]
    return g


def standard_stack(Z, lo, hi):
    """Conventional stacking: non-negative linear meta-model on DL/2-substituted labels."""
    from sklearn.linear_model import LinearRegression
    y = np.where(np.isfinite(lo), lo, hi - np.log10(2))
    try:
        return LinearRegression(positive=True).fit(Z, y)
    except ValueError:  # scipy NNLS edge case; fall back to bounded least squares
        from scipy.optimize import lsq_linear
        A = np.c_[Z, np.ones(len(Z))]
        r = lsq_linear(A, y, bounds=(np.r_[np.zeros(Z.shape[1]), -np.inf], np.inf))
        m = LinearRegression()
        m.coef_, m.intercept_ = r.x[:-1], r.x[-1]
        return m


# ------------------------------------------------------------------- conformal
def conformal_upper(mu_cal, s_cal, hi_cal, alpha=0.1, groups_cal=None, min_group=200):
    """Return a function (mu, sigma, groups) -> upper bound using censoring-robust scores."""
    sc = (hi_cal - mu_cal) / s_cal

    def q(scores):
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - alpha)))
        return np.inf if k > n else np.sort(scores)[k - 1]
    q_all = q(sc)
    q_g = {}
    if groups_cal is not None:
        for g in np.unique(groups_cal):
            m = groups_cal == g
            if m.sum() >= min_group:
                q_g[g] = q(sc[m])

    def upper(mu, s, groups=None):
        if groups is None or not q_g:
            return mu + q_all * s
        qq = np.array([q_g.get(g, q_all) for g in groups])
        return mu + qq * s
    return upper, q_all, q_g


def coverage_bounds(U, lo, hi):
    """Identified bounds on P(y <= U) from interval labels: certain if hi<=U, impossible if lo>U."""
    sure = hi <= U
    fail = lo > U
    return float(sure.mean()), float(1 - fail.mean())


def parametric_upper(mu, s, alpha=0.1):
    from scipy.stats import norm
    return mu + norm.ppf(1 - alpha) * s


def p_exceed(mu, s, tau):
    return 1 - ndtr((tau - mu) / s)
