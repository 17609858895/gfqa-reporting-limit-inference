"""Feasibility benchmark: substitution vs censoring-aware boosting.

Two settings per target, station-grouped 3-fold CV (feasibility run):
  real  - observed non-detects; metrics identified by interval labels
  synth - detected-only events re-censored at heterogeneous artificial limits
          (country x 5-yr block draws q ~ U(0.2, 0.8)); truth known for every row
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from censml import (LN10, exceed_metrics, fit_sigma, interval_cindex, interval_nll, load_target,
                    tobit_grad_hess)

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
RES = ROOT / "results"
TAU = {"Pb": np.log10(1.2), "Cd": np.log10(0.25), "Hg": np.log10(0.07), "Ni": np.log10(4.0),
       "As": np.log10(10.0), "NH4N": np.log10(0.3), "TP": np.log10(0.1)}
LGB_P = dict(learning_rate=0.1, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)


def lgb_l2(Xtr, ytr, Xva, yva):
    m = lgb.train({**LGB_P, "objective": "l2"}, lgb.Dataset(Xtr, ytr), 1500,
                  valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return lambda X: m.predict(X, num_iteration=m.best_iteration)


def lgb_tobit(Xtr, lotr, hitr, Xva, lova, hiva, sigma0):
    det = np.isfinite(lotr) & (lotr == hitr)
    mu0 = float(np.median(np.where(det, lotr, hitr)))
    sigma = sigma0
    for _ in range(2):  # alternate trees | sigma
        def fobj(preds, data, s=sigma):
            return tobit_grad_hess(preds + mu0, lotr, hitr, s)

        def feval(preds, data, s=sigma):
            return "cnll", float(interval_nll(preds + mu0, lova, hiva, s).mean()), False

        dtr = lgb.Dataset(Xtr, np.zeros(len(Xtr)), free_raw_data=False)
        dva = lgb.Dataset(Xva, np.zeros(len(Xva)), reference=dtr)
        m = lgb.train({**LGB_P, "objective": fobj, "metric": "None"}, dtr, 1500, valid_sets=[dva], feval=feval,
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        sigma = fit_sigma(m.predict(Xva, num_iteration=m.best_iteration) + mu0, lova, hiva)
    return (lambda X: m.predict(X, num_iteration=m.best_iteration) + mu0), sigma


def xgb_aft(Xtr, lotr, hitr, Xva, lova, hiva, scale=1.0):
    def dm(X, lo, hi):
        d = xgb.DMatrix(X, enable_categorical=True)
        d.set_float_info("label_lower_bound", np.where(np.isfinite(lo), 10.0 ** lo, 0.0))
        d.set_float_info("label_upper_bound", np.where(np.isfinite(hi), 10.0 ** hi, np.inf))
        return d
    p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution="normal",
             aft_loss_distribution_scale=scale, tree_method="hist", eta=0.1, max_depth=8,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, nthread=16, max_cat_to_onehot=1)
    dva = dm(Xva, lova, hiva)
    m = xgb.train(p, dm(Xtr, lotr, hitr), 1500, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
    f = lambda X: m.predict(xgb.DMatrix(X, enable_categorical=True), output_margin=True,
                            iteration_range=(0, m.best_iteration + 1)) / LN10
    return f, fit_sigma(f(Xva), lova, hiva)


def run_methods(X, lo, hi, groups, tau, truth=None):
    rows = []
    for k, (tr, te) in enumerate(GroupKFold(3).split(X, groups=groups)):
        itr, iva = next(GroupShuffleSplit(1, test_size=0.12, random_state=k).split(tr, groups=groups[tr]))
        tr_, va_ = tr[itr], tr[iva]
        Xtr, Xva, Xte = X.iloc[tr_], X.iloc[va_], X.iloc[te]
        lotr, hitr, lova, hiva, lote, hite = lo[tr_], hi[tr_], lo[va_], hi[va_], lo[te], hi[te]
        det_tr, det_va = np.isfinite(lotr), np.isfinite(lova)
        preds = {}
        t0 = time.time()
        for nm, shift in [("Sub-DL/2", np.log10(2)), ("Sub-DL", 0.0), ("Sub-DL/√2", np.log10(2) / 2)]:
            ytr, yva = np.where(det_tr, lotr, hitr - shift), np.where(det_va, lova, hiva - shift)
            f = lgb_l2(Xtr, ytr, Xva, yva)
            preds[nm] = (f(Xte), fit_sigma(f(Xva), lova, hiva))
        f = lgb_l2(Xtr[det_tr], lotr[det_tr], Xva[det_va], lova[det_va])
        preds["Drop-ND"] = (f(Xte), fit_sigma(f(Xva), lova, hiva))
        f, s = xgb_aft(Xtr, lotr, hitr, Xva, lova, hiva)
        preds["XGB-AFT"] = (f(Xte), s)
        f, s = lgb_tobit(Xtr, lotr, hitr, Xva, lova, hiva, sigma0=preds["Sub-DL/2"][1])
        preds["LGB-Tobit"] = (f(Xte), s)
        for nm, (mu, s) in preds.items():
            r = {"fold": k, "method": nm, "sigma": s, "cindex": interval_cindex(mu, lote, hite),
                 "cnll": float(interval_nll(mu, lote, hite, s).mean())}
            d = np.isfinite(lote)
            r["rmse_detected"] = float(np.sqrt(np.mean((mu[d] - lote[d]) ** 2)))
            r.update({f"exc_{a}": b for a, b in exceed_metrics(mu, s, lote, hite, tau).items()})
            if truth is not None:
                yt = truth[te]
                hid = ~np.isfinite(lote)
                r["rmse_truth_all"] = float(np.sqrt(np.mean((mu - yt) ** 2)))
                r["rmse_truth_hidden"] = float(np.sqrt(np.mean((mu[hid] - yt[hid]) ** 2)))
                r["bias_truth_hidden"] = float(np.mean(mu[hid] - yt[hid]))
                r["prev_true"] = float(np.mean(yt > tau))
                from scipy.special import ndtr
                r["prev_pred"] = float(np.mean(1 - ndtr((tau - mu) / s)))
            rows.append(r)
        print(f"  fold {k} done {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def synth_censor(d, lo, seed=0):
    rng = np.random.default_rng(seed)
    key = d["country"].astype(str) + "_" + (d["date"].dt.year // 5).astype(str)
    q = key.map({k: rng.uniform(0.2, 0.8) for k in key.unique()}).to_numpy()
    # group-specific limit at quantile q of the group's detected distribution
    grp = key.to_numpy()
    dl = np.empty(len(lo))
    for g in np.unique(grp):
        m = grp == g
        dl[m] = np.quantile(lo[m], q[m][0])
    cen = lo < dl
    return np.where(cen, -np.inf, lo), np.where(cen, dl, lo)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["Pb", "NH4N"]
    out = []
    for t in targets:
        d, feats = load_target(DP, t)
        if len(d) > 220_000:
            st = d["station"].unique()
            rng = np.random.default_rng(1)
            keep = set(rng.choice(st, int(len(st) * 220_000 / len(d)), replace=False))
            d = d[d["station"].isin(keep)].reset_index(drop=True)
        X, lo, hi, g = d[feats], d["y_lo"].to_numpy(), d["y_hi"].to_numpy(), d["station"].to_numpy()
        print(t, len(d), "censored", np.mean(~np.isfinite(lo)).round(3), flush=True)
        r = run_methods(X, lo, hi, g, TAU[t])
        r["target"], r["setting"] = t, "real"
        out.append(r)
        det = np.isfinite(lo)
        ds_ = d[det].reset_index(drop=True)
        slo, shi = synth_censor(ds_, lo[det])
        print(t, "synthetic censored share", np.mean(~np.isfinite(slo)).round(3), flush=True)
        r = run_methods(ds_[feats], slo, shi, ds_["station"].to_numpy(), TAU[t], truth=lo[det])
        r["target"], r["setting"] = t, "synth"
        out.append(r)
        res = pd.concat(out)
        res.to_csv(RES / "s04_quick_benchmark_folds.csv", index=False)
        agg = res.groupby(["target", "setting", "method"]).mean(numeric_only=True).drop(columns="fold")
        print(agg.round(3).to_string(), flush=True)
