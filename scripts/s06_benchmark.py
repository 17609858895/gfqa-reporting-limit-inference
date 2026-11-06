"""22-learner benchmark + censored stacking + censoring-robust conformal certification.

Design per target (and per setting):
  * stations split 80/20 into development / held-out test (GroupShuffleSplit, seed 2026)
  * 5-fold station-grouped CV on development -> out-of-fold (OOF) predictions for stacking;
    test predictions are the average of the five fold models (test never used for fitting)
  * inner 12% station split of each training fold for early stopping and sigma profiling
  * ensembles fitted on OOF only; test stations split in half for conformal calibration/evaluation
Settings:
  real  - observed non-detects (metrics identified by interval labels)
  synth - detected-only events re-censored at heterogeneous artificial limits (truth known)

usage: python s06_benchmark.py real Cd Pb ...   |   python s06_benchmark.py synth Cd ...
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, str(Path(__file__).parent))
from censml import exceed_metrics, fit_sigma, interval_cindex, interval_nll, load_target
from ensembles import (CensoredSuperLearner, conformal_upper, coverage_bounds, p_exceed, parametric_upper,
                       standard_stack)
from learners import make_learners

ROOT = Path(__file__).resolve().parents[1]
DP, RES = ROOT / "data_processed", ROOT / "results"
(RES / "preds").mkdir(parents=True, exist_ok=True)
THRESH = {"Cd": 0.08, "Pb": 1.2, "Ni": 4.0, "Hg": 0.07, "As": 10.0, "PFOS": 0.00065, "Atrazine": 0.6,
          "NH4N": 0.2, "TP": 0.1, "Glyphosate": 0.1, "PFOA": 0.1, "NO2N": 0.03}
N_MAX = int(__import__("os").environ.get("N_MAX", 150_000))
ALPHA = 0.10


def synth_censor(d, lo, seed=0):
    rng = np.random.default_rng(seed)
    grp = (d["country"].astype(str) + "_" + (d["date"].dt.year // 5).astype(str)).to_numpy()
    dl = np.empty(len(lo))
    for g in np.unique(grp):
        m = grp == g
        dl[m] = np.quantile(lo[m], rng.uniform(0.2, 0.8))
    cen = lo < dl
    return np.where(cen, -np.inf, lo), np.where(cen, dl, lo)


def evaluate(mu, s, lo, hi, tau, truth=None):
    s = np.broadcast_to(s, mu.shape)
    r = {"cindex": interval_cindex(mu, lo, hi), "cnll": float(interval_nll(mu, lo, hi, s).mean())}
    det = np.isfinite(lo)
    r["rmse_detected"] = float(np.sqrt(np.mean((mu[det] - lo[det]) ** 2)))
    ss = 1 - np.sum((lo[det] - mu[det]) ** 2) / np.sum((lo[det] - lo[det].mean()) ** 2)
    r["r2_detected"] = float(ss)
    r.update({f"exc_{a}": b for a, b in exceed_metrics(mu, s, lo, hi, tau).items()})
    if truth is not None:
        hid = ~det
        r["rmse_truth"] = float(np.sqrt(np.mean((mu - truth) ** 2)))
        r["r2_truth"] = float(1 - np.sum((truth - mu) ** 2) / np.sum((truth - truth.mean()) ** 2))
        r["rmse_hidden"] = float(np.sqrt(np.mean((mu[hid] - truth[hid]) ** 2)))
        r["bias_hidden"] = float(np.mean(mu[hid] - truth[hid]))
        r["prev_true"] = float(np.mean(truth > tau))
        r["prev_pred"] = float(np.mean(p_exceed(mu, s, tau)))
    return r


def run(t, setting):
    t0 = time.time()
    d, feats = load_target(DP, t)
    if len(d) > N_MAX:
        st = d["station"].unique()
        keep = set(np.random.default_rng(1).choice(st, int(len(st) * N_MAX / len(d)), replace=False))
        d = d[d["station"].isin(keep)].reset_index(drop=True)
    lo, hi = d["y_lo"].to_numpy(), d["y_hi"].to_numpy()
    truth = None
    if setting == "synth":
        det = np.isfinite(lo)
        d = d[det].reset_index(drop=True)
        truth = lo[det]
        lo, hi = synth_censor(d, truth)
    tau = np.log10(THRESH[t])
    g = d["station"].to_numpy()
    dev, test = next(GroupShuffleSplit(1, test_size=0.2, random_state=2026).split(d, groups=g))
    X = d[feats]
    learners = make_learners(feats)
    names = list(learners)
    K = len(names)
    oof = np.full((len(dev), K), np.nan)
    tst = np.zeros((len(test), K))
    cost = {n: 0.0 for n in names}
    fails = {}
    for k, (tr, va) in enumerate(GroupKFold(5).split(dev, groups=g[dev])):
        tr_abs, va_abs = dev[tr], dev[va]
        itr, ival = next(GroupShuffleSplit(1, test_size=0.12, random_state=k).split(tr_abs, groups=g[tr_abs]))
        a, b = tr_abs[itr], tr_abs[ival]
        for j, nm in enumerate(names):
            if nm in fails:
                continue
            t1 = time.time()
            try:
                f = learners[nm][1](X.iloc[a], lo[a], hi[a], X.iloc[b], lo[b], hi[b])
                # identical safeguard for every learner: clip to the training label range +/- 1 log10 unit
                ylo = np.nanmin(np.where(np.isfinite(lo[a]), lo[a], hi[a])) - 1.0
                yhi = np.nanmax(np.where(np.isfinite(hi[a]), hi[a], lo[a])) + 1.0
                oof[va, j] = np.clip(f(X.iloc[va_abs]), ylo, yhi)
                tst[:, j] += np.clip(f(X.iloc[test]), ylo, yhi) / 5
            except Exception as e:  # keep the benchmark running; record the failure
                fails[nm] = repr(e)
                traceback.print_exc()
            cost[nm] += time.time() - t1
        print(f"[{t}/{setting}] fold {k} {time.time()-t0:.0f}s", flush=True)
    ok = [j for j, nm in enumerate(names) if nm not in fails]
    names_ok = [names[j] for j in ok]
    oof, tst = oof[:, ok], tst[:, ok]
    lo_d, hi_d, lo_t, hi_t = lo[dev], hi[dev], lo[test], hi[test]
    tr_t = truth[test] if truth is not None else None
    pd.DataFrame(oof, columns=names_ok).assign(y_lo=lo_d, y_hi=hi_d, station=g[dev],
                                               truth=truth[dev] if truth is not None else np.nan).to_parquet(
        RES / "preds" / f"oof_{t}_{setting}.parquet", index=False)
    pd.DataFrame(tst, columns=names_ok).assign(y_lo=lo_t, y_hi=hi_t, station=g[test],
                                               truth=tr_t if tr_t is not None else np.nan).to_parquet(
        RES / "preds" / f"test_{t}_{setting}.parquet", index=False)

    rows, fam = [], {nm: learners[nm][0] for nm in names_ok}
    preds, oof_cnll = {}, {}
    for j, nm in enumerate(names_ok):
        s = fit_sigma(oof[:, j], lo_d, hi_d)
        oof_cnll[nm] = float(interval_nll(oof[:, j], lo_d, hi_d, s).mean())
        preds[nm] = (tst[:, j], np.full(len(test), s))
        rows.append({"model": nm, "family": fam[nm], "sigma": s, "cost_s": cost[nm], "oof_cnll": oof_cnll[nm], **evaluate(tst[:, j], s, lo_t, hi_t, tau, tr_t)})
    # ensembles (all fitted on OOF predictions of the development stations)
    # top-5 chosen on development OOF likelihood only (never on test)
    top_idx = [names_ok.index(nm) for nm in sorted(oof_cnll, key=oof_cnll.get)[:5]]
    mu = tst[:, top_idx].mean(1)
    s = fit_sigma(oof[:, top_idx].mean(1), lo_d, hi_d)
    preds["Mean-Top5"] = (mu, np.full(len(test), s))
    sk = standard_stack(oof, lo_d, hi_d)
    s = fit_sigma(sk.predict(oof), lo_d, hi_d)
    preds["Stack-L2"] = (sk.predict(tst), np.full(len(test), s))
    csl = CensoredSuperLearner(hetero=False).fit(oof, lo_d, hi_d)
    preds["CSL"] = csl.predict(tst)
    dcsl = CensoredSuperLearner(hetero=True).fit(oof, lo_d, hi_d)
    preds["D-CSL"] = dcsl.predict(tst)
    for nm in ["Mean-Top5", "Stack-L2", "CSL", "D-CSL"]:
        mu, s = preds[nm]
        rows.append({"model": nm, "family": "Ensemble", "sigma": float(np.median(s)), "cost_s": np.nan,
                     **evaluate(mu, s, lo_t, hi_t, tau, tr_t)})
    weights = {"CSL": dict(zip(names_ok, csl.w.round(4))), "D-CSL": dict(zip(names_ok, dcsl.w.round(4))),
               "D-CSL_scale": dcsl.c.tolist(), "Stack-L2": dict(zip(names_ok, sk.coef_.round(4)))}

    # conformal compliance certification on held-out test stations (calibration / evaluation halves)
    gt = g[test]
    cal, ev = next(GroupShuffleSplit(1, test_size=0.5, random_state=7).split(gt, groups=gt))
    country = d["country"].astype(str).to_numpy()[test]
    conf_rows = []
    for nm in ["LightGBM", "XGB-AFT-N", "LGB-Tobit", "CSL", "D-CSL"]:
        if nm not in preds:
            continue
        mu, s = preds[nm]
        for mode in ["parametric", "conformal", "mondrian"]:
            if mode == "parametric":
                U = parametric_upper(mu[ev], s[ev], ALPHA)
            else:
                up, q_all, _ = conformal_upper(mu[cal], s[cal], hi_t[cal], ALPHA,
                                               groups_cal=country[cal] if mode == "mondrian" else None)
                U = up(mu[ev], s[ev], country[ev] if mode == "mondrian" else None)
            l, h = lo_t[ev], hi_t[ev]
            cov_lo, cov_hi = coverage_bounds(U, l, h)
            und = (~np.isfinite(l)) & (h > tau)            # non-detect with DL above the threshold
            cert = und & (np.minimum(U, h) <= tau)
            r = {"model": nm, "mode": mode, "cov_lower": cov_lo, "cov_upper": cov_hi,
                 "n_undetermined": int(und.sum()), "certified_share": float(cert.sum() / max(und.sum(), 1)),
                 "median_bound_gap": float(np.median(U - mu[ev]))}
            if tr_t is not None:
                yt = tr_t[ev]
                r["cov_true"] = float(np.mean(yt <= U))
                r["cov_true_undetermined"] = float(np.mean(yt[und] <= U[und])) if und.any() else np.nan
                r["cert_false_rate"] = float(np.mean(yt[cert] > tau)) if cert.any() else np.nan
            conf_rows.append(r)
    out = {"target": t, "setting": setting, "n_dev": int(len(dev)), "n_test": int(len(test)),
           "cens_share": float(np.mean(~np.isfinite(lo))), "fails": fails, "weights": weights,
           "runtime_s": time.time() - t0}
    met = pd.DataFrame(rows).assign(target=t, setting=setting)
    conf = pd.DataFrame(conf_rows).assign(target=t, setting=setting)
    met.to_csv(RES / f"bench_{t}_{setting}.csv", index=False)
    conf.to_csv(RES / f"conformal_{t}_{setting}.csv", index=False)
    (RES / f"bench_{t}_{setting}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 250)
    print(met.sort_values("cnll")[["model", "family", "cindex", "cnll", "r2_detected", "exc_auc", "exc_brier"]
                                  + (["r2_truth", "rmse_hidden", "bias_hidden"] if tr_t is not None else [])].round(3).to_string())
    print(conf.round(3).to_string(), flush=True)


if __name__ == "__main__":
    setting, targets = sys.argv[1], sys.argv[2:]
    for t in targets:
        run(t, setting)
