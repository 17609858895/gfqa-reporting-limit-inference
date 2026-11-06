"""Cross-fitted censored model over the full archive: hidden exceedance, certification, SHAP.

* 5-fold station cross-fitting of LGB-Tobit over every event of a target (each event is
  predicted by a model that never saw its station).
* Posterior exceedance of a non-detect, using the information Y < DL:
      P(Y > tau | Y < DL, x) = [Phi((DL-mu)/s) - Phi((tau-mu)/s)] / Phi((DL-mu)/s)   (DL > tau)
* Exceedance prevalence per country and 5-yr block under four reporting rules:
      ND-as-compliant | DL/2 substitution | DL substitution | censored-model posterior
* Censoring-robust conformal certification (alpha = 0.1) with 30% of stations as calibration.
* TreeSHAP of the latent-mean model on a stratified subsample.
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
from censml import fit_sigma, load_target, tobit_grad_hess, interval_nll
from ensembles import conformal_upper

ROOT = Path(__file__).resolve().parents[1]
DP, RES = ROOT / "data_processed", ROOT / "results"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Hg": 0.07, "Ni": 4.0, "NH4N": 0.2, "PFOS": 0.00065, "PFOA": 0.1,
          "Glyphosate": 0.1, "Atrazine": 0.6, "As": 10.0,
          "CdDis": 0.08, "PbDis": 1.2, "NiDis": 4.0, "HgDis": 0.07,
          "CdLQ": 0.08, "PbLQ": 1.2, "HgLQ": 0.07}
LGB_P = dict(learning_rate=0.08, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=6, max_cat_to_onehot=8)


def fit_tobit(X, lo, hi, Xv, lov, hiv):
    import lightgbm as lgb
    y0 = np.where(np.isfinite(lo), lo, hi - np.log10(2))
    mu0, sigma = float(np.median(y0)), float(np.std(y0))
    for _ in range(2):
        def fobj(p, d, s=sigma):
            return tobit_grad_hess(p + mu0, lo, hi, s)

        def feval(p, d, s=sigma):
            return "cnll", float(interval_nll(p + mu0, lov, hiv, s).mean()), False
        dtr = lgb.Dataset(X, np.zeros(len(X)), free_raw_data=False)
        m = lgb.train({**LGB_P, "objective": fobj, "metric": "None"}, dtr, 2000,
                      valid_sets=[lgb.Dataset(Xv, np.zeros(len(Xv)), reference=dtr)], feval=feval,
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        sigma = fit_sigma(m.predict(Xv, num_iteration=m.best_iteration) + mu0, lov, hiv)
    return m, mu0, sigma


def run(t, tag="", exclude=(), drop_country=False, with_shap=True, drop_features=()):
    d, feats = load_target(DP, t)
    if exclude:
        d = d[~d["country"].astype(str).isin(exclude)].reset_index(drop=True)
    if drop_country:
        feats = [f for f in feats if f != "country"]
    feats = [f for f in feats if f not in drop_features]
    tau = np.log10(THRESH[t])
    X, lo, hi, g = d[feats], d["y_lo"].to_numpy(), d["y_hi"].to_numpy(), d["station"].to_numpy()
    mu, sig = np.zeros(len(d)), np.zeros(len(d))
    model = None
    for k, (tr, te) in enumerate(GroupKFold(5).split(d, groups=g)):
        a_, v_ = next(GroupShuffleSplit(1, test_size=0.1, random_state=k).split(tr, groups=g[tr]))
        a, v = tr[a_], tr[v_]
        m, mu0, s = fit_tobit(X.iloc[a], lo[a], hi[a], X.iloc[v], lo[v], hi[v])
        mu[te] = m.predict(X.iloc[te], num_iteration=m.best_iteration) + mu0
        sig[te] = s
        model = model or (m, mu0)
        print(t, "fold", k, flush=True)
    nd = ~np.isfinite(lo)
    dl = np.where(nd, hi, np.nan)
    p_le_dl = ndtr((dl - mu) / sig)
    p_le_tau = ndtr((tau - mu) / sig)
    post = np.where(nd & (dl > tau), (p_le_dl - p_le_tau) / np.clip(p_le_dl, 1e-12, None), 0.0)
    known_exc = (~nd) & (lo > tau)
    out = pd.DataFrame({"station": g, "date": d["date"], "country": d["country"].astype(str),
                        "wtype": d["wtype"].astype(str), "lat": d["lat"], "lon": d["lon"],
                        "y_lo": lo, "y_hi": hi, "mu": mu, "sigma": sig, "nd": nd,
                        "undetermined": nd & (dl > tau), "known_exceed": known_exc, "post_exceed": post})
    # censoring-robust conformal certification
    cal, ev = next(GroupShuffleSplit(1, test_size=0.7, random_state=11).split(out, groups=g))
    up, q, _ = conformal_upper(mu[cal], sig[cal], hi[cal], 0.1)
    U = np.minimum(up(mu, sig), hi)
    out["U90"] = U
    out["certified"] = out["undetermined"] & (U <= tau)
    out["is_cal"] = False
    out.loc[out.index[cal], "is_cal"] = True
    out.to_parquet(RES / f"crossfit_{t}{tag}.parquet", index=False)
    blk = (out["date"].dt.year // 5) * 5
    rules = out.assign(block=blk,
                       r_nd_compliant=known_exc,
                       r_half=known_exc | (nd & (dl - np.log10(2) > tau)),
                       r_dl=known_exc | (nd & (dl > tau)),
                       r_model=known_exc + post)
    agg = rules.groupby(["country", "block"]).agg(n=("nd", "size"), nd=("nd", "mean"), und=("undetermined", "mean"),
                                                  cert=("certified", "sum"), und_n=("undetermined", "sum"),
                                                  nd_compliant=("r_nd_compliant", "mean"), half=("r_half", "mean"),
                                                  dl=("r_dl", "mean"), model=("r_model", "mean")).reset_index()
    agg["target"] = t
    agg.to_csv(RES / f"prevalence_rules_{t}{tag}.csv", index=False)
    tot = {"target": t, "variant": tag or "main", "events": len(out), "undetermined": float(out["undetermined"].mean()),
           "certified_of_undetermined": float(out.loc[~out["is_cal"], "certified"].sum() / max(out.loc[~out["is_cal"], "undetermined"].sum(), 1)),
           "prev_nd_compliant": float(known_exc.mean()), "prev_half": float(rules["r_half"].mean()),
           "prev_dl": float(rules["r_dl"].mean()), "prev_model": float(rules["r_model"].mean()),
           "mean_post_undetermined": float(post[out["undetermined"]].mean()), "conformal_q": float(q)}
    print(tot, flush=True)
    if not with_shap:
        return tot
    # SHAP on the first fold model (latent mean on log10 scale)
    import shap
    m, mu0 = model
    idx = np.random.default_rng(0).choice(len(d), min(4000, len(d)), replace=False)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(X.iloc[idx])
    pd.DataFrame(sv, columns=feats).assign(row=idx).to_parquet(RES / f"shap_{t}.parquet", index=False)
    X.iloc[idx].reset_index(drop=True).to_parquet(RES / f"shap_X_{t}.parquet", index=False)
    return tot


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=["Pb"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--drop-country", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--drop-features", nargs="*", default=[])
    a = ap.parse_args()
    res = [run(t, a.tag, tuple(a.exclude), a.drop_country, not a.no_shap, tuple(a.drop_features)) for t in a.targets]
    old = RES / "hidden_exceedance_summary.csv"
    df = pd.DataFrame(res)
    if old.exists():
        prev = pd.read_csv(old)
        if "variant" not in prev:
            prev["variant"] = "main"
        key = set(zip(df["target"], df["variant"]))
        df = pd.concat([prev[[k not in key for k in zip(prev["target"], prev["variant"])]], df])
    df.to_csv(old, index=False)
    print(df.round(4).to_string())
