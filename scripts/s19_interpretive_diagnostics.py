"""Diagnostics that test explanations for the reporting-rule contrasts.

All quantities use existing cross-fitted predictions and benchmark test predictions (no refitting).
  limit_variance      share of the variance of log10(D/T) among non-detects lying between countries,
                      between stations within a country and within stations
  period_composition  events, contributing countries, blind-spot share and median D/T by decade
  detected_reference  detected values re-censored at blind-spot limits of their own country (as in s11):
                      observed exceedance share, share the half-limit rule would count, and the position
                      of the measured value relative to the assigned limit
  conditional_mean    model expected concentration of a blind-spot non-detect relative to its limit
  limit_band          blind-spot events, mean conditional exceedance probability and expected hidden
                      exceedances by band of D/T
  station_intensity   blind-spot share by quartile of events per station
  ranking             dispersion of uncertain-event counts and of mean probabilities among stations
  learner_limit       correlation between held-out predictions for blind-spot non-detects and log10 D
  lower_tail          detected values below the threshold that support the lower tail
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables"
THRESH = {"Cd": 0.08, "Hg": 0.07, "Pb": 1.2, "Ni": 4.0, "PFOS": 0.00065, "Glyphosate": 0.1}
METALS = ["Cd", "Hg", "Pb"]
SUBSTITUTION = ["Ridge", "Lasso", "ElasticNet", "Huber", "LinearSVR", "KNN", "DecisionTree", "AdaBoost",
                "RandomForest", "ExtraTrees", "HistGB", "XGBoost", "LightGBM", "CatBoost", "MLP"]
CENSORED = ["Tobit", "XGB-AFT-N", "XGB-AFT-L", "CatBoost-AFT", "LGB-Tobit", "DeepTobit", "RSF"]
BANDS = [(1, 2, "T < D <= 2T"), (2, 5, "2T < D <= 5T"), (5, 10, "5T < D <= 10T"), (10, np.inf, "D > 10T")]
rows = []


def add(diagnostic, target, group, quantity, value):
    rows.append({"diagnostic": diagnostic, "target": target, "group": group, "quantity": quantity, "value": float(value)})


def load(t):
    d = pd.read_parquet(RES / f"crossfit_{t}.parquet")
    d["year"] = d["date"].dt.year
    d["ratio"] = np.where(d["nd"], 10 ** d["y_hi"] / THRESH[t], np.nan)
    return d


def recensored(d, tau, seed=0):
    """Detected values that fall below a blind-spot limit drawn from their own country (same draw as s11)."""
    rng = np.random.default_rng(seed)
    pools = {c: g["y_hi"].to_numpy() for c, g in d[d["undetermined"]].groupby("country")}
    det = d[~d["nd"]].copy()
    det = det[det["country"].isin(pools)]
    det["dl_em"] = [pools[c][rng.integers(len(pools[c]))] for c in det["country"]]
    return det[det["y_lo"] < det["dl_em"]].copy()


for t, thr in THRESH.items():
    d, tau = load(t), np.log10(thr)
    nd, bs = d[d["nd"]], d[d["undetermined"]]

    # lower-tail support (all six targets)
    det = d[~d["nd"] & np.isfinite(d["y_lo"])]
    add("lower_tail", t, "Detected values", "events", len(det))
    add("lower_tail", t, "Detected values", "share below T", (det["y_lo"] < tau).mean())
    add("lower_tail", t, "Detected values", "events below T", (det["y_lo"] < tau).sum())
    add("lower_tail", t, "Blind-spot non-detects", "median D/T", bs["ratio"].median())
    add("lower_tail", t, "Blind-spot non-detects", "share with D > 10T", (bs["ratio"] > 10).mean())

    # detected-value reference (all six targets)
    em = recensored(d, tau)
    y, dl = em["y_lo"].to_numpy(), em["dl_em"].to_numpy()
    truth, half = y > tau, dl > tau + np.log10(2)
    add("detected_reference", t, "Re-censored detected values", "events", len(em))
    add("detected_reference", t, "Re-censored detected values", "observed exceedance share", truth.mean())
    add("detected_reference", t, "Re-censored detected values", "share counted by half-limit rule", half.mean())
    add("detected_reference", t, "Re-censored detected values", "observed exceedance share where D > 2T", truth[half].mean())
    add("detected_reference", t, "Re-censored detected values", "median measured value / D", np.median(10 ** (y - dl)))
    add("detected_reference", t, "Re-censored detected values", "share with measured value > D/2", (y > dl - np.log10(2)).mean())
    known, blind = d["known_exceed"].mean(), d["undetermined"].mean()
    add("detected_reference", t, "All events", "prevalence if hidden share equals detected reference", known + blind * truth.mean())

    if t not in METALS:
        continue

    # reporting-limit variance
    x = pd.DataFrame({"x": np.log10(nd["ratio"]), "country": nd["country"], "station": nd["station"]})
    gm = x["x"].mean()
    cm, sm = x.groupby("country")["x"].transform("mean"), x.groupby("station")["x"].transform("mean")
    ss = {"between countries": ((cm - gm) ** 2).sum(), "between stations within countries": ((sm - cm) ** 2).sum(),
          "within stations": ((x["x"] - sm) ** 2).sum()}
    tot = sum(ss.values())
    for k, v in ss.items():
        add("limit_variance", t, "Non-detects", f"variance share {k}", v / tot)
    add("limit_variance", t, "Non-detects", "events", len(x))

    # period composition
    d["decade"] = np.select([d["year"] < 1990, d["year"] < 2000, d["year"] < 2010, d["year"] < 2020],
                            ["1980-1989", "1990-1999", "2000-2009", "2010-2019"], "2020-2021")
    parts = [(dec, g) for dec, g in d.groupby("decade")]
    parts += [("1980-1999", d[d["year"] < 2000]), ("2000-2021", d[d["year"] >= 2000])]
    for label, g in parts:
        add("period_composition", t, label, "events", len(g))
        add("period_composition", t, label, "countries", g["country"].nunique())
        add("period_composition", t, label, "blind-spot share", g["undetermined"].mean())
        add("period_composition", t, label, "median D/T of non-detects", g["ratio"].median())
        add("period_composition", t, label, "blind-spot share excluding Mexico", g[g["country"] != "Mexico"]["undetermined"].mean())

    # model conditional mean relative to the limit
    m, s, ld = bs["mu"].to_numpy() * np.log(10), bs["sigma"].to_numpy() * np.log(10), bs["y_hi"].to_numpy() * np.log(10)
    tm = np.exp(m + s ** 2 / 2) * ndtr((ld - m - s ** 2) / s) / np.clip(ndtr((ld - m) / s), 1e-300, None)
    tm = np.minimum(tm, np.exp(ld))
    rel = tm / np.exp(ld)
    add("conditional_mean", t, "Blind-spot non-detects", "median expected concentration / D", np.median(rel))
    add("conditional_mean", t, "Blind-spot non-detects", "share with expected concentration < D/2", (rel < 0.5).mean())
    add("conditional_mean", t, "Blind-spot non-detects", "share with expected concentration < T", (tm < thr).mean())

    # bands of D/T
    w = bs["post_exceed"].to_numpy()
    for lo, hi, label in BANDS:
        k = ((bs["ratio"] > lo) & (bs["ratio"] <= hi)).to_numpy()
        add("limit_band", t, label, "share of blind-spot events", k.mean())
        add("limit_band", t, label, "mean conditional exceedance probability", w[k].mean() if k.any() else np.nan)
        add("limit_band", t, label, "share of expected hidden exceedances", w[k].sum() / w.sum())

    # monitoring intensity and sensitivity
    stn = d.groupby("station").agg(n=("nd", "size"), blind=("undetermined", "mean")).reset_index()
    stn["q"] = pd.qcut(stn["n"].rank(method="first"), 4, labels=["Q1 (fewest events)", "Q2", "Q3", "Q4 (most events)"])
    for q, g in stn.groupby("q", observed=True):
        add("station_intensity", t, str(q), "stations", len(g))
        add("station_intensity", t, str(q), "median events per station", g["n"].median())
        add("station_intensity", t, str(q), "mean station blind-spot share", g["blind"].mean())
    add("station_intensity", t, "All stations", "mean station blind-spot share", stn["blind"].mean())
    add("station_intensity", t, "All stations", "event-weighted blind-spot share", d["undetermined"].mean())
    add("station_intensity", t, "All stations", "Spearman correlation of events and blind-spot share", spearmanr(stn["n"], stn["blind"]).statistic)

    # count-based and probability-weighted ranking
    u = bs[(bs["year"] >= 2015) & (bs["post_exceed"] >= 0.05) & (bs["post_exceed"] <= 0.5)]
    sg = u.groupby("station").agg(n=("post_exceed", "size"), e=("post_exceed", "sum"), p=("post_exceed", "mean"))
    grp = "Stations with uncertain events since 2015"
    add("ranking", t, grp, "stations", len(sg))
    add("ranking", t, grp, "coefficient of variation of uncertain-event count", sg["n"].std() / sg["n"].mean())
    add("ranking", t, grp, "coefficient of variation of mean probability", sg["p"].std() / sg["p"].mean())
    add("ranking", t, grp, "Spearman correlation of count and expected hidden exceedances", spearmanr(sg["n"], sg["e"]).statistic)

    # held-out learners and the reporting limit
    p = pd.read_parquet(RES / "preds" / f"test_{t}_real.parquet")
    b = p[np.isneginf(p["y_lo"]) & (p["y_hi"] > tau)]
    add("learner_limit", t, "Held-out blind-spot non-detects", "events", len(b))
    for fam, cols in [("Substitution learners", SUBSTITUTION), ("Censored learners", CENSORED)]:
        r = [np.corrcoef(b[c], b["y_hi"])[0, 1] for c in cols]
        off = [(b[c] - (b["y_hi"] - np.log10(2))).mean() for c in cols]
        above = [(b[c] > tau).mean() for c in cols]
        add("learner_limit", t, fam, "median correlation of prediction with log10 D", np.median(r))
        add("learner_limit", t, fam, "median mean difference from log10(D/2)", np.median(off))
        add("learner_limit", t, fam, "median share of predictions above T", np.median(above))

out = pd.DataFrame(rows)
out.to_csv(TAB / "T_interpretive_diagnostics.csv", index=False)
pd.set_option("display.width", 220, "display.max_rows", 500)
print(out.round(4).to_string(index=False))
