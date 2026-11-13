"""Station-year annual-mean screening against annual-average thresholds.

The EU half-limit rule (Directive 2009/90/EC, Art. 5) is specified for mean values, and annual-average
standards are assessed on annual means. For each station-year with >= MIN_EVENTS events:
  * half-limit mean      : detected values, non-detects at DL/2 (regulatory convention)
  * interval bounds      : non-detects at 0 (lower) or DL (upper); assumption-free classification
  * censored-model mean  : detected values, non-detects at E[C | C < DL, x] (truncated log-normal)
  * P(mean > T)          : Monte Carlo over truncated latent draws for every non-detect
Uses cross-fitted latent predictions from s09 (no refitting).
Thresholds are the screening values used for the event-level analysis; for dissolved Cd the
hardness class of the station-year median hardness is applied (EU AA-EQS classes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
DP, RES, TAB = ROOT / "data_processed", ROOT / "results", ROOT / "tables"
THRESH = {"Cd": 0.08, "Pb": 1.2, "Ni": 4.0, "PFOS": 0.00065, "CdDis": 0.08}
MIN_EVENTS = 4
R = 200


def cd_class_eqs(h):
    return np.select([h < 50, h < 100, h < 200, h >= 200], [0.08, 0.09, 0.15, 0.25], 0.08)


def run(t, hardness=False, seed=0):
    d = pd.read_parquet(RES / f"crossfit_{t}.parquet", columns=["station", "date", "country", "nd", "y_lo", "y_hi", "mu", "sigma"])
    d["year"] = d["date"].dt.year
    if hardness:
        h = pd.read_parquet(DP / "events_hardness.parquet")
        d = d.merge(h, on=["station", "date"], how="left")
        d["hardness"] = d.groupby(["station", "year"])["hardness"].transform("median")
        d["hardness"] = d["hardness"].fillna(d.groupby("station")["hardness"].transform("median"))
        d["T"] = cd_class_eqs(d["hardness"].to_numpy())
    else:
        d["T"] = THRESH[t]
    n_ev = d.groupby(["station", "year"])["nd"].transform("size")
    d = d[n_ev >= MIN_EVENTS].reset_index(drop=True)
    gid, keys = pd.factorize(pd.MultiIndex.from_frame(d[["station", "year"]]))
    G = sparse.csr_matrix((np.ones(len(d)), (gid, np.arange(len(d)))), shape=(len(keys), len(d)))
    cnt = np.asarray(G.sum(1)).ravel()
    nd = d["nd"].to_numpy()
    det_c = np.where(nd, 0.0, 10 ** np.where(np.isfinite(d["y_lo"]), d["y_lo"], 0))
    dl = np.where(nd, 10 ** d["y_hi"].to_numpy(), 0.0)
    mu, s = d["mu"].to_numpy(), d["sigma"].to_numpy()
    # truncated log-normal mean on natural-log scale
    m, sl, ld = mu * np.log(10), s * np.log(10), np.log(np.where(nd, dl, 1.0))
    trunc_mean = np.exp(m + sl ** 2 / 2) * ndtr((ld - m - sl ** 2) / sl) / np.clip(ndtr((ld - m) / sl), 1e-300, None)
    trunc_mean = np.minimum(trunc_mean, dl)
    T = d.groupby(gid)["T"].first().to_numpy() if hardness else np.full(len(keys), THRESH[t])
    mean = lambda v: np.asarray(G @ v).ravel() / cnt
    half = mean(det_c + np.where(nd, dl / 2, 0))
    lo_b, hi_b = mean(det_c), mean(det_c + np.where(nd, dl, 0))
    model_mean = mean(det_c + np.where(nd, trunc_mean, 0))
    # Monte Carlo posterior of the annual mean
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(nd)
    pdl = ndtr((np.log10(dl[idx]) - mu[idx]) / s[idx])
    exceed = np.zeros(len(keys))
    base = np.asarray(G @ det_c).ravel()
    Gn = G[:, idx]
    for _ in range(R // 20):
        u = rng.uniform(0, 1, (len(idx), 20)) * pdl[:, None]
        y = mu[idx, None] + s[idx, None] * ndtri(np.clip(u, 1e-300, 1 - 1e-12))
        c = np.minimum(10 ** y, dl[idx, None])
        sums = base[:, None] + np.asarray(Gn @ c)
        exceed += ((sums / cnt[:, None]) > T[:, None]).sum(1)
    p = exceed / R
    out = pd.DataFrame({"station": keys.get_level_values(0), "year": keys.get_level_values(1), "events": cnt,
                        "T": T, "half": half, "lower": lo_b, "upper": hi_b, "model_mean": model_mean, "p_exceed": p})
    out = out.merge(d.groupby(gid).agg(country=("country", "first"), nd_share=("nd", "mean")).reset_index(drop=True),
                    left_index=True, right_index=True)
    out["cls_bounds"] = np.select([out["upper"] <= out["T"], out["lower"] > out["T"]], ["compliant", "exceeding"], "undetermined")
    out["half_exceed"] = out["half"] > out["T"]
    out["model_exceed"] = out["model_mean"] > out["T"]
    out["model_cls"] = np.select([out["p_exceed"] < 0.05, out["p_exceed"] > 0.95], ["compliant", "exceeding"], "uncertain")
    return out


def summarise(out, label):
    und = out["cls_bounds"] == "undetermined"
    flip_down = out["half_exceed"] & (out["model_cls"] == "compliant")
    return {"target": label, "station_years": len(out), "stations": out["station"].nunique(),
            "undetermined_share": und.mean(), "half_exceed_share": out["half_exceed"].mean(),
            "model_mean_exceed_share": out["model_exceed"].mean(), "expected_exceed_share": out["p_exceed"].mean(),
            "model_compliant_share": (out["model_cls"] == "compliant").mean(),
            "model_uncertain_share": (out["model_cls"] == "uncertain").mean(),
            "model_exceed_share": (out["model_cls"] == "exceeding").mean(),
            "half_exceed_but_model_compliant": flip_down.mean(),
            "half_exceed_but_model_compliant_of_half": flip_down.sum() / max(out["half_exceed"].sum(), 1),
            "half_compliant_but_model_exceeding": ((~out["half_exceed"]) & (out["model_cls"] == "exceeding")).mean()}


def bootstrap_ci(F, B=500, seed=0):
    """95% percentile intervals from resampling whole stations (station-years held fixed)."""
    rng = np.random.default_rng(seed)
    out = []
    for label, g in F.groupby("target", sort=False):
        ind = pd.DataFrame({"station": g["station"], "half": g["half_exceed"].astype(float),
                            "expected": g["p_exceed"], "lower": (g["cls_bounds"] == "exceeding").astype(float),
                            "upper": (g["cls_bounds"] != "compliant").astype(float),
                            "flip": (g["half_exceed"] & (g["model_cls"] == "compliant")).astype(float),
                            "n": 1.0}).groupby("station").sum()
        S = ind.to_numpy()
        point = S.sum(0)
        draws = np.empty((B, S.shape[1]))
        for b in range(B):
            w = np.bincount(rng.integers(0, len(S), len(S)), minlength=len(S))
            draws[b] = w @ S
        share = lambda v: v[..., :5] / v[..., [5]]
        p, d = share(point), share(draws)
        flip_of_half = lambda v: v[..., 4] / v[..., 0]
        row = {"target": label, "station_years": int(point[5]), "stations": len(S)}
        for k, name in enumerate(["half", "expected", "lower", "upper", "flip"]):
            row[name] = p[k]
            row[name + "_lo"], row[name + "_hi"] = np.percentile(d[:, k], [2.5, 97.5])
        row["flip_of_half"] = flip_of_half(point)
        row["flip_of_half_lo"], row["flip_of_half_hi"] = np.percentile(flip_of_half(draws), [2.5, 97.5])
        cls = g.loc[g["half_exceed"], "model_cls"].value_counts(normalize=True)
        for c in ["compliant", "uncertain", "exceeding"]:
            row[f"half_exceed_model_{c}"] = cls.get(c, 0.0)
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    if sys.argv[1:] == ["--ci"]:
        F = pd.read_parquet(RES / "annual_compliance_station_years.parquet")
        C = bootstrap_ci(F)
        C.to_csv(TAB / "T_annual_compliance_ci.csv", index=False)
        pd.set_option("display.width", 250)
        print(C.round(4).T.to_string())
        sys.exit()
    rows, frames = [], []
    jobs = [("Cd", "Cd", False), ("CdDis", "Cd (dissolved), hardness", True), ("Pb", "Pb", False),
            ("Ni", "Ni", False), ("PFOS", "PFOS", False)]
    for t, label, hard in jobs:
        if len(sys.argv) > 1 and t not in sys.argv[1:]:
            continue
        o = run(t, hard)
        o["target"] = label
        frames.append(o)
        rows.append(summarise(o, label))
        print(rows[-1], flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(TAB / "T_annual_compliance.csv", index=False)
    F = pd.concat(frames)
    F.to_parquet(RES / "annual_compliance_station_years.parquet", index=False)
    by_c = F.groupby(["target", "country"]).agg(station_years=("half", "size"), half_exceed=("half_exceed", "mean"),
                                                expected_exceed=("p_exceed", "mean"),
                                                undetermined=("cls_bounds", lambda s: (s == "undetermined").mean())).reset_index()
    by_c.to_csv(TAB / "T_annual_compliance_by_country.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
