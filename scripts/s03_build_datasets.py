"""Build sampling-event covariates and interval-censored target tables.

Sampling event = (station, date). Routine covariates are harmonised to one unit
per parameter, '<' covariate values are replaced by half the reported limit
(covariates are not the inferential target), and same-day replicates are
reduced to the median.

Each target table keeps every accepted measurement of one contaminant with an
interval label on log10 scale:
    detected / '~'  -> [log v, log v]
    '<' DL          -> (-inf, log DL]
    '>' v           -> [log v, +inf)
Suspect, Contamination and Pending-review rows, non-positive detected values and
non-convertible units are dropped and counted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "data_processed"
MEAS = ds.dataset(DP / "measurements", format="parquet")

# canonical unit factors (to the unit in the dict value)
TO_UG = {"µg/l": 1.0, "mg/l": 1e3, "ng/l": 1e-3}
TO_MG = {"mg/l": 1.0, "µg/l": 1e-3, "ng/l": 1e-6}

ROUTINE = {  # feature name -> (codes coalesced in order, unit map)
    "TEMP": (["TEMP"], {"°C": 1.0}),
    "pH": (["pH"], {"---": 1.0}),
    "EC": (["EC"], {"µS/cm": 1.0}),
    "DO": (["O2-Dis"], TO_MG),
    "NO3N": (["NO3N"], TO_MG),
    "NH4N": (["NH4N"], TO_MG),
    "TP": (["TP"], TO_MG),
    "TSS": (["TSS"], TO_MG),
    "DOC": (["DOC"], TO_MG),
    "TOC": (["TOC"], TO_MG),
    "Cl": (["Cl-Tot", "Cl-Dis"], TO_MG),
    "Ca": (["Ca-Tot", "Ca-Dis"], TO_MG),
    "Alk": (["Alk-Tot"], TO_MG),
    "HCO3": (["HCO3"], TO_MG),
    "BOD": (["BOD"], TO_MG),
    "COD": (["COD"], TO_MG),
    "TURB": (["TURB"], {"NTU": 1.0}),
    "Fe": (["Fe-Tot"], TO_MG),
    "Mn": (["Mn-Tot"], TO_MG),
}

TARGETS = {  # target -> (param code, unit map, unit label)
    "Pb": ("Pb-Tot", TO_UG, "µg/L"), "Cd": ("Cd-Tot", TO_UG, "µg/L"),
    "Ni": ("Ni-Tot", TO_UG, "µg/L"), "As": ("As-Tot", TO_UG, "µg/L"),
    "Cr": ("Cr-Tot", TO_UG, "µg/L"), "Cu": ("Cu-Tot", TO_UG, "µg/L"),
    "Zn": ("Zn-Tot", TO_UG, "µg/L"), "Hg": ("Hg-Tot", TO_UG, "µg/L"),
    "NH4N": ("NH4N", TO_MG, "mg/L"), "TP": ("TP", TO_MG, "mg/L"),
    "NO2N": ("NO2N", TO_MG, "mg/L"),
    "Atrazine": ("ATRAZINE", TO_UG, "µg/L"), "Glyphosate": ("GLYPHOSATE", TO_UG, "µg/L"),
    "PFOS": ("PFOS", TO_UG, "µg/L"), "PFOA": ("PFOA", TO_UG, "µg/L"),
    # dissolved fractions (the EU EQS for metals refer to the dissolved concentration)
    "CdDis": ("Cd-Dis", TO_UG, "µg/L"), "PbDis": ("Pb-Dis", TO_UG, "µg/L"),
    "NiDis": ("Ni-Dis", TO_UG, "µg/L"), "HgDis": ("Hg-Dis", TO_UG, "µg/L"),
}
BAD_Q = {"Suspect", "Contamination", "Pending review"}
COLS = ["station", "date", "param", "flag", "value", "unit", "quality", "depth", "method"]


def load(codes):
    return MEAS.to_table(filter=ds.field("param").isin(codes), columns=COLS).to_pandas()


def harmonise(df, umap):
    f = df["unit"].map(umap)
    return df.assign(v=df["value"] * f)[f.notna()]


def build_events():
    parts = []
    for name, (codes, umap) in ROUTINE.items():
        df = harmonise(load(codes), umap)
        df = df[~df["quality"].isin(BAD_Q) & df["v"].notna()]
        df.loc[df["flag"] == "<", "v"] *= 0.5
        if name not in ("TEMP", "pH"):
            df = df[df["v"] > 0]
        prio = {c: i for i, c in enumerate(codes)}
        df["prio"] = df["param"].map(prio)
        df = df[df["prio"] == df.groupby(["station", "date"])["prio"].transform("min")]
        parts.append(df.groupby(["station", "date"])["v"].median().rename(name))
        print("routine", name, len(df), flush=True)
    ev = pd.concat(parts, axis=1).reset_index()
    ev.to_parquet(DP / "events_routine.parquet", index=False)
    return ev


def build_station():
    st = pd.read_parquet(DP / "GEMStat_station_metadata.parquet")
    st = st.drop_duplicates("GEMS Station Number")
    out = pd.DataFrame({
        "station": st["GEMS Station Number"],
        "country": st["Country Name"],
        "wtype": st["Water Type"].str.replace(" station", "", regex=False),
        "mtype": st["Monitoring Type"].fillna("NA"),
        "basin": st["Main Basin"].fillna("NA"),
        "lat": pd.to_numeric(st["Latitude"], errors="coerce"),
        "lon": pd.to_numeric(st["Longitude"], errors="coerce"),
        "elev": pd.to_numeric(st["Elevation"], errors="coerce"),
        "area": pd.to_numeric(st["Upstream Basin Area"], errors="coerce"),
        "discharge": pd.to_numeric(st["Discharge"], errors="coerce"),
    })
    out.to_parquet(DP / "stations_clean.parquet", index=False)
    return out


def build_target(name, code, umap):
    raw = load([code])
    n0 = len(raw)
    df = harmonise(raw, umap)
    n_unit = len(df)
    df = df[~df["quality"].isin(BAD_Q) & df["v"].notna()]
    flag = df["flag"].fillna("")
    df = df[(df["v"] > 0)]
    flag = df["flag"].fillna("")
    lv = np.log10(df["v"].to_numpy())
    lo = np.where(flag == "<", -np.inf, lv)
    hi = np.where(flag == ">", np.inf, lv)
    out = pd.DataFrame({
        "station": df["station"].to_numpy(), "date": df["date"].to_numpy(),
        "depth": df["depth"].to_numpy(), "method": df["method"].astype(str).to_numpy(),
        "y_lo": lo, "y_hi": hi, "cens": np.select([flag == "<", flag == ">"], [1, 2], 0),
        "raw_v": df["v"].to_numpy(),
    })
    # one value per event: keep first replicate per (station, date) to avoid duplicated rows
    out = out.sort_values(["station", "date", "cens"]).drop_duplicates(["station", "date"], keep="first")
    out.to_parquet(DP / f"target_{name}.parquet", index=False)
    return {"target": name, "code": code, "raw_rows": n0, "after_unit": n_unit,
            "events": len(out), "lt_share": float((out["cens"] == 1).mean()),
            "gt_share": float((out["cens"] == 2).mean())}


if __name__ == "__main__":
    build_station()
    if not (DP / "events_routine.parquet").exists():
        build_events()
    only = __import__("sys").argv[1:]
    log = [build_target(k, *v[:2]) for k, v in TARGETS.items() if not only or k in only]
    for r in log:
        print(r)
    f = ROOT / "results" / "s03_target_counts.json"
    old = json.loads(f.read_text()) if f.exists() else []
    names = {r["target"] for r in log}
    f.write_text(json.dumps([r for r in old if r["target"] not in names] + log, indent=1, ensure_ascii=False))
