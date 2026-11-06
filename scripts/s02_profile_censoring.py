"""Per-parameter profile of the GFQA v3 measurement archive.

Outputs (tables/):
  T_param_profile.csv      rows, stations, countries, water types, flag shares, units, years
  T_dl_by_decade.csv       median reported limit of '<' values by parameter x decade
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MEAS = ROOT / "data_processed" / "measurements"
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)

st = pd.read_parquet(ROOT / "data_processed" / "GEMStat_station_metadata.parquet")
st = st.drop_duplicates("GEMS Station Number")[["GEMS Station Number", "Country Name", "Water Type"]]
st.columns = ["station", "country", "wtype"]

prof, dec_rows = [], []
for f in sorted(MEAS.glob("*.parquet")):
    df = pq.read_table(f, columns=["station", "date", "param", "flag", "value", "unit", "quality"]).to_pandas()
    df = df.merge(st, on="station", how="left")
    df["decade"] = (df["date"].dt.year // 10) * 10
    for code, g in df.groupby("param", sort=False):
        fl = g["flag"].fillna("")
        cen = g[fl == "<"]
        prof.append({
            "file": f.stem, "param": code, "rows": len(g),
            "stations": g["station"].nunique(), "countries": g["country"].nunique(),
            "river_share": (g["wtype"] == "River station").mean(),
            "gw_share": (g["wtype"] == "Groundwater station").mean(),
            "lt_share": (fl == "<").mean(), "gt_share": (fl == ">").mean(),
            "est_share": (fl == "~").mean(),
            "nonpos_detected": ((fl == "") & (g["value"] <= 0)).mean(),
            "units": "|".join(g["unit"].value_counts().index[:3].astype(str)),
            "year_min": g["date"].dt.year.min(), "year_max": g["date"].dt.year.max(),
            "suspect_share": g["quality"].isin(["Suspect", "Contamination", "Pending review"]).mean(),
            "n_distinct_dl": cen["value"].round(8).nunique(),
            "median_dl": cen["value"].median(),
            "median_detected": g.loc[fl == "", "value"].median(),
        })
        if len(cen) > 100:
            d = cen.groupby("decade")["value"].agg(["median", "size"]).reset_index()
            d["param"] = code
            d["lt_share_decade"] = g.groupby("decade")["flag"].apply(lambda s: (s == "<").mean()).reindex(d["decade"]).values
            dec_rows.append(d)
    print(f.stem, len(df), flush=True)

prof = pd.DataFrame(prof).sort_values("rows", ascending=False)
prof.to_csv(TAB / "T_param_profile.csv", index=False)
pd.concat(dec_rows).to_csv(TAB / "T_dl_by_decade.csv", index=False)
print(prof.head(80).to_string())
