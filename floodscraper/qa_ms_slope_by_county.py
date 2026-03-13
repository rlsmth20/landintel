from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"

INPUT_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood_and_slope.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_slope.gpkg",
]
OUTPUT_CSV = PARCELS_DIR / "mississippi_slope_county_qa.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build county-level Mississippi slope QA report.")
    parser.add_argument("--input-file", type=str, default="", help="Slope-enriched parcel GPKG.")
    parser.add_argument("--output-csv", type=str, default=str(OUTPUT_CSV), help="County QA CSV output.")
    return parser.parse_args()


def choose_input(path_arg: str) -> Path:
    if path_arg:
        p = Path(path_arg)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return INPUT_CANDIDATES[0]


def main() -> None:
    args = parse_args()
    input_file = choose_input(args.input_file)
    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = BASE_DIR / output_csv

    if not input_file.exists():
        print(f"ERROR: Missing slope parcel file: {input_file}")
        return

    gdf = gpd.read_file(input_file)
    if "county_name" not in gdf.columns:
        print("ERROR: county_name column missing in parcel file.")
        return

    mean_vals = pd.to_numeric(gdf.get("mean_slope_pct"), errors="coerce")
    slope_class = gdf.get("slope_class", pd.Series(index=gdf.index, dtype="object")).astype(str)

    df = pd.DataFrame(
        {
            "county_name": gdf["county_name"].astype(str),
            "mean_slope_pct": mean_vals,
            "slope_class": slope_class,
        }
    )
    df["is_unknown"] = (~np.isfinite(df["mean_slope_pct"])) | (df["slope_class"].str.lower() == "unknown")

    grouped = df.groupby("county_name", as_index=False)
    out = grouped.agg(
        parcels_total=("county_name", "size"),
        parcels_with_slope=("mean_slope_pct", lambda s: int(np.isfinite(s).sum())),
        unknown_rows=("is_unknown", "sum"),
        mean_slope_avg=("mean_slope_pct", "mean"),
        mean_slope_p50=("mean_slope_pct", "median"),
        mean_slope_p95=("mean_slope_pct", lambda s: float(np.nanquantile(s.to_numpy(dtype=float), 0.95)) if np.isfinite(s).any() else np.nan),
        max_slope_max=("mean_slope_pct", "max"),
    )
    out["unknown_pct"] = (out["unknown_rows"] / out["parcels_total"]) * 100.0

    class_counts = (
        df.groupby(["county_name", "slope_class"], as_index=False)
        .size()
        .pivot(index="county_name", columns="slope_class", values="size")
        .fillna(0)
        .reset_index()
    )
    class_counts.columns = [str(c) if c == "county_name" else f"class_{c}" for c in class_counts.columns]
    out = out.merge(class_counts, on="county_name", how="left")
    out = out.sort_values(["unknown_pct", "county_name"], ascending=[False, True]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"QA rows: {len(out):,}")
    print(f"Report written: {output_csv}")
    print("Top counties by unknown pct:")
    print(out[["county_name", "parcels_total", "unknown_rows", "unknown_pct"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
