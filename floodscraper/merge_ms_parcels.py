from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "parcel_raw"
PARCELS_DIR = BASE_DIR / "data" / "parcels"
PARCELS_DIR.mkdir(parents=True, exist_ok=True)

# State-specific items for easy future config extraction.
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
COUNTY_FILE_PATTERN = "*_parcels.gpkg"
LIKELY_PARCEL_ID_FIELDS = ["PARCEL_ID", "PARCELID", "PARCELNO", "PIN", "GLOBALID", "OBJECTID", "FID"]

TARGET_CRS = "EPSG:4326"
STATEWIDE_OUT = PARCELS_DIR / "mississippi_parcels.gpkg"
SUMMARY_CSV = PARCELS_DIR / "mississippi_parcels_merge_summary.csv"
COUNTY_QA_CSV = PARCELS_DIR / "mississippi_parcels_county_qa.csv"


def is_polygon_geometry(geom) -> bool:
    return isinstance(geom, (Polygon, MultiPolygon))


def detect_county_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_parcels"):
        return name.replace("_parcels", "")
    return name


def standardize_parcel_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["parcel_id"] = pd.Series(dtype="object")
        return gdf

    out = gdf.copy()
    cols_map = {str(c).upper(): c for c in out.columns}
    parcel_col = None
    for candidate in LIKELY_PARCEL_ID_FIELDS:
        hit = cols_map.get(candidate.upper())
        if hit:
            parcel_col = hit
            break

    if parcel_col:
        out["parcel_id"] = out[parcel_col].astype(str).str.strip()
    else:
        out["parcel_id"] = pd.NA
    return out


def clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf[gdf.geometry.notnull()].copy()
    out = out[~out.geometry.is_empty].copy()
    out = out[out.geometry.apply(is_polygon_geometry)].copy()
    return out


def dedupe_statewide(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    out = gdf.copy()
    has_parcel_id = "parcel_id" in out.columns
    if has_parcel_id:
        valid = out["parcel_id"].notnull() & (out["parcel_id"].astype(str).str.len() > 0)
        dedupe_subset = ["county_name", "parcel_id"] if "county_name" in out.columns else ["parcel_id"]
        out_valid = out[valid].drop_duplicates(subset=dedupe_subset).copy()
        out_null = out[~valid].copy()
        out = pd.concat([out_valid, out_null], ignore_index=True)

    out["_geom_wkb"] = out.geometry.apply(lambda geom: geom.wkb_hex if geom is not None else None)
    out = out.drop_duplicates(subset=["_geom_wkb"]).drop(columns=["_geom_wkb"]).copy()
    return out


def build_county_qa(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out = out[out["status"].eq("ok")].copy()
    if out.empty:
        return out
    median_rows = float(out["rows"].median())
    out["statewide_row_median"] = median_rows
    out["row_ratio_to_median"] = (out["rows"] / median_rows).round(6)
    out["low_row_warning"] = out["rows"] < 500
    out["very_low_row_warning"] = out["rows"] < 100
    out["notes"] = np.select(
        [
            out["rows"] < 100,
            out["rows"] < 500,
        ],
        [
            "County row count is extremely low and likely indicates a raw ingest failure.",
            "County row count is unusually low and should be reviewed before downstream statewide analysis.",
        ],
        default="ok",
    )
    return out.sort_values("rows").reset_index(drop=True)


def main() -> None:
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATE:    {STATE_NAME} ({STATE_ABBR})")
    print(f"RAW_DIR:  {RAW_DIR}")
    print(f"Output:   {STATEWIDE_OUT}\n")

    county_files = sorted(RAW_DIR.glob(COUNTY_FILE_PATTERN))
    if not county_files:
        print(f"No county parcel files found in {RAW_DIR} using pattern {COUNTY_FILE_PATTERN}")
        return

    print(f"County parcel files found: {len(county_files)}")
    frames: list[gpd.GeoDataFrame] = []
    summary_rows: list[dict] = []

    for path in county_files:
        county_slug = detect_county_from_filename(path)
        print(f"Loading {path.name}...")
        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            print(f"Failed to read {path.name}: {exc}")
            summary_rows.append(
                {"county": county_slug, "file_name": path.name, "rows": 0, "status": f"read_failed: {exc}"}
            )
            continue

        if gdf.crs is None:
            print(f"Missing CRS for {path.name}; assuming {TARGET_CRS}.")
            gdf = gdf.set_crs(TARGET_CRS, allow_override=True)
        elif str(gdf.crs).upper() != TARGET_CRS:
            gdf = gdf.to_crs(TARGET_CRS)

        gdf = standardize_parcel_id(gdf)
        gdf["county_name"] = county_slug
        gdf = clean_geometries(gdf)

        rows = len(gdf)
        print(f"Rows kept: {rows:,}")
        summary_rows.append({"county": county_slug, "file_name": path.name, "rows": rows, "status": "ok"})

        if not gdf.empty:
            frames.append(gdf)

    summary_df = pd.DataFrame(summary_rows)

    if not frames:
        print("No valid county parcel data to merge.")
        summary_df.to_csv(SUMMARY_CSV, index=False)
        print(f"Summary written: {SUMMARY_CSV}")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    print(f"\nMerged rows before dedupe: {len(merged):,}")

    merged = dedupe_statewide(merged)
    print(f"Merged rows after dedupe: {len(merged):,}")

    merged.to_file(STATEWIDE_OUT, driver="GPKG", engine="pyogrio")
    print(f"Saved statewide parcel layer: {STATEWIDE_OUT}")

    total_row = pd.DataFrame(
        [{"county": "TOTAL", "file_name": "-", "rows": len(merged), "status": "statewide_after_dedupe"}]
    )
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved merge summary: {SUMMARY_CSV}")

    county_qa = build_county_qa(summary_df.rename(columns={"county": "county_name"}))
    county_qa.to_csv(COUNTY_QA_CSV, index=False)
    print(f"Saved county QA: {COUNTY_QA_CSV}")


if __name__ == "__main__":
    main()
