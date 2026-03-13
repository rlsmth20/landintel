from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = Path(__file__).resolve().parents[1]
ELEVATION_RAW_DIR = BASE_DIR / "data" / "elevation_raw"
ELEVATION_PROCESSED_DIR = BASE_DIR / "data" / "elevation_processed"
PARCELS_DIR = BASE_DIR / "data" / "parcels"

RAW_MANIFEST = ELEVATION_RAW_DIR / "mississippi_elevation_download_manifest.csv"
DEM_FILE = ELEVATION_PROCESSED_DIR / "mississippi_dem.tif"
SLOPE_FILE = ELEVATION_PROCESSED_DIR / "mississippi_slope.tif"
PARCEL_OUTPUT_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood_and_slope.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_slope.gpkg",
    PARCELS_DIR / "adams_parcels_with_flood_and_slope.gpkg",
]
SUMMARY_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood_and_slope_summary.csv",
    PARCELS_DIR / "mississippi_parcels_with_slope_summary.csv",
    PARCELS_DIR / "adams_parcels_with_flood_and_slope_summary.csv",
]


def print_raster_info(path: Path, label: str) -> None:
    print(f"\n[{label}] {path}")
    if not path.exists():
        print("  MISSING")
        return
    with rasterio.open(path) as src:
        print(f"  CRS: {src.crs}")
        print(f"  Size: {src.width} x {src.height} x {src.count}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Resolution: {src.res}")
        print(f"  Nodata: {src.nodata}")


def pick_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def print_parcel_info(path: Path) -> None:
    print(f"\n[Parcel Output] {path}")
    if not path.exists():
        print("  MISSING")
        return
    gdf = gpd.read_file(path)
    print(f"  Rows: {len(gdf):,}")
    print(f"  CRS: {gdf.crs}")
    geom_types = gdf.geometry.geom_type.value_counts(dropna=False).to_dict()
    print(f"  Geometry types: {geom_types}")

    if "slope_class" in gdf.columns:
        counts = gdf["slope_class"].astype(str).value_counts(dropna=False).head(10)
        print("  Slope class counts:")
        for key, value in counts.items():
            print(f"    {key}: {value:,}")

    if "mean_slope_pct" in gdf.columns:
        vals = pd.to_numeric(gdf["mean_slope_pct"], errors="coerce")
        finite = vals[np.isfinite(vals)]
        print(f"  Non-null mean slope rows: {int(np.isfinite(vals).sum()):,}")
        if not finite.empty:
            print(
                "  Mean slope stats: "
                f"min={float(finite.min()):.2f}, median={float(finite.median()):.2f}, max={float(finite.max()):.2f}"
            )
    sample_cols = [c for c in ["parcel_id", "county_name", "mean_slope_pct", "max_slope_pct", "slope_class", "slope_score"] if c in gdf.columns]
    if sample_cols:
        print("  Sample rows:")
        print(gdf[sample_cols].head(5).to_string(index=False))


def main() -> None:
    print(f"BASE_DIR: {BASE_DIR}")
    print("Running Mississippi slope pipeline validation...")

    print(f"\n[Raw Manifest] {RAW_MANIFEST}")
    if RAW_MANIFEST.exists():
        manifest = pd.read_csv(RAW_MANIFEST)
        print(f"  Rows: {len(manifest):,}")
        if "status" in manifest.columns:
            print(f"  Status counts: {manifest['status'].value_counts(dropna=False).to_dict()}")
    else:
        print("  MISSING")

    print_raster_info(DEM_FILE, "DEM Raster")
    print_raster_info(SLOPE_FILE, "Slope Raster")

    parcel_output = pick_existing(PARCEL_OUTPUT_CANDIDATES)
    if parcel_output is None:
        print("\nNo slope parcel output found.")
    else:
        print_parcel_info(parcel_output)

    summary_csv = pick_existing(SUMMARY_CANDIDATES)
    print(f"\n[Summary CSV] {summary_csv}")
    if summary_csv and summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        print(f"  Rows: {len(summary):,}")
        print(summary.head(20).to_string(index=False))
    else:
        print("  MISSING")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
