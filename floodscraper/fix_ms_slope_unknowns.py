from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"
ELEVATION_PROCESSED_DIR = BASE_DIR / "data" / "elevation_processed"

PARTS_DIR = PARCELS_DIR / "ms_parcels_slope_parts"
SLOPE_RASTER = ELEVATION_PROCESSED_DIR / "mississippi_slope.tif"
REPORT_CSV = PARCELS_DIR / "mississippi_slope_unknown_fix_report.csv"


def classify_slope(mean_slope_pct: float | None) -> tuple[str, int]:
    if mean_slope_pct is None or not np.isfinite(mean_slope_pct):
        return "unknown", 0
    value = float(mean_slope_pct)
    if value <= 5.0:
        return "excellent", 0
    if value <= 10.0:
        return "good", 2
    if value <= 15.0:
        return "moderate", 4
    if value <= 25.0:
        return "difficult", 7
    return "poor", 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill unknown parcel slope rows using centroid-neighborhood fallback.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Slope parts directory.")
    parser.add_argument("--slope-raster", type=str, default=str(SLOPE_RASTER), help="Slope raster path.")
    parser.add_argument("--report-csv", type=str, default=str(REPORT_CSV), help="Per-part fix summary CSV.")
    parser.add_argument("--window-radius", type=int, default=2, help="Centroid neighborhood radius in pixels.")
    parser.add_argument("--limit-parts", type=int, default=0, help="Optional cap on number of part files for testing.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county slug filters (e.g., simpson jones).")
    return parser.parse_args()


def sample_centroid_fallback(src: rasterio.DatasetReader, geom, radius: int) -> tuple[float, float]:
    if geom is None or geom.is_empty:
        return np.nan, np.nan
    cx, cy = geom.centroid.x, geom.centroid.y
    col, row = src.index(cx, cy)
    r0 = max(0, row - radius)
    c0 = max(0, col - radius)
    r1 = min(src.height, row + radius + 1)
    c1 = min(src.width, col + radius + 1)
    if r1 <= r0 or c1 <= c0:
        return np.nan, np.nan
    window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
    arr = src.read(1, window=window).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.max(vals))


def main() -> None:
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    slope_raster = Path(args.slope_raster)
    report_csv = Path(args.report_csv)
    if not parts_dir.is_absolute():
        parts_dir = BASE_DIR / parts_dir
    if not slope_raster.is_absolute():
        slope_raster = BASE_DIR / slope_raster
    if not report_csv.is_absolute():
        report_csv = BASE_DIR / report_csv

    if not parts_dir.exists():
        print(f"ERROR: parts dir missing: {parts_dir}")
        return
    if not slope_raster.exists():
        print(f"ERROR: slope raster missing: {slope_raster}")
        return

    part_files = sorted(parts_dir.glob("*_with_slope.gpkg"))
    if args.counties:
        wanted = {str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in args.counties}
        part_files = [p for p in part_files if p.name.split("__")[0].lower() in wanted]
    if args.limit_parts > 0:
        part_files = part_files[: int(args.limit_parts)]
    print(f"Part files to check: {len(part_files)}")

    report_rows: list[dict] = []
    updated_total = 0
    unknown_before_total = 0
    unknown_after_total = 0

    with rasterio.open(slope_raster) as src:
        raster_crs = src.crs
        if raster_crs is None:
            print("ERROR: slope raster CRS missing")
            return

        for idx, part_path in enumerate(part_files, start=1):
            gdf = gpd.read_file(part_path)
            if gdf.empty:
                report_rows.append(
                    {"part_file": part_path.name, "rows": 0, "unknown_before": 0, "updated": 0, "unknown_after": 0}
                )
                continue

            unknown_mask = (~np.isfinite(pd.to_numeric(gdf.get("mean_slope_pct"), errors="coerce"))) | (
                gdf.get("slope_class", pd.Series(index=gdf.index, dtype="object")).astype(str).str.lower() == "unknown"
            )
            unknown_before = int(unknown_mask.sum())
            unknown_before_total += unknown_before
            if unknown_before == 0:
                report_rows.append(
                    {
                        "part_file": part_path.name,
                        "rows": int(len(gdf)),
                        "unknown_before": 0,
                        "updated": 0,
                        "unknown_after": 0,
                    }
                )
                if idx % 50 == 0 or idx == len(part_files):
                    print(f"[{idx}/{len(part_files)}] {part_path.name}: no unknown rows")
                continue

            subset = gdf.loc[unknown_mask].copy().to_crs(raster_crs)
            updates: list[tuple[int, float, float, str, int]] = []
            for row_idx, rec in zip(gdf.loc[unknown_mask].index.tolist(), subset.itertuples()):
                mean_v, max_v = sample_centroid_fallback(src, rec.geometry, radius=int(args.window_radius))
                if np.isfinite(mean_v):
                    slope_class, slope_score = classify_slope(mean_v)
                    updates.append((row_idx, mean_v, max_v, slope_class, slope_score))

            for row_idx, mean_v, max_v, slope_class, slope_score in updates:
                gdf.at[row_idx, "mean_slope_pct"] = mean_v
                gdf.at[row_idx, "max_slope_pct"] = max_v
                gdf.at[row_idx, "slope_class"] = slope_class
                gdf.at[row_idx, "slope_score"] = slope_score

            unknown_after_mask = (~np.isfinite(pd.to_numeric(gdf.get("mean_slope_pct"), errors="coerce"))) | (
                gdf.get("slope_class", pd.Series(index=gdf.index, dtype="object")).astype(str).str.lower() == "unknown"
            )
            unknown_after = int(unknown_after_mask.sum())
            unknown_after_total += unknown_after
            updated = int(len(updates))
            updated_total += updated

            if updated > 0:
                gdf.to_file(part_path, driver="GPKG", engine="pyogrio")

            report_rows.append(
                {
                    "part_file": part_path.name,
                    "rows": int(len(gdf)),
                    "unknown_before": unknown_before,
                    "updated": updated,
                    "unknown_after": unknown_after,
                }
            )
            print(
                f"[{idx}/{len(part_files)}] {part_path.name}: unknown {unknown_before} -> {unknown_after} (updated {updated})"
            )

    report_df = pd.DataFrame(report_rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)
    print(f"\nUnknown before total: {unknown_before_total:,}")
    print(f"Updated total:        {updated_total:,}")
    print(f"Unknown after total:  {unknown_after_total:,}")
    print(f"Report written: {report_csv}")


if __name__ == "__main__":
    main()
