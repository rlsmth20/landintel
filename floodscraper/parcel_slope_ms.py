from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"
ELEVATION_PROCESSED_DIR = BASE_DIR / "data" / "elevation_processed"

# Mississippi-specific values (easy future config extraction).
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
TARGET_CRS = "EPSG:4326"
LIKELY_PARCEL_ID_FIELDS = ["parcel_id", "PARCEL_ID", "PARCELID", "PARCELNO", "PIN", "GLOBALID", "OBJECTID", "FID"]
INPUT_PARCEL_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood.gpkg",
    PARCELS_DIR / "mississippi_parcels.gpkg",
]
SLOPE_RASTER_DEFAULT = ELEVATION_PROCESSED_DIR / "mississippi_slope.tif"
OUTPUT_WITH_FLOOD = PARCELS_DIR / "mississippi_parcels_with_flood_and_slope.gpkg"
OUTPUT_WITHOUT_FLOOD = PARCELS_DIR / "mississippi_parcels_with_slope.gpkg"
SUMMARY_WITH_FLOOD = PARCELS_DIR / "mississippi_parcels_with_flood_and_slope_summary.csv"
SUMMARY_WITHOUT_FLOOD = PARCELS_DIR / "mississippi_parcels_with_slope_summary.csv"
PARTS_DIR = PARCELS_DIR / "ms_parcels_slope_parts"
CHECKPOINT_CSV = PARCELS_DIR / "mississippi_parcels_with_slope_progress.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Mississippi parcel-level slope metrics from slope raster.")
    parser.add_argument("--parcel-file", type=str, default="", help="Parcel GPKG input path. Defaults to flood-enriched if present.")
    parser.add_argument("--slope-raster", type=str, default=str(SLOPE_RASTER_DEFAULT), help="Processed slope raster path.")
    parser.add_argument("--output-file", type=str, default="", help="Parcel output GPKG path.")
    parser.add_argument("--summary-csv", type=str, default="", help="Summary CSV path.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Chunk output directory.")
    parser.add_argument("--checkpoint-csv", type=str, default=str(CHECKPOINT_CSV), help="Progress CSV path.")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Parcels per chunk per county.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county_name filter values.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and recompute all chunks.")
    return parser.parse_args()


def resolve_parcel_input(path_arg: str) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path
    for candidate in INPUT_PARCEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return INPUT_PARCEL_CANDIDATES[-1]


def resolve_output_paths(parcel_path: Path, output_arg: str, summary_arg: str) -> tuple[Path, Path]:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
    else:
        output_path = OUTPUT_WITH_FLOOD if "with_flood" in parcel_path.name.lower() else OUTPUT_WITHOUT_FLOOD

    if summary_arg:
        summary_path = Path(summary_arg)
        if not summary_path.is_absolute():
            summary_path = BASE_DIR / summary_path
    else:
        summary_path = SUMMARY_WITH_FLOOD if "with_flood" in parcel_path.name.lower() else SUMMARY_WITHOUT_FLOOD
    return output_path, summary_path


def normalize_to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(crs, allow_override=True)
    return gdf.to_crs(crs)


def ensure_parcel_keys(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = parcels.copy()
    out["parcel_row_id"] = out.index.astype(str).map(lambda i: f"row_{i}")

    cols_map = {str(c).upper(): c for c in out.columns}
    source_col = None
    for candidate in LIKELY_PARCEL_ID_FIELDS:
        hit = cols_map.get(candidate.upper())
        if hit:
            source_col = hit
            break

    if source_col:
        out["parcel_id"] = out[source_col].astype(str).str.strip()
        blank = out["parcel_id"].isnull() | (out["parcel_id"] == "") | (out["parcel_id"].str.lower() == "nan")
        out.loc[blank, "parcel_id"] = pd.NA
    elif "parcel_id" not in out.columns:
        out["parcel_id"] = pd.NA
    return out


def sanitize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


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


def load_checkpoint(path: Path) -> pd.DataFrame:
    cols = ["chunk_id", "status", "rows", "part_file"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=cols)
    if not set(cols).issubset(df.columns):
        return pd.DataFrame(columns=cols)
    return df[cols].copy()


def save_checkpoint(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def update_checkpoint(df: pd.DataFrame, chunk_id: str, status: str, rows: int, part_file: str) -> pd.DataFrame:
    row = pd.DataFrame([{"chunk_id": chunk_id, "status": status, "rows": rows, "part_file": part_file}])
    df = df[df["chunk_id"] != chunk_id].copy()
    return pd.concat([df, row], ignore_index=True)


def extract_chunk_slope_stats(
    src: rasterio.DatasetReader,
    chunk_proj: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if chunk_proj.empty:
        return pd.DataFrame(columns=["parcel_row_id", "mean_slope_pct", "max_slope_pct"])

    bounds = chunk_proj.total_bounds
    window = from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()

    col_off = max(0, int(window.col_off))
    row_off = max(0, int(window.row_off))
    width = max(1, int(min(window.width, src.width - col_off)))
    height = max(1, int(min(window.height, src.height - row_off)))
    clipped_window = rasterio.windows.Window(col_off=col_off, row_off=row_off, width=width, height=height)

    arr = src.read(1, window=clipped_window).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    window_transform = src.window_transform(clipped_window)

    out_rows: list[dict] = []
    for rec in chunk_proj.itertuples():
        geom = rec.geometry
        if geom is None or geom.is_empty:
            out_rows.append({"parcel_row_id": rec.parcel_row_id, "mean_slope_pct": np.nan, "max_slope_pct": np.nan})
            continue
        mask = geometry_mask([geom], transform=window_transform, out_shape=arr.shape, invert=True, all_touched=False)
        values = arr[mask]
        values = values[np.isfinite(values)]

        if values.size == 0:
            # Fallback for tiny polygons near cell edges.
            mask_touch = geometry_mask([geom], transform=window_transform, out_shape=arr.shape, invert=True, all_touched=True)
            values = arr[mask_touch]
            values = values[np.isfinite(values)]

        if values.size == 0:
            # Final fallback: sample a small neighborhood around centroid.
            cx, cy = geom.centroid.x, geom.centroid.y
            col, row = src.index(cx, cy)
            r0 = max(0, row - 2)
            c0 = max(0, col - 2)
            r1 = min(src.height, row + 3)
            c1 = min(src.width, col + 3)
            if r1 > r0 and c1 > c0:
                centroid_win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                local = src.read(1, window=centroid_win).astype(np.float32)
                local_nodata = src.nodata
                if local_nodata is not None:
                    local[local == local_nodata] = np.nan
                local = local[np.isfinite(local)]
                values = local

        if values.size == 0:
            mean_v = np.nan
            max_v = np.nan
        else:
            mean_v = float(np.mean(values))
            max_v = float(np.max(values))
        out_rows.append({"parcel_row_id": rec.parcel_row_id, "mean_slope_pct": mean_v, "max_slope_pct": max_v})
    return pd.DataFrame(out_rows)


def write_part_with_retry(gdf: gpd.GeoDataFrame, path: Path, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            gdf.to_file(path, driver="GPKG", engine="pyogrio")
            return
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed writing {path}") from last_exc


def merge_with_existing_counties(
    refreshed: gpd.GeoDataFrame,
    output_file: Path,
    selected_counties: list[str] | None,
) -> gpd.GeoDataFrame:
    if not selected_counties or not output_file.exists():
        return refreshed
    existing = gpd.read_file(output_file, engine="pyogrio")
    county_set = {str(value).lower() for value in selected_counties}
    keep_existing = existing[~existing["county_name"].astype(str).str.lower().isin(county_set)].copy()
    combined = pd.concat([keep_existing, refreshed], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=TARGET_CRS)


def build_expected_parts(parcels_4326: gpd.GeoDataFrame, county_col: str | None, county_values: list[str], chunk_size: int, parts_dir: Path) -> list[tuple[str, Path]]:
    expected: list[tuple[str, Path]] = []
    for county_value in county_values:
        group_label = county_value if county_col else "all_rows"
        if county_col:
            total_rows = int((parcels_4326[county_col].astype(str) == county_value).sum())
        else:
            total_rows = len(parcels_4326)
        n_parts = max(1, math.ceil(total_rows / chunk_size))
        county_slug = sanitize_name(group_label)
        for part_idx in range(n_parts):
            part_id = f"{group_label}__{part_idx + 1:03d}"
            part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_slope.gpkg"
            expected.append((part_id, part_path))
    return expected


def process_chunk(
    chunk_4326: gpd.GeoDataFrame,
    chunk_proj: gpd.GeoDataFrame,
    src: rasterio.DatasetReader,
) -> gpd.GeoDataFrame:
    stats = extract_chunk_slope_stats(src, chunk_proj)
    out = chunk_4326.merge(stats, on="parcel_row_id", how="left")
    class_score = out["mean_slope_pct"].apply(classify_slope)
    out["slope_class"] = class_score.apply(lambda x: x[0])
    out["slope_score"] = class_score.apply(lambda x: x[1]).astype(int)
    return out


def main() -> None:
    args = parse_args()
    parcel_file = resolve_parcel_input(args.parcel_file)
    slope_raster = Path(args.slope_raster)
    if not slope_raster.is_absolute():
        slope_raster = BASE_DIR / slope_raster
    output_file, summary_csv = resolve_output_paths(parcel_file, args.output_file, args.summary_csv)

    parts_dir = Path(args.parts_dir)
    checkpoint_csv = Path(args.checkpoint_csv)
    if not parts_dir.is_absolute():
        parts_dir = BASE_DIR / parts_dir
    if not checkpoint_csv.is_absolute():
        checkpoint_csv = BASE_DIR / checkpoint_csv
    chunk_size = max(1, int(args.chunk_size))
    resume = not args.no_resume

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATE: {STATE_NAME} ({STATE_ABBR})")
    print(f"Parcel file: {parcel_file}")
    print(f"Slope raster: {slope_raster}")
    print(f"Output file: {output_file}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Chunk size: {chunk_size:,}")

    if not parcel_file.exists():
        print(f"ERROR: Missing parcel file: {parcel_file}")
        return
    if not slope_raster.exists():
        print(f"ERROR: Missing slope raster: {slope_raster}")
        return

    print("Loading parcels...")
    parcels_4326 = normalize_to_crs(gpd.read_file(parcel_file), TARGET_CRS)
    parcels_4326 = parcels_4326[parcels_4326.geometry.notnull()].copy()
    parcels_4326 = parcels_4326[~parcels_4326.geometry.is_empty].copy()
    parcels_4326 = ensure_parcel_keys(parcels_4326)
    print(f"Parcels rows: {len(parcels_4326):,}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_df = load_checkpoint(checkpoint_csv)

    with rasterio.open(slope_raster) as src:
        raster_crs = src.crs
        if raster_crs is None:
            print("ERROR: Slope raster CRS is missing.")
            return
        print(f"Slope raster CRS: {raster_crs}")
        parcels_proj = parcels_4326.to_crs(raster_crs)

        county_col = "county_name" if "county_name" in parcels_4326.columns else None
        county_values = sorted(parcels_4326[county_col].astype(str).unique().tolist()) if county_col else ["all_rows"]
        if county_col and args.counties:
            wanted = {str(c).strip().lower() for c in args.counties}
            county_values = [c for c in county_values if c.lower() in wanted]
            print(f"County filter active: {len(county_values)} counties")
        print(f"Processing chunk groups: {len(county_values)}")
        expected_parts = build_expected_parts(parcels_4326, county_col, county_values, chunk_size, parts_dir)
        expected_ids = {chunk_id for chunk_id, _ in expected_parts}

        for county_value in county_values:
            group_label = county_value if county_col else "all_rows"
            if county_col:
                mask = parcels_4326[county_col].astype(str) == county_value
                county_4326 = parcels_4326[mask].copy()
                county_proj = parcels_proj[mask].copy()
            else:
                county_4326 = parcels_4326.copy()
                county_proj = parcels_proj.copy()

            total_rows = len(county_4326)
            n_parts = max(1, math.ceil(total_rows / chunk_size))
            county_slug = sanitize_name(group_label)
            print(f"\nGroup: {group_label} ({total_rows:,} parcels, {n_parts} part(s))")

            for part_idx in range(n_parts):
                start_i = part_idx * chunk_size
                end_i = min((part_idx + 1) * chunk_size, total_rows)
                part_id = f"{group_label}__{part_idx + 1:03d}"
                part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_slope.gpkg"

                if resume and part_path.exists():
                    done = checkpoint_df[
                        (checkpoint_df["chunk_id"] == part_id) & (checkpoint_df["status"] == "completed")
                    ]
                    if not done.empty:
                        print(f"  Skipping completed part: {part_id}")
                        continue

                chunk_4326 = county_4326.iloc[start_i:end_i].copy()
                chunk_proj = county_proj.iloc[start_i:end_i].copy()
                print(f"  Processing {part_id} ({len(chunk_4326):,} parcels)")

                t0 = time.time()
                try:
                    result = process_chunk(chunk_4326, chunk_proj, src)
                    write_part_with_retry(result, part_path)
                    elapsed = time.time() - t0
                    print(f"  Saved {part_path.name} ({len(result):,} rows, {elapsed:.1f}s)")
                    checkpoint_df = update_checkpoint(
                        checkpoint_df,
                        chunk_id=part_id,
                        status="completed",
                        rows=len(result),
                        part_file=str(part_path.relative_to(BASE_DIR)),
                    )
                    save_checkpoint(checkpoint_csv, checkpoint_df)
                except Exception as exc:
                    print(f"  Failed {part_id}: {exc}")
                    checkpoint_df = update_checkpoint(
                        checkpoint_df,
                        chunk_id=part_id,
                        status="failed",
                        rows=0,
                        part_file=str(part_path.relative_to(BASE_DIR)),
                    )
                    save_checkpoint(checkpoint_csv, checkpoint_df)

    completed_map: dict[str, Path] = {}
    completed_rows = checkpoint_df[checkpoint_df["status"] == "completed"].copy()
    for row in completed_rows.itertuples():
        if row.chunk_id not in expected_ids:
            continue
        part_path = BASE_DIR / str(row.part_file)
        if part_path.exists():
            completed_map[str(row.chunk_id)] = part_path

    missing = [chunk_id for chunk_id, _ in expected_parts if chunk_id not in completed_map]
    if missing:
        print(f"Missing {len(missing):,} expected slope parts; skipping merge.")
        print(f"First missing parts: {', '.join(missing[:10])}")
        return

    completed_parts = [completed_map[chunk_id] for chunk_id, _ in expected_parts]
    if not completed_parts:
        print("No completed slope parts found; nothing to merge.")
        return

    print(f"\nMerging {len(completed_parts)} completed part files...")
    frames = [gpd.read_file(path) for path in completed_parts]
    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    if county_col and args.counties:
        merged = merge_with_existing_counties(merged, output_file, county_values)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_file, driver="GPKG", engine="pyogrio")
    print(f"Saved output: {output_file}")

    valid_mean = merged["mean_slope_pct"][np.isfinite(merged["mean_slope_pct"])]
    summary_rows = [
        {"metric": "rows_total", "value": int(len(merged))},
        {"metric": "rows_with_mean_slope", "value": int(np.isfinite(merged["mean_slope_pct"]).sum())},
        {"metric": "mean_slope_min", "value": float(valid_mean.min()) if not valid_mean.empty else None},
        {"metric": "mean_slope_median", "value": float(valid_mean.median()) if not valid_mean.empty else None},
        {"metric": "mean_slope_p95", "value": float(valid_mean.quantile(0.95)) if not valid_mean.empty else None},
        {"metric": "mean_slope_max", "value": float(valid_mean.max()) if not valid_mean.empty else None},
    ]
    class_counts = merged["slope_class"].value_counts(dropna=False).to_dict()
    for key, value in class_counts.items():
        summary_rows.append({"metric": f"class_{key}", "value": int(value)})
    summary_df = pd.DataFrame(summary_rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
