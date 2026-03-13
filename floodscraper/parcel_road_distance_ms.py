from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, box

BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"
ROADS_DIR = BASE_DIR / "data" / "roads" / "mississippi-260305-free.shp"

PARCEL_FILE = PARCELS_DIR / "mississippi_parcels.gpkg"
ROAD_FILE = ROADS_DIR / "gis_osm_roads_free_1.shp"
OUTPUT_FILE = PARCELS_DIR / "mississippi_parcels_with_roads.gpkg"
SUMMARY_CSV = PARCELS_DIR / "mississippi_parcels_with_roads_summary.csv"
PARTS_DIR = PARCELS_DIR / "ms_parcels_road_parts"
CHECKPOINT_CSV = PARCELS_DIR / "mississippi_parcels_with_roads_progress.csv"

TARGET_CRS = "EPSG:4326"
DISTANCE_CRS = "EPSG:3857"  # Projected CRS needed for distance in meters.
LIKELY_PARCEL_ID_FIELDS = ["parcel_id", "PARCEL_ID", "PARCELID", "PARCELNO", "PIN", "GLOBALID", "OBJECTID", "FID"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate nearest-road distance for Mississippi parcels.")
    parser.add_argument("--parcel-file", type=str, default=str(PARCEL_FILE), help="Parcel GPKG input path.")
    parser.add_argument("--road-file", type=str, default=str(ROAD_FILE), help="Road line dataset input path.")
    parser.add_argument("--output-file", type=str, default=str(OUTPUT_FILE), help="Output GPKG path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Summary CSV path.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Directory for per-county chunk outputs.")
    parser.add_argument("--checkpoint-csv", type=str, default=str(CHECKPOINT_CSV), help="Checkpoint CSV path.")
    parser.add_argument("--chunk-size", type=int, default=15000, help="Parcels per sub-chunk within each county.")
    parser.add_argument("--bbox-buffer-m", type=float, default=5000.0, help="Road subset bbox buffer in meters.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county_name filters.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and recompute all chunks.")
    return parser.parse_args()


def normalize_to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(crs, allow_override=True)
    return gdf.to_crs(crs)


def is_line_geom(geom) -> bool:
    return isinstance(geom, (LineString, MultiLineString))


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
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
    )


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


def write_gpkg_with_retry(gdf: gpd.GeoDataFrame, path: Path, retries: int = 3) -> None:
    last_exc = None
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            gdf.to_file(path, driver="GPKG", engine="pyogrio")
            return
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed writing GPKG after {retries} attempts: {path}") from last_exc


def build_expected_parts(parcels_4326: gpd.GeoDataFrame, county_col: str | None, county_values: list[str], chunk_size: int, parts_dir: Path) -> list[tuple[str, Path]]:
    expected: list[tuple[str, Path]] = []
    for county_value in county_values:
        county_id = county_value if county_col else "all_rows"
        if county_col:
            total_rows = int((parcels_4326[county_col].astype(str) == county_value).sum())
        else:
            total_rows = len(parcels_4326)
        n_parts = max(1, math.ceil(total_rows / chunk_size))
        county_slug = sanitize_name(county_id)
        for part_idx in range(n_parts):
            part_id = f"{county_id}__{part_idx + 1:03d}"
            part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_road_distance.gpkg"
            expected.append((part_id, part_path))
    return expected


def process_chunk(
    chunk_4326: gpd.GeoDataFrame,
    chunk_3857: gpd.GeoDataFrame,
    roads_3857: gpd.GeoDataFrame,
    roads_sindex,
    bbox_buffer_m: float,
    chunk_label: str,
) -> gpd.GeoDataFrame:
    if chunk_4326.empty:
        return chunk_4326

    bounds = chunk_3857.total_bounds
    query_geom = box(bounds[0] - bbox_buffer_m, bounds[1] - bbox_buffer_m, bounds[2] + bbox_buffer_m, bounds[3] + bbox_buffer_m)
    road_idx = roads_sindex.query(query_geom, predicate="intersects")
    roads_subset = roads_3857.iloc[road_idx].copy() if len(road_idx) else roads_3857.iloc[0:0].copy()

    if roads_subset.empty:
        print(f"  No roads in subset for {chunk_label}; falling back to full roads.")
        roads_subset = roads_3857

    left = chunk_3857[["parcel_row_id", "geometry"]].copy()
    right_cols = [c for c in ["osm_id", "fclass", "name", "ref", "geometry"] if c in roads_subset.columns]
    right = roads_subset[right_cols].copy()

    joined = gpd.sjoin_nearest(left, right, how="left", distance_col="road_distance_m")
    joined = joined.sort_values("road_distance_m", na_position="last").drop_duplicates(subset=["parcel_row_id"]).copy()

    keep_cols = [c for c in ["parcel_row_id", "osm_id", "fclass", "name", "ref", "road_distance_m"] if c in joined.columns]
    road_attrs = joined[keep_cols].copy()
    out = chunk_4326.merge(road_attrs, on="parcel_row_id", how="left")
    out["road_distance_m"] = out["road_distance_m"].fillna(-1.0)
    out["road_distance_ft"] = out["road_distance_m"].apply(lambda v: -1.0 if v < 0 else float(v) * 3.28084)
    return out


def merge_with_existing_counties(
    refreshed: gpd.GeoDataFrame,
    output_file: Path,
    selected_counties: list[str],
) -> gpd.GeoDataFrame:
    if not selected_counties or not output_file.exists():
        return refreshed
    existing = gpd.read_file(output_file, engine="pyogrio")
    county_set = {str(value).lower() for value in selected_counties}
    keep_existing = existing[~existing["county_name"].astype(str).str.lower().isin(county_set)].copy()
    combined = pd.concat([keep_existing, refreshed], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=TARGET_CRS)


def main() -> None:
    args = parse_args()

    parcel_file = Path(args.parcel_file)
    road_file = Path(args.road_file)
    output_file = Path(args.output_file)
    summary_csv = Path(args.summary_csv)
    parts_dir = Path(args.parts_dir)
    checkpoint_csv = Path(args.checkpoint_csv)

    if not parcel_file.is_absolute():
        parcel_file = BASE_DIR / parcel_file
    if not road_file.is_absolute():
        road_file = BASE_DIR / road_file
    if not output_file.is_absolute():
        output_file = BASE_DIR / output_file
    if not summary_csv.is_absolute():
        summary_csv = BASE_DIR / summary_csv
    if not parts_dir.is_absolute():
        parts_dir = BASE_DIR / parts_dir
    if not checkpoint_csv.is_absolute():
        checkpoint_csv = BASE_DIR / checkpoint_csv
    chunk_size = max(1, int(args.chunk_size))
    bbox_buffer_m = float(args.bbox_buffer_m)
    resume = not args.no_resume

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Parcel file: {parcel_file}")
    print(f"Road file:   {road_file}")
    print(f"Output file: {output_file}")
    print(f"Chunk size:  {chunk_size:,}")
    print(f"BBox buffer: {bbox_buffer_m:,.0f} m\n")

    if not parcel_file.exists():
        print(f"ERROR: Missing parcel file: {parcel_file}")
        return
    if not road_file.exists():
        print(f"ERROR: Missing road file: {road_file}")
        return

    print("Loading parcels...")
    parcels_4326 = normalize_to_crs(gpd.read_file(parcel_file), TARGET_CRS)
    parcels_4326 = ensure_parcel_keys(parcels_4326)
    print(f"Parcels rows: {len(parcels_4326):,}")

    print("Loading roads...")
    roads_4326 = normalize_to_crs(gpd.read_file(road_file), TARGET_CRS)
    roads_4326 = roads_4326[roads_4326.geometry.notnull()].copy()
    roads_4326 = roads_4326[~roads_4326.geometry.is_empty].copy()
    roads_4326 = roads_4326[roads_4326.geometry.apply(is_line_geom)].copy()
    print(f"Road rows: {len(roads_4326):,}")

    print(f"Projecting parcels and roads to {DISTANCE_CRS} for distance calculations...")
    parcels_3857 = parcels_4326.to_crs(DISTANCE_CRS)
    roads_3857 = roads_4326.to_crs(DISTANCE_CRS)
    roads_sindex = roads_3857.sindex

    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_df = load_checkpoint(checkpoint_csv)

    county_col = "county_name" if "county_name" in parcels_4326.columns else None
    if county_col:
        county_values = sorted(parcels_4326[county_col].astype(str).unique().tolist())
        if args.counties:
            wanted = {str(c).strip().lower() for c in args.counties}
            county_values = [c for c in county_values if c.lower() in wanted]
        print(f"Processing by county chunks: {len(county_values)} counties")
    else:
        county_values = ["all_rows"]
        print("No county_name column found; processing single chunk group.")

    expected_parts = build_expected_parts(parcels_4326, county_col, county_values, chunk_size, parts_dir)
    expected_ids = {chunk_id for chunk_id, _ in expected_parts}

    for county_value in county_values:
        county_id = county_value if county_col else "all_rows"
        if county_col:
            mask = parcels_4326[county_col].astype(str) == county_value
            county_4326 = parcels_4326[mask].copy()
            county_3857 = parcels_3857[mask].copy()
        else:
            county_4326 = parcels_4326.copy()
            county_3857 = parcels_3857.copy()

        total_rows = len(county_4326)
        n_parts = max(1, math.ceil(total_rows / chunk_size))
        county_slug = sanitize_name(county_id)
        print(f"\nCounty/chunk group: {county_id} ({total_rows:,} parcels, {n_parts} part(s))")

        for part_idx in range(n_parts):
            start_i = part_idx * chunk_size
            end_i = min((part_idx + 1) * chunk_size, total_rows)
            part_id = f"{county_id}__{part_idx + 1:03d}"
            part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_road_distance.gpkg"

            if resume and part_path.exists():
                done = checkpoint_df[
                    (checkpoint_df["chunk_id"] == part_id) & (checkpoint_df["status"] == "completed")
                ]
                if not done.empty:
                    print(f"  Skipping completed part: {part_id}")
                    continue

            part_4326 = county_4326.iloc[start_i:end_i].copy()
            part_3857 = county_3857.iloc[start_i:end_i].copy()

            print(f"  Processing part {part_idx + 1}/{n_parts}: {part_id} ({len(part_4326):,} parcels)")
            t0 = time.time()
            try:
                part_result = process_chunk(
                    part_4326,
                    part_3857,
                    roads_3857,
                    roads_sindex,
                    bbox_buffer_m=bbox_buffer_m,
                    chunk_label=part_id,
                )
                write_gpkg_with_retry(part_result, part_path)
                elapsed = time.time() - t0
                print(f"  Saved: {part_path.name} ({len(part_result):,} rows, {elapsed:.1f}s)")
                checkpoint_df = update_checkpoint(
                    checkpoint_df,
                    chunk_id=part_id,
                    status="completed",
                    rows=len(part_result),
                    part_file=str(part_path.relative_to(BASE_DIR)),
                )
                save_checkpoint(checkpoint_csv, checkpoint_df)
            except Exception as exc:
                print(f"  Part failed: {part_id} -> {exc}")
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
        print(f"Missing {len(missing):,} expected road parts; skipping statewide merge.")
        print(f"First missing parts: {', '.join(missing[:10])}")
        return

    completed_parts = [completed_map[chunk_id] for chunk_id, _ in expected_parts]
    if not completed_parts:
        print("No completed part files found; nothing to merge.")
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

    valid_dist = merged["road_distance_m"][merged["road_distance_m"] >= 0].copy()
    summary_rows = [
        {"metric": "rows_total", "value": int(len(merged))},
        {"metric": "rows_with_road_match", "value": int((merged["road_distance_m"] >= 0).sum())},
        {"metric": "distance_m_min", "value": float(valid_dist.min()) if not valid_dist.empty else None},
        {"metric": "distance_m_median", "value": float(valid_dist.median()) if not valid_dist.empty else None},
        {"metric": "distance_m_p95", "value": float(valid_dist.quantile(0.95)) if not valid_dist.empty else None},
        {"metric": "distance_m_max", "value": float(valid_dist.max()) if not valid_dist.empty else None},
        {"metric": "within_100m", "value": int((valid_dist <= 100).sum()) if not valid_dist.empty else 0},
        {"metric": "within_500m", "value": int((valid_dist <= 500).sum()) if not valid_dist.empty else 0},
        {"metric": "within_1000m", "value": int((valid_dist <= 1000).sum()) if not valid_dist.empty else 0},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
