from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"
FLOOD_DIR = BASE_DIR / "data" / "flood_layers"
FEMA_DOWNLOADS_DIR = BASE_DIR / "data" / "fema_downloads"

STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
LIKELY_PARCEL_ID_FIELDS = ["parcel_id", "PARCEL_ID", "PARCELID", "PARCELNO", "PIN", "GLOBALID", "OBJECTID", "FID"]

PARCEL_FILE = PARCELS_DIR / "mississippi_parcels.gpkg"
FLOOD_FILE = FLOOD_DIR / "mississippi_fema_flood.gpkg"
OUTPUT_FILE = PARCELS_DIR / "mississippi_parcels_with_flood.gpkg"
SUMMARY_CSV = PARCELS_DIR / "mississippi_parcels_with_flood_summary.csv"
PARTS_DIR = PARCELS_DIR / "ms_parcels_flood_parts"
CHECKPOINT_CSV = PARCELS_DIR / "mississippi_parcels_with_flood_progress.csv"
TARGET_CRS = "EPSG:4326"
AREA_CRS = "EPSG:5070"
ACRES_PER_SQM = 0.0002471053814671653

RISK_MAP = {
    "UNKNOWN": 0,
    "X": 1,
    "X_500": 2,
    "AREA_NOT_INCLUDED": 0,
    "OPEN_WATER": 2,
    "D": 4,
    "A": 8,
    "AO": 8,
    "AH": 8,
    "AE": 10,
    "VE": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join Mississippi parcels with FEMA flood polygons.")
    parser.add_argument("--parcel-file", type=str, default=str(PARCEL_FILE), help="Parcel GPKG input path.")
    parser.add_argument("--flood-file", type=str, default=str(FLOOD_FILE), help="Flood GPKG input path.")
    parser.add_argument("--output-file", type=str, default=str(OUTPUT_FILE), help="Output GPKG path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Summary CSV path.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Directory for per-county join parts.")
    parser.add_argument("--checkpoint-csv", type=str, default=str(CHECKPOINT_CSV), help="CSV tracking completed chunks.")
    parser.add_argument("--chunk-size", type=int, default=20000, help="Parcels per processing sub-chunk within each county.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county_name filters.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint data.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and rebuild all part files.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def normalize_to_target_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(TARGET_CRS, allow_override=True)
    return gdf.to_crs(TARGET_CRS)


def sanitize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


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

    if source_col is not None:
        raw = out[source_col].astype("string").str.strip()
        bad = raw.isna() | (raw == "") | (raw.str.lower() == "nan") | (raw.str.lower() == "<na>")
        out["parcel_id"] = raw
        out.loc[bad, "parcel_id"] = pd.NA
    elif "parcel_id" not in out.columns:
        out["parcel_id"] = pd.NA
    else:
        raw = out["parcel_id"].astype("string").str.strip()
        bad = raw.isna() | (raw == "") | (raw.str.lower() == "nan") | (raw.str.lower() == "<na>")
        out["parcel_id"] = raw
        out.loc[bad, "parcel_id"] = pd.NA

    out["parcel_key"] = out["parcel_row_id"]
    return out


def normalize_zone_subtype(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_flood_zone(zone_value, subtype_value=None) -> str:
    zone = "UNKNOWN" if zone_value is None or pd.isna(zone_value) else str(zone_value).strip().upper()
    subtype = normalize_zone_subtype(subtype_value)
    if zone in {"", "NONE", "NULL", "UNKNOWN"}:
        return "UNKNOWN"
    if zone == "X" and "0.2 PCT" in subtype:
        return "X_500"
    if zone == "OPEN WATER":
        return "OPEN_WATER"
    if zone == "AREA NOT INCLUDED":
        return "AREA_NOT_INCLUDED"
    return zone.replace(" ", "_")


def risk_score_for_zone(zone: str) -> int:
    return int(RISK_MAP.get(str(zone), 0))


def zone_sort_key(zone: str) -> tuple[int, str]:
    return (risk_score_for_zone(zone), str(zone))


def print_layer_debug(label: str, gdf: gpd.GeoDataFrame) -> None:
    print(f"{label} CRS: {gdf.crs}")
    print(f"{label} bounds: {gdf.total_bounds}")
    print(f"{label} rows: {len(gdf):,}")


def bbox_overlap(bounds_a, bounds_b) -> bool:
    return not (
        bounds_a[2] < bounds_b[0]
        or bounds_a[0] > bounds_b[2]
        or bounds_a[3] < bounds_b[1]
        or bounds_a[1] > bounds_b[3]
    )


def extract_county_fips(parcels: gpd.GeoDataFrame) -> str | None:
    candidates = ["STCNTYFIPS", "CNTYFIPS", "COUNTYFP", "COUNTY_FIPS"]
    cols_map = {str(c).upper(): c for c in parcels.columns}
    for candidate in candidates:
        source_col = cols_map.get(candidate.upper())
        if source_col is None:
            continue
        vals = (
            parcels[source_col]
            .dropna()
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.replace(r"\D+", "", regex=True)
            .str.zfill(5)
        )
        vals = vals[vals.str.len() == 5]
        if vals.empty:
            continue
        return vals.value_counts().index[0]
    return None


def load_county_fema_flood(county_fips: str) -> gpd.GeoDataFrame | None:
    candidates = sorted(FEMA_DOWNLOADS_DIR.glob(f"{county_fips}C_*.zip"))
    if not candidates:
        return None
    zip_path = candidates[-1]
    zip_uri = f"zip://{zip_path}!S_FLD_HAZ_AR.shp"
    try:
        flood = gpd.read_file(zip_uri, engine="pyogrio")
    except Exception:
        return None
    if "FLD_ZONE" not in flood.columns:
        return None
    return normalize_to_target_crs(flood)


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
    raise RuntimeError(f"Failed writing part file: {path}") from last_exc


def read_part_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        df = gpd.read_file(path, columns=["parcel_row_id"], ignore_geometry=True, engine="pyogrio")
    except Exception:
        return None
    return int(len(df))


def build_expected_parts(
    parcels: gpd.GeoDataFrame,
    county_col: str | None,
    chunk_values: list[str],
    chunk_size: int,
    parts_dir: Path,
) -> list[tuple[str, Path]]:
    expected: list[tuple[str, Path]] = []
    for county_value in chunk_values:
        county_id = county_value if county_col else "all_rows"
        if county_col:
            total_rows = int((parcels[county_col].astype(str) == county_value).sum())
        else:
            total_rows = len(parcels)
        n_parts = max(1, math.ceil(total_rows / chunk_size))
        county_slug = sanitize_name(county_id)
        for part_idx in range(n_parts):
            part_id = f"{county_id}__{part_idx + 1:03d}"
            part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_flood.gpkg"
            expected.append((part_id, part_path))
    return expected


def process_chunk(
    parcels_chunk: gpd.GeoDataFrame,
    flood_small: gpd.GeoDataFrame,
    flood_sindex,
    chunk_label: str,
) -> gpd.GeoDataFrame:
    if parcels_chunk.empty:
        return parcels_chunk

    bounds_geom = box(*parcels_chunk.total_bounds)
    flood_idx = flood_sindex.query(bounds_geom, predicate="intersects")
    flood_subset = flood_small.iloc[flood_idx].copy() if len(flood_idx) else flood_small.iloc[0:0].copy()
    print(f"  Flood subset rows for {chunk_label}: {len(flood_subset):,}")

    out = parcels_chunk.copy()
    if flood_subset.empty:
        out["flood_zone_primary"] = "UNKNOWN"
        out["flood_zone_list"] = "UNKNOWN"
        out["flood_risk_score"] = 0
        out["has_flood_overlap"] = False
        out["sfha_overlap"] = False
        out["flood_overlap_acres"] = 0.0
        out["flood_overlap_pct"] = 0.0
        return out

    joined = gpd.sjoin(
        parcels_chunk[["parcel_row_id", "geometry"]],
        flood_subset[["flood_zone", "flood_risk_score", "sfha_flag", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined["flood_zone"] = joined["flood_zone"].fillna("UNKNOWN")
    joined["flood_risk_score"] = joined["flood_risk_score"].fillna(0).astype(int)
    joined["sfha_flag"] = joined["sfha_flag"].fillna(False).astype(bool)

    parcel_area_acres = parcels_chunk[["parcel_row_id", "geometry"]].to_crs(AREA_CRS)
    parcel_area_acres["parcel_area_acres"] = parcel_area_acres.geometry.area * ACRES_PER_SQM
    flood_area = flood_subset[["flood_zone", "geometry"]].to_crs(AREA_CRS)
    intersections = gpd.overlay(parcel_area_acres[["parcel_row_id", "geometry"]], flood_area, how="intersection", keep_geom_type=False)
    if intersections.empty:
        flood_coverage = parcel_area_acres[["parcel_row_id", "parcel_area_acres"]].copy()
        flood_coverage["flood_overlap_acres"] = 0.0
    else:
        intersections["int_area_acres"] = intersections.geometry.area * ACRES_PER_SQM
        flood_coverage = intersections.groupby("parcel_row_id", as_index=False).agg(flood_overlap_acres=("int_area_acres", "sum"))
        flood_coverage = parcel_area_acres[["parcel_row_id", "parcel_area_acres"]].merge(flood_coverage, on="parcel_row_id", how="left")
        flood_coverage["flood_overlap_acres"] = flood_coverage["flood_overlap_acres"].fillna(0.0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(
            flood_coverage["parcel_area_acres"] > 0,
            (flood_coverage["flood_overlap_acres"] / flood_coverage["parcel_area_acres"]) * 100.0,
            0.0,
        )
    flood_coverage["flood_overlap_pct"] = np.clip(pct, 0.0, 100.0)

    grouped = (
        joined.groupby("parcel_row_id", as_index=False)
        .agg(
            flood_risk_score=("flood_risk_score", "max"),
            flood_zone_list=(
                "flood_zone",
                lambda s: "|".join(
                    zone
                    for zone in sorted(
                        {str(zone) for zone in s if str(zone) != "UNKNOWN"},
                        key=zone_sort_key,
                        reverse=True,
                    )
                )
                or "UNKNOWN",
            ),
            has_flood_overlap=("flood_zone", lambda s: any(str(zone) != "UNKNOWN" for zone in s)),
            sfha_overlap=("sfha_flag", "max"),
        )
    )

    joined_sorted = joined.assign(zone_rank=joined["flood_zone"].map(lambda z: zone_sort_key(str(z))[0]))
    joined_sorted = joined_sorted.sort_values(
        ["parcel_row_id", "zone_rank", "sfha_flag", "flood_zone"],
        ascending=[True, False, False, True],
    )
    best_zone = joined_sorted.drop_duplicates(subset=["parcel_row_id"])[["parcel_row_id", "flood_zone"]].copy()
    best_zone = best_zone.rename(columns={"flood_zone": "flood_zone_primary"})

    out = out.merge(grouped, on="parcel_row_id", how="left").merge(best_zone, on="parcel_row_id", how="left").merge(
        flood_coverage[["parcel_row_id", "flood_overlap_acres", "flood_overlap_pct"]],
        on="parcel_row_id",
        how="left",
    )
    out["flood_zone_primary"] = out["flood_zone_primary"].fillna("UNKNOWN")
    out["flood_risk_score"] = out["flood_risk_score"].fillna(0).astype(int)
    out["flood_zone_list"] = out["flood_zone_list"].fillna("UNKNOWN")
    out["has_flood_overlap"] = out["has_flood_overlap"].fillna(False).astype(bool)
    out["sfha_overlap"] = out["sfha_overlap"].fillna(False).astype(bool)
    out["flood_overlap_acres"] = out["flood_overlap_acres"].fillna(0.0).astype(float)
    out["flood_overlap_pct"] = out["flood_overlap_pct"].fillna(0.0).astype(float)
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

    parcel_file = resolve_path(args.parcel_file)
    flood_file = resolve_path(args.flood_file)
    output_file = resolve_path(args.output_file)
    summary_csv = resolve_path(args.summary_csv)
    parts_dir = resolve_path(args.parts_dir)
    checkpoint_csv = resolve_path(args.checkpoint_csv)
    chunk_size = max(1, int(args.chunk_size))
    resume = (not args.no_resume) or args.resume

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATE: {STATE_NAME} ({STATE_ABBR})")
    print(f"Parcel file: {parcel_file}")
    print(f"Flood file: {flood_file}")
    print(f"Output file: {output_file}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Resume mode: {resume}")

    if not parcel_file.exists() or not flood_file.exists():
        print("ERROR: Parcel or flood input file missing.")
        return

    print("Loading parcels...")
    parcels = normalize_to_target_crs(gpd.read_file(parcel_file, engine="pyogrio"))
    parcels = ensure_parcel_keys(parcels)

    print("Loading FEMA flood polygons...")
    flood = normalize_to_target_crs(gpd.read_file(flood_file, engine="pyogrio"))
    if "FLD_ZONE" not in flood.columns:
        print("ERROR: FLD_ZONE field missing from FEMA layer.")
        return

    print("\nAlignment debug:")
    print_layer_debug("Parcel layer", parcels)
    print_layer_debug("Flood layer", flood)
    print(f"BBox overlap: {bbox_overlap(parcels.total_bounds, flood.total_bounds)}")

    if not bbox_overlap(parcels.total_bounds, flood.total_bounds):
        county_fips = extract_county_fips(parcels)
        if county_fips:
            fallback = load_county_fema_flood(county_fips)
            if fallback is not None:
                print("Using county FEMA fallback due to no statewide bbox overlap.")
                flood = fallback
            else:
                print("WARNING: No fallback flood data found.")

    flood_small = flood[["FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "geometry"]].copy()
    flood_small["flood_zone"] = [
        normalize_flood_zone(zone, subtype)
        for zone, subtype in zip(flood_small["FLD_ZONE"], flood_small["ZONE_SUBTY"], strict=False)
    ]
    flood_small["flood_risk_score"] = flood_small["flood_zone"].apply(risk_score_for_zone)
    flood_small["sfha_flag"] = flood_small["SFHA_TF"].fillna("").astype(str).str.upper().eq("T")
    flood_sindex = flood_small.sindex

    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_df = load_checkpoint(checkpoint_csv)
    print(
        f"Checkpoint state: {(checkpoint_df['status'] == 'completed').sum():,} completed, "
        f"{(checkpoint_df['status'] == 'failed').sum():,} failed"
    )

    county_col = "county_name" if "county_name" in parcels.columns else None
    if county_col is not None:
        chunk_values = sorted(parcels[county_col].astype(str).unique().tolist())
        if args.counties:
            wanted = {str(c).strip().lower() for c in args.counties}
            chunk_values = [c for c in chunk_values if c.lower() in wanted]
        print(f"Processing by county chunks: {len(chunk_values)}")
    else:
        chunk_values = ["all_rows"]
        print("No county_name column found. Processing one chunk.")

    expected_parts = build_expected_parts(parcels, county_col, chunk_values, chunk_size, parts_dir)
    expected_ids = {chunk_id for chunk_id, _ in expected_parts}

    for county_value in chunk_values:
        county_id = county_value if county_col else "all_rows"
        if county_col:
            county_parcels = parcels[parcels[county_col].astype(str) == county_value].copy()
        else:
            county_parcels = parcels.copy()

        total_rows = len(county_parcels)
        n_parts = max(1, math.ceil(total_rows / chunk_size))
        county_slug = sanitize_name(county_id)
        print(f"\nCounty/chunk group: {county_id} ({total_rows:,} parcels, {n_parts} part(s))")

        for part_idx in range(n_parts):
            start_i = part_idx * chunk_size
            end_i = min((part_idx + 1) * chunk_size, total_rows)
            part_id = f"{county_id}__{part_idx + 1:03d}"
            part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_flood.gpkg"

            if resume:
                done = checkpoint_df[
                    (checkpoint_df["chunk_id"] == part_id) & (checkpoint_df["status"] == "completed")
                ]
                expected_rows = end_i - start_i
                existing_rows = read_part_row_count(part_path)
                if not done.empty and existing_rows == expected_rows:
                    print(f"  Skipping completed part: {part_id}")
                    continue
                if not done.empty and not part_path.exists():
                    print(f"  Recomputing {part_id}; checkpoint exists but part file is missing.")
                elif not done.empty and existing_rows != expected_rows:
                    print(
                        f"  Recomputing {part_id}; expected {expected_rows:,} rows but found "
                        f"{0 if existing_rows is None else existing_rows:,}."
                    )

            part_parcels = county_parcels.iloc[start_i:end_i].copy()
            print(f"  Processing part {part_idx + 1}/{n_parts}: {part_id} ({len(part_parcels):,} parcels)")
            part_start = time.time()
            try:
                part_result = process_chunk(part_parcels, flood_small, flood_sindex, part_id)
                write_part_with_retry(part_result, part_path)
                elapsed = time.time() - part_start
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
        print(f"Missing {len(missing):,} expected flood parts; skipping statewide merge.")
        print(f"First missing parts: {', '.join(missing[:10])}")
        return

    merge_paths = [completed_map[chunk_id] for chunk_id, _ in expected_parts]
    print(f"\nMerging {len(merge_paths)} completed part files...")
    part_frames = [gpd.read_file(path, engine="pyogrio") for path in merge_paths]
    merged = pd.concat(part_frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    if county_col and args.counties:
        merged = merge_with_existing_counties(merged, output_file, chunk_values)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_file, driver="GPKG", engine="pyogrio")
    print(f"Saved parcel flood output: {output_file}")

    summary_rows = [
        {"metric": "rows_total", "value": int(len(merged))},
        {"metric": "rows_with_flood_overlap", "value": int(merged["has_flood_overlap"].sum())},
        {"metric": "rows_with_sfha_overlap", "value": int(merged["sfha_overlap"].sum())},
    ]
    zone_counts = merged["flood_zone_primary"].value_counts(dropna=False).sort_index()
    for zone, count in zone_counts.items():
        summary_rows.append({"metric": f"zone_{zone}", "value": int(count)})
    score_counts = merged["flood_risk_score"].value_counts(dropna=False).sort_index()
    for score, count in score_counts.items():
        summary_rows.append({"metric": f"score_{score}", "value": int(count)})
    summary = pd.DataFrame(summary_rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")

    print(f"Parcels with any flood-zone overlap: {int(merged['has_flood_overlap'].sum()):,}")
    print(f"Parcels with SFHA overlap: {int(merged['sfha_overlap'].sum()):,}")
    print("Done.")


if __name__ == "__main__":
    main()
