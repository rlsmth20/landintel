from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"

INPUT_PARCEL_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood_slope_wetlands.gpkg",
]
ROADS_FILE = PARCELS_DIR / "mississippi_parcels_with_roads.gpkg"
OUTPUT_FILE = PARCELS_DIR / "mississippi_parcels_scored.gpkg"
SUMMARY_CSV = PARCELS_DIR / "mississippi_parcel_scores_summary.csv"
PARTS_DIR = PARCELS_DIR / "ms_parcels_scored_parts"
CHECKPOINT_CSV = PARCELS_DIR / "mississippi_parcels_scored_progress.csv"
TARGET_CRS = "EPSG:4326"

ROAD_COLUMNS = ["county_name", "parcel_row_id", "road_distance_m", "road_distance_ft", "fclass", "name", "ref"]
SUMMARY_SCORE_COLUMNS = ["buildability_score", "environment_score", "investment_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Mississippi parcel scoring from flood, slope, wetlands, and road metrics.")
    parser.add_argument("--parcel-file", type=str, default="", help="Wetlands-enriched parcel GPKG input path.")
    parser.add_argument("--roads-file", type=str, default=str(ROADS_FILE), help="Road-distance parcel GPKG input path.")
    parser.add_argument("--output-file", type=str, default=str(OUTPUT_FILE), help="Scored parcel GPKG output path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Score summary CSV output path.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Directory for per-county scored chunk outputs.")
    parser.add_argument("--checkpoint-csv", type=str, default=str(CHECKPOINT_CSV), help="Checkpoint CSV path.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county_name filters.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint data.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and recompute chunks.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def choose_parcel_input(path_arg: str) -> Path:
    if path_arg:
        return resolve_path(path_arg)
    for candidate in INPUT_PARCEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return INPUT_PARCEL_CANDIDATES[0]


def sanitize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


def sql_quote(value: str) -> str:
    return str(value).replace("'", "''")


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


def load_county_index(parcel_file: Path, counties: list[str] | None) -> list[str]:
    index_df = gpd.read_file(parcel_file, columns=["county_name"], ignore_geometry=True, engine="pyogrio")
    county_values = sorted(index_df["county_name"].astype(str).unique().tolist())
    if counties:
        wanted = {str(c).strip().lower() for c in counties}
        county_values = [c for c in county_values if c.lower() in wanted]
    return county_values


def read_county_parcels(parcel_file: Path, county_name: str) -> gpd.GeoDataFrame:
    where = f"county_name = '{sql_quote(county_name)}'"
    gdf = gpd.read_file(parcel_file, where=where, engine="pyogrio")
    if gdf.crs is None:
        return gdf.set_crs(TARGET_CRS, allow_override=True)
    return gdf.to_crs(TARGET_CRS)


def read_county_roads(roads_file: Path, county_name: str) -> pd.DataFrame:
    where = f"county_name = '{sql_quote(county_name)}'"
    return gpd.read_file(
        roads_file,
        where=where,
        columns=ROAD_COLUMNS,
        ignore_geometry=True,
        engine="pyogrio",
    )


def compute_road_component(distance_m: pd.Series) -> np.ndarray:
    values = pd.to_numeric(distance_m, errors="coerce").fillna(-1.0).to_numpy(dtype=float)
    conditions = [
        values < 0,
        values <= 50,
        values <= 100,
        values <= 250,
        values <= 500,
        values <= 1000,
        values <= 2500,
    ]
    choices = [0.0, 100.0, 95.0, 85.0, 70.0, 50.0, 30.0]
    return np.select(conditions, choices, default=10.0)


def compute_slope_component(mean_slope_pct: pd.Series, slope_score: pd.Series) -> np.ndarray:
    mean_values = pd.to_numeric(mean_slope_pct, errors="coerce").to_numpy(dtype=float)
    slope_values = pd.to_numeric(slope_score, errors="coerce").to_numpy(dtype=float)
    fallback = np.clip(100.0 - np.nan_to_num(slope_values, nan=10.0) * 10.0, 0.0, 100.0)
    component = np.select(
        [
            np.isnan(mean_values),
            mean_values <= 5.0,
            mean_values <= 10.0,
            mean_values <= 15.0,
            mean_values <= 25.0,
        ],
        [
            fallback,
            100.0,
            85.0,
            65.0,
            35.0,
        ],
        default=10.0,
    )
    return np.clip(component, 0.0, 100.0)


def compute_wetland_component(wetland_overlap_pct: pd.Series, wetland_score: pd.Series) -> np.ndarray:
    pct_values = pd.to_numeric(wetland_overlap_pct, errors="coerce").to_numpy(dtype=float)
    score_values = pd.to_numeric(wetland_score, errors="coerce").to_numpy(dtype=float)
    fallback = np.clip(100.0 - np.nan_to_num(score_values, nan=10.0) * 10.0, 0.0, 100.0)
    component = np.select(
        [
            np.isnan(pct_values),
            pct_values <= 0.0,
            pct_values <= 5.0,
            pct_values <= 30.0,
            pct_values <= 60.0,
        ],
        [
            fallback,
            100.0,
            85.0,
            55.0,
            25.0,
        ],
        default=10.0,
    )
    return np.clip(component, 0.0, 100.0)


def compute_flood_component(flood_zone_primary: pd.Series, flood_risk_score: pd.Series) -> np.ndarray:
    zones = flood_zone_primary.fillna("").astype(str).str.upper().str.strip()
    risk_values = pd.to_numeric(flood_risk_score, errors="coerce").to_numpy(dtype=float)

    zone_penalty = np.select(
        [
            zones.str.startswith("VE").to_numpy(),
            zones.isin(["AE", "A", "AO", "AH", "A99"]).to_numpy(),
            zones.str.startswith("X").to_numpy(),
            zones.isin(["B", "C"]).to_numpy(),
            zones.isin(["D"]).to_numpy(),
        ],
        [
            95.0,
            85.0,
            5.0,
            10.0,
            40.0,
        ],
        default=50.0,
    )
    risk_penalty = np.clip(np.nan_to_num(risk_values, nan=5.0) * 10.0, 0.0, 100.0)
    penalty = np.maximum(zone_penalty, risk_penalty)
    return np.clip(100.0 - penalty, 0.0, 100.0)


def score_county(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = df.copy()

    flood_component = compute_flood_component(out["flood_zone_primary"], out["flood_risk_score"])
    slope_component = compute_slope_component(out["mean_slope_pct"], out["slope_score"])
    wetland_component = compute_wetland_component(out["wetland_overlap_pct"], out["wetland_score"])
    road_component = compute_road_component(out["road_distance_m"])

    flood_penalty = 100.0 - flood_component
    slope_penalty = 100.0 - slope_component
    wetland_penalty = 100.0 - wetland_component

    buildability = (
        (0.35 * flood_component)
        + (0.25 * slope_component)
        + (0.25 * wetland_component)
        + (0.15 * road_component)
    )
    environment = (
        (0.50 * wetland_penalty)
        + (0.35 * flood_penalty)
        + (0.15 * slope_penalty)
    )
    investment = (
        (0.55 * buildability)
        + (0.25 * road_component)
        + (0.20 * (100.0 - environment))
    )

    out["buildability_score"] = np.clip(np.round(buildability, 2), 0.0, 100.0)
    out["environment_score"] = np.clip(np.round(environment, 2), 0.0, 100.0)
    out["investment_score"] = np.clip(np.round(investment, 2), 0.0, 100.0)
    return out


def build_summary(scored: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [{"metric": "rows_total", "value": int(len(scored))}]
    for score_col in SUMMARY_SCORE_COLUMNS:
        values = pd.to_numeric(scored[score_col], errors="coerce")
        rows.extend(
            [
                {"metric": f"{score_col}_min", "value": float(values.min())},
                {"metric": f"{score_col}_median", "value": float(values.median())},
                {"metric": f"{score_col}_p95", "value": float(values.quantile(0.95))},
                {"metric": f"{score_col}_max", "value": float(values.max())},
            ]
        )
        bins = pd.cut(values, bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
        counts = bins.value_counts(sort=False)
        for bucket, count in counts.items():
            rows.append({"metric": f"{score_col}_bin_{bucket}", "value": int(count)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    parcel_file = choose_parcel_input(args.parcel_file)
    roads_file = resolve_path(args.roads_file)
    output_file = resolve_path(args.output_file)
    summary_csv = resolve_path(args.summary_csv)
    parts_dir = resolve_path(args.parts_dir)
    checkpoint_csv = resolve_path(args.checkpoint_csv)
    resume = (not args.no_resume) or args.resume

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Parcel file: {parcel_file}")
    print(f"Roads file: {roads_file}")
    print(f"Output file: {output_file}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Resume mode: {resume}")

    if not parcel_file.exists():
        print(f"ERROR: Missing parcel file: {parcel_file}")
        return
    if not roads_file.exists():
        print(f"ERROR: Missing roads file: {roads_file}")
        return

    county_values = load_county_index(parcel_file, args.counties)
    print(f"County groups to score: {len(county_values)}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_df = load_checkpoint(checkpoint_csv)
    print(
        f"Checkpoint state: {(checkpoint_df['status'] == 'completed').sum():,} completed, "
        f"{(checkpoint_df['status'] == 'failed').sum():,} failed"
    )

    expected_parts = [(county_name, parts_dir / f"{sanitize_name(county_name)}_scored.gpkg") for county_name in county_values]
    expected_ids = {county_name for county_name, _ in expected_parts}

    for county_name, part_path in expected_parts:
        if resume:
            done = checkpoint_df[
                (checkpoint_df["chunk_id"] == county_name) & (checkpoint_df["status"] == "completed")
            ]
            if not done.empty and part_path.exists():
                print(f"Skipping completed county: {county_name}")
                continue
            if not done.empty and not part_path.exists():
                print(f"Recomputing {county_name}; checkpoint exists but part file is missing.")

        print(f"\nScoring county: {county_name}")
        t0 = time.time()
        try:
            parcel_county = read_county_parcels(parcel_file, county_name)
            road_county = read_county_roads(roads_file, county_name)
            merged = parcel_county.merge(
                road_county,
                on=["county_name", "parcel_row_id"],
                how="left",
                suffixes=("", "_road"),
            )
            if "road_distance_m" not in merged.columns:
                merged["road_distance_m"] = np.nan
            if "road_distance_ft" not in merged.columns:
                merged["road_distance_ft"] = np.nan

            scored = score_county(merged)
            write_part_with_retry(scored, part_path)
            elapsed = time.time() - t0
            print(f"Saved {part_path.name} ({len(scored):,} rows, {elapsed:.1f}s)")
            checkpoint_df = update_checkpoint(
                checkpoint_df,
                chunk_id=county_name,
                status="completed",
                rows=len(scored),
                part_file=str(part_path.relative_to(BASE_DIR)),
            )
            save_checkpoint(checkpoint_csv, checkpoint_df)
        except Exception as exc:
            print(f"Failed {county_name}: {exc}")
            checkpoint_df = update_checkpoint(
                checkpoint_df,
                chunk_id=county_name,
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

    missing = [county_name for county_name, _ in expected_parts if county_name not in completed_map]
    if missing:
        print(f"Missing {len(missing):,} county score parts; skipping statewide merge.")
        print(f"First missing counties: {', '.join(missing[:10])}")
        return

    merge_paths = [completed_map[county_name] for county_name, _ in expected_parts]
    print(f"\nMerging {len(merge_paths)} scored county parts...")
    frames = [gpd.read_file(path, engine="pyogrio") for path in merge_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    if args.counties:
        merged = merge_with_existing_counties(merged, output_file, county_values)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_file, driver="GPKG", engine="pyogrio")
    print(f"Saved scored parcels: {output_file}")

    summary = build_summary(merged)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved score summary CSV: {summary_csv}")

    top_cols = [c for c in ["county_name", "parcel_row_id", "parcel_id", "investment_score", "buildability_score", "environment_score"] if c in merged.columns]
    print("\nTop parcels by investment_score:")
    print(merged[top_cols].sort_values("investment_score", ascending=False).head(10).to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
