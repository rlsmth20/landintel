from __future__ import annotations

import argparse
import math
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException
from shapely.geometry import MultiPolygon, Polygon, box

BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"
WETLANDS_DIR = BASE_DIR / "data" / "wetlands"

# Mississippi-specific values (easy future config extraction).
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
USFWS_STATE_WETLANDS_ZIP_URL = "https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/MS_geopackage_wetlands.zip"
TARGET_CRS = "EPSG:4326"
AREA_CRS = "EPSG:5070"
ACRES_PER_SQM = 0.00024710538146716534

INPUT_PARCEL_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood_and_slope.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_flood.gpkg",
    PARCELS_DIR / "mississippi_parcels.gpkg",
]
WETLANDS_ZIP = WETLANDS_DIR / "MS_geopackage_wetlands.zip"
WETLANDS_GPKG = WETLANDS_DIR / "mississippi_nwi_wetlands.gpkg"
OUTPUT_FILE = PARCELS_DIR / "mississippi_parcels_with_flood_slope_wetlands.gpkg"
SUMMARY_CSV = PARCELS_DIR / "mississippi_wetlands_summary.csv"
PARTS_DIR = PARCELS_DIR / "ms_parcels_wetlands_parts"
CHECKPOINT_CSV = PARCELS_DIR / "mississippi_parcels_with_wetlands_progress.csv"

REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
CHUNK_SIZE_BYTES = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Mississippi USFWS wetlands and compute parcel-level wetland overlap."
    )
    parser.add_argument("--parcel-file", type=str, default="", help="Input parcels GPKG path.")
    parser.add_argument("--wetlands-zip-url", type=str, default=USFWS_STATE_WETLANDS_ZIP_URL, help="Wetlands ZIP URL.")
    parser.add_argument("--wetlands-zip", type=str, default=str(WETLANDS_ZIP), help="Local wetlands ZIP path.")
    parser.add_argument("--wetlands-gpkg", type=str, default=str(WETLANDS_GPKG), help="Extracted wetlands GPKG path.")
    parser.add_argument("--output-file", type=str, default=str(OUTPUT_FILE), help="Output parcel GPKG path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Summary CSV output path.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Per-county part file directory.")
    parser.add_argument("--checkpoint-csv", type=str, default=str(CHECKPOINT_CSV), help="Checkpoint CSV path.")
    parser.add_argument("--chunk-size", type=int, default=6000, help="Parcels per county chunk.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county_name filters.")
    parser.add_argument("--download-only", action="store_true", help="Only download/extract wetlands dataset.")
    parser.add_argument("--skip-download", action="store_true", help="Skip wetland download/extract step.")
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


def normalize_to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(crs, allow_override=True)
    return gdf.to_crs(crs)


def is_polygon_geom(geom) -> bool:
    return isinstance(geom, (Polygon, MultiPolygon))


def request_head(url: str) -> requests.Response:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"HEAD failed (attempt {attempt}/{MAX_RETRIES}): {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    raise RuntimeError(f"HEAD request failed: {url}") from last_error


def download_with_resume(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        print(f"Wetlands ZIP already exists: {dest}")
        return

    remote_size = None
    try:
        head = request_head(url)
        remote_size = int(head.headers.get("content-length", "0")) or None
    except Exception as exc:
        print(f"Warning: could not read remote size: {exc}")

    downloaded = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
    mode = "ab" if downloaded > 0 else "wb"

    print(f"Downloading wetlands ZIP from USFWS...")
    print(f"URL: {url}")
    print(f"Local file: {dest}")
    if remote_size:
        print(f"Remote size: {remote_size:,} bytes")
    if downloaded > 0:
        print(f"Resuming from byte: {downloaded:,}")

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT, headers=headers) as response:
                if downloaded > 0 and response.status_code == 200:
                    print("Server ignored Range header; restarting full download.")
                    downloaded = 0
                    headers = {}
                    mode = "wb"
                response.raise_for_status()
                with part_path.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                        if chunk:
                            f.write(chunk)
            part_path.replace(dest)
            print(f"Download complete: {dest} ({dest.stat().st_size:,} bytes)")
            return
        except (RequestException, OSError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Download failed (attempt {attempt}/{MAX_RETRIES}): {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
            downloaded = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
            mode = "ab" if downloaded > 0 else "wb"
    raise RuntimeError(f"Wetlands download failed: {url}") from last_error


def extract_wetlands_gpkg(zip_path: Path, gpkg_out: Path) -> None:
    if gpkg_out.exists():
        print(f"Wetlands GPKG already exists: {gpkg_out}")
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing wetlands ZIP: {zip_path}")

    print(f"Extracting wetlands GPKG from ZIP: {zip_path}")
    gpkg_out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        gpkg_members = [m for m in zf.namelist() if m.lower().endswith(".gpkg")]
        if not gpkg_members:
            raise RuntimeError("No .gpkg file found inside wetlands ZIP.")
        member = gpkg_members[0]
        with zf.open(member, "r") as src, gpkg_out.open("wb") as dst:
            dst.write(src.read())
    print(f"Extracted wetlands GPKG: {gpkg_out}")


def pick_wetlands_layer(gpkg_path: Path) -> str | None:
    try:
        layers = gpd.list_layers(gpkg_path)
    except Exception:
        return None
    if layers is None or layers.empty:
        return None
    name_lower = layers["name"].astype(str).str.lower()
    exact = layers[name_lower == "ms_wetlands"]
    if not exact.empty:
        return str(exact.iloc[0]["name"])
    contains = layers[name_lower.str.contains("wetlands", na=False)]
    if not contains.empty:
        return str(contains.iloc[0]["name"])
    preferred = layers[layers["geometry_type"].astype(str).str.contains("Polygon", case=False, na=False)]
    if not preferred.empty:
        return str(preferred.iloc[0]["name"])
    return str(layers.iloc[0]["name"])


def read_layer_crs(gpkg_path: Path, layer: str | None):
    if layer:
        g = gpd.read_file(gpkg_path, layer=layer, rows=1, engine="pyogrio")
    else:
        g = gpd.read_file(gpkg_path, rows=1, engine="pyogrio")
    return g.crs


def ensure_parcel_keys(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = parcels.copy()
    out["parcel_row_id"] = out.index.astype(str).map(lambda x: f"row_{x}")
    return out


def wetland_score_from_pct(pct: float | int | None) -> int:
    if pct is None or not np.isfinite(pct):
        return 0
    v = float(pct)
    if v <= 0:
        return 0
    if v <= 5:
        return 3
    if v <= 25:
        return 7
    return 10


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


def load_wetlands_subset(
    wetlands_gpkg: Path,
    wetlands_layer: str | None,
    wetlands_crs,
    bbox_4326: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    bbox_geom = gpd.GeoSeries([box(*bbox_4326)], crs=TARGET_CRS)
    if wetlands_crs is not None:
        bbox_geom = bbox_geom.to_crs(wetlands_crs)
    query_bbox = tuple(bbox_geom.total_bounds.tolist())

    if wetlands_layer:
        subset = gpd.read_file(wetlands_gpkg, layer=wetlands_layer, bbox=query_bbox, engine="pyogrio")
    else:
        subset = gpd.read_file(wetlands_gpkg, bbox=query_bbox, engine="pyogrio")
    if subset.empty:
        return subset
    subset = normalize_to_crs(subset, TARGET_CRS)
    subset = subset[subset.geometry.notnull()].copy()
    subset = subset[~subset.geometry.is_empty].copy()
    subset = subset[subset.geometry.apply(is_polygon_geom)].copy()
    return subset


def process_chunk(
    chunk_4326: gpd.GeoDataFrame,
    wetlands_gpkg: Path,
    wetlands_layer: str | None,
    wetlands_crs,
) -> gpd.GeoDataFrame:
    if chunk_4326.empty:
        return chunk_4326

    bounds = tuple(chunk_4326.total_bounds.tolist())
    query_geom = box(*bounds)
    wetlands = load_wetlands_subset(wetlands_gpkg, wetlands_layer, wetlands_crs, bounds)
    if not wetlands.empty:
        wetlands = wetlands[wetlands.intersects(query_geom)].copy()

    out = chunk_4326.copy()
    parcel_area = out.to_crs(AREA_CRS).geometry.area * ACRES_PER_SQM
    out["parcel_area_acres"] = parcel_area.astype(float)

    if wetlands.empty:
        out["wetland_overlap_acres"] = 0.0
        out["wetland_overlap_pct"] = 0.0
        out["wetland_flag"] = False
        out["wetland_score"] = 0
        return out

    chunk_area = out[["parcel_row_id", "geometry"]].to_crs(AREA_CRS)
    wetlands_area = wetlands[["geometry"]].to_crs(AREA_CRS)

    intersections = gpd.overlay(chunk_area, wetlands_area, how="intersection", keep_geom_type=False)
    if intersections.empty:
        out["wetland_overlap_acres"] = 0.0
        out["wetland_overlap_pct"] = 0.0
        out["wetland_flag"] = False
        out["wetland_score"] = 0
        return out

    intersections["int_area_acres"] = intersections.geometry.area * ACRES_PER_SQM
    wetland_agg = intersections.groupby("parcel_row_id", as_index=False).agg(
        wetland_overlap_acres=("int_area_acres", "sum")
    )

    out = out.merge(wetland_agg, on="parcel_row_id", how="left")
    out["wetland_overlap_acres"] = out["wetland_overlap_acres"].fillna(0.0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(
            out["parcel_area_acres"] > 0,
            (out["wetland_overlap_acres"] / out["parcel_area_acres"]) * 100.0,
            0.0,
        )
    out["wetland_overlap_pct"] = np.clip(pct, 0.0, 100.0)
    out["wetland_flag"] = out["wetland_overlap_pct"] > 0
    out["wetland_score"] = out["wetland_overlap_pct"].apply(wetland_score_from_pct).astype(int)
    return out


def build_expected_parts(
    parcels: gpd.GeoDataFrame,
    county_col: str | None,
    county_values: list[str],
    chunk_size: int,
    parts_dir: Path,
) -> list[tuple[str, Path]]:
    expected: list[tuple[str, Path]] = []
    for county_value in county_values:
        group_id = county_value if county_col else "all_rows"
        if county_col:
            total_rows = int((parcels[county_col].astype(str) == county_value).sum())
        else:
            total_rows = len(parcels)
        n_parts = max(1, math.ceil(total_rows / chunk_size))
        county_slug = sanitize_name(group_id)
        for part_idx in range(n_parts):
            part_id = f"{group_id}__{part_idx + 1:03d}"
            part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_wetlands.gpkg"
            expected.append((part_id, part_path))
    return expected


def main() -> None:
    args = parse_args()

    parcel_file = choose_parcel_input(args.parcel_file)
    wetlands_zip = resolve_path(args.wetlands_zip)
    wetlands_gpkg = resolve_path(args.wetlands_gpkg)
    output_file = resolve_path(args.output_file)
    summary_csv = resolve_path(args.summary_csv)
    parts_dir = resolve_path(args.parts_dir)
    checkpoint_csv = resolve_path(args.checkpoint_csv)
    chunk_size = max(1, int(args.chunk_size))
    resume = (not args.no_resume) or args.resume

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATE: {STATE_NAME} ({STATE_ABBR})")
    print(f"Parcel file: {parcel_file}")
    print(f"Wetlands ZIP: {wetlands_zip}")
    print(f"Wetlands GPKG: {wetlands_gpkg}")
    print(f"Output file: {output_file}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Resume mode: {resume}")

    WETLANDS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download_with_resume(args.wetlands_zip_url, wetlands_zip)
        extract_wetlands_gpkg(wetlands_zip, wetlands_gpkg)
    else:
        print("Skipping wetlands download/extract step.")

    if args.download_only:
        print("Download-only mode complete.")
        return

    if not parcel_file.exists():
        print(f"ERROR: missing parcel file: {parcel_file}")
        return
    if not wetlands_gpkg.exists():
        print(f"ERROR: missing wetlands GPKG: {wetlands_gpkg}")
        return

    wetlands_layer = pick_wetlands_layer(wetlands_gpkg)
    wetlands_crs = read_layer_crs(wetlands_gpkg, wetlands_layer)
    print(f"Wetlands layer selected: {wetlands_layer}")
    print(f"Wetlands CRS: {wetlands_crs}")

    parcels = normalize_to_crs(gpd.read_file(parcel_file, engine="pyogrio"), TARGET_CRS)
    parcels = parcels[parcels.geometry.notnull()].copy()
    parcels = parcels[~parcels.geometry.is_empty].copy()
    parcels = ensure_parcel_keys(parcels)
    print(f"Parcel rows: {len(parcels):,}")

    county_col = "county_name" if "county_name" in parcels.columns else None
    county_values = sorted(parcels[county_col].astype(str).unique().tolist()) if county_col else ["all_rows"]
    if county_col and args.counties:
        wanted = {str(c).strip().lower() for c in args.counties}
        county_values = [c for c in county_values if c.lower() in wanted]
        print(f"County filter active: {len(county_values)} county groups")
    print(f"Processing county groups: {len(county_values)}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_df = load_checkpoint(checkpoint_csv)
    checkpoint_completed = checkpoint_df[checkpoint_df["status"] == "completed"]
    checkpoint_failed = checkpoint_df[checkpoint_df["status"] == "failed"]
    print(
        f"Checkpoint state: {len(checkpoint_completed):,} completed, "
        f"{len(checkpoint_failed):,} failed"
    )

    expected_parts = build_expected_parts(parcels, county_col, county_values, chunk_size, parts_dir)
    expected_part_ids = {chunk_id for chunk_id, _ in expected_parts}
    print(f"Expected wetland parts this run: {len(expected_parts):,}")

    for county_value in county_values:
        group_id = county_value if county_col else "all_rows"
        if county_col:
            mask = parcels[county_col].astype(str) == county_value
            county_df = parcels[mask].copy()
        else:
            county_df = parcels.copy()

        total_rows = len(county_df)
        n_parts = max(1, math.ceil(total_rows / chunk_size))
        county_slug = sanitize_name(group_id)
        print(f"\nGroup: {group_id} ({total_rows:,} parcels, {n_parts} part(s))")

        for part_idx in range(n_parts):
            start_i = part_idx * chunk_size
            end_i = min((part_idx + 1) * chunk_size, total_rows)
            part_id = f"{group_id}__{part_idx + 1:03d}"
            part_path = parts_dir / f"{county_slug}__{part_idx + 1:03d}_with_wetlands.gpkg"

            if resume:
                done = checkpoint_df[
                    (checkpoint_df["chunk_id"] == part_id) & (checkpoint_df["status"] == "completed")
                ]
                if not done.empty and part_path.exists():
                    print(f"  Skipping completed part: {part_id}")
                    continue
                if not done.empty and not part_path.exists():
                    print(f"  Recomputing {part_id}; checkpoint exists but part file is missing.")

            chunk = county_df.iloc[start_i:end_i].copy()
            print(f"  Processing {part_id} ({len(chunk):,} parcels)")
            t0 = time.time()
            try:
                result = process_chunk(chunk, wetlands_gpkg, wetlands_layer, wetlands_crs)
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
        if row.chunk_id not in expected_part_ids:
            continue
        part_path = BASE_DIR / str(row.part_file)
        if part_path.exists():
            completed_map[str(row.chunk_id)] = part_path

    missing_parts = [chunk_id for chunk_id, _ in expected_parts if chunk_id not in completed_map]
    if missing_parts:
        print(
            f"Missing {len(missing_parts):,} of {len(expected_parts):,} expected wetland parts; "
            "skipping statewide merge."
        )
        print(f"First missing parts: {', '.join(missing_parts[:10])}")
        return

    completed_parts = [completed_map[chunk_id] for chunk_id, _ in expected_parts]
    if not completed_parts:
        print("No completed wetland parts found; nothing to merge.")
        return

    print(f"\nMerging {len(completed_parts)} completed wetland parts...")
    frames = [gpd.read_file(path, engine="pyogrio") for path in completed_parts]
    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    if county_col and args.counties:
        merged = merge_with_existing_counties(merged, output_file, county_values)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_file, driver="GPKG", engine="pyogrio")
    print(f"Saved parcel wetlands output: {output_file}")

    valid_pct = pd.to_numeric(merged["wetland_overlap_pct"], errors="coerce")
    summary_rows = [
        {"metric": "rows_total", "value": int(len(merged))},
        {"metric": "rows_with_wetland_overlap", "value": int((merged["wetland_overlap_pct"] > 0).sum())},
        {"metric": "wetland_pct_min", "value": float(valid_pct.min()) if valid_pct.notna().any() else None},
        {"metric": "wetland_pct_median", "value": float(valid_pct.median()) if valid_pct.notna().any() else None},
        {"metric": "wetland_pct_p95", "value": float(valid_pct.quantile(0.95)) if valid_pct.notna().any() else None},
        {"metric": "wetland_pct_max", "value": float(valid_pct.max()) if valid_pct.notna().any() else None},
    ]
    score_counts = merged["wetland_score"].value_counts(dropna=False).to_dict()
    for score, count in sorted(score_counts.items(), key=lambda x: x[0]):
        summary_rows.append({"metric": f"score_{score}", "value": int(count)})
    summary = pd.DataFrame(summary_rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved wetlands summary CSV: {summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
