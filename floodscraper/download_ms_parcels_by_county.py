from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from requests.exceptions import RequestException
from shapely.geometry import MultiPolygon, Polygon

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "parcel_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_CSV = RAW_DIR / "ms_parcel_download_progress.csv"

# State-specific items for easy future config extraction.
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
COUNTY_LAYER_URL = "https://opcgis.deq.state.ms.us/opcgis/rest/services/Government/MS_County/MapServer/1/query"
PARCEL_SERVICE_URLS = [
    "https://gis.mississippi.edu/server/rest/services/Cadastral/MS_West_Parcels/MapServer/0/query",
    "https://gis.mississippi.edu/server/rest/services/MS_East_Parcels/MapServer/0/query",
]
LIKELY_PARCEL_ID_FIELDS = ["PARNO", "ALTPARNO", "PPIN", "GLOBALID", "OBJECTID", "FID", "PARCELID", "PARCEL_ID", "PARCELNO", "PIN"]

# Defaults: test Adams first. Use --all-counties for full run.
DEFAULT_TEST_COUNTIES = ["Adams"]
MAX_FEATURES_PER_REQUEST = 1800
REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
SLEEP_SECONDS = 0.2
MAX_DEPTH = 12
MAX_OBJECTIDS_PER_REQUEST = 500
TARGET_CRS = "EPSG:4326"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Mississippi parcel polygons by county.")
    parser.add_argument(
        "--all-counties",
        action="store_true",
        help="Process all Mississippi counties (default is Adams-only test mode).",
    )
    parser.add_argument(
        "--counties",
        nargs="+",
        default=None,
        help="Optional explicit county names to process (overrides default test mode).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download counties even if output file already exists.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
    )


def is_polygon_geometry(geom) -> bool:
    return isinstance(geom, (Polygon, MultiPolygon))


def get_json(url: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                raise RuntimeError(f"ArcGIS error: {data['error']}")
            return data
        except (RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Request failed (attempt {attempt}/{retries}): {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    raise RuntimeError(f"Failed request after {retries} attempts: {url}") from last_error


def verify_service_endpoint(url: str) -> bool:
    params = {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    try:
        data = get_json(url, params=params)
        count = data.get("count")
        print(f"Endpoint OK: {url} (count={count})")
        return True
    except Exception as exc:
        print(f"Endpoint FAILED: {url} ({exc})")
        return False


def verify_endpoints() -> bool:
    print("\nVerifying ArcGIS endpoints...")
    ok_county = verify_service_endpoint(COUNTY_LAYER_URL)
    ok_services = [verify_service_endpoint(url) for url in PARCEL_SERVICE_URLS]
    all_ok = ok_county and any(ok_services)
    if not all_ok:
        print("Warning: one or more endpoints failed. Script will still run and skip failing services.")
    return all_ok


def get_ms_counties() -> gpd.GeoDataFrame:
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }

    data = get_json(COUNTY_LAYER_URL, params=params)
    features = data.get("features", [])
    if not features:
        raise RuntimeError("County boundary endpoint returned no features.")

    gdf = gpd.GeoDataFrame.from_features(features, crs=TARGET_CRS)
    if gdf.empty:
        raise RuntimeError("County boundary layer is empty after parsing.")

    possible_cols = [c for c in gdf.columns if "name" in c.lower() or "county" in c.lower()]
    print("Possible county columns:", possible_cols)

    county_col = None
    for col_name in gdf.columns:
        vals = gdf[col_name].astype(str).str.lower()
        if vals.str.contains("adams").any() and vals.str.contains("hinds").any():
            county_col = col_name
            break

    if county_col is None:
        raise ValueError("Could not identify county name column in county boundary layer.")

    gdf = gdf.rename(columns={county_col: "county_name"})
    gdf["county_name"] = gdf["county_name"].astype(str).str.strip()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf[["county_name", "geometry"]].copy()


def fetch_geojson_bbox(service_url: str, bbox_tuple: tuple[float, float, float, float]) -> dict:
    xmin, ymin, xmax, ymax = bbox_tuple
    params = {
        "where": "1=1",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    return get_json(service_url, params=params)


def fetch_object_id_info_for_bbox(service_url: str, bbox_tuple: tuple[float, float, float, float]) -> tuple[str | None, list[int]]:
    xmin, ymin, xmax, ymax = bbox_tuple
    params = {
        "where": "1=1",
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
        "returnGeometry": "false",
        "f": "json",
    }
    data = get_json(service_url, params=params)
    object_ids = data.get("objectIds") or []
    object_field = data.get("objectIdFieldName")
    normalized_ids: list[int] = []
    for value in object_ids:
        try:
            normalized_ids.append(int(value))
        except Exception:
            continue
    return object_field, sorted(set(normalized_ids))


def chunk_values(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[idx: idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def fetch_geojson_object_ids(service_url: str, object_field: str, object_ids: list[int]) -> dict:
    params = {
        "where": "1=1",
        "objectIds": ",".join(str(value) for value in object_ids),
        "objectIdField": object_field,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    return get_json(service_url, params=params)


def split_bbox(bbox_tuple: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    xmin, ymin, xmax, ymax = bbox_tuple
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    return [
        (xmin, ymin, xmid, ymid),
        (xmid, ymin, xmax, ymid),
        (xmin, ymid, xmid, ymax),
        (xmid, ymid, xmax, ymax),
    ]


def bbox_area(bbox_tuple: tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = bbox_tuple
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def recursive_fetch(
    service_url: str,
    bbox_tuple: tuple[float, float, float, float],
    depth: int = 0,
    max_depth: int = MAX_DEPTH,
) -> list[dict]:
    data = fetch_geojson_bbox(service_url, bbox_tuple)
    features = data.get("features", [])
    count = len(features)

    indent = "  " * depth
    print(f"{indent}tile depth={depth} bbox={bbox_tuple} -> {count} features")

    if count < MAX_FEATURES_PER_REQUEST:
        return features

    if depth >= max_depth or bbox_area(bbox_tuple) < 1e-10:
        print(f"{indent}Reached split limit; keeping capped tile with {count} features.")
        return features

    all_features: list[dict] = []
    for sub_bbox in split_bbox(bbox_tuple):
        time.sleep(SLEEP_SECONDS)
        all_features.extend(recursive_fetch(service_url, sub_bbox, depth=depth + 1, max_depth=max_depth))
    return all_features


def features_to_gdf(features: list[dict]) -> gpd.GeoDataFrame:
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    gdf = gpd.GeoDataFrame.from_features(features, crs=TARGET_CRS)
    if gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.apply(is_polygon_geometry)].copy()
    return gdf


def dedupe_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    deduped = gdf.copy()
    columns_upper = {str(c).upper(): c for c in deduped.columns}
    for candidate_col in LIKELY_PARCEL_ID_FIELDS:
        source_col = columns_upper.get(candidate_col.upper())
        if source_col is not None:
            before = len(deduped)
            deduped = deduped.drop_duplicates(subset=[source_col]).copy()
            print(f"Deduped by {source_col}: {before:,} -> {len(deduped):,}")
            return deduped

    deduped["_geom_wkb"] = deduped.geometry.apply(lambda geom: geom.wkb_hex if geom is not None else None)
    before = len(deduped)
    deduped = deduped.drop_duplicates(subset=["_geom_wkb"]).drop(columns=["_geom_wkb"]).copy()
    print(f"Deduped by geometry hash: {before:,} -> {len(deduped):,}")
    return deduped


def clip_to_county(gdf: gpd.GeoDataFrame, county_name: str, county_geom) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    county_gdf = gpd.GeoDataFrame([{"county_name": county_name, "geometry": county_geom}], crs=TARGET_CRS)
    try:
        clipped = gpd.overlay(gdf, county_gdf, how="intersection", keep_geom_type=False)
    except Exception as exc:
        print(f"Overlay failed, using intersects fallback: {exc}")
        clipped = gdf[gdf.intersects(county_geom)].copy()

    if clipped.empty:
        return clipped

    clipped = clipped[clipped.geometry.notnull()].copy()
    clipped = clipped[~clipped.geometry.is_empty].copy()
    clipped = clipped[clipped.geometry.apply(is_polygon_geometry)].copy()
    return clipped


def load_progress() -> pd.DataFrame:
    expected_cols = ["county_name", "status", "rows", "output_file", "message"]
    if not PROGRESS_CSV.exists():
        return pd.DataFrame(columns=expected_cols)

    try:
        progress_df = pd.read_csv(PROGRESS_CSV)
    except Exception as exc:
        print(f"Progress CSV unreadable, resetting: {exc}")
        return pd.DataFrame(columns=expected_cols)

    if progress_df.empty:
        return pd.DataFrame(columns=expected_cols)

    if not set(expected_cols).issubset(progress_df.columns):
        print("Progress CSV missing expected columns, resetting.")
        return pd.DataFrame(columns=expected_cols)

    return progress_df[expected_cols].copy()


def save_progress(progress_df: pd.DataFrame) -> None:
    progress_df.to_csv(PROGRESS_CSV, index=False)


def update_progress(
    progress_df: pd.DataFrame,
    county_name: str,
    status: str,
    rows: int,
    output_file: str,
    message: str,
) -> pd.DataFrame:
    entry = pd.DataFrame(
        [
            {
                "county_name": county_name,
                "status": status,
                "rows": rows,
                "output_file": output_file,
                "message": message,
            }
        ]
    )
    progress_df = progress_df[progress_df["county_name"].str.lower() != county_name.lower()].copy()
    progress_df = pd.concat([progress_df, entry], ignore_index=True)
    save_progress(progress_df)
    return progress_df


def download_county_from_service(county_name: str, county_geom, service_url: str) -> gpd.GeoDataFrame:
    bbox_tuple = county_geom.bounds
    object_field, object_ids = fetch_object_id_info_for_bbox(service_url, bbox_tuple)
    features: list[dict]
    if object_field and object_ids:
        print(f"Object ID fetch for {county_name}: {len(object_ids):,} ids from {service_url}")
        features = []
        for batch in chunk_values(object_ids, MAX_OBJECTIDS_PER_REQUEST):
            time.sleep(SLEEP_SECONDS)
            batch_data = fetch_geojson_object_ids(service_url, object_field, batch)
            features.extend(batch_data.get("features", []))
        print(f"Fetched {len(features):,} features by object IDs before county clip.")
    else:
        print(f"Object ID fetch unavailable for {county_name}; falling back to recursive bbox fetch.")
        features = recursive_fetch(service_url, bbox_tuple)
    gdf = features_to_gdf(features)
    if gdf.empty:
        return gdf
    clipped = clip_to_county(gdf, county_name, county_geom)
    clipped = dedupe_gdf(clipped)
    if "county_name" not in clipped.columns:
        clipped["county_name"] = county_name
    return clipped


def save_gpkg(gdf: gpd.GeoDataFrame, out_path: Path) -> None:
    gdf.to_file(out_path, driver="GPKG", engine="pyogrio")


def choose_counties(counties_gdf: gpd.GeoDataFrame, args: argparse.Namespace) -> gpd.GeoDataFrame:
    if args.counties:
        selected = {name.strip().lower() for name in args.counties}
        return counties_gdf[counties_gdf["county_name"].str.lower().isin(selected)].copy()
    if args.all_counties:
        return counties_gdf.copy()
    selected = {name.lower() for name in DEFAULT_TEST_COUNTIES}
    return counties_gdf[counties_gdf["county_name"].str.lower().isin(selected)].copy()


def main() -> None:
    args = parse_args()

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"RAW_DIR:  {RAW_DIR}")
    print(f"STATE:    {STATE_NAME} ({STATE_ABBR})")
    print(f"Mode:     {'all counties' if args.all_counties else 'test/specified counties'}\n")

    verify_endpoints()
    counties = get_ms_counties()
    counties = choose_counties(counties, args)

    print(f"Counties to process: {len(counties)}")
    if counties.empty:
        print("No counties matched selection.")
        return

    progress_df = load_progress()
    if not progress_df.empty:
        print(f"Loaded progress file: {PROGRESS_CSV}")

    for _, row in counties.iterrows():
        county_name = row["county_name"]
        county_geom = row["geometry"]
        slug = sanitize_name(county_name)
        out_path = RAW_DIR / f"{slug}_parcels.gpkg"

        if out_path.exists() and not args.force:
            print(f"\nSkipping existing file: {out_path.name}")
            progress_df = update_progress(
                progress_df,
                county_name=county_name,
                status="skipped_existing",
                rows=0,
                output_file=str(out_path.relative_to(BASE_DIR)),
                message="Output already exists.",
            )
            continue

        print(f"\n=== Processing {county_name} County ===")
        county_layers: list[gpd.GeoDataFrame] = []
        failed_services: list[str] = []

        for service_url in PARCEL_SERVICE_URLS:
            print(f"Querying parcel service: {service_url}")
            try:
                gdf = download_county_from_service(county_name, county_geom, service_url)
                print(f"Service features after clip/dedupe: {len(gdf):,}")
                if not gdf.empty:
                    county_layers.append(gdf)
            except Exception as exc:
                failed_services.append(service_url)
                print(f"Service failed for {county_name}: {exc}")

        if not county_layers:
            print(f"No parcel features saved for {county_name}.")
            progress_df = update_progress(
                progress_df,
                county_name=county_name,
                status="failed",
                rows=0,
                output_file=str(out_path.relative_to(BASE_DIR)),
                message=f"No data. Failed services: {len(failed_services)}",
            )
            continue

        county_parcels = pd.concat(county_layers, ignore_index=True)
        county_parcels = gpd.GeoDataFrame(county_parcels, geometry="geometry", crs=TARGET_CRS)
        county_parcels = county_parcels[county_parcels.geometry.apply(is_polygon_geometry)].copy()
        county_parcels = dedupe_gdf(county_parcels)

        print(f"Saving {len(county_parcels):,} parcels -> {out_path}")
        save_gpkg(county_parcels, out_path)
        print("Saved successfully.")

        message = "Completed."
        if failed_services:
            message = f"Completed with {len(failed_services)} failed service(s)."
        progress_df = update_progress(
            progress_df,
            county_name=county_name,
            status="completed",
            rows=len(county_parcels),
            output_file=str(out_path.relative_to(BASE_DIR)),
            message=message,
        )

    print("\nDone.")
    print(f"Progress CSV: {PROGRESS_CSV}")


if __name__ == "__main__":
    main()
