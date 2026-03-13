from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import requests
from requests.exceptions import RequestException
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)

BASE_DIR = Path(__file__).resolve().parents[1]
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
INFRASTRUCTURE_DATA_PATH = BASE_DIR / "data"
PARCELS_DIR = INFRASTRUCTURE_DATA_PATH / "parcels"
INFRASTRUCTURE_RAW_DIR = INFRASTRUCTURE_DATA_PATH / "infrastructure_raw"
INFRASTRUCTURE_PROCESSED_DIR = INFRASTRUCTURE_DATA_PATH / "infrastructure_processed"

TARGET_CRS = "EPSG:4326"
DISTANCE_CRS = "EPSG:3857"
ONE_MILE_METERS = 1609.344

INPUT_PARCEL_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_scored.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_flood_slope_wetlands.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_flood_and_slope.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_flood.gpkg",
    PARCELS_DIR / "mississippi_parcels.gpkg",
]

TRANSMISSION_RAW = INFRASTRUCTURE_RAW_DIR / "mississippi_transmission_lines_official.geojson"
SUBSTATIONS_RAW = INFRASTRUCTURE_RAW_DIR / "mississippi_electrical_substations_official.geojson"
PIPELINES_RAW = INFRASTRUCTURE_RAW_DIR / "mississippi_natural_gas_pipelines_official.geojson"
WATER_RAW = INFRASTRUCTURE_RAW_DIR / "mississippi_water_service_areas_official.geojson"
SEWER_RAW = INFRASTRUCTURE_RAW_DIR / "mississippi_sewer_service_areas_official.geojson"
ELECTRIC_CA_RAW = INFRASTRUCTURE_RAW_DIR / "mississippi_electric_service_areas_official.geojson"
GAS_CA_RAW = INFRASTRUCTURE_RAW_DIR / "mississippi_gas_service_areas_official.geojson"

TRANSMISSION_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_transmission_lines_official.gpkg"
SUBSTATIONS_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_electrical_substations_official.gpkg"
PIPELINES_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_natural_gas_pipelines_official.gpkg"
WATER_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_water_service_areas_official.gpkg"
SEWER_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_sewer_service_areas_official.gpkg"
ELECTRIC_CA_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_electric_service_areas_official.gpkg"
GAS_CA_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_gas_service_areas_official.gpkg"
DEFAULT_BROADBAND_PROCESSED = INFRASTRUCTURE_PROCESSED_DIR / "mississippi_broadband.gpkg"

OUTPUT_FILE = PARCELS_DIR / "mississippi_parcels_with_utilities.gpkg"
SUMMARY_CSV = PARCELS_DIR / "ms_utilities_summary.csv"
UTILITY_SIGNALS_CSV = PARCELS_DIR / "mississippi_parcel_utility_signals.csv"
PARTS_DIR = PARCELS_DIR / "ms_parcels_utilities_parts"
CHECKPOINT_CSV = PARCELS_DIR / "mississippi_parcels_with_utilities_progress.csv"

MISSISSIPPI_TRANSMISSION_LAYER_URL = "https://services5.arcgis.com/TL4SApMiX57zN87P/arcgis/rest/services/MS_TransmissionLines_2021/FeatureServer/0"
MISSISSIPPI_SUBSTATIONS_LAYER_URL = "https://services5.arcgis.com/TL4SApMiX57zN87P/arcgis/rest/services/MS_Electric_Substations/FeatureServer/0"
MISSISSIPPI_PIPELINES_LAYER_URL = "https://services5.arcgis.com/TL4SApMiX57zN87P/arcgis/rest/services/MS_Natural_Gas_Pipelines/FeatureServer/0"
PUBLIC_WATER_SERVICE_BOUNDARIES_LAYER_URL = "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/Water_System_Boundaries/FeatureServer/0"
PSC_SERVICE_ROOT = "https://services2.arcgis.com/tONuKShmVp7yWQJL/arcgis/rest/services/PSC_CurrentCAs/FeatureServer"
PSC_WATER_LAYER_ID = 1
PSC_SEWER_LAYER_ID = 2
PSC_GAS_LAYER_ID = 3
PSC_ELECTRIC_LAYER_ID = 4

REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
CHUNK_SIZE_BYTES = 1024 * 1024

DISTANCE_COLUMNS = [
    "distance_to_powerline",
    "distance_to_substation",
    "distance_to_pipeline",
    "distance_to_fiber",
]
SERVICE_COLUMNS = [
    "broadband_available",
    "electric_in_service_territory",
    "gas_in_service_territory",
    "water_service_area",
    "sewer_service_area",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach Mississippi utility and infrastructure metrics to parcel geometries."
    )
    parser.add_argument("--parcel-file", type=str, default="", help="Input parcel GeoPackage path.")
    parser.add_argument("--transmission-file", type=str, default=str(TRANSMISSION_RAW), help="Transmission lines input path.")
    parser.add_argument("--substations-file", type=str, default=str(SUBSTATIONS_RAW), help="Electrical substations input path.")
    parser.add_argument("--pipelines-file", type=str, default=str(PIPELINES_RAW), help="Natural gas pipelines input path.")
    parser.add_argument("--broadband-file", type=str, default="", help="Optional local broadband or fiber layer path.")
    parser.add_argument("--electric-ca-file", type=str, default=str(ELECTRIC_CA_RAW), help="Electric certificated area input path.")
    parser.add_argument("--gas-ca-file", type=str, default=str(GAS_CA_RAW), help="Gas certificated area input path.")
    parser.add_argument("--water-file", type=str, default=str(WATER_RAW), help="Water service area input path.")
    parser.add_argument("--sewer-file", type=str, default=str(SEWER_RAW), help="Sewer service area input path.")
    parser.add_argument("--transmission-url", type=str, default=MISSISSIPPI_TRANSMISSION_LAYER_URL, help="Transmission layer URL or download URL.")
    parser.add_argument("--substations-url", type=str, default=MISSISSIPPI_SUBSTATIONS_LAYER_URL, help="Substations layer URL or download URL.")
    parser.add_argument("--pipelines-url", type=str, default=MISSISSIPPI_PIPELINES_LAYER_URL, help="Pipelines layer URL or download URL.")
    parser.add_argument("--electric-ca-layer-id", type=int, default=PSC_ELECTRIC_LAYER_ID, help="Electric certificated area layer ID.")
    parser.add_argument("--gas-ca-layer-id", type=int, default=PSC_GAS_LAYER_ID, help="Gas certificated area layer ID.")
    parser.add_argument("--water-layer-url", type=str, default=PUBLIC_WATER_SERVICE_BOUNDARIES_LAYER_URL, help="Water service area layer URL or download URL.")
    parser.add_argument("--sewer-layer-url", type=str, default="", help="Optional sewer service area layer URL or download URL.")
    parser.add_argument("--service-root", type=str, default=PSC_SERVICE_ROOT, help="ArcGIS FeatureServer root for certificated service areas.")
    parser.add_argument("--water-layer-id", type=int, default=PSC_WATER_LAYER_ID, help="Water service area layer ID.")
    parser.add_argument("--sewer-layer-id", type=int, default=PSC_SEWER_LAYER_ID, help="Sewer service area layer ID.")
    parser.add_argument("--output-file", type=str, default=str(OUTPUT_FILE), help="Parcel output GeoPackage path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Summary CSV output path.")
    parser.add_argument("--utility-signals-csv", type=str, default=str(UTILITY_SIGNALS_CSV), help="Normalized parcel utility signals CSV output path.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Per-county output directory.")
    parser.add_argument("--checkpoint-csv", type=str, default=str(CHECKPOINT_CSV), help="Checkpoint CSV path.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county_name filters.")
    parser.add_argument("--chunk-size", type=int, default=12000, help="Parcels per internal county sub-chunk.")
    parser.add_argument("--bbox-buffer-m", type=float, default=16000.0, help="Infrastructure subset buffer in meters.")
    parser.add_argument("--download-only", action="store_true", help="Only fetch raw infrastructure inputs.")
    parser.add_argument("--skip-download", action="store_true", help="Skip download steps and use local inputs.")
    parser.add_argument("--allow-partial", action="store_true", help="Merge completed counties even if some counties fail.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint parts.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint state and recompute all counties.")
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


def normalize_to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(crs, allow_override=True)
    return gdf.to_crs(crs)


def is_line_geom(geom) -> bool:
    return isinstance(geom, (LineString, MultiLineString))


def is_point_geom(geom) -> bool:
    return isinstance(geom, (Point, MultiPoint))


def is_polygon_geom(geom) -> bool:
    return isinstance(geom, (Polygon, MultiPolygon))


def empty_gdf(crs: str = TARGET_CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"source_id": pd.Series(dtype=str)}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def ensure_parcel_keys(parcels: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = parcels.copy()
    if "parcel_row_id" not in out.columns:
        out["parcel_row_id"] = out.index.astype(str).map(lambda value: f"row_{value}")
    if "parcel_id" not in out.columns:
        out["parcel_id"] = pd.NA
    return out


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
    raise RuntimeError(f"Failed writing GeoPackage: {path}") from last_exc


def request_json(url: str, params: dict[str, object] | None = None, method: str = "get") -> dict:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if method.lower() == "post":
                response = requests.post(url, data=params, timeout=REQUEST_TIMEOUT)
            else:
                response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Request failed ({attempt}/{MAX_RETRIES}) for {url}: {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    raise RuntimeError(f"Request failed: {url}") from last_error


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Raw file already exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                with dest.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                        if chunk:
                            handle.write(chunk)
            print(f"Saved raw file: {dest}")
            return
        except (RequestException, OSError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Download failed ({attempt}/{MAX_RETRIES}): {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    raise RuntimeError(f"Failed downloading {url}") from last_error


def write_feature_collection(dest: Path, payload: dict[str, object]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def read_dataset_bounds(dataset_path: Path) -> tuple[float, float, float, float]:
    _, bounds = pyogrio.read_bounds(dataset_path)
    return (
        float(bounds[0].min()),
        float(bounds[1].min()),
        float(bounds[2].max()),
        float(bounds[3].max()),
    )


def bbox_query_params(bounds: tuple[float, float, float, float]) -> dict[str, object]:
    xmin, ymin, xmax, ymax = bounds
    return {
        "geometry": f"{xmin},{ymin},{xmax},{ymax}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }


def download_arcgis_geojson_from_layer_url(
    layer_url: str,
    dest: Path,
    query_params: dict[str, object] | None = None,
) -> None:
    if dest.exists():
        print(f"Raw file already exists: {dest}")
        return

    metadata = request_json(layer_url, params={"f": "json"})
    ids_payload = request_json(
        f"{layer_url}/query",
        params={"where": "1=1", "returnIdsOnly": "true", "f": "json", **(query_params or {})},
    )

    object_ids = sorted(ids_payload.get("objectIds") or [])
    max_record_count = int(metadata.get("maxRecordCount") or 1000)
    features: list[dict[str, object]] = []

    if not object_ids:
        print(f"No features returned for ArcGIS layer {layer_url}; writing empty GeoJSON.")
        write_feature_collection(dest, {"type": "FeatureCollection", "features": []})
        return

    total_chunks = math.ceil(len(object_ids) / max_record_count)
    print(f"Downloading ArcGIS layer {layer_url} in {total_chunks} chunk(s).")
    for offset in range(0, len(object_ids), max_record_count):
        chunk_ids = object_ids[offset : offset + max_record_count]
        payload = request_json(
            f"{layer_url}/query",
            params={
                "objectIds": ",".join(str(value) for value in chunk_ids),
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "outSR": "4326",
                **(query_params or {}),
            },
            method="post",
        )
        features.extend(payload.get("features") or [])

    write_feature_collection(dest, {"type": "FeatureCollection", "features": features})
    print(f"Saved raw file: {dest}")


def download_arcgis_layer_geojson(service_root: str, layer_id: int, dest: Path) -> None:
    layer_url = f"{service_root.rstrip('/')}/{layer_id}"
    download_arcgis_geojson_from_layer_url(layer_url, dest)


def ensure_raw_inputs(args: argparse.Namespace, parcel_file: Path) -> None:
    INFRASTRUCTURE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    INFRASTRUCTURE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    transmission_file = resolve_path(args.transmission_file)
    substations_file = resolve_path(args.substations_file)
    pipelines_file = resolve_path(args.pipelines_file)
    electric_ca_file = resolve_path(args.electric_ca_file)
    gas_ca_file = resolve_path(args.gas_ca_file)
    water_file = resolve_path(args.water_file)
    sewer_file = resolve_path(args.sewer_file)
    parcel_bounds = read_dataset_bounds(parcel_file) if parcel_file.exists() else None

    if args.skip_download:
        print("Skipping infrastructure downloads.")
        return

    for url, path in [
        (args.transmission_url, transmission_file),
        (args.substations_url, substations_file),
        (args.pipelines_url, pipelines_file),
    ]:
        if "/FeatureServer/" in url:
            download_arcgis_geojson_from_layer_url(url, path)
        else:
            download_file(url, path)

    if args.service_root:
        download_arcgis_layer_geojson(args.service_root, int(args.electric_ca_layer_id), electric_ca_file)
        download_arcgis_layer_geojson(args.service_root, int(args.gas_ca_layer_id), gas_ca_file)

    if args.water_layer_url:
        if "/FeatureServer/" in args.water_layer_url:
            download_arcgis_geojson_from_layer_url(
                args.water_layer_url,
                water_file,
                query_params=bbox_query_params(parcel_bounds) if parcel_bounds else None,
            )
        else:
            download_file(args.water_layer_url, water_file)
    elif args.service_root:
        download_arcgis_layer_geojson(args.service_root, int(args.water_layer_id), water_file)

    if args.sewer_layer_url:
        if "/FeatureServer/" in args.sewer_layer_url:
            download_arcgis_geojson_from_layer_url(args.sewer_layer_url, sewer_file)
        else:
            download_file(args.sewer_layer_url, sewer_file)
    elif args.service_root:
        download_arcgis_layer_geojson(args.service_root, int(args.sewer_layer_id), sewer_file)


def read_vector_file(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path, engine="pyogrio")


def prepare_vector_layer(
    raw_path: Path,
    processed_path: Path,
    geometry_mode: str | None,
    source_prefix: str,
    required: bool,
) -> gpd.GeoDataFrame:
    if processed_path.exists():
        gdf = gpd.read_file(processed_path, engine="pyogrio")
    else:
        if not raw_path.exists():
            if required:
                raise FileNotFoundError(f"Missing required input layer: {raw_path}")
            return empty_gdf()

        gdf = read_vector_file(raw_path)
        if gdf.empty:
            return empty_gdf()
        gdf = normalize_to_crs(gdf, TARGET_CRS)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        if geometry_mode == "line":
            gdf = gdf[gdf.geometry.apply(is_line_geom)].copy()
        elif geometry_mode == "point":
            gdf = gdf[gdf.geometry.apply(is_point_geom)].copy()
        elif geometry_mode == "polygon":
            gdf = gdf[gdf.geometry.apply(is_polygon_geom)].copy()
        if gdf.empty:
            return empty_gdf()
        gdf = gdf.reset_index(drop=True)
        gdf["source_id"] = gdf.index.astype(str).map(lambda value: f"{source_prefix}_{value}")
        write_gpkg_with_retry(gdf, processed_path)

    gdf = normalize_to_crs(gdf, TARGET_CRS)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if "source_id" not in gdf.columns:
        gdf = gdf.reset_index(drop=True)
        gdf["source_id"] = gdf.index.astype(str).map(lambda value: f"{source_prefix}_{value}")
    return gdf


def bundle_layer(gdf: gpd.GeoDataFrame) -> dict[str, object]:
    if gdf.empty:
        return {
            "gdf_4326": empty_gdf(),
            "gdf_3857": empty_gdf(DISTANCE_CRS),
            "sindex_4326": None,
            "sindex_3857": None,
        }

    gdf_4326 = normalize_to_crs(gdf, TARGET_CRS)
    gdf_3857 = gdf_4326.to_crs(DISTANCE_CRS)
    return {
        "gdf_4326": gdf_4326,
        "gdf_3857": gdf_3857,
        "sindex_4326": gdf_4326.sindex,
        "sindex_3857": gdf_3857.sindex,
    }


def load_county_index(parcel_file: Path, counties: list[str] | None) -> list[str]:
    index_df = gpd.read_file(parcel_file, columns=["county_name"], ignore_geometry=True, engine="pyogrio")
    county_values = sorted(index_df["county_name"].astype(str).unique().tolist())
    if counties:
        wanted = {str(c).strip().lower() for c in counties}
        county_values = [value for value in county_values if value.lower() in wanted]
    return county_values


def read_county_parcels(parcel_file: Path, county_name: str) -> gpd.GeoDataFrame:
    where = f"county_name = '{sql_quote(county_name)}'"
    gdf = gpd.read_file(parcel_file, where=where, engine="pyogrio")
    gdf = ensure_parcel_keys(gdf)
    return normalize_to_crs(gdf, TARGET_CRS)


def subset_by_bbox(
    gdf: gpd.GeoDataFrame,
    sindex,
    bounds: tuple[float, float, float, float],
    buffer_m: float = 0.0,
    projected: bool = False,
) -> gpd.GeoDataFrame:
    if gdf.empty or sindex is None:
        return gdf
    xmin, ymin, xmax, ymax = bounds
    if projected and buffer_m:
        xmin -= buffer_m
        ymin -= buffer_m
        xmax += buffer_m
        ymax += buffer_m
    idx = sindex.query(box(xmin, ymin, xmax, ymax), predicate="intersects")
    if len(idx) == 0:
        return gdf.iloc[0:0].copy()
    return gdf.iloc[idx].copy()


def nearest_distance(
    chunk_3857: gpd.GeoDataFrame,
    layer_bundle: dict[str, object],
    buffer_m: float,
    result_column: str,
) -> pd.Series:
    if chunk_3857.empty:
        return pd.Series(dtype=float)

    layer_3857 = layer_bundle["gdf_3857"]
    if layer_3857.empty:
        return pd.Series(np.nan, index=chunk_3857["parcel_row_id"], dtype=float)

    subset = subset_by_bbox(
        layer_3857,
        layer_bundle["sindex_3857"],
        tuple(chunk_3857.total_bounds.tolist()),
        buffer_m=buffer_m,
        projected=True,
    )
    if subset.empty:
        subset = layer_3857

    left = chunk_3857[["parcel_row_id", "geometry"]].copy()
    right = subset[["source_id", "geometry"]].copy()
    joined = gpd.sjoin_nearest(left, right, how="left", distance_col=result_column)
    joined = joined.sort_values(result_column, na_position="last").drop_duplicates(subset=["parcel_row_id"]).copy()
    return chunk_3857["parcel_row_id"].map(joined.set_index("parcel_row_id")[result_column]).astype(float)


def intersects_flag(
    chunk_4326: gpd.GeoDataFrame,
    layer_bundle: dict[str, object],
    missing_value: object,
) -> pd.Series:
    if chunk_4326.empty:
        return pd.Series(dtype="boolean")

    layer_4326 = layer_bundle["gdf_4326"]
    if layer_4326.empty:
        return pd.Series([missing_value] * len(chunk_4326), index=chunk_4326.index, dtype="boolean")

    subset = subset_by_bbox(
        layer_4326,
        layer_bundle["sindex_4326"],
        tuple(chunk_4326.total_bounds.tolist()),
        buffer_m=0.0,
        projected=False,
    )
    if subset.empty:
        return pd.Series(False, index=chunk_4326.index, dtype="boolean")

    joined = gpd.sjoin(
        chunk_4326[["parcel_row_id", "geometry"]],
        subset[["source_id", "geometry"]],
        how="left",
        predicate="intersects",
    )
    matched_ids = set(joined.loc[joined["index_right"].notna(), "parcel_row_id"].astype(str).tolist())
    return pd.Series(chunk_4326["parcel_row_id"].astype(str).isin(matched_ids), index=chunk_4326.index, dtype="boolean")


def intersect_attributes(
    chunk_4326: gpd.GeoDataFrame,
    layer_bundle: dict[str, object],
    attribute_column: str,
    missing_value: object,
) -> tuple[pd.Series, pd.Series]:
    if chunk_4326.empty:
        return pd.Series(dtype="boolean"), pd.Series(dtype="string")

    layer_4326 = layer_bundle["gdf_4326"]
    if layer_4326.empty or attribute_column not in layer_4326.columns:
        return (
            pd.Series([missing_value] * len(chunk_4326), index=chunk_4326.index, dtype="boolean"),
            pd.Series([pd.NA] * len(chunk_4326), index=chunk_4326.index, dtype="string"),
        )

    subset = subset_by_bbox(
        layer_4326,
        layer_bundle["sindex_4326"],
        tuple(chunk_4326.total_bounds.tolist()),
        buffer_m=0.0,
        projected=False,
    )
    if subset.empty:
        return (
            pd.Series(False, index=chunk_4326.index, dtype="boolean"),
            pd.Series([pd.NA] * len(chunk_4326), index=chunk_4326.index, dtype="string"),
        )

    joined = gpd.sjoin(
        chunk_4326[["parcel_row_id", "geometry"]],
        subset[[attribute_column, "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined[attribute_column] = joined[attribute_column].astype("string")
    hits = joined.loc[joined["index_right"].notna(), ["parcel_row_id", attribute_column]].copy()
    hits = hits.dropna(subset=[attribute_column]).drop_duplicates(subset=["parcel_row_id"])
    provider_map = hits.set_index("parcel_row_id")[attribute_column] if not hits.empty else pd.Series(dtype="string")
    flag = pd.Series(chunk_4326["parcel_row_id"].astype(str).isin(set(hits["parcel_row_id"].astype(str))), index=chunk_4326.index, dtype="boolean")
    provider = chunk_4326["parcel_row_id"].map(provider_map).astype("string")
    return flag, provider


def process_county(
    county_4326: gpd.GeoDataFrame,
    layers: dict[str, dict[str, object]],
    chunk_size: int,
    bbox_buffer_m: float,
) -> gpd.GeoDataFrame:
    county_4326 = county_4326.reset_index(drop=True)
    county_3857 = county_4326.to_crs(DISTANCE_CRS).reset_index(drop=True)

    chunk_frames: list[gpd.GeoDataFrame] = []
    total_rows = len(county_4326)
    num_parts = max(1, math.ceil(total_rows / chunk_size))

    for part_idx in range(num_parts):
        start_i = part_idx * chunk_size
        end_i = min((part_idx + 1) * chunk_size, total_rows)
        chunk_4326 = county_4326.iloc[start_i:end_i].copy()
        chunk_3857 = county_3857.iloc[start_i:end_i].copy()

        chunk_4326["distance_to_powerline"] = nearest_distance(
            chunk_3857,
            layers["powerline"],
            bbox_buffer_m,
            "distance_to_powerline",
        )
        chunk_4326["distance_to_substation"] = nearest_distance(
            chunk_3857,
            layers["substation"],
            bbox_buffer_m,
            "distance_to_substation",
        )
        chunk_4326["distance_to_pipeline"] = nearest_distance(
            chunk_3857,
            layers["pipeline"],
            bbox_buffer_m,
            "distance_to_pipeline",
        )

        if layers["broadband"]["gdf_4326"].empty:
            chunk_4326["distance_to_fiber"] = np.nan
            chunk_4326["broadband_available"] = pd.Series([pd.NA] * len(chunk_4326), index=chunk_4326.index, dtype="boolean")
        else:
            chunk_4326["distance_to_fiber"] = nearest_distance(
                chunk_3857,
                layers["broadband"],
                bbox_buffer_m,
                "distance_to_fiber",
            )
            chunk_4326["broadband_available"] = intersects_flag(chunk_4326, layers["broadband"], pd.NA)

        electric_flag, electric_provider = intersect_attributes(chunk_4326, layers["electric_service"], "UTILITY_NA", pd.NA)
        gas_flag, gas_provider = intersect_attributes(chunk_4326, layers["gas_service"], "UTILITY_NA", pd.NA)
        water_flag, water_provider = intersect_attributes(chunk_4326, layers["water"], "PWS_Name", pd.NA)
        sewer_flag, sewer_provider = intersect_attributes(chunk_4326, layers["sewer"], "UTILITY_NA", pd.NA)

        chunk_4326["electric_in_service_territory"] = electric_flag
        chunk_4326["electric_provider_name"] = electric_provider
        chunk_4326["gas_in_service_territory"] = gas_flag
        chunk_4326["gas_provider_name"] = gas_provider
        chunk_4326["water_service_area"] = water_flag
        chunk_4326["water_provider_name"] = water_provider
        chunk_4326["sewer_service_area"] = sewer_flag
        chunk_4326["sewer_provider_name"] = sewer_provider
        chunk_frames.append(chunk_4326)

    merged = pd.concat(chunk_frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    for column in DISTANCE_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    for column in SERVICE_COLUMNS:
        if column in merged.columns:
            merged[column] = merged[column].astype("boolean")
    return merged


def pct_from_boolean(series: pd.Series) -> float | None:
    values = series.astype("boolean")
    valid = values.notna()
    if valid.sum() == 0:
        return None
    return float((values[valid].fillna(False).sum() / valid.sum()) * 100.0)


def pct_within_distance(series: pd.Series, threshold_m: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    if valid.sum() == 0:
        return None
    return float(((values[valid] <= threshold_m).sum() / valid.sum()) * 100.0)


def build_summary(merged: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"metric": "parcel_count", "value": int(len(merged))},
        {"metric": "parcels_in_electric_territory", "value": int(merged["electric_in_service_territory"].fillna(False).sum())},
        {"metric": "parcels_in_gas_territory", "value": int(merged["gas_in_service_territory"].fillna(False).sum())},
        {"metric": "parcels_in_water_service_area", "value": int(merged["water_service_area"].fillna(False).sum())},
        {"metric": "parcels_in_sewer_service_area", "value": int(merged["sewer_service_area"].fillna(False).sum())},
        {
            "metric": "percent_with_power_within_1_mile",
            "value": pct_within_distance(merged["distance_to_powerline"], ONE_MILE_METERS),
        },
        {
            "metric": "percent_with_pipeline_within_1_mile",
            "value": pct_within_distance(merged["distance_to_pipeline"], ONE_MILE_METERS),
        },
        {"metric": "percent_with_broadband", "value": pct_from_boolean(merged["broadband_available"])},
        {"metric": "percent_in_electric_territory", "value": pct_from_boolean(merged["electric_in_service_territory"])},
        {"metric": "percent_in_gas_territory", "value": pct_from_boolean(merged["gas_in_service_territory"])},
        {"metric": "percent_with_water_service", "value": pct_from_boolean(merged["water_service_area"])},
        {"metric": "percent_with_sewer_service", "value": pct_from_boolean(merged["sewer_service_area"])},
    ]

    for column in DISTANCE_COLUMNS:
        values = pd.to_numeric(merged[column], errors="coerce")
        if not values.notna().any():
            rows.extend(
                [
                    {"metric": f"{column}_min", "value": None},
                    {"metric": f"{column}_median", "value": None},
                    {"metric": f"{column}_p95", "value": None},
                    {"metric": f"{column}_max", "value": None},
                ]
            )
            continue
        rows.extend(
            [
                {"metric": f"{column}_min", "value": float(values.min())},
                {"metric": f"{column}_median", "value": float(values.median())},
                {"metric": f"{column}_p95", "value": float(values.quantile(0.95))},
                {"metric": f"{column}_max", "value": float(values.max())},
            ]
        )
    return pd.DataFrame(rows)


def meters_to_miles(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values / ONE_MILE_METERS


def blank_text(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="string")


def build_status_series(
    index: pd.Index,
    exact_available: pd.Series | None = None,
    likely_available: pd.Series | None = None,
    nearby: pd.Series | None = None,
) -> pd.Series:
    status = pd.Series("unknown", index=index, dtype="string")
    if nearby is not None:
        status.loc[nearby.fillna(False)] = "nearby"
    if likely_available is not None:
        status.loc[likely_available.fillna(False)] = "likely_available"
    if exact_available is not None:
        status.loc[exact_available.fillna(False)] = "available_exact"
    return status


def build_exact_only_status(index: pd.Index, exact_available: pd.Series) -> pd.Series:
    status = pd.Series("unknown", index=index, dtype="string")
    status.loc[exact_available.fillna(False)] = "available_exact"
    return status


def attach_utility_fields(merged: gpd.GeoDataFrame, run_timestamp: pd.Timestamp) -> gpd.GeoDataFrame:
    out = merged.copy()
    index = out.index
    run_ts_value = run_timestamp.isoformat()

    out["state"] = STATE_ABBR
    out["distance_to_powerline_miles"] = meters_to_miles(out["distance_to_powerline"])
    out["distance_to_substation_miles"] = meters_to_miles(out["distance_to_substation"])
    out["distance_to_pipeline_miles"] = meters_to_miles(out["distance_to_pipeline"])
    out["distance_to_fiber_miles"] = meters_to_miles(out["distance_to_fiber"])

    power_exact = out["electric_in_service_territory"].astype("boolean") if "electric_in_service_territory" in out.columns else pd.Series(pd.NA, index=index, dtype="boolean")
    gas_exact = out["gas_in_service_territory"].astype("boolean") if "gas_in_service_territory" in out.columns else pd.Series(pd.NA, index=index, dtype="boolean")
    water_exact = out["water_service_area"].astype("boolean")
    sewer_exact = out["sewer_service_area"].astype("boolean")
    broadband_exact = out["broadband_available"].astype("boolean")

    out["power_status"] = build_exact_only_status(index, power_exact)
    out["power_provider_name"] = out["electric_provider_name"].astype("string") if "electric_provider_name" in out.columns else blank_text(index)
    out["power_source_name"] = pd.Series(np.where(power_exact.notna(), "PSC_CurrentCAs_Electric", pd.NA), index=index, dtype="string")
    out["power_source_type"] = pd.Series(np.where(power_exact.notna(), "official_service_boundary", pd.NA), index=index, dtype="string")
    out["power_source_confidence"] = pd.Series(np.where(power_exact.notna(), "high", pd.NA), index=index, dtype="string")
    out["power_is_inferred"] = pd.Series(np.where(power_exact.notna(), False, pd.NA), index=index, dtype="boolean")
    out["power_last_updated_at"] = pd.Series(np.where(power_exact.notna(), run_ts_value, pd.NA), index=index, dtype="string")

    out["gas_status"] = build_exact_only_status(index, gas_exact)
    out["gas_source_name"] = pd.Series(np.where(gas_exact.notna(), "PSC_CurrentCAs_Gas", pd.NA), index=index, dtype="string")
    out["gas_source_type"] = pd.Series(np.where(gas_exact.notna(), "official_service_boundary", pd.NA), index=index, dtype="string")
    out["gas_source_confidence"] = pd.Series(np.where(gas_exact.notna(), "high", pd.NA), index=index, dtype="string")
    out["gas_is_inferred"] = pd.Series(np.where(gas_exact.notna(), False, pd.NA), index=index, dtype="boolean")
    out["gas_last_updated_at"] = pd.Series(np.where(gas_exact.notna(), run_ts_value, pd.NA), index=index, dtype="string")

    out["water_status"] = build_exact_only_status(index, water_exact)
    out["water_source_name"] = pd.Series(np.where(water_exact.notna(), "Water_System_Boundaries", pd.NA), index=index, dtype="string")
    out["water_source_type"] = pd.Series(np.where(water_exact.notna(), "official_service_boundary", pd.NA), index=index, dtype="string")
    out["water_source_confidence"] = pd.Series(np.where(water_exact.notna(), "high", pd.NA), index=index, dtype="string")
    out["water_is_inferred"] = pd.Series(np.where(water_exact.notna(), False, pd.NA), index=index, dtype="boolean")
    out["water_last_updated_at"] = pd.Series(np.where(water_exact.notna(), run_ts_value, pd.NA), index=index, dtype="string")

    out["sewer_status"] = build_exact_only_status(index, sewer_exact)
    out["sewer_source_name"] = pd.Series(np.where(sewer_exact.notna(), "PSC_CurrentCAs_Sewer", pd.NA), index=index, dtype="string")
    out["sewer_source_type"] = pd.Series(np.where(sewer_exact.notna(), "official_service_boundary", pd.NA), index=index, dtype="string")
    out["sewer_source_confidence"] = pd.Series(np.where(sewer_exact.notna(), "high", pd.NA), index=index, dtype="string")
    out["sewer_is_inferred"] = pd.Series(np.where(sewer_exact.notna(), False, pd.NA), index=index, dtype="boolean")
    out["sewer_last_updated_at"] = pd.Series(np.where(sewer_exact.notna(), run_ts_value, pd.NA), index=index, dtype="string")

    out["internet_status"] = build_exact_only_status(index, broadband_exact)
    out["internet_provider_name"] = blank_text(index)
    out["internet_source_name"] = pd.Series(np.where(broadband_exact.notna(), "broadband_or_fiber_layer", pd.NA), index=index, dtype="string")
    out["internet_source_type"] = pd.Series(np.where(broadband_exact.notna(), "broadband_service_or_fiber_network", pd.NA), index=index, dtype="string")
    out["internet_source_confidence"] = pd.Series(np.where(out["internet_status"] == "available_exact", "high", pd.NA), index=index, dtype="string")
    out["internet_is_inferred"] = pd.Series(np.where(broadband_exact.notna(), False, pd.NA), index=index, dtype="boolean")
    out["internet_last_updated_at"] = pd.Series(
        np.where(broadband_exact.notna(), run_ts_value, pd.NA),
        index=index,
        dtype="string",
    )
    return out


def build_normalized_utility_signals(merged: gpd.GeoDataFrame) -> pd.DataFrame:
    utility_specs = [
        ("power", "power_status", "power_provider_name", "power_source_name", "power_source_type", "power_source_confidence", "power_is_inferred", "distance_to_powerline", "distance_to_powerline_miles", "power_last_updated_at"),
        ("gas", "gas_status", "gas_provider_name", "gas_source_name", "gas_source_type", "gas_source_confidence", "gas_is_inferred", "distance_to_pipeline", "distance_to_pipeline_miles", "gas_last_updated_at"),
        ("water", "water_status", "water_provider_name", "water_source_name", "water_source_type", "water_source_confidence", "water_is_inferred", None, None, "water_last_updated_at"),
        ("sewer", "sewer_status", "sewer_provider_name", "sewer_source_name", "sewer_source_type", "sewer_source_confidence", "sewer_is_inferred", None, None, "sewer_last_updated_at"),
        ("internet", "internet_status", "internet_provider_name", "internet_source_name", "internet_source_type", "internet_source_confidence", "internet_is_inferred", "distance_to_fiber", "distance_to_fiber_miles", "internet_last_updated_at"),
    ]

    frames: list[pd.DataFrame] = []
    base_cols = [column for column in ["parcel_id", "parcel_row_id", "state", "county_name"] if column in merged.columns]
    for utility_type, status_col, provider_col, source_name_col, source_type_col, confidence_col, inferred_col, distance_m_col, distance_miles_col, updated_col in utility_specs:
        frame = merged[base_cols].copy()
        frame["utility_type"] = utility_type
        frame["status"] = merged[status_col]
        frame["provider_name"] = merged[provider_col]
        frame["source_name"] = merged[source_name_col]
        frame["source_type"] = merged[source_type_col]
        frame["source_confidence"] = merged[confidence_col]
        frame["is_inferred"] = merged[inferred_col]
        frame["distance_meters"] = merged[distance_m_col] if distance_m_col else np.nan
        frame["distance_miles"] = merged[distance_miles_col] if distance_miles_col else np.nan
        frame["geometry_reference"] = pd.NA
        frame["last_updated_at"] = merged[updated_col]
        frame = frame.rename(columns={"county_name": "county", "parcel_row_id": "parcel_key"})
        frames.append(frame)
    normalized = pd.concat(frames, ignore_index=True)
    for column in [
        "parcel_id",
        "parcel_key",
        "state",
        "county",
        "utility_type",
        "status",
        "provider_name",
        "source_name",
        "source_type",
        "source_confidence",
        "geometry_reference",
        "last_updated_at",
    ]:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("string")
    if "is_inferred" in normalized.columns:
        normalized["is_inferred"] = normalized["is_inferred"].astype("boolean")
    return normalized


def main() -> None:
    args = parse_args()

    parcel_file = choose_parcel_input(args.parcel_file)
    transmission_file = resolve_path(args.transmission_file)
    substations_file = resolve_path(args.substations_file)
    pipelines_file = resolve_path(args.pipelines_file)
    electric_ca_file = resolve_path(args.electric_ca_file)
    gas_ca_file = resolve_path(args.gas_ca_file)
    broadband_file = resolve_path(args.broadband_file) if args.broadband_file else Path()
    water_file = resolve_path(args.water_file)
    sewer_file = resolve_path(args.sewer_file)
    output_file = resolve_path(args.output_file)
    summary_csv = resolve_path(args.summary_csv)
    utility_signals_csv = resolve_path(args.utility_signals_csv)
    parts_dir = resolve_path(args.parts_dir)
    checkpoint_csv = resolve_path(args.checkpoint_csv)
    chunk_size = max(1, int(args.chunk_size))
    resume = (not args.no_resume) or args.resume
    run_timestamp = pd.Timestamp.now(tz="UTC")

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATE: {STATE_NAME} ({STATE_ABBR})")
    print(f"Parcel file: {parcel_file}")
    print(f"Output file: {output_file}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Utility signals CSV: {utility_signals_csv}")
    print(f"Chunk size: {chunk_size:,}")
    print(f"Resume mode: {resume}")

    if not args.skip_download:
        ensure_raw_inputs(args, parcel_file)
    else:
        print("Download step disabled by flag.")

    if args.download_only:
        print("Download-only mode complete.")
        return

    if not parcel_file.exists():
        print(f"ERROR: Missing parcel file: {parcel_file}")
        return

    parts_dir.mkdir(parents=True, exist_ok=True)
    INFRASTRUCTURE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing infrastructure layers...")
    powerline_gdf = prepare_vector_layer(
        transmission_file,
        TRANSMISSION_PROCESSED,
        "line",
        "powerline",
        required=True,
    )
    substation_gdf = prepare_vector_layer(
        substations_file,
        SUBSTATIONS_PROCESSED,
        "point",
        "substation",
        required=True,
    )
    pipeline_gdf = prepare_vector_layer(
        pipelines_file,
        PIPELINES_PROCESSED,
        "line",
        "pipeline",
        required=True,
    )
    electric_ca_gdf = prepare_vector_layer(
        electric_ca_file,
        ELECTRIC_CA_PROCESSED,
        "polygon",
        "electric_service",
        required=False,
    )
    gas_ca_gdf = prepare_vector_layer(
        gas_ca_file,
        GAS_CA_PROCESSED,
        "polygon",
        "gas_service",
        required=False,
    )
    water_gdf = prepare_vector_layer(
        water_file,
        WATER_PROCESSED,
        "polygon",
        "water",
        required=False,
    )
    sewer_gdf = prepare_vector_layer(
        sewer_file,
        SEWER_PROCESSED,
        "polygon",
        "sewer",
        required=False,
    )

    broadband_raw_exists = bool(args.broadband_file) and broadband_file.exists()
    broadband_processed_exists = DEFAULT_BROADBAND_PROCESSED.exists()
    if broadband_raw_exists:
        broadband_gdf = prepare_vector_layer(
            broadband_file,
            DEFAULT_BROADBAND_PROCESSED,
            None,
            "broadband",
            required=False,
        )
    elif broadband_processed_exists:
        broadband_gdf = prepare_vector_layer(
            DEFAULT_BROADBAND_PROCESSED,
            DEFAULT_BROADBAND_PROCESSED,
            None,
            "broadband",
            required=False,
        )
    else:
        print("No local broadband or fiber layer found. Broadband metrics will remain null.")
        broadband_gdf = empty_gdf()

    print(
        f"Infrastructure rows: powerlines={len(powerline_gdf):,}, "
        f"substations={len(substation_gdf):,}, pipelines={len(pipeline_gdf):,}, "
        f"electric_ca={len(electric_ca_gdf):,}, gas_ca={len(gas_ca_gdf):,}, "
        f"water={len(water_gdf):,}, sewer={len(sewer_gdf):,}, broadband={len(broadband_gdf):,}"
    )

    layers = {
        "powerline": bundle_layer(powerline_gdf),
        "substation": bundle_layer(substation_gdf),
        "pipeline": bundle_layer(pipeline_gdf),
        "electric_service": bundle_layer(electric_ca_gdf),
        "gas_service": bundle_layer(gas_ca_gdf),
        "broadband": bundle_layer(broadband_gdf),
        "water": bundle_layer(water_gdf),
        "sewer": bundle_layer(sewer_gdf),
    }

    county_values = load_county_index(parcel_file, args.counties)
    print(f"County groups to process: {len(county_values)}")

    checkpoint_df = load_checkpoint(checkpoint_csv)
    print(
        f"Checkpoint state: {(checkpoint_df['status'] == 'completed').sum():,} completed, "
        f"{(checkpoint_df['status'] == 'failed').sum():,} failed"
    )

    expected_parts = [(county_name, parts_dir / f"{sanitize_name(county_name)}_with_utilities.gpkg") for county_name in county_values]
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
                print(f"Recomputing {county_name}; checkpoint exists but county output is missing.")

        print(f"\nProcessing county: {county_name}")
        t0 = time.time()
        try:
            county_parcels = read_county_parcels(parcel_file, county_name)
            county_result = process_county(
                county_parcels,
                layers=layers,
                chunk_size=chunk_size,
                bbox_buffer_m=float(args.bbox_buffer_m),
            )
            write_gpkg_with_retry(county_result, part_path)
            elapsed = time.time() - t0
            print(f"Saved {part_path.name} ({len(county_result):,} rows, {elapsed:.1f}s)")
            checkpoint_df = update_checkpoint(
                checkpoint_df,
                chunk_id=county_name,
                status="completed",
                rows=len(county_result),
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
    if missing and not args.allow_partial:
        print(f"Missing {len(missing):,} county parts; skipping statewide merge.")
        print(f"First missing counties: {', '.join(missing[:10])}")
        return

    merge_paths = [completed_map[county_name] for county_name, _ in expected_parts if county_name in completed_map]
    if not merge_paths:
        print("No completed county outputs found; nothing to merge.")
        return

    print(f"\nMerging {len(merge_paths)} county outputs...")
    frames = [gpd.read_file(path, engine="pyogrio") for path in merge_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    merged = attach_utility_fields(merged, run_timestamp=run_timestamp)
    utility_signals = build_normalized_utility_signals(merged)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_file, driver="GPKG", engine="pyogrio")
    print(f"Saved utilities parcel output: {output_file}")

    summary = build_summary(merged)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved utilities summary CSV: {summary_csv}")

    utility_signals_csv.parent.mkdir(parents=True, exist_ok=True)
    utility_signals.to_csv(utility_signals_csv, index=False)
    print(f"Saved normalized utility signals CSV: {utility_signals_csv}")

    sample_columns = [
        column
        for column in [
            "county_name",
            "parcel_row_id",
            "electric_provider_name",
            "electric_in_service_territory",
            "gas_provider_name",
            "gas_in_service_territory",
            "water_provider_name",
            "water_service_area",
            "sewer_provider_name",
            "sewer_service_area",
            "distance_to_powerline",
            "distance_to_substation",
            "distance_to_pipeline",
            "distance_to_fiber",
        ]
        if column in merged.columns
    ]
    print("\nSample rows:")
    print(merged[sample_columns].head(10).to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
