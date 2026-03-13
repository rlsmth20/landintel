from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import requests
from requests.exceptions import RequestException
from shapely import make_valid
from shapely import wkt as shapely_wkt

BASE_DIR = Path(__file__).resolve().parents[1]
STATE_CONFIG_DIR = BASE_DIR / "floodscraper" / "state_configs"
PARCELS_RAW_DIR = BASE_DIR / "data" / "parcels_raw"
PARCELS_DIR = BASE_DIR / "data" / "parcels"
PARCELS_STANDARDIZED_DIR = BASE_DIR / "data" / "parcels_standardized"
PARCELS_METADATA_DIR = BASE_DIR / "data" / "parcels_metadata"

REQUEST_TIMEOUT = 90
MAX_RETRIES = 4
BACKOFF_SECONDS = 2.0
OBJECTID_BATCH_SIZE = 500
SLEEP_SECONDS = 0.2
TARGET_CRS = "EPSG:4326"

CANONICAL_COLUMNS = [
    "state_code",
    "county_name",
    "source_name",
    "source_dataset_path",
    "source_layer_name",
    "source_row_identifier",
    "source_parcel_number",
    "source_alt_parcel_number",
    "source_ppin",
    "owner_name",
    "mail_address_1",
    "mail_address_2",
    "mail_city_1",
    "mail_state_1",
    "mail_zip_1",
    "site_address",
    "site_city",
    "site_state",
    "site_zip",
    "tax_acres",
    "gis_acres",
    "total_acres",
    "zoning",
    "tax_status",
    "tax_year",
    "land_value",
    "total_value",
    "geometry",
]
POSITIVE_KEYWORDS = ["PARCEL", "PARCELS", "TAX PARCEL", "CADASTRAL", "PROPERTY", "ASSESSOR"]
NEGATIVE_KEYWORDS = ["ADDRESS", "ZONING", "ROAD", "WETLAND", "FLOOD", "SUBDIVISION", "SALES", "POINT"]
FIELD_HINTS = ["PARCEL", "APN", "PARNO", "OWNER", "MAIL", "SITUS", "ACRES", "PPIN"]
REGISTRY_OUTPUT_COLUMNS = [
    "county_name",
    "state_code",
    "source_name",
    "source_url",
    "source_type",
    "organization",
    "layer_name",
    "item_title",
    "item_description",
    "access_method",
    "last_modified",
    "license_usage_notes",
    "parcel_confidence_score",
    "validation_notes",
    "access_status",
    "discovered_at",
    "download_status",
    "geometry_type",
    "row_count",
    "lineage_record_type",
    "lineage_layer_url",
    "lineage_service_url",
    "lineage_object_id_field",
    "lineage_object_id_count",
    "notes",
]
COVERAGE_OUTPUT_COLUMNS = [
    "county_name",
    "source_name",
    "source_url",
    "source_type",
    "download_status",
    "row_count",
    "geometry_type",
    "parcel_confidence_score",
    "lineage_layer_url",
    "lineage_service_url",
    "lineage_object_id_field",
    "lineage_object_id_count",
    "notes",
]
FAILURE_OUTPUT_COLUMNS = ["county_name", "source_url", "failure_stage", "error"]
ARCGIS_SOURCE_TYPES = {"feature_service", "map_service"}
DIRECT_DOWNLOAD_SOURCE_TYPES = {"shapefile", "zip", "geojson", "gpkg", "csv"}
SOURCE_TYPE_EXTENSIONS = {
    "zip": ".zip",
    "shapefile": ".zip",
    "geojson": ".geojson",
    "gpkg": ".gpkg",
    "csv": ".csv",
}
ARCGIS_ONLINE_SEARCH_URL = "https://www.arcgis.com/sharing/rest/search"
STANDARDIZED_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "record_hash",
    "state_code",
    "county_fips",
    "county_name",
    "source_name",
    "source_dataset_path",
    "source_record_id",
    "geometry",
    "acreage",
    "owner_name",
    "assessed_value",
    "land_use",
]
GEOMETRY_LOG_COLUMNS = [
    "county_name",
    "parcel_row_id",
    "source_record_id",
    "geometry_issue",
    "repair_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and download public parcel datasets into a normalized raw parcel layer.")
    parser.add_argument("--state-code", required=True, help="Target two-letter state code.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county whitelist.")
    parser.add_argument("--max-counties", type=int, default=None, help="Optional maximum counties to process.")
    parser.add_argument("--refresh-existing", action="store_true", help="Refresh existing successful county downloads.")
    parser.add_argument("--config", type=str, default=None, help="Optional explicit state config path.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    request_params = params or {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=request_params, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "LANDRISK Parcel Discovery/1.0"})
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(f"ArcGIS error: {payload['error']}")
            return payload
        except (RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Failed request for {url}") from last_error


def request_bytes(url: str, params: dict[str, Any] | None = None) -> bytes:
    last_error: Exception | None = None
    request_params = params or {}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=request_params, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "LANDRISK Parcel Discovery/1.0"})
            response.raise_for_status()
            return response.content
        except RequestException as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            time.sleep(BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Failed request for {url}") from last_error


def load_state_config(state_code: str, explicit_path: str | None) -> dict[str, Any]:
    if explicit_path:
        return read_json(resolve_path(explicit_path))
    config_path = STATE_CONFIG_DIR / f"parcel_source_{state_code.lower()}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"State config not found: {config_path}")
    return read_json(config_path)


def get_county_boundaries(config: dict[str, Any]) -> gpd.GeoDataFrame:
    service_url = str(config["county_boundary_service_url"]).rstrip("/")
    payload = request_json(
        f"{service_url}/query",
        params={
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        },
    )
    features = payload.get("features", [])
    if not features:
        raise RuntimeError("County boundary service returned no features.")
    counties = gpd.GeoDataFrame.from_features(features, crs=TARGET_CRS)
    counties.columns = [str(column) for column in counties.columns]
    county_name_field = choose_field(counties.columns, config.get("county_name_field_candidates", []))
    county_fips_field = choose_field(counties.columns, config.get("county_fips_field_candidates", []), optional=True)
    counties = counties.rename(columns={county_name_field: "county_name"})
    counties["county_name"] = counties["county_name"].astype("string").map(sanitize_name)
    if county_fips_field:
        counties = counties.rename(columns={county_fips_field: "county_fips"})
    else:
        counties["county_fips"] = pd.Series(pd.NA, index=counties.index, dtype="string")
    counties = counties.loc[counties.geometry.notna() & ~counties.geometry.is_empty, ["county_name", "county_fips", "geometry"]].copy()
    counties = counties.to_crs(TARGET_CRS)
    return counties.sort_values("county_name").reset_index(drop=True)


def choose_field(columns: list[str] | pd.Index, candidates: list[str], optional: bool = False) -> str | None:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    if optional:
        return None
    raise KeyError(f"Could not locate any of the expected fields: {candidates}")


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def derive_query_url(source_url: str) -> str:
    base = source_url.rstrip("/")
    return base if base.lower().endswith("/query") else f"{base}/query"


def fetch_arcgis_metadata(source_url: str) -> dict[str, Any]:
    return request_json(source_url.rstrip("/"), params={"f": "json"})


def derive_service_url(source_url: str) -> str:
    base = source_url.rstrip("/")
    tail = base.rsplit("/", 1)[-1]
    return base.rsplit("/", 1)[0] if tail.isdigit() else base


def fetch_arcgis_service_metadata(source_url: str) -> dict[str, Any]:
    return request_json(derive_service_url(source_url), params={"f": "json"})


def infer_source_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = str(value).strip().lower()
    mapping = {
        "feature service": "feature_service",
        "map service": "map_service",
        "shapefile": "shapefile",
        "zip": "zip",
        "geojson": "geojson",
        "geopackage": "gpkg",
        "gpkg": "gpkg",
        "csv": "csv",
    }
    if normalized in mapping:
        return mapping[normalized]
    parsed = urlparse(str(value))
    path = parsed.path.lower()
    if path.endswith(".geojson") or path.endswith(".json"):
        return "geojson"
    if path.endswith(".gpkg"):
        return "gpkg"
    if path.endswith(".zip"):
        return "zip"
    if path.endswith(".csv"):
        return "csv"
    return normalized.replace(" ", "_")


def download_method_for_source_type(source_type: str | None) -> str:
    normalized = infer_source_type(source_type)
    if normalized in ARCGIS_SOURCE_TYPES:
        return "arcgis_rest"
    if normalized in DIRECT_DOWNLOAD_SOURCE_TYPES:
        return "direct_download"
    return "unknown"


def arcgis_online_item_download_url(item_id: str) -> str:
    return f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/data"


def arcgis_online_item_to_candidate(item: dict[str, Any], query: str, county_name: str | None = None) -> dict[str, Any] | None:
    source_type = infer_source_type(item.get("type"))
    if source_type is None:
        return None
    if source_type not in ARCGIS_SOURCE_TYPES and source_type not in DIRECT_DOWNLOAD_SOURCE_TYPES:
        return None
    keyword_text = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("name", "")),
            str(item.get("snippet", "")),
            str(item.get("description", "")),
            " ".join(str(value) for value in item.get("tags", [])),
            str(item.get("url", "")),
        ]
    ).upper()
    if not any(keyword in keyword_text for keyword in ["PARCEL", "PARCELS", "CADASTRAL", "ASSESSOR", "APN"]):
        return None
    query_tokens = [
        token
        for token in re.findall(r"[A-Z0-9]+", str(query).upper())
        if token not in {"PARCEL", "PARCELS", "TAX", "ASSESSOR", "CADASTRAL", "GIS", "ARCGIS", "FEATURE", "SERVICE"}
        and len(token) > 2
    ]
    if query_tokens and not any(token in keyword_text for token in query_tokens):
        return None
    source_url = str(item.get("url") or "").strip()
    if source_type in DIRECT_DOWNLOAD_SOURCE_TYPES and item.get("id"):
        source_url = arcgis_online_item_download_url(str(item["id"]))
    if not source_url:
        return None
    return {
        "county_name": county_name,
        "source_name": item.get("title") or item.get("name"),
        "source_url": source_url,
        "source_type": source_type,
        "organization": item.get("owner"),
        "layer_name": item.get("title") or item.get("name"),
        "item_title": item.get("title") or item.get("name"),
        "item_description": item.get("snippet") or item.get("description"),
        "access_method": "arcgis_online_search",
        "last_modified": item.get("modified"),
        "notes": f"ArcGIS Online search query: {query}",
        "license_usage_notes": item.get("licenseInfo"),
        "search_tags": item.get("tags", []),
        "portal_item_id": item.get("id"),
    }


def discover_arcgis_online_candidates(config: dict[str, Any], counties: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    queries = list(config.get("search_queries", []))
    if not queries:
        queries = [f"{config.get('state_name', config['state_code'])} parcels"]

    discovered: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for query in queries:
        payload = request_json(
            ARCGIS_ONLINE_SEARCH_URL,
            params={
                "q": f'({query}) AND (type:"Feature Service" OR type:"Map Service" OR type:"Shapefile" OR type:"GeoJson" OR type:"CSV" OR type:"GeoPackage")',
                "sortField": "modified",
                "sortOrder": "desc",
                "num": int(config.get("search_results_limit", 25)),
                "f": "json",
            },
        )
        for item in payload.get("results", []):
            candidate = arcgis_online_item_to_candidate(item, query=query)
            if candidate is None:
                continue
            source_url = str(candidate["source_url"])
            if source_url in seen_urls:
                continue
            seen_urls.add(source_url)
            discovered.append(candidate)

    county_templates = list(config.get("county_search_templates", []))
    county_limit = int(config.get("county_search_limit", 0))
    if county_templates and county_limit > 0 and not counties.empty:
        state_name = str(config.get("state_name", config["state_code"]))
        for county_name in counties["county_name"].head(county_limit).tolist():
            county_label = county_name.replace("_", " ").title()
            for template in county_templates:
                query = str(template).format(county_name=county_label, state_name=state_name, state_code=config["state_code"])
                payload = request_json(
                    ARCGIS_ONLINE_SEARCH_URL,
                    params={
                        "q": f'({query}) AND (type:"Feature Service" OR type:"Map Service" OR type:"Shapefile" OR type:"GeoJson" OR type:"CSV" OR type:"GeoPackage")',
                        "sortField": "modified",
                        "sortOrder": "desc",
                        "num": int(config.get("search_results_limit", 15)),
                        "f": "json",
                    },
                )
                for item in payload.get("results", []):
                    candidate = arcgis_online_item_to_candidate(item, query=query, county_name=county_name)
                    if candidate is None:
                        continue
                    source_url = str(candidate["source_url"])
                    if source_url in seen_urls:
                        continue
                    seen_urls.add(source_url)
                    discovered.append(candidate)
    return discovered


def discover_root_candidates(root_url: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    payload = request_json(root_url.rstrip("/"), params={"f": "json"})
    for service in payload.get("services", []):
        service_name = str(service.get("name", ""))
        service_type = str(service.get("type", ""))
        if not service_name:
            continue
        full_service_url = f"{root_url.rstrip('/')}/{service_name}/{service_type}"
        if not any(keyword in service_name.upper() for keyword in ["PARCEL", "CADASTRAL", "PROPERTY"]):
            continue
        try:
            metadata = fetch_arcgis_metadata(full_service_url)
        except Exception:
            continue
        for layer in metadata.get("layers", []):
            layer_name = str(layer.get("name", ""))
            if not any(keyword in layer_name.upper() for keyword in ["PARCEL", "CADASTRAL", "PROPERTY"]):
                continue
            candidates.append(
                {
                    "source_name": layer_name or service_name.split("/")[-1],
                    "source_url": f"{full_service_url}/{layer.get('id')}",
                    "source_type": "map_service" if service_type.lower() == "mapserver" else "feature_service",
                    "organization": str(payload.get("currentVersion", "")),
                    "notes": f"Discovered from ArcGIS root {root_url}",
                }
            )
    return candidates


def score_candidate_tokens(tokens: str, field_tokens: str = "", geometry_type: str = "") -> tuple[int, str]:
    score = 0
    notes: list[str] = []
    geometry_type_lower = str(geometry_type).lower()
    if "polygon" in geometry_type_lower:
        score += 30
        notes.append("polygon_geometry")
    for keyword in POSITIVE_KEYWORDS:
        if keyword in tokens.upper():
            score += 12
            notes.append(f"keyword:{keyword}")
    for keyword in FIELD_HINTS:
        if keyword in field_tokens.upper():
            score += 6
            notes.append(f"field:{keyword}")
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in tokens.upper() and "PARCEL" not in keyword:
            score -= 8
            notes.append(f"negative:{keyword}")
    return max(0, min(score, 100)), "|".join(notes)


def score_candidate(metadata: dict[str, Any]) -> tuple[int, str]:
    tokens = " ".join(
        [
            str(metadata.get("name", "")),
            str(metadata.get("description", "")),
            str(metadata.get("serviceDescription", "")),
            str(metadata.get("displayField", "")),
        ]
    )
    field_tokens = " ".join(str(field.get("name", "")) for field in metadata.get("fields", []))
    return score_candidate_tokens(tokens=tokens, field_tokens=field_tokens, geometry_type=str(metadata.get("geometryType", "")))


def score_non_arcgis_candidate(candidate: dict[str, Any]) -> tuple[int, str]:
    tokens = " ".join(
        [
            str(candidate.get("source_name", "")),
            str(candidate.get("item_title", "")),
            str(candidate.get("item_description", "")),
            str(candidate.get("source_url", "")),
            " ".join(str(value) for value in candidate.get("search_tags", [])),
        ]
    )
    return score_candidate_tokens(tokens=tokens, geometry_type=str(candidate.get("geometry_type_hint", "")))


def candidate_records(config: dict[str, Any], discovered_at: str, counties: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    candidates = list(config.get("candidate_layers", []))
    candidates.extend(config.get("direct_download_candidates", []))
    for root_url in config.get("arcgis_root_urls", []):
        try:
            discovered = discover_root_candidates(root_url)
        except Exception:
            discovered = []
        for candidate in discovered:
            if candidate["source_url"] not in {item["source_url"] for item in candidates}:
                candidates.append(candidate)
    try:
        discovered_online = discover_arcgis_online_candidates(config, counties)
    except Exception:
        discovered_online = []
    for candidate in discovered_online:
        if candidate["source_url"] not in {item["source_url"] for item in candidates}:
            candidates.append(candidate)

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        source_type = infer_source_type(candidate.get("source_type") or candidate.get("source_url"))
        record = {
            "county_name": sanitize_name(candidate["county_name"]) if candidate.get("county_name") else pd.NA,
            "state_code": config["state_code"],
            "source_name": candidate.get("source_name"),
            "source_url": candidate.get("source_url"),
            "source_type": source_type,
            "organization": candidate.get("organization"),
            "layer_name": candidate.get("layer_name", pd.NA),
            "item_title": candidate.get("item_title") or candidate.get("source_name"),
            "item_description": candidate.get("item_description", pd.NA),
            "access_method": candidate.get("access_method") or download_method_for_source_type(source_type),
            "last_modified": candidate.get("last_modified", pd.NA),
            "license_usage_notes": candidate.get("license_usage_notes") or candidate.get("notes"),
            "parcel_confidence_score": 0,
            "validation_notes": "",
            "access_status": "candidate",
            "discovered_at": discovered_at,
            "download_status": "not_attempted",
            "geometry_type": candidate.get("geometry_type_hint", pd.NA),
            "row_count": pd.NA,
            "lineage_record_type": "candidate_source",
            "lineage_layer_url": candidate.get("source_url"),
            "lineage_service_url": derive_service_url(str(candidate.get("source_url", ""))) if candidate.get("source_url") else pd.NA,
            "lineage_object_id_field": pd.NA,
            "lineage_object_id_count": pd.NA,
            "notes": candidate.get("notes"),
        }
        try:
            if source_type in ARCGIS_SOURCE_TYPES:
                metadata = fetch_arcgis_metadata(str(candidate["source_url"]))
                confidence, notes = score_candidate(metadata)
                record["layer_name"] = metadata.get("name")
                record["item_description"] = metadata.get("description") or metadata.get("serviceDescription")
                record["last_modified"] = metadata.get("editingInfo", {}).get("lastEditDate")
                record["geometry_type"] = metadata.get("geometryType")
                record["parcel_confidence_score"] = confidence
                record["validation_notes"] = notes
                record["access_status"] = "validated" if confidence >= 35 and "polygon" in str(metadata.get("geometryType", "")).lower() else "rejected"
            else:
                confidence, notes = score_non_arcgis_candidate(candidate)
                record["parcel_confidence_score"] = confidence
                record["validation_notes"] = notes
                record["access_status"] = "validated" if source_type in DIRECT_DOWNLOAD_SOURCE_TYPES and confidence >= 20 else "rejected"
        except Exception as exc:
            record["access_status"] = "metadata_failed"
            record["validation_notes"] = str(exc)
        records.append(record)
    return records


def filter_counties(counties: gpd.GeoDataFrame, county_whitelist: list[str] | None, max_counties: int | None) -> gpd.GeoDataFrame:
    out = counties.copy()
    if county_whitelist:
        wanted = {sanitize_name(value) for value in county_whitelist}
        out = out.loc[out["county_name"].isin(wanted)].copy()
    if max_counties is not None:
        out = out.head(max_counties).copy()
    return out.reset_index(drop=True)


def county_relevance_score(candidate: pd.Series, county_name: str) -> int:
    score = int(pd.to_numeric(pd.Series([candidate.get("parcel_confidence_score")]), errors="coerce").fillna(0).iloc[0])
    candidate_county = sanitize_name(candidate.get("county_name")) if pd.notna(candidate.get("county_name")) else ""
    if candidate_county:
        if candidate_county != county_name:
            return -1000
        score += 100

    haystack = " ".join(
        [
            str(candidate.get("source_name", "")),
            str(candidate.get("item_title", "")),
            str(candidate.get("item_description", "")),
            str(candidate.get("source_url", "")),
        ]
    ).lower()
    county_label = county_name.replace("_", " ").lower()
    if county_label in haystack:
        score += 25
    if infer_source_type(candidate.get("source_type")) in ARCGIS_SOURCE_TYPES:
        score += 5
    return score


def fetch_object_ids(source_url: str, county_geometry) -> tuple[str, list[int]]:
    query_url = derive_query_url(source_url)
    xmin, ymin, xmax, ymax = county_geometry.bounds
    payload = request_json(
        query_url,
        params={
            "where": "1=1",
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "returnIdsOnly": "true",
            "returnGeometry": "false",
            "f": "json",
        },
    )
    object_id_field = str(payload.get("objectIdFieldName") or "")
    object_ids = sorted({int(value) for value in payload.get("objectIds", []) if str(value).strip()})
    return object_id_field, object_ids


def fetch_features_by_ids(source_url: str, object_id_field: str, object_ids: list[int]) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    query_url = derive_query_url(source_url)
    for index in range(0, len(object_ids), OBJECTID_BATCH_SIZE):
        batch = object_ids[index: index + OBJECTID_BATCH_SIZE]
        payload = request_json(
            query_url,
            params={
                "where": "1=1",
                "objectIds": ",".join(str(value) for value in batch),
                "objectIdField": object_id_field,
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
            },
        )
        features = payload.get("features", [])
        if features:
            frames.append(gpd.GeoDataFrame.from_features(features, crs=TARGET_CRS))
        time.sleep(SLEEP_SECONDS)
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
    merged = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)


def filter_to_county_geometry(gdf: gpd.GeoDataFrame, county_geometry) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    representative_points = gdf.geometry.representative_point()
    mask = representative_points.within(county_geometry)
    return gdf.loc[mask].copy()


def keep_polygon_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    polygon_mask = gdf.geometry.geom_type.fillna("").str.contains("Polygon", case=False, regex=False)
    return gdf.loc[polygon_mask].copy()


def infer_download_path(candidate: pd.Series, county_dir: Path) -> Path:
    source_type = infer_source_type(candidate.get("source_type"))
    parsed = urlparse(str(candidate.get("source_url", "")))
    filename = Path(unquote(parsed.path)).name
    suffix = Path(filename).suffix if filename else ""
    if not suffix:
        suffix = SOURCE_TYPE_EXTENSIONS.get(str(source_type), ".bin")
    return county_dir / f"source_download{suffix}"


def download_candidate_file(candidate: pd.Series, county_dir: Path, refresh_existing: bool) -> Path:
    destination = infer_download_path(candidate, county_dir)
    if destination.exists() and not refresh_existing:
        return destination
    payload = request_bytes(str(candidate["source_url"]))
    destination.write_bytes(payload)
    return destination


def inspect_direct_download_layers(download_path: Path, source_type: str, configured_layer: str | None) -> list[str]:
    if source_type in {"zip", "shapefile", "gpkg"}:
        try:
            layer_info = pyogrio.list_layers(str(download_path))
            layer_names = [str(row[0]) for row in layer_info] if layer_info is not None else []
        except Exception:
            layer_names = []
        if configured_layer:
            return [configured_layer]
        if layer_names:
            return layer_names
    return [configured_layer] if configured_layer else [""]


def read_csv_candidate(download_path: Path, candidate: pd.Series) -> gpd.GeoDataFrame:
    frame = pd.read_csv(download_path)
    wkt_field = str(candidate.get("geometry_wkt_field") or "").strip()
    if not wkt_field or wkt_field not in frame.columns:
        raise RuntimeError("CSV candidate requires a configured geometry_wkt_field present in the file.")
    geometry = frame[wkt_field].map(lambda value: shapely_wkt.loads(value) if pd.notna(value) and str(value).strip() else None)
    source_crs = str(candidate.get("source_crs") or TARGET_CRS)
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=source_crs)


def read_direct_download(download_path: Path, candidate: pd.Series) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    source_type = str(infer_source_type(candidate.get("source_type")) or "")
    configured_layer = candidate.get("layer_name")
    if source_type == "csv":
        gdf = read_csv_candidate(download_path, candidate)
        return gdf, {"downloaded_file_path": download_path.relative_to(BASE_DIR).as_posix(), "downloaded_file_size_bytes": download_path.stat().st_size}

    layers = inspect_direct_download_layers(download_path, source_type, configured_layer if pd.notna(configured_layer) else None)
    read_target = f"zip://{download_path}" if source_type in {"zip", "shapefile"} else str(download_path)
    frames: list[gpd.GeoDataFrame] = []
    layer_names_used: list[str] = []
    for layer_name in layers:
        read_kwargs: dict[str, Any] = {"engine": "pyogrio"}
        if layer_name:
            read_kwargs["layer"] = layer_name
        try:
            frame = gpd.read_file(read_target, **read_kwargs)
        except Exception:
            continue
        if frame.empty:
            continue
        frames.append(frame)
        layer_names_used.append(layer_name or "default")
    if not frames:
        raise RuntimeError(f"Unable to read a supported layer from {download_path.name}")
    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    metadata = {
        "downloaded_file_path": download_path.relative_to(BASE_DIR).as_posix(),
        "downloaded_file_size_bytes": download_path.stat().st_size,
        "downloaded_layers": layer_names_used,
    }
    return merged, metadata


def load_direct_candidate_features(candidate: pd.Series, county_row: pd.Series, county_dir: Path, refresh_existing: bool) -> tuple[gpd.GeoDataFrame, dict[str, Any], Path]:
    download_path = download_candidate_file(candidate, county_dir, refresh_existing=refresh_existing)
    gdf, file_metadata = read_direct_download(download_path, candidate)
    if gdf.crs is None:
        source_crs = candidate.get("source_crs")
        if source_crs:
            gdf = gdf.set_crs(str(source_crs))
        else:
            gdf = gdf.set_crs(TARGET_CRS)
    gdf = gdf.to_crs(TARGET_CRS)
    gdf = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = keep_polygon_geometries(gdf)
    gdf = filter_to_county_geometry(gdf, county_row.geometry)
    if gdf.empty:
        raise RuntimeError("Direct-download candidate returned no polygon features for county geometry.")
    return gdf, file_metadata, download_path


def select_candidate_for_county(county_row: pd.Series, registry_df: pd.DataFrame) -> tuple[pd.Series | None, dict[str, Any]]:
    validated = registry_df.loc[registry_df["access_status"].eq("validated")].copy()
    if validated.empty:
        return None, {}
    best_candidate: pd.Series | None = None
    best_context: dict[str, Any] = {}
    best_score = -10**9
    county_name = str(county_row["county_name"])
    sorted_candidates = validated.copy()
    sorted_candidates["county_relevance_score"] = sorted_candidates.apply(lambda row: county_relevance_score(row, county_name), axis=1)
    sorted_candidates = sorted_candidates.sort_values(["county_relevance_score", "parcel_confidence_score", "source_name"], ascending=[False, False, True])
    for candidate in sorted_candidates.itertuples(index=False):
        candidate_series = pd.Series(candidate._asdict())
        relevance = int(candidate_series.get("county_relevance_score", 0))
        if relevance < 0:
            continue
        source_type = infer_source_type(candidate_series.get("source_type"))
        if source_type in ARCGIS_SOURCE_TYPES:
            try:
                object_id_field, object_ids = fetch_object_ids(str(candidate_series["source_url"]), county_row.geometry)
            except Exception:
                continue
            if not object_id_field or not object_ids:
                continue
            selection_score = (relevance * 1_000_000) + len(object_ids)
            context = {
                "download_strategy": "arcgis_ids",
                "object_id_field": object_id_field,
                "object_ids": object_ids,
            }
        elif source_type in DIRECT_DOWNLOAD_SOURCE_TYPES:
            selection_score = relevance
            context = {
                "download_strategy": "direct_download",
                "object_id_field": None,
                "object_ids": [],
            }
        else:
            continue

        if selection_score > best_score:
            best_candidate = candidate_series
            best_context = context
            best_score = selection_score
    return best_candidate, best_context


def lookup_candidate(registry_df: pd.DataFrame, source_name: str | None = None, source_url: str | None = None) -> pd.Series | None:
    candidates = registry_df.copy()
    if source_url:
        matched = candidates.loc[candidates["source_url"].eq(source_url)]
        if not matched.empty:
            return matched.sort_values(["parcel_confidence_score", "source_name"], ascending=[False, True]).iloc[0]
    if source_name:
        matched = candidates.loc[candidates["source_name"].astype("string").str.casefold().eq(str(source_name).casefold())]
        if not matched.empty:
            return matched.sort_values(["parcel_confidence_score", "source_name"], ascending=[False, True]).iloc[0]
    return None


def coalesce_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if value is pd.NA:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return pd.NA


def geometry_type_label(gdf: gpd.GeoDataFrame) -> str | pd._libs.missing.NAType:
    if gdf.empty or "geometry" not in gdf.columns:
        return pd.NA
    geom_types = gdf.geometry.geom_type.dropna().unique().tolist()
    return "|".join(sorted(str(value) for value in geom_types)) if geom_types else pd.NA


def normalize_text_value(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return re.sub(r"\s+", " ", str(value).strip())


def coerce_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def choose_first_nonempty_series(frame: pd.DataFrame, *column_names: str) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column_name in column_names:
        if column_name not in frame.columns:
            continue
        candidate = frame[column_name].astype("string")
        mask = out.isna() & candidate.fillna("").str.strip().ne("")
        out.loc[mask] = candidate.loc[mask]
    return out


def build_parcel_row_id(frame: pd.DataFrame) -> pd.Series:
    source_name = frame["source_name"].astype("string").fillna("")
    state_code = frame["state_code"].astype("string").fillna("")
    county_name = frame["county_name"].astype("string").fillna("")
    source_record_id = frame["source_record_id"].astype("string").fillna("")
    parcel_id = frame["parcel_id"].astype("string").fillna("")
    tokens = state_code + "|" + county_name + "|" + source_name + "|" + source_record_id + "|" + parcel_id
    return tokens.map(lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()[:24]).astype("string")


def build_record_hash(frame: gpd.GeoDataFrame) -> pd.Series:
    def row_hash(row: pd.Series) -> str:
        geometry = row.get("geometry")
        geometry_token = geometry.wkb_hex if geometry is not None else ""
        payload = "|".join(
            [
                normalize_text_value(row.get("parcel_id")),
                normalize_text_value(row.get("state_code")),
                normalize_text_value(row.get("county_fips")),
                normalize_text_value(row.get("county_name")),
                normalize_text_value(row.get("owner_name")),
                normalize_text_value(row.get("acreage")),
                normalize_text_value(row.get("assessed_value")),
                normalize_text_value(row.get("land_use")),
                geometry_token,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return frame.apply(row_hash, axis=1).astype("string")


def extract_polygonal_geometry(geometry) -> Any:
    if geometry is None:
        return None
    geom_type = getattr(geometry, "geom_type", "")
    if "Polygon" in geom_type:
        return geometry
    if geom_type == "GeometryCollection":
        polygons = [geom for geom in getattr(geometry, "geoms", []) if "Polygon" in getattr(geom, "geom_type", "")]
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]
        from shapely.geometry import MultiPolygon

        flattened = []
        for polygon in polygons:
            if polygon.geom_type == "Polygon":
                flattened.append(polygon)
            elif polygon.geom_type == "MultiPolygon":
                flattened.extend(list(polygon.geoms))
        return MultiPolygon(flattened) if flattened else None
    return None


def validate_and_repair_standardized_geometries(gdf: gpd.GeoDataFrame, county_name: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    repaired = gdf.copy()
    logs: list[dict[str, Any]] = []
    if repaired.empty:
        return repaired, pd.DataFrame(columns=GEOMETRY_LOG_COLUMNS)

    for index, geometry in repaired.geometry.items():
        parcel_row_id = repaired.at[index, "parcel_row_id"] if "parcel_row_id" in repaired.columns else pd.NA
        source_record_id = repaired.at[index, "source_record_id"] if "source_record_id" in repaired.columns else pd.NA
        if geometry is None or geometry.is_empty:
            logs.append(
                {
                    "county_name": county_name,
                    "parcel_row_id": parcel_row_id,
                    "source_record_id": source_record_id,
                    "geometry_issue": "missing_or_empty",
                    "repair_status": "dropped",
                    "notes": "",
                }
            )
            continue
        if geometry.is_valid:
            continue

        repaired_geometry = None
        repair_status = "unrepaired"
        notes = ""
        try:
            repaired_geometry = make_valid(geometry)
            repair_status = "make_valid"
        except Exception as exc:
            notes = f"make_valid_failed:{exc}"
        if repaired_geometry is None or getattr(repaired_geometry, "is_empty", True):
            try:
                repaired_geometry = geometry.buffer(0)
                repair_status = "buffer0"
            except Exception as exc:
                notes = f"{notes}|buffer0_failed:{exc}".strip("|")
        repaired_geometry = extract_polygonal_geometry(repaired_geometry)
        if repaired_geometry is None or getattr(repaired_geometry, "is_empty", True):
            logs.append(
                {
                    "county_name": county_name,
                    "parcel_row_id": parcel_row_id,
                    "source_record_id": source_record_id,
                    "geometry_issue": "invalid_geometry",
                    "repair_status": "dropped",
                    "notes": notes,
                }
            )
            repaired.at[index, "geometry"] = None
            continue

        repaired.at[index, "geometry"] = repaired_geometry
        logs.append(
            {
                "county_name": county_name,
                "parcel_row_id": parcel_row_id,
                "source_record_id": source_record_id,
                "geometry_issue": "invalid_geometry",
                "repair_status": repair_status,
                "notes": notes,
            }
        )

    repaired = repaired.loc[repaired.geometry.notna() & ~repaired.geometry.is_empty].copy()
    repaired = keep_polygon_geometries(repaired)
    return repaired, pd.DataFrame.from_records(logs, columns=GEOMETRY_LOG_COLUMNS)


def standardize_parcel_columns(
    normalized: gpd.GeoDataFrame,
    county_fips: str | None,
    county_name: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    standardized = normalized.copy()
    standardized["county_fips"] = pd.Series(county_fips if county_fips else pd.NA, index=standardized.index, dtype="string")
    standardized["source_record_id"] = choose_first_nonempty_series(standardized, "source_row_identifier", "source_parcel_number", "source_alt_parcel_number", "source_ppin")
    standardized["parcel_id"] = choose_first_nonempty_series(standardized, "source_parcel_number", "source_alt_parcel_number", "source_ppin", "source_record_id")
    empty_parcel_mask = standardized["parcel_id"].fillna("").str.strip().eq("")
    standardized.loc[empty_parcel_mask, "parcel_id"] = standardized.loc[empty_parcel_mask, "source_record_id"]
    acreage = choose_first_nonempty_series(standardized, "total_acres", "gis_acres", "tax_acres")
    standardized["acreage"] = coerce_float_series(acreage)
    assessed_value = choose_first_nonempty_series(standardized, "total_value", "land_value")
    standardized["assessed_value"] = coerce_float_series(assessed_value)
    standardized["land_use"] = choose_first_nonempty_series(standardized, "zoning").astype("string")
    standardized["owner_name"] = choose_first_nonempty_series(standardized, "owner_name").astype("string")
    standardized["parcel_row_id"] = build_parcel_row_id(
        pd.DataFrame(
            {
                "state_code": standardized["state_code"],
                "county_name": standardized["county_name"],
                "source_name": standardized["source_name"],
                "source_record_id": standardized["source_record_id"],
                "parcel_id": standardized["parcel_id"],
            }
        )
    )
    standardized, geometry_log = validate_and_repair_standardized_geometries(standardized, county_name=county_name)
    standardized["record_hash"] = build_record_hash(standardized)
    for column in STANDARDIZED_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = pd.NA
    ordered = STANDARDIZED_COLUMNS + [column for column in standardized.columns if column not in STANDARDIZED_COLUMNS]
    standardized = standardized.loc[:, ordered].copy()
    return standardized, geometry_log


def snapshot_timestamp() -> str:
    return pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def make_json_compatible(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): make_json_compatible(inner_value) for key, inner_value in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [make_json_compatible(inner_value) for inner_value in value]
        if value is None or value is pd.NA:
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        try:
            if pd.isna(value):
                return None
        except TypeError:
            pass
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as outfile:
        json.dump(make_json_compatible(payload), outfile, indent=2)
    temp_path.replace(path)


def rename_reserved_write_columns(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    reserved_names = {"fid", "ogc_fid"}
    rename_map: dict[str, str] = {}
    used_names = {str(column) for column in frame.columns}
    for column in frame.columns:
        column_str = str(column)
        if column_str.lower() not in reserved_names:
            continue
        candidate = f"src_{column_str}"
        counter = 1
        while candidate in used_names or candidate in rename_map.values():
            counter += 1
            candidate = f"src_{column_str}_{counter}"
        rename_map[column_str] = candidate
        used_names.add(candidate)
    if rename_map:
        return frame.rename(columns=rename_map)
    return frame


def write_gpkg_replace(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rename_reserved_write_columns(gdf).to_file(path, driver="GPKG", engine="pyogrio")


def write_csv_replace(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def publish_versioned_raw_snapshot(
    county_dir: Path,
    downloaded: gpd.GeoDataFrame,
    metadata_payload: dict[str, Any],
    raw_output_path: Path,
    metadata_output_path: Path,
    staged_download_path: Path | None,
    geometry_log: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    version_id = snapshot_timestamp()
    snapshot_dir = county_dir / "raw_snapshots" / version_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_raw_path = snapshot_dir / "raw_parcels.gpkg"
    snapshot_metadata_path = snapshot_dir / "source_metadata.json"
    metadata_payload["raw_snapshot_path"] = snapshot_raw_path.relative_to(BASE_DIR).as_posix()
    metadata_payload["metadata_snapshot_path"] = snapshot_metadata_path.relative_to(BASE_DIR).as_posix()
    write_gpkg_replace(downloaded, snapshot_raw_path)
    write_json(snapshot_metadata_path, metadata_payload)
    if staged_download_path and staged_download_path.exists():
        shutil.copy2(staged_download_path, snapshot_dir / staged_download_path.name)
    if geometry_log is not None and not geometry_log.empty:
        write_csv_replace(geometry_log, snapshot_dir / "geometry_validation_log.csv")

    write_gpkg_replace(downloaded, raw_output_path)
    write_json(metadata_output_path, metadata_payload)
    return snapshot_raw_path, snapshot_metadata_path


def canonical_series(frame: pd.DataFrame, aliases: list[str]) -> pd.Series:
    normalized_columns = {normalize_column_name(column): column for column in frame.columns}
    for alias in aliases:
        key = normalize_column_name(alias)
        if key in normalized_columns:
            return frame[normalized_columns[key]]
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def rename_conflicting_source_columns(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    canonical_keys = {normalize_column_name(column): column for column in CANONICAL_COLUMNS if column != "geometry"}
    rename_map: dict[str, str] = {}
    used_names = set(frame.columns)
    for column in frame.columns:
        if str(column) == "geometry":
            continue
        canonical_name = canonical_keys.get(normalize_column_name(column))
        if canonical_name and str(column) != canonical_name:
            candidate = f"src_{column}"
            counter = 1
            while candidate in used_names or candidate in rename_map.values():
                counter += 1
                candidate = f"src_{column}_{counter}"
            rename_map[str(column)] = candidate
            used_names.add(candidate)
    if rename_map:
        return frame.rename(columns=rename_map)
    return frame


def normalize_downloaded_parcels(
    gdf: gpd.GeoDataFrame,
    state_code: str,
    county_name: str,
    source_name: str,
    source_dataset_path: Path,
    config: dict[str, Any],
    object_id_field: str,
) -> gpd.GeoDataFrame:
    normalized = rename_conflicting_source_columns(gdf.copy())
    aliases = config.get("normalization_aliases", {})
    normalized["state_code"] = pd.Series(state_code, index=normalized.index, dtype="string")
    normalized["county_name"] = pd.Series(county_name, index=normalized.index, dtype="string")
    normalized["source_name"] = pd.Series(source_name, index=normalized.index, dtype="string")
    normalized["source_dataset_path"] = pd.Series(source_dataset_path.relative_to(BASE_DIR).as_posix(), index=normalized.index, dtype="string")
    normalized["source_layer_name"] = pd.Series(source_name, index=normalized.index, dtype="string")
    row_id_series = canonical_series(normalized, [object_id_field, "OBJECTID", "FID"])
    normalized["source_row_identifier"] = row_id_series.astype("string")
    for field_name in [column for column in CANONICAL_COLUMNS if column not in {"state_code", "county_name", "source_name", "source_dataset_path", "source_layer_name", "source_row_identifier", "geometry"}]:
        normalized[field_name] = canonical_series(normalized, aliases.get(field_name, [])).astype("string")
    for numeric_field in ["tax_acres", "gis_acres", "total_acres", "land_value", "total_value"]:
        normalized[numeric_field] = pd.to_numeric(normalized[numeric_field], errors="coerce")
    normalized["tax_year"] = pd.to_numeric(normalized["tax_year"], errors="coerce").astype("Int64")
    return normalized


def build_county_metadata_payload(
    *,
    state_code: str,
    county_name: str,
    candidate: pd.Series,
    layer_metadata: dict[str, Any],
    service_metadata: dict[str, Any],
    county_output: Path,
    normalized_output: Path,
    object_id_field: str | None,
    object_id_count: int | None,
    row_count: int,
    download_method: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "state_code": state_code,
        "county_name": county_name,
        "source_name": candidate.get("source_name"),
        "source_url": candidate.get("source_url"),
        "source_type": candidate.get("source_type"),
        "organization": candidate.get("organization"),
        "access_method": candidate.get("access_method"),
        "parcel_confidence_score": candidate.get("parcel_confidence_score"),
        "layer_name": layer_metadata.get("name"),
        "service_name": service_metadata.get("mapName") or service_metadata.get("serviceDescription") or service_metadata.get("name"),
        "geometry_type": layer_metadata.get("geometryType"),
        "download_status": "downloaded",
        "row_count": row_count,
        "lineage_layer_url": candidate.get("source_url"),
        "lineage_service_url": derive_service_url(str(candidate.get("source_url"))),
        "lineage_object_id_field": object_id_field or pd.NA,
        "lineage_object_id_count": object_id_count if object_id_count is not None else pd.NA,
        "source_dataset_path": county_output.relative_to(BASE_DIR).as_posix(),
        "normalized_dataset_path": normalized_output.relative_to(BASE_DIR).as_posix(),
        "download_method": download_method,
        "downloaded_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "layer_metadata": layer_metadata,
        "service_metadata": service_metadata,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    return payload


def read_county_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_registry_output(
    registry_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> pd.DataFrame:
    candidate_registry = registry_df.copy()
    if not candidate_registry.empty:
        candidate_registry["notes"] = candidate_registry["notes"].fillna(candidate_registry["license_usage_notes"])

    county_registry = coverage_df.copy()
    if not county_registry.empty:
        county_registry["state_code"] = pd.NA
        county_registry["organization"] = pd.NA
        county_registry["layer_name"] = pd.NA
        county_registry["item_title"] = county_registry["source_name"]
        county_registry["item_description"] = pd.NA
        county_registry["access_method"] = pd.NA
        county_registry["last_modified"] = pd.NA
        county_registry["license_usage_notes"] = pd.NA
        county_registry["validation_notes"] = pd.NA
        county_registry["access_status"] = np.where(
            county_registry["download_status"].eq("downloaded"),
            "used_for_download",
            county_registry["download_status"],
        )
        county_registry["discovered_at"] = pd.NA
        county_registry["lineage_record_type"] = "county_download"
        county_registry["lineage_layer_url"] = county_registry["lineage_layer_url"].fillna(county_registry["source_url"])
        county_registry["lineage_service_url"] = county_registry["lineage_service_url"].fillna(
            county_registry["source_url"].map(lambda value: derive_service_url(str(value)) if pd.notna(value) else pd.NA)
        )
        if not registry_df.empty:
            merge_columns = [
                "source_url",
                "state_code",
                "source_type",
                "organization",
                "layer_name",
                "item_title",
                "item_description",
                "access_method",
                "last_modified",
                "license_usage_notes",
                "parcel_confidence_score",
                "validation_notes",
            ]
            county_registry = county_registry.merge(
                registry_df[merge_columns].drop_duplicates(subset=["source_url"]),
                how="left",
                on="source_url",
                suffixes=("", "_candidate"),
            )
            for column in [value for value in merge_columns if value != "source_url"]:
                candidate_column = f"{column}_candidate"
                if candidate_column in county_registry.columns:
                    county_registry[column] = county_registry[column].fillna(county_registry[candidate_column])
                    county_registry = county_registry.drop(columns=[candidate_column])

    combined = pd.concat([candidate_registry, county_registry], ignore_index=True, sort=False)
    for column in REGISTRY_OUTPUT_COLUMNS:
        if column not in combined.columns:
            combined[column] = pd.NA
    return combined.loc[:, REGISTRY_OUTPUT_COLUMNS].copy()


def build_outputs(
    state_code: str,
    registry_rows: list[dict[str, Any]],
    county_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> tuple[Path, Path, Path, Path]:
    state_suffix = state_code.lower()
    PARCELS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    registry_path = PARCELS_METADATA_DIR / f"parcel_source_registry_{state_suffix}.csv"
    summary_path = PARCELS_METADATA_DIR / f"parcel_discovery_summary_{state_suffix}.csv"
    failures_path = PARCELS_METADATA_DIR / f"parcel_download_failures_{state_suffix}.csv"
    coverage_path = PARCELS_METADATA_DIR / f"parcel_county_coverage_{state_suffix}.csv"
    legacy_registry_path = PARCELS_DIR / f"parcel_source_registry_{state_suffix}.csv"
    legacy_summary_path = PARCELS_DIR / f"parcel_discovery_summary_{state_suffix}.csv"
    legacy_failures_path = PARCELS_DIR / f"parcel_download_failures_{state_suffix}.csv"
    legacy_coverage_path = PARCELS_DIR / f"parcel_county_coverage_{state_suffix}.csv"

    registry_df = pd.DataFrame.from_records(registry_rows)
    coverage_df = pd.DataFrame.from_records(county_rows)
    failures_df = pd.DataFrame.from_records(failure_rows, columns=FAILURE_OUTPUT_COLUMNS)

    for column in COVERAGE_OUTPUT_COLUMNS:
        if column not in coverage_df.columns:
            coverage_df[column] = pd.NA
    coverage_df = coverage_df.loc[:, COVERAGE_OUTPUT_COLUMNS].copy()
    registry_output_df = build_registry_output(registry_df, coverage_df)

    summary_rows = [
        {"metric": "counties_searched", "value": int(coverage_df["county_name"].nunique()) if not coverage_df.empty else 0},
        {"metric": "candidate_sources_found", "value": int(registry_df["source_url"].nunique()) if not registry_df.empty else 0},
        {"metric": "counties_with_usable_parcel_downloads", "value": int(coverage_df["download_status"].eq("downloaded").sum()) if not coverage_df.empty else 0},
        {"metric": "counties_with_no_usable_public_source_found", "value": int(coverage_df["download_status"].ne("downloaded").sum()) if not coverage_df.empty else 0},
        {"metric": "likely_duplicate_sources", "value": int(registry_df["source_url"].duplicated().sum()) if not registry_df.empty else 0},
        {"metric": "partial_or_limited_counties", "value": int(coverage_df["notes"].fillna("").str.contains("partial|fallback|no_features", case=False, regex=True).sum()) if not coverage_df.empty else 0},
    ]

    registry_output_df.to_csv(registry_path, index=False)
    pd.DataFrame.from_records(summary_rows).to_csv(summary_path, index=False)
    failures_df.to_csv(failures_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    shutil.copy2(registry_path, legacy_registry_path)
    shutil.copy2(summary_path, legacy_summary_path)
    shutil.copy2(failures_path, legacy_failures_path)
    shutil.copy2(coverage_path, legacy_coverage_path)
    return registry_path, summary_path, failures_path, coverage_path


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    config = load_state_config(args.state_code.upper(), args.config)
    state_code = str(config["state_code"]).upper()
    discovered_at = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Loading county boundaries for {state_code}")
    counties = filter_counties(get_county_boundaries(config), args.counties, args.max_counties)
    registry_rows = candidate_records(config, discovered_at, counties)
    registry_df = pd.DataFrame.from_records(registry_rows)

    state_raw_dir = PARCELS_RAW_DIR / state_code.lower()
    state_raw_dir.mkdir(parents=True, exist_ok=True)
    state_standardized_dir = PARCELS_STANDARDIZED_DIR / state_code.lower()
    state_standardized_dir.mkdir(parents=True, exist_ok=True)
    PARCELS_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    county_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    normalized_frames: list[gpd.GeoDataFrame] = []
    standardized_frames: list[gpd.GeoDataFrame] = []
    geometry_logs: list[pd.DataFrame] = []

    for county_row in counties.itertuples(index=False):
        county_name = county_row.county_name
        county_dir = state_raw_dir / county_name
        county_dir.mkdir(parents=True, exist_ok=True)
        county_standardized_dir = state_standardized_dir / county_name
        county_standardized_dir.mkdir(parents=True, exist_ok=True)
        county_output = county_dir / "raw_parcels.gpkg"
        normalized_output = county_dir / "normalized_raw_parcels.gpkg"
        metadata_output = county_dir / "source_metadata.json"
        standardized_output = county_standardized_dir / "standardized_parcels.gpkg"
        geometry_log_output = county_standardized_dir / "geometry_validation_log.csv"
        if county_output.exists() and normalized_output.exists() and not args.refresh_existing:
            try:
                cached_normalized = gpd.read_file(normalized_output, engine="pyogrio")
                normalized_frames.append(cached_normalized)
                standardized_cached, geometry_log = standardize_parcel_columns(
                    cached_normalized,
                    county_fips=county_row.county_fips,
                    county_name=county_name,
                )
                write_gpkg_replace(standardized_cached, standardized_output)
                write_csv_replace(geometry_log, geometry_log_output)
                standardized_frames.append(standardized_cached)
                geometry_logs.append(geometry_log)

                cached_source_name = cached_normalized["source_name"].astype("string").dropna().iloc[0] if "source_name" in cached_normalized.columns and not cached_normalized.empty else "cached_download"
                cached_metadata = read_county_metadata(metadata_output)
                cached_candidate = lookup_candidate(
                    registry_df,
                    source_name=cached_source_name,
                    source_url=cached_metadata.get("source_url"),
                )
                needs_metadata_backfill = cached_candidate is not None and (
                    "layer_metadata" not in cached_metadata
                    or "service_metadata" not in cached_metadata
                    or pd.isna(coalesce_value(cached_metadata.get("source_url")))
                )
                if needs_metadata_backfill:
                    try:
                        cached_source_type = infer_source_type(cached_candidate.get("source_type"))
                        if cached_source_type in ARCGIS_SOURCE_TYPES:
                            cached_layer_metadata = fetch_arcgis_metadata(str(cached_candidate["source_url"]))
                            cached_service_metadata = fetch_arcgis_service_metadata(str(cached_candidate["source_url"]))
                        else:
                            cached_layer_metadata = {"name": cached_candidate.get("layer_name"), "geometryType": cached_metadata.get("geometry_type")}
                            cached_service_metadata = {"serviceDescription": cached_candidate.get("item_description")}
                        cached_metadata_payload = build_county_metadata_payload(
                            state_code=state_code,
                            county_name=county_name,
                            candidate=cached_candidate,
                            layer_metadata=cached_layer_metadata,
                            service_metadata=cached_service_metadata,
                            county_output=county_output,
                            normalized_output=normalized_output,
                            object_id_field=cached_metadata.get("lineage_object_id_field"),
                            object_id_count=cached_metadata.get("lineage_object_id_count"),
                            row_count=len(cached_normalized),
                            download_method="cached_existing_download",
                            extra_metadata={
                                "standardized_dataset_path": standardized_output.relative_to(BASE_DIR).as_posix(),
                                "geometry_validation_log_path": geometry_log_output.relative_to(BASE_DIR).as_posix(),
                                "invalid_geometry_count": int(len(geometry_log)),
                                "repaired_geometry_count": int(geometry_log["repair_status"].isin(["make_valid", "buffer0"]).sum()) if not geometry_log.empty else 0,
                            },
                        )
                        write_json(metadata_output, cached_metadata_payload)
                        cached_metadata = read_county_metadata(metadata_output)
                    except Exception:
                        pass
                cached_geometry_type = coalesce_value(cached_metadata.get("geometry_type"), geometry_type_label(cached_normalized))
                county_rows.append(
                    {
                        "county_name": county_name,
                        "source_name": cached_source_name,
                        "source_url": coalesce_value(cached_metadata.get("source_url"), cached_candidate.get("source_url") if cached_candidate is not None else pd.NA),
                        "source_type": coalesce_value(cached_metadata.get("source_type"), cached_candidate.get("source_type") if cached_candidate is not None else pd.NA),
                        "download_status": "downloaded",
                        "row_count": len(standardized_cached),
                        "geometry_type": coalesce_value(geometry_type_label(standardized_cached), cached_geometry_type),
                        "parcel_confidence_score": coalesce_value(cached_metadata.get("parcel_confidence_score"), cached_candidate.get("parcel_confidence_score") if cached_candidate is not None else pd.NA),
                        "lineage_layer_url": coalesce_value(cached_metadata.get("lineage_layer_url"), cached_candidate.get("source_url") if cached_candidate is not None else pd.NA),
                        "lineage_service_url": coalesce_value(
                            cached_metadata.get("lineage_service_url"),
                            derive_service_url(str(cached_candidate.get("source_url"))) if cached_candidate is not None and pd.notna(cached_candidate.get("source_url")) else pd.NA,
                        ),
                        "lineage_object_id_field": coalesce_value(cached_metadata.get("lineage_object_id_field")),
                        "lineage_object_id_count": coalesce_value(cached_metadata.get("lineage_object_id_count")),
                        "notes": f"used_existing_cached_download; standardized_rows={len(standardized_cached)}; invalid_geometries={len(geometry_log)}",
                    }
                )
                print(f"[skip] {county_name}: using existing cached download")
                continue
            except Exception as exc:
                county_rows.append(
                    {
                        "county_name": county_name,
                        "source_name": pd.NA,
                        "source_url": pd.NA,
                        "download_status": "failed",
                        "row_count": 0,
                        "geometry_type": pd.NA,
                        "parcel_confidence_score": pd.NA,
                        "notes": f"cached_standardization_failed:{exc}",
                    }
                )
                failure_rows.append({"county_name": county_name, "source_url": pd.NA, "failure_stage": "standardize_cached", "error": str(exc)})
                print(f"[fail] {county_name}: cached standardization failed ({exc})")
                continue

        candidate, selection_context = select_candidate_for_county(pd.Series(county_row._asdict()), registry_df)
        object_id_field = selection_context.get("object_id_field")
        object_ids = selection_context.get("object_ids", [])
        if candidate is None:
            county_rows.append(
                {
                    "county_name": county_name,
                    "source_name": pd.NA,
                    "source_url": pd.NA,
                    "download_status": "no_usable_source",
                    "row_count": 0,
                    "geometry_type": pd.NA,
                    "parcel_confidence_score": pd.NA,
                    "notes": "no_validated_candidate_with_features",
                }
            )
            failure_rows.append({"county_name": county_name, "source_url": pd.NA, "failure_stage": "selection", "error": "No validated candidate returned parcel features."})
            print(f"[miss] {county_name}: no usable public source found")
            continue

        try:
            source_type = infer_source_type(candidate.get("source_type"))
            extra_metadata: dict[str, Any] = {}
            staged_download_path: Path | None = None
            if selection_context.get("download_strategy") == "arcgis_ids":
                if not object_id_field or not object_ids:
                    raise RuntimeError("ArcGIS candidate did not return object IDs for county selection.")
                layer_metadata = fetch_arcgis_metadata(candidate["source_url"])
                service_metadata = fetch_arcgis_service_metadata(candidate["source_url"])
                downloaded = fetch_features_by_ids(candidate["source_url"], object_id_field, object_ids)
                downloaded = filter_to_county_geometry(downloaded, county_row.geometry)
                downloaded = keep_polygon_geometries(downloaded)
                if downloaded.empty:
                    raise RuntimeError("No parcel polygon features returned for county bbox.")
                downloaded = downloaded.to_crs(TARGET_CRS)
                download_method = "arcgis_return_ids_only_batches"
            elif selection_context.get("download_strategy") == "direct_download":
                downloaded, file_metadata, staged_download_path = load_direct_candidate_features(candidate, pd.Series(county_row._asdict()), county_dir, args.refresh_existing)
                layer_metadata = {
                    "name": candidate.get("layer_name") or candidate.get("source_name"),
                    "geometryType": geometry_type_label(downloaded),
                    "fields": [{"name": str(column)} for column in downloaded.columns if str(column) != "geometry"],
                }
                service_metadata = {
                    "serviceDescription": candidate.get("item_description"),
                    "downloadUrl": candidate.get("source_url"),
                }
                extra_metadata = file_metadata
                download_method = "direct_download_file"
            else:
                raise RuntimeError(f"Unsupported selection strategy: {selection_context.get('download_strategy')}")

            normalized = normalize_downloaded_parcels(
                downloaded,
                state_code=state_code,
                county_name=county_name,
                source_name=str(candidate["source_name"]),
                source_dataset_path=county_output,
                config=config,
                object_id_field=object_id_field,
            )
            standardized, geometry_log = standardize_parcel_columns(
                normalized,
                county_fips=county_row.county_fips,
                county_name=county_name,
            )

            extra_metadata.update(
                {
                    "standardized_dataset_path": standardized_output.relative_to(BASE_DIR).as_posix(),
                    "geometry_validation_log_path": geometry_log_output.relative_to(BASE_DIR).as_posix(),
                    "invalid_geometry_count": int(len(geometry_log)),
                    "repaired_geometry_count": int(geometry_log["repair_status"].isin(["make_valid", "buffer0"]).sum()) if not geometry_log.empty else 0,
                }
            )
            metadata_payload = build_county_metadata_payload(
                state_code=state_code,
                county_name=county_name,
                candidate=candidate,
                layer_metadata=layer_metadata,
                service_metadata=service_metadata,
                county_output=county_output,
                normalized_output=normalized_output,
                object_id_field=object_id_field,
                object_id_count=len(object_ids) if object_ids else None,
                row_count=len(downloaded),
                download_method=download_method,
                extra_metadata=extra_metadata,
            )
            publish_versioned_raw_snapshot(
                county_dir=county_dir,
                downloaded=downloaded,
                metadata_payload=metadata_payload,
                raw_output_path=county_output,
                metadata_output_path=metadata_output,
                staged_download_path=staged_download_path,
                geometry_log=geometry_log,
            )
            write_gpkg_replace(normalized, normalized_output)
            write_gpkg_replace(standardized, standardized_output)
            write_csv_replace(geometry_log, geometry_log_output)
            normalized_frames.append(normalized)
            standardized_frames.append(standardized)
            geometry_logs.append(geometry_log)
            county_rows.append(
                {
                    "county_name": county_name,
                    "source_name": candidate["source_name"],
                    "source_url": candidate["source_url"],
                    "source_type": source_type,
                    "download_status": "downloaded",
                    "row_count": len(standardized),
                    "geometry_type": coalesce_value(layer_metadata.get("geometryType"), geometry_type_label(downloaded)),
                    "parcel_confidence_score": candidate["parcel_confidence_score"],
                    "lineage_layer_url": candidate["source_url"],
                    "lineage_service_url": derive_service_url(str(candidate["source_url"])),
                    "lineage_object_id_field": object_id_field,
                    "lineage_object_id_count": len(object_ids) if object_ids else pd.NA,
                    "notes": (
                        f"download_method={download_method}; object_ids={len(object_ids)}; invalid_geometries={len(geometry_log)}"
                        if object_ids
                        else f"download_method={download_method}; invalid_geometries={len(geometry_log)}"
                    ),
                }
            )
            print(f"[ok] {county_name}: {len(standardized):,} standardized rows from {candidate['source_name']}")
        except Exception as exc:
            county_rows.append(
                {
                    "county_name": county_name,
                    "source_name": candidate["source_name"],
                    "source_url": candidate["source_url"],
                    "source_type": candidate.get("source_type"),
                    "download_status": "failed",
                    "row_count": 0,
                    "geometry_type": candidate.get("geometry_type"),
                    "parcel_confidence_score": candidate.get("parcel_confidence_score"),
                    "lineage_layer_url": candidate["source_url"],
                    "lineage_service_url": derive_service_url(str(candidate["source_url"])),
                    "lineage_object_id_field": object_id_field,
                    "lineage_object_id_count": len(object_ids),
                    "notes": str(exc),
                }
            )
            failure_rows.append({"county_name": county_name, "source_url": candidate["source_url"], "failure_stage": "download", "error": str(exc)})
            print(f"[fail] {county_name}: {exc}")

    if normalized_frames:
        statewide_normalized = gpd.GeoDataFrame(pd.concat(normalized_frames, ignore_index=True), geometry="geometry", crs=TARGET_CRS)
        statewide_path = state_raw_dir / f"normalized_raw_parcels_{state_code.lower()}.gpkg"
        write_gpkg_replace(statewide_normalized, statewide_path)
    if standardized_frames:
        statewide_standardized = gpd.GeoDataFrame(pd.concat(standardized_frames, ignore_index=True), geometry="geometry", crs=TARGET_CRS)
        statewide_standardized_path = state_standardized_dir / f"standardized_parcels_{state_code.lower()}.gpkg"
        write_gpkg_replace(statewide_standardized, statewide_standardized_path)
    else:
        statewide_standardized_path = state_standardized_dir / f"standardized_parcels_{state_code.lower()}.gpkg"
    if geometry_logs:
        statewide_geometry_log = pd.concat(geometry_logs, ignore_index=True)
    else:
        statewide_geometry_log = pd.DataFrame(columns=GEOMETRY_LOG_COLUMNS)
    geometry_log_path = PARCELS_METADATA_DIR / f"parcel_geometry_validation_{state_code.lower()}.csv"
    write_csv_replace(statewide_geometry_log, geometry_log_path)

    registry_path, summary_path, failures_path, coverage_path = build_outputs(state_code, registry_rows, county_rows, failure_rows)
    runtime_minutes = (time.perf_counter() - start) / 60.0
    print(f"Discovery complete in {runtime_minutes:.2f} minutes")
    print(f"Registry: {registry_path.relative_to(BASE_DIR)}")
    print(f"Summary: {summary_path.relative_to(BASE_DIR)}")
    print(f"Failures: {failures_path.relative_to(BASE_DIR)}")
    print(f"Coverage: {coverage_path.relative_to(BASE_DIR)}")
    print(f"Standardized: {statewide_standardized_path.relative_to(BASE_DIR)}")
    print(f"Geometry QA: {geometry_log_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
