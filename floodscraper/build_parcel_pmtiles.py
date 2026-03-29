from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mapbox_vector_tile
import mercantile
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests
from mapbox_vector_tile.encoder import on_invalid_geometry_make_valid
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from shapely import wkb
from shapely.geometry import LinearRing, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from state_artifacts import load_state_artifacts
from state_registry import ROOT, load_state_definition


DEFAULT_BATCH_SIZE = 2000
DEFAULT_OBJECTID_BATCH_SIZE = 250
DEFAULT_TIMEOUT_SECONDS = 60.0
FEATURE_CACHE_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "wetland_flag",
    "flood_risk_score",
    "road_access_tier",
    "source_object_id",
    "geometry_wkb",
]


@dataclass(frozen=True)
class TileBuildSettings:
    state_code: str
    state_name: str
    build_source: str
    layer: str
    min_zoom: int
    max_zoom: int
    frontend_url: str
    public_url: str
    output_path: Path
    geometry_cache_path: Path
    summary_output_path: Path
    publish_manifest_output_path: Path
    cloudflare_object_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a state-aware parcel PMTiles archive.")
    parser.add_argument("--state-code", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--geometry-cache", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--publish-manifest-output", type=Path, default=None)
    parser.add_argument("--min-zoom", type=int, default=None)
    parser.add_argument("--max-zoom", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--objectid-batch-size", type=int, default=DEFAULT_OBJECTID_BATCH_SIZE)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh-geometry-cache", action="store_true")
    return parser.parse_args()


def _safe_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if pd.isna(numeric):
        return None
    return float(numeric)


def _safe_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return bool(value)


def _safe_int(value: Any) -> int | None:
    numeric = _safe_float(value)
    return None if numeric is None else int(numeric)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_output_path(path)
    if temp_path.exists():
        temp_path.unlink()
    temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _temp_output_path(path: Path) -> Path:
    return path.parent / f"{path.name}.building"


def _parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _tile_settings(state_code: str, args: argparse.Namespace) -> TileBuildSettings:
    definition = load_state_definition(state_code)
    artifacts = load_state_artifacts(state_code)
    tile_config = definition.raw.get("parcel_tiles", {})
    return TileBuildSettings(
        state_code=definition.state_code,
        state_name=definition.state_name,
        build_source=str(tile_config.get("build_source", "parcel_master")),
        layer=str(tile_config.get("layer", "parcels")),
        min_zoom=int(args.min_zoom if args.min_zoom is not None else tile_config.get("min_zoom", 6)),
        max_zoom=int(args.max_zoom if args.max_zoom is not None else tile_config.get("max_zoom", 15)),
        frontend_url=str(tile_config.get("frontend_url", f"/tiles/{definition.state_code}_parcels.pmtiles")),
        public_url=str(tile_config.get("public_url", f"https://landintel.vercel.app/tiles/{definition.state_code}_parcels.pmtiles")),
        output_path=args.output or artifacts.frontend_parcel_pmtiles_path,
        geometry_cache_path=args.geometry_cache or ROOT / str(
            tile_config.get("geometry_cache_path", f"data/parcels/{definition.state_code}/{definition.state_code}_parcel_tile_features.parquet")
        ),
        summary_output_path=args.summary_output or ROOT / str(
            tile_config.get("summary_output", f"data/runtime/{definition.state_code}/parcel_pmtiles_build_summary.json")
        ),
        publish_manifest_output_path=args.publish_manifest_output or ROOT / str(
            tile_config.get("publish_manifest_output", f"data/runtime/{definition.state_code}/parcel_pmtiles_publish_manifest.json")
        ),
        cloudflare_object_key=str(tile_config.get("cloudflare_object_key", f"tiles/{definition.state_code}_parcels.pmtiles")),
    )


def _dataset_path(settings: TileBuildSettings) -> Path:
    artifacts = load_state_artifacts(settings.state_code)
    return artifacts.app_ready_path if settings.build_source == "app_ready" else artifacts.parcel_master_path


def _dataset_coverage_label(build_source: str) -> str:
    return "full" if str(build_source).strip().lower() == "parcel_master" else "subset"


def _source_service(state_code: str) -> dict[str, Any] | None:
    definition = load_state_definition(state_code)
    registry_path = definition.source_registry_path("parcel_source")
    if registry_path is None or not registry_path.exists():
        return None
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    for item in payload.get("parcel_sources", []):
        if item.get("primary"):
            return item
    return None


def _source_service_metadata(service_url: str, *, timeout_seconds: float) -> dict[str, Any]:
    response = requests.get(service_url.rstrip("/"), params={"f": "json"}, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"ArcGIS service metadata response was not an object: {service_url}")
    return payload


def _row_point_geometry(row: Mapping[str, Any]) -> Point | None:
    latitude = _safe_float(row.get("latitude"))
    longitude = _safe_float(row.get("longitude"))
    if latitude is None or longitude is None:
        return None
    return Point(longitude, latitude)


def _feature_object_id(feature: dict[str, Any], object_id_field: str) -> int | None:
    properties = feature.get("attributes") or feature.get("properties") or {}
    normalized = {str(key).lower(): value for key, value in properties.items()}
    return _safe_int(normalized.get(object_id_field.lower()) or normalized.get("objectid"))


def _query_arcgis_geometry_features_json(
    service_url: str,
    *,
    object_id_field: str,
    object_ids: list[int],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    response = requests.post(
        service_url.rstrip("/") + "/query",
        data={
            "objectIds": ",".join(str(value) for value in object_ids),
            "returnGeometry": "true",
            "outFields": object_id_field,
            "outSR": "4326",
            "resultType": "standard",
            "f": "json",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json().get("features", [])


def _query_arcgis_centroid_features_json(
    service_url: str,
    *,
    object_id_field: str,
    object_ids: list[int],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    response = requests.post(
        service_url.rstrip("/") + "/query",
        data={
            "objectIds": ",".join(str(value) for value in object_ids),
            "returnGeometry": "false",
            "returnCentroid": "true",
            "outFields": object_id_field,
            "outSR": "4326",
            "f": "json",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json().get("features", [])


def _dataset_has_geometry(path: Path) -> bool:
    dataset = ds.dataset(path, format="parquet")
    return "geometry" in dataset.schema.names


def _arcgis_geometry_to_shape(geometry: dict[str, Any] | None) -> BaseGeometry | None:
    if not geometry:
        return None
    if "x" in geometry and "y" in geometry:
        return Point(float(geometry["x"]), float(geometry["y"]))
    if "paths" in geometry:
        paths = geometry.get("paths") or []
        if not paths:
            return None
        lines = [LineString(path) for path in paths if len(path) >= 2]
        if not lines:
            return None
        return lines[0] if len(lines) == 1 else MultiLineString(lines)
    if "rings" in geometry:
        rings = [ring for ring in (geometry.get("rings") or []) if len(ring) >= 4]
        if not rings:
            return None
        shells: list[Polygon] = []
        holes: list[Polygon] = []
        for ring in rings:
            try:
                linear_ring = LinearRing(ring)
            except Exception:
                continue
            polygon = Polygon(linear_ring)
            if polygon.is_empty or polygon.area <= 0:
                continue
            if linear_ring.is_ccw:
                holes.append(polygon)
            else:
                shells.append(polygon)
        if not shells:
            shells = holes
            holes = []
        polygons: list[Polygon] = []
        unused_holes = holes.copy()
        for shell in shells:
            shell_holes: list[list[tuple[float, float]]] = []
            still_unused: list[Polygon] = []
            for hole in unused_holes:
                try:
                    contained = shell.contains(hole.representative_point())
                except Exception:
                    contained = False
                if contained:
                    shell_holes.append(list(hole.exterior.coords))
                else:
                    still_unused.append(hole)
            unused_holes = still_unused
            polygon = Polygon(shell.exterior.coords, shell_holes)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            polygon = _normalize_tile_geometry(polygon)
            if polygon is None:
                continue
            if polygon.geom_type == "Polygon":
                polygons.append(polygon)
            elif polygon.geom_type == "MultiPolygon":
                polygons.extend(list(polygon.geoms))
        if not polygons:
            return None
        return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    return None


def _centroid_to_shape(feature: dict[str, Any]) -> Point | None:
    centroid = feature.get("centroid")
    if not isinstance(centroid, dict):
        return None
    x = _safe_float(centroid.get("x"))
    y = _safe_float(centroid.get("y"))
    if x is None or y is None:
        return None
    return Point(x, y)


def _append_feature_cache_rows(
    cache_path: Path,
    writer: pq.ParquetWriter | None,
    rows: list[dict[str, Any]],
) -> tuple[pq.ParquetWriter | None, int, set[int]]:
    prepared = pd.DataFrame(rows, columns=FEATURE_CACHE_COLUMNS).dropna(subset=["parcel_row_id", "geometry_wkb"])
    if prepared.empty:
        return writer, 0, set()
    table_out = pa.Table.from_pandas(prepared[FEATURE_CACHE_COLUMNS], preserve_index=False)
    writer = writer or pq.ParquetWriter(cache_path, table_out.schema, compression="snappy")
    writer.write_table(table_out)
    written_ids = {
        int(value)
        for value in pd.to_numeric(prepared["source_object_id"], errors="coerce").dropna().astype(int).tolist()
    }
    return writer, int(len(prepared)), written_ids


def _parcel_id_prefix_candidates(parcel_id: str | None) -> list[str]:
    normalized = _safe_string(parcel_id)
    if not normalized:
        return []
    candidates: list[str] = []
    seen: set[str] = set()
    for index in range(len(normalized) - 1, 0, -1):
        if normalized[index] in {"-", " ", "_", "/"}:
            prefix = normalized[:index]
            if len(prefix) >= 4 and prefix not in seen:
                candidates.append(prefix)
                seen.add(prefix)
    for trim in range(1, min(8, len(normalized) - 3)):
        prefix = normalized[:-trim]
        if len(prefix) >= 4 and prefix not in seen:
            candidates.append(prefix)
            seen.add(prefix)
    return candidates


def _prefix_centroid_fallback_rows(
    *,
    missing_object_ids: list[int],
    row_lookup: dict[int, dict[str, Any]],
    cache_path: Path,
) -> list[dict[str, Any]]:
    if not missing_object_ids:
        return []
    cache_dataset = ds.dataset(cache_path, format="parquet")
    cache_frame = cache_dataset.to_table(columns=["parcel_id", "county_name", "geometry_wkb"]).to_pandas()
    if cache_frame.empty:
        return []
    cache_frame["parcel_id"] = cache_frame["parcel_id"].astype("string")
    cache_frame["county_name"] = cache_frame["county_name"].astype("string")
    centroids = cache_frame["geometry_wkb"].map(lambda value: wkb.loads(bytes(value)).centroid if value is not None else None)
    cache_frame["centroid_x"] = centroids.map(lambda value: value.x if value is not None else None)
    cache_frame["centroid_y"] = centroids.map(lambda value: value.y if value is not None else None)

    prepared_rows: list[dict[str, Any]] = []
    for object_id in missing_object_ids:
        row = row_lookup.get(object_id)
        if row is None:
            continue
        county_name = _safe_string(row.get("county_name"))
        parcel_id = _safe_string(row.get("parcel_id"))
        if not county_name or not parcel_id:
            continue
        county_matches = cache_frame.loc[cache_frame["county_name"].fillna("").str.lower() == county_name.lower()].copy()
        if county_matches.empty:
            continue
        centroid_point: Point | None = None
        for prefix in _parcel_id_prefix_candidates(parcel_id):
            matches = county_matches.loc[county_matches["parcel_id"].fillna("").str.startswith(prefix)].copy()
            if matches.empty:
                continue
            centroid_x = _safe_float(matches["centroid_x"].mean())
            centroid_y = _safe_float(matches["centroid_y"].mean())
            if centroid_x is None or centroid_y is None:
                continue
            centroid_point = Point(centroid_x, centroid_y)
            break
        if centroid_point is None:
            continue
        prepared_rows.append(
            {
                "parcel_row_id": _safe_string(row.get("parcel_row_id")),
                "parcel_id": parcel_id,
                "county_name": county_name,
                "wetland_flag": _safe_bool(row.get("wetland_flag")),
                "flood_risk_score": _safe_float(row.get("flood_risk_score")),
                "road_access_tier": _safe_string(row.get("road_access_tier")),
                "source_object_id": object_id,
                "geometry_wkb": centroid_point.wkb,
            }
        )
    return prepared_rows


def _write_feature_cache_from_local_geometry(
    dataset_path: Path,
    cache_path: Path,
    *,
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    dataset = ds.dataset(dataset_path, format="parquet")
    available = [
        column
        for column in ["parcel_row_id", "parcel_id", "county_name", "wetland_flag", "flood_risk_score", "road_access_tier", "source_object_id", "latitude", "longitude", "geometry"]
        if column in dataset.schema.names
    ]
    table = dataset.to_table(columns=available)
    if limit is not None:
        table = table.slice(0, limit)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    for batch in table.to_batches(max_chunksize=batch_size):
        frame = batch.to_pandas()
        frame = frame.dropna(subset=["parcel_row_id", "geometry"]).copy()
        if frame.empty:
            continue
        prepared = pd.DataFrame(
            {
                "parcel_row_id": frame["parcel_row_id"].map(_safe_string),
                "parcel_id": frame.get("parcel_id", pd.Series([None] * len(frame))).map(_safe_string),
                "county_name": frame.get("county_name", pd.Series([None] * len(frame))).map(_safe_string),
                "wetland_flag": frame.get("wetland_flag", pd.Series([None] * len(frame))).map(_safe_bool),
                "flood_risk_score": frame.get("flood_risk_score", pd.Series([None] * len(frame))).map(_safe_float),
                "road_access_tier": frame.get("road_access_tier", pd.Series([None] * len(frame))).map(_safe_string),
                "source_object_id": frame.get("source_object_id", pd.Series([None] * len(frame))).map(_safe_int),
                "geometry_wkb": frame.apply(
                    lambda row: bytes(row["geometry"])
                    if row.get("geometry") is not None
                    else (_row_point_geometry(row).wkb if _row_point_geometry(row) is not None else None),
                    axis=1,
                ),
            }
        ).dropna(subset=["parcel_row_id", "geometry_wkb"])
        if prepared.empty:
            continue
        writer, written_count, _ = _append_feature_cache_rows(cache_path, writer, prepared.to_dict(orient="records"))
        rows_written += written_count
    if writer is not None:
        writer.close()
    return {
        "geometry_strategy": "local_parquet_geometry",
        "geometry_cache_rows_written": rows_written,
        "geometry_fetch_batches": 0,
        "geometry_fetch_missing_count": 0,
    }


def _query_arcgis_geometries(
    service_url: str,
    *,
    object_id_field: str,
    object_ids: list[int],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    response = requests.post(
        service_url.rstrip("/") + "/query",
        data={
            "objectIds": ",".join(str(value) for value in object_ids),
            "returnGeometry": "true",
            "outFields": object_id_field,
            "outSR": "4326",
            "geometryPrecision": "6",
            "f": "geojson",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response.json().get("features", [])


def _write_feature_cache_from_live_arcgis(
    state_code: str,
    dataset_path: Path,
    cache_path: Path,
    *,
    objectid_batch_size: int,
    timeout_seconds: float,
    limit: int | None,
) -> dict[str, Any]:
    source = _source_service(state_code)
    if source is None:
        raise RuntimeError(f"No primary parcel source is configured for state: {state_code}")
    object_id_field = str(source.get("object_id_field", "objectid"))
    source_max_object_ids_per_query = _safe_int(source.get("max_object_ids_per_query"))
    dataset = ds.dataset(dataset_path, format="parquet")
    available = [
        column
        for column in ["parcel_row_id", "parcel_id", "county_name", "wetland_flag", "flood_risk_score", "road_access_tier", "source_object_id", "latitude", "longitude"]
        if column in dataset.schema.names
    ]
    frame = dataset.to_table(columns=available).to_pandas()
    if limit is not None:
        frame = frame.head(limit).copy()
    frame["source_object_id"] = pd.to_numeric(frame["source_object_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["parcel_row_id", "source_object_id"]).copy()
    frame["source_object_id"] = frame["source_object_id"].astype(int)
    frame = frame.drop_duplicates(subset=["source_object_id"], keep="first").copy()
    row_lookup = frame.set_index("source_object_id", drop=False).to_dict(orient="index")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    fetch_batches = 0
    missing_count = 0
    effective_batch_size = objectid_batch_size
    if source_max_object_ids_per_query is not None and source_max_object_ids_per_query > 0:
        effective_batch_size = min(effective_batch_size, int(source_max_object_ids_per_query))
    object_ids = frame["source_object_id"].tolist()
    for start in range(0, len(object_ids), effective_batch_size):
        batch_ids = object_ids[start : start + effective_batch_size]
        features = _query_arcgis_geometries(
            str(source["service_url"]),
            object_id_field=object_id_field,
            object_ids=batch_ids,
            timeout_seconds=timeout_seconds,
        )
        fetch_batches += 1
        found_ids: set[int] = set()
        prepared_rows: list[dict[str, Any]] = []
        for feature in features:
            properties = {str(key).lower(): value for key, value in (feature.get("properties") or {}).items()}
            object_id = _safe_int(properties.get(object_id_field.lower()) or properties.get("objectid"))
            row = row_lookup.get(object_id) if object_id is not None else None
            if row is None:
                continue
            geometry_shape = shape(feature["geometry"]) if feature.get("geometry") is not None else _row_point_geometry(row)
            if geometry_shape is None:
                continue
            if geometry_shape.is_empty:
                continue
            if not geometry_shape.is_valid:
                geometry_shape = geometry_shape.buffer(0)
            if geometry_shape.is_empty:
                continue
            found_ids.add(object_id)
            prepared_rows.append(
                {
                    "parcel_row_id": _safe_string(row.get("parcel_row_id")),
                    "parcel_id": _safe_string(row.get("parcel_id")),
                    "county_name": _safe_string(row.get("county_name")),
                    "wetland_flag": _safe_bool(row.get("wetland_flag")),
                    "flood_risk_score": _safe_float(row.get("flood_risk_score")),
                    "road_access_tier": _safe_string(row.get("road_access_tier")),
                    "source_object_id": object_id,
                    "geometry_wkb": geometry_shape.wkb,
                }
            )
        missing_count += len(batch_ids) - len(found_ids)
        writer, written_count, _ = _append_feature_cache_rows(cache_path, writer, prepared_rows)
        rows_written += written_count
        if (start // effective_batch_size) % 20 == 0:
            print(f"Fetched {min(start + len(batch_ids), len(object_ids)):,}/{len(object_ids):,} {state_code.upper()} parcel geometries")
    if writer is not None:
        writer.close()
    return {
        "geometry_strategy": "live_arcgis_geometry_cache",
        "geometry_cache_rows_written": rows_written,
        "geometry_fetch_batches": fetch_batches,
        "geometry_fetch_missing_count": missing_count,
        "geometry_fetch_objectid_batch_size": effective_batch_size,
    }


def _write_feature_cache_from_paginated_arcgis(
    state_code: str,
    dataset_path: Path,
    cache_path: Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    source = _source_service(state_code)
    if source is None:
        raise RuntimeError(f"No primary parcel source is configured for state: {state_code}")
    object_id_field = str(source.get("object_id_field", "objectid"))
    source_url = str(source["service_url"])
    service_metadata = _source_service_metadata(source_url, timeout_seconds=timeout_seconds)
    page_size = max(
        1,
        _safe_int(service_metadata.get("standardMaxRecordCount"))
        or _safe_int(service_metadata.get("maxRecordCount"))
        or 2000,
    )
    dataset = ds.dataset(dataset_path, format="parquet")
    available = [
        column
        for column in ["parcel_row_id", "parcel_id", "county_name", "wetland_flag", "flood_risk_score", "road_access_tier", "source_object_id", "latitude", "longitude"]
        if column in dataset.schema.names
    ]
    frame = dataset.to_table(columns=available).to_pandas()
    frame["source_object_id"] = pd.to_numeric(frame["source_object_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["parcel_row_id", "source_object_id"]).copy()
    frame["source_object_id"] = frame["source_object_id"].astype(int)
    frame = frame.drop_duplicates(subset=["source_object_id"], keep="first").copy()
    row_lookup = frame.set_index("source_object_id", drop=False).to_dict(orient="index")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    fetch_batches = 0
    written_object_ids: set[int] = set()
    reconciled_objectid_rows = 0
    centroid_fallback_rows = 0
    prefix_fallback_rows = 0
    session = requests.Session()
    try:
        offset = 0
        while True:
            response = session.get(
                source_url.rstrip("/") + "/query",
                params={
                    "where": "1=1",
                    "outFields": object_id_field,
                    "returnGeometry": "true",
                    "outSR": "4326",
                    "orderByFields": f"{object_id_field} ASC",
                    "resultOffset": offset,
                    "resultRecordCount": page_size,
                    "resultType": "standard",
                    "f": "json",
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            features = payload.get("features", [])
            if not features:
                break
            fetch_batches += 1
            prepared_rows: list[dict[str, Any]] = []
            for feature in features:
                object_id = _feature_object_id(feature, object_id_field)
                row = row_lookup.get(object_id) if object_id is not None else None
                if row is None:
                    continue
                geometry_shape = _arcgis_geometry_to_shape(feature.get("geometry")) or _row_point_geometry(row)
                if geometry_shape is None or geometry_shape.is_empty:
                    continue
                if not geometry_shape.is_valid:
                    geometry_shape = geometry_shape.buffer(0)
                if geometry_shape.is_empty:
                    continue
                prepared_rows.append(
                    {
                        "parcel_row_id": _safe_string(row.get("parcel_row_id")),
                        "parcel_id": _safe_string(row.get("parcel_id")),
                        "county_name": _safe_string(row.get("county_name")),
                        "wetland_flag": _safe_bool(row.get("wetland_flag")),
                        "flood_risk_score": _safe_float(row.get("flood_risk_score")),
                        "road_access_tier": _safe_string(row.get("road_access_tier")),
                        "source_object_id": object_id,
                        "geometry_wkb": geometry_shape.wkb,
                    }
                )
            writer, written_count, batch_written_ids = _append_feature_cache_rows(cache_path, writer, prepared_rows)
            rows_written += written_count
            written_object_ids.update(batch_written_ids)
            offset += page_size
            if fetch_batches % 20 == 0:
                print(f"Fetched {min(offset, len(row_lookup)):,}/{len(row_lookup):,} {state_code.upper()} parcel geometries")
            if not payload.get("exceededTransferLimit") and len(features) < page_size:
                break

        missing_object_ids = sorted(set(row_lookup.keys()) - written_object_ids)
        if missing_object_ids:
            for start in range(0, len(missing_object_ids), page_size):
                batch_ids = missing_object_ids[start : start + page_size]
                features = _query_arcgis_geometry_features_json(
                    source_url,
                    object_id_field=object_id_field,
                    object_ids=batch_ids,
                    timeout_seconds=timeout_seconds,
                )
                prepared_rows = []
                for feature in features:
                    object_id = _feature_object_id(feature, object_id_field)
                    row = row_lookup.get(object_id) if object_id is not None else None
                    if row is None:
                        continue
                    geometry_shape = _arcgis_geometry_to_shape(feature.get("geometry")) or _row_point_geometry(row)
                    if geometry_shape is None or geometry_shape.is_empty:
                        continue
                    if not geometry_shape.is_valid:
                        geometry_shape = geometry_shape.buffer(0)
                    if geometry_shape.is_empty:
                        continue
                    prepared_rows.append(
                        {
                            "parcel_row_id": _safe_string(row.get("parcel_row_id")),
                            "parcel_id": _safe_string(row.get("parcel_id")),
                            "county_name": _safe_string(row.get("county_name")),
                            "wetland_flag": _safe_bool(row.get("wetland_flag")),
                            "flood_risk_score": _safe_float(row.get("flood_risk_score")),
                            "road_access_tier": _safe_string(row.get("road_access_tier")),
                            "source_object_id": object_id,
                            "geometry_wkb": geometry_shape.wkb,
                        }
                    )
                writer, written_count, batch_written_ids = _append_feature_cache_rows(cache_path, writer, prepared_rows)
                rows_written += written_count
                reconciled_objectid_rows += written_count
                written_object_ids.update(batch_written_ids)

        missing_object_ids = sorted(set(row_lookup.keys()) - written_object_ids)
        if missing_object_ids:
            for start in range(0, len(missing_object_ids), page_size):
                batch_ids = missing_object_ids[start : start + page_size]
                features = _query_arcgis_centroid_features_json(
                    source_url,
                    object_id_field=object_id_field,
                    object_ids=batch_ids,
                    timeout_seconds=timeout_seconds,
                )
                prepared_rows = []
                for feature in features:
                    object_id = _feature_object_id(feature, object_id_field)
                    row = row_lookup.get(object_id) if object_id is not None else None
                    if row is None:
                        continue
                    geometry_shape = _centroid_to_shape(feature) or _row_point_geometry(row)
                    if geometry_shape is None or geometry_shape.is_empty:
                        continue
                    prepared_rows.append(
                        {
                            "parcel_row_id": _safe_string(row.get("parcel_row_id")),
                            "parcel_id": _safe_string(row.get("parcel_id")),
                            "county_name": _safe_string(row.get("county_name")),
                            "wetland_flag": _safe_bool(row.get("wetland_flag")),
                            "flood_risk_score": _safe_float(row.get("flood_risk_score")),
                            "road_access_tier": _safe_string(row.get("road_access_tier")),
                            "source_object_id": object_id,
                            "geometry_wkb": geometry_shape.wkb,
                        }
                    )
                writer, written_count, batch_written_ids = _append_feature_cache_rows(cache_path, writer, prepared_rows)
                rows_written += written_count
                centroid_fallback_rows += written_count
                written_object_ids.update(batch_written_ids)

    finally:
        session.close()
        if writer is not None:
            writer.close()
    remaining_missing_object_ids = sorted(set(row_lookup.keys()) - written_object_ids)
    if remaining_missing_object_ids and len(remaining_missing_object_ids) <= 500 and cache_path.exists():
        prepared_rows = _prefix_centroid_fallback_rows(
            missing_object_ids=remaining_missing_object_ids,
            row_lookup=row_lookup,
            cache_path=cache_path,
        )
        if prepared_rows:
            existing_table = ds.dataset(cache_path, format="parquet").to_table(columns=FEATURE_CACHE_COLUMNS)
            fallback_table = pa.Table.from_pandas(
                pd.DataFrame(prepared_rows, columns=FEATURE_CACHE_COLUMNS).dropna(subset=["parcel_row_id", "geometry_wkb"]),
                preserve_index=False,
            )
            if fallback_table.num_rows > 0:
                pq.write_table(pa.concat_tables([existing_table, fallback_table]), cache_path, compression="snappy")
                written_object_ids.update(
                    int(value)
                    for value in pd.to_numeric(fallback_table.column("source_object_id").to_pandas(), errors="coerce").dropna().astype(int).tolist()
                )
                rows_written += int(fallback_table.num_rows)
                prefix_fallback_rows += int(fallback_table.num_rows)
        remaining_missing_object_ids = sorted(set(row_lookup.keys()) - written_object_ids)
    blocker_reason = None
    if remaining_missing_object_ids:
        blocker_reason = (
            f"{len(remaining_missing_object_ids):,} parcel rows did not return polygon geometry, source centroid, or usable fallback geometry."
        )
    return {
        "geometry_strategy": "paged_arcgis_geometry_cache",
        "geometry_cache_rows_written": rows_written,
        "geometry_fetch_batches": fetch_batches,
        "geometry_fetch_missing_count": max(0, int(len(row_lookup) - rows_written)),
        "geometry_fetch_page_size": page_size,
        "geometry_objectid_reconciliation_rows_written": reconciled_objectid_rows,
        "geometry_centroid_fallback_rows_written": centroid_fallback_rows,
        "geometry_prefix_fallback_rows_written": prefix_fallback_rows,
        "geometry_remaining_missing_count": len(remaining_missing_object_ids),
        "geometry_missing_object_id_samples": remaining_missing_object_ids[:10],
        "geometry_blocker_reason": blocker_reason,
    }


def ensure_feature_cache(settings: TileBuildSettings, args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = _dataset_path(settings)
    source_dataset_row_count = _parquet_row_count(dataset_path)
    if settings.geometry_cache_path.exists() and not args.refresh_geometry_cache:
        return {
            "geometry_strategy": "cached_feature_cache",
            "geometry_cache_rows_written": _parquet_row_count(settings.geometry_cache_path),
            "geometry_fetch_batches": 0,
            "geometry_fetch_missing_count": 0,
            "source_dataset_path": str(dataset_path),
            "source_dataset_row_count": source_dataset_row_count,
        }
    temp_cache_path = _temp_output_path(settings.geometry_cache_path)
    if temp_cache_path.exists():
        temp_cache_path.unlink()
    if _dataset_has_geometry(dataset_path):
        result = _write_feature_cache_from_local_geometry(dataset_path, temp_cache_path, batch_size=args.batch_size, limit=args.limit)
    elif settings.build_source == "parcel_master" and args.limit is None:
        result = _write_feature_cache_from_paginated_arcgis(
            settings.state_code,
            dataset_path,
            temp_cache_path,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        result = _write_feature_cache_from_live_arcgis(
            settings.state_code,
            dataset_path,
            temp_cache_path,
            objectid_batch_size=args.objectid_batch_size,
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
        )
    if settings.geometry_cache_path.exists():
        settings.geometry_cache_path.unlink()
    temp_cache_path.replace(settings.geometry_cache_path)
    result["source_dataset_path"] = str(dataset_path)
    result["source_dataset_row_count"] = source_dataset_row_count
    return result


def _normalize_tile_geometry(geometry_shape: BaseGeometry | None) -> BaseGeometry | None:
    if geometry_shape is None or geometry_shape.is_empty:
        return None
    if geometry_shape.geom_type in {"Polygon", "MultiPolygon", "Point", "MultiPoint"}:
        return geometry_shape
    if geometry_shape.geom_type == "GeometryCollection":
        polygons = [part for part in geometry_shape.geoms if not part.is_empty and part.geom_type in {"Polygon", "MultiPolygon"}]
        points = [part for part in geometry_shape.geoms if not part.is_empty and part.geom_type in {"Point", "MultiPoint"}]
        if polygons and not points:
            return polygons[0] if len(polygons) == 1 else unary_union(polygons)
        if points and not polygons:
            point_geoms: list[Point] = []
            for part in points:
                if part.geom_type == "Point":
                    point_geoms.append(part)
                else:
                    point_geoms.extend(list(part.geoms))
            if not point_geoms:
                return None
            return point_geoms[0] if len(point_geoms) == 1 else MultiPoint(point_geoms)
        if not polygons and not points:
            return None
        return polygons[0] if len(polygons) == 1 else unary_union(polygons)
    repaired = geometry_shape.buffer(0)
    return repaired if not repaired.is_empty and repaired.geom_type in {"Polygon", "MultiPolygon"} else None


def _simplify_tolerance(tile_bounds: tuple[float, float, float, float], zoom: int) -> float:
    width = max(tile_bounds[2] - tile_bounds[0], tile_bounds[3] - tile_bounds[1])
    if zoom <= 6:
        return width / 96.0
    if zoom <= 8:
        return width / 192.0
    if zoom <= 10:
        return width / 384.0
    if zoom <= 12:
        return width / 768.0
    return 0.0


def build_pmtiles_from_feature_cache(
    settings: TileBuildSettings,
    *,
    batch_size: int,
) -> dict[str, Any]:
    dataset = ds.dataset(settings.geometry_cache_path, format="parquet")
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    global_bounds: list[float] | None = None
    tile_counts_by_zoom: dict[str, int] = {}
    encoded_feature_hits_by_zoom: dict[str, int] = {}
    started = time.perf_counter()
    temp_output_path = _temp_output_path(settings.output_path)
    if temp_output_path.exists():
        temp_output_path.unlink()
    with temp_output_path.open("wb") as handle:
        writer = Writer(handle)
        for zoom in range(settings.min_zoom, settings.max_zoom + 1):
            tile_buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
            scanner = dataset.scanner(columns=FEATURE_CACHE_COLUMNS, batch_size=batch_size)
            for batch in scanner.to_batches():
                frame = batch.to_pandas()
                for _, row in frame.iterrows():
                    geometry_shape = _normalize_tile_geometry(wkb.loads(bytes(row["geometry_wkb"])))
                    if geometry_shape is None:
                        continue
                    minx, miny, maxx, maxy = geometry_shape.bounds
                    if global_bounds is None:
                        global_bounds = [minx, miny, maxx, maxy]
                    else:
                        global_bounds[0] = min(global_bounds[0], minx)
                        global_bounds[1] = min(global_bounds[1], miny)
                        global_bounds[2] = max(global_bounds[2], maxx)
                        global_bounds[3] = max(global_bounds[3], maxy)
                    for tile in mercantile.tiles(minx, miny, maxx, maxy, zooms=[zoom]):
                        bounds = mercantile.bounds(tile)
                        tile_bounds = (bounds.west, bounds.south, bounds.east, bounds.north)
                        clipped = _normalize_tile_geometry(geometry_shape.intersection(box(*tile_bounds)))
                        if clipped is None:
                            continue
                        tolerance = _simplify_tolerance(tile_bounds, zoom)
                        if tolerance > 0 and clipped.geom_type in {"Polygon", "MultiPolygon"}:
                            clipped = _normalize_tile_geometry(clipped.simplify(tolerance, preserve_topology=True))
                        if clipped is None:
                            continue
                        tile_buckets.setdefault((tile.x, tile.y), []).append(
                            {
                                "geometry": clipped,
                                "properties": {
                                    "parcel_row_id": _safe_string(row.get("parcel_row_id")),
                                    "parcel_id": _safe_string(row.get("parcel_id")),
                                    "county_name": _safe_string(row.get("county_name")),
                                    "wetland_flag": _safe_bool(row.get("wetland_flag")),
                                    "flood_risk_score": _safe_float(row.get("flood_risk_score")),
                                    "road_access_tier": _safe_string(row.get("road_access_tier")),
                                },
                            }
                        )
            tile_ids = sorted(((zxy_to_tileid(zoom, x, y), x, y) for x, y in tile_buckets.keys()), key=lambda item: item[0])
            tile_counts_by_zoom[str(zoom)] = len(tile_ids)
            encoded_feature_hits_by_zoom[str(zoom)] = int(sum(len(tile_buckets[(x, y)]) for _, x, y in tile_ids))
            for tile_id, x, y in tile_ids:
                bounds = mercantile.bounds(x, y, zoom)
                quantize_bounds = (bounds.west, bounds.south, bounds.east, bounds.north)
                encoded = mapbox_vector_tile.encode(
                    {"name": settings.layer, "features": tile_buckets[(x, y)]},
                    default_options={
                        "quantize_bounds": quantize_bounds,
                        "extents": 4096,
                        "on_invalid_geometry": on_invalid_geometry_make_valid,
                    },
                )
                if encoded:
                    writer.write_tile(tile_id, gzip.compress(encoded))
            print(f"Built {settings.state_name} parcel tiles for zoom {zoom}: {tile_counts_by_zoom[str(zoom)]} tiles")

        if global_bounds is None:
            raise RuntimeError(f"No parcel geometry was available to build {settings.state_code} PMTiles.")
        header = {
            "version": 3,
            "tile_compression": Compression.GZIP,
            "tile_type": TileType.MVT,
            "min_lon_e7": int(round(global_bounds[0] * 10_000_000)),
            "min_lat_e7": int(round(global_bounds[1] * 10_000_000)),
            "max_lon_e7": int(round(global_bounds[2] * 10_000_000)),
            "max_lat_e7": int(round(global_bounds[3] * 10_000_000)),
            "center_zoom": settings.min_zoom,
            "center_lon_e7": int(round(((global_bounds[0] + global_bounds[2]) / 2.0) * 10_000_000)),
            "center_lat_e7": int(round(((global_bounds[1] + global_bounds[3]) / 2.0) * 10_000_000)),
        }
        writer.finalize(
            header,
            {
                "name": f"{settings.state_name} parcel overlay",
                "description": f"{settings.state_name} parcel overlay archive for LandIntel.",
                "type": "overlay",
                "format": "pbf",
                "state_code": settings.state_code,
                "frontend_url": settings.frontend_url,
                "public_url": settings.public_url,
                "vector_layers": [
                    {
                        "id": settings.layer,
                        "fields": {
                            "parcel_row_id": "String",
                            "parcel_id": "String",
                            "county_name": "String",
                            "wetland_flag": "Boolean",
                            "flood_risk_score": "Number",
                            "road_access_tier": "String",
                        },
                    }
                ],
            },
        )
    temp_output_path.replace(settings.output_path)

    return {
        "tile_build_method": "python_pmtiles_writer",
        "tile_count": int(sum(tile_counts_by_zoom.values())),
        "tile_counts_by_zoom": tile_counts_by_zoom,
        "encoded_feature_hits_by_zoom": encoded_feature_hits_by_zoom,
        "artifact_size_bytes": settings.output_path.stat().st_size if settings.output_path.exists() else 0,
        "build_elapsed_seconds": round(time.perf_counter() - started, 3),
        "global_bounds": [round(value, 6) for value in global_bounds],
    }


def _publish_manifest(settings: TileBuildSettings, build_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_code": settings.state_code,
        "state_name": settings.state_name,
        "artifact_path": str(settings.output_path),
        "artifact_size_bytes": settings.output_path.stat().st_size if settings.output_path.exists() else 0,
        "frontend_url": settings.frontend_url,
        "public_url": settings.public_url,
        "cloudflare_object_key": settings.cloudflare_object_key,
        "content_type": "application/vnd.pmtiles",
        "cache_control": "public, max-age=31536000, immutable",
        "build_summary_path": str(settings.summary_output_path),
        "build_method": build_summary.get("tile_build_method"),
        "statewide_parcel_tile_coverage": build_summary.get("statewide_parcel_tile_coverage"),
        "map_shows_all_parcels": build_summary.get("map_shows_all_parcels"),
        "public_url_status": "configured" if settings.public_url else "missing",
        "notes": [
            "Upload this PMTiles artifact to the configured public URL contract using the object key above.",
            "The local frontend URL is a development fallback and should not be treated as the production hosting contract.",
            "The parcel overlay must be interpreted separately from the app_ready/default-lead business layer.",
        ],
    }


def main() -> None:
    args = parse_args()
    settings = _tile_settings(args.state_code, args)
    artifacts = load_state_artifacts(settings.state_code)
    build_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stage_runtimes: dict[str, float] = {}

    stage_start = time.perf_counter()
    cache_summary = ensure_feature_cache(settings, args)
    stage_runtimes["geometry_cache_seconds"] = round(time.perf_counter() - stage_start, 3)

    stage_start = time.perf_counter()
    tile_summary = build_pmtiles_from_feature_cache(settings, batch_size=args.batch_size)
    stage_runtimes["pmtiles_build_seconds"] = round(time.perf_counter() - stage_start, 3)

    build_summary = {
        "state_code": settings.state_code,
        "state_name": settings.state_name,
        "build_started_timestamp": build_started,
        "build_finished_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_source": settings.build_source,
        "statewide_parcel_tile_coverage": _dataset_coverage_label(settings.build_source),
        "map_shows_all_parcels": False,
        "layer": settings.layer,
        "min_zoom": settings.min_zoom,
        "max_zoom": settings.max_zoom,
        "output_path": str(settings.output_path),
        "geometry_cache_path": str(settings.geometry_cache_path),
        "frontend_url": settings.frontend_url,
        "public_url": settings.public_url,
        "cloudflare_object_key": settings.cloudflare_object_key,
        "statewide_parcel_row_count": _parquet_row_count(artifacts.parcel_master_path) if artifacts.parcel_master_path.exists() else None,
        "stage_runtimes_seconds": stage_runtimes,
        **cache_summary,
        **tile_summary,
    }
    geometry_cache_rows = int(build_summary.get("geometry_cache_rows_written") or 0)
    statewide_parcel_rows = int(build_summary.get("statewide_parcel_row_count") or 0)
    build_summary["statewide_geometry_coverage"] = (
        "full"
        if build_summary.get("statewide_parcel_tile_coverage") == "full" and geometry_cache_rows >= statewide_parcel_rows and statewide_parcel_rows > 0
        else "subset"
    )
    build_summary["map_shows_all_parcels"] = (
        build_summary.get("statewide_parcel_tile_coverage") == "full"
        and build_summary.get("statewide_geometry_coverage") == "full"
    )
    _write_json(settings.summary_output_path, build_summary)
    _write_json(settings.publish_manifest_output_path, _publish_manifest(settings, build_summary))
    print(json.dumps(build_summary, indent=2))


if __name__ == "__main__":
    main()
