from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from shapely.geometry import box, mapping
from shapely import wkb

from app.settings import GEOMETRY_DEFAULT_LIMIT, GEOMETRY_MAX_LIMIT, LEADS_DEFAULT_LIMIT, LEADS_MAX_LIMIT


def _discover_project_root() -> Path:
    explicit_root = os.getenv("MISSISSIPPI_EXPLORER_DATA_ROOT")
    if explicit_root:
        return Path(explicit_root).expanduser().resolve(strict=False)

    cwd = Path.cwd().resolve(strict=False)
    service_repo_root = Path(__file__).resolve().parents[3]
    candidates = [cwd, cwd.parent, service_repo_root]
    for candidate in candidates:
        if (candidate / "frontend" / "public" / "data").exists():
            return candidate
    return cwd


PROJECT_ROOT = _discover_project_root()
DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
APP_READY_PATH = PROJECT_ROOT / "data" / "tax_published" / "ms" / "app_ready_mississippi_leads.parquet"
STATIC_FEED_PATH = DATA_DIR / "mississippi_lead_explorer.json"
META_PATH = DATA_DIR / "mississippi_lead_explorer_meta.json"
GEOMETRY_PATH = DATA_DIR / "mississippi_lead_explorer_geometries.json"

BOOL_FILTER_FIELDS = {
    "parcel_vacant_flag",
    "county_hosted_flag",
    "high_confidence_link_flag",
    "corporate_owner_flag",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
}
LIST_FILTER_FIELDS = {
    "lead_score_tier",
    "amount_trust_tier",
    "growth_pressure_bucket",
    "recommended_view_bucket",
    "road_access_tier",
}

SUMMARY_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "acreage",
    "owner_name",
    "lead_score_total",
    "lead_score_tier",
    "parcel_vacant_flag",
    "road_access_tier",
    "growth_pressure_bucket",
    "best_source_type",
    "source_confidence_tier",
    "delinquent_amount",
    "amount_trust_tier",
    "recommended_sort_reason",
    "county_hosted_flag",
    "high_confidence_link_flag",
    "recommended_view_bucket",
]


def _to_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _serialize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _geometry_payload(value: bytes | None) -> dict[str, Any] | None:
    if not value:
        return None
    geometry = wkb.loads(value)
    centroid = geometry.centroid
    bounds = geometry.bounds
    return {
        "type": geometry.geom_type,
        "centroid": {"type": "Point", "coordinates": [round(float(centroid.x), 6), round(float(centroid.y), 6)]},
        "bounds": [round(float(v), 6) for v in bounds],
    }


def _serialize_bounds(bounds: tuple[float, float, float, float] | None) -> list[float] | None:
    if bounds is None:
        return None
    return [round(float(value), 6) for value in bounds]


def runtime_file_diagnostics() -> dict[str, dict[str, int | bool | str | None]]:
    diagnostics: dict[str, dict[str, int | bool | str | None]] = {}
    paths = {
        "app_ready_parquet": APP_READY_PATH,
        "static_feed_json": STATIC_FEED_PATH,
        "meta_json": META_PATH,
        "geometry_json": GEOMETRY_PATH,
    }
    for name, path in paths.items():
        diagnostics[name] = {
            "cwd": str(Path.cwd()),
            "project_root": str(PROJECT_ROOT),
            "path": str(path.resolve(strict=False)),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return diagnostics


def _missing_runtime_file_error(label: str, path_key: str) -> FileNotFoundError:
    diagnostics = runtime_file_diagnostics().get(path_key, {})
    return FileNotFoundError(
        f"{label} not found: {diagnostics.get('path')} | cwd={diagnostics.get('cwd')} | "
        f"project_root={diagnostics.get('project_root')}"
    )


@lru_cache(maxsize=1)
def load_app_ready_frame() -> pd.DataFrame:
    if not APP_READY_PATH.exists():
        raise _missing_runtime_file_error("Mississippi leads dataset", "app_ready_parquet")
    frame = pd.read_parquet(APP_READY_PATH)
    return frame


def _attach_spatial_columns(frame: pd.DataFrame) -> pd.DataFrame:
    spatial = frame.copy()
    spatial["_shape"] = spatial["geometry"].apply(lambda value: wkb.loads(value) if value else None)
    spatial["_minx"] = spatial["_shape"].apply(lambda geom: geom.bounds[0] if geom else None)
    spatial["_miny"] = spatial["_shape"].apply(lambda geom: geom.bounds[1] if geom else None)
    spatial["_maxx"] = spatial["_shape"].apply(lambda geom: geom.bounds[2] if geom else None)
    spatial["_maxy"] = spatial["_shape"].apply(lambda geom: geom.bounds[3] if geom else None)
    spatial["_centroid_x"] = spatial["_shape"].apply(lambda geom: geom.centroid.x if geom else None)
    spatial["_centroid_y"] = spatial["_shape"].apply(lambda geom: geom.centroid.y if geom else None)
    return spatial


@lru_cache(maxsize=1)
def load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        raise _missing_runtime_file_error("Mississippi explorer meta file", "meta_json")
    with META_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_geometry_lookup() -> dict[str, str]:
    if not GEOMETRY_PATH.exists():
        raise _missing_runtime_file_error("Mississippi geometry file", "geometry_json")
    with GEOMETRY_PATH.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {str(row["parcel_row_id"]): row.get("path") or "" for row in rows}


def _clamp_limit(requested_limit: int | None, *, default: int, max_limit: int) -> int:
    if requested_limit is None:
        return default
    return max(1, min(int(requested_limit), max_limit))


def _apply_filters(
    frame: pd.DataFrame,
    *,
    county_name: str | None = None,
    lead_score_tier: list[str] | None = None,
    min_lead_score_total: float | None = None,
    acreage_min: float | None = None,
    acreage_max: float | None = None,
    parcel_vacant_flag: bool | None = None,
    county_hosted_flag: bool | None = None,
    high_confidence_link_flag: bool | None = None,
    wetland_flag: bool | None = None,
    amount_trust_tier: list[str] | None = None,
    corporate_owner_flag: bool | None = None,
    absentee_owner_flag: bool | None = None,
    out_of_state_owner_flag: bool | None = None,
    growth_pressure_bucket: list[str] | None = None,
    recommended_view_bucket: list[str] | None = None,
    road_access_tier: list[str] | None = None,
    road_distance_ft_max: float | None = None,
) -> pd.DataFrame:
    filtered = frame
    if county_name:
        filtered = filtered.loc[_normalize_string(filtered["county_name"]).eq(county_name)].copy()
    if lead_score_tier:
        filtered = filtered.loc[_normalize_string(filtered["lead_score_tier"]).isin(lead_score_tier)].copy()
    if min_lead_score_total is not None:
        filtered = filtered.loc[pd.to_numeric(filtered["lead_score_total"], errors="coerce").ge(min_lead_score_total).fillna(False)].copy()
    if acreage_min is not None:
        filtered = filtered.loc[pd.to_numeric(filtered["acreage"], errors="coerce").ge(acreage_min).fillna(False)].copy()
    if acreage_max is not None:
        filtered = filtered.loc[pd.to_numeric(filtered["acreage"], errors="coerce").le(acreage_max).fillna(False)].copy()
    if parcel_vacant_flag is not None:
        filtered = filtered.loc[filtered["parcel_vacant_flag"].fillna(False).eq(parcel_vacant_flag)].copy()
    if county_hosted_flag is not None:
        filtered = filtered.loc[filtered["county_hosted_flag"].fillna(False).eq(county_hosted_flag)].copy()
    if high_confidence_link_flag is not None:
        filtered = filtered.loc[filtered["high_confidence_link_flag"].fillna(False).eq(high_confidence_link_flag)].copy()
    if wetland_flag is not None:
        filtered = filtered.loc[filtered["wetland_flag"].fillna(False).eq(wetland_flag)].copy()
    if amount_trust_tier:
        filtered = filtered.loc[_normalize_string(filtered["amount_trust_tier"]).isin(amount_trust_tier)].copy()
    if corporate_owner_flag is not None:
        filtered = filtered.loc[filtered["corporate_owner_flag"].fillna(False).eq(corporate_owner_flag)].copy()
    if absentee_owner_flag is not None:
        filtered = filtered.loc[filtered["absentee_owner_flag"].fillna(False).eq(absentee_owner_flag)].copy()
    if out_of_state_owner_flag is not None:
        filtered = filtered.loc[filtered["out_of_state_owner_flag"].fillna(False).eq(out_of_state_owner_flag)].copy()
    if growth_pressure_bucket:
        filtered = filtered.loc[_normalize_string(filtered["growth_pressure_bucket"]).isin(growth_pressure_bucket)].copy()
    if recommended_view_bucket:
        filtered = filtered.loc[_normalize_string(filtered["recommended_view_bucket"]).isin(recommended_view_bucket)].copy()
    if road_access_tier:
        filtered = filtered.loc[_normalize_string(filtered["road_access_tier"]).isin(road_access_tier)].copy()
    if road_distance_ft_max is not None:
        filtered = filtered.loc[pd.to_numeric(filtered["road_distance_ft"], errors="coerce").le(road_distance_ft_max).fillna(False)].copy()
    return filtered


def _apply_bounds_filter(
    frame: pd.DataFrame,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    selected_parcel_id: str | None = None,
) -> pd.DataFrame:
    if bounds is None:
        return frame

    min_lng, min_lat, max_lng, max_lat = bounds
    bbox_mask = (
        pd.to_numeric(frame["_maxx"], errors="coerce").ge(min_lng).fillna(False)
        & pd.to_numeric(frame["_minx"], errors="coerce").le(max_lng).fillna(False)
        & pd.to_numeric(frame["_maxy"], errors="coerce").ge(min_lat).fillna(False)
        & pd.to_numeric(frame["_miny"], errors="coerce").le(max_lat).fillna(False)
    )
    if selected_parcel_id:
        bbox_mask = bbox_mask | frame["parcel_row_id"].astype("string").eq(selected_parcel_id)
    return frame.loc[bbox_mask].copy()


def _feature_bounds(frame: pd.DataFrame) -> tuple[float, float, float, float] | None:
    if frame.empty:
        return None
    minx = pd.to_numeric(frame["_minx"], errors="coerce").min()
    miny = pd.to_numeric(frame["_miny"], errors="coerce").min()
    maxx = pd.to_numeric(frame["_maxx"], errors="coerce").max()
    maxy = pd.to_numeric(frame["_maxy"], errors="coerce").max()
    if not all(pd.notna(value) for value in [minx, miny, maxx, maxy]):
        return None
    return (float(minx), float(miny), float(maxx), float(maxy))


def _render_mode_for_zoom(zoom: float | None) -> str:
    if zoom is None:
        return "polygons"
    if zoom < 8:
        return "points"
    if zoom < 11:
        return "centroids"
    return "polygons"


def _simplification_tolerance(zoom: float | None) -> float:
    if zoom is None:
        return 0.0
    if zoom >= 14:
        return 0.0
    if zoom >= 12:
        return 0.00002
    if zoom >= 10:
        return 0.00008
    return 0.0003


def _geometry_feature(row: pd.Series, *, render_mode: str, zoom: float | None, selected_parcel_id: str | None) -> dict[str, Any] | None:
    shape = row.get("_shape")
    if shape is None:
        return None

    if render_mode == "points":
        geometry = {
            "type": "Point",
            "coordinates": [round(float(row["_centroid_x"]), 6), round(float(row["_centroid_y"]), 6)],
        }
    elif render_mode == "centroids":
        geometry = {
            "type": "Point",
            "coordinates": [round(float(row["_centroid_x"]), 6), round(float(row["_centroid_y"]), 6)],
        }
    else:
        tolerance = _simplification_tolerance(zoom)
        geometry_shape = shape.simplify(tolerance, preserve_topology=True) if tolerance > 0 else shape
        geometry = mapping(geometry_shape)

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "parcel_row_id": str(row["parcel_row_id"]),
            "parcel_id": _serialize_scalar(row.get("parcel_id")),
            "county_name": _serialize_scalar(row.get("county_name")),
            "lead_score_total": _serialize_scalar(row.get("lead_score_total")),
            "lead_score_tier": _serialize_scalar(row.get("lead_score_tier")),
            "parcel_vacant_flag": _serialize_scalar(row.get("parcel_vacant_flag")),
            "wetland_flag": _serialize_scalar(row.get("wetland_flag")),
            "flood_risk_score": _serialize_scalar(row.get("flood_risk_score")),
            "road_access_tier": _serialize_scalar(row.get("road_access_tier")),
            "county_hosted_flag": _serialize_scalar(row.get("county_hosted_flag")),
            "best_source_type": _serialize_scalar(row.get("best_source_type")),
            "selected": str(row["parcel_row_id"]) == (selected_parcel_id or ""),
        },
    }


def get_leads(
    *,
    county_name: str | None = None,
    lead_score_tier: list[str] | None = None,
    min_lead_score_total: float | None = None,
    acreage_min: float | None = None,
    acreage_max: float | None = None,
    parcel_vacant_flag: bool | None = None,
    county_hosted_flag: bool | None = None,
    high_confidence_link_flag: bool | None = None,
    wetland_flag: bool | None = None,
    amount_trust_tier: list[str] | None = None,
    corporate_owner_flag: bool | None = None,
    absentee_owner_flag: bool | None = None,
    out_of_state_owner_flag: bool | None = None,
    growth_pressure_bucket: list[str] | None = None,
    recommended_view_bucket: list[str] | None = None,
    road_access_tier: list[str] | None = None,
    road_distance_ft_max: float | None = None,
    sort_by: str = "lead_score_total",
    sort_direction: str = "desc",
    limit: int = LEADS_DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    frame = load_app_ready_frame()
    filtered = _apply_filters(
        frame,
        county_name=county_name,
        lead_score_tier=lead_score_tier,
        min_lead_score_total=min_lead_score_total,
        acreage_min=acreage_min,
        acreage_max=acreage_max,
        parcel_vacant_flag=parcel_vacant_flag,
        county_hosted_flag=county_hosted_flag,
        high_confidence_link_flag=high_confidence_link_flag,
        wetland_flag=wetland_flag,
        amount_trust_tier=amount_trust_tier,
        corporate_owner_flag=corporate_owner_flag,
        absentee_owner_flag=absentee_owner_flag,
        out_of_state_owner_flag=out_of_state_owner_flag,
        growth_pressure_bucket=growth_pressure_bucket,
        recommended_view_bucket=recommended_view_bucket,
        road_access_tier=road_access_tier,
        road_distance_ft_max=road_distance_ft_max,
    )
    total_count = len(filtered)

    if sort_by not in filtered.columns:
        sort_by = "lead_score_total"
    ascending = sort_direction.lower() == "asc"
    filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")
    safe_limit = _clamp_limit(limit, default=LEADS_DEFAULT_LIMIT, max_limit=LEADS_MAX_LIMIT)
    safe_offset = max(int(offset), 0)
    paged = filtered.iloc[safe_offset : safe_offset + safe_limit].copy()

    items: list[dict[str, Any]] = []
    for _, row in paged.loc[:, SUMMARY_FIELDS].iterrows():
        items.append({column: _serialize_scalar(row[column]) for column in SUMMARY_FIELDS})
    return {"total_count": total_count, "limit": safe_limit, "offset": safe_offset, "items": items}


def get_lead_detail(parcel_row_id: str) -> dict[str, Any] | None:
    frame = load_app_ready_frame()
    match = frame.loc[frame["parcel_row_id"].astype("string").eq(parcel_row_id)]
    if match.empty:
        return None
    row = match.iloc[0]
    payload = {column: _serialize_scalar(row[column]) for column in frame.columns if column != "geometry"}
    payload["geometry"] = _geometry_payload(row.get("geometry"))
    return payload


def get_geometry(
    *,
    parcel_row_ids: list[str] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    zoom: float | None = None,
    selected_parcel_id: str | None = None,
    county_name: str | None = None,
    lead_score_tier: list[str] | None = None,
    min_lead_score_total: float | None = None,
    acreage_min: float | None = None,
    acreage_max: float | None = None,
    parcel_vacant_flag: bool | None = None,
    county_hosted_flag: bool | None = None,
    high_confidence_link_flag: bool | None = None,
    wetland_flag: bool | None = None,
    amount_trust_tier: list[str] | None = None,
    corporate_owner_flag: bool | None = None,
    absentee_owner_flag: bool | None = None,
    out_of_state_owner_flag: bool | None = None,
    growth_pressure_bucket: list[str] | None = None,
    recommended_view_bucket: list[str] | None = None,
    road_access_tier: list[str] | None = None,
    road_distance_ft_max: float | None = None,
    limit: int = GEOMETRY_DEFAULT_LIMIT,
) -> dict[str, Any]:
    base_frame = load_app_ready_frame()
    if parcel_row_ids:
        filtered = base_frame.loc[base_frame["parcel_row_id"].astype("string").isin(parcel_row_ids)].copy()
        filtered = _attach_spatial_columns(filtered)
    else:
        filtered = _apply_filters(
            base_frame,
            county_name=county_name,
            lead_score_tier=lead_score_tier,
            min_lead_score_total=min_lead_score_total,
            acreage_min=acreage_min,
            acreage_max=acreage_max,
            parcel_vacant_flag=parcel_vacant_flag,
            county_hosted_flag=county_hosted_flag,
            high_confidence_link_flag=high_confidence_link_flag,
            wetland_flag=wetland_flag,
            amount_trust_tier=amount_trust_tier,
            corporate_owner_flag=corporate_owner_flag,
            absentee_owner_flag=absentee_owner_flag,
            out_of_state_owner_flag=out_of_state_owner_flag,
            growth_pressure_bucket=growth_pressure_bucket,
            recommended_view_bucket=recommended_view_bucket,
            road_access_tier=road_access_tier,
            road_distance_ft_max=road_distance_ft_max,
        )
        safe_limit = _clamp_limit(limit, default=GEOMETRY_DEFAULT_LIMIT, max_limit=GEOMETRY_MAX_LIMIT)
        filtered = filtered.sort_values("lead_score_total", ascending=False, na_position="last").head(safe_limit)
        filtered = _attach_spatial_columns(filtered)
        filtered = _apply_bounds_filter(filtered, bounds=bounds, selected_parcel_id=selected_parcel_id)

    render_mode = _render_mode_for_zoom(zoom)
    feature_bounds = _feature_bounds(filtered)
    features = [
        feature
        for _, row in filtered.iterrows()
        if (feature := _geometry_feature(row, render_mode=render_mode, zoom=zoom, selected_parcel_id=selected_parcel_id))
        is not None
    ]
    items = [
        {
            "parcel_row_id": str(row["parcel_row_id"]),
            "path": None,
            "lead_score_total": _serialize_scalar(row["lead_score_total"]),
        }
        for _, row in filtered.iterrows()
    ]
    return {
        "geometry_mode": "viewport_geojson",
        "render_mode": render_mode,
        "geometry_bounds": _serialize_bounds(feature_bounds),
        "geometry_view_box": None,
        "requested_bounds": _serialize_bounds(bounds),
        "zoom": zoom,
        "feature_count": len(features),
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "items": items,
    }


def get_presets() -> list[dict[str, Any]]:
    meta = load_meta()
    grouped: dict[str, dict[str, Any]] = {}
    for row in meta.get("defaultViews", []):
        name = row.get("view_name")
        if not name:
            continue
        preset = grouped.setdefault(
            name,
            {
                "view_name": name,
                "description": row.get("description"),
                "filter_expression": row.get("filter_expression"),
            },
        )
        preset[row.get("metric", "value")] = row.get("value")
    return list(grouped.values())


def get_summary() -> dict[str, Any]:
    meta = load_meta()
    summary_rows = meta.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(row.get("section", "unknown"), []).append(row)
    return {
        "row_count": meta.get("rowCount"),
        "source": meta.get("source"),
        "geometry_mode": meta.get("geometryMode"),
        "sections": grouped,
    }
