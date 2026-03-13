from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd
from shapely import wkb

from app.settings import (
    GEOMETRY_DEFAULT_LIMIT,
    GEOMETRY_MAX_LIMIT,
    LEADS_DEFAULT_LIMIT,
    LEADS_MAX_LIMIT,
    MISSISSIPPI_APP_READY_PATH,
    MISSISSIPPI_GEOMETRY_PATH,
    MISSISSIPPI_META_PATH,
)


APP_READY_PATH = MISSISSIPPI_APP_READY_PATH
META_PATH = MISSISSIPPI_META_PATH
GEOMETRY_PATH = MISSISSIPPI_GEOMETRY_PATH

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


@lru_cache(maxsize=1)
def load_app_ready_frame() -> pd.DataFrame:
    if not APP_READY_PATH.exists():
        raise FileNotFoundError(f"Mississippi leads dataset not found: {APP_READY_PATH}")
    frame = pd.read_parquet(APP_READY_PATH)
    return frame


@lru_cache(maxsize=1)
def load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Mississippi explorer meta file not found: {META_PATH}")
    with META_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_geometry_lookup() -> dict[str, str]:
    if not GEOMETRY_PATH.exists():
        raise FileNotFoundError(f"Mississippi geometry file not found: {GEOMETRY_PATH}")
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
    frame = load_app_ready_frame()
    if parcel_row_ids:
        filtered = frame.loc[frame["parcel_row_id"].astype("string").isin(parcel_row_ids)].copy()
    else:
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
        safe_limit = _clamp_limit(limit, default=GEOMETRY_DEFAULT_LIMIT, max_limit=GEOMETRY_MAX_LIMIT)
        filtered = filtered.sort_values("lead_score_total", ascending=False, na_position="last").head(safe_limit)

    lookup = load_geometry_lookup()
    items = [
        {
            "parcel_row_id": str(row["parcel_row_id"]),
            "path": lookup.get(str(row["parcel_row_id"])) or None,
            "lead_score_total": _serialize_scalar(row["lead_score_total"]),
        }
        for _, row in filtered.iterrows()
    ]
    meta = load_meta()
    return {
        "geometry_mode": meta.get("geometryMode"),
        "geometry_bounds": meta.get("geometryBounds"),
        "geometry_view_box": meta.get("geometryViewBox"),
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
