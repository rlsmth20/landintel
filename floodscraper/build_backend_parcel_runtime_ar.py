from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests

from parcel_contract_ms import (
    API_LEADS_SUMMARY_FIELDS,
    GEOMETRY_FEATURE_PROPERTY_FIELDS,
    GEOMETRY_ITEM_FIELDS,
    serialize_contract_row,
)
from state_artifacts import load_state_artifacts
from state_diagnostics import build_state_diagnostics
from state_registry import ensure_state_directories, load_state_definition


ROOT = Path(__file__).resolve().parents[1]
STATE_CODE = "ar"
STATE_NAME = "Arkansas"
DEFAULT_BATCH_SIZE = 4000
DEFAULT_MAX_APP_READY = 50_000
DEFAULT_DEFAULT_LEADS_LIMIT = 200
DEFAULT_FRONTEND_FALLBACK_LIMIT = 5_000
SQ_METERS_PER_ACRE = 4046.8564224
FEET_PER_METER = 3.28084
SOURCE_CONFIDENCE_TIER = "medium"
BEST_SOURCE_TYPE = "statewide_arcgis_feature_layer"
BEST_SOURCE_NAME = "Arkansas GIS Office statewide parcel layer"
COUNTY_SOURCE_COVERAGE_TIER = "statewide_primary"
SOURCE_WARNING = (
    "Arkansas MVP uses statewide parcel attributes plus centroid references. "
    "Selected parcel polygons are fetched live from the source feature layer."
)

ATTRIBUTE_FIELDS = [
    "objectid",
    "countyfips",
    "county",
    "parcelid",
    "ownername",
    "adrlabel",
    "parceltype",
    "assessvalue",
    "impvalue",
    "landvalue",
    "totalvalue",
    "subdivision",
    "Shape__Area",
    "Shape__Length",
]

STATIC_FIELD_READINESS = [
    {
        "field_name": "parcel_vacant_flag",
        "readiness": "partial",
        "ui_guidance": "Expose as a parcel-attribute proxy, not as a vacancy-model replacement.",
        "notes": "Arkansas MVP vacancy status is inferred from assessed improvement value, not imagery.",
    },
    {
        "field_name": "road_access_tier",
        "readiness": "hide_from_default_ui",
        "ui_guidance": "Hide until Arkansas road-distance enrichment exists.",
        "notes": "No Arkansas road-access enrichment is included in the first MVP runtime.",
    },
    {
        "field_name": "delinquent_amount",
        "readiness": "hide_from_default_ui",
        "ui_guidance": "Hide until Arkansas tax-source onboarding is implemented.",
        "notes": "County-hosted tax or delinquency feeds are not active in the first Arkansas pass.",
    },
    {
        "field_name": "geometry_marketability_flag",
        "readiness": "production_ready",
        "ui_guidance": "Safe to surface as contextual lead-quality guidance.",
        "notes": "Arkansas MVP marketability uses proxy dimensions derived from statewide area/perimeter attributes.",
    },
]

PRESET_DEFINITIONS = {
    "general_ranked": {
        "description": "Top-ranked Arkansas MVP parcel leads from the statewide parcel layer.",
        "filter_expression": "recommended_view_bucket = 'general_ranked'",
    },
    "vacant_land_targeting": {
        "description": "Likely vacant parcels with acceptable geometry and usable centroid coverage.",
        "filter_expression": "parcel_vacant_flag = true AND recommended_view_bucket = 'vacant_land_targeting'",
    },
    "larger_land_targeting": {
        "description": "Larger-acreage Arkansas land parcels in the current app_ready subset.",
        "filter_expression": "acreage >= 10 AND recommended_view_bucket = 'larger_land_targeting'",
    },
}

STRING_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "state_code",
    "county_name",
    "county_name_display",
    "county_fips",
    "owner_name",
    "site_address",
    "land_use",
    "subdivision_name",
    "geometry_quality_flag",
    "geometry_marketability_base_flag",
    "geometry_marketability_flag",
    "geometry_marketability_context",
    "geometry_marketability_action",
    "geometry_penalty_reason",
    "county_tax_coverage_status",
    "county_tax_coverage_reason",
    "parcel_tax_data_warning",
    "best_source_type",
    "best_source_name",
    "source_confidence_tier",
    "county_source_coverage_tier",
    "lead_score_tier",
    "lead_score_driver_1",
    "lead_score_driver_2",
    "lead_score_driver_3",
    "lead_score_explanation",
    "recommended_sort_reason",
    "top_score_driver",
    "caution_flags",
    "vacant_reason",
    "recommended_use_case",
    "recommended_view_bucket",
    "parcel_improvement_status",
    "parcel_improvement_reason",
    "parcel_improvement_evidence_summary",
    "ai_vacancy_source",
    "ai_vacancy_status_note",
    "overall_vacancy_assessment",
]

BOOLEAN_COLUMNS = [
    "is_multipart",
    "geometry_review_excluded_flag",
    "geometry_training_excluded_flag",
    "geometry_default_leads_excluded_flag",
    "geometry_effective_buildable_flag",
    "geometry_marketability_default_leads_excluded_flag",
    "parcel_vacant_flag",
    "corporate_owner_flag",
    "county_tax_source_configured_flag",
    "county_tax_source_loaded_flag",
    "tax_data_available_flag",
    "parcel_tax_is_actionable_current",
    "parcel_tax_is_historical_only",
    "high_confidence_link_flag",
    "county_hosted_flag",
    "ai_vacancy_available_flag",
]

INTEGER_COLUMNS = ["part_count", "geometry_penalty_points", "building_count"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Arkansas MVP runtime artifacts from the official statewide parcel layer.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-app-ready", type=int, default=DEFAULT_MAX_APP_READY)
    parser.add_argument("--default-leads-limit", type=int, default=DEFAULT_DEFAULT_LEADS_LIMIT)
    parser.add_argument("--frontend-fallback-limit", type=int, default=DEFAULT_FRONTEND_FALLBACK_LIMIT)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _slug_county_name(value: Any) -> str | None:
    text = _safe_string(value)
    if not text:
        return None
    return text.lower().replace(".", "").replace("&", "and").replace(" ", "_")


def _county_fips(value: Any) -> str | None:
    text = _safe_string(value)
    if not text:
        return None
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    return digits.zfill(5)


def _parcel_row_id(county_fips: str | None, parcel_id: str | None) -> str | None:
    if not county_fips or not parcel_id:
        return None
    digest = hashlib.sha1(f"{county_fips}|{parcel_id.strip().upper()}".encode("utf-8")).hexdigest()[:16]
    return f"ar_{digest}"


def _corporate_owner_flag(owner_name: str | None) -> bool | None:
    if not owner_name:
        return None
    normalized = owner_name.upper()
    return any(token in normalized for token in (" LLC", " INC", " CORP", " CO", " LP", " LTD", " TRUST", " BANK"))


def _point_geometry_payload(latitude: float | None, longitude: float | None) -> dict[str, Any] | None:
    if latitude is None or longitude is None:
        return None
    return {
        "type": "point_reference",
        "centroid": {
            "type": "Point",
            "coordinates": [round(longitude, 6), round(latitude, 6)],
        },
        "bounds": None,
    }


def _proxy_dimensions(area_square_meters: float | None, perimeter_meters: float | None) -> tuple[float | None, float | None, float | None]:
    if area_square_meters is None or perimeter_meters is None or area_square_meters <= 0 or perimeter_meters <= 0:
        return None, None, None
    semi_perimeter = perimeter_meters / 2.0
    discriminant = max((semi_perimeter**2) - (4.0 * area_square_meters), 0.0)
    root = math.sqrt(discriminant)
    short_dimension = (semi_perimeter - root) / 2.0
    long_dimension = (semi_perimeter + root) / 2.0
    if short_dimension <= 0 or long_dimension <= 0:
        return None, None, None
    return short_dimension, long_dimension, long_dimension / short_dimension


def _compactness(area_square_meters: float | None, perimeter_meters: float | None) -> float | None:
    if area_square_meters is None or perimeter_meters is None or area_square_meters <= 0 or perimeter_meters <= 0:
        return None
    value = (4.0 * math.pi * area_square_meters) / (perimeter_meters**2)
    return max(0.0, min(float(value), 1.0))


def _approximate_frontage_to_width(max_dimension_feet: float | None, min_dimension_feet: float | None) -> float | None:
    if max_dimension_feet is None or min_dimension_feet is None or min_dimension_feet <= 0:
        return None
    return max_dimension_feet / min_dimension_feet


def _land_use_tokens(land_use: str | None) -> str:
    return (land_use or "").strip().upper()


def _geometry_quality_flag(
    *,
    area_acres: float | None,
    aspect_ratio: float | None,
    compactness: float | None,
    min_dimension_meters: float | None,
    land_use: str | None,
) -> str:
    tokens = _land_use_tokens(land_use)
    row_like = any(token in tokens for token in ("ROW", "RIGHT OF WAY", "R/W", "ROAD", "EASEMENT", "RAIL"))
    if row_like:
        return "access_strip"
    if aspect_ratio is not None and compactness is not None and (aspect_ratio > 10.0 or (aspect_ratio > 6.0 and compactness < 0.1)):
        return "access_strip"
    if area_acres is not None and area_acres < 0.25 and aspect_ratio is not None and aspect_ratio > 4.0:
        return "irregular"
    if compactness is not None and compactness < 0.15:
        return "irregular"
    if min_dimension_meters is not None and min_dimension_meters < 20.0:
        return "irregular"
    return "good"


def _marketability(
    *,
    geometry_quality_flag: str,
    area_acres: float | None,
    compactness: float | None,
    aspect_ratio: float | None,
    min_dimension_feet: float | None,
    width_feet: float | None,
    frontage_to_width_ratio: float | None,
    land_use: str | None,
) -> tuple[str, str, str, int, str]:
    tokens = _land_use_tokens(land_use)
    remnant_like = any(token in tokens for token in ("ROW", "RIGHT OF WAY", "ROAD", "EASEMENT", "RAIL"))
    if remnant_like:
        return "unbuildable_candidate", "unbuildable_candidate", "exclude", -60, "parcel_type_row_or_easement"
    if min_dimension_feet is not None and min_dimension_feet < 20.0:
        return "unbuildable_candidate", "unbuildable_candidate", "exclude", -60, "proxy_min_dimension_below_20ft"
    if width_feet is not None and width_feet < 25.0:
        return "unbuildable_candidate", "unbuildable_candidate", "exclude", -60, "proxy_width_below_25ft"
    if compactness is not None and area_acres is not None and compactness < 0.25 and area_acres < 0.5:
        return "unbuildable_candidate", "unbuildable_candidate", "exclude", -60, "small_low_compactness_proxy"
    if aspect_ratio is not None and compactness is not None and aspect_ratio > 6.0 and compactness < 0.18:
        return "unbuildable_candidate", "unbuildable_candidate", "exclude", -60, "high_aspect_low_compactness_proxy"
    if frontage_to_width_ratio is not None and width_feet is not None and frontage_to_width_ratio > 7.0 and width_feet < 35.0:
        return "unbuildable_candidate", "unbuildable_candidate", "exclude", -60, "frontage_width_ratio_proxy"
    if geometry_quality_flag == "access_strip":
        return "unbuildable_candidate", "unbuildable_candidate", "exclude", -60, "geometry_quality_access_strip"
    if geometry_quality_flag == "irregular" or (compactness is not None and compactness < 0.28 and (area_acres or 0) < 1.0):
        action = "exclude" if (area_acres or 0) < 0.35 else "penalize"
        penalty = -45 if action == "exclude" else -28
        return "poor_geometry", "poor_geometry", action, penalty, "small_or_irregular_proxy_geometry"
    if compactness is not None and compactness < 0.35:
        return "constrained", "constrained", "penalize", -8, "constrained_proxy_compactness"
    if width_feet is not None and width_feet < 50.0:
        return "constrained", "constrained", "penalize", -8, "constrained_proxy_width"
    return "marketable", "marketable", "keep", 0, "marketable_proxy_geometry"


def _vacancy_proxy(
    *,
    assessed_improvement_value: float | None,
    assessed_total_value: float | None,
) -> tuple[bool | None, str, float | None, str, str]:
    improvement_value = assessed_improvement_value or 0.0
    total_value = assessed_total_value or 0.0
    improvement_ratio = (improvement_value / total_value) if total_value > 0 else None
    if improvement_value <= 0 or (improvement_ratio is not None and improvement_ratio <= 0.02):
        return True, "likely_vacant", 84.0, "assessed_improvement_value_zero_or_minimal", "Assessed improvement value is zero or negligible."
    if improvement_ratio is not None and improvement_ratio >= 0.35:
        return False, "likely_improved", 76.0, "assessed_improvement_value_substantial", "Assessed improvement value is substantial relative to total value."
    return None, "needs_review", 55.0, "assessed_improvement_value_ambiguous", "Assessed improvement value is present but not decisive."


def _lead_score_components(
    *,
    acreage: float | None,
    parcel_vacant_flag: bool | None,
    assessed_total_value: float | None,
    marketability_penalty_points: int,
    corporate_owner_flag: bool | None,
) -> tuple[float, float, float, float, float]:
    vacancy_score = 22.0 if parcel_vacant_flag is True else -8.0 if parcel_vacant_flag is False else 6.0
    if acreage is None or acreage <= 0:
        size_score = 0.0
    elif acreage < 0.25:
        size_score = -6.0
    elif acreage < 1.0:
        size_score = 6.0
    elif acreage <= 10.0:
        size_score = 12.0
    elif acreage <= 80.0:
        size_score = 10.0
    elif acreage <= 250.0:
        size_score = 6.0
    else:
        size_score = 2.0

    if assessed_total_value is None or assessed_total_value <= 0:
        value_score = 4.0
    elif assessed_total_value <= 50_000:
        value_score = 10.0
    elif assessed_total_value <= 150_000:
        value_score = 6.0
    else:
        value_score = 2.0

    owner_score = 4.0 if corporate_owner_flag else 0.0
    source_score = 8.0
    return vacancy_score, size_score, value_score, owner_score, source_score + float(marketability_penalty_points)


def _lead_score_tier(score: float) -> str:
    if score >= 75.0:
        return "very_high"
    if score >= 60.0:
        return "high"
    if score >= 45.0:
        return "medium"
    return "low"


def _recommended_view_bucket(acreage: float | None, parcel_vacant_flag: bool | None) -> str:
    if acreage is not None and acreage >= 10.0:
        return "larger_land_targeting"
    if parcel_vacant_flag is True:
        return "vacant_land_targeting"
    return "general_ranked"


def _json_ready(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _feature_payload(row: dict[str, Any], *, selected: bool = False) -> dict[str, Any]:
    payload = serialize_contract_row(row, GEOMETRY_FEATURE_PROPERTY_FIELDS, serializer=_json_ready)
    payload["selected"] = selected
    return payload


def _geometry_item(row: dict[str, Any]) -> dict[str, Any]:
    return serialize_contract_row(row, GEOMETRY_ITEM_FIELDS, serializer=_json_ready)


def _top_records(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.sort_values(
        by=["lead_score_total_effective", "lead_score_total", "acreage", "parcel_row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.head(limit).copy()


def _fetch_json(session: requests.Session, url: str, *, data: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.post(url, data=data, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if "error" in payload:
                raise RuntimeError(f"ArcGIS query error: {payload['error']}")
            return payload
        except Exception as error:  # noqa: PERF203
            last_error = error
            if attempt == 2:
                break
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to query Arkansas parcel source: {last_error}") from last_error


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_partitioned_index(frame: pd.DataFrame, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(output_root),
        format="parquet",
        partitioning=["county_name"],
        existing_data_behavior="delete_matching",
    )


def _preset_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for view_name, meta in PRESET_DEFINITIONS.items():
        if "recommended_view_bucket" in frame.columns:
            subset = frame.loc[frame["recommended_view_bucket"] == view_name].copy()
        else:
            subset = pd.DataFrame(columns=frame.columns)
        average_score = float(subset["lead_score_total_effective"].mean()) if not subset.empty else None
        items.append(
            {
                "view_name": view_name,
                "description": meta["description"],
                "filter_expression": meta["filter_expression"],
                "row_count": str(int(len(subset))),
                "average_lead_score": f"{average_score:.4f}" if average_score is not None and not math.isnan(average_score) else None,
            }
        )
    return items


def _summary_sections(app_ready: pd.DataFrame, geometry_flag_counts: Counter[str], marketability_action_counts: Counter[str]) -> dict[str, list[dict[str, str]]]:
    statewide: list[dict[str, str]] = [
        {"section": "statewide", "metric": "lead_count", "value": str(int(len(app_ready)))},
        {
            "section": "statewide",
            "metric": "average_lead_score",
            "value": f"{float(app_ready['lead_score_total_effective'].mean()):.4f}" if not app_ready.empty else "0.0000",
        },
        {
            "section": "statewide",
            "metric": "likely_vacant_count",
            "value": str(int(app_ready["parcel_vacant_flag"].fillna(False).astype(bool).sum())) if "parcel_vacant_flag" in app_ready.columns else "0",
        },
        {
            "section": "statewide",
            "metric": "default_leads_marketability_excluded_count",
            "value": str(int(marketability_action_counts.get("exclude", 0))),
        },
    ]
    top_counties: list[dict[str, str]] = []
    if not app_ready.empty and "county_name" in app_ready.columns:
        for county_name, count in app_ready["county_name"].value_counts().head(20).items():
            top_counties.append(
                {
                    "section": "top_counties",
                    "key": str(county_name),
                    "metric": "lead_count",
                    "value": str(int(count)),
                }
            )
    recommended_view_bucket: list[dict[str, str]] = []
    if not app_ready.empty and "recommended_view_bucket" in app_ready.columns:
        for bucket, count in app_ready["recommended_view_bucket"].value_counts().items():
            recommended_view_bucket.append(
                {
                    "section": "recommended_view_bucket",
                    "key": str(bucket),
                    "metric": "lead_count",
                    "value": str(int(count)),
                }
            )
    geometry_quality_section = [
        {
            "section": "geometry_quality_flag",
            "key": flag,
            "metric": "parcel_count",
            "value": str(int(count)),
        }
        for flag, count in sorted(geometry_flag_counts.items())
    ]
    marketability_action_section = [
        {
            "section": "geometry_marketability_action",
            "key": action,
            "metric": "parcel_count",
            "value": str(int(count)),
        }
        for action, count in sorted(marketability_action_counts.items())
    ]
    return {
        "statewide": statewide,
        "top_counties": top_counties,
        "recommended_view_bucket": recommended_view_bucket,
        "geometry_quality_flag": geometry_quality_section,
        "geometry_marketability_action": marketability_action_section,
    }


def _meta_payload(
    *,
    app_ready: pd.DataFrame,
    summary_payload: dict[str, Any],
    presets_payload: list[dict[str, Any]],
    app_ready_path: Path,
) -> dict[str, Any]:
    bounds = None
    if not app_ready.empty and {"longitude", "latitude"}.issubset(app_ready.columns):
        lon = pd.to_numeric(app_ready["longitude"], errors="coerce")
        lat = pd.to_numeric(app_ready["latitude"], errors="coerce")
        if lon.notna().any() and lat.notna().any():
            bounds = [
                round(float(lon.min()), 6),
                round(float(lat.min()), 6),
                round(float(lon.max()), 6),
                round(float(lat.max()), 6),
            ]
    default_views: list[dict[str, str]] = []
    for item in presets_payload:
        for metric_name in ("row_count", "average_lead_score"):
            metric_value = item.get(metric_name)
            if metric_value is None:
                continue
            default_views.append(
                {
                    "view_name": str(item["view_name"]),
                    "description": str(item.get("description") or ""),
                    "filter_expression": str(item.get("filter_expression") or ""),
                    "metric": metric_name,
                    "value": str(metric_value),
                }
            )
    flattened_summary: list[dict[str, str]] = []
    for section_name, section_items in summary_payload.get("sections", {}).items():
        for item in section_items:
            flattened_summary.append(
                {
                    "section": section_name,
                    "metric": str(item.get("metric") or ""),
                    "key": str(item.get("key") or ""),
                    "value": str(item.get("value") or ""),
                }
            )
    return {
        "defaultViews": default_views,
        "fieldReadiness": STATIC_FIELD_READINESS,
        "summary": flattened_summary,
        "rowCount": int(len(app_ready)),
        "source": _relative_path(app_ready_path),
        "geometryMode": "selected_parcel_geojson",
        "geometryBounds": bounds,
        "geometryViewBox": [0, 0, 1000, 700] if bounds else None,
        "warnings": [
            SOURCE_WARNING,
            "Arkansas PMTiles parcel basemap generation is deferred in the first MVP pass.",
        ],
    }


def _default_geometry_payload(frame: pd.DataFrame) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        latitude = _safe_float(record.get("latitude"))
        longitude = _safe_float(record.get("longitude"))
        if latitude is None or longitude is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(longitude, 6), round(latitude, 6)],
                },
                "properties": _feature_payload(record, selected=False),
            }
        )
        items.append(_geometry_item(record))
    bounds = None
    if features:
        longitudes = [feature["geometry"]["coordinates"][0] for feature in features]
        latitudes = [feature["geometry"]["coordinates"][1] for feature in features]
        bounds = [round(min(longitudes), 6), round(min(latitudes), 6), round(max(longitudes), 6), round(max(latitudes), 6)]
    return {
        "geometry_mode": "selected_parcel_geojson",
        "render_mode": "points",
        "geometry_bounds": bounds,
        "geometry_view_box": None,
        "requested_bounds": None,
        "zoom": None,
        "feature_count": len(features),
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "items": items,
    }


def _frontend_fallback_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        payload = {key: _json_ready(value) for key, value in record.items()}
        payload["geometry"] = _point_geometry_payload(_safe_float(record.get("latitude")), _safe_float(record.get("longitude")))
        rows.append(payload)
    return rows


def _transform_feature_batch(features: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        attributes = feature.get("attributes", {})
        centroid = feature.get("centroid") or feature.get("geometry") or {}
        county_name_display = _safe_string(attributes.get("county"))
        county_name = _slug_county_name(county_name_display)
        county_fips = _county_fips(attributes.get("countyfips"))
        parcel_id = _safe_string(attributes.get("parcelid"))
        parcel_row_id = _parcel_row_id(county_fips, parcel_id)
        area_square_meters = _safe_float(attributes.get("Shape__Area"))
        perimeter_meters = _safe_float(attributes.get("Shape__Length"))
        acreage = (area_square_meters / SQ_METERS_PER_ACRE) if area_square_meters is not None else None
        compactness = _compactness(area_square_meters, perimeter_meters)
        min_dimension_meters, max_dimension_meters, aspect_ratio = _proxy_dimensions(area_square_meters, perimeter_meters)
        min_dimension_feet = min_dimension_meters * FEET_PER_METER if min_dimension_meters is not None else None
        max_dimension_feet = max_dimension_meters * FEET_PER_METER if max_dimension_meters is not None else None
        width_feet = min_dimension_feet
        frontage_feet = max_dimension_feet
        frontage_ratio = _approximate_frontage_to_width(frontage_feet, width_feet)
        land_use = _safe_string(attributes.get("parceltype"))
        geometry_quality_flag = _geometry_quality_flag(
            area_acres=acreage,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            min_dimension_meters=min_dimension_meters,
            land_use=land_use,
        )
        geometry_review_excluded_flag = geometry_quality_flag != "good"
        geometry_training_excluded_flag = geometry_quality_flag in {"access_strip", "multipart_complex"}
        geometry_default_leads_excluded_flag = geometry_quality_flag == "access_strip"
        (
            geometry_marketability_base_flag,
            geometry_marketability_flag,
            geometry_marketability_action,
            geometry_penalty_points,
            geometry_penalty_reason,
        ) = _marketability(
            geometry_quality_flag=geometry_quality_flag,
            area_acres=acreage,
            compactness=compactness,
            aspect_ratio=aspect_ratio,
            min_dimension_feet=min_dimension_feet,
            width_feet=width_feet,
            frontage_to_width_ratio=frontage_ratio,
            land_use=land_use,
        )
        assessed_total_value = _safe_float(attributes.get("totalvalue")) or _safe_float(attributes.get("assessvalue"))
        assessed_land_value = _safe_float(attributes.get("landvalue"))
        assessed_improvement_value = _safe_float(attributes.get("impvalue"))
        parcel_vacant_flag, parcel_improvement_status, parcel_improvement_confidence, parcel_improvement_reason, parcel_improvement_summary = _vacancy_proxy(
            assessed_improvement_value=assessed_improvement_value,
            assessed_total_value=assessed_total_value,
        )
        corporate_owner_flag = _corporate_owner_flag(_safe_string(attributes.get("ownername")))
        vacancy_score, size_score, value_score, owner_score, source_and_penalty = _lead_score_components(
            acreage=acreage,
            parcel_vacant_flag=parcel_vacant_flag,
            assessed_total_value=assessed_total_value,
            marketability_penalty_points=geometry_penalty_points,
            corporate_owner_flag=corporate_owner_flag,
        )
        raw_score = 45.0 + vacancy_score + size_score + value_score + owner_score + source_and_penalty
        lead_score_total = float(max(0.0, min(raw_score, 100.0)))
        recommended_view_bucket = _recommended_view_bucket(acreage, parcel_vacant_flag)
        top_score_driver = "vacancy_proxy" if parcel_vacant_flag is True else "marketability_penalty" if geometry_penalty_points < 0 else "size_and_value"
        warning_parts = [SOURCE_WARNING]
        if geometry_marketability_action == "exclude":
            warning_parts.append(f"Proxy geometry marked parcel as {geometry_marketability_flag}.")
        elif geometry_marketability_action == "penalize":
            warning_parts.append(f"Proxy geometry penalized parcel as {geometry_marketability_flag}.")
        if parcel_improvement_status == "needs_review":
            warning_parts.append("Assessed improvement value is ambiguous for vacant/improved classification.")
        rows.append(
            {
                "parcel_row_id": parcel_row_id,
                "parcel_id": parcel_id,
                "state_code": STATE_CODE,
                "county_name": county_name,
                "county_name_display": county_name_display,
                "county_fips": county_fips,
                "source_object_id": _safe_float(attributes.get("objectid")),
                "owner_name": _safe_string(attributes.get("ownername")),
                "site_address": _safe_string(attributes.get("adrlabel")),
                "land_use": land_use,
                "subdivision_name": _safe_string(attributes.get("subdivision")),
                "acreage": acreage,
                "acreage_bucket": None,
                "area_acres": acreage,
                "perimeter_meters": perimeter_meters,
                "bounding_box_width_meters": min_dimension_meters,
                "bounding_box_height_meters": max_dimension_meters,
                "aspect_ratio": aspect_ratio,
                "compactness": compactness,
                "is_multipart": False,
                "part_count": 1,
                "geometry_quality_flag": geometry_quality_flag,
                "geometry_review_excluded_flag": geometry_review_excluded_flag,
                "geometry_training_excluded_flag": geometry_training_excluded_flag,
                "geometry_default_leads_excluded_flag": geometry_default_leads_excluded_flag,
                "geometry_estimated_frontage_feet": frontage_feet,
                "geometry_estimated_width_feet": width_feet,
                "geometry_min_dimension_feet": min_dimension_feet,
                "geometry_max_dimension_feet": max_dimension_feet,
                "geometry_frontage_to_width_ratio": frontage_ratio,
                "geometry_effective_buildable_flag": geometry_marketability_action != "exclude",
                "geometry_marketability_base_flag": geometry_marketability_base_flag,
                "geometry_marketability_flag": geometry_marketability_flag,
                "geometry_marketability_context": "unknown",
                "geometry_marketability_action": geometry_marketability_action,
                "geometry_penalty_points": geometry_penalty_points,
                "geometry_penalty_reason": geometry_penalty_reason,
                "geometry_marketability_default_leads_excluded_flag": geometry_marketability_action == "exclude",
                "assessed_land_value": assessed_land_value,
                "assessed_improvement_value": assessed_improvement_value,
                "assessed_total_value": assessed_total_value,
                "parcel_vacant_flag": parcel_vacant_flag,
                "county_vacant_flag": None,
                "building_count": 0 if parcel_vacant_flag is True else 1 if parcel_vacant_flag is False else None,
                "building_area_total": None,
                "nearby_building_count_1km": None,
                "nearby_building_density": None,
                "growth_pressure_bucket": None,
                "road_distance_ft": None,
                "road_access_tier": None,
                "wetland_flag": None,
                "flood_risk_score": None,
                "wetland_pct": None,
                "wetland_area_sqft": None,
                "flood_pct": None,
                "flood_area_sqft": None,
                "primary_fema_zone": None,
                "mean_slope_pct": None,
                "max_slope_pct": None,
                "slope_class": None,
                "slope_score": None,
                "elevation_mean_ft": None,
                "shape_compactness": compactness,
                "parcel_frontage_ft_estimate": frontage_feet,
                "parcel_width_ft_estimate": width_feet,
                "buildability_score": max(0.0, min(100.0, 85.0 + geometry_penalty_points)),
                "environment_score": None,
                "investment_score": max(0.0, min(100.0, lead_score_total)),
                "electric_provider_name": None,
                "owner_type": None,
                "corporate_owner_flag": corporate_owner_flag,
                "absentee_owner_flag": None,
                "out_of_state_owner_flag": None,
                "owner_parcel_count": None,
                "owner_total_acres": None,
                "mailer_target_score": 4.0 if corporate_owner_flag else 0.0,
                "delinquent_amount": None,
                "delinquent_amount_bucket": None,
                "delinquent_flag": None,
                "delinquent_year": None,
                "county_tax_source_configured_flag": False,
                "county_tax_source_loaded_flag": False,
                "tax_data_available_flag": False,
                "county_tax_coverage_status": "not_configured",
                "county_tax_coverage_reason": "Arkansas tax sources are not onboarded in the first MVP runtime.",
                "parcel_tax_status": None,
                "parcel_tax_status_label": None,
                "parcel_tax_status_confidence": None,
                "parcel_tax_status_category": None,
                "parcel_tax_actionability": None,
                "parcel_tax_data_warning": "Arkansas tax-source enrichment is not active in the first MVP runtime.",
                "parcel_tax_freshness_bucket": None,
                "parcel_tax_years_stale": None,
                "parcel_tax_is_actionable_current": False,
                "parcel_tax_is_historical_only": False,
                "parcel_tax_freshness_reason": None,
                "tax_data_upload_date": None,
                "tax_data_year": None,
                "tax_data_source": None,
                "delinquency_last_verified": None,
                "forfeited_flag": None,
                "best_source_type": BEST_SOURCE_TYPE,
                "best_source_name": BEST_SOURCE_NAME,
                "source_confidence_tier": SOURCE_CONFIDENCE_TIER,
                "county_source_coverage_tier": COUNTY_SOURCE_COVERAGE_TIER,
                "amount_trust_tier": None,
                "high_confidence_link_flag": False,
                "county_hosted_flag": False,
                "lead_score_total": lead_score_total,
                "lead_score_total_effective": lead_score_total,
                "lead_score_tier": _lead_score_tier(lead_score_total),
                "lead_score_driver_1": "vacancy_proxy",
                "lead_score_driver_2": "proxy_geometry_marketability",
                "lead_score_driver_3": "size_and_value",
                "lead_score_explanation": "Arkansas MVP score combines parcel-value vacancy proxies, proxy geometry marketability, parcel size, and source confidence.",
                "size_score": size_score,
                "access_score": None,
                "buildability_component": max(0.0, min(100.0, 85.0 + geometry_penalty_points)),
                "environmental_component": None,
                "owner_targeting_component": owner_score,
                "delinquency_component": None,
                "source_confidence_component": 8.0,
                "vacant_land_component": vacancy_score,
                "growth_pressure_component": None,
                "recommended_sort_reason": "arkansas_mvp_parcel_runtime",
                "top_score_driver": top_score_driver,
                "caution_flags": "; ".join(warning_parts),
                "vacant_reason": parcel_improvement_summary,
                "growth_pressure_reason": None,
                "recommended_use_case": "Arkansas MVP statewide parcel screening",
                "recommended_view_bucket": recommended_view_bucket,
                "parcel_improvement_status": parcel_improvement_status,
                "parcel_improvement_confidence": parcel_improvement_confidence,
                "parcel_improvement_reason": parcel_improvement_reason,
                "parcel_improvement_evidence_summary": parcel_improvement_summary,
                "building_signal_conflict_flag": None,
                "ai_building_present_probability": None,
                "ai_building_present_flag": None,
                "building_present_confidence": None,
                "building_presence_reason": None,
                "ai_vacancy_available_flag": False,
                "ai_vacancy_source": "unavailable",
                "ai_vacancy_status_note": "Arkansas vacancy imagery scoring is not onboarded in the first MVP runtime.",
                "vacancy_confidence_score": None,
                "vacancy_model_version": None,
                "overall_vacancy_assessment": parcel_improvement_status,
                "latitude": _safe_float(centroid.get("y")),
                "longitude": _safe_float(centroid.get("x")),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for column in STRING_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column].astype("string")
        for column in BOOLEAN_COLUMNS:
            if column in frame.columns:
                frame[column] = frame[column].astype("boolean")
        for column in INTEGER_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["parcel_row_id", "parcel_id", "county_name", "county_fips"]).copy()
        frame = frame.drop_duplicates(subset=["parcel_row_id"], keep="first").copy()
    return frame


def main() -> None:
    args = parse_args()
    definition = ensure_state_directories(STATE_CODE)
    artifacts = load_state_artifacts(STATE_CODE)
    source_registry_path = definition.source_registry_path("parcel_source")
    if source_registry_path is None or not source_registry_path.exists():
        raise FileNotFoundError(f"Arkansas parcel source registry is missing: {source_registry_path}")
    registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    source = next((item for item in registry.get("parcel_sources", []) if item.get("primary")), None)
    if source is None:
        raise RuntimeError("Arkansas parcel source registry has no primary parcel source configured.")

    for path in [
        artifacts.parcel_master_path.parent,
        artifacts.app_ready_path.parent,
        artifacts.runtime_root,
        artifacts.frontend_meta_path.parent,
        artifacts.frontend_detail_fallback_path.parent,
        artifacts.geometry_quality_artifact_path.parent,
        artifacts.training_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    build_started = _now_iso()
    stage_runtimes: dict[str, float] = {}
    stage_start = time.perf_counter()
    session = requests.Session()
    query_url = str(source["service_url"]).rstrip("/") + "/query"
    timeout_seconds = float(args.timeout_seconds)

    count_payload = _fetch_json(
        session,
        query_url,
        data={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout_seconds=timeout_seconds,
    )
    total_source_rows = int(count_payload.get("count") or 0)
    if total_source_rows <= 0:
        raise RuntimeError("Arkansas parcel source returned no rows.")
    if args.limit_records is not None:
        total_source_rows = min(total_source_rows, int(args.limit_records))
    stage_runtimes["fetch_count_seconds"] = round(time.perf_counter() - stage_start, 3)

    parcel_writer: pq.ParquetWriter | None = None
    geometry_writer: pq.ParquetWriter | None = None
    app_ready_accumulator = pd.DataFrame()
    county_counts: Counter[str] = Counter()
    geometry_flag_counts: Counter[str] = Counter()
    marketability_flag_counts: Counter[str] = Counter()
    marketability_action_counts: Counter[str] = Counter()
    review_excluded_by_county: Counter[str] = Counter()
    training_excluded_by_county: Counter[str] = Counter()
    default_excluded_by_county: Counter[str] = Counter()
    missing_centroid_count = 0
    processed_rows = 0

    stage_start = time.perf_counter()
    for index in range(0, total_source_rows, int(args.batch_size)):
        batch_size = min(int(args.batch_size), total_source_rows - index)
        payload = _fetch_json(
            session,
            query_url,
            data={
                "where": "1=1",
                "outFields": ",".join(ATTRIBUTE_FIELDS),
                "returnGeometry": "false",
                "returnCentroid": "true",
                "outSR": "4326",
                "orderByFields": "objectid ASC",
                "resultOffset": str(index),
                "resultRecordCount": str(batch_size),
                "resultType": "standard",
                "f": "json",
            },
            timeout_seconds=timeout_seconds,
        )
        batch_frame = _transform_feature_batch(payload.get("features", []))
        if batch_frame.empty:
            continue

        processed_rows += int(len(batch_frame))
        missing_centroid_count += int(batch_frame["latitude"].isna().sum() + batch_frame["longitude"].isna().sum())
        county_counts.update(batch_frame["county_name"].dropna().astype(str).tolist())
        geometry_flag_counts.update(batch_frame["geometry_quality_flag"].dropna().astype(str).tolist())
        marketability_flag_counts.update(batch_frame["geometry_marketability_flag"].dropna().astype(str).tolist())
        marketability_action_counts.update(batch_frame["geometry_marketability_action"].dropna().astype(str).tolist())
        review_excluded_by_county.update(
            batch_frame.loc[batch_frame["geometry_review_excluded_flag"].fillna(False), "county_name"].dropna().astype(str).tolist()
        )
        training_excluded_by_county.update(
            batch_frame.loc[batch_frame["geometry_training_excluded_flag"].fillna(False), "county_name"].dropna().astype(str).tolist()
        )
        default_excluded_by_county.update(
            batch_frame.loc[
                batch_frame["geometry_marketability_default_leads_excluded_flag"].fillna(False),
                "county_name",
            ].dropna().astype(str).tolist()
        )

        parcel_table = pa.Table.from_pandas(batch_frame, preserve_index=False)
        parcel_writer = parcel_writer or pq.ParquetWriter(artifacts.parcel_master_path, parcel_table.schema, compression="snappy")
        parcel_writer.write_table(parcel_table)

        geometry_columns = [
            "parcel_row_id",
            "area_acres",
            "perimeter_meters",
            "bounding_box_width_meters",
            "bounding_box_height_meters",
            "aspect_ratio",
            "compactness",
            "is_multipart",
            "part_count",
            "geometry_quality_flag",
            "geometry_review_excluded_flag",
            "geometry_training_excluded_flag",
            "geometry_default_leads_excluded_flag",
        ]
        geometry_table = pa.Table.from_pandas(batch_frame[geometry_columns], preserve_index=False)
        geometry_writer = geometry_writer or pq.ParquetWriter(
            artifacts.geometry_quality_artifact_path,
            geometry_table.schema,
            compression="snappy",
        )
        geometry_writer.write_table(geometry_table)

        candidate_mask = (
            batch_frame["parcel_row_id"].notna()
            & batch_frame["parcel_id"].notna()
            & batch_frame["county_name"].notna()
            & batch_frame["latitude"].notna()
            & batch_frame["longitude"].notna()
            & batch_frame["lead_score_total_effective"].notna()
            & ~batch_frame["geometry_marketability_default_leads_excluded_flag"].fillna(False)
            & pd.to_numeric(batch_frame["acreage"], errors="coerce").fillna(0).ge(0.25)
            & pd.to_numeric(batch_frame["lead_score_total_effective"], errors="coerce").fillna(0).ge(35.0)
        )
        candidate_frame = batch_frame.loc[candidate_mask].copy()
        if not candidate_frame.empty:
            app_ready_accumulator = pd.concat([app_ready_accumulator, candidate_frame], ignore_index=True)
            app_ready_accumulator = _top_records(app_ready_accumulator, limit=int(args.max_app_ready))

        if (index // int(args.batch_size)) % 25 == 0:
            print(
                f"Processed Arkansas parcels {min(index + batch_size, total_source_rows):,}/{total_source_rows:,} "
                f"master_rows={processed_rows:,} app_ready_pool={len(app_ready_accumulator):,}"
            )

    if parcel_writer is not None:
        parcel_writer.close()
    if geometry_writer is not None:
        geometry_writer.close()
    stage_runtimes["stream_statewide_source_seconds"] = round(time.perf_counter() - stage_start, 3)

    app_ready = _top_records(app_ready_accumulator, limit=int(args.max_app_ready))
    app_ready = app_ready.sort_values(
        by=["lead_score_total_effective", "lead_score_total", "acreage", "parcel_row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    stage_start = time.perf_counter()
    app_ready.to_parquet(artifacts.app_ready_path, index=False)
    app_ready.to_parquet(artifacts.runtime_detail_metrics_path, index=False)
    _write_partitioned_index(app_ready, artifacts.runtime_parcel_index_root)
    stage_runtimes["write_runtime_parquet_seconds"] = round(time.perf_counter() - stage_start, 3)

    geometry_quality_summary = {
        "state_code": STATE_CODE,
        "state_name": STATE_NAME,
        "row_count": processed_rows,
        "processed_rows": processed_rows,
        "missing_centroid_count": missing_centroid_count,
        "proxy_dimensions_from_area_perimeter_flag": True,
        "warnings": [
            SOURCE_WARNING,
            "Bounding-box width/height and aspect ratio are proxy dimensions derived from statewide area/perimeter attributes in the first Arkansas pass.",
        ],
        "build_start_timestamp": build_started,
        "build_end_timestamp": _now_iso(),
        "runtime_by_stage_seconds": stage_runtimes,
        "geometry_quality_flag_counts": dict(sorted(geometry_flag_counts.items())),
        "review_excluded_count": int(sum(review_excluded_by_county.values())),
        "review_excluded_pct": round((sum(review_excluded_by_county.values()) / processed_rows) * 100.0, 2) if processed_rows else 0.0,
        "training_excluded_count": int(sum(training_excluded_by_county.values())),
        "training_excluded_pct": round((sum(training_excluded_by_county.values()) / processed_rows) * 100.0, 2) if processed_rows else 0.0,
        "default_leads_excluded_count": int(sum(default_excluded_by_county.values())),
        "default_leads_excluded_pct": round((sum(default_excluded_by_county.values()) / processed_rows) * 100.0, 2) if processed_rows else 0.0,
        "top_counties_by_review_excluded_count": [
            {"county_name": county_name, "row_count": int(count)}
            for county_name, count in review_excluded_by_county.most_common(10)
        ],
        "top_counties_by_training_excluded_count": [
            {"county_name": county_name, "row_count": int(count)}
            for county_name, count in training_excluded_by_county.most_common(10)
        ],
    }
    _write_json(artifacts.geometry_quality_summary_path, geometry_quality_summary)

    runtime_sections = _summary_sections(app_ready, geometry_flag_counts, marketability_action_counts)
    runtime_summary = {
        "row_count": int(len(app_ready)),
        "source": f"Arkansas MVP app_ready parcel runtime built from {BEST_SOURCE_NAME}",
        "geometry_mode": "selected_parcel_geojson",
        "mvp_warnings": [
            SOURCE_WARNING,
            "Arkansas PMTiles parcel basemap generation is deferred.",
            "Arkansas tax and vacancy-imagery enrichment are deferred in the first pass.",
        ],
        "statewide_master_row_count": processed_rows,
        "app_ready_subset_limit": int(args.max_app_ready),
        "geometry_quality_diagnostics": geometry_quality_summary,
        "geometry_marketability_diagnostics": {
            "row_count": processed_rows,
            "proxy_dimensions_from_area_perimeter_flag": True,
            "geometry_marketability_flag_counts": dict(sorted(marketability_flag_counts.items())),
            "geometry_marketability_action_counts": dict(sorted(marketability_action_counts.items())),
            "default_leads_excluded_count": int(sum(default_excluded_by_county.values())),
            "default_leads_excluded_pct": round((sum(default_excluded_by_county.values()) / processed_rows) * 100.0, 2) if processed_rows else 0.0,
            "top_counties_affected": {
                county_name: int(count) for county_name, count in default_excluded_by_county.most_common(10)
            },
        },
        "sections": runtime_sections,
    }
    presets_payload = _preset_payload(app_ready)
    default_leads_frame = _top_records(app_ready, limit=int(args.default_leads_limit))
    default_leads_payload = {
        "total_count": int(len(app_ready)),
        "limit": int(args.default_leads_limit),
        "offset": 0,
        "items": [
            serialize_contract_row(record, API_LEADS_SUMMARY_FIELDS, serializer=_json_ready)
            for record in default_leads_frame.to_dict(orient="records")
        ],
    }
    default_geometry_payload = _default_geometry_payload(default_leads_frame)
    meta_payload = _meta_payload(
        app_ready=app_ready,
        summary_payload=runtime_summary,
        presets_payload=presets_payload,
        app_ready_path=artifacts.app_ready_path,
    )
    fallback_frame = _top_records(app_ready, limit=int(args.frontend_fallback_limit))
    fallback_payload = _frontend_fallback_rows(fallback_frame)

    stage_start = time.perf_counter()
    _write_json(artifacts.runtime_summary_path, runtime_summary)
    _write_json(artifacts.runtime_presets_path, presets_payload)
    _write_json(artifacts.runtime_default_leads_path, default_leads_payload)
    _write_json(artifacts.runtime_default_geometry_path, default_geometry_payload)
    _write_json(artifacts.frontend_meta_path, meta_payload)
    _write_json(artifacts.frontend_detail_fallback_path, fallback_payload)
    _write_json(artifacts.frontend_static_feed_path, default_leads_payload)
    stage_runtimes["write_runtime_json_seconds"] = round(time.perf_counter() - stage_start, 3)

    diagnostics_output = artifacts.training_root / "state_diagnostics.json"
    diagnostics_payload = build_state_diagnostics(STATE_CODE)
    _write_json(diagnostics_output, diagnostics_payload)

    print(
        json.dumps(
            {
                "state_code": STATE_CODE,
                "parcel_master_rows": processed_rows,
                "app_ready_rows": int(len(app_ready)),
                "county_coverage_count": len(county_counts),
                "top_counties": [
                    {"county_name": county_name, "row_count": int(count)}
                    for county_name, count in county_counts.most_common(10)
                ],
                "artifact_paths": {
                    "parcel_master": _relative_path(artifacts.parcel_master_path),
                    "app_ready": _relative_path(artifacts.app_ready_path),
                    "runtime_summary": _relative_path(artifacts.runtime_summary_path),
                    "frontend_detail_fallback": _relative_path(artifacts.frontend_detail_fallback_path),
                    "state_diagnostics": _relative_path(diagnostics_output),
                },
                "stage_runtimes_seconds": stage_runtimes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
