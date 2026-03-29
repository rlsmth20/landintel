from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass
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
    FRONTEND_FALLBACK_REQUIRED_FIELDS,
    GEOMETRY_FEATURE_PROPERTY_FIELDS,
    GEOMETRY_ITEM_FIELDS,
    serialize_contract_row,
    validate_required_columns,
)
from state_artifacts import load_state_artifacts
from state_diagnostics import build_state_diagnostics
from state_registry import ROOT, ensure_state_directories, load_state_definition


DEFAULT_BATCH_SIZE = 32_000
DEFAULT_GEOMETRY_BATCH_SIZE = 8_000
DEFAULT_COORDINATE_BATCH_SIZE = 250
DEFAULT_MAX_APP_READY = 50_000
DEFAULT_DEFAULT_LEADS_LIMIT = 200
DEFAULT_FRONTEND_FALLBACK_LIMIT = 5_000
DEFAULT_MIN_APP_READY_SCORE = 35.0
DEFAULT_MIN_APP_READY_ACRES = 0.25
SQ_METERS_PER_ACRE = 4046.8564224
FEET_PER_METER = 3.28084
BEST_SOURCE_TYPE = "statewide_arcgis_feature_layer"

DEFAULT_FIELD_READINESS = [
    {
        "field_name": "parcel_vacant_flag",
        "readiness": "partial",
        "ui_guidance": "Use as a parcel-attribute proxy, not as an imagery-vacancy replacement.",
        "notes": "First-pass non-Mississippi runtimes infer improvement status from parcel value fields when available.",
    },
    {
        "field_name": "road_access_tier",
        "readiness": "hide_from_default_ui",
        "ui_guidance": "Hide until state-specific road enrichment exists.",
        "notes": "Road-distance enrichment is not included in the first statewide ArcGIS MVP profile.",
    },
    {
        "field_name": "delinquent_amount",
        "readiness": "hide_from_default_ui",
        "ui_guidance": "Hide until tax-source onboarding is implemented for the state.",
        "notes": "Parcel runtime MVP does not include delinquent-tax enrichment.",
    },
    {
        "field_name": "geometry_marketability_flag",
        "readiness": "production_ready",
        "ui_guidance": "Safe to surface as contextual lead-quality guidance.",
        "notes": "Marketability uses proxy dimensions from source area/perimeter attributes in the statewide ArcGIS profile.",
    },
]

DEFAULT_PRESET_DEFINITIONS = {
    "general_ranked": {
        "description": "Top-ranked statewide parcel leads from the official parcel source.",
        "filter_expression": "recommended_view_bucket = 'general_ranked'",
    },
    "vacant_land_targeting": {
        "description": "Likely vacant parcels with acceptable geometry and usable centroid coverage.",
        "filter_expression": "parcel_vacant_flag = true AND recommended_view_bucket = 'vacant_land_targeting'",
    },
    "larger_land_targeting": {
        "description": "Larger-acreage parcels in the current app_ready subset.",
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
FLOAT_COLUMNS = [
    "source_object_id",
    "acreage",
    "area_acres",
    "perimeter_meters",
    "bounding_box_width_meters",
    "bounding_box_height_meters",
    "aspect_ratio",
    "compactness",
    "geometry_estimated_frontage_feet",
    "geometry_estimated_width_feet",
    "geometry_min_dimension_feet",
    "geometry_max_dimension_feet",
    "geometry_frontage_to_width_ratio",
    "assessed_land_value",
    "assessed_improvement_value",
    "assessed_total_value",
    "building_area_total",
    "road_distance_ft",
    "flood_risk_score",
    "shape_compactness",
    "parcel_frontage_ft_estimate",
    "parcel_width_ft_estimate",
    "buildability_score",
    "environment_score",
    "investment_score",
    "delinquent_amount",
    "parcel_tax_status_confidence",
    "parcel_tax_years_stale",
    "lead_score_total",
    "lead_score_total_effective",
    "parcel_improvement_confidence",
    "latitude",
    "longitude",
]


@dataclass(frozen=True)
class RuntimeProfile:
    state_code: str
    state_name: str
    source_name: str
    service_url: str
    object_id_field: str
    geometry_out_fields: str
    attribute_out_fields: list[str]
    count_url: str
    query_url: str
    county_division_label: str
    coordinate_mode: str
    batch_size_hint: int
    geometry_batch_size_hint: int
    coordinate_batch_size_hint: int
    field_map: dict[str, Any]
    county_name_domain: dict[str, str]
    county_fips_lookup: dict[str, str]
    source_confidence_tier: str
    county_source_coverage_tier: str
    source_warning: str
    notes: list[str]
    field_readiness: list[dict[str, Any]]
    preset_definitions: dict[str, dict[str, str]]
    app_ready_min_score: float
    app_ready_min_acres: float
    default_bounds: list[float] | None
    default_lead_limit: int
    frontend_fallback_limit: int
    null_parcel_id_values: set[str]
    parcel_pmtiles_ready: bool
    vacancy_proxy_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build statewide ArcGIS parcel runtime artifacts for a configured state.")
    parser.add_argument("--state-code", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--geometry-batch-size", type=int, default=DEFAULT_GEOMETRY_BATCH_SIZE)
    parser.add_argument("--coordinate-batch-size", type=int, default=DEFAULT_COORDINATE_BATCH_SIZE)
    parser.add_argument("--max-app-ready", type=int, default=DEFAULT_MAX_APP_READY)
    parser.add_argument("--default-leads-limit", type=int, default=None)
    parser.add_argument("--frontend-fallback-limit", type=int, default=None)
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
    return float(numeric)


def _safe_int(value: Any) -> int | None:
    numeric = _safe_float(value)
    return None if numeric is None else int(numeric)


def _json_ready(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_slug(value: Any) -> str | None:
    text = _safe_string(value)
    if not text:
        return None
    normalized = text.lower().replace(".", "").replace("&", "and")
    normalized = normalized.replace("-", " ").replace("/", " ")
    normalized = "_".join(part for part in normalized.split() if part)
    return normalized or None


def _expanded_county_fips_lookup(raw_lookup: dict[str, Any]) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for key, value in raw_lookup.items():
        if value is None:
            continue
        normalized = _normalize_slug(key) or str(key).strip().lower()
        if not normalized:
            continue
        normalized_value = str(value).zfill(5)
        expanded[normalized] = normalized_value
        condensed = normalized.replace("_", "")
        if condensed:
            expanded[condensed] = normalized_value
    return expanded


def _county_fips_from_display(
    county_name_display: str | None,
    *,
    lookup: dict[str, str],
    raw_value: Any = None,
    prefix: str | None = None,
) -> str | None:
    raw_text = _safe_string(raw_value)
    if raw_text:
        digits = "".join(character for character in raw_text if character.isdigit())
        if digits:
            if prefix and len(digits) <= 3:
                return f"{prefix}{digits.zfill(3)}"
            return digits.zfill(5)
    county_slug = _normalize_slug(county_name_display)
    if not county_slug:
        return None
    mapped = lookup.get(county_slug)
    if mapped is None:
        mapped = lookup.get(county_slug.replace("_", ""))
    return str(mapped).zfill(5) if mapped else None


def _parcel_row_id(state_code: str, county_fips: str | None, parcel_id: str | None) -> str | None:
    if not county_fips or not parcel_id:
        return None
    digest = hashlib.sha1(f"{state_code}|{county_fips}|{parcel_id.strip().upper()}".encode("utf-8")).hexdigest()[:16]
    return f"{state_code}_{digest}"


def _corporate_owner_flag(owner_name: str | None) -> bool | None:
    if not owner_name:
        return None
    normalized = owner_name.upper()
    return any(token in normalized for token in (" LLC", " INC", " CORP", " CO", " LP", " LTD", " TRUST", " BANK", " STATE OF "))


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


def _resolve_vacancy_proxy(
    *,
    assessed_improvement_value: float | None,
    assessed_total_value: float | None,
    mode: str,
    state_name: str,
) -> tuple[bool | None, str, float | None, str, str]:
    normalized_mode = str(mode).strip().lower() or "assessed_improvement_ratio"
    if normalized_mode == "disabled":
        return (
            None,
            "needs_review",
            None,
            "improvement_valuation_unavailable",
            f"{state_name} statewide parcel source does not provide separate improvement valuation in this MVP runtime.",
        )
    return _vacancy_proxy(
        assessed_improvement_value=assessed_improvement_value,
        assessed_total_value=assessed_total_value,
    )


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_output_path(path)
    if temp_path.exists():
        temp_path.unlink()
    temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def _temp_output_path(path: Path) -> Path:
    return path.parent / f"{path.name}.building"


def _write_partitioned_index(frame: pd.DataFrame, output_root: Path) -> None:
    temp_root = _temp_output_path(output_root)
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(temp_root),
        format="parquet",
        partitioning=["county_name"],
        existing_data_behavior="overwrite_or_ignore",
    )
    if output_root.exists():
        shutil.rmtree(output_root, ignore_errors=True)
    temp_root.replace(output_root)


def _top_records(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.sort_values(
        by=["lead_score_total_effective", "lead_score_total", "acreage", "parcel_row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.head(limit).copy()


def _feature_payload(row: dict[str, Any], *, selected: bool = False) -> dict[str, Any]:
    payload = serialize_contract_row(row, GEOMETRY_FEATURE_PROPERTY_FIELDS, serializer=_json_ready)
    payload["selected"] = selected
    return payload


def _geometry_item(row: dict[str, Any]) -> dict[str, Any]:
    return serialize_contract_row(row, GEOMETRY_ITEM_FIELDS, serializer=_json_ready)


def _fetch_json(session: requests.Session, url: str, *, params: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)
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
    raise RuntimeError(f"ArcGIS query failed: {last_error}") from last_error


def _load_source_registry(state_code: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    definition = load_state_definition(state_code)
    registry_path = definition.source_registry_path("parcel_source")
    if registry_path is None or not registry_path.exists():
        raise FileNotFoundError(f"Parcel source registry is missing for {state_code}: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    source = next((item for item in registry.get("parcel_sources", []) if item.get("primary")), None)
    if source is None:
        raise RuntimeError(f"Parcel source registry has no primary parcel source configured for {state_code}.")
    runtime_builder = source.get("runtime_builder") or {}
    if str(runtime_builder.get("profile", "")).lower() != "statewide_arcgis_parcel_mvp_v1":
        raise RuntimeError(f"Primary parcel source for {state_code} is missing runtime_builder.profile=statewide_arcgis_parcel_mvp_v1")
    return registry, source, runtime_builder


def _service_metadata(session: requests.Session, source: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    return _fetch_json(session, str(source["service_url"]).rstrip("/"), params={"f": "json"}, timeout_seconds=timeout_seconds)


def _domain_map_from_metadata(metadata: dict[str, Any], field_name: str) -> dict[str, str]:
    normalized_field = str(field_name).lower()
    for field in metadata.get("fields", []):
        if str(field.get("name", "")).lower() != normalized_field:
            continue
        domain = field.get("domain") or {}
        coded_values = domain.get("codedValues") or []
        return {str(item.get("code")): str(item.get("name")) for item in coded_values if item.get("name") is not None}
    return {}


def _load_profile(state_code: str, *, session: requests.Session, timeout_seconds: float) -> RuntimeProfile:
    definition = load_state_definition(state_code)
    _, source, runtime_builder = _load_source_registry(state_code)
    metadata = _service_metadata(session, source, timeout_seconds=timeout_seconds)
    field_map = dict(runtime_builder.get("field_map") or {})
    county_name_domain = {
        str(key): str(value)
        for key, value in (runtime_builder.get("county_name_domain") or {}).items()
        if value is not None
    }
    county_name_domain_field = _safe_string(runtime_builder.get("county_name_domain_field"))
    if not county_name_domain and county_name_domain_field:
        county_name_domain = _domain_map_from_metadata(metadata, county_name_domain_field)

    county_fips_lookup = _expanded_county_fips_lookup(runtime_builder.get("county_fips_lookup") or {})
    attribute_out_fields = list(source.get("attribute_out_fields") or runtime_builder.get("attribute_out_fields") or [])
    if not attribute_out_fields:
        raise RuntimeError(f"Primary parcel source for {state_code} has no attribute_out_fields configured.")

    object_id_field = str(source.get("object_id_field") or metadata.get("objectIdField") or "OBJECTID")
    geometry_out_fields = str(source.get("geometry_out_fields") or runtime_builder.get("geometry_out_fields") or object_id_field)
    batch_hint = _safe_int(runtime_builder.get("batch_size_hint")) or _safe_int(metadata.get("standardMaxRecordCountNoGeometry")) or DEFAULT_BATCH_SIZE
    geometry_batch_hint = _safe_int(runtime_builder.get("geometry_batch_size_hint")) or _safe_int(metadata.get("standardMaxRecordCount")) or DEFAULT_GEOMETRY_BATCH_SIZE
    coordinate_batch_hint = _safe_int(runtime_builder.get("coordinate_batch_size_hint")) or _safe_int(source.get("max_object_ids_per_query")) or DEFAULT_COORDINATE_BATCH_SIZE
    null_values = {str(value).strip().upper() for value in (runtime_builder.get("parcel_id_null_values") or []) if str(value).strip()}
    notes = [str(note) for note in runtime_builder.get("notes", []) if str(note).strip()]

    return RuntimeProfile(
        state_code=definition.state_code,
        state_name=definition.state_name,
        source_name=str(runtime_builder.get("source_name") or source.get("source_name") or definition.state_name),
        service_url=str(source["service_url"]).rstrip("/"),
        object_id_field=object_id_field,
        geometry_out_fields=geometry_out_fields,
        attribute_out_fields=attribute_out_fields,
        count_url=str(source["service_url"]).rstrip("/") + "/query",
        query_url=str(source["service_url"]).rstrip("/") + "/query",
        county_division_label=definition.county_division_label,
        coordinate_mode=str(runtime_builder.get("coordinate_mode") or "fields"),
        batch_size_hint=int(batch_hint),
        geometry_batch_size_hint=int(geometry_batch_hint),
        coordinate_batch_size_hint=int(coordinate_batch_hint),
        field_map=field_map,
        county_name_domain=county_name_domain,
        county_fips_lookup=county_fips_lookup,
        source_confidence_tier=str(runtime_builder.get("source_confidence_tier") or "medium"),
        county_source_coverage_tier=str(runtime_builder.get("county_source_coverage_tier") or "statewide_primary"),
        source_warning=str(
            runtime_builder.get("source_warning")
            or f"{definition.state_name} MVP uses the official statewide ArcGIS parcel source with proxy geometry metrics."
        ),
        notes=notes,
        field_readiness=list(runtime_builder.get("field_readiness") or DEFAULT_FIELD_READINESS),
        preset_definitions=dict(runtime_builder.get("preset_definitions") or DEFAULT_PRESET_DEFINITIONS),
        app_ready_min_score=float(runtime_builder.get("app_ready_min_score") or DEFAULT_MIN_APP_READY_SCORE),
        app_ready_min_acres=float(runtime_builder.get("app_ready_min_acres") or DEFAULT_MIN_APP_READY_ACRES),
        default_bounds=list(runtime_builder.get("default_bounds") or []) or None,
        default_lead_limit=_safe_int(runtime_builder.get("default_lead_limit")) or DEFAULT_DEFAULT_LEADS_LIMIT,
        frontend_fallback_limit=_safe_int(runtime_builder.get("frontend_fallback_limit")) or DEFAULT_FRONTEND_FALLBACK_LIMIT,
        null_parcel_id_values=null_values,
        parcel_pmtiles_ready=bool(runtime_builder.get("parcel_pmtiles_ready", True)),
        vacancy_proxy_mode=str(runtime_builder.get("vacancy_proxy_mode") or "assessed_improvement_ratio"),
    )


def _source_attributes(feature: dict[str, Any]) -> dict[str, Any]:
    attributes = feature.get("attributes") or feature.get("properties") or {}
    return {str(key).lower(): value for key, value in attributes.items()}


def _mapped_value(attributes: dict[str, Any], mapping: Any) -> Any:
    if mapping is None:
        return None
    if isinstance(mapping, list):
        for field_name in mapping:
            value = _mapped_value(attributes, field_name)
            if _safe_string(value):
                return value
        return None
    if isinstance(mapping, dict):
        if "coalesce" in mapping:
            return _mapped_value(attributes, mapping.get("coalesce"))
        value = _mapped_value(attributes, mapping.get("field"))
        if value is None:
            return None
        text = str(value)
        split_token = mapping.get("split")
        if split_token is not None:
            parts = text.split(str(split_token))
            index = int(mapping.get("index", 0))
            if index < 0 or index >= len(parts):
                return None
            text = parts[index]
        prefix_length = mapping.get("prefix_length")
        if prefix_length is not None:
            text = text[: int(prefix_length)]
        suffix_length = mapping.get("suffix_length")
        if suffix_length is not None:
            text = text[-int(suffix_length) :]
        if mapping.get("strip", True):
            text = text.strip()
        return text or None
    return attributes.get(str(mapping).lower())


def _parcel_id(attributes: dict[str, Any], profile: RuntimeProfile) -> str | None:
    candidates = profile.field_map.get("parcel_id_fields")
    if not candidates:
        candidates = [profile.field_map.get("parcel_id")]
    if not isinstance(candidates, list):
        candidates = [candidates]
    for field_name in candidates:
        raw = _mapped_value(attributes, field_name)
        text = _safe_string(raw)
        if not text:
            continue
        if text.strip().upper() in profile.null_parcel_id_values:
            continue
        return text
    return None


def _county_display(attributes: dict[str, Any], profile: RuntimeProfile) -> str | None:
    county_name_field = profile.field_map.get("county_name")
    county_code_field = profile.field_map.get("county_name_code") or profile.field_map.get("county_code")
    display = _safe_string(_mapped_value(attributes, county_name_field))
    if display:
        return display
    if county_code_field:
        code_value = _mapped_value(attributes, county_code_field)
        return profile.county_name_domain.get(str(code_value))
    return None


def _coordinates_from_feature(feature: dict[str, Any], profile: RuntimeProfile) -> tuple[float | None, float | None]:
    geometry = feature.get("geometry") or {}
    if profile.coordinate_mode in {"point_geometry", "selected_return_geometry_point"} and geometry:
        return _safe_float(geometry.get("y")), _safe_float(geometry.get("x"))
    if profile.coordinate_mode in {"fields", "selected_return_centroid"}:
        attributes = _source_attributes(feature)
        latitude = _safe_float(_mapped_value(attributes, profile.field_map.get("latitude")))
        longitude = _safe_float(_mapped_value(attributes, profile.field_map.get("longitude")))
        if latitude is not None and longitude is not None:
            return latitude, longitude
    centroid = feature.get("centroid") or {}
    return _safe_float(centroid.get("y")), _safe_float(centroid.get("x"))


def _transform_feature_batch(
    features: list[dict[str, Any]],
    *,
    profile: RuntimeProfile,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    county_fips_field = profile.field_map.get("county_fips")
    county_fips_prefix = _safe_string(profile.field_map.get("county_fips_prefix"))
    for feature in features:
        attributes = _source_attributes(feature)
        source_object_id = _safe_int(_mapped_value(attributes, profile.field_map.get("source_object_id") or profile.object_id_field))
        county_name_display = _county_display(attributes, profile)
        county_name = _normalize_slug(county_name_display)
        county_fips = _county_fips_from_display(
            county_name_display,
            lookup=profile.county_fips_lookup,
            raw_value=_mapped_value(attributes, county_fips_field),
            prefix=county_fips_prefix,
        )
        parcel_id = _parcel_id(attributes, profile)
        if not county_name or not county_fips or not parcel_id:
            continue

        owner_name = _safe_string(_mapped_value(attributes, profile.field_map.get("owner_name")))
        site_address = _safe_string(_mapped_value(attributes, profile.field_map.get("site_address")))
        land_use = _safe_string(_mapped_value(attributes, profile.field_map.get("land_use")))
        subdivision_name = _safe_string(_mapped_value(attributes, profile.field_map.get("subdivision_name")))
        acreage = _safe_float(_mapped_value(attributes, profile.field_map.get("acreage")))
        area_square_meters = _safe_float(_mapped_value(attributes, profile.field_map.get("area_square_meters")))
        perimeter_meters = _safe_float(_mapped_value(attributes, profile.field_map.get("perimeter_meters")))
        if acreage is None and area_square_meters is not None and area_square_meters > 0:
            acreage = area_square_meters / SQ_METERS_PER_ACRE

        min_dimension_meters, max_dimension_meters, aspect_ratio = _proxy_dimensions(area_square_meters, perimeter_meters)
        compactness = _compactness(area_square_meters, perimeter_meters)
        min_dimension_feet = min_dimension_meters * FEET_PER_METER if min_dimension_meters is not None else None
        max_dimension_feet = max_dimension_meters * FEET_PER_METER if max_dimension_meters is not None else None
        geometry_quality_flag = _geometry_quality_flag(
            area_acres=acreage,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            min_dimension_meters=min_dimension_meters,
            land_use=land_use,
        )
        geometry_marketability_base_flag, geometry_marketability_flag, geometry_marketability_action, geometry_penalty_points, geometry_penalty_reason = _marketability(
            geometry_quality_flag=geometry_quality_flag,
            area_acres=acreage,
            compactness=compactness,
            aspect_ratio=aspect_ratio,
            min_dimension_feet=min_dimension_feet,
            width_feet=min_dimension_feet,
            frontage_to_width_ratio=aspect_ratio,
            land_use=land_use,
        )

        assessed_land_value = _safe_float(_mapped_value(attributes, profile.field_map.get("assessed_land_value")))
        assessed_improvement_value = _safe_float(_mapped_value(attributes, profile.field_map.get("assessed_improvement_value")))
        assessed_total_value = _safe_float(_mapped_value(attributes, profile.field_map.get("assessed_total_value")))
        parcel_vacant_flag, parcel_improvement_status, parcel_improvement_confidence, parcel_improvement_reason, parcel_improvement_summary = _resolve_vacancy_proxy(
            assessed_improvement_value=assessed_improvement_value,
            assessed_total_value=assessed_total_value,
            mode=profile.vacancy_proxy_mode,
            state_name=profile.state_name,
        )
        corporate_owner_flag = _corporate_owner_flag(owner_name)
        vacancy_score, size_score, value_score, owner_score, score_component = _lead_score_components(
            acreage=acreage,
            parcel_vacant_flag=parcel_vacant_flag,
            assessed_total_value=assessed_total_value,
            marketability_penalty_points=geometry_penalty_points,
            corporate_owner_flag=corporate_owner_flag,
        )
        lead_score_total = round(60.0 + vacancy_score + size_score + value_score + owner_score + score_component, 2)
        recommended_view_bucket = _recommended_view_bucket(acreage, parcel_vacant_flag)
        latitude, longitude = _coordinates_from_feature(feature, profile)
        caution_flags = "; ".join(part for part in [profile.source_warning] + profile.notes if part)
        top_score_driver = "vacancy_proxy" if parcel_vacant_flag is True else "proxy_geometry_marketability"

        rows.append(
            {
                "parcel_row_id": _parcel_row_id(profile.state_code, county_fips, parcel_id),
                "parcel_id": parcel_id,
                "state_code": profile.state_code,
                "county_name": county_name,
                "county_name_display": county_name_display,
                "county_fips": county_fips,
                "source_object_id": source_object_id,
                "owner_name": owner_name,
                "site_address": site_address,
                "land_use": land_use,
                "subdivision_name": subdivision_name,
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
                "geometry_review_excluded_flag": False,
                "geometry_training_excluded_flag": False,
                "geometry_default_leads_excluded_flag": False,
                "geometry_estimated_frontage_feet": max_dimension_feet,
                "geometry_estimated_width_feet": min_dimension_feet,
                "geometry_min_dimension_feet": min_dimension_feet,
                "geometry_max_dimension_feet": max_dimension_feet,
                "geometry_frontage_to_width_ratio": aspect_ratio,
                "geometry_effective_buildable_flag": geometry_marketability_action != "exclude",
                "geometry_marketability_base_flag": geometry_marketability_base_flag,
                "geometry_marketability_flag": geometry_marketability_flag,
                "geometry_marketability_context": "proxy_statewide_arcgis",
                "geometry_marketability_action": geometry_marketability_action,
                "geometry_penalty_points": geometry_penalty_points,
                "geometry_penalty_reason": geometry_penalty_reason,
                "geometry_marketability_default_leads_excluded_flag": geometry_marketability_action == "exclude",
                "assessed_land_value": assessed_land_value,
                "assessed_improvement_value": assessed_improvement_value,
                "assessed_total_value": assessed_total_value,
                "parcel_vacant_flag": parcel_vacant_flag,
                "county_vacant_flag": None,
                "building_count": 0,
                "building_area_total": None,
                "growth_pressure_bucket": None,
                "road_distance_ft": None,
                "road_access_tier": None,
                "wetland_flag": None,
                "flood_risk_score": None,
                "shape_compactness": compactness,
                "parcel_frontage_ft_estimate": max_dimension_feet,
                "parcel_width_ft_estimate": min_dimension_feet,
                "buildability_score": max(0.0, min(100.0, 85.0 + geometry_penalty_points)),
                "environment_score": None,
                "investment_score": lead_score_total,
                "corporate_owner_flag": corporate_owner_flag,
                "absentee_owner_flag": None,
                "out_of_state_owner_flag": None,
                "delinquent_amount": None,
                "county_tax_source_configured_flag": False,
                "county_tax_source_loaded_flag": False,
                "tax_data_available_flag": False,
                "county_tax_coverage_status": "not_configured",
                "county_tax_coverage_reason": f"{profile.state_name} tax sources are not onboarded in the first MVP runtime.",
                "parcel_tax_status": None,
                "parcel_tax_status_label": None,
                "parcel_tax_status_confidence": None,
                "parcel_tax_status_category": None,
                "parcel_tax_actionability": None,
                "parcel_tax_data_warning": f"{profile.state_name} tax-source enrichment is not active in the first MVP runtime.",
                "parcel_tax_freshness_bucket": None,
                "parcel_tax_years_stale": None,
                "parcel_tax_is_actionable_current": False,
                "parcel_tax_is_historical_only": False,
                "parcel_tax_freshness_reason": None,
                "best_source_type": BEST_SOURCE_TYPE,
                "best_source_name": profile.source_name,
                "source_confidence_tier": profile.source_confidence_tier,
                "county_source_coverage_tier": profile.county_source_coverage_tier,
                "amount_trust_tier": None,
                "high_confidence_link_flag": False,
                "county_hosted_flag": False,
                "lead_score_total": lead_score_total,
                "lead_score_total_effective": lead_score_total,
                "lead_score_tier": _lead_score_tier(lead_score_total),
                "lead_score_driver_1": "vacancy_proxy",
                "lead_score_driver_2": "proxy_geometry_marketability",
                "lead_score_driver_3": "size_and_value",
                "lead_score_explanation": f"{profile.state_name} MVP score combines parcel-value vacancy proxies, proxy geometry marketability, parcel size, and source confidence.",
                "recommended_sort_reason": f"{profile.state_code}_statewide_arcgis_mvp_runtime",
                "top_score_driver": top_score_driver,
                "caution_flags": caution_flags,
                "vacant_reason": parcel_improvement_summary,
                "recommended_use_case": f"{profile.state_name} MVP statewide parcel screening",
                "recommended_view_bucket": recommended_view_bucket,
                "parcel_improvement_status": parcel_improvement_status,
                "parcel_improvement_confidence": parcel_improvement_confidence,
                "parcel_improvement_reason": parcel_improvement_reason,
                "parcel_improvement_evidence_summary": parcel_improvement_summary,
                "ai_vacancy_available_flag": False,
                "ai_vacancy_source": "unavailable",
                "ai_vacancy_status_note": f"{profile.state_name} vacancy imagery scoring is not onboarded in the first MVP runtime.",
                "overall_vacancy_assessment": parcel_improvement_status,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in STRING_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("string")
    for column in FLOAT_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("boolean")
    for column in INTEGER_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["parcel_row_id", "parcel_id", "county_name", "county_fips"]).copy()
    frame = frame.drop_duplicates(subset=["parcel_row_id"], keep="first").copy()
    return frame


def _query_batch(
    session: requests.Session,
    profile: RuntimeProfile,
    *,
    offset: int,
    batch_size: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    params = {
        "where": "1=1",
        "outFields": ",".join(profile.attribute_out_fields),
        "orderByFields": f"{profile.object_id_field} ASC",
        "resultOffset": str(offset),
        "resultRecordCount": str(batch_size),
        "resultType": "standard",
        "f": "json",
    }
    if profile.coordinate_mode == "point_geometry":
        params["returnGeometry"] = "true"
        params["outSR"] = "4326"
    else:
        params["returnGeometry"] = "false"
    return _fetch_json(session, profile.query_url, params=params, timeout_seconds=timeout_seconds)


def _escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def _source_alignment_audit(
    session: requests.Session,
    profile: RuntimeProfile,
    *,
    total_source_rows: int,
    local_county_counts: Counter[str],
    local_parcel_row_count: int,
    duplicate_parcel_row_id_rows_removed: int,
    timeout_seconds: float,
    limited_build: bool,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "state_code": profile.state_code,
        "state_name": profile.state_name,
        "source_name": profile.source_name,
        "service_url": profile.service_url,
        "official_source_row_count": int(total_source_rows),
        "local_parcel_master_row_count": int(local_parcel_row_count),
        "row_count_gap": int(total_source_rows - local_parcel_row_count),
        "row_count_gap_pct": round(((total_source_rows - local_parcel_row_count) / total_source_rows) * 100.0, 4) if total_source_rows else None,
        "duplicate_parcel_row_id_rows_removed": int(duplicate_parcel_row_id_rows_removed),
        "local_county_count": int(len(local_county_counts)),
        "build_limited_by_request": bool(limited_build),
        "generated_at": _now_iso(),
    }
    county_name_field = profile.field_map.get("county_name")
    if not isinstance(county_name_field, str) or not county_name_field.strip():
        audit["official_county_count"] = None
        audit["missing_counties"] = []
        audit["county_row_gaps"] = []
        audit["source_parity_status"] = "limited_by_request" if limited_build else ("aligned" if local_parcel_row_count >= total_source_rows else "subset")
        return audit

    distinct_payload = _fetch_json(
        session,
        profile.query_url,
        params={
            "where": "1=1",
            "returnDistinctValues": "true",
            "returnGeometry": "false",
            "outFields": county_name_field,
            "orderByFields": f"{county_name_field} ASC",
            "f": "json",
        },
        timeout_seconds=timeout_seconds,
    )
    distinct_display_values: list[str] = []
    for feature in distinct_payload.get("features", []):
        attributes = _source_attributes(feature)
        display_value = _safe_string(attributes.get(county_name_field.lower()))
        if display_value:
            distinct_display_values.append(display_value)
    ordered_display_values = sorted(dict.fromkeys(distinct_display_values))

    county_row_gaps: list[dict[str, Any]] = []
    for index, display_value in enumerate(ordered_display_values):
        count_payload = _fetch_json(
            session,
            profile.count_url,
            params={
                "where": f"{county_name_field} = '{_escape_sql_string(display_value)}'",
                "returnCountOnly": "true",
                "f": "json",
            },
            timeout_seconds=timeout_seconds,
        )
        official_row_count = int(count_payload.get("count") or 0)
        normalized_county_name = _normalize_slug(display_value) or display_value.strip().lower()
        local_row_count = int(local_county_counts.get(normalized_county_name, 0))
        county_row_gaps.append(
            {
                "county_name_display": display_value,
                "county_name": normalized_county_name,
                "official_row_count": official_row_count,
                "local_row_count": local_row_count,
                "row_count_gap": int(official_row_count - local_row_count),
                "coverage": "aligned" if local_row_count >= official_row_count else "subset" if local_row_count > 0 else "missing",
            }
        )
        if index and index % 10 == 0:
            print(f"Audited {index + 1:,}/{len(ordered_display_values):,} {profile.state_code.upper()} source counties")

    missing_counties = [row["county_name"] for row in county_row_gaps if row["coverage"] == "missing"]
    partial_counties = [row["county_name"] for row in county_row_gaps if row["coverage"] == "subset"]
    audit["official_county_count"] = int(len(ordered_display_values))
    audit["missing_counties"] = missing_counties
    audit["partial_counties"] = partial_counties
    audit["county_row_gaps"] = county_row_gaps
    audit["top_row_gaps"] = sorted(county_row_gaps, key=lambda row: row["row_count_gap"], reverse=True)[:20]
    audit["canonical_county_coverage"] = "full" if not missing_counties and len(local_county_counts) >= len(ordered_display_values) else "subset"
    audit["raw_source_parity_status"] = (
        "limited_by_request"
        if limited_build
        else "aligned"
        if not missing_counties and not partial_counties and local_parcel_row_count >= total_source_rows
        else "subset"
    )
    audit["source_parity_status"] = audit["raw_source_parity_status"]
    return audit


def _coordinate_lookup(
    session: requests.Session,
    profile: RuntimeProfile,
    *,
    object_ids: list[int],
    timeout_seconds: float,
) -> dict[int, tuple[float | None, float | None]]:
    if not object_ids:
        return {}
    params = {
        "objectIds": ",".join(str(value) for value in object_ids),
        "outFields": profile.object_id_field,
        "outSR": "4326",
        "f": "json",
    }
    if profile.coordinate_mode == "selected_return_centroid":
        params["returnGeometry"] = "false"
        params["returnCentroid"] = "true"
    elif profile.coordinate_mode == "selected_return_geometry_point":
        params["returnGeometry"] = "true"
    else:
        return {}
    response = session.post(profile.query_url, data=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS coordinate query error: {payload['error']}")
    lookup: dict[int, tuple[float | None, float | None]] = {}
    for feature in payload.get("features", []):
        attributes = _source_attributes(feature)
        object_id = _safe_int(attributes.get(profile.object_id_field.lower()) or attributes.get("objectid"))
        if object_id is None:
            continue
        lookup[object_id] = _coordinates_from_feature(feature, profile)
    return lookup


def _enrich_selected_coordinates(
    frame: pd.DataFrame,
    *,
    session: requests.Session,
    profile: RuntimeProfile,
    timeout_seconds: float,
    batch_size: int,
) -> pd.DataFrame:
    if frame.empty or profile.coordinate_mode not in {"selected_return_centroid", "selected_return_geometry_point"}:
        return frame
    working = frame.copy()
    working["source_object_id"] = pd.to_numeric(working["source_object_id"], errors="coerce").astype("Int64")
    object_ids = working["source_object_id"].dropna().astype(int).tolist()
    coordinate_map: dict[int, tuple[float | None, float | None]] = {}
    for start in range(0, len(object_ids), batch_size):
        batch_ids = object_ids[start : start + batch_size]
        coordinate_map.update(_coordinate_lookup(session, profile, object_ids=batch_ids, timeout_seconds=timeout_seconds))
        if (start // batch_size) % 20 == 0:
            print(f"Fetched {min(start + len(batch_ids), len(object_ids)):,}/{len(object_ids):,} {profile.state_code.upper()} app_ready centroids")

    def _lookup(row: pd.Series, index: int) -> float | None:
        if pd.isna(row.get("source_object_id")):
            return _safe_float(row.get("latitude" if index == 0 else "longitude"))
        latitude, longitude = coordinate_map.get(int(row["source_object_id"]), (row.get("latitude"), row.get("longitude")))
        return latitude if index == 0 else longitude

    working["latitude"] = working.apply(lambda row: _lookup(row, 0), axis=1)
    working["longitude"] = working.apply(lambda row: _lookup(row, 1), axis=1)
    return working


def _preset_payload(frame: pd.DataFrame, preset_definitions: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for view_name, meta in preset_definitions.items():
        subset = frame.loc[frame.get("recommended_view_bucket") == view_name].copy() if "recommended_view_bucket" in frame.columns else pd.DataFrame(columns=frame.columns)
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
    county_counts = Counter(app_ready["county_name"].dropna().astype(str).tolist()) if "county_name" in app_ready.columns else Counter()
    bucket_counts = Counter(app_ready["recommended_view_bucket"].dropna().astype(str).tolist()) if "recommended_view_bucket" in app_ready.columns else Counter()
    statewide: list[dict[str, str]] = [
        {"section": "statewide", "metric": "lead_count", "key": "", "value": str(int(len(app_ready)))},
        {
            "section": "statewide",
            "metric": "average_lead_score",
            "key": "",
            "value": f"{float(app_ready['lead_score_total_effective'].mean()):.4f}" if not app_ready.empty else "0.0000",
        },
        {
            "section": "statewide",
            "metric": "likely_vacant_count",
            "key": "",
            "value": str(int(app_ready["parcel_vacant_flag"].fillna(False).astype(bool).sum())) if "parcel_vacant_flag" in app_ready.columns else "0",
        },
    ]
    top_counties = [
        {"section": "top_counties", "metric": "lead_count", "key": county_name, "value": str(int(count))}
        for county_name, count in county_counts.most_common(20)
    ]
    recommended_view_bucket = [
        {"section": "recommended_view_bucket", "metric": "lead_count", "key": bucket, "value": str(int(count))}
        for bucket, count in sorted(bucket_counts.items())
    ]
    geometry_quality_rows = [
        {"section": "geometry_quality_flag", "metric": "parcel_count", "key": key, "value": str(int(count))}
        for key, count in sorted(geometry_flag_counts.items())
    ]
    marketability_rows = [
        {"section": "geometry_marketability_action", "metric": "parcel_count", "key": key, "value": str(int(count))}
        for key, count in sorted(marketability_action_counts.items())
    ]
    return {
        "statewide": statewide,
        "top_counties": top_counties,
        "recommended_view_bucket": recommended_view_bucket,
        "geometry_quality_flag": geometry_quality_rows,
        "geometry_marketability_action": marketability_rows,
    }


def _summary_row_list(runtime_sections: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for items in runtime_sections.values():
        rows.extend(items)
    return rows


def _default_geometry_payload(frame: pd.DataFrame) -> dict[str, Any]:
    records = frame.to_dict(orient="records")
    features: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    bounds: list[float] | None = None
    for record in records:
        latitude = _safe_float(record.get("latitude"))
        longitude = _safe_float(record.get("longitude"))
        if latitude is None or longitude is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(longitude, 6), round(latitude, 6)]},
                "properties": _feature_payload(record, selected=True),
            }
        )
        items.append(_geometry_item(record))
        if bounds is None:
            bounds = [longitude, latitude, longitude, latitude]
        else:
            bounds = [min(bounds[0], longitude), min(bounds[1], latitude), max(bounds[2], longitude), max(bounds[3], latitude)]
    return {
        "geometry_mode": "selected_parcel_geojson",
        "render_mode": "points" if features else "none",
        "geometry_bounds": bounds,
        "geometry_view_box": None,
        "requested_bounds": None,
        "zoom": None,
        "feature_count": len(features),
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "items": items,
    }


def _meta_payload(
    *,
    app_ready: pd.DataFrame,
    summary_payload: dict[str, Any],
    presets_payload: list[dict[str, Any]],
    app_ready_path: Path,
    profile: RuntimeProfile,
) -> dict[str, Any]:
    summary_rows = _summary_row_list(summary_payload.get("sections", {}))
    default_views: list[dict[str, str]] = []
    for item in presets_payload:
        for metric in ("row_count", "average_lead_score"):
            value = item.get(metric)
            if value is None:
                continue
            default_views.append(
                {
                    "view_name": str(item["view_name"]),
                    "description": str(item["description"]),
                    "filter_expression": str(item["filter_expression"]),
                    "metric": metric,
                    "value": str(value),
                }
            )
    bounded = app_ready.dropna(subset=["longitude", "latitude"]).copy() if {"longitude", "latitude"}.issubset(app_ready.columns) else pd.DataFrame()
    geometry_bounds = (
        [
            round(float(bounded["longitude"].min()), 6),
            round(float(bounded["latitude"].min()), 6),
            round(float(bounded["longitude"].max()), 6),
            round(float(bounded["latitude"].max()), 6),
        ]
        if not bounded.empty
        else profile.default_bounds
    )
    return {
        "defaultViews": default_views,
        "fieldReadiness": profile.field_readiness,
        "summary": summary_rows,
        "rowCount": int(len(app_ready)),
        "source": str(app_ready_path),
        "geometryMode": "selected_parcel_geojson",
        "geometryBounds": geometry_bounds,
        "geometryViewBox": [0, 0, 1000, 700],
        "warnings": summary_payload.get("mvp_warnings", []),
        "geometrySimplifyTolerance": None,
    }


def _frontend_fallback_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    payload: list[dict[str, Any]] = []
    for record in records:
        row = {key: _json_ready(value) for key, value in record.items()}
        row["geometry"] = _point_geometry_payload(_safe_float(record.get("latitude")), _safe_float(record.get("longitude")))
        payload.append(row)
    validation_frame = pd.DataFrame(payload)
    if validation_frame.empty:
        validation_frame = pd.DataFrame(columns=FRONTEND_FALLBACK_REQUIRED_FIELDS)
    validate_required_columns(
        validation_frame,
        required_columns=FRONTEND_FALLBACK_REQUIRED_FIELDS,
        non_null_columns=["parcel_row_id", "parcel_id"],
        context="_frontend_fallback_rows",
    )
    return payload


def main() -> None:
    args = parse_args()
    state_code = str(args.state_code).strip().lower()
    definition = ensure_state_directories(state_code)
    artifacts = load_state_artifacts(state_code)

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
    session = requests.Session()
    profile = _load_profile(state_code, session=session, timeout_seconds=float(args.timeout_seconds))
    batch_size = min(int(args.batch_size), int(profile.batch_size_hint))
    default_leads_limit = int(args.default_leads_limit or profile.default_lead_limit)
    frontend_fallback_limit = int(args.frontend_fallback_limit or profile.frontend_fallback_limit)

    stage_start = time.perf_counter()
    count_payload = _fetch_json(
        session,
        profile.count_url,
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout_seconds=float(args.timeout_seconds),
    )
    total_source_rows = int(count_payload.get("count") or 0)
    if total_source_rows <= 0:
        raise RuntimeError(f"{profile.state_name} parcel source returned no rows.")
    if args.limit_records is not None:
        total_source_rows = min(total_source_rows, int(args.limit_records))
    stage_runtimes["fetch_count_seconds"] = round(time.perf_counter() - stage_start, 3)

    parcel_writer: pq.ParquetWriter | None = None
    geometry_writer: pq.ParquetWriter | None = None
    app_ready_accumulator = pd.DataFrame()
    county_counts: Counter[str] = Counter()
    geometry_flag_counts: Counter[str] = Counter()
    marketability_action_counts: Counter[str] = Counter()
    missing_coordinate_count = 0
    processed_rows = 0
    duplicate_parcel_row_id_rows_removed = 0
    source_alignment_audit: dict[str, Any] | None = None
    source_alignment_output = artifacts.runtime_root / "source_alignment_audit.json"
    parcel_master_output = artifacts.parcel_master_path
    parcel_master_temp_output = _temp_output_path(parcel_master_output)
    geometry_quality_output = artifacts.geometry_quality_artifact_path
    geometry_quality_temp_output = _temp_output_path(geometry_quality_output)
    app_ready_output = artifacts.app_ready_path
    app_ready_temp_output = _temp_output_path(app_ready_output)
    runtime_detail_output = artifacts.runtime_detail_metrics_path
    runtime_detail_temp_output = _temp_output_path(runtime_detail_output)
    for temp_path in [
        parcel_master_temp_output,
        geometry_quality_temp_output,
        app_ready_temp_output,
        runtime_detail_temp_output,
    ]:
        if temp_path.exists():
            temp_path.unlink()

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
    seen_parcel_row_ids: set[str] = set()

    stage_start = time.perf_counter()
    for index in range(0, total_source_rows, batch_size):
        current_batch_size = min(batch_size, total_source_rows - index)
        payload = _query_batch(
            session,
            profile,
            offset=index,
            batch_size=current_batch_size,
            timeout_seconds=float(args.timeout_seconds),
        )
        batch_frame = _transform_feature_batch(payload.get("features", []), profile=profile)
        if not batch_frame.empty:
            duplicate_mask = batch_frame["parcel_row_id"].isin(seen_parcel_row_ids)
            duplicate_parcel_row_id_rows_removed += int(duplicate_mask.sum())
            if duplicate_mask.any():
                batch_frame = batch_frame.loc[~duplicate_mask].copy()
        if batch_frame.empty:
            continue
        seen_parcel_row_ids.update(batch_frame["parcel_row_id"].astype(str).tolist())

        processed_rows += int(len(batch_frame))
        missing_coordinate_count += int(batch_frame["latitude"].isna().sum() + batch_frame["longitude"].isna().sum())
        county_counts.update(batch_frame["county_name"].dropna().astype(str).tolist())
        geometry_flag_counts.update(batch_frame["geometry_quality_flag"].dropna().astype(str).tolist())
        marketability_action_counts.update(batch_frame["geometry_marketability_action"].dropna().astype(str).tolist())

        parcel_table = pa.Table.from_pandas(batch_frame, preserve_index=False)
        parcel_writer = parcel_writer or pq.ParquetWriter(parcel_master_temp_output, parcel_table.schema, compression="snappy")
        parcel_writer.write_table(parcel_table)

        geometry_table = pa.Table.from_pandas(batch_frame[geometry_columns], preserve_index=False)
        geometry_writer = geometry_writer or pq.ParquetWriter(geometry_quality_temp_output, geometry_table.schema, compression="snappy")
        geometry_writer.write_table(geometry_table)

        candidate_mask = (
            batch_frame["parcel_row_id"].notna()
            & batch_frame["parcel_id"].notna()
            & batch_frame["county_name"].notna()
            & batch_frame["lead_score_total_effective"].notna()
            & ~batch_frame["geometry_marketability_default_leads_excluded_flag"].fillna(False)
            & pd.to_numeric(batch_frame["acreage"], errors="coerce").fillna(0).ge(profile.app_ready_min_acres)
            & pd.to_numeric(batch_frame["lead_score_total_effective"], errors="coerce").fillna(0).ge(profile.app_ready_min_score)
        )
        candidate_frame = batch_frame.loc[candidate_mask].copy()
        if not candidate_frame.empty:
            app_ready_accumulator = pd.concat([app_ready_accumulator, candidate_frame], ignore_index=True)
            app_ready_accumulator = _top_records(app_ready_accumulator, limit=int(args.max_app_ready))

        if (index // batch_size) % 25 == 0:
            print(
                f"Processed {profile.state_name} parcels {min(index + current_batch_size, total_source_rows):,}/{total_source_rows:,} "
                f"master_rows={processed_rows:,} app_ready_pool={len(app_ready_accumulator):,}"
            )

    if parcel_writer is not None:
        parcel_writer.close()
        parcel_master_temp_output.replace(parcel_master_output)
    if geometry_writer is not None:
        geometry_writer.close()
        geometry_quality_temp_output.replace(geometry_quality_output)
    stage_runtimes["stream_statewide_source_seconds"] = round(time.perf_counter() - stage_start, 3)

    stage_start = time.perf_counter()
    source_alignment_audit = _source_alignment_audit(
        session,
        profile,
        total_source_rows=total_source_rows,
        local_county_counts=county_counts,
        local_parcel_row_count=processed_rows,
        duplicate_parcel_row_id_rows_removed=duplicate_parcel_row_id_rows_removed,
        timeout_seconds=float(args.timeout_seconds),
        limited_build=args.limit_records is not None,
    )
    _write_json(source_alignment_output, source_alignment_audit)
    stage_runtimes["source_alignment_audit_seconds"] = round(time.perf_counter() - stage_start, 3)

    app_ready = _top_records(app_ready_accumulator, limit=int(args.max_app_ready))
    app_ready = _enrich_selected_coordinates(
        app_ready,
        session=session,
        profile=profile,
        timeout_seconds=float(args.timeout_seconds),
        batch_size=min(int(args.coordinate_batch_size), profile.coordinate_batch_size_hint),
    )
    if "latitude" in app_ready.columns and "longitude" in app_ready.columns:
        app_ready = app_ready.dropna(subset=["latitude", "longitude"]).copy()
    app_ready = app_ready.sort_values(
        by=["lead_score_total_effective", "lead_score_total", "acreage", "parcel_row_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    stage_start = time.perf_counter()
    app_ready.to_parquet(app_ready_temp_output, index=False)
    app_ready.to_parquet(runtime_detail_temp_output, index=False)
    app_ready_temp_output.replace(app_ready_output)
    runtime_detail_temp_output.replace(runtime_detail_output)
    _write_partitioned_index(app_ready, artifacts.runtime_parcel_index_root)
    stage_runtimes["write_runtime_parquet_seconds"] = round(time.perf_counter() - stage_start, 3)

    geometry_quality_summary = {
        "state_code": definition.state_code,
        "state_name": definition.state_name,
        "row_count": processed_rows,
        "processed_rows": processed_rows,
        "missing_coordinate_count": missing_coordinate_count,
        "proxy_dimensions_from_area_perimeter_flag": True,
        "warnings": [profile.source_warning],
        "build_start_timestamp": build_started,
        "build_end_timestamp": _now_iso(),
        "runtime_by_stage_seconds": stage_runtimes,
        "geometry_quality_flag_counts": dict(sorted(geometry_flag_counts.items())),
    }
    _write_json(artifacts.geometry_quality_summary_path, geometry_quality_summary)

    runtime_sections = _summary_sections(app_ready, geometry_flag_counts, marketability_action_counts)
    runtime_summary = {
        "row_count": int(len(app_ready)),
        "source": f"{profile.state_name} MVP app_ready parcel runtime built from {profile.source_name}",
        "geometry_mode": "selected_parcel_geojson",
        "mvp_warnings": [
            profile.source_warning,
            *profile.notes,
            "Tax and imagery-vacancy enrichment are deferred in the statewide ArcGIS MVP profile.",
            *([] if profile.parcel_pmtiles_ready else ["State parcel PMTiles overlay is not enabled for this source yet."]),
        ],
        "statewide_master_row_count": processed_rows,
        "official_source_alignment": source_alignment_audit,
        "app_ready_subset_limit": int(args.max_app_ready),
        "geometry_quality_diagnostics": geometry_quality_summary,
        "geometry_marketability_diagnostics": {
            "row_count": processed_rows,
            "proxy_dimensions_from_area_perimeter_flag": True,
            "geometry_marketability_action_counts": dict(sorted(marketability_action_counts.items())),
        },
        "sections": runtime_sections,
    }
    presets_payload = _preset_payload(app_ready, profile.preset_definitions)
    default_leads_frame = _top_records(app_ready, limit=default_leads_limit)
    default_leads_payload = {
        "total_count": int(len(app_ready)),
        "limit": default_leads_limit,
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
        profile=profile,
    )
    fallback_frame = _top_records(app_ready, limit=frontend_fallback_limit)
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
    diagnostics_payload = build_state_diagnostics(definition.state_code)
    _write_json(diagnostics_output, diagnostics_payload)

    print(
        json.dumps(
            {
                "state_code": definition.state_code,
                "state_name": definition.state_name,
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
                    "source_alignment_audit": _relative_path(source_alignment_output),
                },
                "stage_runtimes_seconds": stage_runtimes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
