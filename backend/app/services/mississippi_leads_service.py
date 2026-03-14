from __future__ import annotations

import json
import logging
import os
import time
import io
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import requests
import mercantile
import mapbox_vector_tile
from PIL import Image
from shapely import wkb
from shapely.geometry import mapping

from app.settings import (
    GEOMETRY_DEFAULT_LIMIT,
    GEOMETRY_MAX_LIMIT,
    LEADS_DEFAULT_LIMIT,
    LEADS_MAX_LIMIT,
    MISSISSIPPI_STATIC_FEED_PATH,
)


def _discover_project_root() -> Path:
    explicit_root = os.getenv("MISSISSIPPI_EXPLORER_DATA_ROOT")
    if explicit_root:
        return Path(explicit_root).expanduser().resolve(strict=False)

    cwd = Path.cwd().resolve(strict=False)
    service_repo_root = Path(__file__).resolve().parents[3]
    candidates = [cwd, cwd.parent, service_repo_root]
    for candidate in candidates:
        if (candidate / "data" / "parcels").exists():
            return candidate
    return cwd


PROJECT_ROOT = _discover_project_root()
BACKEND_DIR = Path(__file__).resolve().parents[2]
EMBEDDED_RUNTIME_DIR = BACKEND_DIR / "runtime" / "mississippi"
EMBEDDED_PARCEL_INDEX_ROOT = EMBEDDED_RUNTIME_DIR / "parcel_index"
EMBEDDED_GEOMETRY_INDEX_ROOT = EMBEDDED_RUNTIME_DIR / "parcel_geometry_index"
EMBEDDED_SUMMARY_PATH = EMBEDDED_RUNTIME_DIR / "summary.json"
EMBEDDED_PRESETS_PATH = EMBEDDED_RUNTIME_DIR / "presets.json"
EMBEDDED_DEFAULT_LEADS_PATH = EMBEDDED_RUNTIME_DIR / "default_leads.json"
EMBEDDED_DEFAULT_GEOMETRY_PATH = EMBEDDED_RUNTIME_DIR / "default_geometry.json"
EMBEDDED_DETAIL_METRICS_PATH = EMBEDDED_RUNTIME_DIR / "parcel_detail_metrics.parquet"
AI_MODEL_PARAMS_PATH = EMBEDDED_RUNTIME_DIR / "ai_building_presence_model_ms.json"
PARCEL_MASTER_PATH = PROJECT_ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
OWNER_LEADS_PATH = PROJECT_ROOT / "data" / "parcels" / "mississippi_parcels_owner_leads.parquet"
BUILDING_METRICS_PATH = PROJECT_ROOT / "data" / "buildings_processed" / "parcel_building_metrics.parquet"
AI_BUILDING_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "buildings_processed" / "ai_building_presence_predictions_ms.parquet"
LEAD_SIGNALS_PATH = PROJECT_ROOT / "data" / "tax_published" / "ms" / "app_ready_mississippi_leads.parquet"
EMBEDDED_LEAD_SIGNALS_PATH = EMBEDDED_RUNTIME_DIR / "app_ready_mississippi_leads.parquet"

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

PRESET_DEFINITIONS = {
    "safest_early_investor_use": {
        "description": "High-confidence county-hosted parcels with stronger amount reliability, no wetlands, and vacancy preference.",
        "filter_expression": "county_hosted_flag=true AND high_confidence_link_flag=true AND parcel_vacant_flag=true AND wetland_flag=false AND amount_trust_tier in trusted/use_with_caution",
        "filters": {
            "parcel_vacant_flag": True,
            "county_hosted_flag": True,
            "high_confidence_link_flag": True,
            "wetland_flag": False,
            "amount_trust_tier": ["trusted", "use_with_caution"],
            "min_lead_score_total": 65,
        },
    },
    "vacant_land_targeting": {
        "description": "Vacant parcels with stronger road access and fewer wetland constraints.",
        "filter_expression": "parcel_vacant_flag=true AND wetland_flag=false AND road_access_tier in direct/near",
        "filters": {
            "parcel_vacant_flag": True,
            "wetland_flag": False,
            "road_access_tier": ["direct", "near"],
            "min_lead_score_total": 65,
        },
    },
    "larger_acreage_land_targeting": {
        "description": "Vacant larger-acreage parcels for land assembly and development exploration.",
        "filter_expression": "parcel_vacant_flag=true AND county_hosted_flag=true AND acreage>=5",
        "filters": {
            "parcel_vacant_flag": True,
            "county_hosted_flag": True,
            "acreage_min": 5,
            "min_lead_score_total": 65,
        },
    },
    "growth_edge_targeting": {
        "description": "Parcels in moderate-to-high growth areas with usable road access.",
        "filter_expression": "growth_pressure_bucket in moderate/high AND road_access_tier in direct/near/moderate",
        "filters": {
            "growth_pressure_bucket": ["moderate", "high"],
            "road_access_tier": ["direct", "near", "moderate"],
            "min_lead_score_total": 65,
        },
    },
}

PARCEL_TILE_LAYER = "parcels"
PARCEL_TILE_MIN_ZOOM = 14
MISSISSIPPI_TILE_BOUNDS = (-91.65, 30.15, -88.0, 35.1)
tile_logger = logging.getLogger("parcel-tiles")
DEFAULT_TILE_URL_TEMPLATE = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SQFT_PER_ACRE = 43560.0


def _normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _serialize_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _coalesce_float_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column not in frame.columns:
            continue
        result = result.fillna(_to_float_series(frame, column))
    return result


def _rectangle_estimates(area_sqft: pd.Series, perimeter_ft: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    compactness = pd.Series(np.nan, index=area_sqft.index, dtype="float64")
    valid = area_sqft.gt(0) & perimeter_ft.gt(0)
    compactness.loc[valid] = ((4.0 * np.pi * area_sqft.loc[valid]) / np.square(perimeter_ft.loc[valid])).clip(0.0, 1.0)

    semi_perimeter = perimeter_ft / 2.0
    discriminant = np.square(semi_perimeter) - (4.0 * area_sqft)
    discriminant = discriminant.where(discriminant.ge(0))
    sqrt_disc = np.sqrt(discriminant)
    frontage = ((semi_perimeter + sqrt_disc) / 2.0).where(valid)
    width = ((semi_perimeter - sqrt_disc) / 2.0).where(valid)
    frontage = frontage.where(frontage.gt(0))
    width = width.where(width.gt(0))
    return compactness, frontage, width


def _vacancy_confidence_series(frame: pd.DataFrame) -> pd.Series:
    building_count = _to_float_series(frame, "building_count").fillna(0)
    building_area_total = _to_float_series(frame, "building_area_total").fillna(0)
    acreage = _to_float_series(frame, "acreage").fillna(0)
    assessed_total_value = _to_float_series(frame, "assessed_total_value").fillna(0)
    road_distance_ft = _to_float_series(frame, "road_distance_ft")
    nearby_building_density = _to_float_series(frame, "nearby_building_density").fillna(0)
    parcel_vacant = frame.get("parcel_vacant_flag")
    vacant_mask = parcel_vacant.fillna(False) if parcel_vacant is not None else pd.Series(False, index=frame.index)

    confidence = pd.Series(45.0, index=frame.index, dtype="float64")
    confidence = confidence.mask(vacant_mask & building_count.eq(0) & building_area_total.le(0), 92.0)
    confidence = confidence.mask(vacant_mask & building_count.le(1) & building_area_total.le(750) & acreage.ge(1), 78.0)
    confidence = confidence.mask(vacant_mask & assessed_total_value.ge(25000), 18.0)
    confidence = confidence.mask(vacant_mask & assessed_total_value.ge(10000) & road_distance_ft.le(150).fillna(False), 28.0)
    confidence = confidence.mask(vacant_mask & nearby_building_density.ge(120) & acreage.le(2), 35.0)
    confidence = confidence.mask(~vacant_mask & building_count.gt(0), 18.0)
    confidence = confidence.mask(~vacant_mask & building_area_total.gt(1500), 8.0)
    return confidence.clip(0, 100)


def _ensure_intelligence_fields(frame: pd.DataFrame) -> pd.DataFrame:
    wetland_pct = _coalesce_float_series(frame, ["wetland_pct", "wetland_overlap_pct"])
    wetland_overlap_acres = _to_float_series(frame, "wetland_overlap_acres")
    wetland_area_sqft = _coalesce_float_series(frame, ["wetland_area_sqft"])
    wetland_area_sqft = wetland_area_sqft.fillna(wetland_overlap_acres * SQFT_PER_ACRE)

    flood_pct = _coalesce_float_series(frame, ["flood_pct", "flood_overlap_pct"])
    flood_overlap_acres = _to_float_series(frame, "flood_overlap_acres")
    flood_area_sqft = _coalesce_float_series(frame, ["flood_area_sqft"])
    flood_area_sqft = flood_area_sqft.fillna(flood_overlap_acres * SQFT_PER_ACRE)

    shape_area = _to_float_series(frame, "shape_area")
    shape_length = _to_float_series(frame, "shape_length")
    derived_compactness, derived_frontage, derived_width = _rectangle_estimates(shape_area, shape_length)

    numeric_defaults = {
        "assessed_total_value": _to_float_series(frame, "assessed_total_value"),
        "mean_slope_pct": _to_float_series(frame, "mean_slope_pct"),
        "max_slope_pct": _to_float_series(frame, "max_slope_pct"),
        "slope_score": _to_float_series(frame, "slope_score"),
        "elevation_mean_ft": _to_float_series(frame, "elevation_mean_ft"),
        "shape_compactness": _coalesce_float_series(frame, ["shape_compactness"]).fillna(derived_compactness),
        "parcel_frontage_ft_estimate": _coalesce_float_series(frame, ["parcel_frontage_ft_estimate"]).fillna(derived_frontage),
        "parcel_width_ft_estimate": _coalesce_float_series(frame, ["parcel_width_ft_estimate"]).fillna(derived_width),
        "wetland_pct": wetland_pct,
        "wetland_area_sqft": wetland_area_sqft,
        "flood_pct": flood_pct,
        "flood_area_sqft": flood_area_sqft,
    }
    for column, series in numeric_defaults.items():
        frame[column] = series

    frame["slope_class"] = _normalize_string(frame.get("slope_class"), index=frame.index)
    frame["primary_fema_zone"] = _normalize_string(frame.get("primary_fema_zone"), index=frame.index).fillna(
        _normalize_string(frame.get("flood_zone_primary"), index=frame.index)
    )

    if "county_vacant_flag" in frame.columns:
        county_vacant = frame["county_vacant_flag"].astype("boolean")
    else:
        county_vacant = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    if "ai_building_present_flag" in frame.columns:
        ai_building_present = frame["ai_building_present_flag"].astype("boolean")
    else:
        ai_building_present = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["county_vacant_flag"] = county_vacant
    frame["ai_building_present_flag"] = ai_building_present
    frame["vacancy_confidence_score"] = _to_float_series(frame, "vacancy_confidence_score").fillna(_vacancy_confidence_series(frame))
    return frame


def _merge_ai_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if not AI_BUILDING_PREDICTIONS_PATH.exists():
        return frame
    ai_columns = [
        "parcel_row_id",
        "ai_building_present_probability",
        "ai_building_present_flag",
        "vacancy_confidence_score",
        "vacancy_model_version",
    ]
    available_columns = []
    ai_schema = ds.dataset(AI_BUILDING_PREDICTIONS_PATH, format="parquet").schema.names
    for column in ai_columns:
        if column in ai_schema:
            available_columns.append(column)
    if "parcel_row_id" not in available_columns:
        return frame
    ai_predictions = pd.read_parquet(AI_BUILDING_PREDICTIONS_PATH, columns=available_columns, engine="pyarrow")
    ai_predictions["parcel_row_id"] = _normalize_string(ai_predictions.get("parcel_row_id"), index=ai_predictions.index)
    return frame.merge(ai_predictions, on="parcel_row_id", how="left")


@lru_cache(maxsize=1)
def _ai_model_params() -> dict[str, Any] | None:
    if not AI_MODEL_PARAMS_PATH.exists():
        return None
    with AI_MODEL_PARAMS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _centroid_tile(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    lat_rad = np.radians(latitude)
    n = 2**zoom
    x = int((longitude + 180.0) / 360.0 * n)
    y = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
    return x, y


def _ai_extract_image_features(image_bytes: bytes) -> dict[str, float]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    flattened = array.reshape(-1, 3)
    features: dict[str, float] = {}

    channel_means = flattened.mean(axis=0)
    channel_stds = flattened.std(axis=0)
    for index, channel in enumerate(("r", "g", "b")):
        features[f"{channel}_mean"] = float(channel_means[index])
        features[f"{channel}_std"] = float(channel_stds[index])

    brightness_hist, _ = np.histogram(gray, bins=12, range=(0.0, 1.0), density=True)
    for index, value in enumerate(brightness_hist):
        features[f"brightness_hist_{index}"] = float(value)

    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    features["edge_density_x"] = float(grad_x)
    features["edge_density_y"] = float(grad_y)
    features["edge_density_total"] = float(grad_x + grad_y)
    features["gray_variance"] = float(gray.var())
    features["green_excess"] = float(channel_means[1] - ((channel_means[0] + channel_means[2]) / 2.0))
    features["roof_tone_pct"] = float(
        np.mean((array[..., 0] > 0.35) & (array[..., 0] < 0.85) & (array[..., 1] > 0.35) & (array[..., 1] < 0.85))
    )
    features["dark_shadow_pct"] = float(np.mean(gray < 0.18))
    return features


def _predict_ai_building_presence(
    longitude: float,
    latitude: float,
    parcel_vacant_flag: bool,
    building_count: float | None = None,
    building_area_total: float | None = None,
) -> dict[str, Any] | None:
    params = _ai_model_params()
    if params is None:
        return None
    zoom = 19
    tile_x, tile_y = _centroid_tile(longitude, latitude, zoom)
    url = DEFAULT_TILE_URL_TEMPLATE.format(z=zoom, x=tile_x, y=tile_y)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    features = _ai_extract_image_features(response.content)
    feature_columns = params["feature_columns"]
    mean = np.asarray(params["scaler_mean"], dtype=np.float64)
    scale = np.asarray(params["scaler_scale"], dtype=np.float64)
    coef = np.asarray(params["coef"], dtype=np.float64)
    intercept = float(params["intercept"])
    values = np.asarray([features[column] for column in feature_columns], dtype=np.float64)
    scaled = (values - mean) / scale
    logit = float(np.dot(scaled, coef) + intercept)
    probability = 1.0 / (1.0 + np.exp(-logit))
    building_count_value = 0.0 if building_count is None or not np.isfinite(building_count) else float(building_count)
    building_area_value = 0.0 if building_area_total is None or not np.isfinite(building_area_total) else float(building_area_total)
    if building_count_value >= 1 or building_area_value >= 400:
        probability = max(probability, 0.75)
    threshold = float(params.get("classification_threshold", 0.5))
    ai_building_present_flag = probability >= threshold
    footprint_vacancy_score = 92.0 if parcel_vacant_flag else 15.0
    imagery_vacancy_score = (1.0 - probability) * 100.0
    vacancy_confidence_score = round((footprint_vacancy_score * 0.55) + (imagery_vacancy_score * 0.45), 2)
    return {
        "ai_building_present_probability": round(probability, 6),
        "ai_building_present_flag": bool(ai_building_present_flag),
        "vacancy_confidence_score": vacancy_confidence_score,
        "vacancy_model_version": params.get("model_version"),
    }


def _bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number


def _apply_vacancy_assessment(payload: dict[str, Any]) -> None:
    parcel_vacant_flag = _bool_or_none(payload.get("parcel_vacant_flag"))
    ai_building_present_flag = _bool_or_none(payload.get("ai_building_present_flag"))
    ai_probability = _float_or_none(payload.get("ai_building_present_probability"))
    building_count = _float_or_none(payload.get("building_count")) or 0.0
    building_area_total = _float_or_none(payload.get("building_area_total")) or 0.0
    assessed_total_value = _float_or_none(payload.get("assessed_total_value")) or 0.0
    road_distance_ft = _float_or_none(payload.get("road_distance_ft"))
    nearby_building_density = _float_or_none(payload.get("nearby_building_density")) or 0.0
    acreage = _float_or_none(payload.get("acreage")) or 0.0

    improved_evidence = 0.0
    vacant_evidence = 0.0

    if parcel_vacant_flag is True:
        vacant_evidence += 42.0
    elif parcel_vacant_flag is False:
        improved_evidence += 18.0

    if building_count >= 1:
        improved_evidence += 55.0
    if building_area_total >= 400:
        improved_evidence += 45.0
    elif building_area_total > 0:
        improved_evidence += 18.0
    if building_count <= 0 and building_area_total <= 0:
        vacant_evidence += 18.0

    if ai_probability is not None:
        if ai_probability >= 0.8:
            improved_evidence += 55.0
        elif ai_probability >= 0.65:
            improved_evidence += 40.0
        elif ai_probability >= 0.5:
            improved_evidence += 22.0
        elif ai_probability <= 0.2:
            vacant_evidence += 18.0
        elif ai_probability <= 0.35:
            vacant_evidence += 10.0
    elif ai_building_present_flag is True:
        improved_evidence += 32.0

    if assessed_total_value >= 25000:
        improved_evidence += 42.0
    elif assessed_total_value >= 10000:
        improved_evidence += 28.0
    elif assessed_total_value >= 3000:
        improved_evidence += 12.0
    else:
        vacant_evidence += 8.0

    if road_distance_ft is not None and road_distance_ft <= 150:
        improved_evidence += 8.0
    if nearby_building_density >= 120 and acreage <= 5:
        improved_evidence += 8.0
    elif nearby_building_density <= 10 and acreage >= 5:
        vacant_evidence += 6.0

    vacancy_likelihood = float(np.clip(50.0 + vacant_evidence - improved_evidence, 0.0, 100.0))

    if improved_evidence >= vacant_evidence + 25.0:
        payload["overall_vacancy_assessment"] = "Likely improved"
        payload["vacancy_confidence_score"] = round(min(vacancy_likelihood, 25.0), 1)
        payload["vacant_reason"] = "Improvement evidence outweighs vacancy signals based on parcel value, structure context, and imagery."
        if parcel_vacant_flag is True:
            payload["overall_vacancy_assessment"] = "Likely improved - conflicting signals"
            payload["vacant_reason"] = "Footprint logic suggests vacancy, but assessed value, access, nearby context, or imagery point to an improved parcel."
        return

    if vacant_evidence >= improved_evidence + 30.0 and assessed_total_value < 10000 and building_count <= 0 and building_area_total <= 0:
        payload["overall_vacancy_assessment"] = "Likely vacant"
        payload["vacancy_confidence_score"] = round(max(vacancy_likelihood, 70.0), 1)
        payload["vacant_reason"] = "Vacancy signals are consistent across parcel footprints, limited improvement evidence, and available imagery."
        return

    payload["overall_vacancy_assessment"] = "Needs review"
    payload["vacancy_confidence_score"] = round(float(np.clip(vacancy_likelihood, 35.0, 65.0)), 1)
    payload["vacant_reason"] = "Signals conflict or are incomplete. Review imagery and local records before treating this parcel as vacant."


def _coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column not in frame.columns:
            continue
        result = result.fillna(_to_float_series(frame, column))
    return result


def _score_tier(score: pd.Series) -> pd.Series:
    return pd.cut(
        score.fillna(0),
        bins=[-1, 49.999, 64.999, 79.999, 100.001],
        labels=["low", "medium", "high", "very_high"],
    ).astype("string")


def _row_top_drivers(frame: pd.DataFrame, components: list[str], top_n: int = 3) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for _, row in frame[components].iterrows():
        ordered = sorted(
            ((component, float(row.get(component) or 0)) for component in components),
            key=lambda item: item[1],
            reverse=True,
        )
        rows.append([ordered[index][0] if index < len(ordered) else None for index in range(top_n)])
    return rows


def _component_reason(component: str | None) -> str | None:
    reasons = {
        "buildability_component": "buildability and development potential",
        "environmental_component": "environmental resilience and lower constraint burden",
        "owner_targeting_component": "owner motivation and outreach potential",
        "vacant_land_component": "vacancy and limited building intensity",
        "growth_pressure_component": "growth edge context and nearby activity",
        "access_score": "usable road access",
        "delinquency_component": "motivation signals from tax distress",
        "source_confidence_component": "data reliability and source strength",
        "size_score": "larger tract size",
    }
    return reasons.get(component)


def _county_hosted_flag(series: pd.Series) -> pd.Series:
    normalized = _normalize_string(series, index=series.index).fillna("")
    return normalized.str.contains("direct_download|county", case=False, regex=True)


def _full_runtime_available() -> bool:
    return PARCEL_MASTER_PATH.exists() and OWNER_LEADS_PATH.exists() and BUILDING_METRICS_PATH.exists()


def _embedded_parcel_runtime_available() -> bool:
    return EMBEDDED_PARCEL_INDEX_ROOT.exists() and any(EMBEDDED_PARCEL_INDEX_ROOT.rglob("*.parquet"))


@lru_cache(maxsize=1)
def _embedded_parcel_dataset() -> ds.Dataset:
    partitioning = ds.partitioning(pa.schema([("county_name", pa.string())]))
    return ds.dataset(EMBEDDED_PARCEL_INDEX_ROOT, format="parquet", partitioning=partitioning)


def _embedded_geometry_runtime_available() -> bool:
    return EMBEDDED_GEOMETRY_INDEX_ROOT.exists() and any(EMBEDDED_GEOMETRY_INDEX_ROOT.rglob("*.parquet"))


@lru_cache(maxsize=1)
def _embedded_geometry_dataset() -> ds.Dataset:
    partitioning = ds.partitioning(pa.schema([("county_name", pa.string())]))
    return ds.dataset(EMBEDDED_GEOMETRY_INDEX_ROOT, format="parquet", partitioning=partitioning)


def _embedded_detail_metrics_runtime_available() -> bool:
    return EMBEDDED_DETAIL_METRICS_PATH.exists()


@lru_cache(maxsize=1)
def _embedded_detail_metrics_dataset() -> ds.Dataset:
    return ds.dataset(EMBEDDED_DETAIL_METRICS_PATH, format="parquet")


def _lookup_embedded_detail_metrics(parcel_row_id: str) -> dict[str, Any]:
    if not _embedded_detail_metrics_runtime_available():
        return {}
    try:
        table = _embedded_detail_metrics_dataset().to_table(filter=ds.field("parcel_row_id") == parcel_row_id)
    except Exception:
        return {}
    if table.num_rows == 0:
        return {}
    row = table.to_pandas().iloc[0]
    return {column: _serialize_scalar(row[column]) for column in row.index if column != "parcel_row_id" and pd.notna(row[column])}


def _using_embedded_runtime() -> bool:
    return (not _full_runtime_available()) and _embedded_parcel_runtime_available()


def _embedded_filter_expression(
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
    parcel_row_ids: list[str] | None = None,
    selected_parcel_id: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> ds.Expression | None:
    expression: ds.Expression | None = None

    def combine(next_expression: ds.Expression | None) -> None:
        nonlocal expression
        if next_expression is None:
            return
        expression = next_expression if expression is None else expression & next_expression

    if county_name:
        combine(ds.field("county_name") == county_name)
    if lead_score_tier:
        combine(ds.field("lead_score_tier").isin(lead_score_tier))
    if min_lead_score_total is not None:
        combine(ds.field("lead_score_total") >= min_lead_score_total)
    if acreage_min is not None:
        combine(ds.field("acreage") >= acreage_min)
    if acreage_max is not None:
        combine(ds.field("acreage") <= acreage_max)
    if parcel_vacant_flag is not None:
        combine(ds.field("parcel_vacant_flag") == parcel_vacant_flag)
    if county_hosted_flag is not None:
        combine(ds.field("county_hosted_flag") == county_hosted_flag)
    if high_confidence_link_flag is not None:
        combine(ds.field("high_confidence_link_flag") == high_confidence_link_flag)
    if wetland_flag is not None:
        combine(ds.field("wetland_flag") == wetland_flag)
    if amount_trust_tier:
        combine(ds.field("amount_trust_tier").isin(amount_trust_tier))
    if corporate_owner_flag is not None:
        combine(ds.field("corporate_owner_flag") == corporate_owner_flag)
    if absentee_owner_flag is not None:
        combine(ds.field("absentee_owner_flag") == absentee_owner_flag)
    if out_of_state_owner_flag is not None:
        combine(ds.field("out_of_state_owner_flag") == out_of_state_owner_flag)
    if growth_pressure_bucket:
        combine(ds.field("growth_pressure_bucket").isin(growth_pressure_bucket))
    if recommended_view_bucket:
        combine(ds.field("recommended_view_bucket").isin(recommended_view_bucket))
    if road_access_tier:
        combine(ds.field("road_access_tier").isin(road_access_tier))
    if road_distance_ft_max is not None:
        combine(ds.field("road_distance_ft") <= road_distance_ft_max)
    if parcel_row_ids:
        combine(ds.field("parcel_row_id").isin(parcel_row_ids))
    if bounds is not None:
        min_lng, min_lat, max_lng, max_lat = bounds
        bounds_expression = (
            (ds.field("longitude") >= min_lng)
            & (ds.field("longitude") <= max_lng)
            & (ds.field("latitude") >= min_lat)
            & (ds.field("latitude") <= max_lat)
        )
        if selected_parcel_id:
            bounds_expression = bounds_expression | (ds.field("parcel_row_id") == selected_parcel_id)
        combine(bounds_expression)

    return expression


def _embedded_to_pandas(columns: list[str], expression: ds.Expression | None = None) -> pd.DataFrame:
    table = _embedded_parcel_dataset().to_table(columns=columns, filter=expression)
    return table.to_pandas()


def _embedded_count_rows(expression: ds.Expression | None = None) -> int:
    return int(_embedded_parcel_dataset().count_rows(filter=expression))


def _bounded_sorted_batches(
    *,
    columns: list[str],
    expression: ds.Expression | None,
    sort_by: str,
    ascending: bool,
    keep_rows: int,
    batch_size: int = 50000,
) -> pd.DataFrame:
    dataset = _embedded_parcel_dataset()
    candidate = pd.DataFrame(columns=columns)
    scanner = dataset.scanner(columns=columns, filter=expression, batch_size=batch_size)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        candidate = pd.concat([candidate, frame], ignore_index=True)
        candidate = candidate.sort_values(sort_by, ascending=ascending, na_position="last").head(keep_rows)
    return candidate.sort_values(sort_by, ascending=ascending, na_position="last").head(keep_rows)


def _bounded_scan_batches(
    *,
    columns: list[str],
    expression: ds.Expression | None,
    keep_rows: int,
    batch_size: int = 50000,
) -> pd.DataFrame:
    dataset = _embedded_parcel_dataset()
    frames: list[pd.DataFrame] = []
    kept = 0
    scanner = dataset.scanner(columns=columns, filter=expression, batch_size=batch_size)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        remaining = keep_rows - kept
        if remaining <= 0:
            break
        sliced = frame.head(remaining)
        frames.append(sliced)
        kept += len(sliced)
        if kept >= keep_rows:
            break
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def _geojson_bounds(geometry: dict[str, Any]) -> list[float] | None:
    coordinates = geometry.get("coordinates")
    if coordinates is None:
        return None
    min_lng = float("inf")
    min_lat = float("inf")
    max_lng = float("-inf")
    max_lat = float("-inf")

    def walk(value: Any) -> None:
        nonlocal min_lng, min_lat, max_lng, max_lat
        if not isinstance(value, (list, tuple)):
            return
        if len(value) >= 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            lng = float(value[0])
            lat = float(value[1])
            min_lng = min(min_lng, lng)
            min_lat = min(min_lat, lat)
            max_lng = max(max_lng, lng)
            max_lat = max(max_lat, lat)
            return
        for item in value:
            walk(item)

    walk(coordinates)
    if not all(np.isfinite(value) for value in [min_lng, min_lat, max_lng, max_lat]):
        return None
    return [min_lng, min_lat, max_lng, max_lat]


def _load_embedded_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_default_filter_state(
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
) -> bool:
    return (
        county_name is None
        and not lead_score_tier
        and (min_lead_score_total is None or float(min_lead_score_total) == 0.0)
        and acreage_min is None
        and acreage_max is None
        and parcel_vacant_flag is None
        and county_hosted_flag is None
        and high_confidence_link_flag is None
        and wetland_flag is None
        and not amount_trust_tier
        and corporate_owner_flag is None
        and absentee_owner_flag is None
        and out_of_state_owner_flag is None
        and not growth_pressure_bucket
        and not recommended_view_bucket
        and not road_access_tier
        and road_distance_ft_max is None
    )


def _load_static_feed_frame() -> pd.DataFrame:
    if not MISSISSIPPI_STATIC_FEED_PATH.exists():
        raise FileNotFoundError(f"Static explorer feed not found: {MISSISSIPPI_STATIC_FEED_PATH}")
    with MISSISSIPPI_STATIC_FEED_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    frame = pd.DataFrame(records)
    if "geometry" in frame.columns:
        frame["longitude"] = frame["geometry"].apply(
            lambda value: value.get("coordinates", [None, None])[0] if isinstance(value, dict) else None
        )
        frame["latitude"] = frame["geometry"].apply(
            lambda value: value.get("coordinates", [None, None])[1] if isinstance(value, dict) else None
        )
    return frame


def _centroids_from_wkb(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    longitudes = pd.Series(np.nan, index=series.index, dtype="float64")
    latitudes = pd.Series(np.nan, index=series.index, dtype="float64")
    for index, geometry_bytes in series.items():
        if not geometry_bytes:
            continue
        try:
            shape = wkb.loads(geometry_bytes)
        except Exception:
            continue
        centroid = shape.centroid
        longitudes.at[index] = float(centroid.x)
        latitudes.at[index] = float(centroid.y)
    return longitudes, latitudes


@lru_cache(maxsize=1)
def load_base_frame() -> pd.DataFrame:
    parquet_path = LEAD_SIGNALS_PATH if LEAD_SIGNALS_PATH.exists() else EMBEDDED_LEAD_SIGNALS_PATH

    if not parquet_path.exists() and not MISSISSIPPI_STATIC_FEED_PATH.exists():
        raise FileNotFoundError(
            f"Lead signals dataset not found: {parquet_path}; static feed not found: {MISSISSIPPI_STATIC_FEED_PATH}"
        )

    if not _full_runtime_available() and _embedded_parcel_runtime_available():
        frame = ds.dataset(EMBEDDED_PARCEL_INDEX_ROOT, format="parquet").to_table().to_pandas()
        frame["county_hosted_flag"] = frame["county_hosted_flag"].fillna(_county_hosted_flag(frame.get("best_source_type")))
        frame["parcel_vacant_flag"] = frame["parcel_vacant_flag"].fillna(False)
        frame["corporate_owner_flag"] = frame["corporate_owner_flag"].fillna(False)
        frame["absentee_owner_flag"] = frame["absentee_owner_flag"].fillna(False)
        frame["out_of_state_owner_flag"] = frame["out_of_state_owner_flag"].fillna(False)
        frame["high_confidence_link_flag"] = frame["high_confidence_link_flag"].fillna(False)
        frame["delinquent_flag"] = frame["delinquent_flag"].fillna(False)
        frame["forfeited_flag"] = frame["forfeited_flag"].fillna(False)
        frame["amount_trust_tier"] = _normalize_string(frame.get("amount_trust_tier"), index=frame.index).fillna("not_reported")
        frame["source_confidence_tier"] = _normalize_string(frame.get("source_confidence_tier"), index=frame.index).fillna("parcel_master_only")
        frame["county_source_coverage_tier"] = _normalize_string(frame.get("county_source_coverage_tier"), index=frame.index).fillna("statewide_parcel_base")
        frame["best_source_type"] = _normalize_string(frame.get("best_source_type"), index=frame.index).fillna("parcel_master")
        frame["best_source_name"] = _normalize_string(frame.get("best_source_name"), index=frame.index).fillna("Mississippi Parcel Runtime")
        frame["growth_pressure_bucket"] = _normalize_string(frame.get("growth_pressure_bucket"), index=frame.index).fillna("unknown")
        frame["recommended_view_bucket"] = _normalize_string(frame.get("recommended_view_bucket"), index=frame.index).fillna("general_ranked")
        frame["owner_type"] = _normalize_string(frame.get("owner_type"), index=frame.index).fillna("unknown")
        frame["state_code"] = _normalize_string(frame.get("state_code"), index=frame.index).fillna("MS")
        frame["county_name"] = _normalize_string(frame.get("county_name"), index=frame.index)
        frame["owner_name"] = _normalize_string(frame.get("owner_name"), index=frame.index)
        frame["parcel_id"] = _normalize_string(frame.get("parcel_id"), index=frame.index)
        frame["electric_provider_name"] = _normalize_string(frame.get("electric_provider_name"), index=frame.index)
        frame = _merge_ai_predictions(frame)
        frame = _ensure_intelligence_fields(frame)
        return frame

    if not _full_runtime_available():
        try:
            frame = pd.read_parquet(parquet_path, engine="pyarrow").copy()
        except Exception:
            frame = _load_static_feed_frame()
        frame["acreage"] = _to_float_series(frame, "acreage")
        frame["land_use"] = _normalize_string(frame.get("land_use"), index=frame.index)
        frame["assessed_total_value"] = _to_float_series(frame, "assessed_total_value")
        frame["county_hosted_flag"] = frame["county_hosted_flag"].fillna(_county_hosted_flag(frame.get("best_source_type")))
        frame["parcel_vacant_flag"] = frame["parcel_vacant_flag"].fillna(False)
        frame["corporate_owner_flag"] = frame["corporate_owner_flag"].fillna(False)
        frame["absentee_owner_flag"] = frame["absentee_owner_flag"].fillna(False)
        frame["out_of_state_owner_flag"] = frame["out_of_state_owner_flag"].fillna(False)
        frame["high_confidence_link_flag"] = frame["high_confidence_link_flag"].fillna(False)
        frame["delinquent_flag"] = frame["delinquent_flag"].fillna(False)
        frame["forfeited_flag"] = frame["forfeited_flag"].fillna(False)
        frame["amount_trust_tier"] = _normalize_string(frame.get("amount_trust_tier"), index=frame.index).fillna("not_reported")
        frame["source_confidence_tier"] = _normalize_string(frame.get("source_confidence_tier"), index=frame.index).fillna("medium")
        frame["county_source_coverage_tier"] = _normalize_string(frame.get("county_source_coverage_tier"), index=frame.index).fillna("county_dataset")
        frame["best_source_type"] = _normalize_string(frame.get("best_source_type"), index=frame.index).fillna("lead_dataset")
        frame["best_source_name"] = _normalize_string(frame.get("best_source_name"), index=frame.index).fillna("Mississippi Lead Dataset")
        frame["growth_pressure_bucket"] = _normalize_string(frame.get("growth_pressure_bucket"), index=frame.index).fillna("unknown")
        frame["recommended_view_bucket"] = _normalize_string(frame.get("recommended_view_bucket"), index=frame.index).fillna("general_ranked")
        frame["owner_type"] = _normalize_string(frame.get("owner_type"), index=frame.index).fillna("unknown")
        frame["state_code"] = _normalize_string(frame.get("state_code"), index=frame.index).fillna("MS")
        frame["county_name"] = _normalize_string(frame.get("county_name"), index=frame.index)
        frame["owner_name"] = _normalize_string(frame.get("owner_name"), index=frame.index)
        frame["parcel_id"] = _normalize_string(frame.get("parcel_id"), index=frame.index)
        frame["electric_provider_name"] = _normalize_string(frame.get("electric_provider_name"), index=frame.index)
        longitudes, latitudes = _centroids_from_wkb(frame.get("geometry"))
        frame["longitude"] = longitudes
        frame["latitude"] = latitudes
        frame = _merge_ai_predictions(frame)
        frame = _ensure_intelligence_fields(frame)
        return frame

    parcel_columns = [
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "county_fips",
        "state_code",
        "owner_name",
        "land_use_raw",
        "tax_acres",
        "gis_acres",
        "total_acres",
        "parcel_area_acres",
        "latitude",
        "longitude",
        "road_distance_ft",
        "road_access_tier",
        "wetland_flag",
        "wetland_overlap_acres",
        "wetland_overlap_pct",
        "flood_risk_score",
        "flood_zone_primary",
        "has_flood_overlap",
        "sfha_overlap",
        "mean_slope_pct",
        "max_slope_pct",
        "slope_class",
        "slope_score",
        "shape_length",
        "shape_area",
        "buildability_score",
        "environment_score",
        "investment_score",
        "electric_provider_name",
        "total_value",
    ]
    parcels = pd.read_parquet(PARCEL_MASTER_PATH, columns=parcel_columns, engine="pyarrow")
    parcels["acreage"] = _coalesce_numeric(parcels, ["total_acres", "parcel_area_acres", "gis_acres", "tax_acres"])
    parcels["land_use"] = _normalize_string(parcels.get("land_use_raw"), index=parcels.index)
    parcels["assessed_total_value"] = _to_float_series(parcels, "total_value")
    parcels = parcels.drop(columns=["land_use_raw", "tax_acres", "gis_acres", "total_acres", "parcel_area_acres", "total_value"])

    owners = pd.read_parquet(
        OWNER_LEADS_PATH,
        columns=[
            "parcel_row_id",
            "owner_type",
            "absentee_owner_flag",
            "out_of_state_owner_flag",
            "owner_parcel_count",
            "owner_total_acres",
            "mailer_target_score",
            "corporate_owner_flag",
        ],
        engine="pyarrow",
    )

    buildings = pd.read_parquet(
        BUILDING_METRICS_PATH,
        columns=[
            "parcel_row_id",
            "building_count",
            "building_area_total",
            "parcel_vacant_flag",
            "nearby_building_count_1km",
            "nearby_building_density",
            "growth_pressure_bucket",
        ],
        engine="pyarrow",
    )

    signals = pd.read_parquet(
        LEAD_SIGNALS_PATH,
        columns=[
            "parcel_row_id",
            "delinquent_amount",
            "delinquent_amount_bucket",
            "delinquent_flag",
            "forfeited_flag",
            "best_source_type",
            "best_source_name",
            "source_confidence_tier",
            "county_source_coverage_tier",
            "amount_trust_tier",
            "high_confidence_link_flag",
            "county_hosted_flag",
            "lead_score_total",
            "lead_score_tier",
            "lead_score_driver_1",
            "lead_score_driver_2",
            "lead_score_driver_3",
            "lead_score_explanation",
            "size_score",
            "access_score",
            "buildability_component",
            "environmental_component",
            "owner_targeting_component",
            "delinquency_component",
            "source_confidence_component",
            "vacant_land_component",
            "growth_pressure_component",
            "recommended_sort_reason",
            "top_score_driver",
            "caution_flags",
            "vacant_reason",
            "growth_pressure_reason",
            "recommended_use_case",
            "recommended_view_bucket",
        ],
        engine="pyarrow",
    )

    frame = parcels.merge(owners, on="parcel_row_id", how="left")
    frame = frame.merge(buildings, on="parcel_row_id", how="left")
    frame = frame.merge(signals, on="parcel_row_id", how="left")

    frame["parcel_vacant_flag"] = frame["parcel_vacant_flag"].fillna(False)
    frame["corporate_owner_flag"] = frame["corporate_owner_flag"].fillna(False)
    frame["absentee_owner_flag"] = frame["absentee_owner_flag"].fillna(False)
    frame["out_of_state_owner_flag"] = frame["out_of_state_owner_flag"].fillna(False)
    frame["high_confidence_link_flag"] = frame["high_confidence_link_flag"].fillna(False)
    frame["county_hosted_flag"] = frame["county_hosted_flag"].fillna(_county_hosted_flag(frame.get("best_source_type")))
    frame["delinquent_flag"] = frame["delinquent_flag"].fillna(False)
    frame["forfeited_flag"] = frame["forfeited_flag"].fillna(False)
    frame["amount_trust_tier"] = _normalize_string(frame.get("amount_trust_tier"), index=frame.index).fillna("not_reported")
    frame["source_confidence_tier"] = _normalize_string(frame.get("source_confidence_tier"), index=frame.index).fillna("parcel_master_only")
    frame["county_source_coverage_tier"] = _normalize_string(frame.get("county_source_coverage_tier"), index=frame.index).fillna("statewide_parcel_base")
    frame["best_source_type"] = _normalize_string(frame.get("best_source_type"), index=frame.index).fillna("parcel_master")
    frame["best_source_name"] = _normalize_string(frame.get("best_source_name"), index=frame.index).fillna("Mississippi Parcel Master")
    frame["growth_pressure_bucket"] = _normalize_string(frame.get("growth_pressure_bucket"), index=frame.index).fillna("unknown")
    frame["recommended_view_bucket"] = _normalize_string(frame.get("recommended_view_bucket"), index=frame.index).fillna("general_ranked")
    frame["owner_type"] = _normalize_string(frame.get("owner_type"), index=frame.index).fillna("unknown")
    frame["state_code"] = _normalize_string(frame.get("state_code"), index=frame.index).fillna("MS")
    frame["county_name"] = _normalize_string(frame.get("county_name"), index=frame.index)
    frame["owner_name"] = _normalize_string(frame.get("owner_name"), index=frame.index)
    frame["parcel_id"] = _normalize_string(frame.get("parcel_id"), index=frame.index)
    frame = _merge_ai_predictions(frame)
    frame = _ensure_intelligence_fields(frame)

    acreage = _to_float_series(frame, "acreage").fillna(0)
    computed_size_score = pd.Series(35.0, index=frame.index)
    computed_size_score = computed_size_score.mask(acreage >= 1, 55.0)
    computed_size_score = computed_size_score.mask(acreage >= 5, 72.0)
    computed_size_score = computed_size_score.mask(acreage >= 20, 86.0)
    frame["size_score"] = pd.to_numeric(frame.get("size_score"), errors="coerce").fillna(computed_size_score)

    road_distance_ft = _to_float_series(frame, "road_distance_ft")
    computed_access_score = pd.Series(25.0, index=frame.index)
    computed_access_score = computed_access_score.mask(road_distance_ft.le(250).fillna(False), 92.0)
    computed_access_score = computed_access_score.mask(road_distance_ft.between(251, 1000).fillna(False), 76.0)
    computed_access_score = computed_access_score.mask(road_distance_ft.between(1001, 2500).fillna(False), 58.0)
    computed_access_score = computed_access_score.mask(road_distance_ft.between(2501, 5280).fillna(False), 42.0)
    frame["access_score"] = pd.to_numeric(frame.get("access_score"), errors="coerce").fillna(computed_access_score)

    frame["buildability_component"] = pd.to_numeric(frame.get("buildability_component"), errors="coerce").fillna(_to_float_series(frame, "buildability_score").fillna(0))
    frame["environmental_component"] = pd.to_numeric(frame.get("environmental_component"), errors="coerce").fillna(_to_float_series(frame, "environment_score").fillna(0))

    owner_targeting_base = _to_float_series(frame, "mailer_target_score").fillna(25)
    owner_targeting_base = owner_targeting_base + frame["absentee_owner_flag"].astype(int) * 12 + frame["out_of_state_owner_flag"].astype(int) * 10 + frame["corporate_owner_flag"].astype(int) * 8
    frame["owner_targeting_component"] = pd.to_numeric(frame.get("owner_targeting_component"), errors="coerce").fillna(owner_targeting_base.clip(0, 100))

    delinquency_base = pd.Series(0.0, index=frame.index)
    delinquency_base = delinquency_base.mask(frame["delinquent_flag"], 78.0)
    delinquency_base = delinquency_base.mask(frame["forfeited_flag"], 92.0)
    frame["delinquency_component"] = pd.to_numeric(frame.get("delinquency_component"), errors="coerce").fillna(delinquency_base)

    source_conf_base = pd.Series(24.0, index=frame.index)
    source_conf_base = source_conf_base.mask(frame["source_confidence_tier"].eq("high"), 90.0)
    source_conf_base = source_conf_base.mask(frame["source_confidence_tier"].eq("medium"), 62.0)
    source_conf_base = source_conf_base.mask(frame["source_confidence_tier"].eq("parcel_master_only"), 38.0)
    frame["source_confidence_component"] = pd.to_numeric(frame.get("source_confidence_component"), errors="coerce").fillna(source_conf_base)

    vacant_base = pd.Series(18.0, index=frame.index)
    vacant_base = vacant_base.mask(frame["parcel_vacant_flag"], 94.0)
    vacant_base = vacant_base.mask((pd.to_numeric(frame.get("building_count"), errors="coerce").fillna(0) <= 1) & acreage.ge(5), 72.0)
    frame["vacant_land_component"] = pd.to_numeric(frame.get("vacant_land_component"), errors="coerce").fillna(vacant_base)

    growth_map = {"very_low": 20.0, "low": 38.0, "moderate": 68.0, "high": 88.0, "unknown": 28.0}
    growth_base = frame["growth_pressure_bucket"].map(growth_map).astype("float64")
    frame["growth_pressure_component"] = pd.to_numeric(frame.get("growth_pressure_component"), errors="coerce").fillna(growth_base.fillna(28.0))

    base_total = (
        frame["size_score"] * 0.08
        + frame["access_score"] * 0.10
        + frame["buildability_component"] * 0.20
        + frame["environmental_component"] * 0.18
        + frame["owner_targeting_component"] * 0.16
        + frame["delinquency_component"] * 0.10
        + frame["source_confidence_component"] * 0.08
        + frame["vacant_land_component"] * 0.05
        + frame["growth_pressure_component"] * 0.05
    ).round(2)
    frame["lead_score_total"] = pd.to_numeric(frame.get("lead_score_total"), errors="coerce").fillna(base_total)
    frame["lead_score_tier"] = _normalize_string(frame.get("lead_score_tier"), index=frame.index).fillna(_score_tier(frame["lead_score_total"]))

    component_columns = [
        "buildability_component",
        "environmental_component",
        "owner_targeting_component",
        "vacant_land_component",
        "growth_pressure_component",
        "access_score",
        "delinquency_component",
        "source_confidence_component",
        "size_score",
    ]
    driver_frame = pd.DataFrame(_row_top_drivers(frame, component_columns), columns=["driver_1", "driver_2", "driver_3"], index=frame.index)
    frame["lead_score_driver_1"] = _normalize_string(frame.get("lead_score_driver_1"), index=frame.index).fillna(driver_frame["driver_1"])
    frame["lead_score_driver_2"] = _normalize_string(frame.get("lead_score_driver_2"), index=frame.index).fillna(driver_frame["driver_2"])
    frame["lead_score_driver_3"] = _normalize_string(frame.get("lead_score_driver_3"), index=frame.index).fillna(driver_frame["driver_3"])
    frame["top_score_driver"] = _normalize_string(frame.get("top_score_driver"), index=frame.index).fillna(frame["lead_score_driver_1"])
    frame["lead_score_explanation"] = _normalize_string(frame.get("lead_score_explanation"), index=frame.index).fillna(
        frame["lead_score_driver_1"].map(_component_reason).fillna("balanced parcel quality and signal coverage")
    )
    frame["recommended_sort_reason"] = _normalize_string(frame.get("recommended_sort_reason"), index=frame.index).fillna(
        frame["lead_score_driver_1"].str.replace("_component", "", regex=False).str.replace("_score", "", regex=False).str.replace("_", " ", regex=False)
    )
    frame["vacant_reason"] = _normalize_string(frame.get("vacant_reason"), index=frame.index).fillna(
        frame["parcel_vacant_flag"].map({True: "No mapped building footprints on parcel.", False: "Existing building footprint detected."})
    )
    frame["growth_pressure_reason"] = _normalize_string(frame.get("growth_pressure_reason"), index=frame.index).fillna(
        frame["growth_pressure_bucket"].map(
            {
                "high": "Strong nearby building density suggests active development pressure.",
                "moderate": "Moderate nearby building density suggests edge growth potential.",
                "low": "Lower nearby building density suggests slower market growth.",
                "very_low": "Minimal nearby building density suggests very limited growth pressure.",
                "unknown": "Growth pressure is not yet classified for this parcel.",
            }
        )
    )
    frame["recommended_use_case"] = _normalize_string(frame.get("recommended_use_case"), index=frame.index).fillna(
        frame["recommended_view_bucket"].map(
            {
                "safest_outreach": "prioritized outreach",
                "larger_land_target": "larger land acquisition",
                "vacant_buildable": "vacant buildable land search",
                "growth_edge_opportunity": "growth-edge targeting",
                "general_ranked": "general parcel review",
            }
        ).fillna("general parcel review")
    )
    frame["caution_flags"] = _normalize_string(frame.get("caution_flags"), index=frame.index).fillna(
        frame["amount_trust_tier"].map(
            {
                "use_with_caution": "Tax amount should be reviewed before outreach.",
                "not_trusted_for_prominent_display": "Tax amount should not be treated as reliable.",
                "not_reported": "No reported delinquent amount is currently available.",
            }
        )
    )
    frame["recommended_view_bucket"] = frame["recommended_view_bucket"].mask(
        frame["parcel_vacant_flag"] & frame["buildability_component"].ge(75),
        "vacant_buildable",
    )
    frame["recommended_view_bucket"] = frame["recommended_view_bucket"].mask(
        frame["growth_pressure_component"].ge(68),
        "growth_edge_opportunity",
    )
    frame["recommended_view_bucket"] = frame["recommended_view_bucket"].mask(
        acreage.ge(5) & frame["parcel_vacant_flag"],
        "larger_land_target",
    )
    return frame


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


def _detail_geometry(parcel_row_id: str) -> dict[str, Any] | None:
    geometry_bytes = None
    if PARCEL_MASTER_PATH.exists():
        dataset = ds.dataset(PARCEL_MASTER_PATH, format="parquet")
        table = dataset.to_table(columns=["parcel_row_id", "geometry"], filter=ds.field("parcel_row_id") == parcel_row_id)
        if table.num_rows:
            frame = table.to_pandas()
            geometry_bytes = frame.iloc[0]["geometry"]
    if geometry_bytes is None and _embedded_geometry_runtime_available():
        frame = _geometry_table_for_ids([parcel_row_id])
        if not frame.empty and "geometry" in frame.columns:
            geometry_bytes = frame.iloc[0]["geometry"]
    if geometry_bytes is None:
        base_frame = load_base_frame()
        if "geometry" in base_frame.columns:
            match = base_frame.loc[base_frame["parcel_row_id"].astype("string").eq(parcel_row_id), ["geometry"]]
            if not match.empty:
                geometry_bytes = match.iloc[0]["geometry"]
        if geometry_bytes is None:
            centroid_match = base_frame.loc[
                base_frame["parcel_row_id"].astype("string").eq(parcel_row_id),
                ["longitude", "latitude"],
            ]
            if not centroid_match.empty:
                longitude = pd.to_numeric(centroid_match.iloc[0].get("longitude"), errors="coerce")
                latitude = pd.to_numeric(centroid_match.iloc[0].get("latitude"), errors="coerce")
                if pd.notna(longitude) and pd.notna(latitude):
                    lng = round(float(longitude), 6)
                    lat = round(float(latitude), 6)
                    return {
                        "type": "Point",
                        "centroid": {"type": "Point", "coordinates": [lng, lat]},
                        "bounds": [lng, lat, lng, lat],
                    }
    if not geometry_bytes:
        return None
    if isinstance(geometry_bytes, dict):
        coordinates = geometry_bytes.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            lng = float(coordinates[0])
            lat = float(coordinates[1])
            return {
                "type": geometry_bytes.get("type", "Point"),
                "centroid": {"type": "Point", "coordinates": [round(lng, 6), round(lat, 6)]},
                "bounds": [round(lng, 6), round(lat, 6), round(lng, 6), round(lat, 6)],
            }
        return None
    geometry = wkb.loads(geometry_bytes)
    centroid = geometry.centroid
    bounds = geometry.bounds
    return {
        "type": geometry.geom_type,
        "centroid": {"type": "Point", "coordinates": [round(float(centroid.x), 6), round(float(centroid.y), 6)]},
        "bounds": [round(float(value), 6) for value in bounds],
    }


def _geometry_table_for_ids(parcel_row_ids: list[str]) -> pd.DataFrame:
    if PARCEL_MASTER_PATH.exists():
        dataset = ds.dataset(PARCEL_MASTER_PATH, format="parquet")
        table = dataset.to_table(columns=["parcel_row_id", "geometry"], filter=ds.field("parcel_row_id").isin(parcel_row_ids))
        return table.to_pandas()
    if _embedded_geometry_runtime_available():
        table = _embedded_geometry_dataset().to_table(columns=["parcel_row_id", "geometry"], filter=ds.field("parcel_row_id").isin(parcel_row_ids))
        return table.to_pandas()
    base_frame = load_base_frame()
    return base_frame.loc[base_frame["parcel_row_id"].astype("string").isin(parcel_row_ids), ["parcel_row_id", "geometry"]].copy()


def _render_mode_for_zoom(zoom: float | None) -> str:
    if zoom is None:
        return "polygons"
    if zoom < 7:
        return "none"
    if zoom < 9:
        return "points"
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


def _tile_simplification_tolerance(z: int) -> float:
    if z >= 15:
        return 0.0
    if z >= 13:
        return 0.00002
    if z >= 11:
        return 0.00008
    if z >= 9:
        return 0.0003
    return 0.0012


def _tile_feature_limit(z: int) -> int:
    if z >= 14:
        return 5000
    if z >= 13:
        return 3000
    if z >= 12:
        return 1800
    return 900


def runtime_file_diagnostics() -> dict[str, dict[str, int | bool | str | None]]:
    diagnostics: dict[str, dict[str, int | bool | str | None]] = {}
    paths = {
        "parcel_master": PARCEL_MASTER_PATH,
        "owner_leads": OWNER_LEADS_PATH,
        "building_metrics": BUILDING_METRICS_PATH,
        "lead_signals": LEAD_SIGNALS_PATH,
    }
    for name, path in paths.items():
        diagnostics[name] = {
            "cwd": str(Path.cwd()),
            "project_root": str(PROJECT_ROOT),
            "path": str(path.resolve(strict=False)),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    diagnostics["embedded_lead_signals"] = {
        "cwd": str(Path.cwd()),
        "project_root": str(PROJECT_ROOT),
        "path": str(EMBEDDED_LEAD_SIGNALS_PATH.resolve(strict=False)),
        "exists": EMBEDDED_LEAD_SIGNALS_PATH.exists(),
        "size_bytes": EMBEDDED_LEAD_SIGNALS_PATH.stat().st_size if EMBEDDED_LEAD_SIGNALS_PATH.exists() else None,
    }
    diagnostics["embedded_parcel_index"] = {
        "cwd": str(Path.cwd()),
        "project_root": str(PROJECT_ROOT),
        "path": str(EMBEDDED_PARCEL_INDEX_ROOT.resolve(strict=False)),
        "exists": _embedded_parcel_runtime_available(),
        "size_bytes": None,
    }
    diagnostics["static_feed"] = {
        "cwd": str(Path.cwd()),
        "project_root": str(PROJECT_ROOT),
        "path": str(MISSISSIPPI_STATIC_FEED_PATH.resolve(strict=False)),
        "exists": MISSISSIPPI_STATIC_FEED_PATH.exists(),
        "size_bytes": MISSISSIPPI_STATIC_FEED_PATH.stat().st_size if MISSISSIPPI_STATIC_FEED_PATH.exists() else None,
    }
    return diagnostics


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
    if _using_embedded_runtime():
        if (
            _is_default_filter_state(
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
            and sort_by == "lead_score_total"
            and sort_direction == "desc"
            and int(offset) == 0
        ):
            cached = _load_embedded_json(EMBEDDED_DEFAULT_LEADS_PATH)
            if cached is not None:
                requested_limit = _clamp_limit(limit, default=LEADS_DEFAULT_LIMIT, max_limit=LEADS_MAX_LIMIT)
                cached["limit"] = requested_limit
                cached["offset"] = 0
                cached["items"] = cached["items"][:requested_limit]
                return cached

        safe_limit = _clamp_limit(limit, default=LEADS_DEFAULT_LIMIT, max_limit=LEADS_MAX_LIMIT)
        safe_offset = max(int(offset), 0)
        expression = _embedded_filter_expression(
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
        total_count = _embedded_count_rows(expression)
        if sort_by not in SUMMARY_FIELDS:
            sort_by = "lead_score_total"
        ascending = sort_direction.lower() == "asc"
        filtered = _bounded_sorted_batches(
            columns=SUMMARY_FIELDS,
            expression=expression,
            sort_by=sort_by,
            ascending=ascending,
            keep_rows=safe_limit + safe_offset,
        )
        paged = filtered.iloc[safe_offset : safe_offset + safe_limit].copy()
        items = [{column: _serialize_scalar(row[column]) for column in SUMMARY_FIELDS} for _, row in paged.loc[:, SUMMARY_FIELDS].iterrows()]
        return {"total_count": total_count, "limit": safe_limit, "offset": safe_offset, "items": items}

    frame = load_base_frame()
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
    items = [{column: _serialize_scalar(row[column]) for column in SUMMARY_FIELDS} for _, row in paged.loc[:, SUMMARY_FIELDS].iterrows()]
    return {"total_count": total_count, "limit": safe_limit, "offset": safe_offset, "items": items}


def get_lead_detail(parcel_row_id: str) -> dict[str, Any] | None:
    if _using_embedded_runtime():
        columns = [column for column in _embedded_parcel_dataset().schema.names if not column.startswith("__")]
        frame = _embedded_to_pandas(columns, ds.field("parcel_row_id") == parcel_row_id)
        if frame.empty:
            return None
        row = frame.iloc[0]
        payload = {column: _serialize_scalar(row[column]) for column in frame.columns if column not in {"latitude", "longitude"}}
        payload.update(_lookup_embedded_detail_metrics(parcel_row_id))
        payload["geometry"] = _detail_geometry(parcel_row_id)
        if payload.get("ai_building_present_flag") is None:
            longitude = pd.to_numeric(row.get("longitude"), errors="coerce")
            latitude = pd.to_numeric(row.get("latitude"), errors="coerce")
            if pd.notna(longitude) and pd.notna(latitude):
                try:
                    ai_payload = _predict_ai_building_presence(
                        float(longitude),
                        float(latitude),
                        bool(payload.get("parcel_vacant_flag")),
                        pd.to_numeric(row.get("building_count"), errors="coerce"),
                        pd.to_numeric(row.get("building_area_total"), errors="coerce"),
                    )
                except Exception:
                    ai_payload = None
                if ai_payload:
                    payload.update(ai_payload)
        _apply_vacancy_assessment(payload)
        return payload

    frame = load_base_frame()
    match = frame.loc[frame["parcel_row_id"].astype("string").eq(parcel_row_id)]
    if match.empty:
        return None
    row = match.iloc[0]
    payload = {column: _serialize_scalar(row[column]) for column in frame.columns if column not in {"latitude", "longitude"}}
    payload["geometry"] = _detail_geometry(parcel_row_id)
    if payload.get("ai_building_present_flag") is None:
        longitude = pd.to_numeric(row.get("longitude"), errors="coerce")
        latitude = pd.to_numeric(row.get("latitude"), errors="coerce")
        if pd.notna(longitude) and pd.notna(latitude):
            try:
                    ai_payload = _predict_ai_building_presence(
                        float(longitude),
                        float(latitude),
                        bool(payload.get("parcel_vacant_flag")),
                        pd.to_numeric(row.get("building_count"), errors="coerce"),
                        pd.to_numeric(row.get("building_area_total"), errors="coerce"),
                    )
            except Exception:
                ai_payload = None
            if ai_payload:
                payload.update(ai_payload)
    _apply_vacancy_assessment(payload)
    return payload


def get_parcel_geometry(parcel_row_id: str, zoom: float | None = None) -> dict[str, Any]:
    effective_zoom = 14.0 if zoom is None else zoom
    return get_geometry(
        parcel_row_ids=[parcel_row_id],
        selected_parcel_id=parcel_row_id,
        zoom=effective_zoom,
        limit=1,
    )


def get_parcel_tile(z: int, x: int, y: int) -> bytes:
    started_at = time.perf_counter()
    tile_logger.info("parcel tile start z=%s x=%s y=%s", z, x, y)
    if z < PARCEL_TILE_MIN_ZOOM:
        payload = mapbox_vector_tile.encode({"name": PARCEL_TILE_LAYER, "features": []})
        tile_logger.info("parcel tile skipped z=%s x=%s y=%s reason=min_zoom bytes=%s", z, x, y, len(payload))
        return payload

    tile = mercantile.bounds(x, y, z)
    bounds = (tile.west, tile.south, tile.east, tile.north)
    if (
        bounds[2] < MISSISSIPPI_TILE_BOUNDS[0]
        or bounds[0] > MISSISSIPPI_TILE_BOUNDS[2]
        or bounds[3] < MISSISSIPPI_TILE_BOUNDS[1]
        or bounds[1] > MISSISSIPPI_TILE_BOUNDS[3]
    ):
        payload = mapbox_vector_tile.encode({"name": PARCEL_TILE_LAYER, "features": []})
        tile_logger.info("parcel tile skipped z=%s x=%s y=%s reason=outside_state bytes=%s", z, x, y, len(payload))
        return payload
    tile_columns = [
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "wetland_flag",
        "flood_risk_score",
        "road_access_tier",
        "county_hosted_flag",
        "best_source_type",
        "latitude",
        "longitude",
    ]

    if _using_embedded_runtime():
        query_started = time.perf_counter()
        expression = _embedded_filter_expression(bounds=bounds)
        frame = _embedded_to_pandas(tile_columns, expression).head(_tile_feature_limit(z))
        tile_logger.info(
            "parcel tile query z=%s x=%s y=%s runtime=embedded candidates=%s elapsed_ms=%s",
            z,
            x,
            y,
            len(frame),
            round((time.perf_counter() - query_started) * 1000, 1),
        )
    else:
        query_started = time.perf_counter()
        frame = load_base_frame()
        bbox_mask = (
            pd.to_numeric(frame["longitude"], errors="coerce").between(bounds[0], bounds[2]).fillna(False)
            & pd.to_numeric(frame["latitude"], errors="coerce").between(bounds[1], bounds[3]).fillna(False)
        )
        frame = frame.loc[bbox_mask, tile_columns].head(_tile_feature_limit(z)).copy()
        tile_logger.info(
            "parcel tile query z=%s x=%s y=%s runtime=full candidates=%s elapsed_ms=%s",
            z,
            x,
            y,
            len(frame),
            round((time.perf_counter() - query_started) * 1000, 1),
        )

    if frame.empty:
        payload = mapbox_vector_tile.encode({"name": PARCEL_TILE_LAYER, "features": []})
        tile_logger.info(
            "parcel tile empty z=%s x=%s y=%s bytes=%s elapsed_ms=%s",
            z,
            x,
            y,
            len(payload),
            round((time.perf_counter() - started_at) * 1000, 1),
        )
        return payload

    geometry_frame = _geometry_table_for_ids(frame["parcel_row_id"].astype(str).tolist())
    geometry_lookup = dict(zip(geometry_frame["parcel_row_id"].astype(str), geometry_frame["geometry"]))
    tolerance = _tile_simplification_tolerance(z)
    features: list[dict[str, Any]] = []
    skipped = 0
    for _, row in frame.iterrows():
        parcel_row_id = str(row["parcel_row_id"])
        geometry_value = geometry_lookup.get(parcel_row_id)
        if geometry_value is None:
            skipped += 1
            continue
        if isinstance(geometry_value, dict):
            skipped += 1
            continue
        try:
            shape = wkb.loads(geometry_value)
            rendered_shape = shape.simplify(tolerance, preserve_topology=True) if tolerance > 0 else shape
            features.append(
                {
                    "id": parcel_row_id,
                    "geometry": mapping(rendered_shape),
                    "properties": {
                        "parcel_row_id": parcel_row_id,
                        "parcel_id": _serialize_scalar(row.get("parcel_id")),
                        "county_name": _serialize_scalar(row.get("county_name")),
                        "wetland_flag": bool(_serialize_scalar(row.get("wetland_flag"))),
                        "flood_risk_score": _serialize_scalar(row.get("flood_risk_score")),
                        "road_access_tier": _serialize_scalar(row.get("road_access_tier")),
                        "county_hosted_flag": bool(_serialize_scalar(row.get("county_hosted_flag"))),
                        "best_source_type": _serialize_scalar(row.get("best_source_type")),
                    },
                }
            )
        except Exception:
            skipped += 1
            tile_logger.exception("parcel tile feature decode failed z=%s x=%s y=%s parcel_row_id=%s", z, x, y, parcel_row_id)

    payload = mapbox_vector_tile.encode(
        {"name": PARCEL_TILE_LAYER, "features": features},
        default_options={"quantize_bounds": bounds, "extents": 4096},
    )
    tile_logger.info(
        "parcel tile complete z=%s x=%s y=%s features=%s skipped=%s bytes=%s elapsed_ms=%s",
        z,
        x,
        y,
        len(features),
        skipped,
        len(payload),
        round((time.perf_counter() - started_at) * 1000, 1),
    )
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
    if _using_embedded_runtime():
        if (
            parcel_row_ids is None
            and selected_parcel_id is None
            and bounds == (-91.65, 30.15, -88.0, 35.1)
            and zoom is not None
            and zoom <= 6.5
            and _is_default_filter_state(
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
        ):
            cached = _load_embedded_json(EMBEDDED_DEFAULT_GEOMETRY_PATH)
            if cached is not None:
                return cached

        geometry_columns = [
            "parcel_row_id",
            "parcel_id",
            "county_name",
            "latitude",
            "longitude",
            "lead_score_total",
            "lead_score_tier",
            "parcel_vacant_flag",
            "wetland_flag",
            "flood_risk_score",
            "road_access_tier",
            "county_hosted_flag",
            "best_source_type",
            "geometry",
        ]
        expression = _embedded_filter_expression(
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
            parcel_row_ids=parcel_row_ids,
            selected_parcel_id=selected_parcel_id,
            bounds=bounds,
        )
        render_mode = _render_mode_for_zoom(zoom)
        safe_limit = _clamp_limit(limit, default=GEOMETRY_DEFAULT_LIMIT, max_limit=GEOMETRY_MAX_LIMIT)
        if render_mode == "none" and not parcel_row_ids and not selected_parcel_id:
            return {
                "geometry_mode": "viewport_geojson",
                "render_mode": "none",
                "geometry_bounds": None,
                "geometry_view_box": None,
                "requested_bounds": [round(value, 6) for value in bounds] if bounds is not None else None,
                "zoom": zoom,
                "feature_count": 0,
                "feature_collection": {"type": "FeatureCollection", "features": []},
                "items": [],
            }

        if parcel_row_ids:
            filtered = _embedded_to_pandas(geometry_columns, expression)
        elif bounds is not None or selected_parcel_id:
            filtered = _bounded_scan_batches(
                columns=geometry_columns,
                expression=expression,
                keep_rows=safe_limit,
            )
        else:
            filtered = _bounded_sorted_batches(
                columns=geometry_columns,
                expression=expression,
                sort_by="lead_score_total",
                ascending=False,
                keep_rows=safe_limit,
            )

        features: list[dict[str, Any]] = []
        if render_mode in {"points", "centroids"}:
            for _, row in filtered.iterrows():
                lng = _serialize_scalar(row.get("longitude"))
                lat = _serialize_scalar(row.get("latitude"))
                if lng is None or lat is None:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [round(float(lng), 6), round(float(lat), 6)]},
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
                )
        else:
            geometry_frame = _geometry_table_for_ids(filtered["parcel_row_id"].astype(str).tolist())
            geometry_lookup = dict(zip(geometry_frame["parcel_row_id"].astype(str), geometry_frame["geometry"]))
            feature_bounds: list[list[float]] = []
            for _, row in filtered.iterrows():
                geometry_value = geometry_lookup.get(str(row["parcel_row_id"]))
                if geometry_value is None:
                    continue
                if isinstance(geometry_value, dict):
                    rendered_geometry = geometry_value
                else:
                    rendered_geometry = mapping(wkb.loads(geometry_value))
                bounds_value = _geojson_bounds(rendered_geometry)
                if bounds_value is not None:
                    feature_bounds.append(bounds_value)
                features.append(
                    {
                        "type": "Feature",
                        "geometry": rendered_geometry,
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
                )

        bounds_payload = None
        if features:
            if render_mode in {"points", "centroids"}:
                lngs = [feature["geometry"]["coordinates"][0] for feature in features]
                lats = [feature["geometry"]["coordinates"][1] for feature in features]
                bounds_payload = [round(min(lngs), 6), round(min(lats), 6), round(max(lngs), 6), round(max(lats), 6)]
            elif feature_bounds:
                bounds_payload = [
                    round(min(bound[0] for bound in feature_bounds), 6),
                    round(min(bound[1] for bound in feature_bounds), 6),
                    round(max(bound[2] for bound in feature_bounds), 6),
                    round(max(bound[3] for bound in feature_bounds), 6),
                ]

        items = [
            {"parcel_row_id": str(row["parcel_row_id"]), "path": None, "lead_score_total": _serialize_scalar(row["lead_score_total"])}
            for _, row in filtered.iterrows()
        ]
        return {
            "geometry_mode": "viewport_geojson",
            "render_mode": render_mode,
            "geometry_bounds": bounds_payload,
            "geometry_view_box": None,
            "requested_bounds": [round(value, 6) for value in bounds] if bounds is not None else None,
            "zoom": zoom,
            "feature_count": len(features),
            "feature_collection": {"type": "FeatureCollection", "features": features},
            "items": items,
        }

    frame = load_base_frame()
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
        if bounds is not None:
            min_lng, min_lat, max_lng, max_lat = bounds
            bbox_mask = (
                pd.to_numeric(filtered["longitude"], errors="coerce").between(min_lng, max_lng).fillna(False)
                & pd.to_numeric(filtered["latitude"], errors="coerce").between(min_lat, max_lat).fillna(False)
            )
            if selected_parcel_id:
                bbox_mask = bbox_mask | filtered["parcel_row_id"].astype("string").eq(selected_parcel_id)
            filtered = filtered.loc[bbox_mask].copy()
    render_mode = _render_mode_for_zoom(zoom)
    safe_limit = _clamp_limit(limit, default=GEOMETRY_DEFAULT_LIMIT, max_limit=GEOMETRY_MAX_LIMIT)
    if render_mode == "none" and not parcel_row_ids and not selected_parcel_id:
        return {
            "geometry_mode": "viewport_geojson",
            "render_mode": "none",
            "geometry_bounds": None,
            "geometry_view_box": None,
            "requested_bounds": [round(value, 6) for value in bounds] if bounds is not None else None,
            "zoom": zoom,
            "feature_count": 0,
            "feature_collection": {"type": "FeatureCollection", "features": []},
            "items": [],
        }
    if not parcel_row_ids and (bounds is not None or selected_parcel_id):
        filtered = filtered.head(safe_limit)
    else:
        filtered = filtered.sort_values("lead_score_total", ascending=False, na_position="last").head(safe_limit)
    features: list[dict[str, Any]] = []

    if render_mode in {"points", "centroids"}:
        for _, row in filtered.iterrows():
            lng = _serialize_scalar(row.get("longitude"))
            lat = _serialize_scalar(row.get("latitude"))
            if lng is None or lat is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [round(float(lng), 6), round(float(lat), 6)]},
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
            )
        bounds_payload = None
        if features:
            lngs = [feature["geometry"]["coordinates"][0] for feature in features]
            lats = [feature["geometry"]["coordinates"][1] for feature in features]
            bounds_payload = [round(min(lngs), 6), round(min(lats), 6), round(max(lngs), 6), round(max(lats), 6)]
    else:
        geometry_frame = _geometry_table_for_ids(filtered["parcel_row_id"].astype(str).tolist())
        geometry_lookup = dict(zip(geometry_frame["parcel_row_id"].astype(str), geometry_frame["geometry"]))
        feature_bounds: list[tuple[float, float, float, float]] = []
        for _, row in filtered.iterrows():
            parcel_row_id = str(row["parcel_row_id"])
            geometry_bytes = geometry_lookup.get(parcel_row_id)
            if not geometry_bytes:
                continue
            if isinstance(geometry_bytes, dict):
                coordinates = geometry_bytes.get("coordinates")
                if not (isinstance(coordinates, list) and len(coordinates) >= 2):
                    continue
                lng = float(coordinates[0])
                lat = float(coordinates[1])
                feature_bounds.append((lng, lat, lng, lat))
                features.append(
                    {
                        "type": "Feature",
                        "geometry": geometry_bytes,
                        "properties": {
                            "parcel_row_id": parcel_row_id,
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
                            "selected": parcel_row_id == (selected_parcel_id or ""),
                        },
                    }
                )
                continue
            shape = wkb.loads(geometry_bytes)
            tolerance = _simplification_tolerance(zoom)
            rendered_shape = shape.simplify(tolerance, preserve_topology=True) if tolerance > 0 else shape
            feature_bounds.append(shape.bounds)
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(rendered_shape),
                    "properties": {
                        "parcel_row_id": parcel_row_id,
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
                        "selected": parcel_row_id == (selected_parcel_id or ""),
                    },
                }
            )
        bounds_payload = None
        if feature_bounds:
            bounds_payload = [
                round(min(bound[0] for bound in feature_bounds), 6),
                round(min(bound[1] for bound in feature_bounds), 6),
                round(max(bound[2] for bound in feature_bounds), 6),
                round(max(bound[3] for bound in feature_bounds), 6),
            ]

    items = [
        {"parcel_row_id": str(row["parcel_row_id"]), "path": None, "lead_score_total": _serialize_scalar(row["lead_score_total"])}
        for _, row in filtered.iterrows()
    ]
    return {
        "geometry_mode": "viewport_geojson",
        "render_mode": render_mode,
        "geometry_bounds": bounds_payload,
        "geometry_view_box": None,
        "requested_bounds": [round(value, 6) for value in bounds] if bounds is not None else None,
        "zoom": zoom,
        "feature_count": len(features),
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "items": items,
    }


def get_presets() -> list[dict[str, Any]]:
    if _using_embedded_runtime():
        cached = _load_embedded_json(EMBEDDED_PRESETS_PATH)
        if cached is not None:
            return cached
        presets: list[dict[str, Any]] = []
        for view_name, definition in PRESET_DEFINITIONS.items():
            expression = _embedded_filter_expression(**definition["filters"])
            row_count = _embedded_count_rows(expression)
            score_sum = 0.0
            score_count = 0
            scanner = _embedded_parcel_dataset().scanner(columns=["lead_score_total"], filter=expression, batch_size=50000)
            for batch in scanner.to_batches():
                frame = batch.to_pandas()
                if frame.empty:
                    continue
                scores = pd.to_numeric(frame["lead_score_total"], errors="coerce")
                score_sum += float(scores.fillna(0).sum())
                score_count += int(scores.notna().sum())
            mean_score = (score_sum / score_count) if score_count else 0
            presets.append(
                {
                    "view_name": view_name,
                    "description": definition["description"],
                    "filter_expression": definition["filter_expression"],
                    "row_count": str(row_count),
                    "average_lead_score": f"{mean_score:.1f}",
                }
            )
        return presets

    frame = load_base_frame()
    presets: list[dict[str, Any]] = []
    for view_name, definition in PRESET_DEFINITIONS.items():
        filtered = _apply_filters(frame, **definition["filters"])
        mean_score = pd.to_numeric(filtered["lead_score_total"], errors="coerce").mean() if len(filtered) else 0
        presets.append(
            {
                "view_name": view_name,
                "description": definition["description"],
                "filter_expression": definition["filter_expression"],
                "row_count": str(len(filtered)),
                "average_lead_score": f"{mean_score:.1f}",
            }
        )
    return presets


def get_summary() -> dict[str, Any]:
    if _using_embedded_runtime():
        cached = _load_embedded_json(EMBEDDED_SUMMARY_PATH)
        if cached is not None:
            return cached
        summary_columns = ["county_name", "recommended_view_bucket", "lead_score_total", "parcel_vacant_flag", "county_hosted_flag"]
        total_rows = 0
        vacant_rows = 0
        county_hosted_rows = 0
        score_sum = 0.0
        score_count = 0
        county_counts: Counter[str] = Counter()
        recommended_counts: Counter[str] = Counter()
        scanner = _embedded_parcel_dataset().scanner(columns=summary_columns, batch_size=50000)
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if frame.empty:
                continue
            total_rows += len(frame)
            vacant_rows += int(frame["parcel_vacant_flag"].fillna(False).sum())
            county_hosted_rows += int(frame["county_hosted_flag"].fillna(False).sum())
            scores = pd.to_numeric(frame["lead_score_total"], errors="coerce")
            score_sum += float(scores.fillna(0).sum())
            score_count += int(scores.notna().sum())
            county_counts.update(frame["county_name"].dropna().astype(str).tolist())
            recommended_counts.update(frame["recommended_view_bucket"].dropna().astype(str).tolist())
        top_counties = county_counts.most_common(20)
        average_score = (score_sum / score_count) if score_count else 0.0
        return {
            "row_count": int(total_rows),
            "source": "mississippi parcel runtime dataset",
            "geometry_mode": "viewport_geojson",
            "sections": {
                "statewide": [
                    {"section": "statewide", "metric": "lead_count", "value": str(total_rows)},
                    {"section": "statewide", "metric": "average_lead_score", "value": f"{average_score:.1f}"},
                    {"section": "statewide", "metric": "likely_vacant_count", "value": str(vacant_rows)},
                    {"section": "statewide", "metric": "vacant_share_pct", "value": f"{(vacant_rows / total_rows * 100) if total_rows else 0:.1f}"},
                    {"section": "statewide", "metric": "county_hosted_share_pct", "value": f"{(county_hosted_rows / total_rows * 100) if total_rows else 0:.1f}"},
                ],
                "top_counties": [
                    {"section": "top_counties", "key": county, "metric": "parcel_count", "value": str(int(count))}
                    for county, count in top_counties
                ],
                "recommended_view_bucket": [
                    {"section": "recommended_view_bucket", "key": bucket, "metric": "parcel_count", "value": str(int(count))}
                    for bucket, count in recommended_counts.most_common()
                ],
            },
        }

    frame = load_base_frame()
    county_counts = frame.groupby("county_name", dropna=True).size().sort_values(ascending=False).head(20)
    recommended_counts = frame.groupby("recommended_view_bucket", dropna=True).size().sort_values(ascending=False)
    average_score = pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()
    source_label = (
        "mississippi_parcels_master + owner leads + building metrics + motivation signals"
        if _full_runtime_available()
        else (
            "mississippi parcel runtime dataset"
            if _embedded_parcel_runtime_available()
            else ("mississippi lead dataset" if LEAD_SIGNALS_PATH.exists() or EMBEDDED_LEAD_SIGNALS_PATH.exists() else "mississippi explorer static feed")
        )
    )
    return {
        "row_count": int(len(frame)),
        "source": source_label,
        "geometry_mode": "viewport_geojson",
        "sections": {
            "statewide": [
                {"section": "statewide", "metric": "lead_count", "value": str(len(frame))},
                {"section": "statewide", "metric": "average_lead_score", "value": f"{average_score:.1f}"},
                {"section": "statewide", "metric": "likely_vacant_count", "value": str(int(frame["parcel_vacant_flag"].fillna(False).sum()))},
                {"section": "statewide", "metric": "vacant_share_pct", "value": f"{frame['parcel_vacant_flag'].fillna(False).mean() * 100:.1f}"},
                {"section": "statewide", "metric": "county_hosted_share_pct", "value": f"{frame['county_hosted_flag'].fillna(False).mean() * 100:.1f}"},
            ],
            "top_counties": [
                {"section": "top_counties", "key": county, "metric": "parcel_count", "value": str(int(count))}
                for county, count in county_counts.items()
            ],
            "recommended_view_bucket": [
                {"section": "recommended_view_bucket", "key": bucket, "metric": "parcel_count", "value": str(int(count))}
                for bucket, count in recommended_counts.items()
            ],
        },
    }
