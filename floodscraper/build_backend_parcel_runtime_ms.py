from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from shapely import wkb


ROOT = Path(__file__).resolve().parents[1]
PARCEL_MASTER_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
OWNER_LEADS_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_owner_leads.parquet"
BUILDING_METRICS_PATH = ROOT / "data" / "buildings_processed" / "parcel_building_metrics.parquet"
AI_BUILDING_PREDICTIONS_PATH = ROOT / "data" / "buildings_processed" / "ai_building_presence_predictions_ms.parquet"
LEAD_SIGNALS_PATH = ROOT / "data" / "tax_published" / "ms" / "app_ready_mississippi_leads.parquet"
DELINQUENT_LEADS_PATH = ROOT / "data" / "tax_published" / "ms" / "delinquent_leads_statewide.parquet"
TAX_DISTRESS_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_tax_distress.parquet"
COUNTY_COVERAGE_MATRIX_PATH = ROOT / "data" / "parcels" / "mississippi_tax_coverage_matrix.parquet"
OUTPUT_ROOT = ROOT / "backend" / "runtime" / "mississippi" / "parcel_index"
RUNTIME_ROOT = ROOT / "backend" / "runtime" / "mississippi"
GEOMETRY_OUTPUT_ROOT = RUNTIME_ROOT / "parcel_geometry_index"
DETAIL_METRICS_OUTPUT_PATH = RUNTIME_ROOT / "parcel_detail_metrics.parquet"
COUNTY_COVERAGE_MATRIX_RUNTIME_PATH = RUNTIME_ROOT / "tax_coverage_matrix.parquet"
SQFT_PER_ACRE = 43560.0


PARCEL_COLUMNS = [
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
    "geometry",
]

OWNER_COLUMNS = [
    "parcel_row_id",
    "owner_type",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "owner_parcel_count",
    "owner_total_acres",
    "mailer_target_score",
    "corporate_owner_flag",
]

BUILDING_COLUMNS = [
    "parcel_row_id",
    "building_count",
    "building_area_total",
    "parcel_vacant_flag",
    "nearby_building_count_1km",
    "nearby_building_density",
    "growth_pressure_bucket",
]

SIGNAL_COLUMNS = [
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
]

DELINQUENT_LEAD_COLUMNS = [
    "parcel_row_id",
    "tax_year",
    "latest_delinquent_year",
    "latest_loaded_at",
    "best_source_name",
    "best_source_file_path",
]

TAX_DISTRESS_COLUMNS = [
    "parcel_row_id",
    "county_tax_source_configured_flag",
    "county_tax_source_loaded_flag",
    "county_tax_source_type",
    "county_tax_source_name",
    "county_tax_source_url",
    "county_tax_source_path",
    "county_tax_coverage_scope",
    "county_tax_quality_flag",
    "county_tax_blocker_reason",
    "county_tax_last_successful_ingest_timestamp",
    "tax_data_available_flag",
    "delinquent_flag",
    "delinquent_amount",
    "delinquent_amount_bucket",
    "delinquent_year",
    "tax_sale_flag",
    "tax_sale_date",
    "latest_delinquent_year",
    "parcel_tax_status",
    "county_tax_coverage_status",
    "county_tax_coverage_note",
    "county_tax_coverage_reason",
    "tax_data_year",
    "tax_data_upload_date",
    "tax_data_source",
    "delinquency_last_verified",
    "tax_source_name",
]


def coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column not in frame.columns:
            continue
        result = result.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return result


def acreage_bucket(series: pd.Series) -> pd.Series:
    acres = pd.to_numeric(series, errors="coerce")
    bucket = pd.Series(pd.NA, index=series.index, dtype="string")
    bucket.loc[acres.lt(1)] = "<1"
    bucket.loc[acres.ge(1) & acres.lt(5)] = "1-4.99"
    bucket.loc[acres.ge(5) & acres.lt(20)] = "5-19.99"
    bucket.loc[acres.ge(20) & acres.lt(100)] = "20-99.99"
    bucket.loc[acres.ge(100)] = "100+"
    return bucket


def point_geometry_from_wkb(value: bytes | None) -> dict[str, object] | None:
    if not value:
        return None
    try:
        shape = wkb.loads(value)
    except Exception:
        return None
    centroid = shape.centroid
    return {
        "type": "Point",
        "coordinates": [round(float(centroid.x), 6), round(float(centroid.y), 6)],
    }


def simplified_polygon_wkb(value: bytes | None, tolerance: float = 0.00002) -> bytes | None:
    if not value:
        return None
    try:
        shape = wkb.loads(value)
    except Exception:
        return None
    return shape.simplify(tolerance, preserve_topology=True).wkb


def json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def normalize_timestamp_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    normalized = normalize_string(series, index=index)
    if normalized.empty:
        return normalized
    parsed = pd.to_datetime(normalized, errors="coerce", utc=True)
    formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype("string")
    return formatted.where(parsed.notna(), normalized)


def iso_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def apply_tax_freshness_fields(frame: pd.DataFrame) -> pd.DataFrame:
    tax_year = pd.to_numeric(frame["tax_year"], errors="coerce").astype("Int64") if "tax_year" in frame.columns else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    latest_delinquent_year = (
        pd.to_numeric(frame["latest_delinquent_year"], errors="coerce").astype("Int64")
        if "latest_delinquent_year" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="Int64")
    )
    if "latest_delinquent_year_taxdistress" in frame.columns:
        latest_delinquent_year = latest_delinquent_year.fillna(
            pd.to_numeric(frame.get("latest_delinquent_year_taxdistress"), errors="coerce").astype("Int64")
        )
    latest_loaded_at = normalize_timestamp_string(frame.get("latest_loaded_at"), index=frame.index)
    default_upload = iso_mtime(DELINQUENT_LEADS_PATH if DELINQUENT_LEADS_PATH.exists() else LEAD_SIGNALS_PATH)
    tax_available_mask = (
        frame.get("delinquent_flag", pd.Series(False, index=frame.index)).fillna(False)
        | frame.get("tax_data_available_flag", pd.Series(False, index=frame.index)).fillna(False)
        | frame.get("county_tax_source_loaded_flag", pd.Series(False, index=frame.index)).fillna(False)
    )
    if default_upload:
        latest_loaded_at = latest_loaded_at.where(latest_loaded_at.notna(), pd.Series(np.where(tax_available_mask, default_upload, pd.NA), index=frame.index, dtype="object"))

    tax_data_source = normalize_string(frame.get("tax_source_name"), index=frame.index)
    tax_data_source = tax_data_source.fillna(normalize_string(frame.get("best_source_name_delinq"), index=frame.index))
    tax_data_source = tax_data_source.fillna(normalize_string(frame.get("best_source_name"), index=frame.index))
    tax_data_source = tax_data_source.fillna(normalize_string(frame.get("county_tax_source_type"), index=frame.index))
    tax_data_source = tax_data_source.where(tax_available_mask, pd.NA)

    frame["tax_year"] = tax_year
    frame["latest_delinquent_year"] = latest_delinquent_year
    frame["delinquent_year"] = latest_delinquent_year.fillna(tax_year)
    frame["tax_data_year"] = tax_year.fillna(latest_delinquent_year)
    frame["tax_data_upload_date"] = latest_loaded_at
    frame["tax_data_source"] = tax_data_source
    frame["delinquency_last_verified"] = latest_loaded_at
    return frame


def build_county_tax_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    county_frame = frame.loc[:, [
        "county_name",
        "county_tax_source_configured_flag",
        "county_tax_source_loaded_flag",
        "tax_data_available_flag",
        "tax_data_upload_date",
        "tax_data_year",
        "delinquent_flag",
    ]].copy()
    county_frame["county_tax_source_configured_flag"] = county_frame["county_tax_source_configured_flag"].fillna(False)
    county_frame["county_tax_source_loaded_flag"] = county_frame["county_tax_source_loaded_flag"].fillna(False)
    county_frame["tax_data_available_flag"] = county_frame["tax_data_available_flag"].fillna(False)
    county_frame["delinquent_flag"] = county_frame["delinquent_flag"].fillna(False)
    grouped = county_frame.groupby("county_name", dropna=False).agg(
        county_tax_source_configured_flag=("county_tax_source_configured_flag", "max"),
        county_tax_source_loaded_flag=("county_tax_source_loaded_flag", "max"),
        tax_data_available_flag=("tax_data_available_flag", "max"),
        observed_delinquency_records=("delinquent_flag", "sum"),
        latest_tax_data_upload_date=("tax_data_upload_date", "max"),
        latest_tax_data_year=("tax_data_year", "max"),
    ).reset_index()
    uploaded = pd.to_datetime(grouped["latest_tax_data_upload_date"], errors="coerce", utc=True)
    current_timestamp = pd.Timestamp.now("UTC")
    current_year = current_timestamp.year
    stale_mask = (
        uploaded.lt(current_timestamp - pd.Timedelta(days=365))
        | pd.to_numeric(grouped["latest_tax_data_year"], errors="coerce").lt(current_year - 1).fillna(False)
    )
    has_full_county_data = grouped["county_tax_source_loaded_flag"].fillna(False) & grouped["tax_data_available_flag"].fillna(False)
    has_partial_data = grouped["observed_delinquency_records"].gt(0)
    grouped["county_tax_coverage_status"] = "unavailable"
    grouped.loc[has_partial_data, "county_tax_coverage_status"] = "partial"
    grouped.loc[has_full_county_data, "county_tax_coverage_status"] = "available"
    grouped.loc[(has_full_county_data | has_partial_data) & stale_mask, "county_tax_coverage_status"] = "stale"
    grouped["county_tax_coverage_reason"] = pd.Series("No usable county tax source is currently available.", index=grouped.index, dtype="string")
    grouped.loc[grouped["county_tax_source_loaded_flag"].fillna(False) & ~grouped["tax_data_available_flag"].fillna(False), "county_tax_coverage_reason"] = "County tax source loaded, but no linked delinquency records were produced yet."
    grouped.loc[grouped["county_tax_coverage_status"].eq("partial"), "county_tax_coverage_reason"] = "Only partial county tax delinquency coverage is currently available."
    grouped.loc[grouped["county_tax_coverage_status"].eq("available"), "county_tax_coverage_reason"] = "County tax delinquency coverage is available."
    grouped.loc[grouped["county_tax_coverage_status"].eq("stale"), "county_tax_coverage_reason"] = "County tax delinquency coverage exists but appears stale."
    grouped["county_tax_coverage_note"] = grouped["county_tax_coverage_reason"]
    return grouped


def apply_county_tax_coverage_fields(frame: pd.DataFrame) -> pd.DataFrame:
    county_coverage = build_county_tax_coverage(frame)
    frame = frame.merge(county_coverage, on="county_name", how="left", suffixes=("", "_county"))
    frame["county_tax_source_configured_flag"] = frame["county_tax_source_configured_flag"].fillna(frame.get("county_tax_source_configured_flag_county"))
    frame["county_tax_source_loaded_flag"] = frame["county_tax_source_loaded_flag"].fillna(frame.get("county_tax_source_loaded_flag_county"))
    frame["tax_data_available_flag"] = frame["tax_data_available_flag"].fillna(frame.get("tax_data_available_flag_county"))
    computed_status = pd.Series("unavailable", index=frame.index, dtype="string")
    computed_status = computed_status.mask(frame.get("county_tax_coverage_status_county", pd.Series(pd.NA, index=frame.index)).notna(), normalize_string(frame.get("county_tax_coverage_status_county"), index=frame.index))
    existing_status = normalize_string(frame.get("county_tax_coverage_status"), index=frame.index)
    frame["county_tax_coverage_status"] = existing_status.fillna(computed_status)
    computed_reason = normalize_string(frame.get("county_tax_coverage_reason_county"), index=frame.index)
    existing_note = normalize_string(frame.get("county_tax_coverage_note"), index=frame.index)
    existing_reason = normalize_string(frame.get("county_tax_coverage_reason"), index=frame.index)
    frame["county_tax_coverage_note"] = existing_note.fillna(existing_reason).fillna(computed_reason)
    frame["county_tax_coverage_reason"] = existing_reason.fillna(frame["county_tax_coverage_note"]).fillna(computed_reason)
    existing_parcel_status = normalize_string(frame.get("parcel_tax_status"), index=frame.index)
    computed_parcel_status = pd.Series("county coverage unavailable", index=frame.index, dtype="string")
    computed_parcel_status.loc[frame["county_tax_coverage_status"].eq("stale")] = "county data stale"
    computed_parcel_status.loc[frame["county_tax_coverage_status"].eq("partial")] = "county coverage partial"
    computed_parcel_status.loc[frame["county_tax_coverage_status"].eq("available")] = "not delinquent"
    computed_parcel_status.loc[frame["delinquent_flag"].fillna(False)] = "delinquent"
    frame["parcel_tax_status"] = existing_parcel_status.fillna(computed_parcel_status)
    return frame.drop(
        columns=[
            "county_tax_source_configured_flag_county",
            "county_tax_source_loaded_flag_county",
            "tax_data_available_flag_county",
            "observed_delinquency_records",
            "latest_tax_data_upload_date",
            "latest_tax_data_year",
            "county_tax_coverage_status_county",
            "county_tax_coverage_reason_county",
        ],
        errors="ignore",
    )


def vacancy_confidence(frame: pd.DataFrame) -> pd.Series:
    building_count = pd.to_numeric(frame.get("building_count"), errors="coerce").fillna(0)
    building_area_total = pd.to_numeric(frame.get("building_area_total"), errors="coerce").fillna(0)
    acreage = pd.to_numeric(frame.get("acreage"), errors="coerce").fillna(0)
    assessed_total_value = pd.to_numeric(frame.get("assessed_total_value"), errors="coerce").fillna(0)
    road_distance_ft = pd.to_numeric(frame.get("road_distance_ft"), errors="coerce")
    nearby_building_density = pd.to_numeric(frame.get("nearby_building_density"), errors="coerce").fillna(0)
    parcel_vacant = frame["parcel_vacant_flag"].fillna(False)

    confidence = pd.Series(45.0, index=frame.index, dtype="float64")
    confidence = confidence.mask(parcel_vacant & building_count.eq(0) & building_area_total.le(0), 92.0)
    confidence = confidence.mask(parcel_vacant & building_count.le(1) & building_area_total.le(750) & acreage.ge(1), 78.0)
    confidence = confidence.mask(parcel_vacant & assessed_total_value.ge(25000), 18.0)
    confidence = confidence.mask(parcel_vacant & assessed_total_value.ge(10000) & road_distance_ft.le(150).fillna(False), 28.0)
    confidence = confidence.mask(parcel_vacant & nearby_building_density.ge(120) & acreage.le(2), 35.0)
    confidence = confidence.mask(~parcel_vacant & building_count.gt(0), 18.0)
    confidence = confidence.mask(~parcel_vacant & building_area_total.gt(1500), 8.0)
    return confidence.clip(0, 100)


def derive_shape_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    area_sqft = pd.to_numeric(frame.get("shape_area"), errors="coerce")
    perimeter_ft = pd.to_numeric(frame.get("shape_length"), errors="coerce")
    compactness = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid_shape = area_sqft.gt(0) & perimeter_ft.gt(0)
    compactness.loc[valid_shape] = ((4.0 * np.pi * area_sqft.loc[valid_shape]) / np.square(perimeter_ft.loc[valid_shape])).clip(0.0, 1.0)

    semi_perimeter = perimeter_ft / 2.0
    discriminant = np.square(semi_perimeter) - (4.0 * area_sqft)
    discriminant = discriminant.where(discriminant.ge(0))
    sqrt_disc = np.sqrt(discriminant)
    frontage = ((semi_perimeter + sqrt_disc) / 2.0).where(valid_shape)
    width = ((semi_perimeter - sqrt_disc) / 2.0).where(valid_shape)
    frame["shape_compactness"] = compactness
    frame["parcel_frontage_ft_estimate"] = frontage.where(frontage.gt(0))
    frame["parcel_width_ft_estimate"] = width.where(width.gt(0))
    return frame


def build_detail_metrics_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    detail_columns = [
        "parcel_row_id",
        "acreage_bucket",
        "county_tax_source_configured_flag",
        "county_tax_source_loaded_flag",
        "county_tax_source_name",
        "county_tax_source_url",
        "county_tax_coverage_scope",
        "county_tax_quality_flag",
        "county_tax_blocker_reason",
        "county_tax_last_successful_ingest_timestamp",
        "tax_data_available_flag",
        "county_tax_coverage_status",
        "county_tax_coverage_note",
        "county_tax_coverage_reason",
        "parcel_tax_status",
        "delinquent_amount",
        "delinquent_amount_bucket",
        "delinquent_year",
        "tax_sale_flag",
        "tax_sale_date",
        "tax_data_upload_date",
        "tax_data_year",
        "tax_data_source",
        "delinquency_last_verified",
        "mean_slope_pct",
        "max_slope_pct",
        "slope_class",
        "slope_score",
        "elevation_mean_ft",
        "shape_compactness",
        "parcel_frontage_ft_estimate",
        "parcel_width_ft_estimate",
        "wetland_pct",
        "wetland_area_sqft",
        "flood_pct",
        "flood_area_sqft",
        "primary_fema_zone",
        "wetland_flag",
        "flood_risk_score",
    ]
    available = [column for column in detail_columns if column in frame.columns]
    return frame.loc[:, available].copy()


def build_runtime_frame() -> pd.DataFrame:
    parcels = pd.read_parquet(PARCEL_MASTER_PATH, columns=PARCEL_COLUMNS, engine="pyarrow")
    parcels["acreage"] = coalesce_numeric(parcels, ["total_acres", "parcel_area_acres", "gis_acres", "tax_acres"])
    parcels["acreage_bucket"] = acreage_bucket(parcels["acreage"])
    parcels["land_use"] = parcels["land_use_raw"].astype("string").str.strip()
    parcels["assessed_total_value"] = pd.to_numeric(parcels["total_value"], errors="coerce")
    parcels["wetland_pct"] = pd.to_numeric(parcels.get("wetland_overlap_pct"), errors="coerce")
    parcels["wetland_area_sqft"] = pd.to_numeric(parcels.get("wetland_overlap_acres"), errors="coerce") * SQFT_PER_ACRE
    parcels["flood_pct"] = pd.to_numeric(parcels.get("flood_overlap_pct"), errors="coerce")
    parcels["flood_area_sqft"] = pd.to_numeric(parcels.get("flood_overlap_acres"), errors="coerce") * SQFT_PER_ACRE
    parcels["primary_fema_zone"] = parcels.get("flood_zone_primary", pd.Series(pd.NA, index=parcels.index)).astype("string").str.strip()
    parcels["elevation_mean_ft"] = pd.to_numeric(parcels.get("elevation_mean_ft"), errors="coerce")
    parcels = derive_shape_metrics(parcels)
    parcels["geometry"] = parcels["geometry"].map(point_geometry_from_wkb)
    parcels = parcels.drop(
        columns=[
            "land_use_raw",
            "tax_acres",
            "gis_acres",
            "total_acres",
            "parcel_area_acres",
            "total_value",
            "wetland_overlap_acres",
            "wetland_overlap_pct",
            "flood_zone_primary",
            "shape_length",
            "shape_area",
        ],
        errors="ignore",
    )

    owners = pd.read_parquet(OWNER_LEADS_PATH, columns=OWNER_COLUMNS, engine="pyarrow")
    buildings = pd.read_parquet(BUILDING_METRICS_PATH, columns=BUILDING_COLUMNS, engine="pyarrow")
    signals = pd.read_parquet(LEAD_SIGNALS_PATH, columns=SIGNAL_COLUMNS, engine="pyarrow")

    frame = parcels.merge(owners, on="parcel_row_id", how="left")
    frame = frame.merge(buildings, on="parcel_row_id", how="left")
    frame = frame.merge(signals, on="parcel_row_id", how="left")
    if DELINQUENT_LEADS_PATH.exists():
        delinquent_leads = pd.read_parquet(DELINQUENT_LEADS_PATH, columns=DELINQUENT_LEAD_COLUMNS, engine="pyarrow")
        frame = frame.merge(delinquent_leads, on="parcel_row_id", how="left", suffixes=("", "_delinq"))
    if TAX_DISTRESS_PATH.exists():
        tax_distress = pd.read_parquet(TAX_DISTRESS_PATH, columns=TAX_DISTRESS_COLUMNS, engine="pyarrow")
        frame = frame.merge(tax_distress, on="parcel_row_id", how="left", suffixes=("", "_taxdistress"))
    if AI_BUILDING_PREDICTIONS_PATH.exists():
        ai_columns = [
            "parcel_row_id",
            "ai_building_present_probability",
            "building_present_confidence",
            "ai_building_present_flag",
            "building_presence_reason",
            "imagery_crop_strategy",
            "imagery_best_crop_label",
            "imagery_crop_count",
            "imagery_driveway_signal",
            "imagery_clearing_signal",
            "parcel_boundary_crop_ready_flag",
            "vacancy_confidence_score",
            "vacancy_model_version",
        ]
        ai_predictions = pd.read_parquet(AI_BUILDING_PREDICTIONS_PATH, columns=ai_columns, engine="pyarrow")
        frame = frame.merge(ai_predictions, on="parcel_row_id", how="left")

    for column in ["parcel_vacant_flag", "corporate_owner_flag", "absentee_owner_flag", "out_of_state_owner_flag", "high_confidence_link_flag", "county_hosted_flag", "delinquent_flag", "forfeited_flag"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna(False)

    frame["state_code"] = frame["state_code"].astype("string").fillna("MS")
    frame["county_name"] = frame["county_name"].astype("string")
    frame["parcel_id"] = frame["parcel_id"].astype("string")
    frame["owner_name"] = frame["owner_name"].astype("string")
    frame["owner_type"] = frame["owner_type"].astype("string").fillna("unknown")
    frame["best_source_type"] = frame["best_source_type"].astype("string").fillna("parcel_master")
    frame["best_source_name"] = frame["best_source_name"].astype("string").fillna("Mississippi Parcel Master")
    frame["source_confidence_tier"] = frame["source_confidence_tier"].astype("string").fillna("parcel_master_only")
    frame["county_source_coverage_tier"] = frame["county_source_coverage_tier"].astype("string").fillna("statewide_parcel_base")
    frame["amount_trust_tier"] = frame["amount_trust_tier"].astype("string").fillna("not_reported")
    frame["growth_pressure_bucket"] = frame["growth_pressure_bucket"].astype("string").fillna("unknown")
    frame["recommended_view_bucket"] = frame["recommended_view_bucket"].astype("string").fillna("general_ranked")
    frame["slope_class"] = frame["slope_class"].astype("string")
    frame["county_vacant_flag"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["ai_building_present_flag"] = (
        frame["ai_building_present_flag"].astype("boolean")
        if "ai_building_present_flag" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="boolean")
    )
    if "vacancy_confidence_score" in frame.columns:
        vacancy_confidence_series = pd.to_numeric(frame["vacancy_confidence_score"], errors="coerce")
    else:
        vacancy_confidence_series = pd.Series(np.nan, index=frame.index, dtype="float64")
    frame["vacancy_confidence_score"] = vacancy_confidence_series.fillna(vacancy_confidence(frame))
    frame = apply_tax_freshness_fields(frame)
    frame = apply_county_tax_coverage_fields(frame)

    return frame


def build_geometry_runtime(parcels: pd.DataFrame) -> pa.Table:
    geometry_frame = parcels.loc[:, ["parcel_row_id", "county_name", "geometry"]].copy()
    geometry_frame["geometry"] = geometry_frame["geometry"].map(simplified_polygon_wkb)
    return pa.Table.from_pandas(geometry_frame, preserve_index=False)


def build_summary_payload(frame: pd.DataFrame) -> dict[str, object]:
    county_counts = frame.groupby("county_name", dropna=True).size().sort_values(ascending=False).head(20)
    recommended_counts = frame.groupby("recommended_view_bucket", dropna=True).size().sort_values(ascending=False)
    average_score = pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()
    vacant_count = int(frame["parcel_vacant_flag"].fillna(False).sum())
    return {
        "row_count": int(len(frame)),
        "source": "mississippi parcel runtime dataset",
        "geometry_mode": "viewport_geojson",
        "sections": {
            "statewide": [
                {"section": "statewide", "metric": "lead_count", "value": str(len(frame))},
                {"section": "statewide", "metric": "average_lead_score", "value": f"{average_score:.1f}"},
                {"section": "statewide", "metric": "likely_vacant_count", "value": str(vacant_count)},
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


def build_presets_payload(frame: pd.DataFrame) -> list[dict[str, str]]:
    definitions = {
        "safest_early_investor_use": {
            "description": "High-confidence county-hosted parcels with stronger amount reliability, no wetlands, and vacancy preference.",
            "filter_expression": "county_hosted_flag=true AND high_confidence_link_flag=true AND parcel_vacant_flag=true AND wetland_flag=false AND amount_trust_tier in trusted/use_with_caution",
            "mask": (
                frame["county_hosted_flag"].fillna(False)
                & frame["high_confidence_link_flag"].fillna(False)
                & frame["parcel_vacant_flag"].fillna(False)
                & ~frame["wetland_flag"].fillna(False)
                & frame["amount_trust_tier"].astype("string").isin(["trusted", "use_with_caution"])
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
        "vacant_land_targeting": {
            "description": "Vacant parcels with stronger road access and fewer wetland constraints.",
            "filter_expression": "parcel_vacant_flag=true AND wetland_flag=false AND road_access_tier in direct/near",
            "mask": (
                frame["parcel_vacant_flag"].fillna(False)
                & ~frame["wetland_flag"].fillna(False)
                & frame["road_access_tier"].astype("string").isin(["direct", "near"])
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
        "larger_acreage_land_targeting": {
            "description": "Vacant larger-acreage parcels for land assembly and development exploration.",
            "filter_expression": "parcel_vacant_flag=true AND county_hosted_flag=true AND acreage>=5",
            "mask": (
                frame["parcel_vacant_flag"].fillna(False)
                & frame["county_hosted_flag"].fillna(False)
                & pd.to_numeric(frame["acreage"], errors="coerce").ge(5).fillna(False)
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
        "growth_edge_targeting": {
            "description": "Parcels in moderate-to-high growth areas with usable road access.",
            "filter_expression": "growth_pressure_bucket in moderate/high AND road_access_tier in direct/near/moderate",
            "mask": (
                frame["growth_pressure_bucket"].astype("string").isin(["moderate", "high"])
                & frame["road_access_tier"].astype("string").isin(["direct", "near", "moderate"])
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
    }
    payload: list[dict[str, str]] = []
    for view_name, definition in definitions.items():
        filtered = frame.loc[definition["mask"]]
        average_score = pd.to_numeric(filtered["lead_score_total"], errors="coerce").mean() if len(filtered) else 0
        payload.append(
            {
                "view_name": view_name,
                "description": definition["description"],
                "filter_expression": definition["filter_expression"],
                "row_count": str(len(filtered)),
                "average_lead_score": f"{average_score:.1f}",
            }
        )
    return payload


def build_default_leads_payload(frame: pd.DataFrame) -> dict[str, object]:
    summary_fields = [
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
    top = frame.sort_values("lead_score_total", ascending=False, na_position="last").head(250)
    return {
        "total_count": int(len(frame)),
        "limit": 200,
        "offset": 0,
        "items": [{key: json_scalar(value) for key, value in record.items()} for record in top.loc[:, summary_fields].to_dict(orient="records")],
    }


def build_default_geometry_payload(frame: pd.DataFrame) -> dict[str, object]:
    top = frame.sort_values(["county_name", "lead_score_total"], ascending=[True, False], na_position="last").groupby("county_name", dropna=True).head(3)
    features = []
    for _, row in top.iterrows():
        geometry = row.get("geometry")
        if not isinstance(geometry, dict):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "parcel_row_id": row["parcel_row_id"],
                    "parcel_id": json_scalar(row.get("parcel_id")),
                    "county_name": json_scalar(row.get("county_name")),
                    "lead_score_total": json_scalar(row.get("lead_score_total")),
                    "lead_score_tier": json_scalar(row.get("lead_score_tier")),
                    "parcel_vacant_flag": json_scalar(row.get("parcel_vacant_flag")),
                    "wetland_flag": json_scalar(row.get("wetland_flag")),
                    "flood_risk_score": json_scalar(row.get("flood_risk_score")),
                    "road_access_tier": json_scalar(row.get("road_access_tier")),
                    "county_hosted_flag": json_scalar(row.get("county_hosted_flag")),
                    "best_source_type": json_scalar(row.get("best_source_type")),
                    "selected": False,
                },
            }
        )
    lngs = [feature["geometry"]["coordinates"][0] for feature in features]
    lats = [feature["geometry"]["coordinates"][1] for feature in features]
    return {
        "geometry_mode": "viewport_geojson",
        "render_mode": "points",
        "geometry_bounds": [round(min(lngs), 6), round(min(lats), 6), round(max(lngs), 6), round(max(lats), 6)] if features else None,
        "geometry_view_box": None,
        "requested_bounds": [-91.65, 30.15, -88.0, 35.1],
        "zoom": 6.1,
        "feature_count": len(features),
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "items": [{"parcel_row_id": row["parcel_row_id"], "path": None, "lead_score_total": json_scalar(row["lead_score_total"])} for _, row in top.iterrows()],
    }


def main() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GEOMETRY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = build_runtime_frame()
    source_parcels = pd.read_parquet(PARCEL_MASTER_PATH, columns=["parcel_row_id", "county_name", "geometry"], engine="pyarrow")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(OUTPUT_ROOT),
        format="parquet",
        existing_data_behavior="delete_matching",
        partitioning=["county_name"],
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        max_rows_per_group=10000,
        max_rows_per_file=50000,
    )
    geometry_table = build_geometry_runtime(source_parcels)
    ds.write_dataset(
        geometry_table,
        base_dir=str(GEOMETRY_OUTPUT_ROOT),
        format="parquet",
        existing_data_behavior="delete_matching",
        partitioning=["county_name"],
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        max_rows_per_group=10000,
        max_rows_per_file=50000,
    )
    detail_metrics = build_detail_metrics_runtime(frame)
    detail_metrics.to_parquet(DETAIL_METRICS_OUTPUT_PATH, index=False, engine="pyarrow")
    if COUNTY_COVERAGE_MATRIX_PATH.exists():
        shutil.copy2(COUNTY_COVERAGE_MATRIX_PATH, COUNTY_COVERAGE_MATRIX_RUNTIME_PATH)
    (RUNTIME_ROOT / "summary.json").write_text(json.dumps(build_summary_payload(frame)), encoding="utf-8")
    (RUNTIME_ROOT / "presets.json").write_text(json.dumps(build_presets_payload(frame)), encoding="utf-8")
    (RUNTIME_ROOT / "default_leads.json").write_text(json.dumps(build_default_leads_payload(frame)), encoding="utf-8")
    (RUNTIME_ROOT / "default_geometry.json").write_text(json.dumps(build_default_geometry_payload(frame)), encoding="utf-8")
    print(f"Wrote {len(frame)} runtime rows to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
