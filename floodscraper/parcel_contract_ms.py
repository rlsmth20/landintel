from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CANONICAL_PARCEL_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "state_code",
    "county_name",
    "county_fips",
]

CANONICAL_PARCEL_FIELDS_WITH_GEOMETRY = [
    *CANONICAL_PARCEL_FIELDS,
    "geometry",
]

CANONICAL_REQUIRED_NON_NULL_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "state_code",
    "county_name",
    "county_fips",
]

PROTECTED_CANONICAL_FIELDS = tuple(CANONICAL_PARCEL_FIELDS_WITH_GEOMETRY)

REVIEW_EXPORT_REQUIRED_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "masked_parcel_tile_path",
    "masked_parcel_core_crop_path",
    "masked_parcel_focus_crop_path",
    "review_tile_manifest_path",
]

DETAIL_METRICS_REQUIRED_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
]

BACKEND_DETAIL_REQUIRED_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
]

FRONTEND_FALLBACK_REQUIRED_FIELDS = BACKEND_DETAIL_REQUIRED_FIELDS.copy()

TILE_DEBUG_REQUIRED_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "tile_label",
    "tile_coordinate",
]

TILE_MANIFEST_REQUIRED_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "tile_label",
]

API_LEADS_SUMMARY_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "acreage",
    "owner_name",
    "lead_score_total",
    "lead_score_total_effective",
    "lead_score_tier",
    "parcel_vacant_flag",
    "parcel_improvement_status",
    "building_signal_conflict_flag",
    "road_access_tier",
    "growth_pressure_bucket",
    "best_source_type",
    "source_confidence_tier",
    "delinquent_amount",
    "amount_trust_tier",
    "parcel_tax_status_label",
    "parcel_tax_status_category",
    "parcel_tax_status_confidence",
    "parcel_tax_actionability",
    "parcel_tax_data_warning",
    "parcel_tax_freshness_bucket",
    "parcel_tax_years_stale",
    "parcel_tax_is_actionable_current",
    "parcel_tax_is_historical_only",
    "parcel_tax_freshness_reason",
    "recommended_sort_reason",
    "county_hosted_flag",
    "high_confidence_link_flag",
    "recommended_view_bucket",
]

SEARCH_SOURCE_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "acreage",
    "owner_name",
    "longitude",
    "latitude",
    "lead_score_total",
    "lead_score_total_effective",
]

SEARCH_OUTPUT_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "acreage",
    "owner_name",
    "centroid",
    "match_field",
]

NEARBY_COMP_SOURCE_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "acreage",
    "land_use",
    "assessed_total_value",
    "lead_score_total",
    "lead_score_total_effective",
    "investment_score",
    "parcel_vacant_flag",
    "county_vacant_flag",
    "building_count",
    "building_area_total",
    "ai_building_present_probability",
    "ai_building_present_flag",
    "building_present_confidence",
    "building_presence_reason",
    "longitude",
    "latitude",
]

NEARBY_COMP_OUTPUT_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "acreage",
    "land_use",
    "distance_to_subject_miles",
    "radius_bucket",
    "assessed_total_value",
    "value_per_acre",
    "lead_score_total",
    "investment_score",
    "parcel_vacant_flag",
    "parcel_improvement_status",
    "parcel_improvement_confidence",
    "parcel_improvement_reason",
    "parcel_improvement_evidence_summary",
    "similarity_score",
    "centroid",
]

GEOMETRY_FEATURE_PROPERTY_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "lead_score_total",
    "lead_score_tier",
    "parcel_vacant_flag",
    "wetland_flag",
    "flood_risk_score",
    "road_access_tier",
    "county_hosted_flag",
    "best_source_type",
    "selected",
]

GEOMETRY_ITEM_FIELDS = [
    "parcel_row_id",
    "path",
    "lead_score_total",
]

PARCEL_TILE_SOURCE_FIELDS = [
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

PARCEL_TILE_FEATURE_PROPERTY_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "wetland_flag",
    "flood_risk_score",
    "road_access_tier",
    "county_hosted_flag",
    "best_source_type",
]

FRONTEND_FALLBACK_RUNTIME_COLUMNS = [
    "parcel_row_id",
    "assessed_total_value",
    "geometry_estimated_frontage_feet",
    "geometry_estimated_width_feet",
    "geometry_min_dimension_feet",
    "geometry_max_dimension_feet",
    "geometry_frontage_to_width_ratio",
    "geometry_effective_buildable_flag",
    "geometry_marketability_base_flag",
    "geometry_marketability_flag",
    "geometry_marketability_context",
    "geometry_marketability_action",
    "geometry_penalty_points",
    "geometry_penalty_reason",
    "geometry_marketability_default_leads_excluded_flag",
    "county_vacant_flag",
    "ai_building_present_probability",
    "ai_building_present_flag",
    "building_present_confidence",
    "building_presence_reason",
    "ai_vacancy_available_flag",
    "ai_vacancy_source",
    "ai_vacancy_status_note",
    "parcel_tile_coverage_pct",
    "parcel_bbox_tile_coverage_pct",
    "full_parcel_visible_flag",
    "parcel_extent_exceeds_tile_flag",
    "parcel_tile_low_coverage_flag",
    "multi_tile_candidate_flag",
    "parcel_covering_tile_count",
    "vacancy_model_version",
    "tiles_scored_count",
    "tiles_with_building_signal_count",
    "multi_tile_inference_used_flag",
    "multi_tile_aggregation_reason",
    "best_tile_label",
    "best_tile_confidence",
    "best_tile_crop_label",
    "best_tile_probability",
    "best_tile_parcel_coverage_pct",
    "negative_tile_coverage_pct",
]

REVIEW_TILE_MANIFEST_OUTPUT_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "tile_label",
    "tile_rank",
    "tile_coordinate",
    "centroid_tile_flag",
    "original_geom_type",
    "clipped_geom_type",
    "polygon_part_count",
    "clipped_polygon_part_count",
    "bounds_before_clip",
    "bounds_after_clip",
    "parcel_tile_coverage_pct",
    "parcel_bbox_tile_coverage_pct",
    "raw_tile_path",
    "masked_parcel_tile_path",
    "masked_parcel_core_crop_path",
    "masked_parcel_focus_crop_path",
]

REVIEW_EXPORT_OUTPUT_COLUMNS = [
    "sample_group",
    "sample_reason",
    "review_priority",
    "sample_score",
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "latitude",
    "longitude",
    "image_url",
    "image_path",
    "raw_centroid_tile_path",
    "review_tile_label",
    "review_tile_rank",
    "review_tile_coordinate",
    "review_tile_image_url",
    "review_tile_image_path",
    "original_geom_type",
    "clipped_geom_type",
    "polygon_part_count",
    "clipped_polygon_part_count",
    "bounds_before_clip",
    "bounds_after_clip",
    "masked_parcel_tile_path",
    "masked_parcel_core_crop_path",
    "masked_parcel_focus_crop_path",
    "review_tile_manifest_path",
    "review_tile_sample_labels",
    "acreage",
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
    "geometry_marketability_base_flag",
    "geometry_marketability_flag",
    "geometry_marketability_context",
    "geometry_marketability_action",
    "geometry_marketability_default_leads_excluded_flag",
    "geometry_penalty_points",
    "geometry_penalty_reason",
    "vacancy_manual_review_eligible_flag",
    "vacancy_manual_review_exclusion_reason",
    "building_count",
    "building_area_total",
    "parcel_vacant_flag",
    "ai_building_present_flag",
    "ai_building_present_probability",
    "building_present_confidence",
    "building_presence_reason",
    "imagery_best_crop_label",
    "imagery_crop_strategy",
    "parcel_boundary_crop_ready_flag",
    "parcel_tile_coverage_pct",
    "parcel_bbox_tile_coverage_pct",
    "full_parcel_visible_flag",
    "parcel_extent_exceeds_tile_flag",
    "parcel_tile_low_coverage_flag",
    "multi_tile_candidate_flag",
    "parcel_covering_tile_count",
    "tile_coordinates",
    "unique_tile_count",
    "duplicate_tile_flag",
    "tiles_scored_count",
    "tiles_with_building_signal_count",
    "multi_tile_inference_used_flag",
    "multi_tile_aggregation_reason",
    "best_tile_label",
    "best_tile_confidence",
    "best_tile_crop_label",
    "best_tile_probability",
    "best_tile_parcel_coverage_pct",
    "negative_tile_coverage_pct",
    "imagery_driveway_signal",
    "imagery_clearing_signal",
    "nearby_building_density",
    "shape_compactness",
    "parcel_width_ft_estimate",
    "parcel_aspect_ratio_estimate",
    "lead_score_total",
    "review_hint",
    "manual_training_label",
    "manual_structure_inside_parcel",
    "manual_neighbor_structure_only",
    "manual_road_or_clearing_only",
    "manual_review_confidence",
    "manual_notes",
]

DEFAULT_CONTRACT_REPORT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "buildings_processed" / "ms_parcel_contract_validation_report.json"
)

FEATURE_NAMESPACE_PREFIXES = (
    "ai_",
    "amount_",
    "apn",
    "assessed_",
    "best_",
    "building_",
    "buildability_",
    "canonical_",
    "caution_",
    "clipped_",
    "corporate_",
    "county_",
    "delinquent_",
    "duplicate_",
    "electric_",
    "environment",
    "feature_",
    "flood_",
    "forfeited_",
    "full_",
    "geometry",
    "growth_",
    "high_",
    "image_",
    "imagery_",
    "investment_",
    "land_",
    "latest_",
    "lead_",
    "mailer_",
    "manual_",
    "masked_",
    "mean_",
    "model_",
    "multi_",
    "negative_",
    "nearby_",
    "original_",
    "out_of_state_",
    "owner_",
    "parcel_",
    "polygon_",
    "primary_",
    "raw_",
    "recommended_",
    "review_",
    "road_",
    "sample_",
    "shape_",
    "size_",
    "slope_",
    "source_",
    "state_",
    "summary",
    "tax_",
    "tile_",
    "tiles_",
    "top_",
    "total_",
    "utilities_",
    "vacancy_",
    "vacant_",
    "value_",
    "wetland_",
)

ALLOWED_UNPREFIXED_COLUMNS = set(CANONICAL_PARCEL_FIELDS) | {
    "acreage",
    "acreage_bucket",
    "area_acres",
    "aspect_ratio",
    "bounds_after_clip",
    "bounds_before_clip",
    "bounding_box_height_meters",
    "bounding_box_width_meters",
    "centroid",
    "compactness",
    "elevation_mean_ft",
    "is_multipart",
    "latitude",
    "longitude",
    "part_count",
    "perimeter_meters",
}


def canonical_identity_payload(row: Mapping[str, Any], *, include_geometry: bool = False) -> dict[str, Any]:
    fields = CANONICAL_PARCEL_FIELDS_WITH_GEOMETRY if include_geometry else CANONICAL_PARCEL_FIELDS
    return {field: row.get(field) for field in fields if field in row}


def serialize_contract_row(
    row: Mapping[str, Any],
    fields: Sequence[str],
    *,
    serializer: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields:
        value = row.get(field)
        payload[field] = serializer(value) if serializer is not None else value
    return payload


def validate_output_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_fields: Sequence[str],
    required_fields: Sequence[str],
    non_null_fields: Sequence[str],
    context: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    expected_columns = list(expected_fields)
    if frame.empty:
        frame = pd.DataFrame(columns=expected_columns)
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in expected_columns]
    if missing_columns or extra_columns:
        raise ValueError(
            f"{context}: output schema drift detected "
            f"missing={missing_columns or 'none'} extra={extra_columns or 'none'}"
        )
    validate_required_columns(
        frame,
        required_columns=required_fields,
        non_null_columns=non_null_fields,
        context=context,
    )
    return frame


def contract_left_merge(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    *,
    on: str | Sequence[str] = "parcel_row_id",
    allow_row_count_change: bool = False,
) -> pd.DataFrame:
    if isinstance(on, str):
        join_columns = [on]
    else:
        join_columns = list(on)
    join_keys = set(join_columns)
    missing_left_keys = [column for column in join_columns if column not in frame.columns]
    missing_right_keys = [column for column in join_columns if column not in source.columns]
    if missing_left_keys or missing_right_keys:
        raise ValueError(
            "contract_left_merge: missing join keys "
            f"left={missing_left_keys or 'none'} right={missing_right_keys or 'none'}"
        )

    duplicate_right_keys = source.duplicated(subset=join_columns, keep=False)
    if bool(duplicate_right_keys.any()):
        duplicate_examples = source.loc[duplicate_right_keys, join_columns].head(5).to_dict(orient="records")
        raise ValueError(
            "contract_left_merge: duplicate right-side join keys detected "
            f"for {join_columns}: {duplicate_examples}"
        )

    left_row_count = int(len(frame))
    protected_left_snapshots = {
        field: frame[field].copy()
        for field in PROTECTED_CANONICAL_FIELDS
        if field in frame.columns and field not in join_keys
    }
    protected_overlaps = [field for field in protected_left_snapshots if field in source.columns]
    protected_null_violations = {
        field: int(frame[field].isna().sum())
        for field in protected_overlaps
        if field in CANONICAL_REQUIRED_NON_NULL_FIELDS and int(frame[field].isna().sum()) > 0
    }
    if protected_null_violations:
        raise ValueError(
            "contract_left_merge: protected canonical fields contain nulls before merge "
            f"and cannot be repaired from the right side: {protected_null_violations}"
        )

    removable_overlaps = [
        column
        for column in source.columns
        if column not in join_keys and column in frame.columns and column not in PROTECTED_CANONICAL_FIELDS
    ]
    protected_source_columns = [
        column for column in source.columns if column not in join_keys and column in PROTECTED_CANONICAL_FIELDS
    ]
    if removable_overlaps:
        frame = frame.drop(columns=removable_overlaps)
    if protected_source_columns:
        source = source.drop(columns=protected_source_columns)

    merged = frame.merge(source, on=on, how="left")
    if not allow_row_count_change and int(len(merged)) != left_row_count:
        raise ValueError(
            "contract_left_merge: row count changed unexpectedly "
            f"from {left_row_count} to {len(merged)} for join keys {join_columns}"
        )

    for field, left_snapshot in protected_left_snapshots.items():
        if field not in merged.columns or not merged[field].equals(left_snapshot):
            raise ValueError(
                "contract_left_merge: protected canonical field changed during merge "
                f"for field {field}"
            )

    collisions = suffix_collision_columns(merged.columns)
    if collisions:
        raise ValueError(f"contract_left_merge: suffix collisions detected {collisions}")
    return merged


def suffix_collision_columns(columns: Iterable[str]) -> list[str]:
    column_set = set(columns)
    collisions: list[str] = []
    for column in columns:
        if not (column.endswith("_x") or column.endswith("_y")):
            continue
        root = column[:-2]
        if root in column_set or f"{root}_x" in column_set or f"{root}_y" in column_set:
            collisions.append(column)
    return sorted(collisions)


def unexpected_unprefixed_columns(columns: Iterable[str]) -> list[str]:
    unexpected: list[str] = []
    for column in columns:
        if column in ALLOWED_UNPREFIXED_COLUMNS:
            continue
        if any(column.startswith(prefix) for prefix in FEATURE_NAMESPACE_PREFIXES):
            continue
        unexpected.append(column)
    return sorted(unexpected)


def validate_required_columns(
    frame: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    non_null_columns: Sequence[str],
    context: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{context}: missing required columns {missing_columns}")
    null_violations = {
        column: int(frame[column].isna().sum())
        for column in non_null_columns
        if column in frame.columns and int(frame[column].isna().sum()) > 0
    }
    if null_violations:
        raise ValueError(f"{context}: canonical null violations {null_violations}")
    collisions = suffix_collision_columns(frame.columns)
    if collisions:
        raise ValueError(f"{context}: suffix collisions detected {collisions}")


def frame_contract_report(
    frame: pd.DataFrame,
    *,
    label: str,
    required_columns: Sequence[str],
    non_null_columns: Sequence[str],
) -> dict[str, Any]:
    columns = list(frame.columns)
    null_counts = {
        column: int(frame[column].isna().sum())
        for column in non_null_columns
        if column in frame.columns
    }
    return {
        "label": label,
        "row_count": int(len(frame)),
        "required_columns": list(required_columns),
        "columns_added": sorted(column for column in columns if column not in required_columns),
        "columns_removed": sorted(column for column in required_columns if column not in columns),
        "canonical_null_counts": null_counts,
        "suffix_collisions": suffix_collision_columns(columns),
        "unexpected_unprefixed_columns": unexpected_unprefixed_columns(columns),
    }


def records_contract_report(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    required_fields: Sequence[str],
    non_null_fields: Sequence[str],
) -> dict[str, Any]:
    columns = sorted({key for record in records for key in record.keys()}) if records else []
    null_counts = {
        field: int(sum(1 for record in records if record.get(field) is None))
        for field in non_null_fields
        if field in columns
    }
    return {
        "label": label,
        "row_count": int(len(records)),
        "required_columns": list(required_fields),
        "columns_added": sorted(column for column in columns if column not in required_fields),
        "columns_removed": sorted(column for column in required_fields if column not in columns),
        "canonical_null_counts": null_counts,
        "suffix_collisions": suffix_collision_columns(columns),
        "unexpected_unprefixed_columns": unexpected_unprefixed_columns(columns),
    }


def write_contract_report(path: Path, reports: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(reports), indent=2), encoding="utf-8")
