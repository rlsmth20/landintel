from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from parcel_geometry_quality_ms import (
    GEOMETRY_QUALITY_ARTIFACT_PATH,
    filter_review_geometry_frame,
    geometry_quality_diagnostics,
    load_geometry_quality_frame,
)
from parcel_marketability_ms import add_geometry_marketability_fields
from parcel_contract_ms import (
    REVIEW_EXPORT_REQUIRED_COLUMNS,
    REVIEW_EXPORT_OUTPUT_COLUMNS,
    REVIEW_TILE_MANIFEST_OUTPUT_FIELDS,
    TILE_MANIFEST_REQUIRED_FIELDS,
    contract_left_merge,
    validate_output_records,
    validate_required_columns,
)
from vacancy_ai_common import (
    APP_READY_PATH,
    DEFAULT_TILE_URL_TEMPLATE,
    PARCEL_MASTER_PATH,
    PREDICTIONS_PATH,
    build_parcel_inference_tile_plan,
    centroid_tile,
    clip_shape_to_tile,
    ensure_tile_image_for_address,
    load_candidate_frame,
    load_geometry_shape,
    load_parcel_geometry_lookup,
    polygon_parts_from_shape,
    prepare_parcel_aware_image,
    prepare_parcel_aware_image_for_tile_set,
    tile_cache_path_for_address,
    tile_url,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "data" / "buildings_processed" / "ms_vacancy_training_review_sample_300.csv"
DEFAULT_SCENE_FALSE_POSITIVE_COUNT = 120
DEFAULT_NEIGHBOR_FALSE_POSITIVE_COUNT = 90
DEFAULT_IMPROVED_CONTROL_COUNT = 45
DEFAULT_VACANT_CONTROL_COUNT = 45
DEFAULT_COUNTY_CAP = 10
DEFAULT_QA_BAD_GEOMETRY_PCT = 0.0
REVIEW_ASSET_DIR = ROOT / "data" / "buildings_processed" / "ai_review_tiles_ms"
MAX_REVIEW_TILE_EXPORT_COUNT = 4
DEFAULT_REVIEW_SELECTION_PADDING_PER_BUCKET = 5
REVIEW_MIN_SINGLE_TILE_PARCEL_COVERAGE_PCT = 15.0
REVIEW_MIN_SINGLE_TILE_BBOX_COVERAGE_PCT = 20.0

MANUAL_LABEL_COLUMNS = [
    "manual_training_label",
    "manual_structure_inside_parcel",
    "manual_neighbor_structure_only",
    "manual_road_or_clearing_only",
    "manual_review_confidence",
    "manual_notes",
]


def _frame_series(frame: pd.DataFrame, column: str, default: object = np.nan) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _blank_string_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series("", index=frame.index, dtype="string")


def _checkpoint_file(checkpoint_dir: Path | None, name: str, suffix: str = ".parquet") -> Path | None:
    if checkpoint_dir is None:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{name}{suffix}"


def _checkpoint_meta_path(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    return checkpoint_path.with_suffix(".meta.json")


def _write_checkpoint_metadata(checkpoint_path: Path | None, payload: dict[str, object]) -> None:
    meta_path = _checkpoint_meta_path(checkpoint_path)
    if meta_path is None:
        return
    meta_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_checkpoint_metadata(checkpoint_path: Path | None) -> dict[str, object]:
    meta_path = _checkpoint_meta_path(checkpoint_path)
    if meta_path is None or not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _merge_optional_frame(frame: pd.DataFrame, source_path: Path, columns: list[str]) -> pd.DataFrame:
    if not source_path.exists():
        return frame
    available_columns = set(pq.read_schema(source_path).names)
    selected_columns = [column for column in columns if column in available_columns]
    if not selected_columns:
        return frame
    source = pd.read_parquet(source_path, columns=selected_columns, engine="pyarrow")
    return contract_left_merge(frame, source, on="parcel_row_id")


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


def prepare_sampling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["parcel_row_id"] = prepared["parcel_row_id"].astype("string")
    prepared["parcel_id"] = _frame_series(prepared, "parcel_id", pd.NA).astype("string")
    prepared["county_name"] = _frame_series(prepared, "county_name", pd.NA).astype("string").str.strip().str.lower()
    prepared["latitude"] = pd.to_numeric(_frame_series(prepared, "latitude"), errors="coerce")
    prepared["longitude"] = pd.to_numeric(_frame_series(prepared, "longitude"), errors="coerce")
    prepared["acreage"] = pd.to_numeric(_frame_series(prepared, "acreage"), errors="coerce")
    prepared["building_count"] = pd.to_numeric(_frame_series(prepared, "building_count", 0), errors="coerce").fillna(0)
    prepared["building_area_total"] = pd.to_numeric(_frame_series(prepared, "building_area_total", 0), errors="coerce").fillna(0)
    prepared["parcel_vacant_flag"] = _frame_series(prepared, "parcel_vacant_flag", False).fillna(False).astype(bool)
    prepared["ai_building_present_flag"] = _frame_series(prepared, "ai_building_present_flag", pd.NA).astype("boolean")
    prepared["ai_building_present_probability"] = pd.to_numeric(_frame_series(prepared, "ai_building_present_probability"), errors="coerce")
    prepared["building_present_confidence"] = pd.to_numeric(_frame_series(prepared, "building_present_confidence"), errors="coerce")
    prepared["building_presence_reason"] = _frame_series(prepared, "building_presence_reason", pd.NA).astype("string")
    prepared["imagery_best_crop_label"] = _frame_series(prepared, "imagery_best_crop_label", pd.NA).astype("string")
    prepared["imagery_crop_strategy"] = _frame_series(prepared, "imagery_crop_strategy", pd.NA).astype("string")
    prepared["parcel_boundary_crop_ready_flag"] = (
        _frame_series(prepared, "parcel_boundary_crop_ready_flag", False).fillna(False).astype(bool)
    )
    prepared["tiles_scored_count"] = pd.to_numeric(_frame_series(prepared, "tiles_scored_count", 0), errors="coerce").fillna(0)
    prepared["tiles_with_building_signal_count"] = (
        pd.to_numeric(_frame_series(prepared, "tiles_with_building_signal_count", 0), errors="coerce").fillna(0)
    )
    prepared["multi_tile_inference_used_flag"] = (
        _frame_series(prepared, "multi_tile_inference_used_flag", False).fillna(False).astype(bool)
    )
    prepared["multi_tile_aggregation_reason"] = _frame_series(prepared, "multi_tile_aggregation_reason", pd.NA).astype("string")
    prepared["best_tile_label"] = _frame_series(prepared, "best_tile_label", pd.NA).astype("string")
    prepared["best_tile_confidence"] = pd.to_numeric(_frame_series(prepared, "best_tile_confidence"), errors="coerce")
    prepared["best_tile_crop_label"] = _frame_series(prepared, "best_tile_crop_label", pd.NA).astype("string")
    prepared["best_tile_probability"] = pd.to_numeric(_frame_series(prepared, "best_tile_probability"), errors="coerce")
    prepared["best_tile_parcel_coverage_pct"] = pd.to_numeric(
        _frame_series(prepared, "best_tile_parcel_coverage_pct"),
        errors="coerce",
    )
    prepared["negative_tile_coverage_pct"] = pd.to_numeric(
        _frame_series(prepared, "negative_tile_coverage_pct"),
        errors="coerce",
    )
    prepared["tile_coordinates"] = _frame_series(prepared, "tile_coordinates", "[]").astype("string")
    prepared["unique_tile_count"] = pd.to_numeric(_frame_series(prepared, "unique_tile_count", 0), errors="coerce").fillna(0)
    prepared["duplicate_tile_flag"] = _frame_series(prepared, "duplicate_tile_flag", False).fillna(False).astype(bool)
    prepared["imagery_driveway_signal"] = pd.to_numeric(_frame_series(prepared, "imagery_driveway_signal"), errors="coerce")
    prepared["imagery_clearing_signal"] = pd.to_numeric(_frame_series(prepared, "imagery_clearing_signal"), errors="coerce")
    prepared["nearby_building_density"] = pd.to_numeric(_frame_series(prepared, "nearby_building_density"), errors="coerce")
    prepared["lead_score_total"] = pd.to_numeric(_frame_series(prepared, "lead_score_total"), errors="coerce")

    shape_area = pd.to_numeric(_frame_series(prepared, "shape_area"), errors="coerce")
    shape_length = pd.to_numeric(_frame_series(prepared, "shape_length"), errors="coerce")
    compactness, frontage, width = _rectangle_estimates(shape_area, shape_length)

    prepared["shape_compactness"] = pd.to_numeric(_frame_series(prepared, "shape_compactness"), errors="coerce").fillna(compactness)
    prepared["parcel_width_ft_estimate"] = pd.to_numeric(_frame_series(prepared, "parcel_width_ft_estimate"), errors="coerce").fillna(width)
    prepared["parcel_frontage_ft_estimate"] = pd.to_numeric(_frame_series(prepared, "parcel_frontage_ft_estimate"), errors="coerce").fillna(frontage)
    prepared["parcel_aspect_ratio_estimate"] = (
        pd.to_numeric(_frame_series(prepared, "parcel_aspect_ratio_estimate"), errors="coerce")
        .fillna(prepared["parcel_frontage_ft_estimate"] / prepared["parcel_width_ft_estimate"])
    )
    prepared["area_acres"] = pd.to_numeric(_frame_series(prepared, "area_acres"), errors="coerce")
    prepared["perimeter_meters"] = pd.to_numeric(_frame_series(prepared, "perimeter_meters"), errors="coerce")
    prepared["bounding_box_width_meters"] = pd.to_numeric(_frame_series(prepared, "bounding_box_width_meters"), errors="coerce")
    prepared["bounding_box_height_meters"] = pd.to_numeric(_frame_series(prepared, "bounding_box_height_meters"), errors="coerce")
    prepared["aspect_ratio"] = pd.to_numeric(_frame_series(prepared, "aspect_ratio"), errors="coerce")
    prepared["compactness"] = pd.to_numeric(_frame_series(prepared, "compactness"), errors="coerce")
    prepared["is_multipart"] = _frame_series(prepared, "is_multipart", False).fillna(False).astype(bool)
    prepared["part_count"] = pd.to_numeric(_frame_series(prepared, "part_count", 0), errors="coerce").fillna(0).astype(int)
    prepared["geometry_quality_flag"] = _frame_series(prepared, "geometry_quality_flag", "good").astype("string").fillna("good")
    prepared["geometry_review_excluded_flag"] = _frame_series(prepared, "geometry_review_excluded_flag", False).fillna(False).astype(bool)
    prepared["geometry_training_excluded_flag"] = _frame_series(prepared, "geometry_training_excluded_flag", False).fillna(False).astype(bool)
    prepared["geometry_default_leads_excluded_flag"] = _frame_series(prepared, "geometry_default_leads_excluded_flag", False).fillna(False).astype(bool)
    if (
        "geometry_marketability_flag" not in prepared.columns
        or "geometry_marketability_action" not in prepared.columns
        or "geometry_marketability_default_leads_excluded_flag" not in prepared.columns
    ):
        prepared = add_geometry_marketability_fields(prepared)
    prepared["geometry_marketability_base_flag"] = _frame_series(prepared, "geometry_marketability_base_flag", "marketable").astype("string").fillna("marketable")
    prepared["geometry_marketability_flag"] = _frame_series(prepared, "geometry_marketability_flag", "marketable").astype("string").fillna("marketable")
    prepared["geometry_marketability_context"] = _frame_series(prepared, "geometry_marketability_context", "rural").astype("string").fillna("rural")
    prepared["geometry_marketability_action"] = _frame_series(prepared, "geometry_marketability_action", "keep").astype("string").fillna("keep")
    prepared["geometry_marketability_default_leads_excluded_flag"] = (
        _frame_series(prepared, "geometry_marketability_default_leads_excluded_flag", False).fillna(False).astype(bool)
    )
    prepared["geometry_penalty_points"] = pd.to_numeric(_frame_series(prepared, "geometry_penalty_points"), errors="coerce")
    prepared["geometry_penalty_reason"] = _frame_series(prepared, "geometry_penalty_reason", pd.NA).astype("string")
    prepared["geometry_estimated_frontage_feet"] = pd.to_numeric(_frame_series(prepared, "geometry_estimated_frontage_feet"), errors="coerce")
    prepared["geometry_estimated_width_feet"] = pd.to_numeric(_frame_series(prepared, "geometry_estimated_width_feet"), errors="coerce")
    prepared["geometry_min_dimension_feet"] = pd.to_numeric(_frame_series(prepared, "geometry_min_dimension_feet"), errors="coerce")
    prepared["geometry_max_dimension_feet"] = pd.to_numeric(_frame_series(prepared, "geometry_max_dimension_feet"), errors="coerce")
    prepared["geometry_frontage_to_width_ratio"] = pd.to_numeric(_frame_series(prepared, "geometry_frontage_to_width_ratio"), errors="coerce")
    prepared["geometry_effective_buildable_flag"] = _frame_series(prepared, "geometry_effective_buildable_flag", True).fillna(False).astype(bool)
    prepared["ai_score_base"] = prepared["building_present_confidence"].fillna(prepared["ai_building_present_probability"] * 100.0)
    prepared["density_percentile"] = prepared["nearby_building_density"].rank(pct=True, method="average").fillna(0.0) * 100.0
    prepared = prepared.loc[prepared["latitude"].notna() & prepared["longitude"].notna()].copy()
    return prepared


def add_vacancy_manual_review_eligibility_fields(
    frame: pd.DataFrame,
    *,
    require_exported_imagery: bool = False,
) -> pd.DataFrame:
    enriched = frame.copy()
    eligible = pd.Series(True, index=enriched.index, dtype="bool")
    exclusion_reason = pd.Series("", index=enriched.index, dtype="string")

    geometry_quality_flag = _frame_series(enriched, "geometry_quality_flag", "good").astype("string").fillna("good").str.lower()
    geometry_review_excluded_flag = _frame_series(enriched, "geometry_review_excluded_flag", False).fillna(False).astype(bool)
    marketability_action = _frame_series(enriched, "geometry_marketability_action", "keep").astype("string").fillna("keep").str.lower()
    marketability_flag = _frame_series(enriched, "geometry_marketability_flag", "marketable").astype("string").fillna("marketable").str.lower()
    parcel_boundary_crop_ready_flag = _frame_series(enriched, "parcel_boundary_crop_ready_flag", False).fillna(False).astype(bool)
    parcel_tile_low_coverage_flag = _frame_series(enriched, "parcel_tile_low_coverage_flag", False).fillna(False).astype(bool)
    multi_tile_candidate_flag = _frame_series(enriched, "multi_tile_candidate_flag", False).fillna(False).astype(bool)
    parcel_tile_coverage_pct = pd.to_numeric(_frame_series(enriched, "parcel_tile_coverage_pct"), errors="coerce")
    parcel_bbox_tile_coverage_pct = pd.to_numeric(_frame_series(enriched, "parcel_bbox_tile_coverage_pct"), errors="coerce")
    best_tile_parcel_coverage_pct = pd.to_numeric(_frame_series(enriched, "best_tile_parcel_coverage_pct"), errors="coerce")
    review_boundary_crop_ready_flag = _frame_series(enriched, "review_parcel_boundary_crop_ready_flag", False).fillna(False).astype(bool)
    review_crop_strategy = _frame_series(enriched, "review_imagery_crop_strategy", pd.NA).astype("string")
    polygon_part_count = pd.to_numeric(_frame_series(enriched, "polygon_part_count", 0), errors="coerce").fillna(0).astype(int)
    clipped_polygon_part_count = pd.to_numeric(_frame_series(enriched, "clipped_polygon_part_count", 0), errors="coerce").fillna(0).astype(int)

    def exclude(mask: pd.Series, reason: str) -> None:
        nonlocal eligible, exclusion_reason
        effective_mask = mask.fillna(False) & eligible
        eligible = eligible.mask(effective_mask, False)
        exclusion_reason = exclusion_reason.mask(effective_mask, reason)

    exclude(geometry_review_excluded_flag | geometry_quality_flag.eq("access_strip"), "geometry_quality_excluded")
    exclude(marketability_action.eq("exclude") | marketability_flag.eq("unbuildable_candidate"), "geometry_marketability_excluded")
    exclude(~parcel_boundary_crop_ready_flag, "missing_parcel_boundary_crop")
    exclude(
        parcel_tile_low_coverage_flag
        & ~multi_tile_candidate_flag
        & parcel_tile_coverage_pct.lt(REVIEW_MIN_SINGLE_TILE_PARCEL_COVERAGE_PCT).fillna(False)
        & parcel_bbox_tile_coverage_pct.lt(REVIEW_MIN_SINGLE_TILE_BBOX_COVERAGE_PCT).fillna(False)
        & best_tile_parcel_coverage_pct.lt(REVIEW_MIN_SINGLE_TILE_PARCEL_COVERAGE_PCT).fillna(False),
        "insufficient_parcel_visibility",
    )

    if require_exported_imagery:
        missing_assets = (
            _frame_series(enriched, "masked_parcel_tile_path", "").astype("string").str.len().eq(0)
            | _frame_series(enriched, "masked_parcel_core_crop_path", "").astype("string").str.len().eq(0)
            | _frame_series(enriched, "masked_parcel_focus_crop_path", "").astype("string").str.len().eq(0)
            | _frame_series(enriched, "review_tile_manifest_path", "").astype("string").str.len().eq(0)
        )
        exclude(missing_assets, "missing_review_imagery_assets")
        exclude(~review_boundary_crop_ready_flag | review_crop_strategy.eq("skipped_image_export"), "unusable_review_imagery")
        exclude(
            polygon_part_count.gt(0) & clipped_polygon_part_count.lt(polygon_part_count),
            "incomplete_multipart_review_imagery",
        )

    enriched["vacancy_manual_review_eligible_flag"] = eligible
    enriched["vacancy_manual_review_exclusion_reason"] = exclusion_reason.mask(exclusion_reason.str.len().eq(0), pd.NA)
    return enriched


def filter_vacancy_manual_review_eligible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "vacancy_manual_review_eligible_flag" not in frame.columns:
        frame = add_vacancy_manual_review_eligibility_fields(frame)
    return frame.loc[_frame_series(frame, "vacancy_manual_review_eligible_flag", False).fillna(False).astype(bool)].copy()


def vacancy_manual_review_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    diagnosed = frame if "vacancy_manual_review_eligible_flag" in frame.columns else add_vacancy_manual_review_eligibility_fields(frame)
    eligible_flag = _frame_series(diagnosed, "vacancy_manual_review_eligible_flag", False).fillna(False).astype(bool)
    exclusion_reason = _frame_series(diagnosed, "vacancy_manual_review_exclusion_reason", "eligible").astype("string").fillna("eligible")
    reason_counts = exclusion_reason.value_counts(dropna=False).sort_index().to_dict()
    excluded = diagnosed.loc[~eligible_flag].copy()
    included = diagnosed.loc[eligible_flag].copy()

    def _example_rows(source: pd.DataFrame, *, limit: int = 5) -> list[dict[str, object]]:
        example_columns = [
            column
            for column in [
                "parcel_row_id",
                "parcel_id",
                "county_name",
                "sample_reason",
                "geometry_quality_flag",
                "geometry_marketability_flag",
                "geometry_marketability_action",
                "parcel_tile_coverage_pct",
                "parcel_bbox_tile_coverage_pct",
                "best_tile_parcel_coverage_pct",
                "vacancy_manual_review_exclusion_reason",
            ]
            if column in source.columns
        ]
        if source.empty or not example_columns:
            return []
        records = source.loc[:, example_columns].head(limit).to_dict(orient="records")
        normalized: list[dict[str, object]] = []
        for record in records:
            normalized.append(
                {
                    str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                    for key, value in record.items()
                }
            )
        return normalized

    return {
        "row_count": int(len(diagnosed)),
        "eligible_count": int(eligible_flag.sum()),
        "eligible_pct": round(float(eligible_flag.mean() * 100.0), 2) if len(diagnosed) else 0.0,
        "excluded_count": int((~eligible_flag).sum()),
        "excluded_pct": round(float((~eligible_flag).mean() * 100.0), 2) if len(diagnosed) else 0.0,
        "exclusion_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
        "top_exclusion_reasons": [
            {"reason": str(key), "count": int(value)}
            for key, value in exclusion_reason.loc[~eligible_flag].value_counts(dropna=False).head(10).to_dict().items()
        ],
        "excluded_examples": _example_rows(excluded),
        "included_examples": _example_rows(included),
    }


def load_sampling_frame(
    *,
    reuse_geometry_quality_artifact: bool = True,
    build_geometry_quality_artifact_if_missing: bool = False,
    geometry_quality_artifact_path: Path = GEOMETRY_QUALITY_ARTIFACT_PATH,
) -> pd.DataFrame:
    frame = load_candidate_frame()
    frame = _merge_optional_frame(
        frame,
        PREDICTIONS_PATH,
        [
            "parcel_row_id",
            "ai_building_present_flag",
            "ai_building_present_probability",
            "building_present_confidence",
            "building_presence_reason",
            "imagery_crop_strategy",
            "imagery_best_crop_label",
            "imagery_driveway_signal",
            "imagery_clearing_signal",
            "parcel_boundary_crop_ready_flag",
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
            "tile_coordinates",
            "unique_tile_count",
            "duplicate_tile_flag",
        ],
    )
    frame = _merge_optional_frame(
        frame,
        APP_READY_PATH,
        ["parcel_row_id", "lead_score_total", "nearby_building_density"],
    )
    frame = _merge_optional_frame(
        frame,
        PARCEL_MASTER_PATH,
        ["parcel_row_id", "parcel_id", "shape_area", "shape_length"],
    )
    geometry_quality = load_geometry_quality_frame(
        frame["parcel_row_id"].astype("string"),
        reuse_artifact=reuse_geometry_quality_artifact,
        artifact_path=geometry_quality_artifact_path,
        build_artifact_if_missing=build_geometry_quality_artifact_if_missing,
    )
    if not geometry_quality.empty:
        frame = contract_left_merge(frame, geometry_quality, on="parcel_row_id")
    return prepare_sampling_frame(frame)


def _exclude_ids(frame: pd.DataFrame, exclude_ids: set[str]) -> pd.DataFrame:
    if not exclude_ids:
        return frame.copy()
    exclude_index = pd.Index(sorted(exclude_ids), dtype="string")
    return frame.loc[~frame["parcel_row_id"].astype("string").isin(exclude_index)].copy()


def _ranked_diverse_sample(
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
    exclude_ids: set[str],
    county_cap: int,
) -> pd.DataFrame:
    if count <= 0:
        return frame.head(0).copy()
    available = _exclude_ids(frame, exclude_ids).copy()
    if available.empty:
        raise ValueError(f"Only found 0 candidate rows; need {count}.")
    rng = np.random.default_rng(seed)
    available["sample_tiebreaker"] = rng.random(len(available))
    available = available.sort_values(["sample_score", "sample_tiebreaker", "parcel_row_id"], ascending=[False, True, True]).reset_index(drop=True)

    selected_positions: list[int] = []
    county_counts: dict[str, int] = {}
    effective_cap = max(1, min(int(county_cap), count))
    for position, row in available.iterrows():
        county = str(row.get("county_name") or "unknown")
        if county_counts.get(county, 0) >= effective_cap:
            continue
        selected_positions.append(position)
        county_counts[county] = county_counts.get(county, 0) + 1
        if len(selected_positions) >= count:
            break

    if len(selected_positions) < count:
        already_selected = set(selected_positions)
        for position in range(len(available)):
            if position in already_selected:
                continue
            selected_positions.append(position)
            if len(selected_positions) >= count:
                break

    if len(selected_positions) < count:
        raise ValueError(f"Only found {len(selected_positions)} candidate rows; need {count}.")

    sampled = available.iloc[selected_positions].drop(columns=["sample_tiebreaker"]).copy()
    return sampled.reset_index(drop=True)


def _no_structure_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["building_count"].le(0) & frame["building_area_total"].le(0)


def scene_false_positive_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    no_structure = _no_structure_mask(frame)
    candidate = frame.loc[
        no_structure
        & frame["parcel_boundary_crop_ready_flag"]
        & frame["ai_score_base"].between(50.0, 89.9, inclusive="both")
        & (
            frame["imagery_driveway_signal"].ge(55.0).fillna(False)
            | frame["imagery_clearing_signal"].ge(22.0).fillna(False)
        )
    ].copy()
    candidate["sample_score"] = (
        candidate["ai_score_base"].fillna(0.0)
        + (candidate["imagery_driveway_signal"].fillna(0.0) - 45.0).clip(lower=0.0) * 0.35
        + (candidate["imagery_clearing_signal"].fillna(0.0) - 15.0).clip(lower=0.0) * 0.75
        + np.where(candidate["ai_building_present_flag"].fillna(False), 6.0, 0.0)
    )
    candidate["sample_group"] = "failure_mode"
    candidate["sample_reason"] = "road_or_clearing_context_false_positive"
    candidate["review_priority"] = "high"
    return candidate


def neighbor_false_positive_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    no_structure = _no_structure_mask(frame)
    density_threshold = float(frame["nearby_building_density"].quantile(0.75))
    narrowness_score = ((60.0 - frame["parcel_width_ft_estimate"]) / 60.0).clip(lower=0.0, upper=1.0) * 100.0
    aspect_score = ((frame["parcel_aspect_ratio_estimate"] - 3.0) / 7.0).clip(lower=0.0, upper=1.0) * 100.0
    compactness_score = ((0.35 - frame["shape_compactness"]) / 0.35).clip(lower=0.0, upper=1.0) * 100.0
    candidate = frame.loc[
        no_structure
        & frame["parcel_boundary_crop_ready_flag"]
        & frame["ai_score_base"].between(45.0, 89.9, inclusive="both")
        & (
            frame["nearby_building_density"].ge(density_threshold).fillna(False)
            | frame["shape_compactness"].le(0.20).fillna(False)
            | frame["parcel_aspect_ratio_estimate"].ge(6.0).fillna(False)
            | frame["parcel_width_ft_estimate"].le(45.0).fillna(False)
        )
    ].copy()
    candidate["sample_score"] = (
        candidate["ai_score_base"].fillna(0.0)
        + candidate["density_percentile"].fillna(0.0) * 0.25
        + aspect_score.loc[candidate.index].fillna(0.0) * 0.35
        + narrowness_score.loc[candidate.index].fillna(0.0) * 0.30
        + compactness_score.loc[candidate.index].fillna(0.0) * 0.20
        + np.where(candidate["ai_building_present_flag"].fillna(False), 4.0, 0.0)
    )
    candidate["sample_group"] = "failure_mode"
    candidate["sample_reason"] = "neighbor_outside_parcel_false_positive"
    candidate["review_priority"] = "high"
    return candidate


def improved_control_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    candidate = frame.loc[
        (
            (frame["building_count"].ge(1) & frame["building_area_total"].ge(400.0))
            | frame["building_area_total"].ge(1200.0)
        )
        & frame["ai_score_base"].ge(82.0)
    ].copy()
    candidate["sample_score"] = (
        candidate["ai_score_base"].fillna(0.0)
        + np.log1p(candidate["building_area_total"].clip(lower=0.0)) * 6.0
        + candidate["building_count"].clip(lower=0.0, upper=4.0) * 6.0
    )
    candidate["sample_group"] = "control"
    candidate["sample_reason"] = "strong_improved_reference"
    candidate["review_priority"] = "medium"
    return candidate


def vacant_control_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    no_structure = _no_structure_mask(frame)
    candidate = frame.loc[
        no_structure
        & frame["ai_score_base"].le(35.0).fillna(False)
        & (
            frame["parcel_vacant_flag"]
            | frame["ai_building_present_probability"].le(0.35).fillna(False)
            | frame["ai_building_present_flag"].eq(False).fillna(False)
        )
    ].copy()
    candidate["sample_score"] = (
        (40.0 - candidate["ai_score_base"].fillna(40.0)).clip(lower=0.0)
        + np.where(candidate["parcel_vacant_flag"], 20.0, 0.0)
        + (45.0 - candidate["imagery_driveway_signal"].fillna(45.0)).clip(lower=0.0) * 0.20
        + (25.0 - candidate["imagery_clearing_signal"].fillna(25.0)).clip(lower=0.0) * 0.40
    )
    candidate["sample_group"] = "control"
    candidate["sample_reason"] = "strong_vacant_reference"
    candidate["review_priority"] = "medium"
    return candidate


def geometry_filter_qa_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    candidate = frame.loc[frame["geometry_review_excluded_flag"]].copy()
    if candidate.empty:
        return candidate
    candidate["sample_score"] = (
        candidate["aspect_ratio"].fillna(0.0) * 8.0
        + (1.0 - candidate["compactness"].fillna(1.0)).clip(lower=0.0) * 100.0
        + candidate["part_count"].clip(lower=0, upper=8) * 6.0
    )
    candidate["sample_group"] = "qa"
    candidate["sample_reason"] = "geometry_filter_excluded"
    candidate["review_priority"] = "low"
    return candidate


def _format_number(value: object, *, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _build_review_hint(row: pd.Series) -> str:
    if row.get("sample_reason") == "road_or_clearing_context_false_positive":
        return (
            "No mapped structure; AI confidence "
            f"{_format_number(row.get('building_present_confidence'))}; driveway "
            f"{_format_number(row.get('imagery_driveway_signal'))}; clearing "
            f"{_format_number(row.get('imagery_clearing_signal'))}."
        )
    if row.get("sample_reason") == "neighbor_outside_parcel_false_positive":
        return (
            "No mapped structure; AI confidence "
            f"{_format_number(row.get('building_present_confidence'))}; nearby-building density "
            f"{_format_number(row.get('nearby_building_density'))}; width est "
            f"{_format_number(row.get('parcel_width_ft_estimate'))} ft."
        )
    if row.get("sample_reason") == "strong_improved_reference":
        return (
            "Mapped building evidence present; count "
            f"{_format_number(row.get('building_count'))}; building area "
            f"{_format_number(row.get('building_area_total'))} sqft; AI confidence "
            f"{_format_number(row.get('building_present_confidence'))}."
        )
    if row.get("sample_reason") == "geometry_filter_excluded":
        return (
            "Geometry QA row; quality "
            f"{row.get('geometry_quality_flag')}; aspect "
            f"{_format_number(row.get('aspect_ratio'))}; compactness "
            f"{_format_number(row.get('compactness'), digits=3)}."
        )
    return (
        "No mapped structure; AI confidence "
        f"{_format_number(row.get('building_present_confidence'))}; vacant flag "
        f"{bool(row.get('parcel_vacant_flag'))}."
    )


def _load_tile_coordinate_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [part.strip() for part in text.split("|") if part.strip()]


def _resolve_review_tile_path(
    *,
    parcel_row_id: str,
    county_name: str | None,
    address: object,
    centroid_address: object,
    fetch_images: bool,
    tile_template: str,
) -> Path | None:
    cache_path = tile_cache_path_for_address(
        parcel_row_id,
        county_name,
        address,
        centroid_address=centroid_address,
    )
    if cache_path.exists():
        return cache_path
    if not fetch_images:
        return None
    return ensure_tile_image_for_address(
        parcel_row_id=parcel_row_id,
        county_name=county_name,
        address=address,
        refresh=False,
        template=tile_template,
        centroid_address=centroid_address,
    )


def _select_review_tile_record(tile_records: list[dict[str, object]], best_tile_label: object) -> dict[str, object]:
    best_label = "" if pd.isna(best_tile_label) else str(best_tile_label)
    for tile_record in tile_records:
        if str(tile_record.get("tile_label")) == best_label:
            return tile_record
    for tile_record in tile_records:
        if bool(tile_record.get("centroid_tile_flag")):
            return tile_record
    return tile_records[0]


def _tile_records_for_manifest(
    tile_records: list[dict[str, object]],
    *,
    selected_tile_label: str,
    max_count: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    centroid: list[dict[str, object]] = []
    others: list[dict[str, object]] = []
    for tile_record in tile_records:
        label = str(tile_record.get("tile_label") or "")
        if label == selected_tile_label:
            selected.append(tile_record)
        elif bool(tile_record.get("centroid_tile_flag")):
            centroid.append(tile_record)
        else:
            others.append(tile_record)

    ordered = selected + centroid + others
    deduped: list[dict[str, object]] = []
    seen_labels: set[str] = set()
    for tile_record in ordered:
        label = str(tile_record.get("tile_label") or "")
        if not label or label in seen_labels:
            continue
        deduped.append(tile_record)
        seen_labels.add(label)
        if len(deduped) >= max_count:
            break
    return deduped


def _save_review_crop_assets(
    *,
    parcel_row_id: str,
    county_name: str | None,
    prepared: dict[str, object],
    tile_label_value: str,
    tile_rank: int,
    asset_stem_override: str | None = None,
) -> dict[str, str]:
    county = (county_name or "unknown").strip().lower()
    asset_dir = REVIEW_ASSET_DIR / county
    asset_dir.mkdir(parents=True, exist_ok=True)

    asset_stem = asset_stem_override or f"{parcel_row_id}_tile{int(tile_rank):02d}_{tile_label_value}"
    masked_tile_path = asset_dir / f"{asset_stem}_masked_tile.png"
    display_image = prepared["image"]
    display_crop_box = prepared.get("display_crop_box")
    if isinstance(display_crop_box, tuple) and len(display_crop_box) == 4:
        display_image = prepared["image"].crop(display_crop_box)
    display_image.save(masked_tile_path, format="PNG")

    crop_paths = {
        "masked_parcel_tile_path": str(masked_tile_path),
        "masked_parcel_core_crop_path": "",
        "masked_parcel_focus_crop_path": "",
    }
    for crop_label, crop_box in prepared["crop_specs"]:
        if crop_label not in {"parcel_core", "parcel_focus"}:
            continue
        crop_path = asset_dir / f"{asset_stem}_{crop_label}.png"
        prepared["image"].crop(crop_box).save(crop_path, format="PNG")
        crop_paths[f"masked_{crop_label}_crop_path"] = str(crop_path)
    return crop_paths


def _composite_tile_records_for_geometry(
    geometry_value: object,
    tile_records: list[dict[str, object]],
    *,
    selected_tile_label: str,
    max_count: int,
) -> list[dict[str, object]]:
    shape = load_geometry_shape(geometry_value)
    polygon_parts = polygon_parts_from_shape(shape)
    if not polygon_parts:
        return []
    selected_map: dict[str, dict[str, object]] = {}
    for polygon_part in polygon_parts:
        if polygon_part.is_empty:
            continue
        part_area = float(polygon_part.area)
        candidate_records: list[tuple[float, float, int, int, dict[str, object]]] = []
        for tile_record in tile_records:
            address = tile_record["address"]
            clipped_part = clip_shape_to_tile(polygon_part, address)
            if clipped_part is None or clipped_part.is_empty:
                continue
            coverage_ratio = float(clipped_part.area / part_area) if part_area > 0 else 0.0
            tile_coverage_ratio = float(tile_record.get("parcel_tile_coverage_ratio", 0.0) or 0.0)
            candidate_records.append(
                (
                    coverage_ratio,
                    tile_coverage_ratio,
                    int(str(tile_record.get("tile_label") or "") == selected_tile_label),
                    int(bool(tile_record.get("centroid_tile_flag", False))),
                    tile_record,
                )
            )
        if not candidate_records:
            continue
        candidate_records.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2],
                -item[3],
                str(item[4].get("tile_label") or ""),
            )
        )
        cumulative_coverage = 0.0
        seen_part_labels: set[str] = set()
        for coverage_ratio, _, _, _, tile_record in candidate_records:
            label = str(tile_record.get("tile_label") or "")
            if not label or label in seen_part_labels:
                continue
            selected_map[label] = tile_record
            seen_part_labels.add(label)
            cumulative_coverage += coverage_ratio
            if cumulative_coverage >= 0.995:
                break

    ordered: list[dict[str, object]] = []
    if selected_tile_label and selected_tile_label in selected_map:
        ordered.append(selected_map.pop(selected_tile_label))
    centroid_records = [item for item in selected_map.values() if bool(item.get("centroid_tile_flag", False))]
    centroid_records.sort(
        key=lambda item: (
            -float(item.get("parcel_tile_coverage_ratio", 0.0) or 0.0),
            str(item.get("tile_label") or ""),
        )
    )
    ordered.extend(centroid_records)
    remaining_records = [item for item in selected_map.values() if not bool(item.get("centroid_tile_flag", False))]
    remaining_records.sort(
        key=lambda item: (
            -float(item.get("parcel_tile_coverage_ratio", 0.0) or 0.0),
            str(item.get("tile_label") or ""),
        )
    )
    ordered.extend(remaining_records)

    deduped: list[dict[str, object]] = []
    seen_labels: set[str] = set()
    for tile_record in ordered:
        label = str(tile_record.get("tile_label") or "")
        if not label or label in seen_labels:
            continue
        deduped.append(tile_record)
        seen_labels.add(label)
    return deduped


def _write_review_tile_manifest(
    *,
    parcel_row_id: str,
    county_name: str | None,
    manifest_rows: list[dict[str, object]],
) -> str:
    if not manifest_rows:
        return ""
    county = (county_name or "unknown").strip().lower()
    asset_dir = REVIEW_ASSET_DIR / county
    asset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = asset_dir / f"{parcel_row_id}_tile_manifest.json"
    validate_output_records(
        manifest_rows,
        expected_fields=REVIEW_TILE_MANIFEST_OUTPUT_FIELDS,
        required_fields=TILE_MANIFEST_REQUIRED_FIELDS,
        non_null_fields=["parcel_row_id", "parcel_id"],
        context=f"sample_vacancy_labeling_ms.tile_manifest_schema[{parcel_row_id}]",
    )
    validate_required_columns(
        pd.DataFrame(manifest_rows),
        required_columns=TILE_MANIFEST_REQUIRED_FIELDS,
        non_null_columns=["parcel_row_id", "parcel_id"],
        context=f"sample_vacancy_labeling_ms.tile_manifest[{parcel_row_id}]",
    )
    manifest_path.write_text(json.dumps(manifest_rows, indent=2, default=str), encoding="utf-8")
    return str(manifest_path)


def attach_imagery_columns(
    frame: pd.DataFrame,
    *,
    zoom: int,
    fetch_images: bool,
    tile_template: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    imagery_started = time.perf_counter()
    total_rows = len(frame)
    for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
        parcel_row_id = str(row["parcel_row_id"])
        longitude = float(row["longitude"])
        latitude = float(row["latitude"])
        geometry_value = row.get("geometry")
        centroid_address = centroid_tile(longitude, latitude, zoom)
        tile_plan = build_parcel_inference_tile_plan(
            geometry_value,
            centroid_address,
            use_multi_tile_extent=True,
        )
        tile_records = list(tile_plan.get("tile_records") or [])
        selected_tile_record = _select_review_tile_record(tile_records, row.get("best_tile_label"))
        selected_address = selected_tile_record["address"]
        centroid_tile_path = _resolve_review_tile_path(
            parcel_row_id=parcel_row_id,
            county_name=row.get("county_name"),
            address=centroid_address,
            centroid_address=centroid_address,
            fetch_images=fetch_images,
            tile_template=tile_template,
        )
        selected_tile_path = _resolve_review_tile_path(
            parcel_row_id=parcel_row_id,
            county_name=row.get("county_name"),
            address=selected_address,
            centroid_address=centroid_address,
            fetch_images=fetch_images,
            tile_template=tile_template,
        )
        image_url = tile_url(centroid_address, tile_template)
        image_path = ""
        asset_paths = {
            "masked_parcel_tile_path": "",
            "masked_parcel_core_crop_path": "",
            "masked_parcel_focus_crop_path": "",
        }
        review_tile_path = ""
        review_tile_manifest_path = ""
        review_tile_sample_labels = ""
        if centroid_tile_path is not None and Path(centroid_tile_path).exists():
            image_path = str(centroid_tile_path)
        if selected_tile_path is not None and Path(selected_tile_path).exists():
            review_tile_path = str(selected_tile_path)
            prepared = prepare_parcel_aware_image(
                selected_tile_path,
                address=selected_address,
                geometry_value=geometry_value,
                acreage=row.get("acreage"),
                coverage_diagnostics=selected_tile_record,
            )
            review_tile_record = selected_tile_record
            review_prepared = prepared
            review_uses_composite = False
            polygon_part_count = len(polygon_parts_from_shape(load_geometry_shape(geometry_value)))
            composite_tile_records = _composite_tile_records_for_geometry(
                geometry_value,
                tile_records,
                selected_tile_label=str(selected_tile_record["tile_label"]),
                max_count=max(MAX_REVIEW_TILE_EXPORT_COUNT, min(12, polygon_part_count)),
            )
            if (
                len(tile_records) > 1
                and not bool(selected_tile_record.get("full_parcel_visible_flag", False))
            ):
                if len(composite_tile_records) == 1:
                    candidate_record = composite_tile_records[0]
                    candidate_tile_path = _resolve_review_tile_path(
                        parcel_row_id=parcel_row_id,
                        county_name=row.get("county_name"),
                        address=candidate_record["address"],
                        centroid_address=centroid_address,
                        fetch_images=fetch_images,
                        tile_template=tile_template,
                    )
                    if candidate_tile_path is not None and Path(candidate_tile_path).exists():
                        candidate_prepared = prepare_parcel_aware_image(
                            candidate_tile_path,
                            address=candidate_record["address"],
                            geometry_value=geometry_value,
                            acreage=row.get("acreage"),
                            coverage_diagnostics=candidate_record,
                        )
                        if int(candidate_prepared.get("clipped_polygon_part_count", 0) or 0) >= int(candidate_prepared.get("polygon_part_count", 0) or 0):
                            review_tile_record = candidate_record
                            review_tile_path = str(candidate_tile_path)
                            review_prepared = candidate_prepared
                elif len(composite_tile_records) > 1:
                    composite_sources: list[tuple[object, Path | str]] = []
                    for composite_tile_record in composite_tile_records:
                        composite_tile_path = _resolve_review_tile_path(
                            parcel_row_id=parcel_row_id,
                            county_name=row.get("county_name"),
                            address=composite_tile_record["address"],
                            centroid_address=centroid_address,
                            fetch_images=fetch_images,
                            tile_template=tile_template,
                        )
                        if composite_tile_path is None or not Path(composite_tile_path).exists():
                            continue
                        composite_sources.append((composite_tile_record["address"], composite_tile_path))
                    if len(composite_sources) > 1:
                        review_prepared = prepare_parcel_aware_image_for_tile_set(
                            composite_sources,
                            geometry_value=geometry_value,
                            acreage=row.get("acreage"),
                        )
                        review_uses_composite = True
            manifest_rows: list[dict[str, object]] = []
            if review_prepared.get("parcel_boundary_crop_ready_flag"):
                asset_paths = _save_review_crop_assets(
                    parcel_row_id=parcel_row_id,
                    county_name=row.get("county_name"),
                    prepared=review_prepared,
                    tile_label_value=str(review_tile_record["tile_label"]),
                    tile_rank=int(review_tile_record["tile_rank"]),
                    asset_stem_override=(
                        f"{parcel_row_id}_parcel_composite"
                        if review_uses_composite
                        else None
                    ),
                )
            for manifest_tile_record in _tile_records_for_manifest(
                tile_records,
                selected_tile_label=str(selected_tile_record["tile_label"]),
                max_count=MAX_REVIEW_TILE_EXPORT_COUNT,
            ):
                manifest_tile_path = selected_tile_path
                if str(manifest_tile_record["tile_label"]) != str(selected_tile_record["tile_label"]):
                    manifest_tile_path = _resolve_review_tile_path(
                        parcel_row_id=parcel_row_id,
                        county_name=row.get("county_name"),
                        address=manifest_tile_record["address"],
                        centroid_address=centroid_address,
                        fetch_images=fetch_images,
                        tile_template=tile_template,
                    )
                if manifest_tile_path is None or not Path(manifest_tile_path).exists():
                    continue
                manifest_prepared = prepared
                if str(manifest_tile_record["tile_label"]) != str(selected_tile_record["tile_label"]):
                    manifest_prepared = prepare_parcel_aware_image(
                        manifest_tile_path,
                        address=manifest_tile_record["address"],
                        geometry_value=geometry_value,
                        acreage=row.get("acreage"),
                        coverage_diagnostics=manifest_tile_record,
                    )
                manifest_assets = {}
                if manifest_prepared.get("parcel_boundary_crop_ready_flag"):
                    manifest_assets = _save_review_crop_assets(
                        parcel_row_id=parcel_row_id,
                        county_name=row.get("county_name"),
                        prepared=manifest_prepared,
                        tile_label_value=str(manifest_tile_record["tile_label"]),
                        tile_rank=int(manifest_tile_record["tile_rank"]),
                    )
                manifest_rows.append(
                    {
                        "parcel_row_id": parcel_row_id,
                        "parcel_id": row.get("parcel_id"),
                        "tile_label": str(manifest_tile_record["tile_label"]),
                        "tile_rank": int(manifest_tile_record["tile_rank"]),
                        "tile_coordinate": str(manifest_tile_record.get("tile_coordinate") or ""),
                        "centroid_tile_flag": bool(manifest_tile_record.get("centroid_tile_flag", False)),
                        "original_geom_type": manifest_prepared.get("original_geom_type"),
                        "clipped_geom_type": manifest_prepared.get("clipped_geom_type"),
                        "polygon_part_count": int(manifest_prepared.get("polygon_part_count", 0) or 0),
                        "clipped_polygon_part_count": int(manifest_prepared.get("clipped_polygon_part_count", 0) or 0),
                        "bounds_before_clip": manifest_prepared.get("bounds_before_clip"),
                        "bounds_after_clip": manifest_prepared.get("bounds_after_clip"),
                        "parcel_tile_coverage_pct": manifest_tile_record.get("parcel_tile_coverage_pct"),
                        "parcel_bbox_tile_coverage_pct": manifest_tile_record.get("parcel_bbox_tile_coverage_pct"),
                        "raw_tile_path": str(manifest_tile_path),
                        **manifest_assets,
                    }
                )
            review_tile_manifest_path = _write_review_tile_manifest(
                parcel_row_id=parcel_row_id,
                county_name=row.get("county_name"),
                manifest_rows=manifest_rows,
            )
            review_tile_sample_labels = "|".join(item["tile_label"] for item in manifest_rows)
            review_crop_strategy = review_prepared.get("imagery_crop_strategy")
            review_boundary_crop_ready_flag = bool(review_prepared.get("parcel_boundary_crop_ready_flag", False))
        else:
            review_crop_strategy = pd.NA
            review_boundary_crop_ready_flag = False
            review_tile_record = selected_tile_record
        records.append(
            {
                "parcel_row_id": parcel_row_id,
                "image_url": image_url,
                "image_path": image_path,
                "raw_centroid_tile_path": image_path,
                "review_tile_label": str(review_tile_record["tile_label"]),
                "review_tile_rank": int(review_tile_record["tile_rank"]),
                "review_tile_coordinate": str(review_tile_record.get("tile_coordinate") or ""),
                "review_tile_image_url": tile_url(review_tile_record["address"], tile_template),
                "review_tile_image_path": review_tile_path,
                "original_geom_type": review_prepared.get("original_geom_type") if selected_tile_path is not None and Path(selected_tile_path).exists() else pd.NA,
                "clipped_geom_type": review_prepared.get("clipped_geom_type") if selected_tile_path is not None and Path(selected_tile_path).exists() else pd.NA,
                "polygon_part_count": int(review_prepared.get("polygon_part_count", 0) or 0) if selected_tile_path is not None and Path(selected_tile_path).exists() else 0,
                "clipped_polygon_part_count": int(review_prepared.get("clipped_polygon_part_count", 0) or 0) if selected_tile_path is not None and Path(selected_tile_path).exists() else 0,
                "bounds_before_clip": review_prepared.get("bounds_before_clip") if selected_tile_path is not None and Path(selected_tile_path).exists() else pd.NA,
                "bounds_after_clip": review_prepared.get("bounds_after_clip") if selected_tile_path is not None and Path(selected_tile_path).exists() else pd.NA,
                "review_imagery_crop_strategy": review_crop_strategy,
                "review_parcel_boundary_crop_ready_flag": review_boundary_crop_ready_flag,
                **asset_paths,
                "review_tile_manifest_path": review_tile_manifest_path,
                "review_tile_sample_labels": review_tile_sample_labels,
                "parcel_tile_coverage_pct": tile_plan.get("parcel_tile_coverage_pct"),
                "parcel_bbox_tile_coverage_pct": tile_plan.get("parcel_bbox_tile_coverage_pct"),
                "full_parcel_visible_flag": bool(tile_plan.get("full_parcel_visible_flag", False)),
                "parcel_extent_exceeds_tile_flag": bool(tile_plan.get("parcel_extent_exceeds_tile_flag", False)),
                "parcel_tile_low_coverage_flag": bool(tile_plan.get("parcel_tile_low_coverage_flag", False)),
                "multi_tile_candidate_flag": bool(tile_plan.get("multi_tile_candidate_flag", False)),
                "parcel_covering_tile_count": int(tile_plan.get("parcel_covering_tile_count", 0) or 0),
                "tile_coordinates": str(tile_plan.get("tile_coordinates") or json.dumps(_load_tile_coordinate_list(row.get("tile_coordinates")))),
                "unique_tile_count": int(tile_plan.get("unique_tile_count", row.get("unique_tile_count", 0)) or 0),
                "duplicate_tile_flag": bool(tile_plan.get("duplicate_tile_flag", row.get("duplicate_tile_flag", False))),
            }
        )
        if row_index % 25 == 0 or row_index == total_rows:
            print(f"[review-imagery] rows={row_index}/{total_rows} elapsed={time.perf_counter() - imagery_started:.1f}s")
    record_frame = pd.DataFrame(records)
    overlapping_columns = [column for column in record_frame.columns if column != "parcel_row_id" and column in frame.columns]
    if overlapping_columns:
        frame = frame.drop(columns=overlapping_columns)
    merged = contract_left_merge(frame, record_frame, on="parcel_row_id")
    validate_required_columns(
        merged,
        required_columns=["parcel_row_id", "parcel_id"],
        non_null_columns=["parcel_row_id", "parcel_id"],
        context="sample_vacancy_labeling_ms.attach_imagery_columns",
    )
    return merged


def select_labeling_rows(
    frame: pd.DataFrame,
    *,
    scene_false_positive_count: int,
    neighbor_false_positive_count: int,
    improved_control_count: int,
    vacant_control_count: int,
    seed: int,
    county_cap: int,
    selection_padding_per_bucket: int = 0,
) -> pd.DataFrame:
    selected_ids: set[str] = set()

    def padded_count(candidate_frame: pd.DataFrame, requested_count: int) -> int:
        if requested_count <= 0:
            return 0
        available_count = len(_exclude_ids(candidate_frame, selected_ids))
        return min(requested_count + max(0, int(selection_padding_per_bucket)), available_count)

    scene_candidates = scene_false_positive_candidates(frame)
    scene_sample = _ranked_diverse_sample(
        scene_candidates,
        count=padded_count(scene_candidates, scene_false_positive_count),
        seed=seed + 11,
        exclude_ids=selected_ids,
        county_cap=county_cap,
    )
    selected_ids.update(scene_sample["parcel_row_id"].astype(str).tolist())

    neighbor_candidates = neighbor_false_positive_candidates(frame)
    neighbor_sample = _ranked_diverse_sample(
        neighbor_candidates,
        count=padded_count(neighbor_candidates, neighbor_false_positive_count),
        seed=seed + 17,
        exclude_ids=selected_ids,
        county_cap=county_cap,
    )
    selected_ids.update(neighbor_sample["parcel_row_id"].astype(str).tolist())

    improved_candidates = improved_control_candidates(frame)
    improved_sample = _ranked_diverse_sample(
        improved_candidates,
        count=padded_count(improved_candidates, improved_control_count),
        seed=seed + 23,
        exclude_ids=selected_ids,
        county_cap=county_cap,
    )
    selected_ids.update(improved_sample["parcel_row_id"].astype(str).tolist())

    vacant_candidates = vacant_control_candidates(frame)
    vacant_sample = _ranked_diverse_sample(
        vacant_candidates,
        count=padded_count(vacant_candidates, vacant_control_count),
        seed=seed + 29,
        exclude_ids=selected_ids,
        county_cap=county_cap,
    )

    sampled = pd.concat([scene_sample, neighbor_sample, improved_sample, vacant_sample], ignore_index=True)
    sampled["sample_score"] = pd.to_numeric(sampled["sample_score"], errors="coerce").round(1)
    sampled["review_hint"] = sampled.apply(_build_review_hint, axis=1).astype("string")
    return sampled


def _trim_sampled_rows_to_targets(
    sampled: pd.DataFrame,
    *,
    scene_false_positive_count: int,
    neighbor_false_positive_count: int,
    improved_control_count: int,
    vacant_control_count: int,
) -> pd.DataFrame:
    targets = {
        "road_or_clearing_context_false_positive": int(scene_false_positive_count),
        "neighbor_outside_parcel_false_positive": int(neighbor_false_positive_count),
        "strong_improved_reference": int(improved_control_count),
        "strong_vacant_reference": int(vacant_control_count),
    }
    ordered_chunks: list[pd.DataFrame] = []
    for sample_reason, target_count in targets.items():
        if target_count <= 0:
            continue
        chunk = sampled.loc[sampled["sample_reason"].astype("string").eq(sample_reason)].copy()
        if chunk.empty:
            continue
        chunk = chunk.sort_values(
            ["review_priority", "sample_score", "county_name", "parcel_row_id"],
            ascending=[True, False, True, True],
            key=lambda series: series.map({"high": 0, "medium": 1, "low": 2}) if series.name == "review_priority" else series,
        ).head(target_count)
        ordered_chunks.append(chunk)
    if not ordered_chunks:
        return sampled.head(0).copy()
    return pd.concat(ordered_chunks, ignore_index=True)


def _attach_placeholder_imagery_columns(sampled: pd.DataFrame, *, zoom: int, tile_template: str) -> pd.DataFrame:
    output = sampled.copy()
    centroid_addresses = [
        centroid_tile(float(row["longitude"]), float(row["latitude"]), zoom)
        for _, row in output.iterrows()
    ]
    output["image_url"] = pd.Series([tile_url(address, tile_template) for address in centroid_addresses], index=output.index, dtype="string")
    output["image_path"] = pd.Series("", index=output.index, dtype="string")
    output["raw_centroid_tile_path"] = pd.Series("", index=output.index, dtype="string")
    output["review_tile_label"] = _frame_series(output, "best_tile_label", pd.NA).astype("string")
    output["review_tile_rank"] = pd.Series(0, index=output.index, dtype="int64")
    output["review_tile_coordinate"] = pd.Series("", index=output.index, dtype="string")
    output["review_tile_image_url"] = output["image_url"].astype("string")
    output["review_tile_image_path"] = pd.Series("", index=output.index, dtype="string")
    output["original_geom_type"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["clipped_geom_type"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["polygon_part_count"] = pd.to_numeric(_frame_series(output, "part_count", 0), errors="coerce").fillna(0).astype(int)
    output["clipped_polygon_part_count"] = pd.to_numeric(_frame_series(output, "part_count", 0), errors="coerce").fillna(0).astype(int)
    output["bounds_before_clip"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["bounds_after_clip"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["review_imagery_crop_strategy"] = pd.Series("skipped_image_export", index=output.index, dtype="string")
    output["review_parcel_boundary_crop_ready_flag"] = pd.Series(False, index=output.index, dtype="bool")
    output["masked_parcel_tile_path"] = pd.Series("", index=output.index, dtype="string")
    output["masked_parcel_core_crop_path"] = pd.Series("", index=output.index, dtype="string")
    output["masked_parcel_focus_crop_path"] = pd.Series("", index=output.index, dtype="string")
    output["review_tile_manifest_path"] = pd.Series("", index=output.index, dtype="string")
    output["review_tile_sample_labels"] = pd.Series("", index=output.index, dtype="string")
    output["parcel_tile_coverage_pct"] = pd.to_numeric(_frame_series(output, "parcel_tile_coverage_pct"), errors="coerce")
    output["parcel_bbox_tile_coverage_pct"] = pd.to_numeric(_frame_series(output, "parcel_bbox_tile_coverage_pct"), errors="coerce")
    output["full_parcel_visible_flag"] = _frame_series(output, "full_parcel_visible_flag", False).fillna(False).astype(bool)
    output["parcel_extent_exceeds_tile_flag"] = _frame_series(output, "parcel_extent_exceeds_tile_flag", False).fillna(False).astype(bool)
    output["parcel_tile_low_coverage_flag"] = _frame_series(output, "parcel_tile_low_coverage_flag", False).fillna(False).astype(bool)
    output["multi_tile_candidate_flag"] = _frame_series(output, "multi_tile_candidate_flag", False).fillna(False).astype(bool)
    output["parcel_covering_tile_count"] = pd.to_numeric(_frame_series(output, "parcel_covering_tile_count", 0), errors="coerce").fillna(0).astype(int)
    output["tile_coordinates"] = _frame_series(output, "tile_coordinates", "[]").astype("string")
    output["unique_tile_count"] = pd.to_numeric(_frame_series(output, "unique_tile_count", 0), errors="coerce").fillna(0).astype(int)
    output["duplicate_tile_flag"] = _frame_series(output, "duplicate_tile_flag", False).fillna(False).astype(bool)
    return output


def _finalize_labeling_sample(
    sampled: pd.DataFrame,
    *,
    zoom: int,
    fetch_images: bool,
    tile_template: str,
    skip_image_export: bool,
) -> pd.DataFrame:
    if skip_image_export:
        sampled = _attach_placeholder_imagery_columns(sampled, zoom=zoom, tile_template=tile_template)
    else:
        if "geometry" not in sampled.columns:
            geometry_lookup = load_parcel_geometry_lookup(sampled["parcel_row_id"].astype("string"))
            sampled["geometry"] = sampled["parcel_row_id"].astype("string").map(geometry_lookup)
        sampled = attach_imagery_columns(
            sampled,
            zoom=zoom,
            fetch_images=fetch_images,
            tile_template=tile_template,
        )
    if "review_imagery_crop_strategy" in sampled.columns:
        sampled["imagery_crop_strategy"] = (
            sampled["review_imagery_crop_strategy"].astype("string").fillna(sampled["imagery_crop_strategy"])
        )
    if "review_parcel_boundary_crop_ready_flag" in sampled.columns:
        sampled["parcel_boundary_crop_ready_flag"] = (
            _frame_series(sampled, "parcel_boundary_crop_ready_flag", False).fillna(False).astype(bool)
            | sampled["review_parcel_boundary_crop_ready_flag"].fillna(False).astype(bool)
        )
    return sampled


def _format_labeling_output(
    sampled: pd.DataFrame,
    *,
    geometry_diagnostics: dict[str, object],
    manual_review_diagnostics: dict[str, object],
    review_pool_row_count: int,
) -> pd.DataFrame:
    sampled = sampled.copy()
    sampled["ai_building_present_probability"] = pd.to_numeric(sampled["ai_building_present_probability"], errors="coerce").round(4)
    sampled["building_present_confidence"] = pd.to_numeric(sampled["building_present_confidence"], errors="coerce").round(1)
    sampled["imagery_driveway_signal"] = pd.to_numeric(sampled["imagery_driveway_signal"], errors="coerce").round(1)
    sampled["imagery_clearing_signal"] = pd.to_numeric(sampled["imagery_clearing_signal"], errors="coerce").round(1)
    sampled["parcel_tile_coverage_pct"] = pd.to_numeric(sampled["parcel_tile_coverage_pct"], errors="coerce").round(1)
    sampled["parcel_bbox_tile_coverage_pct"] = pd.to_numeric(sampled["parcel_bbox_tile_coverage_pct"], errors="coerce").round(1)
    sampled["nearby_building_density"] = pd.to_numeric(sampled["nearby_building_density"], errors="coerce").round(2)
    sampled["shape_compactness"] = pd.to_numeric(sampled["shape_compactness"], errors="coerce").round(3)
    sampled["parcel_width_ft_estimate"] = pd.to_numeric(sampled["parcel_width_ft_estimate"], errors="coerce").round(1)
    sampled["parcel_aspect_ratio_estimate"] = pd.to_numeric(sampled["parcel_aspect_ratio_estimate"], errors="coerce").round(2)
    sampled["area_acres"] = pd.to_numeric(_frame_series(sampled, "area_acres"), errors="coerce").round(3)
    sampled["perimeter_meters"] = pd.to_numeric(_frame_series(sampled, "perimeter_meters"), errors="coerce").round(2)
    sampled["bounding_box_width_meters"] = pd.to_numeric(_frame_series(sampled, "bounding_box_width_meters"), errors="coerce").round(2)
    sampled["bounding_box_height_meters"] = pd.to_numeric(_frame_series(sampled, "bounding_box_height_meters"), errors="coerce").round(2)
    sampled["aspect_ratio"] = pd.to_numeric(_frame_series(sampled, "aspect_ratio"), errors="coerce").round(2)
    sampled["compactness"] = pd.to_numeric(_frame_series(sampled, "compactness"), errors="coerce").round(3)
    sampled["is_multipart"] = _frame_series(sampled, "is_multipart", False).fillna(False).astype(bool)
    sampled["part_count"] = pd.to_numeric(_frame_series(sampled, "part_count", 0), errors="coerce").fillna(0).astype(int)
    sampled["geometry_quality_flag"] = _frame_series(sampled, "geometry_quality_flag", "good").astype("string").fillna("good")
    sampled["geometry_review_excluded_flag"] = _frame_series(sampled, "geometry_review_excluded_flag", False).fillna(False).astype(bool)
    sampled["geometry_training_excluded_flag"] = _frame_series(sampled, "geometry_training_excluded_flag", False).fillna(False).astype(bool)
    sampled["geometry_default_leads_excluded_flag"] = _frame_series(sampled, "geometry_default_leads_excluded_flag", False).fillna(False).astype(bool)
    sampled["geometry_marketability_base_flag"] = _frame_series(sampled, "geometry_marketability_base_flag", "marketable").astype("string").fillna("marketable")
    sampled["geometry_marketability_flag"] = _frame_series(sampled, "geometry_marketability_flag", "marketable").astype("string").fillna("marketable")
    sampled["geometry_marketability_context"] = _frame_series(sampled, "geometry_marketability_context", "rural").astype("string").fillna("rural")
    sampled["geometry_marketability_action"] = _frame_series(sampled, "geometry_marketability_action", "keep").astype("string").fillna("keep")
    sampled["geometry_marketability_default_leads_excluded_flag"] = _frame_series(sampled, "geometry_marketability_default_leads_excluded_flag", False).fillna(False).astype(bool)
    sampled["geometry_penalty_points"] = pd.to_numeric(_frame_series(sampled, "geometry_penalty_points"), errors="coerce").round(1)
    sampled["geometry_penalty_reason"] = _frame_series(sampled, "geometry_penalty_reason", pd.NA).astype("string")
    sampled["vacancy_manual_review_eligible_flag"] = _frame_series(sampled, "vacancy_manual_review_eligible_flag", True).fillna(False).astype(bool)
    sampled["vacancy_manual_review_exclusion_reason"] = _frame_series(sampled, "vacancy_manual_review_exclusion_reason", pd.NA).astype("string")
    sampled["parcel_covering_tile_count"] = pd.to_numeric(sampled["parcel_covering_tile_count"], errors="coerce").fillna(0).astype(int)
    sampled["unique_tile_count"] = pd.to_numeric(_frame_series(sampled, "unique_tile_count", 0), errors="coerce").fillna(0).astype(int)
    sampled["review_tile_rank"] = pd.to_numeric(_frame_series(sampled, "review_tile_rank", 0), errors="coerce").fillna(0).astype(int)
    sampled["tiles_scored_count"] = pd.to_numeric(sampled["tiles_scored_count"], errors="coerce").fillna(0).astype(int)
    sampled["tiles_with_building_signal_count"] = pd.to_numeric(sampled["tiles_with_building_signal_count"], errors="coerce").fillna(0).astype(int)
    sampled["tile_coordinates"] = _frame_series(sampled, "tile_coordinates", "[]").astype("string").fillna("[]")
    sampled["review_tile_label"] = _frame_series(sampled, "review_tile_label", pd.NA).astype("string")
    sampled["review_tile_coordinate"] = _frame_series(sampled, "review_tile_coordinate", pd.NA).astype("string")
    sampled["review_tile_image_url"] = _frame_series(sampled, "review_tile_image_url", pd.NA).astype("string")
    sampled["review_tile_image_path"] = _frame_series(sampled, "review_tile_image_path", pd.NA).astype("string")
    sampled["original_geom_type"] = _frame_series(sampled, "original_geom_type", pd.NA).astype("string")
    sampled["clipped_geom_type"] = _frame_series(sampled, "clipped_geom_type", pd.NA).astype("string")
    sampled["polygon_part_count"] = pd.to_numeric(_frame_series(sampled, "polygon_part_count", 0), errors="coerce").fillna(0).astype(int)
    sampled["clipped_polygon_part_count"] = pd.to_numeric(_frame_series(sampled, "clipped_polygon_part_count", 0), errors="coerce").fillna(0).astype(int)
    sampled["bounds_before_clip"] = _frame_series(sampled, "bounds_before_clip", pd.NA).astype("string")
    sampled["bounds_after_clip"] = _frame_series(sampled, "bounds_after_clip", pd.NA).astype("string")
    sampled["review_tile_manifest_path"] = _frame_series(sampled, "review_tile_manifest_path", pd.NA).astype("string")
    sampled["review_tile_sample_labels"] = _frame_series(sampled, "review_tile_sample_labels", pd.NA).astype("string")
    sampled["duplicate_tile_flag"] = _frame_series(sampled, "duplicate_tile_flag", False).fillna(False).astype(bool)
    sampled["best_tile_confidence"] = pd.to_numeric(sampled["best_tile_confidence"], errors="coerce").round(1)
    sampled["best_tile_probability"] = pd.to_numeric(sampled["best_tile_probability"], errors="coerce").round(4)
    sampled["best_tile_parcel_coverage_pct"] = pd.to_numeric(sampled["best_tile_parcel_coverage_pct"], errors="coerce").round(1)
    sampled["negative_tile_coverage_pct"] = pd.to_numeric(sampled["negative_tile_coverage_pct"], errors="coerce").round(1)
    sampled["lead_score_total"] = pd.to_numeric(sampled["lead_score_total"], errors="coerce").round(2)
    sampled["acreage"] = pd.to_numeric(sampled["acreage"], errors="coerce").round(3)
    sampled["building_area_total"] = pd.to_numeric(sampled["building_area_total"], errors="coerce").round(1)
    sampled["building_count"] = pd.to_numeric(sampled["building_count"], errors="coerce").fillna(0).astype(int)
    for column in MANUAL_LABEL_COLUMNS:
        sampled[column] = _blank_string_series(sampled)

    priority_order = {"high": 0, "medium": 1, "low": 2}
    sampled = sampled.sort_values(
        ["review_priority", "sample_score", "county_name", "parcel_row_id"],
        ascending=[True, False, True, True],
        key=lambda series: series.map(priority_order) if series.name == "review_priority" else series,
    ).reset_index(drop=True)
    output = sampled.loc[:, REVIEW_EXPORT_OUTPUT_COLUMNS].copy()
    validate_output_records(
        output.to_dict(orient="records"),
        expected_fields=REVIEW_EXPORT_OUTPUT_COLUMNS,
        required_fields=REVIEW_EXPORT_REQUIRED_COLUMNS,
        non_null_fields=["parcel_row_id", "parcel_id"],
        context="sample_vacancy_labeling_ms.build_labeling_sample_from_frame.schema",
    )
    validate_required_columns(
        output,
        required_columns=REVIEW_EXPORT_REQUIRED_COLUMNS,
        non_null_columns=["parcel_row_id", "parcel_id"],
        context="sample_vacancy_labeling_ms.build_labeling_sample_from_frame",
    )
    output.attrs["geometry_quality_diagnostics"] = geometry_diagnostics
    output.attrs["manual_review_diagnostics"] = manual_review_diagnostics
    output.attrs["review_geometry_pool_row_count"] = int(review_pool_row_count)
    return output


def build_labeling_sample_from_frame(
    frame: pd.DataFrame,
    *,
    scene_false_positive_count: int,
    neighbor_false_positive_count: int,
    improved_control_count: int,
    vacant_control_count: int,
    seed: int,
    zoom: int,
    fetch_images: bool,
    tile_template: str,
    county_cap: int,
    qa_bad_geometry_pct: float = DEFAULT_QA_BAD_GEOMETRY_PCT,
    skip_image_export: bool = False,
) -> pd.DataFrame:
    prepared = prepare_sampling_frame(frame)
    prepared = add_vacancy_manual_review_eligibility_fields(prepared, require_exported_imagery=False)
    geometry_diagnostics = geometry_quality_diagnostics(prepared)
    manual_review_diagnostics = vacancy_manual_review_diagnostics(prepared)
    review_pool = filter_vacancy_manual_review_eligible_frame(filter_review_geometry_frame(prepared))
    sampled = select_labeling_rows(
        review_pool,
        scene_false_positive_count=scene_false_positive_count,
        neighbor_false_positive_count=neighbor_false_positive_count,
        improved_control_count=improved_control_count,
        vacant_control_count=vacant_control_count,
        seed=seed,
        county_cap=county_cap,
        selection_padding_per_bucket=DEFAULT_REVIEW_SELECTION_PADDING_PER_BUCKET if fetch_images and not skip_image_export else 0,
    )
    if qa_bad_geometry_pct > 0:
        qa_candidates = geometry_filter_qa_candidates(prepared)
        qa_count = int(
            np.ceil(
                (
                    scene_false_positive_count
                    + neighbor_false_positive_count
                    + improved_control_count
                    + vacant_control_count
                )
                * float(qa_bad_geometry_pct)
            )
        )
        if qa_count > 0 and not qa_candidates.empty:
            qa_sample = _ranked_diverse_sample(
                qa_candidates,
                count=min(qa_count, len(qa_candidates)),
                seed=seed + 101,
                exclude_ids=set(sampled["parcel_row_id"].astype("string")),
                county_cap=max(1, min(county_cap, qa_count)),
            )
            sampled = pd.concat([sampled, qa_sample], ignore_index=True)
    sampled = _finalize_labeling_sample(
        sampled,
        zoom=zoom,
        fetch_images=fetch_images,
        tile_template=tile_template,
        skip_image_export=skip_image_export,
    )
    sampled = add_vacancy_manual_review_eligibility_fields(
        sampled,
        require_exported_imagery=fetch_images and not skip_image_export,
    )
    sampled = filter_vacancy_manual_review_eligible_frame(sampled)
    sampled = _trim_sampled_rows_to_targets(
        sampled,
        scene_false_positive_count=scene_false_positive_count,
        neighbor_false_positive_count=neighbor_false_positive_count,
        improved_control_count=improved_control_count,
        vacant_control_count=vacant_control_count,
    )
    return _format_labeling_output(
        sampled,
        geometry_diagnostics=geometry_diagnostics,
        manual_review_diagnostics=manual_review_diagnostics,
        review_pool_row_count=int(len(review_pool)),
    )


def build_labeling_sample(
    *,
    scene_false_positive_count: int,
    neighbor_false_positive_count: int,
    improved_control_count: int,
    vacant_control_count: int,
    seed: int,
    zoom: int,
    fetch_images: bool,
    tile_template: str,
    county_cap: int,
    qa_bad_geometry_pct: float = DEFAULT_QA_BAD_GEOMETRY_PCT,
    skip_image_export: bool = False,
    reuse_geometry_quality_artifact: bool = True,
    build_geometry_quality_artifact_if_missing: bool = False,
    geometry_quality_artifact_path: Path = GEOMETRY_QUALITY_ARTIFACT_PATH,
    limit: int | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    build_started = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    stage_started = time.perf_counter()
    runtime_by_stage_seconds: dict[str, float] = {}
    sampling_checkpoint = _checkpoint_file(checkpoint_dir, "review_sampling_frame")
    selected_checkpoint = _checkpoint_file(checkpoint_dir, "review_selected_rows")

    if resume and selected_checkpoint is not None and selected_checkpoint.exists():
        sampled = pd.read_parquet(selected_checkpoint, engine="pyarrow")
        checkpoint_metadata = _read_checkpoint_metadata(selected_checkpoint)
        review_pool_row_count = int(checkpoint_metadata.get("review_geometry_pool_row_count", len(sampled)))
        geometry_diagnostics = checkpoint_metadata.get("geometry_quality_diagnostics") or geometry_quality_diagnostics(sampled)
        runtime_by_stage_seconds["resume_selected_rows_seconds"] = round(time.perf_counter() - stage_started, 3)
    else:
        if resume and sampling_checkpoint is not None and sampling_checkpoint.exists():
            frame = pd.read_parquet(sampling_checkpoint, engine="pyarrow")
        else:
            frame = load_sampling_frame(
                reuse_geometry_quality_artifact=reuse_geometry_quality_artifact,
                build_geometry_quality_artifact_if_missing=build_geometry_quality_artifact_if_missing,
                geometry_quality_artifact_path=geometry_quality_artifact_path,
            )
            if limit is not None:
                frame = frame.head(int(limit)).copy()
            if sampling_checkpoint is not None:
                frame.to_parquet(sampling_checkpoint, index=False)
                _write_checkpoint_metadata(
                    sampling_checkpoint,
                    {
                        "geometry_quality_diagnostics": geometry_quality_diagnostics(frame),
                    },
                )
        runtime_by_stage_seconds["load_sampling_frame_seconds"] = round(time.perf_counter() - stage_started, 3)

        selection_started = time.perf_counter()
        prepared = prepare_sampling_frame(frame)
        prepared = add_vacancy_manual_review_eligibility_fields(prepared, require_exported_imagery=False)
        geometry_diagnostics = geometry_quality_diagnostics(prepared)
        manual_review_diagnostics = vacancy_manual_review_diagnostics(prepared)
        review_pool = filter_vacancy_manual_review_eligible_frame(filter_review_geometry_frame(prepared))
        sampled = select_labeling_rows(
            review_pool,
            scene_false_positive_count=scene_false_positive_count,
            neighbor_false_positive_count=neighbor_false_positive_count,
            improved_control_count=improved_control_count,
            vacant_control_count=vacant_control_count,
            seed=seed,
            county_cap=county_cap,
            selection_padding_per_bucket=DEFAULT_REVIEW_SELECTION_PADDING_PER_BUCKET if fetch_images and not skip_image_export else 0,
        )
        if qa_bad_geometry_pct > 0:
            qa_candidates = geometry_filter_qa_candidates(prepared)
            qa_count = int(
                np.ceil(
                    (
                        scene_false_positive_count
                        + neighbor_false_positive_count
                        + improved_control_count
                        + vacant_control_count
                    )
                    * float(qa_bad_geometry_pct)
                )
            )
            if qa_count > 0 and not qa_candidates.empty:
                qa_sample = _ranked_diverse_sample(
                    qa_candidates,
                    count=min(qa_count, len(qa_candidates)),
                    seed=seed + 101,
                    exclude_ids=set(sampled["parcel_row_id"].astype("string")),
                    county_cap=max(1, min(county_cap, qa_count)),
                )
                sampled = pd.concat([sampled, qa_sample], ignore_index=True)
        review_pool_row_count = int(len(review_pool))
        if selected_checkpoint is not None:
            sampled.to_parquet(selected_checkpoint, index=False)
            _write_checkpoint_metadata(
                selected_checkpoint,
                {
                    "review_geometry_pool_row_count": review_pool_row_count,
                    "geometry_quality_diagnostics": geometry_diagnostics,
                    "manual_review_diagnostics": manual_review_diagnostics,
                },
            )
        runtime_by_stage_seconds["select_review_rows_seconds"] = round(time.perf_counter() - selection_started, 3)
    if "manual_review_diagnostics" not in locals():
        manual_review_diagnostics = checkpoint_metadata.get("manual_review_diagnostics") if "checkpoint_metadata" in locals() else vacancy_manual_review_diagnostics(sampled)

    imagery_started = time.perf_counter()
    finalized = _finalize_labeling_sample(
        sampled.copy(),
        zoom=zoom,
        fetch_images=fetch_images,
        tile_template=tile_template,
        skip_image_export=skip_image_export,
    )
    finalized = add_vacancy_manual_review_eligibility_fields(
        finalized,
        require_exported_imagery=fetch_images and not skip_image_export,
    )
    finalized = filter_vacancy_manual_review_eligible_frame(finalized)
    finalized = _trim_sampled_rows_to_targets(
        finalized,
        scene_false_positive_count=scene_false_positive_count,
        neighbor_false_positive_count=neighbor_false_positive_count,
        improved_control_count=improved_control_count,
        vacant_control_count=vacant_control_count,
    )
    runtime_by_stage_seconds["review_image_export_seconds"] = round(time.perf_counter() - imagery_started, 3)
    output = _format_labeling_output(
        finalized,
        geometry_diagnostics=geometry_diagnostics,
        manual_review_diagnostics=manual_review_diagnostics,
        review_pool_row_count=review_pool_row_count,
    )
    output.attrs["geometry_quality_diagnostics"] = geometry_diagnostics
    output.attrs["manual_review_diagnostics"] = manual_review_diagnostics
    output.attrs["review_geometry_pool_row_count"] = review_pool_row_count
    output.attrs["runtime_by_stage_seconds"] = runtime_by_stage_seconds
    output.attrs["build_start_timestamp"] = build_started
    output.attrs["build_end_timestamp"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 300-row Mississippi parcel review sample focused on parcel-context false positives and clean control cases."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--scene-false-positive-count", type=int, default=DEFAULT_SCENE_FALSE_POSITIVE_COUNT)
    parser.add_argument("--neighbor-false-positive-count", type=int, default=DEFAULT_NEIGHBOR_FALSE_POSITIVE_COUNT)
    parser.add_argument("--improved-control-count", type=int, default=DEFAULT_IMPROVED_CONTROL_COUNT)
    parser.add_argument("--vacant-control-count", type=int, default=DEFAULT_VACANT_CONTROL_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--fetch-images", action="store_true")
    parser.add_argument("--tile-template", default=DEFAULT_TILE_URL_TEMPLATE)
    parser.add_argument("--county-cap", type=int, default=DEFAULT_COUNTY_CAP)
    parser.add_argument("--qa-bad-geometry-pct", type=float, default=DEFAULT_QA_BAD_GEOMETRY_PCT)
    parser.add_argument("--skip-image-export", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-geometry-quality-artifact", dest="reuse_geometry_quality_artifact", action="store_true", default=True)
    parser.add_argument("--recompute-geometry-quality", dest="reuse_geometry_quality_artifact", action="store_false")
    parser.add_argument("--build-geometry-quality-artifact-if-missing", action="store_true")
    parser.add_argument("--geometry-quality-artifact-path", default=str(GEOMETRY_QUALITY_ARTIFACT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = build_labeling_sample(
        scene_false_positive_count=args.scene_false_positive_count,
        neighbor_false_positive_count=args.neighbor_false_positive_count,
        improved_control_count=args.improved_control_count,
        vacant_control_count=args.vacant_control_count,
        seed=args.seed,
        zoom=args.zoom,
        fetch_images=args.fetch_images,
        tile_template=args.tile_template,
        county_cap=args.county_cap,
        qa_bad_geometry_pct=args.qa_bad_geometry_pct,
        skip_image_export=args.skip_image_export,
        reuse_geometry_quality_artifact=args.reuse_geometry_quality_artifact,
        build_geometry_quality_artifact_if_missing=args.build_geometry_quality_artifact_if_missing,
        geometry_quality_artifact_path=Path(args.geometry_quality_artifact_path),
        limit=args.limit,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        resume=args.resume,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_payload = {
        "row_count": int(len(sample)),
        "build_start_timestamp": sample.attrs.get("build_start_timestamp"),
        "build_end_timestamp": sample.attrs.get("build_end_timestamp"),
        "runtime_by_stage_seconds": sample.attrs.get("runtime_by_stage_seconds", {}),
        "review_geometry_pool_row_count": int(sample.attrs.get("review_geometry_pool_row_count", len(sample))),
        "geometry_quality_diagnostics": sample.attrs.get("geometry_quality_diagnostics", {}),
        "manual_review_diagnostics": sample.attrs.get("manual_review_diagnostics", {}),
        "sample_reason_counts": sample.groupby(["sample_group", "sample_reason"]).size().reset_index(name="count").to_dict(orient="records"),
        "review_priority_counts": sample["review_priority"].astype("string").value_counts().to_dict(),
        "county_distribution_top_20": sample["county_name"].astype("string").value_counts().head(20).to_dict(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(sample)} labeling rows to {output_path}")
    print(f"Wrote labeling summary to {summary_path}")
    print(sample.groupby(["sample_group", "sample_reason"]).size().to_string())


if __name__ == "__main__":
    main()
