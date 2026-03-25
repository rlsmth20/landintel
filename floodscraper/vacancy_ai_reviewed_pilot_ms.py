from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from state_registry import load_state_definition, reviewed_pilot_default_outputs
from vacancy_ai_common import (
    AI_DATA_DIR,
    DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD,
    DEFAULT_TILE_NEGATIVE_CONFIDENCE_THRESHOLD,
    MODEL_VERSION,
    TRAINING_MANIFEST_PATH,
    aggregate_parcel_tile_predictions,
    building_present_confidence_from_probability,
    extract_image_features,
    feature_columns,
    imagery_context_signals,
    load_tile_image,
    write_metrics,
)


DEFAULT_STATE_CODE = "ms"
DEFAULT_RUN_NAME = "reviewed50"
DEFAULT_STATE_OUTPUTS = reviewed_pilot_default_outputs(DEFAULT_STATE_CODE, run_name=DEFAULT_RUN_NAME)
DEFAULT_REVIEWED_PILOT_PATH = DEFAULT_STATE_OUTPUTS["review_input"]
DEFAULT_PILOT_MANIFEST_PATH = DEFAULT_STATE_OUTPUTS["manifest"]
DEFAULT_PILOT_SUMMARY_PATH = DEFAULT_STATE_OUTPUTS["summary"]
DEFAULT_PILOT_MODEL_PATH = DEFAULT_STATE_OUTPUTS["model"]
DEFAULT_PILOT_CV_PREDICTIONS_PATH = DEFAULT_STATE_OUTPUTS["cv_predictions"]
DEFAULT_PILOT_ERROR_ANALYSIS_PATH = DEFAULT_STATE_OUTPUTS["error_analysis"]
DEFAULT_PILOT_ERROR_SUMMARY_PATH = DEFAULT_STATE_OUTPUTS["error_summary"]
DEFAULT_CURRENT_FEATURE_MANIFEST_PATH = DEFAULT_STATE_OUTPUTS["feature_manifest"]
DEFAULT_CV_SPLITS = 5
THRESHOLD_GUIDANCE_VALUES = [0.50, 0.60, 0.70, 0.82]

POSITIVE_MANUAL_LABELS = {
    "improved": 1,
    "structure_present": 1,
    "structure_inside_parcel": 1,
    "structureinsideparcel": 1,
    "present": 1,
}
NEGATIVE_MANUAL_LABELS = {
    "vacant": 0,
    "likely_vacant": 0,
    "no_structure": 0,
    "no_structure_inside_parcel": 0,
    "structure_absent": 0,
    "absent": 0,
}
AMBIGUOUS_MANUAL_LABELS = {
    "unknown",
    "needs_review",
    "ambiguous",
    "unclear",
    "unsure",
}
CONFIDENCE_NORMALIZATION = {
    "high": "high",
    "medium": "medium",
    "low": "low",
}
REVIEW_NOTE_TEXT_COLUMNS = ["manual_review_notes", "review_hint"]


def _normalize_token(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    token = str(value).strip().lower()
    if not token:
        return None
    for old, new in (("-", "_"), ("/", "_"), (" ", "_")):
        token = token.replace(old, new)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or None


def normalize_manual_training_label(value: object) -> str | None:
    return _normalize_token(value)


def normalize_manual_review_confidence(value: object) -> str | None:
    token = _normalize_token(value)
    if token is None:
        return None
    return CONFIDENCE_NORMALIZATION.get(token, token)


def map_manual_training_label_to_target(value: object) -> tuple[str | None, int | None, str | None]:
    normalized = normalize_manual_training_label(value)
    if normalized is None:
        return None, None, "missing_manual_training_label"
    if normalized in POSITIVE_MANUAL_LABELS:
        return normalized, POSITIVE_MANUAL_LABELS[normalized], None
    if normalized in NEGATIVE_MANUAL_LABELS:
        return normalized, NEGATIVE_MANUAL_LABELS[normalized], None
    if normalized in AMBIGUOUS_MANUAL_LABELS:
        return normalized, None, "ambiguous_manual_training_label"
    return normalized, None, "invalid_manual_training_label"


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _text_blob(row: pd.Series) -> str:
    parts = [str(row.get(column) or "").lower() for column in REVIEW_NOTE_TEXT_COLUMNS]
    return " ".join(part for part in parts if part).strip()


def _bool_error_type(target: int, predicted_positive: bool) -> str:
    if int(target) == 1 and not bool(predicted_positive):
        return "false_negative"
    if int(target) == 0 and bool(predicted_positive):
        return "false_positive"
    return "correct"


def _series_max(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    series = pd.to_numeric(frame[column], errors="coerce")
    if series.dropna().empty:
        return default
    return float(series.max())


def _threshold_metrics(y_true: pd.Series, probability: pd.Series, threshold: float) -> dict[str, Any]:
    y_true_int = pd.to_numeric(y_true, errors="coerce").astype(int)
    y_pred = probability.astype(float).ge(threshold)
    tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred.astype(int), labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(round(accuracy_score(y_true_int, y_pred), 4)),
        "precision": float(round(precision_score(y_true_int, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_true_int, y_pred, zero_division=0), 4)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def prepare_review_label_frame(review_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "parcel_row_id" not in review_frame.columns or "manual_training_label" not in review_frame.columns:
        raise ValueError("Pilot review file must contain parcel_row_id and manual_training_label columns.")

    prepared = review_frame.copy()
    mapped = prepared["manual_training_label"].apply(map_manual_training_label_to_target)
    prepared["manual_training_label_normalized"] = mapped.apply(lambda item: item[0])
    prepared["structure_present_target"] = mapped.apply(lambda item: item[1])
    prepared["manual_training_label_exclusion_reason"] = mapped.apply(lambda item: item[2])
    prepared["manual_review_confidence_normalized"] = prepared["manual_review_confidence"].apply(
        normalize_manual_review_confidence
    )

    labeled = prepared.loc[prepared["manual_training_label"].notna()].copy()
    labeled["parcel_row_id"] = labeled["parcel_row_id"].astype("string")
    if labeled["parcel_row_id"].duplicated().any():
        duplicates = labeled.loc[labeled["parcel_row_id"].duplicated(keep=False), "parcel_row_id"].astype("string").tolist()
        raise ValueError(f"Pilot review file contains duplicate parcel_row_id values: {duplicates[:10]}")

    usable = labeled.loc[labeled["structure_present_target"].notna()].copy()
    usable["structure_present_target"] = pd.to_numeric(usable["structure_present_target"], errors="coerce").astype(int)

    summary = {
        "total_input_rows": int(len(prepared)),
        "labeled_rows_non_null": int(len(labeled)),
        "rows_used_after_label_validation": int(len(usable)),
        "rows_excluded_before_join": {
            "missing_manual_training_label": int(prepared["manual_training_label"].isna().sum()),
            "ambiguous_manual_training_label": int(
                labeled["manual_training_label_exclusion_reason"].eq("ambiguous_manual_training_label").sum()
            ),
            "invalid_manual_training_label": int(
                labeled["manual_training_label_exclusion_reason"].eq("invalid_manual_training_label").sum()
            ),
        },
        "manual_training_label_counts_raw": {
            str(key): int(value)
            for key, value in prepared["manual_training_label"].value_counts(dropna=False).items()
        },
        "manual_training_label_counts_used": {
            str(key): int(value)
            for key, value in usable["manual_training_label_normalized"].value_counts(dropna=False).items()
        },
        "manual_review_confidence_counts_raw": {
            str(key): int(value)
            for key, value in labeled["manual_review_confidence"].value_counts(dropna=False).items()
        },
        "manual_review_confidence_counts_used": {
            str(key): int(value)
            for key, value in usable["manual_review_confidence_normalized"].value_counts(dropna=False).items()
        },
        "target_mapping_table": [
            {"manual_training_label": "Improved", "normalized_label": "improved", "structure_present_target": 1},
            {"manual_training_label": "Vacant", "normalized_label": "vacant", "structure_present_target": 0},
            {"manual_training_label": "Unknown", "normalized_label": "unknown", "structure_present_target": None},
        ],
    }
    return usable, summary


def _default_feature_manifest_path(state_code: str, *, run_name: str) -> Path:
    configured = reviewed_pilot_default_outputs(state_code, run_name=run_name)["feature_manifest"]
    if configured.exists():
        return configured
    return TRAINING_MANIFEST_PATH


def _derive_output_paths(output_prefix: Path | None, *, state_code: str, run_name: str) -> dict[str, Path]:
    if output_prefix is None:
        return reviewed_pilot_default_outputs(state_code, run_name=run_name)
    prefix_parent = output_prefix.parent
    prefix_name = output_prefix.name
    return {
        "review_input": reviewed_pilot_default_outputs(state_code, run_name=run_name)["review_input"],
        "feature_manifest": reviewed_pilot_default_outputs(state_code, run_name=run_name)["feature_manifest"],
        "manifest": prefix_parent / f"{prefix_name}_training_manifest.parquet",
        "summary": prefix_parent / f"{prefix_name}_summary.json",
        "cv_predictions": prefix_parent / f"{prefix_name}_cv_predictions.csv",
        "error_analysis": prefix_parent / f"{prefix_name}_error_analysis.csv",
        "error_summary": prefix_parent / f"{prefix_name}_error_analysis_summary.json",
        "model": prefix_parent / f"{prefix_name}_model.joblib",
    }


def build_pilot_training_manifest_frame(
    *,
    review_frame: pd.DataFrame,
    feature_manifest_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    feature_manifest = feature_manifest_frame.copy()
    if not feature_manifest.empty and "parcel_row_id" in feature_manifest.columns:
        feature_manifest["parcel_row_id"] = feature_manifest["parcel_row_id"].astype("string")
        feature_columns_used = feature_columns(feature_manifest)
        supplemental_feature_lookup = (
            feature_manifest.sort_values(["parcel_row_id"])
            .drop_duplicates(subset=["parcel_row_id"], keep="first")
            .set_index("parcel_row_id", drop=False)
        )
    else:
        feature_columns_used = []
        supplemental_feature_lookup = pd.DataFrame()

    review_source_frame = review_frame.copy()
    review_source_frame["parcel_row_id"] = review_source_frame["parcel_row_id"].astype("string")
    review_source_frame["review_parcel_id"] = review_source_frame.get("parcel_id")
    review_source_frame["review_county_name"] = review_source_frame.get("county_name")
    parcel_ids_requested = review_source_frame["parcel_row_id"].astype("string").dropna().unique().tolist()

    sampled_feature_manifest_intersection = 0
    if not supplemental_feature_lookup.empty:
        sampled_feature_manifest_intersection = int(
            review_source_frame["parcel_row_id"].astype("string").isin(supplemental_feature_lookup.index).sum()
        )

    pilot_manifest = _build_generated_rows_from_review_exports(
        review_frame=review_source_frame,
        parcel_row_ids=parcel_ids_requested,
        supplemental_feature_lookup=supplemental_feature_lookup,
    )
    if pilot_manifest.empty:
        raise ValueError("No reviewed parcels produced pilot feature rows from the review export crops.")

    pilot_manifest["structure_present_target"] = pd.to_numeric(
        pilot_manifest["structure_present_target"],
        errors="coerce",
    ).astype(int)
    pilot_manifest["manual_review_confidence_normalized"] = pilot_manifest[
        "manual_review_confidence_normalized"
    ].astype("string")

    parcel_id_mismatch_count = int(
        (
            pilot_manifest["review_parcel_id"].notna()
            & pilot_manifest["parcel_id"].notna()
            & pilot_manifest["review_parcel_id"].astype("string").ne(pilot_manifest["parcel_id"].astype("string"))
        ).sum()
    )
    county_mismatch_count = int(
        (
            pilot_manifest["review_county_name"].notna()
            & pilot_manifest["county_name"].notna()
            & pilot_manifest["review_county_name"].astype("string").str.lower().ne(
                pilot_manifest["county_name"].astype("string").str.lower()
            )
        ).sum()
    )

    missing_image_rows = int(pilot_manifest["image_path"].isna().sum()) if "image_path" in pilot_manifest.columns else int(len(pilot_manifest))
    if "image_path" in pilot_manifest.columns:
        pilot_manifest = pilot_manifest.loc[pilot_manifest["image_path"].notna()].copy()

    if not feature_columns_used:
        feature_columns_used = feature_columns(
            pilot_manifest.drop(
                columns=[
                    column
                    for column in [
                        "manual_training_label",
                        "manual_training_label_normalized",
                        "structure_present_target",
                        "manual_review_confidence",
                        "manual_review_confidence_normalized",
                        "manual_review_notes",
                        "review_parcel_id",
                        "review_county_name",
                        "review_hint",
                    ]
                    if column in pilot_manifest.columns
                ],
                errors="ignore",
            )
        )

    manifest_summary = {
        "pilot_manifest_row_count": int(len(pilot_manifest)),
        "pilot_manifest_parcel_count": int(pilot_manifest["parcel_row_id"].astype("string").nunique()),
        "pilot_manifest_class_balance_rows": {
            str(key): int(value)
            for key, value in pilot_manifest["structure_present_target"].value_counts().sort_index().items()
        },
        "pilot_manifest_class_balance_parcels": {
            str(key): int(value)
            for key, value in pilot_manifest[["parcel_row_id", "structure_present_target"]]
            .drop_duplicates()["structure_present_target"]
            .value_counts()
            .sort_index()
            .items()
        },
        "feature_column_count": int(len(feature_columns_used)),
        "feature_columns": feature_columns_used,
        "feature_source_workflow": "review_export_crops_only",
        "feature_source_parcel_counts": {
            "review_export_crops": int(pilot_manifest["parcel_row_id"].astype("string").nunique()),
            "sampled_feature_manifest_intersection": int(sampled_feature_manifest_intersection),
        },
        "feature_source_investigation": {
            "reviewed_labeled_parcel_count": int(len(parcel_ids_requested)),
            "sampled_feature_manifest_intersection_parcel_count": int(sampled_feature_manifest_intersection),
            "reason_for_partial_intersection": (
                "The geometry-quality statewide feature manifest is a sampled 6,000-parcel training subset, "
                "so it does not contain every manually reviewed parcel."
            ),
            "implemented_reviewed_feature_workflow": (
                "All reviewed parcels now generate features directly from the current review-export "
                "parcel_core and parcel_focus crops for a consistent reviewed-label training path."
            ),
        },
        "rows_excluded_after_join": {
            "missing_model_image_rows": int(missing_image_rows),
        },
        "parcel_id_mismatch_count": parcel_id_mismatch_count,
        "county_name_mismatch_count": county_mismatch_count,
    }
    return pilot_manifest, manifest_summary, feature_columns_used


def _parse_review_tile_coordinate(value: object) -> tuple[int | None, int | None, int | None]:
    token = str(value or "").strip()
    if not token:
        return None, None, None
    parts = token.split("/")
    if len(parts) != 3:
        return None, None, None
    try:
        zoom = int(parts[0])
        tile_x = int(parts[1])
        tile_y = int(parts[2])
    except Exception:
        return None, None, None
    return zoom, tile_x, tile_y


def _build_generated_rows_from_review_exports(
    *,
    review_frame: pd.DataFrame,
    parcel_row_ids: list[str],
    supplemental_feature_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not parcel_row_ids:
        return pd.DataFrame()

    review_lookup = review_frame.set_index(review_frame["parcel_row_id"].astype("string"), drop=False)
    generated_rows: list[dict[str, Any]] = []
    crop_specs = [
        ("parcel_core", "masked_parcel_core_crop_path"),
        ("parcel_focus", "masked_parcel_focus_crop_path"),
    ]
    for parcel_row_id in parcel_row_ids:
        if parcel_row_id not in review_lookup.index:
            continue
        row = review_lookup.loc[parcel_row_id]
        supplemental_row = None
        if supplemental_feature_lookup is not None and not supplemental_feature_lookup.empty:
            if parcel_row_id in supplemental_feature_lookup.index:
                supplemental_row = supplemental_feature_lookup.loc[parcel_row_id]
        tile_label = str(row.get("review_tile_label") or f"review_export_{parcel_row_id}")
        tile_coordinate_value = row.get("review_tile_coordinate")
        zoom, tile_x, tile_y = _parse_review_tile_coordinate(tile_coordinate_value)
        for crop_label, path_column in crop_specs:
            image_path = row.get(path_column)
            if image_path is None or (isinstance(image_path, float) and np.isnan(image_path)):
                continue
            path = Path(str(image_path))
            if not path.exists():
                continue
            image = load_tile_image(path)
            features = extract_image_features(image, (0, 0, image.width, image.height))
            context_signals = imagery_context_signals(features)
            generated_rows.append(
                {
                    "parcel_row_id": parcel_row_id,
                    "parcel_id": row.get("review_parcel_id") or row.get("parcel_id") or (supplemental_row.get("parcel_id") if supplemental_row is not None else None),
                    "state_code": row.get("state_code") or (supplemental_row.get("state_code") if supplemental_row is not None else None) or "MS",
                    "county_name": row.get("review_county_name") or row.get("county_name") or (supplemental_row.get("county_name") if supplemental_row is not None else None),
                    "county_fips": row.get("county_fips") if row.get("county_fips") is not None else (supplemental_row.get("county_fips") if supplemental_row is not None else None),
                    "image_path": str(path),
                    "weak_building_label": row.get("weak_building_label") if row.get("weak_building_label") is not None else row.get("structure_present_target"),
                    "weak_label_source": "reviewed_label_target",
                    "weak_label_rule": "manual_review_training_label",
                    "label_reliability_tier": row.get("label_reliability_tier") or "reviewed",
                    "building_count": row.get("building_count") if row.get("building_count") is not None else (supplemental_row.get("building_count") if supplemental_row is not None else None),
                    "building_area_total": row.get("building_area_total") if row.get("building_area_total") is not None else (supplemental_row.get("building_area_total") if supplemental_row is not None else None),
                    "parcel_vacant_flag": (
                        row.get("parcel_vacant_flag")
                        if row.get("parcel_vacant_flag") is not None
                        else (supplemental_row.get("parcel_vacant_flag") if supplemental_row is not None else None)
                    ),
                    "total_value": row.get("total_value") if row.get("total_value") is not None else (supplemental_row.get("total_value") if supplemental_row is not None else None),
                    "improvement_value_1": row.get("improvement_value_1") if row.get("improvement_value_1") is not None else (supplemental_row.get("improvement_value_1") if supplemental_row is not None else None),
                    "improvement_value_2": row.get("improvement_value_2") if row.get("improvement_value_2") is not None else (supplemental_row.get("improvement_value_2") if supplemental_row is not None else None),
                    "area_acres": row.get("area_acres"),
                    "perimeter_meters": row.get("perimeter_meters"),
                    "bounding_box_width_meters": row.get("bounding_box_width_meters"),
                    "bounding_box_height_meters": row.get("bounding_box_height_meters"),
                    "aspect_ratio": row.get("aspect_ratio"),
                    "compactness": row.get("compactness"),
                    "is_multipart": row.get("is_multipart"),
                    "part_count": row.get("part_count"),
                    "geometry_quality_flag": row.get("geometry_quality_flag"),
                    "geometry_training_excluded_flag": row.get("geometry_training_excluded_flag"),
                    "imagery_source": "review_export_crop",
                    "imagery_zoom": zoom,
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "tile_label": tile_label,
                    "tile_coordinate": tile_coordinate_value,
                    "tile_rank": row.get("review_tile_rank"),
                    "centroid_tile_flag": False,
                    "tile_selection_strategy": "review_export_selected_crop",
                    "tile_selection_role": "review_export",
                    "tile_source_mode": "review_export_existing_crop",
                    "selected_tile_count": 1,
                    "selected_tile_labels": json.dumps([tile_label]),
                    "manifest_version": MODEL_VERSION,
                    "dataset_scope": "reviewed50_pilot",
                    "app_ready_only": True,
                    "cached_tile_required_flag": True,
                    "use_multi_tile_extent": _as_bool(row.get("multi_tile_candidate_flag"), False),
                    "model_version": MODEL_VERSION,
                    "imagery_crop_strategy": row.get("imagery_crop_strategy") or "review_export_crop",
                    "imagery_crop_label": crop_label,
                    "parcel_boundary_crop_ready_flag": _as_bool(row.get("parcel_boundary_crop_ready_flag"), False),
                    "imagery_driveway_signal": round(float(context_signals["imagery_driveway_signal"]), 1),
                    "imagery_clearing_signal": round(float(context_signals["imagery_clearing_signal"]), 1),
                    "crop_parcel_coverage_ratio": 1.0,
                    "parcel_tile_coverage_ratio": _as_float(row.get("parcel_tile_coverage_pct"), 0.0) / 100.0,
                    "parcel_tile_coverage_pct": _as_float(row.get("parcel_tile_coverage_pct"), np.nan),
                    "parcel_bbox_tile_coverage_ratio": _as_float(row.get("parcel_bbox_tile_coverage_pct"), 0.0) / 100.0,
                    "parcel_bbox_tile_coverage_pct": _as_float(row.get("parcel_bbox_tile_coverage_pct"), np.nan),
                    "full_parcel_visible_flag": _as_bool(row.get("full_parcel_visible_flag"), False),
                    "parcel_extent_exceeds_tile_flag": _as_bool(row.get("parcel_extent_exceeds_tile_flag"), False),
                    "parcel_tile_low_coverage_flag": _as_bool(row.get("parcel_tile_low_coverage_flag"), False),
                    "multi_tile_candidate_flag": _as_bool(row.get("multi_tile_candidate_flag"), False),
                    "parcel_covering_tile_count": row.get("parcel_covering_tile_count"),
                    "tile_coordinates": row.get("tile_coordinates"),
                    "unique_tile_count": row.get("unique_tile_count"),
                    "duplicate_tile_flag": row.get("duplicate_tile_flag"),
                    "original_geom_type": row.get("original_geom_type"),
                    "clipped_geom_type": row.get("clipped_geom_type"),
                    "polygon_part_count": row.get("polygon_part_count"),
                    "clipped_polygon_part_count": row.get("clipped_polygon_part_count"),
                    "bounds_before_clip": row.get("bounds_before_clip"),
                    "bounds_after_clip": row.get("bounds_after_clip"),
                    "manual_training_label": row.get("manual_training_label"),
                    "manual_training_label_normalized": row.get("manual_training_label_normalized"),
                    "structure_present_target": row.get("structure_present_target"),
                    "manual_review_confidence": row.get("manual_review_confidence"),
                    "manual_review_confidence_normalized": row.get("manual_review_confidence_normalized"),
                    "manual_review_notes": row.get("manual_review_notes"),
                    "review_hint": row.get("review_hint"),
                    "review_parcel_id": row.get("review_parcel_id") or row.get("parcel_id"),
                    "review_county_name": row.get("review_county_name") or row.get("county_name"),
                    "raw_centroid_tile_path": row.get("raw_centroid_tile_path"),
                    "masked_parcel_tile_path": row.get("masked_parcel_tile_path"),
                    "masked_parcel_core_crop_path": row.get("masked_parcel_core_crop_path"),
                    "masked_parcel_focus_crop_path": row.get("masked_parcel_focus_crop_path"),
                    "review_tile_label": row.get("review_tile_label"),
                    "review_tile_rank": row.get("review_tile_rank"),
                    "review_tile_coordinate": row.get("review_tile_coordinate"),
                    "review_tile_manifest_path": row.get("review_tile_manifest_path"),
                    "nearby_building_density": row.get("nearby_building_density"),
                    "shape_compactness": row.get("shape_compactness"),
                    "parcel_width_ft_estimate": row.get("parcel_width_ft_estimate"),
                    "parcel_aspect_ratio_estimate": row.get("parcel_aspect_ratio_estimate"),
                    "geometry_marketability_flag": row.get("geometry_marketability_flag"),
                    "geometry_marketability_action": row.get("geometry_marketability_action"),
                    "geometry_marketability_context": row.get("geometry_marketability_context"),
                    "best_tile_confidence": row.get("best_tile_confidence"),
                    "best_tile_parcel_coverage_pct": row.get("best_tile_parcel_coverage_pct"),
                    "negative_tile_coverage_pct": row.get("negative_tile_coverage_pct"),
                    **features,
                }
            )
    return pd.DataFrame(generated_rows)


def _aggregate_scored_rows_to_parcels(scored_rows: pd.DataFrame) -> pd.DataFrame:
    parcel_records: list[dict[str, Any]] = []
    for parcel_row_id, parcel_group in scored_rows.groupby("parcel_row_id", sort=False):
        tile_predictions: list[dict[str, Any]] = []
        for current_tile_label, tile_group in parcel_group.groupby("tile_label", sort=False):
            best_index = tile_group["predicted_probability"].astype(float).idxmax()
            best_row = tile_group.loc[best_index]
            probability = _as_float(best_row["predicted_probability"])
            confidence = building_present_confidence_from_probability(probability)
            crop_label = str(best_row.get("imagery_crop_label") or "")
            crop_coverage_ratio = _as_float(best_row.get("crop_parcel_coverage_ratio"))
            tile_predictions.append(
                {
                    "tile_label": str(current_tile_label),
                    "best_crop_label": crop_label,
                    "probability": probability,
                    "building_present_confidence": confidence,
                    "tile_parcel_coverage_ratio": _as_float(best_row.get("parcel_tile_coverage_ratio")),
                    "parcel_coverage_ratio": crop_coverage_ratio,
                    "tile_building_signal_flag": bool(
                        confidence >= DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD
                        and (crop_label == "parcel_core" or crop_coverage_ratio >= 0.35 or probability >= 0.90)
                    ),
                    "tile_negative_signal_flag": bool(
                        confidence <= DEFAULT_TILE_NEGATIVE_CONFIDENCE_THRESHOLD
                    ),
                }
            )

        aggregation = aggregate_parcel_tile_predictions(tile_predictions)
        first_row = parcel_group.iloc[0]
        predicted_probability = float(aggregation["probability"])
        predicted_confidence = float(aggregation["building_present_confidence"])
        parcel_records.append(
            {
                "parcel_row_id": str(parcel_row_id),
                "parcel_id": first_row.get("parcel_id"),
                "county_name": first_row.get("county_name"),
                "structure_present_target": int(first_row["structure_present_target"]),
                "manual_training_label": first_row.get("manual_training_label"),
                "manual_review_confidence": first_row.get("manual_review_confidence"),
                "manual_review_confidence_normalized": first_row.get("manual_review_confidence_normalized"),
                "manual_review_notes": first_row.get("manual_review_notes"),
                "review_hint": first_row.get("review_hint"),
                "predicted_probability": predicted_probability,
                "predicted_confidence": predicted_confidence,
                "predicted_positive_050": bool(predicted_probability >= 0.50),
                "predicted_positive_060": bool(predicted_probability >= 0.60),
                "predicted_positive_070": bool(predicted_probability >= 0.70),
                "predicted_positive_082": bool(predicted_probability >= 0.82),
                "best_tile_label": aggregation["best_tile_label"],
                "best_tile_crop_label": aggregation["best_tile_crop_label"],
                "best_tile_probability": aggregation["best_tile_probability"],
                "best_tile_parcel_coverage_pct": aggregation["best_tile_parcel_coverage_pct"],
                "tiles_scored_count": int(aggregation["tiles_scored_count"]),
                "tiles_with_building_signal_count": int(aggregation["tiles_with_building_signal_count"]),
                "cv_fold": int(_as_float(first_row.get("cv_fold"), 0)),
                "masked_parcel_tile_path": first_row.get("masked_parcel_tile_path"),
                "masked_parcel_core_crop_path": first_row.get("masked_parcel_core_crop_path"),
                "masked_parcel_focus_crop_path": first_row.get("masked_parcel_focus_crop_path"),
                "raw_centroid_tile_path": first_row.get("raw_centroid_tile_path"),
                "review_tile_manifest_path": first_row.get("review_tile_manifest_path"),
                "tile_label": first_row.get("tile_label"),
                "tile_coordinate": first_row.get("tile_coordinate"),
                "full_parcel_visible_flag": _as_bool(first_row.get("full_parcel_visible_flag"), False),
                "parcel_extent_exceeds_tile_flag": _as_bool(first_row.get("parcel_extent_exceeds_tile_flag"), False),
                "parcel_tile_low_coverage_flag": _as_bool(first_row.get("parcel_tile_low_coverage_flag"), False),
                "multi_tile_candidate_flag": _as_bool(first_row.get("multi_tile_candidate_flag"), False),
                "parcel_covering_tile_count": int(_as_float(first_row.get("parcel_covering_tile_count"), 0)),
                "polygon_part_count": int(_as_float(first_row.get("polygon_part_count"), 0)),
                "clipped_polygon_part_count": int(_as_float(first_row.get("clipped_polygon_part_count"), 0)),
                "geometry_quality_flag": first_row.get("geometry_quality_flag"),
                "geometry_marketability_flag": first_row.get("geometry_marketability_flag"),
                "geometry_marketability_action": first_row.get("geometry_marketability_action"),
                "geometry_marketability_context": first_row.get("geometry_marketability_context"),
                "area_acres": _as_float(first_row.get("area_acres"), np.nan),
                "aspect_ratio": _as_float(first_row.get("aspect_ratio"), np.nan),
                "compactness": _as_float(first_row.get("compactness"), np.nan),
                "nearby_building_density": _as_float(first_row.get("nearby_building_density"), np.nan),
                "shape_compactness": _as_float(first_row.get("shape_compactness"), np.nan),
                "parcel_width_ft_estimate": _as_float(first_row.get("parcel_width_ft_estimate"), np.nan),
                "parcel_aspect_ratio_estimate": _as_float(first_row.get("parcel_aspect_ratio_estimate"), np.nan),
                "imagery_driveway_signal_max": _series_max(parcel_group, "imagery_driveway_signal"),
                "imagery_clearing_signal_max": _series_max(parcel_group, "imagery_clearing_signal"),
                "green_excess_max": _series_max(parcel_group, "green_excess"),
                "dark_shadow_pct_max": _series_max(parcel_group, "dark_shadow_pct"),
                "roof_tone_pct_max": _series_max(parcel_group, "roof_tone_pct"),
            }
        )
    return pd.DataFrame(parcel_records)


def _error_examples(parcel_eval: pd.DataFrame, *, threshold_column: str) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    example_specs = {
        "true_positive": parcel_eval[parcel_eval[threshold_column] & parcel_eval["structure_present_target"].eq(1)],
        "false_positive": parcel_eval[parcel_eval[threshold_column] & parcel_eval["structure_present_target"].eq(0)],
        "false_negative": parcel_eval[~parcel_eval[threshold_column] & parcel_eval["structure_present_target"].eq(1)],
    }
    output_columns = [
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "manual_training_label",
        "manual_review_confidence",
        "predicted_probability",
        "predicted_confidence",
        "best_tile_label",
        "best_tile_crop_label",
        "best_tile_parcel_coverage_pct",
        "masked_parcel_tile_path",
        "masked_parcel_core_crop_path",
        "masked_parcel_focus_crop_path",
        "manual_review_notes",
    ]
    for key, frame in example_specs.items():
        if frame.empty:
            examples[key] = []
            continue
        sorted_frame = frame.sort_values(
            ["predicted_probability", "predicted_confidence"],
            ascending=[False, False],
        )
        examples[key] = _json_safe(sorted_frame.loc[:, output_columns].head(3).to_dict(orient="records"))
    return examples


def _threshold_grid(parcel_eval: pd.DataFrame, thresholds: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _threshold_metrics(parcel_eval["structure_present_target"], parcel_eval["predicted_probability"], threshold)
        confusion = metrics["confusion_matrix"]
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f1 = float(round((2.0 * precision * recall / (precision + recall)), 4)) if (precision + recall) else 0.0
        rows.append(
            {
                **metrics,
                "f1": f1,
                "parcel_count": int(len(parcel_eval)),
                "false_positive": int(confusion["false_positive"]),
                "false_negative": int(confusion["false_negative"]),
            }
        )
    return rows


def _recommended_threshold(threshold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [
        row for row in threshold_rows if float(row["precision"]) >= 0.75 and float(row["recall"]) >= 0.6
    ]
    candidate_rows = preferred or threshold_rows
    best_row = max(candidate_rows, key=lambda row: (float(row["f1"]), float(row["precision"]), float(row["accuracy"])))
    rationale = (
        "Selected the highest-F1 threshold among thresholds that keep precision >= 0.75 and recall >= 0.6."
        if preferred
        else "Selected the highest-F1 threshold because no tested threshold met the preferred precision/recall floor."
    )
    return {
        "recommended_threshold": float(best_row["threshold"]),
        "recommended_metrics": _json_safe(best_row),
        "rationale": rationale,
    }


def _classify_likely_error_cause(row: pd.Series) -> tuple[str, str]:
    text_blob = _text_blob(row)
    if "neighbor" in text_blob or "outside parcel" in text_blob or "outside boundary" in text_blob:
        return "neighbor structure confusion", "Manual notes or review hint indicate a nearby structure outside the parcel."
    if "road" in text_blob or "clearing" in text_blob or "driveway" in text_blob:
        return "road/clearing confusion", "Review hint or notes indicate road, clearing, or driveway context."
    if any(token in text_blob for token in ["unclear", "ambiguous", "possible", "maybe", "unsure"]):
        return "label ambiguity", "Reviewer notes indicate ambiguity."

    manual_confidence = str(row.get("manual_review_confidence_normalized") or "").lower()
    if manual_confidence and manual_confidence != "high":
        return "label ambiguity", "Manual review confidence is below high."

    if (
        _as_float(row.get("green_excess_max"), 0.0) >= 0.18
        or _as_float(row.get("dark_shadow_pct_max"), 0.0) >= 0.22
        or "tree" in text_blob
        or "vegetation" in text_blob
        or "occluded" in text_blob
    ):
        return "tree cover / occlusion", "Vegetation or shadow signals suggest the structure is obscured."

    if (
        bool(row.get("parcel_tile_low_coverage_flag", False))
        or not bool(row.get("full_parcel_visible_flag", False))
        or bool(row.get("multi_tile_candidate_flag", False))
        or _as_float(row.get("best_tile_parcel_coverage_pct"), 100.0) < 25.0
    ):
        return "crop/context issue", "Parcel coverage is limited or the parcel spans multiple tiles."

    if (
        int(_as_float(row.get("polygon_part_count"), 0)) > 1
        or int(_as_float(row.get("clipped_polygon_part_count"), 0)) > 1
        or "boundary" in text_blob
    ):
        return "parcel boundary ambiguity", "Multipart geometry or boundary language suggests parcel-boundary ambiguity."

    if (
        _as_float(row.get("imagery_driveway_signal_max"), 0.0) >= 55.0
        or _as_float(row.get("imagery_clearing_signal_max"), 0.0) >= 25.0
    ):
        return "road/clearing confusion", "Imagery context signals are dominated by driveway or clearing cues."

    if (
        _as_float(row.get("parcel_width_ft_estimate"), np.nan) < 35.0
        or _as_float(row.get("best_tile_parcel_coverage_pct"), 100.0) < 35.0
        or _as_float(row.get("area_acres"), np.nan) < 0.35
    ):
        return "tiny or partial structure", "The parcel or visible structure footprint is small or only partially visible."

    return "other", "No dominant error-cause heuristic matched the parcel context."


def build_error_analysis_frame(parcel_eval: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis = parcel_eval.copy()
    for threshold in THRESHOLD_GUIDANCE_VALUES:
        suffix = f"{int(round(threshold * 100)):03d}"
        predicted_column = f"predicted_class_{suffix}"
        error_column = f"error_type_{suffix}"
        analysis[predicted_column] = analysis["predicted_probability"].astype(float).ge(threshold)
        analysis[error_column] = [
            _bool_error_type(target, predicted)
            for target, predicted in zip(analysis["structure_present_target"], analysis[predicted_column], strict=False)
        ]

    error_union_mask = analysis["error_type_050"].ne("correct") | analysis["error_type_082"].ne("correct")
    error_frame = analysis.loc[error_union_mask].copy()
    likely_causes = error_frame.apply(_classify_likely_error_cause, axis=1)
    error_frame["likely_error_cause"] = likely_causes.apply(lambda item: item[0])
    error_frame["likely_error_cause_reason"] = likely_causes.apply(lambda item: item[1])

    export_columns = [
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "manual_training_label",
        "manual_review_confidence",
        "review_hint",
        "manual_review_notes",
        "predicted_probability",
        "predicted_confidence",
        "predicted_class_050",
        "predicted_class_060",
        "predicted_class_070",
        "predicted_class_082",
        "error_type_050",
        "error_type_082",
        "masked_parcel_tile_path",
        "masked_parcel_core_crop_path",
        "masked_parcel_focus_crop_path",
        "best_tile_label",
        "best_tile_crop_label",
        "best_tile_parcel_coverage_pct",
        "full_parcel_visible_flag",
        "parcel_tile_low_coverage_flag",
        "multi_tile_candidate_flag",
        "polygon_part_count",
        "clipped_polygon_part_count",
        "geometry_quality_flag",
        "geometry_marketability_flag",
        "geometry_marketability_action",
        "area_acres",
        "aspect_ratio",
        "compactness",
        "nearby_building_density",
        "parcel_width_ft_estimate",
        "parcel_aspect_ratio_estimate",
        "imagery_driveway_signal_max",
        "imagery_clearing_signal_max",
        "green_excess_max",
        "dark_shadow_pct_max",
        "likely_error_cause",
        "likely_error_cause_reason",
    ]
    error_frame = error_frame.loc[:, export_columns].copy()

    summary = {
        "error_row_count": int(len(error_frame)),
        "error_counts_by_threshold": {
            "threshold_050": {
                "false_positive": int(error_frame["error_type_050"].eq("false_positive").sum()),
                "false_negative": int(error_frame["error_type_050"].eq("false_negative").sum()),
            },
            "threshold_082": {
                "false_positive": int(error_frame["error_type_082"].eq("false_positive").sum()),
                "false_negative": int(error_frame["error_type_082"].eq("false_negative").sum()),
            },
        },
        "likely_error_cause_counts": {
            str(key): int(value) for key, value in error_frame["likely_error_cause"].value_counts().items()
        },
        "likely_error_cause_by_threshold_050": (
            error_frame.groupby(["error_type_050", "likely_error_cause"], dropna=False)
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
        ),
        "example_error_rows": _json_safe(error_frame.head(10).to_dict(orient="records")),
    }
    return error_frame, _json_safe(summary)


def evaluate_pilot_manifest(
    *,
    pilot_manifest: pd.DataFrame,
    feature_columns_used: list[str],
    random_state: int,
    cv_splits: int,
    output_model_path: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    parcel_frame = pilot_manifest[["parcel_row_id", "structure_present_target"]].drop_duplicates().reset_index(drop=True)
    positive_count = int(parcel_frame["structure_present_target"].eq(1).sum())
    negative_count = int(parcel_frame["structure_present_target"].eq(0).sum())
    split_count = min(cv_splits, positive_count, negative_count)
    if split_count < 2:
        raise ValueError("Pilot dataset needs at least two parcels in each class for cross-validation.")

    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=random_state)
    scored_parts: list[pd.DataFrame] = []
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(parcel_frame[["parcel_row_id"]], parcel_frame["structure_present_target"]),
        start=1,
    ):
        train_parcels = set(parcel_frame.iloc[train_index]["parcel_row_id"].astype("string"))
        test_parcels = set(parcel_frame.iloc[test_index]["parcel_row_id"].astype("string"))
        train_frame = pilot_manifest.loc[pilot_manifest["parcel_row_id"].astype("string").isin(train_parcels)].copy()
        test_frame = pilot_manifest.loc[pilot_manifest["parcel_row_id"].astype("string").isin(test_parcels)].copy()

        x_train = _numeric_feature_frame(train_frame, feature_columns_used)
        y_train = pd.to_numeric(train_frame["structure_present_target"], errors="coerce").astype(int)
        x_test = _numeric_feature_frame(test_frame, feature_columns_used)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state + fold_index)),
            ]
        )
        pipeline.fit(x_train, y_train)
        fold_scored = test_frame.copy()
        fold_scored["predicted_probability"] = pipeline.predict_proba(x_test)[:, 1]
        fold_scored["cv_fold"] = fold_index
        scored_parts.append(fold_scored)

    scored_rows = pd.concat(scored_parts, ignore_index=True)
    parcel_eval = _aggregate_scored_rows_to_parcels(scored_rows)
    threshold_rows = _threshold_grid(parcel_eval, THRESHOLD_GUIDANCE_VALUES)
    threshold_lookup = {float(row["threshold"]): row for row in threshold_rows}
    for threshold in THRESHOLD_GUIDANCE_VALUES:
        suffix = f"{int(round(threshold * 100)):03d}"
        parcel_eval[f"predicted_class_{suffix}"] = parcel_eval["predicted_probability"].astype(float).ge(threshold)
    evaluation = {
        "evaluation_method": f"{split_count}-fold stratified parcel cross_validation",
        "cv_splits": int(split_count),
        "parcel_count_evaluated": int(len(parcel_eval)),
        "row_count_evaluated": int(len(scored_rows)),
        "parcel_level_auc": float(
            round(roc_auc_score(parcel_eval["structure_present_target"], parcel_eval["predicted_probability"]), 4)
        ),
        "parcel_level_threshold_050": _json_safe(threshold_lookup[0.50]),
        "parcel_level_threshold_082": _json_safe(threshold_lookup[0.82]),
        "threshold_guidance": {
            "tested_thresholds": _json_safe(threshold_rows),
            **_recommended_threshold(threshold_rows),
        },
        "example_rows": _error_examples(parcel_eval, threshold_column="predicted_positive_050"),
    }

    if output_model_path is not None:
        full_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        )
        full_pipeline.fit(
            _numeric_feature_frame(pilot_manifest, feature_columns_used),
            pd.to_numeric(pilot_manifest["structure_present_target"], errors="coerce").astype(int),
        )
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": full_pipeline,
                "feature_columns": feature_columns_used,
                "model_version": f"{MODEL_VERSION}_reviewed50_pilot",
            },
            output_model_path,
        )
        evaluation["output_model_path"] = str(output_model_path)

    return _json_safe(evaluation), scored_rows, parcel_eval


def build_reviewed50_pilot(
    *,
    state_code: str,
    run_name: str,
    review_path: Path,
    feature_manifest_path: Path,
    output_manifest_path: Path,
    output_summary_path: Path,
    output_cv_predictions_path: Path,
    output_error_analysis_path: Path,
    output_error_summary_path: Path,
    output_model_path: Path | None,
    random_state: int,
    cv_splits: int,
) -> dict[str, Any]:
    state_definition = load_state_definition(state_code)
    review_frame = pd.read_csv(review_path)
    usable_review_frame, validation_summary = prepare_review_label_frame(review_frame)
    feature_manifest_frame = pd.read_parquet(feature_manifest_path)
    pilot_manifest, manifest_summary, feature_columns_used = build_pilot_training_manifest_frame(
        review_frame=usable_review_frame,
        feature_manifest_frame=feature_manifest_frame,
    )

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_manifest.to_parquet(output_manifest_path, index=False)

    evaluation_summary, _scored_rows, parcel_eval = evaluate_pilot_manifest(
        pilot_manifest=pilot_manifest,
        feature_columns_used=feature_columns_used,
        random_state=random_state,
        cv_splits=cv_splits,
        output_model_path=output_model_path,
    )
    parcel_eval = parcel_eval.sort_values(["predicted_probability", "parcel_row_id"], ascending=[False, True]).reset_index(drop=True)
    output_cv_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    parcel_eval.to_csv(output_cv_predictions_path, index=False)

    error_frame, error_summary = build_error_analysis_frame(parcel_eval)
    output_error_analysis_path.parent.mkdir(parents=True, exist_ok=True)
    error_frame.to_csv(output_error_analysis_path, index=False)
    write_metrics(output_error_summary_path, error_summary)

    summary = {
        "state_code": state_definition.state_code,
        "state_name": state_definition.state_name,
        "run_name": run_name,
        "review_source_path": str(review_path),
        "feature_manifest_path": str(feature_manifest_path),
        "output_manifest_path": str(output_manifest_path),
        "output_cv_predictions_path": str(output_cv_predictions_path),
        "output_error_analysis_path": str(output_error_analysis_path),
        "output_error_summary_path": str(output_error_summary_path),
        "model_version": f"{MODEL_VERSION}_{run_name}_pilot",
        **validation_summary,
        **manifest_summary,
        "training_evaluation": evaluation_summary,
        "error_analysis_overview": error_summary,
    }
    write_metrics(output_summary_path, _json_safe(summary))
    return _json_safe(summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a reviewed building-presence pilot manifest from state-configured manual labels."
    )
    parser.add_argument("--state-code", default=DEFAULT_STATE_CODE)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--review-input", default=str(DEFAULT_REVIEWED_PILOT_PATH))
    parser.add_argument("--feature-manifest", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--output-cv-predictions", default=None)
    parser.add_argument("--output-error-analysis", default=None)
    parser.add_argument("--output-error-summary", default=None)
    parser.add_argument("--output-model", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=DEFAULT_CV_SPLITS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    state_code = str(arguments.state_code).strip().lower()
    run_name = str(arguments.run_name).strip() or DEFAULT_RUN_NAME
    default_outputs = _derive_output_paths(
        Path(arguments.output_prefix) if arguments.output_prefix else None,
        state_code=state_code,
        run_name=run_name,
    )
    feature_manifest_path = (
        Path(arguments.feature_manifest)
        if arguments.feature_manifest
        else _default_feature_manifest_path(state_code, run_name=run_name)
    )
    result = build_reviewed50_pilot(
        state_code=state_code,
        run_name=run_name,
        review_path=Path(arguments.review_input),
        feature_manifest_path=feature_manifest_path,
        output_manifest_path=Path(arguments.output_manifest) if arguments.output_manifest else default_outputs["manifest"],
        output_summary_path=Path(arguments.output_summary) if arguments.output_summary else default_outputs["summary"],
        output_cv_predictions_path=Path(arguments.output_cv_predictions) if arguments.output_cv_predictions else default_outputs["cv_predictions"],
        output_error_analysis_path=Path(arguments.output_error_analysis) if arguments.output_error_analysis else default_outputs["error_analysis"],
        output_error_summary_path=Path(arguments.output_error_summary) if arguments.output_error_summary else default_outputs["error_summary"],
        output_model_path=Path(arguments.output_model) if arguments.output_model else default_outputs["model"],
        random_state=arguments.random_state,
        cv_splits=arguments.cv_splits,
    )
    print(json.dumps(result, indent=2))
