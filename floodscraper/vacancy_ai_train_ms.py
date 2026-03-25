from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - fallback for older sklearn
    StratifiedGroupKFold = None

from vacancy_ai_common import (
    DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD,
    DEFAULT_TILE_NEGATIVE_CONFIDENCE_THRESHOLD,
    MODEL_METRICS_PATH,
    MODEL_PARAMS_PATH,
    MODEL_PATH,
    MODEL_VERSION,
    TRAINING_MANIFEST_PATH,
    aggregate_parcel_tile_predictions,
    building_present_confidence_from_probability,
    feature_columns,
    write_metrics,
)


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _split_manifest_by_parcel(
    frame: pd.DataFrame,
    *,
    label_column: str,
    group_column: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parcel_frame = frame[[group_column, label_column]].drop_duplicates().reset_index(drop=True)
    if parcel_frame.empty:
        raise ValueError("Training manifest is empty.")
    if parcel_frame[label_column].nunique() < 2:
        raise ValueError("Training manifest needs at least two label classes.")

    if StratifiedGroupKFold is not None and len(parcel_frame) >= 10:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        train_index, test_index = next(
            splitter.split(
                X=parcel_frame[[group_column]],
                y=parcel_frame[label_column],
                groups=parcel_frame[group_column],
            )
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_index, test_index = next(
            splitter.split(
                X=parcel_frame[[group_column]],
                y=parcel_frame[label_column],
                groups=parcel_frame[group_column],
            )
        )

    train_groups = set(parcel_frame.iloc[train_index][group_column].astype("string"))
    test_groups = set(parcel_frame.iloc[test_index][group_column].astype("string"))
    train_frame = frame.loc[frame[group_column].astype("string").isin(train_groups)].copy()
    test_frame = frame.loc[frame[group_column].astype("string").isin(test_groups)].copy()
    return train_frame, test_frame


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return default
    if not np.isfinite(numeric):
        return default
    return numeric


def _aggregate_test_rows_to_parcels(scored_rows: pd.DataFrame) -> pd.DataFrame:
    parcel_records: list[dict[str, Any]] = []
    for parcel_row_id, parcel_group in scored_rows.groupby("parcel_row_id", sort=False):
        tile_predictions: list[dict[str, Any]] = []
        for tile_label, tile_group in parcel_group.groupby("tile_label", sort=False):
            best_index = tile_group["predicted_probability"].astype(float).idxmax()
            best_row = tile_group.loc[best_index]
            probability = _as_float(best_row["predicted_probability"])
            confidence = building_present_confidence_from_probability(probability)
            crop_label = str(best_row.get("imagery_crop_label") or "")
            crop_coverage_ratio = _as_float(best_row.get("crop_parcel_coverage_ratio"))
            tile_predictions.append(
                {
                    "tile_label": str(tile_label),
                    "best_crop_label": crop_label,
                    "probability": probability,
                    "building_present_confidence": confidence,
                    "tile_parcel_coverage_ratio": _as_float(best_row.get("parcel_tile_coverage_ratio")),
                    "parcel_coverage_ratio": crop_coverage_ratio,
                    "tile_building_signal_flag": bool(
                        confidence >= DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD
                        and (
                            crop_label == "parcel_core"
                            or crop_coverage_ratio >= 0.35
                            or probability >= 0.90
                        )
                    ),
                    "tile_negative_signal_flag": bool(
                        confidence <= DEFAULT_TILE_NEGATIVE_CONFIDENCE_THRESHOLD
                    ),
                }
            )

        aggregation = aggregate_parcel_tile_predictions(tile_predictions)
        first_row = parcel_group.iloc[0]
        probability = _as_float(aggregation["probability"])
        weak_label = int(first_row["weak_building_label"])
        total_value = _as_float(first_row.get("total_value"), np.nan)
        improvement_value_total = _as_float(first_row.get("improvement_value_1"), 0.0) + _as_float(
            first_row.get("improvement_value_2"),
            0.0,
        )
        parcel_records.append(
            {
                "parcel_row_id": str(parcel_row_id),
                "parcel_id": first_row.get("parcel_id"),
                "county_name": first_row.get("county_name"),
                "weak_building_label": weak_label,
                "predicted_probability": probability,
                "predicted_confidence": _as_float(aggregation["building_present_confidence"]),
                "predicted_positive_050": bool(probability >= 0.50),
                "predicted_positive_082": bool(aggregation["ai_building_present_flag"]),
                "building_count": _as_float(first_row.get("building_count")),
                "building_area_total": _as_float(first_row.get("building_area_total")),
                "parcel_vacant_flag": bool(first_row.get("parcel_vacant_flag")),
                "total_value": total_value,
                "improvement_value_total": improvement_value_total,
                "label_reliability_tier": first_row.get("label_reliability_tier"),
                "full_parcel_visible_flag": bool(first_row.get("full_parcel_visible_flag")),
                "parcel_extent_exceeds_tile_flag": bool(first_row.get("parcel_extent_exceeds_tile_flag")),
                "parcel_tile_low_coverage_flag": bool(first_row.get("parcel_tile_low_coverage_flag")),
                "multi_tile_candidate_flag": bool(first_row.get("multi_tile_candidate_flag")),
                "polygon_part_count": int(_as_float(first_row.get("polygon_part_count"))),
                "clipped_polygon_part_count": int(_as_float(first_row.get("clipped_polygon_part_count"))),
                "road_context_like_flag": bool(
                    parcel_group["imagery_driveway_signal"].fillna(0).ge(55.0).any()
                    or parcel_group["imagery_clearing_signal"].fillna(0).ge(22.0).any()
                ),
                "tile_labels_scored": json.dumps([str(item["tile_label"]) for item in tile_predictions]),
                "tiles_scored_count": int(aggregation["tiles_scored_count"]),
                "tiles_with_building_signal_count": int(aggregation["tiles_with_building_signal_count"]),
                "best_tile_label": aggregation["best_tile_label"],
                "best_tile_crop_label": aggregation["best_tile_crop_label"],
                "best_tile_probability": _as_float(aggregation["best_tile_probability"], np.nan),
                "best_tile_parcel_coverage_pct": _as_float(aggregation["best_tile_parcel_coverage_pct"], np.nan),
                "negative_tile_coverage_pct": _as_float(aggregation["negative_tile_coverage_pct"], np.nan),
                "footprint_value_disagreement_flag": bool(
                    (weak_label == 0 and np.isfinite(total_value) and total_value >= 5000.0)
                    or (weak_label == 1 and np.isfinite(total_value) and total_value <= 1000.0)
                ),
            }
        )
    return pd.DataFrame(parcel_records)


def _threshold_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    y_true = frame["weak_building_label"].astype(int)
    y_pred = frame[prediction_column].astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return {
        "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
        "specificity": float(round(specificity, 4)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def _calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.000001, 0.25, 0.50, 0.82, 0.90, 1.000001]
    labels = ["<0.25", "0.25-0.49", "0.50-0.81", "0.82-0.89", "0.90+"]
    calibration = frame.copy()
    calibration["confidence_bucket"] = pd.cut(
        calibration["predicted_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    summary = (
        calibration.groupby("confidence_bucket", observed=False)
        .agg(
            parcel_count=("parcel_row_id", "count"),
            avg_predicted_probability=("predicted_probability", "mean"),
            observed_positive_rate=("weak_building_label", "mean"),
        )
        .reset_index()
    )
    summary["avg_predicted_probability"] = summary["avg_predicted_probability"].round(4)
    summary["observed_positive_rate"] = summary["observed_positive_rate"].round(4)
    return summary


def _slice_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    slice_definitions = {
        "all": pd.Series(True, index=frame.index),
        "full_parcel_visible": frame["full_parcel_visible_flag"].fillna(False),
        "low_coverage": frame["parcel_tile_low_coverage_flag"].fillna(False),
        "multi_tile_candidate": frame["multi_tile_candidate_flag"].fillna(False),
        "multipart": frame["polygon_part_count"].fillna(0).gt(1),
        "clipped_multipart": frame["clipped_polygon_part_count"].fillna(0).gt(1),
        "road_context_like": frame["road_context_like_flag"].fillna(False),
        "footprint_value_disagreement": frame["footprint_value_disagreement_flag"].fillna(False),
    }
    rows: list[dict[str, Any]] = []
    for slice_name, mask in slice_definitions.items():
        subset = frame.loc[mask].copy()
        if subset.empty or subset["weak_building_label"].nunique() < 2:
            rows.append({"slice_name": slice_name, "parcel_count": int(len(subset))})
            continue
        threshold_metrics = _threshold_metrics(subset, "predicted_positive_082")
        rows.append(
            {
                "slice_name": slice_name,
                "parcel_count": int(len(subset)),
                "positive_rate": float(round(subset["weak_building_label"].mean(), 4)),
                "avg_probability": float(round(subset["predicted_probability"].mean(), 4)),
                "roc_auc": float(round(roc_auc_score(subset["weak_building_label"], subset["predicted_probability"]), 4)),
                **threshold_metrics,
            }
        )
    return pd.DataFrame(rows)


def _error_examples(frame: pd.DataFrame) -> pd.DataFrame:
    false_positives = frame.loc[
        frame["predicted_positive_082"] & frame["weak_building_label"].eq(0)
    ].copy()
    false_positives["error_type"] = "false_positive"
    false_negatives = frame.loc[
        ~frame["predicted_positive_082"] & frame["weak_building_label"].eq(1)
    ].copy()
    false_negatives["error_type"] = "false_negative"
    errors = pd.concat([false_positives, false_negatives], ignore_index=True)
    if errors.empty:
        return errors
    errors = errors.sort_values(
        ["error_type", "predicted_confidence", "predicted_probability"],
        ascending=[True, False, False],
    )
    columns = [
        "error_type",
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "weak_building_label",
        "predicted_probability",
        "predicted_confidence",
        "building_count",
        "building_area_total",
        "total_value",
        "improvement_value_total",
        "full_parcel_visible_flag",
        "parcel_tile_low_coverage_flag",
        "multi_tile_candidate_flag",
        "polygon_part_count",
        "road_context_like_flag",
        "footprint_value_disagreement_flag",
        "tile_labels_scored",
        "best_tile_label",
        "best_tile_crop_label",
        "best_tile_probability",
        "best_tile_parcel_coverage_pct",
    ]
    return errors.loc[:, columns].head(100).reset_index(drop=True)


def _default_output_path(base_path: str, suffix: str, extension: str) -> str:
    path = Path(base_path)
    return str(path.with_name(f"{path.stem}{suffix}{extension}"))


def train_model(
    *,
    manifest_path: str,
    output_model: str,
    output_metrics: str,
    output_params: str,
    output_calibration: str,
    output_slices: str,
    output_errors: str,
    random_state: int,
) -> None:
    frame = pd.read_parquet(manifest_path)
    columns = feature_columns(frame)

    train_frame, test_frame = _split_manifest_by_parcel(
        frame,
        label_column="weak_building_label",
        group_column="parcel_row_id",
        random_state=random_state,
    )
    x_train = _numeric_feature_frame(train_frame, columns)
    y_train = pd.to_numeric(train_frame["weak_building_label"], errors="coerce").astype(int)
    x_test = _numeric_feature_frame(test_frame, columns)
    y_test = pd.to_numeric(test_frame["weak_building_label"], errors="coerce").astype(int)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]
    )
    pipeline.fit(x_train, y_train)

    crop_probability = pipeline.predict_proba(x_test)[:, 1]
    crop_predicted = (crop_probability >= 0.5).astype(int)
    scored_test = test_frame.copy()
    scored_test["predicted_probability"] = crop_probability
    parcel_eval = _aggregate_test_rows_to_parcels(scored_test)
    parcel_threshold_050 = _threshold_metrics(parcel_eval, "predicted_positive_050")
    parcel_threshold_082 = _threshold_metrics(parcel_eval, "predicted_positive_082")
    calibration = _calibration_table(parcel_eval)
    slice_metrics = _slice_metrics(parcel_eval)
    error_examples = _error_examples(parcel_eval)

    metrics = {
        "model_version": MODEL_VERSION,
        "manifest_path": manifest_path,
        "feature_count": len(columns),
        "feature_columns": columns,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_parcels": int(train_frame["parcel_row_id"].astype("string").nunique()),
        "test_parcels": int(test_frame["parcel_row_id"].astype("string").nunique()),
        "train_class_balance_rows": y_train.value_counts().sort_index().to_dict(),
        "test_class_balance_rows": y_test.value_counts().sort_index().to_dict(),
        "crop_level_metrics": {
            "accuracy": float(round(accuracy_score(y_test, crop_predicted), 4)),
            "precision": float(round(precision_score(y_test, crop_predicted, zero_division=0), 4)),
            "recall": float(round(recall_score(y_test, crop_predicted, zero_division=0), 4)),
            "roc_auc": float(round(roc_auc_score(y_test, crop_probability), 4)),
            "brier": float(round(brier_score_loss(y_test, crop_probability), 4)),
        },
        "parcel_level_metrics": {
            "roc_auc": float(round(roc_auc_score(parcel_eval["weak_building_label"], parcel_eval["predicted_probability"]), 4)),
            "brier": float(round(brier_score_loss(parcel_eval["weak_building_label"], parcel_eval["predicted_probability"]), 4)),
            "threshold_050": parcel_threshold_050,
            "threshold_082": parcel_threshold_082,
        },
        "evaluation_slices_path": output_slices,
        "evaluation_calibration_path": output_calibration,
        "evaluation_error_examples_path": output_errors,
    }
    joblib.dump({"pipeline": pipeline, "feature_columns": columns, "model_version": MODEL_VERSION}, output_model)
    write_metrics(Path(output_metrics), metrics)

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    Path(output_params).parent.mkdir(parents=True, exist_ok=True)
    Path(output_params).write_text(
        pd.Series(
            {
                "model_version": MODEL_VERSION,
                "feature_columns": columns,
                "scaler_mean": [float(value) for value in scaler.mean_],
                "scaler_scale": [float(value) for value in scaler.scale_],
                "coef": [float(value) for value in model.coef_[0]],
                "intercept": float(model.intercept_[0]),
                "classification_threshold": 0.5,
                "operational_building_confidence_threshold": DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD,
            }
        ).to_json(),
        encoding="utf-8",
    )
    Path(output_calibration).parent.mkdir(parents=True, exist_ok=True)
    Path(output_slices).parent.mkdir(parents=True, exist_ok=True)
    Path(output_errors).parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_calibration, index=False)
    slice_metrics.to_csv(output_slices, index=False)
    error_examples.to_csv(output_errors, index=False)
    print(f"Trained {MODEL_VERSION} with {len(columns)} features using manifest {manifest_path}")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a Mississippi building-presence classifier from the current parcel-aware weak-label manifest.")
    parser.add_argument("--manifest", default=str(TRAINING_MANIFEST_PATH))
    parser.add_argument("--output-model", default=str(MODEL_PATH))
    parser.add_argument("--output-metrics", default=str(MODEL_METRICS_PATH))
    parser.add_argument("--output-params", default=str(MODEL_PARAMS_PATH))
    parser.add_argument("--output-calibration", default=None)
    parser.add_argument("--output-slices", default=None)
    parser.add_argument("--output-errors", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        manifest_path=args.manifest,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
        output_params=args.output_params,
        output_calibration=args.output_calibration or _default_output_path(args.output_metrics, "_calibration", ".csv"),
        output_slices=args.output_slices or _default_output_path(args.output_metrics, "_slices", ".csv"),
        output_errors=args.output_errors or _default_output_path(args.output_metrics, "_errors", ".csv"),
        random_state=args.random_state,
    )
