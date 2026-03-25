from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.mississippi_leads_service import _apply_parcel_improvement_fields_frame  # noqa: E402
from vacancy_ai_common import (  # noqa: E402
    APP_READY_PATH,
    PARCEL_MASTER_PATH,
    PREDICTIONS_PATH,
    load_app_ready_row_ids,
    load_candidate_frame,
)


DEFAULT_SUMMARY_OUTPUT = ROOT / "data" / "buildings_processed" / "ms_vacancy_evaluation_summary.csv"
DEFAULT_SHORTLIST_OUTPUT = ROOT / "data" / "buildings_processed" / "ms_vacancy_labeling_shortlist.csv"
DEFAULT_REVIEW_SAMPLE_PATH = ROOT / "data" / "buildings_processed" / "ms_vacancy_training_review_sample_300.csv"
DEFAULT_COUNTY_CAP = 8
DEFAULT_DISAGREEMENT_COUNT = 45
DEFAULT_MULTI_TILE_REVIEW_COUNT = 45
DEFAULT_HIGH_CONFIDENCE_SUSPICIOUS_COUNT = 30
DEFAULT_REVIEW_GRADE_COUNT = 30


def _frame_series(frame: pd.DataFrame, column: str, default: object = np.nan) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(default, index=frame.index)


def _normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _confidence_bucket(series: pd.Series) -> pd.Series:
    confidence = pd.to_numeric(series, errors="coerce")
    bucket = pd.Series("unknown", index=series.index, dtype="string")
    bucket.loc[confidence.lt(25)] = "<25"
    bucket.loc[confidence.ge(25) & confidence.lt(50)] = "25-49.9"
    bucket.loc[confidence.ge(50) & confidence.lt(82)] = "50-81.9"
    bucket.loc[confidence.ge(82) & confidence.lt(90)] = "82-89.9"
    bucket.loc[confidence.ge(90)] = "90+"
    return bucket


def load_evaluation_frame() -> pd.DataFrame:
    app_ready_ids = load_app_ready_row_ids()
    frame = load_candidate_frame()
    frame = frame.loc[frame["parcel_row_id"].astype("string").isin(app_ready_ids)].copy()

    predictions = pd.read_parquet(PREDICTIONS_PATH, engine="pyarrow")
    predictions["parcel_row_id"] = predictions["parcel_row_id"].astype("string")
    frame = frame.merge(predictions, on=["parcel_row_id", "county_name"], how="inner")

    parcel_frame = pd.read_parquet(
        PARCEL_MASTER_PATH,
        columns=["parcel_row_id", "parcel_id", "land_use_raw", "total_value"],
        engine="pyarrow",
    )
    parcel_frame["parcel_row_id"] = parcel_frame["parcel_row_id"].astype("string")
    frame = frame.merge(parcel_frame, on="parcel_row_id", how="left")

    if APP_READY_PATH.exists():
        available_columns = [
            column
            for column in ["parcel_row_id", "lead_score_total", "recommended_view_bucket", "nearby_building_density"]
            if column in pd.read_parquet(APP_READY_PATH, engine="pyarrow", columns=None).columns
        ]
        if available_columns:
            lead_frame = pd.read_parquet(APP_READY_PATH, columns=available_columns, engine="pyarrow")
            lead_frame["parcel_row_id"] = lead_frame["parcel_row_id"].astype("string")
            frame = frame.merge(lead_frame, on="parcel_row_id", how="left", suffixes=("", "_lead"))

    frame["assessed_total_value"] = pd.to_numeric(frame.get("total_value"), errors="coerce")
    frame["land_use"] = _normalize_string(frame.get("land_use_raw"), index=frame.index)
    frame["county_vacant_flag"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["ai_building_present_flag"] = _frame_series(frame, "ai_building_present_flag", pd.NA).astype("boolean")
    frame["building_present_confidence"] = pd.to_numeric(frame.get("building_present_confidence"), errors="coerce")
    frame["ai_building_present_probability"] = pd.to_numeric(frame.get("ai_building_present_probability"), errors="coerce")
    frame["multi_tile_inference_used_flag"] = _frame_series(frame, "multi_tile_inference_used_flag", False).fillna(False).astype(bool)
    frame["multi_tile_candidate_flag"] = _frame_series(frame, "multi_tile_candidate_flag", False).fillna(False).astype(bool)
    frame["parcel_tile_low_coverage_flag"] = _frame_series(frame, "parcel_tile_low_coverage_flag", False).fillna(False).astype(bool)
    frame["parcel_extent_exceeds_tile_flag"] = _frame_series(frame, "parcel_extent_exceeds_tile_flag", False).fillna(False).astype(bool)
    frame["tiles_scored_count"] = pd.to_numeric(frame.get("tiles_scored_count"), errors="coerce").fillna(0).astype(int)
    frame["tiles_with_building_signal_count"] = (
        pd.to_numeric(frame.get("tiles_with_building_signal_count"), errors="coerce").fillna(0).astype(int)
    )
    frame["parcel_covering_tile_count"] = pd.to_numeric(frame.get("parcel_covering_tile_count"), errors="coerce").fillna(0).astype(int)
    frame["best_tile_confidence"] = pd.to_numeric(frame.get("best_tile_confidence"), errors="coerce")
    frame["best_tile_probability"] = pd.to_numeric(frame.get("best_tile_probability"), errors="coerce")
    frame["best_tile_parcel_coverage_pct"] = pd.to_numeric(frame.get("best_tile_parcel_coverage_pct"), errors="coerce")
    frame["negative_tile_coverage_pct"] = pd.to_numeric(frame.get("negative_tile_coverage_pct"), errors="coerce")
    frame = _apply_parcel_improvement_fields_frame(frame)
    frame["building_present_confidence_bucket"] = _confidence_bucket(frame["building_present_confidence"])
    frame["inference_mode"] = pd.Series(
        np.where(frame["multi_tile_inference_used_flag"], "multi_tile", "single_tile"),
        index=frame.index,
        dtype="string",
    )
    return frame


def build_summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({"section": "overview", "bucket": "evaluation_row_count", "sub_bucket": "", "count": int(len(frame))})

    def add_counts(section: str, series: pd.Series, *, bucket_label: str = "bucket") -> None:
        counts = series.astype("string").fillna("unknown").value_counts(dropna=False)
        for bucket, count in counts.items():
            rows.append(
                {
                    "section": section,
                    "bucket": bucket,
                    "sub_bucket": "",
                    "count": int(count),
                }
            )

    add_counts("inference_mode", frame["inference_mode"])
    add_counts("parcel_improvement_status", frame["parcel_improvement_status"])
    add_counts("building_present_confidence_bucket", frame["building_present_confidence_bucket"])

    for row in frame.groupby(["inference_mode", "parcel_improvement_status"], dropna=False).size().reset_index(name="count").itertuples(index=False):
        rows.append(
            {
                "section": "inference_mode_by_status",
                "bucket": row.inference_mode,
                "sub_bucket": row.parcel_improvement_status,
                "count": int(row.count),
            }
        )

    for label, mask in {
        "parcel_tile_low_coverage_flag": frame["parcel_tile_low_coverage_flag"],
        "multi_tile_candidate_flag": frame["multi_tile_candidate_flag"],
        "parcel_extent_exceeds_tile_flag": frame["parcel_extent_exceeds_tile_flag"],
        "multi_tile_inference_used_flag": frame["multi_tile_inference_used_flag"],
    }.items():
        rows.append({"section": "coverage_flag", "bucket": label, "sub_bucket": "true", "count": int(mask.fillna(False).sum())})

    suspicious_no_structure = (
        frame["parcel_improvement_status"].eq("likely_improved")
        & frame["building_count"].le(0)
        & frame["building_area_total"].le(0)
    )
    suspicious_with_structure = (
        frame["parcel_improvement_status"].eq("likely_vacant")
        & (frame["building_count"].ge(1) | frame["building_area_total"].ge(400))
    )
    rows.append(
        {
            "section": "labeling_priority_pool",
            "bucket": "likely_improved_without_mapped_building",
            "sub_bucket": "",
            "count": int(suspicious_no_structure.sum()),
        }
    )
    rows.append(
        {
            "section": "labeling_priority_pool",
            "bucket": "likely_vacant_with_mapped_building",
            "sub_bucket": "",
            "count": int(suspicious_with_structure.sum()),
        }
    )
    rows.append(
        {
            "section": "labeling_priority_pool",
            "bucket": "needs_review_multi_tile_candidates",
            "sub_bucket": "",
            "count": int((frame["multi_tile_candidate_flag"] & frame["parcel_improvement_status"].eq("needs_review")).sum()),
        }
    )

    return pd.DataFrame(rows, columns=["section", "bucket", "sub_bucket", "count"])


def _ranked_diverse_subset(
    frame: pd.DataFrame,
    *,
    count: int,
    sort_columns: list[str],
    ascending: list[bool],
    county_cap: int,
    exclude_ids: set[str],
) -> pd.DataFrame:
    if count <= 0:
        return frame.head(0).copy()
    available = frame.loc[~frame["parcel_row_id"].astype("string").isin(pd.Index(sorted(exclude_ids), dtype="string"))].copy()
    if available.empty:
        return available
    available = available.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)

    selected_positions: list[int] = []
    county_counts: dict[str, int] = {}
    for position, row in available.iterrows():
        county = str(row.get("county_name") or "unknown")
        if county_counts.get(county, 0) >= county_cap:
            continue
        selected_positions.append(position)
        county_counts[county] = county_counts.get(county, 0) + 1
        if len(selected_positions) >= count:
            break

    if len(selected_positions) < count:
        remaining = [position for position in range(len(available)) if position not in set(selected_positions)]
        selected_positions.extend(remaining[: max(0, count - len(selected_positions))])

    return available.iloc[selected_positions].copy()


def build_shortlist(
    evaluation_frame: pd.DataFrame,
    review_sample_path: Path,
    *,
    disagreement_count: int,
    multi_tile_review_count: int,
    high_confidence_suspicious_count: int,
    review_grade_count: int,
    county_cap: int,
) -> pd.DataFrame:
    review = pd.read_csv(review_sample_path)
    review["parcel_row_id"] = review["parcel_row_id"].astype("string")
    available = review.merge(
        evaluation_frame.loc[
            :,
            [
                "parcel_row_id",
                "parcel_improvement_status",
                "parcel_improvement_confidence",
                "parcel_improvement_reason",
                "building_count",
                "building_area_total",
                "multi_tile_inference_used_flag",
                "multi_tile_candidate_flag",
                "parcel_tile_low_coverage_flag",
                "parcel_extent_exceeds_tile_flag",
                "tiles_scored_count",
                "tiles_with_building_signal_count",
                "building_present_confidence",
                "ai_building_present_probability",
                "best_tile_confidence",
                "best_tile_probability",
                "best_tile_parcel_coverage_pct",
                "negative_tile_coverage_pct",
            ],
        ],
        on="parcel_row_id",
        how="left",
        suffixes=("", "_eval"),
    )

    selected_ids: set[str] = set()
    groups: list[pd.DataFrame] = []

    disagreement = available.loc[
        (
            available["parcel_improvement_status"].eq("likely_improved")
            & pd.to_numeric(available["building_count"], errors="coerce").fillna(0).le(0)
            & pd.to_numeric(available["building_area_total"], errors="coerce").fillna(0).le(0)
        )
        | (
            available["parcel_improvement_status"].eq("likely_vacant")
            & (
                pd.to_numeric(available["building_count"], errors="coerce").fillna(0).ge(1)
                | pd.to_numeric(available["building_area_total"], errors="coerce").fillna(0).ge(400)
            )
        )
    ].copy()
    disagreement["labeling_group"] = "disagreement"
    disagreement["labeling_reason"] = np.where(
        disagreement["parcel_improvement_status"].eq("likely_improved"),
        "likely_improved_without_mapped_building",
        "likely_vacant_with_mapped_building",
    )
    disagreement["labeling_priority_score"] = (
        pd.to_numeric(disagreement["parcel_improvement_confidence"], errors="coerce").fillna(0.0)
        + pd.to_numeric(disagreement["building_present_confidence"], errors="coerce").fillna(0.0) * 0.4
        + pd.to_numeric(disagreement["tiles_scored_count"], errors="coerce").fillna(0.0) * 0.3
    )
    disagreement_sample = _ranked_diverse_subset(
        disagreement,
        count=disagreement_count,
        sort_columns=["labeling_priority_score", "county_name", "parcel_row_id"],
        ascending=[False, True, True],
        county_cap=county_cap,
        exclude_ids=selected_ids,
    )
    groups.append(disagreement_sample)
    selected_ids.update(disagreement_sample["parcel_row_id"].astype(str).tolist())

    multi_tile_review = available.loc[
        available["multi_tile_candidate_flag"].fillna(False)
        & available["parcel_improvement_status"].eq("needs_review")
    ].copy()
    multi_tile_review["labeling_group"] = "multi_tile_review"
    multi_tile_review["labeling_reason"] = "needs_review_multi_tile_candidate"
    multi_tile_review["labeling_priority_score"] = (
        pd.to_numeric(multi_tile_review["tiles_scored_count"], errors="coerce").fillna(0.0) * 1.5
        + pd.to_numeric(multi_tile_review["parcel_covering_tile_count"], errors="coerce").fillna(0.0)
        + pd.to_numeric(multi_tile_review["building_present_confidence"], errors="coerce").sub(50.0).abs().fillna(0.0) * -0.2
    )
    multi_tile_review_sample = _ranked_diverse_subset(
        multi_tile_review,
        count=multi_tile_review_count,
        sort_columns=["labeling_priority_score", "county_name", "parcel_row_id"],
        ascending=[False, True, True],
        county_cap=county_cap,
        exclude_ids=selected_ids,
    )
    groups.append(multi_tile_review_sample)
    selected_ids.update(multi_tile_review_sample["parcel_row_id"].astype(str).tolist())

    high_confidence_suspicious = available.loc[
        (
            pd.to_numeric(available["building_present_confidence"], errors="coerce").fillna(0).ge(90)
            & pd.to_numeric(available["building_count"], errors="coerce").fillna(0).le(0)
            & pd.to_numeric(available["building_area_total"], errors="coerce").fillna(0).le(0)
        )
        | (
            pd.to_numeric(available["building_present_confidence"], errors="coerce").fillna(100).le(10)
            & (
                pd.to_numeric(available["building_count"], errors="coerce").fillna(0).ge(1)
                | pd.to_numeric(available["building_area_total"], errors="coerce").fillna(0).ge(400)
            )
        )
    ].copy()
    high_confidence_suspicious["labeling_group"] = "high_confidence_suspicious"
    high_confidence_suspicious["labeling_reason"] = "high_confidence_output_conflicts_with_structure_baseline"
    high_confidence_suspicious["labeling_priority_score"] = (
        pd.to_numeric(high_confidence_suspicious["building_present_confidence"], errors="coerce").sub(50.0).abs().fillna(0.0)
        + pd.to_numeric(high_confidence_suspicious["best_tile_confidence"], errors="coerce").fillna(0.0) * 0.3
    )
    high_confidence_suspicious_sample = _ranked_diverse_subset(
        high_confidence_suspicious,
        count=high_confidence_suspicious_count,
        sort_columns=["labeling_priority_score", "county_name", "parcel_row_id"],
        ascending=[False, True, True],
        county_cap=county_cap,
        exclude_ids=selected_ids,
    )
    groups.append(high_confidence_suspicious_sample)
    selected_ids.update(high_confidence_suspicious_sample["parcel_row_id"].astype(str).tolist())

    review_grade = available.loc[
        available["parcel_improvement_status"].eq("needs_review")
        & pd.to_numeric(available["building_present_confidence"], errors="coerce").between(50.0, 81.9, inclusive="both")
    ].copy()
    review_grade["labeling_group"] = "review_grade"
    review_grade["labeling_reason"] = "mid_confidence_review_case"
    review_grade["labeling_priority_score"] = (
        pd.to_numeric(review_grade["building_present_confidence"], errors="coerce").sub(66.0).abs().fillna(0.0) * -1.0
        + np.where(review_grade["multi_tile_candidate_flag"].fillna(False), 12.0, 0.0)
        + np.where(review_grade["parcel_tile_low_coverage_flag"].fillna(False), 8.0, 0.0)
    )
    review_grade_sample = _ranked_diverse_subset(
        review_grade,
        count=review_grade_count,
        sort_columns=["labeling_priority_score", "county_name", "parcel_row_id"],
        ascending=[False, True, True],
        county_cap=county_cap,
        exclude_ids=selected_ids,
    )
    groups.append(review_grade_sample)

    shortlist = pd.concat(groups, ignore_index=True)
    shortlist["labeling_priority_score"] = pd.to_numeric(shortlist["labeling_priority_score"], errors="coerce").round(1)
    shortlist["manual_label"] = _normalize_string(shortlist.get("manual_training_label"), index=shortlist.index)
    shortlist["reviewer_notes"] = _normalize_string(shortlist.get("manual_notes"), index=shortlist.index)
    shortlist = shortlist.sort_values(
        ["labeling_group", "labeling_priority_score", "county_name", "parcel_row_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    output_columns = [
        "labeling_group",
        "labeling_reason",
        "labeling_priority_score",
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "parcel_improvement_status",
        "parcel_improvement_confidence",
        "parcel_improvement_reason",
        "ai_building_present_probability",
        "building_present_confidence",
        "ai_building_present_flag",
        "multi_tile_inference_used_flag",
        "multi_tile_candidate_flag",
        "parcel_tile_low_coverage_flag",
        "parcel_extent_exceeds_tile_flag",
        "tiles_scored_count",
        "tiles_with_building_signal_count",
        "best_tile_confidence",
        "best_tile_probability",
        "best_tile_parcel_coverage_pct",
        "negative_tile_coverage_pct",
        "image_url",
        "raw_centroid_tile_path",
        "masked_parcel_tile_path",
        "masked_parcel_core_crop_path",
        "masked_parcel_focus_crop_path",
        "manual_label",
        "reviewer_notes",
        "review_hint",
        "building_presence_reason",
        "manual_training_label",
        "manual_structure_inside_parcel",
        "manual_neighbor_structure_only",
        "manual_road_or_clearing_only",
        "manual_review_confidence",
        "manual_notes",
    ]
    available_columns = [column for column in output_columns if column in shortlist.columns]
    return shortlist.loc[:, available_columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mississippi vacancy evaluation summary and labeling shortlist artifacts.")
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--shortlist-output", default=str(DEFAULT_SHORTLIST_OUTPUT))
    parser.add_argument("--review-sample", default=str(DEFAULT_REVIEW_SAMPLE_PATH))
    parser.add_argument("--county-cap", type=int, default=DEFAULT_COUNTY_CAP)
    parser.add_argument("--disagreement-count", type=int, default=DEFAULT_DISAGREEMENT_COUNT)
    parser.add_argument("--multi-tile-review-count", type=int, default=DEFAULT_MULTI_TILE_REVIEW_COUNT)
    parser.add_argument("--high-confidence-suspicious-count", type=int, default=DEFAULT_HIGH_CONFIDENCE_SUSPICIOUS_COUNT)
    parser.add_argument("--review-grade-count", type=int, default=DEFAULT_REVIEW_GRADE_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_frame = load_evaluation_frame()
    summary = build_summary_rows(evaluation_frame)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    shortlist = build_shortlist(
        evaluation_frame,
        Path(args.review_sample),
        disagreement_count=args.disagreement_count,
        multi_tile_review_count=args.multi_tile_review_count,
        high_confidence_suspicious_count=args.high_confidence_suspicious_count,
        review_grade_count=args.review_grade_count,
        county_cap=args.county_cap,
    )
    shortlist_path = Path(args.shortlist_output)
    shortlist_path.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(shortlist_path, index=False)

    print(f"Wrote {len(evaluation_frame)} evaluation rows to summary source frame.")
    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    print(f"Wrote {len(shortlist)} shortlist rows to {shortlist_path}")


if __name__ == "__main__":
    main()
