from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHORTLIST_PATH = ROOT / "data" / "buildings_processed" / "ms_vacancy_labeling_shortlist.csv"
DEFAULT_SUMMARY_OUTPUT = ROOT / "data" / "buildings_processed" / "ms_vacancy_labeling_shortlist_evaluation_summary.csv"
DEFAULT_CONFUSION_OUTPUT = ROOT / "data" / "buildings_processed" / "ms_vacancy_labeling_shortlist_confusion_matrix.csv"
DEFAULT_REASON_OUTPUT = ROOT / "data" / "buildings_processed" / "ms_vacancy_labeling_shortlist_reason_breakdown.csv"
DEFAULT_DISAGREEMENT_OUTPUT = ROOT / "data" / "buildings_processed" / "ms_vacancy_labeling_shortlist_disagreement_analysis.csv"


def _normalize_label(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    label_map = {
        "improved": "improved",
        "likely_improved": "improved",
        "structure": "improved",
        "building": "improved",
        "vacant": "vacant",
        "likely_vacant": "vacant",
        "no_structure": "vacant",
        "needs_review": "needs_review",
        "review": "needs_review",
        "unclear": "needs_review",
        "unsure": "needs_review",
    }
    return label_map.get(text)


def _predicted_label(series: pd.Series) -> pd.Series:
    status = series.astype("string").str.strip().str.lower()
    mapped = status.map(
        {
            "likely_improved": "improved",
            "likely_vacant": "vacant",
            "needs_review": "needs_review",
        }
    )
    return mapped.astype("string")


def load_labeled_shortlist(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
    if "manual_label" in frame.columns:
        manual_source = frame["manual_label"]
    elif "manual_training_label" in frame.columns:
        manual_source = frame["manual_training_label"]
    else:
        manual_source = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["manual_label_normalized"] = manual_source.map(_normalize_label).astype("string")
    frame["predicted_label"] = _predicted_label(frame["parcel_improvement_status"])
    frame["reviewer_notes"] = (
        frame["reviewer_notes"].astype("string")
        if "reviewer_notes" in frame.columns
        else frame.get("manual_notes", pd.Series(pd.NA, index=frame.index)).astype("string")
    )
    return frame


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.loc[frame["manual_label_normalized"].notna()].copy()
    rows: list[dict[str, object]] = []

    rows.append({"metric": "total_rows", "value": int(len(frame))})
    rows.append({"metric": "total_labeled", "value": int(len(labeled))})
    rows.append({"metric": "total_unlabeled", "value": int(len(frame) - len(labeled))})

    improved_predictions = labeled.loc[labeled["predicted_label"].eq("improved")]
    vacant_predictions = labeled.loc[labeled["predicted_label"].eq("vacant")]
    review_predictions = labeled.loc[labeled["predicted_label"].eq("needs_review")]

    improved_precision = (
        float(improved_predictions["manual_label_normalized"].eq("improved").mean())
        if not improved_predictions.empty
        else float("nan")
    )
    vacant_precision = (
        float(vacant_predictions["manual_label_normalized"].eq("vacant").mean())
        if not vacant_predictions.empty
        else float("nan")
    )

    rows.append({"metric": "improved_prediction_count_labeled", "value": int(len(improved_predictions))})
    rows.append({"metric": "vacant_prediction_count_labeled", "value": int(len(vacant_predictions))})
    rows.append({"metric": "needs_review_prediction_count_labeled", "value": int(len(review_predictions))})
    rows.append({"metric": "improved_precision", "value": round(improved_precision, 4) if pd.notna(improved_precision) else pd.NA})
    rows.append({"metric": "vacant_precision", "value": round(vacant_precision, 4) if pd.notna(vacant_precision) else pd.NA})
    rows.append(
        {
            "metric": "needs_review_manual_improved_count",
            "value": int((review_predictions["manual_label_normalized"].eq("improved")).sum()),
        }
    )
    rows.append(
        {
            "metric": "needs_review_manual_vacant_count",
            "value": int((review_predictions["manual_label_normalized"].eq("vacant")).sum()),
        }
    )

    return pd.DataFrame(rows)


def build_confusion_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.loc[frame["manual_label_normalized"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(columns=["predicted_label", "manual_label", "count"])
    confusion = (
        labeled.groupby(["predicted_label", "manual_label_normalized"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"manual_label_normalized": "manual_label"})
        .sort_values(["predicted_label", "manual_label"])
        .reset_index(drop=True)
    )
    return confusion


def build_reason_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.loc[frame["manual_label_normalized"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(columns=["labeling_reason", "predicted_label", "manual_label", "count"])
    return (
        labeled.groupby(["labeling_reason", "predicted_label", "manual_label_normalized"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"manual_label_normalized": "manual_label"})
        .sort_values(["labeling_reason", "predicted_label", "manual_label"])
        .reset_index(drop=True)
    )


def build_disagreement_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    labeled = frame.loc[frame["manual_label_normalized"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(
            columns=[
                "analysis_bucket",
                "parcel_row_id",
                "parcel_id",
                "county_name",
                "labeling_reason",
                "predicted_label",
                "manual_label",
                "building_present_confidence",
                "ai_building_present_probability",
                "tiles_scored_count",
                "reviewer_notes",
            ]
        )

    analysis_bucket = pd.Series(pd.NA, index=labeled.index, dtype="string")
    analysis_bucket.loc[
        labeled["labeling_reason"].eq("likely_improved_without_mapped_building")
        & labeled["predicted_label"].eq("improved")
        & labeled["manual_label_normalized"].eq("improved")
    ] = "ai_improved_no_mapped_building_human_improved"
    analysis_bucket.loc[
        labeled["labeling_reason"].eq("likely_improved_without_mapped_building")
        & labeled["predicted_label"].eq("improved")
        & labeled["manual_label_normalized"].eq("vacant")
    ] = "ai_improved_no_mapped_building_human_vacant"
    analysis_bucket.loc[
        labeled["predicted_label"].eq("needs_review")
        & labeled["manual_label_normalized"].eq("improved")
    ] = "needs_review_human_improved"
    analysis_bucket.loc[
        labeled["predicted_label"].eq("needs_review")
        & labeled["manual_label_normalized"].eq("vacant")
    ] = "needs_review_human_vacant"

    labeled["analysis_bucket"] = analysis_bucket
    analysis = labeled.loc[labeled["analysis_bucket"].notna()].copy()
    if analysis.empty:
        return pd.DataFrame(
            columns=[
                "analysis_bucket",
                "parcel_row_id",
                "parcel_id",
                "county_name",
                "labeling_reason",
                "predicted_label",
                "manual_label",
                "building_present_confidence",
                "ai_building_present_probability",
                "tiles_scored_count",
                "reviewer_notes",
            ]
        )

    analysis = analysis.rename(columns={"manual_label_normalized": "manual_label"})
    analysis = analysis.loc[
        :,
        [
            "analysis_bucket",
            "parcel_row_id",
            "parcel_id",
            "county_name",
            "labeling_reason",
            "predicted_label",
            "manual_label",
            "parcel_improvement_status",
            "building_present_confidence",
            "ai_building_present_probability",
            "tiles_scored_count",
            "masked_parcel_core_crop_path",
            "masked_parcel_focus_crop_path",
            "review_hint",
            "reviewer_notes",
        ],
    ].copy()
    return analysis.sort_values(
        ["analysis_bucket", "building_present_confidence", "county_name", "parcel_row_id"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the labeled Mississippi vacancy shortlist.")
    parser.add_argument("--shortlist", default=str(DEFAULT_SHORTLIST_PATH))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT))
    parser.add_argument("--confusion-output", default=str(DEFAULT_CONFUSION_OUTPUT))
    parser.add_argument("--reason-output", default=str(DEFAULT_REASON_OUTPUT))
    parser.add_argument("--disagreement-output", default=str(DEFAULT_DISAGREEMENT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shortlist_path = Path(args.shortlist)
    frame = load_labeled_shortlist(shortlist_path)

    summary = build_summary(frame)
    confusion = build_confusion_matrix(frame)
    reason_breakdown = build_reason_breakdown(frame)
    disagreement = build_disagreement_analysis(frame)

    for output_path, data_frame in [
        (Path(args.summary_output), summary),
        (Path(args.confusion_output), confusion),
        (Path(args.reason_output), reason_breakdown),
        (Path(args.disagreement_output), disagreement),
    ]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_frame.to_csv(output_path, index=False)

    labeled = frame.loc[frame["manual_label_normalized"].notna()].copy()
    print(f"Loaded {len(frame)} shortlist rows from {shortlist_path}")
    print(f"Labeled rows: {len(labeled)}")
    if not summary.empty:
        print(summary.to_string(index=False))
    if not confusion.empty:
        print("\nConfusion matrix:")
        print(confusion.to_string(index=False))
    print(f"\nWrote summary to {args.summary_output}")
    print(f"Wrote confusion matrix to {args.confusion_output}")
    print(f"Wrote reason breakdown to {args.reason_output}")
    print(f"Wrote disagreement analysis to {args.disagreement_output}")


if __name__ == "__main__":
    main()
