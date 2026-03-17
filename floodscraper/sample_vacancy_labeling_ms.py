from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from vacancy_ai_common import (
    APP_READY_PATH,
    DEFAULT_TILE_URL_TEMPLATE,
    PARCEL_MASTER_PATH,
    PREDICTIONS_PATH,
    centroid_tile,
    ensure_tile_image,
    load_candidate_frame,
    tile_url,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT / "data" / "buildings_processed" / "ms_vacancy_labeling_sample_300.csv"


def load_sampling_frame() -> pd.DataFrame:
    frame = load_candidate_frame()

    if PREDICTIONS_PATH.exists():
        predictions = pd.read_parquet(
            PREDICTIONS_PATH,
            columns=[
                "parcel_row_id",
                "ai_building_present_flag",
                "building_present_confidence",
            ],
            engine="pyarrow",
        )
        frame = frame.merge(predictions, on="parcel_row_id", how="left")
    else:
        frame["ai_building_present_flag"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        frame["building_present_confidence"] = np.nan

    if APP_READY_PATH.exists():
        leads = pd.read_parquet(
            APP_READY_PATH,
            columns=["parcel_row_id", "lead_score_total"],
            engine="pyarrow",
        )
        frame = frame.merge(leads, on="parcel_row_id", how="left")
    else:
        frame["lead_score_total"] = np.nan

    if PARCEL_MASTER_PATH.exists():
        parcels = pd.read_parquet(
            PARCEL_MASTER_PATH,
            columns=["parcel_row_id", "parcel_id", "county_name", "latitude", "longitude"],
            engine="pyarrow",
        )
        frame = parcels.merge(
            frame.drop(columns=[column for column in ("county_name", "latitude", "longitude") if column in frame.columns]),
            on="parcel_row_id",
            how="left",
        )
    else:
        frame["parcel_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")

    frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
    frame["parcel_id"] = frame.get("parcel_id", pd.Series(pd.NA, index=frame.index)).astype("string")
    frame["county_name"] = frame.get("county_name", pd.Series(pd.NA, index=frame.index)).astype("string")
    frame["building_count"] = pd.to_numeric(frame.get("building_count"), errors="coerce").fillna(0)
    frame["building_present_confidence"] = pd.to_numeric(frame.get("building_present_confidence"), errors="coerce")
    frame["lead_score_total"] = pd.to_numeric(frame.get("lead_score_total"), errors="coerce")
    frame["latitude"] = pd.to_numeric(frame.get("latitude"), errors="coerce")
    frame["longitude"] = pd.to_numeric(frame.get("longitude"), errors="coerce")
    frame["ai_building_present_flag"] = (
        frame["ai_building_present_flag"].astype("boolean")
        if "ai_building_present_flag" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="boolean")
    )
    frame = frame.loc[frame["latitude"].notna() & frame["longitude"].notna()].copy()
    return frame


def _sample_frame(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.head(0).copy()
    if len(frame) <= count:
        return frame.sample(frac=1.0, random_state=seed).copy()
    return frame.sample(n=count, random_state=seed).copy()


def _exclude_ids(frame: pd.DataFrame, exclude_ids: Iterable[str]) -> pd.DataFrame:
    exclude_index = pd.Index([str(value) for value in exclude_ids], dtype="string")
    return frame.loc[~frame["parcel_row_id"].astype("string").isin(exclude_index)].copy()


def sample_random_cases(frame: pd.DataFrame, count: int, seed: int, exclude_ids: set[str]) -> pd.DataFrame:
    ai_available = frame["ai_building_present_flag"].notna() | frame["building_present_confidence"].notna()
    pool = _exclude_ids(frame.loc[ai_available].copy(), exclude_ids)
    sampled = _sample_frame(pool, count, seed)
    sampled["sample_group"] = "random"
    sampled["sample_reason"] = "random_ai_available"
    return sampled


def sample_disagreement_cases(frame: pd.DataFrame, count: int, seed: int, exclude_ids: set[str]) -> pd.DataFrame:
    available = _exclude_ids(frame, exclude_ids)
    case_a = available.loc[
        available["building_count"].le(0)
        & available["ai_building_present_flag"].fillna(False)
    ].copy()
    case_b = available.loc[
        available["building_count"].gt(0)
        & available["ai_building_present_flag"].eq(False).fillna(False)
    ].copy()

    target_each = count // 2
    sample_a = _sample_frame(case_a, min(len(case_a), target_each), seed + 11)
    sample_a["sample_group"] = "disagreement"
    sample_a["sample_reason"] = "building_count_0_ai_true"

    used = set(sample_a["parcel_row_id"].astype(str).tolist())
    case_b = _exclude_ids(case_b, used)
    sample_b = _sample_frame(case_b, min(len(case_b), target_each), seed + 17)
    sample_b["sample_group"] = "disagreement"
    sample_b["sample_reason"] = "building_count_positive_ai_false"

    sampled = pd.concat([sample_a, sample_b], ignore_index=True)
    remainder = count - len(sampled)
    if remainder > 0:
        remaining_pool = _exclude_ids(
            available.loc[
                (available["building_count"].le(0) & available["ai_building_present_flag"].fillna(False))
                | (available["building_count"].gt(0) & available["ai_building_present_flag"].eq(False).fillna(False))
            ].copy(),
            set(sampled["parcel_row_id"].astype(str).tolist()),
        )
        backfill = _sample_frame(remaining_pool, min(len(remaining_pool), remainder), seed + 23)
        if not backfill.empty:
            backfill["sample_group"] = "disagreement"
            backfill["sample_reason"] = np.where(
                backfill["building_count"].le(0),
                "building_count_0_ai_true",
                "building_count_positive_ai_false",
            )
            sampled = pd.concat([sampled, backfill], ignore_index=True)
    if len(sampled) < count:
        raise ValueError(f"Only found {len(sampled)} disagreement cases; need {count}.")
    return sampled.head(count).copy()


def collect_comp_parcel_ids(subject_ids: list[str], limit_per_subject: int) -> list[str]:
    sys.path.insert(0, str(ROOT / "backend"))
    from app.services import mississippi_leads_service as service  # noqa: PLC0415

    comp_ids: list[str] = []
    for parcel_row_id in subject_ids:
        response = service.get_nearby_comps(parcel_row_id, limit=limit_per_subject)
        if not response:
            continue
        for item in response.get("items", []):
            comp_id = str(item.get("parcel_row_id") or "").strip()
            if comp_id:
                comp_ids.append(comp_id)
    return list(dict.fromkeys(comp_ids))


def sample_high_impact_cases(
    frame: pd.DataFrame,
    count: int,
    seed: int,
    exclude_ids: set[str],
    comp_subject_count: int,
    comp_limit_per_subject: int,
) -> pd.DataFrame:
    available = _exclude_ids(frame, exclude_ids)
    ranked = available.sort_values(["lead_score_total", "parcel_row_id"], ascending=[False, True], na_position="last").copy()

    subject_target = min(count // 2, len(ranked))
    top_subjects = ranked.head(max(subject_target, comp_subject_count)).copy()
    selected_subjects = top_subjects.head(subject_target).copy()
    selected_subjects["sample_group"] = "high_impact"
    selected_subjects["sample_reason"] = "top_lead_score"

    comp_subject_ids = top_subjects.head(comp_subject_count)["parcel_row_id"].astype(str).tolist()
    comp_ids = collect_comp_parcel_ids(comp_subject_ids, comp_limit_per_subject)
    comp_pool = available.loc[available["parcel_row_id"].astype("string").isin(pd.Index(comp_ids, dtype="string"))].copy()
    comp_pool = _exclude_ids(comp_pool, set(selected_subjects["parcel_row_id"].astype(str).tolist()))
    comp_selected = comp_pool.head(max(count - len(selected_subjects), 0)).copy()
    comp_selected["sample_group"] = "high_impact"
    comp_selected["sample_reason"] = "used_in_nearby_comps"

    sampled = pd.concat([selected_subjects, comp_selected], ignore_index=True)
    remainder = count - len(sampled)
    if remainder > 0:
        backfill_pool = _exclude_ids(ranked, set(sampled["parcel_row_id"].astype(str).tolist()))
        backfill = _sample_frame(backfill_pool.head(max(remainder * 4, remainder)), remainder, seed + 31)
        if not backfill.empty:
            backfill["sample_group"] = "high_impact"
            backfill["sample_reason"] = "top_lead_score_backfill"
            sampled = pd.concat([sampled, backfill], ignore_index=True)
    if len(sampled) < count:
        raise ValueError(f"Only found {len(sampled)} high-impact cases; need {count}.")
    return sampled.head(count).copy()


def attach_imagery_columns(
    frame: pd.DataFrame,
    *,
    zoom: int,
    fetch_images: bool,
    tile_template: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        longitude = float(row["longitude"])
        latitude = float(row["latitude"])
        address = centroid_tile(longitude, latitude, zoom)
        image_url = tile_url(address, tile_template)
        image_path = ""
        if fetch_images:
            path, _ = ensure_tile_image(
                parcel_row_id=str(row["parcel_row_id"]),
                county_name=row.get("county_name"),
                longitude=longitude,
                latitude=latitude,
                zoom=zoom,
                refresh=False,
                template=tile_template,
            )
            image_path = str(path)
        records.append(
            {
                "parcel_row_id": str(row["parcel_row_id"]),
                "image_url": image_url,
                "image_path": image_path,
            }
        )
    return frame.merge(pd.DataFrame(records), on="parcel_row_id", how="left")


def build_labeling_sample(
    *,
    random_count: int,
    disagreement_count: int,
    high_impact_count: int,
    seed: int,
    zoom: int,
    fetch_images: bool,
    tile_template: str,
    comp_subject_count: int,
    comp_limit_per_subject: int,
) -> pd.DataFrame:
    frame = load_sampling_frame()
    selected_ids: set[str] = set()

    random_sample = sample_random_cases(frame, random_count, seed, selected_ids)
    selected_ids.update(random_sample["parcel_row_id"].astype(str).tolist())

    disagreement_sample = sample_disagreement_cases(frame, disagreement_count, seed + 101, selected_ids)
    selected_ids.update(disagreement_sample["parcel_row_id"].astype(str).tolist())

    high_impact_sample = sample_high_impact_cases(
        frame,
        high_impact_count,
        seed + 211,
        selected_ids,
        comp_subject_count,
        comp_limit_per_subject,
    )

    sampled = pd.concat([random_sample, disagreement_sample, high_impact_sample], ignore_index=True)
    sampled = attach_imagery_columns(
        sampled,
        zoom=zoom,
        fetch_images=fetch_images,
        tile_template=tile_template,
    )
    sampled["ai_building_present_flag"] = sampled["ai_building_present_flag"].astype("boolean")
    sampled["building_present_confidence"] = pd.to_numeric(sampled["building_present_confidence"], errors="coerce").round(1)
    sampled["lead_score_total"] = pd.to_numeric(sampled["lead_score_total"], errors="coerce").round(2)
    sampled = sampled.loc[
        :,
        [
            "sample_group",
            "sample_reason",
            "parcel_row_id",
            "parcel_id",
            "county_name",
            "image_url",
            "image_path",
            "building_count",
            "ai_building_present_flag",
            "building_present_confidence",
            "lead_score_total",
        ],
    ].copy()
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Mississippi parcels for vacancy labeling with emphasis on disagreement and high-impact cases.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--random-count", type=int, default=100)
    parser.add_argument("--disagreement-count", type=int, default=100)
    parser.add_argument("--high-impact-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--fetch-images", action="store_true")
    parser.add_argument("--tile-template", default=DEFAULT_TILE_URL_TEMPLATE)
    parser.add_argument("--comp-subject-count", type=int, default=15)
    parser.add_argument("--comp-limit-per-subject", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = build_labeling_sample(
        random_count=args.random_count,
        disagreement_count=args.disagreement_count,
        high_impact_count=args.high_impact_count,
        seed=args.seed,
        zoom=args.zoom,
        fetch_images=args.fetch_images,
        tile_template=args.tile_template,
        comp_subject_count=args.comp_subject_count,
        comp_limit_per_subject=args.comp_limit_per_subject,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    print(f"Wrote {len(sample)} labeling rows to {output_path}")
    print(sample.groupby(["sample_group", "sample_reason"]).size().to_string())


if __name__ == "__main__":
    main()
