from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import joblib
import pandas as pd

from vacancy_ai_common import (
    DEFAULT_TILE_URL_TEMPLATE,
    MODEL_PATH,
    PREDICTIONS_PATH,
    combined_vacancy_confidence,
    crop_specs_for_acreage,
    ensure_tile_image,
    extract_image_features,
    imagery_context_signals,
    load_app_ready_row_ids,
    load_tile_image,
    load_candidate_frame,
)


def infer_prediction_row(
    row: dict[str, object],
    *,
    pipeline,
    columns: list[str],
    model_version: str,
    zoom: int,
    refresh: bool,
    tile_template: str,
) -> dict[str, object]:
    image_path, address = ensure_tile_image(
        parcel_row_id=str(row["parcel_row_id"]),
        county_name=row.get("county_name"),
        longitude=float(row["longitude"]),
        latitude=float(row["latitude"]),
        zoom=zoom,
        refresh=refresh,
        template=tile_template,
    )
    image = load_tile_image(image_path)
    crop_predictions: list[dict[str, object]] = []
    for crop_label, crop_box in crop_specs_for_acreage(row.get("acreage")):
        features = extract_image_features(image, crop_box)
        feature_frame = pd.DataFrame([{column: features[column] for column in columns}])
        building_probability = float(pipeline.predict_proba(feature_frame)[0, 1])
        crop_predictions.append(
            {
                "crop_label": crop_label,
                "probability": building_probability,
                **imagery_context_signals(features),
            }
        )
    best_crop = max(crop_predictions, key=lambda item: float(item["probability"]))
    building_probability = float(best_crop["probability"])
    building_present_confidence = round(
        float(
            min(
                100.0,
                (building_probability * 100.0 * 0.72)
                + (float(best_crop["imagery_driveway_signal"]) * 0.18)
                + (float(best_crop["imagery_clearing_signal"]) * 0.10),
            )
        ),
        1,
    )
    ai_building_present_flag = building_present_confidence >= 60.0
    building_presence_reason = f"Best crop {best_crop['crop_label']} with imagery confidence {building_present_confidence:.1f}."
    return {
        "parcel_row_id": str(row["parcel_row_id"]),
        "county_name": row.get("county_name"),
        "imagery_source": "esri_world_imagery",
        "imagery_zoom": zoom,
        "tile_x": address.x,
        "tile_y": address.y,
        "ai_building_present_probability": round(building_probability, 6),
        "building_present_confidence": building_present_confidence,
        "ai_building_present_flag": ai_building_present_flag,
        "building_presence_reason": building_presence_reason,
        "ai_vacancy_available_flag": True,
        "ai_vacancy_source": "precomputed",
        "ai_vacancy_status_note": "Precomputed AI vacancy prediction is available for this parcel.",
        "imagery_crop_strategy": "multi_crop_v2",
        "imagery_best_crop_label": str(best_crop["crop_label"]),
        "imagery_crop_count": len(crop_predictions),
        "imagery_driveway_signal": round(float(best_crop["imagery_driveway_signal"]), 1),
        "imagery_clearing_signal": round(float(best_crop["imagery_clearing_signal"]), 1),
        "parcel_boundary_crop_ready_flag": False,
        "vacancy_confidence_score": combined_vacancy_confidence(
            bool(row["parcel_vacant_flag"]),
            building_probability,
            building_present_confidence,
        ),
        "vacancy_model_version": model_version,
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def infer_predictions(
    *,
    model_path: str,
    output_path: str,
    zoom: int,
    limit: int | None,
    county_name: str | None,
    app_ready_only: bool,
    current_vacant_only: bool,
    refresh: bool,
    tile_template: str,
    workers: int,
    resume: bool,
) -> None:
    model_bundle = joblib.load(model_path)
    pipeline = model_bundle["pipeline"]
    columns: list[str] = model_bundle["feature_columns"]
    model_version: str = model_bundle.get("model_version", "ms_building_presence_v1")

    frame = load_candidate_frame()
    if app_ready_only:
        app_ready_ids = load_app_ready_row_ids()
        frame = frame.loc[frame["parcel_row_id"].astype("string").isin(app_ready_ids)].copy()
    if county_name:
        frame = frame.loc[frame["county_name"].astype("string").eq(county_name.lower())].copy()
    if current_vacant_only:
        frame = frame.loc[frame["parcel_vacant_flag"].fillna(False)].copy()
    if limit is not None:
        frame = frame.head(limit).copy()

    existing_predictions = pd.DataFrame()
    if resume and pd.io.common.file_exists(output_path):
        existing_predictions = pd.read_parquet(output_path, engine="pyarrow")
        if "parcel_row_id" in existing_predictions.columns:
            existing_ids = existing_predictions["parcel_row_id"].astype("string")
            frame = frame.loc[~frame["parcel_row_id"].astype("string").isin(existing_ids)].copy()

    records = frame.to_dict(orient="records")
    rows: list[dict[str, object]] = []
    completed = 0
    total = len(records)
    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        future_map = {
            executor.submit(
                infer_prediction_row,
                row,
                pipeline=pipeline,
                columns=columns,
                model_version=model_version,
                zoom=zoom,
                refresh=refresh,
                tile_template=tile_template,
            ): row["parcel_row_id"]
            for row in records
        }
        for future in as_completed(future_map):
            parcel_row_id = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"AI inference failed for {parcel_row_id}: {exc}")
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"AI inference progress: {completed}/{total}")

    predictions = pd.DataFrame(rows)
    if not existing_predictions.empty:
        predictions = pd.concat([existing_predictions, predictions], ignore_index=True)
        predictions = predictions.drop_duplicates(subset=["parcel_row_id"], keep="last")
    predictions.to_parquet(output_path, index=False)
    print(f"Wrote {len(predictions)} AI vacancy prediction rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mississippi building-presence inference and write parcel-level AI vacancy signals.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output", default=str(PREDICTIONS_PATH))
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--county-name", default=None)
    parser.add_argument("--app-ready-only", action="store_true")
    parser.add_argument("--current-vacant-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tile-template", default=DEFAULT_TILE_URL_TEMPLATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer_predictions(
        model_path=args.model,
        output_path=args.output,
        zoom=args.zoom,
        limit=args.limit,
        county_name=args.county_name,
        app_ready_only=args.app_ready_only,
        current_vacant_only=args.current_vacant_only,
        refresh=args.refresh,
        tile_template=args.tile_template,
        workers=args.workers,
        resume=args.resume,
    )
