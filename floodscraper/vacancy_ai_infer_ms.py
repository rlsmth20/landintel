from __future__ import annotations

import argparse
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
    load_tile_image,
    load_candidate_frame,
)


def infer_predictions(
    *,
    model_path: str,
    output_path: str,
    zoom: int,
    limit: int | None,
    county_name: str | None,
    current_vacant_only: bool,
    refresh: bool,
    tile_template: str,
) -> None:
    model_bundle = joblib.load(model_path)
    pipeline = model_bundle["pipeline"]
    columns: list[str] = model_bundle["feature_columns"]
    model_version: str = model_bundle.get("model_version", "ms_building_presence_v1")

    frame = load_candidate_frame()
    if county_name:
        frame = frame.loc[frame["county_name"].astype("string").eq(county_name.lower())].copy()
    if current_vacant_only:
        frame = frame.loc[frame["parcel_vacant_flag"].fillna(False)].copy()
    if limit is not None:
        frame = frame.head(limit).copy()

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
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
        rows.append(
            {
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
        )

    predictions = pd.DataFrame(rows)
    predictions.to_parquet(output_path, index=False)
    print(f"Wrote {len(predictions)} AI vacancy prediction rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mississippi building-presence inference and write parcel-level AI vacancy signals.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output", default=str(PREDICTIONS_PATH))
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--county-name", default=None)
    parser.add_argument("--current-vacant-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
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
        current_vacant_only=args.current_vacant_only,
        refresh=args.refresh,
        tile_template=args.tile_template,
    )
