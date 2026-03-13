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
    ensure_tile_image,
    extract_image_features,
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
        features = extract_image_features(image_path)
        feature_frame = pd.DataFrame([{column: features[column] for column in columns}])
        building_probability = float(pipeline.predict_proba(feature_frame)[0, 1])
        ai_building_present_flag = building_probability >= 0.5
        rows.append(
            {
                "parcel_row_id": str(row["parcel_row_id"]),
                "county_name": row.get("county_name"),
                "imagery_source": "esri_world_imagery",
                "imagery_zoom": zoom,
                "tile_x": address.x,
                "tile_y": address.y,
                "ai_building_present_probability": round(building_probability, 6),
                "ai_building_present_flag": ai_building_present_flag,
                "vacancy_confidence_score": combined_vacancy_confidence(bool(row["parcel_vacant_flag"]), building_probability),
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
