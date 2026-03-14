from __future__ import annotations

import argparse

import pandas as pd

from vacancy_ai_common import (
    DEFAULT_TILE_URL_TEMPLATE,
    MODEL_VERSION,
    TRAINING_MANIFEST_PATH,
    crop_specs_for_acreage,
    ensure_tile_image,
    extract_image_features,
    load_tile_image,
    load_candidate_frame,
    weak_label_frame,
)


def build_dataset(
    *,
    output: str,
    zoom: int,
    positive_limit: int,
    negative_limit: int,
    crop_strategy: str,
    refresh: bool,
    tile_template: str,
) -> None:
    frame = weak_label_frame(load_candidate_frame())
    positive = frame.loc[frame["weak_building_label"].eq(1)].sample(n=min(positive_limit, int(frame["weak_building_label"].eq(1).sum())), random_state=42)
    negative = frame.loc[frame["weak_building_label"].eq(0)].sample(n=min(negative_limit, int(frame["weak_building_label"].eq(0).sum())), random_state=42)
    sampled = pd.concat([positive, negative], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for _, row in sampled.iterrows():
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
        if crop_strategy == "center_tight":
            crop_label, crop_box = crop_specs_for_acreage(row.get("acreage"))[1]
        else:
            crop_label, crop_box = crop_specs_for_acreage(row.get("acreage"))[0]
        features = extract_image_features(image, crop_box)
        rows.append(
            {
                "parcel_row_id": str(row["parcel_row_id"]),
                "county_name": row.get("county_name"),
                "image_path": str(image_path),
                "weak_building_label": int(row["weak_building_label"]),
                "building_count": float(row["building_count"]),
                "building_area_total": float(row["building_area_total"]),
                "parcel_vacant_flag": bool(row["parcel_vacant_flag"]),
                "imagery_source": "esri_world_imagery",
                "imagery_zoom": zoom,
                "tile_x": address.x,
                "tile_y": address.y,
                "imagery_crop_strategy": crop_strategy,
                "imagery_crop_label": crop_label,
                "parcel_boundary_crop_ready_flag": False,
                "model_version": MODEL_VERSION,
                **features,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(output, index=False)
    print(f"Wrote {len(manifest)} labeled imagery rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a weakly labeled Mississippi building-presence imagery dataset.")
    parser.add_argument("--output", default=str(TRAINING_MANIFEST_PATH))
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--positive-limit", type=int, default=3000)
    parser.add_argument("--negative-limit", type=int, default=3000)
    parser.add_argument("--crop-strategy", choices=["tile_full", "center_tight"], default="center_tight")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tile-template", default=DEFAULT_TILE_URL_TEMPLATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        output=args.output,
        zoom=args.zoom,
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
        crop_strategy=args.crop_strategy,
        refresh=args.refresh,
        tile_template=args.tile_template,
    )
