from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import pandas as pd

from parcel_contract_ms import (
    TILE_DEBUG_REQUIRED_COLUMNS,
    canonical_identity_payload,
    validate_required_columns,
)
from vacancy_ai_common import (
    aggregate_parcel_tile_predictions,
    adjust_confidence_for_tile_coverage,
    build_ai_vacancy_status_note,
    build_multi_tile_status_note,
    build_parcel_inference_tile_plan,
    DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD,
    DEFAULT_OUTSIDE_MASK_DIM_FACTOR,
    DEFAULT_OUTSIDE_MASK_FILL,
    DEFAULT_PARCEL_BUFFER_PIXELS,
    DEFAULT_TILE_NEGATIVE_CONFIDENCE_THRESHOLD,
    DEFAULT_TILE_URL_TEMPLATE,
    DEFAULT_USE_PARCEL_MASK,
    building_present_confidence_from_probability,
    calibrated_building_probability,
    MODEL_PATH,
    PREDICTIONS_PATH,
    combined_vacancy_confidence,
    centroid_tile,
    crop_mask_coverage,
    ensure_tile_image_for_address,
    extract_image_features,
    imagery_false_positive_risk,
    imagery_context_signals,
    load_app_ready_row_ids,
    load_candidate_frame,
    load_parcel_geometry_lookup,
    prepare_parcel_aware_image,
    tile_coordinate,
    tile_url,
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
    use_parcel_mask: bool,
    outside_mask_fill: str,
    outside_mask_dim_factor: float,
    parcel_buffer_pixels: int,
    use_multi_tile_extent: bool,
    include_tile_debug_rows: bool,
) -> dict[str, object]:
    address = centroid_tile(
        longitude=float(row["longitude"]),
        latitude=float(row["latitude"]),
        zoom=zoom,
    )
    tile_plan = build_parcel_inference_tile_plan(
        row.get("geometry"),
        address,
        use_multi_tile_extent=use_multi_tile_extent and use_parcel_mask,
    )
    tile_predictions: list[dict[str, object]] = []
    for tile_record in tile_plan["tile_records"]:
        tile_address = tile_record["address"]
        tile_path = ensure_tile_image_for_address(
            parcel_row_id=str(row["parcel_row_id"]),
            county_name=row.get("county_name"),
            address=tile_address,
            refresh=refresh,
            template=tile_template,
            centroid_address=address,
        )
        tile_request_url = tile_url(tile_address, tile_template)
        prepared = prepare_parcel_aware_image(
            tile_path,
            address=tile_address,
            geometry_value=row.get("geometry"),
            acreage=row.get("acreage"),
            coverage_diagnostics=tile_record,
            use_parcel_mask=use_parcel_mask,
            outside_mask_fill=outside_mask_fill,
            outside_mask_dim_factor=outside_mask_dim_factor,
            parcel_buffer_pixels=parcel_buffer_pixels,
        )
        image = prepared["image"]
        parcel_mask = prepared.get("parcel_mask")
        crop_predictions: list[dict[str, object]] = []
        crop_lookup: dict[str, float] = {}
        for crop_label, crop_box in prepared["crop_specs"]:
            features = extract_image_features(image, crop_box)
            feature_frame = pd.DataFrame([{column: features[column] for column in columns}])
            raw_probability = float(pipeline.predict_proba(feature_frame)[0, 1])
            context_signals = imagery_context_signals(features)
            parcel_coverage_ratio = crop_mask_coverage(parcel_mask, crop_box)
            false_positive_risk = imagery_false_positive_risk(
                raw_probability,
                driveway_signal=float(context_signals["imagery_driveway_signal"]),
                clearing_signal=float(context_signals["imagery_clearing_signal"]),
                features=features,
                crop_label=str(crop_label),
                parcel_coverage_ratio=parcel_coverage_ratio,
            )
            building_probability = calibrated_building_probability(
                raw_probability,
                false_positive_risk=false_positive_risk,
                crop_label=str(crop_label),
                parcel_coverage_ratio=parcel_coverage_ratio,
            )
            crop_lookup[f"{crop_label}_probability"] = round(building_probability, 6)
            crop_lookup[f"{crop_label}_confidence"] = building_present_confidence_from_probability(building_probability)
            crop_predictions.append(
                {
                    "crop_label": crop_label,
                    "raw_probability": raw_probability,
                    "probability": building_probability,
                    "parcel_coverage_ratio": round(parcel_coverage_ratio, 4),
                    "false_positive_risk": round(false_positive_risk, 1),
                    **context_signals,
                }
            )
        best_crop = max(crop_predictions, key=lambda item: float(item["probability"]))
        tile_building_confidence = building_present_confidence_from_probability(float(best_crop["probability"]))
        tile_predictions.append(
            {
                "tile_label": str(tile_record["tile_label"]),
                "tile_coordinate": str(tile_record.get("tile_coordinate") or tile_coordinate(tile_address)),
                "tile_z": int(tile_address.z),
                "tile_x": int(tile_address.x),
                "tile_y": int(tile_address.y),
                "tile_rank": int(tile_record["tile_rank"]),
                "centroid_tile_flag": bool(tile_record["centroid_tile_flag"]),
                "tile_parcel_coverage_ratio": float(tile_record["parcel_tile_coverage_ratio"]),
                "tile_parcel_coverage_pct": float(tile_record["parcel_tile_coverage_pct"]),
                "tile_bbox_coverage_pct": float(tile_record["parcel_bbox_tile_coverage_pct"]),
                "tile_cache_path": str(tile_path),
                "tile_request_url": tile_request_url,
                "best_crop_label": str(best_crop["crop_label"]),
                "probability": float(best_crop["probability"]),
                "building_present_confidence": tile_building_confidence,
                "parcel_coverage_ratio": float(best_crop["parcel_coverage_ratio"]),
                "false_positive_risk": float(best_crop["false_positive_risk"]),
                "imagery_driveway_signal": float(best_crop["imagery_driveway_signal"]),
                "imagery_clearing_signal": float(best_crop["imagery_clearing_signal"]),
                "tile_crop_count": len(crop_predictions),
                "parcel_boundary_crop_ready_flag": bool(prepared.get("parcel_boundary_crop_ready_flag", False)),
                "original_geom_type": prepared.get("original_geom_type"),
                "clipped_geom_type": prepared.get("clipped_geom_type"),
                "polygon_part_count": int(prepared.get("polygon_part_count", 0) or 0),
                "clipped_polygon_part_count": int(prepared.get("clipped_polygon_part_count", 0) or 0),
                "bounds_before_clip": json.dumps(prepared.get("bounds_before_clip")) if prepared.get("bounds_before_clip") is not None else None,
                "bounds_after_clip": json.dumps(prepared.get("bounds_after_clip")) if prepared.get("bounds_after_clip") is not None else None,
                "tile_building_signal_flag": bool(
                    tile_building_confidence >= DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD
                    and (
                        str(best_crop["crop_label"]) == "parcel_core"
                        or float(best_crop["parcel_coverage_ratio"]) >= 0.35
                        or float(best_crop["probability"]) >= 0.90
                    )
                ),
                "tile_negative_signal_flag": bool(tile_building_confidence <= DEFAULT_TILE_NEGATIVE_CONFIDENCE_THRESHOLD),
                **crop_lookup,
            }
        )

    aggregation = aggregate_parcel_tile_predictions(tile_predictions)
    best_tile = next(
        (item for item in tile_predictions if item["tile_label"] == aggregation["best_tile_label"]),
        tile_predictions[0],
    )
    building_probability = float(aggregation["probability"])
    building_present_confidence = float(aggregation["building_present_confidence"])
    ai_building_present_flag = bool(aggregation["ai_building_present_flag"])
    penalty_note = ""
    if float(best_tile["false_positive_risk"]) >= 30.0:
        penalty_note = " Non-building context penalty applied for road/clearing-style features."
    if aggregation["multi_tile_inference_used_flag"]:
        building_presence_reason = (
            f"Multi-tile parcel inference scored {aggregation['tiles_scored_count']} tiles; "
            f"best tile {aggregation['best_tile_label']} crop {best_tile['best_crop_label']} produced imagery confidence "
            f"{building_present_confidence:.1f}. {aggregation['multi_tile_aggregation_reason']}{penalty_note}"
        )
    else:
        strict_best_crop = bool(best_tile.get("parcel_boundary_crop_ready_flag")) and str(best_tile["best_crop_label"]) in {
            "parcel_core",
            "parcel_focus",
        }
        building_presence_reason = (
            f"Best strict parcel crop {best_tile['best_crop_label']} with imagery confidence {building_present_confidence:.1f}.{penalty_note}"
            if strict_best_crop
            else f"Best crop {best_tile['best_crop_label']} with imagery confidence {building_present_confidence:.1f}.{penalty_note}"
        )
    vacancy_confidence_score = combined_vacancy_confidence(
        bool(row["parcel_vacant_flag"]),
        building_probability,
        building_present_confidence,
    )
    if not aggregation["multi_tile_inference_used_flag"]:
        vacancy_confidence_score = adjust_confidence_for_tile_coverage(
            vacancy_confidence_score,
            parcel_tile_coverage_ratio=tile_plan.get("parcel_tile_coverage_ratio"),
            parcel_bbox_tile_coverage_ratio=tile_plan.get("parcel_bbox_tile_coverage_ratio"),
        )
    ai_vacancy_status_note = (
        build_multi_tile_status_note(
            "Precomputed AI vacancy prediction is available for this parcel.",
            aggregation,
        )
        if aggregation["multi_tile_inference_used_flag"]
        else build_ai_vacancy_status_note(
            "Precomputed AI vacancy prediction is available for this parcel.",
            tile_plan,
        )
    )
    result = {
        **canonical_identity_payload(row),
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
        "ai_vacancy_status_note": ai_vacancy_status_note,
        "imagery_crop_strategy": "parcel_mask_multi_tile_v1" if aggregation["multi_tile_inference_used_flag"] else "parcel_mask_tight_crop_v2",
        "imagery_best_crop_label": str(best_tile["best_crop_label"]),
        "imagery_crop_count": int(sum(int(item.get("tile_crop_count", 0)) for item in tile_predictions)),
        "imagery_driveway_signal": round(float(best_tile["imagery_driveway_signal"]), 1),
        "imagery_clearing_signal": round(float(best_tile["imagery_clearing_signal"]), 1),
        "imagery_false_positive_risk": round(float(best_tile["false_positive_risk"]), 1),
        "imagery_parcel_coverage_ratio": round(float(best_tile["parcel_coverage_ratio"]), 4),
        "parcel_tile_coverage_pct": tile_plan.get("parcel_tile_coverage_pct"),
        "parcel_bbox_tile_coverage_pct": tile_plan.get("parcel_bbox_tile_coverage_pct"),
        "full_parcel_visible_flag": bool(tile_plan.get("full_parcel_visible_flag", False)),
        "parcel_extent_exceeds_tile_flag": bool(tile_plan.get("parcel_extent_exceeds_tile_flag", False)),
        "parcel_tile_low_coverage_flag": bool(tile_plan.get("parcel_tile_low_coverage_flag", False)),
        "multi_tile_candidate_flag": bool(tile_plan.get("multi_tile_candidate_flag", False)),
        "parcel_covering_tile_count": int(tile_plan.get("parcel_covering_tile_count", 0) or 0),
        "tile_coordinates": str(tile_plan.get("tile_coordinates") or "[]"),
        "unique_tile_count": int(tile_plan.get("unique_tile_count", len(tile_predictions)) or 0),
        "duplicate_tile_flag": bool(tile_plan.get("duplicate_tile_flag", False)),
        "original_geom_type": best_tile.get("original_geom_type"),
        "clipped_geom_type": best_tile.get("clipped_geom_type"),
        "polygon_part_count": int(best_tile.get("polygon_part_count", 0) or 0),
        "clipped_polygon_part_count": int(best_tile.get("clipped_polygon_part_count", 0) or 0),
        "bounds_before_clip": best_tile.get("bounds_before_clip"),
        "bounds_after_clip": best_tile.get("bounds_after_clip"),
        "parcel_boundary_crop_ready_flag": bool(
            any(bool(item.get("parcel_boundary_crop_ready_flag")) for item in tile_predictions)
        ),
        "tiles_scored_count": int(aggregation["tiles_scored_count"]),
        "tiles_with_building_signal_count": int(aggregation["tiles_with_building_signal_count"]),
        "multi_tile_inference_used_flag": bool(aggregation["multi_tile_inference_used_flag"]),
        "multi_tile_aggregation_reason": str(aggregation["multi_tile_aggregation_reason"]),
        "best_tile_label": aggregation["best_tile_label"],
        "best_tile_confidence": aggregation["best_tile_confidence"],
        "best_tile_crop_label": aggregation["best_tile_crop_label"],
        "best_tile_probability": aggregation["best_tile_probability"],
        "best_tile_parcel_coverage_pct": aggregation["best_tile_parcel_coverage_pct"],
        "negative_tile_coverage_pct": aggregation["negative_tile_coverage_pct"],
        "vacancy_confidence_score": vacancy_confidence_score,
        "vacancy_model_version": model_version,
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if include_tile_debug_rows:
        result["_tile_debug_rows"] = [
            {
                **canonical_identity_payload(row),
                "tile_label": tile_prediction["tile_label"],
                "tile_coordinate": tile_prediction["tile_coordinate"],
                "tile_z": tile_prediction["tile_z"],
                "tile_x": tile_prediction["tile_x"],
                "tile_y": tile_prediction["tile_y"],
                "tile_rank": tile_prediction["tile_rank"],
                "centroid_tile_flag": tile_prediction["centroid_tile_flag"],
                "tile_cache_path": tile_prediction["tile_cache_path"],
                "tile_request_url": tile_prediction["tile_request_url"],
                "tile_parcel_coverage_pct": round(float(tile_prediction["tile_parcel_coverage_pct"]), 1),
                "tile_bbox_coverage_pct": round(float(tile_prediction["tile_bbox_coverage_pct"]), 1),
                "original_geom_type": tile_prediction["original_geom_type"],
                "clipped_geom_type": tile_prediction["clipped_geom_type"],
                "polygon_part_count": int(tile_prediction["polygon_part_count"]),
                "clipped_polygon_part_count": int(tile_prediction["clipped_polygon_part_count"]),
                "bounds_before_clip": tile_prediction["bounds_before_clip"],
                "bounds_after_clip": tile_prediction["bounds_after_clip"],
                "best_crop_label": tile_prediction["best_crop_label"],
                "tile_probability": round(float(tile_prediction["probability"]), 6),
                "tile_building_present_confidence": round(float(tile_prediction["building_present_confidence"]), 1),
                "tile_building_signal_flag": bool(tile_prediction["tile_building_signal_flag"]),
                "tile_negative_signal_flag": bool(tile_prediction["tile_negative_signal_flag"]),
                "tile_false_positive_risk": round(float(tile_prediction["false_positive_risk"]), 1),
                "tile_crop_parcel_coverage_ratio": round(float(tile_prediction["parcel_coverage_ratio"]), 4),
                "parcel_boundary_crop_ready_flag": bool(tile_prediction["parcel_boundary_crop_ready_flag"]),
                "parcel_core_probability": tile_prediction.get("parcel_core_probability"),
                "parcel_focus_probability": tile_prediction.get("parcel_focus_probability"),
                "parcel_probability": round(building_probability, 6),
                "parcel_building_present_confidence": round(building_present_confidence, 1),
                "multi_tile_inference_used_flag": bool(aggregation["multi_tile_inference_used_flag"]),
                "tile_coordinates": str(tile_plan.get("tile_coordinates") or "[]"),
                "unique_tile_count": int(tile_plan.get("unique_tile_count", len(tile_predictions)) or 0),
                "duplicate_tile_flag": bool(tile_plan.get("duplicate_tile_flag", False)),
                "multi_tile_aggregation_reason": aggregation["multi_tile_aggregation_reason"],
                "best_tile_label": aggregation["best_tile_label"],
                "best_tile_confidence": aggregation["best_tile_confidence"],
            }
            for tile_prediction in tile_predictions
        ]
    return result


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
    use_parcel_mask: bool,
    outside_mask_fill: str,
    outside_mask_dim_factor: float,
    parcel_buffer_pixels: int,
    use_multi_tile_extent: bool,
    tile_debug_output: str | None,
    parcel_row_ids: list[str] | None,
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
    if parcel_row_ids:
        parcel_index = pd.Index(parcel_row_ids, dtype="string")
        frame = frame.loc[frame["parcel_row_id"].astype("string").isin(parcel_index)].copy()
    if current_vacant_only:
        frame = frame.loc[frame["parcel_vacant_flag"].fillna(False)].copy()
    if limit is not None:
        frame = frame.head(limit).copy()
    if use_parcel_mask and not frame.empty:
        geometry_lookup = load_parcel_geometry_lookup(frame["parcel_row_id"].astype("string"))
        frame["geometry"] = frame["parcel_row_id"].astype("string").map(geometry_lookup)

    existing_predictions = pd.DataFrame()
    if resume and pd.io.common.file_exists(output_path):
        existing_predictions = pd.read_parquet(output_path, engine="pyarrow")
        if "parcel_row_id" in existing_predictions.columns:
            existing_ids = existing_predictions["parcel_row_id"].astype("string")
            frame = frame.loc[~frame["parcel_row_id"].astype("string").isin(existing_ids)].copy()

    records = frame.to_dict(orient="records")
    rows: list[dict[str, object]] = []
    tile_debug_rows: list[dict[str, object]] = []
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
                use_parcel_mask=use_parcel_mask,
                outside_mask_fill=outside_mask_fill,
                outside_mask_dim_factor=outside_mask_dim_factor,
                parcel_buffer_pixels=parcel_buffer_pixels,
                use_multi_tile_extent=use_multi_tile_extent,
                include_tile_debug_rows=bool(tile_debug_output),
            ): row["parcel_row_id"]
            for row in records
        }
        for future in as_completed(future_map):
            parcel_row_id = future_map[future]
            try:
                result = future.result()
                if tile_debug_output and "_tile_debug_rows" in result:
                    tile_debug_rows.extend(result.pop("_tile_debug_rows"))
                rows.append(result)
            except Exception as exc:
                print(f"AI inference failed for {parcel_row_id}: {exc}")
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"AI inference progress: {completed}/{total}")

    predictions = pd.DataFrame(rows)
    if not existing_predictions.empty:
        predictions = pd.concat([existing_predictions, predictions], ignore_index=True)
        predictions = predictions.drop_duplicates(subset=["parcel_row_id"], keep="last")
    validate_required_columns(
        predictions,
        required_columns=["parcel_row_id", "parcel_id"],
        non_null_columns=["parcel_row_id", "parcel_id"],
        context="vacancy_ai_infer_ms.predictions",
    )
    predictions.to_parquet(output_path, index=False)
    print(f"Wrote {len(predictions)} AI vacancy prediction rows to {output_path}")
    if tile_debug_output:
        tile_debug_frame = pd.DataFrame(tile_debug_rows)
        if not tile_debug_frame.empty:
            validate_required_columns(
                tile_debug_frame,
                required_columns=TILE_DEBUG_REQUIRED_COLUMNS,
                non_null_columns=["parcel_row_id", "parcel_id"],
                context="vacancy_ai_infer_ms.tile_debug",
            )
        debug_path = Path(tile_debug_output)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        if debug_path.suffix.lower() == ".csv":
            tile_debug_frame.to_csv(debug_path, index=False)
        else:
            tile_debug_frame.to_parquet(debug_path, index=False)
        print(f"Wrote {len(tile_debug_frame)} tile debug rows to {debug_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mississippi building-presence inference and write parcel-level AI vacancy signals.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output", default=str(PREDICTIONS_PATH))
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--county-name", default=None)
    parser.add_argument("--parcel-row-ids", default=None)
    parser.add_argument("--app-ready-only", action="store_true")
    parser.add_argument("--current-vacant-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tile-template", default=DEFAULT_TILE_URL_TEMPLATE)
    parser.add_argument("--use-parcel-mask", dest="use_parcel_mask", action="store_true", default=DEFAULT_USE_PARCEL_MASK)
    parser.add_argument("--no-parcel-mask", dest="use_parcel_mask", action="store_false")
    parser.add_argument("--outside-mask-fill", choices=["dim", "black"], default=DEFAULT_OUTSIDE_MASK_FILL)
    parser.add_argument("--outside-mask-dim-factor", type=float, default=DEFAULT_OUTSIDE_MASK_DIM_FACTOR)
    parser.add_argument("--parcel-buffer-pixels", type=int, default=DEFAULT_PARCEL_BUFFER_PIXELS)
    parser.add_argument("--use-multi-tile-extent", dest="use_multi_tile_extent", action="store_true", default=True)
    parser.add_argument("--single-tile-only", dest="use_multi_tile_extent", action="store_false")
    parser.add_argument("--tile-debug-output", default=None)
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
        use_parcel_mask=args.use_parcel_mask,
        outside_mask_fill=args.outside_mask_fill,
        outside_mask_dim_factor=args.outside_mask_dim_factor,
        parcel_buffer_pixels=args.parcel_buffer_pixels,
        use_multi_tile_extent=args.use_multi_tile_extent,
        tile_debug_output=args.tile_debug_output,
        parcel_row_ids=[value.strip() for value in str(args.parcel_row_ids).split(",") if value.strip()] if args.parcel_row_ids else None,
    )
