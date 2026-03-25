from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from parcel_contract_ms import contract_left_merge
from parcel_geometry_quality_ms import (
    GEOMETRY_QUALITY_ARTIFACT_PATH,
    filter_training_geometry_frame,
    geometry_quality_diagnostics,
    load_geometry_quality_frame,
)
from vacancy_ai_common import (
    APP_READY_PATH,
    DEFAULT_OUTSIDE_MASK_DIM_FACTOR,
    DEFAULT_OUTSIDE_MASK_FILL,
    DEFAULT_PARCEL_BUFFER_PIXELS,
    DEFAULT_TILE_URL_TEMPLATE,
    DEFAULT_USE_PARCEL_MASK,
    MODEL_VERSION,
    PARCEL_MASTER_PATH,
    TRAINING_MANIFEST_PATH,
    TILE_CACHE_DIR,
    TileAddress,
    build_parcel_inference_tile_plan,
    crop_mask_coverage,
    ensure_tile_image_for_address,
    extract_image_features,
    imagery_context_signals,
    load_candidate_frame,
    load_parcel_geometry_lookup,
    prepare_parcel_aware_image,
    tile_cache_path_for_address,
    tile_coordinate,
    tile_label,
    weak_label_frame,
    write_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = TRAINING_MANIFEST_PATH.with_name(f"{TRAINING_MANIFEST_PATH.stem}_summary.json")
DEFAULT_POSITIVE_LIMIT = 3000
DEFAULT_NEGATIVE_LIMIT = 3000
DEFAULT_MAX_TILES_PER_PARCEL = 2
DEFAULT_MIN_PRIMARY_TILE_COVERAGE_RATIO = 0.10
DEFAULT_MIN_SECONDARY_TILE_COVERAGE_RATIO = 0.15
DEFAULT_USE_MULTI_TILE_EXTENT = True
DEFAULT_MANIFEST_PART_PARCEL_BATCH_SIZE = 250
DEFAULT_PROGRESS_EVERY_PARCELS = 100


def _checkpoint_file(checkpoint_dir: Path | None, name: str, suffix: str = ".parquet") -> Path | None:
    if checkpoint_dir is None:
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{name}{suffix}"


def _manifest_parts_dir(checkpoint_dir: Path | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    parts_dir = checkpoint_dir / "training_manifest_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    return parts_dir


def _checkpoint_meta_path(checkpoint_path: Path | None) -> Path | None:
    if checkpoint_path is None:
        return None
    return checkpoint_path.with_suffix(".meta.json")


def _write_checkpoint_metadata(checkpoint_path: Path | None, payload: dict[str, Any]) -> None:
    meta_path = _checkpoint_meta_path(checkpoint_path)
    if meta_path is None:
        return
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_checkpoint_metadata(checkpoint_path: Path | None) -> dict[str, Any]:
    meta_path = _checkpoint_meta_path(checkpoint_path)
    if meta_path is None or not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _manifest_part_paths(parts_dir: Path | None) -> list[Path]:
    if parts_dir is None or not parts_dir.exists():
        return []
    return sorted(parts_dir.glob("manifest_part_*.parquet"))


def _read_manifest_parts(parts_dir: Path | None) -> pd.DataFrame:
    part_paths = _manifest_part_paths(parts_dir)
    if not part_paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path, engine="pyarrow") for path in part_paths], ignore_index=True)


def _write_manifest_part(part_rows: list[dict[str, Any]], *, parts_dir: Path | None, part_index: int) -> Path | None:
    if parts_dir is None or not part_rows:
        return None
    part_frame = pd.DataFrame(part_rows)
    part_path = parts_dir / f"manifest_part_{part_index:04d}.parquet"
    part_frame.to_parquet(part_path, index=False)
    return part_path


def _load_optional_parcel_fields(parcel_row_ids: pd.Series | pd.Index) -> pd.DataFrame:
    normalized_ids = pd.Index(parcel_row_ids, dtype="string").dropna().unique()
    if normalized_ids.empty:
        return pd.DataFrame(columns=["parcel_row_id", "total_value", "improvement_value_1", "improvement_value_2"])
    dataset = ds.dataset(PARCEL_MASTER_PATH, format="parquet")
    rows: list[pd.DataFrame] = []
    chunk_size = 1000
    for start in range(0, len(normalized_ids), chunk_size):
        chunk = normalized_ids[start : start + chunk_size].tolist()
        table = dataset.to_table(
            columns=["parcel_row_id", "total_value", "improvement_value_1", "improvement_value_2"],
            filter=ds.field("parcel_row_id").isin(chunk),
        )
        if table.num_rows:
            rows.append(table.to_pandas())
    if not rows:
        return pd.DataFrame(columns=["parcel_row_id", "total_value", "improvement_value_1", "improvement_value_2"])
    return pd.concat(rows, ignore_index=True)


def _shuffle_class_rows(frame: pd.DataFrame, label: int, seed: int) -> pd.DataFrame:
    class_frame = frame.loc[frame["weak_building_label"].eq(label)].copy()
    if class_frame.empty:
        return class_frame
    return class_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _label_reliability_tier(tile_plan: dict[str, Any], primary_tile_record: dict[str, Any]) -> str:
    if bool(tile_plan.get("full_parcel_visible_flag")) and not bool(tile_plan.get("multi_tile_candidate_flag", False)):
        return "high"
    if float(primary_tile_record.get("parcel_tile_coverage_ratio", 0.0) or 0.0) >= 0.35:
        return "medium"
    return "low"


def _choose_training_tiles(
    tile_plan: dict[str, Any],
    *,
    max_tiles_per_parcel: int,
    min_primary_tile_coverage_ratio: float,
    min_secondary_tile_coverage_ratio: float,
) -> list[dict[str, Any]]:
    tile_records = list(tile_plan.get("tile_records") or [])
    if not tile_records:
        return []

    centroid_record = next((item for item in tile_records if bool(item.get("centroid_tile_flag"))), tile_records[0])
    best_record = max(
        tile_records,
        key=lambda item: (
            float(item.get("parcel_tile_coverage_ratio", 0.0) or 0.0),
            float(item.get("parcel_bbox_tile_coverage_ratio", 0.0) or 0.0),
            -int(item.get("tile_rank", 0) or 0),
        ),
    )
    centroid_coverage = float(tile_plan.get("parcel_tile_coverage_ratio", 0.0) or 0.0)
    if bool(tile_plan.get("multi_tile_candidate_flag")) and centroid_coverage < min_primary_tile_coverage_ratio:
        primary = best_record
        primary_strategy = "best_coverage_tile"
    else:
        primary = centroid_record
        primary_strategy = "centroid_tile"

    if float(primary.get("parcel_tile_coverage_ratio", 0.0) or 0.0) < min_primary_tile_coverage_ratio:
        return []

    selected: list[dict[str, Any]] = [
        {
            **primary,
            "tile_selection_role": "primary",
            "tile_selection_strategy": primary_strategy,
        }
    ]

    if max_tiles_per_parcel <= 1:
        return selected

    secondary_candidates = [
        item
        for item in tile_records
        if str(item.get("tile_label")) != str(primary.get("tile_label"))
        and float(item.get("parcel_tile_coverage_ratio", 0.0) or 0.0) >= min_secondary_tile_coverage_ratio
    ]
    if not secondary_candidates:
        return selected

    secondary = max(
        secondary_candidates,
        key=lambda item: (
            float(item.get("parcel_tile_coverage_ratio", 0.0) or 0.0),
            float(item.get("parcel_bbox_tile_coverage_ratio", 0.0) or 0.0),
            -int(item.get("tile_rank", 0) or 0),
        ),
    )
    selected.append(
        {
            **secondary,
            "tile_selection_role": "secondary",
            "tile_selection_strategy": "supporting_best_coverage_tile",
        }
    )
    return selected[:max_tiles_per_parcel]


def _resolve_tile_path(
    *,
    parcel_row_id: str,
    county_name: str | None,
    address: TileAddress,
    centroid_address: TileAddress,
    require_existing_tiles: bool,
    tile_template: str,
) -> tuple[Path | None, str]:
    cache_path = tile_cache_path_for_address(
        parcel_row_id,
        county_name,
        address,
        centroid_address=centroid_address,
    )
    if cache_path.exists():
        return cache_path, "cached"
    if require_existing_tiles:
        return None, "missing"
    resolved = ensure_tile_image_for_address(
        parcel_row_id=parcel_row_id,
        county_name=county_name,
        address=address,
        refresh=False,
        template=tile_template,
        centroid_address=centroid_address,
    )
    return resolved, "fetched"


def _build_manifest_rows_for_parcel(
    row: dict[str, Any],
    *,
    geometry_value: bytes | bytearray | memoryview | None,
    zoom: int,
    tile_template: str,
    use_parcel_mask: bool,
    outside_mask_fill: str,
    outside_mask_dim_factor: float,
    parcel_buffer_pixels: int,
    use_multi_tile_extent: bool,
    max_tiles_per_parcel: int,
    min_primary_tile_coverage_ratio: float,
    min_secondary_tile_coverage_ratio: float,
    require_existing_tiles: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    counters: Counter[str] = Counter()
    parcel_row_id = str(row["parcel_row_id"])
    centroid_address = TileAddress(
        x=int(row["tile_x"]),
        y=int(row["tile_y"]),
        z=zoom,
    )
    tile_plan = build_parcel_inference_tile_plan(
        geometry_value,
        centroid_address,
        use_multi_tile_extent=use_multi_tile_extent and use_parcel_mask,
    )
    if not tile_plan.get("tile_records"):
        counters["skipped_no_tile_records"] += 1
        return [], counters

    selected_tiles = _choose_training_tiles(
        tile_plan,
        max_tiles_per_parcel=max_tiles_per_parcel,
        min_primary_tile_coverage_ratio=min_primary_tile_coverage_ratio,
        min_secondary_tile_coverage_ratio=min_secondary_tile_coverage_ratio,
    )
    if not selected_tiles:
        counters["skipped_low_primary_coverage"] += 1
        return [], counters

    manifest_rows: list[dict[str, Any]] = []
    label_tier = _label_reliability_tier(tile_plan, selected_tiles[0])
    selected_tile_labels = [str(item["tile_label"]) for item in selected_tiles]
    for tile_record in selected_tiles:
        address = tile_record["address"]
        tile_path, tile_source_mode = _resolve_tile_path(
            parcel_row_id=parcel_row_id,
            county_name=row.get("county_name"),
            address=address,
            centroid_address=centroid_address,
            require_existing_tiles=require_existing_tiles,
            tile_template=tile_template,
        )
        if tile_path is None or not tile_path.exists():
            counters["skipped_missing_tile_image"] += 1
            continue

        prepared = prepare_parcel_aware_image(
            tile_path,
            address=address,
            geometry_value=geometry_value,
            acreage=row.get("acreage"),
            coverage_diagnostics=tile_record,
            use_parcel_mask=use_parcel_mask,
            outside_mask_fill=outside_mask_fill,
            outside_mask_dim_factor=outside_mask_dim_factor,
            parcel_buffer_pixels=parcel_buffer_pixels,
        )
        if not bool(prepared.get("parcel_boundary_crop_ready_flag", False)):
            counters["skipped_no_parcel_mask"] += 1
            continue

        image = prepared["image"]
        parcel_mask = prepared.get("parcel_mask")
        for crop_label, crop_box in prepared["crop_specs"]:
            features = extract_image_features(image, crop_box)
            context_signals = imagery_context_signals(features)
            crop_parcel_coverage_ratio = crop_mask_coverage(parcel_mask, crop_box)
            manifest_rows.append(
                {
                    "parcel_row_id": parcel_row_id,
                    "parcel_id": row.get("parcel_id"),
                    "state_code": row.get("state_code"),
                    "county_name": row.get("county_name"),
                    "county_fips": row.get("county_fips"),
                    "image_path": str(tile_path),
                    "weak_building_label": int(row["weak_building_label"]),
                    "weak_label_source": "footprint_weak_label",
                    "weak_label_rule": (
                        "building_count>=1_and_building_area_total>=400"
                        if int(row["weak_building_label"]) == 1
                        else "parcel_vacant_flag_and_no_mapped_building"
                    ),
                    "label_reliability_tier": label_tier,
                    "building_count": float(row["building_count"]),
                    "building_area_total": float(row["building_area_total"]),
                    "parcel_vacant_flag": bool(row["parcel_vacant_flag"]),
                    "total_value": row.get("total_value"),
                    "improvement_value_1": row.get("improvement_value_1"),
                    "improvement_value_2": row.get("improvement_value_2"),
                    "area_acres": row.get("area_acres"),
                    "perimeter_meters": row.get("perimeter_meters"),
                    "bounding_box_width_meters": row.get("bounding_box_width_meters"),
                    "bounding_box_height_meters": row.get("bounding_box_height_meters"),
                    "aspect_ratio": row.get("aspect_ratio"),
                    "compactness": row.get("compactness"),
                    "is_multipart": bool(row.get("is_multipart", False)),
                    "part_count": int(row.get("part_count", 0) or 0),
                    "geometry_quality_flag": row.get("geometry_quality_flag"),
                    "geometry_training_excluded_flag": bool(row.get("geometry_training_excluded_flag", False)),
                    "imagery_source": "esri_world_imagery",
                    "imagery_zoom": zoom,
                    "tile_x": int(address.x),
                    "tile_y": int(address.y),
                    "tile_label": str(tile_record["tile_label"]),
                    "tile_coordinate": str(tile_record.get("tile_coordinate") or tile_coordinate(address)),
                    "tile_rank": int(tile_record.get("tile_rank", 0) or 0),
                    "centroid_tile_flag": bool(tile_record.get("centroid_tile_flag", False)),
                    "tile_selection_strategy": str(tile_record["tile_selection_strategy"]),
                    "tile_selection_role": str(tile_record["tile_selection_role"]),
                    "tile_source_mode": tile_source_mode,
                    "selected_tile_count": len(selected_tiles),
                    "selected_tile_labels": json.dumps(selected_tile_labels),
                    "manifest_version": MODEL_VERSION,
                    "dataset_scope": "app_ready" if bool(row.get("app_ready_flag", False)) else "statewide_candidate",
                    "app_ready_only": bool(row.get("app_ready_flag", False)),
                    "cached_tile_required_flag": bool(require_existing_tiles),
                    "use_multi_tile_extent": bool(use_multi_tile_extent),
                    "model_version": MODEL_VERSION,
                    "imagery_crop_strategy": str(prepared.get("imagery_crop_strategy")),
                    "imagery_crop_label": str(crop_label),
                    "parcel_boundary_crop_ready_flag": True,
                    "imagery_driveway_signal": round(float(context_signals["imagery_driveway_signal"]), 1),
                    "imagery_clearing_signal": round(float(context_signals["imagery_clearing_signal"]), 1),
                    "crop_parcel_coverage_ratio": round(float(crop_parcel_coverage_ratio), 4),
                    "parcel_tile_coverage_ratio": round(float(tile_record.get("parcel_tile_coverage_ratio", np.nan)), 4),
                    "parcel_tile_coverage_pct": float(tile_record.get("parcel_tile_coverage_pct", np.nan)),
                    "parcel_bbox_tile_coverage_ratio": round(float(tile_record.get("parcel_bbox_tile_coverage_ratio", np.nan)), 4),
                    "parcel_bbox_tile_coverage_pct": float(tile_record.get("parcel_bbox_tile_coverage_pct", np.nan)),
                    "full_parcel_visible_flag": bool(tile_plan.get("full_parcel_visible_flag", False)),
                    "parcel_extent_exceeds_tile_flag": bool(tile_plan.get("parcel_extent_exceeds_tile_flag", False)),
                    "parcel_tile_low_coverage_flag": bool(tile_plan.get("parcel_tile_low_coverage_flag", False)),
                    "multi_tile_candidate_flag": bool(tile_plan.get("multi_tile_candidate_flag", False)),
                    "parcel_covering_tile_count": int(tile_plan.get("parcel_covering_tile_count", 0) or 0),
                    "tile_coordinates": str(tile_plan.get("tile_coordinates") or "[]"),
                    "unique_tile_count": int(tile_plan.get("unique_tile_count", 0) or 0),
                    "duplicate_tile_flag": bool(tile_plan.get("duplicate_tile_flag", False)),
                    "original_geom_type": prepared.get("original_geom_type"),
                    "clipped_geom_type": prepared.get("clipped_geom_type"),
                    "polygon_part_count": int(prepared.get("polygon_part_count", 0) or 0),
                    "clipped_polygon_part_count": int(prepared.get("clipped_polygon_part_count", 0) or 0),
                    "bounds_before_clip": json.dumps(prepared.get("bounds_before_clip")) if prepared.get("bounds_before_clip") is not None else None,
                    "bounds_after_clip": json.dumps(prepared.get("bounds_after_clip")) if prepared.get("bounds_after_clip") is not None else None,
                    **features,
                }
            )

    if manifest_rows:
        counters["accepted_parcels"] += 1
        counters["accepted_rows"] += len(manifest_rows)
    else:
        counters["skipped_empty_after_tile_resolution"] += 1
    return manifest_rows, counters


def _prepare_source_frame(
    *,
    app_ready_only: bool,
    reuse_geometry_quality_artifact: bool = True,
    build_geometry_quality_artifact_if_missing: bool = False,
    geometry_quality_artifact_path: Path = GEOMETRY_QUALITY_ARTIFACT_PATH,
    limit: int | None = None,
) -> pd.DataFrame:
    frame = weak_label_frame(load_candidate_frame())
    geometry_quality = load_geometry_quality_frame(
        frame["parcel_row_id"].astype("string"),
        reuse_artifact=reuse_geometry_quality_artifact,
        artifact_path=geometry_quality_artifact_path,
        build_artifact_if_missing=build_geometry_quality_artifact_if_missing,
    )
    if not geometry_quality.empty:
        frame = frame.reset_index(drop=True)
        frame = contract_left_merge(frame, geometry_quality, on="parcel_row_id")
    prefilter_diagnostics = geometry_quality_diagnostics(frame) if "geometry_quality_flag" in frame.columns else {}
    frame = filter_training_geometry_frame(frame)
    if app_ready_only:
        app_ready_ids = pd.read_parquet(APP_READY_PATH, columns=["parcel_row_id"], engine="pyarrow")["parcel_row_id"].astype("string")
        frame = frame.loc[frame["parcel_row_id"].astype("string").isin(app_ready_ids)].copy()
        frame["app_ready_flag"] = True
    else:
        frame["app_ready_flag"] = False
    if limit is not None:
        frame = frame.head(int(limit)).copy()
    frame.attrs["geometry_quality_prefilter_diagnostics"] = prefilter_diagnostics
    frame.attrs["geometry_quality_postfilter_diagnostics"] = geometry_quality_diagnostics(frame) if "geometry_quality_flag" in frame.columns else {}
    return frame


def _prepare_sampled_source(
    source: pd.DataFrame,
    *,
    positive_limit: int,
    negative_limit: int,
    zoom: int,
) -> pd.DataFrame:
    positive_source = _shuffle_class_rows(source, label=1, seed=42)
    negative_source = _shuffle_class_rows(source, label=0, seed=43)
    sampled_source = pd.concat(
        [
            positive_source.head(max(positive_limit * 3, positive_limit)),
            negative_source.head(max(negative_limit * 3, negative_limit)),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["parcel_row_id"])
    geometry_lookup = load_parcel_geometry_lookup(sampled_source["parcel_row_id"].astype("string"))
    sampled_source["geometry"] = sampled_source["parcel_row_id"].astype("string").map(geometry_lookup)
    optional_fields = _load_optional_parcel_fields(sampled_source["parcel_row_id"].astype("string"))
    if not optional_fields.empty:
        sampled_source = contract_left_merge(sampled_source, optional_fields, on="parcel_row_id")
    sampled_source["tile_x"] = (((sampled_source["longitude"] + 180.0) / 360.0) * (2**zoom)).astype(int)
    sampled_source["tile_y"] = (
        ((1.0 - np.arcsinh(np.tan(np.radians(sampled_source["latitude"]))) / np.pi) / 2.0 * (2**zoom))
        .astype(int)
    )
    return sampled_source


def build_dataset(
    *,
    output: str,
    summary_output: str,
    zoom: int,
    positive_limit: int,
    negative_limit: int,
    tile_template: str,
    use_parcel_mask: bool,
    outside_mask_fill: str,
    outside_mask_dim_factor: float,
    parcel_buffer_pixels: int,
    use_multi_tile_extent: bool,
    max_tiles_per_parcel: int,
    min_primary_tile_coverage_ratio: float,
    min_secondary_tile_coverage_ratio: float,
    require_existing_tiles: bool,
    app_ready_only: bool,
    reuse_geometry_quality_artifact: bool,
    build_geometry_quality_artifact_if_missing: bool,
    geometry_quality_artifact_path: Path,
    limit: int | None,
    checkpoint_dir: Path | None,
    resume: bool,
    manifest_part_parcel_batch_size: int,
    progress_every_parcels: int,
) -> pd.DataFrame:
    build_started = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    runtime_by_stage_seconds: dict[str, float] = {}
    source_checkpoint = _checkpoint_file(checkpoint_dir, "training_source_frame")
    sampled_source_checkpoint = _checkpoint_file(checkpoint_dir, "training_sampled_source")
    parts_dir = _manifest_parts_dir(checkpoint_dir)
    if parts_dir is not None and not resume:
        for part_path in _manifest_part_paths(parts_dir):
            part_path.unlink(missing_ok=True)

    source_stage_started = time.perf_counter()
    if resume and source_checkpoint is not None and source_checkpoint.exists():
        source = pd.read_parquet(source_checkpoint, engine="pyarrow")
        source.attrs.update(_read_checkpoint_metadata(source_checkpoint))
        runtime_by_stage_seconds["resume_source_frame_seconds"] = round(time.perf_counter() - source_stage_started, 3)
    else:
        source = _prepare_source_frame(
            app_ready_only=app_ready_only,
            reuse_geometry_quality_artifact=reuse_geometry_quality_artifact,
            build_geometry_quality_artifact_if_missing=build_geometry_quality_artifact_if_missing,
            geometry_quality_artifact_path=geometry_quality_artifact_path,
            limit=limit,
        )
        if source_checkpoint is not None:
            source.to_parquet(source_checkpoint, index=False)
            _write_checkpoint_metadata(
                source_checkpoint,
                {
                    "geometry_quality_prefilter_diagnostics": source.attrs.get("geometry_quality_prefilter_diagnostics", {}),
                    "geometry_quality_postfilter_diagnostics": source.attrs.get("geometry_quality_postfilter_diagnostics", {}),
                },
            )
        runtime_by_stage_seconds["load_source_frame_seconds"] = round(time.perf_counter() - source_stage_started, 3)
    if "geometry_quality_prefilter_diagnostics" not in source.attrs:
        source.attrs["geometry_quality_prefilter_diagnostics"] = geometry_quality_diagnostics(source)
    if "geometry_quality_postfilter_diagnostics" not in source.attrs:
        source.attrs["geometry_quality_postfilter_diagnostics"] = geometry_quality_diagnostics(source)

    sampled_stage_started = time.perf_counter()
    if resume and sampled_source_checkpoint is not None and sampled_source_checkpoint.exists():
        sampled_source = pd.read_parquet(sampled_source_checkpoint, engine="pyarrow")
        runtime_by_stage_seconds["resume_sampled_source_seconds"] = round(time.perf_counter() - sampled_stage_started, 3)
    else:
        sampled_source = _prepare_sampled_source(
            source,
            positive_limit=positive_limit,
            negative_limit=negative_limit,
            zoom=zoom,
        )
        if sampled_source_checkpoint is not None:
            sampled_source.to_parquet(sampled_source_checkpoint, index=False)
        runtime_by_stage_seconds["prepare_sampled_source_seconds"] = round(time.perf_counter() - sampled_stage_started, 3)

    build_counters: Counter[str] = Counter()
    collected_parcels: Counter[int] = Counter()
    targets = {1: int(positive_limit), 0: int(negative_limit)}
    processed_parcel_ids: set[str] = set()
    existing_manifest = pd.DataFrame()
    next_part_index = 1
    if resume and parts_dir is not None:
        existing_manifest = _read_manifest_parts(parts_dir)
        if not existing_manifest.empty:
            processed_parcel_ids = set(existing_manifest["parcel_row_id"].astype("string").tolist())
            existing_unique = existing_manifest[["parcel_row_id", "weak_building_label"]].drop_duplicates()
            collected_parcels.update(existing_unique["weak_building_label"].value_counts().to_dict())
            build_counters["resumed_existing_manifest_rows"] = int(len(existing_manifest))
            build_counters["resumed_existing_manifest_parcels"] = int(existing_unique["parcel_row_id"].astype("string").nunique())
            build_counters["resumed_existing_manifest_parts"] = len(_manifest_part_paths(parts_dir))
        next_part_index = len(_manifest_part_paths(parts_dir)) + 1

    manifest_stage_started = time.perf_counter()
    part_rows: list[dict[str, Any]] = []
    part_parcel_count = 0
    attempted_parcels = 0
    for label in (1, 0):
        class_frame = sampled_source.loc[sampled_source["weak_building_label"].eq(label)].copy()
        for _, row in class_frame.iterrows():
            if collected_parcels[label] >= targets[label]:
                break
            parcel_row_id = str(row["parcel_row_id"])
            if parcel_row_id in processed_parcel_ids:
                continue
            attempted_parcels += 1
            if row.get("geometry") is None or pd.isna(row.get("geometry")):
                build_counters["skipped_missing_geometry"] += 1
                continue
            parcel_rows, parcel_counters = _build_manifest_rows_for_parcel(
                row.to_dict(),
                geometry_value=row.get("geometry"),
                zoom=zoom,
                tile_template=tile_template,
                use_parcel_mask=use_parcel_mask,
                outside_mask_fill=outside_mask_fill,
                outside_mask_dim_factor=outside_mask_dim_factor,
                parcel_buffer_pixels=parcel_buffer_pixels,
                use_multi_tile_extent=use_multi_tile_extent,
                max_tiles_per_parcel=max_tiles_per_parcel,
                min_primary_tile_coverage_ratio=min_primary_tile_coverage_ratio,
                min_secondary_tile_coverage_ratio=min_secondary_tile_coverage_ratio,
                require_existing_tiles=require_existing_tiles,
            )
            build_counters.update(parcel_counters)
            if not parcel_rows:
                continue
            part_rows.extend(parcel_rows)
            processed_parcel_ids.add(parcel_row_id)
            collected_parcels[label] += 1
            part_parcel_count += 1
            if parts_dir is not None and part_parcel_count >= max(int(manifest_part_parcel_batch_size), 1):
                _write_manifest_part(part_rows, parts_dir=parts_dir, part_index=next_part_index)
                next_part_index += 1
                part_rows = []
                part_parcel_count = 0
            if progress_every_parcels > 0 and attempted_parcels % int(progress_every_parcels) == 0:
                print(
                    f"[training-manifest] attempted={attempted_parcels} "
                    f"accepted_pos={collected_parcels[1]}/{targets[1]} "
                    f"accepted_neg={collected_parcels[0]}/{targets[0]} "
                    f"elapsed={time.perf_counter() - manifest_stage_started:.1f}s"
                )
    if part_rows and parts_dir is not None:
        _write_manifest_part(part_rows, parts_dir=parts_dir, part_index=next_part_index)
        next_part_index += 1
    runtime_by_stage_seconds["build_manifest_rows_seconds"] = round(time.perf_counter() - manifest_stage_started, 3)

    write_stage_started = time.perf_counter()
    if parts_dir is not None:
        manifest = _read_manifest_parts(parts_dir)
    else:
        manifest = pd.DataFrame(part_rows if part_rows else [])
    required_columns = ["parcel_row_id", "parcel_id", "weak_building_label", "imagery_crop_label"]
    missing_columns = [column for column in required_columns if column not in manifest.columns]
    if missing_columns:
        raise ValueError(f"vacancy_ai_build_dataset_ms.build_dataset: missing required columns {missing_columns}")
    for column in required_columns:
        if manifest[column].isna().any():
            raise ValueError(f"vacancy_ai_build_dataset_ms.build_dataset: null values detected in {column}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)
    runtime_by_stage_seconds["write_manifest_seconds"] = round(time.perf_counter() - write_stage_started, 3)

    parcel_counts = manifest[["parcel_row_id", "weak_building_label"]].drop_duplicates()
    summary = {
        "manifest_path": str(output_path),
        "manifest_rows": int(len(manifest)),
        "manifest_parcels": int(parcel_counts["parcel_row_id"].nunique()),
        "build_start_timestamp": build_started,
        "build_end_timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_by_stage_seconds": runtime_by_stage_seconds,
        "model_version": MODEL_VERSION,
        "dataset_scope": "app_ready" if app_ready_only else "statewide_candidate",
        "tile_cache_dir": str(TILE_CACHE_DIR),
        "use_parcel_mask": bool(use_parcel_mask),
        "use_multi_tile_extent": bool(use_multi_tile_extent),
        "require_existing_tiles": bool(require_existing_tiles),
        "reuse_geometry_quality_artifact": bool(reuse_geometry_quality_artifact),
        "geometry_quality_artifact_path": str(geometry_quality_artifact_path),
        "limit": None if limit is None else int(limit),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "resume": bool(resume),
        "positive_target_parcels": int(positive_limit),
        "negative_target_parcels": int(negative_limit),
        "positive_collected_parcels": int(collected_parcels[1]),
        "negative_collected_parcels": int(collected_parcels[0]),
        "class_balance_rows": manifest["weak_building_label"].value_counts().sort_index().to_dict(),
        "class_balance_parcels": parcel_counts["weak_building_label"].value_counts().sort_index().to_dict(),
        "county_distribution_top_20": manifest["county_name"].astype("string").value_counts().head(20).to_dict(),
        "geometry_quality_prefilter_diagnostics": source.attrs.get("geometry_quality_prefilter_diagnostics", {}),
        "geometry_quality_postfilter_diagnostics": source.attrs.get("geometry_quality_postfilter_diagnostics", {}),
        "geometry_quality_flag_counts": manifest["geometry_quality_flag"].astype("string").value_counts().to_dict() if "geometry_quality_flag" in manifest.columns else {},
        "multipart_row_count": int(manifest["polygon_part_count"].fillna(0).gt(1).sum()),
        "multipart_parcel_count": int(
            manifest.loc[manifest["polygon_part_count"].fillna(0).gt(1), "parcel_row_id"].astype("string").nunique()
        ),
        "edge_tile_row_count": int(manifest["parcel_extent_exceeds_tile_flag"].fillna(False).sum()),
        "edge_tile_parcel_count": int(
            manifest.loc[manifest["parcel_extent_exceeds_tile_flag"].fillna(False), "parcel_row_id"].astype("string").nunique()
        ),
        "low_coverage_row_count": int(manifest["parcel_tile_low_coverage_flag"].fillna(False).sum()),
        "low_coverage_parcel_count": int(
            manifest.loc[manifest["parcel_tile_low_coverage_flag"].fillna(False), "parcel_row_id"].astype("string").nunique()
        ),
        "multi_tile_candidate_row_count": int(manifest["multi_tile_candidate_flag"].fillna(False).sum()),
        "multi_tile_candidate_parcel_count": int(
            manifest.loc[manifest["multi_tile_candidate_flag"].fillna(False), "parcel_row_id"].astype("string").nunique()
        ),
        "label_reliability_tier_counts": manifest["label_reliability_tier"].astype("string").value_counts().to_dict(),
        "tile_source_mode_counts": manifest["tile_source_mode"].astype("string").value_counts().to_dict(),
        "build_counters": dict(sorted(build_counters.items())),
    }
    write_metrics(Path(summary_output), summary)
    print(f"Wrote {len(manifest)} training rows across {parcel_counts['parcel_row_id'].nunique()} parcels to {output_path}")
    print(json.dumps(summary, indent=2))
    manifest.attrs["runtime_by_stage_seconds"] = runtime_by_stage_seconds
    manifest.attrs["build_start_timestamp"] = build_started
    manifest.attrs["build_end_timestamp"] = summary["build_end_timestamp"]
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a current Mississippi weak-label building-presence manifest using the shared parcel imagery path.")
    parser.add_argument("--output", default=str(TRAINING_MANIFEST_PATH))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--positive-limit", type=int, default=DEFAULT_POSITIVE_LIMIT)
    parser.add_argument("--negative-limit", type=int, default=DEFAULT_NEGATIVE_LIMIT)
    parser.add_argument("--tile-template", default=DEFAULT_TILE_URL_TEMPLATE)
    parser.add_argument("--use-parcel-mask", dest="use_parcel_mask", action="store_true", default=DEFAULT_USE_PARCEL_MASK)
    parser.add_argument("--no-parcel-mask", dest="use_parcel_mask", action="store_false")
    parser.add_argument("--outside-mask-fill", choices=["dim", "black"], default=DEFAULT_OUTSIDE_MASK_FILL)
    parser.add_argument("--outside-mask-dim-factor", type=float, default=DEFAULT_OUTSIDE_MASK_DIM_FACTOR)
    parser.add_argument("--parcel-buffer-pixels", type=int, default=DEFAULT_PARCEL_BUFFER_PIXELS)
    parser.add_argument("--use-multi-tile-extent", dest="use_multi_tile_extent", action="store_true", default=DEFAULT_USE_MULTI_TILE_EXTENT)
    parser.add_argument("--single-tile-only", dest="use_multi_tile_extent", action="store_false")
    parser.add_argument("--max-tiles-per-parcel", type=int, default=DEFAULT_MAX_TILES_PER_PARCEL)
    parser.add_argument("--min-primary-tile-coverage-ratio", type=float, default=DEFAULT_MIN_PRIMARY_TILE_COVERAGE_RATIO)
    parser.add_argument("--min-secondary-tile-coverage-ratio", type=float, default=DEFAULT_MIN_SECONDARY_TILE_COVERAGE_RATIO)
    parser.add_argument("--fetch-missing-tiles", dest="require_existing_tiles", action="store_false", default=True)
    parser.add_argument("--allow-statewide-candidates", dest="app_ready_only", action="store_false", default=True)
    parser.add_argument("--reuse-geometry-quality-artifact", dest="reuse_geometry_quality_artifact", action="store_true", default=True)
    parser.add_argument("--recompute-geometry-quality", dest="reuse_geometry_quality_artifact", action="store_false")
    parser.add_argument("--build-geometry-quality-artifact-if-missing", action="store_true")
    parser.add_argument("--geometry-quality-artifact-path", default=str(GEOMETRY_QUALITY_ARTIFACT_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-part-parcel-batch-size", type=int, default=DEFAULT_MANIFEST_PART_PARCEL_BATCH_SIZE)
    parser.add_argument("--progress-every-parcels", type=int, default=DEFAULT_PROGRESS_EVERY_PARCELS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        output=args.output,
        summary_output=args.summary_output,
        zoom=args.zoom,
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
        tile_template=args.tile_template,
        use_parcel_mask=args.use_parcel_mask,
        outside_mask_fill=args.outside_mask_fill,
        outside_mask_dim_factor=args.outside_mask_dim_factor,
        parcel_buffer_pixels=args.parcel_buffer_pixels,
        use_multi_tile_extent=args.use_multi_tile_extent,
        max_tiles_per_parcel=args.max_tiles_per_parcel,
        min_primary_tile_coverage_ratio=args.min_primary_tile_coverage_ratio,
        min_secondary_tile_coverage_ratio=args.min_secondary_tile_coverage_ratio,
        require_existing_tiles=args.require_existing_tiles,
        app_ready_only=args.app_ready_only,
        reuse_geometry_quality_artifact=args.reuse_geometry_quality_artifact,
        build_geometry_quality_artifact_if_missing=args.build_geometry_quality_artifact_if_missing,
        geometry_quality_artifact_path=Path(args.geometry_quality_artifact_path),
        limit=args.limit,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        resume=args.resume,
        manifest_part_parcel_batch_size=args.manifest_part_parcel_batch_size,
        progress_every_parcels=args.progress_every_parcels,
    )
