from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import shapely


ROOT = Path(__file__).resolve().parents[1]
PARCEL_MASTER_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
GEOMETRY_QUALITY_ARTIFACT_PATH = ROOT / "data" / "parcels" / "mississippi_parcel_geometry_quality.parquet"
GEOMETRY_QUALITY_SUMMARY_PATH = ROOT / "data" / "parcels" / "mississippi_parcel_geometry_quality_summary.json"
SQFT_PER_ACRE = 43560.0
SQFT_TO_SQM = 0.09290304
FEET_TO_METERS = 0.3048


GEOMETRY_QUALITY_COLUMNS = [
    "area_acres",
    "perimeter_meters",
    "bounding_box_width_meters",
    "bounding_box_height_meters",
    "aspect_ratio",
    "compactness",
    "is_multipart",
    "part_count",
    "geometry_quality_flag",
    "geometry_review_excluded_flag",
    "geometry_training_excluded_flag",
    "geometry_default_leads_excluded_flag",
]

GEOMETRY_QUALITY_ARTIFACT_COLUMNS = ["parcel_row_id", *GEOMETRY_QUALITY_COLUMNS]


@dataclass(frozen=True)
class GeometryQualityConfig:
    max_aspect_ratio_exclusion: float = 8.0
    min_compactness_exclusion: float = 0.15
    small_area_acres_threshold: float = 0.25
    small_area_aspect_ratio_exclusion: float = 4.0
    min_bounding_box_dimension_meters: float = 20.0
    multipart_complex_part_count_threshold: int = 3
    multipart_complex_max_area_acres: float = 2.0
    access_strip_aspect_ratio: float = 10.0
    access_strip_aspect_ratio_compactness: float = 6.0
    access_strip_compactness: float = 0.1
    irregular_compactness: float = 0.2


DEFAULT_GEOMETRY_QUALITY_CONFIG = GeometryQualityConfig()


def _utc_timestamp() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _polygon_part_count(shape: Any) -> int:
    if shape is None or getattr(shape, "is_empty", False):
        return 0
    geom_type = getattr(shape, "geom_type", None)
    if geom_type == "Polygon":
        return 1
    if geom_type == "MultiPolygon":
        return sum(_polygon_part_count(part) for part in getattr(shape, "geoms", ()))
    if geom_type == "GeometryCollection":
        return sum(_polygon_part_count(part) for part in getattr(shape, "geoms", ()))
    return 0


def _meters_per_degree_latitude(latitude_degrees: np.ndarray) -> np.ndarray:
    latitude_radians = np.deg2rad(latitude_degrees)
    return (
        111132.92
        - (559.82 * np.cos(2.0 * latitude_radians))
        + (1.175 * np.cos(4.0 * latitude_radians))
        - (0.0023 * np.cos(6.0 * latitude_radians))
    )


def _meters_per_degree_longitude(latitude_degrees: np.ndarray) -> np.ndarray:
    latitude_radians = np.deg2rad(latitude_degrees)
    return (
        (111412.84 * np.cos(latitude_radians))
        - (93.5 * np.cos(3.0 * latitude_radians))
        + (0.118 * np.cos(5.0 * latitude_radians))
    )


def _safe_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)


def _frame_bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(bool)
    return pd.Series(default, index=frame.index, dtype="bool")


def _frame_string_series(frame: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column in frame.columns:
        return frame[column].astype("string").fillna(default)
    return pd.Series(default, index=frame.index, dtype="string")


def _classify_geometry_quality(
    *,
    area_acres: np.ndarray,
    aspect_ratio: np.ndarray,
    compactness: np.ndarray,
    width_meters: np.ndarray,
    height_meters: np.ndarray,
    part_count: np.ndarray,
    config: GeometryQualityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    is_multipart = part_count > 1
    multipart_complex = (
        is_multipart
        & (part_count > int(config.multipart_complex_part_count_threshold))
        & np.isfinite(area_acres)
        & (area_acres < float(config.multipart_complex_max_area_acres))
    )
    dimension_sliver = (
        (np.isfinite(width_meters) & (width_meters < float(config.min_bounding_box_dimension_meters)))
        | (np.isfinite(height_meters) & (height_meters < float(config.min_bounding_box_dimension_meters)))
    )
    small_area_sliver = (
        np.isfinite(area_acres)
        & np.isfinite(aspect_ratio)
        & (area_acres < float(config.small_area_acres_threshold))
        & (aspect_ratio > float(config.small_area_aspect_ratio_exclusion))
    )
    access_strip = (
        (np.isfinite(aspect_ratio) & (aspect_ratio > float(config.access_strip_aspect_ratio)))
        | (
            np.isfinite(aspect_ratio)
            & np.isfinite(compactness)
            & (aspect_ratio > float(config.access_strip_aspect_ratio_compactness))
            & (compactness < float(config.access_strip_compactness))
        )
        | dimension_sliver
        | small_area_sliver
    )
    irregular = (
        (np.isfinite(compactness) & (compactness < float(config.irregular_compactness)))
        | is_multipart
        | (np.isfinite(aspect_ratio) & (aspect_ratio > float(config.max_aspect_ratio_exclusion)))
        | dimension_sliver
    )

    geometry_quality_flag = np.full(len(area_acres), "good", dtype=object)
    geometry_quality_flag[irregular] = "irregular"
    geometry_quality_flag[access_strip] = "access_strip"
    geometry_quality_flag[multipart_complex] = "multipart_complex"

    review_excluded = (
        multipart_complex
        | access_strip
        | irregular
        | (np.isfinite(compactness) & (compactness < float(config.min_compactness_exclusion)))
        | (np.isfinite(aspect_ratio) & (aspect_ratio > float(config.max_aspect_ratio_exclusion)))
    )
    training_excluded = access_strip | multipart_complex
    default_leads_excluded = access_strip
    return geometry_quality_flag, review_excluded, training_excluded, default_leads_excluded


def add_geometry_quality_fields(
    frame: pd.DataFrame,
    *,
    geometry_column: str = "geometry",
    area_column: str = "shape_area",
    perimeter_column: str = "shape_length",
    config: GeometryQualityConfig | None = None,
    chunk_size: int = 50000,
) -> pd.DataFrame:
    if geometry_column not in frame.columns:
        raise ValueError(f"parcel_geometry_quality_ms.add_geometry_quality_fields: missing geometry column {geometry_column}")
    config = config or DEFAULT_GEOMETRY_QUALITY_CONFIG
    enriched = frame.copy()
    metric_parts: list[pd.DataFrame] = []

    for start in range(0, len(enriched), max(int(chunk_size), 1)):
        chunk = enriched.iloc[start : start + max(int(chunk_size), 1)].copy()
        raw_geometry = chunk[geometry_column].to_numpy(copy=False)
        geometries = shapely.from_wkb(raw_geometry)
        bounds = shapely.bounds(geometries)
        min_x = bounds[:, 0]
        min_y = bounds[:, 1]
        max_x = bounds[:, 2]
        max_y = bounds[:, 3]
        center_latitude = (min_y + max_y) / 2.0
        width_meters = (max_x - min_x) * _meters_per_degree_longitude(center_latitude)
        height_meters = (max_y - min_y) * _meters_per_degree_latitude(center_latitude)
        width_meters = np.abs(width_meters)
        height_meters = np.abs(height_meters)

        area_sqft = _safe_float_array(chunk[area_column]) if area_column in chunk.columns else np.full(len(chunk), np.nan, dtype="float64")
        perimeter_ft = _safe_float_array(chunk[perimeter_column]) if perimeter_column in chunk.columns else np.full(len(chunk), np.nan, dtype="float64")
        area_acres = np.where(area_sqft > 0, area_sqft / SQFT_PER_ACRE, np.nan)
        area_sqm = np.where(area_sqft > 0, area_sqft * SQFT_TO_SQM, np.nan)
        perimeter_meters = np.where(perimeter_ft > 0, perimeter_ft * FEET_TO_METERS, np.nan)
        compactness = np.full(len(chunk), np.nan, dtype="float64")
        valid_compactness = np.isfinite(area_sqm) & np.isfinite(perimeter_meters) & (area_sqm > 0) & (perimeter_meters > 0)
        compactness[valid_compactness] = np.clip(
            (4.0 * math.pi * area_sqm[valid_compactness]) / np.square(perimeter_meters[valid_compactness]),
            0.0,
            1.0,
        )
        smallest_dimension = np.minimum(width_meters, height_meters)
        largest_dimension = np.maximum(width_meters, height_meters)
        aspect_ratio = np.where(smallest_dimension > 0, largest_dimension / smallest_dimension, np.nan)
        part_count = np.fromiter((_polygon_part_count(shape) for shape in geometries), dtype=np.int32, count=len(chunk))
        is_multipart = part_count > 1
        geometry_quality_flag, review_excluded, training_excluded, default_leads_excluded = _classify_geometry_quality(
            area_acres=area_acres,
            aspect_ratio=aspect_ratio,
            compactness=compactness,
            width_meters=width_meters,
            height_meters=height_meters,
            part_count=part_count,
            config=config,
        )
        metric_parts.append(
            pd.DataFrame(
                {
                    "area_acres": area_acres,
                    "perimeter_meters": perimeter_meters,
                    "bounding_box_width_meters": width_meters,
                    "bounding_box_height_meters": height_meters,
                    "aspect_ratio": aspect_ratio,
                    "compactness": compactness,
                    "is_multipart": is_multipart,
                    "part_count": part_count,
                    "geometry_quality_flag": pd.Series(geometry_quality_flag, dtype="string"),
                    "geometry_review_excluded_flag": review_excluded,
                    "geometry_training_excluded_flag": training_excluded,
                    "geometry_default_leads_excluded_flag": default_leads_excluded,
                },
                index=chunk.index,
            )
        )

    metrics = pd.concat(metric_parts).sort_index()
    for column in GEOMETRY_QUALITY_COLUMNS:
        enriched[column] = metrics[column]
    return enriched


def geometry_quality_artifact_exists(path: Path = GEOMETRY_QUALITY_ARTIFACT_PATH) -> bool:
    return path.exists()


def read_geometry_quality_artifact(
    *,
    parcel_row_ids: pd.Series | pd.Index | list[str] | None = None,
    artifact_path: Path = GEOMETRY_QUALITY_ARTIFACT_PATH,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if not artifact_path.exists():
        return pd.DataFrame(columns=columns or GEOMETRY_QUALITY_ARTIFACT_COLUMNS)
    available_columns = pq.read_schema(artifact_path).names
    selected_columns = [column for column in (columns or GEOMETRY_QUALITY_ARTIFACT_COLUMNS) if column in available_columns]
    if "parcel_row_id" not in selected_columns:
        selected_columns = ["parcel_row_id", *selected_columns]
    if parcel_row_ids is None:
        frame = pd.read_parquet(artifact_path, columns=selected_columns, engine="pyarrow")
    else:
        normalized_ids = pd.Index(parcel_row_ids, dtype="string").dropna().unique()
        if normalized_ids.empty:
            return pd.DataFrame(columns=selected_columns)
        dataset = ds.dataset(artifact_path, format="parquet")
        chunks: list[pd.DataFrame] = []
        step = 5000
        for start in range(0, len(normalized_ids), step):
            requested_ids = normalized_ids[start : start + step].tolist()
            table = dataset.to_table(columns=selected_columns, filter=ds.field("parcel_row_id").isin(requested_ids))
            if table.num_rows:
                chunks.append(table.to_pandas())
        if not chunks:
            return pd.DataFrame(columns=selected_columns)
        frame = pd.concat(chunks, ignore_index=True)
    frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
    return frame


def build_geometry_quality_artifact(
    *,
    output_path: Path = GEOMETRY_QUALITY_ARTIFACT_PATH,
    summary_output_path: Path = GEOMETRY_QUALITY_SUMMARY_PATH,
    config: GeometryQualityConfig | None = None,
    chunk_size: int = 50000,
    limit: int | None = None,
    force: bool = False,
    log_every_batches: int = 10,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force and limit is None:
        frame = pd.read_parquet(output_path, engine="pyarrow")
        summary = {
            "artifact_path": str(output_path),
            "summary_path": str(summary_output_path),
            "reused_existing_artifact": True,
            "build_start_timestamp": _utc_timestamp(),
            "build_end_timestamp": _utc_timestamp(),
            "runtime_by_stage_seconds": {"artifact_read_seconds": 0.0},
            **geometry_quality_diagnostics(frame, config=config),
        }
        _write_json(summary_output_path, summary)
        return summary

    dataset = ds.dataset(PARCEL_MASTER_PATH, format="parquet")
    scanner = dataset.scanner(
        columns=["parcel_row_id", "county_name", "shape_area", "shape_length", "geometry"],
        batch_size=max(int(chunk_size), 1),
    )
    started_at = time.perf_counter()
    build_start_timestamp = _utc_timestamp()
    writer: pq.ParquetWriter | None = None
    processed_rows = 0
    batches_written = 0
    read_seconds = 0.0
    compute_seconds = 0.0
    write_seconds = 0.0
    county_review_excluded: Counter[str] = Counter()
    county_training_excluded: Counter[str] = Counter()
    county_default_excluded: Counter[str] = Counter()
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temp_output_path.unlink(missing_ok=True)

    try:
        for batch_index, batch in enumerate(scanner.to_batches(), start=1):
            batch_read_started = time.perf_counter()
            chunk = batch.to_pandas()
            read_seconds += time.perf_counter() - batch_read_started
            if chunk.empty:
                continue
            if limit is not None and processed_rows >= int(limit):
                break
            if limit is not None:
                remaining = int(limit) - processed_rows
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining).copy()

            compute_started = time.perf_counter()
            enriched = add_geometry_quality_fields(
                chunk,
                config=config,
                chunk_size=max(int(chunk_size), 1),
            ).loc[:, ["parcel_row_id", "county_name", *GEOMETRY_QUALITY_COLUMNS]]
            compute_seconds += time.perf_counter() - compute_started

            county_names = enriched["county_name"].astype("string").fillna("unknown")
            review_mask = enriched["geometry_review_excluded_flag"].fillna(False).astype(bool)
            training_mask = enriched["geometry_training_excluded_flag"].fillna(False).astype(bool)
            default_mask = enriched["geometry_default_leads_excluded_flag"].fillna(False).astype(bool)
            county_review_excluded.update(county_names.loc[review_mask].value_counts().to_dict())
            county_training_excluded.update(county_names.loc[training_mask].value_counts().to_dict())
            county_default_excluded.update(county_names.loc[default_mask].value_counts().to_dict())

            write_started = time.perf_counter()
            write_frame = enriched.loc[:, GEOMETRY_QUALITY_ARTIFACT_COLUMNS].copy()
            table = pa.Table.from_pandas(write_frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp_output_path, table.schema, compression="zstd")
            writer.write_table(table)
            write_seconds += time.perf_counter() - write_started
            processed_rows += len(write_frame)
            batches_written += 1

            if log_every_batches > 0 and batch_index % int(log_every_batches) == 0:
                print(
                    f"[geometry-quality] batches={batch_index} rows={processed_rows:,} "
                    f"elapsed={time.perf_counter() - started_at:.1f}s"
                )
    finally:
        if writer is not None:
            writer.close()

    if temp_output_path.exists():
        temp_output_path.replace(output_path)
    artifact_frame = pd.read_parquet(output_path, engine="pyarrow") if output_path.exists() else pd.DataFrame(columns=GEOMETRY_QUALITY_ARTIFACT_COLUMNS)
    diagnostics = geometry_quality_diagnostics(artifact_frame, config=config)
    summary = {
        "artifact_path": str(output_path),
        "summary_path": str(summary_output_path),
        "reused_existing_artifact": False,
        "build_start_timestamp": build_start_timestamp,
        "build_end_timestamp": _utc_timestamp(),
        "runtime_by_stage_seconds": {
            "parcel_scan_read_seconds": round(read_seconds, 3),
            "geometry_quality_compute_seconds": round(compute_seconds, 3),
            "artifact_write_seconds": round(write_seconds, 3),
            "total_seconds": round(time.perf_counter() - started_at, 3),
        },
        "processed_rows": int(processed_rows),
        "batches_written": int(batches_written),
        "top_counties_by_review_excluded_count": [{"county_name": key, "count": int(value)} for key, value in county_review_excluded.most_common(20)],
        "top_counties_by_training_excluded_count": [{"county_name": key, "count": int(value)} for key, value in county_training_excluded.most_common(20)],
        "top_counties_by_default_excluded_count": [{"county_name": key, "count": int(value)} for key, value in county_default_excluded.most_common(20)],
        **diagnostics,
    }
    _write_json(summary_output_path, summary)
    return summary


def load_geometry_quality_frame(
    parcel_row_ids: pd.Series | pd.Index | list[str] | None = None,
    *,
    config: GeometryQualityConfig | None = None,
    chunk_size: int = 50000,
    reuse_artifact: bool = True,
    artifact_path: Path = GEOMETRY_QUALITY_ARTIFACT_PATH,
    build_artifact_if_missing: bool = False,
    summary_output_path: Path = GEOMETRY_QUALITY_SUMMARY_PATH,
) -> pd.DataFrame:
    if reuse_artifact and artifact_path.exists():
        try:
            return read_geometry_quality_artifact(parcel_row_ids=parcel_row_ids, artifact_path=artifact_path)
        except Exception:
            if not build_artifact_if_missing:
                raise
            artifact_path.unlink(missing_ok=True)
    if reuse_artifact and build_artifact_if_missing:
        build_geometry_quality_artifact(
            output_path=artifact_path,
            summary_output_path=summary_output_path,
            config=config,
            chunk_size=chunk_size,
        )
        return read_geometry_quality_artifact(parcel_row_ids=parcel_row_ids, artifact_path=artifact_path)

    dataset = ds.dataset(PARCEL_MASTER_PATH, format="parquet")
    columns = ["parcel_row_id", "shape_area", "shape_length", "geometry"]
    chunks: list[pd.DataFrame] = []
    normalized_ids = pd.Index([], dtype="string") if parcel_row_ids is None else pd.Index(parcel_row_ids, dtype="string").dropna().unique()
    if parcel_row_ids is None:
        scanner = dataset.scanner(columns=columns, batch_size=max(int(chunk_size), 1))
        for batch in scanner.to_batches():
            chunk = batch.to_pandas()
            chunks.append(
                add_geometry_quality_fields(
                    chunk,
                    config=config,
                    chunk_size=max(int(chunk_size), 1),
                ).loc[:, GEOMETRY_QUALITY_ARTIFACT_COLUMNS]
            )
    elif not normalized_ids.empty:
        step = 1000
        for start in range(0, len(normalized_ids), step):
            requested_ids = normalized_ids[start : start + step].tolist()
            table = dataset.to_table(columns=columns, filter=ds.field("parcel_row_id").isin(requested_ids))
            if table.num_rows == 0:
                continue
            chunk = table.to_pandas()
            chunks.append(
                add_geometry_quality_fields(
                    chunk,
                    config=config,
                    chunk_size=max(int(chunk_size), 1),
                ).loc[:, GEOMETRY_QUALITY_ARTIFACT_COLUMNS]
            )
    if not chunks:
        return pd.DataFrame(columns=GEOMETRY_QUALITY_ARTIFACT_COLUMNS)
    merged = pd.concat(chunks, ignore_index=True)
    merged["parcel_row_id"] = merged["parcel_row_id"].astype("string")
    return merged


def geometry_quality_diagnostics(
    frame: pd.DataFrame,
    *,
    config: GeometryQualityConfig | None = None,
) -> dict[str, Any]:
    config = config or DEFAULT_GEOMETRY_QUALITY_CONFIG

    def distribution(series: pd.Series) -> dict[str, float | None]:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return {"min": None, "p50": None, "p75": None, "p90": None, "p95": None, "max": None}
        return {
            "min": round(float(numeric.min()), 3),
            "p50": round(float(numeric.quantile(0.50)), 3),
            "p75": round(float(numeric.quantile(0.75)), 3),
            "p90": round(float(numeric.quantile(0.90)), 3),
            "p95": round(float(numeric.quantile(0.95)), 3),
            "max": round(float(numeric.max()), 3),
        }

    row_count = int(len(frame))
    review_excluded = int(_frame_bool_series(frame, "geometry_review_excluded_flag").sum())
    training_excluded = int(_frame_bool_series(frame, "geometry_training_excluded_flag").sum())
    default_excluded = int(_frame_bool_series(frame, "geometry_default_leads_excluded_flag").sum())
    flag_counts = _frame_string_series(frame, "geometry_quality_flag").value_counts(dropna=False).sort_index().to_dict()
    return {
        "config": asdict(config),
        "row_count": row_count,
        "geometry_quality_flag_counts": {str(key): int(value) for key, value in flag_counts.items()},
        "review_excluded_count": review_excluded,
        "review_excluded_pct": round((review_excluded / row_count) * 100.0, 2) if row_count else 0.0,
        "training_excluded_count": training_excluded,
        "training_excluded_pct": round((training_excluded / row_count) * 100.0, 2) if row_count else 0.0,
        "default_leads_excluded_count": default_excluded,
        "default_leads_excluded_pct": round((default_excluded / row_count) * 100.0, 2) if row_count else 0.0,
        "aspect_ratio_distribution": distribution(pd.Series(frame.get("aspect_ratio"))),
        "compactness_distribution": distribution(pd.Series(frame.get("compactness"))),
        "area_acres_distribution": distribution(pd.Series(frame.get("area_acres"))),
    }


def filter_review_geometry_frame(
    frame: pd.DataFrame,
    *,
    qa_bad_geometry_pct: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    good = frame.loc[_frame_string_series(frame, "geometry_quality_flag", "good").eq("good")].copy()
    if qa_bad_geometry_pct <= 0:
        good.attrs["geometry_quality_diagnostics"] = geometry_quality_diagnostics(frame)
        return good
    bad = frame.loc[~_frame_string_series(frame, "geometry_quality_flag", "good").eq("good")].copy()
    qa_count = min(len(bad), int(math.ceil(len(frame) * float(qa_bad_geometry_pct))))
    if qa_count > 0:
        qa_rows = bad.sample(n=qa_count, random_state=seed).copy()
        qa_rows["sample_group"] = qa_rows.get("sample_group", pd.Series(index=qa_rows.index, dtype="string")).fillna("qa")
        qa_rows["sample_reason"] = qa_rows.get("sample_reason", pd.Series(index=qa_rows.index, dtype="string")).fillna("geometry_filter_qa")
        qa_rows["review_priority"] = qa_rows.get("review_priority", pd.Series(index=qa_rows.index, dtype="string")).fillna("low")
        filtered = pd.concat([good, qa_rows], ignore_index=False)
    else:
        filtered = good
    filtered.attrs["geometry_quality_diagnostics"] = geometry_quality_diagnostics(frame)
    return filtered


def filter_training_geometry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.loc[~_frame_bool_series(frame, "geometry_training_excluded_flag")].copy()
    filtered.attrs["geometry_quality_diagnostics"] = geometry_quality_diagnostics(frame)
    return filtered


def filter_default_leads_geometry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.loc[~_frame_bool_series(frame, "geometry_default_leads_excluded_flag")].copy()
    filtered.attrs["geometry_quality_diagnostics"] = geometry_quality_diagnostics(frame)
    return filtered
