from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
from PIL import Image, ImageDraw
from shapely import wkb
from shapely.geometry import box as geometry_box

try:
    from floodscraper.parcel_contract_ms import (
        CANONICAL_PARCEL_FIELDS,
        CANONICAL_REQUIRED_NON_NULL_FIELDS,
        validate_required_columns,
    )
    from floodscraper.state_artifacts import load_state_artifacts
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from parcel_contract_ms import (
        CANONICAL_PARCEL_FIELDS,
        CANONICAL_REQUIRED_NON_NULL_FIELDS,
        validate_required_columns,
    )
    from state_artifacts import load_state_artifacts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_ARTIFACTS = load_state_artifacts("ms")
PARCEL_MASTER_PATH = DEFAULT_STATE_ARTIFACTS.parcel_master_path
BUILDING_METRICS_PATH = DEFAULT_STATE_ARTIFACTS.parcel_building_metrics_path
AI_DATA_DIR = DEFAULT_STATE_ARTIFACTS.ai_training_manifest_path.parent
APP_READY_PATH = DEFAULT_STATE_ARTIFACTS.app_ready_path
BACKEND_RUNTIME_DIR = DEFAULT_STATE_ARTIFACTS.runtime_root
TRAINING_MANIFEST_PATH = DEFAULT_STATE_ARTIFACTS.ai_training_manifest_path
MODEL_PATH = DEFAULT_STATE_ARTIFACTS.ai_model_path
MODEL_METRICS_PATH = DEFAULT_STATE_ARTIFACTS.ai_model_metrics_path
MODEL_PARAMS_PATH = DEFAULT_STATE_ARTIFACTS.ai_runtime_model_params_path
RUNTIME_MODEL_METRICS_PATH = DEFAULT_STATE_ARTIFACTS.ai_runtime_model_metrics_path
PREDICTIONS_PATH = DEFAULT_STATE_ARTIFACTS.ai_predictions_path
TILE_CACHE_DIR = DEFAULT_STATE_ARTIFACTS.ai_tile_cache_dir
DEFAULT_TILE_URL_TEMPLATE = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
MODEL_VERSION = "ms_building_presence_v2_multi_crop"
DEFAULT_USE_PARCEL_MASK = True
DEFAULT_OUTSIDE_MASK_FILL = "black"
DEFAULT_OUTSIDE_MASK_DIM_FACTOR = 0.0
DEFAULT_PARCEL_BUFFER_PIXELS = 8
DEFAULT_PARCEL_MIN_CROP_PIXELS = 48
DEFAULT_MULTI_TILE_CANVAS_PADDING_PIXELS = 24
DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD = 82.0
DEFAULT_LOW_COVERAGE_TILE_THRESHOLD = 0.75
DEFAULT_LOW_COVERAGE_BBOX_THRESHOLD = 0.85
DEFAULT_TILE_NEGATIVE_CONFIDENCE_THRESHOLD = 40.0


@dataclass
class TileAddress:
    x: int
    y: int
    z: int


def load_candidate_frame() -> pd.DataFrame:
    parcels = pd.read_parquet(
        PARCEL_MASTER_PATH,
        columns=["parcel_row_id", "parcel_id", "state_code", "county_name", "county_fips", "latitude", "longitude", "total_acres", "parcel_area_acres", "gis_acres", "tax_acres"],
        engine="pyarrow",
    )
    buildings = pd.read_parquet(
        BUILDING_METRICS_PATH,
        columns=["parcel_row_id", "building_count", "building_area_total", "parcel_vacant_flag"],
        engine="pyarrow",
    )
    frame = parcels.merge(buildings, on="parcel_row_id", how="left")
    frame["county_name"] = frame["county_name"].astype("string").str.strip().str.lower()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["acreage"] = (
        pd.to_numeric(frame["total_acres"], errors="coerce")
        .fillna(pd.to_numeric(frame["parcel_area_acres"], errors="coerce"))
        .fillna(pd.to_numeric(frame["gis_acres"], errors="coerce"))
        .fillna(pd.to_numeric(frame["tax_acres"], errors="coerce"))
    )
    frame["building_count"] = pd.to_numeric(frame["building_count"], errors="coerce").fillna(0)
    frame["building_area_total"] = pd.to_numeric(frame["building_area_total"], errors="coerce").fillna(0)
    frame["parcel_vacant_flag"] = frame["parcel_vacant_flag"].fillna(False)
    frame = frame.loc[frame["latitude"].notna() & frame["longitude"].notna()].copy()
    validate_required_columns(
        frame,
        required_columns=CANONICAL_PARCEL_FIELDS,
        non_null_columns=CANONICAL_REQUIRED_NON_NULL_FIELDS,
        context="vacancy_ai_common.load_candidate_frame",
    )
    return frame


def load_parcel_geometry_lookup(parcel_row_ids: list[str] | pd.Series | pd.Index) -> dict[str, bytes]:
    normalized_ids = pd.Index(parcel_row_ids, dtype="string").dropna().unique()
    if normalized_ids.empty:
        return {}
    dataset = ds.dataset(PARCEL_MASTER_PATH, format="parquet")
    rows: list[pd.DataFrame] = []
    chunk_size = 1000
    for start in range(0, len(normalized_ids), chunk_size):
        chunk = normalized_ids[start : start + chunk_size].tolist()
        table = dataset.to_table(columns=["parcel_row_id", "geometry"], filter=ds.field("parcel_row_id").isin(chunk))
        if table.num_rows:
            rows.append(table.to_pandas())
    if not rows:
        return {}
    frame = pd.concat(rows, ignore_index=True)
    return {str(row["parcel_row_id"]): row["geometry"] for _, row in frame.iterrows() if pd.notna(row["geometry"])}


def load_app_ready_row_ids() -> pd.Index:
    if not APP_READY_PATH.exists():
        return pd.Index([], dtype="string")
    frame = pd.read_parquet(APP_READY_PATH, columns=["parcel_row_id"], engine="pyarrow")
    return frame["parcel_row_id"].astype("string")


def weak_label_frame(frame: pd.DataFrame) -> pd.DataFrame:
    positive = frame["building_count"].ge(1) & frame["building_area_total"].ge(400)
    negative = frame["parcel_vacant_flag"].fillna(False) & frame["building_count"].eq(0) & frame["building_area_total"].le(0)
    labeled = frame.loc[positive | negative].copy()
    labeled["weak_building_label"] = np.where(positive.loc[labeled.index], 1, 0)
    return labeled


def centroid_tile(longitude: float, latitude: float, zoom: int) -> TileAddress:
    lat_rad = math.radians(latitude)
    n = 2**zoom
    xtile = int((longitude + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return TileAddress(x=xtile, y=ytile, z=zoom)


def tile_url(address: TileAddress, template: str = DEFAULT_TILE_URL_TEMPLATE) -> str:
    return template.format(z=address.z, x=address.x, y=address.y)


def tile_cache_path(parcel_row_id: str, county_name: str | None, zoom: int, suffix: str = ".jpg") -> Path:
    county = (county_name or "unknown").strip().lower()
    return TILE_CACHE_DIR / county / f"{parcel_row_id}_z{zoom}{suffix}"


def tile_cache_path_for_address(
    parcel_row_id: str,
    county_name: str | None,
    address: TileAddress,
    *,
    centroid_address: TileAddress | None = None,
    suffix: str = ".jpg",
) -> Path:
    if centroid_address is not None and (address.x, address.y, address.z) == (centroid_address.x, centroid_address.y, centroid_address.z):
        return tile_cache_path(parcel_row_id, county_name, address.z, suffix=suffix)
    county = (county_name or "unknown").strip().lower()
    return TILE_CACHE_DIR / county / f"{parcel_row_id}_z{address.z}_x{address.x}_y{address.y}{suffix}"


def fetch_tile_image(url: str, timeout: int = 20) -> Image.Image:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def ensure_tile_image(
    *,
    parcel_row_id: str,
    county_name: str | None,
    longitude: float,
    latitude: float,
    zoom: int,
    refresh: bool = False,
    template: str = DEFAULT_TILE_URL_TEMPLATE,
) -> tuple[Path, TileAddress]:
    address = centroid_tile(longitude, latitude, zoom)
    path = tile_cache_path_for_address(parcel_row_id, county_name, address, centroid_address=address)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        image = fetch_tile_image(tile_url(address, template))
        image.save(path, format="JPEG", quality=88)
    return path, address


def ensure_tile_image_for_address(
    *,
    parcel_row_id: str,
    county_name: str | None,
    address: TileAddress,
    refresh: bool = False,
    template: str = DEFAULT_TILE_URL_TEMPLATE,
    centroid_address: TileAddress | None = None,
) -> Path:
    path = tile_cache_path_for_address(parcel_row_id, county_name, address, centroid_address=centroid_address)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        image = fetch_tile_image(tile_url(address, template))
        image.save(path, format="JPEG", quality=88)
    return path


def load_tile_image(image_source: Path | Image.Image) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    return Image.open(image_source).convert("RGB")


def load_geometry_shape(geometry_value: bytes | bytearray | memoryview | None) -> Any | None:
    if geometry_value is None:
        return None
    try:
        return wkb.loads(bytes(geometry_value))
    except Exception:
        return None


def crop_specs_for_acreage(acreage: float | None) -> list[tuple[str, tuple[int, int, int, int]]]:
    # Smaller, shifted crops help find off-center rural houses that disappear in full-tile scoring.
    specs = [
        ("tile_full", (0, 0, 256, 256)),
        ("center_tight", (48, 48, 208, 208)),
        ("northwest", (0, 0, 160, 160)),
        ("northeast", (96, 0, 256, 160)),
        ("southwest", (0, 96, 160, 256)),
        ("southeast", (96, 96, 256, 256)),
    ]
    if acreage is not None and float(acreage) >= 5:
        specs.extend(
            [
                ("north_center", (48, 0, 208, 160)),
                ("south_center", (48, 96, 208, 256)),
                ("west_center", (0, 48, 160, 208)),
                ("east_center", (96, 48, 256, 208)),
            ]
        )
    return specs


def _tile_pixel(longitude: float, latitude: float, address: TileAddress, tile_size: int = 256) -> tuple[float, float]:
    lat_rad = math.radians(latitude)
    n = 2**address.z
    world_x = ((longitude + 180.0) / 360.0) * n * tile_size
    world_y = ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0) * n * tile_size
    return world_x - (address.x * tile_size), world_y - (address.y * tile_size)


def _tile_canvas_pixel(
    longitude: float,
    latitude: float,
    *,
    origin_x: int,
    origin_y: int,
    zoom: int,
    tile_size: int = 256,
) -> tuple[float, float]:
    lat_rad = math.radians(latitude)
    n = 2**zoom
    world_x = ((longitude + 180.0) / 360.0) * n * tile_size
    world_y = ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0) * n * tile_size
    return world_x - (origin_x * tile_size), world_y - (origin_y * tile_size)


def _tile_longitude(tile_x: float, zoom: int) -> float:
    n = 2**zoom
    return (float(tile_x) / n) * 360.0 - 180.0


def _tile_latitude(tile_y: float, zoom: int) -> float:
    n = 2**zoom
    mercator = math.pi - ((2.0 * math.pi * float(tile_y)) / n)
    return math.degrees(math.atan(math.sinh(mercator)))


def tile_bounds(address: TileAddress) -> tuple[float, float, float, float]:
    west = _tile_longitude(address.x, address.z)
    east = _tile_longitude(address.x + 1, address.z)
    north = _tile_latitude(address.y, address.z)
    south = _tile_latitude(address.y + 1, address.z)
    return (west, south, east, north)


def tile_set_bounds(addresses: list[TileAddress]) -> tuple[float, float, float, float] | None:
    normalized_addresses = list(addresses)
    if not normalized_addresses:
        return None
    zooms = {int(address.z) for address in normalized_addresses}
    if len(zooms) != 1:
        raise ValueError("tile_set_bounds requires all tile addresses to share the same zoom level")
    min_x = min(int(address.x) for address in normalized_addresses)
    max_x = max(int(address.x) for address in normalized_addresses)
    min_y = min(int(address.y) for address in normalized_addresses)
    max_y = max(int(address.y) for address in normalized_addresses)
    zoom = next(iter(zooms))
    west = _tile_longitude(min_x, zoom)
    east = _tile_longitude(max_x + 1, zoom)
    north = _tile_latitude(min_y, zoom)
    south = _tile_latitude(max_y + 1, zoom)
    return (west, south, east, north)


def tile_label(address: TileAddress) -> str:
    return f"z{address.z}_x{address.x}_y{address.y}"


def tile_coordinate(address: TileAddress) -> str:
    return f"{address.z}/{address.x}/{address.y}"


def polygon_parts_from_shape(shape: Any | None) -> list[Any]:
    if shape is None or shape.is_empty:
        return []
    geom_type = getattr(shape, "geom_type", None)
    if geom_type == "Polygon":
        return [shape]
    if geom_type == "MultiPolygon":
        return [part for part in getattr(shape, "geoms", []) if not part.is_empty]
    if hasattr(shape, "geoms"):
        polygon_parts: list[Any] = []
        for part in getattr(shape, "geoms", []):
            polygon_parts.extend(polygon_parts_from_shape(part))
        return polygon_parts
    return []


def _polygon_parts_bounds(parts: list[Any]) -> list[float] | None:
    if not parts:
        return None
    min_x = min(float(part.bounds[0]) for part in parts)
    min_y = min(float(part.bounds[1]) for part in parts)
    max_x = max(float(part.bounds[2]) for part in parts)
    max_y = max(float(part.bounds[3]) for part in parts)
    return [round(value, 8) for value in (min_x, min_y, max_x, max_y)]


def clip_shape_to_tile(shape: Any | None, address: TileAddress) -> Any | None:
    if shape is None or shape.is_empty:
        return None
    clipped_shape = shape.intersection(geometry_box(*tile_bounds(address)))
    if clipped_shape.is_empty:
        return None
    return clipped_shape


def clip_shape_to_tile_set(shape: Any | None, addresses: list[TileAddress]) -> Any | None:
    if shape is None or shape.is_empty:
        return None
    bounds = tile_set_bounds(addresses)
    if bounds is None:
        return None
    clipped_shape = shape.intersection(geometry_box(*bounds))
    if clipped_shape.is_empty:
        return None
    return clipped_shape


def parcel_geometry_debug_metadata(shape: Any | None, clipped_shape: Any | None) -> dict[str, Any]:
    polygon_parts = polygon_parts_from_shape(shape)
    clipped_polygon_parts = polygon_parts_from_shape(clipped_shape)
    return {
        "original_geom_type": getattr(shape, "geom_type", None),
        "clipped_geom_type": getattr(clipped_shape, "geom_type", None) if clipped_shape is not None else None,
        "polygon_part_count": len(polygon_parts),
        "clipped_polygon_part_count": len(clipped_polygon_parts),
        "bounds_before_clip": _polygon_parts_bounds(polygon_parts),
        "bounds_after_clip": _polygon_parts_bounds(clipped_polygon_parts),
    }


def _tile_address_ranges(bounds: tuple[float, float, float, float], zoom: int) -> tuple[range, range]:
    west, south, east, north = bounds
    max_latitude = 85.05112878
    n = 2**zoom

    clamped_west = float(np.clip(west, -180.0, 180.0))
    clamped_east = float(np.clip(east, -180.0, 180.0))
    clamped_south = float(np.clip(south, -max_latitude, max_latitude))
    clamped_north = float(np.clip(north, -max_latitude, max_latitude))

    west_index = int(math.floor(((clamped_west + 180.0) / 360.0) * n))
    east_fraction = ((clamped_east + 180.0) / 360.0) * n
    east_index = int(math.floor(max(east_fraction - 1e-9, 0.0)))

    north_fraction = ((1.0 - math.asinh(math.tan(math.radians(clamped_north))) / math.pi) / 2.0) * n
    south_fraction = ((1.0 - math.asinh(math.tan(math.radians(clamped_south))) / math.pi) / 2.0) * n
    north_index = int(math.floor(min(north_fraction, south_fraction)))
    south_index = int(math.floor(max(north_fraction, south_fraction) - 1e-9))

    west_index = max(0, min(west_index, n - 1))
    east_index = max(0, min(east_index, n - 1))
    north_index = max(0, min(north_index, n - 1))
    south_index = max(0, min(south_index, n - 1))

    return range(west_index, east_index + 1), range(north_index, south_index + 1)


def parcel_covering_tiles(geometry_value: bytes | bytearray | memoryview | None, *, zoom: int) -> list[TileAddress]:
    return [item["address"] for item in parcel_covering_tile_records(geometry_value, zoom=zoom)]


def _tile_coverage_diagnostics_for_shape(shape: Any | None, address: TileAddress) -> dict[str, Any]:
    if shape is None or shape.is_empty:
        return {
            "parcel_tile_coverage_ratio": np.nan,
            "parcel_tile_coverage_pct": np.nan,
            "parcel_bbox_tile_coverage_ratio": np.nan,
            "parcel_bbox_tile_coverage_pct": np.nan,
            "full_parcel_visible_flag": False,
            "parcel_extent_exceeds_tile_flag": False,
            "parcel_tile_low_coverage_flag": False,
            "multi_tile_candidate_flag": False,
            "parcel_covering_tile_count": 0,
            "parcel_coverage_diagnostics_ready_flag": False,
        }

    polygon_bounds = _polygon_parts_bounds(polygon_parts_from_shape(shape))
    if polygon_bounds is None:
        return {
            "parcel_tile_coverage_ratio": np.nan,
            "parcel_tile_coverage_pct": np.nan,
            "parcel_bbox_tile_coverage_ratio": np.nan,
            "parcel_bbox_tile_coverage_pct": np.nan,
            "full_parcel_visible_flag": False,
            "parcel_extent_exceeds_tile_flag": False,
            "parcel_tile_low_coverage_flag": False,
            "multi_tile_candidate_flag": False,
            "parcel_covering_tile_count": 0,
            "parcel_coverage_diagnostics_ready_flag": False,
        }
    tile_polygon = geometry_box(*tile_bounds(address))
    parcel_area = float(shape.area)
    intersection_area = float(shape.intersection(tile_polygon).area) if parcel_area > 0 else np.nan
    parcel_tile_coverage_ratio = float(np.clip(intersection_area / parcel_area, 0.0, 1.0)) if parcel_area > 0 else np.nan

    parcel_bbox = geometry_box(*polygon_bounds)
    bbox_area = float(parcel_bbox.area)
    bbox_intersection_area = float(parcel_bbox.intersection(tile_polygon).area) if bbox_area > 0 else np.nan
    parcel_bbox_tile_coverage_ratio = float(np.clip(bbox_intersection_area / bbox_area, 0.0, 1.0)) if bbox_area > 0 else np.nan

    return {
        "parcel_tile_coverage_ratio": parcel_tile_coverage_ratio,
        "parcel_tile_coverage_pct": round(parcel_tile_coverage_ratio * 100.0, 1) if np.isfinite(parcel_tile_coverage_ratio) else np.nan,
        "parcel_bbox_tile_coverage_ratio": parcel_bbox_tile_coverage_ratio,
        "parcel_bbox_tile_coverage_pct": round(parcel_bbox_tile_coverage_ratio * 100.0, 1) if np.isfinite(parcel_bbox_tile_coverage_ratio) else np.nan,
    }


def deduplicate_tile_records(tile_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[int, int, int], dict[str, Any]] = {}
    for item in tile_records:
        address = item["address"]
        key = (int(address.z), int(address.x), int(address.y))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {**item}
            continue
        item_coverage = float(item.get("parcel_tile_coverage_ratio", 0.0))
        existing_coverage = float(existing.get("parcel_tile_coverage_ratio", 0.0))
        item_bbox_coverage = float(item.get("parcel_bbox_tile_coverage_ratio", 0.0))
        existing_bbox_coverage = float(existing.get("parcel_bbox_tile_coverage_ratio", 0.0))
        if item_coverage > existing_coverage or (
            item_coverage == existing_coverage and item_bbox_coverage > existing_bbox_coverage
        ):
            deduped[key] = {**item}
    return list(deduped.values())


def parcel_covering_tile_records(geometry_value: bytes | bytearray | memoryview | None, *, zoom: int) -> list[dict[str, Any]]:
    shape = load_geometry_shape(geometry_value)
    if shape is None or shape.is_empty:
        return []
    polygon_bounds = _polygon_parts_bounds(polygon_parts_from_shape(shape))
    if polygon_bounds is None:
        return []
    x_range, y_range = _tile_address_ranges(tuple(polygon_bounds), zoom)
    records: list[dict[str, Any]] = []
    for tile_x in x_range:
        for tile_y in y_range:
            address = TileAddress(x=tile_x, y=tile_y, z=zoom)
            diagnostics = _tile_coverage_diagnostics_for_shape(shape, address)
            if not np.isfinite(diagnostics["parcel_tile_coverage_ratio"]) or diagnostics["parcel_tile_coverage_ratio"] <= 0.0:
                continue
            records.append(
                {
                    "address": address,
                    "tile_label": tile_label(address),
                    "tile_coordinate": tile_coordinate(address),
                    **diagnostics,
                }
            )
    records.sort(
        key=lambda item: (
            -float(item["parcel_tile_coverage_ratio"]),
            -float(item["parcel_bbox_tile_coverage_ratio"]),
            item["address"].y,
            item["address"].x,
        )
    )
    return records


def parcel_tile_coverage_diagnostics(
    geometry_value: bytes | bytearray | memoryview | None,
    address: TileAddress,
) -> dict[str, Any]:
    shape = load_geometry_shape(geometry_value)
    base = _tile_coverage_diagnostics_for_shape(shape, address)
    parcel_tile_coverage_ratio = base.get("parcel_tile_coverage_ratio")
    parcel_bbox_tile_coverage_ratio = base.get("parcel_bbox_tile_coverage_ratio")
    if not np.isfinite(parcel_tile_coverage_ratio) or not np.isfinite(parcel_bbox_tile_coverage_ratio):
        return {
            **base,
            "full_parcel_visible_flag": False,
            "parcel_extent_exceeds_tile_flag": False,
            "parcel_tile_low_coverage_flag": False,
            "multi_tile_candidate_flag": False,
            "parcel_covering_tile_count": 0,
            "parcel_coverage_diagnostics_ready_flag": False,
        }
    parcel_tile_coverage_ratio = float(parcel_tile_coverage_ratio)
    parcel_bbox_tile_coverage_ratio = float(parcel_bbox_tile_coverage_ratio)
    parcel_covering_tile_count = len(deduplicate_tile_records(parcel_covering_tile_records(geometry_value, zoom=address.z)))
    full_parcel_visible_flag = bool(
        parcel_tile_coverage_ratio >= 0.995
        and parcel_bbox_tile_coverage_ratio >= 0.995
    )
    parcel_extent_exceeds_tile_flag = bool(
        np.isfinite(parcel_bbox_tile_coverage_ratio) and parcel_bbox_tile_coverage_ratio < 0.995
    )
    parcel_tile_low_coverage_flag = bool(
        (
            np.isfinite(parcel_tile_coverage_ratio)
            and parcel_tile_coverage_ratio < DEFAULT_LOW_COVERAGE_TILE_THRESHOLD
        )
        or (
            np.isfinite(parcel_bbox_tile_coverage_ratio)
            and parcel_bbox_tile_coverage_ratio < DEFAULT_LOW_COVERAGE_BBOX_THRESHOLD
        )
    )
    multi_tile_candidate_flag = bool(parcel_covering_tile_count > 1)
    if multi_tile_candidate_flag and np.isfinite(parcel_tile_coverage_ratio) and parcel_tile_coverage_ratio < 0.9:
        parcel_tile_low_coverage_flag = True

    return {
        **base,
        "full_parcel_visible_flag": full_parcel_visible_flag,
        "parcel_extent_exceeds_tile_flag": parcel_extent_exceeds_tile_flag,
        "parcel_tile_low_coverage_flag": parcel_tile_low_coverage_flag,
        "multi_tile_candidate_flag": multi_tile_candidate_flag,
        "parcel_covering_tile_count": parcel_covering_tile_count,
        "parcel_coverage_diagnostics_ready_flag": True,
    }


def build_parcel_inference_tile_plan(
    geometry_value: bytes | bytearray | memoryview | None,
    centroid_address: TileAddress,
    *,
    use_multi_tile_extent: bool = True,
) -> dict[str, Any]:
    centroid_diagnostics = parcel_tile_coverage_diagnostics(geometry_value, centroid_address)
    centroid_record = {
        "address": centroid_address,
        "tile_label": tile_label(centroid_address),
        "tile_coordinate": tile_coordinate(centroid_address),
        "centroid_tile_flag": True,
        "tile_rank": 1,
        **centroid_diagnostics,
    }
    if not use_multi_tile_extent:
        return {
            **centroid_diagnostics,
            "tile_records": [centroid_record],
            "multi_tile_inference_used_flag": False,
            "tile_coordinates": json.dumps([tile_coordinate(centroid_address)]),
            "unique_tile_count": 1,
            "duplicate_tile_flag": False,
            "tiles_requested_count": 1,
        }

    tile_records = parcel_covering_tile_records(geometry_value, zoom=centroid_address.z)
    if not tile_records:
        return {
            **centroid_diagnostics,
            "tile_records": [centroid_record],
            "multi_tile_inference_used_flag": False,
            "tile_coordinates": json.dumps([tile_coordinate(centroid_address)]),
            "unique_tile_count": 1,
            "duplicate_tile_flag": False,
            "tiles_requested_count": 1,
        }

    raw_tile_count = len(tile_records)
    tile_records = deduplicate_tile_records(tile_records)
    parcel_covering_tile_count = len(tile_records)
    multi_tile_candidate_flag = parcel_covering_tile_count > 1
    duplicate_tile_flag = parcel_covering_tile_count != raw_tile_count
    normalized_records: list[dict[str, Any]] = []
    for index, item in enumerate(tile_records, start=1):
        address = item["address"]
        is_centroid = (address.x, address.y, address.z) == (centroid_address.x, centroid_address.y, centroid_address.z)
        tile_parcel_coverage_ratio = float(item["parcel_tile_coverage_ratio"])
        tile_bbox_coverage_ratio = float(item["parcel_bbox_tile_coverage_ratio"])
        normalized_records.append(
            {
                **item,
                "centroid_tile_flag": is_centroid,
                "tile_rank": index,
                "full_parcel_visible_flag": bool(
                    tile_parcel_coverage_ratio >= 0.995 and tile_bbox_coverage_ratio >= 0.995
                ),
                "parcel_extent_exceeds_tile_flag": bool(tile_bbox_coverage_ratio < 0.995),
                "parcel_tile_low_coverage_flag": bool(
                    tile_parcel_coverage_ratio < DEFAULT_LOW_COVERAGE_TILE_THRESHOLD
                    or tile_bbox_coverage_ratio < DEFAULT_LOW_COVERAGE_BBOX_THRESHOLD
                    or (multi_tile_candidate_flag and tile_parcel_coverage_ratio < 0.9)
                ),
                "multi_tile_candidate_flag": multi_tile_candidate_flag,
                "parcel_covering_tile_count": parcel_covering_tile_count,
                "parcel_coverage_diagnostics_ready_flag": True,
            }
        )
    normalized_records.sort(
        key=lambda item: (
            not bool(item["centroid_tile_flag"]),
            -float(item["parcel_tile_coverage_ratio"]),
            item["address"].y,
            item["address"].x,
        )
    )
    for index, item in enumerate(normalized_records, start=1):
        item["tile_rank"] = index
    tile_coordinates = [str(item["tile_coordinate"]) for item in normalized_records]
    return {
        **centroid_diagnostics,
        "tile_records": normalized_records,
        "multi_tile_inference_used_flag": len(normalized_records) > 1,
        "tile_coordinates": json.dumps(tile_coordinates),
        "unique_tile_count": len(normalized_records),
        "duplicate_tile_flag": duplicate_tile_flag,
        "tiles_requested_count": len(normalized_records),
    }


def parcel_tile_reliability_factor(
    parcel_tile_coverage_ratio: float | None,
    parcel_bbox_tile_coverage_ratio: float | None,
) -> float:
    ratios: list[float] = []
    for value in (parcel_tile_coverage_ratio, parcel_bbox_tile_coverage_ratio):
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if np.isfinite(numeric):
            ratios.append(float(np.clip(numeric, 0.0, 1.0)))
    if not ratios:
        return 1.0
    visible_ratio = min(ratios)
    return float(np.clip(0.35 + (visible_ratio * 0.65), 0.35, 1.0))


def adjust_confidence_for_tile_coverage(
    confidence_score: float,
    *,
    parcel_tile_coverage_ratio: float | None,
    parcel_bbox_tile_coverage_ratio: float | None,
) -> float:
    reliability = parcel_tile_reliability_factor(parcel_tile_coverage_ratio, parcel_bbox_tile_coverage_ratio)
    adjusted = 50.0 + ((float(confidence_score) - 50.0) * reliability)
    return float(round(np.clip(adjusted, 0.0, 100.0), 2))


def build_ai_vacancy_status_note(base_note: str, coverage_diagnostics: dict[str, Any]) -> str:
    if not coverage_diagnostics.get("parcel_coverage_diagnostics_ready_flag"):
        return base_note
    parcel_tile_coverage_pct = coverage_diagnostics.get("parcel_tile_coverage_pct")
    parcel_bbox_tile_coverage_pct = coverage_diagnostics.get("parcel_bbox_tile_coverage_pct")
    if coverage_diagnostics.get("parcel_tile_low_coverage_flag"):
        return (
            f"{base_note} Limited centroid-tile coverage: {parcel_tile_coverage_pct:.1f}% of parcel area "
            f"and {parcel_bbox_tile_coverage_pct:.1f}% of parcel extent are visible; multi-tile inference is recommended."
        )
    if coverage_diagnostics.get("parcel_extent_exceeds_tile_flag"):
        return (
            f"{base_note} Parcel extends beyond the centroid tile; visible coverage is {parcel_tile_coverage_pct:.1f}% "
            f"of parcel area and {parcel_bbox_tile_coverage_pct:.1f}% of parcel extent."
        )
    return base_note


def build_multi_tile_status_note(base_note: str, aggregation: dict[str, Any]) -> str:
    if not aggregation.get("multi_tile_inference_used_flag"):
        return base_note
    return (
        f"{base_note} Multi-tile parcel inference scored {int(aggregation.get('tiles_scored_count', 0) or 0)} "
        f"parcel-covering tiles; {int(aggregation.get('tiles_with_building_signal_count', 0) or 0)} tile(s) "
        f"showed building-positive evidence."
    )


def build_parcel_mask_result_from_shape(
    shape: Any | None,
    address: TileAddress,
    *,
    tile_size: int = 256,
) -> dict[str, Any]:
    clipped_shape = clip_shape_to_tile(shape, address)
    debug_metadata = parcel_geometry_debug_metadata(shape, clipped_shape)
    polygon_parts = polygon_parts_from_shape(clipped_shape)
    if not polygon_parts:
        return {
            "parcel_mask": None,
            "mask_bbox": None,
            **debug_metadata,
        }
    mask = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(mask)
    drew_any = False
    for polygon in polygon_parts:
        if polygon.is_empty:
            continue
        exterior = [_tile_pixel(lng, lat, address, tile_size) for lng, lat in polygon.exterior.coords]
        if len(exterior) < 3:
            continue
        draw.polygon(exterior, fill=255)
        for ring in polygon.interiors:
            interior = [_tile_pixel(lng, lat, address, tile_size) for lng, lat in ring.coords]
            if len(interior) >= 3:
                draw.polygon(interior, fill=0)
        drew_any = True
    mask_bbox = mask.getbbox() if drew_any else None
    if not drew_any or mask_bbox is None:
        return {
            "parcel_mask": None,
            "mask_bbox": None,
            **debug_metadata,
        }
    return {
        "parcel_mask": mask,
        "mask_bbox": mask_bbox,
        **debug_metadata,
    }


def build_parcel_mask_result_from_shape_for_tile_set(
    shape: Any | None,
    addresses: list[TileAddress],
    *,
    tile_size: int = 256,
    canvas_padding_pixels: int = 0,
) -> dict[str, Any]:
    normalized_addresses = list(addresses)
    if not normalized_addresses:
        return {
            "parcel_mask": None,
            "mask_bbox": None,
            "original_geom_type": getattr(shape, "geom_type", None),
            "clipped_geom_type": None,
            "polygon_part_count": len(polygon_parts_from_shape(shape)),
            "clipped_polygon_part_count": 0,
            "bounds_before_clip": _polygon_parts_bounds(polygon_parts_from_shape(shape)),
            "bounds_after_clip": None,
        }
    zooms = {int(address.z) for address in normalized_addresses}
    if len(zooms) != 1:
        raise ValueError("build_parcel_mask_result_from_shape_for_tile_set requires a single zoom level")
    clipped_shape = clip_shape_to_tile_set(shape, normalized_addresses)
    debug_metadata = parcel_geometry_debug_metadata(shape, clipped_shape)
    polygon_parts = polygon_parts_from_shape(clipped_shape)
    if not polygon_parts:
        return {
            "parcel_mask": None,
            "mask_bbox": None,
            **debug_metadata,
        }
    min_x = min(int(address.x) for address in normalized_addresses)
    max_x = max(int(address.x) for address in normalized_addresses)
    min_y = min(int(address.y) for address in normalized_addresses)
    max_y = max(int(address.y) for address in normalized_addresses)
    canvas_width = tile_size * ((max_x - min_x) + 1)
    canvas_height = tile_size * ((max_y - min_y) + 1)
    canvas_width += int(canvas_padding_pixels * 2)
    canvas_height += int(canvas_padding_pixels * 2)
    zoom = next(iter(zooms))
    mask = Image.new("L", (canvas_width, canvas_height), 0)
    draw = ImageDraw.Draw(mask)
    drew_any = False
    for polygon in polygon_parts:
        if polygon.is_empty:
            continue
        exterior = [
            _tile_canvas_pixel(
                lng,
                lat,
                origin_x=min_x,
                origin_y=min_y,
                zoom=zoom,
                tile_size=tile_size,
            )
            for lng, lat in polygon.exterior.coords
        ]
        exterior = [(x + canvas_padding_pixels, y + canvas_padding_pixels) for x, y in exterior]
        if len(exterior) < 3:
            continue
        draw.polygon(exterior, fill=255)
        for ring in polygon.interiors:
            interior = [
                _tile_canvas_pixel(
                    lng,
                    lat,
                    origin_x=min_x,
                    origin_y=min_y,
                    zoom=zoom,
                    tile_size=tile_size,
                )
                for lng, lat in ring.coords
            ]
            interior = [(x + canvas_padding_pixels, y + canvas_padding_pixels) for x, y in interior]
            if len(interior) >= 3:
                draw.polygon(interior, fill=0)
        drew_any = True
    mask_bbox = mask.getbbox() if drew_any else None
    if not drew_any or mask_bbox is None:
        return {
            "parcel_mask": None,
            "mask_bbox": None,
            **debug_metadata,
        }
    return {
        "parcel_mask": mask,
        "mask_bbox": mask_bbox,
        **debug_metadata,
    }


def build_parcel_mask_from_shape(
    shape: Any | None,
    address: TileAddress,
    *,
    tile_size: int = 256,
) -> Image.Image | None:
    return build_parcel_mask_result_from_shape(shape, address, tile_size=tile_size).get("parcel_mask")


def build_parcel_mask(
    geometry_value: bytes | bytearray | memoryview | None,
    address: TileAddress,
    *,
    tile_size: int = 256,
) -> Image.Image | None:
    return build_parcel_mask_from_shape(load_geometry_shape(geometry_value), address, tile_size=tile_size)


def apply_outside_mask(
    image: Image.Image,
    mask: Image.Image,
    *,
    outside_mask_fill: str = DEFAULT_OUTSIDE_MASK_FILL,
    outside_mask_dim_factor: float = DEFAULT_OUTSIDE_MASK_DIM_FACTOR,
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    mask_array = np.asarray(mask, dtype=np.uint8) > 0
    if outside_mask_fill == "black":
        array[~mask_array] = 0
    else:
        dim_factor = float(np.clip(outside_mask_dim_factor, 0.0, 1.0))
        array[~mask_array] = (array[~mask_array].astype(np.float32) * dim_factor).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _expand_crop_box(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    buffer_pixels: int,
    min_pixels: int,
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    left, top, right, bottom = bbox
    width = max(right - left, 1)
    height = max(bottom - top, 1)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_width = max(width + (buffer_pixels * 2), min_pixels)
    crop_height = max(height + (buffer_pixels * 2), min_pixels)
    crop_width = min(int(round(crop_width)), image_width)
    crop_height = min(int(round(crop_height)), image_height)
    crop_left = int(round(center_x - (crop_width / 2.0)))
    crop_top = int(round(center_y - (crop_height / 2.0)))
    crop_left = max(0, min(crop_left, image_width - crop_width))
    crop_top = max(0, min(crop_top, image_height - crop_height))
    return (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)


def parcel_aware_crop_specs(
    mask_bbox: tuple[int, int, int, int],
    acreage: float | None,
    *,
    image_size: tuple[int, int],
    parcel_buffer_pixels: int = DEFAULT_PARCEL_BUFFER_PIXELS,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    if acreage is None or not np.isfinite(float(acreage)):
        core_min_pixels = max(40, DEFAULT_PARCEL_MIN_CROP_PIXELS - 8)
        focus_min_pixels = DEFAULT_PARCEL_MIN_CROP_PIXELS
        buffer_pixels = parcel_buffer_pixels
    else:
        acreage_value = float(acreage)
        if acreage_value <= 0.25:
            core_min_pixels = 40
            focus_min_pixels = 48
            buffer_pixels = max(2, int(parcel_buffer_pixels * 0.35))
        elif acreage_value <= 1.0:
            core_min_pixels = 44
            focus_min_pixels = 56
            buffer_pixels = max(4, int(parcel_buffer_pixels * 0.5))
        elif acreage_value <= 5.0:
            core_min_pixels = 56
            focus_min_pixels = 72
            buffer_pixels = max(6, int(parcel_buffer_pixels * 0.75))
        else:
            core_min_pixels = 64
            focus_min_pixels = 88
            buffer_pixels = max(8, parcel_buffer_pixels)
    core = _expand_crop_box(
        mask_bbox,
        image_size=image_size,
        buffer_pixels=max(0, min(2, buffer_pixels // 2)),
        min_pixels=core_min_pixels,
    )
    focus = _expand_crop_box(
        mask_bbox,
        image_size=image_size,
        buffer_pixels=buffer_pixels,
        min_pixels=focus_min_pixels,
    )
    return [
        ("parcel_core", core),
        ("parcel_focus", focus),
    ]


def crop_mask_coverage(mask: Image.Image | None, crop_box: tuple[int, int, int, int] | None) -> float:
    if mask is None or crop_box is None:
        return 0.0
    cropped = np.asarray(mask.crop(crop_box), dtype=np.uint8)
    if cropped.size == 0:
        return 0.0
    return float((cropped > 0).mean())


def imagery_false_positive_risk(
    probability: float,
    *,
    driveway_signal: float,
    clearing_signal: float,
    features: dict[str, float],
    crop_label: str,
    parcel_coverage_ratio: float,
) -> float:
    brightness = float(
        ((features.get("r_mean", 0.0) + features.get("g_mean", 0.0) + features.get("b_mean", 0.0)) / 3.0) * 100.0
    )
    shadow_pct = float(features.get("dark_shadow_pct", 0.0) * 100.0)
    variance = float(features.get("gray_variance", 0.0))
    edge_density = float(features.get("edge_density_total", 0.0))

    road_risk = (
        (driveway_signal * 0.36)
        + (edge_density * 520.0)
        + (max(18.0 - shadow_pct, 0.0) * 0.9)
    )
    clearing_risk = (
        (clearing_signal * 0.48)
        + (max(brightness - 55.0, 0.0) * 0.35)
        + (max(0.02 - variance, 0.0) * 1200.0)
        + (max(18.0 - shadow_pct, 0.0) * 0.8)
    )
    crop_penalty = 0.0 if crop_label == "parcel_core" else 8.0
    coverage_penalty = np.clip((0.35 - parcel_coverage_ratio) * 70.0, 0.0, 20.0)
    mid_confidence_penalty = 0.0
    if probability < 0.8:
        mid_confidence_penalty = float(np.clip((0.8 - probability) / 0.3, 0.0, 1.0) * 18.0)
    return float(np.clip(max(road_risk, clearing_risk) + crop_penalty + coverage_penalty + mid_confidence_penalty, 0.0, 100.0))


def calibrated_building_probability(
    probability: float,
    *,
    false_positive_risk: float,
    crop_label: str,
    parcel_coverage_ratio: float,
) -> float:
    penalty = (float(false_positive_risk) / 100.0) * 0.40
    if crop_label != "parcel_core":
        penalty += 0.02
    penalty += float(np.clip(0.30 - parcel_coverage_ratio, 0.0, 0.30) * 0.25)
    return float(np.clip(probability - penalty, 0.0, 1.0))


def building_present_confidence_from_probability(probability: float) -> float:
    return float(round(np.clip(probability * 100.0, 0.0, 100.0), 1))


def aggregate_parcel_tile_predictions(tile_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not tile_predictions:
        return {
            "probability": 0.0,
            "building_present_confidence": 0.0,
            "ai_building_present_flag": False,
            "tiles_scored_count": 0,
            "tiles_with_building_signal_count": 0,
            "multi_tile_inference_used_flag": False,
            "multi_tile_aggregation_reason": "No parcel-covering tiles were scored.",
            "best_tile_label": None,
            "best_tile_confidence": np.nan,
            "best_tile_crop_label": None,
            "best_tile_probability": np.nan,
            "best_tile_parcel_coverage_pct": np.nan,
            "negative_tile_coverage_pct": 0.0,
        }

    sorted_tiles = sorted(
        tile_predictions,
        key=lambda item: (
            float(item["probability"]),
            float(item.get("building_present_confidence", 0.0)),
            float(item.get("tile_parcel_coverage_ratio", 0.0)),
            float(item.get("parcel_coverage_ratio", 0.0)),
        ),
        reverse=True,
    )
    best_tile = sorted_tiles[0]
    multi_tile_inference_used_flag = len(tile_predictions) > 1
    weighted_denominator = sum(max(float(item.get("tile_parcel_coverage_ratio", 0.0)), 0.01) for item in tile_predictions)
    weighted_probability = (
        sum(float(item["probability"]) * max(float(item.get("tile_parcel_coverage_ratio", 0.0)), 0.01) for item in tile_predictions)
        / weighted_denominator
        if weighted_denominator > 0
        else float(best_tile["probability"])
    )

    positive_tiles = [item for item in tile_predictions if bool(item.get("tile_building_signal_flag"))]
    negative_tiles = [item for item in tile_predictions if bool(item.get("tile_negative_signal_flag"))]
    negative_tile_coverage_ratio = float(
        np.clip(sum(float(item.get("tile_parcel_coverage_ratio", 0.0)) for item in negative_tiles), 0.0, 1.0)
    )

    if not multi_tile_inference_used_flag:
        parcel_probability = float(best_tile["probability"])
        aggregation_reason = (
            f"Single centroid tile scored with best crop {best_tile.get('best_crop_label')} on {best_tile.get('tile_label')}."
        )
    elif positive_tiles:
        strongest_positive_tile = max(
            positive_tiles,
            key=lambda item: (
                float(item["probability"]),
                float(item.get("building_present_confidence", 0.0)),
                float(item.get("parcel_coverage_ratio", 0.0)),
            ),
        )
        parcel_probability = float(
            np.clip(
                max(float(strongest_positive_tile["probability"]), weighted_probability, 0.84),
                0.0,
                1.0,
            )
        )
        aggregation_reason = (
            f"{len(positive_tiles)} tile(s) show strong in-parcel building evidence; "
            f"best tile {strongest_positive_tile.get('tile_label')} uses crop {strongest_positive_tile.get('best_crop_label')}."
        )
    elif negative_tile_coverage_ratio >= 0.85 and float(best_tile["probability"]) < 0.50:
        parcel_probability = float(
            np.clip(min(float(best_tile["probability"]), 0.22 + ((1.0 - negative_tile_coverage_ratio) * 0.20)), 0.0, 0.45)
        )
        aggregation_reason = (
            f"All sufficiently covered parcel tiles are negative across {negative_tile_coverage_ratio * 100.0:.1f}% "
            f"of parcel area."
        )
    else:
        parcel_probability = float(np.clip(max(min(float(best_tile["probability"]), 0.68), weighted_probability), 0.0, 0.75))
        aggregation_reason = (
            f"Mixed or weak tile evidence across {len(tile_predictions)} scored tiles; "
            f"best tile {best_tile.get('tile_label')} remains review-grade."
        )

    building_present_confidence = building_present_confidence_from_probability(parcel_probability)
    return {
        "probability": parcel_probability,
        "building_present_confidence": building_present_confidence,
        "ai_building_present_flag": building_present_confidence >= DEFAULT_BUILDING_PRESENT_CONFIDENCE_THRESHOLD,
        "tiles_scored_count": len(tile_predictions),
        "tiles_with_building_signal_count": len(positive_tiles),
        "multi_tile_inference_used_flag": multi_tile_inference_used_flag,
        "multi_tile_aggregation_reason": aggregation_reason,
        "best_tile_label": best_tile.get("tile_label"),
        "best_tile_confidence": round(float(best_tile.get("building_present_confidence", 0.0)), 1),
        "best_tile_crop_label": best_tile.get("best_crop_label"),
        "best_tile_probability": round(float(best_tile["probability"]), 6),
        "best_tile_parcel_coverage_pct": round(float(best_tile.get("tile_parcel_coverage_ratio", 0.0)) * 100.0, 1),
        "negative_tile_coverage_pct": round(negative_tile_coverage_ratio * 100.0, 1),
    }


def prepare_parcel_aware_image(
    image_source: Path | Image.Image,
    *,
    address: TileAddress,
    geometry_value: bytes | bytearray | memoryview | None,
    acreage: float | None,
    coverage_diagnostics: dict[str, Any] | None = None,
    use_parcel_mask: bool = DEFAULT_USE_PARCEL_MASK,
    outside_mask_fill: str = DEFAULT_OUTSIDE_MASK_FILL,
    outside_mask_dim_factor: float = DEFAULT_OUTSIDE_MASK_DIM_FACTOR,
    parcel_buffer_pixels: int = DEFAULT_PARCEL_BUFFER_PIXELS,
) -> dict[str, Any]:
    raw_image = load_tile_image(image_source)
    shape = load_geometry_shape(geometry_value)
    clipped_shape = clip_shape_to_tile(shape, address)
    geometry_debug = parcel_geometry_debug_metadata(shape, clipped_shape)
    if coverage_diagnostics is None:
        coverage_diagnostics = parcel_tile_coverage_diagnostics(geometry_value, address)
    if not use_parcel_mask:
        return {
            "raw_image": raw_image,
            "image": raw_image,
            "crop_specs": crop_specs_for_acreage(acreage),
            "parcel_mask": None,
            "parcel_boundary_crop_ready_flag": False,
            "imagery_crop_strategy": "multi_crop_v2",
            **geometry_debug,
            **coverage_diagnostics,
        }
    mask_result = build_parcel_mask_result_from_shape(shape, address, tile_size=raw_image.size[0])
    mask = mask_result.get("parcel_mask")
    mask_bbox = mask_result.get("mask_bbox")
    if mask is None or mask_bbox is None:
        return {
            "raw_image": raw_image,
            "image": raw_image,
            "crop_specs": crop_specs_for_acreage(acreage),
            "parcel_mask": None,
            "parcel_boundary_crop_ready_flag": False,
            "imagery_crop_strategy": "multi_crop_v2",
            **mask_result,
            **coverage_diagnostics,
        }
    masked_image = apply_outside_mask(
        raw_image,
        mask,
        outside_mask_fill=outside_mask_fill,
        outside_mask_dim_factor=outside_mask_dim_factor,
    )
    return {
        "raw_image": raw_image,
        "image": masked_image,
        "crop_specs": parcel_aware_crop_specs(
            mask_bbox,
            acreage,
            image_size=raw_image.size,
            parcel_buffer_pixels=parcel_buffer_pixels,
        ),
        "parcel_mask": mask,
        "parcel_boundary_crop_ready_flag": True,
        "imagery_crop_strategy": "parcel_mask_tight_crop_v2",
        **mask_result,
        **coverage_diagnostics,
    }


def prepare_parcel_aware_image_for_tile_set(
    tile_image_sources: list[tuple[TileAddress, Path | Image.Image]],
    *,
    geometry_value: bytes | bytearray | memoryview | None,
    acreage: float | None,
    use_parcel_mask: bool = DEFAULT_USE_PARCEL_MASK,
    outside_mask_fill: str = DEFAULT_OUTSIDE_MASK_FILL,
    outside_mask_dim_factor: float = DEFAULT_OUTSIDE_MASK_DIM_FACTOR,
    parcel_buffer_pixels: int = DEFAULT_PARCEL_BUFFER_PIXELS,
    canvas_padding_pixels: int = DEFAULT_MULTI_TILE_CANVAS_PADDING_PIXELS,
) -> dict[str, Any]:
    if not tile_image_sources:
        raise ValueError("prepare_parcel_aware_image_for_tile_set requires at least one tile image")
    deduped_sources: dict[tuple[int, int, int], tuple[TileAddress, Path | Image.Image]] = {}
    for address, image_source in tile_image_sources:
        deduped_sources[(int(address.z), int(address.x), int(address.y))] = (address, image_source)
    normalized_sources = list(deduped_sources.values())
    if len(normalized_sources) == 1:
        address, image_source = normalized_sources[0]
        return prepare_parcel_aware_image(
            image_source,
            address=address,
            geometry_value=geometry_value,
            acreage=acreage,
            use_parcel_mask=use_parcel_mask,
            outside_mask_fill=outside_mask_fill,
            outside_mask_dim_factor=outside_mask_dim_factor,
            parcel_buffer_pixels=parcel_buffer_pixels,
        )

    addresses = [item[0] for item in normalized_sources]
    zooms = {int(address.z) for address in addresses}
    if len(zooms) != 1:
        raise ValueError("prepare_parcel_aware_image_for_tile_set requires a single zoom level")
    first_image = load_tile_image(normalized_sources[0][1]).convert("RGB")
    tile_size = first_image.size[0]
    canvas_padding_pixels = max(int(canvas_padding_pixels), int(parcel_buffer_pixels), 0)
    min_x = min(int(address.x) for address in addresses)
    max_x = max(int(address.x) for address in addresses)
    min_y = min(int(address.y) for address in addresses)
    max_y = max(int(address.y) for address in addresses)
    canvas_width = (tile_size * ((max_x - min_x) + 1)) + (canvas_padding_pixels * 2)
    canvas_height = (tile_size * ((max_y - min_y) + 1)) + (canvas_padding_pixels * 2)
    raw_image = Image.new("RGB", (canvas_width, canvas_height), (0, 0, 0))
    for address, image_source in normalized_sources:
        tile_image = first_image if image_source is normalized_sources[0][1] else load_tile_image(image_source).convert("RGB")
        paste_x = ((int(address.x) - min_x) * tile_size) + canvas_padding_pixels
        paste_y = ((int(address.y) - min_y) * tile_size) + canvas_padding_pixels
        raw_image.paste(tile_image, (paste_x, paste_y))

    shape = load_geometry_shape(geometry_value)
    mask_result = build_parcel_mask_result_from_shape_for_tile_set(
        shape,
        addresses,
        tile_size=tile_size,
        canvas_padding_pixels=canvas_padding_pixels,
    )
    mask = mask_result.get("parcel_mask")
    mask_bbox = mask_result.get("mask_bbox")
    if not use_parcel_mask or mask is None or mask_bbox is None:
        return {
            "raw_image": raw_image,
            "image": raw_image,
            "crop_specs": crop_specs_for_acreage(acreage),
            "parcel_mask": mask,
            "parcel_boundary_crop_ready_flag": False,
            "imagery_crop_strategy": "multi_tile_crop_v1",
            "display_crop_box": None,
            "canvas_padding_pixels": canvas_padding_pixels,
            **mask_result,
        }
    masked_image = apply_outside_mask(
        raw_image,
        mask,
        outside_mask_fill=outside_mask_fill,
        outside_mask_dim_factor=outside_mask_dim_factor,
    )
    display_crop_box = _expand_crop_box(
        mask_bbox,
        image_size=raw_image.size,
        buffer_pixels=max(parcel_buffer_pixels * 2, canvas_padding_pixels // 2),
        min_pixels=max(DEFAULT_PARCEL_MIN_CROP_PIXELS + (parcel_buffer_pixels * 2), 64),
    )
    return {
        "raw_image": raw_image,
        "image": masked_image,
        "crop_specs": parcel_aware_crop_specs(
            mask_bbox,
            acreage,
            image_size=raw_image.size,
            parcel_buffer_pixels=parcel_buffer_pixels,
        ),
        "parcel_mask": mask,
        "parcel_boundary_crop_ready_flag": True,
        "imagery_crop_strategy": "parcel_mask_tight_crop_multi_tile_v1",
        "display_crop_box": display_crop_box,
        "canvas_padding_pixels": canvas_padding_pixels,
        **mask_result,
    }


def extract_image_features(image_source: Path | Image.Image, crop_box: tuple[int, int, int, int] | None = None) -> dict[str, float]:
    image = load_tile_image(image_source)
    if crop_box is not None:
        image = image.crop(crop_box)
    image = image.resize((128, 128))
    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = array.mean(axis=2)
    flattened = array.reshape(-1, 3)
    features: dict[str, float] = {}

    channel_means = flattened.mean(axis=0)
    channel_stds = flattened.std(axis=0)
    for index, channel in enumerate(("r", "g", "b")):
        features[f"{channel}_mean"] = float(channel_means[index])
        features[f"{channel}_std"] = float(channel_stds[index])

    brightness_hist, _ = np.histogram(gray, bins=12, range=(0.0, 1.0), density=True)
    for index, value in enumerate(brightness_hist):
        features[f"brightness_hist_{index}"] = float(value)

    grad_x = np.abs(np.diff(gray, axis=1)).mean()
    grad_y = np.abs(np.diff(gray, axis=0)).mean()
    features["edge_density_x"] = float(grad_x)
    features["edge_density_y"] = float(grad_y)
    features["edge_density_total"] = float(grad_x + grad_y)
    features["gray_variance"] = float(gray.var())
    features["green_excess"] = float(channel_means[1] - ((channel_means[0] + channel_means[2]) / 2.0))
    features["roof_tone_pct"] = float(
        np.mean((array[..., 0] > 0.35) & (array[..., 0] < 0.85) & (array[..., 1] > 0.35) & (array[..., 1] < 0.85))
    )
    features["dark_shadow_pct"] = float(np.mean(gray < 0.18))
    return features


def imagery_context_signals(features: dict[str, float]) -> dict[str, float]:
    driveway_signal = float(
        np.clip(
            (features["roof_tone_pct"] * 120.0)
            + (features["edge_density_total"] * 280.0)
            + (features["dark_shadow_pct"] * 35.0)
            - (max(features["green_excess"], 0.0) * 55.0),
            0.0,
            100.0,
        )
    )
    clearing_signal = float(
        np.clip(
            ((features["r_mean"] + features["g_mean"] + features["b_mean"]) / 3.0 * 100.0)
            + (features["roof_tone_pct"] * 40.0)
            - (max(features["green_excess"], 0.0) * 45.0),
            0.0,
            100.0,
        )
    )
    return {
        "imagery_driveway_signal": driveway_signal,
        "imagery_clearing_signal": clearing_signal,
    }


def feature_columns(frame: pd.DataFrame) -> list[str]:
    base_columns = {
        "parcel_row_id",
        "parcel_id",
        "state_code",
        "county_name",
        "county_fips",
        "image_path",
        "weak_building_label",
        "weak_label_source",
        "weak_label_rule",
        "label_reliability_tier",
        "building_count",
        "building_area_total",
        "parcel_vacant_flag",
        "total_value",
        "improvement_value_1",
        "improvement_value_2",
        "imagery_source",
        "imagery_zoom",
        "tile_x",
        "tile_y",
        "tile_label",
        "tile_coordinate",
        "tile_rank",
        "centroid_tile_flag",
        "tile_selection_strategy",
        "tile_selection_role",
        "tile_source_mode",
        "crop_parcel_coverage_ratio",
        "model_version",
        "imagery_crop_strategy",
        "imagery_crop_label",
        "imagery_best_crop_label",
        "imagery_crop_count",
        "parcel_boundary_crop_ready_flag",
        "building_present_confidence",
        "building_presence_reason",
        "imagery_driveway_signal",
        "imagery_clearing_signal",
        "ai_building_present_probability",
        "ai_building_present_flag",
        "parcel_tile_coverage_ratio",
        "parcel_tile_coverage_pct",
        "parcel_bbox_tile_coverage_ratio",
        "parcel_bbox_tile_coverage_pct",
        "full_parcel_visible_flag",
        "parcel_extent_exceeds_tile_flag",
        "parcel_tile_low_coverage_flag",
        "multi_tile_candidate_flag",
        "parcel_covering_tile_count",
        "tile_coordinates",
        "unique_tile_count",
        "duplicate_tile_flag",
        "original_geom_type",
        "clipped_geom_type",
        "polygon_part_count",
        "clipped_polygon_part_count",
        "bounds_before_clip",
        "bounds_after_clip",
        "parcel_coverage_diagnostics_ready_flag",
        "tiles_scored_count",
        "tiles_with_building_signal_count",
        "multi_tile_inference_used_flag",
        "multi_tile_aggregation_reason",
        "best_tile_label",
        "best_tile_confidence",
        "best_tile_crop_label",
        "best_tile_probability",
        "best_tile_parcel_coverage_pct",
        "negative_tile_coverage_pct",
        "vacancy_confidence_score",
        "inference_timestamp",
        "geometry",
        "tile_records",
        "manifest_version",
        "dataset_scope",
        "app_ready_only",
        "cached_tile_required_flag",
        "use_multi_tile_extent",
        "selected_tile_count",
        "selected_tile_labels",
    }
    return [column for column in frame.columns if column not in base_columns]


def combined_vacancy_confidence(
    parcel_vacant_flag: bool,
    building_probability: float,
    building_present_confidence: float | None = None,
) -> float:
    footprint_vacancy_score = 92.0 if parcel_vacant_flag else 15.0
    imagery_vacancy_score = (1.0 - building_probability) * 100.0
    if building_present_confidence is not None:
        imagery_vacancy_score = 100.0 - float(building_present_confidence)
    return float(round((footprint_vacancy_score * 0.45) + (imagery_vacancy_score * 0.55), 2))


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
