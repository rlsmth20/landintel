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


ROOT = Path(__file__).resolve().parents[1]
PARCEL_MASTER_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
BUILDING_METRICS_PATH = ROOT / "data" / "buildings_processed" / "parcel_building_metrics.parquet"
AI_DATA_DIR = ROOT / "data" / "buildings_processed"
APP_READY_PATH = ROOT / "data" / "tax_published" / "ms" / "app_ready_mississippi_leads.parquet"
BACKEND_RUNTIME_DIR = ROOT / "backend" / "runtime" / "mississippi"
TRAINING_MANIFEST_PATH = AI_DATA_DIR / "ai_building_presence_training_manifest_ms.parquet"
MODEL_PATH = AI_DATA_DIR / "ai_building_presence_model_ms.joblib"
MODEL_METRICS_PATH = AI_DATA_DIR / "ai_building_presence_model_metrics_ms.json"
MODEL_PARAMS_PATH = BACKEND_RUNTIME_DIR / "ai_building_presence_model_ms.json"
RUNTIME_MODEL_METRICS_PATH = BACKEND_RUNTIME_DIR / "ai_building_presence_model_metrics_ms.json"
PREDICTIONS_PATH = AI_DATA_DIR / "ai_building_presence_predictions_ms.parquet"
TILE_CACHE_DIR = AI_DATA_DIR / "ai_building_tiles_ms"
DEFAULT_TILE_URL_TEMPLATE = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
MODEL_VERSION = "ms_building_presence_v2_multi_crop"
DEFAULT_USE_PARCEL_MASK = True
DEFAULT_OUTSIDE_MASK_FILL = "dim"
DEFAULT_OUTSIDE_MASK_DIM_FACTOR = 0.15
DEFAULT_PARCEL_BUFFER_PIXELS = 18
DEFAULT_PARCEL_MIN_CROP_PIXELS = 72


@dataclass
class TileAddress:
    x: int
    y: int
    z: int


def load_candidate_frame() -> pd.DataFrame:
    parcels = pd.read_parquet(
        PARCEL_MASTER_PATH,
        columns=["parcel_row_id", "county_name", "latitude", "longitude", "total_acres", "parcel_area_acres", "gis_acres", "tax_acres"],
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
    path = tile_cache_path(parcel_row_id, county_name, zoom)
    path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not path.exists():
        image = fetch_tile_image(tile_url(address, template))
        image.save(path, format="JPEG", quality=88)
    return path, address


def load_tile_image(image_source: Path | Image.Image) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    return Image.open(image_source).convert("RGB")


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


def build_parcel_mask(
    geometry_value: bytes | bytearray | memoryview | None,
    address: TileAddress,
    *,
    tile_size: int = 256,
) -> Image.Image | None:
    if geometry_value is None:
        return None
    try:
        shape = wkb.loads(bytes(geometry_value))
    except Exception:
        return None
    mask = Image.new("L", (tile_size, tile_size), 0)
    draw = ImageDraw.Draw(mask)
    polygons = getattr(shape, "geoms", [shape]) if shape.geom_type == "MultiPolygon" else [shape]
    drew_any = False
    for polygon in polygons:
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
    if not drew_any or mask.getbbox() is None:
        return None
    return mask


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
        min_pixels = DEFAULT_PARCEL_MIN_CROP_PIXELS
        buffer_pixels = parcel_buffer_pixels
    else:
        acreage_value = float(acreage)
        if acreage_value <= 0.25:
            min_pixels = 64
            buffer_pixels = max(6, int(parcel_buffer_pixels * 0.5))
        elif acreage_value <= 1.0:
            min_pixels = 72
            buffer_pixels = max(8, int(parcel_buffer_pixels * 0.65))
        elif acreage_value <= 5.0:
            min_pixels = 88
            buffer_pixels = parcel_buffer_pixels
        else:
            min_pixels = 112
            buffer_pixels = parcel_buffer_pixels + 10
    focus = _expand_crop_box(mask_bbox, image_size=image_size, buffer_pixels=buffer_pixels, min_pixels=min_pixels)
    context = _expand_crop_box(
        mask_bbox,
        image_size=image_size,
        buffer_pixels=buffer_pixels + 14,
        min_pixels=min(min_pixels + 36, max(image_size)),
    )
    full = (0, 0, image_size[0], image_size[1])
    return [
        ("parcel_focus", focus),
        ("parcel_context", context),
        ("parcel_mask_full", full),
    ]


def prepare_parcel_aware_image(
    image_source: Path | Image.Image,
    *,
    address: TileAddress,
    geometry_value: bytes | bytearray | memoryview | None,
    acreage: float | None,
    use_parcel_mask: bool = DEFAULT_USE_PARCEL_MASK,
    outside_mask_fill: str = DEFAULT_OUTSIDE_MASK_FILL,
    outside_mask_dim_factor: float = DEFAULT_OUTSIDE_MASK_DIM_FACTOR,
    parcel_buffer_pixels: int = DEFAULT_PARCEL_BUFFER_PIXELS,
) -> dict[str, Any]:
    image = load_tile_image(image_source)
    if not use_parcel_mask:
        return {
            "image": image,
            "crop_specs": crop_specs_for_acreage(acreage),
            "parcel_boundary_crop_ready_flag": False,
            "imagery_crop_strategy": "multi_crop_v2",
        }
    mask = build_parcel_mask(geometry_value, address, tile_size=image.size[0])
    if mask is None or mask.getbbox() is None:
        return {
            "image": image,
            "crop_specs": crop_specs_for_acreage(acreage),
            "parcel_boundary_crop_ready_flag": False,
            "imagery_crop_strategy": "multi_crop_v2",
        }
    masked_image = apply_outside_mask(
        image,
        mask,
        outside_mask_fill=outside_mask_fill,
        outside_mask_dim_factor=outside_mask_dim_factor,
    )
    return {
        "image": masked_image,
        "crop_specs": parcel_aware_crop_specs(
            mask.getbbox(),
            acreage,
            image_size=image.size,
            parcel_buffer_pixels=parcel_buffer_pixels,
        ),
        "parcel_boundary_crop_ready_flag": True,
        "imagery_crop_strategy": "parcel_mask_multi_crop_v1",
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
        "county_name",
        "image_path",
        "weak_building_label",
        "building_count",
        "building_area_total",
        "parcel_vacant_flag",
        "imagery_source",
        "imagery_zoom",
        "tile_x",
        "tile_y",
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
        "vacancy_confidence_score",
        "inference_timestamp",
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
