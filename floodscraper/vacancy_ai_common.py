from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PARCEL_MASTER_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
BUILDING_METRICS_PATH = ROOT / "data" / "buildings_processed" / "parcel_building_metrics.parquet"
AI_DATA_DIR = ROOT / "data" / "buildings_processed"
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
MODEL_VERSION = "ms_building_presence_v1"


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


def extract_image_features(image_path: Path) -> dict[str, float]:
    image = Image.open(image_path).convert("RGB").resize((128, 128))
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
        "ai_building_present_probability",
        "ai_building_present_flag",
        "vacancy_confidence_score",
        "inference_timestamp",
    }
    return [column for column in frame.columns if column not in base_columns]


def combined_vacancy_confidence(parcel_vacant_flag: bool, building_probability: float) -> float:
    footprint_vacancy_score = 92.0 if parcel_vacant_flag else 15.0
    imagery_vacancy_score = (1.0 - building_probability) * 100.0
    return float(round((footprint_vacancy_score * 0.55) + (imagery_vacancy_score * 0.45), 2))


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
