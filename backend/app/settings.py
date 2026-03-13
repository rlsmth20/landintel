from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [value.strip() for value in raw.split(",")]
    return [value for value in values if value]


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
ALLOWED_CORS_ORIGINS = _csv_env(
    "ALLOWED_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)

TAX_PUBLISHED_DIR = _path_env("MISSISSIPPI_TAX_PUBLISHED_DIR", BASE_DIR / "data" / "tax_published" / "ms")
FRONTEND_PUBLIC_DATA_DIR = _path_env("MISSISSIPPI_FRONTEND_PUBLIC_DATA_DIR", BASE_DIR / "frontend" / "public" / "data")

MISSISSIPPI_APP_READY_PATH = _path_env(
    "MISSISSIPPI_APP_READY_PATH",
    TAX_PUBLISHED_DIR / "app_ready_mississippi_leads.parquet",
)
MISSISSIPPI_META_PATH = _path_env(
    "MISSISSIPPI_META_PATH",
    FRONTEND_PUBLIC_DATA_DIR / "mississippi_lead_explorer_meta.json",
)
MISSISSIPPI_GEOMETRY_PATH = _path_env(
    "MISSISSIPPI_GEOMETRY_PATH",
    FRONTEND_PUBLIC_DATA_DIR / "mississippi_lead_explorer_geometries.json",
)

LEADS_DEFAULT_LIMIT = int(os.getenv("LEADS_DEFAULT_LIMIT", "200"))
LEADS_MAX_LIMIT = int(os.getenv("LEADS_MAX_LIMIT", "250"))
GEOMETRY_DEFAULT_LIMIT = int(os.getenv("GEOMETRY_DEFAULT_LIMIT", "350"))
GEOMETRY_MAX_LIMIT = int(os.getenv("GEOMETRY_MAX_LIMIT", "500"))
GZIP_MINIMUM_SIZE = int(os.getenv("GZIP_MINIMUM_SIZE", "1024"))
