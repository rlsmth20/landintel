from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [value.strip() for value in raw.split(",")]
    return [value for value in values if value]


def _absolute_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _repo_relative_candidate(raw: str) -> Path | None:
    stripped = raw.lstrip("/\\")
    if not stripped:
        return None
    return _absolute_path(BASE_DIR / Path(stripped))


def _resolve_runtime_path(env_name: str, relative_path: str | Path, *, root: Path | None = None) -> Path:
    relative = Path(relative_path)
    search_root = _absolute_path(root or BASE_DIR)
    raw = os.getenv(env_name)

    candidates: list[Path] = []
    if raw:
        explicit = Path(raw).expanduser()
        if explicit.is_absolute():
            candidates.append(_absolute_path(explicit))
            repo_relative = _repo_relative_candidate(raw)
            if repo_relative is not None:
                candidates.append(repo_relative)
        else:
            candidates.append(_absolute_path(search_root / explicit))

    candidates.append(_absolute_path(search_root / relative))
    candidates.append(_absolute_path(BASE_DIR / relative))

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    return unique_candidates[0]


APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
ALLOWED_CORS_ORIGINS = _csv_env(
    "ALLOWED_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)

MISSISSIPPI_EXPLORER_DATA_ROOT = _resolve_runtime_path(
    "MISSISSIPPI_EXPLORER_DATA_ROOT",
    ".",
    root=BASE_DIR,
)

MISSISSIPPI_APP_READY_PATH = _resolve_runtime_path(
    "MISSISSIPPI_APP_READY_PATH",
    "data/tax_published/ms/app_ready_mississippi_leads.parquet",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)
MISSISSIPPI_STATIC_FEED_PATH = _resolve_runtime_path(
    "MISSISSIPPI_STATIC_FEED_PATH",
    "frontend/public/data/mississippi_lead_explorer.json",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)
MISSISSIPPI_META_PATH = _resolve_runtime_path(
    "MISSISSIPPI_META_PATH",
    "frontend/public/data/mississippi_lead_explorer_meta.json",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)
MISSISSIPPI_GEOMETRY_PATH = _resolve_runtime_path(
    "MISSISSIPPI_GEOMETRY_PATH",
    "frontend/public/data/mississippi_lead_explorer_geometries.json",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)

LEADS_DEFAULT_LIMIT = int(os.getenv("LEADS_DEFAULT_LIMIT", "200"))
LEADS_MAX_LIMIT = int(os.getenv("LEADS_MAX_LIMIT", "250"))
GEOMETRY_DEFAULT_LIMIT = int(os.getenv("GEOMETRY_DEFAULT_LIMIT", "350"))
GEOMETRY_MAX_LIMIT = int(os.getenv("GEOMETRY_MAX_LIMIT", "500"))
GZIP_MINIMUM_SIZE = int(os.getenv("GZIP_MINIMUM_SIZE", "1024"))


def runtime_file_diagnostics() -> dict[str, dict[str, int | bool | str | None]]:
    paths = {
        "app_ready_parquet": MISSISSIPPI_APP_READY_PATH,
        "static_feed_json": MISSISSIPPI_STATIC_FEED_PATH,
        "meta_json": MISSISSIPPI_META_PATH,
        "geometry_json": MISSISSIPPI_GEOMETRY_PATH,
    }
    diagnostics: dict[str, dict[str, int | bool | str | None]] = {}
    for name, path in paths.items():
        diagnostics[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return diagnostics
