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


def runtime_path_candidates(env_name: str, relative_path: str | Path, *, root: Path | None = None) -> list[Path]:
    relative = Path(relative_path)
    search_root = _absolute_path(root or BASE_DIR)
    raw = os.getenv(env_name)

    preferred_candidates: list[Path] = [
        _absolute_path(search_root / relative),
        _absolute_path(BASE_DIR / relative),
    ]
    env_candidates: list[Path] = []

    if raw:
        explicit = Path(raw).expanduser()
        if explicit.is_absolute():
            repo_relative = _repo_relative_candidate(raw)
            if repo_relative is not None:
                env_candidates.append(repo_relative)
            env_candidates.append(_absolute_path(explicit))
        else:
            env_candidates.append(_absolute_path(search_root / explicit))

    candidates = preferred_candidates + env_candidates

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def _resolve_runtime_path(env_name: str, relative_path: str | Path, *, root: Path | None = None) -> Path:
    candidates = runtime_path_candidates(env_name, relative_path, root=root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
ALLOWED_CORS_ORIGINS = _csv_env(
    "ALLOWED_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://landintel.vercel.app",
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
GEOMETRY_DEFAULT_LIMIT = int(os.getenv("GEOMETRY_DEFAULT_LIMIT", "800"))
GEOMETRY_MAX_LIMIT = int(os.getenv("GEOMETRY_MAX_LIMIT", "1500"))
GZIP_MINIMUM_SIZE = int(os.getenv("GZIP_MINIMUM_SIZE", "1024"))


def runtime_file_diagnostics() -> dict[str, dict[str, int | bool | str | list[str] | None]]:
    specs = {
        "app_ready_parquet": (
            MISSISSIPPI_APP_READY_PATH,
            "MISSISSIPPI_APP_READY_PATH",
            "data/tax_published/ms/app_ready_mississippi_leads.parquet",
        ),
        "static_feed_json": (
            MISSISSIPPI_STATIC_FEED_PATH,
            "MISSISSIPPI_STATIC_FEED_PATH",
            "frontend/public/data/mississippi_lead_explorer.json",
        ),
        "meta_json": (
            MISSISSIPPI_META_PATH,
            "MISSISSIPPI_META_PATH",
            "frontend/public/data/mississippi_lead_explorer_meta.json",
        ),
        "geometry_json": (
            MISSISSIPPI_GEOMETRY_PATH,
            "MISSISSIPPI_GEOMETRY_PATH",
            "frontend/public/data/mississippi_lead_explorer_geometries.json",
        ),
    }
    diagnostics: dict[str, dict[str, int | bool | str | list[str] | None]] = {}
    cwd = str(Path.cwd())
    for name, (path, env_name, relative_path) in specs.items():
        diagnostics[name] = {
            "cwd": cwd,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "candidates": [str(candidate) for candidate in runtime_path_candidates(env_name, relative_path, root=MISSISSIPPI_EXPLORER_DATA_ROOT)],
        }
    return diagnostics
