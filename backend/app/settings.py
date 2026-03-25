from __future__ import annotations

import json
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


def _state_registry_path() -> Path:
    return BASE_DIR / "config" / "states" / "registry.json"


def _state_config_payload(state_code: str) -> dict | None:
    registry_path = _state_registry_path()
    if not registry_path.exists():
        return None
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    normalized = state_code.strip().lower()
    states = registry.get("states", {})
    state_entry = states.get(normalized)
    if not isinstance(state_entry, dict):
        return None
    config_path_raw = state_entry.get("config_path")
    if not config_path_raw:
        return None
    config_path = BASE_DIR / str(config_path_raw)
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _state_legacy_path(state_code: str, legacy_key: str) -> Path | None:
    payload = _state_config_payload(state_code)
    if not payload:
        return None
    legacy_paths = payload.get("legacy_paths", {})
    raw_path = legacy_paths.get(legacy_key)
    if not raw_path:
        return None
    return _absolute_path(BASE_DIR / Path(str(raw_path)))


def resolve_state_runtime_path(
    state_code: str,
    *,
    env_name: str | None,
    legacy_key: str | None,
    default_relative_path: str | Path,
    root: Path | None = None,
) -> Path:
    relative = Path(default_relative_path)
    search_root = _absolute_path(root or BASE_DIR)
    candidates: list[Path] = []

    if legacy_key:
        legacy_candidate = _state_legacy_path(state_code, legacy_key)
        if legacy_candidate is not None:
            candidates.append(legacy_candidate)

    candidates.extend(
        [
            _absolute_path(search_root / relative),
            _absolute_path(BASE_DIR / relative),
        ]
    )

    raw = os.getenv(env_name) if env_name else None
    if raw:
        explicit = Path(raw).expanduser()
        if explicit.is_absolute():
            repo_relative = _repo_relative_candidate(raw)
            if repo_relative is not None:
                candidates.append(repo_relative)
            candidates.append(_absolute_path(explicit))
        else:
            candidates.append(_absolute_path(search_root / explicit))

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
    "http://localhost:3000,http://127.0.0.1:3000,https://landintel.vercel.app",
)

MISSISSIPPI_EXPLORER_DATA_ROOT = _resolve_runtime_path(
    "MISSISSIPPI_EXPLORER_DATA_ROOT",
    ".",
    root=BASE_DIR,
)

MISSISSIPPI_APP_READY_PATH = resolve_state_runtime_path(
    "ms",
    env_name="MISSISSIPPI_APP_READY_PATH",
    legacy_key="app_ready_leads",
    default_relative_path="data/tax_published/ms/app_ready_mississippi_leads.parquet",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)
MISSISSIPPI_STATIC_FEED_PATH = resolve_state_runtime_path(
    "ms",
    env_name="MISSISSIPPI_STATIC_FEED_PATH",
    legacy_key="frontend_static_feed",
    default_relative_path="frontend/public/data/mississippi_lead_explorer.json",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)
MISSISSIPPI_META_PATH = resolve_state_runtime_path(
    "ms",
    env_name="MISSISSIPPI_META_PATH",
    legacy_key="frontend_meta",
    default_relative_path="frontend/public/data/mississippi_lead_explorer_meta.json",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)
MISSISSIPPI_GEOMETRY_PATH = resolve_state_runtime_path(
    "ms",
    env_name="MISSISSIPPI_GEOMETRY_PATH",
    legacy_key="frontend_geometry",
    default_relative_path="frontend/public/data/mississippi_lead_explorer_geometries.json",
    root=MISSISSIPPI_EXPLORER_DATA_ROOT,
)

LEADS_DEFAULT_LIMIT = int(os.getenv("LEADS_DEFAULT_LIMIT", "200"))
LEADS_MAX_LIMIT = int(os.getenv("LEADS_MAX_LIMIT", "250"))
GEOMETRY_DEFAULT_LIMIT = int(os.getenv("GEOMETRY_DEFAULT_LIMIT", "250"))
GEOMETRY_MAX_LIMIT = int(os.getenv("GEOMETRY_MAX_LIMIT", "400"))
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


def state_runtime_file_diagnostics(state_code: str) -> dict[str, dict[str, int | bool | str | list[str] | None]]:
    normalized = state_code.strip().lower()
    specs = {
        "app_ready_parquet": (
            resolve_state_runtime_path(
                normalized,
                env_name="MISSISSIPPI_APP_READY_PATH" if normalized == "ms" else None,
                legacy_key="app_ready_leads",
                default_relative_path=f"data/tax_published/{normalized}/app_ready_{normalized}_leads.parquet",
                root=MISSISSIPPI_EXPLORER_DATA_ROOT,
            ),
            "app_ready_leads",
            f"data/tax_published/{normalized}/app_ready_{normalized}_leads.parquet",
        ),
        "static_feed_json": (
            resolve_state_runtime_path(
                normalized,
                env_name="MISSISSIPPI_STATIC_FEED_PATH" if normalized == "ms" else None,
                legacy_key="frontend_static_feed",
                default_relative_path=f"frontend/public/data/{normalized}_lead_explorer.json",
                root=MISSISSIPPI_EXPLORER_DATA_ROOT,
            ),
            "frontend_static_feed",
            f"frontend/public/data/{normalized}_lead_explorer.json",
        ),
        "meta_json": (
            resolve_state_runtime_path(
                normalized,
                env_name="MISSISSIPPI_META_PATH" if normalized == "ms" else None,
                legacy_key="frontend_meta",
                default_relative_path=f"frontend/public/data/{normalized}_lead_explorer_meta.json",
                root=MISSISSIPPI_EXPLORER_DATA_ROOT,
            ),
            "frontend_meta",
            f"frontend/public/data/{normalized}_lead_explorer_meta.json",
        ),
        "geometry_json": (
            resolve_state_runtime_path(
                normalized,
                env_name="MISSISSIPPI_GEOMETRY_PATH" if normalized == "ms" else None,
                legacy_key="frontend_geometry",
                default_relative_path=f"frontend/public/data/{normalized}_lead_explorer_geometries.json",
                root=MISSISSIPPI_EXPLORER_DATA_ROOT,
            ),
            "frontend_geometry",
            f"frontend/public/data/{normalized}_lead_explorer_geometries.json",
        ),
    }
    diagnostics: dict[str, dict[str, int | bool | str | list[str] | None]] = {}
    cwd = str(Path.cwd())
    project_root = str(MISSISSIPPI_EXPLORER_DATA_ROOT)
    for name, (path, _legacy_key, relative_path) in specs.items():
        diagnostics[name] = {
            "cwd": cwd,
            "project_root": project_root,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "candidates": [str(path), str(BASE_DIR / Path(relative_path))],
        }
    return diagnostics
