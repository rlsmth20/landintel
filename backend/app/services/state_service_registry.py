from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

from app.services.runtime_state_service import get_runtime_state_service


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_REGISTRY_PATH = PROJECT_ROOT / "config" / "states" / "registry.json"
logger = logging.getLogger("state-runtime")
SPECIAL_STATE_SERVICE_MODULES = {
    "ms": "app.services.mississippi_leads_service",
}


def configured_state_codes(*, include_blocked: bool = False) -> list[str]:
    if not STATE_REGISTRY_PATH.exists():
        return []
    payload = json.loads(STATE_REGISTRY_PATH.read_text(encoding="utf-8"))
    states = payload.get("states", {})
    configured: list[str] = []
    for state_code, metadata in states.items():
        normalized = str(state_code).lower()
        status = str((metadata or {}).get("status", "")).strip().lower()
        if not include_blocked and status.startswith("blocked"):
            continue
        configured.append(normalized)
    return sorted(configured)


def supported_state_codes() -> list[str]:
    return configured_state_codes()


def get_state_service(state_code: str) -> Any:
    normalized = state_code.strip().lower()
    if normalized not in configured_state_codes():
        raise KeyError(f"State code is not configured: {normalized}")
    module_name = SPECIAL_STATE_SERVICE_MODULES.get(normalized)
    if module_name is not None:
        logger.info("Resolving special-case state service state=%s module=%s", normalized, module_name)
        return importlib.import_module(module_name)
    logger.info("Resolving runtime-backed state service state=%s", normalized)
    return get_runtime_state_service(normalized)


def get_state_service_module(state_code: str) -> Any:
    return get_state_service(state_code)
