from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_REGISTRY_PATH = PROJECT_ROOT / "config" / "states" / "registry.json"
logger = logging.getLogger("state-runtime")
STATE_SERVICE_MODULES = {
    "ar": "app.services.arkansas_leads_service",
    "ms": "app.services.mississippi_leads_service",
}


def configured_state_codes() -> list[str]:
    if not STATE_REGISTRY_PATH.exists():
        return []
    payload = json.loads(STATE_REGISTRY_PATH.read_text(encoding="utf-8"))
    states = payload.get("states", {})
    return sorted(str(state_code).lower() for state_code in states.keys())


def supported_state_codes() -> list[str]:
    configured = set(configured_state_codes())
    return sorted(state_code for state_code in STATE_SERVICE_MODULES.keys() if state_code in configured)


def get_state_service_module(state_code: str) -> ModuleType:
    normalized = state_code.strip().lower()
    if normalized not in configured_state_codes():
        raise KeyError(f"State code is not configured: {normalized}")
    module_name = STATE_SERVICE_MODULES.get(normalized)
    if module_name is None:
        raise NotImplementedError(f"State code is configured but no backend service module is registered yet: {normalized}")
    logger.info("Resolving state service state=%s module=%s", normalized, module_name)
    return importlib.import_module(module_name)
