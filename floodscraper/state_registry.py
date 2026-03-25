from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_CONFIG_DIR = ROOT / "config" / "states"
STATE_REGISTRY_PATH = STATE_CONFIG_DIR / "registry.json"


@dataclass(frozen=True)
class StateDefinition:
    state_code: str
    state_name: str
    county_division_label: str
    artifact_roots: dict[str, Path]
    legacy_paths: dict[str, Path]
    source_registry_paths: dict[str, Path]
    schema_mapping_templates: dict[str, Path]
    raw: dict[str, Any]

    def artifact_root(self, key: str) -> Path:
        return self.artifact_roots[key]

    def legacy_path(self, key: str) -> Path | None:
        return self.legacy_paths.get(key)

    def source_registry_path(self, key: str) -> Path | None:
        return self.source_registry_paths.get(key)

    def schema_mapping_path(self, key: str) -> Path | None:
        return self.schema_mapping_templates.get(key)


def _resolve_path_map(values: dict[str, Any]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key, value in values.items():
        resolved[key] = ROOT / str(value)
    return resolved


def load_state_registry() -> dict[str, Any]:
    if not STATE_REGISTRY_PATH.exists():
        raise FileNotFoundError(f"State registry not found: {STATE_REGISTRY_PATH}")
    return json.loads(STATE_REGISTRY_PATH.read_text(encoding="utf-8"))


def load_state_definition(state_code: str) -> StateDefinition:
    normalized = state_code.strip().lower()
    registry = load_state_registry()
    states = registry.get("states", {})
    if normalized not in states:
        raise KeyError(f"State code not registered: {normalized}")
    config_path = ROOT / str(states[normalized]["config_path"])
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return StateDefinition(
        state_code=str(payload["state_code"]).lower(),
        state_name=str(payload["state_name"]),
        county_division_label=str(payload.get("county_division_label", "county")),
        artifact_roots=_resolve_path_map(payload.get("artifact_roots", {})),
        legacy_paths=_resolve_path_map(payload.get("legacy_paths", {})),
        source_registry_paths=_resolve_path_map(payload.get("source_registry_paths", {})),
        schema_mapping_templates=_resolve_path_map(payload.get("schema_mapping_templates", {})),
        raw=payload,
    )


def ensure_state_directories(state_code: str) -> StateDefinition:
    definition = load_state_definition(state_code)
    for path in definition.artifact_roots.values():
        path.mkdir(parents=True, exist_ok=True)
    return definition


def reviewed_pilot_default_outputs(state_code: str, *, run_name: str = "reviewed_pilot") -> dict[str, Path]:
    definition = load_state_definition(state_code)
    training_root = definition.artifact_root("training")
    legacy_keys = {
        "review_input": "reviewed_pilot_input",
        "feature_manifest": "training_feature_manifest",
        "manifest": "reviewed_pilot_manifest",
        "summary": "reviewed_pilot_summary",
        "cv_predictions": "reviewed_pilot_predictions",
        "error_analysis": "reviewed_pilot_error_analysis",
        "error_summary": "reviewed_pilot_error_summary",
        "model": "reviewed_pilot_model",
    }
    defaults: dict[str, Path] = {}
    for output_key, legacy_key in legacy_keys.items():
        legacy_path = definition.legacy_path(legacy_key)
        if legacy_path is not None:
            defaults[output_key] = legacy_path
    if "review_input" not in defaults:
        defaults["review_input"] = training_root / f"{run_name}_input.csv"
    if "feature_manifest" not in defaults:
        defaults["feature_manifest"] = training_root / f"{run_name}_feature_manifest.parquet"
    if "manifest" not in defaults:
        defaults["manifest"] = training_root / f"{run_name}_training_manifest.parquet"
    if "summary" not in defaults:
        defaults["summary"] = training_root / f"{run_name}_summary.json"
    if "cv_predictions" not in defaults:
        defaults["cv_predictions"] = training_root / f"{run_name}_cv_predictions.csv"
    if "error_analysis" not in defaults:
        defaults["error_analysis"] = training_root / f"{run_name}_error_analysis.csv"
    if "error_summary" not in defaults:
        defaults["error_summary"] = training_root / f"{run_name}_error_analysis_summary.json"
    if "model" not in defaults:
        defaults["model"] = training_root / f"{run_name}_model.joblib"
    return defaults


def state_bootstrap_template(state_code: str, state_name: str, *, county_division_label: str = "county") -> dict[str, Any]:
    normalized = state_code.strip().lower()
    return {
        "state_code": normalized,
        "state_name": state_name,
        "county_division_label": county_division_label,
        "artifact_roots": {
            "parcels": f"data/parcels/{normalized}",
            "runtime": f"data/runtime/{normalized}",
            "review": f"data/review/{normalized}",
            "training": f"data/training/{normalized}",
        },
        "legacy_paths": {},
        "source_registry_paths": {
            "parcel_source": f"floodscraper/state_configs/parcel_source_{normalized}.json",
            "tax_source": f"floodscraper/state_configs/tax_source_{normalized}.json",
            "tax_linkage": f"floodscraper/state_configs/tax_linkage_{normalized}.json",
        },
        "schema_mapping_templates": {
            "parcel_schema_mapping": f"config/states/{normalized}_parcel_schema_mapping.json",
        },
        "notes": {
            "bootstrap_status": "placeholder",
            "reference_state": "ms",
        },
    }
