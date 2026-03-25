from __future__ import annotations

import argparse
import json
from pathlib import Path

from state_registry import ROOT, STATE_CONFIG_DIR, STATE_REGISTRY_PATH, state_bootstrap_template


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def bootstrap_state(
    *,
    state_code: str,
    state_name: str,
    county_division_label: str,
    force: bool,
) -> dict[str, str]:
    normalized = state_code.strip().lower()
    template = state_bootstrap_template(normalized, state_name, county_division_label=county_division_label)

    config_path = STATE_CONFIG_DIR / f"{normalized}.json"
    registry_payload = {"states": {}}
    if STATE_REGISTRY_PATH.exists():
        registry_payload = json.loads(STATE_REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_payload.setdefault("states", {})
    registry_payload["states"][normalized] = {
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "status": "bootstrapped",
    }

    if config_path.exists() and not force:
        raise FileExistsError(f"State config already exists: {config_path}")

    artifact_dirs = {
        key: ROOT / relative_path
        for key, relative_path in template["artifact_roots"].items()
    }
    for directory in artifact_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    parcel_source_path = ROOT / template["source_registry_paths"]["parcel_source"]
    tax_source_path = ROOT / template["source_registry_paths"]["tax_source"]
    tax_linkage_path = ROOT / template["source_registry_paths"]["tax_linkage"]
    schema_mapping_path = ROOT / template["schema_mapping_templates"]["parcel_schema_mapping"]

    _write_json(config_path, template)
    _write_json(STATE_REGISTRY_PATH, registry_payload)
    _write_json(
        parcel_source_path,
        {
            "state_code": normalized,
            "state_name": state_name,
            "status": "placeholder",
            "parcel_sources": [],
        },
    )
    _write_json(
        tax_source_path,
        {
            "state_code": normalized,
            "state_name": state_name,
            "status": "placeholder",
            "tax_sources": [],
        },
    )
    _write_json(
        tax_linkage_path,
        {
            "state_code": normalized,
            "state_name": state_name,
            "status": "placeholder",
            "linkage_rules": [],
        },
    )
    _write_json(
        schema_mapping_path,
        {
            "state_code": normalized,
            "state_name": state_name,
            "canonical_identity": {
                "parcel_row_id": "",
                "parcel_id": "",
                "state_code": "state_code",
                "county_name": "",
                "county_fips": "",
                "geometry": "",
            },
            "parcel_source_fields": {},
            "notes": {
                "status": "placeholder",
                "reference_state": "ms",
            },
        },
    )

    return {
        "state_code": normalized,
        "config_path": str(config_path),
        "parcel_source_registry_path": str(parcel_source_path),
        "tax_source_registry_path": str(tax_source_path),
        "tax_linkage_registry_path": str(tax_linkage_path),
        "schema_mapping_path": str(schema_mapping_path),
        "artifact_root_parcels": str(artifact_dirs["parcels"]),
        "artifact_root_runtime": str(artifact_dirs["runtime"]),
        "artifact_root_review": str(artifact_dirs["review"]),
        "artifact_root_training": str(artifact_dirs["training"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a new state config and folder structure.")
    parser.add_argument("state_code")
    parser.add_argument("state_name")
    parser.add_argument("--county-division-label", default="county")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = bootstrap_state(
        state_code=args.state_code,
        state_name=args.state_name,
        county_division_label=args.county_division_label,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
