from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

import build_parcel_pmtiles as parcel_pmtiles
from state_artifacts import load_state_artifacts
from state_registry import ROOT, load_state_definition, load_state_registry


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tile_settings(state_code: str) -> parcel_pmtiles.TileBuildSettings:
    args = SimpleNamespace(
        output=None,
        geometry_cache=None,
        summary_output=None,
        publish_manifest_output=None,
        min_zoom=None,
        max_zoom=None,
    )
    return parcel_pmtiles._tile_settings(state_code, args)  # noqa: SLF001


def _state_codes(args: argparse.Namespace) -> list[str]:
    if args.state_code:
        return [str(value).strip().lower() for value in args.state_code]
    registry = load_state_registry()
    return sorted(str(state_code).strip().lower() for state_code in registry.get("states", {}).keys())


def _diagnostics_payload(state_code: str) -> dict[str, Any] | None:
    return _safe_json(ROOT / "data" / "training" / state_code / "state_diagnostics.json")


def _build_summary_for_manifest(settings: parcel_pmtiles.TileBuildSettings) -> dict[str, Any]:
    existing_summary = _safe_json(settings.summary_output_path)
    if existing_summary:
        return existing_summary
    diagnostics = _diagnostics_payload(settings.state_code) or {}
    return {
        "state_code": settings.state_code,
        "state_name": settings.state_name,
        "tile_build_method": "existing_pmtiles_artifact",
        "statewide_parcel_tile_coverage": diagnostics.get("statewide_parcel_tile_coverage")
        or ("full" if settings.build_source == "parcel_master" else "subset"),
        "map_shows_all_parcels": diagnostics.get("map_shows_all_parcels"),
    }


def _ensure_publish_manifest(state_code: str) -> dict[str, Any] | None:
    definition = load_state_definition(state_code)
    if not definition.raw.get("parcel_tiles"):
        return None
    settings = _tile_settings(state_code)
    manifest = parcel_pmtiles._publish_manifest(settings, _build_summary_for_manifest(settings))  # noqa: SLF001
    settings.publish_manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.publish_manifest_output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _check_public_url(public_url: str | None, *, timeout_seconds: float) -> dict[str, Any]:
    if not public_url:
        return {
            "reachable": False,
            "status": "missing_public_url",
            "http_status": None,
            "content_type": None,
            "final_url": None,
            "content_signature": None,
            "error": None,
        }
    try:
        response = requests.get(
            public_url,
            headers={"Range": "bytes=0-15"},
            allow_redirects=True,
            stream=True,
            timeout=timeout_seconds,
        )
        content = response.content[:16]
        content_signature = "pmtiles" if content.startswith(b"PMTiles") else "git_lfs" if content.startswith(b"version https://") else "html" if content.startswith(b"<!DOCTYPE html") else "unknown"
        reachable = 200 <= response.status_code < 300 and content_signature == "pmtiles"
        return {
            "reachable": reachable,
            "status": "reachable" if reachable else "invalid_content",
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "final_url": response.url,
            "content_signature": content_signature,
            "error": None,
        }
    except Exception as exc:
        return {
            "reachable": False,
            "status": "request_failed",
            "http_status": None,
            "content_type": None,
            "final_url": None,
            "content_signature": None,
            "error": str(exc),
        }


def _state_validation_record(state_code: str, *, refresh_manifests: bool, timeout_seconds: float) -> dict[str, Any]:
    definition = load_state_definition(state_code)
    artifacts = load_state_artifacts(state_code)
    tile_config = definition.raw.get("parcel_tiles", {})
    if not tile_config:
        return {
            "state_code": state_code,
            "state_name": definition.state_name,
            "tile_contract_present": False,
            "status": "no_tile_contract",
            "notes": ["State has no parcel PMTiles contract configured."],
        }

    settings = _tile_settings(state_code)
    manifest = _ensure_publish_manifest(state_code) if refresh_manifests else _safe_json(settings.publish_manifest_output_path)
    public_check = _check_public_url(settings.public_url, timeout_seconds=timeout_seconds)

    local_artifact_exists = settings.output_path.exists()
    local_artifact_size_bytes = settings.output_path.stat().st_size if local_artifact_exists else None
    manifest_exists = settings.publish_manifest_output_path.exists()
    notes: list[str] = []
    if not manifest_exists:
        notes.append("Publish manifest is missing.")
    if not local_artifact_exists:
        notes.append("Local PMTiles artifact is missing.")
    if not settings.public_url:
        notes.append("Production public URL is not configured.")
    if public_check["status"] == "invalid_content":
        notes.append(f"Configured public PMTiles URL responded with non-PMTiles content ({public_check['content_signature']}).")
    elif not public_check["reachable"]:
        notes.append("Configured public PMTiles URL did not resolve successfully.")

    return {
        "state_code": state_code,
        "state_name": definition.state_name,
        "tile_contract_present": True,
        "local_dev_url": settings.frontend_url,
        "public_url": settings.public_url,
        "resolved_production_url": settings.public_url,
        "local_artifact_path": str(settings.output_path),
        "local_artifact_exists": local_artifact_exists,
        "local_artifact_size_bytes": local_artifact_size_bytes,
        "manifest_path": str(settings.publish_manifest_output_path),
        "manifest_exists": manifest_exists,
        "manifest_public_url": manifest.get("public_url") if manifest else None,
        "manifest_matches_config": bool(manifest and manifest.get("public_url") == settings.public_url),
        "cloudflare_object_key": settings.cloudflare_object_key,
        "public_asset_status": "available" if public_check["reachable"] else "missing_or_unreachable",
        "public_http_status": public_check["http_status"],
        "public_content_type": public_check["content_type"],
        "public_content_signature": public_check["content_signature"],
        "public_final_url": public_check["final_url"],
        "public_request_error": public_check["error"],
        "status": (
            "public_ready"
            if manifest_exists and local_artifact_exists and public_check["reachable"]
            else "local_only"
            if manifest_exists and local_artifact_exists and not public_check["reachable"]
            else "partial"
        ),
        "notes": notes,
        "runtime_root": str(artifacts.runtime_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the checked-in PMTiles hosting contract across states.")
    parser.add_argument("--state-code", action="append", default=None, help="Optional state code to validate. Repeatable.")
    parser.add_argument("--refresh-manifests", action="store_true", help="Rewrite publish manifests from the checked-in contract before validating.")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "runtime" / "pmtiles_hosting_validation.json"),
        help="Path to write the validation summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states = _state_codes(args)
    records = [
        _state_validation_record(state_code, refresh_manifests=args.refresh_manifests, timeout_seconds=args.timeout_seconds)
        for state_code in states
    ]
    summary = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "states": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
