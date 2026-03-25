from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from state_artifacts import load_state_artifacts
from state_registry import ROOT, load_state_definition


def _safe_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        try:
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return int(len(frame))


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _schema_mapping_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _safe_json(path)
    if not payload:
        return None
    canonical_identity = payload.get("canonical_identity", {})
    parcel_source_fields = payload.get("parcel_source_fields", {})
    return {
        "path": str(path),
        "canonical_identity_fields": sorted(str(key) for key in canonical_identity.keys()),
        "canonical_identity_mapping": canonical_identity,
        "parcel_source_field_count": int(len(parcel_source_fields)),
        "parcel_source_fields": sorted(str(key) for key in parcel_source_fields.keys()),
    }


def _county_coverage_summary(path: Path | None, *, max_counties: int = 10) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() != ".parquet":
        return None
    try:
        frame = pd.read_parquet(path, columns=["county_name"])
    except Exception:
        return None
    if "county_name" not in frame.columns:
        return None
    normalized = frame["county_name"].astype("string").dropna().str.strip().str.lower()
    counts = normalized.value_counts()
    return {
        "row_count": int(len(frame)),
        "county_count": int(counts.shape[0]),
        "top_counties_by_row_count": [
            {"county_name": str(county), "row_count": int(count)}
            for county, count in counts.head(max_counties).items()
        ],
    }


def _geometry_quality_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "row_count": payload.get("processed_rows") or payload.get("row_count"),
        "build_start_timestamp": payload.get("build_start_timestamp"),
        "build_end_timestamp": payload.get("build_end_timestamp"),
        "runtime_by_stage_seconds": payload.get("runtime_by_stage_seconds"),
        "top_counties_by_review_excluded_count": payload.get("top_counties_by_review_excluded_count"),
        "top_counties_by_training_excluded_count": payload.get("top_counties_by_training_excluded_count"),
    }


def _marketability_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    diagnostics = payload.get("geometry_marketability_diagnostics", {})
    return {
        "geometry_marketability_flag_counts": diagnostics.get("geometry_marketability_flag_counts"),
        "geometry_marketability_action_counts": diagnostics.get("geometry_marketability_action_counts"),
        "default_leads_excluded_count": diagnostics.get("default_leads_excluded_count"),
        "default_leads_excluded_pct": diagnostics.get("default_leads_excluded_pct"),
        "top_counties_affected": diagnostics.get("top_counties_affected"),
    }


def _review_eligibility_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    diagnostics = payload.get("manual_review_diagnostics", {})
    return {
        "eligible_count": diagnostics.get("eligible_count"),
        "eligible_pct": diagnostics.get("eligible_pct"),
        "excluded_count": diagnostics.get("excluded_count"),
        "excluded_pct": diagnostics.get("excluded_pct"),
        "exclusion_reason_counts": diagnostics.get("exclusion_reason_counts"),
        "top_exclusion_reasons": diagnostics.get("top_exclusion_reasons"),
    }


def _reviewed_pilot_metrics_summary(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    training_eval = payload.get("training_evaluation", {})
    threshold_guidance = training_eval.get("threshold_guidance", {})
    return {
        "rows_used_after_label_validation": payload.get("rows_used_after_label_validation"),
        "pilot_manifest_parcel_count": payload.get("pilot_manifest_parcel_count"),
        "pilot_manifest_row_count": payload.get("pilot_manifest_row_count"),
        "feature_source_workflow": payload.get("feature_source_workflow"),
        "feature_source_parcel_counts": payload.get("feature_source_parcel_counts"),
        "evaluation_method": training_eval.get("evaluation_method"),
        "parcel_level_auc": training_eval.get("parcel_level_auc"),
        "recommended_threshold": threshold_guidance.get("recommended_threshold"),
        "recommended_metrics": threshold_guidance.get("recommended_metrics"),
    }


def build_state_diagnostics(state_code: str) -> dict[str, Any]:
    definition = load_state_definition(state_code)
    artifacts = load_state_artifacts(state_code)
    diagnostics: dict[str, Any] = {
        "state_code": definition.state_code,
        "state_name": definition.state_name,
        "county_division_label": definition.county_division_label,
        "artifact_roots": {key: str(path) for key, path in definition.artifact_roots.items()},
        "legacy_paths": {key: str(path) for key, path in definition.legacy_paths.items()},
        "source_registry_paths": {key: str(path) for key, path in definition.source_registry_paths.items()},
        "schema_mapping_summary": _schema_mapping_summary(definition.schema_mapping_path("parcel_schema_mapping")),
    }

    parcel_master = definition.legacy_path("parcel_master") or artifacts.parcel_master_path
    if parcel_master.exists():
        diagnostics["parcel_master_row_count"] = _safe_row_count(parcel_master)
        diagnostics["parcel_master_county_coverage"] = _county_coverage_summary(parcel_master)

    app_ready = definition.legacy_path("app_ready_leads") or artifacts.app_ready_path
    if app_ready.exists():
        diagnostics["app_ready_row_count"] = _safe_row_count(app_ready)
        diagnostics["app_ready_county_coverage"] = _county_coverage_summary(app_ready)

    review_summary = definition.legacy_path("review_sample_summary") or artifacts.review_sample_summary_path
    if review_summary.exists():
        review_payload = _safe_json(review_summary)
        diagnostics["review_summary"] = review_payload
        diagnostics["review_eligibility_summary"] = _review_eligibility_summary(review_payload)

    reviewed_pilot_summary = definition.legacy_path("reviewed_pilot_summary") or artifacts.reviewed_pilot_summary_path
    if reviewed_pilot_summary.exists():
        reviewed_pilot_payload = _safe_json(reviewed_pilot_summary)
        diagnostics["reviewed_pilot_summary"] = reviewed_pilot_payload
        diagnostics["reviewed_pilot_metrics_summary"] = _reviewed_pilot_metrics_summary(reviewed_pilot_payload)

    geometry_quality_summary = definition.legacy_path("geometry_quality_summary") or artifacts.geometry_quality_summary_path
    if geometry_quality_summary.exists():
        geometry_quality_payload = _safe_json(geometry_quality_summary)
        diagnostics["geometry_quality_summary"] = geometry_quality_payload
        diagnostics["geometry_quality_overview"] = _geometry_quality_summary(geometry_quality_payload)

    runtime_summary = definition.legacy_path("runtime_summary") or artifacts.runtime_summary_path
    if runtime_summary.exists():
        runtime_payload = _safe_json(runtime_summary)
        diagnostics["runtime_summary"] = runtime_payload
        diagnostics["marketability_summary"] = _marketability_summary(runtime_payload)

    tile_summary_path = ROOT / str(definition.raw.get("parcel_tiles", {}).get("summary_output", ""))
    tile_manifest_path = ROOT / str(definition.raw.get("parcel_tiles", {}).get("publish_manifest_output", ""))
    tile_summary = _safe_json(tile_summary_path) if str(tile_summary_path) != str(ROOT) else None
    tile_manifest = _safe_json(tile_manifest_path) if str(tile_manifest_path) != str(ROOT) else None
    diagnostics["parcel_tile_artifact"] = {
        "path": str(artifacts.frontend_parcel_pmtiles_path),
        "exists": artifacts.frontend_parcel_pmtiles_path.exists(),
        "size_bytes": artifacts.frontend_parcel_pmtiles_path.stat().st_size if artifacts.frontend_parcel_pmtiles_path.exists() else None,
        "frontend_url": definition.raw.get("parcel_tiles", {}).get("frontend_url"),
        "build_summary": tile_summary,
        "publish_manifest": tile_manifest,
    }

    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit standard diagnostics for a configured state.")
    parser.add_argument("state_code")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build_state_diagnostics(args.state_code)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
