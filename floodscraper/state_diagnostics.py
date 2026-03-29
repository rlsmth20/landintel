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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coverage_label(*, numerator: int | None, denominator: int | None) -> str | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return "full" if numerator >= denominator else "subset"


def _configured_tile_coverage(build_source: str | None) -> str:
    return "full" if str(build_source or "").strip().lower() == "parcel_master" else "subset"


def _geometry_source_type(*, state_code: str, tile_summary: dict[str, Any] | None, tile_config: dict[str, Any], tile_artifact_exists: bool) -> str:
    if state_code == "ms":
        return "local cached"
    strategy = str((tile_summary or {}).get("geometry_strategy") or tile_config.get("geometry_strategy") or "").strip().lower()
    if "local" in strategy:
        return "local cached"
    if "arcgis" in strategy or tile_artifact_exists:
        return "mixed"
    return "remote fetch"


def _blocker_reason(
    *,
    state_code: str,
    tile_artifact_exists: bool,
    tile_coverage: str | None,
    geometry_coverage: str | None,
    tile_summary: dict[str, Any] | None,
    tile_config: dict[str, Any],
) -> str | None:
    if tile_artifact_exists and tile_coverage == "full" and geometry_coverage == "full":
        return None
    build_source = str((tile_summary or {}).get("build_source") or tile_config.get("build_source") or "").strip().lower()
    if build_source and build_source != "parcel_master":
        return "Parcel overlay is still built from an app_ready or other subset dataset instead of the statewide parcel master."
    if tile_summary and tile_summary.get("geometry_blocker_reason"):
        return str(tile_summary["geometry_blocker_reason"])
    remaining_missing = (tile_summary or {}).get("geometry_remaining_missing_count")
    if remaining_missing:
        return f"{int(remaining_missing):,} parcel rows are still missing usable geometry in the parcel overlay cache."
    if state_code == "ny":
        return "The current official statewide source is centroid-only; polygon coverage is source-limited unless a hybrid or alternate polygon source is added."
    if not tile_artifact_exists:
        return "Statewide parcel overlay artifact is not built yet."
    return "Statewide parcel geometry coverage is still below parcel-master row count."


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
    parcel_master_row_count = None
    if parcel_master.exists():
        parcel_master_row_count = _safe_row_count(parcel_master)
        diagnostics["parcel_master_row_count"] = parcel_master_row_count
        diagnostics["parcel_master_county_coverage"] = _county_coverage_summary(parcel_master)

    app_ready = definition.legacy_path("app_ready_leads") or artifacts.app_ready_path
    app_ready_row_count = None
    if app_ready.exists():
        app_ready_row_count = _safe_row_count(app_ready)
        diagnostics["app_ready_row_count"] = app_ready_row_count
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

    source_alignment_path = artifacts.runtime_root / "source_alignment_audit.json"
    source_alignment_payload = _safe_json(source_alignment_path)
    if source_alignment_payload:
        diagnostics["official_source_alignment"] = source_alignment_payload

    tile_config = definition.raw.get("parcel_tiles", {})
    tile_summary_path = ROOT / str(tile_config.get("summary_output", ""))
    tile_manifest_path = ROOT / str(tile_config.get("publish_manifest_output", ""))
    tile_summary = _safe_json(tile_summary_path) if str(tile_summary_path) != str(ROOT) else None
    tile_manifest = _safe_json(tile_manifest_path) if str(tile_manifest_path) != str(ROOT) else None
    configured_tile_coverage = _configured_tile_coverage(
        tile_summary.get("build_source") if tile_summary and tile_summary.get("build_source") is not None else tile_config.get("build_source")
    )
    tile_artifact_exists = artifacts.frontend_parcel_pmtiles_path.exists()
    tile_coverage = (
        tile_summary.get("statewide_parcel_tile_coverage")
        if tile_summary and tile_summary.get("statewide_parcel_tile_coverage") is not None
        else (configured_tile_coverage if tile_artifact_exists else "subset")
    )
    geometry_coverage = (
        tile_summary.get("statewide_geometry_coverage")
        if tile_summary and tile_summary.get("statewide_geometry_coverage") is not None
        else (configured_tile_coverage if tile_artifact_exists else "subset")
    )
    map_shows_all_parcels = bool(tile_artifact_exists and tile_coverage == "full" and geometry_coverage == "full")
    geometry_source_type = _geometry_source_type(
        state_code=state_code,
        tile_summary=tile_summary,
        tile_config=tile_config,
        tile_artifact_exists=tile_artifact_exists,
    )
    blocker_reason = _blocker_reason(
        state_code=state_code,
        tile_artifact_exists=tile_artifact_exists,
        tile_coverage=tile_coverage,
        geometry_coverage=geometry_coverage,
        tile_summary=tile_summary,
        tile_config=tile_config,
    )
    official_source_row_count = _safe_int((source_alignment_payload or {}).get("official_source_row_count"))
    official_county_count = _safe_int((source_alignment_payload or {}).get("official_county_count"))
    canonical_county_coverage = (source_alignment_payload or {}).get("canonical_county_coverage")
    missing_counties = list((source_alignment_payload or {}).get("missing_counties") or [])
    partial_counties = list((source_alignment_payload or {}).get("partial_counties") or [])
    if canonical_county_coverage != "full" and official_source_row_count and parcel_master_row_count and parcel_master_row_count < official_source_row_count:
        missing_rows = int(official_source_row_count - parcel_master_row_count)
        county_gap_count = int(len(missing_counties) + len(partial_counties))
        blocker_reason = (
            f"Local parcel master is short by {missing_rows:,} rows versus the live official statewide source"
            f"{f' and still has {county_gap_count} counties with missing or partial coverage' if county_gap_count else ''}."
        )
    diagnostics["parcel_tile_artifact"] = {
        "path": str(artifacts.frontend_parcel_pmtiles_path),
        "exists": tile_artifact_exists,
        "size_bytes": artifacts.frontend_parcel_pmtiles_path.stat().st_size if tile_artifact_exists else None,
        "frontend_url": tile_config.get("frontend_url"),
        "public_url": tile_config.get("public_url"),
        "statewide_parcel_tile_coverage": tile_coverage,
        "statewide_geometry_coverage": geometry_coverage,
        "map_shows_all_parcels": map_shows_all_parcels,
        "geometry_source_type": geometry_source_type,
        "blocker_reason": blocker_reason,
        "build_summary": tile_summary,
        "publish_manifest": tile_manifest,
    }

    runtime_detail_row_count = None
    runtime_detail_path = definition.legacy_path("runtime_detail_metrics") or artifacts.runtime_detail_metrics_path
    if runtime_detail_path.exists():
        runtime_detail_row_count = _safe_row_count(runtime_detail_path)
    diagnostics["statewide_parcel_base"] = {
        "row_count": parcel_master_row_count,
        "coverage": canonical_county_coverage or _coverage_label(numerator=parcel_master_row_count, denominator=official_source_row_count) or ("full" if parcel_master_row_count else None),
        "official_source_row_count": official_source_row_count,
        "official_source_county_count": official_county_count,
        "raw_source_row_gap": official_source_row_count - parcel_master_row_count if official_source_row_count and parcel_master_row_count else None,
    }
    diagnostics["statewide_parcel_tile_coverage"] = tile_coverage
    diagnostics["statewide_geometry_coverage"] = geometry_coverage
    diagnostics["app_ready_default_lead_row_count"] = app_ready_row_count
    diagnostics["lead_coverage"] = _coverage_label(numerator=app_ready_row_count, denominator=parcel_master_row_count)
    diagnostics["map_shows_all_parcels"] = map_shows_all_parcels
    diagnostics["geometry_source_type"] = geometry_source_type
    diagnostics["blocker_reason"] = blocker_reason
    diagnostics["runtime_parcel_detail_coverage"] = (
        "full"
        if parcel_master_row_count and runtime_detail_row_count and runtime_detail_row_count >= parcel_master_row_count
        else "full"
        if parcel_master_row_count and state_code != "ms"
        else _coverage_label(numerator=runtime_detail_row_count, denominator=parcel_master_row_count)
    )

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
