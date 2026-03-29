from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from floodscraper.state_registry import ROOT, load_state_definition
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from state_registry import ROOT, load_state_definition


FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"
FRONTEND_TILES_DIR = ROOT / "frontend" / "public" / "tiles"
BUILDINGS_PROCESSED_DIR = ROOT / "data" / "buildings_processed"


@dataclass(frozen=True)
class StateArtifactPaths:
    state_code: str
    state_name: str
    parcels_root: Path
    runtime_root: Path
    review_root: Path
    training_root: Path
    parcel_master_path: Path
    owner_leads_path: Path
    parcel_building_metrics_path: Path
    app_ready_path: Path
    delinquent_leads_statewide_path: Path
    tax_distress_path: Path
    county_coverage_matrix_path: Path
    runtime_parcel_index_root: Path
    runtime_geometry_index_root: Path
    runtime_detail_metrics_path: Path
    runtime_tax_coverage_matrix_path: Path
    runtime_summary_path: Path
    runtime_presets_path: Path
    runtime_default_leads_path: Path
    runtime_default_geometry_path: Path
    frontend_static_feed_path: Path
    frontend_meta_path: Path
    frontend_geometry_path: Path
    frontend_detail_fallback_path: Path
    frontend_parcel_pmtiles_path: Path
    review_sample_path: Path
    review_sample_summary_path: Path
    reviewed_pilot_input_path: Path
    reviewed_pilot_manifest_path: Path
    reviewed_pilot_summary_path: Path
    reviewed_pilot_predictions_path: Path
    reviewed_pilot_error_analysis_path: Path
    reviewed_pilot_error_summary_path: Path
    reviewed_pilot_model_path: Path
    training_feature_manifest_path: Path
    ai_training_manifest_path: Path
    ai_model_path: Path
    ai_model_metrics_path: Path
    ai_runtime_model_params_path: Path
    ai_runtime_model_metrics_path: Path
    ai_predictions_path: Path
    ai_tile_cache_dir: Path
    geometry_quality_artifact_path: Path
    geometry_quality_summary_path: Path


def _legacy_or_default(definition, legacy_key: str, default_path: Path) -> Path:
    configured = definition.legacy_path(legacy_key)
    return configured if configured is not None else default_path


def load_state_artifacts(state_code: str) -> StateArtifactPaths:
    definition = load_state_definition(state_code)
    state_code = definition.state_code
    parcels_root = definition.artifact_root("parcels")
    runtime_root = _legacy_or_default(definition, "backend_runtime_dir", definition.artifact_root("runtime"))
    review_root = definition.artifact_root("review")
    training_root = definition.artifact_root("training")

    return StateArtifactPaths(
        state_code=state_code,
        state_name=definition.state_name,
        parcels_root=parcels_root,
        runtime_root=runtime_root,
        review_root=review_root,
        training_root=training_root,
        parcel_master_path=_legacy_or_default(definition, "parcel_master", parcels_root / f"{state_code}_parcels_master.parquet"),
        owner_leads_path=_legacy_or_default(definition, "owner_leads", parcels_root / f"{state_code}_parcels_owner_leads.parquet"),
        parcel_building_metrics_path=_legacy_or_default(
            definition,
            "parcel_building_metrics",
            BUILDINGS_PROCESSED_DIR / f"parcel_building_metrics_{state_code}.parquet",
        ),
        app_ready_path=_legacy_or_default(
            definition,
            "app_ready_leads",
            ROOT / "data" / "tax_published" / state_code / f"app_ready_{state_code}_leads.parquet",
        ),
        delinquent_leads_statewide_path=_legacy_or_default(
            definition,
            "delinquent_leads_statewide",
            ROOT / "data" / "tax_published" / state_code / "delinquent_leads_statewide.parquet",
        ),
        tax_distress_path=_legacy_or_default(definition, "tax_distress", parcels_root / f"{state_code}_parcels_tax_distress.parquet"),
        county_coverage_matrix_path=_legacy_or_default(
            definition,
            "county_coverage_matrix",
            parcels_root / f"{state_code}_tax_coverage_matrix.parquet",
        ),
        runtime_parcel_index_root=_legacy_or_default(definition, "runtime_parcel_index", runtime_root / "parcel_index"),
        runtime_geometry_index_root=_legacy_or_default(definition, "runtime_geometry_index", runtime_root / "parcel_geometry_index"),
        runtime_detail_metrics_path=_legacy_or_default(definition, "runtime_detail_metrics", runtime_root / "parcel_detail_metrics.parquet"),
        runtime_tax_coverage_matrix_path=runtime_root / "tax_coverage_matrix.parquet",
        runtime_summary_path=_legacy_or_default(definition, "runtime_summary", runtime_root / "summary.json"),
        runtime_presets_path=_legacy_or_default(definition, "runtime_presets", runtime_root / "presets.json"),
        runtime_default_leads_path=_legacy_or_default(definition, "runtime_default_leads", runtime_root / "default_leads.json"),
        runtime_default_geometry_path=_legacy_or_default(definition, "runtime_default_geometry", runtime_root / "default_geometry.json"),
        frontend_static_feed_path=_legacy_or_default(definition, "frontend_static_feed", FRONTEND_DATA_DIR / f"{state_code}_lead_explorer.json"),
        frontend_meta_path=_legacy_or_default(definition, "frontend_meta", FRONTEND_DATA_DIR / f"{state_code}_lead_explorer_meta.json"),
        frontend_geometry_path=_legacy_or_default(
            definition,
            "frontend_geometry",
            FRONTEND_DATA_DIR / f"{state_code}_lead_explorer_geometries.json",
        ),
        frontend_detail_fallback_path=_legacy_or_default(
            definition,
            "frontend_detail_fallback",
            FRONTEND_DATA_DIR / f"{state_code}_lead_detail_fallback.json",
        ),
        frontend_parcel_pmtiles_path=_legacy_or_default(definition, "frontend_parcel_pmtiles", FRONTEND_TILES_DIR / f"{state_code}_parcels.pmtiles"),
        review_sample_path=_legacy_or_default(definition, "review_sample", review_root / "vacancy_training_review_sample_300.csv"),
        review_sample_summary_path=_legacy_or_default(
            definition,
            "review_sample_summary",
            review_root / "vacancy_training_review_sample_300_summary.json",
        ),
        reviewed_pilot_input_path=_legacy_or_default(definition, "reviewed_pilot_input", training_root / "reviewed_pilot_input.csv"),
        reviewed_pilot_manifest_path=_legacy_or_default(definition, "reviewed_pilot_manifest", training_root / f"ai_building_presence_training_manifest_{state_code}_reviewed_pilot.parquet"),
        reviewed_pilot_summary_path=_legacy_or_default(definition, "reviewed_pilot_summary", training_root / f"ai_building_presence_training_manifest_{state_code}_reviewed_pilot_summary.json"),
        reviewed_pilot_predictions_path=_legacy_or_default(definition, "reviewed_pilot_predictions", training_root / f"ai_building_presence_reviewed_pilot_cv_predictions_{state_code}.csv"),
        reviewed_pilot_error_analysis_path=_legacy_or_default(definition, "reviewed_pilot_error_analysis", training_root / f"ai_building_presence_reviewed_pilot_error_analysis_{state_code}.csv"),
        reviewed_pilot_error_summary_path=_legacy_or_default(definition, "reviewed_pilot_error_summary", training_root / f"ai_building_presence_reviewed_pilot_error_analysis_summary_{state_code}.json"),
        reviewed_pilot_model_path=_legacy_or_default(definition, "reviewed_pilot_model", training_root / f"ai_building_presence_model_{state_code}_reviewed_pilot.joblib"),
        training_feature_manifest_path=_legacy_or_default(definition, "training_feature_manifest", training_root / f"ai_building_presence_training_manifest_{state_code}_geometry_quality.parquet"),
        ai_training_manifest_path=_legacy_or_default(definition, "ai_training_manifest", training_root / f"ai_building_presence_training_manifest_{state_code}.parquet"),
        ai_model_path=_legacy_or_default(definition, "ai_model", training_root / f"ai_building_presence_model_{state_code}.joblib"),
        ai_model_metrics_path=_legacy_or_default(definition, "ai_model_metrics", training_root / f"ai_building_presence_model_metrics_{state_code}.json"),
        ai_runtime_model_params_path=_legacy_or_default(definition, "ai_runtime_model_params", runtime_root / f"ai_building_presence_model_{state_code}.json"),
        ai_runtime_model_metrics_path=_legacy_or_default(definition, "ai_runtime_model_metrics", runtime_root / f"ai_building_presence_model_metrics_{state_code}.json"),
        ai_predictions_path=_legacy_or_default(definition, "ai_predictions", training_root / f"ai_building_presence_predictions_{state_code}.parquet"),
        ai_tile_cache_dir=_legacy_or_default(definition, "ai_tile_cache_dir", review_root / f"ai_building_tiles_{state_code}"),
        geometry_quality_artifact_path=_legacy_or_default(definition, "geometry_quality_artifact", parcels_root / f"{state_code}_parcel_geometry_quality.parquet"),
        geometry_quality_summary_path=_legacy_or_default(definition, "geometry_quality_summary", parcels_root / f"{state_code}_parcel_geometry_quality_summary.json"),
    )
