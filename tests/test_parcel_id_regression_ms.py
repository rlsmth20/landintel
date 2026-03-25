from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))
sys.path.insert(0, str(ROOT / "backend"))

import build_backend_parcel_runtime_ms as runtime_builder  # noqa: E402
import build_frontend_detail_fallback_ms as detail_fallback  # noqa: E402
import parcel_contract_ms as parcel_contract  # noqa: E402
import sample_vacancy_labeling_ms as labeling_sample  # noqa: E402
import vacancy_ai_common as vacancy_common  # noqa: E402
import vacancy_ai_infer_ms as vacancy_infer  # noqa: E402
from app.services import mississippi_leads_service as leads_service  # noqa: E402


class _DummyPipeline:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.25, 0.75] for _ in range(len(frame))], dtype="float64")


class ParcelIdRegressionTests(unittest.TestCase):
    def test_load_candidate_frame_preserves_parcel_id(self) -> None:
        parcels = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "state_code": "MS",
                    "county_name": "alpha",
                    "county_fips": "001",
                    "latitude": 32.0,
                    "longitude": -90.0,
                    "total_acres": 1.0,
                    "parcel_area_acres": np.nan,
                    "gis_acres": np.nan,
                    "tax_acres": np.nan,
                }
            ]
        )
        buildings = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "building_count": 0,
                    "building_area_total": 0.0,
                    "parcel_vacant_flag": True,
                }
            ]
        )

        def fake_read_parquet(path, columns=None, engine=None):  # type: ignore[no-untyped-def]
            if Path(path) == vacancy_common.PARCEL_MASTER_PATH:
                return parcels.loc[:, columns].copy()
            if Path(path) == vacancy_common.BUILDING_METRICS_PATH:
                return buildings.loc[:, columns].copy()
            raise AssertionError(f"Unexpected parquet read: {path}")

        with mock.patch.object(vacancy_common.pd, "read_parquet", side_effect=fake_read_parquet):
            frame = vacancy_common.load_candidate_frame()

        self.assertIn("parcel_id", frame.columns)
        self.assertEqual(frame.iloc[0]["parcel_id"], "pid-1")
        self.assertEqual(frame.iloc[0]["parcel_row_id"], "row_1")

    def test_infer_prediction_row_preserves_parcel_id_in_parcel_and_tile_outputs(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_test_parcel_id_infer"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            tile_path = temp_path / "tile.png"
            Image.new("RGB", (256, 256), (194, 170, 138)).save(tile_path, format="PNG")
            address = vacancy_common.TileAddress(x=100, y=200, z=19)
            tile_plan = {
                "tile_records": [
                    {
                        "address": address,
                        "tile_label": vacancy_common.tile_label(address),
                        "tile_coordinate": vacancy_common.tile_coordinate(address),
                        "tile_rank": 1,
                        "centroid_tile_flag": True,
                        "parcel_tile_coverage_ratio": 1.0,
                        "parcel_tile_coverage_pct": 100.0,
                        "parcel_bbox_tile_coverage_ratio": 1.0,
                        "parcel_bbox_tile_coverage_pct": 100.0,
                        "full_parcel_visible_flag": True,
                        "parcel_extent_exceeds_tile_flag": False,
                        "parcel_tile_low_coverage_flag": False,
                        "multi_tile_candidate_flag": False,
                        "parcel_covering_tile_count": 1,
                        "parcel_coverage_diagnostics_ready_flag": True,
                    }
                ],
                "parcel_tile_coverage_pct": 100.0,
                "parcel_bbox_tile_coverage_pct": 100.0,
                "full_parcel_visible_flag": True,
                "parcel_extent_exceeds_tile_flag": False,
                "parcel_tile_low_coverage_flag": False,
                "multi_tile_candidate_flag": False,
                "parcel_covering_tile_count": 1,
                "tile_coordinates": json.dumps([vacancy_common.tile_coordinate(address)]),
                "unique_tile_count": 1,
                "duplicate_tile_flag": False,
            }
            row = {
                "parcel_row_id": "row_1",
                "parcel_id": "pid-1",
                "county_name": "alpha",
                "longitude": -90.0,
                "latitude": 32.0,
                "acreage": 1.0,
                "parcel_vacant_flag": True,
                "geometry": None,
            }

            with (
                mock.patch.object(vacancy_infer, "build_parcel_inference_tile_plan", return_value=tile_plan),
                mock.patch.object(vacancy_infer, "ensure_tile_image_for_address", return_value=tile_path),
            ):
                result = vacancy_infer.infer_prediction_row(
                    row,
                    pipeline=_DummyPipeline(),
                    columns=["r_mean"],
                    model_version="test_model",
                    zoom=19,
                    refresh=False,
                    tile_template=vacancy_common.DEFAULT_TILE_URL_TEMPLATE,
                    use_parcel_mask=True,
                    outside_mask_fill="black",
                    outside_mask_dim_factor=0.0,
                    parcel_buffer_pixels=8,
                    use_multi_tile_extent=True,
                    include_tile_debug_rows=True,
                )

            self.assertEqual(result["parcel_id"], "pid-1")
            self.assertEqual(result["parcel_row_id"], "row_1")
            self.assertEqual(result["_tile_debug_rows"][0]["parcel_id"], "pid-1")
            self.assertEqual(result["_tile_debug_rows"][0]["parcel_row_id"], "row_1")
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_review_export_and_manifest_preserve_parcel_id(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_test_parcel_id_review"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            centroid_image_path = temp_path / "centroid.png"
            Image.new("RGB", (256, 256), (10, 20, 30)).save(centroid_image_path, format="PNG")
            address = vacancy_common.TileAddress(x=100, y=200, z=19)
            tile_plan = {
                "tile_records": [
                    {
                        "address": address,
                        "tile_label": "z19_x100_y200",
                        "tile_coordinate": "19/100/200",
                        "centroid_tile_flag": True,
                        "tile_rank": 1,
                        "parcel_tile_coverage_pct": 100.0,
                        "parcel_bbox_tile_coverage_pct": 100.0,
                    }
                ],
                "parcel_tile_coverage_pct": 100.0,
                "parcel_bbox_tile_coverage_pct": 100.0,
                "full_parcel_visible_flag": True,
                "parcel_extent_exceeds_tile_flag": False,
                "parcel_tile_low_coverage_flag": False,
                "multi_tile_candidate_flag": False,
                "parcel_covering_tile_count": 1,
                "tile_coordinates": json.dumps(["19/100/200"]),
                "unique_tile_count": 1,
                "duplicate_tile_flag": False,
            }
            frame = pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_1",
                        "parcel_id": "pid-1",
                        "county_name": "alpha",
                        "longitude": -90.0,
                        "latitude": 32.0,
                        "acreage": 1.0,
                        "best_tile_label": "z19_x100_y200",
                        "geometry": None,
                    }
                ]
            )

            def fake_prepare(image_source, **kwargs):  # type: ignore[no-untyped-def]
                image = Image.open(image_source).convert("RGB")
                return {
                    "image": image,
                    "crop_specs": [("parcel_core", (0, 0, 64, 64)), ("parcel_focus", (0, 0, 96, 96))],
                    "parcel_boundary_crop_ready_flag": True,
                    "imagery_crop_strategy": "parcel_mask_tight_crop_v2",
                    "original_geom_type": "Polygon",
                    "clipped_geom_type": "Polygon",
                    "polygon_part_count": 1,
                    "clipped_polygon_part_count": 1,
                    "bounds_before_clip": [0.0, 0.0, 1.0, 1.0],
                    "bounds_after_clip": [0.0, 0.0, 1.0, 1.0],
                }

            original_asset_dir = labeling_sample.REVIEW_ASSET_DIR
            try:
                labeling_sample.REVIEW_ASSET_DIR = temp_path / "review_assets"
                with (
                    mock.patch.object(labeling_sample, "build_parcel_inference_tile_plan", return_value=tile_plan),
                    mock.patch.object(labeling_sample, "_resolve_review_tile_path", return_value=centroid_image_path),
                    mock.patch.object(labeling_sample, "prepare_parcel_aware_image", side_effect=fake_prepare),
                ):
                    attached = labeling_sample.attach_imagery_columns(
                        frame,
                        zoom=19,
                        fetch_images=False,
                        tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
                    )
            finally:
                labeling_sample.REVIEW_ASSET_DIR = original_asset_dir

            row = attached.iloc[0]
            self.assertEqual(row["parcel_id"], "pid-1")
            manifest = json.loads(Path(row["review_tile_manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["parcel_id"], "pid-1")
            self.assertEqual(manifest[0]["parcel_row_id"], "row_1")
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_merge_optional_frame_preserves_left_parcel_id_without_suffix(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_test_parcel_id_merge"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            source_path = temp_path / "source.parquet"
            pd.DataFrame([{"parcel_row_id": "row_1", "parcel_id": "pid-source"}]).to_parquet(
                source_path,
                index=False,
                engine="pyarrow",
            )
            frame = pd.DataFrame([{"parcel_row_id": "row_1", "parcel_id": "pid-original"}])

            merged = labeling_sample._merge_optional_frame(frame, source_path, ["parcel_row_id", "parcel_id"])

            self.assertIn("parcel_id", merged.columns)
            self.assertNotIn("parcel_id_x", merged.columns)
            self.assertNotIn("parcel_id_y", merged.columns)
            self.assertEqual(merged.iloc[0]["parcel_id"], "pid-original")
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_contract_left_merge_rejects_duplicate_right_join_keys(self) -> None:
        left = pd.DataFrame([{"parcel_row_id": "row_1", "parcel_id": "pid-1"}])
        right = pd.DataFrame(
            [
                {"parcel_row_id": "row_1", "owner_name": "first"},
                {"parcel_row_id": "row_1", "owner_name": "second"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate right-side join keys"):
            parcel_contract.contract_left_merge(left, right, on="parcel_row_id")

    def test_contract_left_merge_rejects_null_canonical_repair_from_right(self) -> None:
        left = pd.DataFrame([{"parcel_row_id": "row_1", "parcel_id": pd.NA}])
        right = pd.DataFrame([{"parcel_row_id": "row_1", "parcel_id": "pid-1"}])

        with self.assertRaisesRegex(ValueError, "cannot be repaired from the right side"):
            parcel_contract.contract_left_merge(left, right, on="parcel_row_id")

    def test_contract_left_merge_preserves_left_canonical_fields(self) -> None:
        left = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-left",
                    "state_code": "MS",
                    "county_name": "alpha",
                    "county_fips": "001",
                    "geometry": b"left-geometry",
                    "lead_score_total": 10.0,
                }
            ]
        )
        right = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-right",
                    "state_code": "LA",
                    "county_name": "beta",
                    "county_fips": "999",
                    "geometry": b"right-geometry",
                    "lead_score_total": 77.0,
                    "owner_name": "Owner A",
                }
            ]
        )

        merged = parcel_contract.contract_left_merge(left, right, on="parcel_row_id")

        self.assertEqual(merged.iloc[0]["parcel_id"], "pid-left")
        self.assertEqual(merged.iloc[0]["state_code"], "MS")
        self.assertEqual(merged.iloc[0]["county_name"], "alpha")
        self.assertEqual(merged.iloc[0]["county_fips"], "001")
        self.assertEqual(merged.iloc[0]["geometry"], b"left-geometry")
        self.assertEqual(merged.iloc[0]["lead_score_total"], 77.0)
        self.assertEqual(merged.iloc[0]["owner_name"], "Owner A")

    def test_contract_left_merge_rejects_suffix_collisions(self) -> None:
        left = pd.DataFrame([{"parcel_row_id": "row_1", "parcel_id_x": "collision"}])
        right = pd.DataFrame([{"parcel_row_id": "row_1", "owner_name": "Owner A"}])

        with self.assertRaisesRegex(ValueError, "suffix collisions detected"):
            parcel_contract.contract_left_merge(left, right, on="parcel_row_id")

    def test_contract_left_merge_rejects_unexpected_row_count_change(self) -> None:
        left = pd.DataFrame([{"parcel_row_id": "row_1", "parcel_id": "pid-1"}])
        right = pd.DataFrame([{"parcel_row_id": "row_1", "owner_name": "Owner A"}])
        original_merge = pd.DataFrame.merge

        def merge_with_extra_row(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            merged = original_merge(self, *args, **kwargs)
            return pd.concat([merged, merged.iloc[[0]].copy()], ignore_index=True)

        with mock.patch.object(pd.DataFrame, "merge", merge_with_extra_row):
            with self.assertRaisesRegex(ValueError, "row count changed unexpectedly"):
                parcel_contract.contract_left_merge(left, right, on="parcel_row_id")

    def test_build_detail_metrics_runtime_preserves_parcel_id(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "acreage_bucket": "1-4.99",
                    "building_present_confidence": 77.0,
                }
            ]
        )

        detail = runtime_builder.build_detail_metrics_runtime(frame)

        self.assertIn("parcel_id", detail.columns)
        self.assertIn("parcel_row_id", detail.columns)
        self.assertEqual(detail.iloc[0]["parcel_id"], "pid-1")

    def test_frontend_detail_fallback_payload_preserves_parcel_id_without_duplicate_field(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_test_parcel_id_fallback"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            app_ready_path = temp_path / "app_ready.parquet"
            detail_metrics_path = temp_path / "detail_metrics.parquet"
            output_path = temp_path / "fallback.json"
            runtime_root = temp_path / "parcel_index"

            pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_1",
                        "parcel_id": "pid-1",
                        "county_name": "alpha",
                        "parcel_vacant_flag": True,
                        "county_hosted_flag": False,
                        "best_source_type": "parcel_master",
                    }
                ]
            ).to_parquet(app_ready_path, index=False, engine="pyarrow")
            pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_1",
                        "parcel_id": "pid-1",
                        "building_present_confidence": 77.0,
                    }
                ]
            ).to_parquet(detail_metrics_path, index=False, engine="pyarrow")

            with (
                mock.patch.object(detail_fallback, "APP_READY_PATH", app_ready_path),
                mock.patch.object(detail_fallback, "DETAIL_METRICS_PATH", detail_metrics_path),
                mock.patch.object(detail_fallback, "PARCEL_INDEX_ROOT", runtime_root),
                mock.patch.object(detail_fallback, "OUTPUT_PATH", output_path),
            ):
                detail_fallback.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["parcel_id"], "pid-1")
            self.assertEqual(payload[0]["parcel_row_id"], "row_1")
            self.assertNotIn("parcel_id_detail", payload[0])
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_backend_detail_payload_preserves_parcel_id(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "county_name": "alpha",
                    "state_code": "MS",
                    "county_fips": "001",
                    "latitude": 32.0,
                    "longitude": -90.0,
                    "parcel_vacant_flag": True,
                }
            ]
        )

        with (
            mock.patch.object(leads_service, "_using_embedded_runtime", return_value=False),
            mock.patch.object(leads_service, "load_base_frame", return_value=frame),
            mock.patch.object(leads_service, "_lookup_tax_freshness_detail", return_value={}),
            mock.patch.object(leads_service, "_detail_geometry", return_value={"type": "Polygon", "coordinates": []}),
            mock.patch.object(leads_service, "_maybe_apply_on_demand_ai"),
            mock.patch.object(leads_service, "_apply_tax_detail_defaults"),
            mock.patch.object(leads_service, "_apply_tax_interpretation_payload"),
            mock.patch.object(leads_service, "_apply_vacancy_assessment"),
            mock.patch.object(leads_service, "_stabilize_detail_payload"),
        ):
            payload = leads_service.get_lead_detail("row_1")

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["parcel_id"], "pid-1")
        self.assertEqual(payload["parcel_row_id"], "row_1")

    def test_load_base_frame_uses_runtime_artifacts_in_production(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "state_code": "MS",
                    "county_name": "alpha",
                    "county_fips": "001",
                }
            ]
        )
        leads_service.load_base_frame.cache_clear()
        try:
            with (
                mock.patch.object(leads_service, "_canonical_runtime_available", return_value=True),
                mock.patch.object(leads_service, "_load_base_frame_from_runtime_artifacts", return_value=frame) as runtime_loader,
                mock.patch.object(leads_service.pd, "read_parquet", side_effect=AssertionError("raw parquet path should not run")),
            ):
                loaded = leads_service.load_base_frame()
        finally:
            leads_service.load_base_frame.cache_clear()

        runtime_loader.assert_called_once_with()
        self.assertEqual(loaded.iloc[0]["parcel_id"], "pid-1")
        self.assertEqual(loaded.iloc[0]["parcel_row_id"], "row_1")

    def test_runtime_artifact_loader_applies_shared_service_finalizer(self) -> None:
        source_frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "state_code": "MS",
                    "county_name": "alpha",
                    "county_fips": "001",
                    "county_hosted_flag": False,
                    "parcel_vacant_flag": True,
                    "corporate_owner_flag": False,
                    "absentee_owner_flag": False,
                    "out_of_state_owner_flag": False,
                    "high_confidence_link_flag": False,
                    "delinquent_flag": False,
                    "forfeited_flag": False,
                    "amount_trust_tier": "not_reported",
                    "source_confidence_tier": "parcel_master_only",
                    "county_source_coverage_tier": "statewide_parcel_base",
                    "best_source_type": "parcel_master",
                    "best_source_name": "Mississippi Parcel Runtime",
                    "growth_pressure_bucket": "unknown",
                    "recommended_view_bucket": "general_ranked",
                    "owner_type": "unknown",
                    "owner_name": "Owner A",
                    "electric_provider_name": "utility",
                }
            ]
        )
        finalized_frame = source_frame.assign(lead_score_total=10.0, lead_score_total_effective=10.0)

        class _FakeTable:
            def to_pandas(self_inner) -> pd.DataFrame:
                return source_frame.copy()

        class _FakeDataset:
            def to_table(self_inner):
                return _FakeTable()

        with (
            mock.patch.object(leads_service, "_embedded_parcel_dataset", return_value=_FakeDataset()),
            mock.patch.object(leads_service, "_ensure_intelligence_fields", side_effect=lambda frame: frame),
            mock.patch.object(leads_service, "_apply_tax_interpretation_fields", side_effect=lambda frame: frame),
            mock.patch.object(
                leads_service,
                "_finalize_load_base_frame_service_view",
                return_value=finalized_frame,
            ) as finalizer,
            mock.patch.object(leads_service, "validate_required_columns"),
        ):
            loaded = leads_service._load_base_frame_from_runtime_artifacts()

        finalizer.assert_called_once()
        self.assertEqual(loaded.iloc[0]["lead_score_total"], 10.0)
        self.assertEqual(loaded.iloc[0]["parcel_id"], "pid-1")

    def test_load_base_frame_rejects_noncanonical_source_rebuild_without_opt_in(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_test_load_base_frame_dispatch"
        missing_parquet = temp_path / "missing_leads.parquet"
        missing_embedded = temp_path / "missing_embedded.parquet"
        missing_static = temp_path / "missing_static.json"
        leads_service.load_base_frame.cache_clear()
        try:
            with (
                mock.patch.object(leads_service, "_canonical_runtime_available", return_value=False),
                mock.patch.object(leads_service, "_full_runtime_available", return_value=True),
                mock.patch.object(leads_service, "LEAD_SIGNALS_PATH", missing_parquet),
                mock.patch.object(leads_service, "EMBEDDED_LEAD_SIGNALS_PATH", missing_embedded),
                mock.patch.object(leads_service, "MISSISSIPPI_STATIC_FEED_PATH", missing_static),
                mock.patch.dict(leads_service.os.environ, {}, clear=False),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "MISSISSIPPI_ALLOW_NONCANONICAL_SOURCE_REBUILD"):
                    leads_service.load_base_frame()
        finally:
            leads_service.load_base_frame.cache_clear()

    def test_load_base_frame_noncanonical_source_rebuild_is_explicit_and_preserves_identifiers(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_test_load_base_frame_noncanonical"
        missing_parquet = temp_path / "missing_leads.parquet"
        missing_embedded = temp_path / "missing_embedded.parquet"
        missing_static = temp_path / "missing_static.json"

        parcels = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "county_name": "alpha",
                    "county_fips": "001",
                    "state_code": "MS",
                    "owner_name": "Owner A",
                    "land_use_raw": "vacant",
                    "tax_acres": 1.0,
                    "gis_acres": np.nan,
                    "total_acres": np.nan,
                    "parcel_area_acres": np.nan,
                    "latitude": 32.0,
                    "longitude": -90.0,
                    "road_distance_ft": 100.0,
                    "road_access_tier": "direct",
                    "wetland_flag": False,
                    "wetland_overlap_acres": 0.0,
                    "wetland_overlap_pct": 0.0,
                    "flood_risk_score": 0.0,
                    "flood_zone_primary": pd.NA,
                    "has_flood_overlap": False,
                    "sfha_overlap": False,
                    "mean_slope_pct": 1.0,
                    "max_slope_pct": 2.0,
                    "slope_class": "low",
                    "slope_score": 90.0,
                    "shape_length": 400.0,
                    "shape_area": 43560.0,
                    "buildability_score": 80.0,
                    "environment_score": 70.0,
                    "investment_score": 65.0,
                    "electric_provider_name": "utility",
                    "total_value": 1000.0,
                }
            ]
        )
        owners = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "owner_type": "individual",
                    "absentee_owner_flag": False,
                    "out_of_state_owner_flag": False,
                    "owner_parcel_count": 1,
                    "owner_total_acres": 1.0,
                    "mailer_target_score": 20.0,
                    "corporate_owner_flag": False,
                }
            ]
        )
        buildings = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "building_count": 0,
                    "building_area_total": 0.0,
                    "parcel_vacant_flag": True,
                    "nearby_building_count_1km": 0,
                    "nearby_building_density": 0.0,
                    "growth_pressure_bucket": "low",
                }
            ]
        )
        signals = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "delinquent_amount": np.nan,
                    "delinquent_amount_bucket": pd.NA,
                    "delinquent_flag": False,
                    "forfeited_flag": False,
                    "best_source_type": "parcel_master",
                    "best_source_name": "parcel master",
                    "source_confidence_tier": "parcel_master_only",
                    "county_source_coverage_tier": "statewide_parcel_base",
                    "amount_trust_tier": "not_reported",
                    "high_confidence_link_flag": False,
                    "county_hosted_flag": False,
                    "lead_score_total": 10.0,
                    "lead_score_tier": "low",
                    "lead_score_driver_1": pd.NA,
                    "lead_score_driver_2": pd.NA,
                    "lead_score_driver_3": pd.NA,
                    "lead_score_explanation": pd.NA,
                    "size_score": 10.0,
                    "access_score": 20.0,
                    "buildability_component": 30.0,
                    "environmental_component": 40.0,
                    "owner_targeting_component": 50.0,
                    "delinquency_component": 0.0,
                    "source_confidence_component": 30.0,
                    "vacant_land_component": 60.0,
                    "growth_pressure_component": 20.0,
                    "recommended_sort_reason": pd.NA,
                    "top_score_driver": pd.NA,
                    "caution_flags": pd.NA,
                    "vacant_reason": pd.NA,
                    "growth_pressure_reason": pd.NA,
                    "recommended_use_case": pd.NA,
                    "recommended_view_bucket": "general_ranked",
                }
            ]
        )

        def fake_read_parquet(path, columns=None, engine=None):  # type: ignore[no-untyped-def]
            path = Path(path)
            if path == leads_service.PARCEL_MASTER_PATH:
                return parcels.loc[:, columns].copy()
            if path == leads_service.OWNER_LEADS_PATH:
                return owners.loc[:, columns].copy()
            if path == leads_service.BUILDING_METRICS_PATH:
                return buildings.loc[:, columns].copy()
            if path == leads_service.LEAD_SIGNALS_PATH:
                return signals.loc[:, columns].copy()
            raise AssertionError(f"Unexpected parquet read: {path}")

        leads_service.load_base_frame.cache_clear()
        try:
            with (
                mock.patch.object(leads_service, "_canonical_runtime_available", return_value=False),
                mock.patch.object(leads_service, "_full_runtime_available", return_value=True),
                mock.patch.object(leads_service, "LEAD_SIGNALS_PATH", missing_parquet),
                mock.patch.object(leads_service, "EMBEDDED_LEAD_SIGNALS_PATH", missing_embedded),
                mock.patch.object(leads_service, "MISSISSIPPI_STATIC_FEED_PATH", missing_static),
                mock.patch.object(leads_service.pd, "read_parquet", side_effect=fake_read_parquet),
                mock.patch.object(leads_service, "_merge_tax_freshness_sources", side_effect=lambda frame: frame),
                mock.patch.object(leads_service, "_merge_ai_predictions", side_effect=lambda frame: frame),
                mock.patch.object(leads_service, "_ensure_intelligence_fields", side_effect=lambda frame: frame),
                mock.patch.object(
                    leads_service,
                    "_apply_tax_interpretation_fields",
                    side_effect=lambda frame: frame.assign(parcel_tax_score_adjustment=0.0),
                ),
                mock.patch.dict(
                    leads_service.os.environ,
                    {"MISSISSIPPI_ALLOW_NONCANONICAL_SOURCE_REBUILD": "true"},
                    clear=False,
                ),
            ):
                loaded = leads_service.load_base_frame()
        finally:
            leads_service.load_base_frame.cache_clear()

        self.assertEqual(loaded.iloc[0]["parcel_id"], "pid-1")
        self.assertEqual(loaded.iloc[0]["parcel_row_id"], "row_1")


if __name__ == "__main__":
    unittest.main()
