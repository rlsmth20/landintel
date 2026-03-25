from __future__ import annotations

import sys
import unittest
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import vacancy_ai_reviewed_pilot_ms as reviewed_pilot  # noqa: E402


class VacancyAiReviewedPilotTests(unittest.TestCase):
    def test_derive_output_paths_uses_state_registry_defaults(self) -> None:
        outputs = reviewed_pilot._derive_output_paths(None, state_code="ms", run_name="reviewed50")

        self.assertTrue(str(outputs["manifest"]).endswith("ai_building_presence_training_manifest_ms_reviewed50.parquet"))
        self.assertTrue(str(outputs["cv_predictions"]).endswith("ai_building_presence_reviewed50_cv_predictions.csv"))

    def test_prepare_review_label_frame_filters_unlabeled_and_ambiguous(self) -> None:
        review_frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_a",
                    "manual_training_label": "Vacant",
                    "manual_review_confidence": "High",
                },
                {
                    "parcel_row_id": "row_b",
                    "manual_training_label": "Improved",
                    "manual_review_confidence": "HIgh",
                },
                {
                    "parcel_row_id": "row_c",
                    "manual_training_label": "Unknown",
                    "manual_review_confidence": "Medium",
                },
                {
                    "parcel_row_id": "row_d",
                    "manual_training_label": None,
                    "manual_review_confidence": None,
                },
            ]
        )

        usable, summary = reviewed_pilot.prepare_review_label_frame(review_frame)

        self.assertEqual(len(usable), 2)
        self.assertEqual(
            usable.set_index("parcel_row_id")["structure_present_target"].astype(int).to_dict(),
            {"row_a": 0, "row_b": 1},
        )
        self.assertEqual(summary["rows_excluded_before_join"]["missing_manual_training_label"], 1)
        self.assertEqual(summary["rows_excluded_before_join"]["ambiguous_manual_training_label"], 1)
        self.assertEqual(summary["manual_review_confidence_counts_used"]["high"], 2)

    def test_build_pilot_training_manifest_frame_joins_review_rows_to_features(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_tmp_reviewed50_join_test_images"
        shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            core_a = temp_path / "core_a.png"
            focus_a = temp_path / "focus_a.png"
            core_b = temp_path / "core_b.png"
            focus_b = temp_path / "focus_b.png"
            Image.new("RGB", (48, 48), (20, 30, 40)).save(core_a)
            Image.new("RGB", (48, 48), (25, 35, 45)).save(focus_a)
            Image.new("RGB", (48, 48), (200, 210, 220)).save(core_b)
            Image.new("RGB", (48, 48), (205, 215, 225)).save(focus_b)

            review_frame = pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_a",
                        "parcel_id": "pid-a",
                        "county_name": "alpha",
                        "manual_training_label": "Vacant",
                        "manual_training_label_normalized": "vacant",
                        "structure_present_target": 0,
                        "manual_review_confidence": "High",
                        "manual_review_confidence_normalized": "high",
                        "manual_review_notes": "clear vacant parcel",
                        "review_tile_label": "tile-a",
                        "review_tile_coordinate": "19/1/1",
                        "masked_parcel_tile_path": str(core_a),
                        "masked_parcel_core_crop_path": str(core_a),
                        "masked_parcel_focus_crop_path": str(focus_a),
                    },
                    {
                        "parcel_row_id": "row_b",
                        "parcel_id": "pid-b",
                        "county_name": "beta",
                        "manual_training_label": "Improved",
                        "manual_training_label_normalized": "improved",
                        "structure_present_target": 1,
                        "manual_review_confidence": "Medium",
                        "manual_review_confidence_normalized": "medium",
                        "manual_review_notes": "structure present",
                        "review_tile_label": "tile-b",
                        "review_tile_coordinate": "19/1/2",
                        "masked_parcel_tile_path": str(core_b),
                        "masked_parcel_core_crop_path": str(core_b),
                        "masked_parcel_focus_crop_path": str(focus_b),
                    },
                ]
            )
            feature_manifest = pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_a",
                        "parcel_id": "pid-a",
                        "county_name": "alpha",
                        "state_code": "MS",
                        "county_fips": "001",
                        "image_path": "tile-a.jpg",
                        "weak_building_label": 0,
                        "tile_label": "tile-a",
                        "imagery_crop_label": "parcel_focus",
                        "tile_coordinate": "19/1/1",
                        "crop_parcel_coverage_ratio": 0.6,
                        "parcel_tile_coverage_ratio": 0.7,
                        "r_mean": 10.0,
                    },
                    {
                        "parcel_row_id": "row_b",
                        "parcel_id": "pid-b",
                        "county_name": "beta",
                        "state_code": "MS",
                        "county_fips": "003",
                        "image_path": "tile-b.jpg",
                        "weak_building_label": 1,
                        "tile_label": "tile-b",
                        "imagery_crop_label": "parcel_core",
                        "tile_coordinate": "19/1/2",
                        "crop_parcel_coverage_ratio": 0.8,
                        "parcel_tile_coverage_ratio": 0.9,
                        "r_mean": 90.0,
                    },
                ]
            )

            pilot_manifest, summary, feature_columns_used = reviewed_pilot.build_pilot_training_manifest_frame(
                review_frame=review_frame,
                feature_manifest_frame=feature_manifest,
            )

            self.assertEqual(len(pilot_manifest), 4)
            self.assertEqual(
                pilot_manifest.drop_duplicates(subset=["parcel_row_id"]).set_index("parcel_row_id")["structure_present_target"].astype(int).to_dict(),
                {"row_a": 0, "row_b": 1},
            )
            self.assertIn("masked_parcel_tile_path", pilot_manifest.columns)
            self.assertIn("r_mean", feature_columns_used)
            self.assertEqual(summary["pilot_manifest_parcel_count"], 2)
            self.assertEqual(summary["feature_source_workflow"], "review_export_crops_only")
            self.assertEqual(summary["feature_source_parcel_counts"]["review_export_crops"], 2)
            self.assertEqual(summary["feature_source_parcel_counts"]["sampled_feature_manifest_intersection"], 2)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_evaluate_pilot_manifest_runs_grouped_cross_validation(self) -> None:
        pilot_manifest = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_a",
                    "parcel_id": "pid-a",
                    "county_name": "alpha",
                    "image_path": "tile-a-1.jpg",
                    "tile_label": "tile-a",
                    "imagery_crop_label": "parcel_core",
                    "crop_parcel_coverage_ratio": 0.8,
                    "parcel_tile_coverage_ratio": 0.8,
                    "structure_present_target": 0,
                    "manual_training_label": "Vacant",
                    "manual_review_confidence": "High",
                    "manual_review_confidence_normalized": "high",
                    "manual_review_notes": None,
                    "r_mean": 5.0,
                },
                {
                    "parcel_row_id": "row_b",
                    "parcel_id": "pid-b",
                    "county_name": "beta",
                    "image_path": "tile-b-1.jpg",
                    "tile_label": "tile-b",
                    "imagery_crop_label": "parcel_core",
                    "crop_parcel_coverage_ratio": 0.8,
                    "parcel_tile_coverage_ratio": 0.8,
                    "structure_present_target": 0,
                    "manual_training_label": "Vacant",
                    "manual_review_confidence": "High",
                    "manual_review_confidence_normalized": "high",
                    "manual_review_notes": None,
                    "r_mean": 8.0,
                },
                {
                    "parcel_row_id": "row_c",
                    "parcel_id": "pid-c",
                    "county_name": "gamma",
                    "image_path": "tile-c-1.jpg",
                    "tile_label": "tile-c",
                    "imagery_crop_label": "parcel_core",
                    "crop_parcel_coverage_ratio": 0.8,
                    "parcel_tile_coverage_ratio": 0.8,
                    "structure_present_target": 1,
                    "manual_training_label": "Improved",
                    "manual_review_confidence": "High",
                    "manual_review_confidence_normalized": "high",
                    "manual_review_notes": None,
                    "r_mean": 85.0,
                },
                {
                    "parcel_row_id": "row_d",
                    "parcel_id": "pid-d",
                    "county_name": "delta",
                    "image_path": "tile-d-1.jpg",
                    "tile_label": "tile-d",
                    "imagery_crop_label": "parcel_core",
                    "crop_parcel_coverage_ratio": 0.8,
                    "parcel_tile_coverage_ratio": 0.8,
                    "structure_present_target": 1,
                    "manual_training_label": "Improved",
                    "manual_review_confidence": "Medium",
                    "manual_review_confidence_normalized": "medium",
                    "manual_review_notes": None,
                    "r_mean": 95.0,
                },
            ]
        )

        evaluation, _scored_rows, parcel_eval = reviewed_pilot.evaluate_pilot_manifest(
            pilot_manifest=pilot_manifest,
            feature_columns_used=["r_mean"],
            random_state=42,
            cv_splits=2,
            output_model_path=None,
        )

        self.assertEqual(evaluation["cv_splits"], 2)
        self.assertEqual(evaluation["parcel_count_evaluated"], 4)
        self.assertIn("accuracy", evaluation["parcel_level_threshold_050"])
        self.assertIn("false_positive", evaluation["parcel_level_threshold_050"]["confusion_matrix"])
        self.assertEqual(len(parcel_eval), 4)

    def test_build_pilot_training_manifest_frame_generates_missing_parcel_rows_from_review_crops(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_tmp_reviewed50_test_images"
        shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            core_path = temp_path / "core.png"
            focus_path = temp_path / "focus.png"
            Image.new("RGB", (48, 48), (120, 140, 160)).save(core_path)
            Image.new("RGB", (48, 48), (100, 110, 120)).save(focus_path)

            review_frame = pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_missing",
                        "parcel_id": "pid-missing",
                        "county_name": "omega",
                        "manual_training_label": "Vacant",
                        "manual_training_label_normalized": "vacant",
                        "structure_present_target": 0,
                        "manual_review_confidence": "High",
                        "manual_review_confidence_normalized": "high",
                        "manual_review_notes": "generated from crops",
                        "review_tile_label": "tile-review",
                        "review_tile_coordinate": "19/10/20",
                        "review_tile_rank": 1,
                        "masked_parcel_tile_path": str(core_path),
                        "masked_parcel_core_crop_path": str(core_path),
                        "masked_parcel_focus_crop_path": str(focus_path),
                        "parcel_tile_coverage_pct": 44.0,
                        "parcel_bbox_tile_coverage_pct": 55.0,
                        "geometry_quality_flag": "good",
                        "geometry_training_excluded_flag": False,
                        "area_acres": 1.0,
                        "perimeter_meters": 100.0,
                        "bounding_box_width_meters": 20.0,
                        "bounding_box_height_meters": 30.0,
                        "aspect_ratio": 1.5,
                        "compactness": 0.6,
                        "is_multipart": False,
                        "part_count": 1,
                        "parcel_boundary_crop_ready_flag": True,
                    }
                ]
            )

            pilot_manifest, summary, _ = reviewed_pilot.build_pilot_training_manifest_frame(
                review_frame=review_frame,
                feature_manifest_frame=pd.DataFrame(columns=["parcel_row_id", "image_path"]),
            )

            self.assertEqual(len(pilot_manifest), 2)
            self.assertEqual(summary["feature_source_parcel_counts"]["review_export_crops"], 1)
            self.assertSetEqual(set(pilot_manifest["imagery_crop_label"]), {"parcel_core", "parcel_focus"})
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_build_error_analysis_frame_groups_neighbor_confusion(self) -> None:
        parcel_eval = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_neighbor",
                    "parcel_id": "pid-neighbor",
                    "county_name": "alpha",
                    "structure_present_target": 0,
                    "manual_training_label": "Vacant",
                    "manual_review_confidence": "High",
                    "manual_review_confidence_normalized": "high",
                    "manual_review_notes": "Structure present just outside parcel boundary",
                    "review_hint": "neighbor_outside_parcel_false_positive",
                    "predicted_probability": 0.91,
                    "predicted_confidence": 91.0,
                    "predicted_class_050": True,
                    "predicted_class_060": True,
                    "predicted_class_070": True,
                    "predicted_class_082": True,
                    "error_type_050": "false_positive",
                    "error_type_082": "false_positive",
                    "masked_parcel_tile_path": "masked.png",
                    "masked_parcel_core_crop_path": "core.png",
                    "masked_parcel_focus_crop_path": "focus.png",
                    "best_tile_label": "tile-a",
                    "best_tile_crop_label": "parcel_core",
                    "best_tile_parcel_coverage_pct": 80.0,
                    "full_parcel_visible_flag": True,
                    "parcel_tile_low_coverage_flag": False,
                    "multi_tile_candidate_flag": False,
                    "polygon_part_count": 1,
                    "clipped_polygon_part_count": 1,
                    "geometry_quality_flag": "good",
                    "geometry_marketability_flag": "marketable",
                    "geometry_marketability_action": "keep",
                    "area_acres": 1.0,
                    "aspect_ratio": 1.5,
                    "compactness": 0.6,
                    "nearby_building_density": 250.0,
                    "parcel_width_ft_estimate": 100.0,
                    "parcel_aspect_ratio_estimate": 1.5,
                    "imagery_driveway_signal_max": 12.0,
                    "imagery_clearing_signal_max": 4.0,
                    "green_excess_max": 0.05,
                    "dark_shadow_pct_max": 0.04,
                }
            ]
        )

        error_frame, summary = reviewed_pilot.build_error_analysis_frame(parcel_eval)

        self.assertEqual(len(error_frame), 1)
        self.assertEqual(error_frame.iloc[0]["likely_error_cause"], "neighbor structure confusion")
        self.assertEqual(summary["likely_error_cause_counts"]["neighbor structure confusion"], 1)

    def test_default_feature_manifest_path_accepts_state_code(self) -> None:
        path = reviewed_pilot._default_feature_manifest_path("ms", run_name="reviewed50")

        self.assertTrue(str(path).endswith("ai_building_presence_training_manifest_ms_geometry_quality.parquet"))


if __name__ == "__main__":
    unittest.main()
