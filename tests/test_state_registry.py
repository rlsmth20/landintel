from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import bootstrap_state as bootstrap  # noqa: E402
import state_registry  # noqa: E402
import state_diagnostics  # noqa: E402


class StateRegistryTests(unittest.TestCase):
    def test_load_state_definition_ms(self) -> None:
        definition = state_registry.load_state_definition("ms")

        self.assertEqual(definition.state_code, "ms")
        self.assertEqual(definition.state_name, "Mississippi")
        self.assertIn("training", definition.artifact_roots)
        self.assertTrue(str(definition.legacy_path("parcel_master")).endswith("mississippi_parcels_master.parquet"))

    def test_load_state_definition_ar(self) -> None:
        definition = state_registry.load_state_definition("ar")

        self.assertEqual(definition.state_code, "ar")
        self.assertEqual(definition.state_name, "Arkansas")
        self.assertTrue(str(definition.legacy_path("parcel_master")).endswith("ar_parcels_master.parquet"))
        self.assertTrue(str(definition.source_registry_path("parcel_source")).endswith("parcel_source_ar.json"))

    def test_reviewed_pilot_default_outputs_ms_uses_legacy_paths(self) -> None:
        outputs = state_registry.reviewed_pilot_default_outputs("ms", run_name="reviewed50")

        self.assertTrue(str(outputs["review_input"]).endswith("_50_Sample.csv"))
        self.assertTrue(str(outputs["manifest"]).endswith("ai_building_presence_training_manifest_ms_reviewed50.parquet"))
        self.assertTrue(str(outputs["error_summary"]).endswith("ai_building_presence_reviewed50_error_analysis_summary.json"))

    def test_bootstrap_state_creates_expected_scaffolding(self) -> None:
        original_root = bootstrap.ROOT
        original_config_dir = bootstrap.STATE_CONFIG_DIR
        original_registry_path = bootstrap.STATE_REGISTRY_PATH
        temp_root = ROOT / "data" / "buildings_processed" / "_tmp_state_registry_test"
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            bootstrap.ROOT = temp_root
            bootstrap.STATE_CONFIG_DIR = temp_root / "config" / "states"
            bootstrap.STATE_REGISTRY_PATH = bootstrap.STATE_CONFIG_DIR / "registry.json"

            result = bootstrap.bootstrap_state(
                state_code="al",
                state_name="Alabama",
                county_division_label="county",
                force=False,
            )

            config_path = Path(result["config_path"])
            registry_path = bootstrap.STATE_REGISTRY_PATH
            parcel_source_path = Path(result["parcel_source_registry_path"])
            schema_mapping_path = Path(result["schema_mapping_path"])

            self.assertTrue(config_path.exists())
            self.assertTrue(registry_path.exists())
            self.assertTrue(parcel_source_path.exists())
            self.assertTrue(schema_mapping_path.exists())

            registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertIn("al", registry_payload["states"])

            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config_payload["state_code"], "al")
            self.assertEqual(config_payload["artifact_roots"]["training"], "data/training/al")
        finally:
            bootstrap.ROOT = original_root
            bootstrap.STATE_CONFIG_DIR = original_config_dir
            bootstrap.STATE_REGISTRY_PATH = original_registry_path
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_build_state_diagnostics_ms_includes_expected_sections(self) -> None:
        diagnostics = state_diagnostics.build_state_diagnostics("ms")

        self.assertEqual(diagnostics["state_code"], "ms")
        self.assertIn("schema_mapping_summary", diagnostics)
        self.assertIn("app_ready_county_coverage", diagnostics)
        self.assertIn("geometry_quality_overview", diagnostics)
        self.assertIn("marketability_summary", diagnostics)

    def test_build_state_diagnostics_ar_includes_schema_and_paths(self) -> None:
        diagnostics = state_diagnostics.build_state_diagnostics("ar")

        self.assertEqual(diagnostics["state_code"], "ar")
        self.assertIn("schema_mapping_summary", diagnostics)
        self.assertIn("artifact_roots", diagnostics)


if __name__ == "__main__":
    unittest.main()
