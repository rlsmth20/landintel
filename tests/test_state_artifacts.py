from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))
sys.path.insert(0, str(ROOT / "backend"))

import state_artifacts  # noqa: E402
from app.services import state_service_registry  # noqa: E402


class StateArtifactsTests(unittest.TestCase):
    def test_load_state_artifacts_ar_uses_state_aware_defaults(self) -> None:
        artifacts = state_artifacts.load_state_artifacts("ar")

        self.assertTrue(str(artifacts.parcel_master_path).endswith("data\\parcels\\ar\\ar_parcels_master.parquet"))
        self.assertTrue(str(artifacts.runtime_root).endswith("data\\runtime\\ar"))
        self.assertTrue(str(artifacts.frontend_detail_fallback_path).endswith("frontend\\public\\data\\ar_lead_detail_fallback.json"))
        self.assertTrue(str(artifacts.app_ready_path).endswith("data\\tax_published\\ar\\app_ready_ar_leads.parquet"))
        self.assertTrue(str(artifacts.frontend_parcel_pmtiles_path).endswith("frontend\\public\\tiles\\ar_parcels.pmtiles"))

    def test_load_state_artifacts_ms_preserves_legacy_paths(self) -> None:
        artifacts = state_artifacts.load_state_artifacts("ms")

        self.assertTrue(str(artifacts.parcel_master_path).endswith("mississippi_parcels_master.parquet"))
        self.assertTrue(str(artifacts.runtime_root).endswith("backend\\runtime\\mississippi"))
        self.assertTrue(str(artifacts.frontend_detail_fallback_path).endswith("mississippi_lead_detail_fallback.json"))
        self.assertTrue(str(artifacts.ai_predictions_path).endswith("ai_building_presence_predictions_ms.parquet"))

    def test_state_service_registry_resolves_ms(self) -> None:
        module = state_service_registry.get_state_service_module("ms")

        self.assertTrue(hasattr(module, "get_leads"))
        self.assertTrue(hasattr(module, "get_parcel_geometry"))
        self.assertIn("ms", state_service_registry.supported_state_codes())

    def test_state_service_registry_resolves_ar(self) -> None:
        module = state_service_registry.get_state_service_module("ar")

        self.assertTrue(hasattr(module, "get_leads"))
        self.assertTrue(hasattr(module, "get_parcel_geometry"))
        self.assertIn("ar", state_service_registry.supported_state_codes())

    def test_state_service_registry_resolves_runtime_backed_wi(self) -> None:
        service = state_service_registry.get_state_service("wi")

        self.assertTrue(hasattr(service, "get_leads"))
        self.assertTrue(hasattr(service, "get_parcel_geometry"))
        self.assertEqual(service.state_code, "wi")
        self.assertIn("wi", state_service_registry.supported_state_codes())

    def test_state_service_registry_resolves_runtime_backed_ct(self) -> None:
        service = state_service_registry.get_state_service("ct")

        self.assertTrue(hasattr(service, "get_leads"))
        self.assertTrue(hasattr(service, "get_parcel_geometry"))
        self.assertEqual(service.state_code, "ct")
        self.assertIn("ct", state_service_registry.supported_state_codes())

    def test_blocked_registry_entries_are_not_supported_runtime_states(self) -> None:
        self.assertNotIn("ut", state_service_registry.supported_state_codes())


if __name__ == "__main__":
    unittest.main()
