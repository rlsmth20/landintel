from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))
sys.path.insert(0, str(ROOT / "backend"))

from app.services import mississippi_leads_service  # noqa: E402
from app.services.runtime_state_service import RuntimeStateService  # noqa: E402
from floodscraper.parcel_contract_ms import API_LEADS_SUMMARY_FIELDS, BACKEND_DETAIL_REQUIRED_FIELDS  # noqa: E402

try:  # noqa: E402
    from fastapi import HTTPException
    from app.api import state_leads
except ModuleNotFoundError:  # pragma: no cover - depends on backend env
    HTTPException = None
    state_leads = None


class RuntimeStateServiceFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = ROOT / "data" / "buildings_processed" / f"runtime-state-fallback-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _write_json(self, name: str, payload: object) -> Path:
        path = self.temp_root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _summary_item(self, parcel_row_id: str) -> dict[str, object]:
        item = {field: None for field in API_LEADS_SUMMARY_FIELDS}
        item.update(
            {
                "parcel_row_id": parcel_row_id,
                "parcel_id": f"{parcel_row_id}-parcel",
                "county_name": "pulaski",
                "acreage": 3.5,
                "lead_score_total": 91.2,
                "lead_score_total_effective": 91.2,
                "lead_score_tier": "high",
                "recommended_view_bucket": "general_ranked",
                "road_access_tier": "direct",
            }
        )
        return item

    def _detail_item(self, parcel_row_id: str) -> dict[str, object]:
        item = {field: None for field in BACKEND_DETAIL_REQUIRED_FIELDS}
        item.update(
            {
                "parcel_row_id": parcel_row_id,
                "parcel_id": f"{parcel_row_id}-parcel",
                "county_name": "pulaski",
                "latitude": 34.7465,
                "longitude": -92.2896,
                "acreage": 4.2,
                "lead_score_total": 83.4,
            }
        )
        return item

    def test_summary_and_presets_fall_back_to_frontend_meta(self) -> None:
        meta_path = self._write_json(
            "ar_meta.json",
            {
                "rowCount": 50000,
                "source": "frontend meta fallback",
                "geometryMode": "selected_parcel_geojson",
                "summary": [
                    {"section": "statewide", "metric": "lead_count", "value": "50000"},
                    {"section": "recommended_view_bucket", "key": "general_ranked", "metric": "parcel_count", "value": "1200"},
                ],
                "defaultViews": [
                    {
                        "view_name": "general_ranked",
                        "description": "Top ranked leads.",
                        "filter_expression": "recommended_view_bucket = 'general_ranked'",
                        "metric": "row_count",
                        "value": "1200",
                    },
                    {
                        "view_name": "general_ranked",
                        "description": "Top ranked leads.",
                        "filter_expression": "recommended_view_bucket = 'general_ranked'",
                        "metric": "average_lead_score",
                        "value": "88.5",
                    },
                ],
            },
        )
        missing_summary = self.temp_root / "missing_summary.json"
        missing_presets = self.temp_root / "missing_presets.json"
        service = RuntimeStateService("ar")
        service.artifacts = replace(
            service.artifacts,
            runtime_summary_path=missing_summary,
            runtime_presets_path=missing_presets,
            frontend_meta_path=meta_path,
        )

        summary = service.get_summary()
        presets = service.get_presets()

        self.assertEqual(summary["row_count"], 50000)
        self.assertEqual(summary["source"], "frontend meta fallback")
        self.assertEqual(summary["sections"]["statewide"][0]["metric"], "lead_count")
        self.assertEqual(summary["sections"]["statewide"][0]["value"], "50000")
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0]["view_name"], "general_ranked")
        self.assertEqual(presets[0]["row_count"], "1200")
        self.assertEqual(presets[0]["average_lead_score"], "88.5")

    def test_default_leads_fall_back_to_frontend_static_feed(self) -> None:
        static_feed_path = self._write_json(
            "ar_leads.json",
            {
                "total_count": 50000,
                "limit": 200,
                "offset": 0,
                "items": [self._summary_item("ar_static_1")],
            },
        )
        missing_path = self.temp_root / "missing.json"
        service = RuntimeStateService("ar")
        service.artifacts = replace(
            service.artifacts,
            runtime_default_leads_path=missing_path,
            app_ready_path=self.temp_root / "missing_app_ready.parquet",
            frontend_static_feed_path=static_feed_path,
            frontend_detail_fallback_path=self.temp_root / "missing_detail.json",
        )

        payload = service.get_leads(limit=5, offset=0)

        self.assertEqual(payload["total_count"], 50000)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["parcel_row_id"], "ar_static_1")

    def test_lead_detail_uses_frontend_detail_fallback_when_runtime_datasets_are_missing(self) -> None:
        detail_path = self._write_json("ar_detail.json", [self._detail_item("ar_detail_1")])
        service = RuntimeStateService("ar")
        service.artifacts = replace(
            service.artifacts,
            runtime_detail_metrics_path=self.temp_root / "missing_detail_metrics.parquet",
            app_ready_path=self.temp_root / "missing_app_ready.parquet",
            parcel_master_path=self.temp_root / "missing_parcel_master.parquet",
            frontend_detail_fallback_path=detail_path,
            frontend_static_feed_path=self.temp_root / "missing_leads.json",
        )

        payload = service.get_lead_detail("ar_detail_1")

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["parcel_row_id"], "ar_detail_1")
        self.assertEqual(payload["parcel_id"], "ar_detail_1-parcel")
        self.assertEqual(payload["geometry"]["type"], "point_reference")


class MississippiLeadFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = ROOT / "data" / "buildings_processed" / f"mississippi-leads-fallback-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)
        mississippi_leads_service._embedded_default_leads_payload.cache_clear()

    def test_default_request_uses_embedded_default_leads_payload(self) -> None:
        item = {field: None for field in mississippi_leads_service.SUMMARY_FIELDS}
        item.update(
            {
                "parcel_row_id": "ms_default_1",
                "parcel_id": "ms_default_1-parcel",
                "county_name": "adams",
            }
        )
        payload_path = self.temp_root / "ms_default_leads.json"
        payload_path.write_text(
            json.dumps({"total_count": 1, "limit": 200, "offset": 0, "items": [item]}),
            encoding="utf-8",
        )
        mississippi_leads_service._embedded_default_leads_payload.cache_clear()

        with mock.patch.object(mississippi_leads_service, "EMBEDDED_DEFAULT_LEADS_PATH", payload_path), mock.patch.object(
            mississippi_leads_service,
            "_using_embedded_runtime",
            side_effect=AssertionError("default request should not evaluate embedded runtime"),
        ):
            payload = mississippi_leads_service.get_leads()

        self.assertEqual(payload["total_count"], 1)
        self.assertEqual(payload["items"][0]["parcel_row_id"], "ms_default_1")


class StateLeadRouteTests(unittest.TestCase):
    def test_leads_route_returns_503_for_missing_artifacts(self) -> None:
        if HTTPException is None or state_leads is None:
            raise unittest.SkipTest("fastapi backend dependencies are not installed in this Python environment")
        mock_service = mock.Mock()
        mock_service.get_leads.side_effect = FileNotFoundError("missing artifacts")

        with mock.patch("app.api.state_leads.get_state_service", return_value=mock_service):
            with self.assertRaises(HTTPException) as error:
                state_leads.leads(state_code="ar")

        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(error.exception.detail, "missing artifacts")


if __name__ == "__main__":
    unittest.main()
