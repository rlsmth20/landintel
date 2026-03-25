from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import mapbox_vector_tile
import mercantile
import pandas as pd
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))
sys.path.insert(0, str(ROOT / "backend"))

import build_frontend_detail_fallback_ms as detail_fallback  # noqa: E402
import parcel_contract_ms as parcel_contract  # noqa: E402
from app.services import mississippi_leads_service as leads_service  # noqa: E402


class OutputContractTests(unittest.TestCase):
    def test_get_leads_items_match_summary_contract(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    **{column: None for column in parcel_contract.API_LEADS_SUMMARY_FIELDS},
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "county_name": "alpha",
                    "lead_score_total": 77.0,
                    "lead_score_total_effective": 77.0,
                }
            ]
        )

        with (
            mock.patch.object(leads_service, "_using_embedded_runtime", return_value=False),
            mock.patch.object(leads_service, "load_base_frame", return_value=frame),
            mock.patch.object(leads_service.pd, "read_parquet", side_effect=AssertionError("raw parquet should not be read here")),
        ):
            payload = leads_service.get_leads(limit=1)

        item = payload["items"][0]
        self.assertEqual(list(item.keys()), parcel_contract.API_LEADS_SUMMARY_FIELDS)
        self.assertEqual(item["parcel_row_id"], "row_1")
        self.assertEqual(item["parcel_id"], "pid-1")

    def test_search_leads_items_match_search_contract(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "county_name": "alpha",
                    "acreage": 2.0,
                    "owner_name": "Owner A",
                    "longitude": -90.0,
                    "latitude": 32.0,
                    "lead_score_total": 70.0,
                    "lead_score_total_effective": 70.0,
                }
            ]
        )

        with (
            mock.patch.object(leads_service, "_using_embedded_runtime", return_value=False),
            mock.patch.object(leads_service, "load_base_frame", return_value=frame),
        ):
            payload = leads_service.search_leads("pid", limit=5)

        item = payload["items"][0]
        self.assertEqual(list(item.keys()), parcel_contract.SEARCH_OUTPUT_FIELDS)
        self.assertEqual(item["parcel_row_id"], "row_1")
        self.assertEqual(item["parcel_id"], "pid-1")

    def test_get_nearby_comps_items_match_contract(self) -> None:
        subject = pd.Series(
            {
                "parcel_row_id": "row_subject",
                "parcel_id": "pid-subject",
                "county_name": "alpha",
                "acreage": 2.0,
                "land_use": "vacant",
                "assessed_total_value": 10000.0,
                "lead_score_total": 70.0,
                "lead_score_total_effective": 70.0,
                "investment_score": 55.0,
                "parcel_vacant_flag": True,
                "county_vacant_flag": True,
                "building_count": 0,
                "building_area_total": 0.0,
                "ai_building_present_probability": 0.1,
                "ai_building_present_flag": False,
                "building_present_confidence": 10.0,
                "building_presence_reason": "none",
                "longitude": -90.0,
                "latitude": 32.0,
            }
        )
        candidates = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_comp",
                    "parcel_id": "pid-comp",
                    "county_name": "alpha",
                    "acreage": 2.2,
                    "land_use": "vacant",
                    "assessed_total_value": 9000.0,
                    "lead_score_total": 68.0,
                    "lead_score_total_effective": 68.0,
                    "investment_score": 50.0,
                    "parcel_vacant_flag": True,
                    "county_vacant_flag": True,
                    "building_count": 0,
                    "building_area_total": 0.0,
                    "ai_building_present_probability": 0.12,
                    "ai_building_present_flag": False,
                    "building_present_confidence": 12.0,
                    "building_presence_reason": "none",
                    "longitude": -90.002,
                    "latitude": 32.001,
                }
            ]
        )

        def classify_subject(payload):  # type: ignore[no-untyped-def]
            payload.update(
                {
                    "parcel_improvement_status": "likely_vacant",
                    "parcel_improvement_confidence": 91.0,
                    "parcel_improvement_reason": "Synthetic subject classification.",
                    "parcel_improvement_evidence_summary": "Synthetic subject evidence.",
                }
            )

        with (
            mock.patch.object(leads_service, "_load_nearby_comp_subject", return_value=subject),
            mock.patch.object(leads_service, "_load_nearby_comp_candidates", return_value=candidates),
            mock.patch.object(leads_service, "_maybe_apply_on_demand_ai"),
            mock.patch.object(leads_service, "_apply_ai_detail_defaults"),
            mock.patch.object(leads_service, "_apply_parcel_improvement_classification", side_effect=classify_subject),
            mock.patch.object(
                leads_service,
                "_nearby_comp_classification",
                return_value={
                    "parcel_improvement_status": "likely_vacant",
                    "parcel_improvement_confidence": 88.0,
                    "parcel_improvement_reason": "Synthetic comp classification.",
                    "parcel_improvement_evidence_summary": "Synthetic comp evidence.",
                },
            ),
        ):
            payload = leads_service.get_nearby_comps("row_subject", limit=1)

        assert payload is not None
        self.assertEqual(list(payload["subject"].keys()), parcel_contract.NEARBY_COMP_OUTPUT_FIELDS)
        self.assertEqual(list(payload["items"][0].keys()), parcel_contract.NEARBY_COMP_OUTPUT_FIELDS)
        self.assertEqual(payload["subject"]["parcel_id"], "pid-subject")
        self.assertEqual(payload["items"][0]["parcel_id"], "pid-comp")

    def test_get_geometry_features_and_items_match_contract(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "county_name": "alpha",
                    "longitude": -90.0,
                    "latitude": 32.0,
                    "lead_score_total": 70.0,
                    "lead_score_tier": "high",
                    "parcel_vacant_flag": True,
                    "wetland_flag": False,
                    "flood_risk_score": 5.0,
                    "road_access_tier": "direct",
                    "county_hosted_flag": True,
                    "best_source_type": "parcel_master",
                }
            ]
        )

        with (
            mock.patch.object(leads_service, "_using_embedded_runtime", return_value=False),
            mock.patch.object(leads_service, "load_base_frame", return_value=frame),
        ):
            payload = leads_service.get_geometry(zoom=8, limit=1)

        feature = payload["feature_collection"]["features"][0]
        self.assertEqual(list(feature["properties"].keys()), parcel_contract.GEOMETRY_FEATURE_PROPERTY_FIELDS)
        self.assertEqual(list(payload["items"][0].keys()), parcel_contract.GEOMETRY_ITEM_FIELDS)
        self.assertEqual(feature["properties"]["parcel_id"], "pid-1")

    def test_get_parcel_tile_features_match_contract(self) -> None:
        tile = mercantile.tile(-90.0, 32.0, 14)
        west, south, east, north = mercantile.bounds(tile)
        geometry = Polygon(
            [
                (west + ((east - west) * 0.35), south + ((north - south) * 0.35)),
                (west + ((east - west) * 0.65), south + ((north - south) * 0.35)),
                (west + ((east - west) * 0.65), south + ((north - south) * 0.65)),
                (west + ((east - west) * 0.35), south + ((north - south) * 0.65)),
            ]
        )
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_1",
                    "parcel_id": "pid-1",
                    "county_name": "alpha",
                    "wetland_flag": False,
                    "flood_risk_score": 5.0,
                    "road_access_tier": "direct",
                    "county_hosted_flag": True,
                    "best_source_type": "parcel_master",
                    "latitude": 32.0,
                    "longitude": -90.0,
                }
            ]
        )
        geometry_frame = pd.DataFrame([{"parcel_row_id": "row_1", "geometry": geometry.wkb}])

        with (
            mock.patch.object(leads_service, "_using_embedded_runtime", return_value=False),
            mock.patch.object(leads_service, "load_base_frame", return_value=frame),
            mock.patch.object(leads_service, "_geometry_table_for_ids", return_value=geometry_frame),
        ):
            payload = leads_service.get_parcel_tile(tile.z, tile.x, tile.y)

        decoded = mapbox_vector_tile.decode(payload)
        feature = decoded[leads_service.PARCEL_TILE_LAYER]["features"][0]
        self.assertEqual(list(feature["properties"].keys()), parcel_contract.PARCEL_TILE_FEATURE_PROPERTY_FIELDS)
        self.assertEqual(feature["properties"]["parcel_row_id"], "row_1")
        self.assertEqual(feature["properties"]["parcel_id"], "pid-1")

    def test_get_leads_excludes_access_strip_by_default(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    **{column: None for column in parcel_contract.API_LEADS_SUMMARY_FIELDS},
                    "parcel_row_id": "row_good",
                    "parcel_id": "pid-good",
                    "county_name": "alpha",
                    "lead_score_total": 80.0,
                    "lead_score_total_effective": 80.0,
                    "geometry_quality_flag": "good",
                    "geometry_default_leads_excluded_flag": False,
                },
                {
                    **{column: None for column in parcel_contract.API_LEADS_SUMMARY_FIELDS},
                    "parcel_row_id": "row_strip",
                    "parcel_id": "pid-strip",
                    "county_name": "alpha",
                    "lead_score_total": 99.0,
                    "lead_score_total_effective": 99.0,
                    "geometry_quality_flag": "access_strip",
                    "geometry_default_leads_excluded_flag": True,
                },
            ]
        )

        with (
            mock.patch.object(leads_service, "_using_embedded_runtime", return_value=False),
            mock.patch.object(leads_service, "load_base_frame", return_value=frame),
        ):
            payload = leads_service.get_leads(limit=10)

        self.assertEqual(payload["total_count"], 1)
        self.assertEqual([item["parcel_row_id"] for item in payload["items"]], ["row_good"])

    def test_get_leads_excludes_unbuildable_marketability_by_default(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    **{column: None for column in parcel_contract.API_LEADS_SUMMARY_FIELDS},
                    "parcel_row_id": "row_good",
                    "parcel_id": "pid-good",
                    "county_name": "alpha",
                    "lead_score_total": 78.0,
                    "lead_score_total_effective": 78.0,
                    "geometry_quality_flag": "good",
                    "geometry_default_leads_excluded_flag": False,
                    "geometry_marketability_flag": "marketable",
                    "geometry_marketability_action": "keep",
                    "geometry_marketability_default_leads_excluded_flag": False,
                },
                {
                    **{column: None for column in parcel_contract.API_LEADS_SUMMARY_FIELDS},
                    "parcel_row_id": "row_wedge",
                    "parcel_id": "pid-wedge",
                    "county_name": "alpha",
                    "lead_score_total": 95.0,
                    "lead_score_total_effective": 95.0,
                    "geometry_quality_flag": "good",
                    "geometry_default_leads_excluded_flag": False,
                    "geometry_marketability_flag": "poor_geometry",
                    "geometry_marketability_action": "exclude",
                    "geometry_marketability_default_leads_excluded_flag": True,
                },
            ]
        )

        with (
            mock.patch.object(leads_service, "_using_embedded_runtime", return_value=False),
            mock.patch.object(leads_service, "load_base_frame", return_value=frame),
        ):
            payload = leads_service.get_leads(limit=10)

        self.assertEqual(payload["total_count"], 1)
        self.assertEqual([item["parcel_row_id"] for item in payload["items"]], ["row_good"])

    def test_frontend_fallback_record_preserves_required_fields(self) -> None:
        row = pd.Series(
            {
                "parcel_row_id": "row_1",
                "parcel_id": "pid-1",
                "county_name": "alpha",
                "geometry_marketability_flag": "poor_geometry",
                "geometry_marketability_action": "exclude",
                "geometry_penalty_points": -28.0,
                "geometry_penalty_reason": "Frontage greatly exceeds usable width.",
                "geometry": b"drop",
            }
        )

        payload = detail_fallback.build_frontend_fallback_record(row)

        self.assertEqual(payload["parcel_row_id"], "row_1")
        self.assertEqual(payload["parcel_id"], "pid-1")
        self.assertEqual(payload["county_name"], "alpha")
        self.assertEqual(payload["geometry_marketability_flag"], "poor_geometry")
        self.assertEqual(payload["geometry_marketability_action"], "exclude")
        self.assertEqual(payload["geometry_penalty_points"], -28.0)
        self.assertNotIn("geometry", payload)


if __name__ == "__main__":
    unittest.main()
