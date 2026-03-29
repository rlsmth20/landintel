from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import build_backend_parcel_runtime_arcgis as runtime_builder  # noqa: E402


def _profile(**overrides):
    payload = {
        "state_code": "ct",
        "state_name": "Connecticut",
        "source_name": "Connecticut parcel layer",
        "service_url": "https://example.com/FeatureServer/0",
        "object_id_field": "OBJECTID",
        "geometry_out_fields": "OBJECTID,Link",
        "attribute_out_fields": ["OBJECTID", "Town_Name", "Link", "Assessed_Total", "Assessed_Land", "Assessed_Building", "Shape__Area", "Shape__Length"],
        "count_url": "https://example.com/FeatureServer/0/query",
        "query_url": "https://example.com/FeatureServer/0/query",
        "county_division_label": "town",
        "coordinate_mode": "selected_return_centroid",
        "batch_size_hint": 2000,
        "geometry_batch_size_hint": 2000,
        "coordinate_batch_size_hint": 250,
        "field_map": {
            "source_object_id": "OBJECTID",
            "parcel_id_fields": ["Link"],
            "county_name": "Town_Name",
            "county_fips": {"field": "Link", "split": "-", "index": 0},
            "owner_name": "Owner",
            "site_address": "Location",
            "assessed_total_value": "Assessed_Total",
            "assessed_land_value": "Assessed_Land",
            "assessed_improvement_value": "Assessed_Building",
            "area_square_meters": "Shape__Area",
            "perimeter_meters": "Shape__Length",
        },
        "county_name_domain": {},
        "county_fips_lookup": {},
        "source_confidence_tier": "high",
        "county_source_coverage_tier": "statewide_primary",
        "source_warning": "Test warning",
        "notes": [],
        "field_readiness": [],
        "preset_definitions": {},
        "app_ready_min_score": 35.0,
        "app_ready_min_acres": 0.25,
        "default_bounds": [-73.75, 40.95, -71.78, 42.05],
        "default_lead_limit": 200,
        "frontend_fallback_limit": 5000,
        "null_parcel_id_values": set(),
        "parcel_pmtiles_ready": True,
        "vacancy_proxy_mode": "assessed_improvement_ratio",
    }
    payload.update(overrides)
    return runtime_builder.RuntimeProfile(**payload)


class ArcgisRuntimeBuilderTests(unittest.TestCase):
    def test_mapped_value_supports_split_index_dict(self) -> None:
        attributes = {"link": "09190-CT-019-16-85-4"}

        value = runtime_builder._mapped_value(attributes, {"field": "Link", "split": "-", "index": 0})

        self.assertEqual(value, "09190")

    def test_resolve_vacancy_proxy_disabled_returns_needs_review(self) -> None:
        parcel_vacant_flag, status, confidence, reason, summary = runtime_builder._resolve_vacancy_proxy(
            assessed_improvement_value=None,
            assessed_total_value=None,
            mode="disabled",
            state_name="Utah",
        )

        self.assertIsNone(parcel_vacant_flag)
        self.assertEqual(status, "needs_review")
        self.assertIsNone(confidence)
        self.assertEqual(reason, "improvement_valuation_unavailable")
        self.assertIn("Utah statewide parcel source", summary)

    def test_transform_feature_batch_supports_derived_division_code(self) -> None:
        frame = runtime_builder._transform_feature_batch(
            [
                {
                    "attributes": {
                        "OBJECTID": 1,
                        "Town_Name": "BROOKLYN",
                        "Link": "09190-CT-019-16-85-4",
                        "Owner": "HAYNES ALYSSA & SEAN",
                        "Location": "135 TATNIC RD",
                        "Assessed_Total": 157700,
                        "Assessed_Land": 34500,
                        "Assessed_Building": 123200,
                        "Shape__Area": 24859.02734375,
                        "Shape__Length": 1420.4350563180556,
                    }
                }
            ],
            profile=_profile(),
        )

        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["county_name"], "brooklyn")
        self.assertEqual(row["county_name_display"], "BROOKLYN")
        self.assertEqual(row["county_fips"], "09190")
        self.assertEqual(row["parcel_id"], "09190-CT-019-16-85-4")
        self.assertEqual(row["parcel_improvement_status"], "likely_improved")
        self.assertFalse(bool(row["parcel_vacant_flag"]))

    def test_county_fips_lookup_supports_compact_aliases(self) -> None:
        profile = _profile(
            field_map={
                "source_object_id": "OBJECTID",
                "parcel_id_fields": ["PRINT_KEY"],
                "county_name": "COUNTY_NAME",
                "owner_name": "OWNER",
                "site_address": "ADDR",
                "assessed_total_value": "TOTAL_AV",
                "assessed_land_value": "LAND_AV",
                "assessed_improvement_value": "TOTAL_AV",
                "acreage": "ACRES",
            },
            county_fips_lookup={
                "new_york": "36061",
                "st_lawrence": "36089",
            },
        )
        profile = runtime_builder.RuntimeProfile(
            **{
                **profile.__dict__,
                "county_fips_lookup": runtime_builder._expanded_county_fips_lookup(profile.county_fips_lookup),
            }
        )

        frame = runtime_builder._transform_feature_batch(
            [
                {"attributes": {"OBJECTID": 1, "COUNTY_NAME": "NewYork", "PRINT_KEY": "A", "TOTAL_AV": 1000, "LAND_AV": 1000, "ACRES": 1.0}},
                {"attributes": {"OBJECTID": 2, "COUNTY_NAME": "StLawrence", "PRINT_KEY": "B", "TOTAL_AV": 1000, "LAND_AV": 1000, "ACRES": 1.0}},
            ],
            profile=profile,
        )

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[0]["county_fips"], "36061")
        self.assertEqual(frame.iloc[1]["county_fips"], "36089")


if __name__ == "__main__":
    unittest.main()
