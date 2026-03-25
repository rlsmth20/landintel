from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import parcel_marketability_ms as marketability  # noqa: E402


class ParcelMarketabilityTests(unittest.TestCase):
    def test_narrow_triangular_wedge_is_unbuildable_and_heavily_penalized(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_wedge",
                    "area_acres": 0.18,
                    "compactness": 0.20,
                    "parcel_width_ft_estimate": 15.0,
                    "parcel_frontage_ft_estimate": 201.0,
                    "aspect_ratio": 13.4,
                    "geometry_quality_flag": "good",
                    "nearby_building_density": 180.0,
                    "growth_pressure_bucket": "high",
                    "lead_score_total_effective": 96.0,
                }
            ]
        )

        enriched = marketability.add_geometry_marketability_fields(frame)
        adjusted = marketability.apply_geometry_marketability_score_adjustment(enriched)
        row = adjusted.iloc[0]

        self.assertEqual(row["geometry_marketability_base_flag"], "unbuildable_candidate")
        self.assertEqual(row["geometry_marketability_flag"], "unbuildable_candidate")
        self.assertEqual(row["geometry_marketability_action"], "exclude")
        self.assertTrue(bool(row["geometry_marketability_default_leads_excluded_flag"]))
        self.assertEqual(float(row["geometry_penalty_points"]), -60.0)
        self.assertFalse(bool(row["geometry_effective_buildable_flag"]))
        self.assertLess(float(row["lead_score_total_effective"]), 40.0)

    def test_urban_triangular_remnant_is_excluded_as_poor_geometry(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_urban_remnant",
                    "area_acres": 0.61,
                    "compactness": 0.23,
                    "parcel_width_ft_estimate": 33.0,
                    "parcel_frontage_ft_estimate": 150.0,
                    "aspect_ratio": 4.8,
                    "geometry_quality_flag": "good",
                    "nearby_building_density": 220.0,
                    "growth_pressure_bucket": "high",
                    "lead_score_total_effective": 84.0,
                }
            ]
        )

        enriched = marketability.add_geometry_marketability_fields(frame)
        adjusted = marketability.apply_geometry_marketability_score_adjustment(enriched)
        row = adjusted.iloc[0]

        self.assertEqual(row["geometry_marketability_base_flag"], "poor_geometry")
        self.assertEqual(row["geometry_marketability_flag"], "poor_geometry")
        self.assertEqual(row["geometry_marketability_context"], "urban_suburban")
        self.assertEqual(row["geometry_marketability_action"], "exclude")
        self.assertTrue(bool(row["geometry_marketability_default_leads_excluded_flag"]))
        self.assertEqual(float(row["geometry_penalty_points"]), -45.0)
        self.assertIn("urban or suburban context", str(row["geometry_penalty_reason"]).lower())

    def test_suburban_frontage_scrap_is_excluded_or_heavily_penalized(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_frontage",
                    "area_acres": 0.62,
                    "compactness": 0.23,
                    "parcel_width_ft_estimate": 38.0,
                    "parcel_frontage_ft_estimate": 205.0,
                    "aspect_ratio": 5.4,
                    "geometry_quality_flag": "good",
                    "nearby_building_density": 95.0,
                    "growth_pressure_bucket": "moderate",
                }
            ]
        )

        enriched = marketability.add_geometry_marketability_fields(frame)
        row = enriched.iloc[0]

        self.assertEqual(row["geometry_marketability_flag"], "poor_geometry")
        self.assertEqual(row["geometry_marketability_action"], "exclude")
        self.assertTrue(bool(row["geometry_marketability_default_leads_excluded_flag"]))
        self.assertEqual(float(row["geometry_penalty_points"]), -45.0)
        self.assertIn("remnant", str(row["geometry_penalty_reason"]).lower())

    def test_normal_rectangular_lot_is_marketable(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_good",
                    "area_acres": 0.75,
                    "compactness": 0.76,
                    "parcel_width_ft_estimate": 115.0,
                    "parcel_frontage_ft_estimate": 132.0,
                    "aspect_ratio": 1.15,
                    "geometry_quality_flag": "good",
                    "nearby_building_density": 80.0,
                    "growth_pressure_bucket": "moderate",
                }
            ]
        )

        enriched = marketability.add_geometry_marketability_fields(frame)
        row = enriched.iloc[0]

        self.assertEqual(row["geometry_marketability_flag"], "marketable")
        self.assertEqual(row["geometry_marketability_action"], "keep")
        self.assertFalse(bool(row["geometry_marketability_default_leads_excluded_flag"]))
        self.assertEqual(float(row["geometry_penalty_points"]), 0.0)
        self.assertTrue(bool(row["geometry_effective_buildable_flag"]))

    def test_larger_irregular_rural_parcel_gets_rural_tolerance(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "parcel_row_id": "row_rural_irregular",
                    "area_acres": 12.0,
                    "compactness": 0.17,
                    "parcel_width_ft_estimate": 180.0,
                    "parcel_frontage_ft_estimate": 820.0,
                    "aspect_ratio": 4.6,
                    "geometry_quality_flag": "irregular",
                    "nearby_building_density": 8.0,
                    "growth_pressure_bucket": "low",
                    "land_use": "timber",
                    "is_multipart": False,
                    "part_count": 1,
                }
            ]
        )

        enriched = marketability.add_geometry_marketability_fields(frame)
        row = enriched.iloc[0]

        self.assertEqual(row["geometry_marketability_base_flag"], "constrained")
        self.assertEqual(row["geometry_marketability_flag"], "constrained")
        self.assertEqual(row["geometry_marketability_context"], "rural")
        self.assertEqual(row["geometry_marketability_action"], "keep")
        self.assertFalse(bool(row["geometry_marketability_default_leads_excluded_flag"]))
        self.assertEqual(float(row["geometry_penalty_points"]), 0.0)
        self.assertTrue(bool(row["geometry_effective_buildable_flag"]))


if __name__ == "__main__":
    unittest.main()
