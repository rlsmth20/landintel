from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas as pd
from shapely.geometry import MultiPolygon, Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import parcel_geometry_quality_ms as geometry_quality  # noqa: E402


def _workspace_temp_dir(name: str) -> Path:
    path = ROOT / "data" / "buildings_processed" / f"_{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _frame_for_geometry(
    parcel_row_id: str,
    geometry,
    *,
    shape_area_sqft: float,
    shape_length_ft: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parcel_row_id": parcel_row_id,
                "geometry": geometry.wkb,
                "shape_area": shape_area_sqft,
                "shape_length": shape_length_ft,
            }
        ]
    )


class ParcelGeometryQualityTests(unittest.TestCase):
    def test_long_narrow_parcel_is_access_strip(self) -> None:
        geometry = Polygon(
            [
                (-90.0000, 32.00000),
                (-89.9975, 32.00000),
                (-89.9975, 32.00008),
                (-90.0000, 32.00008),
            ]
        )
        frame = _frame_for_geometry("row_strip", geometry, shape_area_sqft=2500.0, shape_length_ft=560.0)

        enriched = geometry_quality.add_geometry_quality_fields(frame)
        row = enriched.iloc[0]

        self.assertEqual(row["geometry_quality_flag"], "access_strip")
        self.assertTrue(bool(row["geometry_review_excluded_flag"]))
        self.assertTrue(bool(row["geometry_training_excluded_flag"]))
        self.assertTrue(bool(row["geometry_default_leads_excluded_flag"]))

    def test_square_residential_lot_passes(self) -> None:
        geometry = Polygon(
            [
                (-90.00020, 32.00000),
                (-89.99988, 32.00000),
                (-89.99988, 32.00030),
                (-90.00020, 32.00030),
            ]
        )
        frame = _frame_for_geometry("row_square", geometry, shape_area_sqft=10890.0, shape_length_ft=420.0)

        enriched = geometry_quality.add_geometry_quality_fields(frame)
        row = enriched.iloc[0]

        self.assertEqual(row["geometry_quality_flag"], "good")
        self.assertFalse(bool(row["geometry_review_excluded_flag"]))
        self.assertFalse(bool(row["geometry_training_excluded_flag"]))
        self.assertFalse(bool(row["geometry_default_leads_excluded_flag"]))

    def test_large_rural_parcel_passes(self) -> None:
        geometry = Polygon(
            [
                (-90.0100, 32.0000),
                (-90.0060, 32.0000),
                (-90.0060, 32.0038),
                (-90.0100, 32.0038),
            ]
        )
        frame = _frame_for_geometry("row_rural", geometry, shape_area_sqft=400000.0, shape_length_ft=2600.0)

        enriched = geometry_quality.add_geometry_quality_fields(frame)
        row = enriched.iloc[0]

        self.assertEqual(row["geometry_quality_flag"], "good")
        self.assertGreater(float(row["area_acres"]), 5.0)
        self.assertFalse(bool(row["geometry_default_leads_excluded_flag"]))

    def test_fragmented_multipart_parcel_is_flagged(self) -> None:
        pieces = [
            Polygon([(-90.0000, 32.0000), (-89.9998, 32.0000), (-89.9998, 32.0002), (-90.0000, 32.0002)]),
            Polygon([(-89.9994, 32.0000), (-89.9992, 32.0000), (-89.9992, 32.0002), (-89.9994, 32.0002)]),
            Polygon([(-89.9988, 32.0000), (-89.9986, 32.0000), (-89.9986, 32.0002), (-89.9988, 32.0002)]),
            Polygon([(-89.9982, 32.0000), (-89.9980, 32.0000), (-89.9980, 32.0002), (-89.9982, 32.0002)]),
        ]
        geometry = MultiPolygon(pieces)
        frame = _frame_for_geometry("row_multi", geometry, shape_area_sqft=40000.0, shape_length_ft=1200.0)

        enriched = geometry_quality.add_geometry_quality_fields(frame)
        row = enriched.iloc[0]

        self.assertEqual(int(row["part_count"]), 4)
        self.assertTrue(bool(row["is_multipart"]))
        self.assertEqual(row["geometry_quality_flag"], "multipart_complex")
        self.assertTrue(bool(row["geometry_review_excluded_flag"]))
        self.assertTrue(bool(row["geometry_training_excluded_flag"]))

    def test_default_review_and_training_filters_apply_expected_flags(self) -> None:
        frame = pd.DataFrame(
            [
                {"parcel_row_id": "row_good", "geometry_quality_flag": "good", "geometry_review_excluded_flag": False, "geometry_training_excluded_flag": False, "geometry_default_leads_excluded_flag": False},
                {"parcel_row_id": "row_irregular", "geometry_quality_flag": "irregular", "geometry_review_excluded_flag": True, "geometry_training_excluded_flag": False, "geometry_default_leads_excluded_flag": False},
                {"parcel_row_id": "row_strip", "geometry_quality_flag": "access_strip", "geometry_review_excluded_flag": True, "geometry_training_excluded_flag": True, "geometry_default_leads_excluded_flag": True},
                {"parcel_row_id": "row_multi", "geometry_quality_flag": "multipart_complex", "geometry_review_excluded_flag": True, "geometry_training_excluded_flag": True, "geometry_default_leads_excluded_flag": False},
            ]
        )

        review = geometry_quality.filter_review_geometry_frame(frame)
        training = geometry_quality.filter_training_geometry_frame(frame)
        default_leads = geometry_quality.filter_default_leads_geometry_frame(frame)

        self.assertEqual(review["parcel_row_id"].astype("string").tolist(), ["row_good"])
        self.assertEqual(training["parcel_row_id"].astype("string").tolist(), ["row_good", "row_irregular"])
        self.assertEqual(default_leads["parcel_row_id"].astype("string").tolist(), ["row_good", "row_irregular", "row_multi"])

    def test_build_geometry_quality_artifact_writes_reusable_artifact(self) -> None:
        geometry_one = Polygon(
            [
                (-90.00020, 32.00000),
                (-89.99988, 32.00000),
                (-89.99988, 32.00030),
                (-90.00020, 32.00030),
            ]
        )
        geometry_two = Polygon(
            [
                (-90.0010, 32.0010),
                (-90.0006, 32.0010),
                (-90.0006, 32.0013),
                (-90.0010, 32.0013),
            ]
        )
        tmpdir_path = _workspace_temp_dir("test_geometry_quality_artifact")
        try:
            master_path = tmpdir_path / "parcel_master.parquet"
            artifact_path = tmpdir_path / "geometry_quality.parquet"
            summary_path = tmpdir_path / "geometry_quality_summary.json"
            pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_one",
                        "county_name": "alpha",
                        "shape_area": 10890.0,
                        "shape_length": 420.0,
                        "geometry": geometry_one.wkb,
                    },
                    {
                        "parcel_row_id": "row_two",
                        "county_name": "beta",
                        "shape_area": 3200.0,
                        "shape_length": 760.0,
                        "geometry": geometry_two.wkb,
                    },
                ]
            ).to_parquet(master_path, index=False)

            with mock.patch.object(geometry_quality, "PARCEL_MASTER_PATH", master_path):
                summary = geometry_quality.build_geometry_quality_artifact(
                    output_path=artifact_path,
                    summary_output_path=summary_path,
                    chunk_size=1,
                    force=True,
                )

            self.assertTrue(artifact_path.exists())
            self.assertTrue(summary_path.exists())
            artifact_frame = pd.read_parquet(artifact_path)
            self.assertEqual(len(artifact_frame), 2)
            self.assertEqual(
                artifact_frame.columns.tolist(),
                geometry_quality.GEOMETRY_QUALITY_ARTIFACT_COLUMNS,
            )
            self.assertEqual(summary["processed_rows"], 2)
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["row_count"], 2)
        finally:
            shutil.rmtree(tmpdir_path, ignore_errors=True)

    def test_load_geometry_quality_frame_reuses_existing_artifact(self) -> None:
        geometry = Polygon(
            [
                (-90.00020, 32.00000),
                (-89.99988, 32.00000),
                (-89.99988, 32.00030),
                (-90.00020, 32.00030),
            ]
        )
        tmpdir_path = _workspace_temp_dir("test_geometry_quality_reuse")
        try:
            master_path = tmpdir_path / "parcel_master.parquet"
            artifact_path = tmpdir_path / "geometry_quality.parquet"
            summary_path = tmpdir_path / "geometry_quality_summary.json"
            pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_one",
                        "county_name": "alpha",
                        "shape_area": 10890.0,
                        "shape_length": 420.0,
                        "geometry": geometry.wkb,
                    }
                ]
            ).to_parquet(master_path, index=False)

            with mock.patch.object(geometry_quality, "PARCEL_MASTER_PATH", master_path):
                geometry_quality.build_geometry_quality_artifact(
                    output_path=artifact_path,
                    summary_output_path=summary_path,
                    chunk_size=1,
                    force=True,
                )

            with mock.patch.object(
                geometry_quality,
                "add_geometry_quality_fields",
                side_effect=AssertionError("geometry quality should not be recomputed when artifact exists"),
            ):
                reused = geometry_quality.load_geometry_quality_frame(
                    ["row_one"],
                    reuse_artifact=True,
                    artifact_path=artifact_path,
                )

            self.assertEqual(reused["parcel_row_id"].astype("string").tolist(), ["row_one"])
            self.assertIn("geometry_quality_flag", reused.columns)
        finally:
            shutil.rmtree(tmpdir_path, ignore_errors=True)

    def test_load_geometry_quality_frame_can_build_artifact_if_missing(self) -> None:
        geometry = Polygon(
            [
                (-90.00020, 32.00000),
                (-89.99988, 32.00000),
                (-89.99988, 32.00030),
                (-90.00020, 32.00030),
            ]
        )
        tmpdir_path = _workspace_temp_dir("test_geometry_quality_build_if_missing")
        try:
            master_path = tmpdir_path / "parcel_master.parquet"
            artifact_path = tmpdir_path / "geometry_quality.parquet"
            summary_path = tmpdir_path / "geometry_quality_summary.json"
            pd.DataFrame(
                [
                    {
                        "parcel_row_id": "row_one",
                        "county_name": "alpha",
                        "shape_area": 10890.0,
                        "shape_length": 420.0,
                        "geometry": geometry.wkb,
                    }
                ]
            ).to_parquet(master_path, index=False)

            with mock.patch.object(geometry_quality, "PARCEL_MASTER_PATH", master_path):
                loaded = geometry_quality.load_geometry_quality_frame(
                    ["row_one"],
                    reuse_artifact=True,
                    artifact_path=artifact_path,
                    build_artifact_if_missing=True,
                    summary_output_path=summary_path,
                )

            self.assertTrue(artifact_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(loaded["parcel_row_id"].astype("string").tolist(), ["row_one"])
            self.assertEqual(len(loaded), 1)
        finally:
            shutil.rmtree(tmpdir_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
