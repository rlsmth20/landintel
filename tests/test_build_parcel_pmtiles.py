from __future__ import annotations

import sys
import unittest
from pathlib import Path
import shutil
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point, box


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import build_parcel_pmtiles as parcel_pmtiles  # noqa: E402


class BuildParcelPmtilesTests(unittest.TestCase):
    def test_build_pmtiles_from_feature_cache_writes_archive(self) -> None:
        temp_root = ROOT / "data" / "buildings_processed"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_root / f"parcel-pmtiles-test-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache_path = temp_dir / "features.parquet"
            output_path = temp_dir / "ar_parcels.pmtiles"
            frame = pd.DataFrame(
                {
                    "parcel_row_id": ["ar_test_1", "ar_test_2"],
                    "parcel_id": ["001-1", "001-2"],
                    "county_name": ["pulaski", "pulaski"],
                    "wetland_flag": [False, False],
                    "flood_risk_score": [0.0, 1.0],
                    "road_access_tier": ["direct", "near"],
                    "source_object_id": [1, 2],
                    "geometry_wkb": [
                        box(-92.35, 34.72, -92.34, 34.73).wkb,
                        box(-92.33, 34.71, -92.32, 34.72).wkb,
                    ],
                }
            )
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), cache_path)
            settings = parcel_pmtiles.TileBuildSettings(
                state_code="ar",
                state_name="Arkansas",
                build_source="app_ready",
                layer="parcels",
                min_zoom=6,
                max_zoom=6,
                frontend_url="/tiles/ar_parcels.pmtiles",
                public_url="https://landintel.vercel.app/tiles/ar_parcels.pmtiles",
                output_path=output_path,
                geometry_cache_path=cache_path,
                summary_output_path=temp_dir / "summary.json",
                publish_manifest_output_path=temp_dir / "publish_manifest.json",
                cloudflare_object_key="tiles/ar_parcels.pmtiles",
            )

            summary = parcel_pmtiles.build_pmtiles_from_feature_cache(settings, batch_size=16)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertGreater(summary["tile_count"], 0)
            self.assertEqual(summary["tile_build_method"], "python_pmtiles_writer")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_arcgis_geometry_to_shape_supports_multi_polygon_rings(self) -> None:
        geometry = {
            "rings": [
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
                [[2.0, 0.0], [2.0, 1.0], [3.0, 1.0], [3.0, 0.0], [2.0, 0.0]],
            ]
        }

        geometry_shape = parcel_pmtiles._arcgis_geometry_to_shape(geometry)  # noqa: SLF001

        self.assertIsNotNone(geometry_shape)
        self.assertEqual(geometry_shape.geom_type, "MultiPolygon")
        self.assertAlmostEqual(float(geometry_shape.area), 2.0, places=6)

    def test_build_pmtiles_supports_point_feature_cache(self) -> None:
        temp_root = ROOT / "data" / "buildings_processed"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = temp_root / f"parcel-pmtiles-point-test-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            cache_path = temp_dir / "features.parquet"
            output_path = temp_dir / "ny_parcels.pmtiles"
            frame = pd.DataFrame(
                {
                    "parcel_row_id": ["ny_test_1"],
                    "parcel_id": ["123.45-1-2"],
                    "county_name": ["albany"],
                    "wetland_flag": [False],
                    "flood_risk_score": [0.0],
                    "road_access_tier": ["direct"],
                    "source_object_id": [1],
                    "geometry_wkb": [Point(-73.75, 42.65).wkb],
                }
            )
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), cache_path)
            settings = parcel_pmtiles.TileBuildSettings(
                state_code="ny",
                state_name="New York",
                build_source="parcel_master",
                layer="parcels",
                min_zoom=7,
                max_zoom=7,
                frontend_url="/tiles/ny_parcels.pmtiles",
                public_url="https://landintel.vercel.app/tiles/ny_parcels.pmtiles",
                output_path=output_path,
                geometry_cache_path=cache_path,
                summary_output_path=temp_dir / "summary.json",
                publish_manifest_output_path=temp_dir / "publish_manifest.json",
                cloudflare_object_key="tiles/ny_parcels.pmtiles",
            )

            summary = parcel_pmtiles.build_pmtiles_from_feature_cache(settings, batch_size=16)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertGreater(summary["tile_count"], 0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
