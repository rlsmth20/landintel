from __future__ import annotations

import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))
sys.path.insert(0, str(ROOT / "backend"))

from app.services.runtime_state_service import RuntimeStateService  # noqa: E402


class RuntimeStateServiceGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = ROOT / "data" / "buildings_processed" / f"runtime-state-geometry-{uuid.uuid4().hex}"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def _write_geometry_cache(self, rows: list[dict[str, object]]) -> Path:
        cache_path = self.temp_root / "geometry_cache.parquet"
        frame = pd.DataFrame(rows)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), cache_path)
        return cache_path

    def test_arkansas_geometry_uses_cache_before_live_fetch(self) -> None:
        cache_path = self._write_geometry_cache(
            [
                {
                    "parcel_row_id": "ar_test_1",
                    "parcel_id": "001-00001-000",
                    "county_name": "pulaski",
                    "source_object_id": 101,
                    "geometry_wkb": box(-92.35, 34.72, -92.34, 34.73).wkb,
                }
            ]
        )
        service = RuntimeStateService("ar")
        service.definition.raw.setdefault("parcel_tiles", {})["geometry_cache_path"] = str(cache_path)

        row = pd.Series(
            {
                "parcel_row_id": "ar_test_1",
                "parcel_id": "001-00001-000",
                "county_name": "pulaski",
                "lead_score_total": 88.0,
                "source_object_id": 101,
            }
        )

        with mock.patch.object(service, "_row_for_parcel", return_value=row), mock.patch.object(
            service,
            "_source_geometry_geojson",
            side_effect=AssertionError("live fetch should not be used when cache is present"),
        ):
            payload = service.get_parcel_geometry("ar_test_1", zoom=14)

        self.assertEqual(payload["feature_count"], 1)
        self.assertEqual(payload["render_mode"], "polygons")
        feature = payload["feature_collection"]["features"][0]
        self.assertEqual(feature["properties"]["parcel_row_id"], "ar_test_1")
        self.assertEqual(feature["properties"]["parcel_id"], "001-00001-000")
        self.assertEqual(feature["geometry"]["type"], "Polygon")

    def test_arkansas_geometry_falls_back_to_point_when_live_fetch_fails(self) -> None:
        service = RuntimeStateService("ar")
        service.definition.raw.setdefault("parcel_tiles", {})["geometry_cache_path"] = str(self.temp_root / "missing.parquet")

        row = pd.Series(
            {
                "parcel_row_id": "ar_test_2",
                "parcel_id": "001-00002-000",
                "county_name": "pulaski",
                "lead_score_total": 77.0,
                "source_object_id": 202,
                "longitude": -92.31,
                "latitude": 34.75,
            }
        )

        with mock.patch.object(service, "_row_for_parcel", return_value=row), mock.patch.object(
            service,
            "_source_geometry_geojson",
            side_effect=RuntimeError("ArcGIS unavailable"),
        ):
            payload = service.get_parcel_geometry("ar_test_2", zoom=14)

        self.assertEqual(payload["feature_count"], 1)
        self.assertEqual(payload["render_mode"], "points")
        feature = payload["feature_collection"]["features"][0]
        self.assertEqual(feature["properties"]["parcel_row_id"], "ar_test_2")
        self.assertEqual(feature["geometry"]["type"], "Point")

    def test_detail_uses_parcel_master_when_not_in_runtime_detail_dataset(self) -> None:
        service = RuntimeStateService("ct")
        row = pd.Series(
            {
                "parcel_row_id": "ct_test_1",
                "parcel_id": "001-0001",
                "county_name": "hartford",
                "latitude": 41.7658,
                "longitude": -72.6734,
            }
        )

        with mock.patch.object(service, "_detail_row", return_value=None), mock.patch.object(
            service,
            "_parcel_master_row",
            return_value=row,
        ):
            payload = service.get_lead_detail("ct_test_1")

        self.assertIsNotNone(payload)
        self.assertEqual(payload["parcel_row_id"], "ct_test_1")
        self.assertEqual(payload["parcel_id"], "001-0001")
        self.assertEqual(payload["county_name"], "hartford")
        self.assertEqual(payload["geometry"]["type"], "point_reference")


if __name__ == "__main__":
    unittest.main()
