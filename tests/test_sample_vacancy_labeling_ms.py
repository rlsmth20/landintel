from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image
from shapely.geometry import GeometryCollection, LineString, Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import sample_vacancy_labeling_ms as labeling_sample  # noqa: E402
import vacancy_ai_common as vacancy_common  # noqa: E402


def _base_row(parcel_row_id: str, county_name: str, latitude: float, longitude: float) -> dict[str, object]:
    return {
        "parcel_row_id": parcel_row_id,
        "parcel_id": f"pid-{parcel_row_id}",
        "county_name": county_name,
        "latitude": latitude,
        "longitude": longitude,
        "acreage": 1.0,
        "building_count": 0,
        "building_area_total": 0.0,
        "parcel_vacant_flag": True,
        "ai_building_present_flag": True,
        "ai_building_present_probability": 0.7,
        "building_present_confidence": 70.0,
        "building_presence_reason": "Synthetic test row.",
        "imagery_best_crop_label": "parcel_focus",
        "imagery_crop_strategy": "parcel_mask_multi_crop_v1",
        "parcel_boundary_crop_ready_flag": True,
        "imagery_driveway_signal": 60.0,
        "imagery_clearing_signal": 20.0,
        "nearby_building_density": 120.0,
        "shape_compactness": 0.45,
        "parcel_width_ft_estimate": 90.0,
        "parcel_aspect_ratio_estimate": 2.0,
        "lead_score_total": 55.0,
    }


def _build_test_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for index, county in enumerate(["alpha", "beta", "gamma"]):
        row = _base_row(f"scene_{index}", county, 32.0 + index / 100.0, -90.0 - index / 100.0)
        row["building_present_confidence"] = 72.0 + index
        row["ai_building_present_probability"] = 0.78 + (index * 0.01)
        row["imagery_driveway_signal"] = 82.0 + index
        row["imagery_clearing_signal"] = 33.0 + index
        rows.append(row)

    for index, county in enumerate(["delta", "epsilon", "zeta"]):
        row = _base_row(f"neighbor_{index}", county, 32.1 + index / 100.0, -90.1 - index / 100.0)
        row["building_present_confidence"] = 66.0 + index
        row["ai_building_present_probability"] = 0.74 + (index * 0.01)
        row["imagery_driveway_signal"] = 42.0 + index
        row["imagery_clearing_signal"] = 18.0 + index
        row["nearby_building_density"] = 460.0 + (index * 10.0)
        row["shape_compactness"] = 0.22
        row["parcel_width_ft_estimate"] = 42.0 + index
        row["parcel_aspect_ratio_estimate"] = 5.5 + index
        rows.append(row)

    for index, county in enumerate(["eta", "theta"]):
        row = _base_row(f"improved_{index}", county, 32.2 + index / 100.0, -90.2 - index / 100.0)
        row["building_count"] = 2
        row["building_area_total"] = 1800.0 + (index * 200.0)
        row["parcel_vacant_flag"] = False
        row["building_present_confidence"] = 91.0 + index
        row["ai_building_present_probability"] = 0.95
        row["imagery_driveway_signal"] = 88.0
        row["imagery_clearing_signal"] = 52.0
        rows.append(row)

    for index, county in enumerate(["iota", "kappa"]):
        row = _base_row(f"vacant_{index}", county, 32.3 + index / 100.0, -90.3 - index / 100.0)
        row["ai_building_present_flag"] = False
        row["building_present_confidence"] = 9.0 + index
        row["ai_building_present_probability"] = 0.08
        row["imagery_driveway_signal"] = 8.0
        row["imagery_clearing_signal"] = 2.0
        rows.append(row)

    return pd.DataFrame(rows)


def _multipart_geometry_collection(address: vacancy_common.TileAddress) -> tuple[GeometryCollection, tuple[float, float], tuple[float, float]]:
    west, south, east, north = vacancy_common.tile_bounds(address)
    dx = east - west
    dy = north - south
    south_polygon = Polygon(
        [
            (west + (dx * 0.35), south + (dy * 0.16)),
            (west + (dx * 0.58), south + (dy * 0.16)),
            (west + (dx * 0.58), south + (dy * 0.32)),
            (west + (dx * 0.35), south + (dy * 0.32)),
        ]
    )
    north_polygon = Polygon(
        [
            (west + (dx * 0.35), south + (dy * 0.68)),
            (west + (dx * 0.58), south + (dy * 0.68)),
            (west + (dx * 0.58), south + (dy * 0.84)),
            (west + (dx * 0.35), south + (dy * 0.84)),
        ]
    )
    road_center_line = LineString(
        [
            (west + (dx * 0.20), south + (dy * 0.50)),
            (west + (dx * 0.80), south + (dy * 0.50)),
        ]
    )
    south_center = (west + (dx * 0.465), south + (dy * 0.24))
    north_center = (west + (dx * 0.465), south + (dy * 0.76))
    return GeometryCollection([south_polygon, north_polygon, road_center_line]), north_center, south_center


def _multipart_cross_tile_geometry(
    south_address: vacancy_common.TileAddress,
) -> tuple[GeometryCollection, vacancy_common.TileAddress, tuple[float, float], tuple[float, float]]:
    north_address = vacancy_common.TileAddress(x=south_address.x, y=south_address.y - 1, z=south_address.z)
    south_west, south_south, south_east, south_north = vacancy_common.tile_bounds(south_address)
    north_west, north_south, north_east, north_north = vacancy_common.tile_bounds(north_address)
    south_dx = south_east - south_west
    south_dy = south_north - south_south
    north_dx = north_east - north_west
    north_dy = north_north - north_south
    south_polygon = Polygon(
        [
            (south_west + (south_dx * 0.30), south_south + (south_dy * 0.68)),
            (south_west + (south_dx * 0.58), south_south + (south_dy * 0.68)),
            (south_west + (south_dx * 0.58), south_south + (south_dy * 0.92)),
            (south_west + (south_dx * 0.30), south_south + (south_dy * 0.92)),
        ]
    )
    north_polygon = Polygon(
        [
            (north_west + (north_dx * 0.30), north_south + (north_dy * 0.08)),
            (north_west + (north_dx * 0.58), north_south + (north_dy * 0.08)),
            (north_west + (north_dx * 0.58), north_south + (north_dy * 0.34)),
            (north_west + (north_dx * 0.30), north_south + (north_dy * 0.34)),
        ]
    )
    road_center_line = LineString(
        [
            (north_west + (north_dx * 0.20), north_south + (north_dy * 0.98)),
            (south_west + (south_dx * 0.80), south_south + (south_dy * 0.02)),
        ]
    )
    north_center = (north_west + (north_dx * 0.44), north_south + (north_dy * 0.21))
    south_center = (south_west + (south_dx * 0.44), south_south + (south_dy * 0.80))
    return GeometryCollection([south_polygon, north_polygon, road_center_line]), north_address, north_center, south_center


def _single_polygon_cross_tile_geometry(
    west_address: vacancy_common.TileAddress,
) -> tuple[Polygon, vacancy_common.TileAddress]:
    east_address = vacancy_common.TileAddress(x=west_address.x + 1, y=west_address.y, z=west_address.z)
    west, south, east, north = vacancy_common.tile_bounds(west_address)
    tile_dx = east - west
    tile_dy = north - south
    return (
        Polygon(
            [
                (west + (tile_dx * 0.70), south + (tile_dy * 0.42)),
                (west + (tile_dx * 1.30), south + (tile_dy * 0.42)),
                (west + (tile_dx * 1.30), south + (tile_dy * 0.58)),
                (west + (tile_dx * 0.70), south + (tile_dy * 0.58)),
            ]
        ),
        east_address,
    )


class LabelingSampleTests(unittest.TestCase):
    def test_save_review_crop_assets_writes_actual_scored_images(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        west, south, east, north = vacancy_common.tile_bounds(address)
        dx = east - west
        dy = north - south
        geometry = Polygon(
            [
                (west + (dx * 0.35), south + (dy * 0.35)),
                (west + (dx * 0.65), south + (dy * 0.35)),
                (west + (dx * 0.65), south + (dy * 0.65)),
                (west + (dx * 0.35), south + (dy * 0.65)),
            ]
        )

        temp_path = ROOT / "data" / "buildings_processed" / "_test_review_assets"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            raw_image_path = temp_path / "raw_tile.jpg"
            Image.new("RGB", (256, 256), (194, 170, 138)).save(raw_image_path, format="JPEG")
            prepared = vacancy_common.prepare_parcel_aware_image(
                raw_image_path,
                address=address,
                geometry_value=geometry.wkb,
                acreage=0.5,
            )
            original_asset_dir = labeling_sample.REVIEW_ASSET_DIR
            try:
                labeling_sample.REVIEW_ASSET_DIR = temp_path / "review_assets"
                assets = labeling_sample._save_review_crop_assets(
                    parcel_row_id="row_test",
                    county_name="alpha",
                    prepared=prepared,
                    tile_label_value="z19_x131072_y204800",
                    tile_rank=2,
                )
            finally:
                labeling_sample.REVIEW_ASSET_DIR = original_asset_dir

            self.assertTrue(prepared["parcel_boundary_crop_ready_flag"])
            self.assertTrue(Path(assets["masked_parcel_tile_path"]).exists())
            self.assertTrue(Path(assets["masked_parcel_core_crop_path"]).exists())
            self.assertTrue(Path(assets["masked_parcel_focus_crop_path"]).exists())
            self.assertIn("tile02_z19_x131072_y204800", Path(assets["masked_parcel_tile_path"]).name)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_attach_imagery_columns_uses_selected_scored_tile_assets(self) -> None:
        temp_path = ROOT / "data" / "buildings_processed" / "_test_review_assets_attach"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            centroid_image_path = temp_path / "centroid.png"
            best_tile_image_path = temp_path / "best_tile.png"
            Image.new("RGB", (256, 256), (10, 20, 30)).save(centroid_image_path, format="PNG")
            Image.new("RGB", (256, 256), (210, 180, 40)).save(best_tile_image_path, format="PNG")

            centroid_address = vacancy_common.TileAddress(x=100, y=200, z=19)
            best_address = vacancy_common.TileAddress(x=101, y=200, z=19)
            tile_plan = {
                "tile_records": [
                    {
                        "address": centroid_address,
                        "tile_label": "z19_x100_y200",
                        "tile_coordinate": "19/100/200",
                        "centroid_tile_flag": True,
                        "tile_rank": 1,
                        "parcel_tile_coverage_pct": 8.5,
                        "parcel_bbox_tile_coverage_pct": 6.1,
                    },
                    {
                        "address": best_address,
                        "tile_label": "z19_x101_y200",
                        "tile_coordinate": "19/101/200",
                        "centroid_tile_flag": False,
                        "tile_rank": 2,
                        "parcel_tile_coverage_pct": 61.4,
                        "parcel_bbox_tile_coverage_pct": 54.8,
                    },
                ],
                "parcel_tile_coverage_pct": 8.5,
                "parcel_bbox_tile_coverage_pct": 6.1,
                "full_parcel_visible_flag": False,
                "parcel_extent_exceeds_tile_flag": True,
                "parcel_tile_low_coverage_flag": True,
                "multi_tile_candidate_flag": True,
                "parcel_covering_tile_count": 2,
                "tile_coordinates": json.dumps(["19/100/200", "19/101/200"]),
                "unique_tile_count": 2,
                "duplicate_tile_flag": False,
            }
            frame = pd.DataFrame(
                [
                    {
                        **_base_row("row_attach", "alpha", 32.0, -90.0),
                        "geometry": b"fake",
                        "best_tile_label": "z19_x101_y200",
                    }
                ]
            )

            def fake_prepare(image_source, **kwargs):  # type: ignore[no-untyped-def]
                image = Image.open(image_source).convert("RGB")
                return {
                    "image": image,
                    "crop_specs": [("parcel_core", (0, 0, 64, 64)), ("parcel_focus", (0, 0, 96, 96))],
                    "parcel_boundary_crop_ready_flag": True,
                    "imagery_crop_strategy": "parcel_mask_tight_crop_v2",
                }

            def fake_resolve(**kwargs):  # type: ignore[no-untyped-def]
                address = kwargs["address"]
                if (address.x, address.y, address.z) == (centroid_address.x, centroid_address.y, centroid_address.z):
                    return centroid_image_path
                return best_tile_image_path

            original_asset_dir = labeling_sample.REVIEW_ASSET_DIR
            try:
                labeling_sample.REVIEW_ASSET_DIR = temp_path / "review_assets"
                with (
                    mock.patch.object(labeling_sample, "build_parcel_inference_tile_plan", return_value=tile_plan),
                    mock.patch.object(labeling_sample, "_resolve_review_tile_path", side_effect=fake_resolve),
                    mock.patch.object(labeling_sample, "prepare_parcel_aware_image", side_effect=fake_prepare),
                ):
                    attached = labeling_sample.attach_imagery_columns(
                        frame,
                        zoom=19,
                        fetch_images=False,
                        tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
                    )
            finally:
                labeling_sample.REVIEW_ASSET_DIR = original_asset_dir

            row = attached.iloc[0]
            self.assertEqual(row["review_tile_label"], "z19_x101_y200")
            self.assertEqual(row["unique_tile_count"], 2)
            self.assertFalse(bool(row["duplicate_tile_flag"]))
            self.assertIn("z19_x101_y200", row["masked_parcel_tile_path"])
            self.assertIn("z19_x100_y200", row["review_tile_sample_labels"])
            self.assertIn("z19_x101_y200", row["review_tile_sample_labels"])
            self.assertTrue(Path(row["review_tile_manifest_path"]).exists())

            selected_image = Image.open(row["masked_parcel_tile_path"]).convert("RGB")
            self.assertEqual(selected_image.getpixel((0, 0)), (210, 180, 40))
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_review_export_preserves_multipart_mask_geometry(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, north_center, south_center = _multipart_geometry_collection(address)

        temp_path = ROOT / "data" / "buildings_processed" / "_test_review_assets_multipart"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            raw_image_path = temp_path / "raw_tile.png"
            Image.new("RGB", (256, 256), (194, 170, 138)).save(raw_image_path, format="PNG")
            prepared = vacancy_common.prepare_parcel_aware_image(
                raw_image_path,
                address=address,
                geometry_value=geometry.wkb,
                acreage=1.0,
            )
            original_asset_dir = labeling_sample.REVIEW_ASSET_DIR
            try:
                labeling_sample.REVIEW_ASSET_DIR = temp_path / "review_assets"
                assets = labeling_sample._save_review_crop_assets(
                    parcel_row_id="row_multipart",
                    county_name="alpha",
                    prepared=prepared,
                    tile_label_value="z19_x131072_y204800",
                    tile_rank=1,
                )
            finally:
                labeling_sample.REVIEW_ASSET_DIR = original_asset_dir

            masked_tile = Image.open(assets["masked_parcel_tile_path"]).convert("RGB")
            north_pixel = tuple(int(round(value)) for value in vacancy_common._tile_pixel(north_center[0], north_center[1], address))
            south_pixel = tuple(int(round(value)) for value in vacancy_common._tile_pixel(south_center[0], south_center[1], address))
            self.assertNotEqual(masked_tile.getpixel(north_pixel), (0, 0, 0))
            self.assertNotEqual(masked_tile.getpixel(south_pixel), (0, 0, 0))
            self.assertEqual(prepared["polygon_part_count"], 2)
            self.assertEqual(prepared["clipped_polygon_part_count"], 2)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_attach_imagery_columns_uses_shared_tile_coverage_diagnostics(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, _, _ = _multipart_geometry_collection(address)
        tile_plan = vacancy_common.build_parcel_inference_tile_plan(
            geometry.wkb,
            address,
            use_multi_tile_extent=True,
        )
        frame = pd.DataFrame(
            [
                {
                    **_base_row("row_shared", "alpha", 32.0, -90.0),
                    "geometry": geometry.wkb,
                    "best_tile_label": tile_plan["tile_records"][0]["tile_label"],
                }
            ]
        )

        temp_path = ROOT / "data" / "buildings_processed" / "_test_review_assets_shared"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            raw_image_path = temp_path / "raw_tile.png"
            Image.new("RGB", (256, 256), (194, 170, 138)).save(raw_image_path, format="PNG")

            def fake_resolve(**kwargs):  # type: ignore[no-untyped-def]
                return raw_image_path

            original_asset_dir = labeling_sample.REVIEW_ASSET_DIR
            try:
                labeling_sample.REVIEW_ASSET_DIR = temp_path / "review_assets"
                with mock.patch.object(labeling_sample, "_resolve_review_tile_path", side_effect=fake_resolve):
                    attached = labeling_sample.attach_imagery_columns(
                        frame,
                        zoom=19,
                        fetch_images=False,
                        tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
                    )
            finally:
                labeling_sample.REVIEW_ASSET_DIR = original_asset_dir

            row = attached.iloc[0]
            self.assertEqual(row["parcel_tile_coverage_pct"], tile_plan["parcel_tile_coverage_pct"])
            self.assertEqual(row["parcel_bbox_tile_coverage_pct"], tile_plan["parcel_bbox_tile_coverage_pct"])
            self.assertEqual(bool(row["multi_tile_candidate_flag"]), bool(tile_plan["multi_tile_candidate_flag"]))
            self.assertEqual(int(row["unique_tile_count"]), int(tile_plan["unique_tile_count"]))
            self.assertEqual(json.loads(row["tile_coordinates"]), json.loads(tile_plan["tile_coordinates"]))
            self.assertEqual(row["parcel_id"], "pid-row_shared")
            manifest = json.loads(Path(row["review_tile_manifest_path"]).read_text(encoding="utf-8"))
            self.assertTrue(all(item["parcel_row_id"] == "row_shared" for item in manifest))
            self.assertTrue(all(item["parcel_id"] == "pid-row_shared" for item in manifest))
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_attach_imagery_columns_builds_multipart_composite_when_selected_tile_misses_other_part(self) -> None:
        south_address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, north_address, north_center, south_center = _multipart_cross_tile_geometry(south_address)
        tile_plan = vacancy_common.build_parcel_inference_tile_plan(
            geometry.wkb,
            south_address,
            use_multi_tile_extent=True,
        )
        selected_tile_record = next(item for item in tile_plan["tile_records"] if item["centroid_tile_flag"])
        frame = pd.DataFrame(
            [
                {
                    **_base_row("row_multipart_cross_tile", "alpha", 32.0, -90.0),
                    "geometry": geometry.wkb,
                    "best_tile_label": selected_tile_record["tile_label"],
                }
            ]
        )

        temp_path = ROOT / "data" / "buildings_processed" / "_test_review_assets_cross_tile"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            north_image_path = temp_path / "north_tile.png"
            south_image_path = temp_path / "south_tile.png"
            Image.new("RGB", (256, 256), (80, 120, 180)).save(north_image_path, format="PNG")
            Image.new("RGB", (256, 256), (180, 120, 80)).save(south_image_path, format="PNG")

            def fake_resolve(**kwargs):  # type: ignore[no-untyped-def]
                address = kwargs["address"]
                if (address.x, address.y, address.z) == (north_address.x, north_address.y, north_address.z):
                    return north_image_path
                return south_image_path

            original_asset_dir = labeling_sample.REVIEW_ASSET_DIR
            try:
                labeling_sample.REVIEW_ASSET_DIR = temp_path / "review_assets"
                with mock.patch.object(labeling_sample, "_resolve_review_tile_path", side_effect=fake_resolve):
                    attached = labeling_sample.attach_imagery_columns(
                        frame,
                        zoom=19,
                        fetch_images=False,
                        tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
                    )
            finally:
                labeling_sample.REVIEW_ASSET_DIR = original_asset_dir

            row = attached.iloc[0]
            self.assertIn("parcel_composite", row["masked_parcel_tile_path"])
            self.assertEqual(int(row["polygon_part_count"]), 2)
            self.assertEqual(int(row["clipped_polygon_part_count"]), 2)

            masked_tile = Image.open(row["masked_parcel_tile_path"]).convert("RGB")
            masked_bbox = masked_tile.getbbox()
            self.assertIsNotNone(masked_bbox)
            self.assertGreater(masked_bbox[0], 0)
            self.assertGreater(masked_bbox[1], 0)
            self.assertLess(masked_bbox[2], masked_tile.size[0])
            self.assertLess(masked_bbox[3], masked_tile.size[1])
            masked_colors = {pixel for pixel in masked_tile.getdata() if pixel != (0, 0, 0)}
            self.assertIn((80, 120, 180), masked_colors)
            self.assertIn((180, 120, 80), masked_colors)

            focus_crop = Image.open(row["masked_parcel_focus_crop_path"]).convert("RGB")
            focus_bbox = focus_crop.getbbox()
            self.assertIsNotNone(focus_bbox)
            self.assertGreater(focus_bbox[0], 0)
            self.assertGreater(focus_bbox[1], 0)
            self.assertLess(focus_bbox[2], focus_crop.size[0])
            self.assertLess(focus_bbox[3], focus_crop.size[1])
            colors = {pixel for pixel in focus_crop.getdata() if pixel != (0, 0, 0)}
            self.assertIn((80, 120, 180), colors)
            self.assertIn((180, 120, 80), colors)
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_attach_imagery_columns_builds_composite_for_single_polygon_cross_tile_parcel(self) -> None:
        west_address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, east_address = _single_polygon_cross_tile_geometry(west_address)
        tile_plan = vacancy_common.build_parcel_inference_tile_plan(
            geometry.wkb,
            west_address,
            use_multi_tile_extent=True,
        )
        selected_tile_record = next(item for item in tile_plan["tile_records"] if str(item["tile_label"]) == str(tile_plan["tile_records"][1]["tile_label"]))
        frame = pd.DataFrame(
            [
                {
                    **_base_row("row_single_cross_tile", "alpha", 32.0, -90.0),
                    "geometry": geometry.wkb,
                    "best_tile_label": selected_tile_record["tile_label"],
                }
            ]
        )

        temp_path = ROOT / "data" / "buildings_processed" / "_test_review_assets_single_cross_tile"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        temp_path.mkdir(parents=True, exist_ok=True)
        try:
            west_image_path = temp_path / "west_tile.png"
            east_image_path = temp_path / "east_tile.png"
            Image.new("RGB", (256, 256), (60, 100, 160)).save(west_image_path, format="PNG")
            Image.new("RGB", (256, 256), (160, 100, 60)).save(east_image_path, format="PNG")

            def fake_resolve(**kwargs):  # type: ignore[no-untyped-def]
                address = kwargs["address"]
                if (address.x, address.y, address.z) == (east_address.x, east_address.y, east_address.z):
                    return east_image_path
                return west_image_path

            original_asset_dir = labeling_sample.REVIEW_ASSET_DIR
            try:
                labeling_sample.REVIEW_ASSET_DIR = temp_path / "review_assets"
                with mock.patch.object(labeling_sample, "_resolve_review_tile_path", side_effect=fake_resolve):
                    attached = labeling_sample.attach_imagery_columns(
                        frame,
                        zoom=19,
                        fetch_images=False,
                        tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
                    )
            finally:
                labeling_sample.REVIEW_ASSET_DIR = original_asset_dir

            row = attached.iloc[0]
            self.assertIn("parcel_composite", row["masked_parcel_tile_path"])
            masked_tile = Image.open(row["masked_parcel_tile_path"]).convert("RGB")
            self.assertIsNotNone(masked_tile.getbbox())
            colors = {pixel for pixel in masked_tile.getdata() if pixel != (0, 0, 0)}
            self.assertIn((60, 100, 160), colors)
            self.assertIn((160, 100, 60), colors)
            self.assertNotEqual(masked_tile.getbbox()[0], 0)
            self.assertNotEqual(masked_tile.getbbox()[2], masked_tile.size[0])
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)

    def test_build_labeling_sample_targets_failure_modes_and_controls(self) -> None:
        frame = _build_test_frame()

        sample = labeling_sample.build_labeling_sample_from_frame(
            frame,
            scene_false_positive_count=2,
            neighbor_false_positive_count=2,
            improved_control_count=1,
            vacant_control_count=1,
            seed=42,
            zoom=19,
            fetch_images=False,
            tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
            county_cap=1,
        )

        counts = sample.groupby("sample_reason").size().to_dict()
        self.assertEqual(
            counts,
            {
                "neighbor_outside_parcel_false_positive": 2,
                "road_or_clearing_context_false_positive": 2,
                "strong_improved_reference": 1,
                "strong_vacant_reference": 1,
            },
        )
        self.assertEqual(len(sample), 6)
        self.assertEqual(sample["parcel_row_id"].nunique(), 6)

    def test_build_labeling_sample_adds_manual_review_columns_and_hints(self) -> None:
        frame = _build_test_frame()

        sample = labeling_sample.build_labeling_sample_from_frame(
            frame,
            scene_false_positive_count=1,
            neighbor_false_positive_count=1,
            improved_control_count=1,
            vacant_control_count=1,
            seed=7,
            zoom=19,
            fetch_images=False,
            tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
            county_cap=2,
        )

        self.assertTrue(set(labeling_sample.MANUAL_LABEL_COLUMNS).issubset(sample.columns))
        self.assertTrue(
            {
                "raw_centroid_tile_path",
                "masked_parcel_tile_path",
                "masked_parcel_core_crop_path",
                "masked_parcel_focus_crop_path",
                "parcel_tile_coverage_pct",
                "parcel_bbox_tile_coverage_pct",
                "full_parcel_visible_flag",
                "parcel_extent_exceeds_tile_flag",
                "parcel_tile_low_coverage_flag",
                "multi_tile_candidate_flag",
                "parcel_covering_tile_count",
                "tile_coordinates",
                "unique_tile_count",
                "duplicate_tile_flag",
                "tiles_scored_count",
                "tiles_with_building_signal_count",
                "multi_tile_inference_used_flag",
                "multi_tile_aggregation_reason",
                "best_tile_label",
                "best_tile_confidence",
                "best_tile_crop_label",
                "best_tile_probability",
                "best_tile_parcel_coverage_pct",
                "negative_tile_coverage_pct",
                "review_tile_label",
                "review_tile_rank",
                "review_tile_coordinate",
                "review_tile_image_url",
                "review_tile_image_path",
                "review_tile_manifest_path",
                "review_tile_sample_labels",
                "original_geom_type",
                "clipped_geom_type",
                "polygon_part_count",
                "clipped_polygon_part_count",
                "bounds_before_clip",
                "bounds_after_clip",
                "area_acres",
                "perimeter_meters",
                "bounding_box_width_meters",
                "bounding_box_height_meters",
                "aspect_ratio",
                "compactness",
                "is_multipart",
                "part_count",
                "geometry_quality_flag",
                "geometry_review_excluded_flag",
                "geometry_training_excluded_flag",
                "geometry_default_leads_excluded_flag",
                "geometry_marketability_base_flag",
                "geometry_marketability_flag",
                "geometry_marketability_context",
                "geometry_marketability_action",
                "geometry_marketability_default_leads_excluded_flag",
                "geometry_penalty_points",
                "geometry_penalty_reason",
                "vacancy_manual_review_eligible_flag",
                "vacancy_manual_review_exclusion_reason",
            }.issubset(sample.columns)
        )
        self.assertTrue(sample["manual_training_label"].fillna("").eq("").all())
        self.assertTrue(sample["image_url"].str.contains("World_Imagery", regex=False).all())
        self.assertTrue(sample["review_hint"].str.len().gt(0).all())
        self.assertTrue(sample["vacancy_manual_review_eligible_flag"].fillna(False).all())
        self.assertTrue(sample["vacancy_manual_review_exclusion_reason"].isna().all())

        neighbor_hint = sample.loc[
            sample["sample_reason"].eq("neighbor_outside_parcel_false_positive"),
            "review_hint",
        ].iloc[0]
        self.assertIn("nearby-building density", neighbor_hint)

    def test_build_labeling_sample_excludes_bad_geometry_by_default(self) -> None:
        frame = _build_test_frame()
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "geometry_quality_flag"] = "access_strip"
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "geometry_review_excluded_flag"] = True
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "geometry_training_excluded_flag"] = True
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "geometry_default_leads_excluded_flag"] = True
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "aspect_ratio"] = 12.0
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "compactness"] = 0.05
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "area_acres"] = 0.1
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "bounding_box_width_meters"] = 180.0
        frame.loc[frame["parcel_row_id"].eq("neighbor_0"), "bounding_box_height_meters"] = 8.0

        sample = labeling_sample.build_labeling_sample_from_frame(
            frame,
            scene_false_positive_count=2,
            neighbor_false_positive_count=1,
            improved_control_count=1,
            vacant_control_count=1,
            seed=11,
            zoom=19,
            fetch_images=False,
            tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
            county_cap=2,
        )

        self.assertNotIn("neighbor_0", sample["parcel_row_id"].astype("string").tolist())

    def test_build_labeling_sample_excludes_marketability_scrap_by_default(self) -> None:
        frame = _build_test_frame()
        frame.loc[frame["parcel_row_id"].eq("scene_0"), "geometry_marketability_flag"] = "poor_geometry"
        frame.loc[frame["parcel_row_id"].eq("scene_0"), "geometry_marketability_action"] = "exclude"
        frame.loc[frame["parcel_row_id"].eq("scene_0"), "geometry_marketability_default_leads_excluded_flag"] = True
        frame.loc[frame["parcel_row_id"].eq("scene_0"), "geometry_penalty_reason"] = "Synthetic remnant parcel."

        sample = labeling_sample.build_labeling_sample_from_frame(
            frame,
            scene_false_positive_count=2,
            neighbor_false_positive_count=1,
            improved_control_count=1,
            vacant_control_count=1,
            seed=5,
            zoom=19,
            fetch_images=False,
            tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
            county_cap=2,
        )

        self.assertNotIn("scene_0", sample["parcel_row_id"].astype("string").tolist())

    def test_neighbor_structure_only_case_remains_review_eligible_when_geometry_and_imagery_are_usable(self) -> None:
        frame = _build_test_frame()

        sample = labeling_sample.build_labeling_sample_from_frame(
            frame,
            scene_false_positive_count=1,
            neighbor_false_positive_count=2,
            improved_control_count=1,
            vacant_control_count=1,
            seed=13,
            zoom=19,
            fetch_images=False,
            tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
            county_cap=2,
        )

        neighbor_rows = sample.loc[sample["sample_reason"].eq("neighbor_outside_parcel_false_positive")].copy()
        self.assertFalse(neighbor_rows.empty)
        self.assertTrue(neighbor_rows["vacancy_manual_review_eligible_flag"].fillna(False).all())

    def test_build_labeling_sample_excludes_bad_exported_imagery(self) -> None:
        frame = _build_test_frame().iloc[[0, 1, 6, 8]].copy()

        def fake_attach(sampled: pd.DataFrame, **_: object) -> pd.DataFrame:
            result = sampled.copy()
            result["image_url"] = "https://example.com/tile.png"
            result["image_path"] = ""
            result["raw_centroid_tile_path"] = ""
            result["review_tile_label"] = "z19_x1_y1"
            result["review_tile_rank"] = 1
            result["review_tile_coordinate"] = "19/1/1"
            result["review_tile_image_url"] = "https://example.com/tile.png"
            result["review_tile_image_path"] = ""
            result["original_geom_type"] = "Polygon"
            result["clipped_geom_type"] = "Polygon"
            result["polygon_part_count"] = 1
            result["clipped_polygon_part_count"] = 1
            result["bounds_before_clip"] = "[0,0,1,1]"
            result["bounds_after_clip"] = "[0,0,1,1]"
            result["review_imagery_crop_strategy"] = "parcel_mask_multi_crop_v1"
            result["review_parcel_boundary_crop_ready_flag"] = False
            result["masked_parcel_tile_path"] = ""
            result["masked_parcel_core_crop_path"] = ""
            result["masked_parcel_focus_crop_path"] = ""
            result["review_tile_manifest_path"] = ""
            result["review_tile_sample_labels"] = ""
            result["parcel_tile_coverage_pct"] = 85.0
            result["parcel_bbox_tile_coverage_pct"] = 85.0
            result["full_parcel_visible_flag"] = True
            result["parcel_extent_exceeds_tile_flag"] = False
            result["parcel_tile_low_coverage_flag"] = False
            result["multi_tile_candidate_flag"] = False
            result["parcel_covering_tile_count"] = 1
            result["tile_coordinates"] = "[]"
            result["unique_tile_count"] = 1
            result["duplicate_tile_flag"] = False
            return result

        with mock.patch.object(labeling_sample, "attach_imagery_columns", side_effect=fake_attach):
            sample = labeling_sample.build_labeling_sample_from_frame(
                frame,
                scene_false_positive_count=1,
                neighbor_false_positive_count=1,
                improved_control_count=1,
                vacant_control_count=1,
                seed=19,
                zoom=19,
                fetch_images=True,
                tile_template=labeling_sample.DEFAULT_TILE_URL_TEMPLATE,
                county_cap=2,
            )

        self.assertEqual(len(sample), 0)


if __name__ == "__main__":
    unittest.main()
