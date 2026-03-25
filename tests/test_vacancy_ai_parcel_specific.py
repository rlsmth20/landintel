from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import GeometryCollection, LineString, Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))
sys.path.insert(0, str(ROOT / "backend"))

import vacancy_ai_common as vacancy_common  # noqa: E402
from app.services import mississippi_leads_service as leads_service  # noqa: E402


def _legacy_building_confidence(raw_probability: float, driveway_signal: float, clearing_signal: float) -> float:
    return min(100.0, (raw_probability * 100.0 * 0.72) + (driveway_signal * 0.18) + (clearing_signal * 0.10))


def _blank_tile(color: tuple[int, int, int] = (194, 170, 138)) -> Image.Image:
    return Image.new("RGB", (256, 256), color)


def _rect_mask(bounds: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bounds, fill=255)
    return mask


def _neighbor_building_tile() -> tuple[Image.Image, Image.Image]:
    image = _blank_tile()
    draw = ImageDraw.Draw(image)
    parcel_mask = _rect_mask((112, 20, 144, 236))
    draw.rectangle((62, 104, 98, 138), fill=(122, 120, 116))
    draw.rectangle((98, 110, 116, 144), fill=(62, 62, 64))
    draw.rectangle((52, 118, 110, 124), fill=(146, 146, 146))
    return image, parcel_mask


def _road_clearing_tile() -> tuple[Image.Image, Image.Image]:
    image = _blank_tile()
    draw = ImageDraw.Draw(image)
    parcel_mask = _rect_mask((36, 36, 220, 220))
    draw.rectangle((36, 36, 220, 220), fill=(206, 182, 142))
    draw.rectangle((112, 36, 144, 220), fill=(128, 128, 128))
    draw.rectangle((36, 112, 220, 144), fill=(128, 128, 128))
    draw.rectangle((112, 112, 144, 144), fill=(110, 110, 110))
    return image, parcel_mask


def _strict_result(
    image: Image.Image,
    mask: Image.Image,
    *,
    raw_probability: float,
    acreage: float,
) -> dict[str, float]:
    masked = vacancy_common.apply_outside_mask(image, mask, outside_mask_fill="black", outside_mask_dim_factor=0.0)
    crop_label, crop_box = vacancy_common.parcel_aware_crop_specs(mask.getbbox(), acreage, image_size=image.size)[0]
    features = vacancy_common.extract_image_features(masked, crop_box)
    context = vacancy_common.imagery_context_signals(features)
    coverage = vacancy_common.crop_mask_coverage(mask, crop_box)
    risk = vacancy_common.imagery_false_positive_risk(
        raw_probability,
        driveway_signal=float(context["imagery_driveway_signal"]),
        clearing_signal=float(context["imagery_clearing_signal"]),
        features=features,
        crop_label=crop_label,
        parcel_coverage_ratio=coverage,
    )
    adjusted_probability = vacancy_common.calibrated_building_probability(
        raw_probability,
        false_positive_risk=risk,
        crop_label=crop_label,
        parcel_coverage_ratio=coverage,
    )
    confidence = vacancy_common.building_present_confidence_from_probability(adjusted_probability)
    return {
        "crop_label": crop_label,
        "coverage": coverage,
        "driveway_signal": float(context["imagery_driveway_signal"]),
        "clearing_signal": float(context["imagery_clearing_signal"]),
        "false_positive_risk": risk,
        "adjusted_probability": adjusted_probability,
        "confidence": confidence,
    }


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


class ParcelSpecificInferenceTests(unittest.TestCase):
    def test_parcel_tile_coverage_diagnostics_flags_extent_beyond_centroid_tile(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        west, south, east, north = vacancy_common.tile_bounds(address)
        dx = east - west
        dy = north - south
        geometry = Polygon(
            [
                (west + (dx * 0.70), south + (dy * 0.25)),
                (east + (dx * 0.35), south + (dy * 0.25)),
                (east + (dx * 0.35), south + (dy * 0.75)),
                (west + (dx * 0.70), south + (dy * 0.75)),
            ]
        )

        diagnostics = vacancy_common.parcel_tile_coverage_diagnostics(geometry.wkb, address)

        self.assertLess(diagnostics["parcel_tile_coverage_pct"], 100.0)
        self.assertLess(diagnostics["parcel_bbox_tile_coverage_pct"], 100.0)
        self.assertFalse(diagnostics["full_parcel_visible_flag"])
        self.assertTrue(diagnostics["parcel_extent_exceeds_tile_flag"])
        self.assertTrue(diagnostics["multi_tile_candidate_flag"])
        self.assertGreaterEqual(diagnostics["parcel_covering_tile_count"], 2)

    def test_ai_vacancy_status_note_mentions_limited_coverage(self) -> None:
        diagnostics = {
            "parcel_coverage_diagnostics_ready_flag": True,
            "parcel_tile_coverage_pct": 41.2,
            "parcel_bbox_tile_coverage_pct": 28.7,
            "parcel_tile_low_coverage_flag": True,
            "parcel_extent_exceeds_tile_flag": True,
        }

        note = vacancy_common.build_ai_vacancy_status_note(
            "Precomputed AI vacancy prediction is available for this parcel.",
            diagnostics,
        )

        self.assertIn("Limited centroid-tile coverage", note)
        self.assertIn("41.2%", note)
        self.assertIn("28.7%", note)

    def test_build_parcel_inference_tile_plan_uses_covering_tile_when_centroid_tile_misses(self) -> None:
        centroid_address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        east_address = vacancy_common.TileAddress(
            x=centroid_address.x + 1,
            y=centroid_address.y,
            z=centroid_address.z,
        )
        west, south, east, north = vacancy_common.tile_bounds(east_address)
        dx = east - west
        dy = north - south
        geometry = Polygon(
            [
                (west + (dx * 0.20), south + (dy * 0.25)),
                (west + (dx * 0.80), south + (dy * 0.25)),
                (west + (dx * 0.80), south + (dy * 0.75)),
                (west + (dx * 0.20), south + (dy * 0.75)),
            ]
        )

        plan = vacancy_common.build_parcel_inference_tile_plan(
            geometry.wkb,
            centroid_address,
            use_multi_tile_extent=True,
        )

        self.assertEqual(len(plan["tile_records"]), 1)
        self.assertFalse(plan["multi_tile_inference_used_flag"])
        self.assertFalse(plan["tile_records"][0]["centroid_tile_flag"])
        self.assertEqual(plan["tile_records"][0]["address"], east_address)
        self.assertEqual(plan["unique_tile_count"], 1)
        self.assertFalse(plan["duplicate_tile_flag"])
        self.assertEqual(json.loads(plan["tile_coordinates"]), ["19/{}/{}".format(east_address.x, east_address.y)])

    def test_deduplicate_tile_records_keeps_best_coverage_per_coordinate(self) -> None:
        address = vacancy_common.TileAddress(x=12, y=34, z=19)
        records = [
            {
                "address": address,
                "tile_label": vacancy_common.tile_label(address),
                "tile_coordinate": vacancy_common.tile_coordinate(address),
                "parcel_tile_coverage_ratio": 0.20,
                "parcel_bbox_tile_coverage_ratio": 0.18,
            },
            {
                "address": address,
                "tile_label": vacancy_common.tile_label(address),
                "tile_coordinate": vacancy_common.tile_coordinate(address),
                "parcel_tile_coverage_ratio": 0.35,
                "parcel_bbox_tile_coverage_ratio": 0.28,
            },
        ]

        deduped = vacancy_common.deduplicate_tile_records(records)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["tile_coordinate"], "19/12/34")
        self.assertEqual(deduped[0]["parcel_tile_coverage_ratio"], 0.35)

    def test_polygon_parts_from_shape_preserves_geometry_collection_parts(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, _, _ = _multipart_geometry_collection(address)

        polygon_parts = vacancy_common.polygon_parts_from_shape(geometry)

        self.assertEqual(geometry.geom_type, "GeometryCollection")
        self.assertEqual(len(polygon_parts), 2)

    def test_prepare_parcel_aware_image_preserves_multipart_components_in_mask_and_crops(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, north_center, south_center = _multipart_geometry_collection(address)

        prepared = vacancy_common.prepare_parcel_aware_image(
            Image.new("RGB", (256, 256), (194, 170, 138)),
            address=address,
            geometry_value=geometry.wkb,
            acreage=1.0,
        )

        mask = prepared["parcel_mask"]
        self.assertIsNotNone(mask)
        self.assertEqual(prepared["original_geom_type"], "GeometryCollection")
        self.assertIn(prepared["clipped_geom_type"], {"GeometryCollection", "MultiPolygon"})
        self.assertEqual(prepared["polygon_part_count"], 2)
        self.assertEqual(prepared["clipped_polygon_part_count"], 2)
        self.assertIsNotNone(prepared["bounds_before_clip"])
        self.assertIsNotNone(prepared["bounds_after_clip"])

        north_pixel = tuple(int(round(value)) for value in vacancy_common._tile_pixel(north_center[0], north_center[1], address))
        south_pixel = tuple(int(round(value)) for value in vacancy_common._tile_pixel(south_center[0], south_center[1], address))
        self.assertGreater(mask.getpixel(north_pixel), 0)
        self.assertGreater(mask.getpixel(south_pixel), 0)

        for crop_label, crop_box in prepared["crop_specs"]:
            self.assertIn(crop_label, {"parcel_core", "parcel_focus"})
            self.assertLessEqual(crop_box[0], north_pixel[0])
            self.assertLessEqual(crop_box[0], south_pixel[0])
            self.assertGreaterEqual(crop_box[2], north_pixel[0])
            self.assertGreaterEqual(crop_box[2], south_pixel[0])
            self.assertLessEqual(crop_box[1], north_pixel[1])
            self.assertLessEqual(crop_box[1], south_pixel[1])
            self.assertGreaterEqual(crop_box[3], north_pixel[1])
            self.assertGreaterEqual(crop_box[3], south_pixel[1])

    def test_service_tile_coverage_diagnostics_match_shared_implementation(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, _, _ = _multipart_geometry_collection(address)

        shared = vacancy_common.parcel_tile_coverage_diagnostics(geometry.wkb, address)
        service = leads_service._parcel_tile_coverage_diagnostics(
            geometry.wkb,
            tile_x=address.x,
            tile_y=address.y,
            zoom=address.z,
        )

        for key in [
            "parcel_tile_coverage_pct",
            "parcel_bbox_tile_coverage_pct",
            "full_parcel_visible_flag",
            "parcel_extent_exceeds_tile_flag",
            "parcel_tile_low_coverage_flag",
            "multi_tile_candidate_flag",
            "parcel_covering_tile_count",
            "parcel_tile_coverage_ratio",
            "parcel_bbox_tile_coverage_ratio",
        ]:
            self.assertEqual(service[key], shared[key])

    def test_service_prepare_parcel_aware_ai_image_matches_shared_prepare_for_multipart_parcel(self) -> None:
        address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, north_center, south_center = _multipart_geometry_collection(address)
        tile_image = Image.new("RGB", (256, 256), (194, 170, 138))

        shared = vacancy_common.prepare_parcel_aware_image(
            tile_image,
            address=address,
            geometry_value=geometry.wkb,
            acreage=1.0,
            use_parcel_mask=leads_service.AI_USE_PARCEL_MASK,
            outside_mask_fill=leads_service.AI_OUTSIDE_MASK_FILL,
            outside_mask_dim_factor=leads_service.AI_OUTSIDE_MASK_DIM_FACTOR,
            parcel_buffer_pixels=leads_service.AI_PARCEL_BUFFER_PIXELS,
        )
        with mock.patch.object(leads_service, "_parcel_geometry_bytes", return_value=geometry.wkb):
            (
                service_image,
                service_mask,
                service_crop_specs,
                service_ready_flag,
                service_crop_strategy,
                service_coverage,
            ) = leads_service._prepare_parcel_aware_ai_image(
                tile_image,
                parcel_row_id="row_multipart",
                tile_x=address.x,
                tile_y=address.y,
                zoom=address.z,
                acreage=1.0,
            )

        self.assertEqual(service_ready_flag, shared["parcel_boundary_crop_ready_flag"])
        self.assertEqual(service_crop_strategy, shared["imagery_crop_strategy"])
        self.assertEqual(service_crop_specs, shared["crop_specs"])
        self.assertIsNotNone(service_mask)
        self.assertEqual(service_mask.tobytes(), shared["parcel_mask"].tobytes())
        self.assertEqual(service_image.tobytes(), shared["image"].tobytes())
        for key in [
            "parcel_tile_coverage_pct",
            "parcel_bbox_tile_coverage_pct",
            "full_parcel_visible_flag",
            "parcel_extent_exceeds_tile_flag",
            "parcel_tile_low_coverage_flag",
            "multi_tile_candidate_flag",
            "parcel_covering_tile_count",
        ]:
            self.assertEqual(service_coverage[key], shared[key])

        north_pixel = tuple(int(round(value)) for value in vacancy_common._tile_pixel(north_center[0], north_center[1], address))
        south_pixel = tuple(int(round(value)) for value in vacancy_common._tile_pixel(south_center[0], south_center[1], address))
        self.assertGreater(service_mask.getpixel(north_pixel), 0)
        self.assertGreater(service_mask.getpixel(south_pixel), 0)

    def test_prepare_parcel_aware_image_for_tile_set_preserves_multipart_components_across_tiles(self) -> None:
        south_address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        geometry, north_address, north_center, south_center = _multipart_cross_tile_geometry(south_address)

        prepared = vacancy_common.prepare_parcel_aware_image_for_tile_set(
            [
                (north_address, Image.new("RGB", (256, 256), (80, 120, 180))),
                (south_address, Image.new("RGB", (256, 256), (180, 120, 80))),
            ],
            geometry_value=geometry.wkb,
            acreage=1.0,
        )

        self.assertTrue(bool(prepared.get("parcel_boundary_crop_ready_flag", False)))
        self.assertEqual(prepared["polygon_part_count"], 2)
        self.assertEqual(prepared["clipped_polygon_part_count"], 2)
        padding = int(prepared.get("canvas_padding_pixels", 0) or 0)
        self.assertEqual(prepared["image"].size, (256 + (padding * 2), 512 + (padding * 2)))

        north_pixel_local = vacancy_common._tile_pixel(north_center[0], north_center[1], north_address)
        south_pixel_local = vacancy_common._tile_pixel(south_center[0], south_center[1], south_address)
        north_pixel = (padding + int(round(north_pixel_local[0])), padding + int(round(north_pixel_local[1])))
        south_pixel = (
            padding + int(round(south_pixel_local[0])),
            padding + 256 + int(round(south_pixel_local[1])),
        )

        mask = prepared["parcel_mask"]
        self.assertGreater(mask.getpixel(north_pixel), 0)
        self.assertGreater(mask.getpixel(south_pixel), 0)
        mask_bbox = prepared["mask_bbox"]
        self.assertIsNotNone(mask_bbox)
        self.assertGreater(mask_bbox[0], 0)
        self.assertGreater(mask_bbox[1], 0)
        self.assertLess(mask_bbox[2], prepared["image"].size[0])
        self.assertLess(mask_bbox[3], prepared["image"].size[1])
        display_crop_box = prepared.get("display_crop_box")
        self.assertIsNotNone(display_crop_box)
        self.assertLessEqual(display_crop_box[0], north_pixel[0])
        self.assertLessEqual(display_crop_box[0], south_pixel[0])
        self.assertGreaterEqual(display_crop_box[2], north_pixel[0])
        self.assertGreaterEqual(display_crop_box[2], south_pixel[0])
        for _, crop_box in prepared["crop_specs"]:
            self.assertLessEqual(crop_box[0], north_pixel[0])
            self.assertLessEqual(crop_box[0], south_pixel[0])
            self.assertGreaterEqual(crop_box[2], north_pixel[0])
            self.assertGreaterEqual(crop_box[2], south_pixel[0])
            self.assertLessEqual(crop_box[1], north_pixel[1])
            self.assertLessEqual(crop_box[1], south_pixel[1])
            self.assertGreaterEqual(crop_box[3], north_pixel[1])
            self.assertGreaterEqual(crop_box[3], south_pixel[1])

    def test_aggregate_parcel_tile_predictions_promotes_strong_positive_tile(self) -> None:
        aggregation = vacancy_common.aggregate_parcel_tile_predictions(
            [
                {
                    "tile_label": "z19_x1_y1",
                    "probability": 0.18,
                    "building_present_confidence": 18.0,
                    "best_crop_label": "parcel_core",
                    "tile_parcel_coverage_ratio": 0.62,
                    "parcel_coverage_ratio": 0.91,
                    "tile_building_signal_flag": False,
                    "tile_negative_signal_flag": True,
                },
                {
                    "tile_label": "z19_x1_y2",
                    "probability": 0.91,
                    "building_present_confidence": 91.0,
                    "best_crop_label": "parcel_core",
                    "tile_parcel_coverage_ratio": 0.24,
                    "parcel_coverage_ratio": 0.88,
                    "tile_building_signal_flag": True,
                    "tile_negative_signal_flag": False,
                },
            ]
        )

        self.assertTrue(aggregation["multi_tile_inference_used_flag"])
        self.assertTrue(aggregation["ai_building_present_flag"])
        self.assertEqual(aggregation["tiles_with_building_signal_count"], 1)
        self.assertEqual(aggregation["best_tile_label"], "z19_x1_y2")
        self.assertIn("strong in-parcel building evidence", aggregation["multi_tile_aggregation_reason"])

    def test_aggregate_parcel_tile_predictions_demotes_all_negative_tiles(self) -> None:
        aggregation = vacancy_common.aggregate_parcel_tile_predictions(
            [
                {
                    "tile_label": "z19_x2_y1",
                    "probability": 0.16,
                    "building_present_confidence": 16.0,
                    "best_crop_label": "parcel_core",
                    "tile_parcel_coverage_ratio": 0.60,
                    "parcel_coverage_ratio": 0.90,
                    "tile_building_signal_flag": False,
                    "tile_negative_signal_flag": True,
                },
                {
                    "tile_label": "z19_x2_y2",
                    "probability": 0.19,
                    "building_present_confidence": 19.0,
                    "best_crop_label": "parcel_focus",
                    "tile_parcel_coverage_ratio": 0.28,
                    "parcel_coverage_ratio": 0.72,
                    "tile_building_signal_flag": False,
                    "tile_negative_signal_flag": True,
                },
            ]
        )

        self.assertTrue(aggregation["multi_tile_inference_used_flag"])
        self.assertFalse(aggregation["ai_building_present_flag"])
        self.assertLessEqual(aggregation["probability"], 0.45)
        self.assertGreaterEqual(aggregation["negative_tile_coverage_pct"], 85.0)
        self.assertIn("All sufficiently covered parcel tiles are negative", aggregation["multi_tile_aggregation_reason"])

    def test_neighbor_building_outside_parcel_is_blacked_out_and_downgraded(self) -> None:
        image, mask = _neighbor_building_tile()
        legacy_features = vacancy_common.extract_image_features(image, (48, 48, 208, 208))
        legacy_context = vacancy_common.imagery_context_signals(legacy_features)
        legacy_confidence = _legacy_building_confidence(0.65, legacy_context["imagery_driveway_signal"], legacy_context["imagery_clearing_signal"])

        masked = vacancy_common.apply_outside_mask(image, mask, outside_mask_fill="black", outside_mask_dim_factor=0.0)
        outside_sample = masked.crop((62, 104, 110, 144))
        self.assertEqual(outside_sample.getbbox(), None)

        strict = _strict_result(image, mask, raw_probability=0.65, acreage=0.4)

        self.assertGreaterEqual(legacy_confidence, 60.0)
        self.assertLess(strict["confidence"], 45.0)
        self.assertGreaterEqual(strict["false_positive_risk"], 30.0)

    def test_road_and_clearing_do_not_count_as_structure(self) -> None:
        image, mask = _road_clearing_tile()
        crop_label, crop_box = vacancy_common.parcel_aware_crop_specs(mask.getbbox(), 1.5, image_size=image.size)[0]
        masked = vacancy_common.apply_outside_mask(image, mask, outside_mask_fill="black", outside_mask_dim_factor=0.0)
        features = vacancy_common.extract_image_features(masked, crop_box)
        context = vacancy_common.imagery_context_signals(features)
        legacy_confidence = _legacy_building_confidence(0.74, context["imagery_driveway_signal"], context["imagery_clearing_signal"])
        strict = _strict_result(image, mask, raw_probability=0.74, acreage=1.5)

        self.assertEqual(crop_label, "parcel_core")
        self.assertGreaterEqual(legacy_confidence, 60.0)
        self.assertLess(strict["confidence"], 50.0)
        self.assertLess(strict["adjusted_probability"], 0.5)

    def test_tight_crop_logic_avoids_context_heavy_full_tile(self) -> None:
        narrow_mask = _rect_mask((112, 20, 144, 236))
        crop_specs = vacancy_common.parcel_aware_crop_specs(narrow_mask.getbbox(), 0.4, image_size=(256, 256))
        labels = [label for label, _ in crop_specs]
        widths = [box[2] - box[0] for _, box in crop_specs]

        self.assertEqual(labels, ["parcel_core", "parcel_focus"])
        self.assertLessEqual(max(widths), 56)

    def test_mid_confidence_ai_signal_stays_needs_review(self) -> None:
        payload = {
            "parcel_vacant_flag": True,
            "building_count": 0,
            "building_area_total": 0,
            "assessed_total_value": 15000,
            "ai_building_present_probability": 0.74,
            "building_present_confidence": 74.0,
            "ai_building_present_flag": None,
        }
        leads_service._apply_ai_detail_defaults(payload)
        leads_service._apply_parcel_improvement_classification(payload)

        self.assertFalse(payload["ai_building_present_flag"])
        self.assertEqual(payload["parcel_improvement_status"], "needs_review")

        frame = pd.DataFrame([payload])
        classified = leads_service._apply_parcel_improvement_fields_frame(frame.copy())
        self.assertEqual(classified.iloc[0]["parcel_improvement_status"], "needs_review")

    def test_on_demand_ai_skips_large_multi_tile_parcel(self) -> None:
        centroid_address = vacancy_common.centroid_tile(-90.0, 32.0, 19)
        west, south, east, north = vacancy_common.tile_bounds(centroid_address)
        dx = east - west
        dy = north - south
        geometry = Polygon(
            [
                (west + (dx * 0.70), south + (dy * 0.25)),
                (east + (dx * 0.35), south + (dy * 0.25)),
                (east + (dx * 0.35), south + (dy * 0.75)),
                (west + (dx * 0.70), south + (dy * 0.75)),
            ]
        )
        payload = {
            "parcel_row_id": "row_large",
            "parcel_vacant_flag": True,
        }
        row = pd.Series(
            {
                "parcel_row_id": "row_large",
                "longitude": -90.0,
                "latitude": 32.0,
            }
        )

        original_lookup = leads_service._parcel_geometry_bytes
        try:
            leads_service._parcel_geometry_bytes = lambda parcel_row_id: geometry.wkb if parcel_row_id == "row_large" else None
            leads_service._maybe_apply_on_demand_ai(payload, row)
        finally:
            leads_service._parcel_geometry_bytes = original_lookup

        self.assertFalse(payload["ai_vacancy_available_flag"])
        self.assertEqual(payload["ai_vacancy_source"], "unavailable")
        self.assertTrue(payload["multi_tile_candidate_flag"])
        self.assertIn("multi-tile inference", payload["ai_vacancy_status_note"])


if __name__ == "__main__":
    unittest.main()
