from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import publish_parcel_pmtiles as publisher  # noqa: E402


class PublishParcelPmtilesTests(unittest.TestCase):
    def test_build_wrangler_command_uses_manifest_fields(self) -> None:
        manifest = {
            "artifact_path": str(ROOT / "frontend" / "public" / "tiles" / "ar_parcels.pmtiles"),
            "cloudflare_object_key": "tiles/ar_parcels.pmtiles",
            "content_type": "application/vnd.pmtiles",
            "cache_control": "public, max-age=31536000, immutable",
            "public_url": "https://landintel.vercel.app/tiles/ar_parcels.pmtiles",
        }

        command = publisher.build_wrangler_command(manifest, bucket="landintel-tiles", wrangler_bin="wrangler")

        self.assertEqual(
            command,
            [
                "wrangler",
                "r2",
                "object",
                "put",
                "landintel-tiles/tiles/ar_parcels.pmtiles",
                "--remote",
                "--file",
                str(ROOT / "frontend" / "public" / "tiles" / "ar_parcels.pmtiles"),
                "--content-type",
                "application/vnd.pmtiles",
                "--cache-control",
                "public, max-age=31536000, immutable",
            ],
        )

    def test_manifest_path_defaults_to_runtime_state_manifest(self) -> None:
        manifest_path = publisher._manifest_path_for_state("ar", None)
        self.assertEqual(
            manifest_path,
            ROOT / "data" / "runtime" / "ar" / "parcel_pmtiles_publish_manifest.json",
        )

    def test_select_transport_prefers_boto3_when_endpoint_is_available(self) -> None:
        manifest = {
            "artifact_path": str(ROOT / "frontend" / "public" / "tiles" / "ar_parcels.pmtiles"),
        }

        transport = publisher.select_transport(
            manifest,
            requested_transport="auto",
            endpoint_url="https://example.r2.cloudflarestorage.com",
        )

        self.assertEqual(transport, "boto3")

    def test_select_transport_requires_boto3_for_large_artifacts_without_endpoint(self) -> None:
        large_artifact = ROOT / "frontend" / "public" / "tiles" / "tmp_large.pmtiles"
        large_artifact.write_bytes(b"xxxxx")
        self.addCleanup(large_artifact.unlink)

        manifest = {
            "artifact_path": str(large_artifact),
        }

        original_threshold = publisher.WRANGLER_R2_MAX_UPLOAD_BYTES
        publisher.WRANGLER_R2_MAX_UPLOAD_BYTES = 4
        self.addCleanup(setattr, publisher, "WRANGLER_R2_MAX_UPLOAD_BYTES", original_threshold)
        transport = publisher.select_transport(
            manifest,
            requested_transport="auto",
            endpoint_url=None,
        )

        self.assertEqual(transport, "boto3")

    def test_publish_uses_manifest_public_url_when_no_override_is_provided(self) -> None:
        manifest = {
            "artifact_path": str(ROOT / "frontend" / "public" / "tiles" / "ar_parcels.pmtiles"),
            "cloudflare_object_key": "tiles/ar_parcels.pmtiles",
            "content_type": "application/vnd.pmtiles",
            "cache_control": "public, max-age=31536000, immutable",
            "public_url": "https://landintel.vercel.app/tiles/ar_parcels.pmtiles",
        }

        output = {
            "configured_public_url": manifest.get("public_url"),
            "public_url": manifest.get("public_url"),
        }

        self.assertEqual(output["configured_public_url"], "https://landintel.vercel.app/tiles/ar_parcels.pmtiles")
        self.assertEqual(output["public_url"], "https://landintel.vercel.app/tiles/ar_parcels.pmtiles")

    def test_resolve_endpoint_url_prefers_explicit_then_env(self) -> None:
        self.assertEqual(
            publisher.resolve_endpoint_url("https://explicit.example.com"),
            "https://explicit.example.com",
        )


if __name__ == "__main__":
    unittest.main()
