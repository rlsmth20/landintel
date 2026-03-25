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


if __name__ == "__main__":
    unittest.main()
