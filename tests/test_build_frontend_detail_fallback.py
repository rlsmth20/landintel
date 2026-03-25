from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "floodscraper"))

import build_frontend_detail_fallback as frontend_fallback  # noqa: E402
import build_frontend_detail_fallback_ar as frontend_fallback_ar  # noqa: E402


class BuildFrontendDetailFallbackTests(unittest.TestCase):
    def test_generic_entrypoint_dispatches_arkansas_module(self) -> None:
        with mock.patch.object(sys, "argv", ["build_frontend_detail_fallback.py", "--state-code", "ar"]), mock.patch(
            "build_frontend_detail_fallback.runpy.run_module"
        ) as run_module:
            frontend_fallback.main()
        run_module.assert_called_once_with("build_frontend_detail_fallback_ar", run_name="__main__")

    def test_generic_entrypoint_dispatches_mississippi_module(self) -> None:
        with mock.patch.object(sys, "argv", ["build_frontend_detail_fallback.py", "--state-code", "ms"]), mock.patch(
            "build_frontend_detail_fallback.runpy.run_module"
        ) as run_module:
            frontend_fallback.main()
        run_module.assert_called_once_with("build_frontend_detail_fallback_ms", run_name="__main__")

    def test_arkansas_meta_normalization_adds_required_shape_fields(self) -> None:
        payload = frontend_fallback_ar._normalize_meta_payload(  # noqa: SLF001
            {
                "defaultViews": [],
                "fieldReadiness": [],
                "summary": [],
                "rowCount": 0,
                "source": "dummy",
                "geometryMode": None,
                "geometryBounds": None,
                "geometryViewBox": None,
            }
        )
        self.assertIn("geometrySimplifyTolerance", payload)
        self.assertIsNone(payload["geometrySimplifyTolerance"])


if __name__ == "__main__":
    unittest.main()
