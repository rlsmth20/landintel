from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

try:
    from fastapi.testclient import TestClient
    import main as backend_main  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - depends on backend env
    TestClient = None
    backend_main = None


class CorsHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if TestClient is None or backend_main is None:
            raise unittest.SkipTest("fastapi backend dependencies are not installed in this Python environment")
        cls.client = TestClient(backend_main.app)

    def test_preflight_allows_vercel_origin_on_state_geometry_route(self) -> None:
        response = self.client.options(
            "/api/states/ar/parcels/ar_test_1/geometry",
            headers={
                "Origin": "https://landintel.vercel.app",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://landintel.vercel.app")
        self.assertIn("GET", response.headers.get("access-control-allow-methods", ""))

    def test_geometry_route_includes_cors_headers_for_allowed_origin(self) -> None:
        mock_service = mock.Mock()
        mock_service.get_parcel_geometry.return_value = {
            "geometry_mode": "selected_parcel_geojson",
            "render_mode": "none",
            "geometry_bounds": None,
            "geometry_view_box": None,
            "requested_bounds": None,
            "zoom": 14,
            "feature_count": 0,
            "feature_collection": {"type": "FeatureCollection", "features": []},
            "items": [],
        }

        with mock.patch("app.api.state_leads.get_state_service", return_value=mock_service):
            response = self.client.get(
                "/api/states/ar/parcels/ar_test_1/geometry?zoom=14",
                headers={"Origin": "https://landintel.vercel.app"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://landintel.vercel.app")


if __name__ == "__main__":
    unittest.main()
