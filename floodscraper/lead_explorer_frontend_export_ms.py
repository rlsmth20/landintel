from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from shapely import wkb
from shapely.geometry import MultiPolygon, Polygon


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"
FRONTEND_PUBLIC_DIR = BASE_DIR / "frontend" / "public" / "data"

APP_READY_PATH = TAX_PUBLISHED_DIR / "app_ready_mississippi_leads.parquet"
DEFAULT_VIEWS_PATH = TAX_PUBLISHED_DIR / "mississippi_lead_explorer_default_views.csv"
FIELD_READINESS_PATH = TAX_PUBLISHED_DIR / "mississippi_lead_explorer_field_readiness.csv"
SUMMARY_PATH = TAX_PUBLISHED_DIR / "mississippi_lead_explorer_summary.csv"

FRONTEND_DATA_PATH = FRONTEND_PUBLIC_DIR / "mississippi_lead_explorer.json"
FRONTEND_META_PATH = FRONTEND_PUBLIC_DIR / "mississippi_lead_explorer_meta.json"
FRONTEND_GEOMETRY_PATH = FRONTEND_PUBLIC_DIR / "mississippi_lead_explorer_geometries.json"
SIMPLIFY_TOLERANCE = 0.00002
VIEWBOX_WIDTH = 1000.0
VIEWBOX_HEIGHT = 700.0


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_serializable(value):
    if pd.isna(value):
        return None
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def geometry_payload(wkb_bytes: bytes | None) -> tuple[dict[str, object] | None, list[float] | None]:
    if not wkb_bytes:
        return None, None
    geometry = wkb.loads(wkb_bytes)
    centroid = geometry.centroid
    bounds = geometry.bounds
    point = {
        "type": "Point",
        "coordinates": [round(float(centroid.x), 6), round(float(centroid.y), 6)],
    }
    return point, [round(float(v), 6) for v in bounds]


def to_normalized_path(geometry, bounds: tuple[float, float, float, float]) -> str | None:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 0.000001)
    height = max(maxy - miny, 0.000001)

    def convert_ring(ring) -> str:
        coords = list(ring.coords)
        commands: list[str] = []
        for index, (x, y) in enumerate(coords):
            px = ((x - minx) / width) * VIEWBOX_WIDTH
            py = VIEWBOX_HEIGHT - ((y - miny) / height) * VIEWBOX_HEIGHT
            command = "M" if index == 0 else "L"
            commands.append(f"{command}{px:.2f},{py:.2f}")
        commands.append("Z")
        return " ".join(commands)

    simplified = geometry.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    if isinstance(simplified, Polygon):
        paths = [convert_ring(simplified.exterior)]
        paths.extend(convert_ring(ring) for ring in simplified.interiors)
        return " ".join(paths)
    if isinstance(simplified, MultiPolygon):
        parts: list[str] = []
        for polygon in simplified.geoms:
            parts.append(convert_ring(polygon.exterior))
            parts.extend(convert_ring(ring) for ring in polygon.interiors)
        return " ".join(parts)
    return None


def row_to_feature(row: pd.Series) -> dict[str, object]:
    point_geometry, geometry_bounds = geometry_payload(row.get("geometry"))
    feature: dict[str, object] = {}
    for column, value in row.items():
        if column == "geometry":
            continue
        feature[column] = to_serializable(value)
    feature["geometry"] = point_geometry
    feature["geometry_bounds"] = geometry_bounds
    feature["geometry_source"] = "parcel_centroid_from_parcel_polygon"
    return feature


def main() -> None:
    FRONTEND_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(APP_READY_PATH)
    geometries = [wkb.loads(value) if value else None for value in frame["geometry"]]
    valid_geometries = [geometry for geometry in geometries if geometry is not None]
    statewide_bounds = (
        min(geometry.bounds[0] for geometry in valid_geometries),
        min(geometry.bounds[1] for geometry in valid_geometries),
        max(geometry.bounds[2] for geometry in valid_geometries),
        max(geometry.bounds[3] for geometry in valid_geometries),
    )

    records = [row_to_feature(row) for _, row in frame.iterrows()]
    geometry_records = [
        {
            "parcel_row_id": str(frame.iloc[index]["parcel_row_id"]),
            "path": to_normalized_path(geometry, statewide_bounds) if geometry is not None else None,
        }
        for index, geometry in enumerate(geometries)
    ]

    metadata = {
        "defaultViews": load_csv_rows(DEFAULT_VIEWS_PATH),
        "fieldReadiness": load_csv_rows(FIELD_READINESS_PATH),
        "summary": load_csv_rows(SUMMARY_PATH),
        "rowCount": len(records),
        "source": str(APP_READY_PATH.relative_to(BASE_DIR)),
        "geometryMode": "simplified_polygon_top_n_plus_selected",
        "geometryBounds": [round(float(v), 6) for v in statewide_bounds],
        "geometryViewBox": [0, 0, VIEWBOX_WIDTH, VIEWBOX_HEIGHT],
        "geometrySimplifyTolerance": SIMPLIFY_TOLERANCE,
    }

    with FRONTEND_DATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, separators=(",", ":"))
    with FRONTEND_META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, separators=(",", ":"))
    with FRONTEND_GEOMETRY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(geometry_records, handle, ensure_ascii=True, separators=(",", ":"))

    print(f"Frontend data: {FRONTEND_DATA_PATH.relative_to(BASE_DIR)}")
    print(f"Frontend metadata: {FRONTEND_META_PATH.relative_to(BASE_DIR)}")
    print(f"Frontend geometries: {FRONTEND_GEOMETRY_PATH.relative_to(BASE_DIR)}")
    print(f"Rows: {len(records):,}")


if __name__ == "__main__":
    main()
