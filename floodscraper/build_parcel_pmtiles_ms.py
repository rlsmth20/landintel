from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow.dataset as ds
from shapely import wkb
from shapely.geometry import mapping


ROOT = Path(__file__).resolve().parents[1]
PARCEL_MASTER_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
DEFAULT_OUTPUT = ROOT / "frontend" / "public" / "tiles" / "mississippi_parcels.pmtiles"
DEFAULT_LAYER = "parcels"
DEFAULT_MIN_ZOOM = 6
DEFAULT_MAX_ZOOM = 15
EXPORT_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "wetland_flag",
    "flood_risk_score",
    "road_access_tier",
    "geometry",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Mississippi parcel PMTiles from the statewide parcel master.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--min-zoom", type=int, default=DEFAULT_MIN_ZOOM)
    parser.add_argument("--max-zoom", type=int, default=DEFAULT_MAX_ZOOM)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--keep-geojsonseq", action="store_true")
    return parser.parse_args()


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise SystemExit(f"Required binary not found on PATH: {name}")
    return binary


def iter_features(batch_size: int):
    dataset = ds.dataset(PARCEL_MASTER_PATH, format="parquet")
    scanner = dataset.scanner(columns=EXPORT_COLUMNS, batch_size=batch_size)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        for _, row in frame.iterrows():
            geometry_value = row.get("geometry")
            if not geometry_value:
                continue
            try:
                shape = wkb.loads(geometry_value)
            except Exception:
                continue
            parcel_row_id = row.get("parcel_row_id")
            if parcel_row_id is None:
                continue
            yield {
                "type": "Feature",
                "properties": {
                    "parcel_row_id": str(parcel_row_id),
                    "parcel_id": row.get("parcel_id"),
                    "county_name": row.get("county_name"),
                    "wetland_flag": None if row.get("wetland_flag") is None else bool(row.get("wetland_flag")),
                    "flood_risk_score": row.get("flood_risk_score"),
                    "road_access_tier": row.get("road_access_tier"),
                },
                "geometry": mapping(shape),
            }


def write_geojsonseq(path: Path, batch_size: int) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for feature in iter_features(batch_size):
            handle.write(json.dumps(feature, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def run(argv: list[str]) -> None:
    subprocess.run(argv, check=True)


def main() -> None:
    args = parse_args()
    if not PARCEL_MASTER_PATH.exists():
        raise SystemExit(f"Parcel master not found: {PARCEL_MASTER_PATH}")

    tippecanoe = require_binary("tippecanoe")
    pmtiles = require_binary("pmtiles")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ms-parcel-pmtiles-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        geojsonseq_path = temp_dir / "mississippi_parcels.geojsonseq"
        mbtiles_path = temp_dir / "mississippi_parcels.mbtiles"

        feature_count = write_geojsonseq(geojsonseq_path, args.batch_size)

        run(
            [
                tippecanoe,
                "--force",
                "--read-parallel",
                "--no-tile-size-limit",
                "--drop-densest-as-needed",
                "--extend-zooms-if-still-dropping",
                "--minimum-zoom",
                str(args.min_zoom),
                "--maximum-zoom",
                str(args.max_zoom),
                "--layer",
                args.layer,
                "--output",
                str(mbtiles_path),
                str(geojsonseq_path),
            ]
        )
        run([pmtiles, "convert", str(mbtiles_path), str(args.output)])

        if args.keep_geojsonseq:
            shutil.copy2(geojsonseq_path, args.output.with_suffix(".geojsonseq"))
            shutil.copy2(mbtiles_path, args.output.with_suffix(".mbtiles"))

    print(f"Wrote {feature_count} parcel features to {args.output}")


if __name__ == "__main__":
    main()
