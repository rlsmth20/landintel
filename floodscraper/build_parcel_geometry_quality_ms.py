from __future__ import annotations

import argparse
import json
from pathlib import Path

from parcel_geometry_quality_ms import (
    GEOMETRY_QUALITY_ARTIFACT_PATH,
    GEOMETRY_QUALITY_SUMMARY_PATH,
    build_geometry_quality_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the statewide Mississippi parcel geometry-quality artifact.")
    parser.add_argument("--output", default=str(GEOMETRY_QUALITY_ARTIFACT_PATH))
    parser.add_argument("--summary-output", default=str(GEOMETRY_QUALITY_SUMMARY_PATH))
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-every-batches", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_geometry_quality_artifact(
        output_path=Path(args.output),
        summary_output_path=Path(args.summary_output),
        chunk_size=args.chunk_size,
        limit=args.limit,
        force=args.force,
        log_every_batches=args.log_every_batches,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
