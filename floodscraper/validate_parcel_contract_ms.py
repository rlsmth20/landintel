from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from parcel_contract_ms import (
    BACKEND_DETAIL_REQUIRED_FIELDS,
    CANONICAL_PARCEL_FIELDS_WITH_GEOMETRY,
    CANONICAL_REQUIRED_NON_NULL_FIELDS,
    DEFAULT_CONTRACT_REPORT_PATH,
    DETAIL_METRICS_REQUIRED_COLUMNS,
    REVIEW_EXPORT_REQUIRED_COLUMNS,
    TILE_DEBUG_REQUIRED_COLUMNS,
    TILE_MANIFEST_REQUIRED_FIELDS,
    frame_contract_report,
    records_contract_report,
    write_contract_report,
)
from vacancy_ai_common import PARCEL_MASTER_PATH


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_EXPORT_PATH = ROOT / "data" / "buildings_processed" / "ms_vacancy_training_review_sample_300_contract.csv"
DEFAULT_TILE_DEBUG_PATH = ROOT / "data" / "buildings_processed" / "_tmp_parcel_contract_tiles.csv"
DEFAULT_TILE_MANIFEST_PATH = ROOT / "data" / "buildings_processed" / "ai_review_tiles_ms" / "desoto" / "row_255174_tile_manifest.json"
DEFAULT_DETAIL_METRICS_PATH = ROOT / "backend" / "runtime" / "mississippi" / "parcel_detail_metrics.parquet"
DEFAULT_DETAIL_FALLBACK_PATH = ROOT / "frontend" / "public" / "data" / "mississippi_lead_detail_fallback.json"


def _load_json_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def build_contract_reports(
    *,
    review_export_path: Path,
    tile_debug_path: Path,
    tile_manifest_path: Path,
    detail_metrics_path: Path,
    detail_fallback_path: Path,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []

    canonical_frame = pd.read_parquet(PARCEL_MASTER_PATH, engine="pyarrow")
    reports.append(
        frame_contract_report(
            canonical_frame,
            label="canonical_parcel_dataset",
            required_columns=CANONICAL_PARCEL_FIELDS_WITH_GEOMETRY,
            non_null_columns=[*CANONICAL_REQUIRED_NON_NULL_FIELDS, "geometry"],
        )
    )

    review_frame = pd.read_csv(review_export_path)
    reports.append(
        frame_contract_report(
            review_frame,
            label="review_export",
            required_columns=REVIEW_EXPORT_REQUIRED_COLUMNS,
            non_null_columns=["parcel_row_id", "parcel_id"],
        )
    )

    detail_metrics_frame = pd.read_parquet(detail_metrics_path, engine="pyarrow")
    reports.append(
        frame_contract_report(
            detail_metrics_frame,
            label="detail_metrics",
            required_columns=DETAIL_METRICS_REQUIRED_COLUMNS,
            non_null_columns=DETAIL_METRICS_REQUIRED_COLUMNS,
        )
    )

    tile_debug_frame = pd.read_csv(tile_debug_path)
    reports.append(
        frame_contract_report(
            tile_debug_frame,
            label="tile_debug",
            required_columns=TILE_DEBUG_REQUIRED_COLUMNS,
            non_null_columns=["parcel_row_id", "parcel_id"],
        )
    )

    tile_manifest_records = _load_json_records(tile_manifest_path)
    reports.append(
        records_contract_report(
            tile_manifest_records,
            label="tile_manifest",
            required_fields=TILE_MANIFEST_REQUIRED_FIELDS,
            non_null_fields=["parcel_row_id", "parcel_id"],
        )
    )

    detail_records = _load_json_records(detail_fallback_path)
    reports.append(
        records_contract_report(
            detail_records,
            label="backend_detail_fallback",
            required_fields=BACKEND_DETAIL_REQUIRED_FIELDS,
            non_null_fields=["parcel_row_id", "parcel_id"],
        )
    )

    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Mississippi parcel contracts and emit schema drift summaries.")
    parser.add_argument("--review-export", default=str(DEFAULT_REVIEW_EXPORT_PATH))
    parser.add_argument("--tile-debug", default=str(DEFAULT_TILE_DEBUG_PATH))
    parser.add_argument("--tile-manifest", default=str(DEFAULT_TILE_MANIFEST_PATH))
    parser.add_argument("--detail-metrics", default=str(DEFAULT_DETAIL_METRICS_PATH))
    parser.add_argument("--detail-fallback", default=str(DEFAULT_DETAIL_FALLBACK_PATH))
    parser.add_argument("--output", default=str(DEFAULT_CONTRACT_REPORT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = build_contract_reports(
        review_export_path=Path(args.review_export),
        tile_debug_path=Path(args.tile_debug),
        tile_manifest_path=Path(args.tile_manifest),
        detail_metrics_path=Path(args.detail_metrics),
        detail_fallback_path=Path(args.detail_fallback),
    )
    output_path = Path(args.output)
    write_contract_report(output_path, reports)
    print(f"Wrote parcel contract report to {output_path}")


if __name__ == "__main__":
    main()
