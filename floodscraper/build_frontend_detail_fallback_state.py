from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from build_backend_parcel_runtime_arcgis import (
    DEFAULT_FRONTEND_FALLBACK_LIMIT,
    _frontend_fallback_rows,
    _load_profile,
    _meta_payload,
    _top_records,
)
from parcel_contract_ms import FRONTEND_FALLBACK_REQUIRED_FIELDS, validate_required_columns
from state_artifacts import load_state_artifacts


BASE_DIR = Path(__file__).resolve().parents[1]
REQUIRED_META_FIELDS = [
    "defaultViews",
    "fieldReadiness",
    "summary",
    "rowCount",
    "source",
    "geometryMode",
    "geometryBounds",
    "geometryViewBox",
    "geometrySimplifyTolerance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build state frontend fallback artifacts from state-aware runtime outputs.")
    parser.add_argument("--state-code", required=True)
    parser.add_argument("--frontend-fallback-limit", type=int, default=DEFAULT_FRONTEND_FALLBACK_LIMIT)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")


def _load_runtime_frame(state_code: str) -> pd.DataFrame:
    artifacts = load_state_artifacts(state_code)
    source_path = artifacts.runtime_detail_metrics_path if artifacts.runtime_detail_metrics_path.exists() else artifacts.app_ready_path
    return pd.read_parquet(source_path, engine="pyarrow").copy()


def _normalize_meta_payload(meta_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(meta_payload)
    normalized.setdefault("geometrySimplifyTolerance", None)
    for field in REQUIRED_META_FIELDS:
        normalized.setdefault(field, None)
    return normalized


def main() -> None:
    args = parse_args()
    state_code = str(args.state_code).strip().lower()
    artifacts = load_state_artifacts(state_code)
    runtime_summary = _read_json(artifacts.runtime_summary_path)
    presets_payload = _read_json(artifacts.runtime_presets_path)
    runtime_frame = _load_runtime_frame(state_code)
    profile = _load_profile(state_code, session=requests.Session(), timeout_seconds=float(args.timeout_seconds))

    meta_payload = _normalize_meta_payload(
        _meta_payload(
            app_ready=runtime_frame,
            summary_payload=runtime_summary,
            presets_payload=presets_payload,
            app_ready_path=artifacts.app_ready_path,
            profile=profile,
        )
    )
    fallback_frame = _top_records(runtime_frame, limit=int(args.frontend_fallback_limit))
    fallback_payload = _frontend_fallback_rows(fallback_frame)

    validate_required_columns(
        fallback_frame,
        required_columns=FRONTEND_FALLBACK_REQUIRED_FIELDS,
        non_null_columns=["parcel_row_id", "parcel_id"],
        context=f"build_frontend_detail_fallback_state[{state_code}].frame",
    )
    validate_required_columns(
        pd.DataFrame(fallback_payload),
        required_columns=FRONTEND_FALLBACK_REQUIRED_FIELDS,
        non_null_columns=["parcel_row_id", "parcel_id"],
        context=f"build_frontend_detail_fallback_state[{state_code}].records",
    )

    _write_json(artifacts.frontend_meta_path, meta_payload)
    _write_json(artifacts.frontend_detail_fallback_path, fallback_payload)

    print(f"Wrote {state_code.upper()} frontend meta to {artifacts.frontend_meta_path.relative_to(BASE_DIR)}")
    print(f"Wrote {state_code.upper()} frontend detail fallback to {artifacts.frontend_detail_fallback_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
