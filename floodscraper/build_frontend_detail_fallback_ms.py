from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from state_artifacts import load_state_artifacts
from parcel_contract_ms import (
    FRONTEND_FALLBACK_REQUIRED_FIELDS,
    FRONTEND_FALLBACK_RUNTIME_COLUMNS,
    validate_required_columns,
)


BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
DEFAULT_STATE_ARTIFACTS = load_state_artifacts("ms")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.mississippi_leads_service import (  # noqa: E402
    _apply_tax_detail_defaults,
    _apply_vacancy_assessment,
    _stabilize_detail_payload,
)

APP_READY_PATH = DEFAULT_STATE_ARTIFACTS.app_ready_path
PARCEL_INDEX_ROOT = DEFAULT_STATE_ARTIFACTS.runtime_parcel_index_root
DETAIL_METRICS_PATH = DEFAULT_STATE_ARTIFACTS.runtime_detail_metrics_path
OUTPUT_PATH = DEFAULT_STATE_ARTIFACTS.frontend_detail_fallback_path

def to_json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, bytes):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def read_runtime_enrichment(parcel_ids: list[str]) -> pd.DataFrame:
    if not PARCEL_INDEX_ROOT.exists():
        return pd.DataFrame(columns=FRONTEND_FALLBACK_RUNTIME_COLUMNS)
    dataset = ds.dataset(PARCEL_INDEX_ROOT, format="parquet")
    available_columns = [column for column in FRONTEND_FALLBACK_RUNTIME_COLUMNS if column in dataset.schema.names]
    if "parcel_row_id" not in available_columns:
        return pd.DataFrame(columns=FRONTEND_FALLBACK_RUNTIME_COLUMNS)
    table = dataset.to_table(columns=available_columns, filter=ds.field("parcel_row_id").isin(parcel_ids))
    return table.to_pandas()


def read_detail_metrics(parcel_ids: list[str]) -> pd.DataFrame:
    if not DETAIL_METRICS_PATH.exists():
        return pd.DataFrame(columns=["parcel_row_id"])
    dataset = ds.dataset(DETAIL_METRICS_PATH, format="parquet")
    available_columns = [column for column in dataset.schema.names if column != "parcel_id"]
    table = dataset.to_table(columns=available_columns, filter=ds.field("parcel_row_id").isin(parcel_ids))
    return table.to_pandas()


def build_frontend_fallback_record(row: pd.Series) -> dict[str, object]:
    return {column: to_json_scalar(value) for column, value in row.items() if column != "geometry"}


def main() -> None:
    base_frame = pd.read_parquet(APP_READY_PATH, engine="pyarrow").copy()
    parcel_ids = base_frame["parcel_row_id"].astype("string").tolist()
    runtime_enrichment = read_runtime_enrichment(parcel_ids)
    detail_metrics = read_detail_metrics(parcel_ids)

    frame = base_frame.merge(runtime_enrichment, on="parcel_row_id", how="left", suffixes=("", "_runtime"))
    frame = frame.merge(detail_metrics, on="parcel_row_id", how="left", suffixes=("", "_detail"))
    validate_required_columns(
        frame,
        required_columns=FRONTEND_FALLBACK_REQUIRED_FIELDS,
        non_null_columns=["parcel_row_id", "parcel_id"],
        context="build_frontend_detail_fallback_ms.frame",
    )

    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        payload = build_frontend_fallback_record(row)
        _apply_tax_detail_defaults(payload)
        _apply_vacancy_assessment(payload)
        _stabilize_detail_payload(payload)
        records.append(payload)
    validate_required_columns(
        pd.DataFrame(records),
        required_columns=FRONTEND_FALLBACK_REQUIRED_FIELDS,
        non_null_columns=["parcel_row_id", "parcel_id"],
        context="build_frontend_detail_fallback_ms.records",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, separators=(",", ":"))

    print(f"Wrote {len(records)} detail fallback rows to {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
