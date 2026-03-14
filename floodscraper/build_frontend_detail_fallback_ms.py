from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.mississippi_leads_service import (  # noqa: E402
    _apply_tax_detail_defaults,
    _apply_vacancy_assessment,
    _stabilize_detail_payload,
)


APP_READY_PATH = BASE_DIR / "data" / "tax_published" / "ms" / "app_ready_mississippi_leads.parquet"
PARCEL_INDEX_ROOT = BASE_DIR / "backend" / "runtime" / "mississippi" / "parcel_index"
DETAIL_METRICS_PATH = BASE_DIR / "backend" / "runtime" / "mississippi" / "parcel_detail_metrics.parquet"
OUTPUT_PATH = BASE_DIR / "frontend" / "public" / "data" / "mississippi_lead_detail_fallback.json"

RUNTIME_COLUMNS = [
    "parcel_row_id",
    "assessed_total_value",
    "county_vacant_flag",
    "ai_building_present_flag",
    "building_present_confidence",
    "building_presence_reason",
]


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
        return pd.DataFrame(columns=RUNTIME_COLUMNS)
    dataset = ds.dataset(PARCEL_INDEX_ROOT, format="parquet")
    available_columns = [column for column in RUNTIME_COLUMNS if column in dataset.schema.names]
    if "parcel_row_id" not in available_columns:
        return pd.DataFrame(columns=RUNTIME_COLUMNS)
    table = dataset.to_table(columns=available_columns, filter=ds.field("parcel_row_id").isin(parcel_ids))
    return table.to_pandas()


def read_detail_metrics(parcel_ids: list[str]) -> pd.DataFrame:
    if not DETAIL_METRICS_PATH.exists():
        return pd.DataFrame(columns=["parcel_row_id"])
    dataset = ds.dataset(DETAIL_METRICS_PATH, format="parquet")
    table = dataset.to_table(filter=ds.field("parcel_row_id").isin(parcel_ids))
    return table.to_pandas()


def main() -> None:
    base_frame = pd.read_parquet(APP_READY_PATH, engine="pyarrow").copy()
    parcel_ids = base_frame["parcel_row_id"].astype("string").tolist()
    runtime_enrichment = read_runtime_enrichment(parcel_ids)
    detail_metrics = read_detail_metrics(parcel_ids)

    frame = base_frame.merge(runtime_enrichment, on="parcel_row_id", how="left", suffixes=("", "_runtime"))
    frame = frame.merge(detail_metrics, on="parcel_row_id", how="left", suffixes=("", "_detail"))

    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        payload = {column: to_json_scalar(value) for column, value in row.items() if column != "geometry"}
        _apply_tax_detail_defaults(payload)
        _apply_vacancy_assessment(payload)
        _stabilize_detail_payload(payload)
        if payload.get("ai_building_present_flag") is None and payload.get("building_present_confidence") is not None:
            payload["ai_building_present_flag"] = bool(float(payload["building_present_confidence"]) >= 60.0)
        payload.setdefault("ai_building_present_flag", None)
        payload.setdefault("county_vacant_flag", None)
        records.append(payload)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=True, separators=(",", ":"))

    print(f"Wrote {len(records)} detail fallback rows to {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
