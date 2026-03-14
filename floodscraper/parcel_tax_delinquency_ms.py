from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from parcel_owner_leads_ms import (
    COUNTY_QA_CSV as OWNER_COUNTY_QA_CSV,
    MAILER_EXPORT_CSV as OWNER_MAILER_EXPORT_CSV,
    MAILER_EXPORT_FIELDS,
    OUTPUT_GPKG as OWNER_OUTPUT_GPKG,
    OUTPUT_PARQUET as OWNER_OUTPUT_PARQUET,
    SCHEMA_CSV as OWNER_SCHEMA_CSV,
    SUMMARY_CSV as OWNER_SUMMARY_CSV,
    build_county_qa as build_owner_county_qa,
    build_mailer_target_score,
    build_schema as build_owner_schema,
    build_summary as build_owner_summary,
    write_outputs as write_owner_outputs,
)

BASE_DIR = Path(__file__).resolve().parents[1]
STATE_ABBR = "MS"
PARCELS_DIR = BASE_DIR / "data" / "parcels"
TAX_PROCESSED_DIR = BASE_DIR / "data" / "tax_processed"

MASTER_INPUT = PARCELS_DIR / "mississippi_parcels_master.parquet"
OWNER_INPUT = PARCELS_DIR / "mississippi_parcels_owner_leads.parquet"
REGISTRY_PATH = BASE_DIR / "mississippi_tax_source_registry.yaml"
OUTPUT_PARQUET = PARCELS_DIR / "mississippi_parcels_tax_distress.parquet"
OUTPUT_GPKG = PARCELS_DIR / "mississippi_parcels_tax_distress.gpkg"
SUMMARY_CSV = PARCELS_DIR / "mississippi_tax_distress_summary.csv"
PROGRESS_CSV = PARCELS_DIR / "mississippi_tax_distress_progress.csv"
COUNTY_COVERAGE_MATRIX_PARQUET = PARCELS_DIR / "mississippi_tax_coverage_matrix.parquet"
COUNTY_COVERAGE_MATRIX_CSV = PARCELS_DIR / "mississippi_tax_coverage_matrix.csv"
COUNTY_COVERAGE_QA_CSV = PARCELS_DIR / "mississippi_tax_coverage_qa.csv"
COUNTY_COVERAGE_QA_SUMMARY_JSON = PARCELS_DIR / "mississippi_tax_coverage_qa_summary.json"
COUNTY_LINKAGE_DIAGNOSTICS_CSV = PARCELS_DIR / "mississippi_tax_county_diagnostics.csv"
COUNTY_REMEDIATION_PRIORITY_CSV = PARCELS_DIR / "mississippi_tax_county_work_queue.csv"
COUNTY_PARTS_DIR = PARCELS_DIR / "ms_parcels_tax_parts"
NORMALIZED_PARTS_DIR = TAX_PROCESSED_DIR / "ms_tax_normalized_parts"

MASTER_COLUMNS = [
    "parcel_row_id", "parcel_id", "state_code", "county_name", "county_fips",
    "source_parcel_id_raw", "source_parcel_id_normalized", "geometry",
]
OWNER_COLUMNS = [
    "parcel_row_id", "owner_name_normalized", "owner_group_id", "absentee_owner_flag",
    "out_of_state_owner_flag", "corporate_owner_flag", "owner_parcel_count",
    "owner_total_acres", "mailer_target_score",
]
FINAL_COLUMNS = [
    "parcel_row_id", "parcel_id", "state_code", "county_name", "county_fips",
    "source_parcel_id_raw", "source_parcel_id_normalized", "owner_name_normalized",
    "owner_group_id", "absentee_owner_flag", "out_of_state_owner_flag",
    "corporate_owner_flag", "owner_parcel_count", "owner_total_acres",
    "county_tax_source_configured_flag", "county_tax_source_loaded_flag",
    "county_tax_source_type", "county_tax_source_name", "county_tax_source_url",
    "county_tax_source_path", "county_tax_coverage_scope", "county_tax_quality_flag",
    "county_tax_blocker_reason", "county_tax_last_successful_ingest_timestamp",
    "tax_data_available_flag",
    "delinquent_flag", "delinquent_amount", "delinquent_amount_bucket",
    "delinquent_year", "tax_sale_date", "parcel_tax_status",
    "county_tax_coverage_status", "county_tax_coverage_note",
    "county_tax_coverage_reason", "tax_data_year", "tax_data_upload_date",
    "tax_data_source", "delinquency_last_verified",
    "tax_delinquent_flag", "delinquent_year_count", "delinquent_tax_amount_total",
    "tax_sale_flag", "latest_delinquent_year", "most_severe_tax_status",
    "tax_record_count", "tax_source_name", "tax_distress_score",
    "distressed_owner_flag", "geometry",
]
SEVERITY_PRIORITY = {"unknown": 0, "current": 1, "redeemed": 2, "delinquent": 3, "tax_sale": 4, "forfeited": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Mississippi parcel tax distress layer.")
    parser.add_argument("--master-input", type=str, default=str(MASTER_INPUT))
    parser.add_argument("--owner-input", type=str, default=str(OWNER_INPUT))
    parser.add_argument("--registry", type=str, default=str(REGISTRY_PATH))
    parser.add_argument("--output-parquet", type=str, default=str(OUTPUT_PARQUET))
    parser.add_argument("--output-gpkg", type=str, default=str(OUTPUT_GPKG))
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV))
    parser.add_argument("--progress-csv", type=str, default=str(PROGRESS_CSV))
    parser.add_argument("--coverage-matrix-parquet", type=str, default=str(COUNTY_COVERAGE_MATRIX_PARQUET))
    parser.add_argument("--coverage-matrix-csv", type=str, default=str(COUNTY_COVERAGE_MATRIX_CSV))
    parser.add_argument("--coverage-qa-csv", type=str, default=str(COUNTY_COVERAGE_QA_CSV))
    parser.add_argument("--coverage-qa-summary-json", type=str, default=str(COUNTY_COVERAGE_QA_SUMMARY_JSON))
    parser.add_argument("--county-diagnostics-csv", type=str, default=str(COUNTY_LINKAGE_DIAGNOSTICS_CSV))
    parser.add_argument("--county-work-queue-csv", type=str, default=str(COUNTY_REMEDIATION_PRIORITY_CSV))
    parser.add_argument("--county-parts-dir", type=str, default=str(COUNTY_PARTS_DIR))
    parser.add_argument("--normalized-parts-dir", type=str, default=str(NORMALIZED_PARTS_DIR))
    parser.add_argument("--distressed-threshold", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


def normalize_identifier(series: pd.Series) -> pd.Series:
    out = (
        series.astype("string").fillna("").str.strip().str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", "-", regex=True).str.strip("- ")
    )
    return out.mask(out.eq(""), pd.NA).astype("string")


def parse_registry(path: Path) -> dict[str, object]:
    registry: dict[str, object] = {"state": None, "counties": {}, "pending": {}}
    if not path.exists():
        return registry
    current_county: str | None = None
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        token = line.strip()
        if indent == 0:
            current_county = None
            if token.startswith("state:"):
                registry["state"] = token.split(":", 1)[1].strip()
            current_section = token[:-1] if token in {"counties:", "pending:"} else None
            continue
        if current_section not in {"counties", "pending"}:
            continue
        if indent == 2 and token.endswith(":"):
            current_county = token[:-1].strip().lower()
            registry[current_section][current_county] = {}
            continue
        if indent >= 4 and current_county and ":" in token:
            key, value = token.split(":", 1)
            registry[current_section][current_county][key.strip()] = value.strip().strip("'\"") or None
    return registry


def read_progress(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    progress = pd.read_csv(path)
    progress["county_name"] = progress["county_name"].astype("string").str.lower()
    return progress


def source_fingerprint(entry: dict[str, object] | None, source_path: Path | None) -> str:
    if entry is None:
        return "missing_registry"
    if source_path is None:
        return "configured_without_path"
    if not source_path.exists():
        return f"missing_source::{source_path.as_posix()}"
    stat = source_path.stat()
    return f"{entry.get('source_type', 'unknown')}::{source_path.as_posix()}::{stat.st_size}::{stat.st_mtime_ns}"


def should_skip(progress: pd.DataFrame, county_name: str, fingerprint: str, part_path: Path) -> bool:
    if progress.empty or not part_path.exists():
        return False
    rows = progress.loc[progress["county_name"].eq(county_name)]
    if rows.empty:
        return False
    latest = rows.sort_values("processed_at_utc").iloc[-1]
    return str(latest.get("source_fingerprint", "")) == fingerprint


def load_source_frame(source_type: str, source_path: Path) -> pd.DataFrame:
    lowered = source_type.lower()
    if lowered == "csv":
        return pd.read_csv(source_path)
    if lowered in {"excel", "xlsx", "xls"}:
        return pd.read_excel(source_path)
    if lowered == "html":
        tables = pd.read_html(source_path)
        if not tables:
            raise ValueError("No tables found in HTML source.")
        return tables[0]
    if lowered == "parquet":
        return pd.read_parquet(source_path)
    if lowered in {"gpkg", "geopackage", "geojson"}:
        return gpd.read_file(source_path, ignore_geometry=True, engine="pyogrio")
    raise ValueError(f"Unsupported source_type: {source_type}")


def source_col(frame: pd.DataFrame, name: object) -> pd.Series:
    if not name:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    if str(name) not in frame.columns:
        raise KeyError(f"Missing source column: {name}")
    return frame[str(name)]


def normalize_tax_status(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.strip().str.upper()
    normalized = np.select(
        [
            raw.str.contains(r"FORFEIT", regex=True),
            raw.str.contains(r"REDEEM", regex=True),
            raw.str.contains(r"TAX\s*SALE|CERTIF", regex=True),
            raw.str.contains(r"DELQ|DELINQ|UNPAID|PAST\s*DUE", regex=True),
            raw.str.contains(r"CURRENT|PAID", regex=True),
        ],
        ["forfeited", "redeemed", "tax_sale", "delinquent", "current"],
        default="unknown",
    )
    return pd.Series(normalized, index=series.index, dtype="string")


def normalize_year(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def normalize_amount(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string").str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def delinquent_amount_bucket(series: pd.Series) -> pd.Series:
    amount = pd.to_numeric(series, errors="coerce")
    bucket = pd.Series(pd.NA, index=series.index, dtype="string")
    bucket.loc[amount.lt(1000)] = "<1k"
    bucket.loc[amount.ge(1000) & amount.lt(5000)] = "1k-4.99k"
    bucket.loc[amount.ge(5000) & amount.lt(25000)] = "5k-24.99k"
    bucket.loc[amount.ge(25000)] = "25k+"
    return bucket


def normalize_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    out = pd.Series(dt.dt.strftime("%Y-%m-%d"), index=series.index, dtype="string")
    return out.mask(dt.isna(), pd.NA)


def normalize_timestamp(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    out = pd.Series(dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ"), index=series.index, dtype="string")
    return out.mask(dt.isna(), pd.NA)


def build_normalized_tax_frame(
    raw_frame: pd.DataFrame,
    county_name: str,
    entry: dict[str, object],
    source_path: Path,
    processed_at: str,
) -> pd.DataFrame:
    parcel_raw = source_col(raw_frame, entry.get("parcel_key_field")).astype("string")
    status_raw = source_col(raw_frame, entry.get("status_field")).astype("string")
    year = normalize_year(source_col(raw_frame, entry.get("tax_year_field")))
    amount = normalize_amount(source_col(raw_frame, entry.get("amount_field"))) if entry.get("amount_field") else pd.Series(np.nan, index=raw_frame.index, dtype="float64")
    sale_date = normalize_date(source_col(raw_frame, entry.get("sale_date_field"))) if entry.get("sale_date_field") else pd.Series(pd.NA, index=raw_frame.index, dtype="string")
    normalized_status = normalize_tax_status(status_raw)
    tax_sale_flag = normalized_status.isin(["tax_sale", "forfeited"]) | sale_date.notna()
    return pd.DataFrame(
        {
            "state_code": pd.Series(STATE_ABBR, index=raw_frame.index, dtype="string"),
            "county_name": pd.Series(county_name, index=raw_frame.index, dtype="string"),
            "source_name": pd.Series(f"ms_tax_{county_name}", index=raw_frame.index, dtype="string"),
            "source_record_id": pd.Series([f"{county_name}_{idx}" for idx in range(len(raw_frame))], dtype="string"),
            "source_parcel_id_raw": parcel_raw,
            "source_parcel_id_normalized": normalize_identifier(parcel_raw),
            "tax_status_raw": status_raw,
            "tax_status_normalized": normalized_status,
            "tax_year": year,
            "delinquent_tax_amount": amount,
            "tax_sale_flag": pd.Series(tax_sale_flag, index=raw_frame.index, dtype="boolean"),
            "tax_sale_date": sale_date,
            "record_source_path": pd.Series(source_path.relative_to(BASE_DIR).as_posix(), index=raw_frame.index, dtype="string"),
            "record_ingest_timestamp": pd.Series(processed_at, index=raw_frame.index, dtype="string"),
        }
    )


def build_linked_tax_frame(
    linked_frame: pd.DataFrame,
    county_name: str,
    entry: dict[str, object],
    source_path: Path,
) -> pd.DataFrame:
    county_field = str(entry.get("source_county_field", "") or "").strip()
    county_value = str(entry.get("source_county_value", county_name) or county_name).strip().lower()
    if county_field and county_field not in linked_frame.columns:
        raise KeyError(f"Missing county filter field in linked source: {county_field}")
    if county_field and county_field in linked_frame.columns:
        linked_frame = linked_frame.loc[
            linked_frame[county_field].astype("string").str.strip().str.lower().eq(county_value)
        ].copy()
    parcel_row_id = pd.Series(pd.NA, index=linked_frame.index, dtype="string")
    for column in ["parcel_row_id", "parcel_row_id_master", "parcel_row_id_tax"]:
        if column in linked_frame.columns:
            parcel_row_id = parcel_row_id.fillna(linked_frame[column].astype("string"))
    tax_status_raw = linked_frame["tax_status"].astype("string") if "tax_status" in linked_frame.columns else pd.Series(pd.NA, index=linked_frame.index, dtype="string")
    normalized_status = normalize_tax_status(tax_status_raw.fillna(""))
    delinquent_flag = (
        linked_frame["delinquent_flag"].fillna(False).astype("boolean")
        if "delinquent_flag" in linked_frame.columns
        else pd.Series(False, index=linked_frame.index, dtype="boolean")
    )
    if "tax_delinquent_flag_standardized" in linked_frame.columns:
        delinquent_flag = delinquent_flag.fillna(False) | linked_frame["tax_delinquent_flag_standardized"].fillna(False).astype("boolean")
    delinquent_flag = delinquent_flag | normalized_status.isin(["delinquent", "tax_sale", "forfeited"])
    delinquent_amount = (
        normalize_amount(linked_frame["delinquent_amount"])
        if "delinquent_amount" in linked_frame.columns
        else pd.Series(np.nan, index=linked_frame.index, dtype="float64")
    )
    if "tax_balance_due" in linked_frame.columns:
        delinquent_amount = delinquent_amount.fillna(normalize_amount(linked_frame["tax_balance_due"]))
    if "tax_amount_due" in linked_frame.columns:
        delinquent_amount = delinquent_amount.fillna(normalize_amount(linked_frame["tax_amount_due"]))
    tax_year = normalize_year(linked_frame["tax_year"]) if "tax_year" in linked_frame.columns else pd.Series(pd.NA, index=linked_frame.index, dtype="Int64")
    tax_sale_flag = (
        linked_frame["tax_sale_flag"].fillna(False).astype("boolean")
        if "tax_sale_flag" in linked_frame.columns
        else pd.Series(False, index=linked_frame.index, dtype="boolean")
    )
    tax_sale_flag = tax_sale_flag | normalized_status.isin(["tax_sale", "forfeited"])
    tax_sale_date = normalize_date(linked_frame["tax_sale_date"]) if "tax_sale_date" in linked_frame.columns else pd.Series(pd.NA, index=linked_frame.index, dtype="string")
    loaded_at = normalize_timestamp(linked_frame["loaded_at"]) if "loaded_at" in linked_frame.columns else pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=linked_frame.index, dtype="string")
    source_name = (
        linked_frame["source_name"].astype("string")
        if "source_name" in linked_frame.columns
        else pd.Series(entry.get("source_name"), index=linked_frame.index, dtype="string")
    )
    source_type = (
        linked_frame["source_type"].astype("string")
        if "source_type" in linked_frame.columns
        else pd.Series(entry.get("source_type"), index=linked_frame.index, dtype="string")
    )
    return pd.DataFrame(
        {
            "parcel_row_id": parcel_row_id,
            "state_code": pd.Series(STATE_ABBR, index=linked_frame.index, dtype="string"),
            "county_name": pd.Series(county_name, index=linked_frame.index, dtype="string"),
            "source_name": source_name,
            "source_type": source_type,
            "tax_status_raw": tax_status_raw,
            "tax_status_normalized": normalized_status,
            "tax_year": tax_year,
            "delinquent_year": tax_year,
            "delinquent_flag": delinquent_flag.astype("boolean"),
            "delinquent_amount": delinquent_amount.astype("float64"),
            "delinquent_amount_bucket": delinquent_amount_bucket(delinquent_amount),
            "tax_sale_flag": tax_sale_flag.astype("boolean"),
            "tax_sale_date": tax_sale_date,
            "tax_data_year": tax_year,
            "tax_data_upload_date": loaded_at,
            "tax_data_source": source_name,
            "delinquency_last_verified": loaded_at,
            "record_source_path": pd.Series(source_path.relative_to(BASE_DIR).as_posix(), index=linked_frame.index, dtype="string"),
        }
    )


def aggregate_tax_frame(normalized: pd.DataFrame) -> pd.DataFrame:
    joinable = normalized.loc[normalized["source_parcel_id_normalized"].notna()].copy()
    if joinable.empty:
        return pd.DataFrame(columns=[
            "state_code", "county_name", "source_parcel_id_normalized", "tax_data_available_flag",
            "tax_delinquent_flag", "delinquent_year_count", "delinquent_tax_amount_total",
            "tax_sale_flag", "latest_delinquent_year", "most_severe_tax_status",
            "tax_record_count", "tax_source_name",
        ])
    joinable["severity_rank"] = joinable["tax_status_normalized"].map(SEVERITY_PRIORITY).fillna(0).astype(int)
    joinable["delinquent_like"] = joinable["tax_status_normalized"].isin(["delinquent", "tax_sale", "forfeited"])
    joinable["strict_delinquent"] = joinable["tax_status_normalized"].isin(["delinquent", "tax_sale"])
    grouped = joinable.groupby(["state_code", "county_name", "source_parcel_id_normalized"], as_index=False).agg(
        tax_data_available_flag=("source_record_id", lambda x: True),
        tax_delinquent_flag=("strict_delinquent", "max"),
        delinquent_year_count=("tax_year", lambda x: int(pd.Series(x)[joinable.loc[x.index, "strict_delinquent"]].dropna().nunique())),
        delinquent_tax_amount_total=("delinquent_tax_amount", lambda x: float(pd.Series(x)[joinable.loc[x.index, "delinquent_like"]].fillna(0.0).sum())),
        tax_sale_flag=("tax_sale_flag", "max"),
        latest_delinquent_year=("tax_year", lambda x: pd.Series(x)[joinable.loc[x.index, "delinquent_like"]].dropna().max() if pd.Series(x)[joinable.loc[x.index, "delinquent_like"]].notna().any() else pd.NA),
        tax_record_count=("source_record_id", "size"),
        max_severity_rank=("severity_rank", "max"),
        tax_source_name=("source_name", "first"),
    )
    inverse = {value: key for key, value in SEVERITY_PRIORITY.items()}
    grouped["most_severe_tax_status"] = grouped["max_severity_rank"].map(inverse).fillna("unknown").astype("string")
    grouped["tax_data_available_flag"] = grouped["tax_data_available_flag"].astype("boolean")
    grouped["tax_delinquent_flag"] = grouped["tax_delinquent_flag"].fillna(False).astype("boolean")
    grouped["tax_sale_flag"] = grouped["tax_sale_flag"].fillna(False).astype("boolean")
    grouped["delinquent_year_count"] = grouped["delinquent_year_count"].fillna(0).astype("int32")
    grouped["delinquent_tax_amount_total"] = grouped["delinquent_tax_amount_total"].fillna(0.0).round(2).astype("float64")
    grouped["latest_delinquent_year"] = pd.to_numeric(grouped["latest_delinquent_year"], errors="coerce").astype("Int64")
    grouped["tax_record_count"] = grouped["tax_record_count"].fillna(0).astype("int32")
    return grouped.drop(columns=["max_severity_rank"])


def aggregate_linked_tax_frame(linked: pd.DataFrame) -> pd.DataFrame:
    joinable = linked.loc[linked["parcel_row_id"].notna()].copy()
    if joinable.empty:
        return pd.DataFrame(
            columns=[
                "parcel_row_id",
                "state_code",
                "county_name",
                "tax_data_available_flag",
                "delinquent_flag",
                "tax_delinquent_flag",
                "delinquent_amount",
                "delinquent_amount_bucket",
                "delinquent_year",
                "delinquent_year_count",
                "delinquent_tax_amount_total",
                "tax_sale_flag",
                "tax_sale_date",
                "latest_delinquent_year",
                "most_severe_tax_status",
                "tax_record_count",
                "tax_source_name",
                "tax_data_year",
                "tax_data_upload_date",
                "tax_data_source",
                "delinquency_last_verified",
            ]
        )
    joinable["severity_rank"] = joinable["tax_status_normalized"].map(SEVERITY_PRIORITY).fillna(0).astype(int)
    delinquent_like = joinable["tax_status_normalized"].isin(["delinquent", "tax_sale", "forfeited"])
    grouped = joinable.groupby(["parcel_row_id", "state_code", "county_name"], as_index=False).agg(
        tax_data_available_flag=("parcel_row_id", lambda x: True),
        delinquent_flag=("delinquent_flag", "max"),
        delinquent_amount=("delinquent_amount", lambda x: float(pd.Series(x)[delinquent_like.loc[x.index]].fillna(0.0).sum())),
        tax_sale_flag=("tax_sale_flag", "max"),
        tax_sale_date=("tax_sale_date", "max"),
        delinquent_year=("delinquent_year", lambda x: pd.Series(x)[delinquent_like.loc[x.index]].dropna().max() if pd.Series(x)[delinquent_like.loc[x.index]].notna().any() else pd.NA),
        delinquent_year_count=("delinquent_year", lambda x: int(pd.Series(x)[delinquent_like.loc[x.index]].dropna().nunique())),
        latest_delinquent_year=("delinquent_year", lambda x: pd.Series(x)[delinquent_like.loc[x.index]].dropna().max() if pd.Series(x)[delinquent_like.loc[x.index]].notna().any() else pd.NA),
        tax_record_count=("parcel_row_id", "size"),
        max_severity_rank=("severity_rank", "max"),
        tax_source_name=("source_name", "first"),
        tax_data_year=("tax_data_year", "max"),
        tax_data_upload_date=("tax_data_upload_date", "max"),
        tax_data_source=("tax_data_source", "first"),
        delinquency_last_verified=("delinquency_last_verified", "max"),
    )
    inverse = {value: key for key, value in SEVERITY_PRIORITY.items()}
    grouped["most_severe_tax_status"] = grouped["max_severity_rank"].map(inverse).fillna("unknown").astype("string")
    grouped["tax_data_available_flag"] = grouped["tax_data_available_flag"].astype("boolean")
    grouped["delinquent_flag"] = grouped["delinquent_flag"].fillna(False).astype("boolean")
    grouped["tax_delinquent_flag"] = grouped["delinquent_flag"].copy()
    grouped["tax_sale_flag"] = grouped["tax_sale_flag"].fillna(False).astype("boolean")
    grouped["delinquent_amount"] = pd.to_numeric(grouped["delinquent_amount"], errors="coerce").round(2).astype("float64")
    grouped["delinquent_amount_bucket"] = delinquent_amount_bucket(grouped["delinquent_amount"])
    grouped["delinquent_tax_amount_total"] = grouped["delinquent_amount"].copy()
    grouped["delinquent_year"] = pd.to_numeric(grouped["delinquent_year"], errors="coerce").astype("Int64")
    grouped["delinquent_year_count"] = grouped["delinquent_year_count"].fillna(0).astype("int32")
    grouped["latest_delinquent_year"] = pd.to_numeric(grouped["latest_delinquent_year"], errors="coerce").astype("Int64")
    grouped["tax_record_count"] = grouped["tax_record_count"].fillna(0).astype("int32")
    return grouped.drop(columns=["max_severity_rank"])


def build_tax_distress_score(df: pd.DataFrame) -> pd.Series:
    score = (
        df["tax_delinquent_flag"].fillna(False).astype(bool).astype(int) * 2
        + df["tax_sale_flag"].fillna(False).astype(bool).astype(int) * 4
        + df["most_severe_tax_status"].astype("string").eq("forfeited").astype(int) * 5
        + pd.to_numeric(df["delinquent_tax_amount_total"], errors="coerce").fillna(0.0).gt(500.0).astype(int)
        + pd.to_numeric(df["delinquent_tax_amount_total"], errors="coerce").fillna(0.0).gt(2000.0).astype(int)
        + df["absentee_owner_flag"].fillna(False).astype(bool).astype(int)
        + df["out_of_state_owner_flag"].fillna(False).astype(bool).astype(int)
        + pd.to_numeric(df["owner_parcel_count"], errors="coerce").fillna(1).eq(1).astype(int)
    )
    score = np.where(df["tax_data_available_flag"].fillna(False).astype(bool), score, 0)
    return pd.Series(np.clip(score, 0, 10), index=df.index, dtype="float64")


def prepare_county_output(
    county_frame: gpd.GeoDataFrame,
    aggregated: pd.DataFrame | None,
    entry: dict[str, object] | None,
    source_path: Path | None,
    source_loaded: bool,
    distressed_threshold: float,
) -> gpd.GeoDataFrame:
    out = county_frame.copy()
    out["county_tax_source_configured_flag"] = pd.Series(entry is not None, index=out.index, dtype="boolean")
    out["county_tax_source_loaded_flag"] = pd.Series(source_loaded, index=out.index, dtype="boolean")
    out["county_tax_source_type"] = pd.Series(entry.get("source_type") if entry else pd.NA, index=out.index, dtype="string")
    out["county_tax_source_name"] = pd.Series(entry.get("source_name") if entry else pd.NA, index=out.index, dtype="string")
    out["county_tax_source_url"] = pd.Series(entry.get("source_url") if entry else pd.NA, index=out.index, dtype="string")
    out["county_tax_source_path"] = pd.Series(source_path.relative_to(BASE_DIR).as_posix() if source_path else pd.NA, index=out.index, dtype="string")
    out["county_tax_coverage_scope"] = pd.Series(entry.get("coverage_scope", "full") if entry else pd.NA, index=out.index, dtype="string")
    out["county_tax_quality_flag"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["county_tax_blocker_reason"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["county_tax_last_successful_ingest_timestamp"] = pd.Series(pd.NA, index=out.index, dtype="string")
    if aggregated is not None and not aggregated.empty:
        if "parcel_row_id" in aggregated.columns:
            out = out.merge(aggregated, on=["parcel_row_id", "state_code", "county_name"], how="left")
        else:
            out = out.merge(aggregated, on=["state_code", "county_name", "source_parcel_id_normalized"], how="left")
    defaults = {
        "tax_data_available_flag": pd.Series(False, index=out.index, dtype="boolean"),
        "delinquent_flag": pd.Series(False, index=out.index, dtype="boolean"),
        "tax_delinquent_flag": pd.Series(False, index=out.index, dtype="boolean"),
        "delinquent_amount": pd.Series(np.nan, index=out.index, dtype="float64"),
        "delinquent_amount_bucket": pd.Series(pd.NA, index=out.index, dtype="string"),
        "delinquent_year": pd.Series(pd.NA, index=out.index, dtype="Int64"),
        "delinquent_year_count": pd.Series(0, index=out.index, dtype="int32"),
        "delinquent_tax_amount_total": pd.Series(0.0, index=out.index, dtype="float64"),
        "tax_sale_flag": pd.Series(False, index=out.index, dtype="boolean"),
        "tax_sale_date": pd.Series(pd.NA, index=out.index, dtype="string"),
        "latest_delinquent_year": pd.Series(pd.NA, index=out.index, dtype="Int64"),
        "most_severe_tax_status": pd.Series("unknown", index=out.index, dtype="string"),
        "tax_record_count": pd.Series(0, index=out.index, dtype="int32"),
        "tax_source_name": pd.Series(pd.NA, index=out.index, dtype="string"),
        "tax_data_year": pd.Series(pd.NA, index=out.index, dtype="Int64"),
        "tax_data_upload_date": pd.Series(pd.NA, index=out.index, dtype="string"),
        "tax_data_source": pd.Series(pd.NA, index=out.index, dtype="string"),
        "delinquency_last_verified": pd.Series(pd.NA, index=out.index, dtype="string"),
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
    out["tax_data_available_flag"] = out["tax_data_available_flag"].fillna(False).astype("boolean")
    out["delinquent_flag"] = out["delinquent_flag"].fillna(out["tax_delinquent_flag"]).fillna(False).astype("boolean")
    out["tax_delinquent_flag"] = out["tax_delinquent_flag"].fillna(out["delinquent_flag"]).fillna(False).astype("boolean")
    out["delinquent_amount"] = pd.to_numeric(out["delinquent_amount"], errors="coerce").astype("float64")
    out["delinquent_amount_bucket"] = out["delinquent_amount_bucket"].astype("string")
    out["delinquent_year"] = pd.to_numeric(out["delinquent_year"], errors="coerce").astype("Int64")
    out["delinquent_year_count"] = pd.to_numeric(out["delinquent_year_count"], errors="coerce").fillna(0).astype("int32")
    out["delinquent_tax_amount_total"] = pd.to_numeric(out["delinquent_tax_amount_total"], errors="coerce").fillna(0.0).round(2).astype("float64")
    out["tax_sale_flag"] = out["tax_sale_flag"].fillna(False).astype("boolean")
    out["tax_sale_date"] = out["tax_sale_date"].astype("string")
    out["latest_delinquent_year"] = pd.to_numeric(out["latest_delinquent_year"], errors="coerce").astype("Int64")
    out["most_severe_tax_status"] = out["most_severe_tax_status"].fillna("unknown").astype("string")
    out["tax_record_count"] = pd.to_numeric(out["tax_record_count"], errors="coerce").fillna(0).astype("int32")
    out["tax_source_name"] = out["tax_source_name"].astype("string")
    out["tax_data_year"] = pd.to_numeric(out["tax_data_year"], errors="coerce").astype("Int64")
    out["tax_data_upload_date"] = normalize_timestamp(out["tax_data_upload_date"])
    out["tax_data_source"] = out["tax_data_source"].astype("string").fillna(out["tax_source_name"])
    out["delinquency_last_verified"] = normalize_timestamp(out["delinquency_last_verified"]).fillna(out["tax_data_upload_date"])
    coverage_scope = str(entry.get("coverage_scope", "full")).strip().lower() if entry else "missing"
    county_has_data = bool(out["tax_data_available_flag"].fillna(False).any())
    county_latest_upload = pd.to_datetime(out["tax_data_upload_date"], errors="coerce", utc=True).max()
    county_latest_year = pd.to_numeric(out["tax_data_year"], errors="coerce").max()
    current_timestamp = pd.Timestamp.now("UTC")
    stale = bool(pd.notna(county_latest_upload) and county_latest_upload < (current_timestamp - pd.Timedelta(days=365)))
    if (not stale) and pd.notna(county_latest_year):
        stale = bool(float(county_latest_year) < (current_timestamp.year - 1))
    coverage_status = "missing"
    coverage_note = "No county tax delinquency dataset is configured yet."
    if entry is not None and not source_loaded:
        coverage_note = "County tax source is configured but has not been loaded yet."
    elif source_loaded and not county_has_data:
        coverage_status = "partial"
        if coverage_scope == "partial":
            coverage_note = "Partial source loaded, but no county-linked delinquency or tax-sale records were produced yet."
        else:
            coverage_note = "County tax source loaded, but no linked delinquency records were produced yet."
    elif county_has_data:
        if coverage_scope == "partial":
            coverage_status = "stale" if stale else "partial"
            coverage_note = "Partial county tax or tax-sale coverage exists but appears stale." if stale else "Partial county tax or tax-sale coverage is available."
        else:
            coverage_status = "stale" if stale else "available"
            coverage_note = "County tax delinquency coverage exists but appears stale." if stale else "County tax delinquency coverage is available."
    out["county_tax_coverage_status"] = pd.Series(coverage_status, index=out.index, dtype="string")
    out["county_tax_coverage_note"] = pd.Series(coverage_note, index=out.index, dtype="string")
    out["county_tax_coverage_reason"] = out["county_tax_coverage_note"].copy()
    out["parcel_tax_status"] = pd.Series("county coverage missing", index=out.index, dtype="string")
    out.loc[out["county_tax_coverage_status"].eq("stale"), "parcel_tax_status"] = "county data stale"
    out.loc[out["county_tax_coverage_status"].eq("partial"), "parcel_tax_status"] = "county coverage partial"
    out.loc[out["county_tax_coverage_status"].eq("available"), "parcel_tax_status"] = "not delinquent"
    out.loc[out["delinquent_flag"].fillna(False), "parcel_tax_status"] = "delinquent"
    out["tax_distress_score"] = build_tax_distress_score(out)
    out["distressed_owner_flag"] = (
        out["tax_delinquent_flag"].fillna(False).astype(bool)
        | out["tax_distress_score"].ge(distressed_threshold)
    ).astype("boolean")
    return out.loc[:, FINAL_COLUMNS].copy()


def build_county_coverage_matrix(
    county_names: list[str],
    registry: dict[str, object],
    progress: pd.DataFrame,
    final_frame: pd.DataFrame,
) -> pd.DataFrame:
    county_entries = registry.get("counties", {}) if isinstance(registry.get("counties"), dict) else {}
    pending_entries = registry.get("pending", {}) if isinstance(registry.get("pending"), dict) else {}
    county_stats = (
        final_frame.groupby("county_name", dropna=False)
        .agg(
            parcel_count=("parcel_row_id", "size"),
            parcel_match_count=("tax_data_available_flag", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
            matched_row_count=("tax_record_count", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(0).sum())),
            tax_data_available_flag=("tax_data_available_flag", lambda x: bool(pd.Series(x).fillna(False).astype(bool).any())),
            tax_data_year=("tax_data_year", "max"),
            tax_data_upload_date=("tax_data_upload_date", "max"),
            delinquency_last_verified=("delinquency_last_verified", "max"),
        )
        .reset_index()
    )
    latest_progress = (
        progress.sort_values(["county_name", "processed_at_utc"])
        .drop_duplicates(subset=["county_name"], keep="last")
        .set_index("county_name")
        if not progress.empty
        else pd.DataFrame().set_index(pd.Index([], name="county_name"))
    )

    records: list[dict[str, object]] = []
    now = pd.Timestamp.now("UTC")
    for county_name in county_names:
        entry = county_entries.get(county_name)
        pending_entry = pending_entries.get(county_name)
        progress_row = latest_progress.loc[county_name] if county_name in latest_progress.index else None
        stat_row = county_stats.loc[county_stats["county_name"].eq(county_name)]
        if stat_row.empty:
            parcel_count = 0
            parcel_match_count = 0
            matched_row_count = 0
            tax_data_available_flag = False
            tax_data_year = pd.NA
            tax_data_upload_date = pd.NA
            delinquency_last_verified = pd.NA
        else:
            record = stat_row.iloc[0]
            parcel_count = int(record["parcel_count"])
            parcel_match_count = int(record["parcel_match_count"])
            matched_row_count = int(record["matched_row_count"])
            tax_data_available_flag = bool(record["tax_data_available_flag"])
            tax_data_year = record["tax_data_year"]
            tax_data_upload_date = record["tax_data_upload_date"]
            delinquency_last_verified = record["delinquency_last_verified"]

        progress_status = str(progress_row["status"]) if progress_row is not None and "status" in progress_row else ""
        source_loaded = progress_status == "processed"
        latest_upload_ts = pd.to_datetime(pd.Series([tax_data_upload_date]), errors="coerce", utc=True).iloc[0]
        latest_year_num = pd.to_numeric(pd.Series([tax_data_year]), errors="coerce").iloc[0]
        stale = False
        if pd.notna(latest_upload_ts):
            stale = bool(latest_upload_ts < (now - pd.Timedelta(days=365)))
        if (not stale) and pd.notna(latest_year_num):
            stale = bool(float(latest_year_num) < (now.year - 1))

        coverage_scope = None
        ingest_mode = None
        source_type = None
        source_name = None
        source_url = None
        source_county_field = None
        source_county_value = None
        blocker_reason = None
        quality_flag = None
        last_successful_ingest_timestamp = None
        discovery_status = None

        if pending_entry:
            coverage_status = "pending"
            coverage_scope = pending_entry.get("coverage_scope") or "pending"
            ingest_mode = pending_entry.get("ingest_mode")
            source_type = pending_entry.get("source_type")
            source_name = pending_entry.get("source_name")
            source_url = pending_entry.get("source_url")
            source_county_field = pending_entry.get("source_county_field")
            source_county_value = pending_entry.get("source_county_value")
            discovery_status = pending_entry.get("discovery_status")
            blocker_reason = pending_entry.get("note") or "County tax source has been discovered but is not yet ingestable."
            quality_flag = discovery_status or "pending_source"
        elif entry is None:
            coverage_status = "unavailable"
            coverage_scope = "unavailable"
            blocker_reason = "No county tax source is currently configured."
            quality_flag = "unavailable_no_source"
        else:
            coverage_scope = entry.get("coverage_scope") or "full"
            ingest_mode = entry.get("ingest_mode")
            source_type = entry.get("source_type")
            source_name = entry.get("source_name")
            source_url = entry.get("source_url")
            source_county_field = entry.get("source_county_field")
            source_county_value = entry.get("source_county_value")
            if source_loaded:
                last_successful_ingest_timestamp = progress_row.get("processed_at_utc") if progress_row is not None else None
                if stale and (parcel_match_count > 0 or str(coverage_scope).lower() == "partial"):
                    coverage_status = "stale"
                elif str(coverage_scope).lower() == "partial":
                    coverage_status = "partial"
                elif parcel_match_count > 0:
                    coverage_status = "covered"
                else:
                    coverage_status = "partial"
                    blocker_reason = "County tax source loaded, but no linked parcel matches were produced."
                quality_flag = "county_specific_loaded"
                if str(coverage_scope).lower() == "partial":
                    quality_flag = "shared_partial_loaded"
                if coverage_status == "stale":
                    quality_flag = "stale_source_review"
                elif parcel_match_count == 0:
                    quality_flag = "zero_match_review"
            else:
                coverage_status = "unavailable"
                blocker_reason = (
                    progress_row.get("notes")
                    if progress_row is not None and "notes" in progress_row
                    else "County tax source is configured but not currently ingestable."
                )
                quality_flag = "unavailable_source_not_loaded"

        if quality_flag == "county_specific_loaded" and matched_row_count > parcel_match_count and parcel_match_count > 0:
            quality_flag = "duplicate_match_review"

        records.append(
            {
                "county_name": county_name,
                "county_slug": county_name,
                "coverage_status": coverage_status,
                "coverage_scope": coverage_scope,
                "ingest_mode": ingest_mode,
                "source_type": source_type,
                "source_name": source_name,
                "source_url": source_url,
                "source_county_field": source_county_field,
                "source_county_value": source_county_value,
                "discovery_status": discovery_status,
                "county_tax_source_configured_flag": bool(entry is not None),
                "county_tax_source_loaded_flag": bool(source_loaded),
                "tax_data_available_flag": bool(tax_data_available_flag),
                "delinquency_last_verified": delinquency_last_verified,
                "tax_data_upload_date": tax_data_upload_date,
                "tax_data_year": tax_data_year,
                "last_successful_ingest_timestamp": last_successful_ingest_timestamp,
                "parcel_count": parcel_count,
                "parcel_match_count": parcel_match_count,
                "matched_row_count": matched_row_count,
                "quality_flag": quality_flag,
                "blocker_reason": blocker_reason,
            }
        )
    return pd.DataFrame.from_records(records).sort_values("county_name").reset_index(drop=True)


def apply_county_coverage_matrix(final_frame: gpd.GeoDataFrame, coverage_matrix: pd.DataFrame) -> gpd.GeoDataFrame:
    matrix = coverage_matrix.rename(
        columns={
            "coverage_scope": "county_tax_coverage_scope",
            "source_name": "county_tax_source_name",
            "source_url": "county_tax_source_url",
            "quality_flag": "county_tax_quality_flag",
            "blocker_reason": "county_tax_blocker_reason",
            "last_successful_ingest_timestamp": "county_tax_last_successful_ingest_timestamp",
        }
    )
    merged = final_frame.merge(
        matrix[
            [
                "county_name",
                "coverage_status",
                "county_tax_coverage_scope",
                "county_tax_source_configured_flag",
                "county_tax_source_loaded_flag",
                "county_tax_source_name",
                "county_tax_source_url",
                "county_tax_quality_flag",
                "county_tax_blocker_reason",
                "county_tax_last_successful_ingest_timestamp",
                "tax_data_upload_date",
                "tax_data_year",
                "delinquency_last_verified",
            ]
        ],
        on="county_name",
        how="left",
        suffixes=("", "_matrix"),
    )
    matrix_status = merged["coverage_status"].astype("string")
    row_status = matrix_status.replace({"covered": "available"}).astype("string")
    row_note = pd.Series("No county tax delinquency dataset is available yet.", index=merged.index, dtype="string")
    row_note.loc[matrix_status.eq("covered")] = "County tax delinquency coverage is available."
    row_note.loc[matrix_status.eq("partial")] = "Only partial county tax delinquency coverage is currently available."
    row_note.loc[matrix_status.eq("stale")] = "County tax delinquency coverage exists but appears stale."
    row_note.loc[matrix_status.eq("pending")] = "County tax source has been discovered but is pending ingest implementation."
    row_note.loc[matrix_status.eq("unavailable")] = "No usable county tax source is currently available."
    row_note = row_note.where(merged["county_tax_blocker_reason"].isna(), merged["county_tax_blocker_reason"].astype("string"))

    parcel_tax_status = pd.Series("county coverage unavailable", index=merged.index, dtype="string")
    parcel_tax_status.loc[matrix_status.eq("stale")] = "county data stale"
    parcel_tax_status.loc[matrix_status.eq("partial")] = "county coverage partial"
    parcel_tax_status.loc[matrix_status.eq("pending")] = "county source pending"
    parcel_tax_status.loc[matrix_status.eq("covered")] = "not delinquent"
    parcel_tax_status.loc[merged["delinquent_flag"].fillna(False)] = "delinquent"

    merged["county_tax_source_configured_flag"] = (
        merged["county_tax_source_configured_flag_matrix"].fillna(merged["county_tax_source_configured_flag"]).fillna(False).astype("boolean")
    )
    merged["county_tax_source_loaded_flag"] = (
        merged["county_tax_source_loaded_flag_matrix"].fillna(merged["county_tax_source_loaded_flag"]).fillna(False).astype("boolean")
    )
    merged["county_tax_source_name"] = merged["county_tax_source_name"].astype("string").fillna(merged["county_tax_source_name_matrix"].astype("string"))
    merged["county_tax_source_url"] = merged["county_tax_source_url"].astype("string").fillna(merged["county_tax_source_url_matrix"].astype("string"))
    merged["county_tax_coverage_scope"] = merged["county_tax_coverage_scope"].astype("string").fillna(merged["county_tax_coverage_scope_matrix"].astype("string"))
    merged["county_tax_quality_flag"] = merged["county_tax_quality_flag"].astype("string").fillna(merged["county_tax_quality_flag_matrix"].astype("string"))
    merged["county_tax_blocker_reason"] = merged["county_tax_blocker_reason"].astype("string").fillna(merged["county_tax_blocker_reason_matrix"].astype("string"))
    merged["county_tax_last_successful_ingest_timestamp"] = normalize_timestamp(merged["county_tax_last_successful_ingest_timestamp"]).fillna(
        normalize_timestamp(merged["county_tax_last_successful_ingest_timestamp_matrix"])
    )
    merged["county_tax_coverage_status"] = row_status
    merged["county_tax_coverage_note"] = row_note
    merged["county_tax_coverage_reason"] = row_note
    merged["parcel_tax_status"] = parcel_tax_status
    merged["tax_data_upload_date"] = normalize_timestamp(merged["tax_data_upload_date"]).fillna(normalize_timestamp(merged["tax_data_upload_date_matrix"]))
    merged["tax_data_year"] = pd.to_numeric(merged["tax_data_year"], errors="coerce").astype("Int64").fillna(
        pd.to_numeric(merged["tax_data_year_matrix"], errors="coerce").astype("Int64")
    )
    merged["delinquency_last_verified"] = normalize_timestamp(merged["delinquency_last_verified"]).fillna(
        normalize_timestamp(merged["delinquency_last_verified_matrix"])
    ).fillna(merged["county_tax_last_successful_ingest_timestamp"])
    merged["tax_data_source"] = merged["tax_data_source"].astype("string").fillna(merged["county_tax_source_name"])
    return gpd.GeoDataFrame(
        merged.drop(
            columns=[
                "coverage_status",
                "county_tax_source_configured_flag_matrix",
                "county_tax_source_loaded_flag_matrix",
                "county_tax_source_name_matrix",
                "county_tax_source_url_matrix",
                "county_tax_coverage_scope_matrix",
                "county_tax_quality_flag_matrix",
                "county_tax_blocker_reason_matrix",
                "county_tax_last_successful_ingest_timestamp_matrix",
                "tax_data_upload_date_matrix",
                "tax_data_year_matrix",
                "delinquency_last_verified_matrix",
            ],
            errors="ignore",
        ),
        geometry="geometry",
        crs=final_frame.crs,
    )


def build_county_coverage_qa(
    coverage_matrix: pd.DataFrame,
    progress: pd.DataFrame,
    final_frame: pd.DataFrame,
    normalized_parts_dir: Path,
) -> pd.DataFrame:
    issue_rows: list[dict[str, object]] = []
    matched = final_frame.loc[final_frame["tax_data_available_flag"].fillna(False).astype(bool)].copy()
    null_check_columns = [
        "delinquent_amount",
        "delinquent_year",
        "tax_data_upload_date",
        "tax_data_source",
        "delinquency_last_verified",
    ]
    latest_progress = (
        progress.sort_values(["county_name", "processed_at_utc"])
        .drop_duplicates(subset=["county_name"], keep="last")
        .set_index("county_name")
        if not progress.empty
        else pd.DataFrame().set_index(pd.Index([], name="county_name"))
    )
    for record in coverage_matrix.to_dict(orient="records"):
        county = str(record["county_name"])
        parcel_match_count = int(record.get("parcel_match_count") or 0)
        matched_row_count = int(record.get("matched_row_count") or 0)
        coverage_status = str(record.get("coverage_status") or "")
        source_type = str(record.get("source_type") or "")
        source_county_field = str(record.get("source_county_field") or "")

        part_path = normalized_parts_dir / f"{county}.parquet"
        duplicate_count = 0
        if part_path.exists():
            try:
                part = pd.read_parquet(part_path, columns=["parcel_row_id"], engine="pyarrow")
                duplicate_count = int(part["parcel_row_id"].dropna().duplicated().sum())
            except Exception:
                duplicate_count = 0
        if duplicate_count > 0:
            issue_rows.append({"county_name": county, "issue_type": "duplicate_parcel_matches", "severity": "warning", "observed_value": duplicate_count, "details": "Normalized tax records contain repeated parcel_row_id values."})

        if bool(record.get("county_tax_source_loaded_flag")) and parcel_match_count == 0:
            issue_rows.append({"county_name": county, "issue_type": "zero_match_count", "severity": "warning", "observed_value": parcel_match_count, "details": "County source loaded, but no linked parcel matches were produced."})
        elif bool(record.get("county_tax_source_loaded_flag")) and parcel_match_count < 5 and int(record.get("parcel_count") or 0) > 10000:
            issue_rows.append({"county_name": county, "issue_type": "suspiciously_low_match_count", "severity": "warning", "observed_value": parcel_match_count, "details": "County source loaded, but matched parcel count is unusually low for county size."})

        if source_type == "statewide_public_inventory" and not source_county_field:
            issue_rows.append({"county_name": county, "issue_type": "county_bleed_through_risk", "severity": "error", "observed_value": 1, "details": "Shared statewide source is missing a county filter field."})

        county_matched = matched.loc[matched["county_name"].astype("string").str.lower().eq(county)]
        if not county_matched.empty:
            for column in null_check_columns:
                null_rate = float(county_matched[column].isna().mean())
                if null_rate > 0.0:
                    issue_rows.append({"county_name": county, "issue_type": f"null_rate::{column}", "severity": "info" if null_rate < 0.2 else "warning", "observed_value": round(null_rate, 4), "details": f"Matched parcel null rate for {column}."})

        inconsistent = (
            (coverage_status == "covered" and parcel_match_count == 0)
            or (coverage_status == "pending" and parcel_match_count > 0)
            or (coverage_status == "unavailable" and parcel_match_count > 0)
        )
        if inconsistent:
            issue_rows.append({"county_name": county, "issue_type": "coverage_status_inconsistent", "severity": "error", "observed_value": coverage_status, "details": "Coverage status does not agree with observed matched parcel counts."})

        if county in latest_progress.index:
            progress_status = str(latest_progress.loc[county, "status"])
            if coverage_status in {"covered", "partial", "stale"} and progress_status != "processed":
                issue_rows.append({"county_name": county, "issue_type": "progress_status_inconsistent", "severity": "error", "observed_value": progress_status, "details": "County coverage indicates ingest, but progress status is not processed."})
    return pd.DataFrame.from_records(issue_rows)


def _diagnostic_identifier_series(frame: pd.DataFrame) -> pd.Series:
    for column in [
        "source_parcel_id_normalized",
        "source_parcel_id_normalized_master",
        "parcel_id_normalized",
        "source_parcel_number",
        "source_parcel_id_raw",
        "parcel_id_raw",
        "source_ppin",
        "account_id",
        "parcel_row_id",
    ]:
        if column in frame.columns:
            series = frame[column].astype("string")
            if column not in {"source_parcel_id_normalized", "parcel_id_normalized", "parcel_row_id"}:
                series = normalize_identifier(series)
            if series.dropna().empty:
                continue
            return series
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def _load_source_frame_for_diagnostics(
    cache: dict[str, pd.DataFrame],
    source_path: Path,
    ingest_mode: str | None,
    source_type: str | None,
) -> pd.DataFrame:
    cache_key = source_path.as_posix()
    if cache_key not in cache:
        cache[cache_key] = load_source_frame("parquet" if str(ingest_mode or "").strip().lower() == "linked_parquet" else str(source_type or ""), source_path)
    return cache[cache_key]


def build_county_linkage_diagnostics(
    county_names: list[str],
    registry: dict[str, object],
    coverage_matrix: pd.DataFrame,
    progress: pd.DataFrame,
    coverage_qa: pd.DataFrame,
) -> pd.DataFrame:
    county_entries = registry.get("counties", {}) if isinstance(registry.get("counties"), dict) else {}
    pending_entries = registry.get("pending", {}) if isinstance(registry.get("pending"), dict) else {}
    coverage_lookup = coverage_matrix.set_index("county_name")
    qa_focus = set(coverage_qa.loc[coverage_qa["issue_type"].isin(["zero_match_count", "suspiciously_low_match_count", "coverage_status_inconsistent"]), "county_name"].astype("string"))
    source_cache: dict[str, pd.DataFrame] = {}
    records: list[dict[str, object]] = []

    for county_name in county_names:
        coverage_record = coverage_lookup.loc[county_name].to_dict() if county_name in coverage_lookup.index else {}
        entry = county_entries.get(county_name)
        pending_entry = pending_entries.get(county_name)
        selected = entry or pending_entry or {}
        source_path_raw = selected.get("source_path")
        source_path = resolve_path(str(source_path_raw)) if source_path_raw else None
        ingest_mode = selected.get("ingest_mode")
        source_type = selected.get("source_type")
        source_county_field = selected.get("source_county_field")
        source_county_value = str(selected.get("source_county_value") or county_name).strip().lower()

        total_source_rows = pd.NA
        filtered_county_rows = pd.NA
        distinct_identifier_count = pd.NA
        normalized_identifier_count = pd.NA
        unmatched_row_count = pd.NA
        ambiguous_row_count = pd.NA
        unmatched_rate = pd.NA
        top_mismatch_patterns = pd.NA

        if source_path and source_path.exists():
            try:
                source_frame = _load_source_frame_for_diagnostics(source_cache, source_path, ingest_mode, source_type)
                total_source_rows = int(len(source_frame))
                county_frame = source_frame
                if source_county_field and source_county_field in source_frame.columns:
                    county_frame = source_frame.loc[
                        source_frame[source_county_field].astype("string").str.strip().str.lower().eq(source_county_value)
                    ].copy()
                filtered_county_rows = int(len(county_frame))
                identifier_series = _diagnostic_identifier_series(county_frame)
                distinct_identifier_count = int(identifier_series.nunique(dropna=True)) if not county_frame.empty else 0
                normalized_identifier_count = int(identifier_series.dropna().nunique()) if not county_frame.empty else 0
                matched_row_count = int(coverage_record.get("matched_row_count") or 0)
                if filtered_county_rows and pd.notna(filtered_county_rows):
                    unmatched_rate = round(max(0.0, 1.0 - (matched_row_count / max(int(filtered_county_rows), 1))), 4)
            except Exception as exc:
                top_mismatch_patterns = f"diagnostic_load_error::{exc}"

        unmatched_path = BASE_DIR / "data" / "tax_linked" / "ms" / county_name / f"{county_name}_unmatched_tax_records.parquet"
        if unmatched_path.exists():
            unmatched_frame = pd.read_parquet(unmatched_path)
            unmatched_row_count = int(len(unmatched_frame))
            if "unmatched_reason" in unmatched_frame.columns and not unmatched_frame.empty:
                top_mismatch_patterns = " | ".join(
                    f"{reason}:{count}"
                    for reason, count in unmatched_frame["unmatched_reason"].astype("string").fillna("unknown").value_counts().head(3).items()
                )

        ambiguous_path = BASE_DIR / "data" / "tax_linked" / "ms" / county_name / f"{county_name}_ambiguous_tax_links.parquet"
        if ambiguous_path.exists():
            ambiguous_row_count = int(len(pd.read_parquet(ambiguous_path)))

        records.append(
            {
                "county_name": county_name,
                "coverage_status": coverage_record.get("coverage_status"),
                "coverage_scope": coverage_record.get("coverage_scope"),
                "quality_flag": coverage_record.get("quality_flag"),
                "discovery_status": coverage_record.get("discovery_status"),
                "source_name": coverage_record.get("source_name"),
                "source_type": coverage_record.get("source_type"),
                "source_path": source_path.relative_to(BASE_DIR).as_posix() if source_path and source_path.exists() else pd.NA,
                "county_focus_flag": bool(county_name in qa_focus or str(coverage_record.get("coverage_status") or "") in {"pending", "partial", "stale"}),
                "parcel_count": int(coverage_record.get("parcel_count") or 0),
                "parcel_match_count": int(coverage_record.get("parcel_match_count") or 0),
                "matched_row_count": int(coverage_record.get("matched_row_count") or 0),
                "total_source_rows": total_source_rows,
                "filtered_county_rows": filtered_county_rows,
                "distinct_identifier_count": distinct_identifier_count,
                "normalized_identifier_count": normalized_identifier_count,
                "unmatched_row_count": unmatched_row_count,
                "ambiguous_row_count": ambiguous_row_count,
                "unmatched_rate": unmatched_rate,
                "top_mismatch_patterns": top_mismatch_patterns,
                "blocker_reason": coverage_record.get("blocker_reason"),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["county_focus_flag", "county_name"], ascending=[False, True]).reset_index(drop=True)


def build_county_remediation_work_queue(
    coverage_matrix: pd.DataFrame,
    coverage_qa: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    qa_map = (
        coverage_qa.groupby("county_name")["issue_type"]
        .agg(lambda values: sorted(set(str(value) for value in values)))
        .to_dict()
        if not coverage_qa.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    for record in coverage_matrix.to_dict(orient="records"):
        county_name = str(record["county_name"])
        quality_flag = str(record.get("quality_flag") or "")
        coverage_status = str(record.get("coverage_status") or "")
        coverage_scope = str(record.get("coverage_scope") or "")
        discovery_status = str(record.get("discovery_status") or "")
        parcel_count = int(record.get("parcel_count") or 0)
        parcel_match_count = int(record.get("parcel_match_count") or 0)
        issue_types = qa_map.get(county_name, [])

        priority_score = 0.0
        recommended_action = "monitor"
        raw_reason = record.get("blocker_reason")
        reason = "" if pd.isna(raw_reason) else str(raw_reason)

        if quality_flag == "zero_match_review":
            priority_score += 120.0 + min(parcel_count / 2500.0, 40.0)
            recommended_action = "repair_linking"
            reason = reason or "County source loaded, but no linked parcel matches are landing."
        elif coverage_status == "stale" and coverage_scope == "full":
            priority_score += 95.0 + min(parcel_match_count / 100.0, 35.0)
            recommended_action = "refresh_county_source"
            reason = reason or "County-specific source has good linkage volume, but freshness is stale."
        elif "suspiciously_low_match_count" in issue_types:
            priority_score += 85.0 + min(parcel_count / 5000.0, 25.0)
            recommended_action = "inspect_linking"
            reason = reason or "County source is loaded but match volume is suspiciously low."
        elif discovery_status == "pending_parser_needed":
            priority_score += 75.0 + min(parcel_count / 5000.0, 20.0)
            recommended_action = "build_parser"
            reason = reason or "County source looks reachable, but parser/linker work is still needed."
        elif discovery_status == "pending_subscription_gated":
            priority_score += 35.0 + min(parcel_count / 10000.0, 15.0)
            recommended_action = "obtain_access"
            reason = reason or "County source exists, but access is subscription-gated."
        elif coverage_status == "stale":
            priority_score += 55.0 + min(parcel_match_count / 150.0, 20.0)
            recommended_action = "refresh_partial_source"
            reason = reason or "Partial county coverage exists but is stale."
        elif coverage_status == "partial":
            priority_score += 40.0 + min(parcel_count / 7500.0, 10.0)
            recommended_action = "diagnose_partial_coverage"
            reason = reason or "County coverage is only partial."

        if quality_flag == "duplicate_match_review":
            priority_score += 20.0
            if recommended_action == "monitor":
                recommended_action = "deduplicate_matches"

        if priority_score <= 0:
            continue

        rows.append(
            {
                "county_name": county_name,
                "priority_score": round(priority_score, 2),
                "recommended_action": recommended_action,
                "priority_reason": reason,
                "coverage_status": coverage_status,
                "coverage_scope": coverage_scope,
                "quality_flag": quality_flag,
                "discovery_status": discovery_status or pd.NA,
                "parcel_count": parcel_count,
                "parcel_match_count": parcel_match_count,
                "matched_row_count": int(record.get("matched_row_count") or 0),
                "source_name": record.get("source_name"),
                "source_type": record.get("source_type"),
                "source_url": record.get("source_url"),
            }
        )

    work_queue = pd.DataFrame.from_records(rows).sort_values(["priority_score", "parcel_match_count", "parcel_count"], ascending=[False, False, False]).reset_index(drop=True)
    if not work_queue.empty:
        work_queue.insert(0, "priority_rank", pd.Series(range(1, len(work_queue) + 1), dtype="Int64"))
    diagnostics_fields = diagnostics[["county_name", "total_source_rows", "filtered_county_rows", "unmatched_row_count", "ambiguous_row_count", "top_mismatch_patterns"]] if not diagnostics.empty else pd.DataFrame(columns=["county_name"])
    return work_queue.merge(diagnostics_fields, on="county_name", how="left")


def build_summary(final_frame: gpd.GeoDataFrame, progress: pd.DataFrame, runtime_seconds: float) -> pd.DataFrame:
    total_rows = len(final_frame)
    matched_rows = int(final_frame["tax_data_available_flag"].fillna(False).sum())
    delinquent_rows = int(final_frame["tax_delinquent_flag"].fillna(False).sum())
    distressed_rows = int(final_frame["distressed_owner_flag"].fillna(False).sum())
    rows: list[dict[str, object]] = [
        {"section": "validation", "metric": "parcel_rows", "county_name": pd.NA, "count": total_rows, "pct": 100.0, "value": total_rows, "notes": "final_tax_rows"},
        {"section": "validation", "metric": "parcel_row_id_duplicates", "county_name": pd.NA, "count": int(final_frame["parcel_row_id"].duplicated().sum()), "pct": pd.NA, "value": int(final_frame["parcel_row_id"].duplicated().sum()), "notes": "expected_zero"},
        {"section": "validation", "metric": "parcel_id_duplicates", "county_name": pd.NA, "count": int(final_frame["parcel_id"].duplicated().sum()), "pct": pd.NA, "value": int(final_frame["parcel_id"].duplicated().sum()), "notes": "expected_zero"},
        {"section": "validation", "metric": "runtime_seconds", "county_name": pd.NA, "count": pd.NA, "pct": pd.NA, "value": round(runtime_seconds, 2), "notes": "tax_distress_stage_runtime"},
        {"section": "statewide", "metric": "parcel_match_rate", "county_name": pd.NA, "count": matched_rows, "pct": round(matched_rows / total_rows * 100.0, 4), "value": matched_rows, "notes": "tax_data_available_flag_true_rows"},
        {"section": "statewide", "metric": "tax_delinquent_rate", "county_name": pd.NA, "count": delinquent_rows, "pct": round(delinquent_rows / total_rows * 100.0, 4), "value": delinquent_rows, "notes": "tax_delinquent_flag_true_rows"},
        {"section": "statewide", "metric": "distressed_owner_rate", "county_name": pd.NA, "count": distressed_rows, "pct": round(distressed_rows / total_rows * 100.0, 4), "value": distressed_rows, "notes": "distressed_owner_flag_true_rows"},
        {"section": "statewide", "metric": "counties_processed_successfully", "county_name": pd.NA, "count": int(progress["status"].eq("processed").sum()), "pct": pd.NA, "value": int(progress["status"].eq("processed").sum()), "notes": "counties_with_loaded_tax_sources"},
        {"section": "statewide", "metric": "counties_missing_tax_sources", "county_name": pd.NA, "count": int(progress["status"].isin(["missing_registry", "missing_source", "configured_without_path"]).sum()), "pct": pd.NA, "value": int(progress["status"].isin(["missing_registry", "missing_source", "configured_without_path"]).sum()), "notes": "counties_without_usable_local_sources"},
    ]
    for value, count in final_frame["most_severe_tax_status"].value_counts(dropna=False).items():
        rows.append({"section": "distribution", "metric": "most_severe_tax_status", "county_name": pd.NA, "count": int(count), "pct": round(int(count) / total_rows * 100.0, 4), "value": value, "notes": "value_count"})
    county_rank = (
        final_frame.assign(distressed_num=final_frame["distressed_owner_flag"].fillna(False).astype(bool).astype(int))
        .groupby("county_name", as_index=False)
        .agg(parcel_rows=("parcel_row_id", "size"), distressed_rows=("distressed_num", "sum"))
    )
    county_rank["distressed_pct"] = county_rank["distressed_rows"] / county_rank["parcel_rows"] * 100.0
    for row in county_rank.sort_values(["distressed_rows", "distressed_pct"], ascending=[False, False]).head(10).itertuples(index=False):
        rows.append({"section": "top_counties", "metric": "distressed_owner_rows", "county_name": row.county_name, "count": int(row.distressed_rows), "pct": round(float(row.distressed_pct), 4), "value": int(row.parcel_rows), "notes": "distressed_rows / parcel_rows"})
    for row in progress.sort_values("county_name").itertuples(index=False):
        rows.append({"section": "county_processing", "metric": row.status, "county_name": row.county_name, "count": int(row.parcel_rows), "pct": pd.NA, "value": int(row.matched_parcels), "notes": row.notes})
    return pd.DataFrame.from_records(rows)


def refresh_owner_outputs(final_frame: gpd.GeoDataFrame) -> None:
    owner_leads = gpd.read_parquet(OWNER_OUTPUT_PARQUET)
    tax_fields = final_frame.loc[:, ["parcel_row_id", "tax_data_available_flag", "tax_delinquent_flag", "tax_distress_score", "distressed_owner_flag"]].copy()
    owner_leads = owner_leads.drop(columns=["tax_data_available_flag", "tax_delinquent_flag", "tax_distress_score", "distressed_owner_flag"], errors="ignore")
    owner_leads = owner_leads.merge(tax_fields, on="parcel_row_id", how="left")
    owner_leads["tax_data_available_flag"] = owner_leads["tax_data_available_flag"].fillna(False).astype("boolean")
    owner_leads["tax_delinquent_flag"] = owner_leads["tax_delinquent_flag"].fillna(False).astype("boolean")
    owner_leads["tax_distress_score"] = pd.to_numeric(owner_leads["tax_distress_score"], errors="coerce").fillna(0.0).astype("float64")
    owner_leads["distressed_owner_flag"] = owner_leads["distressed_owner_flag"].fillna(False).astype("boolean")
    owner_leads["mailer_target_score"] = build_mailer_target_score(owner_leads)
    mailer_export = owner_leads.loc[:, MAILER_EXPORT_FIELDS].copy()
    summary = build_owner_summary(
        owner_leads,
        audit_fields=[
            "source_parcel_id_raw", "source_parcel_id_normalized", "owner_name_raw",
            "mailing_address_line1", "mailing_city", "mailing_state", "mailing_zip",
            "property_address_raw", "property_city", "property_state", "land_use_raw",
            "tax_status", "tax_data_available_flag", "tax_delinquent_flag", "distressed_owner_flag",
        ],
        runtime_seconds=0.0,
    )
    schema = build_owner_schema()
    county_qa = build_owner_county_qa(owner_leads)
    write_owner_outputs(
        owner_leads=owner_leads,
        mailer_export=mailer_export,
        summary=summary,
        schema=schema,
        county_qa=county_qa,
        output_parquet=OWNER_OUTPUT_PARQUET,
        output_gpkg=OWNER_OUTPUT_GPKG,
        mailer_export_csv=OWNER_MAILER_EXPORT_CSV,
        summary_csv=OWNER_SUMMARY_CSV,
        schema_csv=OWNER_SCHEMA_CSV,
        county_qa_csv=OWNER_COUNTY_QA_CSV,
    )


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    master_input = resolve_path(args.master_input)
    owner_input = resolve_path(args.owner_input)
    registry_path = resolve_path(args.registry)
    output_parquet = resolve_path(args.output_parquet)
    output_gpkg = resolve_path(args.output_gpkg)
    summary_csv = resolve_path(args.summary_csv)
    progress_csv = resolve_path(args.progress_csv)
    coverage_matrix_parquet = resolve_path(args.coverage_matrix_parquet)
    coverage_matrix_csv = resolve_path(args.coverage_matrix_csv)
    coverage_qa_csv = resolve_path(args.coverage_qa_csv)
    coverage_qa_summary_json = resolve_path(args.coverage_qa_summary_json)
    county_diagnostics_csv = resolve_path(args.county_diagnostics_csv)
    county_work_queue_csv = resolve_path(args.county_work_queue_csv)
    county_parts_dir = resolve_path(args.county_parts_dir)
    normalized_parts_dir = resolve_path(args.normalized_parts_dir)

    processed_at = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    registry = parse_registry(registry_path)

    print(f"Loading master parcels from {master_input.relative_to(BASE_DIR)}")
    master = gpd.read_parquet(master_input, columns=MASTER_COLUMNS)
    print(f"Loading owner leads from {owner_input.relative_to(BASE_DIR)}")
    owner = pd.read_parquet(owner_input, columns=OWNER_COLUMNS)
    master["parcel_row_id"] = master["parcel_row_id"].astype("string")
    master["county_name"] = master["county_name"].astype("string").str.lower()
    master["source_parcel_id_normalized"] = master["source_parcel_id_normalized"].astype("string")
    master["master_row_order"] = np.arange(len(master), dtype=np.int64)
    owner["parcel_row_id"] = owner["parcel_row_id"].astype("string")
    master = master.merge(owner, on="parcel_row_id", how="left")

    county_parts_dir.mkdir(parents=True, exist_ok=True)
    normalized_parts_dir.mkdir(parents=True, exist_ok=True)
    progress_existing = read_progress(progress_csv) if args.resume else pd.DataFrame()
    progress_rows: list[dict[str, object]] = []
    county_names = master["county_name"].dropna().astype("string").sort_values().unique().tolist()

    for county_name in county_names:
        entry = registry.get("counties", {}).get(county_name)
        source_path = resolve_path(str(entry.get("source_path"))) if entry and entry.get("source_path") else None
        fingerprint = source_fingerprint(entry, source_path)
        county_part = county_parts_dir / f"{county_name}.parquet"
        normalized_part = normalized_parts_dir / f"{county_name}.parquet"
        county_frame = master.loc[master["county_name"].eq(county_name)].copy()

        if args.resume and should_skip(progress_existing, county_name, fingerprint, county_part):
            latest = progress_existing.loc[progress_existing["county_name"].eq(county_name)].sort_values("processed_at_utc").iloc[-1].to_dict()
            progress_rows.append(latest)
            print(f"[resume] {county_name}: {latest['status']}")
            continue

        status = "processed"
        notes = "tax source normalized and aggregated"
        raw_record_count = 0
        normalized_record_count = 0
        aggregated: pd.DataFrame | None = None
        source_loaded = False

        ingest_mode = str(entry.get("ingest_mode", "")).strip().lower() if entry else ""

        if entry is None:
            status = "missing_registry"
            notes = "No county registry entry present."
        elif source_path is None:
            status = "configured_without_path"
            notes = "County registry entry has no source_path."
        elif not source_path.exists():
            status = "missing_source"
            notes = f"Source file not found: {source_path.relative_to(BASE_DIR).as_posix()}"
        else:
            try:
                raw_source = load_source_frame("parquet" if ingest_mode == "linked_parquet" else str(entry.get("source_type", "")), source_path)
                raw_record_count = len(raw_source)
                if ingest_mode == "linked_parquet":
                    normalized = build_linked_tax_frame(raw_source, county_name, entry, source_path)
                    normalized_record_count = int(normalized["parcel_row_id"].notna().sum())
                    normalized.to_parquet(normalized_part, index=False)
                    aggregated = aggregate_linked_tax_frame(normalized)
                else:
                    normalized = build_normalized_tax_frame(raw_source, county_name, entry, source_path, processed_at)
                    normalized_record_count = int(normalized["source_parcel_id_normalized"].notna().sum())
                    normalized.to_parquet(normalized_part, index=False)
                    aggregated = aggregate_tax_frame(normalized)
                source_loaded = True
            except Exception as exc:
                status = "load_error"
                notes = str(exc)

        county_output = prepare_county_output(county_frame, aggregated, entry, source_path, source_loaded, float(args.distressed_threshold))
        county_output.to_parquet(county_part, index=False)
        matched_parcels = int(county_output["tax_data_available_flag"].fillna(False).sum())
        progress_rows.append(
            {
                "county_name": county_name,
                "status": status,
                "source_type": entry.get("source_type") if entry else pd.NA,
                "source_path": source_path.relative_to(BASE_DIR).as_posix() if source_path else pd.NA,
                "source_fingerprint": fingerprint,
                "parcel_rows": len(county_output),
                "raw_record_count": raw_record_count,
                "normalized_record_count": normalized_record_count,
                "matched_parcels": matched_parcels,
                "processed_at_utc": processed_at,
                "notes": notes,
            }
        )
        print(f"[{county_name}] {status} | parcels={len(county_output):,} | matched={matched_parcels:,}")

    progress = pd.DataFrame.from_records(progress_rows).sort_values(["county_name", "processed_at_utc"]).reset_index(drop=True)
    progress.to_csv(progress_csv, index=False)
    final_parts = [gpd.read_parquet(county_parts_dir / f"{county_name}.parquet") for county_name in county_names]
    final_frame = gpd.GeoDataFrame(pd.concat(final_parts, ignore_index=True), geometry="geometry", crs=master.crs)
    final_frame = final_frame.merge(master.loc[:, ["parcel_row_id", "master_row_order"]], on="parcel_row_id", how="left")
    final_frame = final_frame.sort_values("master_row_order").drop(columns=["master_row_order"]).reset_index(drop=True)
    coverage_matrix = build_county_coverage_matrix(county_names, registry, progress, final_frame)
    final_frame = apply_county_coverage_matrix(final_frame, coverage_matrix)
    coverage_qa = build_county_coverage_qa(coverage_matrix, progress, final_frame, normalized_parts_dir)
    county_diagnostics = build_county_linkage_diagnostics(county_names, registry, coverage_matrix, progress, coverage_qa)
    county_work_queue = build_county_remediation_work_queue(coverage_matrix, coverage_qa, county_diagnostics)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    final_frame.to_parquet(output_parquet, index=False)
    final_frame.to_file(output_gpkg, driver="GPKG", engine="pyogrio")
    coverage_matrix_parquet.parent.mkdir(parents=True, exist_ok=True)
    coverage_matrix.to_parquet(coverage_matrix_parquet, index=False)
    coverage_matrix.to_csv(coverage_matrix_csv, index=False)
    coverage_qa.to_csv(coverage_qa_csv, index=False)
    county_diagnostics.to_csv(county_diagnostics_csv, index=False)
    county_work_queue.to_csv(county_work_queue_csv, index=False)
    coverage_summary = {
        "county_count": int(len(coverage_matrix)),
        "coverage_status_counts": {str(key): int(value) for key, value in coverage_matrix["coverage_status"].fillna("unknown").value_counts().items()},
        "qa_issue_counts": {str(key): int(value) for key, value in coverage_qa["issue_type"].fillna("unknown").value_counts().items()} if not coverage_qa.empty else {},
    }
    coverage_qa_summary_json.write_text(json.dumps(coverage_summary, indent=2), encoding="utf-8")
    summary = build_summary(final_frame, progress, time.perf_counter() - start)
    summary.to_csv(summary_csv, index=False)
    refresh_owner_outputs(final_frame)

    runtime_seconds = time.perf_counter() - start
    print(f"Tax distress build complete in {runtime_seconds / 60.0:.2f} minutes")
    print(f"Rows: {len(final_frame):,}")
    print(f"Matched tax rows: {int(final_frame['tax_data_available_flag'].fillna(False).sum()):,}")
    print(f"Delinquent parcels: {int(final_frame['tax_delinquent_flag'].fillna(False).sum()):,}")
    print(f"Distressed owners: {int(final_frame['distressed_owner_flag'].fillna(False).sum()):,}")


if __name__ == "__main__":
    main()
