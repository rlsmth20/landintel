from __future__ import annotations

import argparse
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
    "county_tax_source_type", "county_tax_source_path", "tax_data_available_flag",
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
    registry: dict[str, object] = {"state": None, "counties": {}}
    if not path.exists():
        return registry
    current_county: str | None = None
    in_counties = False
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
            in_counties = token == "counties:"
            continue
        if not in_counties:
            continue
        if indent == 2 and token.endswith(":"):
            current_county = token[:-1].strip().lower()
            registry["counties"][current_county] = {}
            continue
        if indent >= 4 and current_county and ":" in token:
            key, value = token.split(":", 1)
            registry["counties"][current_county][key.strip()] = value.strip().strip("'\"") or None
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


def normalize_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    out = pd.Series(dt.dt.strftime("%Y-%m-%d"), index=series.index, dtype="string")
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
    out["county_tax_source_path"] = pd.Series(source_path.relative_to(BASE_DIR).as_posix() if source_path else pd.NA, index=out.index, dtype="string")
    if aggregated is not None and not aggregated.empty:
        out = out.merge(aggregated, on=["state_code", "county_name", "source_parcel_id_normalized"], how="left")
    defaults = {
        "tax_data_available_flag": pd.Series(False, index=out.index, dtype="boolean"),
        "tax_delinquent_flag": pd.Series(False, index=out.index, dtype="boolean"),
        "delinquent_year_count": pd.Series(0, index=out.index, dtype="int32"),
        "delinquent_tax_amount_total": pd.Series(0.0, index=out.index, dtype="float64"),
        "tax_sale_flag": pd.Series(False, index=out.index, dtype="boolean"),
        "latest_delinquent_year": pd.Series(pd.NA, index=out.index, dtype="Int64"),
        "most_severe_tax_status": pd.Series("unknown", index=out.index, dtype="string"),
        "tax_record_count": pd.Series(0, index=out.index, dtype="int32"),
        "tax_source_name": pd.Series(pd.NA, index=out.index, dtype="string"),
    }
    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default
    out["tax_data_available_flag"] = out["tax_data_available_flag"].fillna(False).astype("boolean")
    out["tax_delinquent_flag"] = out["tax_delinquent_flag"].fillna(False).astype("boolean")
    out["delinquent_year_count"] = pd.to_numeric(out["delinquent_year_count"], errors="coerce").fillna(0).astype("int32")
    out["delinquent_tax_amount_total"] = pd.to_numeric(out["delinquent_tax_amount_total"], errors="coerce").fillna(0.0).round(2).astype("float64")
    out["tax_sale_flag"] = out["tax_sale_flag"].fillna(False).astype("boolean")
    out["latest_delinquent_year"] = pd.to_numeric(out["latest_delinquent_year"], errors="coerce").astype("Int64")
    out["most_severe_tax_status"] = out["most_severe_tax_status"].fillna("unknown").astype("string")
    out["tax_record_count"] = pd.to_numeric(out["tax_record_count"], errors="coerce").fillna(0).astype("int32")
    out["tax_source_name"] = out["tax_source_name"].astype("string")
    out["tax_distress_score"] = build_tax_distress_score(out)
    out["distressed_owner_flag"] = (
        out["tax_delinquent_flag"].fillna(False).astype(bool)
        | out["tax_distress_score"].ge(distressed_threshold)
    ).astype("boolean")
    return out.loc[:, FINAL_COLUMNS].copy()


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
                raw_source = load_source_frame(str(entry.get("source_type", "")), source_path)
                raw_record_count = len(raw_source)
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

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    final_frame.to_parquet(output_parquet, index=False)
    final_frame.to_file(output_gpkg, driver="GPKG", engine="pyogrio")
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
