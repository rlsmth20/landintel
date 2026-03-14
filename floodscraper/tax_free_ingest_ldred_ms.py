from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from tax_common import (
    BASE_DIR,
    RAW_TAX_DIR,
    TAX_LINKED_DIR,
    TAX_METADATA_DIR,
    TAX_STANDARDIZED_DIR,
    build_record_hash,
    build_row_hash,
    clean_string,
    infer_corporate_owner,
    link_standardized_tax_records,
    load_master_index,
    normalize_identifier,
    update_registry_row,
    write_json,
)

MASTER_PARQUET = BASE_DIR / "data" / "parcels" / "mississippi_parcels_master.parquet"
REGISTRY_CSV = BASE_DIR / "data" / "tax_metadata" / "tax_source_registry_ms.csv"
STATE_CODE = "MS"

COUNTY_CONFIGS: list[dict[str, str | int]] = [
    {
        "county_name": "calhoun",
        "county_fips": "013",
        "county_code": 7,
        "source_id": "ms_013_dsm_land_redemption",
        "source_name": "calhoun_land_redemption",
        "source_url": "https://cs.datasysmgt.com/tax?state=MS&county=7",
    },
    {
        "county_name": "clay",
        "county_fips": "025",
        "county_code": 13,
        "source_id": "ms_025_dsm_land_redemption",
        "source_name": "clay_land_redemption",
        "source_url": "https://cs.datasysmgt.com/tax?state=MS&county=13",
    },
    {
        "county_name": "coahoma",
        "county_fips": "027",
        "county_code": 14,
        "source_id": "ms_027_dsm_land_redemption",
        "source_name": "coahoma_land_redemption",
        "source_url": "https://cs.datasysmgt.com/tax?state=MS&county=14",
    },
]

STATUS_MAP = {
    "": ("land_redemption_open", True, False, "unpaid"),
    "H": ("hold_for_sale", True, False, "unpaid"),
    "F": ("forfeited", True, True, "unpaid"),
    "M": ("matured_to_state", True, True, "unpaid"),
    "T": ("tax_deed_issued", True, True, "unpaid"),
    "R": ("released", False, False, "released"),
    "V": ("void", False, False, "void"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest public Mississippi land-redemption county sources.")
    parser.add_argument("--download-dir", default=str(RAW_TAX_DIR / "ms"), help="Base raw tax directory.")
    return parser.parse_args()


def fetch_json(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode(params)
    with urlopen(f"{base_url}?{query}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def source_paths(download_dir: Path, county_fips: str, source_name: str, county_name: str) -> dict[str, Path]:
    run_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    raw_dir = download_dir / county_fips / source_name / run_date
    standardized_dir = TAX_STANDARDIZED_DIR / "ms" / county_name
    linked_dir = TAX_LINKED_DIR / "ms" / county_name
    return {
        "raw_dir": raw_dir,
        "raw_json": raw_dir / "land_redemption_list.json",
        "manifest": raw_dir / "manifest.json",
        "standardized": standardized_dir / f"{county_name}_land_redemption_records.parquet",
        "linked": linked_dir / f"{county_name}_land_redemption_linked_tax_records.parquet",
        "unmatched": linked_dir / f"{county_name}_land_redemption_unmatched_tax_records.parquet",
        "ambiguous": linked_dir / f"{county_name}_land_redemption_ambiguous_tax_links.parquet",
        "summary": TAX_METADATA_DIR / f"tax_free_{county_name}_land_redemption_linkage_summary_ms.csv",
        "unmatched_reason_summary": TAX_METADATA_DIR / f"tax_free_{county_name}_land_redemption_unmatched_reason_summary_ms.csv",
        "ambiguity_reason_summary": TAX_METADATA_DIR / f"tax_free_{county_name}_land_redemption_ambiguity_reason_summary_ms.csv",
        "identifier_diagnostics": TAX_METADATA_DIR / f"tax_free_{county_name}_land_redemption_identifier_diagnostics_ms.csv",
        "qa_summary": TAX_METADATA_DIR / f"tax_free_{county_name}_land_redemption_qa_summary_ms.csv",
    }


def load_cached_raw_records(download_dir: Path, county_fips: str, source_name: str) -> pd.DataFrame:
    source_dir = download_dir / county_fips / source_name
    if not source_dir.exists():
        return pd.DataFrame()
    raw_candidates = sorted(source_dir.glob("*/land_redemption_list.json"), reverse=True)
    if not raw_candidates:
        return pd.DataFrame()
    payload = json.loads(raw_candidates[0].read_text(encoding="utf-8"))
    records = payload.get("records") or payload.get("data") or []
    return pd.DataFrame(records)


def fetch_county_records(county_code: int) -> pd.DataFrame:
    base_url = "https://cs.datasysmgt.com/lrdsmp/ldredweb"
    offset = 0
    total = None
    all_rows: list[dict[str, Any]] = []
    while True:
        payload = fetch_json(
            base_url,
            {
                "task": "getlist",
                "state": STATE_CODE,
                "county": county_code,
                "start": offset,
                "limit": 100,
            },
        )
        if total is None:
            total = int(payload.get("totalcount", 0))
        page = payload.get("data") or payload.get("records") or []
        if not page:
            break
        all_rows.extend(page)
        offset += 100
        if total is not None and offset >= total:
            break
    return pd.DataFrame(all_rows)


def build_raw_payload(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame.astype("string").fillna("").apply(lambda row: json.dumps(row.to_dict(), separators=(",", ":")), axis=1),
        index=frame.index,
        dtype="string",
    )


def standardize_records(frame: pd.DataFrame, config: dict[str, str | int], run_id: str, raw_json_path: Path) -> pd.DataFrame:
    county_name = str(config["county_name"])
    county_fips = str(config["county_fips"])
    source_name = str(config["source_name"])
    parcel_id_raw = clean_string(frame["lmaprf"].astype("string").str.replace(r"</?pre>", "", regex=True))
    owner_name = clean_string(frame["lname"])
    situs_address = clean_string(frame["laddress"])
    situs_city = clean_string(frame["lcity"])
    tax_year = pd.to_numeric(frame["lyear"], errors="coerce").astype("Int64")
    exception_code = clean_string(frame["lexception"]).fillna("")
    status_info = exception_code.map(lambda code: STATUS_MAP.get(code, ("land_redemption_open", True, False, "unpaid")))

    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash([STATE_CODE, county_fips, source_name, parcel_id, year, rec, recsb])
                    for parcel_id, year, rec, recsb in zip(parcel_id_raw, frame["lyear"], frame["lrec"], frame["lrecsb"])
                ),
                index=frame.index,
                dtype="string",
            ),
            "parcel_row_id": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "parcel_id_raw": parcel_id_raw,
            "parcel_id_normalized": normalize_identifier(parcel_id_raw),
            "state_code": pd.Series(STATE_CODE, index=frame.index, dtype="string"),
            "county_fips": pd.Series(county_fips, index=frame.index, dtype="string"),
            "county_name": pd.Series(county_name, index=frame.index, dtype="string"),
            "source_name": pd.Series(source_name, index=frame.index, dtype="string"),
            "source_type": pd.Series("direct_download_page", index=frame.index, dtype="string"),
            "source_dataset_path": pd.Series(raw_json_path.relative_to(BASE_DIR).as_posix(), index=frame.index, dtype="string"),
            "source_record_id": pd.Series(
                [f"{rec}:{recsb or '0'}:{year}" for rec, recsb, year in zip(frame["lrec"], frame["lrecsb"], frame["lyear"])],
                index=frame.index,
                dtype="string",
            ),
            "source_ppin": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "ingestion_run_id": pd.Series(run_id, index=frame.index, dtype="string"),
            "source_file_version": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%d"), index=frame.index, dtype="string"),
            "loaded_at": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=frame.index, dtype="string"),
            "owner_name": owner_name,
            "owner_name_2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line1": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_city": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_state": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_zip": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_address": situs_address,
            "situs_city": situs_city,
            "situs_state": pd.Series("MS", index=frame.index, dtype="string"),
            "situs_zip": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "assessed_land_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "assessed_improvement_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "assessed_total_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_land_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_improvement_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_total_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "taxable_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "exemptions_text": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "exemptions_amount": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_year": tax_year,
            "bill_year": tax_year,
            "tax_amount_due": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_amount_paid": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_balance_due": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_status": pd.Series([item[0] for item in status_info], index=frame.index, dtype="string"),
            "payment_status": pd.Series([item[3] for item in status_info], index=frame.index, dtype="string"),
            "delinquent_flag": pd.Series([item[1] for item in status_info], index=frame.index, dtype="boolean"),
            "forfeited_flag": pd.Series([item[2] for item in status_info], index=frame.index, dtype="boolean"),
            "delinquent_amount": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "delinquent_years": frame["lyear"].astype("string"),
            "delinquent_as_of_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "last_payment_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "due_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "absentee_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "out_of_state_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "owner_corporate_flag": infer_corporate_owner(owner_name),
            "mailing_matches_situs_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "tax_delinquent_flag_standardized": pd.Series([item[1] for item in status_info], index=frame.index, dtype="boolean"),
            "raw_payload_json": build_raw_payload(frame),
        }
    )
    standardized["exception_code"] = exception_code.replace({"": pd.NA}).astype("string")
    standardized["record_hash"] = build_record_hash(
        standardized,
        ["parcel_id_normalized", "owner_name", "tax_year", "tax_status", "source_record_id"],
    )
    return standardized


def build_reason_summaries(unmatched: pd.DataFrame, ambiguous: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unmatched_summary = (
        unmatched.groupby(["county_name", "county_fips", "unmatched_reason"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values(["row_count"], ascending=False)
        .reset_index(drop=True)
        if not unmatched.empty
        else pd.DataFrame(columns=["county_name", "county_fips", "unmatched_reason", "row_count"])
    )
    ambiguity_summary = (
        ambiguous.groupby(["county_name", "county_fips", "ambiguity_reason"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values(["row_count"], ascending=False)
        .reset_index(drop=True)
        if not ambiguous.empty
        else pd.DataFrame(columns=["county_name", "county_fips", "ambiguity_reason", "row_count"])
    )
    return unmatched_summary, ambiguity_summary


def build_identifier_diagnostics(standardized: pd.DataFrame, master: pd.DataFrame, county_name: str) -> pd.DataFrame:
    county_master = master.loc[master["county_name"].eq(county_name)].copy()
    compact_source = standardized["parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    compact_master = county_master["source_parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    direct_overlap = int(standardized["parcel_id_normalized"].isin(set(county_master["source_parcel_id_normalized"].dropna())).sum())
    compact_overlap = int(compact_source.isin(set(compact_master.dropna())).sum())
    return pd.DataFrame(
        [
            {"metric": f"{county_name}_rows", "value": int(len(standardized))},
            {"metric": f"{county_name}_unique_parcel_ids", "value": int(standardized["parcel_id_normalized"].nunique())},
            {"metric": f"{county_name}_duplicate_parcel_rows", "value": int(standardized["parcel_id_normalized"].duplicated(keep=False).fillna(False).sum())},
            {"metric": f"{county_name}_direct_identifier_overlap_rows", "value": direct_overlap},
            {"metric": f"{county_name}_compact_identifier_overlap_rows", "value": compact_overlap},
            {"metric": f"{county_name}_safe_adapter_justified", "value": "not_needed"},
        ]
    )


def build_qa_summary(standardized: pd.DataFrame, linked: pd.DataFrame, ambiguous: pd.DataFrame, county_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "rows", "value": int(len(standardized))},
            {"metric": "unique_parcel_ids", "value": int(standardized["parcel_id_normalized"].nunique())},
            {"metric": "null_parcel_id_rate", "value": round(float(standardized["parcel_id_normalized"].isna().mean() * 100.0), 4)},
            {"metric": "duplicate_record_rate", "value": round(float(standardized["parcel_id_normalized"].duplicated(keep=False).mean() * 100.0), 4)},
            {"metric": "linked_rate", "value": round(float(len(linked) / len(standardized) * 100.0), 4) if len(standardized) else 0.0},
            {"metric": "ambiguity_rate", "value": round(float(len(ambiguous) / len(standardized) * 100.0), 4) if len(standardized) else 0.0},
            {"metric": "county_name", "value": county_name},
        ]
    )


def ingest_county(config: dict[str, str | int], download_dir: Path, master: pd.DataFrame) -> dict[str, Any]:
    county_name = str(config["county_name"])
    county_fips = str(config["county_fips"])
    source_name = str(config["source_name"])
    source_id = str(config["source_id"])
    source_url = str(config["source_url"])
    county_code = int(config["county_code"])
    run_id = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    paths = source_paths(download_dir, county_fips, source_name, county_name)

    raw_frame = fetch_county_records(county_code)
    if raw_frame.empty:
        raw_frame = load_cached_raw_records(download_dir, county_fips, source_name)
    if raw_frame.empty:
        return {"county_name": county_name, "rows": 0, "linked_rows": 0}

    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    paths["standardized"].parent.mkdir(parents=True, exist_ok=True)
    paths["linked"].parent.mkdir(parents=True, exist_ok=True)
    raw_payload = {
        "county_name": county_name,
        "county_fips": county_fips,
        "county_code": county_code,
        "source_url": source_url,
        "retrieved_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_rows": int(len(raw_frame)),
        "records": raw_frame.to_dict("records"),
    }
    write_json(paths["raw_json"], raw_payload)
    write_json(
        paths["manifest"],
        {
            "county_name": county_name,
            "county_fips": county_fips,
            "source_id": source_id,
            "source_name": source_name,
            "source_url": source_url,
            "raw_json": paths["raw_json"].relative_to(BASE_DIR).as_posix(),
            "retrieved_at": raw_payload["retrieved_at"],
            "row_count": int(len(raw_frame)),
        },
    )

    standardized = standardize_records(raw_frame, config, run_id, paths["raw_json"])
    standardized.to_parquet(paths["standardized"], index=False)

    linked, unmatched, ambiguous, _county_summary = link_standardized_tax_records(
        standardized,
        master,
        heuristic_variants_by_county={county_name: []},
    )
    linked.to_parquet(paths["linked"], index=False)
    unmatched.to_parquet(paths["unmatched"], index=False)
    ambiguous.to_parquet(paths["ambiguous"], index=False)

    prefix = county_name
    summary_df = pd.DataFrame(
        [
            {"metric": f"{prefix}_standardized_rows", "value": int(len(standardized))},
            {"metric": f"{prefix}_linked_rows", "value": int(len(linked))},
            {"metric": f"{prefix}_unmatched_rows", "value": int(len(unmatched))},
            {"metric": f"{prefix}_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": f"{prefix}_linkage_rate", "value": round(float(len(linked) / len(standardized) * 100.0), 4) if len(standardized) else 0.0},
            {"metric": f"{prefix}_exact_match_rows", "value": int(linked["linkage_method"].isin(["exact_ppin", "exact_normalized_parcel_id"]).sum()) if not linked.empty else 0},
            {"metric": f"{prefix}_heuristic_match_rows", "value": int(linked["linkage_method"].astype("string").str.startswith("heuristic_").sum()) if not linked.empty else 0},
            {"metric": f"{prefix}_active_delinquent_rows", "value": int(standardized["tax_delinquent_flag_standardized"].fillna(False).sum())},
        ]
    )
    unmatched_summary, ambiguity_summary = build_reason_summaries(unmatched, ambiguous)
    diagnostics_df = build_identifier_diagnostics(standardized, master, county_name)
    qa_df = build_qa_summary(standardized, linked, ambiguous, county_name)

    summary_df.to_csv(paths["summary"], index=False)
    unmatched_summary.to_csv(paths["unmatched_reason_summary"], index=False)
    ambiguity_summary.to_csv(paths["ambiguity_reason_summary"], index=False)
    diagnostics_df.to_csv(paths["identifier_diagnostics"], index=False)
    qa_df.to_csv(paths["qa_summary"], index=False)

    downloaded_at = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    update_registry_row(
        REGISTRY_CSV,
        source_id,
        downloaded_at,
        f"Downloaded land redemption JSON to {paths['raw_json'].relative_to(BASE_DIR).as_posix()}.",
    )

    return {
        "county_name": county_name,
        "rows": int(len(standardized)),
        "linked_rows": int(len(linked)),
        "unmatched_rows": int(len(unmatched)),
        "ambiguous_rows": int(len(ambiguous)),
        "linkage_rate": round(float(len(linked) / len(standardized) * 100.0), 4) if len(standardized) else 0.0,
    }


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir)
    master = load_master_index(MASTER_PARQUET)
    results = [ingest_county(config, download_dir, master) for config in COUNTY_CONFIGS]
    summary = pd.DataFrame(results).sort_values(["linkage_rate", "rows"], ascending=[False, False]).reset_index(drop=True)
    summary_path = TAX_METADATA_DIR / "tax_free_land_redemption_counties_summary_ms.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path.relative_to(BASE_DIR)}")
    for row in results:
        print(
            f"{row['county_name']}: rows={row['rows']:,}, linked={row['linked_rows']:,}, "
            f"unmatched={row.get('unmatched_rows', 0):,}, ambiguous={row.get('ambiguous_rows', 0):,}, "
            f"rate={row.get('linkage_rate', 0.0):.4f}"
        )


if __name__ == "__main__":
    main()
