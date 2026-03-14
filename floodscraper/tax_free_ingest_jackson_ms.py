from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

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
    normalize_ppin,
    update_registry_row,
    write_json,
)

MASTER_PARQUET = BASE_DIR / "data" / "parcels" / "mississippi_parcels_master.parquet"
REGISTRY_CSV = BASE_DIR / "data" / "tax_metadata" / "tax_source_registry_ms.csv"

SOURCE_ID = "ms_059_jackson_taxsale_preliminary_list"
SOURCE_NAME = "jackson_taxsale_preliminary_list"
SOURCE_URL = "https://co.jackson.ms.us/DocumentCenter/View/1450/Tax-Sale-Ad-for-2024-Real-Property-Taxes---Preliminary-List"
COUNTY_NAME = "jackson"
COUNTY_FIPS = "059"
STATE_CODE = "MS"
TAX_YEAR = 2024
AS_OF_DATE = "2025-08-25"
SOURCE_FILE_NAME = "jackson_tax_sale_2024_preliminary_list.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Jackson County preliminary tax-sale list.")
    parser.add_argument("--download-dir", default=str(RAW_TAX_DIR / "ms"), help="Base raw tax directory.")
    return parser.parse_args()


def download_text(url: str) -> str:
    completed = subprocess.run(
        ["curl.exe", "-L", "--fail", url],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=300,
    )
    return completed.stdout


def source_paths(download_dir: Path) -> dict[str, Path]:
    run_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    raw_dir = download_dir / COUNTY_FIPS / SOURCE_NAME / run_date
    standardized_dir = TAX_STANDARDIZED_DIR / "ms" / COUNTY_NAME
    linked_dir = TAX_LINKED_DIR / "ms" / COUNTY_NAME
    return {
        "raw_dir": raw_dir,
        "raw_text": raw_dir / SOURCE_FILE_NAME,
        "manifest": raw_dir / "manifest.json",
        "standardized": standardized_dir / "jackson_tax_sale_records.parquet",
        "linked": linked_dir / "jackson_linked_tax_records.parquet",
        "unmatched": linked_dir / "jackson_unmatched_tax_records.parquet",
        "ambiguous": linked_dir / "jackson_ambiguous_tax_links.parquet",
        "summary": TAX_METADATA_DIR / "tax_free_jackson_linkage_summary_ms.csv",
        "unmatched_reason_summary": TAX_METADATA_DIR / "tax_free_jackson_unmatched_reason_summary_ms.csv",
        "ambiguity_reason_summary": TAX_METADATA_DIR / "tax_free_jackson_ambiguity_reason_summary_ms.csv",
        "identifier_diagnostics": TAX_METADATA_DIR / "tax_free_jackson_identifier_diagnostics_ms.csv",
        "qa_summary": TAX_METADATA_DIR / "tax_free_jackson_qa_summary_ms.csv",
    }


def split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line.startswith("I, KEVIN MILLER") or line.startswith("**TO AVOID THE TAX SALE") or line.startswith("****PERSONAL CHECKS"):
            continue
        if set(line) == {"-"}:
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith("Account:") and current:
            current.append(line)
            continue
        if current or not line.startswith("------------------------------------------"):
            current.append(line)
    if current:
        blocks.append(current)
    cleaned: list[list[str]] = []
    for block in blocks:
        if any(item.startswith("Account:") for item in block) and any("Parcel#" in item for item in block):
            cleaned.append(block)
    return cleaned


def parse_block(block: list[str], record_index: int) -> dict[str, Any] | None:
    owner_name = block[0] if block else pd.NA
    account_line = next((line for line in block if line.startswith("Account:")), None)
    parcel_line = next((line for line in block if line.startswith("Parcel#")), None)
    acres_line = next((line for line in block if line.startswith("Acres -")), None)
    amount_line = next((line for line in block if "TOTAL TAX & COST" in line), None)
    sale_sequence_line = next((line for line in block if line.startswith("Sale Sequence")), None)
    sec_line = next((line for line in block if line.startswith("Sec-")), None)
    if parcel_line is None:
        return None

    parcel_match = re.search(r"Parcel#\s*-\s*(.+)$", parcel_line)
    if parcel_match is None:
        return None

    legal_lines: list[str] = []
    seen_sale = False
    for line in block:
        if line.startswith("Sale Sequence"):
            seen_sale = True
            continue
        if not seen_sale:
            continue
        if line.startswith("Parcel#") or line.startswith("Acres -") or "TOTAL TAX & COST" in line:
            continue
        legal_lines.append(line)

    amount_match = re.search(r"TOTAL TAX & COST\s+\$?([0-9,]+(?:\.[0-9]{2})?)", amount_line or "")
    acres_match = re.search(r"Acres -\s*([0-9.]+)", acres_line or "")
    account_match = re.search(r"Account:\s*(.+)$", account_line or "")
    sale_sequence_match = re.search(r"Sale Sequence\s+([0-9]+)", sale_sequence_line or "")

    return {
        "source_record_id": f"jackson_{TAX_YEAR}_{record_index:06d}",
        "owner_name": owner_name,
        "account_id": account_match.group(1).strip() if account_match else pd.NA,
        "parcel_id_raw": parcel_match.group(1).strip(),
        "acres_raw": acres_match.group(1) if acres_match else pd.NA,
        "delinquent_amount_raw": amount_match.group(1) if amount_match else pd.NA,
        "sale_sequence": sale_sequence_match.group(1) if sale_sequence_match else pd.NA,
        "sec_township_range": sec_line if sec_line else pd.NA,
        "legal_description": " ".join(legal_lines).strip() or pd.NA,
        "tax_year": TAX_YEAR,
        "delinquent_as_of_date": AS_OF_DATE,
        "raw_text_block": "\n".join(block),
    }


def parse_text(raw_text: str) -> pd.DataFrame:
    blocks = split_blocks(raw_text)
    rows = [parse_block(block, idx) for idx, block in enumerate(blocks, start=1)]
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def coerce_amount(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce").astype("float64")


def standardize_jackson(frame: pd.DataFrame, run_id: str, raw_dir: Path) -> pd.DataFrame:
    owner_name = clean_string(frame["owner_name"])
    parcel_id_raw = clean_string(frame["parcel_id_raw"])
    source_ppin = parcel_id_raw.copy()
    delinquent_amount = coerce_amount(frame["delinquent_amount_raw"])
    acreage = coerce_amount(frame["acres_raw"])

    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash([STATE_CODE, COUNTY_FIPS, SOURCE_NAME, record_id, parcel_id])
                    for record_id, parcel_id in zip(frame["source_record_id"], parcel_id_raw)
                ),
                index=frame.index,
                dtype="string",
            ),
            "parcel_row_id": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "parcel_id_raw": parcel_id_raw,
            "parcel_id_normalized": normalize_identifier(parcel_id_raw),
            "state_code": pd.Series(STATE_CODE, index=frame.index, dtype="string"),
            "county_fips": pd.Series(COUNTY_FIPS, index=frame.index, dtype="string"),
            "county_name": pd.Series(COUNTY_NAME, index=frame.index, dtype="string"),
            "source_name": pd.Series(SOURCE_NAME, index=frame.index, dtype="string"),
            "source_type": pd.Series("direct_download_page", index=frame.index, dtype="string"),
            "source_dataset_path": pd.Series(raw_dir.relative_to(BASE_DIR).as_posix(), index=frame.index, dtype="string"),
            "source_record_id": frame["source_record_id"].astype("string"),
            "source_ppin": source_ppin,
            "ingestion_run_id": pd.Series(run_id, index=frame.index, dtype="string"),
            "source_file_version": pd.Series(str(TAX_YEAR), index=frame.index, dtype="string"),
            "loaded_at": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=frame.index, dtype="string"),
            "owner_name": owner_name,
            "owner_name_2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line1": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_city": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_state": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_zip": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_address": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_city": pd.Series(pd.NA, index=frame.index, dtype="string"),
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
            "tax_year": pd.Series(TAX_YEAR, index=frame.index, dtype="Int64"),
            "bill_year": pd.Series(TAX_YEAR, index=frame.index, dtype="Int64"),
            "tax_amount_due": delinquent_amount,
            "tax_amount_paid": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_balance_due": delinquent_amount,
            "tax_status": pd.Series("preliminary_tax_sale_list", index=frame.index, dtype="string"),
            "payment_status": pd.Series("unpaid", index=frame.index, dtype="string"),
            "delinquent_flag": pd.Series(True, index=frame.index, dtype="boolean"),
            "forfeited_flag": pd.Series(False, index=frame.index, dtype="boolean"),
            "delinquent_amount": delinquent_amount,
            "delinquent_years": pd.Series(str(TAX_YEAR), index=frame.index, dtype="string"),
            "delinquent_as_of_date": pd.Series(AS_OF_DATE, index=frame.index, dtype="string"),
            "last_payment_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "due_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "absentee_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "out_of_state_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "owner_corporate_flag": infer_corporate_owner(owner_name),
            "mailing_matches_situs_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "tax_delinquent_flag_standardized": pd.Series(True, index=frame.index, dtype="boolean"),
            "raw_payload_json": pd.Series(
                frame.astype("string").fillna("").apply(lambda row: json.dumps(row.to_dict(), separators=(",", ":")), axis=1),
                index=frame.index,
                dtype="string",
            ),
        }
    )
    standardized["account_id"] = clean_string(frame["account_id"])
    standardized["sale_sequence"] = clean_string(frame["sale_sequence"])
    standardized["sec_township_range"] = clean_string(frame["sec_township_range"])
    standardized["legal_description"] = clean_string(frame["legal_description"])
    standardized["acreage_raw"] = acreage
    standardized["record_hash"] = build_record_hash(
        standardized,
        ["parcel_id_normalized", "owner_name", "tax_year", "delinquent_amount"],
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


def build_identifier_diagnostics(standardized: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    county_master = master.loc[master["county_name"].eq(COUNTY_NAME)].copy()
    compact_source = standardized["parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    compact_master = county_master["source_parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    direct_overlap = int(standardized["parcel_id_normalized"].isin(set(county_master["source_parcel_id_normalized"].dropna())).sum())
    compact_overlap = int(compact_source.isin(set(compact_master.dropna())).sum())
    ppin_overlap = int(normalize_ppin(standardized["source_ppin"]).isin(set(normalize_ppin(county_master["source_ppin"]).dropna())).sum())
    return pd.DataFrame(
        [
            {"metric": "jackson_rows", "value": int(len(standardized))},
            {"metric": "jackson_unique_parcel_ids", "value": int(standardized["parcel_id_normalized"].nunique())},
            {"metric": "jackson_direct_identifier_overlap_rows", "value": direct_overlap},
            {"metric": "jackson_compact_identifier_overlap_rows", "value": compact_overlap},
            {"metric": "jackson_ppin_overlap_rows", "value": ppin_overlap},
            {
                "metric": "jackson_identifier_assessment",
                "value": "Jackson preliminary list parcel numbers align to county PPIN values, so PPIN linkage is the authoritative join path for this county source.",
            },
        ]
    )


def build_qa_summary(standardized: pd.DataFrame, linked: pd.DataFrame, ambiguous: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "jackson_standardized_rows", "value": int(len(standardized))},
            {"metric": "jackson_linked_rows", "value": int(len(linked))},
            {"metric": "jackson_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "jackson_linked_rate", "value": round(float(len(linked) / len(standardized) * 100.0), 4) if len(standardized) else 0.0},
            {"metric": "jackson_delinquent_amount_nonnull_rate", "value": round(float(standardized["delinquent_amount"].notna().mean() * 100.0), 4) if len(standardized) else 0.0},
        ]
    )


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir)
    paths = source_paths(download_dir)
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    paths["standardized"].parent.mkdir(parents=True, exist_ok=True)
    paths["linked"].parent.mkdir(parents=True, exist_ok=True)

    run_id = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    raw_text = download_text(SOURCE_URL)
    paths["raw_text"].write_text(raw_text, encoding="utf-8")

    raw_frame = parse_text(raw_text)
    if raw_frame.empty:
        raise RuntimeError("No Jackson County tax-sale rows were parsed from the source text.")

    write_json(
        paths["manifest"],
        {
            "source_id": SOURCE_ID,
            "source_name": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "retrieved_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "raw_text": paths["raw_text"].relative_to(BASE_DIR).as_posix(),
            "row_count": int(len(raw_frame)),
            "tax_year": TAX_YEAR,
            "as_of_date": AS_OF_DATE,
        },
    )

    standardized = standardize_jackson(raw_frame, run_id, paths["raw_dir"])
    standardized.to_parquet(paths["standardized"], index=False)

    master = load_master_index(MASTER_PARQUET)
    linked, unmatched, ambiguous, _county_summary = link_standardized_tax_records(
        standardized,
        master,
        heuristic_variants_by_county={COUNTY_NAME: []},
    )
    linked.to_parquet(paths["linked"], index=False)
    unmatched.to_parquet(paths["unmatched"], index=False)
    ambiguous.to_parquet(paths["ambiguous"], index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "jackson_standardized_rows", "value": int(len(standardized))},
            {"metric": "jackson_linked_rows", "value": int(len(linked))},
            {"metric": "jackson_unmatched_rows", "value": int(len(unmatched))},
            {"metric": "jackson_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "jackson_linkage_rate", "value": round(float(len(linked) / len(standardized) * 100.0), 4)},
            {"metric": "jackson_exact_match_rows", "value": int(linked["linkage_method"].isin(["exact_ppin", "exact_normalized_parcel_id"]).sum()) if not linked.empty else 0},
            {"metric": "jackson_heuristic_match_rows", "value": int(linked["linkage_method"].astype("string").str.startswith("heuristic_").sum()) if not linked.empty else 0},
            {"metric": "jackson_total_delinquent_amount", "value": round(float(pd.to_numeric(standardized["delinquent_amount"], errors="coerce").fillna(0.0).sum()), 2)},
        ]
    )
    unmatched_summary, ambiguity_summary = build_reason_summaries(unmatched, ambiguous)
    diagnostics_df = build_identifier_diagnostics(standardized, master)
    qa_df = build_qa_summary(standardized, linked, ambiguous)

    summary_df.to_csv(paths["summary"], index=False)
    unmatched_summary.to_csv(paths["unmatched_reason_summary"], index=False)
    ambiguity_summary.to_csv(paths["ambiguity_reason_summary"], index=False)
    diagnostics_df.to_csv(paths["identifier_diagnostics"], index=False)
    qa_df.to_csv(paths["qa_summary"], index=False)

    downloaded_at = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    update_registry_row(
        REGISTRY_CSV,
        SOURCE_ID,
        downloaded_at,
        f"Downloaded Jackson County preliminary list to {paths['raw_dir'].relative_to(BASE_DIR).as_posix()}.",
    )

    print(f"Jackson rows: {len(standardized):,}")
    print(f"Linked rows: {len(linked):,}")
    print(f"Unmatched rows: {len(unmatched):,}")
    print(f"Ambiguous rows: {len(ambiguous):,}")
    print(f"Summary: {paths['summary'].relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
