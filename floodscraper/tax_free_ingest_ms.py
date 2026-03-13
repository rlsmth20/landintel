from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
    coerce_float,
    coerce_year,
    link_standardized_tax_records,
    load_master_index,
    normalize_identifier,
    normalize_ppin,
    update_registry_row,
    write_json,
)

MASTER_PARQUET = BASE_DIR / "data" / "parcels" / "mississippi_parcels_master.parquet"
REGISTRY_CSV = BASE_DIR / "data" / "tax_metadata" / "tax_source_registry_ms.csv"

PIKE_SOURCE = {
    "state_code": "MS",
    "county_name": "pike",
    "county_fips": "113",
    "source_name": "pike_tax_sale_list",
    "source_url": "https://www.co.pike.ms.us/wp-content/uploads/2023/08/LANDSALE-2023_2.csv",
    "source_page_url": "https://www.co.pike.ms.us/download-tax-sale-list/",
    "file_name": "LANDSALE-2023_2.csv",
    "file_type": "csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest free Mississippi tax sources.")
    parser.add_argument("--download-dir", default=str(RAW_TAX_DIR / "ms"), help="Base raw tax directory.")
    return parser.parse_args()


def download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def source_dirs(download_dir: Path) -> dict[str, Path]:
    run_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    raw_dir = download_dir / PIKE_SOURCE["county_fips"] / PIKE_SOURCE["source_name"] / run_date
    standardized_dir = TAX_STANDARDIZED_DIR / "ms" / PIKE_SOURCE["county_name"]
    linked_dir = TAX_LINKED_DIR / "ms" / PIKE_SOURCE["county_name"]
    return {
        "raw_dir": raw_dir,
        "raw_file": raw_dir / PIKE_SOURCE["file_name"],
        "manifest": raw_dir / "manifest.json",
        "standardized": standardized_dir / "free_tax_records.parquet",
        "linked": linked_dir / "free_linked_tax_records.parquet",
        "unmatched": linked_dir / "free_unmatched_tax_records.parquet",
        "ambiguous": linked_dir / "free_ambiguous_tax_links.parquet",
    }

def standardize_pike(frame: pd.DataFrame, run_id: str, raw_path: Path) -> pd.DataFrame:
    parcel_id_raw = clean_string(frame["PARCEL_NUMBER"])
    parcel_id_normalized = normalize_identifier(parcel_id_raw)
    owner_name = clean_string(frame["NAME"])
    assessed_land_value = coerce_float(frame["LAND_VALUE"])
    assessed_total_value = coerce_float(frame["ASSESSED_TOTAL_VALUE"])
    market_total_value = coerce_float(frame["TRUE_TOTAL_VALUE"])
    delinquent_amount = coerce_float(frame["BALANCE"])
    tax_year = coerce_year(frame["YEAR"])
    source_record_id = clean_string(frame["PPIN"].astype("string"))
    source_ppin = normalize_ppin(frame["PPIN"].astype("string"))

    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash(["MS", "113", "pike_tax_sale_list", record_id, parcel_id])
                    for record_id, parcel_id in zip(source_record_id, parcel_id_normalized)
                ),
                index=frame.index,
                dtype="string",
            ),
            "parcel_row_id": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "parcel_id_raw": parcel_id_raw,
            "parcel_id_normalized": parcel_id_normalized,
            "state_code": pd.Series("MS", index=frame.index, dtype="string"),
            "county_fips": pd.Series("113", index=frame.index, dtype="string"),
            "county_name": pd.Series("pike", index=frame.index, dtype="string"),
            "source_name": pd.Series("pike_tax_sale_list", index=frame.index, dtype="string"),
            "source_type": pd.Series("free_direct_download", index=frame.index, dtype="string"),
            "source_dataset_path": pd.Series(raw_path.relative_to(BASE_DIR).as_posix(), index=frame.index, dtype="string"),
            "source_record_id": source_record_id,
            "source_ppin": source_ppin,
            "ingestion_run_id": pd.Series(run_id, index=frame.index, dtype="string"),
            "source_file_version": pd.Series("2023-08-21", index=frame.index, dtype="string"),
            "loaded_at": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=frame.index, dtype="string"),
            "owner_name": owner_name,
            "owner_name_2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line1": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_city": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_state": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_zip": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_address": clean_string(frame["STR"]),
            "situs_city": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_state": pd.Series("MS", index=frame.index, dtype="string"),
            "situs_zip": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "assessed_land_value": assessed_land_value,
            "assessed_improvement_value": coerce_float(frame["BLDG_VALUE"]),
            "assessed_total_value": assessed_total_value,
            "market_land_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_improvement_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_total_value": market_total_value,
            "taxable_value": assessed_total_value,
            "exemptions_text": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "exemptions_amount": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_year": tax_year,
            "bill_year": tax_year,
            "tax_amount_due": delinquent_amount,
            "tax_amount_paid": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_balance_due": delinquent_amount,
            "tax_status": pd.Series("delinquent_tax_sale_list", index=frame.index, dtype="string"),
            "payment_status": pd.Series("unpaid", index=frame.index, dtype="string"),
            "delinquent_flag": pd.Series(True, index=frame.index, dtype="boolean"),
            "forfeited_flag": pd.Series(False, index=frame.index, dtype="boolean"),
            "delinquent_amount": delinquent_amount,
            "delinquent_years": frame["YEAR"].astype("string"),
            "delinquent_as_of_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "last_payment_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "due_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "absentee_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "out_of_state_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "owner_corporate_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "mailing_matches_situs_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "tax_delinquent_flag_standardized": pd.Series(True, index=frame.index, dtype="boolean"),
            "raw_payload_json": pd.Series(
                frame.astype("string").fillna("").apply(lambda row: json.dumps(row.to_dict(), separators=(",", ":")), axis=1),
                index=frame.index,
                dtype="string",
            ),
        }
    )
    standardized["record_hash"] = build_record_hash(
        standardized,
        [
            "parcel_id_normalized",
            "owner_name",
            "assessed_land_value",
            "assessed_total_value",
            "market_total_value",
            "tax_year",
            "tax_balance_due",
            "tax_status",
        ],
    )
    return standardized


def write_summary(linked: pd.DataFrame, unmatched: pd.DataFrame, ambiguous: pd.DataFrame, standardized: pd.DataFrame) -> Path:
    TAX_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TAX_METADATA_DIR / "tax_free_ingest_summary_ms.csv"
    summary_df = pd.DataFrame(
        [
            {"metric": "pike_raw_rows", "value": int(len(standardized))},
            {"metric": "pike_linked_rows", "value": int(len(linked))},
            {"metric": "pike_unmatched_rows", "value": int(len(unmatched))},
            {"metric": "pike_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "pike_linkage_rate", "value": round(float(len(linked) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "pike_total_delinquent_amount", "value": round(float(pd.to_numeric(standardized["delinquent_amount"], errors="coerce").fillna(0.0).sum()), 2)},
        ]
    )
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    args = parse_args()
    run_id = hashlib.sha1(pd.Timestamp.now("UTC").isoformat().encode("utf-8")).hexdigest()[:12]
    download_dir = Path(args.download_dir)
    paths = source_dirs(download_dir)
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    paths["standardized"].parent.mkdir(parents=True, exist_ok=True)
    paths["linked"].parent.mkdir(parents=True, exist_ok=True)

    raw_bytes = download_bytes(PIKE_SOURCE["source_url"])
    paths["raw_file"].write_bytes(raw_bytes)
    frame = pd.read_csv(paths["raw_file"])
    standardized = standardize_pike(frame, run_id, paths["raw_file"])
    master = load_master_index(MASTER_PARQUET)
    linked, unmatched, ambiguous, _county_summary = link_standardized_tax_records(standardized, master)

    standardized.to_parquet(paths["standardized"], index=False)
    linked.to_parquet(paths["linked"], index=False)
    unmatched.to_parquet(paths["unmatched"], index=False)
    ambiguous.to_parquet(paths["ambiguous"], index=False)
    summary_path = write_summary(linked, unmatched, ambiguous, standardized)

    write_json(
        paths["manifest"],
        {
            "ingestion_run_id": run_id,
            "state_code": PIKE_SOURCE["state_code"],
            "county_name": PIKE_SOURCE["county_name"],
            "county_fips": PIKE_SOURCE["county_fips"],
            "source_name": PIKE_SOURCE["source_name"],
            "source_url": PIKE_SOURCE["source_url"],
            "source_page_url": PIKE_SOURCE["source_page_url"],
            "raw_file_path": paths["raw_file"].relative_to(BASE_DIR).as_posix(),
            "standardized_path": paths["standardized"].relative_to(BASE_DIR).as_posix(),
            "linked_path": paths["linked"].relative_to(BASE_DIR).as_posix(),
            "unmatched_path": paths["unmatched"].relative_to(BASE_DIR).as_posix(),
            "ambiguous_path": paths["ambiguous"].relative_to(BASE_DIR).as_posix(),
            "summary_path": summary_path.relative_to(BASE_DIR).as_posix(),
            "row_count": int(len(frame)),
        },
    )
    update_registry_row(
        REGISTRY_CSV,
        "ms_113_pike_taxsale_direct_download",
        pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        f"Downloaded free CSV to {paths['raw_file'].relative_to(BASE_DIR).as_posix()}.",
    )

    print(f"Downloaded raw Pike CSV: {paths['raw_file'].relative_to(BASE_DIR)}")
    print(f"Standardized rows: {len(standardized):,}")
    print(f"Linked rows: {len(linked):,}")
    print(f"Unmatched rows: {len(unmatched):,}")
    print(f"Ambiguous rows: {len(ambiguous):,}")
    print(f"Summary: {summary_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
