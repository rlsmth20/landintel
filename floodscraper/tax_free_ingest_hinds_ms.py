from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

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
SOURCE_ID = "ms_049_hinds_taxsale_direct_download"
SOURCE_NAME = "hinds_tax_sale_files"
COUNTY_NAME = "hinds"
COUNTY_FIPS = "049"
STATE_CODE = "MS"
TAX_YEAR = 2025

HINDS_FILES = [
    {
        "district": "district_1",
        "file_name": "RPTS12-PRINT1-2025-dist1.xlsx",
        "source_url": "https://www.co.hinds.ms.us/download/RPTS12-PRINT1-2025-dist1.xlsx",
    },
    {
        "district": "district_2",
        "file_name": "RPTS12-PRINT1-2025-dist2.xlsx",
        "source_url": "https://www.co.hinds.ms.us/download/RPTS12-PRINT1-2025-dist2.xlsx",
    },
]

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Hinds County free structured tax sale workbooks.")
    parser.add_argument("--download-dir", default=str(RAW_TAX_DIR / "ms"), help="Base raw tax directory.")
    return parser.parse_args()


def download_bytes(url: str) -> bytes:
    completed = subprocess.run(
        ["curl.exe", "-L", "--fail", url],
        check=True,
        capture_output=True,
        text=False,
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
        "manifest": raw_dir / "manifest.json",
        "standardized": standardized_dir / "hinds_tax_sale_records.parquet",
        "linked": linked_dir / "hinds_linked_tax_records.parquet",
        "unmatched": linked_dir / "hinds_unmatched_tax_records.parquet",
        "ambiguous": linked_dir / "hinds_ambiguous_tax_links.parquet",
        "summary": TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv",
        "unmatched_reason_summary": TAX_METADATA_DIR / "tax_free_hinds_unmatched_reason_summary_ms.csv",
        "ambiguity_reason_summary": TAX_METADATA_DIR / "tax_free_hinds_ambiguity_reason_summary_ms.csv",
        "identifier_diagnostics": TAX_METADATA_DIR / "tax_free_hinds_identifier_diagnostics_ms.csv",
        "qa_summary": TAX_METADATA_DIR / "tax_free_hinds_qa_summary_ms.csv",
        "comparison_summary": TAX_METADATA_DIR / "tax_free_hinds_vs_sos_comparison_ms.csv",
        "free_statewide_summary": TAX_METADATA_DIR / "tax_free_statewide_source_summary_ms.csv",
    }


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", NS)
    if value_node is None:
        inline = cell.find("a:is", NS)
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.findall(".//a:t", NS))
    value = value_node.text or ""
    if cell_type == "s" and value.isdigit():
        index = int(value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
    return value


def read_hinds_workbook(path: Path, district_label: str) -> pd.DataFrame:
    with ZipFile(path) as workbook:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for entry in shared_root.findall("a:si", NS):
                shared_strings.append("".join(node.text or "" for node in entry.findall(".//a:t", NS)))
        sheet_root = ET.fromstring(workbook.read("xl/worksheets/sheet.xml"))
        rows: list[dict[str, str]] = []
        for row in sheet_root.find("a:sheetData", NS).findall("a:row", NS):
            row_map: dict[str, str] = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in ref if ch.isalpha())
                row_map[col] = _cell_text(cell, shared_strings)
            rows.append(row_map)
    ordered_cols = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    matrix = [[row.get(col, "") for col in ordered_cols] for row in rows]
    frame = pd.DataFrame(matrix[1:], columns=matrix[0])
    frame["district_label"] = district_label
    return frame


def normalize_hinds_parcel_id(series: pd.Series) -> pd.Series:
    cleaned = clean_string(series)
    return cleaned.astype("string")


def standardize_hinds(frame: pd.DataFrame, run_id: str, raw_dir: Path) -> pd.DataFrame:
    parcel_id_raw = normalize_hinds_parcel_id(frame["Parcel_Number"])
    owner_name = clean_string(frame["Owner_Name"])
    owner_add1 = clean_string(frame["Owner_Add1"])
    owner_add2 = clean_string(frame["Owner_Add2"])
    owner_add3 = clean_string(frame["Owner_Add3"])
    legal1 = clean_string(frame["Legal1"])
    legal2 = clean_string(frame["Legal2"])
    legal3 = clean_string(frame["Legal3"])
    situs_address = clean_string(frame["Property_Location"])

    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash([STATE_CODE, COUNTY_FIPS, SOURCE_NAME, district, record_id, parcel_id])
                    for district, record_id, parcel_id in zip(
                        frame["district_label"].astype("string"),
                        pd.Series(range(1, len(frame) + 1), index=frame.index).astype("string"),
                        parcel_id_raw,
                    )
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
            "source_record_id": pd.Series(
                [f"{district}:{idx}" for district, idx in zip(frame["district_label"], range(1, len(frame) + 1))],
                index=frame.index,
                dtype="string",
            ),
            "source_ppin": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "ingestion_run_id": pd.Series(run_id, index=frame.index, dtype="string"),
            "source_file_version": pd.Series(str(TAX_YEAR), index=frame.index, dtype="string"),
            "loaded_at": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=frame.index, dtype="string"),
            "owner_name": owner_name,
            "owner_name_2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line1": owner_add1,
            "owner_address_line2": owner_add2,
            "owner_city": owner_add3.astype("string").str.extract(r"^(.+?)\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?$", expand=False).astype("string"),
            "owner_state": owner_add3.astype("string").str.extract(r"\b([A-Z]{2})\b", expand=False).astype("string"),
            "owner_zip": owner_add3.astype("string").str.extract(r"(\d{5}(?:-\d{4})?)", expand=False).astype("string"),
            "situs_address": situs_address,
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
            "tax_amount_due": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_amount_paid": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_balance_due": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_status": pd.Series("advertised_tax_sale_list", index=frame.index, dtype="string"),
            "payment_status": pd.Series("unpaid", index=frame.index, dtype="string"),
            "delinquent_flag": pd.Series(True, index=frame.index, dtype="boolean"),
            "forfeited_flag": pd.Series(False, index=frame.index, dtype="boolean"),
            "delinquent_amount": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "delinquent_years": pd.Series(str(TAX_YEAR), index=frame.index, dtype="string"),
            "delinquent_as_of_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "last_payment_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "due_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "absentee_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "out_of_state_owner_flag": owner_add3.astype("string").str.extract(r"\b([A-Z]{2})\b", expand=False).astype("string").ne("MS").astype("boolean"),
            "owner_corporate_flag": infer_corporate_owner(owner_name),
            "mailing_matches_situs_flag": owner_add1.eq(situs_address).astype("boolean"),
            "tax_delinquent_flag_standardized": pd.Series(True, index=frame.index, dtype="boolean"),
            "raw_payload_json": pd.Series(
                frame.astype("string").fillna("").apply(lambda row: json.dumps(row.to_dict(), separators=(",", ":")), axis=1),
                index=frame.index,
                dtype="string",
            ),
        }
    )
    standardized["legal_description"] = clean_string(legal1.fillna("") + " " + legal2.fillna("") + " " + legal3.fillna(""))
    standardized["record_hash"] = build_record_hash(
        standardized,
        ["parcel_id_normalized", "owner_name", "owner_address_line1", "situs_address", "tax_year", "tax_status"],
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
    std = standardized.copy()
    master_hinds = master.loc[master["county_name"].eq(COUNTY_NAME)].copy()
    std["compact_source"] = std["parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    master_hinds["compact_master"] = master_hinds["source_parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    compact_overlap = int(std["compact_source"].isin(set(master_hinds["compact_master"].dropna())).sum())
    direct_overlap = int(std["parcel_id_normalized"].isin(set(master_hinds["source_parcel_id_normalized"].dropna())).sum())
    diagnostics = pd.DataFrame(
        [
            {"metric": "hinds_structured_rows", "value": int(len(std))},
            {"metric": "hinds_structured_unique_parcel_ids", "value": int(std["parcel_id_normalized"].nunique())},
            {"metric": "hinds_structured_null_parcel_id_rows", "value": int(std["parcel_id_normalized"].isna().sum())},
            {"metric": "hinds_structured_null_ppin_rows", "value": int(std["source_ppin"].isna().sum())},
            {"metric": "hinds_structured_duplicate_parcel_rows", "value": int(std["parcel_id_normalized"].duplicated(keep=False).fillna(False).sum())},
            {"metric": "hinds_master_unique_parcel_ids", "value": int(master_hinds["source_parcel_id_normalized"].nunique())},
            {"metric": "hinds_master_unique_ppins", "value": int(normalize_ppin(master_hinds["source_ppin"]).nunique())},
            {"metric": "hinds_direct_identifier_overlap_rows", "value": direct_overlap},
            {"metric": "hinds_compact_identifier_overlap_rows", "value": compact_overlap},
            {
                "metric": "hinds_safe_adapter_justified",
                "value": "no" if compact_overlap == 0 and direct_overlap == 0 else "review",
            },
            {
                "metric": "hinds_adapter_assessment",
                "value": "No safe county adapter justified from current structured file; source parcel numbers do not align with parcel master IDs and PPIN is absent.",
            },
        ]
    )
    return diagnostics


def build_qa_summary(standardized: pd.DataFrame, linked: pd.DataFrame, unmatched: pd.DataFrame, ambiguous: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "row_count", "value": int(len(standardized))},
            {"metric": "null_parcel_id_rate", "value": round(float(standardized["parcel_id_normalized"].isna().mean() * 100.0), 4)},
            {"metric": "null_ppin_rate", "value": round(float(standardized["source_ppin"].isna().mean() * 100.0), 4)},
            {
                "metric": "delinquent_amount_parse_nonnull_rate",
                "value": round(float(standardized["delinquent_amount"].notna().mean() * 100.0), 4),
            },
            {
                "metric": "duplicate_record_rate",
                "value": round(float(standardized["record_hash"].duplicated(keep=False).mean() * 100.0), 4),
            },
            {"metric": "linked_rate", "value": round(float(len(linked) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "exact_match_rate", "value": round(float(linked["match_confidence_tier"].eq("high").mean() * 100.0), 4) if not linked.empty else 0.0},
            {"metric": "heuristic_match_rate", "value": round(float(linked["match_confidence_tier"].eq("low").mean() * 100.0), 4) if not linked.empty else 0.0},
            {"metric": "unmatched_rate", "value": round(float(len(unmatched) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "ambiguous_rate", "value": round(float(len(ambiguous) / max(len(standardized), 1) * 100.0), 4)},
        ]
    )


def build_comparison_summary(standardized: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    sos_summary_path = TAX_METADATA_DIR / "tax_free_sos_county_summary_ms.csv"
    sos_county = pd.read_csv(sos_summary_path)
    sos_hinds = sos_county.loc[sos_county["county_name"].eq(COUNTY_NAME)].copy()
    sos_linked = int(sos_hinds["linked_rows"].iloc[0]) if not sos_hinds.empty else 0
    sos_rows = int(sos_hinds["standardized_rows"].iloc[0]) if not sos_hinds.empty else 0
    sos_rate = float(sos_hinds["linkage_rate"].iloc[0]) if not sos_hinds.empty else 0.0
    hinds_rate = float(len(linked) / max(len(standardized), 1) * 100.0)
    return pd.DataFrame(
        [
            {"metric": "hinds_structured_rows", "value": int(len(standardized))},
            {"metric": "hinds_structured_linked_rows", "value": int(len(linked))},
            {"metric": "hinds_structured_linkage_rate", "value": round(hinds_rate, 4)},
            {"metric": "sos_hinds_rows", "value": sos_rows},
            {"metric": "sos_hinds_linked_rows", "value": sos_linked},
            {"metric": "sos_hinds_linkage_rate", "value": round(sos_rate, 4)},
            {
                "metric": "comparison_assessment",
                "value": "Hinds structured source is cleaner as a county-hosted delinquency list, but without PPIN or master-aligned parcel IDs it does not currently improve linkage over SOS for Hinds.",
            },
        ]
    )


def update_free_statewide_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    source_files = [
        ("pike", TAX_METADATA_DIR / "tax_free_ingest_summary_ms.csv"),
        ("sos_statewide", TAX_METADATA_DIR / "tax_free_sos_ingest_summary_ms.csv"),
        ("hinds", TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv"),
    ]
    metric_map = {
        "pike": {
            "standardized_rows": "pike_raw_rows",
            "linked_rows": "pike_linked_rows",
            "unmatched_rows": "pike_unmatched_rows",
            "ambiguous_rows": "pike_ambiguous_rows",
            "linkage_rate": "pike_linkage_rate",
        },
        "sos_statewide": {
            "standardized_rows": "sos_standardized_rows",
            "linked_rows": "sos_linked_rows",
            "unmatched_rows": "sos_unmatched_rows",
            "ambiguous_rows": "sos_ambiguous_rows",
            "linkage_rate": "sos_linkage_rate",
        },
        "hinds": {
            "standardized_rows": "hinds_standardized_rows",
            "linked_rows": "hinds_linked_rows",
            "unmatched_rows": "hinds_unmatched_rows",
            "ambiguous_rows": "hinds_ambiguous_rows",
            "linkage_rate": "hinds_linkage_rate",
        },
    }
    for source_name, path in source_files:
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        if "metric" not in summary.columns:
            continue
        lookup = dict(zip(summary["metric"].astype("string"), summary["value"]))
        keys = metric_map[source_name]
        rows.append(
            {
                "source_name": source_name,
                "standardized_rows": lookup.get(keys["standardized_rows"], pd.NA),
                "linked_rows": lookup.get(keys["linked_rows"], pd.NA),
                "unmatched_rows": lookup.get(keys["unmatched_rows"], pd.NA),
                "ambiguous_rows": lookup.get(keys["ambiguous_rows"], pd.NA),
                "linkage_rate": lookup.get(keys["linkage_rate"], pd.NA),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_id = hashlib.sha1(pd.Timestamp.now("UTC").isoformat().encode("utf-8")).hexdigest()[:12]
    paths = source_paths(Path(args.download_dir))
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
    paths["standardized"].parent.mkdir(parents=True, exist_ok=True)
    paths["linked"].parent.mkdir(parents=True, exist_ok=True)

    raw_files: list[str] = []
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for file_info in HINDS_FILES:
        raw_path = paths["raw_dir"] / file_info["file_name"]
        raw_bytes = download_bytes(file_info["source_url"])
        raw_path.write_bytes(raw_bytes)
        raw_files.append(raw_path.relative_to(BASE_DIR).as_posix())
        frame = read_hinds_workbook(raw_path, file_info["district"])
        frame["source_file_name"] = file_info["file_name"]
        frames.append(frame)
        manifests.append(
            {
                "district": file_info["district"],
                "file_name": file_info["file_name"],
                "source_url": file_info["source_url"],
                "row_count": int(len(frame)),
                "columns": frame.columns.tolist(),
            }
        )
    raw_frame = pd.concat(frames, ignore_index=True)
    standardized = standardize_hinds(raw_frame, run_id, paths["raw_dir"])
    master = load_master_index(MASTER_PARQUET)
    linked, unmatched, ambiguous, _county_summary = link_standardized_tax_records(
        standardized,
        master,
        heuristic_variants_by_county={COUNTY_NAME: []},
    )

    standardized.to_parquet(paths["standardized"], index=False)
    linked.to_parquet(paths["linked"], index=False)
    unmatched.to_parquet(paths["unmatched"], index=False)
    ambiguous.to_parquet(paths["ambiguous"], index=False)

    summary = pd.DataFrame(
        [
            {"metric": "hinds_standardized_rows", "value": int(len(standardized))},
            {"metric": "hinds_linked_rows", "value": int(len(linked))},
            {"metric": "hinds_unmatched_rows", "value": int(len(unmatched))},
            {"metric": "hinds_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "hinds_linkage_rate", "value": round(float(len(linked) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "hinds_exact_match_rows", "value": int(linked["match_confidence_tier"].eq("high").sum()) if not linked.empty else 0},
            {"metric": "hinds_heuristic_match_rows", "value": int(linked["match_confidence_tier"].eq("low").sum()) if not linked.empty else 0},
        ]
    )
    summary.to_csv(paths["summary"], index=False)

    unmatched_reason_summary, ambiguity_reason_summary = build_reason_summaries(unmatched, ambiguous)
    unmatched_reason_summary.to_csv(paths["unmatched_reason_summary"], index=False)
    ambiguity_reason_summary.to_csv(paths["ambiguity_reason_summary"], index=False)
    diagnostics = build_identifier_diagnostics(standardized, master)
    diagnostics.to_csv(paths["identifier_diagnostics"], index=False)
    qa_summary = build_qa_summary(standardized, linked, unmatched, ambiguous)
    qa_summary.to_csv(paths["qa_summary"], index=False)
    comparison_summary = build_comparison_summary(standardized, linked)
    comparison_summary.to_csv(paths["comparison_summary"], index=False)
    free_summary = update_free_statewide_summary()
    free_summary.to_csv(paths["free_statewide_summary"], index=False)

    write_json(
        paths["manifest"],
        {
            "ingestion_run_id": run_id,
            "state_code": STATE_CODE,
            "county_name": COUNTY_NAME,
            "county_fips": COUNTY_FIPS,
            "source_name": SOURCE_NAME,
            "source_files": manifests,
            "raw_file_paths": raw_files,
            "standardized_path": paths["standardized"].relative_to(BASE_DIR).as_posix(),
            "linked_path": paths["linked"].relative_to(BASE_DIR).as_posix(),
            "unmatched_path": paths["unmatched"].relative_to(BASE_DIR).as_posix(),
            "ambiguous_path": paths["ambiguous"].relative_to(BASE_DIR).as_posix(),
            "summary_path": paths["summary"].relative_to(BASE_DIR).as_posix(),
            "qa_summary_path": paths["qa_summary"].relative_to(BASE_DIR).as_posix(),
            "comparison_summary_path": paths["comparison_summary"].relative_to(BASE_DIR).as_posix(),
        },
    )
    update_registry_row(
        REGISTRY_CSV,
        SOURCE_ID,
        pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        f"Downloaded structured Hinds XLSX files to {paths['raw_dir'].relative_to(BASE_DIR).as_posix()}.",
    )

    print(f"Hinds structured rows: {len(standardized):,}")
    print(f"Linked rows: {len(linked):,}")
    print(f"Unmatched rows: {len(unmatched):,}")
    print(f"Ambiguous rows: {len(ambiguous):,}")
    print(f"Summary: {paths['summary'].relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
