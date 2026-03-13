from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypdf import PdfReader

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
SOURCE_ID = "ms_061_jasper_delinquent_tax_pdfs"
SOURCE_NAME = "jasper_delinquent_tax_pdfs"
SOURCE_URL = "https://co.jasper.ms.us/delinquent-taxes/"
COUNTY_NAME = "jasper"
COUNTY_FIPS = "061"
STATE_CODE = "MS"

JASPER_FILES = [
    {
        "tax_year": 2023,
        "file_name": "2023-Taxes.pdf",
        "source_url": "https://co.jasper.ms.us/wp-content/uploads/2025/09/2023-Taxes.pdf",
        "as_of_date": "2025-09-05",
    },
    {
        "tax_year": 2024,
        "file_name": "2024-Taxes.pdf",
        "source_url": "https://co.jasper.ms.us/wp-content/uploads/2025/09/2024-Taxes.pdf",
        "as_of_date": "2025-09-05",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Jasper County delinquent-tax PDFs.")
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
        "standardized": standardized_dir / "jasper_delinquent_tax_records.parquet",
        "linked": linked_dir / "jasper_linked_tax_records.parquet",
        "unmatched": linked_dir / "jasper_unmatched_tax_records.parquet",
        "ambiguous": linked_dir / "jasper_ambiguous_tax_links.parquet",
        "summary": TAX_METADATA_DIR / "tax_free_jasper_linkage_summary_ms.csv",
        "unmatched_reason_summary": TAX_METADATA_DIR / "tax_free_jasper_unmatched_reason_summary_ms.csv",
        "ambiguity_reason_summary": TAX_METADATA_DIR / "tax_free_jasper_ambiguity_reason_summary_ms.csv",
        "identifier_diagnostics": TAX_METADATA_DIR / "tax_free_jasper_identifier_diagnostics_ms.csv",
        "qa_summary": TAX_METADATA_DIR / "tax_free_jasper_qa_summary_ms.csv",
    }


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        normalized = normalize_whitespace(line)
        if not normalized:
            continue
        if normalized.startswith("JESSICA ") or normalized.startswith("Parcels Sold and Not Redeemed") or normalized.startswith("Tax Year ") or normalized.startswith("Parcel Option ") or normalized.startswith("PPIN "):
            continue
        if set(normalized) == {"-"}:
            continue
        if re.match(r"^\d{4,6}(?:\s+\d+)?\s+\S", normalized):
            if current:
                blocks.append(current)
            current = [normalized]
        else:
            if current:
                current.append(normalized)
    if current:
        blocks.append(current)
    return blocks


def parse_first_line(line: str) -> dict[str, Any]:
    tokens = line.split()
    ppin = tokens[0]
    suffix = None
    start_index = 1
    if len(tokens) > 1 and re.fullmatch(r"\d+", tokens[1]) and len(tokens[1]) <= 2:
        suffix = tokens[1]
        start_index = 2
    numeric_pattern = re.compile(r"^\d+(?:\.\d+)?$|^\.\d+$")
    end_index = len(tokens)
    while end_index > start_index and numeric_pattern.fullmatch(tokens[end_index - 1]):
        end_index -= 1
    owner_name = " ".join(tokens[start_index:end_index]).strip()
    numeric_tail = tokens[end_index:]

    acres = pd.NA
    assessed_total_value = pd.NA
    if len(numeric_tail) >= 2:
        acres = numeric_tail[0]
        assessed_total_value = numeric_tail[1]
    return {
        "ppin_raw": ppin,
        "record_suffix": suffix,
        "owner_name": owner_name or pd.NA,
        "amount_due_raw": pd.NA,
        "excess_bid_raw": pd.NA,
        "acres_raw": acres,
        "assessed_total_value_raw": assessed_total_value,
        "numeric_tail_tokens": "|".join(numeric_tail) if numeric_tail else pd.NA,
    }


def extract_decimal_tokens(line: str) -> list[str]:
    return re.findall(r"(?<!\d)(?:\d+\.\d+|\.\d+)(?!\d)", line)


def detect_summed_amount(tokens: list[str]) -> str | None:
    values = [float(token) for token in tokens]
    for idx in range(len(values) - 1, 1, -1):
        if idx >= 2 and abs((values[idx - 1] + values[idx - 2]) - values[idx]) < 0.02:
            return tokens[idx]
        if idx >= 3 and abs((values[idx - 1] + values[idx - 2] + values[idx - 3]) - values[idx]) < 0.02:
            return tokens[idx]
    return None


def choose_structured_amount(
    city_state_zip_line: str | None,
    header_detail_line: str | None,
    recovery_lines: list[str],
    acres_raw: Any,
    assessed_total_value_raw: Any,
) -> dict[str, Any]:
    parser_flags: list[str] = []
    parser_evidence: list[str] = []

    acres_val = pd.to_numeric(pd.Series([acres_raw]), errors="coerce").iloc[0]
    assessed_val = pd.to_numeric(pd.Series([assessed_total_value_raw]), errors="coerce").iloc[0]

    line_candidates: list[tuple[str, list[str], str]] = []
    if header_detail_line:
        line_candidates.append(("header_detail_line", extract_decimal_tokens(header_detail_line), header_detail_line))
    if city_state_zip_line:
        line_candidates.append(("city_state_zip_line", extract_decimal_tokens(city_state_zip_line), city_state_zip_line))

    chosen_amount = pd.NA
    chosen_source = pd.NA
    recovery_method = pd.NA
    raw_tax_line = header_detail_line or city_state_zip_line or pd.NA

    preferred = [item for item in line_candidates if item[0] == "header_detail_line"]
    fallback = [item for item in line_candidates if item[0] == "city_state_zip_line"]

    for source_name, tokens, source_line in preferred + fallback:
        if not tokens:
            continue
        token = tokens[-1]
        value = float(token)
        if pd.notna(acres_val) and abs(value - float(acres_val)) < 1e-9:
            parser_flags.append("rejected_amount_equal_to_acres")
            parser_evidence.append(f"{source_name}:{token}")
            continue
        if pd.notna(assessed_val) and assessed_val > 0 and value > assessed_val * 50:
            parser_flags.append("rejected_amount_implausible_vs_assessed")
            parser_evidence.append(f"{source_name}:{token}")
            continue
        chosen_amount = token
        chosen_source = source_name
        recovery_method = f"{source_name}_last_decimal"
        break

    if pd.isna(chosen_amount):
        for line in recovery_lines:
            tokens = extract_decimal_tokens(line)
            if len(tokens) < 3:
                continue
            token = detect_summed_amount(tokens)
            if token is None:
                continue
            value = float(token)
            if pd.notna(acres_val) and abs(value - float(acres_val)) < 1e-9:
                parser_flags.append("rejected_amount_equal_to_acres")
                parser_evidence.append(f"decimal_sum_pattern_line:{token}")
                continue
            if pd.notna(assessed_val) and assessed_val > 0 and value > assessed_val * 50:
                parser_flags.append("rejected_amount_implausible_vs_assessed")
                parser_evidence.append(f"decimal_sum_pattern_line:{token}")
                continue
            chosen_amount = token
            chosen_source = "decimal_sum_pattern_line"
            recovery_method = "decimal_sum_pattern_line"
            raw_tax_line = line
            parser_flags.append("used_decimal_sum_pattern_line")
            break

    if pd.isna(chosen_amount):
        parser_flags.append("missing_structured_decimal_amount")

    if header_detail_line:
        parser_flags.append("used_header_detail_line" if pd.notna(chosen_amount) and chosen_source == "header_detail_line" else "header_detail_line_present")
    elif city_state_zip_line and pd.notna(chosen_amount):
        parser_flags.append("used_city_state_zip_line")

    return {
        "amount_due_raw": chosen_amount,
        "amount_source_line": chosen_source,
        "raw_tax_detail_line": raw_tax_line,
        "recovery_method": recovery_method,
        "parser_qa_flags": "|".join(dict.fromkeys(parser_flags)) if parser_flags else pd.NA,
        "parser_qa_evidence": "|".join(dict.fromkeys(parser_evidence)) if parser_evidence else pd.NA,
    }


def parse_block(block: list[str], tax_year: int, as_of_date: str, record_index: int) -> dict[str, Any] | None:
    first = parse_first_line(block[0])
    parcel_line = next((line for line in block if "PARCEL-" in line.upper()), None)
    if parcel_line is None:
        return None
    parcel_match = re.search(r"PARCEL-([A-Z0-9\-\. ]+)", parcel_line.upper())
    if parcel_match is None:
        return None
    parcel_id_raw = parcel_match.group(1).strip()
    sold_to_line = next((line for line in block if line.startswith("Sold To:")), None)

    address_lines: list[str] = []
    legal_lines: list[str] = []
    city_state_zip = None
    header_line_index = None
    header_detail_line = None
    recovery_lines: list[str] = []
    seen_legal = False
    for idx, line in enumerate(block[1:], start=1):
        if line.startswith("Sold To:"):
            continue
        if "PARCEL-" in line.upper():
            continue
        if line.startswith("Total Assessed/ Reg 100 Taxes Printers Amount Amount Excess"):
            header_line_index = idx
            if idx + 1 < len(block):
                header_detail_line = block[idx + 1]
            continue
        if line.startswith("SEC-") or line.startswith("Deed Bk") or line.startswith("DB "):
            continue
        if not seen_legal:
            recovery_lines.append(line)
        if re.search(r"\bMS\b\s+\d{5}", line) or re.search(r"\b[A-Z]{2}\b\s+\d{5}", line):
            city_state_zip = line
            address_lines.append(line)
            continue
        if not seen_legal and re.match(r"^\d+\s+", line):
            address_lines.append(line)
            continue
        seen_legal = True
        legal_lines.append(line)

    chosen_amount = choose_structured_amount(
        city_state_zip_line=city_state_zip,
        header_detail_line=header_detail_line,
        recovery_lines=recovery_lines,
        acres_raw=first["acres_raw"],
        assessed_total_value_raw=first["assessed_total_value_raw"],
    )

    return {
        "source_record_id": f"jasper_{tax_year}_{record_index:06d}",
        "tax_year": tax_year,
        "delinquent_as_of_date": as_of_date,
        "parcel_id_raw": parcel_id_raw,
        "source_ppin": first["ppin_raw"],
        "record_suffix": first["record_suffix"],
        "owner_name": first["owner_name"],
        "owner_address_line1": address_lines[0] if address_lines else pd.NA,
        "owner_address_line2": city_state_zip if city_state_zip and (not address_lines or city_state_zip != address_lines[0]) else pd.NA,
        "legal_description": " ".join(legal_lines).strip() or pd.NA,
        "sold_to": sold_to_line.replace("Sold To:", "").strip() if sold_to_line else pd.NA,
        "amount_due_raw": chosen_amount["amount_due_raw"],
        "excess_bid_raw": first["excess_bid_raw"],
        "acres_raw": first["acres_raw"],
        "assessed_total_value_raw": first["assessed_total_value_raw"],
        "amount_source_line": chosen_amount["amount_source_line"],
        "raw_tax_detail_line": chosen_amount["raw_tax_detail_line"],
        "recovery_method": chosen_amount["recovery_method"],
        "numeric_tail_tokens": first["numeric_tail_tokens"],
        "parser_qa_flags": chosen_amount["parser_qa_flags"],
        "parser_qa_evidence": chosen_amount["parser_qa_evidence"],
        "raw_text_block": "\n".join(block),
    }


def parse_jasper_pdf(pdf_path: Path, tax_year: int, as_of_date: str) -> pd.DataFrame:
    text = extract_pdf_text(pdf_path)
    blocks = split_blocks(text)
    rows = [parse_block(block, tax_year, as_of_date, idx) for idx, block in enumerate(blocks, start=1)]
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def coerce_amount(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.replace(",", "", regex=False), errors="coerce").astype("float64")


def standardize_jasper(frame: pd.DataFrame, run_id: str, raw_dir: Path) -> pd.DataFrame:
    owner_name = clean_string(frame["owner_name"])
    owner_add1 = clean_string(frame["owner_address_line1"])
    owner_add2 = clean_string(frame["owner_address_line2"])
    owner_city = owner_add2.astype("string").str.extract(r"^(.+?)\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?$", expand=False).astype("string")
    owner_state = owner_add2.astype("string").str.extract(r"\b([A-Z]{2})\b", expand=False).astype("string")
    owner_zip = owner_add2.astype("string").str.extract(r"(\d{5}(?:-\d{4})?)", expand=False).astype("string")
    parcel_id_raw = clean_string(frame["parcel_id_raw"])
    delinquent_amount = coerce_amount(frame["amount_due_raw"])
    assessed_total_value = coerce_amount(frame["assessed_total_value_raw"])
    acreage = coerce_amount(frame["acres_raw"])

    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash([STATE_CODE, COUNTY_FIPS, SOURCE_NAME, record_id, parcel_id, tax_year])
                    for record_id, parcel_id, tax_year in zip(frame["source_record_id"], parcel_id_raw, frame["tax_year"])
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
            "source_ppin": normalize_ppin(frame["source_ppin"]),
            "ingestion_run_id": pd.Series(run_id, index=frame.index, dtype="string"),
            "source_file_version": frame["tax_year"].astype("string"),
            "loaded_at": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=frame.index, dtype="string"),
            "owner_name": owner_name,
            "owner_name_2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line1": owner_add1,
            "owner_address_line2": owner_add2,
            "owner_city": owner_city,
            "owner_state": owner_state,
            "owner_zip": owner_zip,
            "situs_address": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_city": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_state": pd.Series("MS", index=frame.index, dtype="string"),
            "situs_zip": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "assessed_land_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "assessed_improvement_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "assessed_total_value": assessed_total_value,
            "market_land_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_improvement_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_total_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "taxable_value": assessed_total_value,
            "exemptions_text": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "exemptions_amount": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_year": pd.to_numeric(frame["tax_year"], errors="coerce").astype("Int64"),
            "bill_year": pd.to_numeric(frame["tax_year"], errors="coerce").astype("Int64"),
            "tax_amount_due": delinquent_amount,
            "tax_amount_paid": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_balance_due": delinquent_amount,
            "tax_status": pd.Series("sold_not_redeemed_delinquent_taxes", index=frame.index, dtype="string"),
            "payment_status": pd.Series("unpaid", index=frame.index, dtype="string"),
            "delinquent_flag": pd.Series(True, index=frame.index, dtype="boolean"),
            "forfeited_flag": frame["sold_to"].astype("string").str.contains("STATE OF MISSISSIPPI", case=False, na=False).astype("boolean"),
            "delinquent_amount": delinquent_amount,
            "delinquent_years": frame["tax_year"].astype("string"),
            "delinquent_as_of_date": frame["delinquent_as_of_date"].astype("string"),
            "last_payment_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "due_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "absentee_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "out_of_state_owner_flag": owner_state.ne("MS").astype("boolean"),
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
    standardized["legal_description"] = clean_string(frame["legal_description"])
    standardized["acreage_raw"] = acreage
    standardized["sold_to"] = clean_string(frame["sold_to"])
    standardized["amount_source_line"] = clean_string(frame["amount_source_line"])
    standardized["raw_tax_detail_line"] = clean_string(frame["raw_tax_detail_line"])
    standardized["recovery_method"] = clean_string(frame["recovery_method"])
    standardized["numeric_tail_tokens"] = clean_string(frame["numeric_tail_tokens"])
    standardized["parser_qa_flags"] = clean_string(frame["parser_qa_flags"])
    standardized["parser_qa_evidence"] = clean_string(frame["parser_qa_evidence"])
    standardized["record_hash"] = build_record_hash(
        standardized,
        ["parcel_id_normalized", "source_ppin", "owner_name", "tax_year", "delinquent_amount"],
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
            {"metric": "jasper_rows", "value": int(len(standardized))},
            {"metric": "jasper_unique_parcel_ids", "value": int(standardized["parcel_id_normalized"].nunique())},
            {"metric": "jasper_direct_identifier_overlap_rows", "value": direct_overlap},
            {"metric": "jasper_compact_identifier_overlap_rows", "value": compact_overlap},
            {"metric": "jasper_ppin_overlap_rows", "value": ppin_overlap},
        ]
    )


def build_qa_summary(standardized: pd.DataFrame, linked: pd.DataFrame, ambiguous: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "jasper_standardized_rows", "value": int(len(standardized))},
            {"metric": "jasper_linked_rows", "value": int(len(linked))},
            {"metric": "jasper_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "jasper_linked_rate", "value": round(float(len(linked) / len(standardized) * 100.0), 4) if len(standardized) else 0.0},
            {"metric": "jasper_delinquent_amount_nonnull_rate", "value": round(float(standardized["delinquent_amount"].notna().mean() * 100.0), 4) if len(standardized) else 0.0},
            {"metric": "jasper_parser_qa_flagged_rows", "value": int(standardized["parser_qa_flags"].astype("string").notna().sum()) if "parser_qa_flags" in standardized.columns else 0},
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
    manifest_files: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for file_cfg in JASPER_FILES:
        pdf_path = paths["raw_dir"] / str(file_cfg["file_name"])
        pdf_path.write_bytes(download_bytes(str(file_cfg["source_url"])))
        manifest_files.append(
            {
                "tax_year": int(file_cfg["tax_year"]),
                "source_url": str(file_cfg["source_url"]),
                "raw_pdf": pdf_path.relative_to(BASE_DIR).as_posix(),
                "as_of_date": str(file_cfg["as_of_date"]),
            }
        )
        frames.append(parse_jasper_pdf(pdf_path, int(file_cfg["tax_year"]), str(file_cfg["as_of_date"])))

    raw_frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw_frame.empty:
        raise RuntimeError("No Jasper delinquent-tax rows were parsed from the source PDFs.")

    write_json(
        paths["manifest"],
        {
            "source_id": SOURCE_ID,
            "source_name": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "retrieved_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": manifest_files,
            "row_count": int(len(raw_frame)),
        },
    )

    standardized = standardize_jasper(raw_frame, run_id, paths["raw_dir"])
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
            {"metric": "jasper_standardized_rows", "value": int(len(standardized))},
            {"metric": "jasper_linked_rows", "value": int(len(linked))},
            {"metric": "jasper_unmatched_rows", "value": int(len(unmatched))},
            {"metric": "jasper_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "jasper_linkage_rate", "value": round(float(len(linked) / len(standardized) * 100.0), 4)},
            {"metric": "jasper_exact_match_rows", "value": int(linked["linkage_method"].isin(["exact_ppin", "exact_normalized_parcel_id"]).sum()) if not linked.empty else 0},
            {"metric": "jasper_heuristic_match_rows", "value": int(linked["linkage_method"].astype("string").str.startswith("heuristic_").sum()) if not linked.empty else 0},
            {"metric": "jasper_total_delinquent_amount", "value": round(float(pd.to_numeric(standardized["delinquent_amount"], errors="coerce").fillna(0.0).sum()), 2)},
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
        f"Downloaded Jasper delinquent-tax PDFs to {paths['raw_dir'].relative_to(BASE_DIR).as_posix()}.",
    )

    print(f"Jasper rows: {len(standardized):,}")
    print(f"Linked rows: {len(linked):,}")
    print(f"Unmatched rows: {len(unmatched):,}")
    print(f"Ambiguous rows: {len(ambiguous):,}")
    print(f"Summary: {paths['summary'].relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
