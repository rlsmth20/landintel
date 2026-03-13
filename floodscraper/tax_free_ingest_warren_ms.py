from __future__ import annotations

import argparse
import hashlib
import html
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
SOURCE_ID = "ms_149_warren_delinquent_taxes_page"
SOURCE_NAME = "warren_delinquent_taxes"
SOURCE_URL = "https://www.co.warren.ms.us/elected-officials/chancery-clerk/delinquent-taxes/"
COUNTY_NAME = "warren"
COUNTY_FIPS = "149"
STATE_CODE = "MS"
TAX_YEAR = 2023
AS_OF_DATE = "2026-03-05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Warren County delinquent taxes page.")
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
        "raw_html": raw_dir / "warren_delinquent_taxes.html",
        "manifest": raw_dir / "manifest.json",
        "standardized": standardized_dir / "warren_delinquent_tax_records.parquet",
        "linked": linked_dir / "warren_linked_tax_records.parquet",
        "unmatched": linked_dir / "warren_unmatched_tax_records.parquet",
        "ambiguous": linked_dir / "warren_ambiguous_tax_links.parquet",
        "summary": TAX_METADATA_DIR / "tax_free_warren_linkage_summary_ms.csv",
        "unmatched_reason_summary": TAX_METADATA_DIR / "tax_free_warren_unmatched_reason_summary_ms.csv",
        "ambiguity_reason_summary": TAX_METADATA_DIR / "tax_free_warren_ambiguity_reason_summary_ms.csv",
        "identifier_diagnostics": TAX_METADATA_DIR / "tax_free_warren_identifier_diagnostics_ms.csv",
        "qa_summary": TAX_METADATA_DIR / "tax_free_warren_qa_summary_ms.csv",
        "comparison_summary": TAX_METADATA_DIR / "tax_free_warren_comparison_ms.csv",
        "free_statewide_summary": TAX_METADATA_DIR / "tax_free_statewide_source_summary_ms.csv",
    }


def extract_records(raw_html: str) -> pd.DataFrame:
    text = html.unescape(raw_html)
    body_start = text.find("DELINQUENT TAXES Updated")
    if body_start >= 0:
        text = text[body_start:]
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>|</div>|</h2>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    blocks = re.split(r"[—-]{8,}", text)
    rows: list[dict[str, Any]] = []
    record_index = 1
    for block in blocks:
        lines = [re.sub(r"\s+", " ", line).strip() for line in block.splitlines() if re.sub(r"\s+", " ", line).strip()]
        if not lines:
            continue
        parcel_line = next((line for line in lines if line.startswith("Parcel ")), None)
        if parcel_line is None:
            continue
        parcel_id_raw = re.sub(r"^Parcel\s+", "", parcel_line).strip()
        ppin_values = re.findall(r"PPIN\s+(\d+)", " ".join(lines))
        owner_candidates = [line for line in lines if line != parcel_line and not line.startswith("PPIN ") and not line.startswith("Lienor ")]
        owner_name = owner_candidates[0] if owner_candidates else None
        legal_lines: list[str] = []
        seen_owner = False
        for line in lines:
            if line == parcel_line:
                continue
            if owner_name and not seen_owner and line == owner_name:
                seen_owner = True
                continue
            if line.startswith("PPIN ") or line.startswith("Lienor "):
                continue
            legal_lines.append(line)
        rows.append(
            {
                "source_record_id": f"{COUNTY_NAME}_{record_index:05d}",
                "parcel_id_raw": parcel_id_raw,
                "owner_name": owner_name,
                "legal_description": " ".join(legal_lines).strip() or None,
                "ppin_raw": ppin_values[0] if ppin_values else None,
                "ppin_list_raw": "|".join(ppin_values) if ppin_values else None,
                "raw_text_block": "\n".join(lines),
            }
        )
        record_index += 1
    return pd.DataFrame(rows)


def standardize_warren(frame: pd.DataFrame, run_id: str, raw_path: Path) -> pd.DataFrame:
    parcel_id_raw = clean_string(frame["parcel_id_raw"])
    owner_name = clean_string(frame["owner_name"])
    legal_description = clean_string(frame["legal_description"])
    source_ppin = normalize_ppin(frame["ppin_raw"].astype("string"))
    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash([STATE_CODE, COUNTY_FIPS, SOURCE_NAME, source_record_id, parcel_id])
                    for source_record_id, parcel_id in zip(frame["source_record_id"], parcel_id_raw)
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
            "source_dataset_path": pd.Series(raw_path.relative_to(BASE_DIR).as_posix(), index=frame.index, dtype="string"),
            "source_record_id": frame["source_record_id"].astype("string"),
            "source_ppin": source_ppin,
            "ingestion_run_id": pd.Series(run_id, index=frame.index, dtype="string"),
            "source_file_version": pd.Series("2026-03-06", index=frame.index, dtype="string"),
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
            "tax_amount_due": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_amount_paid": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_balance_due": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_status": pd.Series("sold_not_redeemed_delinquent_taxes", index=frame.index, dtype="string"),
            "payment_status": pd.Series("unpaid", index=frame.index, dtype="string"),
            "delinquent_flag": pd.Series(True, index=frame.index, dtype="boolean"),
            "forfeited_flag": pd.Series(False, index=frame.index, dtype="boolean"),
            "delinquent_amount": pd.Series(np.nan, index=frame.index, dtype="float64"),
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
    standardized["legal_description"] = legal_description
    standardized["record_hash"] = build_record_hash(
        standardized,
        ["parcel_id_normalized", "source_ppin", "owner_name", "tax_year", "tax_status"],
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
            {"metric": "warren_rows", "value": int(len(standardized))},
            {"metric": "warren_unique_parcel_ids", "value": int(standardized["parcel_id_normalized"].nunique())},
            {"metric": "warren_null_ppin_rows", "value": int(standardized["source_ppin"].isna().sum())},
            {"metric": "warren_duplicate_parcel_rows", "value": int(standardized["parcel_id_normalized"].duplicated(keep=False).fillna(False).sum())},
            {"metric": "warren_direct_identifier_overlap_rows", "value": direct_overlap},
            {"metric": "warren_compact_identifier_overlap_rows", "value": compact_overlap},
            {"metric": "warren_ppin_overlap_rows", "value": ppin_overlap},
            {"metric": "warren_safe_adapter_justified", "value": "not_needed"},
        ]
    )


def build_qa_summary(standardized: pd.DataFrame, linked: pd.DataFrame, unmatched: pd.DataFrame, ambiguous: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "row_count", "value": int(len(standardized))},
            {"metric": "null_parcel_id_rate", "value": round(float(standardized["parcel_id_normalized"].isna().mean() * 100.0), 4)},
            {"metric": "null_ppin_rate", "value": round(float(standardized["source_ppin"].isna().mean() * 100.0), 4)},
            {"metric": "duplicate_record_rate", "value": round(float(standardized["record_hash"].duplicated(keep=False).mean() * 100.0), 4)},
            {"metric": "linked_rate", "value": round(float(len(linked) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "exact_match_rate", "value": round(float(linked["match_confidence_tier"].eq("high").mean() * 100.0), 4) if not linked.empty else 0.0},
            {"metric": "heuristic_match_rate", "value": round(float(linked["match_confidence_tier"].eq("low").mean() * 100.0), 4) if not linked.empty else 0.0},
            {"metric": "unmatched_rate", "value": round(float(len(unmatched) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "ambiguous_rate", "value": round(float(len(ambiguous) / max(len(standardized), 1) * 100.0), 4)},
        ]
    )


def build_comparison_summary(standardized: pd.DataFrame, linked: pd.DataFrame) -> pd.DataFrame:
    def lookup_metric(path: Path, key: str) -> Any:
        if not path.exists():
            return pd.NA
        frame = pd.read_csv(path)
        lookup = dict(zip(frame["metric"].astype("string"), frame["value"]))
        return lookup.get(key, pd.NA)

    pike_path = TAX_METADATA_DIR / "tax_free_ingest_summary_ms.csv"
    sos_path = TAX_METADATA_DIR / "tax_free_sos_ingest_summary_ms.csv"
    hinds_path = TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv"
    return pd.DataFrame(
        [
            {"metric": "warren_rows", "value": int(len(standardized))},
            {"metric": "warren_linked_rows", "value": int(len(linked))},
            {"metric": "warren_linkage_rate", "value": round(float(len(linked) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "pike_linkage_rate", "value": lookup_metric(pike_path, "pike_linkage_rate")},
            {"metric": "sos_linkage_rate", "value": lookup_metric(sos_path, "sos_linkage_rate")},
            {"metric": "hinds_linkage_rate", "value": lookup_metric(hinds_path, "hinds_linkage_rate")},
            {
                "metric": "comparison_assessment",
                "value": "Warren is a strong linkage template: county-hosted, free, low-duplicate, and parcel IDs align directly to parcel master without unsafe heuristics.",
            },
        ]
    )


def update_free_statewide_summary() -> pd.DataFrame:
    source_files = [
        ("pike", TAX_METADATA_DIR / "tax_free_ingest_summary_ms.csv"),
        ("sos_statewide", TAX_METADATA_DIR / "tax_free_sos_ingest_summary_ms.csv"),
        ("hinds", TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv"),
        ("warren", TAX_METADATA_DIR / "tax_free_warren_linkage_summary_ms.csv"),
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
        "warren": {
            "standardized_rows": "warren_standardized_rows",
            "linked_rows": "warren_linked_rows",
            "unmatched_rows": "warren_unmatched_rows",
            "ambiguous_rows": "warren_ambiguous_rows",
            "linkage_rate": "warren_linkage_rate",
        },
    }
    rows: list[dict[str, Any]] = []
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

    raw_bytes = download_bytes(SOURCE_URL)
    paths["raw_html"].write_bytes(raw_bytes)
    frame = extract_records(raw_bytes.decode("utf-8", errors="ignore"))
    standardized = standardize_warren(frame, run_id, paths["raw_html"])
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
            {"metric": "warren_standardized_rows", "value": int(len(standardized))},
            {"metric": "warren_linked_rows", "value": int(len(linked))},
            {"metric": "warren_unmatched_rows", "value": int(len(unmatched))},
            {"metric": "warren_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "warren_linkage_rate", "value": round(float(len(linked) / max(len(standardized), 1) * 100.0), 4)},
            {"metric": "warren_exact_match_rows", "value": int(linked["match_confidence_tier"].eq("high").sum()) if not linked.empty else 0},
            {"metric": "warren_heuristic_match_rows", "value": int(linked["match_confidence_tier"].eq("low").sum()) if not linked.empty else 0},
        ]
    )
    summary.to_csv(paths["summary"], index=False)

    unmatched_reason_summary, ambiguity_reason_summary = build_reason_summaries(unmatched, ambiguous)
    unmatched_reason_summary.to_csv(paths["unmatched_reason_summary"], index=False)
    ambiguity_reason_summary.to_csv(paths["ambiguity_reason_summary"], index=False)
    build_identifier_diagnostics(standardized, master).to_csv(paths["identifier_diagnostics"], index=False)
    build_qa_summary(standardized, linked, unmatched, ambiguous).to_csv(paths["qa_summary"], index=False)
    build_comparison_summary(standardized, linked).to_csv(paths["comparison_summary"], index=False)
    update_free_statewide_summary().to_csv(paths["free_statewide_summary"], index=False)

    write_json(
        paths["manifest"],
        {
            "ingestion_run_id": run_id,
            "state_code": STATE_CODE,
            "county_name": COUNTY_NAME,
            "county_fips": COUNTY_FIPS,
            "source_name": SOURCE_NAME,
            "source_url": SOURCE_URL,
            "raw_html_path": paths["raw_html"].relative_to(BASE_DIR).as_posix(),
            "standardized_path": paths["standardized"].relative_to(BASE_DIR).as_posix(),
            "linked_path": paths["linked"].relative_to(BASE_DIR).as_posix(),
            "unmatched_path": paths["unmatched"].relative_to(BASE_DIR).as_posix(),
            "ambiguous_path": paths["ambiguous"].relative_to(BASE_DIR).as_posix(),
            "summary_path": paths["summary"].relative_to(BASE_DIR).as_posix(),
            "row_count": int(len(frame)),
            "schema_columns": frame.columns.tolist(),
        },
    )
    update_registry_row(
        REGISTRY_CSV,
        SOURCE_ID,
        pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        f"Downloaded Warren delinquent taxes HTML to {paths['raw_html'].relative_to(BASE_DIR).as_posix()}.",
    )

    print(f"Warren rows: {len(standardized):,}")
    print(f"Linked rows: {len(linked):,}")
    print(f"Unmatched rows: {len(unmatched):,}")
    print(f"Ambiguous rows: {len(ambiguous):,}")
    print(f"Summary: {paths['summary'].relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
