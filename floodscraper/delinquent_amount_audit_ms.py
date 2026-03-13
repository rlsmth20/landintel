from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"
TAX_STANDARDIZED_DIR = BASE_DIR / "data" / "tax_standardized" / "ms"

LEADS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_statewide.parquet"
OUTLIERS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_amount_outliers.csv"

FLAGGED_RECORDS_PATH = TAX_PUBLISHED_DIR / "delinquent_amount_audit_flagged_records.csv"
COUNTY_SUMMARY_PATH = TAX_PUBLISHED_DIR / "delinquent_amount_audit_county_summary.csv"
SOURCE_SUMMARY_PATH = TAX_PUBLISHED_DIR / "delinquent_amount_audit_source_summary.csv"


def normalize_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def load_standardized_records() -> pd.DataFrame:
    wanted_columns = [
        "source_name",
        "source_type",
        "county_name",
        "county_fips",
        "source_record_id",
        "parcel_id_raw",
        "parcel_id_normalized",
        "source_ppin",
        "owner_name",
        "tax_year",
        "bill_year",
        "delinquent_amount",
        "assessed_total_value",
        "taxable_value",
        "tax_amount_due",
        "tax_balance_due",
        "delinquent_years",
        "raw_payload_json",
        "source_dataset_path",
        "loaded_at",
    ]
    frames: list[pd.DataFrame] = []
    for path in TAX_STANDARDIZED_DIR.rglob("*.parquet"):
        frame = pd.read_parquet(path)
        for column in wanted_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame.loc[:, wanted_columns].copy()
        frame["standardized_file_path"] = path.relative_to(BASE_DIR).as_posix()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["county_name"] = normalize_string(combined["county_name"])
    combined["source_name"] = normalize_string(combined["source_name"])
    combined["source_type"] = normalize_string(combined["source_type"])
    combined["source_record_id"] = normalize_string(combined["source_record_id"])
    return combined


def parse_payload_columns(frame: pd.DataFrame) -> pd.DataFrame:
    payloads = frame["raw_payload_json"].map(lambda value: json.loads(value) if pd.notna(value) else {})
    extracted = pd.DataFrame(
        {
            "payload_amount_due_raw": payloads.map(lambda p: p.get("amount_due_raw")),
            "payload_excess_bid_raw": payloads.map(lambda p: p.get("excess_bid_raw")),
            "payload_acres_raw": payloads.map(lambda p: p.get("acres_raw")),
            "payload_assessed_total_value_raw": payloads.map(lambda p: p.get("assessed_total_value_raw")),
            "payload_record_suffix": payloads.map(lambda p: p.get("record_suffix")),
            "payload_raw_text_block": payloads.map(lambda p: p.get("raw_text_block")),
        },
        index=frame.index,
    )
    return pd.concat([frame, extracted], axis=1)


def detect_record_flags(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    alias_map = {
        "delinquent_amount": ["delinquent_amount_std", "delinquent_amount_lead"],
        "assessed_total_value": ["assessed_total_value_std", "assessed_total_value_lead"],
        "tax_amount_due": ["tax_amount_due_std", "tax_amount_due_lead"],
        "tax_balance_due": ["tax_balance_due_std", "tax_balance_due_lead"],
        "source_record_id": ["best_source_record_id"],
    }
    for canonical, candidates in alias_map.items():
        if canonical in working.columns:
            continue
        for candidate in candidates:
            if candidate in working.columns:
                working[canonical] = working[candidate]
                break
    working["delinquent_amount"] = pd.to_numeric(working["delinquent_amount"], errors="coerce")
    working["assessed_total_value"] = pd.to_numeric(working["assessed_total_value"], errors="coerce")
    working["tax_amount_due"] = pd.to_numeric(working["tax_amount_due"], errors="coerce")
    working["tax_balance_due"] = pd.to_numeric(working["tax_balance_due"], errors="coerce")
    working["payload_excess_bid_num"] = pd.to_numeric(working["payload_excess_bid_raw"], errors="coerce")
    working["payload_amount_due_num"] = pd.to_numeric(working["payload_amount_due_raw"], errors="coerce")
    working["payload_acres_num"] = pd.to_numeric(working["payload_acres_raw"], errors="coerce")

    source_record_counts = working.groupby("source_record_id").size().rename("source_record_lead_count")
    same_amount_counts = (
        working.groupby(["county_name", "source_name", "delinquent_amount"], dropna=False)
        .size()
        .rename("same_amount_count")
        .reset_index()
    )
    working = working.merge(same_amount_counts, on=["county_name", "source_name", "delinquent_amount"], how="left")
    working["source_record_lead_count"] = working["source_record_id"].map(source_record_counts).fillna(0).astype("int32")

    flags: list[list[str]] = []
    evidence: list[list[str]] = []
    for row in working.itertuples(index=False):
        row_flags: list[str] = []
        row_evidence: list[str] = []
        raw_block = str(row.payload_raw_text_block or "")
        if pd.notna(row.delinquent_amount) and row.delinquent_amount >= 100000:
            row_flags.append("extreme_amount")
            row_evidence.append(f"delinquent_amount={row.delinquent_amount:.2f}")
        if pd.notna(row.delinquent_amount) and pd.notna(row.assessed_total_value) and row.assessed_total_value > 0 and row.delinquent_amount / row.assessed_total_value >= 50:
            row_flags.append("amount_far_exceeds_assessed_value")
            row_evidence.append(f"amount_to_assessed_ratio={row.delinquent_amount / row.assessed_total_value:.2f}")
        if pd.notna(row.payload_amount_due_num) and pd.notna(row.payload_acres_num) and row.payload_amount_due_num == row.payload_acres_num and row.payload_amount_due_num >= 100000:
            row_flags.append("amount_equals_acres_raw_parser_error")
            row_evidence.append(f"payload_amount_due_raw={row.payload_amount_due_raw}")
        if pd.notna(row.payload_excess_bid_num) and pd.notna(row.assessed_total_value) and row.payload_excess_bid_num == row.assessed_total_value and pd.notna(row.payload_amount_due_num) and row.payload_amount_due_num >= 100000:
            row_flags.append("amount_likely_from_wrong_header_field")
            row_evidence.append(f"payload_excess_bid_raw={row.payload_excess_bid_raw}")
        if raw_block and "Total Assessed/ Reg 100 Taxes Printers Amount Amount Excess" in raw_block and pd.notna(row.payload_amount_due_num) and row.payload_amount_due_num >= 100000:
            row_flags.append("header_row_shift_detected")
            row_evidence.append("raw_text_contains_total_assessed_header")
        if pd.notna(row.same_amount_count) and int(row.same_amount_count) >= 3 and pd.notna(row.delinquent_amount) and row.delinquent_amount >= 100000:
            row_flags.append("repeated_extreme_amount")
            row_evidence.append(f"same_amount_count={int(row.same_amount_count)}")
        if int(row.source_record_lead_count) > 1:
            row_flags.append("single_source_record_to_multiple_leads")
            row_evidence.append(f"source_record_lead_count={int(row.source_record_lead_count)}")
        flags.append(row_flags)
        evidence.append(row_evidence)

    working["audit_flags"] = ["|".join(items) if items else pd.NA for items in flags]
    working["audit_evidence"] = ["|".join(items) if items else pd.NA for items in evidence]
    return working


def build_flagged_record_output(leads: pd.DataFrame, standardized: pd.DataFrame, flagged_counties: list[str]) -> pd.DataFrame:
    county_slice = leads.loc[leads["county_name"].isin(flagged_counties)].copy()
    county_slice = county_slice.sort_values(["county_name", "delinquent_amount"], ascending=[True, False])
    top_candidates = county_slice.groupby("county_name", group_keys=False).head(50).copy()

    merged = top_candidates.merge(
        standardized,
        left_on=["best_source_name", "best_source_record_id", "county_name"],
        right_on=["source_name", "source_record_id", "county_name"],
        how="left",
        suffixes=("_lead", "_std"),
    )
    merged = detect_record_flags(merged)
    alias_map = {
        "county_fips": ["county_fips_lead", "county_fips_std"],
        "parcel_row_id": ["parcel_row_id_lead", "parcel_row_id_std"],
        "source_dataset_path": ["source_dataset_path_std", "source_dataset_path_lead"],
        "tax_year_lead": ["tax_year"],
        "owner_name_lead": ["owner_name"],
        "owner_name_std": ["owner_name_std", "owner_name"],
    }
    for canonical, candidates in alias_map.items():
        if canonical in merged.columns:
            continue
        for candidate in candidates:
            if candidate in merged.columns:
                merged[canonical] = merged[candidate]
                break
    keep = merged["audit_flags"].notna() | merged["delinquent_amount_lead"].ge(100000).fillna(False)
    merged = merged.loc[keep].copy()
    merged["recommendation"] = "trusted"
    merged.loc[merged["audit_flags"].notna(), "recommendation"] = "use_with_caution"
    severe_mask = merged["audit_flags"].astype("string").str.contains(
        "amount_equals_acres_raw_parser_error|amount_likely_from_wrong_header_field|header_row_shift_detected|repeated_extreme_amount",
        regex=True,
        na=False,
    )
    merged.loc[severe_mask, "recommendation"] = "not_trusted_for_prominent_display"

    output = pd.DataFrame(
        {
            "county_fips": merged["county_fips"],
            "county_name": merged["county_name"],
            "parcel_row_id": merged["parcel_row_id"],
            "best_source_type": merged["best_source_type"],
            "best_source_name": merged["best_source_name"],
            "best_source_record_id": merged["best_source_record_id"],
            "source_record_id": merged["source_record_id"],
            "source_dataset_path": merged["source_dataset_path"],
            "standardized_file_path": merged["standardized_file_path"],
            "delinquent_amount_lead": pd.to_numeric(merged["delinquent_amount_lead"], errors="coerce"),
            "delinquent_amount_standardized": pd.to_numeric(merged["delinquent_amount_std"], errors="coerce"),
            "tax_amount_due": pd.to_numeric(merged["tax_amount_due"], errors="coerce"),
            "tax_balance_due": pd.to_numeric(merged["tax_balance_due"], errors="coerce"),
            "assessed_total_value": pd.to_numeric(merged["assessed_total_value"], errors="coerce"),
            "payload_amount_due_raw": merged["payload_amount_due_raw"],
            "payload_excess_bid_raw": merged["payload_excess_bid_raw"],
            "payload_acres_raw": merged["payload_acres_raw"],
            "payload_assessed_total_value_raw": merged["payload_assessed_total_value_raw"],
            "tax_year": merged["tax_year_std"].fillna(merged["tax_year_lead"]),
            "reported_delinquent_years": merged["reported_delinquent_years"],
            "parcel_id_raw": merged["parcel_id_raw"],
            "source_ppin": merged["source_ppin"],
            "owner_name_lead": merged["owner_name_lead"],
            "owner_name_standardized": merged["owner_name_std"],
            "source_record_lead_count": merged["source_record_lead_count"],
            "same_amount_count": merged["same_amount_count"],
            "audit_flags": merged["audit_flags"],
            "audit_evidence": merged["audit_evidence"],
            "recommendation": merged["recommendation"],
            "raw_text_block": merged["payload_raw_text_block"],
        }
    )
    return output.sort_values(["county_name", "delinquent_amount_lead"], ascending=[True, False]).reset_index(drop=True)


def build_county_summary(flagged_records: pd.DataFrame, outliers: pd.DataFrame, leads: pd.DataFrame) -> pd.DataFrame:
    county_outliers = outliers.loc[outliers["outlier_scope"].eq("county")].copy()
    summaries: list[dict[str, Any]] = []
    for county_name, group in flagged_records.groupby("county_name", dropna=False):
        county_leads = leads.loc[leads["county_name"].eq(county_name)].copy()
        amounts = pd.to_numeric(county_leads["delinquent_amount"], errors="coerce").fillna(0.0)
        outlier_row = county_outliers.loc[county_outliers["county_name"].eq(county_name)].head(1)
        rec = "trusted"
        flags = normalize_string(group["audit_flags"]).dropna().tolist()
        if any("amount_equals_acres_raw_parser_error" in flag or "header_row_shift_detected" in flag for flag in flags):
            rec = "not_trusted_for_prominent_display"
        elif any(flags):
            rec = "use_with_caution"
        summaries.append(
            {
                "county_name": county_name,
                "county_fips": group["county_fips"].iloc[0],
                "lead_count": int(len(county_leads)),
                "reported_amount_count": int(county_leads["has_reported_delinquent_amount_flag"].fillna(False).astype(bool).sum()),
                "county_total_delinquent_amount": round(float(amounts.sum()), 2),
                "county_median_delinquent_amount": round(float(amounts.loc[amounts.gt(0)].median()), 2) if amounts.gt(0).any() else 0.0,
                "top10_share": float(outlier_row["county_top10_share"].iloc[0]) if not outlier_row.empty else pd.NA,
                "flagged_record_count": int(len(group)),
                "flagged_source_count": int(group["best_source_name"].nunique()),
                "recommendation": rec,
                "outlier_anomaly_flag": outlier_row["anomaly_flag"].iloc[0] if not outlier_row.empty else pd.NA,
            }
        )
    return pd.DataFrame(summaries).sort_values(["county_total_delinquent_amount"], ascending=False).reset_index(drop=True)


def build_source_summary(flagged_records: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for (county_name, source_name, source_type), group in flagged_records.groupby(["county_name", "best_source_name", "best_source_type"], dropna=False):
        flags = normalize_string(group["audit_flags"]).dropna().tolist()
        recommendation = "trusted"
        if any("amount_equals_acres_raw_parser_error" in flag or "header_row_shift_detected" in flag for flag in flags):
            recommendation = "not_trusted_for_prominent_display"
        elif any(flags):
            recommendation = "use_with_caution"
        summaries.append(
            {
                "county_name": county_name,
                "source_name": source_name,
                "source_type": source_type,
                "lead_count": int(group["parcel_row_id"].nunique()),
                "flagged_record_count": int(len(group)),
                "extreme_amount_record_count": int(pd.to_numeric(group["delinquent_amount_lead"], errors="coerce").ge(100000).fillna(False).sum()),
                "repeated_extreme_amount_count": int(group["audit_flags"].astype("string").str.contains("repeated_extreme_amount", regex=False, na=False).sum()),
                "parser_error_flag_count": int(group["audit_flags"].astype("string").str.contains("amount_equals_acres_raw_parser_error|amount_likely_from_wrong_header_field|header_row_shift_detected", regex=True, na=False).sum()),
                "recommendation": recommendation,
                "flag_examples": "|".join(sorted(set(flags))[:5]) if flags else pd.NA,
            }
        )
    return pd.DataFrame(summaries).sort_values(["parser_error_flag_count", "flagged_record_count"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    TAX_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    leads = pd.read_parquet(LEADS_PATH)
    leads["county_name"] = normalize_string(leads["county_name"])
    leads["best_source_name"] = normalize_string(leads["best_source_name"])
    leads["best_source_type"] = normalize_string(leads["best_source_type"])
    leads["best_source_record_id"] = normalize_string(leads["best_source_record_id"])
    leads["delinquent_amount"] = pd.to_numeric(leads["delinquent_amount"], errors="coerce")

    outliers = pd.read_csv(OUTLIERS_PATH)
    flagged_counties = (
        outliers.loc[outliers["outlier_scope"].eq("county") & outliers["anomaly_flag"].notna(), "county_name"]
        .astype("string")
        .dropna()
        .unique()
        .tolist()
    )

    standardized = parse_payload_columns(load_standardized_records())
    flagged_records = build_flagged_record_output(leads, standardized, flagged_counties)
    county_summary = build_county_summary(flagged_records, outliers, leads)
    source_summary = build_source_summary(flagged_records)

    flagged_records.to_csv(FLAGGED_RECORDS_PATH, index=False)
    county_summary.to_csv(COUNTY_SUMMARY_PATH, index=False)
    source_summary.to_csv(SOURCE_SUMMARY_PATH, index=False)

    print(f"Flagged records: {FLAGGED_RECORDS_PATH.relative_to(BASE_DIR)}")
    print(f"County summary: {COUNTY_SUMMARY_PATH.relative_to(BASE_DIR)}")
    print(f"Source summary: {SOURCE_SUMMARY_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
