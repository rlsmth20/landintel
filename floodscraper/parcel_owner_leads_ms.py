from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
STATE_ABBR = "MS"
PARCELS_DIR = BASE_DIR / "data" / "parcels"

INPUT_FILE = PARCELS_DIR / "mississippi_parcels_master.gpkg"
OUTPUT_PARQUET = PARCELS_DIR / "mississippi_parcels_owner_leads.parquet"
OUTPUT_GPKG = PARCELS_DIR / "mississippi_parcels_owner_leads.gpkg"
MAILER_EXPORT_CSV = PARCELS_DIR / "mississippi_mailer_export.csv"
SUMMARY_CSV = PARCELS_DIR / "mississippi_owner_leads_summary.csv"
SCHEMA_CSV = PARCELS_DIR / "mississippi_owner_schema.csv"
COUNTY_QA_CSV = PARCELS_DIR / "mississippi_owner_county_qa.csv"
TAX_DISTRESS_INPUT = PARCELS_DIR / "mississippi_parcels_tax_distress.parquet"

OWNER_SOURCE_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "state_code",
    "county_name",
    "county_fips",
    "apn",
    "source_parcel_id_raw",
    "source_parcel_id_normalized",
    "source_parcel_number",
    "source_alt_parcel_number",
    "owner_name_raw",
    "owner_name_2_raw",
    "mailing_address_line1_raw",
    "mailing_address_line2_raw",
    "mailing_city_raw",
    "mailing_state_raw",
    "mailing_zip_raw",
    "property_address_raw",
    "property_city_raw",
    "property_state_raw",
    "land_use_raw",
    "owner_name",
    "mail_address_1",
    "mail_address_2",
    "mail_city_1",
    "mail_state_1",
    "mail_zip_1",
    "mail_city_2",
    "mail_state_2",
    "mail_zip_2",
    "site_address",
    "site_city",
    "site_state",
    "site_zip",
    "zoning",
    "tax_status",
    "tax_acres",
    "gis_acres",
    "total_acres",
    "buildability_score",
    "investment_score",
    "environment_score",
    "parcel_constraint_summary",
    "geometry",
]

MAILER_EXPORT_FIELDS = [
    "parcel_row_id",
    "parcel_id",
    "apn",
    "county_name",
    "owner_name_normalized",
    "mailing_address_line1",
    "mailing_address_line2",
    "mailing_city",
    "mailing_state",
    "mailing_zip",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "owner_type",
    "corporate_owner_flag",
    "owner_parcel_count",
    "tax_delinquent_flag",
    "tax_distress_score",
    "distressed_owner_flag",
    "mailer_target_score",
    "buildability_score",
    "investment_score",
    "environment_score",
    "parcel_constraint_summary",
]

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL",
    "ALASKA": "AK",
    "ARIZONA": "AZ",
    "ARKANSAS": "AR",
    "CALIFORNIA": "CA",
    "COLORADO": "CO",
    "CONNECTICUT": "CT",
    "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL",
    "GEORGIA": "GA",
    "HAWAII": "HI",
    "IDAHO": "ID",
    "ILLINOIS": "IL",
    "INDIANA": "IN",
    "IOWA": "IA",
    "KANSAS": "KS",
    "KENTUCKY": "KY",
    "LOUISIANA": "LA",
    "MAINE": "ME",
    "MARYLAND": "MD",
    "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI",
    "MINNESOTA": "MN",
    "MISSISSIPPI": "MS",
    "MISSOURI": "MO",
    "MONTANA": "MT",
    "NEBRASKA": "NE",
    "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH",
    "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM",
    "NEW YORK": "NY",
    "NORTH CAROLINA": "NC",
    "NORTH DAKOTA": "ND",
    "OHIO": "OH",
    "OKLAHOMA": "OK",
    "OREGON": "OR",
    "PENNSYLVANIA": "PA",
    "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN",
    "TEXAS": "TX",
    "UTAH": "UT",
    "VERMONT": "VT",
    "VIRGINIA": "VA",
    "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI",
    "WYOMING": "WY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Mississippi owner and mailer lead parcel layer.")
    parser.add_argument("--input-file", type=str, default=str(INPUT_FILE), help="Master parcel GeoPackage input path.")
    parser.add_argument("--output-parquet", type=str, default=str(OUTPUT_PARQUET), help="Owner leads Parquet output path.")
    parser.add_argument("--output-gpkg", type=str, default=str(OUTPUT_GPKG), help="Owner leads GeoPackage output path.")
    parser.add_argument("--mailer-export-csv", type=str, default=str(MAILER_EXPORT_CSV), help="Mailer-ready CSV export path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Summary CSV output path.")
    parser.add_argument("--schema-csv", type=str, default=str(SCHEMA_CSV), help="Schema CSV output path.")
    parser.add_argument("--county-qa-csv", type=str, default=str(COUNTY_QA_CSV), help="County QA CSV output path.")
    parser.add_argument("--tax-distress-file", type=str, default=str(TAX_DISTRESS_INPUT), help="Optional parcel tax distress parquet for score integration.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def blank_to_na(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out


def collapse_spaces(series: pd.Series) -> pd.Series:
    out = blank_to_na(series)
    out = out.str.replace(r"\s+", " ", regex=True)
    return out.astype("string")


def normalize_state(series: pd.Series) -> pd.Series:
    out = collapse_spaces(series).str.upper()
    out = out.replace(STATE_NAME_TO_ABBR)
    out = out.str.replace(r"[^A-Z]", "", regex=True)
    out = out.mask(out.str.len().eq(0), pd.NA)
    out = out.mask(out.str.len().gt(2), pd.NA)
    return out.astype("string")


def normalize_zip5(series: pd.Series) -> pd.Series:
    out = collapse_spaces(series)
    out = out.str.extract(r"(\d{5})", expand=False)
    out = out.astype("string")
    return out.mask(out.str.len().eq(0), pd.NA)


def normalize_owner_name(series: pd.Series) -> pd.Series:
    out = collapse_spaces(series).str.upper()
    out = out.str.replace("&", " AND ", regex=False)
    out = out.str.replace(r"[.,/\\()]", " ", regex=True)
    out = out.str.replace(r"\bC\/O\b", "CARE OF", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out.astype("string")


def standardize_owner_group_key(owner_name_normalized: pd.Series, owner_type: pd.Series) -> pd.Series:
    out = owner_name_normalized.fillna("").astype("string")
    replacements = {
        r"\bSTATE OF MISS\b": "STATE OF MISSISSIPPI",
        r"\bSTATE OF MISSISSIPPI\b": "STATE OF MISSISSIPPI",
        r"\bSTATE\b": "STATE OF MISSISSIPPI",
        r"\bU S A\b": "UNITED STATES OF AMERICA",
        r"\bU S GOVERNMENT\b": "UNITED STATES OF AMERICA",
        r"\bUNITED STATES\b": "UNITED STATES OF AMERICA",
        r"\bUNITED STATES OF AMERICA\b": "UNITED STATES OF AMERICA",
    }
    for pattern, replacement in replacements.items():
        out = out.str.replace(pattern, replacement, regex=True)
    out = out.str.replace(r"\bCITY OF ([A-Z ]+)\b", r"CITY OF \1", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out.astype("string")


def normalize_address_for_compare(series: pd.Series) -> pd.Series:
    out = collapse_spaces(series).str.upper()
    out = out.str.replace("&", " AND ", regex=False)
    out = out.str.replace(r"[#.,]", " ", regex=True)
    out = out.str.replace(r"\bP O BOX\b", "PO BOX", regex=True)
    replacements = {
        r"\bROAD\b": "RD",
        r"\bRD\b": "RD",
        r"\bSTREET\b": "ST",
        r"\bST\b": "ST",
        r"\bAVENUE\b": "AVE",
        r"\bAVE\b": "AVE",
        r"\bDRIVE\b": "DR",
        r"\bDR\b": "DR",
        r"\bLANE\b": "LN",
        r"\bLN\b": "LN",
        r"\bBOULEVARD\b": "BLVD",
        r"\bHIGHWAY\b": "HWY",
        r"\bHWY\b": "HWY",
        r"\bCOUNTY ROAD\b": "CR",
        r"\bCOURT\b": "CT",
        r"\bPLACE\b": "PL",
        r"\bCIRCLE\b": "CIR",
        r"\bESTATES\b": "EST",
        r"\bESTATE\b": "EST",
        r"\bAPARTMENT\b": "APT",
        r"\bSUITE\b": "STE",
        r"\bNORTH\b": "N",
        r"\bSOUTH\b": "S",
        r"\bEAST\b": "E",
        r"\bWEST\b": "W",
    }
    for pattern, replacement in replacements.items():
        out = out.str.replace(pattern, replacement, regex=True)
    out = out.str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.mask(out.eq(""), pd.NA)
    return out.astype("string")


def choose_with_fallback(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    first = collapse_spaces(primary)
    second = collapse_spaces(secondary)
    return first.where(first.notna(), second).astype("string")


def select_series(frame: pd.DataFrame, preferred: str, fallback: str | None = None) -> pd.Series:
    if preferred in frame.columns:
        value = frame[preferred]
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        return value.astype("string")
    if fallback and fallback in frame.columns:
        value = frame[fallback]
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        return value.astype("string")
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def build_owner_type(owner_name_normalized: pd.Series) -> pd.Series:
    owner = owner_name_normalized.fillna("")
    patterns = {
        "government": r"\b(?:COUNTY|CITY|STATE OF|UNITED STATES|U S A|US GOV|GOVERNMENT|DEPARTMENT|BOARD OF|SCHOOL DIST|SCHOOL DISTRICT|AUTHORITY|HOUSING AUTHORITY|TOWN OF|VILLAGE OF)\b",
        "nonprofit": r"\b(?:CHURCH|MINISTRY|MINISTRIES|MISSIONARY|TEMPLE|FOUNDATION|CHARITIES|CHARITY|NONPROFIT|ASSOCIATION|SYNAGOGUE|DIOCESE|BAPTIST|METHODIST|PRESBYTERIAN|CATHOLIC)\b",
        "trust": r"\b(?:TRUST|TRUSTEE|REVOCABLE|IRREVOCABLE|U\/A|U\/T\/A|ESTATE OF)\b",
        "llc": r"\b(?:LLC|L L C|LLP|L L P|LP|L P|LTD|LIMITED PARTNERSHIP|LIMITED LIABILITY)\b",
        "corporation": r"\b(?:INC|INCORPORATED|CORP|CORPORATION|COMPANY|CO|BANK|BANCORP|PROPERTIES|PROPERTY|HOLDINGS|ENTERPRISES|ENTERPRISE|INVESTMENTS|INVESTMENT|REALTY|GROUP|PARTNERS|PARTNERSHIP)\b",
    }
    out = pd.Series("individual", index=owner_name_normalized.index, dtype="string")
    out = out.mask(owner.eq(""), "unknown")
    for label, pattern in patterns.items():
        out = out.mask(owner.str.contains(pattern, regex=True, na=False), label)
    return out.astype("string")


def build_owner_name_confidence(owner_name_normalized: pd.Series) -> pd.Series:
    owner = owner_name_normalized.fillna("")
    low = (
        owner.eq("")
        | owner.str.contains(r"\b(?:UNKNOWN|UNKNOWN OWNER|OWNER UNKNOWN|CURRENT OWNER|OCCUPANT|VACANT|NONE)\b", regex=True, na=False)
        | owner.str.fullmatch(r"[\W\d]+", na=False)
    )
    medium = owner.str.contains(r"\b(?:CARE OF|ET AL|HEIRS|ESTATE OF|TRUSTEE|REVOCABLE|IRREVOCABLE)\b", regex=True, na=False)
    out = pd.Series("high", index=owner_name_normalized.index, dtype="string")
    out = out.mask(medium, "medium")
    out = out.mask(low, "low")
    return out.astype("string")


def build_mailing_completeness_score(line1: pd.Series, city: pd.Series, state: pd.Series, zip5: pd.Series) -> pd.Series:
    score = (
        line1.notna().astype("int8")
        + city.notna().astype("int8")
        + state.notna().astype("int8")
        + zip5.notna().astype("int8")
    )
    return pd.Series(score, index=line1.index, dtype="int8")


def build_address_match(
    mailing_address_line1: pd.Series,
    property_address_raw: pd.Series,
    mailing_city: pd.Series,
    property_city: pd.Series,
    mailing_zip5: pd.Series,
    mailing_state: pd.Series,
    property_state: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    mail_addr_key = normalize_address_for_compare(mailing_address_line1)
    prop_addr_key = normalize_address_for_compare(property_address_raw)
    mail_city_key = normalize_address_for_compare(mailing_city)
    prop_city_key = normalize_address_for_compare(property_city)
    mail_state_key = normalize_state(mailing_state)
    prop_state_key = normalize_state(property_state)

    sufficient = mail_addr_key.notna() & prop_addr_key.notna()
    addr_match = mail_addr_key.eq(prop_addr_key)
    city_match = mail_city_key.eq(prop_city_key) | mail_city_key.isna() | prop_city_key.isna()
    state_match = mail_state_key.eq(prop_state_key) | mail_state_key.isna() | prop_state_key.isna()
    match = addr_match & city_match & state_match
    out = pd.Series(pd.NA, index=mailing_address_line1.index, dtype="boolean")
    reason = pd.Series(pd.NA, index=mailing_address_line1.index, dtype="string")
    out.loc[sufficient] = match.loc[sufficient]
    reason.loc[mail_addr_key.isna()] = "missing_mailing_address"
    reason.loc[mail_addr_key.notna() & prop_addr_key.isna()] = "missing_property_address"
    po_box_mask = mail_addr_key.fillna("").str.contains(r"\bPO BOX\b", regex=True)
    reason.loc[po_box_mask & prop_addr_key.notna()] = "po_box_only"
    reason.loc[sufficient & match] = "matched_address"
    reason.loc[sufficient & ~match] = "different_address"
    return out, reason


def build_owner_group_basis(owner_name_normalized: pd.Series, mailing_key: pd.Series, parcel_row_id: pd.Series) -> pd.Series:
    owner_key = owner_name_normalized.fillna("")
    mail_key = mailing_key.fillna("")
    basis = np.select(
        [
            (owner_key != "") & (mail_key != ""),
            (owner_key != "") & (mail_key == ""),
            (owner_key == "") & (mail_key != ""),
        ],
        [
            "OWNER|" + owner_key + "|MAIL|" + mail_key,
            "OWNER|" + owner_key,
            "MAIL|" + mail_key,
        ],
        default="PARCEL|" + parcel_row_id.astype("string").fillna(""),
    )
    return pd.Series(basis, index=parcel_row_id.index, dtype="string")


def build_mailer_target_score(df: pd.DataFrame) -> pd.Series:
    score = (
        df["absentee_owner_flag"].fillna(False).astype(bool).astype(int) * 25
        + df["out_of_state_owner_flag"].fillna(False).astype(bool).astype(int) * 15
        + df["corporate_owner_flag"].fillna(False).astype(bool).astype(int) * 10
        + df["tax_delinquent_flag"].fillna(False).astype(bool).astype(int) * 12
        + pd.to_numeric(df["tax_distress_score"], errors="coerce").fillna(0.0) * 4.0
        + df["mailing_completeness_score"].fillna(0).astype(float) * 5.0
        + pd.to_numeric(df["investment_score"], errors="coerce").fillna(0.0) * 0.20
        + pd.to_numeric(df["buildability_score"], errors="coerce").fillna(0.0) * 0.15
        + pd.to_numeric(df["environment_score"], errors="coerce").fillna(0.0) * 0.05
        + np.minimum(pd.to_numeric(df["owner_parcel_count"], errors="coerce").fillna(1.0) - 1.0, 5.0) * 2.0
        - df["owner_occupied_proxy_flag"].fillna(False).astype(bool).astype(int) * 20
    )
    return pd.Series(np.clip(score, 0.0, 100.0).round(2), index=df.index, dtype="float64")


def audit_field_availability(df: pd.DataFrame, field_names: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    total_rows = len(df)
    for field_name in field_names:
        series = df[field_name].astype("string")
        nonnull = int(series.notna().sum())
        blank_like = int(series.fillna("").str.strip().eq("").sum())
        records.append(
            {
                "section": "source_audit_statewide",
                "metric": field_name,
                "county_name": pd.NA,
                "count": nonnull,
                "pct": round(nonnull / total_rows * 100.0, 4),
                "value": nonnull,
                "notes": f"blank_like_rows={blank_like}",
            }
        )
    return pd.DataFrame.from_records(records)


def audit_county_inconsistency(df: pd.DataFrame, field_names: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = df.groupby("county_name", dropna=False)
    for field_name in field_names:
        county_blank = grouped[field_name].apply(lambda x: x.astype("string").fillna("").str.strip().eq("").mean() * 100.0)
        for county_name, pct in county_blank[county_blank > 0.0].sort_values(ascending=False).head(10).items():
            records.append(
                {
                    "section": "source_audit_county_inconsistency",
                    "metric": field_name,
                    "county_name": county_name,
                    "count": pd.NA,
                    "pct": round(float(pct), 4),
                    "value": round(float(pct), 4),
                    "notes": "blank_like_pct",
                }
            )
    return pd.DataFrame.from_records(records)


def build_summary(owner_leads: gpd.GeoDataFrame, audit_fields: list[str], runtime_seconds: float) -> pd.DataFrame:
    total_rows = len(owner_leads)
    rows: list[dict[str, object]] = []

    rows.extend(audit_field_availability(owner_leads, audit_fields).to_dict("records"))
    rows.extend(audit_county_inconsistency(owner_leads, audit_fields).to_dict("records"))

    rows.extend(
        [
            {"section": "validation", "metric": "parcel_rows", "county_name": pd.NA, "count": total_rows, "pct": 100.0, "value": total_rows, "notes": "final_owner_layer_rows"},
            {"section": "validation", "metric": "parcel_id_duplicates", "county_name": pd.NA, "count": int(owner_leads["parcel_id"].duplicated().sum()), "pct": pd.NA, "value": int(owner_leads["parcel_id"].duplicated().sum()), "notes": "expected_zero"},
            {"section": "validation", "metric": "parcel_row_id_duplicates", "county_name": pd.NA, "count": int(owner_leads["parcel_row_id"].duplicated().sum()), "pct": pd.NA, "value": int(owner_leads["parcel_row_id"].duplicated().sum()), "notes": "expected_zero"},
            {"section": "validation", "metric": "owner_layer_columns", "county_name": pd.NA, "count": owner_leads.shape[1], "pct": pd.NA, "value": owner_leads.shape[1], "notes": "final_column_count"},
            {"section": "validation", "metric": "runtime_seconds", "county_name": pd.NA, "count": pd.NA, "pct": pd.NA, "value": round(runtime_seconds, 2), "notes": "owner_leads_build_runtime"},
        ]
    )

    key_null_fields = [
        "owner_name_raw",
        "owner_name_normalized",
        "mailing_address_line1",
        "mailing_city",
        "mailing_state",
        "mailing_zip",
        "property_address_raw",
        "property_city",
        "property_state",
    ]
    for field_name in key_null_fields:
        null_count = int(owner_leads[field_name].isna().sum())
        rows.append(
            {
                "section": "null_rates",
                "metric": field_name,
                "county_name": pd.NA,
                "count": null_count,
                "pct": round(null_count / total_rows * 100.0, 4),
                "value": null_count,
                "notes": "null_count",
            }
        )

    absentee_true = owner_leads["absentee_owner_flag"].eq(True).sum()
    out_of_state_true = owner_leads["out_of_state_owner_flag"].eq(True).sum()
    owner_occupied_true = owner_leads["owner_occupied_proxy_flag"].eq(True).sum()
    tax_delinquent_true = owner_leads["tax_delinquent_flag"].eq(True).sum()
    distressed_true = owner_leads["distressed_owner_flag"].eq(True).sum()
    multi_owner_groups = int(owner_leads.loc[owner_leads["owner_parcel_count"] > 1, "owner_group_id"].nunique())
    rows.extend(
        [
            {"section": "rates", "metric": "absentee_owner_rate", "county_name": pd.NA, "count": int(absentee_true), "pct": round(absentee_true / total_rows * 100.0, 4), "value": int(absentee_true), "notes": "true_rows_only"},
            {"section": "rates", "metric": "out_of_state_owner_rate", "county_name": pd.NA, "count": int(out_of_state_true), "pct": round(out_of_state_true / total_rows * 100.0, 4), "value": int(out_of_state_true), "notes": "true_rows_only"},
            {"section": "rates", "metric": "owner_occupied_proxy_rate", "county_name": pd.NA, "count": int(owner_occupied_true), "pct": round(owner_occupied_true / total_rows * 100.0, 4), "value": int(owner_occupied_true), "notes": "true_rows_only"},
            {"section": "rates", "metric": "tax_delinquent_rate", "county_name": pd.NA, "count": int(tax_delinquent_true), "pct": round(tax_delinquent_true / total_rows * 100.0, 4), "value": int(tax_delinquent_true), "notes": "true_rows_only"},
            {"section": "rates", "metric": "distressed_owner_rate", "county_name": pd.NA, "count": int(distressed_true), "pct": round(distressed_true / total_rows * 100.0, 4), "value": int(distressed_true), "notes": "true_rows_only"},
            {"section": "rates", "metric": "multi_parcel_owner_groups", "county_name": pd.NA, "count": multi_owner_groups, "pct": pd.NA, "value": multi_owner_groups, "notes": "owner_group_id_with_owner_parcel_count_gt_1"},
        ]
    )

    for field_name in ["owner_type", "owner_name_confidence_tier", "mailing_completeness_score"]:
        distribution = owner_leads[field_name].value_counts(dropna=False)
        for label, count in distribution.items():
            rows.append(
                {
                    "section": "distribution",
                    "metric": field_name,
                    "county_name": pd.NA,
                    "count": int(count),
                    "pct": round(int(count) / total_rows * 100.0, 4),
                    "value": label if pd.notna(label) else "<NA>",
                    "notes": "value_count",
                }
            )

    absentee_by_county = (
        owner_leads.assign(absentee_num=owner_leads["absentee_owner_flag"].fillna(False).astype(bool).astype(int))
        .groupby("county_name", as_index=False)
        .agg(parcel_rows=("parcel_row_id", "size"), absentee_rows=("absentee_num", "sum"))
    )
    absentee_by_county["absentee_pct"] = absentee_by_county["absentee_rows"] / absentee_by_county["parcel_rows"] * 100.0
    for row in absentee_by_county.sort_values("absentee_pct", ascending=False).head(10).itertuples(index=False):
        rows.append(
            {
                "section": "top_counties",
                "metric": "absentee_owner_pct",
                "county_name": row.county_name,
                "count": int(row.absentee_rows),
                "pct": round(float(row.absentee_pct), 4),
                "value": int(row.parcel_rows),
                "notes": "absentee_rows / parcel_rows",
            }
        )

    out_of_state_by_county = (
        owner_leads.assign(out_of_state_num=owner_leads["out_of_state_owner_flag"].fillna(False).astype(bool).astype(int))
        .groupby("county_name", as_index=False)
        .agg(parcel_rows=("parcel_row_id", "size"), out_of_state_rows=("out_of_state_num", "sum"))
    )
    out_of_state_by_county["out_of_state_pct"] = out_of_state_by_county["out_of_state_rows"] / out_of_state_by_county["parcel_rows"] * 100.0
    for row in out_of_state_by_county.sort_values("out_of_state_pct", ascending=False).head(10).itertuples(index=False):
        rows.append(
            {
                "section": "top_counties",
                "metric": "out_of_state_owner_pct",
                "county_name": row.county_name,
                "count": int(row.out_of_state_rows),
                "pct": round(float(row.out_of_state_pct), 4),
                "value": int(row.parcel_rows),
                "notes": "out_of_state_rows / parcel_rows",
            }
        )

    top_groups = (
        owner_leads.loc[:, ["owner_group_id", "owner_name_normalized", "mailing_city", "mailing_state", "owner_parcel_count", "owner_total_acres"]]
        .drop_duplicates("owner_group_id")
        .sort_values(["owner_parcel_count", "owner_total_acres"], ascending=[False, False])
        .head(20)
    )
    for row in top_groups.itertuples(index=False):
        mailing_city = "" if pd.isna(row.mailing_city) else row.mailing_city
        mailing_state = "" if pd.isna(row.mailing_state) else row.mailing_state
        rows.append(
            {
                "section": "top_owner_groups",
                "metric": row.owner_group_id,
                "county_name": pd.NA,
                "count": int(row.owner_parcel_count),
                "pct": pd.NA,
                "value": row.owner_name_normalized,
                "notes": f"{mailing_city}, {mailing_state}; total_acres={row.owner_total_acres}",
            }
        )

    quality_issues = [
        (
            "unusable_owner_names",
            int(owner_leads["owner_name_confidence_tier"].eq("low").sum()),
            "blank, generic, or otherwise weak owner_name_normalized values",
        ),
        (
            "null_mailing_states",
            int(owner_leads["mailing_state"].isna().sum()),
            "mailing_state missing after primary/fallback normalization",
        ),
        (
            "po_box_rows",
            int(owner_leads["mailing_address_line1"].fillna("").str.contains(r"\bPO BOX\b", regex=True).sum()),
            "PO Box mailing rows may overstate absentee detection precision",
        ),
        (
            "missing_property_addresses",
            int(owner_leads["property_address_raw"].isna().sum()),
            "missing situs address prevents exact owner occupancy proxy",
        ),
        (
            "blank_site_city_rows",
            int(owner_leads["property_city"].isna().sum()),
            "site city is structurally sparse in source parcel records",
        ),
        (
            "tax_data_unavailable_rows",
            int(owner_leads["tax_data_available_flag"].fillna(False).eq(False).sum()),
            "No matched county tax source record for the parcel.",
        ),
    ]
    for metric, count, note in quality_issues:
        rows.append(
            {
                "section": "data_quality",
                "metric": metric,
                "county_name": pd.NA,
                "count": count,
                "pct": round(count / total_rows * 100.0, 4),
                "value": count,
                "notes": note,
            }
        )

    scoring_rules = [
        ("absentee_owner_flag", 25.0),
        ("out_of_state_owner_flag", 15.0),
        ("corporate_owner_flag", 10.0),
        ("tax_delinquent_flag", 12.0),
        ("tax_distress_score", 4.0),
        ("mailing_completeness_score", 5.0),
        ("investment_score", 0.20),
        ("buildability_score", 0.15),
        ("environment_score", 0.05),
        ("owner_parcel_count_extra", 2.0),
        ("owner_occupied_proxy_flag", -20.0),
    ]
    for metric, weight in scoring_rules:
        rows.append(
            {
                "section": "mailer_target_rules",
                "metric": metric,
                "county_name": pd.NA,
                "count": pd.NA,
                "pct": pd.NA,
                "value": weight,
                "notes": "deterministic scoring rule input weight",
            }
        )

    return pd.DataFrame.from_records(rows)


def build_schema() -> pd.DataFrame:
    records = [
        {"field_name": "parcel_row_id", "source_fields": "parcel_row_id", "field_status": "source", "meaning": "Stable internal parcel row key.", "datatype": "string", "null_handling": "must be non-null", "assumptions": "Inherited from master parcel identity model."},
        {"field_name": "parcel_id", "source_fields": "parcel_id", "field_status": "source", "meaning": "Current canonical parcel identifier for Mississippi product outputs.", "datatype": "string", "null_handling": "must be non-null", "assumptions": "Equals parcel_row_id in the current master layer."},
        {"field_name": "state_code", "source_fields": "state_code", "field_status": "source", "meaning": "Two-letter state abbreviation.", "datatype": "string", "null_handling": "must be non-null", "assumptions": "Mississippi rows only in this layer."},
        {"field_name": "county_name", "source_fields": "county_name", "field_status": "source", "meaning": "County name from parcel source.", "datatype": "string", "null_handling": "must be non-null", "assumptions": "Used for county aggregations."},
        {"field_name": "county_fips", "source_fields": "county_fips", "field_status": "source", "meaning": "County FIPS code.", "datatype": "string", "null_handling": "must be non-null", "assumptions": "Derived in master from county name."},
        {"field_name": "apn", "source_fields": "source_parcel_id_raw", "field_status": "normalized", "meaning": "Best available assessor parcel number string for outreach exports.", "datatype": "string", "null_handling": "nullable if source ID ever missing", "assumptions": "Uses raw parcel source identifier instead of synthetic parcel_row_id."},
        {"field_name": "source_parcel_id_raw", "source_fields": "source_parcel_id_raw", "field_status": "source", "meaning": "Raw parcel source identifier.", "datatype": "string", "null_handling": "expected non-null", "assumptions": "Current Mississippi source uses PARNO."},
        {"field_name": "source_parcel_id_normalized", "source_fields": "source_parcel_id_normalized", "field_status": "normalized", "meaning": "Normalized parcel source identifier for multi-state identity readiness.", "datatype": "string", "null_handling": "expected non-null", "assumptions": "Token-preserving normalization from master stage."},
        {"field_name": "owner_name_raw", "source_fields": "owner_name", "field_status": "source", "meaning": "Raw owner name string from parcel source.", "datatype": "string", "null_handling": "nullable when blank-like", "assumptions": "Whitespace collapsed and blank strings converted to null."},
        {"field_name": "owner_name_normalized", "source_fields": "owner_name", "field_status": "normalized", "meaning": "Normalized owner name used for grouping and owner typing.", "datatype": "string", "null_handling": "nullable when unusable", "assumptions": "Uppercase, spacing normalized, light punctuation cleanup only."},
        {"field_name": "owner_name_2_raw", "source_fields": "", "field_status": "derived", "meaning": "Secondary owner name placeholder.", "datatype": "string", "null_handling": "currently always null", "assumptions": "No reliable co-owner field exists statewide in current source."},
        {"field_name": "mailing_address_raw", "source_fields": "mail_address_1, mail_address_2", "field_status": "normalized", "meaning": "Concatenated raw mailing address string for reference.", "datatype": "string", "null_handling": "nullable when both components blank", "assumptions": "Built from line1 + line2 without aggressive parsing."},
        {"field_name": "mailing_address_line1", "source_fields": "mail_address_1", "field_status": "normalized", "meaning": "Mailing address first line.", "datatype": "string", "null_handling": "nullable when blank", "assumptions": "Whitespace normalized only."},
        {"field_name": "mailing_address_line2", "source_fields": "mail_address_2", "field_status": "normalized", "meaning": "Mailing address second line.", "datatype": "string", "null_handling": "nullable when blank", "assumptions": "Whitespace normalized only."},
        {"field_name": "mailing_city", "source_fields": "mail_city_1, mail_city_2", "field_status": "normalized", "meaning": "Best available mailing city.", "datatype": "string", "null_handling": "nullable when both source cities blank", "assumptions": "Uses primary then secondary fallback."},
        {"field_name": "mailing_state", "source_fields": "mail_state_1, mail_state_2", "field_status": "normalized", "meaning": "Best available mailing state abbreviation.", "datatype": "string", "null_handling": "nullable when both source states blank or invalid", "assumptions": "Normalized to USPS-style two-letter abbreviation when possible."},
        {"field_name": "mailing_zip", "source_fields": "mail_zip_1, mail_zip_2", "field_status": "normalized", "meaning": "Best available mailing ZIP string.", "datatype": "string", "null_handling": "nullable when both source zips blank", "assumptions": "Preserves source formatting if present."},
        {"field_name": "mailing_zip5", "source_fields": "mail_zip_1, mail_zip_2", "field_status": "normalized", "meaning": "Five-digit mailing ZIP for grouping and QA.", "datatype": "string", "null_handling": "nullable when a 5-digit ZIP cannot be extracted", "assumptions": "Extracts the first 5 digits only."},
        {"field_name": "property_address_raw", "source_fields": "site_address", "field_status": "normalized", "meaning": "Situs/property address.", "datatype": "string", "null_handling": "nullable when blank", "assumptions": "Whitespace normalized only."},
        {"field_name": "property_city", "source_fields": "site_city", "field_status": "normalized", "meaning": "Situs/property city.", "datatype": "string", "null_handling": "nullable when blank", "assumptions": "Source coverage is structurally sparse statewide."},
        {"field_name": "property_state", "source_fields": "site_state", "field_status": "normalized", "meaning": "Situs/property state abbreviation.", "datatype": "string", "null_handling": "nullable when invalid", "assumptions": "Expected to normalize to MS for all Mississippi parcels."},
        {"field_name": "land_use_raw", "source_fields": "zoning", "field_status": "source", "meaning": "Best available land use or zoning-like parcel classifier.", "datatype": "string", "null_handling": "nullable when blank", "assumptions": "Current statewide zoning coverage is weak and should not be treated as canonical land use."},
        {"field_name": "tax_status", "source_fields": "tax_status", "field_status": "source", "meaning": "Tax status text from parcel source.", "datatype": "string", "null_handling": "nullable when blank", "assumptions": "Useful for later tax delinquency additions but not yet standardized."},
        {"field_name": "parcel_acres", "source_fields": "total_acres, tax_acres, gis_acres", "field_status": "normalized", "meaning": "Best available acreage for the parcel.", "datatype": "float64", "null_handling": "nullable when all acreage fields missing", "assumptions": "Uses total_acres first, then tax_acres, then gis_acres."},
        {"field_name": "owner_total_acres", "source_fields": "total_acres", "field_status": "derived", "meaning": "Total acres across parcels in the owner group.", "datatype": "float64", "null_handling": "0.0 when acreage unavailable for every group parcel", "assumptions": "Uses parcel_acres rollup within owner_group_id."},
        {"field_name": "absentee_owner_flag", "source_fields": "mailing address fields + situs address fields", "field_status": "derived", "meaning": "True when mailing and situs addresses differ after light normalization, or when mailing state is out of state.", "datatype": "boolean", "null_handling": "nullable when address comparison is not possible and no out-of-state signal exists", "assumptions": "Comparison is deterministic, not fuzzy matching or skip tracing."},
        {"field_name": "absentee_owner_reason", "source_fields": "mailing address fields + situs address fields", "field_status": "derived", "meaning": "Reason code explaining absentee_owner_flag state.", "datatype": "string", "null_handling": "nullable only if rule did not run", "assumptions": "Values include matched_address, different_address, missing_mailing_address, missing_property_address, po_box_only, out_of_state_owner."},
        {"field_name": "out_of_state_owner_flag", "source_fields": "mailing_state", "field_status": "derived", "meaning": "True when mailing state is present and not Mississippi.", "datatype": "boolean", "null_handling": "nullable when mailing_state missing", "assumptions": "Property state is expected to be Mississippi."},
        {"field_name": "owner_occupied_proxy_flag", "source_fields": "mailing address fields + situs address fields", "field_status": "derived", "meaning": "Proxy flag for likely owner occupancy when mailing and situs addresses match.", "datatype": "boolean", "null_handling": "nullable when address comparison is not possible", "assumptions": "Proxy only; not occupancy truth."},
        {"field_name": "owner_occupied_proxy_reason", "source_fields": "mailing address fields + situs address fields", "field_status": "derived", "meaning": "Reason code explaining owner_occupied_proxy_flag state.", "datatype": "string", "null_handling": "nullable only if rule did not run", "assumptions": "Mirrors address comparison reason codes."},
        {"field_name": "owner_type", "source_fields": "owner_name_normalized", "field_status": "derived", "meaning": "Owner category for targeting filters.", "datatype": "string", "null_handling": "never null; defaults to unknown or individual", "assumptions": "Regex-based classification into individual, llc, corporation, trust, government, nonprofit, unknown."},
        {"field_name": "corporate_owner_flag", "source_fields": "owner_type", "field_status": "derived", "meaning": "True for llc, corporation, trust, government, and nonprofit owner types.", "datatype": "boolean", "null_handling": "never null", "assumptions": "Entity-style ownership proxy for lead targeting."},
        {"field_name": "mailing_completeness_score", "source_fields": "mailing_address_line1, mailing_city, mailing_state, mailing_zip5", "field_status": "derived", "meaning": "Count of core mailing components present from 0 to 4.", "datatype": "int8", "null_handling": "never null", "assumptions": "One point each for line1, city, state, and zip5."},
        {"field_name": "owner_name_confidence_tier", "source_fields": "owner_name_normalized", "field_status": "derived", "meaning": "Usability tier for outreach based on owner name quality.", "datatype": "string", "null_handling": "never null", "assumptions": "High/Medium/Low deterministic rule set."},
        {"field_name": "owner_group_id", "source_fields": "owner_name_normalized + mailing fields", "field_status": "derived", "meaning": "Stable hashed group identifier for likely same owner.", "datatype": "string", "null_handling": "never null", "assumptions": "Based on normalized owner plus normalized mailing key, with parcel fallback when both missing."},
        {"field_name": "owner_parcel_count", "source_fields": "owner_group_id", "field_status": "derived", "meaning": "Count of parcels in the owner group.", "datatype": "int32", "null_handling": "never null", "assumptions": "Computed statewide across the final owner layer."},
        {"field_name": "tax_data_available_flag", "source_fields": "tax distress stage parcel join", "field_status": "derived", "meaning": "True when a matched normalized tax record exists for the parcel.", "datatype": "boolean", "null_handling": "never null", "assumptions": "False includes both unmatched parcels and counties without local tax source files."},
        {"field_name": "tax_delinquent_flag", "source_fields": "tax distress stage parcel join", "field_status": "derived", "meaning": "True when any matched tax record is delinquent or tax sale related.", "datatype": "boolean", "null_handling": "never null", "assumptions": "Derived from canonical normalized tax statuses."},
        {"field_name": "tax_distress_score", "source_fields": "tax distress stage + owner signals", "field_status": "derived", "meaning": "Deterministic 0-10 tax distress score.", "datatype": "float64", "null_handling": "never null", "assumptions": "Uses delinquency severity, tax amount, absentee, out-of-state, and single-parcel ownership."},
        {"field_name": "distressed_owner_flag", "source_fields": "tax distress stage", "field_status": "derived", "meaning": "True when the owner meets the parcel distress threshold or has delinquent tax status.", "datatype": "boolean", "null_handling": "never null", "assumptions": "Configured in the tax distress stage."},
        {"field_name": "mailer_target_score", "source_fields": "ownership flags + mailing completeness + parcel scores", "field_status": "derived", "meaning": "Deterministic 0-100 outreach targeting score.", "datatype": "float64", "null_handling": "never null", "assumptions": "Weighted sum clipped to 0-100; see summary CSV for exact rule weights."},
        {"field_name": "buildability_score", "source_fields": "buildability_score", "field_status": "source", "meaning": "Composite buildability score from earlier parcel scoring stage.", "datatype": "float64", "null_handling": "nullable if score missing upstream", "assumptions": "Used directly in mailer target scoring."},
        {"field_name": "investment_score", "source_fields": "investment_score", "field_status": "source", "meaning": "Composite investment score from earlier parcel scoring stage.", "datatype": "float64", "null_handling": "nullable if score missing upstream", "assumptions": "Used directly in mailer target scoring."},
        {"field_name": "environment_score", "source_fields": "environment_score", "field_status": "source", "meaning": "Composite environmental score from earlier parcel scoring stage.", "datatype": "float64", "null_handling": "nullable if score missing upstream", "assumptions": "Used directly in mailer target scoring."},
        {"field_name": "parcel_constraint_summary", "source_fields": "parcel_constraint_summary", "field_status": "source", "meaning": "Human-readable parcel constraints summary.", "datatype": "string", "null_handling": "nullable", "assumptions": "Carried through from master for downstream filtering and exports."},
        {"field_name": "geometry", "source_fields": "geometry", "field_status": "source", "meaning": "Parcel geometry.", "datatype": "geometry", "null_handling": "must be non-null for GPKG output", "assumptions": "Retained for geospatial product use."},
    ]
    return pd.DataFrame.from_records(records)


def build_county_qa(owner_leads: pd.DataFrame) -> pd.DataFrame:
    grouped = owner_leads.groupby("county_name", as_index=False).agg(
        parcel_rows=("parcel_row_id", "size"),
        owner_name_raw_populated=("owner_name_raw", lambda x: int(pd.Series(x).notna().sum())),
        mailing_address_line1_populated=("mailing_address_line1", lambda x: int(pd.Series(x).notna().sum())),
        mailing_state_populated=("mailing_state", lambda x: int(pd.Series(x).notna().sum())),
        property_address_raw_populated=("property_address_raw", lambda x: int(pd.Series(x).notna().sum())),
        property_city_populated=("property_city", lambda x: int(pd.Series(x).notna().sum())),
        absentee_rows=("absentee_owner_flag", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
        out_of_state_rows=("out_of_state_owner_flag", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
        multi_parcel_rows=("owner_parcel_count", lambda x: int((pd.Series(x) > 1).sum())),
        tax_delinquent_rows=("tax_delinquent_flag", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
        distressed_owner_rows=("distressed_owner_flag", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
    )
    for source_col, rate_col in [
        ("owner_name_raw_populated", "owner_name_raw_pct"),
        ("mailing_address_line1_populated", "mailing_address_line1_pct"),
        ("mailing_state_populated", "mailing_state_pct"),
        ("property_address_raw_populated", "property_address_raw_pct"),
        ("property_city_populated", "property_city_pct"),
        ("absentee_rows", "absentee_owner_pct"),
        ("out_of_state_rows", "out_of_state_owner_pct"),
        ("multi_parcel_rows", "multi_parcel_owner_row_pct"),
        ("tax_delinquent_rows", "tax_delinquent_pct"),
        ("distressed_owner_rows", "distressed_owner_pct"),
    ]:
        grouped[rate_col] = (grouped[source_col] / grouped["parcel_rows"] * 100.0).round(4)
    return grouped.sort_values("parcel_rows", ascending=False).reset_index(drop=True)


def write_outputs(
    owner_leads: gpd.GeoDataFrame,
    mailer_export: pd.DataFrame,
    summary: pd.DataFrame,
    schema: pd.DataFrame,
    county_qa: pd.DataFrame,
    output_parquet: Path,
    output_gpkg: Path,
    mailer_export_csv: Path,
    summary_csv: Path,
    schema_csv: Path,
    county_qa_csv: Path,
) -> None:
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    owner_leads.to_parquet(output_parquet, index=False)
    owner_leads.to_file(output_gpkg, driver="GPKG", engine="pyogrio")
    mailer_export.to_csv(mailer_export_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    schema.to_csv(schema_csv, index=False)
    county_qa.to_csv(county_qa_csv, index=False)


def main() -> None:
    args = parse_args()
    input_file = resolve_path(args.input_file)
    output_parquet = resolve_path(args.output_parquet)
    output_gpkg = resolve_path(args.output_gpkg)
    mailer_export_csv = resolve_path(args.mailer_export_csv)
    summary_csv = resolve_path(args.summary_csv)
    schema_csv = resolve_path(args.schema_csv)
    county_qa_csv = resolve_path(args.county_qa_csv)
    tax_distress_file = resolve_path(args.tax_distress_file)

    start = time.perf_counter()
    print(f"Loading master parcels from {input_file.relative_to(BASE_DIR)}")
    master = gpd.read_file(input_file, columns=OWNER_SOURCE_FIELDS, engine="pyogrio")

    owner_leads = master.copy()
    owner_leads["apn"] = blank_to_na(select_series(owner_leads, "apn", "source_parcel_id_raw")).where(
        blank_to_na(select_series(owner_leads, "apn", "source_parcel_id_raw")).notna(),
        blank_to_na(owner_leads["source_parcel_id_raw"]),
    )
    owner_leads["source_parcel_id_raw"] = blank_to_na(owner_leads["source_parcel_id_raw"])
    owner_leads["source_parcel_id_normalized"] = blank_to_na(owner_leads["source_parcel_id_normalized"])
    owner_leads["owner_name_raw"] = collapse_spaces(select_series(owner_leads, "owner_name_raw", "owner_name"))
    owner_leads["owner_name_normalized"] = normalize_owner_name(owner_leads["owner_name_raw"])
    owner_leads["owner_name_2_raw"] = collapse_spaces(select_series(owner_leads, "owner_name_2_raw"))
    owner_leads["mailing_address_line1"] = collapse_spaces(select_series(owner_leads, "mailing_address_line1_raw", "mail_address_1"))
    owner_leads["mailing_address_line2"] = collapse_spaces(select_series(owner_leads, "mailing_address_line2_raw", "mail_address_2"))
    owner_leads["mailing_address_raw"] = (
        owner_leads["mailing_address_line1"].fillna("") + " " + owner_leads["mailing_address_line2"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    owner_leads["mailing_address_raw"] = owner_leads["mailing_address_raw"].mask(owner_leads["mailing_address_raw"].eq(""), pd.NA).astype("string")
    owner_leads["mailing_city"] = choose_with_fallback(
        select_series(owner_leads, "mailing_city_raw", "mail_city_1"),
        select_series(owner_leads, "mail_city_2"),
    )
    owner_leads["mailing_state"] = normalize_state(
        choose_with_fallback(
            select_series(owner_leads, "mailing_state_raw", "mail_state_1"),
            select_series(owner_leads, "mail_state_2"),
        )
    )
    owner_leads["mailing_zip"] = choose_with_fallback(
        select_series(owner_leads, "mailing_zip_raw", "mail_zip_1"),
        select_series(owner_leads, "mail_zip_2"),
    )
    owner_leads["mailing_zip5"] = normalize_zip5(owner_leads["mailing_zip"])
    owner_leads["property_address_raw"] = collapse_spaces(select_series(owner_leads, "property_address_raw", "site_address"))
    owner_leads["property_city"] = collapse_spaces(select_series(owner_leads, "property_city_raw", "site_city"))
    owner_leads["property_state"] = normalize_state(select_series(owner_leads, "property_state_raw", "site_state"))
    owner_leads["land_use_raw"] = collapse_spaces(select_series(owner_leads, "land_use_raw", "zoning"))
    owner_leads["tax_status"] = collapse_spaces(owner_leads["tax_status"])

    owner_leads["parcel_acres"] = pd.to_numeric(owner_leads["total_acres"], errors="coerce")
    owner_leads["parcel_acres"] = owner_leads["parcel_acres"].where(owner_leads["parcel_acres"].notna(), pd.to_numeric(owner_leads["tax_acres"], errors="coerce"))
    owner_leads["parcel_acres"] = owner_leads["parcel_acres"].where(owner_leads["parcel_acres"].notna(), pd.to_numeric(owner_leads["gis_acres"], errors="coerce"))

    owner_leads["owner_type"] = build_owner_type(owner_leads["owner_name_normalized"])
    owner_leads["corporate_owner_flag"] = owner_leads["owner_type"].isin(["llc", "corporation", "trust", "government", "nonprofit"]).astype("boolean")
    owner_leads["owner_name_confidence_tier"] = build_owner_name_confidence(owner_leads["owner_name_normalized"])
    owner_leads["mailing_completeness_score"] = build_mailing_completeness_score(
        owner_leads["mailing_address_line1"],
        owner_leads["mailing_city"],
        owner_leads["mailing_state"],
        owner_leads["mailing_zip5"],
    )
    owner_leads["out_of_state_owner_flag"] = pd.Series(pd.NA, index=owner_leads.index, dtype="boolean")
    has_state = owner_leads["mailing_state"].notna()
    owner_leads.loc[has_state, "out_of_state_owner_flag"] = owner_leads.loc[has_state, "mailing_state"].ne(STATE_ABBR)

    owner_leads["owner_occupied_proxy_flag"], owner_leads["owner_occupied_proxy_reason"] = build_address_match(
        owner_leads["mailing_address_line1"],
        owner_leads["property_address_raw"],
        owner_leads["mailing_city"],
        owner_leads["property_city"],
        owner_leads["mailing_zip5"],
        owner_leads["mailing_state"],
        owner_leads["property_state"],
    )

    owner_leads["absentee_owner_flag"] = pd.Series(pd.NA, index=owner_leads.index, dtype="boolean")
    owner_leads["absentee_owner_reason"] = owner_leads["owner_occupied_proxy_reason"].astype("string")
    comparable = owner_leads["owner_occupied_proxy_flag"].notna()
    owner_leads.loc[comparable, "absentee_owner_flag"] = ~owner_leads.loc[comparable, "owner_occupied_proxy_flag"]
    out_of_state_true = owner_leads["out_of_state_owner_flag"].eq(True)
    owner_leads.loc[out_of_state_true, "absentee_owner_flag"] = True
    owner_leads.loc[out_of_state_true, "absentee_owner_reason"] = "out_of_state_owner"

    mailing_key = (
        normalize_address_for_compare(owner_leads["mailing_address_line1"]).fillna("")
        + "|"
        + normalize_address_for_compare(owner_leads["mailing_city"]).fillna("")
        + "|"
        + normalize_state(owner_leads["mailing_state"]).fillna("")
        + "|"
        + owner_leads["mailing_zip5"].fillna("")
    )
    mailing_key = mailing_key.str.strip("|").mask(mailing_key.eq(""), pd.NA).astype("string")
    owner_group_key = standardize_owner_group_key(owner_leads["owner_name_normalized"], owner_leads["owner_type"])
    group_basis = build_owner_group_basis(owner_group_key, mailing_key, owner_leads["parcel_row_id"])
    group_hash = pd.util.hash_pandas_object(group_basis.fillna(""), index=False).astype("uint64").astype(str)
    owner_leads["owner_group_id"] = pd.Series("msown_" + group_hash, index=owner_leads.index, dtype="string")

    owner_agg = owner_leads.groupby("owner_group_id", dropna=False).agg(
        owner_parcel_count=("parcel_row_id", "size"),
        owner_total_acres=("parcel_acres", "sum"),
    )
    owner_agg["owner_total_acres"] = owner_agg["owner_total_acres"].fillna(0.0).round(4)
    owner_leads = owner_leads.merge(owner_agg, left_on="owner_group_id", right_index=True, how="left")
    owner_leads["owner_parcel_count"] = owner_leads["owner_parcel_count"].astype("int32")

    owner_leads["tax_data_available_flag"] = pd.Series(False, index=owner_leads.index, dtype="boolean")
    owner_leads["tax_delinquent_flag"] = pd.Series(False, index=owner_leads.index, dtype="boolean")
    owner_leads["tax_distress_score"] = pd.Series(0.0, index=owner_leads.index, dtype="float64")
    owner_leads["distressed_owner_flag"] = pd.Series(False, index=owner_leads.index, dtype="boolean")
    if tax_distress_file.exists():
        tax_fields = [
            "parcel_row_id",
            "tax_data_available_flag",
            "tax_delinquent_flag",
            "tax_distress_score",
            "distressed_owner_flag",
        ]
        tax_distress = gpd.read_parquet(tax_distress_file, columns=tax_fields).drop(columns="geometry", errors="ignore")
        tax_distress["parcel_row_id"] = tax_distress["parcel_row_id"].astype("string")
        owner_leads = owner_leads.drop(columns=tax_fields[1:], errors="ignore").merge(tax_distress, on="parcel_row_id", how="left")
        owner_leads["tax_data_available_flag"] = owner_leads["tax_data_available_flag"].fillna(False).astype("boolean")
        owner_leads["tax_delinquent_flag"] = owner_leads["tax_delinquent_flag"].fillna(False).astype("boolean")
        owner_leads["tax_distress_score"] = pd.to_numeric(owner_leads["tax_distress_score"], errors="coerce").fillna(0.0).astype("float64")
        owner_leads["distressed_owner_flag"] = owner_leads["distressed_owner_flag"].fillna(False).astype("boolean")

    owner_leads["mailer_target_score"] = build_mailer_target_score(owner_leads)

    final_columns = [
        "parcel_row_id",
        "parcel_id",
        "state_code",
        "county_name",
        "county_fips",
        "apn",
        "source_parcel_id_raw",
        "source_parcel_id_normalized",
        "owner_name_raw",
        "owner_name_normalized",
        "owner_name_2_raw",
        "mailing_address_raw",
        "mailing_address_line1",
        "mailing_address_line2",
        "mailing_city",
        "mailing_state",
        "mailing_zip",
        "mailing_zip5",
        "property_address_raw",
        "property_city",
        "property_state",
        "land_use_raw",
        "tax_status",
        "parcel_acres",
        "owner_total_acres",
        "absentee_owner_flag",
        "absentee_owner_reason",
        "out_of_state_owner_flag",
        "owner_occupied_proxy_flag",
        "owner_occupied_proxy_reason",
        "owner_type",
        "corporate_owner_flag",
        "mailing_completeness_score",
        "owner_name_confidence_tier",
        "owner_group_id",
        "owner_parcel_count",
        "tax_data_available_flag",
        "tax_delinquent_flag",
        "tax_distress_score",
        "distressed_owner_flag",
        "mailer_target_score",
        "buildability_score",
        "investment_score",
        "environment_score",
        "parcel_constraint_summary",
        "geometry",
    ]
    owner_leads = owner_leads.loc[:, final_columns].copy()

    summary = build_summary(
        owner_leads,
        audit_fields=[
            "source_parcel_id_raw",
            "source_parcel_id_normalized",
            "owner_name_raw",
            "mailing_address_line1",
            "mailing_city",
            "mailing_state",
            "mailing_zip",
            "property_address_raw",
            "property_city",
            "property_state",
            "land_use_raw",
            "tax_status",
            "tax_data_available_flag",
            "tax_delinquent_flag",
            "distressed_owner_flag",
        ],
        runtime_seconds=time.perf_counter() - start,
    )
    schema = build_schema()
    county_qa = build_county_qa(owner_leads)
    mailer_export = owner_leads.loc[:, MAILER_EXPORT_FIELDS].copy()

    write_outputs(
        owner_leads=owner_leads,
        mailer_export=mailer_export,
        summary=summary,
        schema=schema,
        county_qa=county_qa,
        output_parquet=output_parquet,
        output_gpkg=output_gpkg,
        mailer_export_csv=mailer_export_csv,
        summary_csv=summary_csv,
        schema_csv=schema_csv,
        county_qa_csv=county_qa_csv,
    )

    runtime_seconds = time.perf_counter() - start
    print(f"Owner leads build complete in {runtime_seconds / 60.0:.2f} minutes")
    print(f"Rows: {len(owner_leads):,}")
    print(f"Columns: {owner_leads.shape[1]}")
    print(f"Absentee owners: {int(owner_leads['absentee_owner_flag'].eq(True).sum()):,}")
    print(f"Out-of-state owners: {int(owner_leads['out_of_state_owner_flag'].eq(True).sum()):,}")
    print(f"Multi-parcel owner groups: {int(owner_leads.loc[owner_leads['owner_parcel_count'] > 1, 'owner_group_id'].nunique()):,}")


if __name__ == "__main__":
    main()
