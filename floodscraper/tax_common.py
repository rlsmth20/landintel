from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_TAX_DIR = DATA_DIR / "raw_tax"
TAX_STANDARDIZED_DIR = DATA_DIR / "tax_standardized"
TAX_LINKED_DIR = DATA_DIR / "tax_linked"
TAX_PUBLISHED_DIR = DATA_DIR / "tax_published"
TAX_METADATA_DIR = DATA_DIR / "tax_metadata"
PARCELS_RAW_DIR = DATA_DIR / "parcels_raw"
PARCELS_STANDARDIZED_DIR = DATA_DIR / "parcels_standardized"
PARCELS_DIR = DATA_DIR / "parcels"

CANONICAL_TAX_COLUMNS = [
    "tax_record_row_id",
    "parcel_row_id",
    "parcel_id_raw",
    "parcel_id_normalized",
    "state_code",
    "county_fips",
    "county_name",
    "source_name",
    "source_type",
    "source_dataset_path",
    "source_record_id",
    "source_ppin",
    "ingestion_run_id",
    "source_file_version",
    "loaded_at",
    "owner_name",
    "owner_name_2",
    "owner_address_line1",
    "owner_address_line2",
    "owner_city",
    "owner_state",
    "owner_zip",
    "situs_address",
    "situs_city",
    "situs_state",
    "situs_zip",
    "assessed_land_value",
    "assessed_improvement_value",
    "assessed_total_value",
    "market_land_value",
    "market_improvement_value",
    "market_total_value",
    "taxable_value",
    "exemptions_text",
    "exemptions_amount",
    "tax_year",
    "bill_year",
    "tax_amount_due",
    "tax_amount_paid",
    "tax_balance_due",
    "tax_status",
    "payment_status",
    "delinquent_flag",
    "forfeited_flag",
    "delinquent_amount",
    "delinquent_years",
    "delinquent_as_of_date",
    "last_payment_date",
    "due_date",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "owner_corporate_flag",
    "mailing_matches_situs_flag",
    "tax_delinquent_flag_standardized",
    "raw_payload_json",
    "record_hash",
]

COUNTY_FIPS_MAP = {
    "adams": "001",
    "alcorn": "003",
    "amite": "005",
    "attala": "007",
    "benton": "009",
    "bolivar": "011",
    "calhoun": "013",
    "carroll": "015",
    "chickasaw": "017",
    "choctaw": "019",
    "claiborne": "021",
    "clarke": "023",
    "clay": "025",
    "coahoma": "027",
    "copiah": "029",
    "covington": "031",
    "desoto": "033",
    "forrest": "035",
    "franklin": "037",
    "george": "039",
    "greene": "041",
    "grenada": "043",
    "hancock": "045",
    "harrison": "047",
    "hinds": "049",
    "holmes": "051",
    "humphreys": "053",
    "issaquena": "055",
    "itawamba": "057",
    "jackson": "059",
    "jasper": "061",
    "jefferson": "063",
    "jefferson_davis": "065",
    "jones": "067",
    "kemper": "069",
    "lafayette": "071",
    "lamar": "073",
    "lauderdale": "075",
    "lawrence": "077",
    "leake": "079",
    "lee": "081",
    "leflore": "083",
    "lincoln": "085",
    "lowndes": "087",
    "madison": "089",
    "marion": "091",
    "marshall": "093",
    "monroe": "095",
    "montgomery": "097",
    "neshoba": "099",
    "newton": "101",
    "noxubee": "103",
    "oktibbeha": "105",
    "panola": "107",
    "pearl_river": "109",
    "perry": "111",
    "pike": "113",
    "pontotoc": "115",
    "prentiss": "117",
    "quitman": "119",
    "rankin": "121",
    "scott": "123",
    "sharkey": "125",
    "simpson": "127",
    "smith": "129",
    "stone": "131",
    "sunflower": "133",
    "tallahatchie": "135",
    "tate": "137",
    "tippah": "139",
    "tishomingo": "141",
    "tunica": "143",
    "union": "145",
    "walthall": "147",
    "warren": "149",
    "washington": "151",
    "wayne": "153",
    "webster": "155",
    "wilkinson": "157",
    "winston": "159",
    "yalobusha": "161",
    "yazoo": "163",
}


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


def sanitize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_county_name(series: pd.Series) -> pd.Series:
    out = series.astype("string").fillna("").map(sanitize_name)
    return out.mask(out.eq(""), pd.NA).astype("string")


def normalize_identifier(series: pd.Series) -> pd.Series:
    out = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", "-", regex=True)
        .str.strip("- ")
    )
    return out.mask(out.eq(""), pd.NA).astype("string")


def clean_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    return out.mask(out.eq(""), pd.NA).astype("string")


def normalize_ppin(series: pd.Series) -> pd.Series:
    extracted = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.replace(r"[^0-9]+", "", regex=True)
        .str.lstrip("0")
    )
    return extracted.mask(extracted.eq(""), pd.NA).astype("string")


def normalize_zip(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d{5})", expand=False)
    return extracted.astype("string").mask(extracted.isna(), pd.NA)


def coerce_float(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def coerce_year(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def choose_first(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate]
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def infer_corporate_owner(owner_name: pd.Series) -> pd.Series:
    raw = owner_name.astype("string").fillna("").str.upper()
    patterns = [
        r"\bLLC\b",
        r"\bINC\b",
        r"\bCORP\b",
        r"\bCORPORATION\b",
        r"\bCOMPANY\b",
        r"\bCO\b",
        r"\bLP\b",
        r"\bLTD\b",
        r"\bTRUST\b",
        r"\bBANK\b",
    ]
    matched = pd.Series(False, index=owner_name.index, dtype="boolean")
    for pattern in patterns:
        matched = (matched | raw.str.contains(pattern, regex=True)).astype("boolean")
    return matched


def infer_payment_status(tax_status: pd.Series) -> pd.Series:
    raw = tax_status.astype("string").fillna("").str.upper()
    values = np.select(
        [
            raw.str.contains(r"PAID|CURRENT", regex=True),
            raw.str.contains(r"PARTIAL", regex=True),
            raw.str.contains(r"DELINQ|PAST\s*DUE|UNPAID|FORFEIT|SALE", regex=True),
        ],
        ["paid", "partial", "unpaid"],
        default=None,
    )
    return pd.Series(values, index=tax_status.index, dtype="string")


def infer_delinquent_flag(tax_status: pd.Series) -> pd.Series:
    raw = tax_status.astype("string").fillna("").str.upper()
    values = raw.str.contains(r"DELINQ|PAST\s*DUE|UNPAID|FORFEIT|SALE", regex=True)
    return pd.Series(values, index=tax_status.index, dtype="boolean")


def build_row_hash(parts: list[Any]) -> str:
    return hashlib.sha1("||".join("" if part is None else str(part) for part in parts).encode("utf-8")).hexdigest()


def build_record_hash(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    subset = frame.loc[:, columns].copy().astype("string").fillna("")
    return pd.Series(
        (
            hashlib.sha1("||".join(str(value) for value in row).encode("utf-8")).hexdigest()
            for row in subset.itertuples(index=False, name=None)
        ),
        index=frame.index,
        dtype="string",
    )


def raw_payload_json(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    payload_rows: list[str] = []
    for row in frame.loc[:, selected].itertuples(index=False, name=None):
        payload = {
            column: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
            for column, value in zip(selected, row)
        }
        payload_rows.append(json.dumps(payload, separators=(",", ":"), default=str))
    return pd.Series(payload_rows, index=frame.index, dtype="string")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def county_fips_from_name(county_name: pd.Series) -> pd.Series:
    normalized = normalize_county_name(county_name)
    return normalized.map(COUNTY_FIPS_MAP).astype("string")


def load_master_index(master_parquet: Path) -> pd.DataFrame:
    columns = [
        "parcel_row_id",
        "county_name",
        "source_parcel_id_normalized",
        "source_ppin",
        "total_acres",
        "owner_name_raw",
        "property_address_raw",
    ]
    master = pd.read_parquet(master_parquet, columns=columns)
    master["county_name"] = normalize_county_name(master["county_name"])
    master["source_parcel_id_normalized"] = master["source_parcel_id_normalized"].astype("string")
    master["source_ppin"] = normalize_ppin(master["source_ppin"])
    master = master.rename(columns={"total_acres": "acreage"})
    return master


def compact_alnum_identifier(series: pd.Series) -> pd.Series:
    compacted = series.astype("string").fillna("").str.upper().str.replace(r"[^A-Z0-9]+", "", regex=True)
    return compacted.mask(compacted.eq(""), pd.NA).astype("string")


def strip_leading_zero_runs(series: pd.Series) -> pd.Series:
    def _transform(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip().upper()
        if not text:
            return None
        parts = re.split(r"([^A-Z0-9]+)", text)
        transformed: list[str] = []
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"[^A-Z0-9]+", part):
                transformed.append(part)
                continue
            runs = re.findall(r"[A-Z]+|\d+", part)
            rebuilt = []
            for run in runs:
                if run.isdigit():
                    rebuilt.append(str(int(run)))
                else:
                    rebuilt.append(run)
            transformed.append("".join(rebuilt))
        candidate = "".join(transformed).strip()
        return candidate or None

    return pd.Series((_transform(value) for value in series), index=series.index, dtype="string")


def apply_identifier_variant(series: pd.Series, variant_name: str) -> pd.Series:
    if variant_name == "compact_alnum":
        return compact_alnum_identifier(series)
    if variant_name == "strip_leading_zero_runs":
        return strip_leading_zero_runs(series)
    if variant_name == "compact_alnum_strip_leading_zero_runs":
        return compact_alnum_identifier(strip_leading_zero_runs(series))
    raise ValueError(f"Unsupported identifier variant: {variant_name}")


def link_standardized_tax_records(
    standardized: pd.DataFrame,
    master: pd.DataFrame,
    heuristic_variants_by_county: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = standardized.copy()
    working["county_name"] = normalize_county_name(working["county_name"])
    working["source_ppin"] = normalize_ppin(working.get("source_ppin", pd.Series(pd.NA, index=working.index)))
    working["parcel_id_normalized"] = working["parcel_id_normalized"].astype("string")
    county_master = master.copy()

    linked_frames: list[pd.DataFrame] = []
    unresolved = working.copy()

    master_ppin_counts = county_master.groupby(["county_name", "source_ppin"]).size().rename("ppin_master_match_count").reset_index()
    tax_ppin_counts = unresolved.groupby(["county_name", "source_ppin"]).size().rename("ppin_tax_match_count").reset_index()
    unresolved = unresolved.merge(tax_ppin_counts, how="left", on=["county_name", "source_ppin"])
    unresolved = unresolved.merge(master_ppin_counts, how="left", on=["county_name", "source_ppin"])
    unresolved["ppin_master_match_count"] = pd.to_numeric(unresolved["ppin_master_match_count"], errors="coerce").fillna(0).astype("int32")
    unresolved["ppin_tax_match_count"] = pd.to_numeric(unresolved["ppin_tax_match_count"], errors="coerce").fillna(0).astype("int32")

    ppin_linked = unresolved.loc[
        unresolved["source_ppin"].notna()
        & unresolved["ppin_master_match_count"].eq(1)
        & unresolved["ppin_tax_match_count"].eq(1)
    ].copy()
    if not ppin_linked.empty:
        ppin_linked = ppin_linked.merge(
            county_master[["parcel_row_id", "county_name", "source_ppin", "source_parcel_id_normalized", "acreage", "owner_name_raw", "property_address_raw"]],
            how="left",
            on=["county_name", "source_ppin"],
            suffixes=("", "_master"),
        )
        ppin_linked["linkage_method"] = "exact_ppin"
        ppin_linked["match_method"] = "exact_ppin"
        ppin_linked["match_confidence"] = 1.0
        ppin_linked["match_confidence_tier"] = "high"
        linked_frames.append(ppin_linked)

    unresolved = unresolved.loc[~unresolved["tax_record_row_id"].isin(ppin_linked["tax_record_row_id"] if not ppin_linked.empty else [])].copy()
    master_pid_counts = county_master.groupby(["county_name", "source_parcel_id_normalized"]).size().rename("parcel_master_match_count").reset_index()
    tax_pid_counts = unresolved.groupby(["county_name", "parcel_id_normalized"]).size().rename("parcel_tax_match_count").reset_index()
    unresolved = unresolved.merge(tax_pid_counts, how="left", on=["county_name", "parcel_id_normalized"])
    unresolved = unresolved.merge(
        master_pid_counts,
        how="left",
        left_on=["county_name", "parcel_id_normalized"],
        right_on=["county_name", "source_parcel_id_normalized"],
    )
    unresolved["parcel_master_match_count"] = pd.to_numeric(unresolved["parcel_master_match_count"], errors="coerce").fillna(0).astype("int32")
    unresolved["parcel_tax_match_count"] = pd.to_numeric(unresolved["parcel_tax_match_count"], errors="coerce").fillna(0).astype("int32")

    exact_pid_linked = unresolved.loc[
        unresolved["parcel_id_normalized"].notna()
        & unresolved["parcel_master_match_count"].eq(1)
        & unresolved["parcel_tax_match_count"].eq(1)
    ].copy()
    if not exact_pid_linked.empty:
        exact_pid_linked = exact_pid_linked.merge(
            county_master[["parcel_row_id", "county_name", "source_parcel_id_normalized", "acreage", "owner_name_raw", "property_address_raw"]],
            how="left",
            left_on=["county_name", "parcel_id_normalized"],
            right_on=["county_name", "source_parcel_id_normalized"],
            suffixes=("", "_master"),
        )
        exact_pid_linked["linkage_method"] = "exact_normalized_parcel_id"
        exact_pid_linked["match_method"] = "exact_normalized_parcel_id"
        exact_pid_linked["match_confidence"] = 0.9
        exact_pid_linked["match_confidence_tier"] = "high"
        linked_frames.append(exact_pid_linked)

    unresolved = unresolved.loc[
        ~unresolved["tax_record_row_id"].isin(exact_pid_linked["tax_record_row_id"] if not exact_pid_linked.empty else [])
    ].copy()

    heuristic_variants_by_county = heuristic_variants_by_county or {}
    county_variant_rows: list[pd.DataFrame] = []
    for county_name, variants in heuristic_variants_by_county.items():
        county_slug = sanitize_name(county_name)
        county_unresolved = unresolved.loc[unresolved["county_name"].eq(county_slug)].copy()
        county_master_subset = county_master.loc[county_master["county_name"].eq(county_slug)].copy()
        if county_unresolved.empty or county_master_subset.empty:
            continue
        for variant_name in variants:
            county_unresolved["variant_value"] = apply_identifier_variant(county_unresolved["parcel_id_normalized"], variant_name)
            county_master_subset["variant_value"] = apply_identifier_variant(county_master_subset["source_parcel_id_normalized"], variant_name)
            variant_tax_counts = county_unresolved.groupby("variant_value").size().rename("variant_tax_match_count").reset_index()
            variant_master_counts = county_master_subset.groupby("variant_value").size().rename("variant_master_match_count").reset_index()
            variant_matches = county_unresolved.merge(variant_tax_counts, how="left", on="variant_value")
            variant_matches = variant_matches.merge(variant_master_counts, how="left", on="variant_value")
            variant_matches["variant_tax_match_count"] = pd.to_numeric(variant_matches["variant_tax_match_count"], errors="coerce").fillna(0).astype("int32")
            variant_matches["variant_master_match_count"] = pd.to_numeric(variant_matches["variant_master_match_count"], errors="coerce").fillna(0).astype("int32")
            variant_matches = variant_matches.loc[
                variant_matches["variant_value"].notna()
                & variant_matches["variant_tax_match_count"].eq(1)
                & variant_matches["variant_master_match_count"].eq(1)
            ].copy()
            if variant_matches.empty:
                continue
            variant_matches = variant_matches.merge(
                county_master_subset[["parcel_row_id", "variant_value", "acreage", "owner_name_raw", "property_address_raw", "source_parcel_id_normalized"]],
                how="left",
                on="variant_value",
                suffixes=("", "_master"),
            )
            variant_matches["linkage_method"] = f"heuristic_{variant_name}"
            variant_matches["match_method"] = (
                "county_adapter_normalized_match" if variant_name.startswith("county_adapter_") else f"heuristic_{variant_name}"
            )
            variant_matches["match_confidence"] = 0.75 if variant_name == "compact_alnum" else 0.7
            variant_matches["match_confidence_tier"] = (
                "medium" if variant_name.startswith("county_adapter_") else "low"
            )
            county_variant_rows.append(variant_matches)
            county_unresolved = county_unresolved.loc[~county_unresolved["tax_record_row_id"].isin(variant_matches["tax_record_row_id"])].copy()

    if county_variant_rows:
        heuristic_linked = pd.concat(county_variant_rows, ignore_index=True, sort=False)
        linked_frames.append(heuristic_linked)
        unresolved = unresolved.loc[~unresolved["tax_record_row_id"].isin(heuristic_linked["tax_record_row_id"])].copy()

    linked = pd.concat(linked_frames, ignore_index=True, sort=False) if linked_frames else standardized.head(0).copy()
    if not linked.empty:
        if "parcel_row_id_master" in linked.columns:
            linked["parcel_row_id"] = linked["parcel_row_id_master"].astype("string").fillna(linked["parcel_row_id"].astype("string"))
        linked["parcel_row_id"] = linked["parcel_row_id"].astype("string")
        linked["candidate_match_count"] = 1
        linked["value_per_acre"] = pd.to_numeric(
            pd.to_numeric(linked.get("market_total_value"), errors="coerce")
            / pd.to_numeric(linked.get("acreage"), errors="coerce").replace({0: np.nan}),
            errors="coerce",
        )
        linked["tax_balance_to_assessed_value_ratio"] = pd.to_numeric(
            pd.to_numeric(linked.get("tax_balance_due"), errors="coerce")
            / pd.to_numeric(linked.get("assessed_total_value"), errors="coerce").replace({0: np.nan}),
            errors="coerce",
        )

    ambiguous = unresolved.loc[
        unresolved["ppin_master_match_count"].gt(1)
        | unresolved["ppin_tax_match_count"].gt(1)
        | unresolved["parcel_master_match_count"].gt(1)
        | unresolved["parcel_tax_match_count"].gt(1)
    ].copy()
    unmatched = unresolved.loc[~unresolved["tax_record_row_id"].isin(ambiguous["tax_record_row_id"])].copy()

    if not ambiguous.empty:
        ambiguous["candidate_match_count"] = ambiguous[
            ["ppin_master_match_count", "ppin_tax_match_count", "parcel_master_match_count", "parcel_tax_match_count"]
        ].max(axis=1)
        ambiguous["ambiguity_reason"] = np.select(
            [
                ambiguous["ppin_tax_match_count"].gt(1) | ambiguous["parcel_tax_match_count"].gt(1),
                ambiguous["ppin_master_match_count"].gt(1) | ambiguous["parcel_master_match_count"].gt(1),
            ],
            ["duplicate_source_identifier", "duplicate_parcel_master_identifier"],
            default="multiple_candidate_matches",
        )
        ambiguous.loc[
            (ambiguous["ppin_tax_match_count"].gt(1) | ambiguous["parcel_tax_match_count"].gt(1))
            & (ambiguous["ppin_master_match_count"].gt(1) | ambiguous["parcel_master_match_count"].gt(1)),
            "ambiguity_reason",
        ] = "multiple_candidate_matches"

    if not unmatched.empty:
        unmatched["candidate_match_count"] = 0
        unmatched["unmatched_reason"] = "no_candidate_match"
        unmatched.loc[
            unmatched["parcel_id_normalized"].isna() & unmatched["source_ppin"].isna(),
            "unmatched_reason",
        ] = "missing_ppin|missing_parcel_id"
        unmatched.loc[
            unmatched["parcel_id_normalized"].isna() & unmatched["source_ppin"].notna(),
            "unmatched_reason",
        ] = "missing_parcel_id"
        unmatched.loc[
            unmatched["source_ppin"].isna() & unmatched["parcel_id_normalized"].notna(),
            "unmatched_reason",
        ] = "missing_ppin"
        unmatched.loc[
            unmatched["source_ppin"].isna() & unmatched["parcel_id_normalized"].notna() & unmatched["county_name"].notna(),
            "unmatched_reason",
        ] = "county_format_divergence"

    summary_rows: list[dict[str, Any]] = []
    county_counts = standardized.groupby("county_name").size().rename("standardized_rows")
    for county_name, standardized_rows in county_counts.items():
        county_linked = linked.loc[linked["county_name"].eq(county_name)] if not linked.empty else linked
        county_unmatched = unmatched.loc[unmatched["county_name"].eq(county_name)]
        county_ambiguous = ambiguous.loc[ambiguous["county_name"].eq(county_name)]
        summary_rows.append(
            {
                "county_name": county_name,
                "county_fips": COUNTY_FIPS_MAP.get(str(county_name), pd.NA),
                "standardized_rows": int(standardized_rows),
                "linked_rows": int(len(county_linked)),
                "unmatched_rows": int(len(county_unmatched)),
                "ambiguous_rows": int(len(county_ambiguous)),
                "exact_ppin_rows": int(county_linked["linkage_method"].eq("exact_ppin").sum()) if not county_linked.empty else 0,
                "exact_parcel_rows": int(county_linked["linkage_method"].eq("exact_normalized_parcel_id").sum()) if not county_linked.empty else 0,
                "heuristic_rows": int(county_linked["linkage_method"].astype("string").str.startswith("heuristic_").sum()) if not county_linked.empty else 0,
            }
        )
    county_summary = pd.DataFrame(summary_rows).sort_values(["county_fips", "county_name"]).reset_index(drop=True)
    if not county_summary.empty:
        county_summary["linkage_rate"] = np.where(
            county_summary["standardized_rows"].gt(0),
            county_summary["linked_rows"] / county_summary["standardized_rows"] * 100.0,
            0.0,
        )
    return linked, unmatched, ambiguous, county_summary


def update_registry_row(registry_csv: Path, source_id: str, downloaded_at: str, note_text: str) -> None:
    if not registry_csv.exists():
        return
    registry = pd.read_csv(registry_csv)
    registry["last_downloaded_at"] = registry["last_downloaded_at"].astype("string")
    registry["notes"] = registry["notes"].astype("string")
    mask = registry["source_id"].astype("string").eq(source_id)
    if not mask.any():
        return
    registry.loc[mask, "last_downloaded_at"] = downloaded_at
    registry.loc[mask, "notes"] = (
        registry.loc[mask, "notes"].astype("string").fillna("").str.strip() + " " + note_text.strip()
    ).str.strip()
    registry.to_csv(registry_csv, index=False)
