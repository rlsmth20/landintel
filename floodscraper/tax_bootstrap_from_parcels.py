from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from tax_common import (
    BASE_DIR,
    CANONICAL_TAX_COLUMNS,
    PARCELS_RAW_DIR,
    RAW_TAX_DIR,
    TAX_LINKED_DIR,
    TAX_METADATA_DIR,
    TAX_PUBLISHED_DIR,
    TAX_STANDARDIZED_DIR,
    build_record_hash,
    build_row_hash,
    choose_first,
    clean_string,
    coerce_float,
    coerce_year,
    infer_corporate_owner,
    infer_delinquent_flag,
    infer_payment_status,
    normalize_identifier,
    normalize_zip,
    raw_payload_json,
    resolve_path,
    sanitize_name,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap tax data from standardized parcel-source tax attributes.")
    parser.add_argument("--state-code", default="MS", help="Two-letter state code.")
    parser.add_argument("--config", default="floodscraper/state_configs/tax_source_ms.json", help="Tax source config path.")
    parser.add_argument("--master-parquet", default="data/parcels/mississippi_parcels_master.parquet", help="Master parcel parquet path.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county whitelist.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> dict[str, object]:
    return read_json(path)


def load_master_index(path: Path) -> pd.DataFrame:
    columns = [
        "parcel_row_id",
        "state_code",
        "county_name",
        "county_fips",
        "source_parcel_id_normalized",
        "total_acres",
    ]
    master = pd.read_parquet(path, columns=columns)
    master["county_name"] = master["county_name"].astype("string").str.lower()
    master["source_parcel_id_normalized"] = master["source_parcel_id_normalized"].astype("string")
    master = master.rename(columns={"total_acres": "acreage"})
    return master


def registry_rows_for_state(state_code: str, config: dict[str, object], counties: list[str] | None) -> pd.DataFrame:
    state_dir = PARCELS_RAW_DIR / state_code.lower()
    county_filter = {sanitize_name(value) for value in counties} if counties else None
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(state_dir.glob("*/source_metadata.json")):
        payload = read_json(metadata_path)
        county_name = sanitize_name(payload.get("county_name"))
        if county_filter and county_name not in county_filter:
            continue
        standardized_dataset_path = payload.get("standardized_dataset_path")
        if not standardized_dataset_path:
            fallback_standardized = (
                BASE_DIR / "data" / "parcels_standardized" / state_code.lower() / county_name / "standardized_parcels.gpkg"
            )
            if fallback_standardized.exists():
                standardized_dataset_path = fallback_standardized.relative_to(BASE_DIR).as_posix()
        county_fips = str(payload.get("county_fips") or "").zfill(3)
        if standardized_dataset_path and not county_fips.strip("0"):
            county_fips = str(detect_county_fips(resolve_path(str(standardized_dataset_path))) or "").zfill(3)
        if not standardized_dataset_path or not county_fips.strip("0"):
            continue
        source_slug = sanitize_name(str(config["source_category"]))
        rows.append(
            {
                "source_id": f"{state_code.lower()}_{county_fips}_{source_slug}",
                "source_name": str(config["source_category"]),
                "source_category": str(config["source_category"]),
                "source_url": payload.get("source_url"),
                "state_code": state_code,
                "county_fips": county_fips,
                "county_name": county_name,
                "discovery_method": str(config["discovery_method"]),
                "file_type": str(config["file_type"]),
                "update_frequency": "unknown",
                "last_checked_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_downloaded_at": payload.get("downloaded_at"),
                "last_successful_run_at": pd.NA,
                "is_active": True,
                "notes": f"Bootstrapped from {payload.get('source_name')}",
                "source_dataset_path": standardized_dataset_path,
                "parcel_metadata_path": metadata_path.relative_to(BASE_DIR).as_posix(),
                "parcel_source_name": payload.get("source_name"),
                "parcel_source_type": payload.get("source_type"),
                "source_file_version": payload.get("downloaded_at"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["county_fips", "county_name"]).reset_index(drop=True)


def load_county_frame(path: Path) -> pd.DataFrame:
    return gpd.read_file(path, ignore_geometry=True, engine="pyogrio")


def detect_county_fips(standardized_path: Path) -> str | None:
    try:
        sample = gpd.read_file(standardized_path, rows=1, ignore_geometry=True, engine="pyogrio")
    except Exception:
        return None
    if sample.empty or "county_fips" not in sample.columns:
        return None
    values = sample["county_fips"].astype("string").dropna()
    if values.empty:
        return None
    return str(values.iloc[0]).zfill(3)


def standardized_tax_frame(
    county_frame: pd.DataFrame,
    registry_row: pd.Series,
    config: dict[str, object],
    run_id: str,
) -> pd.DataFrame:
    mapping = config["mapping"]
    parcel_id_raw = choose_first(county_frame, mapping["parcel_id_raw"]).astype("string")
    parcel_id_alt = choose_first(county_frame, mapping["parcel_id_alt"]).astype("string")
    parcel_id_normalized = normalize_identifier(parcel_id_raw)
    alt_normalized = normalize_identifier(parcel_id_alt)
    fill_alt = parcel_id_normalized.isna() & alt_normalized.notna()
    parcel_id_normalized.loc[fill_alt] = alt_normalized.loc[fill_alt]
    if str(registry_row["county_name"]).lower() == "adams":
        combine_mask = parcel_id_normalized.notna() & alt_normalized.notna()
        parcel_id_normalized.loc[combine_mask] = (
            parcel_id_normalized.loc[combine_mask] + "|" + alt_normalized.loc[combine_mask]
        )

    owner_name = clean_string(choose_first(county_frame, mapping["owner_name"]))
    owner_address_line1 = clean_string(choose_first(county_frame, mapping["owner_address_line1"]))
    owner_address_line2 = clean_string(choose_first(county_frame, mapping["owner_address_line2"]))
    owner_city = clean_string(choose_first(county_frame, mapping["owner_city"]))
    owner_state = clean_string(choose_first(county_frame, mapping["owner_state"])).str.upper()
    owner_zip = normalize_zip(choose_first(county_frame, mapping["owner_zip"]))
    situs_address = clean_string(choose_first(county_frame, mapping["situs_address"]))
    situs_city = clean_string(choose_first(county_frame, mapping["situs_city"]))
    situs_state = clean_string(choose_first(county_frame, mapping["situs_state"])).str.upper()
    situs_zip = normalize_zip(choose_first(county_frame, mapping["situs_zip"]))
    assessed_land_value = coerce_float(choose_first(county_frame, mapping["assessed_land_value"]))
    assessed_total_value = coerce_float(choose_first(county_frame, mapping["assessed_total_value"]))
    improvement_1 = coerce_float(choose_first(county_frame, mapping["assessed_improvement_value_1"]))
    improvement_2 = coerce_float(choose_first(county_frame, mapping["assessed_improvement_value_2"]))
    assessed_improvement_value = (improvement_1.fillna(0.0) + improvement_2.fillna(0.0)).astype("float64")
    assessed_improvement_value = assessed_improvement_value.mask(improvement_1.isna() & improvement_2.isna())
    taxable_value = coerce_float(choose_first(county_frame, mapping["taxable_value"]))
    tax_year = coerce_year(choose_first(county_frame, mapping["tax_year"]))
    tax_status = clean_string(choose_first(county_frame, mapping["tax_status"]))
    delinquent_flag = infer_delinquent_flag(tax_status)
    payment_status = infer_payment_status(tax_status)

    mailing_key = (
        owner_address_line1.fillna("")
        + "|"
        + owner_city.fillna("")
        + "|"
        + owner_state.fillna("")
        + "|"
        + owner_zip.fillna("")
    )
    situs_key = (
        situs_address.fillna("")
        + "|"
        + situs_city.fillna("")
        + "|"
        + situs_state.fillna("")
        + "|"
        + situs_zip.fillna("")
    )
    mailing_matches_situs = mailing_key.ne("|||") & mailing_key.eq(situs_key)
    absentee_owner_flag = (~mailing_matches_situs) & mailing_key.ne("|||")
    out_of_state_owner_flag = owner_state.notna() & owner_state.ne("MS")
    owner_corporate_flag = infer_corporate_owner(owner_name)

    source_record_id = clean_string(choose_first(county_frame, mapping["source_record_id"]))
    generated_ids = pd.Series(
        (f"{registry_row['county_name']}_{index}" for index in county_frame.index),
        index=county_frame.index,
        dtype="string",
    )
    source_record_id = source_record_id.fillna(generated_ids)

    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash([registry_row["state_code"], registry_row["county_fips"], registry_row["source_name"], record_id, parcel_id])
                    for record_id, parcel_id in zip(source_record_id, parcel_id_normalized)
                ),
                index=county_frame.index,
                dtype="string",
            ),
            "parcel_row_id": pd.Series(pd.NA, index=county_frame.index, dtype="string"),
            "parcel_id_raw": parcel_id_raw,
            "parcel_id_normalized": parcel_id_normalized,
            "state_code": pd.Series(registry_row["state_code"], index=county_frame.index, dtype="string"),
            "county_fips": pd.Series(registry_row["county_fips"], index=county_frame.index, dtype="string"),
            "county_name": pd.Series(registry_row["county_name"], index=county_frame.index, dtype="string"),
            "source_name": pd.Series(registry_row["source_name"], index=county_frame.index, dtype="string"),
            "source_type": pd.Series(registry_row["source_category"], index=county_frame.index, dtype="string"),
            "source_dataset_path": pd.Series(registry_row["source_dataset_path"], index=county_frame.index, dtype="string"),
            "source_record_id": source_record_id,
            "ingestion_run_id": pd.Series(run_id, index=county_frame.index, dtype="string"),
            "source_file_version": pd.Series(registry_row["source_file_version"], index=county_frame.index, dtype="string"),
            "loaded_at": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=county_frame.index, dtype="string"),
            "owner_name": owner_name,
            "owner_name_2": pd.Series(pd.NA, index=county_frame.index, dtype="string"),
            "owner_address_line1": owner_address_line1,
            "owner_address_line2": owner_address_line2,
            "owner_city": owner_city,
            "owner_state": owner_state,
            "owner_zip": owner_zip,
            "situs_address": situs_address,
            "situs_city": situs_city,
            "situs_state": situs_state,
            "situs_zip": situs_zip,
            "assessed_land_value": assessed_land_value,
            "assessed_improvement_value": assessed_improvement_value,
            "assessed_total_value": assessed_total_value,
            "market_land_value": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "market_improvement_value": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "market_total_value": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "taxable_value": taxable_value,
            "exemptions_text": pd.Series(pd.NA, index=county_frame.index, dtype="string"),
            "exemptions_amount": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "tax_year": tax_year,
            "bill_year": tax_year,
            "tax_amount_due": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "tax_amount_paid": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "tax_balance_due": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "tax_status": tax_status,
            "payment_status": payment_status,
            "delinquent_flag": delinquent_flag,
            "delinquent_amount": pd.Series(np.nan, index=county_frame.index, dtype="float64"),
            "delinquent_years": pd.Series(pd.NA, index=county_frame.index, dtype="string"),
            "delinquent_as_of_date": pd.Series(pd.NA, index=county_frame.index, dtype="string"),
            "last_payment_date": pd.Series(pd.NA, index=county_frame.index, dtype="string"),
            "due_date": pd.Series(pd.NA, index=county_frame.index, dtype="string"),
            "absentee_owner_flag": absentee_owner_flag.astype("boolean"),
            "out_of_state_owner_flag": out_of_state_owner_flag.astype("boolean"),
            "owner_corporate_flag": owner_corporate_flag.astype("boolean"),
            "mailing_matches_situs_flag": mailing_matches_situs.astype("boolean"),
            "tax_delinquent_flag_standardized": delinquent_flag.astype("boolean"),
            "raw_payload_json": raw_payload_json(county_frame, mapping["raw_payload_columns"]),
        }
    )
    standardized["record_hash"] = build_record_hash(
        standardized,
        [
            "parcel_id_normalized",
            "owner_name",
            "owner_address_line1",
            "owner_city",
            "owner_state",
            "assessed_land_value",
            "assessed_improvement_value",
            "assessed_total_value",
            "tax_year",
            "tax_status",
            "delinquent_flag",
        ],
    )
    for column in CANONICAL_TAX_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = pd.NA
    return standardized.loc[:, CANONICAL_TAX_COLUMNS].copy()


def county_output_paths(registry_row: pd.Series, run_id: str) -> dict[str, Path]:
    state_code = str(registry_row["state_code"]).lower()
    county_fips = str(registry_row["county_fips"]).zfill(3)
    source_slug = sanitize_name(registry_row["source_name"])
    run_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    raw_dir = RAW_TAX_DIR / state_code / county_fips / source_slug / run_date
    standardized_dir = TAX_STANDARDIZED_DIR / state_code / registry_row["county_name"]
    linked_dir = TAX_LINKED_DIR / state_code / registry_row["county_name"]
    published_dir = TAX_PUBLISHED_DIR / state_code / registry_row["county_name"]
    return {
        "raw_dir": raw_dir,
        "raw_extract": raw_dir / "extracted_raw_tax_records.parquet",
        "manifest": raw_dir / "manifest.json",
        "standardized": standardized_dir / "standardized_tax_records.parquet",
        "linked": linked_dir / "linked_tax_records.parquet",
        "unmatched": linked_dir / "unmatched_tax_records.parquet",
        "ambiguous": linked_dir / "ambiguous_tax_links.parquet",
        "published": published_dir / "published_tax_records.parquet",
        "run_id": published_dir / f"ingestion_run_{run_id}.txt",
    }


def link_tax_records(standardized: pd.DataFrame, master_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if standardized.empty:
        empty = standardized.copy()
        return empty, empty, empty, {
            "raw_tax_rows": 0,
            "standardized_rows": 0,
            "linked_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "linkage_rate": 0.0,
            "null_parcel_id_rate": 0.0,
            "null_owner_name_rate": 0.0,
            "null_assessed_total_value_rate": 0.0,
            "null_tax_year_rate": 0.0,
            "delinquent_flag_count": 0,
            "duplicate_tax_record_rows": 0,
        }

    county_name = standardized["county_name"].astype("string").dropna().iloc[0]
    county_master = master_index.loc[master_index["county_name"].eq(county_name)].copy()
    master_counts = county_master.groupby("source_parcel_id_normalized").size().rename("master_match_count").reset_index()
    tax_counts = standardized.groupby("parcel_id_normalized").size().rename("tax_match_count").reset_index()
    working = standardized.merge(tax_counts, how="left", on="parcel_id_normalized")
    working = working.merge(master_counts, how="left", left_on="parcel_id_normalized", right_on="source_parcel_id_normalized")
    working = working.drop(columns=["source_parcel_id_normalized"], errors="ignore")
    working["master_match_count"] = pd.to_numeric(working["master_match_count"], errors="coerce").fillna(0).astype("int32")
    working["tax_match_count"] = pd.to_numeric(working["tax_match_count"], errors="coerce").fillna(0).astype("int32")

    linked = working.loc[working["master_match_count"].eq(1) & working["tax_match_count"].eq(1)].copy()
    linked = linked.merge(
        county_master[["parcel_row_id", "source_parcel_id_normalized", "acreage"]],
        how="left",
        left_on="parcel_id_normalized",
        right_on="source_parcel_id_normalized",
        suffixes=("_tax", "_master"),
    )
    linked["linkage_method"] = "exact_normalized_parcel_id"
    linked["match_confidence"] = 1.0
    if "parcel_row_id_master" in linked.columns:
        linked["parcel_row_id"] = linked["parcel_row_id_master"].astype("string")
    elif "parcel_row_id" in linked.columns:
        linked["parcel_row_id"] = linked["parcel_row_id"].astype("string")
    else:
        linked["parcel_row_id"] = pd.Series(pd.NA, index=linked.index, dtype="string")
    linked["value_per_acre"] = pd.to_numeric(
        pd.to_numeric(linked["assessed_total_value"], errors="coerce")
        / pd.to_numeric(linked["acreage"], errors="coerce").replace({0: np.nan}),
        errors="coerce",
    )
    linked["tax_balance_to_assessed_value_ratio"] = pd.to_numeric(
        pd.to_numeric(linked["tax_balance_due"], errors="coerce")
        / pd.to_numeric(linked["assessed_total_value"], errors="coerce").replace({0: np.nan}),
        errors="coerce",
    )

    unmatched = working.loc[working["master_match_count"].eq(0)].copy()
    ambiguous = working.loc[working["master_match_count"].gt(1) | working["tax_match_count"].gt(1)].copy()
    summary = {
        "raw_tax_rows": int(len(standardized)),
        "standardized_rows": int(len(standardized)),
        "linked_rows": int(len(linked)),
        "unmatched_rows": int(len(unmatched)),
        "ambiguous_rows": int(len(ambiguous)),
        "linkage_rate": round(float(len(linked) / len(standardized) * 100.0), 4),
        "null_parcel_id_rate": round(float(standardized["parcel_id_normalized"].isna().mean() * 100.0), 4),
        "null_owner_name_rate": round(float(standardized["owner_name"].isna().mean() * 100.0), 4),
        "null_assessed_total_value_rate": round(float(standardized["assessed_total_value"].isna().mean() * 100.0), 4),
        "null_tax_year_rate": round(float(standardized["tax_year"].isna().mean() * 100.0), 4),
        "delinquent_flag_count": int(standardized["tax_delinquent_flag_standardized"].fillna(False).sum()),
        "duplicate_tax_record_rows": int(standardized["tax_record_row_id"].duplicated().sum()),
    }
    return linked, unmatched, ambiguous, summary


def main() -> None:
    args = parse_args()
    state_code = args.state_code.upper()
    run_id = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    config = load_config(resolve_path(args.config))
    registry_df = registry_rows_for_state(state_code, config, args.counties)
    if registry_df.empty:
        raise RuntimeError(f"No parcel-derived tax sources found for {state_code}.")

    master_index = load_master_index(resolve_path(args.master_parquet))
    TAX_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    published_index_rows: list[dict[str, object]] = []

    for registry_row in registry_df.itertuples(index=False):
        registry_series = pd.Series(registry_row._asdict())
        source_path = resolve_path(str(registry_series["source_dataset_path"]))
        county_frame = load_county_frame(source_path)
        standardized = standardized_tax_frame(county_frame, registry_series, config, run_id)
        linked, unmatched, ambiguous, summary = link_tax_records(standardized, master_index)
        paths = county_output_paths(registry_series, run_id)
        paths["raw_dir"].mkdir(parents=True, exist_ok=True)
        paths["standardized"].parent.mkdir(parents=True, exist_ok=True)
        paths["linked"].parent.mkdir(parents=True, exist_ok=True)
        paths["published"].parent.mkdir(parents=True, exist_ok=True)

        county_frame.to_parquet(paths["raw_extract"], index=False)
        standardized.to_parquet(paths["standardized"], index=False)
        linked.to_parquet(paths["linked"], index=False)
        unmatched.to_parquet(paths["unmatched"], index=False)
        ambiguous.to_parquet(paths["ambiguous"], index=False)
        linked.to_parquet(paths["published"], index=False)
        paths["run_id"].write_text(run_id, encoding="utf-8")

        write_json(
            paths["manifest"],
            {
                "ingestion_run_id": run_id,
                "state_code": registry_series["state_code"],
                "county_name": registry_series["county_name"],
                "county_fips": registry_series["county_fips"],
                "source_name": registry_series["source_name"],
                "source_category": registry_series["source_category"],
                "source_dataset_path": registry_series["source_dataset_path"],
                "parcel_metadata_path": registry_series["parcel_metadata_path"],
                "raw_extract_path": paths["raw_extract"].relative_to(BASE_DIR).as_posix(),
                "standardized_path": paths["standardized"].relative_to(BASE_DIR).as_posix(),
                "linked_path": paths["linked"].relative_to(BASE_DIR).as_posix(),
                "unmatched_path": paths["unmatched"].relative_to(BASE_DIR).as_posix(),
                "ambiguous_path": paths["ambiguous"].relative_to(BASE_DIR).as_posix(),
                "row_count": int(len(county_frame)),
                "standardized_row_count": int(len(standardized)),
                "linked_row_count": int(len(linked)),
            },
        )

        registry_df.loc[registry_df["source_id"].eq(registry_series["source_id"]), "last_successful_run_at"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        summary_rows.append(
            {
                "state_code": registry_series["state_code"],
                "county_fips": registry_series["county_fips"],
                "county_name": registry_series["county_name"],
                "source_id": registry_series["source_id"],
                "source_name": registry_series["source_name"],
                **summary,
            }
        )
        published_index_rows.append(
            {
                "state_code": registry_series["state_code"],
                "county_fips": registry_series["county_fips"],
                "county_name": registry_series["county_name"],
                "published_tax_path": paths["published"].relative_to(BASE_DIR).as_posix(),
                "linked_rows": int(len(linked)),
                "unmatched_rows": int(len(unmatched)),
                "ambiguous_rows": int(len(ambiguous)),
            }
        )
        print(
            f"[tax] {registry_series['county_name']}: raw={len(county_frame):,} "
            f"standardized={len(standardized):,} linked={len(linked):,} "
            f"unmatched={len(unmatched):,} ambiguous={len(ambiguous):,}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["county_fips", "county_name"]).reset_index(drop=True)
    statewide_row = {
        "state_code": state_code,
        "county_fips": "ALL",
        "county_name": "statewide",
        "source_id": f"{state_code.lower()}_statewide_summary",
        "source_name": str(config["source_category"]),
        "raw_tax_rows": int(summary_df["raw_tax_rows"].sum()),
        "standardized_rows": int(summary_df["standardized_rows"].sum()),
        "linked_rows": int(summary_df["linked_rows"].sum()),
        "unmatched_rows": int(summary_df["unmatched_rows"].sum()),
        "ambiguous_rows": int(summary_df["ambiguous_rows"].sum()),
        "linkage_rate": round(float(summary_df["linked_rows"].sum() / max(summary_df["standardized_rows"].sum(), 1) * 100.0), 4),
        "null_parcel_id_rate": round(float(summary_df["null_parcel_id_rate"].mean()), 4),
        "null_owner_name_rate": round(float(summary_df["null_owner_name_rate"].mean()), 4),
        "null_assessed_total_value_rate": round(float(summary_df["null_assessed_total_value_rate"].mean()), 4),
        "null_tax_year_rate": round(float(summary_df["null_tax_year_rate"].mean()), 4),
        "delinquent_flag_count": int(summary_df["delinquent_flag_count"].sum()),
        "duplicate_tax_record_rows": int(summary_df["duplicate_tax_record_rows"].sum()),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([statewide_row])], ignore_index=True)
    published_index_df = pd.DataFrame(published_index_rows).sort_values(["county_fips", "county_name"]).reset_index(drop=True)

    registry_path = TAX_METADATA_DIR / f"tax_source_registry_{state_code.lower()}.csv"
    summary_path = TAX_METADATA_DIR / f"tax_linkage_summary_{state_code.lower()}.csv"
    qa_path = TAX_METADATA_DIR / f"tax_standardization_qa_{state_code.lower()}.csv"
    published_index_path = TAX_PUBLISHED_DIR / state_code.lower() / "published_tax_county_index.csv"
    registry_df.to_csv(registry_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(qa_path, index=False)
    published_index_path.parent.mkdir(parents=True, exist_ok=True)
    published_index_df.to_csv(published_index_path, index=False)

    print(f"Tax registry: {registry_path.relative_to(BASE_DIR).as_posix()}")
    print(f"Tax linkage summary: {summary_path.relative_to(BASE_DIR).as_posix()}")
    print(f"Published county index: {published_index_path.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    main()
