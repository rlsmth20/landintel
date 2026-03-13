from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_LINKED_DIR = BASE_DIR / "data" / "tax_linked" / "ms"
OUTPUT_DIR = BASE_DIR / "data" / "tax_published" / "ms"

OUTPUT_COLUMNS = [
    "parcel_row_id",
    "county_fips",
    "county_name",
    "parcel_id_raw",
    "parcel_id_normalized",
    "owner_name",
    "situs_address",
    "tax_year",
    "bill_year",
    "best_source_name",
    "best_source_type",
    "best_source_dataset_path",
    "best_source_file_path",
    "best_source_record_id",
    "best_linkage_method",
    "best_match_confidence",
    "best_match_confidence_tier",
    "best_delinquent_amount",
    "max_reported_delinquent_amount",
    "reported_delinquent_years",
    "source_count",
    "source_names",
    "source_types",
    "record_count",
    "county_hosted_source_count",
    "statewide_source_count",
    "has_forfeited_source",
    "has_county_hosted_source",
    "has_statewide_source",
    "latest_loaded_at",
]

READ_COLUMNS = [
    "tax_record_row_id",
    "parcel_row_id",
    "parcel_row_id_master",
    "parcel_row_id_tax",
    "county_fips",
    "county_name",
    "parcel_id_raw",
    "parcel_id_normalized",
    "owner_name",
    "situs_address",
    "source_name",
    "source_type",
    "source_dataset_path",
    "source_record_id",
    "loaded_at",
    "tax_year",
    "bill_year",
    "delinquent_amount",
    "tax_balance_due",
    "tax_status",
    "delinquent_flag",
    "forfeited_flag",
    "tax_delinquent_flag_standardized",
    "delinquent_years",
    "match_method",
    "linkage_method",
    "match_confidence",
    "match_confidence_tier",
]

COUNTY_HOSTED_SOURCE_TYPES = {
    "free_direct_download",
    "direct_download_page",
}
STATEWIDE_SOURCE_TYPES = {
    "statewide_public_inventory",
    "statewide_public_map",
}
SOURCE_TYPE_PRIORITY = {
    "free_direct_download": 400,
    "direct_download_page": 390,
    "statewide_public_inventory": 250,
    "statewide_public_map": 240,
    "parcel_service_tax_attributes": 100,
}
MATCH_METHOD_PRIORITY = {
    "exact_ppin": 50,
    "exact_normalized_parcel_id": 45,
    "compact_alnum": 20,
}


def parquet_columns(path: Path) -> list[str]:
    if pq is None:
        frame = pd.read_parquet(path)
        return list(frame.columns)
    return list(pq.ParquetFile(path).schema.names)


def delinquent_mask(frame: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=frame.index, dtype="boolean")
    if "tax_delinquent_flag_standardized" in frame.columns:
        mask = mask | frame["tax_delinquent_flag_standardized"].fillna(False).astype("boolean")
    if "delinquent_flag" in frame.columns:
        mask = mask | frame["delinquent_flag"].fillna(False).astype("boolean")
    if "forfeited_flag" in frame.columns:
        mask = mask | frame["forfeited_flag"].fillna(False).astype("boolean")
    if "tax_status" in frame.columns:
        status = frame["tax_status"].astype("string").str.lower().fillna("")
        mask = mask | status.str.contains("delinquent|forfeit|tax_sale|sold_not_redeemed", regex=True)
    return mask.fillna(False)


def safe_read_delinquent_rows(path: Path) -> pd.DataFrame:
    available = [column for column in READ_COLUMNS if column in parquet_columns(path)]
    if not available:
        return pd.DataFrame()
    frame: pd.DataFrame
    if "tax_delinquent_flag_standardized" in available:
        try:
            frame = pd.read_parquet(
                path,
                columns=available,
                filters=[("tax_delinquent_flag_standardized", "==", True)],
            )
        except Exception:
            frame = pd.read_parquet(path, columns=available)
    else:
        frame = pd.read_parquet(path, columns=available)
    if frame.empty:
        return frame
    frame = frame.loc[delinquent_mask(frame)].copy()
    if frame.empty:
        return frame
    frame["source_file_path"] = path.relative_to(BASE_DIR).as_posix()
    frame["parcel_row_id"] = first_non_null(frame, ["parcel_row_id", "parcel_row_id_master", "parcel_row_id_tax"])
    frame = frame.loc[frame["parcel_row_id"].astype("string").notna()].copy()
    if frame.empty:
        return frame
    frame["delinquent_amount_value"] = coerce_amount(frame)
    frame["loaded_at"] = pd.to_datetime(frame.get("loaded_at"), errors="coerce", utc=True)
    frame["record_key"] = (
        frame.get("source_name", pd.Series(pd.NA, index=frame.index, dtype="string")).astype("string").fillna("")
        + "|"
        + frame.get("source_record_id", pd.Series(pd.NA, index=frame.index, dtype="string")).astype("string").fillna("")
        + "|"
        + frame.get("tax_record_row_id", pd.Series(pd.NA, index=frame.index, dtype="string")).astype("string").fillna("")
    )
    return frame.drop_duplicates(subset=["record_key", "source_file_path"], keep="first").reset_index(drop=True)


def first_non_null(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = frame[column].astype("string")
        result = result.where(result.notna(), candidate)
    return result


def coerce_amount(frame: pd.DataFrame) -> pd.Series:
    amount = pd.Series(float("nan"), index=frame.index, dtype="float64")
    for column in ["delinquent_amount", "tax_balance_due"]:
        if column not in frame.columns:
            continue
        candidate = pd.to_numeric(frame[column], errors="coerce")
        amount = amount.where(amount.notna(), candidate)
    return amount


def load_delinquent_records() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(TAX_LINKED_DIR.rglob("*linked_tax_records.parquet")):
        frame = safe_read_delinquent_rows(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=READ_COLUMNS + ["source_file_path", "delinquent_amount_value", "record_key"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["source_name", "source_record_id", "tax_record_row_id", "parcel_row_id"], keep="first").reset_index(drop=True)


def choose_best_record(records: pd.DataFrame) -> pd.DataFrame:
    ranked = records.copy()
    ranked["source_priority"] = ranked.get("source_type", pd.Series("", index=ranked.index, dtype="string")).map(SOURCE_TYPE_PRIORITY).fillna(150)
    ranked["match_priority"] = (
        first_non_null(ranked, ["match_method", "linkage_method"]).map(MATCH_METHOD_PRIORITY).fillna(0)
    )
    ranked["amount_priority"] = ranked["delinquent_amount_value"].notna().astype(int)
    ranked["loaded_priority"] = (
        pd.to_datetime(ranked["loaded_at"], errors="coerce", utc=True)
        .fillna(pd.Timestamp("1970-01-01T00:00:00Z"))
        .astype("int64")
    )
    ranked = ranked.sort_values(
        ["parcel_row_id", "source_priority", "match_priority", "amount_priority", "loaded_priority"],
        ascending=[True, False, False, False, False],
    )
    return ranked.drop_duplicates(subset=["parcel_row_id"], keep="first").reset_index(drop=True)


def aggregate_parcel_records(records: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby("parcel_row_id", dropna=False)
    summary = grouped.agg(
        county_fips=("county_fips", "first"),
        county_name=("county_name", "first"),
        max_reported_delinquent_amount=("delinquent_amount_value", "max"),
        record_count=("parcel_row_id", "size"),
        latest_loaded_at=("loaded_at", "max"),
    ).reset_index()
    summary["source_count"] = grouped["source_name"].nunique().to_numpy()
    summary["source_names"] = grouped["source_name"].apply(join_unique_strings).to_numpy()
    summary["source_types"] = grouped["source_type"].apply(join_unique_strings).to_numpy()
    summary["reported_delinquent_years"] = grouped["delinquent_years"].apply(join_unique_strings).to_numpy()
    summary["county_hosted_source_count"] = grouped["source_type"].apply(lambda s: int(s.astype("string").isin(COUNTY_HOSTED_SOURCE_TYPES).sum())).to_numpy()
    summary["statewide_source_count"] = grouped["source_type"].apply(lambda s: int(s.astype("string").isin(STATEWIDE_SOURCE_TYPES).sum())).to_numpy()
    summary["has_forfeited_source"] = grouped["forfeited_flag"].apply(series_any_true).to_numpy()
    summary["has_county_hosted_source"] = summary["county_hosted_source_count"].gt(0)
    summary["has_statewide_source"] = summary["statewide_source_count"].gt(0)

    parcel_frame = best.merge(summary, on=["parcel_row_id", "county_fips", "county_name"], how="left")
    parcel_frame = parcel_frame.assign(
        best_source_name=parcel_frame["source_name"].astype("string"),
        best_source_type=parcel_frame["source_type"].astype("string"),
        best_source_dataset_path=parcel_frame["source_dataset_path"].astype("string"),
        best_source_file_path=parcel_frame["source_file_path"].astype("string"),
        best_source_record_id=parcel_frame["source_record_id"].astype("string"),
        best_linkage_method=first_non_null(parcel_frame, ["match_method", "linkage_method"]),
        best_match_confidence=pd.to_numeric(parcel_frame.get("match_confidence"), errors="coerce"),
        best_match_confidence_tier=parcel_frame.get("match_confidence_tier", pd.Series(pd.NA, index=parcel_frame.index, dtype="string")).astype("string"),
        best_delinquent_amount=parcel_frame["delinquent_amount_value"].astype("float64"),
    )
    parcel_frame["latest_loaded_at"] = pd.to_datetime(parcel_frame["latest_loaded_at"], errors="coerce", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for column in OUTPUT_COLUMNS:
        if column not in parcel_frame.columns:
            parcel_frame[column] = pd.NA
    parcel_frame = parcel_frame.loc[:, OUTPUT_COLUMNS].sort_values(["county_name", "parcel_row_id"]).reset_index(drop=True)
    return parcel_frame


def build_county_summary(parcel_frame: pd.DataFrame) -> pd.DataFrame:
    grouped = parcel_frame.groupby(["county_fips", "county_name"], dropna=False)
    county = grouped.agg(
        delinquent_parcel_count=("parcel_row_id", "size"),
        parcels_with_reported_amount=("best_delinquent_amount", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        total_best_delinquent_amount=("best_delinquent_amount", lambda s: round(float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum()), 2)),
        max_best_delinquent_amount=("best_delinquent_amount", lambda s: round(float(pd.to_numeric(s, errors="coerce").max()), 2) if pd.to_numeric(s, errors="coerce").notna().any() else 0.0),
        multi_source_parcel_count=("source_count", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).gt(1).sum())),
        forfeited_parcel_count=("has_forfeited_source", lambda s: int(s.fillna(False).astype(bool).sum())),
        latest_loaded_at=("latest_loaded_at", "max"),
    ).reset_index()
    county["source_names"] = grouped["source_names"].apply(join_nested_unique_strings).to_numpy()
    county["county_hosted_best_source_parcels"] = grouped["has_county_hosted_source"].apply(lambda s: int(s.fillna(False).astype(bool).sum())).to_numpy()
    county["statewide_best_source_parcels"] = grouped["has_statewide_source"].apply(lambda s: int(s.fillna(False).astype(bool).sum())).to_numpy()
    county["average_best_delinquent_amount"] = county.apply(
        lambda row: round(row["total_best_delinquent_amount"] / row["parcels_with_reported_amount"], 2)
        if row["parcels_with_reported_amount"]
        else 0.0,
        axis=1,
    )
    return county.sort_values(["delinquent_parcel_count", "county_name"], ascending=[False, True]).reset_index(drop=True)


def build_statewide_summary(records: pd.DataFrame, parcel_frame: pd.DataFrame) -> pd.DataFrame:
    best_linkage = parcel_frame["best_linkage_method"].astype("string")
    summary_rows = [
        {"metric": "delinquent_source_file_count", "value": int(records["source_file_path"].astype("string").nunique())},
        {"metric": "delinquent_source_name_count", "value": int(records["source_name"].astype("string").nunique())},
        {"metric": "delinquent_record_count", "value": int(len(records))},
        {"metric": "unique_delinquent_parcel_count", "value": int(len(parcel_frame))},
        {"metric": "counties_with_delinquent_parcels", "value": int(parcel_frame["county_name"].astype("string").nunique())},
        {"metric": "parcels_with_reported_amount", "value": int(pd.to_numeric(parcel_frame["best_delinquent_amount"], errors="coerce").notna().sum())},
        {
            "metric": "total_best_delinquent_amount",
            "value": round(float(pd.to_numeric(parcel_frame["best_delinquent_amount"], errors="coerce").fillna(0.0).sum()), 2),
        },
        {"metric": "multi_source_parcel_count", "value": int(pd.to_numeric(parcel_frame["source_count"], errors="coerce").fillna(0).gt(1).sum())},
        {"metric": "forfeited_parcel_count", "value": int(parcel_frame["has_forfeited_source"].fillna(False).astype(bool).sum())},
        {"metric": "exact_ppin_best_source_parcels", "value": int(best_linkage.eq("exact_ppin").sum())},
        {"metric": "exact_normalized_parcel_id_best_source_parcels", "value": int(best_linkage.eq("exact_normalized_parcel_id").sum())},
        {"metric": "heuristic_best_source_parcels", "value": int(best_linkage.isin(["compact_alnum"]).sum())},
        {"metric": "county_hosted_best_source_parcels", "value": int(parcel_frame["has_county_hosted_source"].fillna(False).astype(bool).sum())},
        {"metric": "statewide_best_source_parcels", "value": int(parcel_frame["has_statewide_source"].fillna(False).astype(bool).sum())},
    ]
    return pd.DataFrame(summary_rows)


def join_unique_strings(values: pd.Series) -> str:
    items = sorted({value for value in values.astype("string").fillna("") if value})
    return "|".join(items)


def join_nested_unique_strings(values: pd.Series) -> str:
    items: set[str] = set()
    for value in values.astype("string").fillna(""):
        if not value:
            continue
        items.update(piece for piece in value.split("|") if piece)
    return "|".join(sorted(items))


def series_any_true(values: pd.Series) -> bool:
    return bool(values.fillna(False).astype(bool).any())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_delinquent_records()
    if records.empty:
        empty_statewide = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty_county = pd.DataFrame(
            columns=[
                "county_fips",
                "county_name",
                "delinquent_parcel_count",
                "parcels_with_reported_amount",
                "total_best_delinquent_amount",
                "max_best_delinquent_amount",
                "multi_source_parcel_count",
                "forfeited_parcel_count",
                "latest_loaded_at",
                "source_names",
                "county_hosted_best_source_parcels",
                "statewide_best_source_parcels",
                "average_best_delinquent_amount",
            ]
        )
        empty_summary = pd.DataFrame(columns=["metric", "value"])
        empty_statewide.to_parquet(OUTPUT_DIR / "delinquent_parcels_statewide.parquet", index=False)
        empty_county.to_csv(OUTPUT_DIR / "delinquent_parcels_by_county.csv", index=False)
        empty_summary.to_csv(OUTPUT_DIR / "delinquent_parcels_summary.csv", index=False)
        return

    best = choose_best_record(records)
    parcel_frame = aggregate_parcel_records(records, best)
    county_frame = build_county_summary(parcel_frame)
    summary_frame = build_statewide_summary(records, parcel_frame)

    statewide_path = OUTPUT_DIR / "delinquent_parcels_statewide.parquet"
    county_path = OUTPUT_DIR / "delinquent_parcels_by_county.csv"
    summary_path = OUTPUT_DIR / "delinquent_parcels_summary.csv"

    parcel_frame.to_parquet(statewide_path, index=False)
    county_frame.to_csv(county_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    print(f"Statewide: {statewide_path.relative_to(BASE_DIR)}")
    print(f"By county: {county_path.relative_to(BASE_DIR)}")
    print(f"Summary: {summary_path.relative_to(BASE_DIR)}")
    print(f"Delinquent source records: {len(records):,}")
    print(f"Unique delinquent parcels: {len(parcel_frame):,}")


if __name__ == "__main__":
    main()
