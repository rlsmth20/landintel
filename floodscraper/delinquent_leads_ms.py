from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PARCELS_DIR = BASE_DIR / "data" / "parcels"
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"

DELINQUENT_PARCELS_PATH = TAX_PUBLISHED_DIR / "delinquent_parcels_statewide.parquet"
PARCEL_MASTER_PATH = PARCELS_DIR / "mississippi_parcels_master.parquet"
OWNER_LEADS_PATH = PARCELS_DIR / "mississippi_parcels_owner_leads.parquet"
TAX_DISTRESS_PATH = PARCELS_DIR / "mississippi_parcels_tax_distress.parquet"

OUTPUT_PARQUET = TAX_PUBLISHED_DIR / "delinquent_leads_statewide.parquet"
OUTPUT_COUNTY_CSV = TAX_PUBLISHED_DIR / "delinquent_leads_by_county.csv"
OUTPUT_SUMMARY_CSV = TAX_PUBLISHED_DIR / "delinquent_leads_summary.csv"

MASTER_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "state_code",
    "county_fips",
    "county_name",
    "geometry",
    "total_acres",
    "tax_acres",
    "gis_acres",
    "land_value",
    "improvement_value_1",
    "improvement_value_2",
    "total_value",
    "land_use_raw",
    "zoning",
    "owner_name",
    "owner_name_raw",
    "mailing_address_line1_raw",
    "mailing_address_line2_raw",
    "mailing_city_raw",
    "mailing_state_raw",
    "mailing_zip_raw",
    "property_address_raw",
    "property_city_raw",
    "property_state_raw",
    "flood_risk_score",
    "wetland_flag",
    "road_distance_ft",
    "buildability_score",
    "environment_score",
    "investment_score",
    "broadband_available",
    "electric_provider_name",
]

OWNER_LEADS_COLUMNS = [
    "parcel_row_id",
    "apn",
    "owner_name_raw",
    "owner_name_normalized",
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
    "parcel_acres",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "owner_occupied_proxy_flag",
    "corporate_owner_flag",
    "owner_type",
    "owner_group_id",
    "owner_parcel_count",
    "owner_total_acres",
    "mailer_target_score",
    "parcel_constraint_summary",
    "tax_distress_score",
    "distressed_owner_flag",
]

TAX_DISTRESS_COLUMNS = [
    "parcel_row_id",
    "owner_group_id",
    "owner_parcel_count",
    "owner_total_acres",
    "county_tax_source_configured_flag",
    "county_tax_source_loaded_flag",
    "county_tax_source_type",
    "tax_data_available_flag",
    "tax_delinquent_flag",
    "delinquent_year_count",
    "delinquent_tax_amount_total",
    "tax_sale_flag",
    "latest_delinquent_year",
    "most_severe_tax_status",
    "tax_record_count",
    "tax_source_name",
    "tax_distress_score",
    "distressed_owner_flag",
]

LEAD_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "state_code",
    "county_fips",
    "county_name",
    "geometry",
    "acreage",
    "acreage_bucket",
    "owner_name",
    "owner_name_normalized",
    "owner_address",
    "mailing_address_line1",
    "mailing_address_line2",
    "mailing_city",
    "mailing_state",
    "mailing_zip",
    "mailing_zip5",
    "property_address",
    "property_city",
    "property_state",
    "source_type",
    "best_source_type",
    "best_source_name",
    "source_priority",
    "source_count",
    "county_hosted_source_count",
    "statewide_source_count",
    "delinquent_amount",
    "delinquent_amount_bucket",
    "delinquent_flag",
    "forfeited_flag",
    "has_reported_delinquent_amount_flag",
    "best_match_method",
    "best_match_confidence",
    "best_match_confidence_tier",
    "high_confidence_link_flag",
    "source_confidence_tier",
    "county_source_coverage_tier",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "mailing_matches_situs_flag",
    "assessed_land_value",
    "assessed_improvement_value",
    "assessed_total_value",
    "land_use",
    "flood_risk_score",
    "wetland_flag",
    "road_distance_ft",
    "buildability_score",
    "environment_score",
    "investment_score",
    "broadband_available",
    "electric_provider_name",
    "owner_group_id",
    "owner_parcel_count",
    "owner_total_acres",
    "owner_type",
    "corporate_owner_flag",
    "mailer_target_score",
    "parcel_constraint_summary",
    "tax_distress_score",
    "distressed_owner_flag",
    "delinquent_year_count",
    "delinquent_tax_amount_total",
    "tax_record_count",
    "latest_delinquent_year",
    "most_severe_tax_status",
    "tax_year",
    "bill_year",
    "reported_delinquent_years",
    "best_source_dataset_path",
    "best_source_file_path",
    "best_source_record_id",
    "latest_loaded_at",
]

SOURCE_PRIORITY = {
    "free_direct_download": 4,
    "direct_download_page": 4,
    "statewide_public_inventory": 2,
    "statewide_public_map": 2,
    "parcel_service_tax_attributes": 1,
}


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def read_filtered_parquet(path: Path, columns: list[str], parcel_ids: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch in chunked(parcel_ids, 1000):
        try:
            frame = pd.read_parquet(path, columns=columns, filters=[("parcel_row_id", "in", batch)])
        except Exception:
            full = pd.read_parquet(path, columns=columns)
            return full.loc[full["parcel_row_id"].astype("string").isin(parcel_ids)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["parcel_row_id"], keep="first").reset_index(drop=True)


def read_filtered_geoparquet(path: Path, columns: list[str], parcel_ids: list[str]) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for batch in chunked(parcel_ids, 1000):
        try:
            frame = gpd.read_parquet(path, columns=columns, filters=[("parcel_row_id", "in", batch)])
        except Exception:
            full = gpd.read_parquet(path, columns=columns)
            return full.loc[full["parcel_row_id"].astype("string").isin(parcel_ids)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:4326")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["parcel_row_id"], keep="first").reset_index(drop=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=getattr(frames[0], "crs", "EPSG:4326"))


def normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def join_line1_line2(line1: pd.Series, line2: pd.Series) -> pd.Series:
    line1 = normalize_string(line1)
    line2 = normalize_string(line2)
    combined = line1.fillna("")
    has_line2 = line2.notna()
    combined = combined.where(~has_line2, (combined + " " + line2.fillna("")).str.strip())
    combined = combined.replace({"": pd.NA})
    return combined.astype("string")


def compute_mailing_matches_situs(frame: pd.DataFrame) -> pd.Series:
    mailing = normalize_string(frame["owner_address"])
    situs = normalize_string(frame["property_address"])
    comparable = mailing.notna() & situs.notna()
    result = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    result.loc[comparable] = mailing.loc[comparable].str.upper().eq(situs.loc[comparable].str.upper())
    return result


def acreage_bucket(series: pd.Series) -> pd.Series:
    acres = pd.to_numeric(series, errors="coerce")
    bucket = pd.Series(pd.NA, index=series.index, dtype="string")
    bucket.loc[acres.lt(1)] = "<1"
    bucket.loc[acres.ge(1) & acres.lt(5)] = "1-4.99"
    bucket.loc[acres.ge(5) & acres.lt(20)] = "5-19.99"
    bucket.loc[acres.ge(20) & acres.lt(100)] = "20-99.99"
    bucket.loc[acres.ge(100)] = "100+"
    return bucket


def delinquent_amount_bucket(series: pd.Series) -> pd.Series:
    amount = pd.to_numeric(series, errors="coerce")
    bucket = pd.Series(pd.NA, index=series.index, dtype="string")
    bucket.loc[amount.lt(1000)] = "<1k"
    bucket.loc[amount.ge(1000) & amount.lt(5000)] = "1k-4.99k"
    bucket.loc[amount.ge(5000) & amount.lt(25000)] = "5k-24.99k"
    bucket.loc[amount.ge(25000)] = "25k+"
    return bucket


def derive_source_confidence(frame: pd.DataFrame) -> pd.Series:
    source_type = normalize_string(frame["best_source_type"])
    match_tier = normalize_string(frame["best_match_confidence_tier"]).fillna("unknown")
    result = pd.Series("low", index=frame.index, dtype="string")
    county_hosted = source_type.isin(["free_direct_download", "direct_download_page"])
    statewide = source_type.isin(["statewide_public_inventory", "statewide_public_map"])
    result.loc[statewide & match_tier.isin(["high", "medium"])] = "medium"
    result.loc[county_hosted & match_tier.eq("high")] = "high"
    result.loc[county_hosted & match_tier.eq("medium")] = "medium"
    return result


def derive_county_coverage_tier(frame: pd.DataFrame) -> pd.Series:
    county_ratio = pd.to_numeric(frame["county_hosted_source_count"], errors="coerce").fillna(0) / pd.to_numeric(frame["source_count"], errors="coerce").replace(0, pd.NA)
    result = pd.Series("low", index=frame.index, dtype="string")
    result.loc[county_ratio.ge(0.75).fillna(False)] = "high"
    result.loc[county_ratio.ge(0.25).fillna(False) & county_ratio.lt(0.75).fillna(False)] = "medium"
    return result


def build_county_output(leads: gpd.GeoDataFrame) -> pd.DataFrame:
    grouped = leads.groupby(["county_fips", "county_name"], dropna=False)
    county = grouped.agg(
        delinquent_lead_count=("parcel_row_id", "size"),
        total_delinquent_amount=("delinquent_amount", lambda s: round(float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum()), 2)),
        average_delinquent_amount=("delinquent_amount", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 2) if pd.to_numeric(s, errors="coerce").notna().any() else 0.0),
        acreage_available_count=("acreage", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        owner_name_available_count=("owner_name", lambda s: int(normalize_string(s).notna().sum())),
        reported_amount_count=("has_reported_delinquent_amount_flag", lambda s: int(s.fillna(False).astype(bool).sum())),
        absentee_owner_count=("absentee_owner_flag", lambda s: int(s.fillna(False).astype(bool).sum())),
        out_of_state_owner_count=("out_of_state_owner_flag", lambda s: int(s.fillna(False).astype(bool).sum())),
        corporate_owner_count=("corporate_owner_flag", lambda s: int(s.fillna(False).astype(bool).sum())),
        distressed_owner_count=("distressed_owner_flag", lambda s: int(s.fillna(False).astype(bool).sum())),
        high_confidence_link_count=("high_confidence_link_flag", lambda s: int(s.fillna(False).astype(bool).sum())),
        average_investment_score=("investment_score", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 2) if pd.to_numeric(s, errors="coerce").notna().any() else 0.0),
        latest_loaded_at=("latest_loaded_at", "max"),
    ).reset_index()
    county["best_source_types"] = grouped["best_source_type"].apply(lambda s: "|".join(sorted({v for v in normalize_string(s).dropna()}))).to_numpy()
    county["county_source_coverage_tier"] = grouped["county_source_coverage_tier"].agg(lambda s: s.value_counts(dropna=False).index[0] if len(s) else pd.NA).to_numpy()
    return county.sort_values(["delinquent_lead_count", "total_delinquent_amount"], ascending=[False, False]).reset_index(drop=True)


def build_summary(leads: gpd.GeoDataFrame, delinquent_parcels: pd.DataFrame, county_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.extend(
        [
            {"section": "statewide", "metric": "delinquent_leads_total", "key": pd.NA, "value": int(len(leads))},
            {"section": "statewide", "metric": "parcel_master_join_rate_pct", "key": pd.NA, "value": round(float(len(leads) / len(delinquent_parcels) * 100.0), 4) if len(delinquent_parcels) else 0.0},
            {"section": "statewide", "metric": "acreage_available_count", "key": pd.NA, "value": int(pd.to_numeric(leads["acreage"], errors="coerce").notna().sum())},
            {"section": "statewide", "metric": "owner_name_available_count", "key": pd.NA, "value": int(normalize_string(leads["owner_name"]).notna().sum())},
            {"section": "statewide", "metric": "delinquent_amount_available_count", "key": pd.NA, "value": int(pd.to_numeric(leads["delinquent_amount"], errors="coerce").notna().sum())},
            {"section": "statewide", "metric": "absentee_owner_count", "key": pd.NA, "value": int(leads["absentee_owner_flag"].fillna(False).astype(bool).sum())},
            {"section": "statewide", "metric": "out_of_state_owner_count", "key": pd.NA, "value": int(leads["out_of_state_owner_flag"].fillna(False).astype(bool).sum())},
            {"section": "statewide", "metric": "corporate_owner_count", "key": pd.NA, "value": int(leads["corporate_owner_flag"].fillna(False).astype(bool).sum())},
            {"section": "statewide", "metric": "distressed_owner_count", "key": pd.NA, "value": int(leads["distressed_owner_flag"].fillna(False).astype(bool).sum())},
            {"section": "statewide", "metric": "multi_parcel_owner_count", "key": pd.NA, "value": int(pd.to_numeric(leads["owner_parcel_count"], errors="coerce").fillna(0).gt(1).sum())},
            {"section": "statewide", "metric": "total_delinquent_amount", "key": pd.NA, "value": round(float(pd.to_numeric(leads["delinquent_amount"], errors="coerce").fillna(0.0).sum()), 2)},
        ]
    )
    for field in ["acreage", "owner_name", "delinquent_amount"]:
        if field == "owner_name":
            null_rate = 100.0 - float(normalize_string(leads[field]).notna().mean() * 100.0)
        else:
            null_rate = float(pd.to_numeric(leads[field], errors="coerce").isna().mean() * 100.0)
        rows.append({"section": "null_rate", "metric": field, "key": pd.NA, "value": round(null_rate, 4)})
    for value, count in normalize_string(leads["best_source_type"]).fillna("unknown").value_counts(dropna=False).items():
        rows.append({"section": "count_by_source_type", "metric": "lead_count", "key": value, "value": int(count)})
    for value, count in normalize_string(leads["best_match_method"]).fillna("unknown").value_counts(dropna=False).items():
        rows.append({"section": "count_by_link_method", "metric": "lead_count", "key": value, "value": int(count)})
    for value, count in normalize_string(leads["source_confidence_tier"]).fillna("unknown").value_counts(dropna=False).items():
        rows.append({"section": "count_by_source_confidence_tier", "metric": "lead_count", "key": value, "value": int(count)})
    for value, count in normalize_string(leads["county_source_coverage_tier"]).fillna("unknown").value_counts(dropna=False).items():
        rows.append({"section": "count_by_county_source_coverage_tier", "metric": "lead_count", "key": value, "value": int(count)})
    for value, count in leads["has_reported_delinquent_amount_flag"].fillna(False).astype(bool).value_counts(dropna=False).items():
        rows.append({"section": "count_by_reported_amount_flag", "metric": "lead_count", "key": str(bool(value)).lower(), "value": int(count)})
    for _, row in county_frame.iterrows():
        rows.append({"section": "count_by_county", "metric": "lead_count", "key": row["county_name"], "value": int(row["delinquent_lead_count"])})
    return pd.DataFrame(rows)


def main() -> None:
    TAX_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    delinquent_parcels = pd.read_parquet(DELINQUENT_PARCELS_PATH)
    delinquent_parcels["parcel_row_id"] = delinquent_parcels["parcel_row_id"].astype("string")
    parcel_ids = delinquent_parcels["parcel_row_id"].dropna().astype("string").unique().tolist()

    master = read_filtered_geoparquet(PARCEL_MASTER_PATH, MASTER_COLUMNS, parcel_ids)
    owner = read_filtered_parquet(OWNER_LEADS_PATH, OWNER_LEADS_COLUMNS, parcel_ids)
    tax_distress = read_filtered_parquet(TAX_DISTRESS_PATH, TAX_DISTRESS_COLUMNS, parcel_ids)
    master["parcel_row_id"] = master["parcel_row_id"].astype("string")
    owner["parcel_row_id"] = owner["parcel_row_id"].astype("string")
    tax_distress["parcel_row_id"] = tax_distress["parcel_row_id"].astype("string")

    leads = delinquent_parcels.merge(master, on="parcel_row_id", how="inner", suffixes=("", "_master"))
    leads = leads.merge(owner, on="parcel_row_id", how="left", suffixes=("", "_owner"))
    leads = leads.merge(tax_distress, on="parcel_row_id", how="left", suffixes=("", "_taxdistress"))

    acreage = pd.to_numeric(leads.get("parcel_acres"), errors="coerce")
    acreage = acreage.where(acreage.notna(), pd.to_numeric(leads.get("total_acres"), errors="coerce"))
    acreage = acreage.where(acreage.notna(), pd.to_numeric(leads.get("tax_acres"), errors="coerce"))
    acreage = acreage.where(acreage.notna(), pd.to_numeric(leads.get("gis_acres"), errors="coerce"))

    leads["parcel_id"] = normalize_string(leads.get("parcel_id"), leads.index).fillna(normalize_string(leads.get("apn"), leads.index))
    leads["state_code"] = normalize_string(leads.get("state_code")).fillna("MS")
    leads["county_fips"] = normalize_string(leads.get("county_fips"), leads.index).fillna(normalize_string(leads.get("county_fips_master"), leads.index))
    leads["county_name"] = normalize_string(leads.get("county_name"), leads.index).fillna(normalize_string(leads.get("county_name_master"), leads.index))
    leads["owner_name"] = normalize_string(leads.get("owner_name_normalized"), leads.index).fillna(normalize_string(leads.get("owner_name_raw_owner"), leads.index)).fillna(normalize_string(leads.get("owner_name_raw"), leads.index)).fillna(normalize_string(leads.get("owner_name_master"), leads.index)).fillna(normalize_string(leads.get("owner_name"), leads.index))
    leads["owner_name_normalized"] = normalize_string(leads.get("owner_name_normalized"), leads.index).fillna(leads["owner_name"])
    leads["owner_address"] = normalize_string(leads.get("mailing_address_raw"), leads.index).fillna(join_line1_line2(leads.get("mailing_address_line1"), leads.get("mailing_address_line2")))
    leads["property_address"] = normalize_string(leads.get("property_address_raw"), leads.index).fillna(normalize_string(leads.get("situs_address"), leads.index)).fillna(normalize_string(leads.get("property_address_raw_master"), leads.index))
    leads["property_city"] = normalize_string(leads.get("property_city"), leads.index).fillna(normalize_string(leads.get("property_city_raw"), leads.index))
    leads["property_state"] = normalize_string(leads.get("property_state"), leads.index).fillna(normalize_string(leads.get("property_state_raw"), leads.index)).fillna("MS")
    leads["mailing_address_line1"] = normalize_string(leads.get("mailing_address_line1"), leads.index).fillna(normalize_string(leads.get("mailing_address_line1_raw"), leads.index))
    leads["mailing_address_line2"] = normalize_string(leads.get("mailing_address_line2"), leads.index).fillna(normalize_string(leads.get("mailing_address_line2_raw"), leads.index))
    leads["mailing_city"] = normalize_string(leads.get("mailing_city"), leads.index).fillna(normalize_string(leads.get("mailing_city_raw"), leads.index))
    leads["mailing_state"] = normalize_string(leads.get("mailing_state"), leads.index).fillna(normalize_string(leads.get("mailing_state_raw"), leads.index))
    leads["mailing_zip"] = normalize_string(leads.get("mailing_zip"), leads.index).fillna(normalize_string(leads.get("mailing_zip_raw"), leads.index))
    leads["mailing_zip5"] = normalize_string(leads.get("mailing_zip5"), leads.index)
    leads["acreage"] = acreage.astype("float64")
    leads["source_type"] = normalize_string(leads.get("best_source_type"), leads.index)
    leads["best_source_type"] = normalize_string(leads.get("best_source_type"), leads.index)
    leads["best_source_name"] = normalize_string(leads.get("best_source_name"), leads.index)
    leads["source_priority"] = leads["best_source_type"].map(SOURCE_PRIORITY).fillna(1).astype("int32")
    leads["delinquent_amount"] = pd.to_numeric(leads.get("best_delinquent_amount"), errors="coerce")
    leads["delinquent_flag"] = pd.Series(True, index=leads.index, dtype="boolean")
    leads["forfeited_flag"] = leads.get("has_forfeited_source", pd.Series(False, index=leads.index)).fillna(False).astype("boolean")
    leads["has_reported_delinquent_amount_flag"] = leads["delinquent_amount"].notna().astype("boolean")
    leads["best_match_method"] = normalize_string(leads.get("best_linkage_method"), leads.index)
    leads["best_match_confidence"] = pd.to_numeric(leads.get("best_match_confidence"), errors="coerce")
    leads["best_match_confidence_tier"] = normalize_string(leads.get("best_match_confidence_tier"), leads.index)
    leads["high_confidence_link_flag"] = leads["best_match_method"].isin(["exact_ppin", "exact_normalized_parcel_id"]).astype("boolean") & leads["best_match_confidence_tier"].fillna("unknown").isin(["high", "medium"]).astype("boolean")
    leads["source_confidence_tier"] = derive_source_confidence(leads)
    leads["county_source_coverage_tier"] = derive_county_coverage_tier(leads)
    leads["absentee_owner_flag"] = leads.get("absentee_owner_flag", pd.Series(pd.NA, index=leads.index, dtype="boolean")).astype("boolean")
    leads["out_of_state_owner_flag"] = leads.get("out_of_state_owner_flag", pd.Series(pd.NA, index=leads.index, dtype="boolean")).astype("boolean")
    leads["mailing_matches_situs_flag"] = compute_mailing_matches_situs(leads)
    leads["assessed_land_value"] = pd.to_numeric(leads.get("land_value"), errors="coerce")
    leads["assessed_improvement_value"] = (
        pd.to_numeric(leads.get("improvement_value_1"), errors="coerce").fillna(0.0)
        + pd.to_numeric(leads.get("improvement_value_2"), errors="coerce").fillna(0.0)
    )
    leads["assessed_total_value"] = pd.to_numeric(leads.get("total_value"), errors="coerce")
    leads["land_use"] = normalize_string(leads.get("land_use_raw_owner"), leads.index).fillna(normalize_string(leads.get("land_use_raw"), leads.index)).fillna(normalize_string(leads.get("zoning"), leads.index))
    leads["flood_risk_score"] = pd.to_numeric(leads.get("flood_risk_score"), errors="coerce")
    leads["wetland_flag"] = leads.get("wetland_flag", pd.Series(pd.NA, index=leads.index, dtype="boolean")).astype("boolean")
    leads["road_distance_ft"] = pd.to_numeric(leads.get("road_distance_ft"), errors="coerce")
    leads["buildability_score"] = pd.to_numeric(leads.get("buildability_score"), errors="coerce")
    leads["environment_score"] = pd.to_numeric(leads.get("environment_score"), errors="coerce")
    leads["investment_score"] = pd.to_numeric(leads.get("investment_score"), errors="coerce")
    leads["broadband_available"] = leads.get("broadband_available", pd.Series(pd.NA, index=leads.index, dtype="boolean")).astype("boolean")
    leads["electric_provider_name"] = normalize_string(leads.get("electric_provider_name"), leads.index)
    leads["owner_group_id"] = normalize_string(leads.get("owner_group_id"), leads.index).fillna(normalize_string(leads.get("owner_group_id_taxdistress"), leads.index))
    leads["owner_parcel_count"] = pd.to_numeric(leads.get("owner_parcel_count"), errors="coerce").fillna(pd.to_numeric(leads.get("owner_parcel_count_taxdistress"), errors="coerce")).astype("Int64")
    leads["owner_total_acres"] = pd.to_numeric(leads.get("owner_total_acres"), errors="coerce").fillna(pd.to_numeric(leads.get("owner_total_acres_taxdistress"), errors="coerce"))
    leads["owner_type"] = normalize_string(leads.get("owner_type"), leads.index)
    leads["corporate_owner_flag"] = leads.get("corporate_owner_flag", pd.Series(pd.NA, index=leads.index, dtype="boolean")).astype("boolean")
    leads["mailer_target_score"] = pd.to_numeric(leads.get("mailer_target_score"), errors="coerce")
    leads["parcel_constraint_summary"] = normalize_string(leads.get("parcel_constraint_summary"), leads.index)
    leads["tax_distress_score"] = pd.to_numeric(leads.get("tax_distress_score"), errors="coerce").fillna(pd.to_numeric(leads.get("tax_distress_score_taxdistress"), errors="coerce"))
    leads["distressed_owner_flag"] = leads.get("distressed_owner_flag", pd.Series(pd.NA, index=leads.index, dtype="boolean")).fillna(leads.get("distressed_owner_flag_taxdistress", pd.Series(pd.NA, index=leads.index, dtype="boolean"))).astype("boolean")
    leads["delinquent_year_count"] = pd.to_numeric(leads.get("delinquent_year_count"), errors="coerce").astype("Int64")
    leads["delinquent_tax_amount_total"] = pd.to_numeric(leads.get("delinquent_tax_amount_total"), errors="coerce")
    leads["tax_record_count"] = pd.to_numeric(leads.get("tax_record_count"), errors="coerce").astype("Int64")
    leads["latest_delinquent_year"] = pd.to_numeric(leads.get("latest_delinquent_year"), errors="coerce").astype("Int64")
    leads["most_severe_tax_status"] = normalize_string(leads.get("most_severe_tax_status"), leads.index)
    leads["acreage_bucket"] = acreage_bucket(leads["acreage"])
    leads["delinquent_amount_bucket"] = delinquent_amount_bucket(leads["delinquent_amount"])
    leads["source_count"] = pd.to_numeric(leads.get("source_count"), errors="coerce").fillna(0).astype("int32")
    leads["county_hosted_source_count"] = pd.to_numeric(leads.get("county_hosted_source_count"), errors="coerce").fillna(0).astype("int32")
    leads["statewide_source_count"] = pd.to_numeric(leads.get("statewide_source_count"), errors="coerce").fillna(0).astype("int32")
    leads["tax_year"] = pd.to_numeric(leads.get("tax_year"), errors="coerce").astype("Int64")
    leads["bill_year"] = pd.to_numeric(leads.get("bill_year"), errors="coerce").astype("Int64")
    leads["reported_delinquent_years"] = normalize_string(leads.get("reported_delinquent_years"), leads.index)
    leads["best_source_dataset_path"] = normalize_string(leads.get("best_source_dataset_path"), leads.index)
    leads["best_source_file_path"] = normalize_string(leads.get("best_source_file_path"), leads.index)
    leads["best_source_record_id"] = normalize_string(leads.get("best_source_record_id"), leads.index)
    leads["latest_loaded_at"] = normalize_string(leads.get("latest_loaded_at"), leads.index)

    lead_frame = gpd.GeoDataFrame(leads.loc[:, LEAD_COLUMNS].copy(), geometry="geometry", crs=master.crs)
    lead_frame = lead_frame.sort_values(["county_name", "parcel_row_id"]).reset_index(drop=True)

    county_frame = build_county_output(lead_frame)
    summary_frame = build_summary(lead_frame, delinquent_parcels, county_frame)

    lead_frame.to_parquet(OUTPUT_PARQUET, index=False)
    county_frame.to_csv(OUTPUT_COUNTY_CSV, index=False)
    summary_frame.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print(f"Statewide leads: {OUTPUT_PARQUET.relative_to(BASE_DIR)}")
    print(f"County leads: {OUTPUT_COUNTY_CSV.relative_to(BASE_DIR)}")
    print(f"Summary: {OUTPUT_SUMMARY_CSV.relative_to(BASE_DIR)}")
    print(f"Delinquent leads: {len(lead_frame):,}")


if __name__ == "__main__":
    main()
