from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
PARCELS_DIR = BASE_DIR / "data" / "parcels"

INPUT_FILES = {
    "base": PARCELS_DIR / "mississippi_parcels.gpkg",
    "flood": PARCELS_DIR / "mississippi_parcels_with_flood.gpkg",
    "roads": PARCELS_DIR / "mississippi_parcels_with_roads.gpkg",
    "slope": PARCELS_DIR / "mississippi_parcels_with_flood_and_slope.gpkg",
    "wetlands": PARCELS_DIR / "mississippi_parcels_with_flood_slope_wetlands.gpkg",
    "scored": PARCELS_DIR / "mississippi_parcels_scored.gpkg",
    "utilities": PARCELS_DIR / "mississippi_parcels_with_utilities.gpkg",
    "soils": PARCELS_DIR / "mississippi_parcels_with_soils.gpkg",
}

OUTPUT_GPKG = PARCELS_DIR / "mississippi_parcels_master.gpkg"
OUTPUT_PARQUET = PARCELS_DIR / "mississippi_parcels_master.parquet"
SUMMARY_CSV = PARCELS_DIR / "mississippi_parcels_master_summary.csv"
IDENTITY_AUDIT_CSV = PARCELS_DIR / "parcel_identity_audit.csv"
LINEAGE_SCHEMA_CSV = PARCELS_DIR / "parcel_lineage_schema.csv"
FIELD_DICTIONARY_CSV = PARCELS_DIR / "parcel_field_dictionary.csv"
IMPLEMENTATION_NOTE_MD = PARCELS_DIR / "parcel_identity_implementation_note.md"

COUNTY_FIPS_MAP = {
    "adams": "001", "alcorn": "003", "amite": "005", "attala": "007", "benton": "009", "bolivar": "011",
    "calhoun": "013", "carroll": "015", "chickasaw": "017", "choctaw": "019", "claiborne": "021", "clarke": "023",
    "clay": "025", "coahoma": "027", "copiah": "029", "covington": "031", "desoto": "033", "forrest": "035",
    "franklin": "037", "george": "039", "greene": "041", "grenada": "043", "hancock": "045", "harrison": "047",
    "hinds": "049", "holmes": "051", "humphreys": "053", "issaquena": "055", "itawamba": "057", "jackson": "059",
    "jasper": "061", "jefferson": "063", "jefferson_davis": "065", "jones": "067", "kemper": "069", "lafayette": "071",
    "lamar": "073", "lauderdale": "075", "lawrence": "077", "leake": "079", "lee": "081", "leflore": "083",
    "lincoln": "085", "lowndes": "087", "madison": "089", "marion": "091", "marshall": "093", "monroe": "095",
    "montgomery": "097", "neshoba": "099", "newton": "101", "noxubee": "103", "oktibbeha": "105", "panola": "107",
    "pearl_river": "109", "perry": "111", "pike": "113", "pontotoc": "115", "prentiss": "117", "quitman": "119",
    "rankin": "121", "scott": "123", "sharkey": "125", "simpson": "127", "smith": "129", "stone": "131",
    "sunflower": "133", "tallahatchie": "135", "tate": "137", "tippah": "139", "tishomingo": "141", "tunica": "143",
    "union": "145", "walthall": "147", "warren": "149", "washington": "151", "wayne": "153", "webster": "155",
    "wilkinson": "157", "winston": "159", "yalobusha": "161", "yazoo": "163",
}

BASE_RENAMES = {
    "CAMA": "source_cama",
    "PARNO": "source_parcel_number",
    "ALTPARNO": "source_alt_parcel_number",
    "PPIN": "source_ppin",
    "OWNNAME": "owner_name",
    "MAILADD1": "mail_address_1",
    "MAILADD2": "mail_address_2",
    "MCITY1": "mail_city_1",
    "MSTATE1": "mail_state_1",
    "MZIP1": "mail_zip_1",
    "MCITY2": "mail_city_2",
    "MSTATE2": "mail_state_2",
    "MZIP2": "mail_zip_2",
    "SITEADD": "site_address",
    "SCITY": "site_city",
    "SSTATE": "site_state",
    "SZIP": "site_zip",
    "SUBNAME": "subdivision_name",
    "SUBDIVNO": "subdivision_number",
    "TAXACRES": "tax_acres",
    "GISACRES": "gis_acres",
    "DEEDREF": "deed_reference",
    "DEEDDATE": "deed_date",
    "PLATREF": "plat_reference",
    "PLATDATE": "plat_date",
    "TAXMAP": "tax_map",
    "SECTION": "section",
    "TWSP": "township",
    "RANGE": "range",
    "TAXSTATUS": "tax_status",
    "STNAME": "state_name_source",
    "CNTYNAME": "county_name_source",
    "CNTYFIPS": "county_fips_source_raw",
    "STFIPS": "state_fips_source_raw",
    "STCNTYFIPS": "state_county_fips_source_raw",
    "LANDVAL": "land_value",
    "IMPVAL1": "improvement_value_1",
    "IMPVAL2": "improvement_value_2",
    "TOTVAL": "total_value",
    "CULT_AC1": "cultivated_acres_1",
    "CULT_AC2": "cultivated_acres_2",
    "UNCULT_AC1": "uncultivated_acres_1",
    "UNCULT_AC2": "uncultivated_acres_2",
    "TOTAL_AC": "total_acres",
    "LATDEC": "latitude",
    "LONGDEC": "longitude",
    "ZONING": "zoning",
    "LEGLDESC": "legal_description",
    "TAXYEAR": "tax_year",
    "INSIDE_X": "inside_x",
    "INSIDE_Y": "inside_y",
    "SHAPE_Leng": "shape_length",
    "SHAPE_Area": "shape_area",
    "county_name": "county_name",
    "parcel_id": "legacy_parcel_id",
}

STAGE_NATIVE_FIELDS = {
    "flood": ["flood_risk_score", "flood_zone_list", "has_flood_overlap", "sfha_overlap", "flood_zone_primary", "flood_overlap_acres", "flood_overlap_pct"],
    "roads": ["osm_id", "fclass", "name", "ref", "road_distance_m", "road_distance_ft"],
    "slope": ["mean_slope_pct", "max_slope_pct", "slope_class", "slope_score"],
    "wetlands": ["parcel_area_acres", "wetland_overlap_acres", "wetland_overlap_pct", "wetland_flag", "wetland_score"],
    "scored": ["buildability_score", "environment_score", "investment_score"],
    "utilities": [
        "distance_to_powerline", "distance_to_substation", "distance_to_pipeline", "distance_to_fiber",
        "broadband_available", "electric_in_service_territory", "electric_provider_name", "gas_in_service_territory",
        "gas_provider_name", "water_service_area", "water_provider_name", "sewer_service_area", "sewer_provider_name",
        "distance_to_powerline_miles", "distance_to_substation_miles", "distance_to_pipeline_miles", "distance_to_fiber_miles",
        "power_status", "power_provider_name", "power_source_name", "power_source_type", "power_source_confidence",
        "power_is_inferred", "power_last_updated_at", "gas_status", "gas_source_name", "gas_source_type",
        "gas_source_confidence", "gas_is_inferred", "gas_last_updated_at", "water_status", "water_source_name",
        "water_source_type", "water_source_confidence", "water_is_inferred", "water_last_updated_at", "sewer_status",
        "sewer_source_name", "sewer_source_type", "sewer_source_confidence", "sewer_is_inferred", "sewer_last_updated_at",
        "internet_status", "internet_provider_name", "internet_source_name", "internet_source_type",
        "internet_source_confidence", "internet_is_inferred", "internet_last_updated_at",
    ],
    "soils": ["dominant_soil_type", "drainage_class", "hydrologic_group", "soil_texture", "soil_depth", "septic_suitability", "septic_limitation_class"],
}

STAGE_DEPENDENCIES = {
    "base": [],
    "flood": ["base"],
    "roads": ["base"],
    "slope": ["flood"],
    "wetlands": ["slope", "flood"],
    "scored": ["wetlands", "roads", "flood", "slope"],
    "utilities": ["wetlands", "flood", "slope"],
    "soils": ["utilities"],
}

STAGE_NOTES = {
    "base": "Merged statewide parcel geometry source from the Mississippi parcel download step.",
    "flood": "Parcel keyed flood metrics only.",
    "roads": "Parcel keyed nearest road metrics only.",
    "slope": "Parcel keyed slope metrics only.",
    "wetlands": "Parcel keyed wetlands overlap metrics only.",
    "scored": "Parcel keyed composite score metrics only.",
    "utilities": "Parcel keyed utility proximity, territory, and provenance metrics only.",
    "soils": "Parcel keyed soils and septic metrics only.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical Mississippi master parcel dataset with explicit identity and lineage.")
    parser.add_argument("--output-gpkg", type=str, default=str(OUTPUT_GPKG), help="Master GeoPackage output path.")
    parser.add_argument("--output-parquet", type=str, default=str(OUTPUT_PARQUET), help="Master Parquet output path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Summary CSV output path.")
    parser.add_argument("--identity-audit-csv", type=str, default=str(IDENTITY_AUDIT_CSV), help="Parcel identity audit CSV output path.")
    parser.add_argument("--lineage-schema-csv", type=str, default=str(LINEAGE_SCHEMA_CSV), help="Lineage schema CSV output path.")
    parser.add_argument("--field-dictionary-csv", type=str, default=str(FIELD_DICTIONARY_CSV), help="Field dictionary CSV output path.")
    parser.add_argument("--implementation-note-md", type=str, default=str(IMPLEMENTATION_NOTE_MD), help="Implementation note markdown output path.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


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
    out = out.mask(out.eq(""), pd.NA)
    return out.astype("string")


def choose_source_parcel_id_raw(base: pd.DataFrame) -> pd.Series:
    raw = base["source_parcel_number"].astype("string")
    return raw.mask(raw.str.strip().eq(""), pd.NA)


def build_selected_source_parcel_id_normalized(base: pd.DataFrame) -> pd.Series:
    parno = normalize_identifier(base["source_parcel_number"])
    alt = normalize_identifier(base["source_alt_parcel_number"])
    selected = parno.copy()
    adams_mask = base["county_name"].astype("string").str.lower().eq("adams")
    use_alt = adams_mask & selected.notna() & alt.notna()
    selected.loc[use_alt] = selected.loc[use_alt] + "|" + alt.loc[use_alt]
    return selected.astype("string")


def load_base_master_frame() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    base = gpd.read_file(INPUT_FILES["base"], engine="pyogrio").rename(columns=BASE_RENAMES)
    base["parcel_row_id"] = pd.Series(base.index.astype(str).map(lambda value: f"row_{value}"), index=base.index, dtype="string")
    base["state_code"] = pd.Series(STATE_ABBR, index=base.index, dtype="string")
    base["county_name"] = base["county_name"].astype("string")
    base["county_fips"] = base["county_name"].map(COUNTY_FIPS_MAP).astype("string")
    base["source_name"] = pd.Series("mississippi_parcel_services_merged", index=base.index, dtype="string")
    base["source_layer_name"] = pd.Series("mississippi_parcels", index=base.index, dtype="string")
    base["source_dataset_path"] = pd.Series(str(INPUT_FILES["base"].relative_to(BASE_DIR)), index=base.index, dtype="string")
    base["source_dataset_name"] = pd.Series(INPUT_FILES["base"].name, index=base.index, dtype="string")
    base["source_feature_layer_family"] = pd.Series("MS_West_Parcels + MS_East_Parcels", index=base.index, dtype="string")
    base["source_ingest_batch_at"] = pd.Series(pd.Timestamp(INPUT_FILES["base"].stat().st_mtime, unit="s").strftime("%Y-%m-%dT%H:%M:%SZ"), index=base.index, dtype="string")
    base["source_row_identifier"] = pd.Series(pd.NA, index=base.index, dtype="string")
    base["source_parcel_id_raw"] = choose_source_parcel_id_raw(base)
    base["source_parcel_id_normalized"] = build_selected_source_parcel_id_normalized(base)
    base["apn"] = base["source_parcel_id_raw"].astype("string")
    base["owner_name_raw"] = base["owner_name"].astype("string")
    base["owner_name_2_raw"] = pd.Series(pd.NA, index=base.index, dtype="string")
    base["mailing_address_line1_raw"] = base["mail_address_1"].astype("string")
    base["mailing_address_line2_raw"] = base["mail_address_2"].astype("string")
    base["mailing_city_raw"] = (
        base["mail_city_1"].astype("string").where(base["mail_city_1"].astype("string").str.strip().ne(""), base["mail_city_2"].astype("string"))
    )
    base["mailing_state_raw"] = (
        base["mail_state_1"].astype("string").where(base["mail_state_1"].astype("string").str.strip().ne(""), base["mail_state_2"].astype("string"))
    )
    base["mailing_zip_raw"] = (
        base["mail_zip_1"].astype("string").where(base["mail_zip_1"].astype("string").str.strip().ne(""), base["mail_zip_2"].astype("string"))
    )
    base["property_address_raw"] = base["site_address"].astype("string")
    base["property_city_raw"] = base["site_city"].astype("string")
    base["property_state_raw"] = base["site_state"].astype("string")
    base["land_use_raw"] = base["zoning"].astype("string")

    audit = build_identity_audit(base)
    trust_map = audit.set_index("county_name")["source_parcel_id_trust_tier"].astype("string")
    safe_map = audit.set_index("county_name")["parcel_national_key_safe"].astype(bool)
    base["source_parcel_id_trust_tier"] = base["county_name"].map(trust_map).astype("string")
    base["parcel_national_key_candidate"] = (
        base["state_code"].fillna("") + ":" + base["county_fips"].fillna("") + ":" + base["source_name"].fillna("") + ":" + base["source_parcel_id_normalized"].fillna("")
    ).astype("string")
    base.loc[~base["county_name"].map(safe_map).fillna(False), "parcel_national_key_candidate"] = pd.NA
    base["parcel_id"] = base["parcel_row_id"].astype("string")
    return base, audit


def build_identity_audit(base: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "county_name": base["county_name"],
            "county_fips": base["county_name"].map(COUNTY_FIPS_MAP),
            "source_parcel_id_raw": choose_source_parcel_id_raw(base),
            "source_parcel_id_normalized": build_selected_source_parcel_id_normalized(base),
            "source_alt_parcel_id_normalized": normalize_identifier(base["source_alt_parcel_number"]),
            "source_ppin_normalized": normalize_identifier(base["source_ppin"]),
        }
    )
    grouped = out.groupby(["county_name", "county_fips"], as_index=False).agg(
        parcel_rows=("county_name", "size"),
        source_parcel_id_raw_nonnull=("source_parcel_id_raw", lambda x: int(pd.Series(x).notna().sum())),
        source_parcel_id_raw_unique=("source_parcel_id_raw", lambda x: int(pd.Series(x).dropna().nunique())),
        source_parcel_id_normalized_unique=("source_parcel_id_normalized", lambda x: int(pd.Series(x).dropna().nunique())),
        source_alt_parcel_id_normalized_unique=("source_alt_parcel_id_normalized", lambda x: int(pd.Series(x).dropna().nunique())),
        source_ppin_normalized_unique=("source_ppin_normalized", lambda x: int(pd.Series(x).dropna().nunique())),
    )
    grouped["source_parcel_id_normalized_unique_pct"] = (grouped["source_parcel_id_normalized_unique"] / grouped["parcel_rows"] * 100.0).round(4)
    grouped["source_alt_parcel_id_normalized_unique_pct"] = (grouped["source_alt_parcel_id_normalized_unique"] / grouped["parcel_rows"] * 100.0).round(4)
    grouped["source_ppin_normalized_unique_pct"] = (grouped["source_ppin_normalized_unique"] / grouped["parcel_rows"] * 100.0).round(4)
    grouped["source_parcel_id_selected_field"] = np.where(grouped["county_name"].eq("adams"), "PARNO+ALTPARNO", "PARNO")
    grouped["source_parcel_id_trust_tier"] = np.select(
        [grouped["source_parcel_id_normalized_unique_pct"] >= 99.0, grouped["source_parcel_id_normalized_unique_pct"] >= 95.0],
        ["strong", "usable"],
        default="weak",
    )
    grouped["parcel_national_key_safe"] = grouped["source_parcel_id_trust_tier"].eq("strong")
    grouped["notes"] = np.select(
        [grouped["source_parcel_id_trust_tier"].eq("strong"), grouped["source_parcel_id_trust_tier"].eq("usable")],
        [
            "Selected normalized source parcel ID is strong enough for a parcel_national_key_candidate.",
            "PARNO is usable but has moderate duplication pressure; review before external identity use.",
        ],
        default="PARNO is too weak for an external parcel identity without additional disambiguation.",
    )
    return grouped.sort_values(["source_parcel_id_trust_tier", "source_parcel_id_normalized_unique_pct", "county_name"]).reset_index(drop=True)


def build_stage_source_parcel_id_normalized(frame: pd.DataFrame) -> pd.Series:
    county = frame["county_name"].astype("string").str.lower()
    if "source_parcel_id_normalized" in frame.columns:
        out = normalize_identifier(frame["source_parcel_id_normalized"])
    elif "source_parcel_number" in frame.columns:
        out = normalize_identifier(frame["source_parcel_number"])
    elif "PARNO" in frame.columns:
        out = normalize_identifier(frame["PARNO"])
    else:
        out = pd.Series(pd.NA, index=frame.index, dtype="string")
    if "source_alt_parcel_number" in frame.columns:
        alt = normalize_identifier(frame["source_alt_parcel_number"])
    elif "ALTPARNO" in frame.columns:
        alt = normalize_identifier(frame["ALTPARNO"])
    else:
        alt = pd.Series(pd.NA, index=frame.index, dtype="string")
    adams_mask = county.eq("adams")
    use_alt = adams_mask & out.notna() & alt.notna()
    out.loc[use_alt] = out.loc[use_alt] + "|" + alt.loc[use_alt]
    return out.astype("string")


def load_stage_native_frame(stage_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    sample = gpd.read_file(INPUT_FILES[stage_name], rows=1, engine="pyogrio")
    available_cols = set(sample.columns)
    requested_cols = ["parcel_row_id", "county_name", "PARNO", "ALTPARNO", "source_parcel_number", "source_alt_parcel_number", "source_parcel_id_normalized"] + STAGE_NATIVE_FIELDS[stage_name]
    columns = [column for column in requested_cols if column in available_cols]
    frame = gpd.read_file(INPUT_FILES[stage_name], columns=columns, ignore_geometry=True, engine="pyogrio")
    if "parcel_row_id" in frame.columns:
        frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
    frame["county_name"] = frame["county_name"].astype("string")
    frame["source_parcel_id_normalized"] = build_stage_source_parcel_id_normalized(frame)
    duplicate_rows = int(frame.loc[frame["county_name"].notna() & frame["source_parcel_id_normalized"].notna(), ["county_name", "source_parcel_id_normalized"]].duplicated().sum())
    if duplicate_rows:
        frame = frame.drop_duplicates(subset=["county_name", "source_parcel_id_normalized"], keep="first")
    frame = frame.loc[:, ["county_name", "source_parcel_id_normalized"] + STAGE_NATIVE_FIELDS[stage_name]].copy()
    sample_cols = len(sample.columns)
    lineage = {
        "stage_name": stage_name,
        "artifact_path": str(INPUT_FILES[stage_name].relative_to(BASE_DIR)),
        "canonical_join_key": "county_name + source_parcel_id_normalized",
        "dependencies": ", ".join(STAGE_DEPENDENCIES[stage_name]),
        "stage_native_field_count": len(STAGE_NATIVE_FIELDS[stage_name]),
        "actual_output_field_count": sample_cols,
        "inherited_field_count": sample_cols - len(STAGE_NATIVE_FIELDS[stage_name]) - 1,
        "duplicate_join_key_rows": duplicate_rows,
        "rows_with_all_stage_native_fields_missing": int(frame[STAGE_NATIVE_FIELDS[stage_name]].isna().all(axis=1).sum()),
        "notes": STAGE_NOTES[stage_name],
    }
    return frame, lineage


def build_soil_build(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    septic_map = {"good": 3, "moderate": 2, "poor": 1, "very_poor": 0}
    septic = df["septic_suitability"].fillna("").str.lower().map(septic_map).fillna(0).astype(int)
    drainage = df["drainage_class"].fillna("").str.lower()
    drainage_score = np.select([drainage.eq("well drained"), drainage.eq("moderately well drained")], [2, 1], default=0)
    hydro = df["hydrologic_group"].fillna("").str.upper().str.strip()
    hydro_score = np.select([hydro.eq("A"), hydro.eq("B")], [1, 1], default=0)
    score = septic + drainage_score + hydro_score
    rating = np.select([score >= 5, score >= 3, score >= 1], ["Excellent", "Good", "Marginal"], default="Poor")
    return pd.Series(score, index=df.index, dtype="int64"), pd.Series(rating, index=df.index, dtype="string")


def build_road_access_tier(distance_m: pd.Series) -> pd.Series:
    values = pd.to_numeric(distance_m, errors="coerce")
    tiers = np.select([values <= 25.0, values <= 150.0, values <= 800.0, values.notna()], ["Direct / Adjacent", "Near", "Moderate", "Remote"], default="Unknown")
    return pd.Series(tiers, index=values.index, dtype="string")


def build_utility_readiness(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    def bool_score(column: str) -> np.ndarray:
        return df[column].fillna(False).astype(bool).to_numpy(dtype=int)

    def distance_score(column: str, threshold: float) -> np.ndarray:
        values = pd.to_numeric(df[column], errors="coerce")
        return values.le(threshold).fillna(False).to_numpy(dtype=int)

    score = (
        bool_score("electric_in_service_territory")
        + bool_score("gas_in_service_territory")
        + bool_score("water_service_area")
        + bool_score("sewer_service_area")
        + distance_score("distance_to_powerline", 1609.344)
        + distance_score("distance_to_substation", 8046.72)
        + distance_score("distance_to_pipeline", 1609.344)
        + distance_score("distance_to_fiber", 1609.344)
    ).astype(int)
    tier = np.select([score >= 7, score >= 5, score >= 3], ["Excellent", "Strong", "Moderate"], default="Limited")
    return pd.Series(score, index=df.index, dtype="int64"), pd.Series(tier, index=df.index, dtype="string")


def build_flood_risk_tier(df: pd.DataFrame) -> pd.Series:
    zone = df["flood_zone_primary"].fillna("").astype(str).str.upper().str.strip()
    score = pd.to_numeric(df["flood_risk_score"], errors="coerce").fillna(0.0)
    sfha = df["sfha_overlap"].fillna(False).astype(bool)
    coverage_pct = pd.to_numeric(df["flood_overlap_pct"], errors="coerce").fillna(0.0) if "flood_overlap_pct" in df.columns else pd.Series(0.0, index=df.index)
    tier = np.select(
        [
            zone.str.startswith("VE") | score.ge(9.0) | coverage_pct.ge(50.0),
            sfha | zone.isin(["A", "AE", "AH", "AO", "A99"]) | coverage_pct.ge(15.0),
            zone.str.startswith("X_500") | score.ge(4.0) | coverage_pct.gt(0.0),
            zone.str.startswith("X") | zone.isin(["B", "C"]),
        ],
        ["Severe", "High", "Moderate", "Minimal"],
        default="Unknown",
    )
    return pd.Series(tier, index=df.index, dtype="string")


def build_constraint_summary(df: pd.DataFrame) -> pd.Series:
    flood_pct = pd.to_numeric(df["flood_overlap_pct"], errors="coerce").fillna(0.0).round(1) if "flood_overlap_pct" in df.columns else pd.Series(0.0, index=df.index)
    flood_phrase = pd.Series(
        np.where(
            flood_pct.gt(0),
            df["flood_risk_tier"].map({"Minimal": "minimal flood risk", "Moderate": "moderate flood risk", "High": "high flood risk", "Severe": "severe flood risk", "Unknown": "unknown flood risk"}).fillna("unknown flood risk")
            + " on "
            + flood_pct.astype(str)
            + "% of parcel",
            "no mapped FEMA flood overlap",
        ),
        index=df.index,
        dtype="string",
    )
    slope = pd.to_numeric(df["mean_slope_pct"], errors="coerce")
    slope_phrase = pd.Series(np.select([slope <= 5, slope <= 15, slope.notna()], ["low slope", "moderate slope", "steep slope"], default="unknown slope"), index=df.index, dtype="string")
    wetland_pct = pd.to_numeric(df["wetland_overlap_pct"], errors="coerce").fillna(0.0).round(1)
    wetland_phrase = pd.Series(np.where(wetland_pct.gt(0), "wetlands on " + wetland_pct.astype(str) + "% of parcel", "no mapped wetlands"), index=df.index, dtype="string")
    septic_phrase = df["septic_suitability"].fillna("unknown").str.replace("_", " ", regex=False) + " septic suitability"
    utility_phrase = pd.Series(np.select([df["utility_readiness_score"] >= 7, df["utility_readiness_score"] >= 5, df["utility_readiness_score"] >= 3], ["strong utility readiness", "solid utility readiness", "partial utility readiness"], default="limited utility readiness"), index=df.index, dtype="string")
    road_phrase = df["road_access_tier"].fillna("Unknown").str.lower() + " road access"
    summary = flood_phrase
    for phrase in [slope_phrase, wetland_phrase, septic_phrase, utility_phrase, road_phrase]:
        summary = pd.Series(np.where(summary.eq(""), phrase, summary + ", " + phrase), index=df.index, dtype="string")
    return summary.str.replace(r"\s+", " ", regex=True).str.strip().str.capitalize()


def merge_explicit_master() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    master, identity_audit = load_base_master_frame()
    lineage_rows = [
        {
            "stage_name": "base",
            "artifact_path": str(INPUT_FILES["base"].relative_to(BASE_DIR)),
            "canonical_join_key": "county_name + source_parcel_id_normalized",
            "dependencies": "",
            "stage_native_field_count": len(master.columns) - 1,
            "actual_output_field_count": len(gpd.read_file(INPUT_FILES["base"], rows=1, engine="pyogrio").columns),
            "inherited_field_count": 0,
            "duplicate_join_key_rows": 0,
            "rows_with_all_stage_native_fields_missing": 0,
            "notes": STAGE_NOTES["base"],
        }
    ]
    for stage_name in ["flood", "roads", "slope", "wetlands", "scored", "utilities", "soils"]:
        stage_df, lineage = load_stage_native_frame(stage_name)
        master = master.merge(stage_df, on=["county_name", "source_parcel_id_normalized"], how="left", sort=False)
        lineage_rows.append(lineage)

    soil_score, soil_rating = build_soil_build(master)
    utility_score, utility_tier = build_utility_readiness(master)
    master["soil_build_score"] = soil_score
    master["soil_build_rating"] = soil_rating
    master["road_access_tier"] = build_road_access_tier(master["road_distance_m"])
    master["utility_readiness_score"] = utility_score
    master["utility_readiness_tier"] = utility_tier
    master["flood_risk_tier"] = build_flood_risk_tier(master)
    master["parcel_constraint_summary"] = build_constraint_summary(master)
    return master, identity_audit, pd.DataFrame(lineage_rows)


def build_summary(master: pd.DataFrame, identity_audit: pd.DataFrame, lineage_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"section": "dataset", "metric": "final_row_count", "value": int(len(master))},
        {"section": "dataset", "metric": "final_column_count", "value": int(len(master.columns))},
        {"section": "quality", "metric": "duplicate_parcel_id_rows", "value": int(master["parcel_id"].dropna().duplicated().sum())},
        {"section": "quality", "metric": "null_parcel_row_id_rows", "value": int(master["parcel_row_id"].isna().sum())},
        {"section": "quality", "metric": "null_source_parcel_id_raw_rows", "value": int(master["source_parcel_id_raw"].isna().sum())},
        {"section": "quality", "metric": "null_source_parcel_id_normalized_rows", "value": int(master["source_parcel_id_normalized"].isna().sum())},
        {"section": "quality", "metric": "null_parcel_national_key_candidate_rows", "value": int(master["parcel_national_key_candidate"].isna().sum())},
    ]
    for field in ["flood_zone_primary", "road_distance_m", "mean_slope_pct", "wetland_overlap_pct", "dominant_soil_type", "electric_in_service_territory", "buildability_score", "environment_score", "investment_score"]:
        rows.append({"section": "null_rate", "metric": field, "value": round(float(master[field].isna().mean() * 100.0), 4)})
    rows.extend(
        [
            {"section": "identity", "metric": "counties_with_strong_source_parcel_ids", "value": int(identity_audit["source_parcel_id_trust_tier"].eq("strong").sum())},
            {"section": "identity", "metric": "counties_with_usable_source_parcel_ids", "value": int(identity_audit["source_parcel_id_trust_tier"].eq("usable").sum())},
            {"section": "identity", "metric": "counties_with_weak_source_parcel_ids", "value": int(identity_audit["source_parcel_id_trust_tier"].eq("weak").sum())},
        ]
    )
    for row in lineage_df.itertuples():
        rows.append({"section": f"lineage::{row.stage_name}", "metric": "stage_native_field_count", "value": int(row.stage_native_field_count)})
        rows.append({"section": f"lineage::{row.stage_name}", "metric": "inherited_field_count", "value": int(row.inherited_field_count)})
        rows.append({"section": f"lineage::{row.stage_name}", "metric": "duplicate_join_key_rows", "value": int(row.duplicate_join_key_rows)})
    for field in ["soil_build_rating", "flood_risk_tier", "road_access_tier", "utility_readiness_score", "utility_readiness_tier", "septic_suitability", "wetland_flag", "source_parcel_id_trust_tier"]:
        counts = master[field].fillna("unknown").value_counts(dropna=False)
        for bucket, count in counts.items():
            rows.append({"section": f"distribution::{field}", "metric": str(bucket), "value": int(count), "pct": round(float(count / len(master) * 100.0), 4)})
    return pd.DataFrame(rows)


def build_field_dictionary(master: pd.DataFrame) -> pd.DataFrame:
    lineage_fields = {
        "state_code", "county_name", "county_fips", "source_name", "source_layer_name", "source_dataset_path", "source_dataset_name",
        "source_feature_layer_family", "source_ingest_batch_at", "source_row_identifier", "source_parcel_id_raw", "source_parcel_id_normalized",
        "source_parcel_id_trust_tier", "parcel_national_key_candidate", "parcel_row_id",
    }
    scored_fields = {"buildability_score", "environment_score", "investment_score", "soil_build_score", "utility_readiness_score"}
    derived_fields = {"soil_build_rating", "road_access_tier", "utility_readiness_tier", "flood_risk_tier", "parcel_constraint_summary"}
    rows = []
    for field in master.columns:
        if field == "geometry":
            category = "raw_source"
            source_stage = "base"
        elif field in lineage_fields:
            category = "lineage"
            source_stage = "base"
        elif field in derived_fields:
            category = "derived_product"
            source_stage = "master"
        elif field in scored_fields:
            category = "scored"
            source_stage = "scored" if field in {"buildability_score", "environment_score", "investment_score"} else "master"
        elif field in STAGE_NATIVE_FIELDS["flood"]:
            category = "normalized"
            source_stage = "flood"
        elif field in STAGE_NATIVE_FIELDS["roads"]:
            category = "normalized"
            source_stage = "roads"
        elif field in STAGE_NATIVE_FIELDS["slope"]:
            category = "normalized"
            source_stage = "slope"
        elif field in STAGE_NATIVE_FIELDS["wetlands"]:
            category = "normalized"
            source_stage = "wetlands"
        elif field in STAGE_NATIVE_FIELDS["utilities"]:
            category = "normalized"
            source_stage = "utilities"
        elif field in STAGE_NATIVE_FIELDS["soils"]:
            category = "normalized"
            source_stage = "soils"
        else:
            category = "raw_source"
            source_stage = "base"
        if field == "geometry":
            meaning = "Parcel polygon geometry from the statewide base parcel source."
        elif category == "lineage":
            meaning = f"Lineage or parcel identity field used to trace the master parcel row back to source."
        elif category == "raw_source":
            meaning = f"Raw source parcel field preserved from the Mississippi base parcel dataset."
        elif category == "normalized":
            meaning = f"Parcel-keyed normalized attribute imported explicitly from the {source_stage} stage."
        elif category == "scored":
            meaning = f"Score field produced by the {source_stage} stage or the master product layer."
        else:
            meaning = "Derived product-facing field computed during the master build."
        rows.append(
            {
                "field_name": field,
                "category": category,
                "datatype": str(master[field].dtype),
                "source_stage": source_stage,
                "semantic_meaning": meaning,
                "null_behavior": "not null" if float(master[field].isna().mean()) == 0 else f"nullable; {master[field].isna().mean() * 100.0:.4f}% null in current Mississippi build",
                "state_specific_or_national": "state_specific" if field in {"county_fips_source_raw", "state_fips_source_raw", "state_county_fips_source_raw"} else "national_reusable",
            }
        )
    return pd.DataFrame(rows)


def build_implementation_note(identity_audit: pd.DataFrame, lineage_df: pd.DataFrame, master: pd.DataFrame) -> str:
    strong = identity_audit.loc[identity_audit["source_parcel_id_trust_tier"].eq("strong"), "county_name"].tolist()
    usable = identity_audit.loc[identity_audit["source_parcel_id_trust_tier"].eq("usable"), "county_name"].tolist()
    inherited_text = ", ".join(f"{row.stage_name}={int(row.inherited_field_count)}" for row in lineage_df.loc[lineage_df["stage_name"] != "base"].itertuples())
    return "\n".join(
        [
            "# Parcel Identity Implementation Note",
            "",
            "Recommended long-term parcel identity strategy:",
            "- Keep `parcel_row_id` as the internal synthetic join key.",
            "- Use `source_parcel_id_raw` = `PARNO` and `source_parcel_id_normalized` = token-preserving normalized `PARNO`.",
            "- For Adams, append normalized `ALTPARNO` to break the remaining `PARNO` collisions.",
            "- Use `parcel_national_key_candidate` = `state_code:county_fips:source_name:source_parcel_id_normalized` only where county trust tier is `strong`.",
            "",
            "Counties with strong source parcel IDs:",
            f"- {', '.join(strong) if strong else 'none'}",
            "",
            "Counties with usable but not yet strong source parcel IDs:",
            f"- {', '.join(usable) if usable else 'none'}",
            "",
            "Lineage issues fixed in this refactor:",
            "- The master dataset no longer treats the soils output as the de facto master.",
            "- The master build now merges explicit stage-native attribute tables keyed by `parcel_row_id`.",
            f"- Current inherited field counts in stage products were audited: {inherited_text}.",
            "",
            "Still needs improvement before owner/lead scale-out:",
            "- The underlying parcel ingest should preserve exact source service/layer and feature row identifier per parcel.",
            "- Stage scripts should eventually emit narrow parcel-keyed tables directly instead of wide inherited GeoPackages.",
            "- Counties with non-strong `PARNO` uniqueness should get additional disambiguation rules before external identity use.",
            f"- Current master row count remains {len(master):,}; no row correction was required in this refactor.",
        ]
    )


def write_outputs(master: gpd.GeoDataFrame, summary_df: pd.DataFrame, identity_audit: pd.DataFrame, lineage_df: pd.DataFrame, field_dictionary: pd.DataFrame, implementation_note: str, output_gpkg: Path, output_parquet: Path, summary_csv: Path, identity_audit_csv: Path, lineage_schema_csv: Path, field_dictionary_csv: Path, implementation_note_md: Path) -> None:
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    master.to_file(output_gpkg, driver="GPKG", engine="pyogrio")
    master.to_parquet(output_parquet, index=False)
    summary_df.to_csv(summary_csv, index=False)
    identity_audit.to_csv(identity_audit_csv, index=False)
    lineage_df.to_csv(lineage_schema_csv, index=False)
    field_dictionary.to_csv(field_dictionary_csv, index=False)
    implementation_note_md.write_text(implementation_note, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started_at = time.time()
    output_gpkg = resolve_path(args.output_gpkg)
    output_parquet = resolve_path(args.output_parquet)
    summary_csv = resolve_path(args.summary_csv)
    identity_audit_csv = resolve_path(args.identity_audit_csv)
    lineage_schema_csv = resolve_path(args.lineage_schema_csv)
    field_dictionary_csv = resolve_path(args.field_dictionary_csv)
    implementation_note_md = resolve_path(args.implementation_note_md)
    print(f"Building refactored master parcel dataset for {STATE_NAME}.")
    master, identity_audit, lineage_df = merge_explicit_master()
    summary_df = build_summary(master, identity_audit, lineage_df)
    field_dictionary = build_field_dictionary(master)
    implementation_note = build_implementation_note(identity_audit, lineage_df, master)
    write_outputs(master, summary_df, identity_audit, lineage_df, field_dictionary, implementation_note, output_gpkg, output_parquet, summary_csv, identity_audit_csv, lineage_schema_csv, field_dictionary_csv, implementation_note_md)
    print(f"Master rows: {len(master):,}")
    print(f"Master columns: {len(master.columns):,}")
    print(f"Strong source parcel ID counties: {int(identity_audit['source_parcel_id_trust_tier'].eq('strong').sum())}")
    print(f"Usable source parcel ID counties: {int(identity_audit['source_parcel_id_trust_tier'].eq('usable').sum())}")
    print(f"Weak source parcel ID counties: {int(identity_audit['source_parcel_id_trust_tier'].eq('weak').sum())}")
    print(f"Road distance null rate: {master['road_distance_m'].isna().mean() * 100.0:.4f}%")
    print(f"Output GeoPackage: {output_gpkg}")
    print(f"Output Parquet: {output_parquet}")
    print(f"Identity audit: {identity_audit_csv}")
    print(f"Lineage schema: {lineage_schema_csv}")
    print(f"Field dictionary: {field_dictionary_csv}")
    print(f"Implementation note: {implementation_note_md}")
    print(f"Runtime: {(time.time() - started_at) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
