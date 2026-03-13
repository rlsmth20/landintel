from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from tax_common import (
    BASE_DIR,
    CANONICAL_TAX_COLUMNS,
    RAW_TAX_DIR,
    TAX_LINKED_DIR,
    TAX_METADATA_DIR,
    TAX_STANDARDIZED_DIR,
    build_record_hash,
    build_row_hash,
    clean_string,
    county_fips_from_name,
    infer_corporate_owner,
    link_standardized_tax_records,
    load_master_index,
    normalize_county_name,
    normalize_identifier,
    normalize_ppin,
    raw_payload_json,
    update_registry_row,
    write_json,
)

MASTER_PARQUET = BASE_DIR / "data" / "parcels" / "mississippi_parcels_master.parquet"
REGISTRY_CSV = BASE_DIR / "data" / "tax_metadata" / "tax_source_registry_ms.csv"
SOS_PAGE_URL = "https://tflgis.sos.ms.gov/"
SOS_WEBMAP_DATA_URL = "https://gisportal.its.ms.gov/portal/sharing/rest/content/items/{mapid}/data?f=json&token={token}"
SOS_LAYER_URL = "https://gisserver.its.ms.gov/arcgis/rest/services/Hosted/Active_Tax_Forfeited_Properties/FeatureServer/0"
SOS_QUERY_URL = SOS_LAYER_URL + "/query"
SOS_LAYER_METADATA_URL = SOS_LAYER_URL + "?f=pjson&token={token}"
SOURCE_NAME = "sos_tax_forfeited_lands_gis"
SOURCE_ID_MAP = "ms_000_sos_tax_forfeited_lands_map"
SOURCE_ID_PAGE = "ms_000_sos_tax_forfeited_lands"
SOURCE_FILE_VERSION = "Active_Tax_Forfeited_Properties/26ff49d7978e49f687e348bb3fc57d2a"
LINKAGE_CONFIG_PATH = BASE_DIR / "floodscraper" / "state_configs" / "tax_linkage_ms.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the Mississippi SOS free tax-forfeited inventory.")
    parser.add_argument("--download-dir", default=str(RAW_TAX_DIR / "ms"), help="Base raw tax directory.")
    parser.add_argument("--page-size", type=int, default=2000, help="ArcGIS page size.")
    return parser.parse_args()


def http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=180) as response:
        return response.read()


def http_get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return json.loads(http_get(url, headers=headers).decode("utf-8"))


def nullable_string(series: pd.Series) -> pd.Series:
    cleaned = clean_string(series)
    upper = cleaned.astype("string").str.upper()
    return cleaned.mask(upper.isin(["NULL", "N/A", "NONE"]))


def join_name_parts(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [nullable_string(frame[column]) for column in columns if column in frame.columns]
    if not parts:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    stacked = pd.concat(parts, axis=1)
    return clean_string(stacked.fillna("").agg(" ".join, axis=1))


def format_arcgis_date(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    dates = pd.to_datetime(values, unit="ms", utc=True, errors="coerce")
    return dates.dt.strftime("%Y-%m-%d").astype("string").mask(dates.isna(), pd.NA)


def source_paths(download_dir: Path) -> dict[str, Path]:
    run_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    raw_dir = download_dir / "000" / SOURCE_NAME / run_date
    standardized_dir = TAX_STANDARDIZED_DIR / "ms" / "statewide"
    linked_dir = TAX_LINKED_DIR / "ms" / "statewide"
    return {
        "raw_dir": raw_dir,
        "raw_page_html": raw_dir / "sos_page.html",
        "raw_webmap_json": raw_dir / "webmap_data.json",
        "raw_layer_metadata": raw_dir / "layer_metadata.json",
        "raw_query_dir": raw_dir / "query_pages",
        "manifest": raw_dir / "manifest.json",
        "standardized": standardized_dir / "sos_forfeited_tax_records.parquet",
        "linked": linked_dir / "sos_forfeited_linked_tax_records.parquet",
        "unmatched": linked_dir / "sos_forfeited_unmatched_tax_records.parquet",
        "ambiguous": linked_dir / "sos_forfeited_ambiguous_tax_links.parquet",
        "county_diagnostics": TAX_METADATA_DIR / "tax_free_sos_county_diagnostics_ms.csv",
        "unmatched_reason_summary": TAX_METADATA_DIR / "tax_free_sos_unmatched_reason_summary_ms.csv",
        "ambiguity_reason_summary": TAX_METADATA_DIR / "tax_free_sos_ambiguity_reason_summary_ms.csv",
    }


def fetch_app_context(raw_dir: Path) -> tuple[str, str, bytes, bytes, bytes]:
    page_headers = {"User-Agent": "Mozilla/5.0"}
    page_bytes = http_get(SOS_PAGE_URL, headers=page_headers)
    html = page_bytes.decode("utf-8", errors="ignore")
    token_match = re.search(r'id="tk"[^>]*value="([^"]+)"', html)
    mapid_match = re.search(r'id="mapid"[^>]*value="([^"]+)"', html)
    if not token_match or not mapid_match:
        raise RuntimeError("Unable to locate SOS app token or map id in page HTML.")
    token = token_match.group(1)
    mapid = mapid_match.group(1)
    referer_headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": SOS_PAGE_URL,
        "Origin": "https://tflgis.sos.ms.gov",
    }
    webmap_bytes = http_get(SOS_WEBMAP_DATA_URL.format(mapid=mapid, token=token), headers=referer_headers)
    layer_bytes = http_get(SOS_LAYER_METADATA_URL.format(token=token), headers=referer_headers)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return token, mapid, page_bytes, webmap_bytes, layer_bytes


def download_query_pages(raw_query_dir: Path, token: str, page_size: int) -> tuple[list[Path], list[dict[str, Any]], int]:
    raw_query_dir.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": SOS_PAGE_URL,
        "Origin": "https://tflgis.sos.ms.gov",
    }
    count_params = {"where": "1=1", "returnCountOnly": "true", "f": "json", "token": token}
    count_url = SOS_QUERY_URL + "?" + urlencode(count_params)
    count_payload = http_get_json(count_url, headers=headers)
    total_count = int(count_payload["count"])

    page_paths: list[Path] = []
    all_features: list[dict[str, Any]] = []
    offset = 0
    page_index = 1
    while offset < total_count:
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "false",
            "orderByFields": "objectid",
            "resultOffset": str(offset),
            "resultRecordCount": str(page_size),
            "f": "json",
            "token": token,
        }
        page_url = SOS_QUERY_URL + "?" + urlencode(params)
        page_bytes = http_get(page_url, headers=headers)
        page_path = raw_query_dir / f"query_page_{page_index:04d}.json"
        page_path.write_bytes(page_bytes)
        page_paths.append(page_path)
        payload = json.loads(page_bytes.decode("utf-8"))
        features = payload.get("features", [])
        all_features.extend(features)
        offset += len(features)
        if not features:
            break
        page_index += 1
    return page_paths, all_features, total_count


def standardize_sos(features: list[dict[str, Any]], run_id: str, raw_dir: Path) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(columns=CANONICAL_TAX_COLUMNS)
    frame = pd.json_normalize(features, sep="_")
    frame.columns = [column.replace("attributes_", "") for column in frame.columns]

    county_name = normalize_county_name(frame["county"])
    county_fips = county_fips_from_name(county_name)
    source_ppin = normalize_ppin(frame["ppin_number"].where(frame["ppin_number"].notna(), frame["ppin"]))
    parcel_id_raw = nullable_string(frame["parcel_number"])
    owner_company = nullable_string(frame["assessedownercompany"])
    owner_person = join_name_parts(
        frame,
        ["assessedownerprefix", "assessedownerfirst", "assessedownermiddle", "assessedownerlast", "assessedownersuffix"],
    )
    owner_name = owner_company.where(owner_company.notna(), owner_person)
    owner_address_line1 = nullable_string(frame["street_line_1"])
    owner_address_line2 = nullable_string(frame["street_line_2"])
    owner_city = nullable_string(frame["city"])
    owner_state = nullable_string(frame["state_abbreviation"]).fillna("MS")
    owner_zip = nullable_string(frame["zip"])
    sale_date = format_arcgis_date(frame["sale_date"])
    tax_year = pd.to_numeric(frame["certificatetaxyear"], errors="coerce").astype("Int64")
    delinquent_amount = pd.to_numeric(frame["sumoftaxfees"], errors="coerce").astype("float64")
    market_total_value = pd.to_numeric(frame["market_value"], errors="coerce").astype("float64")
    parcel_status = nullable_string(frame["parcel_status_description"]).fillna("active_tax_forfeited_inventory")

    standardized = pd.DataFrame(
        {
            "tax_record_row_id": pd.Series(
                (
                    build_row_hash(["MS", county_fips.iloc[i], SOURCE_NAME, frame.iloc[i]["objectid"], parcel_id_raw.iloc[i]])
                    for i in range(len(frame))
                ),
                index=frame.index,
                dtype="string",
            ),
            "parcel_row_id": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "parcel_id_raw": parcel_id_raw,
            "parcel_id_normalized": normalize_identifier(parcel_id_raw),
            "state_code": pd.Series("MS", index=frame.index, dtype="string"),
            "county_fips": county_fips,
            "county_name": county_name,
            "source_name": pd.Series(SOURCE_NAME, index=frame.index, dtype="string"),
            "source_type": pd.Series("statewide_public_inventory", index=frame.index, dtype="string"),
            "source_dataset_path": pd.Series(raw_dir.relative_to(BASE_DIR).as_posix(), index=frame.index, dtype="string"),
            "source_record_id": nullable_string(frame["objectid"].astype("Int64").astype("string")),
            "source_ppin": source_ppin,
            "ingestion_run_id": pd.Series(run_id, index=frame.index, dtype="string"),
            "source_file_version": pd.Series(SOURCE_FILE_VERSION, index=frame.index, dtype="string"),
            "loaded_at": pd.Series(pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), index=frame.index, dtype="string"),
            "owner_name": owner_name,
            "owner_name_2": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "owner_address_line1": owner_address_line1,
            "owner_address_line2": owner_address_line2,
            "owner_city": owner_city,
            "owner_state": owner_state.astype("string"),
            "owner_zip": owner_zip,
            "situs_address": nullable_string(frame["propertyaddress"]),
            "situs_city": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "situs_state": pd.Series("MS", index=frame.index, dtype="string"),
            "situs_zip": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "assessed_land_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "assessed_improvement_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "assessed_total_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_land_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_improvement_value": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "market_total_value": market_total_value,
            "taxable_value": market_total_value,
            "exemptions_text": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "exemptions_amount": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_year": tax_year,
            "bill_year": tax_year,
            "tax_amount_due": delinquent_amount,
            "tax_amount_paid": pd.Series(np.nan, index=frame.index, dtype="float64"),
            "tax_balance_due": delinquent_amount,
            "tax_status": parcel_status,
            "payment_status": pd.Series("unpaid", index=frame.index, dtype="string"),
            "delinquent_flag": pd.Series(True, index=frame.index, dtype="boolean"),
            "forfeited_flag": pd.Series(True, index=frame.index, dtype="boolean"),
            "delinquent_amount": delinquent_amount,
            "delinquent_years": tax_year.astype("string"),
            "delinquent_as_of_date": sale_date,
            "last_payment_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "due_date": pd.Series(pd.NA, index=frame.index, dtype="string"),
            "absentee_owner_flag": pd.Series(pd.NA, index=frame.index, dtype="boolean"),
            "out_of_state_owner_flag": owner_state.astype("string").ne("MS").astype("boolean"),
            "owner_corporate_flag": infer_corporate_owner(owner_name),
            "mailing_matches_situs_flag": (owner_address_line1 == nullable_string(frame["propertyaddress"])).astype("boolean"),
            "tax_delinquent_flag_standardized": pd.Series(True, index=frame.index, dtype="boolean"),
            "raw_payload_json": raw_payload_json(
                frame,
                [
                    "objectid",
                    "county",
                    "parcel_number",
                    "ppin",
                    "ppin_number",
                    "sale_date",
                    "certificatetaxyear",
                    "sumoftaxfees",
                    "market_value",
                    "parcel_status_description",
                    "assessedownercompany",
                    "assessedownerfirst",
                    "assessedownerlast",
                    "propertyaddress",
                    "street_line_1",
                    "city",
                    "state_abbreviation",
                    "zip",
                    "legal_description",
                    "certificate_number",
                    "link",
                    "parcel_id",
                ],
            ),
        }
    )
    standardized["record_hash"] = build_record_hash(
        standardized,
        [
            "county_fips",
            "parcel_id_normalized",
            "source_ppin",
            "owner_name",
            "market_total_value",
            "tax_year",
            "tax_balance_due",
            "tax_status",
        ],
    )
    return standardized


def write_summaries(
    standardized: pd.DataFrame,
    linked: pd.DataFrame,
    unmatched: pd.DataFrame,
    ambiguous: pd.DataFrame,
    county_summary: pd.DataFrame,
    unmatched_reason_summary: pd.DataFrame,
    ambiguity_reason_summary: pd.DataFrame,
    county_diagnostics: pd.DataFrame,
    paths: dict[str, Path],
) -> tuple[Path, Path, Path, Path, Path]:
    TAX_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    county_summary_path = TAX_METADATA_DIR / "tax_free_sos_county_summary_ms.csv"
    statewide_summary_path = TAX_METADATA_DIR / "tax_free_sos_ingest_summary_ms.csv"
    county_summary.to_csv(county_summary_path, index=False)
    unmatched_reason_summary.to_csv(paths["unmatched_reason_summary"], index=False)
    ambiguity_reason_summary.to_csv(paths["ambiguity_reason_summary"], index=False)
    county_diagnostics.to_csv(paths["county_diagnostics"], index=False)

    summary_df = pd.DataFrame(
        [
            {"metric": "sos_standardized_rows", "value": int(len(standardized))},
            {"metric": "sos_linked_rows", "value": int(len(linked))},
            {"metric": "sos_unmatched_rows", "value": int(len(unmatched))},
            {"metric": "sos_ambiguous_rows", "value": int(len(ambiguous))},
            {"metric": "sos_exact_ppin_rows", "value": int(linked["linkage_method"].eq("exact_ppin").sum()) if not linked.empty else 0},
            {"metric": "sos_exact_parcel_rows", "value": int(linked["linkage_method"].eq("exact_normalized_parcel_id").sum()) if not linked.empty else 0},
            {
                "metric": "sos_heuristic_rows",
                "value": int(linked["linkage_method"].astype("string").str.startswith("heuristic_").sum()) if not linked.empty else 0,
            },
            {"metric": "sos_linkage_rate", "value": round(float(len(linked) / max(len(standardized), 1) * 100.0), 4)},
            {
                "metric": "sos_total_delinquent_amount",
                "value": round(float(pd.to_numeric(standardized["delinquent_amount"], errors="coerce").fillna(0.0).sum()), 2),
            },
            {"metric": "sos_null_owner_name_rows", "value": int(standardized["owner_name"].isna().sum())},
            {"metric": "sos_null_situs_address_rows", "value": int(standardized["situs_address"].isna().sum())},
            {"metric": "sos_null_ppin_rows", "value": int(standardized["source_ppin"].isna().sum())},
        ]
    )
    summary_df.to_csv(statewide_summary_path, index=False)
    return (
        county_summary_path,
        statewide_summary_path,
        paths["unmatched_reason_summary"],
        paths["ambiguity_reason_summary"],
        paths["county_diagnostics"],
    )


def load_linkage_config(standardized: pd.DataFrame) -> dict[str, list[str]]:
    if LINKAGE_CONFIG_PATH.exists():
        payload = json.loads(LINKAGE_CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        payload = {}
    default_variants = payload.get("default_heuristic_variants", ["compact_alnum"])
    county_overrides = payload.get("county_identifier_variants", {})
    counties = standardized["county_name"].dropna().astype("string").unique().tolist()
    config: dict[str, list[str]] = {}
    for county in counties:
        config[county] = list(county_overrides.get(county, default_variants))
    return config


def build_reason_summaries(unmatched: pd.DataFrame, ambiguous: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if unmatched.empty:
        unmatched_reason_summary = pd.DataFrame(columns=["county_name", "county_fips", "unmatched_reason", "row_count"])
    else:
        unmatched_reason_summary = (
            unmatched.groupby(["county_name", "county_fips", "unmatched_reason"], dropna=False)
            .size()
            .rename("row_count")
            .reset_index()
            .sort_values(["county_fips", "county_name", "row_count"], ascending=[True, True, False])
            .reset_index(drop=True)
        )
    if ambiguous.empty:
        ambiguity_reason_summary = pd.DataFrame(columns=["county_name", "county_fips", "ambiguity_reason", "row_count"])
    else:
        ambiguity_reason_summary = (
            ambiguous.groupby(["county_name", "county_fips", "ambiguity_reason"], dropna=False)
            .size()
            .rename("row_count")
            .reset_index()
            .sort_values(["county_fips", "county_name", "row_count"], ascending=[True, True, False])
            .reset_index(drop=True)
        )
    return unmatched_reason_summary, ambiguity_reason_summary


def build_county_diagnostics(
    standardized: pd.DataFrame,
    linked: pd.DataFrame,
    unmatched: pd.DataFrame,
    ambiguous: pd.DataFrame,
    master: pd.DataFrame,
) -> pd.DataFrame:
    diag_rows: list[dict[str, Any]] = []
    standardized = standardized.copy()
    standardized["compact_source_id"] = standardized["parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    master = master.copy()
    master["compact_master_id"] = master["source_parcel_id_normalized"].astype("string").str.replace(r"[^A-Z0-9]+", "", regex=True)
    for county in sorted(standardized["county_name"].dropna().astype("string").unique().tolist()):
        county_std = standardized.loc[standardized["county_name"].eq(county)].copy()
        county_master = master.loc[master["county_name"].eq(county)].copy()
        county_linked = linked.loc[linked["county_name"].eq(county)] if not linked.empty else linked
        county_unmatched = unmatched.loc[unmatched["county_name"].eq(county)] if not unmatched.empty else unmatched
        county_ambiguous = ambiguous.loc[ambiguous["county_name"].eq(county)] if not ambiguous.empty else ambiguous

        compact_overlap = 0
        if not county_std.empty and not county_master.empty:
            compact_overlap = int(county_std["compact_source_id"].isin(set(county_master["compact_master_id"].dropna())).sum())
        likely_causes: list[str] = []
        missing_ppin_rate = float(county_std["source_ppin"].isna().mean() * 100.0)
        if missing_ppin_rate >= 50.0:
            likely_causes.append("missing_ppin")
        if compact_overlap == 0 and len(county_std) > 0:
            likely_causes.append("county_format_divergence")
        if not county_ambiguous.empty and int(county_ambiguous["ambiguity_reason"].eq("duplicate_source_identifier").sum()) > 0:
            likely_causes.append("duplicate_source_identifier")
        if not county_ambiguous.empty and int(county_ambiguous["ambiguity_reason"].eq("duplicate_parcel_master_identifier").sum()) > 0:
            likely_causes.append("duplicate_parcel_master_identifier")
        if not likely_causes:
            likely_causes.append("mixed_minor_issues")

        diag_rows.append(
            {
                "county_name": county,
                "county_fips": county_std["county_fips"].dropna().iloc[0] if not county_std.empty else pd.NA,
                "standardized_rows": int(len(county_std)),
                "linked_rows": int(len(county_linked)),
                "unmatched_rows": int(len(county_unmatched)),
                "ambiguous_rows": int(len(county_ambiguous)),
                "source_ppin_nonnull_rate": round(float(county_std["source_ppin"].notna().mean() * 100.0), 4),
                "source_duplicate_ppin_rows": int(county_std["source_ppin"].duplicated(keep=False).fillna(False).sum()),
                "source_duplicate_parcel_rows": int(county_std["parcel_id_normalized"].duplicated(keep=False).fillna(False).sum()),
                "master_duplicate_ppin_rows": int(county_master["source_ppin"].duplicated(keep=False).fillna(False).sum()),
                "master_duplicate_parcel_rows": int(county_master["source_parcel_id_normalized"].duplicated(keep=False).fillna(False).sum()),
                "compact_identifier_overlap_rows": compact_overlap,
                "compact_identifier_overlap_rate": round(float(compact_overlap / max(len(county_std), 1) * 100.0), 4),
                "avg_source_parcel_id_length": round(float(county_std["parcel_id_normalized"].astype("string").str.len().dropna().mean()), 2),
                "avg_master_parcel_id_length": round(float(county_master["source_parcel_id_normalized"].astype("string").str.len().dropna().mean()), 2),
                "likely_causes": "|".join(dict.fromkeys(likely_causes)),
            }
        )
    return pd.DataFrame(diag_rows).sort_values(["county_fips", "county_name"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run_id = hashlib.sha1(pd.Timestamp.now("UTC").isoformat().encode("utf-8")).hexdigest()[:12]
    paths = source_paths(Path(args.download_dir))
    for key in ["raw_dir", "standardized", "linked"]:
        path = paths[key]
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)

    token, mapid, page_bytes, webmap_bytes, layer_bytes = fetch_app_context(paths["raw_dir"])
    paths["raw_page_html"].write_bytes(page_bytes)
    paths["raw_webmap_json"].write_bytes(webmap_bytes)
    paths["raw_layer_metadata"].write_bytes(layer_bytes)
    page_paths, features, total_count = download_query_pages(paths["raw_query_dir"], token, args.page_size)

    standardized = standardize_sos(features, run_id, paths["raw_dir"])
    master = load_master_index(MASTER_PARQUET)
    linked, unmatched, ambiguous, county_summary = link_standardized_tax_records(
        standardized,
        master,
        heuristic_variants_by_county=load_linkage_config(standardized),
    )
    unmatched_reason_summary, ambiguity_reason_summary = build_reason_summaries(unmatched, ambiguous)
    county_diagnostics = build_county_diagnostics(standardized, linked, unmatched, ambiguous, master)

    standardized.to_parquet(paths["standardized"], index=False)
    linked.to_parquet(paths["linked"], index=False)
    unmatched.to_parquet(paths["unmatched"], index=False)
    ambiguous.to_parquet(paths["ambiguous"], index=False)
    county_summary_path, statewide_summary_path, unmatched_reason_path, ambiguity_reason_path, county_diagnostics_path = write_summaries(
        standardized,
        linked,
        unmatched,
        ambiguous,
        county_summary,
        unmatched_reason_summary,
        ambiguity_reason_summary,
        county_diagnostics,
        paths,
    )

    write_json(
        paths["manifest"],
        {
            "ingestion_run_id": run_id,
            "state_code": "MS",
            "county_fips": "000",
            "county_name": "statewide",
            "source_name": SOURCE_NAME,
            "source_page_url": SOS_PAGE_URL,
            "source_layer_url": SOS_LAYER_URL,
            "source_map_id": mapid,
            "source_file_version": SOURCE_FILE_VERSION,
            "raw_page_html": paths["raw_page_html"].relative_to(BASE_DIR).as_posix(),
            "raw_webmap_json": paths["raw_webmap_json"].relative_to(BASE_DIR).as_posix(),
            "raw_layer_metadata": paths["raw_layer_metadata"].relative_to(BASE_DIR).as_posix(),
            "raw_query_pages": [path.relative_to(BASE_DIR).as_posix() for path in page_paths],
            "standardized_path": paths["standardized"].relative_to(BASE_DIR).as_posix(),
            "linked_path": paths["linked"].relative_to(BASE_DIR).as_posix(),
            "unmatched_path": paths["unmatched"].relative_to(BASE_DIR).as_posix(),
            "ambiguous_path": paths["ambiguous"].relative_to(BASE_DIR).as_posix(),
            "county_summary_path": county_summary_path.relative_to(BASE_DIR).as_posix(),
            "statewide_summary_path": statewide_summary_path.relative_to(BASE_DIR).as_posix(),
            "unmatched_reason_summary_path": unmatched_reason_path.relative_to(BASE_DIR).as_posix(),
            "ambiguity_reason_summary_path": ambiguity_reason_path.relative_to(BASE_DIR).as_posix(),
            "county_diagnostics_path": county_diagnostics_path.relative_to(BASE_DIR).as_posix(),
            "row_count": int(len(standardized)),
            "expected_count": int(total_count),
        },
    )
    downloaded_at = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    note_text = (
        f"Downloaded SOS page, webmap, layer metadata, and {len(page_paths)} paged JSON query snapshots "
        f"to {paths['raw_dir'].relative_to(BASE_DIR).as_posix()}. Layer URL: {SOS_LAYER_URL}."
    )
    update_registry_row(REGISTRY_CSV, SOURCE_ID_MAP, downloaded_at, note_text)
    update_registry_row(REGISTRY_CSV, SOURCE_ID_PAGE, downloaded_at, note_text)

    print(f"SOS forfeited records standardized: {len(standardized):,}")
    print(f"Linked rows: {len(linked):,}")
    print(f"Unmatched rows: {len(unmatched):,}")
    print(f"Ambiguous rows: {len(ambiguous):,}")
    print(f"County summary: {county_summary_path.relative_to(BASE_DIR)}")
    print(f"Statewide summary: {statewide_summary_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
