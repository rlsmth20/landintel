from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from parcel_tax_delinquency_ms import parse_registry, resolve_path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "mississippi_tax_source_registry.yaml"
COVERAGE_MATRIX_PATH = ROOT / "data" / "parcels" / "mississippi_tax_coverage_matrix.csv"
OUTPUT_CSV = ROOT / "data" / "parcels" / "mississippi_tax_source_discovery_audit.csv"
OUTPUT_JSON = ROOT / "data" / "parcels" / "mississippi_tax_source_discovery_audit.json"
LEGACY_OUTPUT_CSV = ROOT / "data" / "parcels" / "mississippi_tax_source_freshness_audit.csv"
LEGACY_OUTPUT_JSON = ROOT / "data" / "parcels" / "mississippi_tax_source_freshness_audit.json"
TARGET_COUNTIES = ["jackson", "madison", "jasper", "pike", "warren", "calhoun", "clay", "coahoma"]
LDRED_LINKED_PATHS = {
    "calhoun": ROOT / "data" / "tax_linked" / "ms" / "calhoun" / "calhoun_land_redemption_linked_tax_records.parquet",
    "clay": ROOT / "data" / "tax_linked" / "ms" / "clay" / "clay_land_redemption_linked_tax_records.parquet",
    "coahoma": ROOT / "data" / "tax_linked" / "ms" / "coahoma" / "coahoma_land_redemption_linked_tax_records.parquet",
}


def _freshness_bucket(tax_year: int | None, current_year: int) -> str:
    if tax_year is None:
        return "unknown"
    if tax_year >= current_year:
        return "current"
    if tax_year == current_year - 1:
        return "actionable_recent"
    if tax_year == current_year - 2:
        return "stale_caution"
    if tax_year == current_year - 3:
        return "historical_only"
    return "non_actionable_historical"


def _safe_get(url: str) -> requests.Response | None:
    try:
        return requests.get(url, timeout=30, allow_redirects=True)
    except Exception:
        return None


def _safe_head(url: str) -> requests.Response | None:
    try:
        return requests.head(url, timeout=20, allow_redirects=True)
    except Exception:
        return None


def _linked_snapshot_summary(county_name: str) -> tuple[int | None, int]:
    path = LDRED_LINKED_PATHS.get(county_name)
    if path is None or not path.exists():
        return None, 0
    frame = pd.read_parquet(path, columns=["parcel_row_id", "tax_year"])
    year = pd.to_numeric(frame["tax_year"], errors="coerce").dropna()
    return (int(year.max()) if not year.empty else None, int(frame["parcel_row_id"].nunique()))


def _clean_optional_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _audit_jackson(current_url: str, current_tax_year: int | None, current_year: int) -> dict[str, Any]:
    response = _safe_get(current_url)
    text = response.text if response is not None and response.ok else ""
    fiscal_year_match = re.search(r"fiscal year\s+(\d{4})", text, flags=re.I)
    sale_year_match = re.search(r"august\s+\d{4}", text, flags=re.I)
    newer_candidates = [
        "https://co.jackson.ms.us/DocumentCenter/View/1549/Tax-Sale-Ad-for-2025-Real-Property-Taxes---Preliminary-List",
        "https://co.jackson.ms.us/DocumentCenter/View/1451/Tax-Sale-Ad-for-2025-Real-Property-Taxes---Preliminary-List",
    ]
    newer_url = next((url for url in newer_candidates if (_safe_head(url) or requests.Response()).status_code == 200), None)
    newer_found = newer_url is not None
    note_parts: list[str] = []
    if fiscal_year_match:
        note_parts.append(f"Current source still describes fiscal year {fiscal_year_match.group(1)} taxes.")
    if sale_year_match:
        note_parts.append(f"Sale cycle text references {sale_year_match.group(0)}.")
    return {
        "newer_public_source_found_flag": newer_found,
        "newer_public_source_url": newer_url,
        "newer_public_source_type": "county_documentcenter_text" if newer_found else None,
        "discovered_tax_year": 2025 if newer_found else None,
        "source_access_type": "public_document" if newer_found else "not_found",
        "notes": " ".join(note_parts) or "No newer public Jackson county tax source was found.",
        "refresh_status": "public_source_found_parser_needed" if newer_found else "no_newer_public_source_found",
        "recommended_action": "build_parser_for_newer_source" if newer_found else ("keep_stale_caution" if _freshness_bucket(current_tax_year, current_year) == "stale_caution" else "keep_historical_only"),
    }


def _audit_madison(current_url: str, current_tax_year: int | None) -> dict[str, Any]:
    candidate_urls = [
        "https://www.madison-co.com/sites/default/files/2025_tax_year_as_of_august_11_0.xlsx",
        "https://www.madison-co.com/sites/default/files/2025_tax_year_as_of_august_11.xlsx",
        "https://www.madison-co.com/sites/default/files/2026_tax_year_as_of_august_11_0.xlsx",
    ]
    newer_url = next((url for url in candidate_urls if (_safe_head(url) or requests.Response()).status_code == 200), None)
    newer_found = newer_url is not None
    return {
        "newer_public_source_found_flag": newer_found,
        "newer_public_source_url": newer_url,
        "newer_public_source_type": "county_xlsx_workbook" if newer_found else None,
        "discovered_tax_year": 2025 if newer_found else None,
        "source_access_type": "public_direct" if newer_found else "not_found",
        "notes": "Current public Madison workbook remains the 2024 tax-year file; 2025/2026 replacement workbook URLs were not publicly reachable." if not newer_found else "Newer public Madison workbook found.",
        "refresh_status": "public_source_found_parser_needed" if newer_found else "no_newer_public_source_found",
        "recommended_action": "build_parser_for_newer_source" if newer_found else "keep_stale_caution",
    }


def _audit_pike(current_url: str, current_tax_year: int | None, current_year: int) -> dict[str, Any]:
    response = _safe_get(current_url)
    text = response.text if response is not None and response.ok else ""
    links = re.findall(r'href=["\']([^"\']+)["\']', text, flags=re.I)
    sale_links = [link for link in links if "landsale" in link.lower()]
    years = sorted({int(match) for link in sale_links for match in re.findall(r"20\d{2}", link)})
    newer_year = next((year for year in reversed(years) if year >= current_year - 2), None)
    newer_url = next((link for link in sale_links if newer_year is not None and str(newer_year) in link), None)
    newer_found = newer_url is not None
    bucket = _freshness_bucket(current_tax_year, current_year)
    return {
        "newer_public_source_found_flag": newer_found,
        "newer_public_source_url": newer_url,
        "newer_public_source_type": "county_taxsale_download" if newer_found else None,
        "discovered_tax_year": newer_year if newer_found else None,
        "source_access_type": "public_direct" if newer_found else "not_found",
        "notes": f"Current Pike page links LANDSALE years: {', '.join(str(year) for year in years) if years else 'none'}." if not newer_found else f"Newer Pike LANDSALE file for {newer_year} found.",
        "refresh_status": "public_source_found_parser_needed" if newer_found else ("still_historical" if bucket in {"historical_only", "non_actionable_historical"} else "no_newer_public_source_found"),
        "recommended_action": "build_parser_for_newer_source" if newer_found else ("keep_historical_only" if bucket in {"historical_only", "non_actionable_historical"} else "keep_stale_caution"),
    }


def _audit_jasper(current_url: str, current_tax_year: int | None) -> dict[str, Any]:
    response = _safe_get(current_url)
    text = response.text if response is not None and response.ok else ""
    pdf_links = re.findall(r'https://co\.jasper\.ms\.us/wp-content/uploads/[^"\']+?/(\d{4})-Taxes\.pdf', text, flags=re.I)
    years = sorted({int(year) for year in pdf_links})
    newer_year = next((year for year in reversed(years) if current_tax_year is not None and year > current_tax_year), None)
    newer_url = f"https://co.jasper.ms.us/wp-content/uploads/2026/09/{newer_year}-Taxes.pdf" if newer_year is not None else None
    newer_found = newer_year is not None
    return {
        "newer_public_source_found_flag": newer_found,
        "newer_public_source_url": newer_url,
        "newer_public_source_type": "county_pdf_notice" if newer_found else None,
        "discovered_tax_year": newer_year if newer_found else None,
        "source_access_type": "public_document" if newer_found else "not_found",
        "notes": f"Current Jasper page exposes tax-year PDFs: {', '.join(str(year) for year in years) if years else 'none'}." if not newer_found else f"Newer Jasper PDF for {newer_year} found.",
        "refresh_status": "public_source_found_parser_needed" if newer_found else "no_newer_public_source_found",
        "recommended_action": "build_parser_for_newer_source" if newer_found else "keep_stale_caution",
    }


def _audit_warren(current_url: str, current_tax_year: int | None, current_year: int) -> dict[str, Any]:
    response = _safe_get(current_url)
    text = response.text if response is not None and response.ok else ""
    fiscal_year_match = re.search(r"fiscal year of\s+(\d{4})", text, flags=re.I)
    newer_found = False
    note = (
        f"Current Warren page still explicitly references fiscal year {fiscal_year_match.group(1)}."
        if fiscal_year_match
        else "No newer public Warren tax-year source was found."
    )
    bucket = _freshness_bucket(current_tax_year, current_year)
    return {
        "newer_public_source_found_flag": newer_found,
        "newer_public_source_url": None,
        "newer_public_source_type": None,
        "discovered_tax_year": None,
        "source_access_type": "not_found",
        "notes": note,
        "refresh_status": "still_historical" if bucket in {"historical_only", "non_actionable_historical"} else "no_newer_public_source_found",
        "recommended_action": "keep_historical_only" if bucket in {"historical_only", "non_actionable_historical"} else "keep_stale_caution",
    }


def _audit_ldred_candidate(county_name: str, current_tax_year: int | None, current_source_url: str, current_match_count: int) -> dict[str, Any]:
    discovered_tax_year, discovered_match_count = _linked_snapshot_summary(county_name)
    newer_found = discovered_tax_year is not None and (current_tax_year is None or discovered_tax_year > current_tax_year)
    url = _clean_optional_string(current_source_url) or {
        "calhoun": "https://cs.datasysmgt.com/tax?state=MS&county=7",
        "clay": "https://cs.datasysmgt.com/tax?state=MS&county=13",
        "coahoma": "https://cs.datasysmgt.com/tax?state=MS&county=14",
    }.get(county_name)
    if county_name == "coahoma":
        recommended_action = "parser_needed"
        refresh_status = "public_source_found_parser_needed"
        notes = (
            f"County-specific public land-redemption snapshot reaches tax year {discovered_tax_year} with {discovered_match_count} exact parcel matches, "
            f"but the currently configured SOS partial source still covers {current_match_count} parcels; a safe multi-source merge is needed before replacement."
        )
    elif county_name == "calhoun":
        recommended_action = "integrate_now"
        refresh_status = "refreshed"
        notes = (
            f"County-specific public land-redemption snapshot reaches tax year {discovered_tax_year} with {discovered_match_count} exact parcel matches; "
            "it safely promoted Calhoun from pending coverage to explicit county-specific stale-caution coverage."
        )
    else:
        recommended_action = "integrate_now"
        refresh_status = "refreshed"
        notes = (
            f"County-specific public land-redemption snapshot reaches tax year {discovered_tax_year} with {discovered_match_count} exact parcel matches; "
            "it safely replaced the older low-match SOS partial source."
        )
    return {
        "newer_public_source_found_flag": bool(newer_found or county_name in {"calhoun", "clay", "coahoma"}),
        "newer_public_source_url": url,
        "newer_public_source_type": "county_land_redemption_json",
        "discovered_tax_year": discovered_tax_year,
        "source_access_type": "public_direct",
        "notes": notes,
        "refresh_status": refresh_status,
        "recommended_action": recommended_action,
    }


def main() -> None:
    coverage = pd.read_csv(COVERAGE_MATRIX_PATH)
    registry = parse_registry(REGISTRY_PATH)
    county_entries = registry.get("counties", {}) if isinstance(registry.get("counties"), dict) else {}
    current_year = pd.Timestamp.now("UTC").year

    audit_rows: list[dict[str, Any]] = []
    for county_name in TARGET_COUNTIES:
        row = coverage.loc[coverage["county_name"].eq(county_name)]
        if row.empty:
            continue
        coverage_row = row.iloc[0].to_dict()
        entry = county_entries.get(county_name, {})
        current_url = _clean_optional_string(coverage_row.get("source_url")) or _clean_optional_string(entry.get("source_url")) or ""
        current_tax_year = pd.to_numeric(pd.Series([coverage_row.get("tax_data_year")]), errors="coerce").iloc[0]
        current_tax_year_int = int(current_tax_year) if pd.notna(current_tax_year) else None

        current_match_count = int(coverage_row.get("parcel_match_count") or 0)
        if county_name == "jackson":
            audit = _audit_jackson(current_url, current_tax_year_int, current_year)
        elif county_name == "madison":
            audit = _audit_madison(current_url, current_tax_year_int)
        elif county_name == "jasper":
            audit = _audit_jasper(current_url, current_tax_year_int)
        elif county_name == "pike":
            audit = _audit_pike(current_url, current_tax_year_int, current_year)
        elif county_name == "warren":
            audit = _audit_warren(current_url, current_tax_year_int, current_year)
        else:
            audit = _audit_ldred_candidate(county_name, current_tax_year_int, current_url, current_match_count)

        bucket = _freshness_bucket(current_tax_year_int, current_year)
        audit_rows.append(
            {
                "county_name": county_name,
                "currently_configured_source_url": current_url or None,
                "current_tax_data_year": current_tax_year_int,
                "freshness_bucket": bucket,
                "parcel_match_count": current_match_count,
                "newer_public_source_found_flag": bool(audit["newer_public_source_found_flag"]),
                "newer_public_source_url": audit["newer_public_source_url"],
                "newer_public_source_type": audit["newer_public_source_type"],
                "discovered_tax_year": audit["discovered_tax_year"],
                "source_access_type": audit["source_access_type"],
                "refresh_status": audit["refresh_status"],
                "recommended_action": audit["recommended_action"],
                "notes": audit["notes"],
            }
        )

    audit_frame = pd.DataFrame(audit_rows).sort_values(["parcel_match_count", "county_name"], ascending=[False, True]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(audit_frame.to_dict(orient="records"), indent=2), encoding="utf-8")
    audit_frame.to_csv(LEGACY_OUTPUT_CSV, index=False)
    LEGACY_OUTPUT_JSON.write_text(json.dumps(audit_frame.to_dict(orient="records"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
