from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
TAX_METADATA_DIR = BASE_DIR / "data" / "tax_metadata"


def load_metric_lookup(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "metric" not in frame.columns:
        return {}
    return dict(zip(frame["metric"].astype("string"), frame["value"]))


def build_candidate_rows() -> list[dict[str, Any]]:
    pike = load_metric_lookup(TAX_METADATA_DIR / "tax_free_ingest_summary_ms.csv")
    sos = load_metric_lookup(TAX_METADATA_DIR / "tax_free_sos_ingest_summary_ms.csv")
    hinds = load_metric_lookup(TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv")
    warren = load_metric_lookup(TAX_METADATA_DIR / "tax_free_warren_linkage_summary_ms.csv")
    madison = load_metric_lookup(TAX_METADATA_DIR / "tax_free_madison_linkage_summary_ms.csv")
    jasper = load_metric_lookup(TAX_METADATA_DIR / "tax_free_jasper_linkage_summary_ms.csv")
    jackson = load_metric_lookup(TAX_METADATA_DIR / "tax_free_jackson_linkage_summary_ms.csv")

    return [
        {
            "county_name": "warren",
            "source_type": "county_html_listing",
            "access_method": "direct_county_page",
            "identifier_fields_present": "parcel_id|ppin",
            "free_public_access": 1,
            "structured_parseable": 1,
            "ppin_present": 1,
            "parcel_overlap_signal": 1.0,
            "duplicate_source_identifier_risk": 0.15,
            "null_identifier_risk": 0.0,
            "source_freshness_score": 1.0,
            "actual_linkage_rate": float(warren.get("warren_linkage_rate", 0.0)),
            "expected_linkage_tier": "high",
            "rank_reason": "County-hosted, free, parseable, includes PPIN, and direct parcel-ID alignment to parcel master is already proven.",
        },
        {
            "county_name": "madison",
            "source_type": "county_xlsx_download",
            "access_method": "direct_county_download",
            "identifier_fields_present": "parcel_id|owner|delinquent_amount",
            "free_public_access": 1,
            "structured_parseable": 1,
            "ppin_present": 0,
            "parcel_overlap_signal": 0.98,
            "duplicate_source_identifier_risk": 0.0,
            "null_identifier_risk": 0.0,
            "source_freshness_score": 0.9,
            "actual_linkage_rate": float(madison.get("madison_linkage_rate", 0.0)),
            "expected_linkage_tier": "high",
            "rank_reason": "County-hosted direct XLSX with complete parcel IDs, full delinquent amounts, and very strong parcel-master overlap despite lacking PPIN.",
        },
        {
            "county_name": "pike",
            "source_type": "county_csv_download",
            "access_method": "direct_county_download",
            "identifier_fields_present": "parcel_id|ppin|delinquent_amount",
            "free_public_access": 1,
            "structured_parseable": 1,
            "ppin_present": 1,
            "parcel_overlap_signal": 0.95,
            "duplicate_source_identifier_risk": 0.2,
            "null_identifier_risk": 0.0,
            "source_freshness_score": 0.9,
            "actual_linkage_rate": float(pike.get("pike_linkage_rate", 0.0)),
            "expected_linkage_tier": "high",
            "rank_reason": "Already validated success pattern: direct county CSV with PPIN-first linkage and strong exact match performance.",
        },
        {
            "county_name": "jackson",
            "source_type": "county_text_document",
            "access_method": "county_documentcenter_download",
            "identifier_fields_present": "parcel_id|account_id|delinquent_amount",
            "free_public_access": 1,
            "structured_parseable": 1,
            "ppin_present": 0,
            "parcel_overlap_signal": 0.0,
            "duplicate_source_identifier_risk": 0.05,
            "null_identifier_risk": 0.0,
            "source_freshness_score": 0.95,
            "actual_linkage_rate": float(jackson.get("jackson_linkage_rate", 0.0)),
            "expected_linkage_tier": "medium",
            "rank_reason": "County-hosted and orderly, but current county document exposes account IDs and parcel numbers that do not align to parcel master keys yet.",
        },
        {
            "county_name": "jasper",
            "source_type": "county_pdf_download",
            "access_method": "direct_county_download",
            "identifier_fields_present": "parcel_id|ppin|owner|delinquent_amount",
            "free_public_access": 1,
            "structured_parseable": 1,
            "ppin_present": 1,
            "parcel_overlap_signal": 0.73,
            "duplicate_source_identifier_risk": 0.23,
            "null_identifier_risk": 0.0,
            "source_freshness_score": 0.95,
            "actual_linkage_rate": float(jasper.get("jasper_linkage_rate", 0.0)),
            "expected_linkage_tier": "high",
            "rank_reason": "County-hosted current PDFs are public and extractable, with both PPIN and parcel IDs; linkage is good enough for production even though the sold-not-redeemed lists produce more duplicate identifiers than Warren or Pike.",
        },
        {
            "county_name": "hinds",
            "source_type": "county_xlsx_download",
            "access_method": "direct_county_download",
            "identifier_fields_present": "parcel_id",
            "free_public_access": 1,
            "structured_parseable": 1,
            "ppin_present": 0,
            "parcel_overlap_signal": 0.0,
            "duplicate_source_identifier_risk": 0.45,
            "null_identifier_risk": 0.2,
            "source_freshness_score": 0.9,
            "actual_linkage_rate": float(hinds.get("hinds_linkage_rate", 0.0)),
            "expected_linkage_tier": "low",
            "rank_reason": "Good county-hosted structured source, but no PPIN and no observed parcel-ID alignment to parcel master.",
        },
        {
            "county_name": "statewide_sos",
            "source_type": "statewide_arcgis_inventory",
            "access_method": "public_arcgis_feature_service",
            "identifier_fields_present": "parcel_id|ppin|owner|market_value|delinquent_amount",
            "free_public_access": 1,
            "structured_parseable": 1,
            "ppin_present": 1,
            "parcel_overlap_signal": 0.55,
            "duplicate_source_identifier_risk": 0.6,
            "null_identifier_risk": 0.35,
            "source_freshness_score": 0.95,
            "actual_linkage_rate": float(sos.get("sos_linkage_rate", 0.0)),
            "expected_linkage_tier": "medium",
            "rank_reason": "Broad coverage and rich fields, but higher duplicate and county-format divergence risk than county-hosted direct sources.",
        },
    ]


def score_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["score"] = (
        frame["free_public_access"] * 15
        + frame["structured_parseable"] * 15
        + frame["ppin_present"] * 20
        + frame["parcel_overlap_signal"] * 25
        + (1 - frame["duplicate_source_identifier_risk"]) * 10
        + (1 - frame["null_identifier_risk"]) * 5
        + frame["source_freshness_score"] * 5
        + (frame["actual_linkage_rate"].fillna(0.0) / 100.0) * 5
    )
    frame = frame.sort_values(["score", "actual_linkage_rate"], ascending=[False, False]).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def build_source_scoring_summary(ranked: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "factor": "ppin_present",
                "importance_tier": "highest",
                "evidence": "Pike and Warren both include PPIN and both exceed 95% linkage.",
            },
            {
                "factor": "direct parcel-id overlap",
                "importance_tier": "highest",
                "evidence": "Warren and Madison both showed very high parcel-ID overlap before ingest and both linked cleanly.",
            },
            {
                "factor": "low duplicate source identifiers",
                "importance_tier": "high",
                "evidence": "Madison had zero duplicate parcel numbers and Warren ambiguity stayed low while Hinds and SOS were much noisier.",
            },
            {
                "factor": "county-hosted direct source",
                "importance_tier": "high",
                "evidence": "County-owned CSV/XLSX/HTML sources have been more predictable than broad statewide or portal-derived inventories.",
            },
            {
                "factor": "freshness",
                "importance_tier": "medium",
                "evidence": "Fresh sources are useful, but freshness alone did not predict linkage without aligned identifiers.",
            },
            {
                "factor": "structured format alone",
                "importance_tier": "low",
                "evidence": "Hinds was structured XLSX but still failed linkage because identifiers did not align.",
            },
        ]
    )


def main() -> None:
    rows = build_candidate_rows()
    ranked = score_candidates(pd.DataFrame(rows))
    ranked_path = TAX_METADATA_DIR / "tax_free_candidate_ranking_ms.csv"
    scoring_path = TAX_METADATA_DIR / "tax_free_source_scoring_summary_ms.csv"
    ranked.to_csv(ranked_path, index=False)
    build_source_scoring_summary(ranked).to_csv(scoring_path, index=False)
    print(f"Ranking: {ranked_path.relative_to(BASE_DIR)}")
    print(f"Scoring summary: {scoring_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
