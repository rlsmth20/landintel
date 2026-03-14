from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.mississippi_leads_service import _apply_tax_interpretation_fields  # noqa: E402


OUTPUT_PATH = ROOT / "data" / "parcels" / "mississippi_tax_interpretation_validation.json"


def main() -> None:
    scenarios = [
        {
            "scenario": "covered_delinquent_flagged",
            "county_tax_coverage_status": "available",
            "county_tax_source_configured_flag": True,
            "delinquent_flag": True,
            "expected_category": "delinquent_confirmed",
            "expected_actionability": "actionable",
        },
        {
            "scenario": "covered_not_flagged",
            "county_tax_coverage_status": "available",
            "county_tax_source_configured_flag": True,
            "delinquent_flag": False,
            "expected_category": "tax_current_or_not_flagged",
            "expected_actionability": "not_actionable",
        },
        {
            "scenario": "partial_matched_delinquent",
            "county_tax_coverage_status": "partial",
            "county_tax_source_configured_flag": True,
            "delinquent_flag": True,
            "expected_category": "delinquent_possible_partial",
            "expected_actionability": "caution",
        },
        {
            "scenario": "partial_no_match",
            "county_tax_coverage_status": "partial",
            "county_tax_source_configured_flag": True,
            "delinquent_flag": False,
            "expected_category": "county_coverage_partial",
            "expected_actionability": "caution",
        },
        {
            "scenario": "pending_county",
            "county_tax_coverage_status": "pending",
            "county_tax_source_configured_flag": True,
            "delinquent_flag": False,
            "expected_category": "county_source_pending",
            "expected_actionability": "unknown",
        },
        {
            "scenario": "unavailable_county",
            "county_tax_coverage_status": "unavailable",
            "county_tax_source_configured_flag": False,
            "delinquent_flag": False,
            "expected_category": "county_source_unavailable",
            "expected_actionability": "unknown",
        },
        {
            "scenario": "stale_county",
            "county_tax_coverage_status": "stale",
            "county_tax_source_configured_flag": True,
            "delinquent_flag": False,
            "expected_category": "county_data_stale",
            "expected_actionability": "caution",
        },
        {
            "scenario": "zero_match_review_county",
            "county_tax_coverage_status": "partial",
            "county_tax_source_configured_flag": True,
            "county_tax_quality_flag": "zero_match_review",
            "delinquent_flag": False,
            "expected_category": "review_needed",
            "expected_actionability": "caution",
        },
    ]

    frame = pd.DataFrame(scenarios)
    interpreted = _apply_tax_interpretation_fields(frame.copy())
    records: list[dict[str, object]] = []
    passed = True
    for _, row in interpreted.iterrows():
        category = str(row["parcel_tax_status_category"])
        actionability = str(row["parcel_tax_actionability"])
        scenario_ok = category == row["expected_category"] and actionability == row["expected_actionability"]
        passed = passed and scenario_ok
        records.append(
            {
                "scenario": row["scenario"],
                "expected_category": row["expected_category"],
                "actual_category": category,
                "expected_actionability": row["expected_actionability"],
                "actual_actionability": actionability,
                "actual_confidence": row["parcel_tax_status_confidence"],
                "actual_label": row["parcel_tax_status_label"],
                "actual_reason": row["parcel_tax_status_reason"],
                "passed": scenario_ok,
            }
        )

    payload = {"passed": passed, "scenario_count": len(records), "results": records}
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
