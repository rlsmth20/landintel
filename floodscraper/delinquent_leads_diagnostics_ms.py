from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"

LEADS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_statewide.parquet"
DIAGNOSTICS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_diagnostics_statewide.csv"
COMPLETENESS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_field_completeness_by_county.csv"
OUTLIERS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_amount_outliers.csv"
PRIORITY_SLICES_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_priority_slices_by_county.csv"
READINESS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_field_readiness_summary.csv"

KEY_FIELDS = [
    "acreage",
    "owner_name",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "corporate_owner_flag",
    "delinquent_amount",
    "flood_risk_score",
    "buildability_score",
    "environment_score",
    "investment_score",
    "wetland_flag",
    "road_distance_ft",
    "electric_provider_name",
    "broadband_available",
    "distressed_owner_flag",
    "tax_distress_score",
    "mailer_target_score",
]

READINESS_ORDER = {"strong": 0, "partial": 1, "weak": 2}

BOOLEAN_FIELDS = {
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "corporate_owner_flag",
    "wetland_flag",
    "broadband_available",
    "distressed_owner_flag",
}

NUMERIC_FIELDS = {
    "acreage",
    "delinquent_amount",
    "flood_risk_score",
    "buildability_score",
    "environment_score",
    "investment_score",
    "road_distance_ft",
    "tax_distress_score",
    "mailer_target_score",
}


def normalize_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def completeness_rate(series: pd.Series) -> float:
    if series.name in BOOLEAN_FIELDS:
        return float(series.notna().mean())
    if series.name in NUMERIC_FIELDS:
        return float(pd.to_numeric(series, errors="coerce").notna().mean())
    return float(normalize_string(series).notna().mean())


def non_null_series(series: pd.Series) -> pd.Series:
    if series.name in BOOLEAN_FIELDS:
        return series.dropna()
    if series.name in NUMERIC_FIELDS:
        return pd.to_numeric(series, errors="coerce").dropna()
    return normalize_string(series).dropna()


def downgrade_readiness(current: str, target: str) -> str:
    return target if READINESS_ORDER[target] > READINESS_ORDER[current] else current


def build_statewide_diagnostics(leads: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            {"section": "statewide", "metric": "lead_rows", "key": pd.NA, "value": int(len(leads))},
            {"section": "statewide", "metric": "county_count", "key": pd.NA, "value": int(normalize_string(leads["county_name"]).nunique())},
            {"section": "statewide", "metric": "source_type_count", "key": pd.NA, "value": int(normalize_string(leads["best_source_type"]).nunique())},
            {"section": "statewide", "metric": "match_method_count", "key": pd.NA, "value": int(normalize_string(leads["best_match_method"]).nunique())},
        ]
    )
    for county, count in normalize_string(leads["county_name"]).value_counts(dropna=False).items():
        rows.append({"section": "count_by_county", "metric": "lead_rows", "key": county, "value": int(count)})
    for source_type, count in normalize_string(leads["best_source_type"]).fillna("unknown").value_counts(dropna=False).items():
        rows.append({"section": "count_by_best_source_type", "metric": "lead_rows", "key": source_type, "value": int(count)})
    for match_method, count in normalize_string(leads["best_match_method"]).fillna("unknown").value_counts(dropna=False).items():
        rows.append({"section": "count_by_best_match_method", "metric": "lead_rows", "key": match_method, "value": int(count)})
    for field in KEY_FIELDS:
        rows.append(
            {
                "section": "field_completeness_statewide",
                "metric": field,
                "key": "completeness_pct",
                "value": round(completeness_rate(leads[field]) * 100.0, 4),
            }
        )
    return pd.DataFrame(rows)


def build_field_completeness_by_county(leads: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (county_fips, county_name), group in leads.groupby(["county_fips", "county_name"], dropna=False):
        for field in KEY_FIELDS:
            rate = completeness_rate(group[field])
            series = non_null_series(group[field])
            row: dict[str, Any] = {
                "county_fips": county_fips,
                "county_name": county_name,
                "field_name": field,
                "row_count": int(len(group)),
                "non_null_count": int(round(rate * len(group))),
                "completeness_pct": round(rate * 100.0, 4),
                "null_pct": round((1.0 - rate) * 100.0, 4),
            }
            if field in BOOLEAN_FIELDS:
                row["true_count"] = int(group[field].fillna(False).astype(bool).sum())
                row["true_pct"] = round(float(group[field].fillna(False).astype(bool).mean() * 100.0), 4)
            else:
                row["true_count"] = pd.NA
                row["true_pct"] = pd.NA
            if field in NUMERIC_FIELDS and not series.empty:
                row["median_value"] = round(float(series.median()), 4)
                row["max_value"] = round(float(series.max()), 4)
            else:
                row["median_value"] = pd.NA
                row["max_value"] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["county_name", "field_name"]).reset_index(drop=True)


def build_amount_outliers(leads: pd.DataFrame) -> pd.DataFrame:
    amounts = pd.to_numeric(leads["delinquent_amount"], errors="coerce")
    working = leads.copy()
    working["delinquent_amount"] = amounts
    working = working.loc[working["delinquent_amount"].notna()].copy()

    county_rows: list[dict[str, Any]] = []
    county_amounts = working.groupby(["county_fips", "county_name"], dropna=False)["delinquent_amount"]
    for (county_fips, county_name), series in county_amounts:
        total = float(series.sum())
        sorted_series = series.sort_values(ascending=False)
        top1 = float(sorted_series.head(1).sum())
        top5 = float(sorted_series.head(5).sum())
        top10 = float(sorted_series.head(10).sum())
        median = float(series.median()) if len(series) else 0.0
        max_amount = float(series.max()) if len(series) else 0.0
        repeated_top_count = int((series == max_amount).sum()) if len(series) else 0
        flags: list[str] = []
        if top10 / total >= 0.4 if total else False:
            flags.append("county_total_dominated_by_top10")
        if top1 / total >= 0.1 if total else False:
            flags.append("county_total_dominated_by_top1")
        if median > 0 and max_amount / median >= 100:
            flags.append("max_to_median_gt_100x")
        if max_amount >= 100000:
            flags.append("max_amount_ge_100k")
        if repeated_top_count >= 3:
            flags.append("repeated_extreme_max_amount")
        county_rows.append(
            {
                "outlier_scope": "county",
                "county_fips": county_fips,
                "county_name": county_name,
                "parcel_row_id": pd.NA,
                "best_source_type": pd.NA,
                "best_source_name": pd.NA,
                "best_source_record_id": pd.NA,
                "delinquent_amount": pd.NA,
                "county_total_delinquent_amount": round(total, 2),
                "county_row_count": int(len(series)),
                "county_top1_share": round(top1 / total, 6) if total else pd.NA,
                "county_top5_share": round(top5 / total, 6) if total else pd.NA,
                "county_top10_share": round(top10 / total, 6) if total else pd.NA,
                "county_median_amount": round(median, 2),
                "county_max_amount": round(max_amount, 2),
                "repeated_extreme_count": repeated_top_count,
                "anomaly_flag": "|".join(flags) if flags else pd.NA,
            }
        )

    parcel_rows = working.sort_values(["delinquent_amount", "county_name"], ascending=[False, True]).head(100).copy()
    county_totals = county_amounts.sum().rename("county_total_delinquent_amount").reset_index()
    parcel_rows = parcel_rows.merge(county_totals, on=["county_fips", "county_name"], how="left")
    max_amount_by_county = county_amounts.max().rename("county_max_amount").reset_index()
    parcel_rows = parcel_rows.merge(max_amount_by_county, on=["county_fips", "county_name"], how="left")
    parcel_rows["county_amount_share"] = parcel_rows["delinquent_amount"] / parcel_rows["county_total_delinquent_amount"]

    repeated_counts = (
        parcel_rows.groupby(["county_name", "delinquent_amount"], dropna=False)
        .size()
        .rename("same_amount_count_in_top100")
        .reset_index()
    )
    parcel_rows = parcel_rows.merge(repeated_counts, on=["county_name", "delinquent_amount"], how="left")

    parcel_outliers: list[dict[str, Any]] = []
    for row in parcel_rows.itertuples(index=False):
        flags: list[str] = []
        if float(row.delinquent_amount) >= 100000:
            flags.append("amount_ge_100k")
        if float(row.county_amount_share) >= 0.1:
            flags.append("parcel_dominates_county_total")
        if int(row.same_amount_count_in_top100) >= 3 and float(row.delinquent_amount) == float(row.county_max_amount):
            flags.append("repeated_extreme_max_amount")
        parcel_outliers.append(
            {
                "outlier_scope": "parcel",
                "county_fips": row.county_fips,
                "county_name": row.county_name,
                "parcel_row_id": row.parcel_row_id,
                "best_source_type": row.best_source_type,
                "best_source_name": row.best_source_name,
                "best_source_record_id": row.best_source_record_id,
                "delinquent_amount": round(float(row.delinquent_amount), 2),
                "county_total_delinquent_amount": round(float(row.county_total_delinquent_amount), 2),
                "county_row_count": pd.NA,
                "county_top1_share": pd.NA,
                "county_top5_share": pd.NA,
                "county_top10_share": pd.NA,
                "county_median_amount": pd.NA,
                "county_max_amount": round(float(row.county_max_amount), 2),
                "repeated_extreme_count": int(row.same_amount_count_in_top100),
                "anomaly_flag": "|".join(flags) if flags else pd.NA,
            }
        )

    outliers = pd.concat([pd.DataFrame(county_rows), pd.DataFrame(parcel_outliers)], ignore_index=True)
    county_only = outliers.loc[outliers["outlier_scope"].eq("county")].sort_values(["county_total_delinquent_amount"], ascending=False)
    parcel_only = outliers.loc[outliers["outlier_scope"].eq("parcel")].sort_values(["delinquent_amount"], ascending=False)
    return pd.concat([county_only, parcel_only], ignore_index=True)


def build_priority_slices(leads: pd.DataFrame) -> pd.DataFrame:
    slices = {
        "absentee_owner": leads["absentee_owner_flag"].fillna(False).astype(bool),
        "out_of_state_owner": leads["out_of_state_owner_flag"].fillna(False).astype(bool),
        "corporate_owner": leads["corporate_owner_flag"].fillna(False).astype(bool),
        "acreage_ge_5": pd.to_numeric(leads["acreage"], errors="coerce").ge(5).fillna(False),
        "no_wetland_flag": leads["wetland_flag"].fillna(False).eq(False),
        "buildability_score_ge_80": pd.to_numeric(leads["buildability_score"], errors="coerce").ge(80).fillna(False),
        "reported_amount_present": leads["has_reported_delinquent_amount_flag"].fillna(False).astype(bool),
    }
    rows: list[dict[str, Any]] = []
    grouped = leads.groupby(["county_fips", "county_name"], dropna=False)
    for (county_fips, county_name), group in grouped:
        total_rows = int(len(group))
        total_amount = float(pd.to_numeric(group["delinquent_amount"], errors="coerce").fillna(0.0).sum())
        for slice_name, mask in slices.items():
            selected = group.loc[mask.loc[group.index]].copy()
            selected_amount = float(pd.to_numeric(selected["delinquent_amount"], errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "county_fips": county_fips,
                    "county_name": county_name,
                    "slice_name": slice_name,
                    "lead_count": int(len(selected)),
                    "lead_share_pct": round((len(selected) / total_rows * 100.0), 4) if total_rows else 0.0,
                    "total_delinquent_amount": round(selected_amount, 2),
                    "amount_share_pct": round((selected_amount / total_amount * 100.0), 4) if total_amount else 0.0,
                    "high_confidence_link_count": int(selected["high_confidence_link_flag"].fillna(False).astype(bool).sum()),
                    "reported_amount_count": int(selected["has_reported_delinquent_amount_flag"].fillna(False).astype(bool).sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["slice_name", "lead_count", "total_delinquent_amount"], ascending=[True, False, False]).reset_index(drop=True)


def classify_field_readiness(leads: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in KEY_FIELDS:
        series = leads[field]
        completeness = completeness_rate(series)
        non_null = non_null_series(series)
        readiness = "strong"
        rationale: list[str] = []
        recommendation = "production_ready"
        if completeness < 0.25:
            readiness = "weak"
            rationale.append("very_low_completeness")
            recommendation = "hide_for_now"
        elif completeness < 0.9:
            readiness = "partial"
            rationale.append("partial_completeness")
            recommendation = "deprioritize_or_badge_as_partial"

        if field in BOOLEAN_FIELDS:
            true_rate = float(series.fillna(False).astype(bool).mean()) if len(series) else 0.0
            if true_rate == 0.0 or true_rate == 1.0:
                readiness = downgrade_readiness(readiness, "weak" if field in {"broadband_available", "distressed_owner_flag"} else "partial")
                rationale.append("no_signal_variation")
                if field in {"broadband_available", "distressed_owner_flag"}:
                    recommendation = "hide_for_now"
        elif field in NUMERIC_FIELDS and not non_null.empty:
            if float(non_null.nunique()) <= 1:
                readiness = "weak"
                rationale.append("no_numeric_variation")
                recommendation = "hide_for_now"

        if field == "delinquent_amount":
            jasper_total = float(pd.to_numeric(leads.loc[leads["county_name"].eq("jasper"), field], errors="coerce").fillna(0.0).sum())
            statewide_total = float(pd.to_numeric(leads[field], errors="coerce").fillna(0.0).sum())
            if statewide_total and jasper_total / statewide_total > 0.5:
                readiness = "partial"
                rationale.append("county_outlier_requires_review")
                recommendation = "show_with_source_context"

        rows.append(
            {
                "field_name": field,
                "completeness_pct": round(completeness * 100.0, 4),
                "non_null_count": int(round(completeness * len(leads))),
                "distinct_non_null_count": int(non_null.nunique()) if not non_null.empty else 0,
                "readiness_tier": readiness,
                "rationale": "|".join(dict.fromkeys(rationale)) if rationale else "complete_and_consistent",
                "recommended_action": recommendation,
            }
        )
    out = pd.DataFrame(rows)
    out["readiness_sort"] = out["readiness_tier"].map(READINESS_ORDER).fillna(9)
    out = out.sort_values(["readiness_sort", "field_name"]).drop(columns=["readiness_sort"]).reset_index(drop=True)
    return out


def main() -> None:
    TAX_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    leads = pd.read_parquet(LEADS_PATH)

    diagnostics = build_statewide_diagnostics(leads)
    completeness = build_field_completeness_by_county(leads)
    outliers = build_amount_outliers(leads)
    priority_slices = build_priority_slices(leads)
    readiness = classify_field_readiness(leads)

    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    completeness.to_csv(COMPLETENESS_PATH, index=False)
    outliers.to_csv(OUTLIERS_PATH, index=False)
    priority_slices.to_csv(PRIORITY_SLICES_PATH, index=False)
    readiness.to_csv(READINESS_PATH, index=False)

    print(f"Diagnostics: {DIAGNOSTICS_PATH.relative_to(BASE_DIR)}")
    print(f"Completeness: {COMPLETENESS_PATH.relative_to(BASE_DIR)}")
    print(f"Outliers: {OUTLIERS_PATH.relative_to(BASE_DIR)}")
    print(f"Priority slices: {PRIORITY_SLICES_PATH.relative_to(BASE_DIR)}")
    print(f"Field readiness: {READINESS_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
