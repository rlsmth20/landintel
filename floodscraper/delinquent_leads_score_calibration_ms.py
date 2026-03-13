from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"

SCORED_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_scored_statewide.parquet"
TOP_SAMPLES_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_top_ranked_samples.csv"
SCORE_SUMMARY_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_summary.csv"
DIAGNOSTICS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_diagnostics_statewide.csv"
AMOUNT_AUDIT_COUNTY_PATH = TAX_PUBLISHED_DIR / "delinquent_amount_audit_county_summary.csv"

REVIEW_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_calibration_review.csv"
DOMINANCE_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_component_dominance.csv"
SANITY_SLICES_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_sanity_slices.csv"
RECOMMENDATIONS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_adjustment_recommendations.csv"
BUILDING_COMPARISON_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_building_comparison.csv"

TARGET_COUNTIES = ["pike", "madison", "warren", "jasper", "washington"]
COMPONENT_COLUMNS = [
    "size_score",
    "access_score",
    "buildability_component",
    "environmental_component",
    "owner_targeting_component",
    "delinquency_component",
    "source_confidence_component",
    "vacant_land_component",
    "growth_pressure_component",
]
BASELINE_COMPARISON_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_tuning_comparison.csv"


def normalize_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def subset_stats(frame: pd.DataFrame, subset_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    rows.append({"subset_name": subset_name, "metric": "row_count", "value": int(len(frame))})
    rows.append({"subset_name": subset_name, "metric": "average_lead_score", "value": round(float(pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()), 4)})
    rows.append({"subset_name": subset_name, "metric": "median_lead_score", "value": round(float(pd.to_numeric(frame["lead_score_total"], errors="coerce").median()), 4)})
    rows.append({"subset_name": subset_name, "metric": "average_delinquent_amount", "value": round(float(pd.to_numeric(frame["delinquent_amount"], errors="coerce").mean()), 4) if pd.to_numeric(frame["delinquent_amount"], errors="coerce").notna().any() else 0.0})
    rows.append({"subset_name": subset_name, "metric": "high_or_very_high_pct", "value": round(float(normalize_string(frame["lead_score_tier"]).isin(["high", "very_high"]).mean() * 100.0), 4)})
    rows.append({"subset_name": subset_name, "metric": "low_coverage_pct", "value": round(float(normalize_string(frame["county_source_coverage_tier"]).eq("low").mean() * 100.0), 4)})
    rows.append({"subset_name": subset_name, "metric": "amount_not_trusted_pct", "value": round(float(normalize_string(frame["amount_trust_tier"]).eq("not_trusted_for_prominent_display").mean() * 100.0), 4)})
    rows.append({"subset_name": subset_name, "metric": "vacant_pct", "value": round(float(frame["parcel_vacant_flag"].fillna(False).mean() * 100.0), 4)})
    rows.append({"subset_name": subset_name, "metric": "county_hosted_high_confidence_pct", "value": round(float((normalize_string(frame["best_source_type"]).isin(["direct_download_page", "free_direct_download"]) & normalize_string(frame["source_confidence_tier"]).eq("high")).mean() * 100.0), 4)})
    rows.append({"subset_name": subset_name, "metric": "sos_only_pct", "value": round(float(normalize_string(frame["best_source_type"]).eq("statewide_public_inventory").mean() * 100.0), 4)})
    rows.append({"subset_name": subset_name, "metric": "high_density_pct", "value": round(float(pd.to_numeric(frame["nearby_building_density"], errors="coerce").ge(300).mean() * 100.0), 4)})
    rows.append({"subset_name": subset_name, "metric": "very_dense_pct", "value": round(float(pd.to_numeric(frame["nearby_building_density"], errors="coerce").ge(800).mean() * 100.0), 4)})
    top_driver = normalize_string(frame["lead_score_driver_1"]).value_counts(dropna=False)
    if not top_driver.empty:
        rows.append({"subset_name": subset_name, "metric": "dominant_top_driver", "value": top_driver.index[0]})
        rows.append({"subset_name": subset_name, "metric": "dominant_top_driver_share_pct", "value": round(float(top_driver.iloc[0] / len(frame) * 100.0), 4)})
    for component in COMPONENT_COLUMNS:
        rows.append({"subset_name": subset_name, "metric": f"avg_{component}", "value": round(float(pd.to_numeric(frame[component], errors="coerce").mean()), 4)})
    return rows


def build_review(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    statewide_top25 = scored.sort_values("lead_score_total", ascending=False).head(25)
    statewide_top100 = scored.sort_values("lead_score_total", ascending=False).head(100)
    rows.extend(subset_stats(statewide_top25, "statewide_top25"))
    rows.extend(subset_stats(statewide_top100, "statewide_top100"))
    for county in TARGET_COUNTIES:
        county_top25 = scored.loc[normalize_string(scored["county_name"]).eq(county)].sort_values("lead_score_total", ascending=False).head(25)
        rows.extend(subset_stats(county_top25, f"{county}_top25"))
    return pd.DataFrame(rows)


def build_component_dominance(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "statewide_all": scored,
        "statewide_top100": scored.sort_values("lead_score_total", ascending=False).head(100),
        "very_high_only": scored.loc[normalize_string(scored["lead_score_tier"]).eq("very_high")].copy(),
        "high_and_very_high": scored.loc[normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])].copy(),
    }
    for name, frame in subsets.items():
        if frame.empty:
            continue
        for driver, count in normalize_string(frame["lead_score_driver_1"]).value_counts(dropna=False).items():
            rows.append(
                {
                    "subset_name": name,
                    "dominance_type": "top_driver_1",
                    "component_name": driver,
                    "lead_count": int(count),
                    "lead_share_pct": round(float(count / len(frame) * 100.0), 4),
                }
            )
        component_means = {component: float(pd.to_numeric(frame[component], errors="coerce").mean()) for component in COMPONENT_COLUMNS}
        total_mean = sum(component_means.values()) or 1.0
        for component, mean_value in component_means.items():
            rows.append(
                {
                    "subset_name": name,
                    "dominance_type": "avg_component_level",
                    "component_name": component,
                    "lead_count": pd.NA,
                    "lead_share_pct": round(mean_value / total_mean * 100.0, 4),
                }
            )
    return pd.DataFrame(rows)


def build_sanity_slices(scored: pd.DataFrame) -> pd.DataFrame:
    high_plus = normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
    slices = {
        "high_score_poor_road_access": high_plus & pd.to_numeric(scored["road_distance_ft"], errors="coerce").gt(1000).fillna(False),
        "high_score_high_flood_or_wetland": high_plus & (pd.to_numeric(scored["flood_risk_score"], errors="coerce").ge(8).fillna(False) | scored["wetland_flag"].fillna(False).astype(bool)),
        "high_score_low_coverage": high_plus & normalize_string(scored["county_source_coverage_tier"]).eq("low"),
        "high_score_missing_amount": high_plus & scored["has_reported_delinquent_amount_flag"].fillna(False).eq(False),
        "high_score_weak_owner_targeting": high_plus & pd.to_numeric(scored["mailer_target_score"], errors="coerce").lt(50).fillna(False),
        "high_score_amount_not_trusted": high_plus & normalize_string(scored["amount_trust_tier"]).eq("not_trusted_for_prominent_display"),
        "high_score_vacant": high_plus & scored["parcel_vacant_flag"].fillna(False).astype(bool),
        "high_score_vacant_acreage_5plus": high_plus & scored["parcel_vacant_flag"].fillna(False).astype(bool) & pd.to_numeric(scored["acreage"], errors="coerce").ge(5).fillna(False),
        "high_score_vacant_no_wetland": high_plus & scored["parcel_vacant_flag"].fillna(False).astype(bool) & scored["wetland_flag"].fillna(False).eq(False),
        "high_score_improved_parcel": high_plus & pd.to_numeric(scored["building_count"], errors="coerce").gt(0).fillna(False),
        "high_score_high_nearby_density": high_plus & pd.to_numeric(scored["nearby_building_density"], errors="coerce").ge(300).fillna(False),
        "very_high_very_dense_urban": normalize_string(scored["lead_score_tier"]).eq("very_high") & pd.to_numeric(scored["nearby_building_density"], errors="coerce").ge(800).fillna(False),
    }
    rows: list[dict[str, Any]] = []
    for name, mask in slices.items():
        frame = scored.loc[mask].copy()
        rows.append(
            {
                "slice_name": name,
                "lead_count": int(len(frame)),
                "lead_share_pct": round(float(len(frame) / len(scored) * 100.0), 4) if len(scored) else 0.0,
                "average_lead_score": round(float(pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()), 4) if not frame.empty else 0.0,
                "dominant_county": normalize_string(frame["county_name"]).value_counts(dropna=False).index[0] if not frame.empty else pd.NA,
                "dominant_source_type": normalize_string(frame["best_source_type"]).value_counts(dropna=False).index[0] if not frame.empty else pd.NA,
                "dominant_amount_trust_tier": normalize_string(frame["amount_trust_tier"]).value_counts(dropna=False).index[0] if not frame.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_recommendations(scored: pd.DataFrame) -> pd.DataFrame:
    recommendations: list[dict[str, Any]] = []
    summary = pd.read_csv(SCORE_SUMMARY_PATH) if SCORE_SUMMARY_PATH.exists() else pd.DataFrame()
    diagnostics = pd.read_csv(DIAGNOSTICS_PATH) if DIAGNOSTICS_PATH.exists() else pd.DataFrame()
    county_audit = pd.read_csv(AMOUNT_AUDIT_COUNTY_PATH) if AMOUNT_AUDIT_COUNTY_PATH.exists() else pd.DataFrame()

    top_driver_share = normalize_string(scored["lead_score_driver_1"]).value_counts(normalize=True, dropna=False)
    buildability_share = float(top_driver_share.get("buildability", 0.0))
    owner_share = float(top_driver_share.get("owner_targeting", 0.0))
    vacant_share = float(top_driver_share.get("vacant_land", 0.0))
    growth_share = float(top_driver_share.get("growth_pressure", 0.0))
    low_cov_high_share = float((normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"]) & normalize_string(scored["county_source_coverage_tier"]).eq("low")).mean())
    dense_urban_very_high_share = float((normalize_string(scored["lead_score_tier"]).eq("very_high") & pd.to_numeric(scored["nearby_building_density"], errors="coerce").ge(800).fillna(False)).mean())
    warren = scored.loc[normalize_string(scored["county_name"]).eq("warren")].copy()
    warren_high_share = float(normalize_string(warren["lead_score_tier"]).isin(["high", "very_high"]).mean()) if not warren.empty else 0.0
    warren_amount_not_trusted_share = float(normalize_string(warren["amount_trust_tier"]).eq("not_trusted_for_prominent_display").mean()) if not warren.empty else 0.0

    recommendations.append(
        {
            "topic": "buildability_weight",
            "current_observation": f"buildability is top driver for {buildability_share * 100.0:.2f}% of scored leads",
            "recommendation": "consider_small_reduction" if buildability_share > 0.70 else "keep_as_is",
            "justification": "Buildability should remain important, but it should not crowd out newer parcel-development signals.",
        }
    )
    recommendations.append(
        {
            "topic": "owner_targeting_weight",
            "current_observation": f"owner_targeting is top driver for {owner_share * 100.0:.2f}% of scored leads",
            "recommendation": "keep_as_is",
            "justification": "Owner targeting is materially influencing top leads without overwhelming the score.",
        }
    )
    recommendations.append(
        {
            "topic": "vacant_land_weight",
            "current_observation": f"vacant_land is top driver for {vacant_share * 100.0:.2f}% of scored leads",
            "recommendation": "keep_as_is" if vacant_share < 0.10 else "consider_small_reduction",
            "justification": "Vacancy should influence rankings, but not overwhelm the existing parcel-quality and owner-targeting signals.",
        }
    )
    recommendations.append(
        {
            "topic": "growth_pressure_weight",
            "current_observation": f"growth_pressure is top driver for {growth_share * 100.0:.2f}% of scored leads; very_high very-dense share {dense_urban_very_high_share * 100.0:.2f}%",
            "recommendation": "keep_as_is" if growth_share < 0.08 and dense_urban_very_high_share < 0.03 else "consider_small_reduction",
            "justification": "Growth pressure should help surface expansion-edge parcels, but should not over-reward dense urban inventory.",
        }
    )
    recommendations.append(
        {
            "topic": "low_coverage_penalty",
            "current_observation": f"high/very_high leads from low-coverage counties are {low_cov_high_share * 100.0:.2f}% of all leads",
            "recommendation": "keep_as_is",
            "justification": "Low-coverage counties are not dominating the high end of the score distribution.",
        }
    )
    recommendations.append(
        {
            "topic": "warren_amount_penalty",
            "current_observation": f"Warren amount not trusted share {warren_amount_not_trusted_share * 100.0:.2f}% with only {warren_high_share * 100.0:.2f}% of leads in high/very_high tiers",
            "recommendation": "consider_small_relaxation" if warren_high_share < 0.05 else "keep_as_is",
            "justification": "Warren may be somewhat suppressed because amount trust is unavailable, but the current behavior is directionally reasonable and not catastrophic.",
        }
    )
    recommendations.append(
        {
            "topic": "amount_trust_gating",
            "current_observation": f"county amount audit rows: {len(county_audit)} flagged counties",
            "recommendation": "keep_as_is",
            "justification": "Amount trust gating appears to prevent sparse or unreliable amount data from dominating score outcomes.",
        }
    )
    return pd.DataFrame(recommendations)


def composition_string(frame: pd.DataFrame, column: str, limit: int = 5) -> str:
    counts = normalize_string(frame[column]).value_counts(dropna=False).head(limit)
    return "|".join(f"{key}:{int(value)}" for key, value in counts.items())


def baseline_lookup() -> dict[str, str]:
    if not BASELINE_COMPARISON_PATH.exists():
        return {}
    baseline = pd.read_csv(BASELINE_COMPARISON_PATH)
    return dict(zip(baseline["metric"].astype("string"), baseline["after"].astype("string")))


def review_metric(review: pd.DataFrame, subset_name: str, metric: str) -> str:
    match = review.loc[review["subset_name"].astype("string").eq(subset_name) & review["metric"].astype("string").eq(metric), "value"]
    if match.empty:
        return ""
    return str(match.iloc[0])


def build_building_comparison(scored: pd.DataFrame, review: pd.DataFrame, dominance: pd.DataFrame, sanity: pd.DataFrame) -> pd.DataFrame:
    baseline = baseline_lookup()
    top25 = scored.sort_values("lead_score_total", ascending=False).head(25)
    top100 = scored.sort_values("lead_score_total", ascending=False).head(100)
    top_driver_share = normalize_string(scored["lead_score_driver_1"]).value_counts(normalize=True, dropna=False)

    sanity_lookup = {str(row["slice_name"]): row for _, row in sanity.iterrows()}
    rows = [
        {"metric": "statewide_average_score", "before_non_building": baseline.get("statewide_average_score", pd.NA), "after_building": round(float(pd.to_numeric(scored["lead_score_total"], errors="coerce").mean()), 4)},
        {"metric": "statewide_median_score", "before_non_building": baseline.get("statewide_median_score", pd.NA), "after_building": round(float(pd.to_numeric(scored["lead_score_total"], errors="coerce").median()), 4)},
        {"metric": "tier_very_high", "before_non_building": baseline.get("tier_very_high", pd.NA), "after_building": int(normalize_string(scored["lead_score_tier"]).eq("very_high").sum())},
        {"metric": "tier_high", "before_non_building": baseline.get("tier_high", pd.NA), "after_building": int(normalize_string(scored["lead_score_tier"]).eq("high").sum())},
        {"metric": "tier_medium", "before_non_building": baseline.get("tier_medium", pd.NA), "after_building": int(normalize_string(scored["lead_score_tier"]).eq("medium").sum())},
        {"metric": "tier_low", "before_non_building": baseline.get("tier_low", pd.NA), "after_building": int(normalize_string(scored["lead_score_tier"]).eq("low").sum())},
        {"metric": "buildability_top_driver_share_pct", "before_non_building": baseline.get("buildability_top_driver_share_pct", pd.NA), "after_building": round(float(top_driver_share.get("buildability", 0.0) * 100.0), 4)},
        {"metric": "owner_targeting_top_driver_share_pct", "before_non_building": baseline.get("owner_targeting_top_driver_share_pct", pd.NA), "after_building": round(float(top_driver_share.get("owner_targeting", 0.0) * 100.0), 4)},
        {"metric": "vacant_land_top_driver_share_pct", "before_non_building": 0.0, "after_building": round(float(top_driver_share.get("vacant_land", 0.0) * 100.0), 4)},
        {"metric": "growth_pressure_top_driver_share_pct", "before_non_building": 0.0, "after_building": round(float(top_driver_share.get("growth_pressure", 0.0) * 100.0), 4)},
        {"metric": "warren_top25_average_score", "before_non_building": baseline.get("warren_top25_average_score", pd.NA), "after_building": review_metric(review, "warren_top25", "average_lead_score")},
        {"metric": "high_score_low_coverage_share_pct", "before_non_building": baseline.get("high_score_low_coverage_share_pct", pd.NA), "after_building": round(float((normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"]) & normalize_string(scored["county_source_coverage_tier"]).eq("low")).mean() * 100.0), 4)},
        {"metric": "top25_county_composition", "before_non_building": baseline.get("top25_county_composition", pd.NA), "after_building": composition_string(top25, "county_name")},
        {"metric": "top100_county_composition", "before_non_building": baseline.get("top100_county_composition", pd.NA), "after_building": composition_string(top100, "county_name")},
        {"metric": "top25_vacant_share_pct", "before_non_building": pd.NA, "after_building": round(float(top25["parcel_vacant_flag"].fillna(False).mean() * 100.0), 4)},
        {"metric": "top100_vacant_share_pct", "before_non_building": pd.NA, "after_building": round(float(top100["parcel_vacant_flag"].fillna(False).mean() * 100.0), 4)},
        {"metric": "top25_county_hosted_high_confidence_pct", "before_non_building": pd.NA, "after_building": review_metric(review, "statewide_top25", "county_hosted_high_confidence_pct")},
        {"metric": "top100_county_hosted_high_confidence_pct", "before_non_building": pd.NA, "after_building": review_metric(review, "statewide_top100", "county_hosted_high_confidence_pct")},
        {"metric": "top25_sos_only_pct", "before_non_building": pd.NA, "after_building": review_metric(review, "statewide_top25", "sos_only_pct")},
        {"metric": "top100_sos_only_pct", "before_non_building": pd.NA, "after_building": review_metric(review, "statewide_top100", "sos_only_pct")},
        {"metric": "high_score_vacant_share_pct", "before_non_building": pd.NA, "after_building": round(float(sanity_lookup.get("high_score_vacant", {}).get("lead_share_pct", 0.0)), 4)},
        {"metric": "high_score_vacant_acreage_5plus_share_pct", "before_non_building": pd.NA, "after_building": round(float(sanity_lookup.get("high_score_vacant_acreage_5plus", {}).get("lead_share_pct", 0.0)), 4)},
        {"metric": "high_score_high_nearby_density_share_pct", "before_non_building": pd.NA, "after_building": round(float(sanity_lookup.get("high_score_high_nearby_density", {}).get("lead_share_pct", 0.0)), 4)},
        {"metric": "very_high_very_dense_urban_share_pct", "before_non_building": pd.NA, "after_building": round(float(sanity_lookup.get("very_high_very_dense_urban", {}).get("lead_share_pct", 0.0)), 4)},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    TAX_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    scored = pd.read_parquet(SCORED_PATH)

    review = build_review(scored)
    dominance = build_component_dominance(scored)
    sanity = build_sanity_slices(scored)
    recommendations = build_recommendations(scored)
    comparison = build_building_comparison(scored, review, dominance, sanity)

    review.to_csv(REVIEW_PATH, index=False)
    dominance.to_csv(DOMINANCE_PATH, index=False)
    sanity.to_csv(SANITY_SLICES_PATH, index=False)
    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)
    comparison.to_csv(BUILDING_COMPARISON_PATH, index=False)

    print(f"Calibration review: {REVIEW_PATH.relative_to(BASE_DIR)}")
    print(f"Component dominance: {DOMINANCE_PATH.relative_to(BASE_DIR)}")
    print(f"Sanity slices: {SANITY_SLICES_PATH.relative_to(BASE_DIR)}")
    print(f"Adjustment recommendations: {RECOMMENDATIONS_PATH.relative_to(BASE_DIR)}")
    print(f"Building comparison: {BUILDING_COMPARISON_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
