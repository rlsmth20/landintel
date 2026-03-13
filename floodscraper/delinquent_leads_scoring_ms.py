from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"
BUILDINGS_PROCESSED_DIR = BASE_DIR / "data" / "buildings_processed"

LEADS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_statewide.parquet"
AMOUNT_AUDIT_COUNTY_PATH = TAX_PUBLISHED_DIR / "delinquent_amount_audit_county_summary.csv"
AMOUNT_AUDIT_SOURCE_PATH = TAX_PUBLISHED_DIR / "delinquent_amount_audit_source_summary.csv"
FIELD_COMPLETENESS_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_field_completeness_by_county.csv"
BUILDING_METRICS_PATH = BUILDINGS_PROCESSED_DIR / "parcel_building_metrics.parquet"

SCORED_STATEWIDE_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_scored_statewide.parquet"
SCORED_BY_COUNTY_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_scored_by_county.csv"
SCORE_SUMMARY_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_score_summary.csv"
TOP_SAMPLES_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_top_ranked_samples.csv"

WEIGHTS = {
    "size_score": 0.10,
    "access_score": 0.10,
    "buildability_component": 0.18,
    "environmental_component": 0.20,
    "owner_targeting_component": 0.22,
    "delinquency_component": 0.10,
    "source_confidence_component": 0.11,
    "vacant_land_component": 0.04,
    "growth_pressure_component": 0.03,
}


def normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def clamp(series: pd.Series, low: float = 0.0, high: float = 100.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=low, upper=high)


def load_building_metrics(parcel_ids: pd.Series) -> pd.DataFrame:
    defaults = pd.DataFrame({"parcel_row_id": parcel_ids.astype("string")})
    defaults["building_count"] = 0
    defaults["building_area_total"] = 0.0
    defaults["parcel_vacant_flag"] = pd.Series(True, index=defaults.index, dtype="boolean")
    defaults["nearby_building_count_1km"] = 0
    defaults["nearby_building_density"] = 0.0
    defaults["growth_pressure_bucket"] = "very_low"
    if not BUILDING_METRICS_PATH.exists():
        return defaults

    metrics = pd.read_parquet(BUILDING_METRICS_PATH)
    metrics["parcel_row_id"] = metrics["parcel_row_id"].astype("string")
    metrics = defaults.merge(metrics, on="parcel_row_id", how="left", suffixes=("_default", ""))
    for column, default_column in [
        ("building_count", "building_count_default"),
        ("building_area_total", "building_area_total_default"),
        ("parcel_vacant_flag", "parcel_vacant_flag_default"),
        ("nearby_building_count_1km", "nearby_building_count_1km_default"),
        ("nearby_building_density", "nearby_building_density_default"),
        ("growth_pressure_bucket", "growth_pressure_bucket_default"),
    ]:
        if column in metrics:
            metrics[column] = metrics[column].fillna(metrics[default_column])
        else:
            metrics[column] = metrics[default_column]
    return metrics.loc[:, ["parcel_row_id", "building_count", "building_area_total", "parcel_vacant_flag", "nearby_building_count_1km", "nearby_building_density", "growth_pressure_bucket"]]


def size_score_from_acreage(acres: pd.Series) -> pd.Series:
    values = pd.to_numeric(acres, errors="coerce").fillna(0.0)
    score = pd.Series(20.0, index=values.index)
    score.loc[values.ge(0.25) & values.lt(1)] = 30.0
    score.loc[values.ge(1) & values.lt(5)] = 55.0
    score.loc[values.ge(5) & values.lt(20)] = 75.0
    score.loc[values.ge(20) & values.lt(100)] = 90.0
    score.loc[values.ge(100)] = 100.0
    return score


def access_score_from_distance(distance_ft: pd.Series) -> pd.Series:
    values = pd.to_numeric(distance_ft, errors="coerce").fillna(99999.0)
    score = pd.Series(25.0, index=values.index)
    score.loc[values.le(50)] = 100.0
    score.loc[values.gt(50) & values.le(200)] = 90.0
    score.loc[values.gt(200) & values.le(500)] = 80.0
    score.loc[values.gt(500) & values.le(1000)] = 65.0
    score.loc[values.gt(1000) & values.le(2500)] = 45.0
    return score


def buildability_component(frame: pd.DataFrame) -> pd.Series:
    buildability = clamp(frame["buildability_score"])
    investment = clamp(frame["investment_score"])
    return clamp(buildability * 0.65 + investment * 0.35)


def environmental_component(frame: pd.DataFrame) -> pd.Series:
    environment = clamp(frame["environment_score"])
    inverse_flood = clamp(100.0 - pd.to_numeric(frame["flood_risk_score"], errors="coerce").fillna(10.0) * 10.0)
    wetland_bonus = pd.Series(100.0, index=frame.index)
    wetland_bonus.loc[frame["wetland_flag"].fillna(False).astype(bool)] = 25.0
    return clamp(environment * 0.50 + inverse_flood * 0.35 + wetland_bonus * 0.15)


def owner_targeting_component(frame: pd.DataFrame) -> pd.Series:
    mailer = clamp(frame["mailer_target_score"])
    absentee_bonus = frame["absentee_owner_flag"].fillna(False).astype(bool).map({True: 12.0, False: 0.0})
    out_state_bonus = frame["out_of_state_owner_flag"].fillna(False).astype(bool).map({True: 8.0, False: 0.0})
    corporate_bonus = frame["corporate_owner_flag"].fillna(False).astype(bool).map({True: 6.0, False: 0.0})
    return clamp(mailer * 0.75 + absentee_bonus + out_state_bonus + corporate_bonus)


def amount_band(values: pd.Series) -> pd.Series:
    amounts = pd.to_numeric(values, errors="coerce")
    score = pd.Series(0.0, index=amounts.index)
    score.loc[amounts.gt(0) & amounts.lt(250)] = 20.0
    score.loc[amounts.ge(250) & amounts.lt(500)] = 40.0
    score.loc[amounts.ge(500) & amounts.lt(1000)] = 60.0
    score.loc[amounts.ge(1000) & amounts.lt(2500)] = 80.0
    score.loc[amounts.ge(2500)] = 100.0
    return score


def derive_amount_trust(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    county_audit = pd.read_csv(AMOUNT_AUDIT_COUNTY_PATH) if AMOUNT_AUDIT_COUNTY_PATH.exists() else pd.DataFrame()
    source_audit = pd.read_csv(AMOUNT_AUDIT_SOURCE_PATH) if AMOUNT_AUDIT_SOURCE_PATH.exists() else pd.DataFrame()
    completeness = pd.read_csv(FIELD_COMPLETENESS_PATH) if FIELD_COMPLETENESS_PATH.exists() else pd.DataFrame()

    county_lookup: dict[str, str] = {}
    if not county_audit.empty:
        county_lookup = dict(zip(county_audit["county_name"].astype("string"), county_audit["recommendation"].astype("string")))

    source_lookup: dict[tuple[str, str], str] = {}
    if not source_audit.empty:
        source_lookup = {
            (str(county), str(source)): str(rec)
            for county, source, rec in zip(
                source_audit["county_name"].astype("string"),
                source_audit["source_name"].astype("string"),
                source_audit["recommendation"].astype("string"),
            )
        }

    completeness_lookup: dict[str, float] = {}
    if not completeness.empty:
        subset = completeness.loc[completeness["field_name"].astype("string").eq("delinquent_amount")].copy()
        completeness_lookup = dict(zip(subset["county_name"].astype("string"), pd.to_numeric(subset["completeness_pct"], errors="coerce").fillna(0.0)))

    trust_tier = pd.Series("trusted", index=frame.index, dtype="string")
    trust_multiplier = pd.Series(1.0, index=frame.index, dtype="float64")

    for idx, row in frame.loc[:, ["county_name", "best_source_name"]].iterrows():
        county = str(row["county_name"])
        source = str(row["best_source_name"])
        source_rec = source_lookup.get((county, source))
        county_rec = county_lookup.get(county)
        completeness_pct = completeness_lookup.get(county, 100.0)

        recommendation = source_rec or county_rec or "trusted"
        if recommendation == "trusted":
            if completeness_pct < 75:
                recommendation = "not_trusted_for_prominent_display"
            elif completeness_pct < 95:
                recommendation = "use_with_caution"

        trust_tier.at[idx] = recommendation
        if recommendation == "trusted":
            trust_multiplier.at[idx] = 1.0
        elif recommendation == "use_with_caution":
            trust_multiplier.at[idx] = 0.5
        else:
            trust_multiplier.at[idx] = 0.0

    return trust_tier, trust_multiplier


def delinquency_component(frame: pd.DataFrame, amount_trust_multiplier: pd.Series) -> pd.Series:
    has_amount = frame["has_reported_delinquent_amount_flag"].fillna(False).astype(bool)
    base = pd.Series(25.0, index=frame.index)
    base.loc[has_amount] = 40.0
    return clamp(base + amount_band(frame["delinquent_amount"]) * amount_trust_multiplier * 0.6)


def source_confidence_component(frame: pd.DataFrame, amount_trust_tier: pd.Series) -> pd.Series:
    tier = normalize_string(frame["source_confidence_tier"], frame.index).fillna("low")
    source_type = normalize_string(frame["best_source_type"], frame.index)
    coverage_tier = normalize_string(frame["county_source_coverage_tier"], frame.index)
    score = pd.Series(45.0, index=frame.index)
    score.loc[tier.eq("medium")] = 75.0
    score.loc[tier.eq("high")] = 100.0

    statewide_low_cov = source_type.eq("statewide_public_inventory") & coverage_tier.eq("low")
    score.loc[statewide_low_cov] = (score.loc[statewide_low_cov] - 10.0).clip(lower=0.0)
    score.loc[amount_trust_tier.eq("use_with_caution")] = (score.loc[amount_trust_tier.eq("use_with_caution")] - 5.0).clip(lower=0.0)
    score.loc[amount_trust_tier.eq("not_trusted_for_prominent_display")] = (score.loc[amount_trust_tier.eq("not_trusted_for_prominent_display")] - 20.0).clip(lower=0.0)
    county_hosted_not_trusted = amount_trust_tier.eq("not_trusted_for_prominent_display") & source_type.isin(["direct_download_page", "free_direct_download"]) & coverage_tier.eq("high")
    score.loc[county_hosted_not_trusted] = (score.loc[county_hosted_not_trusted] + 5.0).clip(upper=100.0)
    return score


def vacant_land_component(frame: pd.DataFrame) -> pd.Series:
    acres = pd.to_numeric(frame["acreage"], errors="coerce").fillna(0.0)
    building_count = pd.to_numeric(frame["building_count"], errors="coerce").fillna(0.0)
    building_area = pd.to_numeric(frame["building_area_total"], errors="coerce").fillna(0.0)
    per_acre = building_count / acres.clip(lower=0.25)
    area_per_acre = building_area / acres.clip(lower=0.25)

    score = pd.Series(15.0, index=frame.index)
    score.loc[building_count.eq(0)] = 100.0
    score.loc[(building_count.eq(1)) & acres.ge(5)] = 85.0
    score.loc[(building_count.eq(1)) & acres.ge(1) & acres.lt(5)] = 70.0
    score.loc[(building_count.le(2)) & acres.ge(10)] = 75.0
    score.loc[(per_acre.le(0.2)) & acres.ge(2)] = score.loc[(per_acre.le(0.2)) & acres.ge(2)].clip(lower=60.0)
    score.loc[(area_per_acre.le(1000)) & acres.ge(2) & building_count.gt(0)] = score.loc[(area_per_acre.le(1000)) & acres.ge(2) & building_count.gt(0)].clip(lower=55.0)
    score.loc[(building_count.ge(3)) | (per_acre.gt(1.0))] = 20.0
    return clamp(score)


def growth_pressure_component(frame: pd.DataFrame) -> pd.Series:
    density = pd.to_numeric(frame["nearby_building_density"], errors="coerce").fillna(0.0)
    score = pd.Series(20.0, index=frame.index)
    score.loc[density.ge(25) & density.lt(100)] = 45.0
    score.loc[density.ge(100) & density.lt(300)] = 70.0
    score.loc[density.ge(300) & density.lt(800)] = 90.0
    score.loc[density.ge(800) & density.lt(1500)] = 60.0
    score.loc[density.ge(1500)] = 50.0
    return clamp(score)


def score_tier(total: pd.Series) -> pd.Series:
    values = pd.to_numeric(total, errors="coerce").fillna(0.0)
    tier = pd.Series("low", index=values.index, dtype="string")
    tier.loc[values.ge(50) & values.lt(65)] = "medium"
    tier.loc[values.ge(65) & values.lt(80)] = "high"
    tier.loc[values.ge(80)] = "very_high"
    return tier


def top_driver_labels(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    driver_map = {
        "size_score": "size",
        "access_score": "road_access",
        "buildability_component": "buildability",
        "environmental_component": "environment",
        "owner_targeting_component": "owner_targeting",
        "delinquency_component": "delinquency_signal",
        "source_confidence_component": "source_confidence",
        "vacant_land_component": "vacant_land",
        "growth_pressure_component": "growth_pressure",
    }
    weighted = pd.DataFrame(index=frame.index)
    for column, weight in WEIGHTS.items():
        weighted[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0) * weight

    top1: list[str] = []
    top2: list[str] = []
    top3: list[str] = []
    explanation: list[str] = []
    for row in weighted.itertuples(index=False, name=None):
        row_series = pd.Series(row, index=weighted.columns).sort_values(ascending=False)
        labels = [driver_map[name] for name in row_series.index[:3]]
        top1.append(labels[0])
        top2.append(labels[1])
        top3.append(labels[2])
        explanation.append("|".join(labels))
    return (
        pd.Series(top1, index=frame.index, dtype="string"),
        pd.Series(top2, index=frame.index, dtype="string"),
        pd.Series(top3, index=frame.index, dtype="string"),
        pd.Series(explanation, index=frame.index, dtype="string"),
    )


def build_county_output(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["county_fips", "county_name"], dropna=False)
    county = grouped.agg(
        lead_count=("parcel_row_id", "size"),
        average_lead_score=("lead_score_total", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 2)),
        median_lead_score=("lead_score_total", lambda s: round(float(pd.to_numeric(s, errors="coerce").median()), 2)),
        very_high_count=("lead_score_tier", lambda s: int(normalize_string(s).eq("very_high").sum())),
        high_count=("lead_score_tier", lambda s: int(normalize_string(s).eq("high").sum())),
        medium_count=("lead_score_tier", lambda s: int(normalize_string(s).eq("medium").sum())),
        low_count=("lead_score_tier", lambda s: int(normalize_string(s).eq("low").sum())),
        average_amount=("delinquent_amount", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 2) if pd.to_numeric(s, errors="coerce").notna().any() else 0.0),
    ).reset_index()
    county["dominant_score_driver"] = grouped["lead_score_driver_1"].agg(lambda s: normalize_string(s).value_counts(dropna=False).index[0] if len(s) else pd.NA).to_numpy()
    county["amount_trust_tier_mode"] = grouped["amount_trust_tier"].agg(lambda s: normalize_string(s).value_counts(dropna=False).index[0] if len(s) else pd.NA).to_numpy()
    county["source_coverage_mode"] = grouped["county_source_coverage_tier"].agg(lambda s: normalize_string(s).value_counts(dropna=False).index[0] if len(s) else pd.NA).to_numpy()
    return county.sort_values(["average_lead_score", "lead_count"], ascending=[False, False]).reset_index(drop=True)


def build_summary(frame: pd.DataFrame, county_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(
        [
            {"section": "statewide", "metric": "lead_count", "key": pd.NA, "value": int(len(frame))},
            {"section": "statewide", "metric": "average_lead_score", "key": pd.NA, "value": round(float(pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()), 4)},
            {"section": "statewide", "metric": "median_lead_score", "key": pd.NA, "value": round(float(pd.to_numeric(frame["lead_score_total"], errors="coerce").median()), 4)},
        ]
    )
    for tier, count in normalize_string(frame["lead_score_tier"]).value_counts(dropna=False).items():
        rows.append({"section": "count_by_score_tier", "metric": "lead_count", "key": tier, "value": int(count)})
    for driver, count in normalize_string(frame["lead_score_driver_1"]).value_counts(dropna=False).items():
        rows.append({"section": "count_by_top_driver", "metric": "lead_count", "key": driver, "value": int(count)})
    for trust, count in normalize_string(frame["amount_trust_tier"]).value_counts(dropna=False).items():
        rows.append({"section": "count_by_amount_trust_tier", "metric": "lead_count", "key": trust, "value": int(count)})
    artifact_mask = normalize_string(frame["county_source_coverage_tier"]).eq("low") & normalize_string(frame["lead_score_tier"]).eq("very_high")
    rows.append({"section": "coverage_artifact_check", "metric": "very_high_low_coverage_count", "key": pd.NA, "value": int(artifact_mask.sum())})
    rows.append({"section": "coverage_artifact_check", "metric": "very_high_low_coverage_pct", "key": pd.NA, "value": round(float(artifact_mask.mean() * 100.0), 4)})
    for _, row in county_frame.head(25).iterrows():
        rows.append({"section": "top_counties_by_average_score", "metric": "average_lead_score", "key": row["county_name"], "value": row["average_lead_score"]})
    return pd.DataFrame(rows)


def build_top_samples(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "parcel_row_id",
        "county_name",
        "parcel_id",
        "lead_score_total",
        "lead_score_tier",
        "lead_score_explanation",
        "lead_score_driver_1",
        "lead_score_driver_2",
        "lead_score_driver_3",
        "size_score",
        "access_score",
        "buildability_component",
        "environmental_component",
        "owner_targeting_component",
        "delinquency_component",
        "source_confidence_component",
        "vacant_land_component",
        "growth_pressure_component",
        "amount_trust_tier",
        "source_confidence_tier",
        "best_source_type",
        "delinquent_amount",
        "acreage",
        "owner_name",
        "building_count",
        "building_area_total",
        "parcel_vacant_flag",
        "nearby_building_density",
    ]
    return frame.sort_values(["lead_score_total", "investment_score", "acreage"], ascending=[False, False, False]).loc[:, columns].head(250).reset_index(drop=True)


def main() -> None:
    TAX_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(LEADS_PATH)
    building_metrics = load_building_metrics(frame["parcel_row_id"])
    frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
    frame = frame.merge(building_metrics, on="parcel_row_id", how="left")

    frame["amount_trust_tier"], amount_trust_multiplier = derive_amount_trust(frame)
    frame["size_score"] = size_score_from_acreage(frame["acreage"])
    frame["access_score"] = access_score_from_distance(frame["road_distance_ft"])
    frame["buildability_component"] = buildability_component(frame)
    frame["environmental_component"] = environmental_component(frame)
    frame["owner_targeting_component"] = owner_targeting_component(frame)
    frame["delinquency_component"] = delinquency_component(frame, amount_trust_multiplier)
    frame["source_confidence_component"] = source_confidence_component(frame, frame["amount_trust_tier"])
    frame["vacant_land_component"] = vacant_land_component(frame)
    frame["growth_pressure_component"] = growth_pressure_component(frame)

    weighted_total = pd.Series(0.0, index=frame.index)
    for column, weight in WEIGHTS.items():
        weighted_total = weighted_total + pd.to_numeric(frame[column], errors="coerce").fillna(0.0) * weight
    frame["lead_score_total"] = weighted_total.round(2)
    frame["lead_score_tier"] = score_tier(frame["lead_score_total"])

    driver1, driver2, driver3, explanation = top_driver_labels(frame)
    frame["lead_score_driver_1"] = driver1
    frame["lead_score_driver_2"] = driver2
    frame["lead_score_driver_3"] = driver3
    frame["lead_score_explanation"] = explanation

    county_frame = build_county_output(frame)
    summary_frame = build_summary(frame, county_frame)
    samples_frame = build_top_samples(frame)

    frame.to_parquet(SCORED_STATEWIDE_PATH, index=False)
    county_frame.to_csv(SCORED_BY_COUNTY_PATH, index=False)
    summary_frame.to_csv(SCORE_SUMMARY_PATH, index=False)
    samples_frame.to_csv(TOP_SAMPLES_PATH, index=False)

    print(f"Statewide scored leads: {SCORED_STATEWIDE_PATH.relative_to(BASE_DIR)}")
    print(f"County scores: {SCORED_BY_COUNTY_PATH.relative_to(BASE_DIR)}")
    print(f"Score summary: {SCORE_SUMMARY_PATH.relative_to(BASE_DIR)}")
    print(f"Top samples: {TOP_SAMPLES_PATH.relative_to(BASE_DIR)}")
    print(f"Scored leads: {len(frame):,}")


if __name__ == "__main__":
    main()
