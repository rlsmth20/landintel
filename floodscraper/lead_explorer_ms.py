from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from delinquent_leads_exports_ms import (
    caution_flags,
    growth_pressure_reason,
    normalize_string,
    recommended_sort_reason,
    recommended_use_case,
    vacant_reason,
)


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"

SCORED_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_scored_statewide.parquet"
APP_READY_PATH = TAX_PUBLISHED_DIR / "app_ready_mississippi_leads.parquet"
SUMMARY_PATH = TAX_PUBLISHED_DIR / "mississippi_lead_explorer_summary.csv"
DEFAULT_VIEWS_PATH = TAX_PUBLISHED_DIR / "mississippi_lead_explorer_default_views.csv"
FIELD_READINESS_PATH = TAX_PUBLISHED_DIR / "mississippi_lead_explorer_field_readiness.csv"


def road_access_tier(distance_ft: pd.Series) -> pd.Series:
    values = pd.to_numeric(distance_ft, errors="coerce")
    tier = pd.Series(pd.NA, index=values.index, dtype="string")
    tier.loc[values.le(50)] = "direct"
    tier.loc[values.gt(50) & values.le(200)] = "near"
    tier.loc[values.gt(200) & values.le(500)] = "moderate"
    tier.loc[values.gt(500) & values.le(1000)] = "limited"
    tier.loc[values.gt(1000)] = "remote"
    return tier


def county_hosted_flag(source_type: pd.Series) -> pd.Series:
    return normalize_string(source_type).isin(["direct_download_page", "free_direct_download"]).astype("boolean")


def recommended_view_bucket(frame: pd.DataFrame) -> pd.Series:
    bucket = pd.Series("general_ranked", index=frame.index, dtype="string")
    safest = frame["county_hosted_flag"].fillna(False).astype(bool) & frame["high_confidence_link_flag"].fillna(False).astype(bool)
    larger = frame["parcel_vacant_flag"].fillna(False).astype(bool) & pd.to_numeric(frame["acreage"], errors="coerce").ge(5).fillna(False)
    vacant_buildable = (
        frame["parcel_vacant_flag"].fillna(False).astype(bool)
        & frame["wetland_flag"].fillna(False).eq(False)
        & pd.to_numeric(frame["buildability_score"], errors="coerce").ge(80).fillna(False)
    )
    growth_edge = normalize_string(frame["growth_pressure_bucket"]).isin(["moderate", "high"]) & frame["parcel_vacant_flag"].fillna(False).astype(bool)

    bucket.loc[safest] = "safest_outreach"
    bucket.loc[larger] = "larger_land_target"
    bucket.loc[vacant_buildable] = "vacant_buildable"
    bucket.loc[growth_edge] = "growth_edge_opportunity"
    return bucket


def top_score_driver(frame: pd.DataFrame) -> pd.Series:
    return normalize_string(frame["lead_score_driver_1"], frame.index)


def build_app_ready_frame(scored: pd.DataFrame) -> pd.DataFrame:
    frame = scored.copy()
    frame["road_access_tier"] = road_access_tier(frame["road_distance_ft"])
    frame["county_hosted_flag"] = county_hosted_flag(frame["best_source_type"])
    frame["recommended_sort_reason"] = recommended_sort_reason(frame)
    frame["top_score_driver"] = top_score_driver(frame)
    frame["caution_flags"] = caution_flags(frame)
    frame["vacant_reason"] = vacant_reason(frame)
    frame["growth_pressure_reason"] = growth_pressure_reason(frame)
    frame["recommended_use_case"] = recommended_use_case(frame)
    frame["recommended_view_bucket"] = recommended_view_bucket(frame)

    columns = [
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "county_fips",
        "state_code",
        "geometry",
        "acreage",
        "acreage_bucket",
        "land_use",
        "parcel_vacant_flag",
        "building_count",
        "building_area_total",
        "nearby_building_count_1km",
        "nearby_building_density",
        "growth_pressure_bucket",
        "road_distance_ft",
        "road_access_tier",
        "wetland_flag",
        "flood_risk_score",
        "buildability_score",
        "environment_score",
        "investment_score",
        "electric_provider_name",
        "owner_name",
        "owner_type",
        "corporate_owner_flag",
        "absentee_owner_flag",
        "out_of_state_owner_flag",
        "owner_parcel_count",
        "owner_total_acres",
        "mailer_target_score",
        "delinquent_amount",
        "delinquent_amount_bucket",
        "delinquent_flag",
        "forfeited_flag",
        "best_source_type",
        "best_source_name",
        "source_confidence_tier",
        "county_source_coverage_tier",
        "amount_trust_tier",
        "high_confidence_link_flag",
        "county_hosted_flag",
        "lead_score_total",
        "lead_score_tier",
        "lead_score_driver_1",
        "lead_score_driver_2",
        "lead_score_driver_3",
        "lead_score_explanation",
        "size_score",
        "access_score",
        "buildability_component",
        "environmental_component",
        "owner_targeting_component",
        "delinquency_component",
        "source_confidence_component",
        "vacant_land_component",
        "growth_pressure_component",
        "recommended_sort_reason",
        "top_score_driver",
        "caution_flags",
        "vacant_reason",
        "growth_pressure_reason",
        "recommended_use_case",
        "recommended_view_bucket",
    ]
    return frame.loc[:, columns].copy()


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"section": "statewide", "metric": "lead_count", "key": pd.NA, "value": int(len(frame))},
        {"section": "statewide", "metric": "average_lead_score", "key": pd.NA, "value": round(float(pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()), 4)},
        {"section": "statewide", "metric": "vacant_share_pct", "key": pd.NA, "value": round(float(frame["parcel_vacant_flag"].fillna(False).mean() * 100.0), 4)},
        {"section": "statewide", "metric": "county_hosted_share_pct", "key": pd.NA, "value": round(float(frame["county_hosted_flag"].fillna(False).mean() * 100.0), 4)},
        {"section": "statewide", "metric": "high_confidence_link_share_pct", "key": pd.NA, "value": round(float(frame["high_confidence_link_flag"].fillna(False).mean() * 100.0), 4)},
    ]

    for key, value in normalize_string(frame["road_access_tier"]).value_counts(dropna=False).items():
        rows.append({"section": "road_access_tier", "metric": "lead_count", "key": key, "value": int(value)})
    for key, value in normalize_string(frame["growth_pressure_bucket"]).value_counts(dropna=False).items():
        rows.append({"section": "growth_pressure_bucket", "metric": "lead_count", "key": key, "value": int(value)})
    for key, value in normalize_string(frame["recommended_view_bucket"]).value_counts(dropna=False).items():
        rows.append({"section": "recommended_view_bucket", "metric": "lead_count", "key": key, "value": int(value)})
    for key, value in normalize_string(frame["lead_score_tier"]).value_counts(dropna=False).items():
        rows.append({"section": "lead_score_tier", "metric": "lead_count", "key": key, "value": int(value)})
    for key, value in normalize_string(frame["county_name"]).value_counts(dropna=False).head(25).items():
        rows.append({"section": "top_counties", "metric": "lead_count", "key": key, "value": int(value)})
    return pd.DataFrame(rows)


def build_default_views(frame: pd.DataFrame) -> pd.DataFrame:
    views = [
        {
            "view_name": "safest_early_investor_use",
            "description": "High-confidence county-hosted vacant leads with conservative source-quality filters.",
            "filter_expression": "lead_score_tier in ('high','very_high') AND parcel_vacant_flag = true AND county_hosted_flag = true AND high_confidence_link_flag = true AND wetland_flag = false AND amount_trust_tier != 'not_trusted_for_prominent_display'",
        },
        {
            "view_name": "vacant_land_targeting",
            "description": "Vacant delinquent land with access and no wetland default applied.",
            "filter_expression": "lead_score_tier in ('high','very_high') AND parcel_vacant_flag = true AND wetland_flag = false AND road_access_tier in ('direct','near')",
        },
        {
            "view_name": "larger_acreage_land_targeting",
            "description": "Larger vacant land acquisition screen.",
            "filter_expression": "lead_score_tier in ('high','very_high') AND parcel_vacant_flag = true AND acreage >= 5 AND county_hosted_flag = true",
        },
        {
            "view_name": "growth_edge_targeting",
            "description": "Vacant or lightly improved leads near moderate or high nearby building density.",
            "filter_expression": "lead_score_tier in ('high','very_high') AND growth_pressure_bucket in ('moderate','high') AND road_access_tier in ('direct','near','moderate')",
        },
    ]
    rows: list[dict[str, Any]] = []
    for view in views:
        rows.append({**view, "metric": "row_count", "value": int(apply_view(frame, view["view_name"]).shape[0])})
        rows.append({**view, "metric": "average_lead_score", "value": round(float(pd.to_numeric(apply_view(frame, view["view_name"])["lead_score_total"], errors="coerce").mean()), 4) if not apply_view(frame, view["view_name"]).empty else 0.0})
    return pd.DataFrame(rows)


def apply_view(frame: pd.DataFrame, view_name: str) -> pd.DataFrame:
    high = normalize_string(frame["lead_score_tier"]).isin(["high", "very_high"])
    county_hosted = frame["county_hosted_flag"].fillna(False).astype(bool)
    vacant = frame["parcel_vacant_flag"].fillna(False).astype(bool)
    no_wetland = frame["wetland_flag"].fillna(False).eq(False)
    good_amount = normalize_string(frame["amount_trust_tier"]).ne("not_trusted_for_prominent_display")
    good_access = normalize_string(frame["road_access_tier"]).isin(["direct", "near"])
    larger = pd.to_numeric(frame["acreage"], errors="coerce").ge(5).fillna(False)
    growth = normalize_string(frame["growth_pressure_bucket"]).isin(["moderate", "high"])
    moderate_access = normalize_string(frame["road_access_tier"]).isin(["direct", "near", "moderate"])

    if view_name == "safest_early_investor_use":
        mask = high & vacant & county_hosted & frame["high_confidence_link_flag"].fillna(False).astype(bool) & no_wetland & good_amount
    elif view_name == "vacant_land_targeting":
        mask = high & vacant & no_wetland & good_access
    elif view_name == "larger_acreage_land_targeting":
        mask = high & vacant & larger & county_hosted
    elif view_name == "growth_edge_targeting":
        mask = high & growth & moderate_access
    else:
        mask = pd.Series(False, index=frame.index)
    return frame.loc[mask].copy()


def build_field_readiness() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_name": "road_distance_ft",
                "readiness": "production_ready",
                "ui_guidance": "Expose as numeric max-distance filter.",
                "notes": "Parcel-side numeric distance exists and is used in scoring.",
            },
            {
                "field_name": "road_access_tier",
                "readiness": "production_ready",
                "ui_guidance": "Expose as default access facet.",
                "notes": "Derived directly from road_distance_ft using deterministic buckets.",
            },
            {
                "field_name": "electric_provider_name",
                "readiness": "partial",
                "ui_guidance": "Expose as contextual label/filter only, not proximity.",
                "notes": "Provider-name presence does not imply numeric utility proximity.",
            },
            {
                "field_name": "electric_proximity_distance",
                "readiness": "hide_from_default_ui",
                "ui_guidance": "Do not expose.",
                "notes": "No numeric electric/power proximity distance field exists in the current lead dataset.",
            },
            {
                "field_name": "absentee_owner_flag",
                "readiness": "partial",
                "ui_guidance": "Expose as optional owner-targeting filter, not default safety gate.",
                "notes": "Useful but derived from mailing-vs-situs heuristics.",
            },
            {
                "field_name": "out_of_state_owner_flag",
                "readiness": "partial",
                "ui_guidance": "Expose as optional owner-targeting filter, not default safety gate.",
                "notes": "Useful but heuristic and secondary.",
            },
            {
                "field_name": "delinquent_amount",
                "readiness": "partial",
                "ui_guidance": "Show with amount_trust_tier badge and keep out of strict default sorting.",
                "notes": "Amount quality varies by county/source; trust tier should always travel with it.",
            },
            {
                "field_name": "amount_trust_tier",
                "readiness": "production_ready",
                "ui_guidance": "Expose as a filter and default guard when amount is displayed prominently.",
                "notes": "This is the correct control for amount-quality gating.",
            },
            {
                "field_name": "parcel_vacant_flag",
                "readiness": "production_ready",
                "ui_guidance": "Expose prominently in default presets.",
                "notes": "Backed by statewide Microsoft building footprint join.",
            },
            {
                "field_name": "growth_pressure_bucket",
                "readiness": "production_ready",
                "ui_guidance": "Expose as optional growth-oriented filter.",
                "notes": "Derived deterministically from nearby building density.",
            },
        ]
    )


def main() -> None:
    TAX_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    scored = pd.read_parquet(SCORED_PATH)
    app_ready = build_app_ready_frame(scored)
    app_ready.to_parquet(APP_READY_PATH, index=False)
    build_summary(app_ready).to_csv(SUMMARY_PATH, index=False)
    build_default_views(app_ready).to_csv(DEFAULT_VIEWS_PATH, index=False)
    build_field_readiness().to_csv(FIELD_READINESS_PATH, index=False)

    print(f"App-ready: {APP_READY_PATH.relative_to(BASE_DIR)}")
    print(f"Explorer summary: {SUMMARY_PATH.relative_to(BASE_DIR)}")
    print(f"Default views: {DEFAULT_VIEWS_PATH.relative_to(BASE_DIR)}")
    print(f"Field readiness: {FIELD_READINESS_PATH.relative_to(BASE_DIR)}")
    print(f"Rows: {len(app_ready):,}")


if __name__ == "__main__":
    main()
