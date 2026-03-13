from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
TAX_PUBLISHED_DIR = BASE_DIR / "data" / "tax_published" / "ms"

SCORED_PATH = TAX_PUBLISHED_DIR / "delinquent_leads_scored_statewide.parquet"
EXPORT_SUMMARY_PATH = TAX_PUBLISHED_DIR / "lead_export_summary.csv"

EXPORT_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "acreage",
    "owner_name",
    "delinquent_amount",
    "best_source_type",
    "source_confidence_tier",
    "amount_trust_tier",
    "parcel_vacant_flag",
    "building_count",
    "building_area_total",
    "nearby_building_count_1km",
    "nearby_building_density",
    "growth_pressure_bucket",
    "lead_score_total",
    "lead_score_tier",
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
]


def normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def caution_flags(frame: pd.DataFrame) -> pd.Series:
    flags: list[str] = []
    output: list[str | pd._libs.missing.NAType] = []
    for row in frame.itertuples(index=False):
        flags = []
        if str(row.amount_trust_tier) != "trusted":
            flags.append(f"amount_{row.amount_trust_tier}")
        if str(row.source_confidence_tier) == "low":
            flags.append("low_source_confidence")
        if str(row.county_source_coverage_tier) == "low":
            flags.append("low_county_coverage")
        if not bool(row.high_confidence_link_flag):
            flags.append("non_high_confidence_link")
        if not bool(row.has_reported_delinquent_amount_flag):
            flags.append("missing_reported_amount")
        if bool(row.wetland_flag):
            flags.append("wetland_flag")
        output.append("|".join(flags) if flags else pd.NA)
    return pd.Series(output, index=frame.index, dtype="string")


def vacant_reason(frame: pd.DataFrame) -> pd.Series:
    building_count = pd.to_numeric(frame["building_count"], errors="coerce").fillna(0)
    acreage = pd.to_numeric(frame["acreage"], errors="coerce").fillna(0.0)
    reasons = pd.Series(pd.NA, index=frame.index, dtype="string")
    reasons.loc[building_count.eq(0)] = "no_buildings_detected"
    sparse_mask = building_count.gt(0) & building_count.le(2) & acreage.ge(5)
    reasons.loc[sparse_mask] = "very_low_improvement_intensity"
    compact_mask = building_count.gt(0) & building_count.le(1) & acreage.ge(1) & acreage.lt(5)
    reasons.loc[compact_mask] = reasons.loc[compact_mask].fillna("lightly_improved")
    return reasons


def growth_pressure_reason(frame: pd.DataFrame) -> pd.Series:
    bucket = normalize_string(frame["growth_pressure_bucket"], frame.index).fillna("very_low")
    mapping = {
        "very_low": "limited_nearby_growth_signal",
        "low": "emerging_growth_signal",
        "moderate": "moderate_growth_pressure",
        "high": "strong_growth_pressure",
        "very_high": "urban_core_density_neutralized",
    }
    return bucket.map(mapping).astype("string")


def recommended_use_case(frame: pd.DataFrame) -> pd.Series:
    use_case = pd.Series("general_ranked_outreach", index=frame.index, dtype="string")
    vacant = frame["parcel_vacant_flag"].fillna(False).astype(bool)
    acreage = pd.to_numeric(frame["acreage"], errors="coerce").fillna(0.0)
    no_wetland = frame["wetland_flag"].fillna(False).eq(False)
    good_access = pd.to_numeric(frame["road_distance_ft"], errors="coerce").le(200).fillna(False)
    growth = normalize_string(frame["growth_pressure_bucket"], frame.index).isin(["moderate", "high", "very_high"])
    county_hosted = normalize_string(frame["best_source_type"], frame.index).isin(["direct_download_page", "free_direct_download"])

    use_case.loc[vacant & acreage.ge(5)] = "larger_vacant_land_screen"
    use_case.loc[vacant & no_wetland & good_access] = "vacant_land_acquisition"
    use_case.loc[growth & no_wetland] = "growth_corridor_screen"
    use_case.loc[county_hosted & vacant] = "high_confidence_vacant_outreach"
    return use_case


def recommended_sort_reason(frame: pd.DataFrame) -> pd.Series:
    reason = normalize_string(frame["lead_score_driver_1"], frame.index).fillna("score")
    trust = normalize_string(frame["amount_trust_tier"], frame.index)
    reason = reason.where(~trust.eq("not_trusted_for_prominent_display"), reason + "|amount_suppressed")
    return reason.astype("string")


def prepare_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    export = frame.copy()
    export["top_score_driver"] = normalize_string(export["lead_score_driver_1"], export.index)
    export["recommended_sort_reason"] = recommended_sort_reason(export)
    export["caution_flags"] = caution_flags(export)
    export["vacant_reason"] = vacant_reason(export)
    export["growth_pressure_reason"] = growth_pressure_reason(export)
    export["recommended_use_case"] = recommended_use_case(export)
    export = export.sort_values(["lead_score_total", "county_name", "parcel_row_id"], ascending=[False, True, True]).reset_index(drop=True)
    return export.loc[:, EXPORT_COLUMNS].copy()


def build_summary_row(export_name: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(
        [
            {"export_name": export_name, "section": "summary", "metric": "row_count", "key": pd.NA, "value": int(len(frame))},
            {"export_name": export_name, "section": "summary", "metric": "average_lead_score", "key": pd.NA, "value": round(float(pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()), 4) if len(frame) else 0.0},
            {"export_name": export_name, "section": "summary", "metric": "average_acreage", "key": pd.NA, "value": round(float(pd.to_numeric(frame["acreage"], errors="coerce").mean()), 4) if len(frame) else 0.0},
            {"export_name": export_name, "section": "summary", "metric": "vacant_share_pct", "key": pd.NA, "value": round(float(frame["parcel_vacant_flag"].fillna(False).mean() * 100.0), 4) if len(frame) else 0.0},
            {"export_name": export_name, "section": "summary", "metric": "county_hosted_share_pct", "key": pd.NA, "value": round(float(normalize_string(frame["best_source_type"]).isin(["direct_download_page", "free_direct_download"]).mean() * 100.0), 4) if len(frame) else 0.0},
            {"export_name": export_name, "section": "summary", "metric": "sos_only_share_pct", "key": pd.NA, "value": round(float(normalize_string(frame["best_source_type"]).eq("statewide_public_inventory").mean() * 100.0), 4) if len(frame) else 0.0},
        ]
    )
    for county, count in normalize_string(frame["county_name"]).value_counts(dropna=False).head(25).items():
        rows.append({"export_name": export_name, "section": "county_distribution", "metric": "row_count", "key": county, "value": int(count)})
    for source_type, count in normalize_string(frame["best_source_type"]).value_counts(dropna=False).items():
        rows.append({"export_name": export_name, "section": "source_type_distribution", "metric": "row_count", "key": source_type, "value": int(count)})
    for trust, count in normalize_string(frame["amount_trust_tier"]).value_counts(dropna=False).items():
        rows.append({"export_name": export_name, "section": "amount_trust_distribution", "metric": "row_count", "key": trust, "value": int(count)})
    return rows


def export_top_500_statewide(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.sort_values(["lead_score_total", "lead_score_tier"], ascending=[False, False]).head(500).copy()


def export_top_100_by_county(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.sort_values(["county_name", "lead_score_total"], ascending=[True, False]).groupby("county_name", group_keys=False).head(100).copy()


def export_absentee_high_score(scored: pd.DataFrame) -> pd.DataFrame:
    mask = normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"]) & scored["absentee_owner_flag"].fillna(False).astype(bool)
    return scored.loc[mask].copy()


def export_corporate_owner_high_score(scored: pd.DataFrame) -> pd.DataFrame:
    mask = normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"]) & scored["corporate_owner_flag"].fillna(False).astype(bool)
    return scored.loc[mask].copy()


def export_acreage_5plus_high_score(scored: pd.DataFrame) -> pd.DataFrame:
    mask = normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"]) & pd.to_numeric(scored["acreage"], errors="coerce").ge(5).fillna(False)
    return scored.loc[mask].copy()


def export_high_score_no_wetland_good_access(scored: pd.DataFrame) -> pd.DataFrame:
    mask = (
        normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
        & scored["wetland_flag"].fillna(False).eq(False)
        & pd.to_numeric(scored["road_distance_ft"], errors="coerce").le(200).fillna(False)
    )
    return scored.loc[mask].copy()


def export_county_hosted_high_confidence(scored: pd.DataFrame) -> pd.DataFrame:
    mask = (
        normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
        & normalize_string(scored["best_source_type"]).isin(["direct_download_page", "free_direct_download"])
        & normalize_string(scored["source_confidence_tier"]).eq("high")
        & scored["high_confidence_link_flag"].fillna(False).astype(bool)
    )
    return scored.loc[mask].copy()


def export_sos_only_high_score(scored: pd.DataFrame) -> pd.DataFrame:
    mask = normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"]) & normalize_string(scored["best_source_type"]).eq("statewide_public_inventory")
    return scored.loc[mask].copy()


def export_vacant_high_score(scored: pd.DataFrame) -> pd.DataFrame:
    mask = normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"]) & scored["parcel_vacant_flag"].fillna(False).astype(bool)
    return scored.loc[mask].copy()


def export_vacant_5plus_high_score(scored: pd.DataFrame) -> pd.DataFrame:
    mask = (
        normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
        & scored["parcel_vacant_flag"].fillna(False).astype(bool)
        & pd.to_numeric(scored["acreage"], errors="coerce").ge(5).fillna(False)
    )
    return scored.loc[mask].copy()


def export_vacant_high_buildability(scored: pd.DataFrame) -> pd.DataFrame:
    mask = (
        normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
        & scored["parcel_vacant_flag"].fillna(False).astype(bool)
        & pd.to_numeric(scored["buildability_score"], errors="coerce").ge(80).fillna(False)
    )
    return scored.loc[mask].copy()


def export_vacant_county_hosted_high_confidence(scored: pd.DataFrame) -> pd.DataFrame:
    mask = (
        normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
        & scored["parcel_vacant_flag"].fillna(False).astype(bool)
        & normalize_string(scored["best_source_type"]).isin(["direct_download_page", "free_direct_download"])
        & normalize_string(scored["source_confidence_tier"]).eq("high")
        & scored["high_confidence_link_flag"].fillna(False).astype(bool)
    )
    return scored.loc[mask].copy()


def export_growth_pressure_high_score(scored: pd.DataFrame) -> pd.DataFrame:
    mask = (
        normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
        & normalize_string(scored["growth_pressure_bucket"]).isin(["moderate", "high"])
    )
    return scored.loc[mask].copy()


def export_vacant_no_wetland_good_access(scored: pd.DataFrame) -> pd.DataFrame:
    mask = (
        normalize_string(scored["lead_score_tier"]).isin(["high", "very_high"])
        & scored["parcel_vacant_flag"].fillna(False).astype(bool)
        & scored["wetland_flag"].fillna(False).eq(False)
        & pd.to_numeric(scored["road_distance_ft"], errors="coerce").le(200).fillna(False)
    )
    return scored.loc[mask].copy()


def build_app_ready_frame(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "parcel_row_id",
        "parcel_id",
        "county_fips",
        "county_name",
        "geometry",
        "acreage",
        "acreage_bucket",
        "owner_name",
        "best_source_type",
        "source_confidence_tier",
        "county_source_coverage_tier",
        "amount_trust_tier",
        "corporate_owner_flag",
        "absentee_owner_flag",
        "out_of_state_owner_flag",
        "wetland_flag",
        "delinquent_amount",
        "has_reported_delinquent_amount_flag",
        "parcel_vacant_flag",
        "building_count",
        "building_area_total",
        "nearby_building_count_1km",
        "nearby_building_density",
        "growth_pressure_bucket",
        "lead_score_total",
        "lead_score_tier",
        "high_confidence_link_flag",
        "size_score",
        "access_score",
        "buildability_component",
        "environmental_component",
        "owner_targeting_component",
        "delinquency_component",
        "source_confidence_component",
        "vacant_land_component",
        "growth_pressure_component",
        "lead_score_driver_1",
        "lead_score_driver_2",
        "lead_score_driver_3",
        "lead_score_explanation",
    ]
    return scored.loc[:, columns].copy()


def main() -> None:
    TAX_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    scored = pd.read_parquet(SCORED_PATH)
    app_ready_path = TAX_PUBLISHED_DIR / "app_ready_mississippi_leads.parquet"

    export_specs: list[tuple[str, str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("top_500_statewide_leads", "top_500_statewide_leads.csv", export_top_500_statewide),
        ("top_100_leads_by_county", "top_100_leads_by_county.csv", export_top_100_by_county),
        ("absentee_high_score_leads", "absentee_high_score_leads.csv", export_absentee_high_score),
        ("corporate_owner_high_score_leads", "corporate_owner_high_score_leads.csv", export_corporate_owner_high_score),
        ("acreage_5plus_high_score_leads", "acreage_5plus_high_score_leads.csv", export_acreage_5plus_high_score),
        ("high_score_no_wetland_good_access_leads", "high_score_no_wetland_good_access_leads.csv", export_high_score_no_wetland_good_access),
        ("county_hosted_high_confidence_leads", "county_hosted_high_confidence_leads.csv", export_county_hosted_high_confidence),
        ("sos_only_high_score_leads", "sos_only_high_score_leads.csv", export_sos_only_high_score),
        ("vacant_high_score_leads", "vacant_high_score_leads.csv", export_vacant_high_score),
        ("vacant_5plus_acres_high_score_leads", "vacant_5plus_acres_high_score_leads.csv", export_vacant_5plus_high_score),
        ("vacant_high_buildability_leads", "vacant_high_buildability_leads.csv", export_vacant_high_buildability),
        ("vacant_county_hosted_high_confidence_leads", "vacant_county_hosted_high_confidence_leads.csv", export_vacant_county_hosted_high_confidence),
        ("growth_pressure_high_score_leads", "growth_pressure_high_score_leads.csv", export_growth_pressure_high_score),
        ("vacant_no_wetland_good_access_leads", "vacant_no_wetland_good_access_leads.csv", export_vacant_no_wetland_good_access),
    ]

    summary_rows: list[dict[str, object]] = []
    for export_name, file_name, builder in export_specs:
        raw_frame = builder(scored)
        prepared = prepare_export_frame(raw_frame)
        prepared.to_csv(TAX_PUBLISHED_DIR / file_name, index=False)
        summary_rows.extend(build_summary_row(export_name, raw_frame))
        print(f"{export_name}: {(TAX_PUBLISHED_DIR / file_name).relative_to(BASE_DIR)} ({len(prepared):,} rows)")

    pd.DataFrame(summary_rows).to_csv(EXPORT_SUMMARY_PATH, index=False)
    build_app_ready_frame(scored).to_parquet(app_ready_path, index=False)
    print(f"Summary: {EXPORT_SUMMARY_PATH.relative_to(BASE_DIR)}")
    print(f"App-ready: {app_ready_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
