from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from shapely import wkb


ROOT = Path(__file__).resolve().parents[1]
PARCEL_MASTER_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
OWNER_LEADS_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_owner_leads.parquet"
BUILDING_METRICS_PATH = ROOT / "data" / "buildings_processed" / "parcel_building_metrics.parquet"
AI_BUILDING_PREDICTIONS_PATH = ROOT / "data" / "buildings_processed" / "ai_building_presence_predictions_ms.parquet"
LEAD_SIGNALS_PATH = ROOT / "data" / "tax_published" / "ms" / "app_ready_mississippi_leads.parquet"
OUTPUT_ROOT = ROOT / "backend" / "runtime" / "mississippi" / "parcel_index"
RUNTIME_ROOT = ROOT / "backend" / "runtime" / "mississippi"
GEOMETRY_OUTPUT_ROOT = RUNTIME_ROOT / "parcel_geometry_index"
DETAIL_METRICS_OUTPUT_PATH = RUNTIME_ROOT / "parcel_detail_metrics.parquet"
SQFT_PER_ACRE = 43560.0


PARCEL_COLUMNS = [
    "parcel_row_id",
    "parcel_id",
    "county_name",
    "county_fips",
    "state_code",
    "owner_name",
    "land_use_raw",
    "tax_acres",
    "gis_acres",
    "total_acres",
    "parcel_area_acres",
    "latitude",
    "longitude",
    "road_distance_ft",
    "road_access_tier",
    "wetland_flag",
    "wetland_overlap_acres",
    "wetland_overlap_pct",
    "flood_risk_score",
    "flood_zone_primary",
    "has_flood_overlap",
    "sfha_overlap",
    "mean_slope_pct",
    "max_slope_pct",
    "slope_class",
    "slope_score",
    "shape_length",
    "shape_area",
    "buildability_score",
    "environment_score",
    "investment_score",
    "electric_provider_name",
    "total_value",
    "geometry",
]

OWNER_COLUMNS = [
    "parcel_row_id",
    "owner_type",
    "absentee_owner_flag",
    "out_of_state_owner_flag",
    "owner_parcel_count",
    "owner_total_acres",
    "mailer_target_score",
    "corporate_owner_flag",
]

BUILDING_COLUMNS = [
    "parcel_row_id",
    "building_count",
    "building_area_total",
    "parcel_vacant_flag",
    "nearby_building_count_1km",
    "nearby_building_density",
    "growth_pressure_bucket",
]

SIGNAL_COLUMNS = [
    "parcel_row_id",
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


def coalesce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column not in frame.columns:
            continue
        result = result.fillna(pd.to_numeric(frame[column], errors="coerce"))
    return result


def point_geometry_from_wkb(value: bytes | None) -> dict[str, object] | None:
    if not value:
        return None
    try:
        shape = wkb.loads(value)
    except Exception:
        return None
    centroid = shape.centroid
    return {
        "type": "Point",
        "coordinates": [round(float(centroid.x), 6), round(float(centroid.y), 6)],
    }


def simplified_polygon_wkb(value: bytes | None, tolerance: float = 0.00002) -> bytes | None:
    if not value:
        return None
    try:
        shape = wkb.loads(value)
    except Exception:
        return None
    return shape.simplify(tolerance, preserve_topology=True).wkb


def json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def vacancy_confidence(frame: pd.DataFrame) -> pd.Series:
    building_count = pd.to_numeric(frame.get("building_count"), errors="coerce").fillna(0)
    building_area_total = pd.to_numeric(frame.get("building_area_total"), errors="coerce").fillna(0)
    acreage = pd.to_numeric(frame.get("acreage"), errors="coerce").fillna(0)
    assessed_total_value = pd.to_numeric(frame.get("assessed_total_value"), errors="coerce").fillna(0)
    road_distance_ft = pd.to_numeric(frame.get("road_distance_ft"), errors="coerce")
    nearby_building_density = pd.to_numeric(frame.get("nearby_building_density"), errors="coerce").fillna(0)
    parcel_vacant = frame["parcel_vacant_flag"].fillna(False)

    confidence = pd.Series(45.0, index=frame.index, dtype="float64")
    confidence = confidence.mask(parcel_vacant & building_count.eq(0) & building_area_total.le(0), 92.0)
    confidence = confidence.mask(parcel_vacant & building_count.le(1) & building_area_total.le(750) & acreage.ge(1), 78.0)
    confidence = confidence.mask(parcel_vacant & assessed_total_value.ge(25000), 18.0)
    confidence = confidence.mask(parcel_vacant & assessed_total_value.ge(10000) & road_distance_ft.le(150).fillna(False), 28.0)
    confidence = confidence.mask(parcel_vacant & nearby_building_density.ge(120) & acreage.le(2), 35.0)
    confidence = confidence.mask(~parcel_vacant & building_count.gt(0), 18.0)
    confidence = confidence.mask(~parcel_vacant & building_area_total.gt(1500), 8.0)
    return confidence.clip(0, 100)


def derive_shape_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    area_sqft = pd.to_numeric(frame.get("shape_area"), errors="coerce")
    perimeter_ft = pd.to_numeric(frame.get("shape_length"), errors="coerce")
    compactness = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid_shape = area_sqft.gt(0) & perimeter_ft.gt(0)
    compactness.loc[valid_shape] = ((4.0 * np.pi * area_sqft.loc[valid_shape]) / np.square(perimeter_ft.loc[valid_shape])).clip(0.0, 1.0)

    semi_perimeter = perimeter_ft / 2.0
    discriminant = np.square(semi_perimeter) - (4.0 * area_sqft)
    discriminant = discriminant.where(discriminant.ge(0))
    sqrt_disc = np.sqrt(discriminant)
    frontage = ((semi_perimeter + sqrt_disc) / 2.0).where(valid_shape)
    width = ((semi_perimeter - sqrt_disc) / 2.0).where(valid_shape)
    frame["shape_compactness"] = compactness
    frame["parcel_frontage_ft_estimate"] = frontage.where(frontage.gt(0))
    frame["parcel_width_ft_estimate"] = width.where(width.gt(0))
    return frame


def build_detail_metrics_runtime(frame: pd.DataFrame) -> pd.DataFrame:
    detail_columns = [
        "parcel_row_id",
        "mean_slope_pct",
        "max_slope_pct",
        "slope_class",
        "slope_score",
        "elevation_mean_ft",
        "shape_compactness",
        "parcel_frontage_ft_estimate",
        "parcel_width_ft_estimate",
        "wetland_pct",
        "wetland_area_sqft",
        "flood_pct",
        "flood_area_sqft",
        "primary_fema_zone",
        "wetland_flag",
        "flood_risk_score",
    ]
    available = [column for column in detail_columns if column in frame.columns]
    return frame.loc[:, available].copy()


def build_runtime_frame() -> pd.DataFrame:
    parcels = pd.read_parquet(PARCEL_MASTER_PATH, columns=PARCEL_COLUMNS, engine="pyarrow")
    parcels["acreage"] = coalesce_numeric(parcels, ["total_acres", "parcel_area_acres", "gis_acres", "tax_acres"])
    parcels["land_use"] = parcels["land_use_raw"].astype("string").str.strip()
    parcels["assessed_total_value"] = pd.to_numeric(parcels["total_value"], errors="coerce")
    parcels["wetland_pct"] = pd.to_numeric(parcels.get("wetland_overlap_pct"), errors="coerce")
    parcels["wetland_area_sqft"] = pd.to_numeric(parcels.get("wetland_overlap_acres"), errors="coerce") * SQFT_PER_ACRE
    parcels["flood_pct"] = pd.to_numeric(parcels.get("flood_overlap_pct"), errors="coerce")
    parcels["flood_area_sqft"] = pd.to_numeric(parcels.get("flood_overlap_acres"), errors="coerce") * SQFT_PER_ACRE
    parcels["primary_fema_zone"] = parcels.get("flood_zone_primary", pd.Series(pd.NA, index=parcels.index)).astype("string").str.strip()
    parcels["elevation_mean_ft"] = pd.to_numeric(parcels.get("elevation_mean_ft"), errors="coerce")
    parcels = derive_shape_metrics(parcels)
    parcels["geometry"] = parcels["geometry"].map(point_geometry_from_wkb)
    parcels = parcels.drop(
        columns=[
            "land_use_raw",
            "tax_acres",
            "gis_acres",
            "total_acres",
            "parcel_area_acres",
            "total_value",
            "wetland_overlap_acres",
            "wetland_overlap_pct",
            "flood_zone_primary",
            "shape_length",
            "shape_area",
        ],
        errors="ignore",
    )

    owners = pd.read_parquet(OWNER_LEADS_PATH, columns=OWNER_COLUMNS, engine="pyarrow")
    buildings = pd.read_parquet(BUILDING_METRICS_PATH, columns=BUILDING_COLUMNS, engine="pyarrow")
    signals = pd.read_parquet(LEAD_SIGNALS_PATH, columns=SIGNAL_COLUMNS, engine="pyarrow")

    frame = parcels.merge(owners, on="parcel_row_id", how="left")
    frame = frame.merge(buildings, on="parcel_row_id", how="left")
    frame = frame.merge(signals, on="parcel_row_id", how="left")
    if AI_BUILDING_PREDICTIONS_PATH.exists():
        ai_columns = [
            "parcel_row_id",
            "ai_building_present_probability",
            "ai_building_present_flag",
            "vacancy_confidence_score",
            "vacancy_model_version",
        ]
        ai_predictions = pd.read_parquet(AI_BUILDING_PREDICTIONS_PATH, columns=ai_columns, engine="pyarrow")
        frame = frame.merge(ai_predictions, on="parcel_row_id", how="left")

    for column in ["parcel_vacant_flag", "corporate_owner_flag", "absentee_owner_flag", "out_of_state_owner_flag", "high_confidence_link_flag", "county_hosted_flag", "delinquent_flag", "forfeited_flag"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna(False)

    frame["state_code"] = frame["state_code"].astype("string").fillna("MS")
    frame["county_name"] = frame["county_name"].astype("string")
    frame["parcel_id"] = frame["parcel_id"].astype("string")
    frame["owner_name"] = frame["owner_name"].astype("string")
    frame["owner_type"] = frame["owner_type"].astype("string").fillna("unknown")
    frame["best_source_type"] = frame["best_source_type"].astype("string").fillna("parcel_master")
    frame["best_source_name"] = frame["best_source_name"].astype("string").fillna("Mississippi Parcel Master")
    frame["source_confidence_tier"] = frame["source_confidence_tier"].astype("string").fillna("parcel_master_only")
    frame["county_source_coverage_tier"] = frame["county_source_coverage_tier"].astype("string").fillna("statewide_parcel_base")
    frame["amount_trust_tier"] = frame["amount_trust_tier"].astype("string").fillna("not_reported")
    frame["growth_pressure_bucket"] = frame["growth_pressure_bucket"].astype("string").fillna("unknown")
    frame["recommended_view_bucket"] = frame["recommended_view_bucket"].astype("string").fillna("general_ranked")
    frame["slope_class"] = frame["slope_class"].astype("string")
    frame["county_vacant_flag"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["ai_building_present_flag"] = (
        frame["ai_building_present_flag"].astype("boolean")
        if "ai_building_present_flag" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="boolean")
    )
    if "vacancy_confidence_score" in frame.columns:
        vacancy_confidence_series = pd.to_numeric(frame["vacancy_confidence_score"], errors="coerce")
    else:
        vacancy_confidence_series = pd.Series(np.nan, index=frame.index, dtype="float64")
    frame["vacancy_confidence_score"] = vacancy_confidence_series.fillna(vacancy_confidence(frame))

    return frame


def build_geometry_runtime(parcels: pd.DataFrame) -> pa.Table:
    geometry_frame = parcels.loc[:, ["parcel_row_id", "county_name", "geometry"]].copy()
    geometry_frame["geometry"] = geometry_frame["geometry"].map(simplified_polygon_wkb)
    return pa.Table.from_pandas(geometry_frame, preserve_index=False)


def build_summary_payload(frame: pd.DataFrame) -> dict[str, object]:
    county_counts = frame.groupby("county_name", dropna=True).size().sort_values(ascending=False).head(20)
    recommended_counts = frame.groupby("recommended_view_bucket", dropna=True).size().sort_values(ascending=False)
    average_score = pd.to_numeric(frame["lead_score_total"], errors="coerce").mean()
    vacant_count = int(frame["parcel_vacant_flag"].fillna(False).sum())
    return {
        "row_count": int(len(frame)),
        "source": "mississippi parcel runtime dataset",
        "geometry_mode": "viewport_geojson",
        "sections": {
            "statewide": [
                {"section": "statewide", "metric": "lead_count", "value": str(len(frame))},
                {"section": "statewide", "metric": "average_lead_score", "value": f"{average_score:.1f}"},
                {"section": "statewide", "metric": "likely_vacant_count", "value": str(vacant_count)},
                {"section": "statewide", "metric": "vacant_share_pct", "value": f"{frame['parcel_vacant_flag'].fillna(False).mean() * 100:.1f}"},
                {"section": "statewide", "metric": "county_hosted_share_pct", "value": f"{frame['county_hosted_flag'].fillna(False).mean() * 100:.1f}"},
            ],
            "top_counties": [
                {"section": "top_counties", "key": county, "metric": "parcel_count", "value": str(int(count))}
                for county, count in county_counts.items()
            ],
            "recommended_view_bucket": [
                {"section": "recommended_view_bucket", "key": bucket, "metric": "parcel_count", "value": str(int(count))}
                for bucket, count in recommended_counts.items()
            ],
        },
    }


def build_presets_payload(frame: pd.DataFrame) -> list[dict[str, str]]:
    definitions = {
        "safest_early_investor_use": {
            "description": "High-confidence county-hosted parcels with stronger amount reliability, no wetlands, and vacancy preference.",
            "filter_expression": "county_hosted_flag=true AND high_confidence_link_flag=true AND parcel_vacant_flag=true AND wetland_flag=false AND amount_trust_tier in trusted/use_with_caution",
            "mask": (
                frame["county_hosted_flag"].fillna(False)
                & frame["high_confidence_link_flag"].fillna(False)
                & frame["parcel_vacant_flag"].fillna(False)
                & ~frame["wetland_flag"].fillna(False)
                & frame["amount_trust_tier"].astype("string").isin(["trusted", "use_with_caution"])
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
        "vacant_land_targeting": {
            "description": "Vacant parcels with stronger road access and fewer wetland constraints.",
            "filter_expression": "parcel_vacant_flag=true AND wetland_flag=false AND road_access_tier in direct/near",
            "mask": (
                frame["parcel_vacant_flag"].fillna(False)
                & ~frame["wetland_flag"].fillna(False)
                & frame["road_access_tier"].astype("string").isin(["direct", "near"])
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
        "larger_acreage_land_targeting": {
            "description": "Vacant larger-acreage parcels for land assembly and development exploration.",
            "filter_expression": "parcel_vacant_flag=true AND county_hosted_flag=true AND acreage>=5",
            "mask": (
                frame["parcel_vacant_flag"].fillna(False)
                & frame["county_hosted_flag"].fillna(False)
                & pd.to_numeric(frame["acreage"], errors="coerce").ge(5).fillna(False)
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
        "growth_edge_targeting": {
            "description": "Parcels in moderate-to-high growth areas with usable road access.",
            "filter_expression": "growth_pressure_bucket in moderate/high AND road_access_tier in direct/near/moderate",
            "mask": (
                frame["growth_pressure_bucket"].astype("string").isin(["moderate", "high"])
                & frame["road_access_tier"].astype("string").isin(["direct", "near", "moderate"])
                & pd.to_numeric(frame["lead_score_total"], errors="coerce").ge(65).fillna(False)
            ),
        },
    }
    payload: list[dict[str, str]] = []
    for view_name, definition in definitions.items():
        filtered = frame.loc[definition["mask"]]
        average_score = pd.to_numeric(filtered["lead_score_total"], errors="coerce").mean() if len(filtered) else 0
        payload.append(
            {
                "view_name": view_name,
                "description": definition["description"],
                "filter_expression": definition["filter_expression"],
                "row_count": str(len(filtered)),
                "average_lead_score": f"{average_score:.1f}",
            }
        )
    return payload


def build_default_leads_payload(frame: pd.DataFrame) -> dict[str, object]:
    summary_fields = [
        "parcel_row_id",
        "parcel_id",
        "county_name",
        "acreage",
        "owner_name",
        "lead_score_total",
        "lead_score_tier",
        "parcel_vacant_flag",
        "road_access_tier",
        "growth_pressure_bucket",
        "best_source_type",
        "source_confidence_tier",
        "delinquent_amount",
        "amount_trust_tier",
        "recommended_sort_reason",
        "county_hosted_flag",
        "high_confidence_link_flag",
        "recommended_view_bucket",
    ]
    top = frame.sort_values("lead_score_total", ascending=False, na_position="last").head(250)
    return {
        "total_count": int(len(frame)),
        "limit": 200,
        "offset": 0,
        "items": [{key: json_scalar(value) for key, value in record.items()} for record in top.loc[:, summary_fields].to_dict(orient="records")],
    }


def build_default_geometry_payload(frame: pd.DataFrame) -> dict[str, object]:
    top = frame.sort_values(["county_name", "lead_score_total"], ascending=[True, False], na_position="last").groupby("county_name", dropna=True).head(3)
    features = []
    for _, row in top.iterrows():
        geometry = row.get("geometry")
        if not isinstance(geometry, dict):
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "parcel_row_id": row["parcel_row_id"],
                    "parcel_id": json_scalar(row.get("parcel_id")),
                    "county_name": json_scalar(row.get("county_name")),
                    "lead_score_total": json_scalar(row.get("lead_score_total")),
                    "lead_score_tier": json_scalar(row.get("lead_score_tier")),
                    "parcel_vacant_flag": json_scalar(row.get("parcel_vacant_flag")),
                    "wetland_flag": json_scalar(row.get("wetland_flag")),
                    "flood_risk_score": json_scalar(row.get("flood_risk_score")),
                    "road_access_tier": json_scalar(row.get("road_access_tier")),
                    "county_hosted_flag": json_scalar(row.get("county_hosted_flag")),
                    "best_source_type": json_scalar(row.get("best_source_type")),
                    "selected": False,
                },
            }
        )
    lngs = [feature["geometry"]["coordinates"][0] for feature in features]
    lats = [feature["geometry"]["coordinates"][1] for feature in features]
    return {
        "geometry_mode": "viewport_geojson",
        "render_mode": "points",
        "geometry_bounds": [round(min(lngs), 6), round(min(lats), 6), round(max(lngs), 6), round(max(lats), 6)] if features else None,
        "geometry_view_box": None,
        "requested_bounds": [-91.65, 30.15, -88.0, 35.1],
        "zoom": 6.1,
        "feature_count": len(features),
        "feature_collection": {"type": "FeatureCollection", "features": features},
        "items": [{"parcel_row_id": row["parcel_row_id"], "path": None, "lead_score_total": json_scalar(row["lead_score_total"])} for _, row in top.iterrows()],
    }


def main() -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    GEOMETRY_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = build_runtime_frame()
    source_parcels = pd.read_parquet(PARCEL_MASTER_PATH, columns=["parcel_row_id", "county_name", "geometry"], engine="pyarrow")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    ds.write_dataset(
        table,
        base_dir=str(OUTPUT_ROOT),
        format="parquet",
        existing_data_behavior="delete_matching",
        partitioning=["county_name"],
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        max_rows_per_group=10000,
        max_rows_per_file=50000,
    )
    geometry_table = build_geometry_runtime(source_parcels)
    ds.write_dataset(
        geometry_table,
        base_dir=str(GEOMETRY_OUTPUT_ROOT),
        format="parquet",
        existing_data_behavior="delete_matching",
        partitioning=["county_name"],
        file_options=ds.ParquetFileFormat().make_write_options(compression="zstd"),
        max_rows_per_group=10000,
        max_rows_per_file=50000,
    )
    detail_metrics = build_detail_metrics_runtime(frame)
    detail_metrics.to_parquet(DETAIL_METRICS_OUTPUT_PATH, index=False, engine="pyarrow")
    (RUNTIME_ROOT / "summary.json").write_text(json.dumps(build_summary_payload(frame)), encoding="utf-8")
    (RUNTIME_ROOT / "presets.json").write_text(json.dumps(build_presets_payload(frame)), encoding="utf-8")
    (RUNTIME_ROOT / "default_leads.json").write_text(json.dumps(build_default_leads_payload(frame)), encoding="utf-8")
    (RUNTIME_ROOT / "default_geometry.json").write_text(json.dumps(build_default_geometry_payload(frame)), encoding="utf-8")
    print(f"Wrote {len(frame)} runtime rows to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
