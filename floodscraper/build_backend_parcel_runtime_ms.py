from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from shapely import wkb


ROOT = Path(__file__).resolve().parents[1]
PARCEL_MASTER_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_master.parquet"
OWNER_LEADS_PATH = ROOT / "data" / "parcels" / "mississippi_parcels_owner_leads.parquet"
BUILDING_METRICS_PATH = ROOT / "data" / "buildings_processed" / "parcel_building_metrics.parquet"
LEAD_SIGNALS_PATH = ROOT / "data" / "tax_published" / "ms" / "app_ready_mississippi_leads.parquet"
OUTPUT_ROOT = ROOT / "backend" / "runtime" / "mississippi" / "parcel_index"


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
    "flood_risk_score",
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


def build_runtime_frame() -> pd.DataFrame:
    parcels = pd.read_parquet(PARCEL_MASTER_PATH, columns=PARCEL_COLUMNS, engine="pyarrow")
    parcels["acreage"] = coalesce_numeric(parcels, ["total_acres", "parcel_area_acres", "gis_acres", "tax_acres"])
    parcels["land_use"] = parcels["land_use_raw"].astype("string").str.strip()
    parcels["assessed_total_value"] = pd.to_numeric(parcels["total_value"], errors="coerce")
    parcels["geometry"] = parcels["geometry"].map(point_geometry_from_wkb)
    parcels = parcels.drop(
        columns=[
            "land_use_raw",
            "tax_acres",
            "gis_acres",
            "total_acres",
            "parcel_area_acres",
            "total_value",
        ]
    )

    owners = pd.read_parquet(OWNER_LEADS_PATH, columns=OWNER_COLUMNS, engine="pyarrow")
    buildings = pd.read_parquet(BUILDING_METRICS_PATH, columns=BUILDING_COLUMNS, engine="pyarrow")
    signals = pd.read_parquet(LEAD_SIGNALS_PATH, columns=SIGNAL_COLUMNS, engine="pyarrow")

    frame = parcels.merge(owners, on="parcel_row_id", how="left")
    frame = frame.merge(buildings, on="parcel_row_id", how="left")
    frame = frame.merge(signals, on="parcel_row_id", how="left")

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

    return frame


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frame = build_runtime_frame()
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
    print(f"Wrote {len(frame)} runtime rows to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
