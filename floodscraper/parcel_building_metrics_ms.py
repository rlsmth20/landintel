from __future__ import annotations

import math
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PARCELS_DIR = DATA_DIR / "parcels"
BUILDINGS_RAW_DIR = DATA_DIR / "buildings_raw"
BUILDINGS_PROCESSED_DIR = DATA_DIR / "buildings_processed"

RAW_ZIP_PATH = BUILDINGS_RAW_DIR / "Mississippi.zip"
RAW_GEOJSON_PATH = BUILDINGS_RAW_DIR / "Mississippi.geojson"
PROCESSED_BUILDINGS_PATH = BUILDINGS_PROCESSED_DIR / "mississippi_building_centroids.parquet"
PARCEL_MASTER_PATH = PARCELS_DIR / "mississippi_parcels_master.parquet"
PARCEL_BUILDING_METRICS_PATH = BUILDINGS_PROCESSED_DIR / "parcel_building_metrics.parquet"
SUMMARY_PATH = BUILDINGS_PROCESSED_DIR / "parcel_building_metrics_summary.csv"
GROWTH_DISTRIBUTION_PATH = BUILDINGS_PROCESSED_DIR / "growth_pressure_distribution.csv"

MICROSOFT_BUILDINGS_URL = "https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/Mississippi.geojson.zip"
EQUAL_AREA_CRS = "EPSG:5070"
SQM_TO_SQFT = 10.763910416709722

PARCEL_COLUMNS = [
    "parcel_row_id",
    "county_fips",
    "county_name",
    "total_acres",
    "tax_acres",
    "gis_acres",
    "geometry",
]


def normalize_string(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype="string")
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def ensure_raw_download() -> None:
    BUILDINGS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_ZIP_PATH.exists() and RAW_ZIP_PATH.stat().st_size < 1024 * 1024:
        RAW_ZIP_PATH.unlink()
    if not RAW_ZIP_PATH.exists():
        print(f"Downloading {MICROSOFT_BUILDINGS_URL} -> {RAW_ZIP_PATH.relative_to(BASE_DIR)}")
        urllib.request.urlretrieve(MICROSOFT_BUILDINGS_URL, RAW_ZIP_PATH)
    if not RAW_GEOJSON_PATH.exists():
        print(f"Extracting {RAW_ZIP_PATH.relative_to(BASE_DIR)}")
        with zipfile.ZipFile(RAW_ZIP_PATH) as archive:
            archive.extractall(BUILDINGS_RAW_DIR)


def build_processed_buildings() -> None:
    BUILDINGS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if PROCESSED_BUILDINGS_PATH.exists():
        return

    print(f"Reading raw footprints: {RAW_GEOJSON_PATH.relative_to(BASE_DIR)}")
    buildings = gpd.read_file(RAW_GEOJSON_PATH)
    buildings = buildings.loc[:, ["geometry"]].copy()
    buildings = buildings.to_crs(EQUAL_AREA_CRS)
    buildings["building_area_total"] = (buildings.geometry.area * SQM_TO_SQFT).round(2)
    buildings["geometry"] = buildings.geometry.centroid
    buildings = buildings.reset_index(drop=True)
    buildings["building_id"] = np.arange(len(buildings), dtype=np.int64)
    buildings = gpd.GeoDataFrame(
        buildings.loc[:, ["building_id", "building_area_total", "geometry"]],
        geometry="geometry",
        crs=EQUAL_AREA_CRS,
    )
    buildings.to_parquet(PROCESSED_BUILDINGS_PATH, index=False)
    print(f"Processed buildings: {PROCESSED_BUILDINGS_PATH.relative_to(BASE_DIR)}")


def read_counties() -> list[str]:
    counties = pd.read_parquet(PARCEL_MASTER_PATH, columns=["county_name"])
    return sorted(normalize_string(counties["county_name"]).dropna().unique().tolist())


def read_county_parcels(county_name: str) -> gpd.GeoDataFrame:
    try:
        parcels = gpd.read_parquet(PARCEL_MASTER_PATH, columns=PARCEL_COLUMNS, filters=[("county_name", "==", county_name)])
    except Exception:
        parcels = gpd.read_parquet(PARCEL_MASTER_PATH, columns=PARCEL_COLUMNS)
        parcels = parcels.loc[normalize_string(parcels["county_name"]).eq(county_name)].copy()
    if parcels.empty:
        return gpd.GeoDataFrame(columns=PARCEL_COLUMNS, geometry="geometry", crs="EPSG:4326")
    parcels = parcels.to_crs(EQUAL_AREA_CRS)
    acreage = pd.to_numeric(parcels["total_acres"], errors="coerce")
    acreage = acreage.fillna(pd.to_numeric(parcels["tax_acres"], errors="coerce"))
    acreage = acreage.fillna(pd.to_numeric(parcels["gis_acres"], errors="coerce"))
    parcels["parcel_acreage"] = acreage
    return parcels


def growth_bucket(values: pd.Series) -> pd.Series:
    density = pd.to_numeric(values, errors="coerce").fillna(0.0)
    bucket = pd.Series("very_low", index=density.index, dtype="string")
    bucket.loc[density.ge(25) & density.lt(100)] = "low"
    bucket.loc[density.ge(100) & density.lt(300)] = "moderate"
    bucket.loc[density.ge(300) & density.lt(800)] = "high"
    bucket.loc[density.ge(800)] = "very_high"
    return bucket


def compute_county_metrics(parcels: gpd.GeoDataFrame, tree: STRtree, building_area: np.ndarray) -> pd.DataFrame:
    if parcels.empty:
        return pd.DataFrame(
            columns=[
                "parcel_row_id",
                "county_fips",
                "county_name",
                "building_count",
                "building_area_total",
                "parcel_vacant_flag",
                "nearby_building_count_1km",
                "nearby_building_density",
                "growth_pressure_bucket",
            ]
        )

    parcel_geoms = np.asarray(parcels.geometry.values)
    contains_pairs = tree.query(parcel_geoms, predicate="contains")
    building_count = np.bincount(contains_pairs[0], minlength=len(parcels)).astype(np.int64)
    building_area_total = np.bincount(
        contains_pairs[0],
        weights=building_area[contains_pairs[1]] if contains_pairs.shape[1] else np.array([], dtype=float),
        minlength=len(parcels),
    )

    parcel_centroids = np.asarray(parcels.geometry.centroid.values)
    nearby_pairs = tree.query(parcel_centroids, predicate="dwithin", distance=1000.0)
    nearby_count = np.bincount(nearby_pairs[0], minlength=len(parcels)).astype(np.int64)
    nearby_density = nearby_count / math.pi

    metrics = pd.DataFrame(
        {
            "parcel_row_id": parcels["parcel_row_id"].astype("string"),
            "county_fips": normalize_string(parcels["county_fips"]),
            "county_name": normalize_string(parcels["county_name"]),
            "building_count": building_count,
            "building_area_total": np.round(building_area_total, 2),
            "parcel_vacant_flag": building_count == 0,
            "nearby_building_count_1km": nearby_count,
            "nearby_building_density": np.round(nearby_density, 2),
        }
    )
    metrics["growth_pressure_bucket"] = growth_bucket(metrics["nearby_building_density"])
    return metrics


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"section": "statewide", "metric": "parcel_count", "key": pd.NA, "value": int(len(metrics))},
        {"section": "statewide", "metric": "vacant_parcel_count", "key": pd.NA, "value": int(metrics["parcel_vacant_flag"].fillna(False).sum())},
        {"section": "statewide", "metric": "vacant_parcel_pct", "key": pd.NA, "value": round(float(metrics["parcel_vacant_flag"].fillna(False).mean() * 100.0), 4)},
        {"section": "statewide", "metric": "total_building_count_assigned", "key": pd.NA, "value": int(pd.to_numeric(metrics["building_count"], errors="coerce").fillna(0).sum())},
        {"section": "statewide", "metric": "average_building_count", "key": pd.NA, "value": round(float(pd.to_numeric(metrics["building_count"], errors="coerce").mean()), 4)},
        {"section": "statewide", "metric": "average_building_area_total", "key": pd.NA, "value": round(float(pd.to_numeric(metrics["building_area_total"], errors="coerce").mean()), 4)},
        {"section": "statewide", "metric": "average_nearby_building_density", "key": pd.NA, "value": round(float(pd.to_numeric(metrics["nearby_building_density"], errors="coerce").mean()), 4)},
    ]

    county_summary = (
        metrics.groupby(["county_fips", "county_name"], dropna=False)
        .agg(
            parcel_count=("parcel_row_id", "size"),
            vacant_parcel_count=("parcel_vacant_flag", lambda s: int(s.fillna(False).sum())),
            avg_building_count=("building_count", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 4)),
            avg_building_area_total=("building_area_total", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 4)),
            avg_nearby_building_density=("nearby_building_density", lambda s: round(float(pd.to_numeric(s, errors="coerce").mean()), 4)),
        )
        .reset_index()
    )
    county_summary["vacant_parcel_pct"] = np.round(
        county_summary["vacant_parcel_count"] / county_summary["parcel_count"] * 100.0,
        4,
    )

    for _, row in county_summary.sort_values("avg_nearby_building_density", ascending=False).head(25).iterrows():
        rows.append({"section": "top_counties_by_growth_pressure", "metric": "avg_nearby_building_density", "key": row["county_name"], "value": row["avg_nearby_building_density"]})
    for _, row in county_summary.sort_values("vacant_parcel_pct", ascending=False).head(25).iterrows():
        rows.append({"section": "top_counties_by_vacancy", "metric": "vacant_parcel_pct", "key": row["county_name"], "value": row["vacant_parcel_pct"]})
    return pd.DataFrame(rows)


def build_growth_distribution(metrics: pd.DataFrame) -> pd.DataFrame:
    statewide = (
        normalize_string(metrics["growth_pressure_bucket"], metrics.index)
        .value_counts(dropna=False)
        .rename_axis("growth_pressure_bucket")
        .reset_index(name="parcel_count")
    )
    statewide.insert(0, "scope", "statewide")
    statewide.insert(1, "county_name", pd.NA)

    county = (
        metrics.groupby("county_name", dropna=False)["growth_pressure_bucket"]
        .value_counts(dropna=False)
        .rename("parcel_count")
        .reset_index()
    )
    county.insert(0, "scope", "county")
    return pd.concat([statewide, county], ignore_index=True)


def main() -> None:
    ensure_raw_download()
    build_processed_buildings()

    buildings = gpd.read_parquet(PROCESSED_BUILDINGS_PATH)
    building_points = np.asarray(buildings.geometry.values)
    building_area = pd.to_numeric(buildings["building_area_total"], errors="coerce").fillna(0.0).to_numpy()
    tree = STRtree(building_points)

    county_frames: list[pd.DataFrame] = []
    for county_name in read_counties():
        parcels = read_county_parcels(county_name)
        county_metrics = compute_county_metrics(parcels, tree, building_area)
        county_frames.append(county_metrics)
        print(f"{county_name}: {len(county_metrics):,} parcels")

    metrics = pd.concat(county_frames, ignore_index=True)
    BUILDINGS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(PARCEL_BUILDING_METRICS_PATH, index=False)
    build_summary(metrics).to_csv(SUMMARY_PATH, index=False)
    build_growth_distribution(metrics).to_csv(GROWTH_DISTRIBUTION_PATH, index=False)

    print(f"Parcel metrics: {PARCEL_BUILDING_METRICS_PATH.relative_to(BASE_DIR)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(BASE_DIR)}")
    print(f"Growth distribution: {GROWTH_DISTRIBUTION_PATH.relative_to(BASE_DIR)}")
    print(f"Rows: {len(metrics):,}")


if __name__ == "__main__":
    main()
