from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "parcel_raw"
PARCELS_DIR = BASE_DIR / "data" / "parcels"
FLOOD_DIR = BASE_DIR / "data" / "flood_layers"

COUNTY_TEST_FILE = RAW_DIR / "adams_parcels.gpkg"
STATEWIDE_FILE = PARCELS_DIR / "mississippi_parcels.gpkg"
FLOOD_FILE = FLOOD_DIR / "mississippi_fema_flood.gpkg"
JOIN_FILE = PARCELS_DIR / "mississippi_parcels_with_flood.gpkg"
SUMMARY_FILE = PARCELS_DIR / "mississippi_parcels_with_flood_summary.csv"


def print_basic_stats(path: Path, label: str, expected_zone_col: str | None = None) -> None:
    print(f"\n[{label}] {path}")
    if not path.exists():
        print("  MISSING")
        return

    gdf = gpd.read_file(path)
    print(f"  Rows: {len(gdf):,}")
    print(f"  CRS:  {gdf.crs}")

    if "geometry" in gdf.columns:
        geom_types = gdf.geometry.geom_type.value_counts(dropna=False).to_dict()
        print(f"  Geometry types: {geom_types}")

    if expected_zone_col and expected_zone_col in gdf.columns:
        top = gdf[expected_zone_col].astype(str).value_counts(dropna=False).head(10)
        print("  Top flood zones:")
        for zone, count in top.items():
            print(f"    {zone}: {count:,}")


def main() -> None:
    print(f"BASE_DIR: {BASE_DIR}")
    print("Running Mississippi pipeline validation...")

    print_basic_stats(COUNTY_TEST_FILE, "County test file")
    print_basic_stats(STATEWIDE_FILE, "Statewide parcels")
    print_basic_stats(FLOOD_FILE, "FEMA flood layer", expected_zone_col="FLD_ZONE")
    print_basic_stats(JOIN_FILE, "Parcel flood join", expected_zone_col="flood_zone_primary")

    if JOIN_FILE.exists():
        joined = gpd.read_file(JOIN_FILE)
        if "flood_zone_primary" in joined.columns:
            matched = int((joined["flood_zone_primary"].astype(str).str.upper() != "UNKNOWN").sum())
            print(f"\nJoin non-UNKNOWN flood zone rows: {matched:,} / {len(joined):,}")
            if matched == 0:
                print("WARNING: Join produced zero non-UNKNOWN flood zones.")
            else:
                print("Join check passed: non-UNKNOWN flood zones exist.")
        else:
            print("\nWARNING: 'flood_zone_primary' column missing in join output.")

    if SUMMARY_FILE.exists():
        summary = pd.read_csv(SUMMARY_FILE)
        print(f"\nSummary CSV rows: {len(summary):,}")
        print(summary.head(10).to_string(index=False))
    else:
        print(f"\nSummary CSV missing: {SUMMARY_FILE}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
