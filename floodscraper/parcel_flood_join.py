from pathlib import Path
import sys
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parents[1]

FLOOD_FILE = BASE_DIR / "data" / "flood_layers" / "mississippi_fema_flood.gpkg"
PARCEL_FILE = BASE_DIR / "data" / "parcels" / "mississippi_parcels.gpkg"
OUTPUT_FILE = BASE_DIR / "data" / "flood_layers" / "parcel_flood_risk.gpkg"


def main():
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Flood file:  {FLOOD_FILE}")
    print(f"Parcel file: {PARCEL_FILE}")
    print()

    if not FLOOD_FILE.exists():
        print(f"ERROR: Flood file not found:\n{FLOOD_FILE}")
        sys.exit(1)

    if not PARCEL_FILE.exists():
        print(f"ERROR: Parcel file not found:\n{PARCEL_FILE}")
        print("\nCreate this folder/file first:")
        print(BASE_DIR / "data" / "parcels")
        sys.exit(1)

    print("Loading FEMA flood layer...")
    flood = gpd.read_file(FLOOD_FILE)

    print("Loading parcel polygons...")
    parcels = gpd.read_file(PARCEL_FILE)

    print("Matching parcel CRS to flood CRS...")
    parcels = parcels.to_crs(flood.crs)

    print("Running spatial join...")
    joined = gpd.sjoin(
        parcels,
        flood[["FLD_ZONE", "geometry"]],
        how="left",
        predicate="intersects"
    )

    risk_map = {
        "X": 0,
        "A": 8,
        "AE": 10,
        "AH": 7
    }

    joined["flood_risk_score"] = joined["FLD_ZONE"].map(risk_map).fillna(0)

    print(f"Saving output to: {OUTPUT_FILE}")
    joined.to_file(OUTPUT_FILE, driver="GPKG")

    print("Done.")


if __name__ == "__main__":
    main()