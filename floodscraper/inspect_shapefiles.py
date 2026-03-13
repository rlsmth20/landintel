from pathlib import Path
import pandas as pd
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parents[1]
UNZIPPED_DIR = BASE_DIR / "data" / "fema_unzipped"
OUTPUT_DIR = BASE_DIR / "data" / "flood_layers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FEMA flood-related fields often found in the useful polygon layer
FLOOD_HINT_COLUMNS = {
    "FLD_ZONE",
    "ZONE_SUBTY",
    "SFHA_TF",
    "STATIC_BFE",
    "V_DATUM",
    "DEPTH",
    "LEN_UNIT",
    "FLOODWAY_TF",
}


def summarize_shapefile(shp_path: Path) -> dict:
    try:
        gdf = gpd.read_file(shp_path)

        columns_upper = [str(c).upper() for c in gdf.columns]
        matching_columns = sorted(list(FLOOD_HINT_COLUMNS.intersection(columns_upper)))

        geom_types = []
        if "geometry" in gdf.columns:
            try:
                geom_types = sorted(gdf.geometry.geom_type.dropna().unique().tolist())
            except Exception:
                geom_types = []

        return {
            "file_name": shp_path.name,
            "full_path": str(shp_path),
            "parent_folder": str(shp_path.parent),
            "rows": len(gdf),
            "cols": len(gdf.columns),
            "crs": str(gdf.crs),
            "geometry_types": ", ".join(geom_types),
            "columns": ", ".join(map(str, gdf.columns)),
            "matching_flood_columns": ", ".join(matching_columns),
            "is_candidate_flood_layer": len(matching_columns) > 0,
        }

    except Exception as e:
        return {
            "file_name": shp_path.name,
            "full_path": str(shp_path),
            "parent_folder": str(shp_path.parent),
            "rows": None,
            "cols": None,
            "crs": None,
            "geometry_types": None,
            "columns": f"ERROR: {e}",
            "matching_flood_columns": "",
            "is_candidate_flood_layer": False,
        }


def main() -> None:
    shp_files = sorted(UNZIPPED_DIR.rglob("*.shp"))

    if not shp_files:
        print(f"No shapefiles found in: {UNZIPPED_DIR}")
        return

    print(f"Found {len(shp_files)} shapefiles. Inspecting...\n")

    records = []
    for i, shp in enumerate(shp_files, start=1):
        print(f"[{i}/{len(shp_files)}] {shp.name}")
        records.append(summarize_shapefile(shp))

    inventory_df = pd.DataFrame(records)
    inventory_csv = OUTPUT_DIR / "fema_shapefile_inventory.csv"
    inventory_df.to_csv(inventory_csv, index=False)

    candidates_df = inventory_df[inventory_df["is_candidate_flood_layer"] == True].copy()
    candidates_csv = OUTPUT_DIR / "fema_candidate_flood_layers.csv"
    candidates_df.to_csv(candidates_csv, index=False)

    print("\nInspection complete.")
    print(f"Inventory saved to:   {inventory_csv}")
    print(f"Candidates saved to:  {candidates_csv}")

    print("\nTop candidate layers:")
    if not candidates_df.empty:
        display_cols = [
            "file_name",
            "rows",
            "geometry_types",
            "matching_flood_columns",
            "full_path",
        ]
        print(candidates_df[display_cols].head(20).to_string(index=False))
    else:
        print("No obvious flood candidate layers were found.")


if __name__ == "__main__":
    main()