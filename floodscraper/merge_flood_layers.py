from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DOWNLOADS_DIR = BASE_DIR / "data" / "fema_downloads"
UNZIPPED_DIR = BASE_DIR / "data" / "fema_unzipped"
OUTPUT_DIR = BASE_DIR / "data" / "flood_layers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:4326"
FLOOD_LAYER_NAME = "S_FLD_HAZ_AR.shp"
ZIP_NAME_PATTERN = re.compile(r"(?P<fips>\d{5})C_(?P<effective>\d{8})\.zip$", re.IGNORECASE)
MISSISSIPPI_FIPS_PREFIX = "28"


def extract_county_fips_from_zip_name(name: str) -> str | None:
    match = ZIP_NAME_PATTERN.match(name)
    if not match:
        return None
    return match.group("fips")


def is_mississippi_fips(county_fips: str | None) -> bool:
    return county_fips is not None and county_fips.startswith(MISSISSIPPI_FIPS_PREFIX)


def load_flood_from_zip(zip_path: Path) -> gpd.GeoDataFrame | None:
    zip_uri = f"zip://{zip_path}!{FLOOD_LAYER_NAME}"
    try:
        gdf = gpd.read_file(zip_uri)
    except Exception as exc:
        print(f"Failed reading {zip_path.name}: {exc}")
        return None

    if gdf.empty:
        print(f"Skipping empty flood layer in {zip_path.name}")
        return None
    if "FLD_ZONE" not in gdf.columns:
        print(f"Skipping {zip_path.name}: FLD_ZONE not found")
        return None
    if gdf.crs is None:
        print(f"Skipping {zip_path.name}: CRS missing")
        return None

    gdf = gdf.to_crs(TARGET_CRS).copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    county_fips = extract_county_fips_from_zip_name(zip_path.name)
    gdf["source_zip"] = zip_path.name
    gdf["source_county_fips"] = county_fips
    gdf["source_path"] = str(zip_path)
    return gdf


def load_fallback_unzipped_flood() -> gpd.GeoDataFrame | None:
    shp_path = UNZIPPED_DIR / FLOOD_LAYER_NAME
    if not shp_path.exists():
        return None
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as exc:
        print(f"Failed fallback read from {shp_path}: {exc}")
        return None
    if gdf.empty or "FLD_ZONE" not in gdf.columns or gdf.crs is None:
        return None
    gdf = gdf.to_crs(TARGET_CRS).copy()
    gdf["source_zip"] = "fallback_unzipped_single_layer"
    gdf["source_county_fips"] = pd.NA
    gdf["source_path"] = str(shp_path)
    return gdf


def main() -> None:
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DOWNLOADS_DIR: {DOWNLOADS_DIR}")
    print(f"TARGET_CRS: {TARGET_CRS}\n")

    all_zip_files = sorted(DOWNLOADS_DIR.glob("*C_*.zip"))
    zip_files = [z for z in all_zip_files if is_mississippi_fips(extract_county_fips_from_zip_name(z.name))]
    merged_layers: list[gpd.GeoDataFrame] = []
    source_rows: list[dict] = []

    if zip_files:
        print(
            f"Found {len(zip_files)} Mississippi FEMA ZIP files "
            f"(from {len(all_zip_files)} total ZIPs). Loading {FLOOD_LAYER_NAME} from each ZIP...\n"
        )
        for idx, zip_path in enumerate(zip_files, start=1):
            print(f"[{idx}/{len(zip_files)}] {zip_path.name}")
            gdf = load_flood_from_zip(zip_path)
            if gdf is None:
                source_rows.append(
                    {
                        "source_zip": zip_path.name,
                        "source_county_fips": extract_county_fips_from_zip_name(zip_path.name),
                        "rows": 0,
                        "status": "failed_or_missing_layer",
                    }
                )
                continue

            merged_layers.append(gdf)
            source_rows.append(
                {
                    "source_zip": zip_path.name,
                    "source_county_fips": extract_county_fips_from_zip_name(zip_path.name),
                    "rows": len(gdf),
                    "status": "ok",
                }
            )
            print(f"  Added {len(gdf):,} rows")
    else:
        print("No FEMA ZIP files found. Trying fallback from data/fema_unzipped...")
        fallback = load_fallback_unzipped_flood()
        if fallback is not None:
            merged_layers.append(fallback)
            source_rows.append(
                {
                    "source_zip": "fallback_unzipped_single_layer",
                    "source_county_fips": pd.NA,
                    "rows": len(fallback),
                    "status": "ok",
                }
            )

    if not merged_layers:
        print("No flood layers were loaded. Nothing to merge.")
        return

    merged = pd.concat(merged_layers, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    merged = merged[merged.geometry.notnull()].copy()
    merged = merged[~merged.geometry.is_empty].copy()
    merged["FLD_ZONE"] = merged["FLD_ZONE"].astype(str).str.strip()

    if "FLD_AR_ID" in merged.columns:
        merged = merged.drop_duplicates(subset=["FLD_AR_ID", "source_county_fips"]).copy()
    else:
        merged["_geom_wkb"] = merged.geometry.apply(lambda geom: geom.wkb_hex if geom is not None else None)
        merged = merged.drop_duplicates(subset=["_geom_wkb"]).drop(columns=["_geom_wkb"]).copy()

    gpkg_path = OUTPUT_DIR / "mississippi_fema_flood.gpkg"
    shp_path = OUTPUT_DIR / "mississippi_fema_flood.shp"
    used_files_csv = OUTPUT_DIR / "merged_flood_source_files.csv"
    zone_summary_csv = OUTPUT_DIR / "flood_zone_summary.csv"

    print("\nSaving merged flood outputs...")
    merged.to_file(gpkg_path, driver="GPKG", engine="pyogrio")
    merged.to_file(shp_path)

    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(used_files_csv, index=False)

    zone_summary = (
        merged["FLD_ZONE"]
        .value_counts(dropna=False)
        .rename_axis("FLD_ZONE")
        .reset_index(name="count")
    )
    zone_summary.to_csv(zone_summary_csv, index=False)

    represented_fips = merged["source_county_fips"].dropna().astype(str).unique()
    print("\nDone.")
    print(f"Merged rows: {len(merged):,}")
    print(f"Represented county FIPS in merged layer: {len(represented_fips)}")
    print(f"Bounds: {merged.total_bounds}")
    print(f"Output GPKG: {gpkg_path}")
    print(f"Output SHP:  {shp_path}")
    print(f"Source CSV:  {used_files_csv}")
    print(f"Zone summary:{zone_summary_csv}")


if __name__ == "__main__":
    main()
