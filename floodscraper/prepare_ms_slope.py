from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

BASE_DIR = Path(__file__).resolve().parents[1]
ELEVATION_RAW_DIR = BASE_DIR / "data" / "elevation_raw"
ELEVATION_PROCESSED_DIR = BASE_DIR / "data" / "elevation_processed"
PARCELS_DIR = BASE_DIR / "data" / "parcels"

# Mississippi-specific values (easy future config extraction).
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
TARGET_VECTOR_CRS = "EPSG:4326"
SLOPE_CALC_CRS = "EPSG:5070"  # Meter-based projection for stable slope calculations.
DEM_OUTPUT = ELEVATION_PROCESSED_DIR / "mississippi_dem.tif"
SLOPE_OUTPUT = ELEVATION_PROCESSED_DIR / "mississippi_slope.tif"
PARCEL_BOUNDS_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood.gpkg",
    PARCELS_DIR / "mississippi_parcels.gpkg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Mississippi DEM and slope rasters.")
    parser.add_argument("--raw-dir", type=str, default=str(ELEVATION_RAW_DIR), help="Directory containing raw DEM tiles.")
    parser.add_argument("--dem-out", type=str, default=str(DEM_OUTPUT), help="Output DEM GeoTIFF path.")
    parser.add_argument("--slope-out", type=str, default=str(SLOPE_OUTPUT), help="Output slope GeoTIFF path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed rasters.")
    return parser.parse_args()


def choose_parcel_bounds_4326() -> tuple[float, float, float, float] | None:
    for path in PARCEL_BOUNDS_CANDIDATES:
        if not path.exists():
            continue
        try:
            gdf = gpd.read_file(path, columns=["geometry"])
            if gdf.empty:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs(TARGET_VECTOR_CRS, allow_override=True)
            else:
                gdf = gdf.to_crs(TARGET_VECTOR_CRS)
            bounds = tuple(gdf.total_bounds.tolist())
            print(f"Using parcel bounds from {path.name}: {bounds}")
            return bounds
        except Exception as exc:
            print(f"Could not read bounds from {path.name}: {exc}")
    return None


def find_raw_dem_files(raw_dir: Path) -> list[Path]:
    patterns = ["*.tif", "*.tiff", "*.img"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_dir.rglob(pattern))
    unique_files = sorted({path.resolve() for path in files})
    return [Path(p) for p in unique_files]


def write_dem_mosaic(
    dem_sources: list[Path],
    dem_out: Path,
    clip_bounds_4326: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, Affine, rasterio.crs.CRS, float | None]:
    src_handles = []
    try:
        for path in dem_sources:
            src_handles.append(rasterio.open(path))

        if not src_handles:
            raise RuntimeError("No readable DEM sources.")

        mosaic_crs = src_handles[0].crs
        if mosaic_crs is None:
            raise RuntimeError("DEM source CRS is missing.")

        mosaic_bounds = None
        if clip_bounds_4326 is not None:
            try:
                mosaic_bounds = transform_bounds("EPSG:4326", mosaic_crs, *clip_bounds_4326, densify_pts=21)
                print(f"Clipping mosaic to bounds in DEM CRS: {mosaic_bounds}")
            except Exception as exc:
                print(f"Could not transform clip bounds to DEM CRS: {exc}. Using full mosaic extent.")
                mosaic_bounds = None

        mosaic_arr, mosaic_transform = merge(src_handles, bounds=mosaic_bounds)
        band1 = mosaic_arr[0].astype(np.float32)
        nodata = src_handles[0].nodata
        if nodata is not None:
            band1[band1 == nodata] = np.nan

        profile = src_handles[0].profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            height=band1.shape[0],
            width=band1.shape[1],
            transform=mosaic_transform,
            compress="lzw",
            tiled=True,
            BIGTIFF="YES",
            nodata=np.nan,
        )
        dem_out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dem_out, "w", **profile) as dst:
            dst.write(band1, 1)

        print(f"Saved DEM mosaic: {dem_out}")
        return band1, mosaic_transform, mosaic_crs, nodata
    finally:
        for src in src_handles:
            try:
                src.close()
            except Exception:
                pass


def reproject_dem_to_metric(
    dem_arr: np.ndarray,
    dem_transform: Affine,
    dem_crs,
    dst_crs: str,
) -> tuple[np.ndarray, Affine]:
    src_height, src_width = dem_arr.shape
    left, bottom, right, top = rasterio.transform.array_bounds(src_height, src_width, dem_transform)

    dst_transform, dst_width, dst_height = calculate_default_transform(
        dem_crs,
        dst_crs,
        src_width,
        src_height,
        left,
        bottom,
        right,
        top,
    )

    dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=dem_arr,
        destination=dst_arr,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst_arr, dst_transform


def compute_slope_percent(dem_metric: np.ndarray, metric_transform: Affine) -> np.ndarray:
    y_res = abs(float(metric_transform.e))
    x_res = abs(float(metric_transform.a))
    if y_res <= 0 or x_res <= 0:
        raise RuntimeError("Invalid raster resolution for slope computation.")

    dem = dem_metric.astype(np.float32)
    dem[~np.isfinite(dem)] = np.nan

    dz_dy, dz_dx = np.gradient(dem, y_res, x_res)
    slope_pct = np.sqrt((dz_dx ** 2) + (dz_dy ** 2)) * 100.0
    slope_pct[~np.isfinite(dem)] = np.nan
    return slope_pct.astype(np.float32)


def write_slope_raster(slope_arr: np.ndarray, transform: Affine, out_path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "height": slope_arr.shape[0],
        "width": slope_arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": SLOPE_CALC_CRS,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "BIGTIFF": "YES",
        "nodata": np.nan,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(slope_arr, 1)
    print(f"Saved slope raster: {out_path}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    dem_out = Path(args.dem_out)
    slope_out = Path(args.slope_out)
    if not raw_dir.is_absolute():
        raw_dir = BASE_DIR / raw_dir
    if not dem_out.is_absolute():
        dem_out = BASE_DIR / dem_out
    if not slope_out.is_absolute():
        slope_out = BASE_DIR / slope_out

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATE: {STATE_NAME} ({STATE_ABBR})")
    print(f"Raw DEM dir: {raw_dir}")
    print(f"DEM out: {dem_out}")
    print(f"Slope out: {slope_out}")

    if dem_out.exists() and slope_out.exists() and not args.overwrite:
        print("Processed DEM and slope outputs already exist. Use --overwrite to rebuild.")
        return

    dem_files = find_raw_dem_files(raw_dir)
    if not dem_files:
        print("ERROR: No raw DEM raster files found. Run download_ms_elevation.py first.")
        return
    print(f"Raw DEM files found: {len(dem_files)}")

    clip_bounds_4326 = choose_parcel_bounds_4326()
    dem_arr, dem_transform, dem_crs, _ = write_dem_mosaic(
        dem_sources=dem_files,
        dem_out=dem_out,
        clip_bounds_4326=clip_bounds_4326,
    )

    print(f"Reprojecting DEM to {SLOPE_CALC_CRS} for slope calculation...")
    dem_metric, metric_transform = reproject_dem_to_metric(dem_arr, dem_transform, dem_crs, SLOPE_CALC_CRS)

    print("Computing slope percent raster...")
    slope_pct = compute_slope_percent(dem_metric, metric_transform)
    write_slope_raster(slope_pct, metric_transform, slope_out)

    valid = slope_pct[np.isfinite(slope_pct)]
    if valid.size > 0:
        print(
            f"Slope stats (%): min={float(np.min(valid)):.2f}, median={float(np.median(valid)):.2f}, max={float(np.max(valid)):.2f}"
        )
    print("Prepare step complete.")


if __name__ == "__main__":
    main()
