from __future__ import annotations

import argparse
import csv
import math
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import requests
from requests.exceptions import RequestException

BASE_DIR = Path(__file__).resolve().parents[1]
ELEVATION_RAW_DIR = BASE_DIR / "data" / "elevation_raw"
PARCELS_DIR = BASE_DIR / "data" / "parcels"

# Mississippi-specific values (easy future config extraction).
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
STATE_BBOX_4326 = (-91.6550, 30.1739, -88.0979, 34.9963)
TNM_PRODUCTS_API = "https://tnmaccess.nationalmap.gov/api/v1/products"
TNM_DATASET_CANDIDATES = [
    "USGS 3DEP 1/3 arc-second DEM",
    "National Elevation Dataset (NED) 1/3 arc-second",
]
TNM_FORMATS = "GeoTIFF,IMG"
USGS_STAGED_TILE_BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
PARCEL_BOUNDS_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_flood.gpkg",
    PARCELS_DIR / "mississippi_parcels.gpkg",
]

MANIFEST_CSV = ELEVATION_RAW_DIR / "mississippi_elevation_download_manifest.csv"
REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Mississippi DEM tiles from official USGS The National Map API.")
    parser.add_argument("--max-items", type=int, default=500, help="Max product records per dataset query.")
    parser.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--download-limit", type=int, default=0, help="Optional cap on number of URLs to download (0 = no cap).")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    parser.add_argument("--no-extract", action="store_true", help="Skip ZIP extraction.")
    return parser.parse_args()


def request_json(url: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Request failed (attempt {attempt}/{retries}): {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def discover_bbox_4326(arg_bbox: tuple[float, float, float, float] | None) -> tuple[float, float, float, float]:
    if arg_bbox is not None:
        return tuple(arg_bbox)

    for parcel_path in PARCEL_BOUNDS_CANDIDATES:
        if not parcel_path.exists():
            continue
        try:
            full = gpd.read_file(parcel_path, columns=["geometry"])
            if full.crs is None:
                full = full.set_crs("EPSG:4326", allow_override=True)
            else:
                full = full.to_crs("EPSG:4326")
            bounds = tuple(full.total_bounds.tolist())
            print(f"Using parcel-driven bbox from {parcel_path.name}: {bounds}")
            return bounds
        except Exception as exc:
            print(f"Could not derive bbox from {parcel_path.name}: {exc}")
            continue

    print(f"Using fallback Mississippi bbox: {STATE_BBOX_4326}")
    return STATE_BBOX_4326


def normalize_url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = f"{STATE_ABBR.lower()}_dem_file.bin"
    return name


def extract_urls_from_item(item: dict) -> list[str]:
    urls: set[str] = set()
    for key in ["downloadURL", "downloadUrl", "url", "download_url"]:
        value = item.get(key)
        if isinstance(value, str) and value.startswith("http"):
            urls.add(value)

    urls_obj = item.get("urls")
    if isinstance(urls_obj, dict):
        for value in urls_obj.values():
            if isinstance(value, str) and value.startswith("http"):
                urls.add(value)
    elif isinstance(urls_obj, list):
        for value in urls_obj:
            if isinstance(value, str) and value.startswith("http"):
                urls.add(value)

    return sorted(urls)


def discover_dem_urls(bbox: tuple[float, float, float, float], max_items: int) -> tuple[list[str], str]:
    bbox_param = ",".join(f"{v:.6f}" for v in bbox)
    for dataset_name in TNM_DATASET_CANDIDATES:
        print(f"Querying The National Map dataset: {dataset_name}")
        params = {
            "datasets": dataset_name,
            "bbox": bbox_param,
            "max": int(max_items),
            "prodFormats": TNM_FORMATS,
            "outputFormat": "JSON",
        }
        try:
            data = request_json(TNM_PRODUCTS_API, params=params)
        except Exception as exc:
            print(f"  TNM API query failed for dataset '{dataset_name}': {exc}")
            continue
        items = data.get("items") or data.get("results") or []
        print(f"  Records returned: {len(items)}")

        urls: set[str] = set()
        for item in items:
            for url in extract_urls_from_item(item):
                urls.add(url)

        ordered = sorted(urls)
        if ordered:
            print(f"  Download URLs discovered: {len(ordered)}")
            return ordered, dataset_name

    staged_urls = discover_staged_tile_urls(bbox)
    if staged_urls:
        print(f"Fallback staged 3DEP tile URLs discovered: {len(staged_urls)}")
        return staged_urls, "USGS StagedProducts Elevation/13/TIFF/current"

    raise RuntimeError("No DEM download URLs discovered from TNM API or staged tile fallback.")


def tile_id_from_cell(south_lat: int, west_lon: int) -> str:
    # USGS 1x1 naming uses north-edge latitude for N tiles.
    if south_lat >= 0:
        lat_prefix = "n"
        lat_code = south_lat + 1
    else:
        lat_prefix = "s"
        lat_code = abs(south_lat)

    # USGS naming uses west-edge longitude for W tiles.
    if west_lon >= 0:
        lon_prefix = "e"
        lon_code = west_lon
    else:
        lon_prefix = "w"
        lon_code = abs(west_lon)
    return f"{lat_prefix}{lat_code:02d}{lon_prefix}{lon_code:03d}"


def discover_staged_tile_urls(bbox: tuple[float, float, float, float]) -> list[str]:
    xmin, ymin, xmax, ymax = bbox
    lon_values = list(range(math.floor(xmin), math.ceil(xmax)))
    lat_values = list(range(math.floor(ymin), math.ceil(ymax)))
    urls: list[str] = []

    print("Trying fallback discovery from USGS staged 3DEP tile URLs...")
    for south_lat in lat_values:
        for west_lon in lon_values:
            tile_id = tile_id_from_cell(south_lat, west_lon)
            url = f"{USGS_STAGED_TILE_BASE}/{tile_id}/USGS_13_{tile_id}.tif"
            try:
                response = requests.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
                if response.status_code == 200:
                    urls.append(url)
                    print(f"  Tile available: {tile_id}")
                else:
                    print(f"  Tile missing: {tile_id} (status {response.status_code})")
            except Exception as exc:
                print(f"  Tile check failed: {tile_id} ({exc})")
                continue
            time.sleep(0.1)
    return sorted(set(urls))


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "url" in row:
                    rows[row["url"]] = row
    except Exception:
        return {}
    return rows


def save_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source_dataset", "url", "file_name", "status", "bytes", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def download_file(url: str, destination: Path, force: bool) -> tuple[str, int, str]:
    if destination.exists() and not force:
        return "skipped_existing", int(destination.stat().st_size), "already present"

    temp_path = destination.with_suffix(destination.suffix + ".part")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                with temp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            temp_path.replace(destination)
            return "downloaded", int(destination.stat().st_size), ""
        except Exception as exc:
            last_error = exc
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Download failed (attempt {attempt}/{MAX_RETRIES}) for {destination.name}: {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    return "failed", 0, str(last_error)


def extract_zip_if_needed(path: Path) -> tuple[int, str]:
    if path.suffix.lower() != ".zip":
        return 0, ""
    extracted = 0
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for member in zf.infolist():
                suffix = Path(member.filename).suffix.lower()
                if suffix not in {".tif", ".tiff", ".img", ".adf"}:
                    continue
                target = ELEVATION_RAW_DIR / Path(member.filename).name
                if target.exists():
                    continue
                with zf.open(member, "r") as src, target.open("wb") as dst:
                    dst.write(src.read())
                extracted += 1
    except Exception as exc:
        return extracted, f"zip_extract_error: {exc}"
    return extracted, ""


def main() -> None:
    args = parse_args()
    ELEVATION_RAW_DIR.mkdir(parents=True, exist_ok=True)

    bbox_4326 = discover_bbox_4326(tuple(args.bbox) if args.bbox else None)
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"STATE: {STATE_NAME} ({STATE_ABBR})")
    print(f"Raw elevation dir: {ELEVATION_RAW_DIR}")
    print(f"BBox (EPSG:4326): {bbox_4326}")

    existing_rasters = sorted(ELEVATION_RAW_DIR.glob("*.tif")) + sorted(ELEVATION_RAW_DIR.glob("*.img"))
    if existing_rasters and not args.force:
        print(f"Found {len(existing_rasters)} existing DEM raster files. Download step is resumable and will skip duplicates.")

    urls, source_dataset = discover_dem_urls(bbox_4326, max_items=int(args.max_items))
    if args.download_limit > 0:
        urls = urls[: int(args.download_limit)]
    print(f"Planned downloads: {len(urls)}")

    manifest_map = load_manifest(MANIFEST_CSV)

    for idx, url in enumerate(urls, start=1):
        file_name = normalize_url_to_filename(url)
        destination = ELEVATION_RAW_DIR / file_name
        print(f"[{idx}/{len(urls)}] {file_name}")
        status, n_bytes, note = download_file(url, destination, force=bool(args.force))
        print(f"  Status: {status} ({n_bytes:,} bytes)")

        if status != "failed" and not args.no_extract:
            extracted_count, extract_note = extract_zip_if_needed(destination)
            if extracted_count:
                print(f"  Extracted raster members: {extracted_count}")
            if extract_note:
                print(f"  Extract note: {extract_note}")
                if note:
                    note = f"{note}; {extract_note}"
                else:
                    note = extract_note

        manifest_map[url] = {
            "source_dataset": source_dataset,
            "url": url,
            "file_name": file_name,
            "status": status,
            "bytes": n_bytes,
            "note": note,
        }
        save_manifest(MANIFEST_CSV, sorted(manifest_map.values(), key=lambda r: str(r.get("file_name", ""))))

    all_rasters = sorted(ELEVATION_RAW_DIR.glob("*.tif")) + sorted(ELEVATION_RAW_DIR.glob("*.img"))
    print(f"\nSource used: USGS The National Map ({source_dataset})")
    print(f"Raster files present in elevation_raw: {len(all_rasters)}")
    print(f"Manifest written: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
