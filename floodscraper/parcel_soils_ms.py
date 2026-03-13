from __future__ import annotations

import argparse
import csv
import io
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.exceptions import RequestException

BASE_DIR = Path(__file__).resolve().parents[1]
STATE_NAME = "Mississippi"
STATE_ABBR = "MS"
SOIL_DATA_PATH = BASE_DIR / "data"
PARCELS_DIR = SOIL_DATA_PATH / "parcels"
SOILS_RAW_DIR = SOIL_DATA_PATH / "soils_raw"
SOILS_PROCESSED_DIR = SOIL_DATA_PATH / "soils_processed"

TARGET_CRS = "EPSG:4326"
AREA_CRS = "EPSG:5070"

INPUT_PARCEL_CANDIDATES = [
    PARCELS_DIR / "mississippi_parcels_with_utilities.gpkg",
    PARCELS_DIR / "mississippi_parcels_scored.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_flood_slope_wetlands.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_flood_and_slope.gpkg",
    PARCELS_DIR / "mississippi_parcels_with_flood.gpkg",
    PARCELS_DIR / "mississippi_parcels.gpkg",
]

SURVEY_CATALOG_CSV = SOILS_PROCESSED_DIR / "mississippi_ssurgo_surveys.csv"
STATEWIDE_SOILS_GPKG = SOILS_PROCESSED_DIR / "mississippi_ssurgo_mapunits.gpkg"
MAPUNIT_ATTR_CSV = SOILS_PROCESSED_DIR / "mississippi_ssurgo_mapunit_attributes.csv"

OUTPUT_FILE = PARCELS_DIR / "mississippi_parcels_with_soils.gpkg"
SUMMARY_CSV = PARCELS_DIR / "ms_soils_summary.csv"
PARTS_DIR = PARCELS_DIR / "ms_parcels_soils_parts"
CHECKPOINT_CSV = PARCELS_DIR / "mississippi_parcels_with_soils_progress.csv"

SDA_POST_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
WSS_ZIP_URL_TEMPLATE = (
    "https://websoilsurvey.sc.egov.usda.gov/DSD/Download/Cache/SSA/"
    "wss_SSA_{areasymbol}_soildb_US_2003_[{save_date}].zip"
)

REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0
CHUNK_SIZE_BYTES = 1024 * 1024

TABLE_FILE_MAP = {
    "mapunit": "mapunit.txt",
    "muaggatt": "muaggatt.txt",
    "component": "comp.txt",
    "chorizon": "chorizon.txt",
    "chtexturegrp": "chtexgrp.txt",
}

PARCEL_OUTPUT_COLUMNS = [
    "dominant_soil_type",
    "drainage_class",
    "hydrologic_group",
    "soil_texture",
    "soil_depth",
    "septic_suitability",
    "septic_limitation_class",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Mississippi SSURGO map units and attach soil/septic attributes to parcel geometries."
    )
    parser.add_argument("--parcel-file", type=str, default="", help="Input parcel GeoPackage path.")
    parser.add_argument("--raw-dir", type=str, default=str(SOILS_RAW_DIR), help="Directory for raw SSURGO downloads.")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(SOILS_PROCESSED_DIR),
        help="Directory for processed soil layers.",
    )
    parser.add_argument("--output-file", type=str, default=str(OUTPUT_FILE), help="Output parcel GeoPackage path.")
    parser.add_argument("--summary-csv", type=str, default=str(SUMMARY_CSV), help="Summary CSV output path.")
    parser.add_argument("--parts-dir", type=str, default=str(PARTS_DIR), help="Per-county output directory.")
    parser.add_argument("--checkpoint-csv", type=str, default=str(CHECKPOINT_CSV), help="Checkpoint CSV path.")
    parser.add_argument("--counties", nargs="+", default=None, help="Optional county_name filters.")
    parser.add_argument("--chunk-size", type=int, default=5000, help="Parcels per county chunk.")
    parser.add_argument("--download-only", action="store_true", help="Only fetch and process statewide SSURGO assets.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloads and use local SSURGO files.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint data.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint state and recompute counties.")
    parser.add_argument("--allow-partial", action="store_true", help="Merge completed counties even if some counties fail.")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def choose_parcel_input(path_arg: str) -> Path:
    if path_arg:
        return resolve_path(path_arg)
    for candidate in INPUT_PARCEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return INPUT_PARCEL_CANDIDATES[0]


def sanitize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


def sql_quote(value: str) -> str:
    return str(value).replace("'", "''")


def normalize_to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(crs, allow_override=True)
    return gdf.to_crs(crs)


def load_checkpoint(path: Path) -> pd.DataFrame:
    cols = ["chunk_id", "status", "rows", "part_file"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=cols)
    if not set(cols).issubset(df.columns):
        return pd.DataFrame(columns=cols)
    return df[cols].copy()


def save_checkpoint(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def update_checkpoint(df: pd.DataFrame, chunk_id: str, status: str, rows: int, part_file: str) -> pd.DataFrame:
    row = pd.DataFrame([{"chunk_id": chunk_id, "status": status, "rows": rows, "part_file": part_file}])
    df = df[df["chunk_id"] != chunk_id].copy()
    return pd.concat([df, row], ignore_index=True)


def write_gpkg_with_retry(gdf: gpd.GeoDataFrame, path: Path, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            gdf.to_file(path, driver="GPKG", engine="pyogrio")
            return
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed writing GeoPackage: {path}") from last_exc


def request_text(url: str, payload: dict[str, object]) -> str:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Request failed ({attempt}/{MAX_RETRIES}) for {url}: {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    raise RuntimeError(f"Request failed: {url}") from last_error


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                with part_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                        if chunk:
                            handle.write(chunk)
            part_path.replace(dest)
            return
        except (RequestException, OSError) as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                break
            wait_s = RETRY_BACKOFF_SECONDS * attempt
            print(f"Download failed ({attempt}/{MAX_RETRIES}) for {url}: {exc}. Retrying in {wait_s:.1f}s.")
            time.sleep(wait_s)
    raise RuntimeError(f"Failed downloading {url}") from last_error


def parse_saverest_date(value: str) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid saverest date: {value}")
    return dt.strftime("%Y-%m-%d")


def fetch_survey_catalog(state_abbr: str, processed_dir: Path) -> pd.DataFrame:
    processed_dir.mkdir(parents=True, exist_ok=True)
    query = (
        "SELECT areasymbol, saverest "
        "FROM sacatalog "
        f"WHERE areasymbol LIKE '{state_abbr}%' "
        "ORDER BY areasymbol"
    )
    payload = {"format": "json", "query": query}
    payload_json = pd.read_json(io.StringIO(request_text(SDA_POST_URL, payload)), typ="series")
    rows = payload_json.get("Table", [])
    if not rows:
        raise RuntimeError(f"No SSURGO surveys returned for state prefix {state_abbr}.")
    df = pd.DataFrame(rows, columns=["areasymbol", "saverest"])
    df["save_date"] = df["saverest"].map(parse_saverest_date)
    df["zip_name"] = df.apply(
        lambda row: f"wss_SSA_{row['areasymbol']}_soildb_US_2003_[{row['save_date']}].zip",
        axis=1,
    )
    df["zip_url"] = df.apply(
        lambda row: WSS_ZIP_URL_TEMPLATE.format(areasymbol=row["areasymbol"], save_date=row["save_date"]),
        axis=1,
    )
    df.to_csv(processed_dir / SURVEY_CATALOG_CSV.name, index=False)
    return df


def ensure_survey_extracted(zip_path: Path, extract_root: Path, areasymbol: str) -> Path:
    survey_dir = extract_root / areasymbol
    spatial_dir = survey_dir / "spatial"
    tabular_dir = survey_dir / "tabular"
    shp = spatial_dir / f"soilmu_a_{areasymbol.lower()}.shp"
    required_tabular = [
        tabular_dir / "mstabcol.txt",
        tabular_dir / "muaggatt.txt",
        tabular_dir / "comp.txt",
        tabular_dir / "chorizon.txt",
        tabular_dir / "chtexgrp.txt",
    ]
    if shp.exists() and all(path.exists() for path in required_tabular):
        return survey_dir

    survey_dir.mkdir(parents=True, exist_ok=True)
    wanted = {
        f"{areasymbol.lower()}/spatial/soilmu_a_{areasymbol.lower()}.shp",
        f"{areasymbol.lower()}/spatial/soilmu_a_{areasymbol.lower()}.shx",
        f"{areasymbol.lower()}/spatial/soilmu_a_{areasymbol.lower()}.dbf",
        f"{areasymbol.lower()}/spatial/soilmu_a_{areasymbol.lower()}.prj",
        f"{areasymbol.lower()}/tabular/mstabcol.txt",
        f"{areasymbol.lower()}/tabular/muaggatt.txt",
        f"{areasymbol.lower()}/tabular/comp.txt",
        f"{areasymbol.lower()}/tabular/chorizon.txt",
        f"{areasymbol.lower()}/tabular/chtexgrp.txt",
    }
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.lower() in wanted:
                zf.extract(member, extract_root)
    return survey_dir


def load_table_columns(mstabcol_path: Path) -> dict[str, list[str]]:
    with mstabcol_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="|", quotechar='"')
        rows = [row for row in reader if row]
    column_map: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        column_map.setdefault(row[0], []).append((int(row[1]), row[2]))
    return {
        table_name: [column_name for _, column_name in sorted(entries, key=lambda item: item[0])]
        for table_name, entries in column_map.items()
    }


def read_ssurgo_table(
    survey_dir: Path,
    table_columns: dict[str, list[str]],
    table_name: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    file_name = TABLE_FILE_MAP[table_name]
    path = survey_dir / "tabular" / file_name
    names = table_columns[table_name]
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=names,
        usecols=usecols,
        dtype=str,
        quotechar='"',
        engine="python",
        keep_default_na=False,
    )
    for column in df.columns:
        df[column] = pd.Series(df[column], dtype="string").str.strip()
        df[column] = df[column].replace({"": pd.NA, "NULL": pd.NA})
    return df


def build_top_texture(chorizon: pd.DataFrame, chtexturegrp: pd.DataFrame) -> pd.DataFrame:
    hz = chorizon.copy()
    hz["hzdept_r_num"] = pd.to_numeric(hz["hzdept_r"], errors="coerce")
    hz["hzdepb_r_num"] = pd.to_numeric(hz["hzdepb_r"], errors="coerce")

    textures = chtexturegrp.copy()
    textures["rv_rank"] = textures["rvindicator"].fillna("").str.upper().eq("YES").astype(int)
    textures["soil_texture"] = textures["texdesc"].fillna(textures["texture"])
    textures = (
        textures.sort_values(["chkey", "rv_rank", "soil_texture"], ascending=[True, False, True])
        .drop_duplicates("chkey", keep="first")
        [["chkey", "soil_texture"]]
    )

    hz = hz.merge(textures, on="chkey", how="left")
    profile_depth = hz.groupby("cokey", as_index=False)["hzdepb_r_num"].max().rename(
        columns={"hzdepb_r_num": "profile_depth_cm"}
    )
    top = (
        hz.sort_values(["cokey", "hzdept_r_num", "hzdepb_r_num"], ascending=[True, True, True])
        .drop_duplicates("cokey", keep="first")
        [["cokey", "soil_texture"]]
    )
    return top.merge(profile_depth, on="cokey", how="left")


def build_dominant_component(component: pd.DataFrame, top_texture: pd.DataFrame) -> pd.DataFrame:
    comp = component.copy()
    comp["comppct_r_num"] = pd.to_numeric(comp["comppct_r"], errors="coerce").fillna(-1.0)
    comp["majcomp_rank"] = comp["majcompflag"].fillna("").str.upper().eq("YES").astype(int)
    comp = (
        comp.sort_values(
            ["mukey", "majcomp_rank", "comppct_r_num", "compname", "cokey"],
            ascending=[True, False, False, True, True],
        )
        .drop_duplicates("mukey", keep="first")
        [["mukey", "cokey", "compname", "drainagecl", "hydgrp"]]
    )
    return comp.merge(top_texture, on="cokey", how="left")


def compute_mapunit_septic_suitability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rating = out["engstafdcd"].fillna("").str.lower()
    drainage = out["drclassdcd"].fillna("").str.lower()
    hydgrp = out["hydgrpdcd"].fillna("").str.upper()
    water_depth = pd.to_numeric(out["wtdepaprjunmin"], errors="coerce").fillna(
        pd.to_numeric(out["wtdepannmin"], errors="coerce")
    )
    soil_depth = pd.to_numeric(out["soil_depth"], errors="coerce")

    extreme = (
        drainage.str.contains("very poorly")
        | (water_depth.notna() & (water_depth <= 25.0))
        | (soil_depth.notna() & (soil_depth <= 50.0))
        | hydgrp.isin(["D", "A/D", "B/D", "C/D"])
    )
    severe = extreme | drainage.str.contains("poorly") | (water_depth.notna() & (water_depth <= 50.0)) | (
        soil_depth.notna() & (soil_depth <= 100.0)
    )

    suitability = np.select(
        [
            rating.eq("not limited") & ~severe,
            rating.eq("not limited"),
            rating.eq("somewhat limited") & ~severe,
            rating.eq("somewhat limited"),
            rating.eq("very limited") & extreme,
            rating.eq("very limited"),
            extreme,
            severe,
        ],
        [
            "good",
            "moderate",
            "moderate",
            "poor",
            "very_poor",
            "poor",
            "very_poor",
            "poor",
        ],
        default="moderate",
    )

    out["septic_suitability"] = suitability
    out["mapunit_septic_score"] = pd.Series(suitability).map(
        {"good": 4.0, "moderate": 3.0, "poor": 2.0, "very_poor": 1.0}
    )
    out["septic_limitation_class"] = out["engstafdcd"].fillna("Unknown")
    return out


def build_mapunit_attributes_for_survey(survey_dir: Path) -> pd.DataFrame:
    table_columns = load_table_columns(survey_dir / "tabular" / "mstabcol.txt")
    muaggatt = read_ssurgo_table(
        survey_dir,
        table_columns,
        "muaggatt",
        usecols=[
            "mukey",
            "muname",
            "drclassdcd",
            "hydgrpdcd",
            "brockdepmin",
            "wtdepannmin",
            "wtdepaprjunmin",
            "engstafdcd",
            "engstafll",
            "engstafml",
        ],
    )
    component = read_ssurgo_table(
        survey_dir,
        table_columns,
        "component",
        usecols=["mukey", "cokey", "compname", "comppct_r", "drainagecl", "hydgrp", "majcompflag"],
    )
    chorizon = read_ssurgo_table(
        survey_dir,
        table_columns,
        "chorizon",
        usecols=["cokey", "chkey", "hzdept_r", "hzdepb_r"],
    )
    chtexturegrp = read_ssurgo_table(
        survey_dir,
        table_columns,
        "chtexturegrp",
        usecols=["chkey", "texture", "texdesc", "rvindicator"],
    )

    dominant_component = build_dominant_component(component, build_top_texture(chorizon, chtexturegrp))
    attrs = muaggatt.merge(
        dominant_component[["mukey", "compname", "drainagecl", "hydgrp", "soil_texture", "profile_depth_cm"]],
        on="mukey",
        how="left",
    )
    attrs["soil_depth"] = pd.to_numeric(attrs["brockdepmin"], errors="coerce")
    attrs["soil_depth"] = attrs["soil_depth"].fillna(pd.to_numeric(attrs["profile_depth_cm"], errors="coerce"))
    attrs["dominant_soil_type"] = attrs["muname"].fillna(attrs["compname"])
    attrs["drainage_class"] = attrs["drclassdcd"].fillna(attrs["drainagecl"])
    attrs["hydrologic_group"] = attrs["hydgrpdcd"].fillna(attrs["hydgrp"])
    attrs = compute_mapunit_septic_suitability(attrs)
    return attrs[
        [
            "mukey",
            "dominant_soil_type",
            "drainage_class",
            "hydrologic_group",
            "soil_texture",
            "soil_depth",
            "septic_limitation_class",
            "septic_suitability",
            "mapunit_septic_score",
        ]
    ].drop_duplicates("mukey", keep="first")


def build_processed_soils(raw_dir: Path, processed_dir: Path, skip_download: bool) -> tuple[Path, Path]:
    statewide_soils_gpkg = processed_dir / STATEWIDE_SOILS_GPKG.name
    mapunit_attr_csv = processed_dir / MAPUNIT_ATTR_CSV.name
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    extract_root = raw_dir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    if statewide_soils_gpkg.exists() and mapunit_attr_csv.exists():
        print(f"Using existing processed soils: {statewide_soils_gpkg}")
        return statewide_soils_gpkg, mapunit_attr_csv

    catalog = fetch_survey_catalog(STATE_ABBR, processed_dir)
    polygons: list[gpd.GeoDataFrame] = []
    attrs: list[pd.DataFrame] = []

    print(f"Preparing {len(catalog):,} SSURGO survey areas for {STATE_NAME}.")
    for idx, row in catalog.iterrows():
        areasymbol = str(row["areasymbol"])
        zip_path = raw_dir / str(row["zip_name"])
        if not zip_path.exists():
            if skip_download:
                raise FileNotFoundError(f"Missing SSURGO ZIP and --skip-download set: {zip_path}")
            print(f"[{idx + 1}/{len(catalog)}] Downloading {areasymbol} -> {zip_path.name}")
            download_file(str(row["zip_url"]), zip_path)
        survey_dir = ensure_survey_extracted(zip_path, extract_root, areasymbol)

        shp_path = survey_dir / "spatial" / f"soilmu_a_{areasymbol.lower()}.shp"
        soil_polygons = gpd.read_file(shp_path, engine="pyogrio")
        soil_polygons = normalize_to_crs(soil_polygons, TARGET_CRS)
        soil_polygons = soil_polygons.rename(columns=str.lower)[["areasymbol", "musym", "mukey", "geometry"]].copy()
        polygons.append(soil_polygons)
        attrs.append(build_mapunit_attributes_for_survey(survey_dir))

    statewide = gpd.GeoDataFrame(pd.concat(polygons, ignore_index=True), geometry="geometry", crs=TARGET_CRS)
    write_gpkg_with_retry(statewide, statewide_soils_gpkg)
    attr_df = pd.concat(attrs, ignore_index=True).drop_duplicates("mukey", keep="first")
    attr_df.to_csv(mapunit_attr_csv, index=False)
    return statewide_soils_gpkg, mapunit_attr_csv


def load_county_index(parcel_file: Path, counties: list[str] | None) -> list[str]:
    index_df = gpd.read_file(parcel_file, columns=["county_name"], ignore_geometry=True, engine="pyogrio")
    county_values = sorted(index_df["county_name"].astype(str).unique().tolist())
    if counties:
        wanted = {str(c).strip().lower() for c in counties}
        county_values = [c for c in county_values if c.lower() in wanted]
    return county_values


def read_county_parcels(parcel_file: Path, county_name: str) -> gpd.GeoDataFrame:
    where = f"county_name = '{sql_quote(county_name)}'"
    gdf = gpd.read_file(parcel_file, where=where, engine="pyogrio")
    if gdf.crs is None:
        return gdf.set_crs(TARGET_CRS, allow_override=True)
    return gdf.to_crs(TARGET_CRS)


def read_soils_subset(soils_file: Path, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(soils_file, bbox=bounds, engine="pyogrio")
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(TARGET_CRS, allow_override=True)
    return gdf.to_crs(TARGET_CRS)


def aggregate_chunk_soils(chunk: gpd.GeoDataFrame, soils_file: Path, attrs_df: pd.DataFrame) -> pd.DataFrame:
    soils = read_soils_subset(soils_file, tuple(float(value) for value in chunk.total_bounds))
    if soils.empty:
        return pd.DataFrame({"parcel_row_id": chunk["parcel_row_id"]})

    soils = soils.rename(columns=str.lower).merge(attrs_df, on="mukey", how="left")
    soils = soils[
        [
            "mukey",
            "dominant_soil_type",
            "drainage_class",
            "hydrologic_group",
            "soil_texture",
            "soil_depth",
            "septic_limitation_class",
            "septic_suitability",
            "mapunit_septic_score",
            "geometry",
        ]
    ].copy()

    chunk_area = normalize_to_crs(chunk[["parcel_row_id", "geometry"]].copy(), AREA_CRS)
    soils_area = normalize_to_crs(soils, AREA_CRS)
    intersections = gpd.overlay(chunk_area, soils_area, how="intersection", keep_geom_type=False)
    if intersections.empty:
        return pd.DataFrame({"parcel_row_id": chunk["parcel_row_id"]})

    intersections["overlap_sqm"] = intersections.geometry.area
    intersections = intersections[intersections["overlap_sqm"] > 0].copy()
    if intersections.empty:
        return pd.DataFrame({"parcel_row_id": chunk["parcel_row_id"]})

    dominant = (
        intersections.sort_values(["parcel_row_id", "overlap_sqm"], ascending=[True, False])
        .drop_duplicates("parcel_row_id", keep="first")
        [
            [
                "parcel_row_id",
                "dominant_soil_type",
                "drainage_class",
                "hydrologic_group",
                "soil_texture",
                "soil_depth",
                "septic_limitation_class",
                "septic_suitability",
                "mapunit_septic_score",
            ]
        ]
        .rename(columns={"soil_depth": "dominant_soil_depth"})
    )

    numeric = intersections[["parcel_row_id", "overlap_sqm", "soil_depth", "mapunit_septic_score"]].copy()
    numeric["soil_depth_num"] = pd.to_numeric(numeric["soil_depth"], errors="coerce")
    numeric["septic_score_num"] = pd.to_numeric(numeric["mapunit_septic_score"], errors="coerce")
    numeric["weighted_depth"] = numeric["overlap_sqm"] * numeric["soil_depth_num"]
    numeric["weighted_septic"] = numeric["overlap_sqm"] * numeric["septic_score_num"]
    numeric["depth_weight"] = numeric["overlap_sqm"] * numeric["soil_depth_num"].notna().astype(float)
    numeric["septic_weight"] = numeric["overlap_sqm"] * numeric["septic_score_num"].notna().astype(float)

    aggregated = numeric.groupby("parcel_row_id", as_index=False).agg(
        weighted_depth=("weighted_depth", "sum"),
        depth_weight=("depth_weight", "sum"),
        weighted_septic=("weighted_septic", "sum"),
        septic_weight=("septic_weight", "sum"),
    )
    aggregated["soil_depth"] = aggregated["weighted_depth"] / aggregated["depth_weight"].replace(0.0, np.nan)
    aggregated["weighted_septic_score"] = aggregated["weighted_septic"] / aggregated["septic_weight"].replace(0.0, np.nan)
    aggregated["weighted_septic_score"] = aggregated["weighted_septic_score"].fillna(2.0)
    aggregated["septic_suitability_weighted"] = np.select(
        [
            aggregated["weighted_septic_score"] >= 3.5,
            aggregated["weighted_septic_score"] >= 2.5,
            aggregated["weighted_septic_score"] >= 1.5,
        ],
        ["good", "moderate", "poor"],
        default="very_poor",
    )
    aggregated = aggregated[["parcel_row_id", "soil_depth", "septic_suitability_weighted"]]

    metrics = dominant.merge(aggregated, on="parcel_row_id", how="left")
    metrics["soil_depth"] = metrics["soil_depth"].fillna(metrics["dominant_soil_depth"])
    metrics["septic_suitability"] = metrics["septic_suitability_weighted"].fillna(metrics["septic_suitability"])
    return metrics[
        [
            "parcel_row_id",
            "dominant_soil_type",
            "drainage_class",
            "hydrologic_group",
            "soil_texture",
            "soil_depth",
            "septic_suitability",
            "septic_limitation_class",
        ]
    ]


def score_county_soils(
    county_parcels: gpd.GeoDataFrame,
    soils_file: Path,
    attrs_df: pd.DataFrame,
    chunk_size: int,
) -> gpd.GeoDataFrame:
    results: list[gpd.GeoDataFrame] = []
    total_chunks = max((len(county_parcels) + chunk_size - 1) // chunk_size, 1)
    for idx, start in enumerate(range(0, len(county_parcels), chunk_size), start=1):
        chunk = county_parcels.iloc[start : start + chunk_size].copy()
        print(f"    Chunk {idx}/{total_chunks}: {len(chunk):,} parcels")
        chunk = chunk.merge(aggregate_chunk_soils(chunk, soils_file, attrs_df), on="parcel_row_id", how="left")
        results.append(chunk)
    return gpd.GeoDataFrame(pd.concat(results, ignore_index=True), geometry="geometry", crs=county_parcels.crs)


def merge_parts(
    counties: list[str],
    parts_dir: Path,
    checkpoint_df: pd.DataFrame,
    output_file: Path,
    allow_partial: bool,
) -> None:
    expected = {sanitize_name(county): county for county in counties}
    completed = checkpoint_df[checkpoint_df["status"] == "completed"].copy()
    completed["county_slug"] = completed["chunk_id"].astype(str).map(sanitize_name)
    done = set(completed["county_slug"].tolist())
    missing = sorted(set(expected) - done)
    if missing and not allow_partial:
        raise RuntimeError(f"Cannot merge statewide soils output; missing counties: {', '.join(missing[:10])}")

    frames: list[gpd.GeoDataFrame] = []
    for county_slug in sorted(done):
        part_path = parts_dir / f"{county_slug}_with_soils.gpkg"
        if not part_path.exists():
            if allow_partial:
                continue
            raise FileNotFoundError(f"Missing county part file: {part_path}")
        frames.append(gpd.read_file(part_path, engine="pyogrio"))
    if not frames:
        raise RuntimeError("No county soil parts available to merge.")

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    write_gpkg_with_retry(merged, output_file)


def build_summary(parcels: gpd.GeoDataFrame) -> pd.DataFrame:
    parcel_count = int(len(parcels))
    septic_pct = parcels["septic_suitability"].fillna("unknown").value_counts(dropna=False, normalize=True).mul(100.0).round(4)
    soil_counts = parcels["dominant_soil_type"].fillna("unknown").value_counts(dropna=False).head(10)
    rows = [
        {"metric": "parcel_count", "value": parcel_count},
        {"metric": "percent_good_septic", "value": float(septic_pct.get("good", 0.0))},
        {"metric": "percent_moderate_septic", "value": float(septic_pct.get("moderate", 0.0))},
        {"metric": "percent_poor_septic", "value": float(septic_pct.get("poor", 0.0))},
        {"metric": "percent_very_poor_septic", "value": float(septic_pct.get("very_poor", 0.0))},
        {"metric": "soil_field_population_pct", "value": float(parcels["dominant_soil_type"].notna().mean() * 100.0)},
    ]
    for soil_name, count in soil_counts.items():
        rows.append({"metric": f"most_common_soil_type::{soil_name}", "value": int(count)})
    return pd.DataFrame(rows)


def print_validation(parcels: gpd.GeoDataFrame) -> None:
    parcel_count = len(parcels)
    septic_counts = parcels["septic_suitability"].fillna("unknown").value_counts(dropna=False)
    soil_population_pct = parcels["dominant_soil_type"].notna().mean() * 100.0
    top_soils = parcels["dominant_soil_type"].fillna("unknown").value_counts(dropna=False).head(10)

    print("\nSoils validation")
    print(f"Parcel rows: {parcel_count:,}")
    print(f"Soil field population: {soil_population_pct:.4f}%")
    for label in ["good", "moderate", "poor", "very_poor", "unknown"]:
        count = int(septic_counts.get(label, 0))
        pct = (count / parcel_count * 100.0) if parcel_count else 0.0
        print(f"Septic {label}: {count:,} ({pct:.4f}%)")

    print("Most common soil types:")
    for soil_name, count in top_soils.items():
        print(f"  {soil_name}: {count:,}")

    sample_cols = [
        "county_name",
        "parcel_row_id",
        "dominant_soil_type",
        "drainage_class",
        "hydrologic_group",
        "soil_texture",
        "soil_depth",
        "septic_suitability",
        "septic_limitation_class",
    ]
    examples = parcels.loc[parcels["dominant_soil_type"].notna(), sample_cols].head(12)
    print("Example soil rows:")
    if examples.empty:
        print("  No populated soil rows found.")
    else:
        print(examples.to_string(index=False))


def main() -> None:
    args = parse_args()
    started_at = time.time()

    parcel_file = choose_parcel_input(args.parcel_file)
    raw_dir = resolve_path(args.raw_dir)
    processed_dir = resolve_path(args.processed_dir)
    output_file = resolve_path(args.output_file)
    summary_csv = resolve_path(args.summary_csv)
    parts_dir = resolve_path(args.parts_dir)
    checkpoint_csv = resolve_path(args.checkpoint_csv)

    soils_file, attrs_file = build_processed_soils(raw_dir, processed_dir, skip_download=args.skip_download)
    print(f"Processed soils layer: {soils_file}")
    print(f"Mapunit attributes: {attrs_file}")

    if args.download_only:
        print(f"Download/process-only complete in {(time.time() - started_at) / 60.0:.2f} minutes.")
        return

    if not parcel_file.exists():
        raise FileNotFoundError(f"Missing parcel input: {parcel_file}")

    attrs_df = pd.read_csv(attrs_file, dtype={"mukey": "string"})
    attrs_df["mukey"] = pd.Series(attrs_df["mukey"], dtype="string").str.strip()

    counties = load_county_index(parcel_file, args.counties)
    checkpoint_df = load_checkpoint(checkpoint_csv)
    resume_enabled = args.resume and not args.no_resume

    print(f"Parcel input: {parcel_file}")
    print(f"County count: {len(counties):,}")

    for county_name in counties:
        county_slug = sanitize_name(county_name)
        part_path = parts_dir / f"{county_slug}_with_soils.gpkg"
        existing = checkpoint_df[checkpoint_df["chunk_id"] == county_slug]
        if resume_enabled and not existing.empty and part_path.exists():
            try:
                expected_rows = int(existing.iloc[0]["rows"])
                part_rows = len(gpd.read_file(part_path, ignore_geometry=True, engine="pyogrio"))
                if str(existing.iloc[0]["status"]).lower() == "completed" and part_rows == expected_rows:
                    print(f"Skipping completed county: {county_name} ({part_rows:,} rows)")
                    continue
            except Exception:
                pass

        print(f"Processing county: {county_name}")
        county_out = score_county_soils(read_county_parcels(parcel_file, county_name), soils_file, attrs_df, chunk_size=args.chunk_size)
        write_gpkg_with_retry(county_out, part_path)
        checkpoint_df = update_checkpoint(checkpoint_df, county_slug, "completed", len(county_out), str(part_path))
        save_checkpoint(checkpoint_csv, checkpoint_df)
        print(f"Completed county: {county_name} ({len(county_out):,} rows)")

    merge_parts(counties, parts_dir, checkpoint_df, output_file, allow_partial=args.allow_partial)
    parcels = gpd.read_file(output_file, engine="pyogrio")
    summary = build_summary(parcels)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print_validation(parcels)

    print(f"Output file: {output_file}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Runtime: {(time.time() - started_at) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()
