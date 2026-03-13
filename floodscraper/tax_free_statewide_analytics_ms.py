from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
TAX_METADATA_DIR = BASE_DIR / "data" / "tax_metadata"
TAX_LINKED_DIR = BASE_DIR / "data" / "tax_linked" / "ms"


def load_metric_lookup(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "metric" not in frame.columns:
        return {}
    return dict(zip(frame["metric"].astype("string"), frame["value"]))


def build_overview() -> pd.DataFrame:
    metric_files = {
        "pike": TAX_METADATA_DIR / "tax_free_ingest_summary_ms.csv",
        "sos_statewide": TAX_METADATA_DIR / "tax_free_sos_ingest_summary_ms.csv",
        "hinds": TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv",
        "warren": TAX_METADATA_DIR / "tax_free_warren_linkage_summary_ms.csv",
        "madison": TAX_METADATA_DIR / "tax_free_madison_linkage_summary_ms.csv",
    }
    lookups = {name: load_metric_lookup(path) for name, path in metric_files.items()}
    linked_frames = []
    linked_paths = {
        "pike": TAX_LINKED_DIR / "pike" / "free_linked_tax_records.parquet",
        "sos_statewide": TAX_LINKED_DIR / "statewide" / "sos_forfeited_linked_tax_records.parquet",
        "warren": TAX_LINKED_DIR / "warren" / "warren_linked_tax_records.parquet",
        "madison": TAX_LINKED_DIR / "madison" / "madison_linked_tax_records.parquet",
    }
    for source_name, path in linked_paths.items():
        if path.exists():
            df = pd.read_parquet(path, columns=["parcel_row_id", "county_name", "delinquent_amount"])
            df["source_name"] = source_name
            linked_frames.append(df)
    all_linked = pd.concat(linked_frames, ignore_index=True) if linked_frames else pd.DataFrame(columns=["parcel_row_id", "county_name", "delinquent_amount", "source_name"])
    available_delinquent_total = float(pd.to_numeric(all_linked["delinquent_amount"], errors="coerce").fillna(0).sum()) if not all_linked.empty else 0.0
    return pd.DataFrame(
        [
            {"metric": "free_source_count", "value": 5},
            {
                "metric": "cumulative_standardized_rows",
                "value": sum(
                    float(lookups[name].get(key, 0.0))
                    for name, key in [
                        ("pike", "pike_raw_rows"),
                        ("sos_statewide", "sos_standardized_rows"),
                        ("hinds", "hinds_standardized_rows"),
                        ("warren", "warren_standardized_rows"),
                        ("madison", "madison_standardized_rows"),
                    ]
                ),
            },
            {"metric": "cumulative_linked_rows", "value": float(len(all_linked))},
            {"metric": "cumulative_unique_linked_parcel_rows", "value": float(all_linked["parcel_row_id"].astype("string").nunique()) if not all_linked.empty else 0.0},
            {"metric": "counties_with_linked_free_tax_rows", "value": float(all_linked["county_name"].astype("string").nunique()) if not all_linked.empty else 0.0},
            {"metric": "cumulative_available_delinquent_dollars", "value": round(available_delinquent_total, 2)},
        ]
    )


def build_source_type_performance() -> pd.DataFrame:
    rows = [
        {"source_name": "pike", "source_type": "county_csv_download", "summary_path": TAX_METADATA_DIR / "tax_free_ingest_summary_ms.csv", "rate_key": "pike_linkage_rate", "rows_key": "pike_raw_rows"},
        {"source_name": "madison", "source_type": "county_xlsx_download", "summary_path": TAX_METADATA_DIR / "tax_free_madison_linkage_summary_ms.csv", "rate_key": "madison_linkage_rate", "rows_key": "madison_standardized_rows"},
        {"source_name": "hinds", "source_type": "county_xlsx_download", "summary_path": TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv", "rate_key": "hinds_linkage_rate", "rows_key": "hinds_standardized_rows"},
        {"source_name": "warren", "source_type": "county_html_listing", "summary_path": TAX_METADATA_DIR / "tax_free_warren_linkage_summary_ms.csv", "rate_key": "warren_linkage_rate", "rows_key": "warren_standardized_rows"},
        {"source_name": "sos_statewide", "source_type": "statewide_arcgis_inventory", "summary_path": TAX_METADATA_DIR / "tax_free_sos_ingest_summary_ms.csv", "rate_key": "sos_linkage_rate", "rows_key": "sos_standardized_rows"},
    ]
    out = []
    for row in rows:
        lookup = load_metric_lookup(row["summary_path"])
        out.append(
            {
                "source_name": row["source_name"],
                "source_type": row["source_type"],
                "standardized_rows": lookup.get(row["rows_key"], pd.NA),
                "linkage_rate": lookup.get(row["rate_key"], pd.NA),
            }
        )
    return pd.DataFrame(out).sort_values(["linkage_rate"], ascending=False).reset_index(drop=True)


def build_county_performance() -> pd.DataFrame:
    paths = [
        TAX_METADATA_DIR / "tax_free_sos_county_summary_ms.csv",
        TAX_METADATA_DIR / "tax_free_warren_linkage_summary_ms.csv",
        TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv",
        TAX_METADATA_DIR / "tax_free_madison_linkage_summary_ms.csv",
    ]
    rows: list[dict[str, Any]] = []
    sos_path = TAX_METADATA_DIR / "tax_free_sos_county_summary_ms.csv"
    if sos_path.exists():
        sos = pd.read_csv(sos_path)
        rows.extend(sos.loc[:, ["county_name", "county_fips", "standardized_rows", "linked_rows", "unmatched_rows", "ambiguous_rows", "linkage_rate"]].to_dict("records"))
    for county_name, path, prefix in [
        ("warren", TAX_METADATA_DIR / "tax_free_warren_linkage_summary_ms.csv", "warren"),
        ("hinds", TAX_METADATA_DIR / "tax_free_hinds_linkage_summary_ms.csv", "hinds"),
        ("madison", TAX_METADATA_DIR / "tax_free_madison_linkage_summary_ms.csv", "madison"),
    ]:
        if path.exists():
            lookup = load_metric_lookup(path)
            rows.append(
                {
                    "county_name": county_name,
                    "county_fips": {"warren": "149", "hinds": "049", "madison": "089"}[county_name],
                    "standardized_rows": lookup.get(f"{prefix}_standardized_rows", pd.NA),
                    "linked_rows": lookup.get(f"{prefix}_linked_rows", pd.NA),
                    "unmatched_rows": lookup.get(f"{prefix}_unmatched_rows", pd.NA),
                    "ambiguous_rows": lookup.get(f"{prefix}_ambiguous_rows", pd.NA),
                    "linkage_rate": lookup.get(f"{prefix}_linkage_rate", pd.NA),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=["county_name"], keep="last").sort_values(["linkage_rate"], ascending=False).reset_index(drop=True)


def main() -> None:
    overview_path = TAX_METADATA_DIR / "tax_free_statewide_overview_ms.csv"
    county_path = TAX_METADATA_DIR / "tax_free_county_linkage_performance_ms.csv"
    source_type_path = TAX_METADATA_DIR / "tax_free_source_type_performance_ms.csv"
    build_overview().to_csv(overview_path, index=False)
    build_county_performance().to_csv(county_path, index=False)
    build_source_type_performance().to_csv(source_type_path, index=False)
    print(f"Overview: {overview_path.relative_to(BASE_DIR)}")
    print(f"County performance: {county_path.relative_to(BASE_DIR)}")
    print(f"Source type performance: {source_type_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
