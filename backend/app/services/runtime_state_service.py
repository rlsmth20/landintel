from __future__ import annotations

import json
import math
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import mapbox_vector_tile
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import requests
from shapely import wkb
from shapely.geometry import mapping, shape

from app.bootstrap import ensure_repo_root_on_path

PROJECT_ROOT = ensure_repo_root_on_path()

from floodscraper.parcel_contract_ms import (  # noqa: E402
    API_LEADS_SUMMARY_FIELDS,
    BACKEND_DETAIL_REQUIRED_FIELDS,
    GEOMETRY_FEATURE_PROPERTY_FIELDS,
    GEOMETRY_ITEM_FIELDS,
    NEARBY_COMP_OUTPUT_FIELDS,
    SEARCH_OUTPUT_FIELDS,
    SEARCH_SOURCE_FIELDS,
    serialize_contract_row,
    validate_output_records,
)
from floodscraper.state_artifacts import load_state_artifacts  # noqa: E402
from floodscraper.state_registry import load_state_definition  # noqa: E402


EARTH_RADIUS_MILES = 3958.7613
DEFAULT_NEARBY_RADIUS_MILES = 3.0
logger = logging.getLogger("state-geometry")


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    stripped = str(value).strip()
    return stripped or None


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _slug_county_name(value: Any) -> str | None:
    normalized = _normalize_string(value)
    if not normalized:
        return None
    return normalized.lower().replace(".", "").replace("&", "and").replace(" ", "_")


def _point_geometry_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    longitude = row.get("longitude")
    latitude = row.get("latitude")
    if longitude is None or latitude is None:
        return None
    try:
        longitude = float(longitude)
        latitude = float(latitude)
    except Exception:
        return None
    if not np.isfinite(longitude) or not np.isfinite(latitude):
        return None
    point = {
        "type": "Point",
        "coordinates": [round(longitude, 6), round(latitude, 6)],
    }
    return {
        "type": "point_reference",
        "centroid": point,
        "bounds": None,
    }


def _haversine_miles(
    latitude: float,
    longitude: float,
    latitudes: pd.Series,
    longitudes: pd.Series,
) -> pd.Series:
    lat1_r = math.radians(latitude)
    lon1_r = math.radians(longitude)
    lat2_r = np.radians(pd.to_numeric(latitudes, errors="coerce"))
    lon2_r = np.radians(pd.to_numeric(longitudes, errors="coerce"))
    delta_lat = lat2_r - lat1_r
    delta_lon = lon2_r - lon1_r
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(delta_lon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return pd.Series(EARTH_RADIUS_MILES * c, index=latitudes.index, dtype="float64")


class RuntimeStateService:
    def __init__(self, state_code: str) -> None:
        self.state_code = state_code.strip().lower()
        self.definition = load_state_definition(self.state_code)
        self.artifacts = load_state_artifacts(self.state_code)
        self._http = requests.Session()

    @property
    def state_name(self) -> str:
        return self.definition.state_name

    def _read_json_artifact(self, path: Path, *, artifact: str) -> Any | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception(
                "State artifact load failed state=%s artifact=%s path=%s",
                self.state_code,
                artifact,
                path,
            )
            return None

    def _state_summary_unavailable_payload(self) -> dict[str, Any]:
        return {
            "row_count": 0,
            "source": f"{self.state_name} runtime artifacts unavailable",
            "geometry_mode": "selected_parcel_geojson",
            "sections": {
                "statewide": [],
                "top_counties": [],
                "recommended_view_bucket": [],
            },
        }

    def _is_usable_summary_payload(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        sections = payload.get("sections")
        if not isinstance(sections, dict):
            return False
        if any(isinstance(items, list) and items for items in sections.values()):
            return True
        try:
            row_count = int(payload.get("row_count") or 0)
        except Exception:
            row_count = 0
        if row_count > 0:
            return True
        source = (_normalize_string(payload.get("source")) or "").lower()
        return bool(source) and "unavailable" not in source

    @lru_cache(maxsize=1)
    def _frontend_meta_payload(self) -> dict[str, Any] | None:
        payload = self._read_json_artifact(self.artifacts.frontend_meta_path, artifact="frontend_meta")
        return payload if isinstance(payload, dict) else None

    def _summary_from_meta_payload(self) -> dict[str, Any] | None:
        payload = self._frontend_meta_payload()
        if not isinstance(payload, dict):
            return None
        rows = payload.get("summary")
        if not isinstance(rows, list) or not rows:
            return None
        sections: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            section = _normalize_string(row.get("section")) or "statewide"
            sections.setdefault(section, []).append(
                {
                    "section": section,
                    "metric": _normalize_string(row.get("metric")),
                    "key": _normalize_string(row.get("key")),
                    "value": "" if row.get("value") is None else str(row.get("value")),
                }
            )
        if not sections:
            return None
        try:
            row_count = int(payload.get("rowCount") or 0)
        except Exception:
            row_count = 0
        source_label = _normalize_string(payload.get("source"))
        if source_label and "frontend meta fallback" not in source_label.lower():
            source_label = f"{source_label} (frontend meta fallback)"
        return {
            "row_count": row_count,
            "source": source_label or f"{self.state_name} frontend meta fallback",
            "geometry_mode": _normalize_string(payload.get("geometryMode")) or "selected_parcel_geojson",
            "sections": sections,
        }

    def _presets_from_meta_payload(self) -> list[dict[str, Any]]:
        payload = self._frontend_meta_payload()
        if not isinstance(payload, dict):
            return []
        default_views = payload.get("defaultViews")
        if not isinstance(default_views, list):
            return []
        grouped: dict[str, dict[str, Any]] = {}
        for item in default_views:
            if not isinstance(item, dict):
                continue
            view_name = _normalize_string(item.get("view_name"))
            if not view_name:
                continue
            current = grouped.setdefault(
                view_name,
                {
                    "view_name": view_name,
                    "description": _normalize_string(item.get("description")),
                    "filter_expression": _normalize_string(item.get("filter_expression")),
                },
            )
            metric = _normalize_string(item.get("metric"))
            value = "" if item.get("value") is None else str(item.get("value"))
            if metric == "row_count":
                current["row_count"] = value
            elif metric == "average_lead_score":
                current["average_lead_score"] = value
        return list(grouped.values())

    def _normalize_leads_payload(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, list):
            items_payload = payload
            total_count = len(items_payload)
            limit = len(items_payload)
            offset = 0
        elif isinstance(payload, dict):
            items_payload = payload.get("items")
            if not isinstance(items_payload, list):
                return None
            try:
                total_count = int(payload.get("total_count") or len(items_payload))
            except Exception:
                total_count = len(items_payload)
            try:
                limit = int(payload.get("limit") or len(items_payload))
            except Exception:
                limit = len(items_payload)
            try:
                offset = int(payload.get("offset") or 0)
            except Exception:
                offset = 0
        else:
            return None
        records = [
            serialize_contract_row(item, API_LEADS_SUMMARY_FIELDS, serializer=_json_scalar)
            for item in items_payload
            if isinstance(item, Mapping)
        ]
        validate_output_records(
            records,
            expected_fields=API_LEADS_SUMMARY_FIELDS,
            required_fields=["parcel_row_id", "parcel_id", "county_name"],
            non_null_fields=["parcel_row_id", "parcel_id"],
            context=f"runtime_state_service[{self.state_code}].normalize_leads_payload",
        )
        return {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "items": records,
        }

    @lru_cache(maxsize=1)
    def _frontend_static_feed_payload(self) -> dict[str, Any] | None:
        payload = self._read_json_artifact(self.artifacts.frontend_static_feed_path, artifact="frontend_static_feed")
        normalized = self._normalize_leads_payload(payload)
        return normalized if normalized is not None else None

    @lru_cache(maxsize=1)
    def _frontend_detail_fallback_payload(self) -> list[dict[str, Any]]:
        payload = self._read_json_artifact(self.artifacts.frontend_detail_fallback_path, artifact="frontend_detail_fallback")
        return payload if isinstance(payload, list) else []

    @lru_cache(maxsize=1)
    def _frontend_static_feed_frame(self) -> pd.DataFrame:
        payload = self._frontend_static_feed_payload()
        items = payload.get("items", []) if isinstance(payload, dict) else []
        frame = pd.DataFrame(items)
        if "parcel_row_id" in frame.columns:
            frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
        return frame

    @lru_cache(maxsize=1)
    def _frontend_detail_fallback_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self._frontend_detail_fallback_payload())
        if "parcel_row_id" in frame.columns:
            frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
        return frame

    @lru_cache(maxsize=1)
    def _fallback_query_frame(self) -> pd.DataFrame:
        detail_frame = self._frontend_detail_fallback_frame()
        if not detail_frame.empty:
            return detail_frame
        return self._frontend_static_feed_frame()

    def _fallback_frame_row(self, parcel_row_id: str) -> pd.Series | None:
        for frame in (self._frontend_detail_fallback_frame(), self._frontend_static_feed_frame()):
            if frame.empty or "parcel_row_id" not in frame.columns:
                continue
            matched = frame.loc[frame["parcel_row_id"].astype("string") == parcel_row_id]
            if not matched.empty:
                return matched.iloc[0]
        return None

    @lru_cache(maxsize=1)
    def _summary_payload(self) -> dict[str, Any]:
        payload = self._read_json_artifact(self.artifacts.runtime_summary_path, artifact="runtime_summary")
        if self._is_usable_summary_payload(payload):
            return payload
        fallback = self._summary_from_meta_payload()
        if fallback is not None:
            logger.warning(
                "State summary fallback activated state=%s runtime_path=%s meta_path=%s",
                self.state_code,
                self.artifacts.runtime_summary_path,
                self.artifacts.frontend_meta_path,
            )
            return fallback
        logger.warning(
            "State summary unavailable state=%s runtime_path=%s meta_path=%s",
            self.state_code,
            self.artifacts.runtime_summary_path,
            self.artifacts.frontend_meta_path,
        )
        return self._state_summary_unavailable_payload()

    @lru_cache(maxsize=1)
    def _presets_payload(self) -> list[dict[str, Any]]:
        payload = self._read_json_artifact(self.artifacts.runtime_presets_path, artifact="runtime_presets")
        if isinstance(payload, list) and payload:
            return payload
        fallback = self._presets_from_meta_payload()
        if fallback:
            logger.warning(
                "State presets fallback activated state=%s runtime_path=%s meta_path=%s",
                self.state_code,
                self.artifacts.runtime_presets_path,
                self.artifacts.frontend_meta_path,
            )
            return fallback
        return []

    @lru_cache(maxsize=1)
    def _default_leads_payload(self) -> dict[str, Any]:
        runtime_payload = self._normalize_leads_payload(
            self._read_json_artifact(self.artifacts.runtime_default_leads_path, artifact="runtime_default_leads")
        )
        if runtime_payload is not None and runtime_payload.get("items"):
            return runtime_payload
        static_payload = self._frontend_static_feed_payload()
        if static_payload is not None and static_payload.get("items"):
            logger.warning(
                "State leads fallback activated state=%s source=frontend_static_feed runtime_path=%s static_path=%s",
                self.state_code,
                self.artifacts.runtime_default_leads_path,
                self.artifacts.frontend_static_feed_path,
            )
            return static_payload
        detail_frame = self._frontend_detail_fallback_frame()
        if not detail_frame.empty:
            records = [
                serialize_contract_row(row.to_dict(), API_LEADS_SUMMARY_FIELDS, serializer=_json_scalar)
                for _, row in detail_frame.head(200).iterrows()
            ]
            validate_output_records(
                records,
                expected_fields=API_LEADS_SUMMARY_FIELDS,
                required_fields=["parcel_row_id", "parcel_id", "county_name"],
                non_null_fields=["parcel_row_id", "parcel_id"],
                context=f"runtime_state_service[{self.state_code}].default_leads_detail_fallback",
            )
            logger.warning(
                "State leads fallback activated state=%s source=frontend_detail_fallback detail_path=%s",
                self.state_code,
                self.artifacts.frontend_detail_fallback_path,
            )
            return {
                "total_count": int(len(detail_frame)),
                "limit": min(200, int(len(detail_frame))),
                "offset": 0,
                "items": records,
            }
        raise FileNotFoundError(
            "State leads artifacts unavailable: "
            f"runtime_default={self.artifacts.runtime_default_leads_path}; "
            f"static_feed={self.artifacts.frontend_static_feed_path}; "
            f"detail_fallback={self.artifacts.frontend_detail_fallback_path}"
        )

    @lru_cache(maxsize=1)
    def _app_ready_dataset(self) -> ds.Dataset:
        if not self.artifacts.app_ready_path.exists():
            raise FileNotFoundError(f"State app-ready dataset not found: {self.artifacts.app_ready_path}")
        return ds.dataset(self.artifacts.app_ready_path, format="parquet")

    @lru_cache(maxsize=1)
    def _detail_dataset(self) -> ds.Dataset:
        target = self.artifacts.runtime_detail_metrics_path if self.artifacts.runtime_detail_metrics_path.exists() else self.artifacts.app_ready_path
        return ds.dataset(target, format="parquet")

    @lru_cache(maxsize=1)
    def _parcel_master_dataset(self) -> ds.Dataset:
        if not self.artifacts.parcel_master_path.exists():
            raise FileNotFoundError(f"State parcel master dataset not found: {self.artifacts.parcel_master_path}")
        return ds.dataset(self.artifacts.parcel_master_path, format="parquet")

    @lru_cache(maxsize=1)
    def _search_frame(self) -> pd.DataFrame:
        if self.artifacts.app_ready_path.exists():
            dataset = self._app_ready_dataset()
            available_columns = [column for column in SEARCH_SOURCE_FIELDS + ["latitude", "longitude"] if column in dataset.schema.names]
            frame = dataset.to_table(columns=available_columns).to_pandas()
        else:
            fallback = self._fallback_query_frame()
            available_columns = [column for column in SEARCH_SOURCE_FIELDS + ["latitude", "longitude"] if column in fallback.columns]
            frame = fallback.loc[:, available_columns].copy() if available_columns else pd.DataFrame(columns=SEARCH_SOURCE_FIELDS)
        if "parcel_row_id" in frame.columns:
            frame["parcel_row_id"] = frame["parcel_row_id"].astype("string")
        return frame

    @lru_cache(maxsize=1)
    def _parcel_source_registry(self) -> dict[str, Any]:
        registry_path = self.definition.source_registry_path("parcel_source")
        if registry_path is None or not registry_path.exists():
            return {}
        return json.loads(registry_path.read_text(encoding="utf-8"))

    def _geometry_cache_path(self) -> Path | None:
        configured = self.definition.raw.get("parcel_tiles", {}).get("geometry_cache_path")
        if not configured:
            return None
        candidate = Path(str(configured))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate

    @lru_cache(maxsize=1)
    def _geometry_cache_dataset(self) -> ds.Dataset | None:
        cache_path = self._geometry_cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        return ds.dataset(cache_path, format="parquet")

    def _primary_arcgis_source(self) -> dict[str, Any] | None:
        registry = self._parcel_source_registry()
        for source in registry.get("parcel_sources", []):
            if str(source.get("service_type", "")).lower() == "arcgis_feature_layer" and source.get("primary", False):
                return source
        return None

    def _dataset_row(self, dataset: ds.Dataset, parcel_row_id: str) -> pd.Series | None:
        table = dataset.to_table(filter=ds.field("parcel_row_id") == parcel_row_id)
        if table.num_rows == 0:
            return None
        return table.to_pandas().iloc[0]

    def _detail_row(self, parcel_row_id: str) -> pd.Series | None:
        if self.artifacts.runtime_detail_metrics_path.exists() or self.artifacts.app_ready_path.exists():
            try:
                row = self._dataset_row(self._detail_dataset(), parcel_row_id)
            except FileNotFoundError:
                row = None
            if row is not None:
                return row
        return self._fallback_frame_row(parcel_row_id)

    def _parcel_master_row(self, parcel_row_id: str) -> pd.Series | None:
        if not self.artifacts.parcel_master_path.exists():
            return None
        return self._dataset_row(self._parcel_master_dataset(), parcel_row_id)

    def _row_for_parcel(self, parcel_row_id: str) -> pd.Series | None:
        row = self._detail_row(parcel_row_id)
        if row is None:
            row = self._parcel_master_row(parcel_row_id)
        return row

    def _query_frame(self, *, columns: list[str], filter_expression: ds.Expression | None = None) -> pd.DataFrame:
        if self.artifacts.app_ready_path.exists():
            dataset = self._app_ready_dataset()
            available_columns = [column for column in columns if column in dataset.schema.names]
            if not available_columns:
                return pd.DataFrame(columns=columns)
            return dataset.to_table(columns=available_columns, filter=filter_expression).to_pandas()
        fallback = self._fallback_query_frame()
        available_columns = [column for column in columns if column in fallback.columns]
        if not available_columns:
            return pd.DataFrame(columns=columns)
        return fallback.loc[:, available_columns].copy()

    def _apply_lead_filters_to_frame(
        self,
        frame: pd.DataFrame,
        *,
        county_name: str | None = None,
        lead_score_tier: list[str] | None = None,
        min_lead_score_total: float | None = None,
        acreage_min: float | None = None,
        acreage_max: float | None = None,
        parcel_vacant_flag: bool | None = None,
        county_hosted_flag: bool | None = None,
        high_confidence_link_flag: bool | None = None,
        wetland_flag: bool | None = None,
        amount_trust_tier: list[str] | None = None,
        corporate_owner_flag: bool | None = None,
        absentee_owner_flag: bool | None = None,
        out_of_state_owner_flag: bool | None = None,
        growth_pressure_bucket: list[str] | None = None,
        recommended_view_bucket: list[str] | None = None,
        road_access_tier: list[str] | None = None,
        road_distance_ft_max: float | None = None,
        sort_by: str = "lead_score_total",
        sort_direction: str = "desc",
    ) -> pd.DataFrame:
        filtered = frame.copy()

        def _string_series(column: str) -> pd.Series:
            if column not in filtered.columns:
                return pd.Series(pd.NA, index=filtered.index, dtype="string")
            return pd.Series(filtered[column], index=filtered.index, dtype="string").str.strip()

        def _apply_membership_filter(column: str, allowed: list[str] | None) -> None:
            nonlocal filtered
            if not allowed or column not in filtered.columns:
                return
            allowed_values = {str(value).strip().lower() for value in allowed if str(value).strip()}
            if not allowed_values:
                return
            normalized = _string_series(column).str.lower()
            filtered = filtered.loc[normalized.isin(allowed_values)].copy()

        def _apply_bool_filter(column: str, expected: bool | None) -> None:
            nonlocal filtered
            if expected is None or column not in filtered.columns:
                return
            filtered = filtered.loc[filtered[column].fillna(False).astype(bool) == bool(expected)].copy()

        if county_name and "county_name" in filtered.columns:
            filtered = filtered.loc[_string_series("county_name").str.lower() == (_slug_county_name(county_name) or "")].copy()
        if acreage_min is not None and "acreage" in filtered.columns:
            filtered = filtered.loc[pd.to_numeric(filtered["acreage"], errors="coerce") >= float(acreage_min)].copy()
        if acreage_max is not None and "acreage" in filtered.columns:
            filtered = filtered.loc[pd.to_numeric(filtered["acreage"], errors="coerce") <= float(acreage_max)].copy()
        if min_lead_score_total is not None:
            score_column = "lead_score_total" if "lead_score_total" in filtered.columns else "lead_score_total_effective"
            if score_column in filtered.columns:
                filtered = filtered.loc[pd.to_numeric(filtered[score_column], errors="coerce") >= float(min_lead_score_total)].copy()
        _apply_bool_filter("parcel_vacant_flag", parcel_vacant_flag)
        _apply_bool_filter("county_hosted_flag", county_hosted_flag)
        _apply_bool_filter("high_confidence_link_flag", high_confidence_link_flag)
        _apply_bool_filter("wetland_flag", wetland_flag)
        _apply_bool_filter("corporate_owner_flag", corporate_owner_flag)
        _apply_bool_filter("absentee_owner_flag", absentee_owner_flag)
        _apply_bool_filter("out_of_state_owner_flag", out_of_state_owner_flag)
        _apply_membership_filter("lead_score_tier", lead_score_tier)
        _apply_membership_filter("amount_trust_tier", amount_trust_tier)
        _apply_membership_filter("growth_pressure_bucket", growth_pressure_bucket)
        _apply_membership_filter("recommended_view_bucket", recommended_view_bucket)
        _apply_membership_filter("road_access_tier", road_access_tier)
        if road_distance_ft_max is not None and "road_distance_ft" in filtered.columns:
            filtered = filtered.loc[pd.to_numeric(filtered["road_distance_ft"], errors="coerce") <= float(road_distance_ft_max)].copy()

        if sort_by not in filtered.columns:
            sort_by = "lead_score_total" if "lead_score_total" in filtered.columns else "acreage"
        if sort_by in filtered.columns and sort_by in {"lead_score_total", "lead_score_total_effective", "acreage", "road_distance_ft"}:
            filtered[sort_by] = pd.to_numeric(filtered[sort_by], errors="coerce")
        ascending = str(sort_direction).lower() == "asc"
        if sort_by in filtered.columns:
            filtered = filtered.sort_values(by=sort_by, ascending=ascending, na_position="last", kind="mergesort")
        return filtered

    def _cached_geometry_geojson(
        self,
        parcel_row_id: str,
        *,
        source_object_id: int | str | None = None,
    ) -> dict[str, Any] | None:
        dataset = self._geometry_cache_dataset()
        if dataset is None:
            return None
        available_columns = [
            column
            for column in ["parcel_row_id", "parcel_id", "county_name", "source_object_id", "geometry_wkb"]
            if column in dataset.schema.names
        ]
        if "geometry_wkb" not in available_columns:
            return None
        table = dataset.to_table(columns=available_columns, filter=ds.field("parcel_row_id") == parcel_row_id)
        if table.num_rows == 0 and source_object_id is not None and "source_object_id" in dataset.schema.names:
            try:
                object_id_value = int(float(source_object_id))
            except Exception:
                object_id_value = None
            if object_id_value is not None:
                table = dataset.to_table(columns=available_columns, filter=ds.field("source_object_id") == object_id_value)
        if table.num_rows == 0:
            return None
        record = table.to_pandas().iloc[0]
        geometry_bytes = record.get("geometry_wkb")
        if geometry_bytes is None:
            return None
        geometry_shape = wkb.loads(bytes(geometry_bytes))
        return {
            "type": "Feature",
            "geometry": mapping(geometry_shape),
            "properties": {
                "parcel_row_id": _normalize_string(record.get("parcel_row_id")),
                "parcel_id": _normalize_string(record.get("parcel_id")),
                "county_name": _normalize_string(record.get("county_name")),
                "source_object_id": _json_scalar(record.get("source_object_id")),
            },
        }

    def _source_geometry_geojson(
        self,
        source_object_id: int | str,
        *,
        parcel_row_id: str | None = None,
        parcel_id: str | None = None,
    ) -> dict[str, Any] | None:
        source = self._primary_arcgis_source()
        if source is None:
            logger.warning(
                "State geometry source missing state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s",
                self.state_code,
                parcel_row_id,
                parcel_id,
                source_object_id,
            )
            return None
        query_url = str(source["service_url"]).rstrip("/") + "/query"
        out_fields = str(source.get("geometry_out_fields", "objectid,countyfips,county,parcelid"))
        object_id_value = int(float(source_object_id))
        params = {
            "objectIds": str(object_id_value),
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        logger.info(
            "State geometry ArcGIS request state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s url=%s params=%s",
            self.state_code,
            parcel_row_id,
            parcel_id,
            object_id_value,
            query_url,
            params,
        )
        response = self._http.get(
            query_url,
            params=params,
            timeout=float(os.getenv("STATE_GEOMETRY_TIMEOUT_SECONDS", "20")),
        )
        if not response.ok:
            logger.error(
                "State geometry ArcGIS error state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s status=%s body=%s",
                self.state_code,
                parcel_row_id,
                parcel_id,
                object_id_value,
                response.status_code,
                response.text[:500],
            )
            response.raise_for_status()
        logger.info(
            "State geometry ArcGIS response state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s status=%s url=%s",
            self.state_code,
            parcel_row_id,
            parcel_id,
            object_id_value,
            response.status_code,
            response.url,
        )
        payload = response.json()
        features = payload.get("features", [])
        if not features:
            logger.warning(
                "State geometry ArcGIS empty state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s body=%s",
                self.state_code,
                parcel_row_id,
                parcel_id,
                object_id_value,
                response.text[:500],
            )
            return None
        return features[0]

    def _geometry_feature_properties(self, row: Mapping[str, Any], *, selected: bool) -> dict[str, Any]:
        properties = serialize_contract_row(row, GEOMETRY_FEATURE_PROPERTY_FIELDS, serializer=_json_scalar)
        properties["selected"] = selected
        return properties

    def _geometry_item(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "parcel_row_id": _normalize_string(row.get("parcel_row_id")),
            "path": None,
            "lead_score_total": _json_scalar(row.get("lead_score_total")),
        }

    def get_leads(
        self,
        *,
        county_name: str | None = None,
        lead_score_tier: list[str] | None = None,
        min_lead_score_total: float | None = None,
        acreage_min: float | None = None,
        acreage_max: float | None = None,
        parcel_vacant_flag: bool | None = None,
        county_hosted_flag: bool | None = None,
        high_confidence_link_flag: bool | None = None,
        wetland_flag: bool | None = None,
        amount_trust_tier: list[str] | None = None,
        corporate_owner_flag: bool | None = None,
        absentee_owner_flag: bool | None = None,
        out_of_state_owner_flag: bool | None = None,
        growth_pressure_bucket: list[str] | None = None,
        recommended_view_bucket: list[str] | None = None,
        road_access_tier: list[str] | None = None,
        road_distance_ft_max: float | None = None,
        sort_by: str = "lead_score_total",
        sort_direction: str = "desc",
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        default_request = not any(
            value
            for value in [
                county_name,
                lead_score_tier,
                min_lead_score_total,
                acreage_min,
                acreage_max,
                parcel_vacant_flag,
                county_hosted_flag,
                high_confidence_link_flag,
                wetland_flag,
                amount_trust_tier,
                corporate_owner_flag,
                absentee_owner_flag,
                out_of_state_owner_flag,
                growth_pressure_bucket,
                recommended_view_bucket,
                road_access_tier,
                road_distance_ft_max,
                offset,
            ]
        ) and sort_by == "lead_score_total" and sort_direction == "desc"
        if default_request:
            return self._default_leads_payload()

        dataset = self._app_ready_dataset() if self.artifacts.app_ready_path.exists() else None
        requested_columns = list(
            dict.fromkeys(
                API_LEADS_SUMMARY_FIELDS
                + [
                    "parcel_row_id",
                    "parcel_id",
                    "county_name",
                    "acreage",
                    "lead_score_total",
                    "lead_score_total_effective",
                    "lead_score_tier",
                    "parcel_vacant_flag",
                    "county_hosted_flag",
                    "high_confidence_link_flag",
                    "wetland_flag",
                    "amount_trust_tier",
                    "corporate_owner_flag",
                    "absentee_owner_flag",
                    "out_of_state_owner_flag",
                    "growth_pressure_bucket",
                    "recommended_view_bucket",
                    "road_access_tier",
                    "road_distance_ft",
                    sort_by,
                ]
            )
        )
        filters: list[ds.Expression] = []
        schema_names = set(dataset.schema.names) if dataset is not None else set()
        if dataset is not None:
            if county_name and "county_name" in schema_names:
                filters.append(ds.field("county_name") == _slug_county_name(county_name))
            if acreage_min is not None and "acreage" in schema_names:
                filters.append(ds.field("acreage") >= float(acreage_min))
            if acreage_max is not None and "acreage" in schema_names:
                filters.append(ds.field("acreage") <= float(acreage_max))
            if min_lead_score_total is not None and "lead_score_total" in schema_names:
                filters.append(ds.field("lead_score_total") >= float(min_lead_score_total))
            if parcel_vacant_flag is not None and "parcel_vacant_flag" in schema_names:
                filters.append(ds.field("parcel_vacant_flag") == bool(parcel_vacant_flag))
            if county_hosted_flag is not None and "county_hosted_flag" in schema_names:
                filters.append(ds.field("county_hosted_flag") == bool(county_hosted_flag))
            if high_confidence_link_flag is not None and "high_confidence_link_flag" in schema_names:
                filters.append(ds.field("high_confidence_link_flag") == bool(high_confidence_link_flag))
            if wetland_flag is not None and "wetland_flag" in schema_names:
                filters.append(ds.field("wetland_flag") == bool(wetland_flag))
            if road_distance_ft_max is not None and "road_distance_ft" in schema_names:
                filters.append(ds.field("road_distance_ft") <= float(road_distance_ft_max))

        filter_expression = None
        for expression in filters:
            filter_expression = expression if filter_expression is None else filter_expression & expression

        frame = self._query_frame(columns=requested_columns, filter_expression=filter_expression)
        if frame.empty and dataset is None and self._fallback_query_frame().empty:
            raise FileNotFoundError(
                "State leads query artifacts unavailable: "
                f"app_ready={self.artifacts.app_ready_path}; "
                f"static_feed={self.artifacts.frontend_static_feed_path}; "
                f"detail_fallback={self.artifacts.frontend_detail_fallback_path}"
            )
        if frame.empty:
            validate_output_records(
                [],
                expected_fields=API_LEADS_SUMMARY_FIELDS,
                required_fields=["parcel_row_id", "parcel_id", "county_name"],
                non_null_fields=["parcel_row_id", "parcel_id"],
                context=f"runtime_state_service[{self.state_code}].get_leads.empty",
            )
            return {"total_count": 0, "limit": int(limit), "offset": int(offset), "items": []}
        frame = self._apply_lead_filters_to_frame(
            frame,
            county_name=county_name,
            lead_score_tier=lead_score_tier,
            min_lead_score_total=min_lead_score_total,
            acreage_min=acreage_min,
            acreage_max=acreage_max,
            parcel_vacant_flag=parcel_vacant_flag,
            county_hosted_flag=county_hosted_flag,
            high_confidence_link_flag=high_confidence_link_flag,
            wetland_flag=wetland_flag,
            amount_trust_tier=amount_trust_tier,
            corporate_owner_flag=corporate_owner_flag,
            absentee_owner_flag=absentee_owner_flag,
            out_of_state_owner_flag=out_of_state_owner_flag,
            growth_pressure_bucket=growth_pressure_bucket,
            recommended_view_bucket=recommended_view_bucket,
            road_access_tier=road_access_tier,
            road_distance_ft_max=road_distance_ft_max,
            sort_by=sort_by,
            sort_direction=sort_direction,
        )

        total_count = int(len(frame))
        paged = frame.iloc[offset : offset + limit].copy()
        records = [
            serialize_contract_row(row.to_dict(), API_LEADS_SUMMARY_FIELDS, serializer=_json_scalar)
            for _, row in paged.iterrows()
        ]
        validate_output_records(
            records,
            expected_fields=API_LEADS_SUMMARY_FIELDS,
            required_fields=["parcel_row_id", "parcel_id", "county_name"],
            non_null_fields=["parcel_row_id", "parcel_id"],
            context=f"runtime_state_service[{self.state_code}].get_leads.items",
        )
        return {"total_count": total_count, "limit": int(limit), "offset": int(offset), "items": records}

    def search_leads(self, q: str, *, limit: int = 10) -> dict[str, Any]:
        query = (q or "").strip().lower()
        if not query:
            return {"query": "", "items": []}
        frame = self._search_frame().copy()
        if frame.empty:
            return {"query": query, "items": []}
        row_id = frame["parcel_row_id"].astype("string").str.lower()
        parcel_id = frame.get("parcel_id", pd.Series("", index=frame.index, dtype="string")).astype("string").str.lower()
        owner_name = frame.get("owner_name", pd.Series("", index=frame.index, dtype="string")).astype("string").str.lower()

        score = pd.Series(np.nan, index=frame.index, dtype="float64")
        match_field = pd.Series(pd.NA, index=frame.index, dtype="string")
        ranked_masks = [
            (row_id.eq(query), "parcel_row_id_exact"),
            (parcel_id.eq(query), "parcel_id_exact"),
            (row_id.str.startswith(query), "parcel_row_id_prefix"),
            (parcel_id.str.startswith(query), "parcel_id_prefix"),
            (row_id.str.contains(query, regex=False), "parcel_row_id_partial"),
            (parcel_id.str.contains(query, regex=False), "parcel_id_partial"),
            (owner_name.eq(query), "owner_name_exact"),
            (owner_name.str.startswith(query), "owner_name_prefix"),
            (owner_name.str.contains(query, regex=False), "owner_name_partial"),
        ]
        for rank, (mask, label) in enumerate(ranked_masks):
            applicable = mask.fillna(False) & score.isna()
            score.loc[applicable] = rank
            match_field.loc[applicable] = label

        matched = frame.loc[score.notna()].copy()
        if matched.empty:
            return {"query": query, "items": []}
        matched["match_rank"] = score.loc[matched.index]
        matched["match_field"] = match_field.loc[matched.index]
        matched = matched.sort_values(by=["match_rank", "parcel_row_id"], kind="mergesort").head(limit)

        records: list[dict[str, Any]] = []
        for _, row in matched.iterrows():
            centroid = None
            if pd.notna(row.get("longitude")) and pd.notna(row.get("latitude")):
                centroid = {
                    "type": "Point",
                    "coordinates": [round(float(row["longitude"]), 6), round(float(row["latitude"]), 6)],
                }
            records.append(
                {
                    "parcel_row_id": _normalize_string(row.get("parcel_row_id")),
                    "parcel_id": _normalize_string(row.get("parcel_id")),
                    "county_name": _normalize_string(row.get("county_name")),
                    "acreage": _json_scalar(row.get("acreage")),
                    "owner_name": _normalize_string(row.get("owner_name")),
                    "centroid": centroid,
                    "match_field": _normalize_string(row.get("match_field")),
                }
            )
        validate_output_records(
            records,
            expected_fields=SEARCH_OUTPUT_FIELDS,
            required_fields=["parcel_row_id", "parcel_id", "county_name"],
            non_null_fields=["parcel_row_id", "parcel_id"],
            context=f"runtime_state_service[{self.state_code}].search_leads.items",
        )
        return {"query": query, "items": records}

    def get_lead_detail(self, parcel_row_id: str) -> dict[str, Any] | None:
        row = self._row_for_parcel(parcel_row_id)
        if row is None:
            return None
        payload = {column: _json_scalar(value) for column, value in row.to_dict().items() if column != "geometry"}
        payload["geometry"] = _point_geometry_payload(row)
        validate_output_records(
            [payload],
            expected_fields=list(payload.keys()),
            required_fields=BACKEND_DETAIL_REQUIRED_FIELDS,
            non_null_fields=["parcel_row_id", "parcel_id"],
            context=f"runtime_state_service[{self.state_code}].get_lead_detail",
        )
        return payload

    def get_nearby_comps(self, parcel_row_id: str, *, limit: int = 8) -> dict[str, Any] | None:
        subject = self._row_for_parcel(parcel_row_id)
        if subject is None:
            return None
        if pd.isna(subject.get("latitude")) or pd.isna(subject.get("longitude")):
            return {
                "subject": None,
                "methodology": {
                    "radius_tiers_miles": [0.5, 1.0, 3.0],
                    "acreage_similarity_floor": 0.25,
                    "prioritize_same_land_use": True,
                    "prefer_same_improvement_status": True,
                    "value_signal": "assessed_total_value",
                    "limit": int(limit),
                    "comp_filtering_mode": "state_runtime_minimal",
                    "mixed_status_included_flag": True,
                    "uncertain_status_included_flag": True,
                    "quality_note": "Subject parcel is missing centroid coordinates.",
                },
                "items": [],
            }

        columns = [
            "parcel_row_id",
            "parcel_id",
            "county_name",
            "acreage",
            "land_use",
            "assessed_total_value",
            "lead_score_total",
            "investment_score",
            "parcel_vacant_flag",
            "parcel_improvement_status",
            "parcel_improvement_confidence",
            "parcel_improvement_reason",
            "parcel_improvement_evidence_summary",
            "parcel_width_ft_estimate",
            "buildability_score",
            "road_access_tier",
            "latitude",
            "longitude",
        ]
        frame = self._query_frame(columns=columns)
        frame = frame.loc[frame["parcel_row_id"].astype("string") != parcel_row_id].copy()
        item_records: list[dict[str, Any]] = []
        if not frame.empty:
            distance = _haversine_miles(float(subject["latitude"]), float(subject["longitude"]), frame["latitude"], frame["longitude"])
            frame["distance_to_subject_miles"] = distance.round(3)
            frame = frame.loc[frame["distance_to_subject_miles"] <= DEFAULT_NEARBY_RADIUS_MILES].copy()
            if not frame.empty:
                acreage_delta = np.abs(pd.to_numeric(frame.get("acreage"), errors="coerce").fillna(0) - float(subject.get("acreage") or 0))
                acreage_score = np.maximum(0.0, 1.0 - (acreage_delta / np.maximum(float(subject.get("acreage") or 1.0), 1.0)))
                same_land_use = (
                    pd.Series(frame.get("land_use"), index=frame.index, dtype="string").fillna("")
                    == str(subject.get("land_use") or "")
                ).astype(float)
                same_status = (
                    pd.Series(frame.get("parcel_improvement_status"), index=frame.index, dtype="string").fillna("")
                    == str(subject.get("parcel_improvement_status") or "")
                ).astype(float)
                distance_score = np.maximum(0.0, 1.0 - (frame["distance_to_subject_miles"] / DEFAULT_NEARBY_RADIUS_MILES))
                frame["similarity_score"] = (0.45 * distance_score + 0.35 * acreage_score + 0.10 * same_land_use + 0.10 * same_status).round(3)
                frame["value_per_acre"] = np.where(
                    pd.to_numeric(frame.get("acreage"), errors="coerce").fillna(0) > 0,
                    pd.to_numeric(frame.get("assessed_total_value"), errors="coerce").fillna(0)
                    / pd.to_numeric(frame.get("acreage"), errors="coerce").replace(0, np.nan),
                    np.nan,
                )
                frame["radius_bucket"] = np.select(
                    [frame["distance_to_subject_miles"] <= 0.5, frame["distance_to_subject_miles"] <= 1.0],
                    ["0.5_mi", "1_mi"],
                    default="3_mi",
                )
                frame = frame.sort_values(by=["similarity_score", "distance_to_subject_miles"], ascending=[False, True]).head(limit)
                for _, row in frame.iterrows():
                    centroid = None
                    if pd.notna(row.get("longitude")) and pd.notna(row.get("latitude")):
                        centroid = {
                            "type": "Point",
                            "coordinates": [round(float(row["longitude"]), 6), round(float(row["latitude"]), 6)],
                        }
                    item_records.append(
                        {
                            "parcel_row_id": _normalize_string(row.get("parcel_row_id")),
                            "parcel_id": _normalize_string(row.get("parcel_id")),
                            "county_name": _normalize_string(row.get("county_name")),
                            "acreage": _json_scalar(row.get("acreage")),
                            "land_use": _normalize_string(row.get("land_use")),
                            "distance_to_subject_miles": _json_scalar(row.get("distance_to_subject_miles")),
                            "radius_bucket": _normalize_string(row.get("radius_bucket")),
                            "assessed_total_value": _json_scalar(row.get("assessed_total_value")),
                            "value_per_acre": _json_scalar(row.get("value_per_acre")),
                            "parcel_width_ft_estimate": _json_scalar(row.get("parcel_width_ft_estimate")),
                            "buildability_score": _json_scalar(row.get("buildability_score")),
                            "road_access_tier": _normalize_string(row.get("road_access_tier")),
                            "lead_score_total": _json_scalar(row.get("lead_score_total")),
                            "investment_score": _json_scalar(row.get("investment_score")),
                            "parcel_vacant_flag": _json_scalar(row.get("parcel_vacant_flag")),
                            "parcel_improvement_status": _normalize_string(row.get("parcel_improvement_status")),
                            "parcel_improvement_confidence": _json_scalar(row.get("parcel_improvement_confidence")),
                            "parcel_improvement_reason": _normalize_string(row.get("parcel_improvement_reason")),
                            "parcel_improvement_evidence_summary": _normalize_string(row.get("parcel_improvement_evidence_summary")),
                            "similarity_score": _json_scalar(row.get("similarity_score")),
                            "centroid": centroid,
                        }
                    )

        validate_output_records(
            item_records,
            expected_fields=NEARBY_COMP_OUTPUT_FIELDS,
            required_fields=["parcel_row_id", "parcel_id", "county_name"],
            non_null_fields=["parcel_row_id", "parcel_id"],
            context=f"runtime_state_service[{self.state_code}].get_nearby_comps.items",
        )
        subject_payload = {
            "parcel_row_id": _normalize_string(subject.get("parcel_row_id")),
            "parcel_id": _normalize_string(subject.get("parcel_id")),
            "county_name": _normalize_string(subject.get("county_name")),
            "acreage": _json_scalar(subject.get("acreage")),
            "land_use": _normalize_string(subject.get("land_use")),
            "distance_to_subject_miles": 0.0,
            "radius_bucket": "subject",
            "assessed_total_value": _json_scalar(subject.get("assessed_total_value")),
            "value_per_acre": None,
            "parcel_width_ft_estimate": _json_scalar(subject.get("parcel_width_ft_estimate")),
            "buildability_score": _json_scalar(subject.get("buildability_score")),
            "road_access_tier": _normalize_string(subject.get("road_access_tier")),
            "lead_score_total": _json_scalar(subject.get("lead_score_total")),
            "investment_score": _json_scalar(subject.get("investment_score")),
            "parcel_vacant_flag": _json_scalar(subject.get("parcel_vacant_flag")),
            "parcel_improvement_status": _normalize_string(subject.get("parcel_improvement_status")),
            "parcel_improvement_confidence": _json_scalar(subject.get("parcel_improvement_confidence")),
            "parcel_improvement_reason": _normalize_string(subject.get("parcel_improvement_reason")),
            "parcel_improvement_evidence_summary": _normalize_string(subject.get("parcel_improvement_evidence_summary")),
            "similarity_score": 1.0,
            "centroid": (
                {
                    "type": "Point",
                    "coordinates": [round(float(subject["longitude"]), 6), round(float(subject["latitude"]), 6)],
                }
                if pd.notna(subject.get("longitude")) and pd.notna(subject.get("latitude"))
                else None
            ),
        }
        return {
            "subject": subject_payload,
            "methodology": {
                "radius_tiers_miles": [0.5, 1.0, 3.0],
                "acreage_similarity_floor": 0.25,
                "prioritize_same_land_use": True,
                "prefer_same_improvement_status": True,
                "value_signal": "assessed_total_value",
                "limit": int(limit),
                "comp_filtering_mode": "state_runtime_minimal",
                "mixed_status_included_flag": True,
                "uncertain_status_included_flag": True,
                "quality_note": "Arkansas MVP comps use runtime parcel centroids and simplified acreage similarity; no sales dataset is included yet.",
            },
            "items": item_records,
        }

    def get_parcel_geometry(self, parcel_row_id: str, *, zoom: float | None = None) -> dict[str, Any]:
        row = self._row_for_parcel(parcel_row_id)
        if row is None:
            return {
                "geometry_mode": "selected_parcel_geojson",
                "render_mode": "none",
                "geometry_bounds": None,
                "geometry_view_box": None,
                "requested_bounds": None,
                "zoom": zoom,
                "feature_count": 0,
                "feature_collection": {"type": "FeatureCollection", "features": []},
                "items": [],
            }

        geometry_feature = None
        row_dict = row.to_dict()
        parcel_id = _normalize_string(row.get("parcel_id"))
        source_object_id = row.get("source_object_id") or row.get("geometry_source_objectid")
        logger.info(
            "State geometry request state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s",
            self.state_code,
            parcel_row_id,
            parcel_id,
            source_object_id,
        )
        try:
            geometry_feature = self._cached_geometry_geojson(parcel_row_id, source_object_id=source_object_id)
            if geometry_feature is not None:
                logger.info(
                    "State geometry cache hit state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s",
                    self.state_code,
                    parcel_row_id,
                    parcel_id,
                    source_object_id,
                )
            else:
                logger.info(
                    "State geometry cache miss state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s cache_path=%s",
                    self.state_code,
                    parcel_row_id,
                    parcel_id,
                    source_object_id,
                    self._geometry_cache_path(),
                )
        except Exception:
            logger.exception(
                "State geometry cache lookup failed state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s",
                self.state_code,
                parcel_row_id,
                parcel_id,
                source_object_id,
            )
            geometry_feature = None

        if geometry_feature is None and source_object_id is not None:
            try:
                geometry_feature = self._source_geometry_geojson(
                    source_object_id,
                    parcel_row_id=parcel_row_id,
                    parcel_id=parcel_id,
                )
            except Exception:
                logger.exception(
                    "State geometry live fetch failed state=%s parcel_row_id=%s parcel_id=%s source_object_id=%s",
                    self.state_code,
                    parcel_row_id,
                    parcel_id,
                    source_object_id,
                )
                geometry_feature = None

        if geometry_feature is None and pd.notna(row.get("longitude")) and pd.notna(row.get("latitude")):
            geometry_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": {},
            }

        if geometry_feature is None:
            features: list[dict[str, Any]] = []
            bounds_payload = None
        else:
            geometry_object = geometry_feature.get("geometry")
            features = [
                {
                    "type": "Feature",
                    "geometry": geometry_object,
                    "properties": self._geometry_feature_properties(row_dict, selected=True),
                }
            ]
            try:
                shape_bounds = shape(geometry_object).bounds if geometry_object else None
            except Exception:
                shape_bounds = None
            bounds_payload = (
                [round(shape_bounds[0], 6), round(shape_bounds[1], 6), round(shape_bounds[2], 6), round(shape_bounds[3], 6)]
                if shape_bounds is not None
                else None
            )

        items = [self._geometry_item(row_dict)]
        validate_output_records(
            [feature["properties"] for feature in features],
            expected_fields=GEOMETRY_FEATURE_PROPERTY_FIELDS,
            required_fields=["parcel_row_id", "parcel_id", "county_name"],
            non_null_fields=["parcel_row_id", "parcel_id"],
            context=f"runtime_state_service[{self.state_code}].get_parcel_geometry.features",
        )
        validate_output_records(
            items,
            expected_fields=GEOMETRY_ITEM_FIELDS,
            required_fields=["parcel_row_id"],
            non_null_fields=["parcel_row_id"],
            context=f"runtime_state_service[{self.state_code}].get_parcel_geometry.items",
        )
        return {
            "geometry_mode": "selected_parcel_geojson",
            "render_mode": (
                "polygons"
                if features and features[0]["geometry"] and features[0]["geometry"].get("type") != "Point"
                else "points"
                if features
                else "none"
            ),
            "geometry_bounds": bounds_payload,
            "geometry_view_box": None,
            "requested_bounds": None,
            "zoom": zoom,
            "feature_count": len(features),
            "feature_collection": {"type": "FeatureCollection", "features": features},
            "items": items,
        }

    def get_parcel_tile(self, z: int, x: int, y: int) -> bytes:
        return mapbox_vector_tile.encode([])

    def get_presets(self) -> list[dict[str, Any]]:
        return self._presets_payload()

    def get_summary(self) -> dict[str, Any]:
        return self._summary_payload()


@lru_cache(maxsize=8)
def get_runtime_state_service(state_code: str) -> RuntimeStateService:
    return RuntimeStateService(state_code)
