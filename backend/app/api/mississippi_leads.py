from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.services.mississippi_leads_service import (
    get_geometry,
    get_lead_detail,
    get_leads,
    get_parcel_geometry,
    get_parcel_tile,
    get_presets,
    get_summary,
)
from app.settings import GEOMETRY_DEFAULT_LIMIT, LEADS_DEFAULT_LIMIT


router = APIRouter(prefix="/api", tags=["mississippi-leads"])


@router.get("/leads")
def leads(
    county_name: str | None = None,
    lead_score_tier: list[str] | None = Query(default=None),
    min_lead_score_total: float | None = None,
    acreage_min: float | None = None,
    acreage_max: float | None = None,
    parcel_vacant_flag: bool | None = None,
    county_hosted_flag: bool | None = None,
    high_confidence_link_flag: bool | None = None,
    wetland_flag: bool | None = None,
    amount_trust_tier: list[str] | None = Query(default=None),
    corporate_owner_flag: bool | None = None,
    absentee_owner_flag: bool | None = None,
    out_of_state_owner_flag: bool | None = None,
    growth_pressure_bucket: list[str] | None = Query(default=None),
    recommended_view_bucket: list[str] | None = Query(default=None),
    road_access_tier: list[str] | None = Query(default=None),
    road_distance_ft_max: float | None = None,
    sort_by: str = "lead_score_total",
    sort_direction: str = "desc",
    limit: int = LEADS_DEFAULT_LIMIT,
    offset: int = 0,
):
    return get_leads(
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
        limit=limit,
        offset=offset,
    )


@router.get("/leads/geometry")
def leads_geometry(
    parcel_row_id: list[str] | None = Query(default=None),
    min_lng: float | None = None,
    min_lat: float | None = None,
    max_lng: float | None = None,
    max_lat: float | None = None,
    zoom: float | None = None,
    selected_parcel_id: str | None = None,
    county_name: str | None = None,
    lead_score_tier: list[str] | None = Query(default=None),
    min_lead_score_total: float | None = None,
    acreage_min: float | None = None,
    acreage_max: float | None = None,
    parcel_vacant_flag: bool | None = None,
    county_hosted_flag: bool | None = None,
    high_confidence_link_flag: bool | None = None,
    wetland_flag: bool | None = None,
    amount_trust_tier: list[str] | None = Query(default=None),
    corporate_owner_flag: bool | None = None,
    absentee_owner_flag: bool | None = None,
    out_of_state_owner_flag: bool | None = None,
    growth_pressure_bucket: list[str] | None = Query(default=None),
    recommended_view_bucket: list[str] | None = Query(default=None),
    road_access_tier: list[str] | None = Query(default=None),
    road_distance_ft_max: float | None = None,
    limit: int = GEOMETRY_DEFAULT_LIMIT,
):
    bounds = None
    if None not in {min_lng, min_lat, max_lng, max_lat}:
        bounds = (min_lng, min_lat, max_lng, max_lat)
    return get_geometry(
        parcel_row_ids=parcel_row_id,
        bounds=bounds,
        zoom=zoom,
        selected_parcel_id=selected_parcel_id,
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
        limit=limit,
    )


@router.get("/leads/{parcel_row_id}")
def lead_detail(parcel_row_id: str):
    result = get_lead_detail(parcel_row_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return result


@router.get("/parcels")
def parcels_geometry(
    min_lng: float | None = None,
    min_lat: float | None = None,
    max_lng: float | None = None,
    max_lat: float | None = None,
    zoom: float | None = None,
    selected_parcel_id: str | None = None,
    county_name: str | None = None,
    lead_score_tier: list[str] | None = Query(default=None),
    min_lead_score_total: float | None = None,
    acreage_min: float | None = None,
    acreage_max: float | None = None,
    parcel_vacant_flag: bool | None = None,
    county_hosted_flag: bool | None = None,
    high_confidence_link_flag: bool | None = None,
    wetland_flag: bool | None = None,
    amount_trust_tier: list[str] | None = Query(default=None),
    corporate_owner_flag: bool | None = None,
    absentee_owner_flag: bool | None = None,
    out_of_state_owner_flag: bool | None = None,
    growth_pressure_bucket: list[str] | None = Query(default=None),
    recommended_view_bucket: list[str] | None = Query(default=None),
    road_access_tier: list[str] | None = Query(default=None),
    road_distance_ft_max: float | None = None,
    limit: int = GEOMETRY_DEFAULT_LIMIT,
):
    bounds = None
    if None not in {min_lng, min_lat, max_lng, max_lat}:
        bounds = (min_lng, min_lat, max_lng, max_lat)
    return get_geometry(
        bounds=bounds,
        zoom=zoom,
        selected_parcel_id=selected_parcel_id,
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
        limit=limit,
    )


@router.get("/parcels/{parcel_row_id}/geometry")
def parcel_geometry(parcel_row_id: str, zoom: float | None = None):
    return get_parcel_geometry(parcel_row_id, zoom=zoom)


@router.get("/tiles/parcels/{z}/{x}/{y}.mvt")
def parcel_tile(z: int, x: int, y: int):
    tile = get_parcel_tile(z, x, y)
    return Response(content=tile, media_type="application/vnd.mapbox-vector-tile")


@router.get("/presets")
def presets():
    return {"items": get_presets()}


@router.get("/summary")
def summary():
    return get_summary()
