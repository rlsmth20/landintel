from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.services.state_service_registry import get_state_service_module
from app.settings import GEOMETRY_DEFAULT_LIMIT, LEADS_DEFAULT_LIMIT


router = APIRouter(prefix="/api/states/{state_code}", tags=["state-leads"])
logger = logging.getLogger("parcel-tiles")


def _service(state_code: str):
    try:
        logger.info("Resolving state route state=%s", state_code)
        return get_state_service_module(state_code)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error


@router.get("/leads")
def leads(
    state_code: str,
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
    service = _service(state_code)
    return service.get_leads(
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


@router.get("/leads/search")
def leads_search(state_code: str, q: str, limit: int = 10):
    return _service(state_code).search_leads(q, limit=limit)


@router.get("/leads/{parcel_row_id}")
def lead_detail(state_code: str, parcel_row_id: str):
    result = _service(state_code).get_lead_detail(parcel_row_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return result


@router.get("/leads/{parcel_row_id}/nearby-comps")
def nearby_comps(state_code: str, parcel_row_id: str, limit: int = 8):
    result = _service(state_code).get_nearby_comps(parcel_row_id, limit=limit)
    if result is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return result


@router.get("/parcels/{parcel_row_id}/geometry")
def parcel_geometry(state_code: str, parcel_row_id: str, zoom: float | None = None):
    return _service(state_code).get_parcel_geometry(parcel_row_id, zoom=zoom)


@router.get("/tiles/parcels/{z}/{x}/{y}.mvt")
def parcel_tile(state_code: str, z: int, x: int, y: int):
    service = _service(state_code)
    try:
        tile = service.get_parcel_tile(z, x, y)
        return Response(content=tile, media_type="application/vnd.mapbox-vector-tile")
    except Exception:
        logger.exception("state parcel tile endpoint failed state=%s z=%s x=%s y=%s", state_code, z, x, y)
        raise


@router.get("/presets")
def presets(state_code: str):
    return {"items": _service(state_code).get_presets()}


@router.get("/summary")
def summary(state_code: str):
    return _service(state_code).get_summary()
