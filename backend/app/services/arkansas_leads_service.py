from __future__ import annotations

from functools import lru_cache

from app.services.runtime_state_service import get_runtime_state_service


@lru_cache(maxsize=1)
def _service():
    return get_runtime_state_service("ar")


def get_leads(**kwargs):
    return _service().get_leads(**kwargs)


def search_leads(q: str, limit: int = 10):
    return _service().search_leads(q, limit=limit)


def get_lead_detail(parcel_row_id: str):
    return _service().get_lead_detail(parcel_row_id)


def get_nearby_comps(parcel_row_id: str, limit: int = 8):
    return _service().get_nearby_comps(parcel_row_id, limit=limit)


def get_parcel_geometry(parcel_row_id: str, zoom: float | None = None):
    return _service().get_parcel_geometry(parcel_row_id, zoom=zoom)


def get_parcel_tile(z: int, x: int, y: int):
    return _service().get_parcel_tile(z, x, y)


def get_presets():
    return _service().get_presets()


def get_summary():
    return _service().get_summary()
