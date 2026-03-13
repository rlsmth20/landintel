# Mississippi Explorer Geometry

The Mississippi lead explorer uses a two-feed geometry model for the MVP.

## Source

- Base product dataset: `data/tax_published/ms/app_ready_mississippi_leads.parquet`
- Frontend point feed: `frontend/public/data/mississippi_lead_explorer.json`
- Frontend polygon feed: `frontend/public/data/mississippi_lead_explorer_geometries.json`

## Preparation

Geometry is exported from the parcel WKB in `app_ready_mississippi_leads.parquet`.

The frontend export script:

- loads parcel geometry by `parcel_row_id`
- simplifies each parcel polygon with `shapely.simplify(..., preserve_topology=True)`
- uses tolerance `0.00002`
- converts the simplified geometry into a normalized SVG path
- writes one path per `parcel_row_id`

Exporter:

- `floodscraper/lead_explorer_frontend_export_ms.py`

## Frontend loading model

The frontend loads:

- `mississippi_lead_explorer.json` for list/detail/filter data and point geometry
- `mississippi_lead_explorer_geometries.json` for simplified parcel shapes

Map rendering strategy:

- render simplified parcel polygons only for the top ranked filtered subset plus the selected parcel
- keep centroid points as broader spatial context

This keeps the MVP responsive while still showing parcel shape instead of centroid-only rendering.

## Tradeoffs

- Shapes are simplified for browser rendering speed and smaller payload size
- The map does not render all filtered parcel polygons at once
- Selected parcels still get full shape visibility
- The geometry feed is static and local, not tiled

## Later production upgrade

For a production-grade statewide map surface, the next step is server-backed geometry tiles or vector tiles with viewport-based loading. That would allow:

- full parcel polygon rendering by zoom/viewport
- better map performance at large result counts
- lower client payload size
- easier multi-state scaling
