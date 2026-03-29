# Parcel PMTiles Workflow

This is the state-aware parcel overlay workflow for LandIntel.

## Build

Use the shared builder:

```powershell
python floodscraper\build_parcel_pmtiles.py --state-code ar
python floodscraper\build_parcel_pmtiles.py --state-code ms
```

The builder resolves state-specific settings from `config/states/<state_code>.json`:

- output PMTiles path
- local development parcel tile URL
- production public parcel tile URL
- build source
- min/max zoom
- geometry cache path
- build summary path
- publish manifest path

## Full-Parcel Rule

The parcel base layer must always be built from the full statewide parcel base:

- `parcel_master` -> base parcel overlay / PMTiles archive
- `app_ready` -> filtered business layer only

Lead filtering must never determine which parcels appear on the map.

## Geometry Sources

- States with local parcel geometry in parquet can build directly from that geometry.
- States without local parcel geometry can build from a live ArcGIS parcel source into a reusable local geometry cache first.

Mississippi currently uses:

- build source: `parcel_master`
- output: `frontend/public/tiles/mississippi_parcels.pmtiles`

Current multi-state parcel PMTiles states also include `ar`, `ct`, `mt`, `vt`, `wi`, and `ny`, all built from `parcel_master`.

## Outputs

Each build writes:

- PMTiles archive in `frontend/public/tiles/`
- build summary JSON
- publish manifest JSON

The publish manifest includes:

- artifact path
- expected object key
- local development URL
- production public URL
- content type
- cache-control header

The publish manifest is publish-ready, not publish-complete. It does not by itself mean the archive has been uploaded or that a public URL exists.

## Publish

Use the shared publish helper to turn the manifest into a repeatable Cloudflare R2 upload plan:

```powershell
python floodscraper\publish_parcel_pmtiles.py --state-code ar --bucket your-r2-bucket
python floodscraper\publish_parcel_pmtiles.py --state-code ms --bucket your-r2-bucket
```

For large statewide archives, prefer the `boto3` transport against the R2 S3 endpoint:

```powershell
python -m pip install -r floodscraper\requirements.publish.txt
python floodscraper\publish_parcel_pmtiles.py --state-code wi --bucket your-r2-bucket --transport boto3 --endpoint-url https://<cloudflare-account-id>.r2.cloudflarestorage.com --execute
```

Wrangler remains available for smaller archives:

```powershell
python floodscraper\publish_parcel_pmtiles.py --state-code ny --bucket your-r2-bucket --transport wrangler --execute
```

The helper reads the state-specific publish manifest and preserves:

- object key
- content type
- cache-control
- artifact path
- transport choice

## Hosting

Production contract:

- each state has a checked-in `public_url`
- the frontend prefers that `public_url` in production
- local `/tiles/...` paths are reserved for local development fallback

If parcel archives are hosted in Cloudflare-backed object storage/CDN instead, upload the artifact using the object key from the publish manifest and preserve:

- `Content-Type: application/vnd.pmtiles`
- `Cache-Control: public, max-age=31536000, immutable`

Then point the frontend state config or deployment routing to the hosted archive URL.

## Validate

Refresh manifests and validate the public URL contract:

```powershell
python floodscraper\validate_pmtiles_hosting_contract.py --refresh-manifests
```

This writes `data/runtime/pmtiles_hosting_validation.json` with:

- per-state public URL
- per-state local development fallback URL
- manifest presence and contract consistency
- local artifact presence
- current public URL reachability
