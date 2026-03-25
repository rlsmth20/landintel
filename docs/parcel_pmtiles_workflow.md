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
- frontend parcel tile URL
- build source (`parcel_master` or `app_ready`)
- min/max zoom
- geometry cache path
- build summary path
- publish manifest path

## Geometry Sources

- States with local parcel geometry in parquet can build directly from that geometry.
- States without local parcel geometry can build from a live ArcGIS parcel source into a reusable local geometry cache first.

Arkansas currently uses:

- build source: `app_ready`
- geometry strategy: live ArcGIS geometry cache
- output: `frontend/public/tiles/ar_parcels.pmtiles`

Mississippi currently uses:

- build source: `parcel_master`
- output: `frontend/public/tiles/mississippi_parcels.pmtiles`

## Outputs

Each build writes:

- PMTiles archive in `frontend/public/tiles/`
- build summary JSON
- publish manifest JSON

The publish manifest includes:

- artifact path
- expected object key
- content type
- cache-control header
- frontend URL

## Publish

Use the shared publish helper to turn the manifest into a repeatable Cloudflare R2 upload command:

```powershell
python floodscraper\publish_parcel_pmtiles.py --state-code ar --bucket your-r2-bucket
python floodscraper\publish_parcel_pmtiles.py --state-code ms --bucket your-r2-bucket
```

To execute the upload through Wrangler instead of printing the command:

```powershell
python floodscraper\publish_parcel_pmtiles.py --state-code ar --bucket your-r2-bucket --execute
```

The helper reads the state-specific publish manifest and preserves:

- object key
- content type
- cache-control
- artifact path

## Hosting

If the frontend deployment serves `frontend/public/tiles` directly, the PMTiles archive can ship with the frontend bundle.

If parcel archives are hosted in Cloudflare-backed object storage/CDN instead, upload the artifact using the object key from the publish manifest and preserve:

- `Content-Type: application/vnd.pmtiles`
- `Cache-Control: public, max-age=31536000, immutable`

Then point the state frontend config to the hosted archive URL.
