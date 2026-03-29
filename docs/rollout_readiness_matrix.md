# Rollout Readiness Matrix

Updated: 2026-03-28

This matrix is the current rollout-readiness view for the active multi-state set. It separates parcel-map compliance from broader product readiness.

Legend:

- `production-ready`: safe to treat as a live reference state
- `MVP-ready`: end-to-end parcel experience works, but key business enrichments are still deferred
- `partial`: significant statewide coverage or product gaps remain
- `blocked`: hard external dependency still unresolved

## State Matrix

| State | Parcel Map Compliance | Runtime / Detail Readiness | Default Leads Readiness | Tax Linkage Readiness | Vacancy Imagery / Review / Training | PMTiles Hosting Readiness | Known Source Limitations | Overall Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ms` | compliant; full parcel master `2,008,479`, map shows all parcels | production-ready; full runtime detail | production-ready; `11,432` default leads | production-ready with county-specific linkage caveats | production-ready; review and reviewed-pilot artifacts exist | public-ready; R2-hosted PMTiles validates by PMTiles signature | legacy runtime layout still preserved; Hinds and Jackson heuristic tax variants remain disabled | production-ready |
| `ar` | compliant; full parcel master `2,103,055`, map shows all parcels | MVP-ready; full runtime detail | MVP-ready; `50,000` default leads | not ready; tax source and linkage both `not_implemented_mvp` | not ready; no review or training outputs beyond diagnostics | public-ready; R2-hosted PMTiles validates by PMTiles signature | live ArcGIS polygon fetch path; no county-hosted tax or vacancy-imagery parity yet | MVP-ready |
| `ct` | compliant; full parcel master `1,222,185`, map shows all parcels | MVP-ready; full runtime detail | MVP-ready; `50,000` default leads | not ready; tax source and linkage are stubs | not ready; no review or training outputs beyond diagnostics | public-ready; R2-hosted PMTiles validates by PMTiles signature | town-based statewide parcel layer; tax workflow still deferred | MVP-ready |
| `mt` | compliant; full parcel master `885,499`, map shows all parcels | MVP-ready; full runtime detail | MVP-ready; `50,000` default leads | not ready; tax source and linkage are placeholders | not ready; no review or training outputs beyond diagnostics | public-ready; R2-hosted PMTiles validates by PMTiles signature | relies on live statewide ArcGIS polygon fetches; no separate tax linkage yet | MVP-ready |
| `vt` | compliant; full parcel master `302,575`, map shows all parcels | MVP-ready; full runtime detail | MVP-ready; `50,000` default leads | not ready; tax source and linkage are stubs | not ready; no review or training outputs beyond diagnostics | public-ready; R2-hosted PMTiles validates by PMTiles signature | town-based statewide parcel layer; tax workflow still deferred | MVP-ready |
| `wi` | compliant; full parcel master `3,507,460`, map shows all parcels | MVP-ready; full runtime detail | MVP-ready; `50,000` default leads | not ready; tax source and linkage are placeholders | not ready; no review or training outputs beyond diagnostics | public-ready; R2-hosted PMTiles validates by PMTiles signature | statewide parcel layer is healthy, but tax and vacancy parity are still deferred | MVP-ready |
| `ny` | compliant; full canonical statewide master `5,225,972` rows across `62` counties | MVP-ready; full runtime detail from the statewide canonical base | MVP-ready; `50,000` default leads from the statewide canonical base | not ready; tax source and linkage are placeholders | not ready; no review or training outputs beyond diagnostics | public-ready; R2-hosted PMTiles validates by PMTiles signature | official statewide source is centroid-only; raw source contains duplicate/null-ID rows, so raw row parity stays below the official count even though canonical county coverage is statewide | MVP-ready |
| `ut` | not compliant; no statewide parcel overlay built | blocked | blocked | blocked | blocked | no PMTiles artifact or publish manifest | candidate source validated during the last pass returned only county FIPS `49049`, not statewide coverage | blocked |

## PMTiles Hosting Audit

Current hosting posture:

- Cloud-hosted PMTiles with checked-in public URLs and PMTiles-signature validation: `ms`, `ar`, `ct`, `mt`, `ny`, `vt`, `wi`
- No PMTiles artifact: `ut`

Current production contract:

- Production resolves parcel overlays from the checked-in per-state `public_url`.
- Local development falls back to `/tiles/<state>_parcels.pmtiles` or `/tiles/mississippi_parcels.pmtiles`.
- `NEXT_PUBLIC_USE_LOCAL_PARCEL_PMTILES=1` forces the local fallback even when a public URL is configured.
- Publish manifests now include the aligned `object_key`, `frontend_url`, and `public_url` for each active state.

Current production blockers for parcel overlays:

- None for the active parcel-map states. The configured public URLs now serve real PMTiles bytes and validate successfully.
- `ut` remains outside the hosting contract because it still has no validated statewide parcel source and no PMTiles artifact.

## Highest-Value Next Move Per State

- `ms`: extend Mississippi-era vacancy/training parity into newer states rather than more parcel-base work.
- `ar`: implement real tax source ingestion and linkage before vacancy parity work.
- `ct`: add a reusable town-based tax linkage workflow.
- `mt`: add real tax linkage and replace placeholder tax configs.
- `vt`: add a reusable town-based tax linkage workflow.
- `wi`: implement real tax linkage; the parcel base is ready for it.
- `ny`: decide whether polygon expansion should be hybrid county-plus-state or remain an explicit centroid-only product limitation.
- `ut`: validate a true statewide official parcel source before doing any more onboarding work.

## Recommended Next Engineering Priority

Close the remaining product-enrichment gaps on top of the now-stable multi-state parcel base.

Reason:

- parcel-map compliance and production PMTiles hosting are now stable for every active state except Utah
- New York now has a statewide canonical parcel base and no longer blocks MVP rollout
- the next rollout risk is business-layer parity: tax linkage and vacancy review/training outside Mississippi

## Recommended Next State To Onboard

`nj` is the best next state to onboard.

Reason:

- New Jersey publishes an official statewide parcel polygon composite
- the state also publishes a statewide parcels-plus-MOD-IV joined layer with parcel identifiers and assessment fields
- the source is maintained as a current ArcGIS Online hosted feature service and also has download options, which fits the existing generic state-registry and ArcGIS-driven onboarding model

## Remaining Production Blockers

- Tax linkage is only materially ready in Mississippi
- Vacancy imagery/review/training parity exists only in Mississippi
