# Multi-State Rollout Status

Updated: 2026-03-28

## Standard

The parcel base layer must represent the full statewide parcel master.

- Parcel master: all parcels statewide
- Parcel overlay or PMTiles: all parcels statewide
- `app_ready` and default leads: separate filtered business layer
- Lead filtering must never determine which parcels appear on the map

## Compliance Audit

| State | Parcel Master | Tile Coverage | Geometry Coverage | Runtime Detail | Map Shows All Parcels | Geometry Source | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ms` | full, `2,008,479` rows | full | full | full | true | local cached | compliant | Reference state preserved. |
| `ar` | full, `2,103,055` rows | full | full | full | true | mixed | compliant | Full parcel-master PMTiles rebuilt from paged statewide polygon pulls. |
| `ct` | full, `1,222,185` rows | full | full | full | true | mixed | compliant | Closed `5` missing rows with `3` official source centroids plus `2` prefix-centroid fallback points. |
| `mt` | full, `885,499` rows | full | full | full | true | mixed | compliant | Closed `68,101` skipped rows with object-id reconciliation after paged fetch. |
| `vt` | full, `302,575` rows | full | full | full | true | mixed | compliant | Town-based full parcel-master PMTiles rebuilt successfully. |
| `wi` | full, `3,507,460` rows | full | full | full | true | mixed | compliant | Removed old lead-subset overlay path and rebuilt from `parcel_master`. |
| `ny` | full canonical statewide master, `5,225,972` rows across `62` counties | full | full | full | true | mixed | compliant | New York now ships as a statewide centroid-based canonical parcel state. The remaining raw-source row gap is due to duplicate parcel identities and null-ID rows in the official feed, not a county-coverage gap. |

## Root Causes Closed In This Pass

- `ct`
  - Root cause: `5` official source rows returned null polygon geometry in the statewide parcel layer.
  - Fix: the shared cache builder now falls back to official source centroids and then to prefix-based centroid hints from nearby same-prefix parcels when the source exposes no polygon.

- `mt`
  - Root cause: offset-based paged geometry pulls skipped a high object-id block even though direct object-id queries returned valid polygons.
  - Fix: the shared cache builder now reconciles missing rows with direct object-id geometry batches after the paged pass.

- `wi`
  - Root cause: Wisconsin was still configured to build parcel tiles from the `app_ready` subset.
  - Fix: the state now builds PMTiles from `parcel_master`, and the full statewide geometry cache was rebuilt.

- `ny`
  - Root cause: the checked-in New York artifacts were stale partial outputs, and the shared ArcGIS builder was not surfacing source-to-local parity clearly enough.
  - Fix: New York now rebuilds from the live statewide centroid service into a canonical parcel master with all `62` counties, global parcel-row-id deduplication, refreshed runtime artifacts, and a refreshed statewide PMTiles overlay.
  - Remaining source-data limitation: the official statewide source is still centroid-only, so statewide polygon parcel boundaries are not available from that feed.

## New York Source Note

New York is now rollout-ready as a statewide centroid parcel state.

Current state:

- the local canonical parcel master now covers all `62` counties
- the runtime and default-lead artifacts are rebuilt from that statewide base
- the parcel PMTiles overlay is rebuilt and published from that statewide base
- the official public statewide source remains centroid-only, so polygon parcel boundaries are still a separate sourcing problem

## Shared Improvements

- Shared PMTiles builder now supports:
  - paged statewide ArcGIS pulls
  - object-id reconciliation for skipped rows
  - official source centroid fallback
  - point-capable PMTiles output for centroid-only states and missing-polygon edge cases
- Shared diagnostics now report:
  - parcel master row count
  - official-source alignment audits for states that build from live ArcGIS parcel services
  - tile coverage vs parcel master
  - geometry coverage vs parcel master
  - runtime detail coverage vs parcel master
  - map shows all parcels true or false
  - geometry source type
  - blocker reason when applicable
- Frontend parcel overlays now render point features as well as polygons, so centroid-only parcel states no longer need to disable the base overlay.

## Remaining Blocker

- `ut`
  - Still blocked during source validation because the candidate source returned only county FIPS `49049` rather than statewide coverage.
