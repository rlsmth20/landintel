# Multi-State Architecture Summary

## Current Mississippi-Specific Hardcoding

The current reference implementation is still Mississippi-first in five main layers:

1. Entry points and module names
- `floodscraper/build_backend_parcel_runtime_ms.py`
- `floodscraper/sample_vacancy_labeling_ms.py`
- `floodscraper/vacancy_ai_build_dataset_ms.py`
- `floodscraper/vacancy_ai_infer_ms.py`
- `floodscraper/vacancy_ai_train_ms.py`
- `floodscraper/vacancy_ai_reviewed_pilot_ms.py`
- `backend/app/services/mississippi_leads_service.py`
- `backend/app/api/mississippi_leads.py`

2. Artifact paths and filenames
- `data/parcels/mississippi_parcels_master.parquet`
- `backend/runtime/mississippi/...`
- `frontend/public/data/mississippi_*`
- `data/buildings_processed/*_ms.*`

3. Runtime and API wiring
- `backend/main.py` mounts `mississippi_leads_router`
- `backend/app/settings.py` defaults to Mississippi feed/meta/geometry paths
- frontend static fallback paths point to Mississippi JSON assets

4. State-specific source registries
- `floodscraper/state_configs/parcel_source_ms.json`
- `floodscraper/state_configs/tax_source_ms.json`
- `floodscraper/state_configs/tax_linkage_ms.json`

5. Schema and naming assumptions
- file/module naming uses `_ms`
- runtime output names embed `mississippi`
- diagnostics and review/training paths assume Mississippi-specific filenames even when the logic is reusable

## Refactor Direction

The codebase now has a small multi-state foundation that preserves Mississippi compatibility while separating state configuration from reusable workflow scaffolding:

### State registry/config layer
- `config/states/registry.json`
- `config/states/ms.json`
- `config/states/ms_parcel_schema_mapping.json`
- `floodscraper/state_registry.py`

This layer defines:
- `state_code`
- `state_name`
- county-division label
- canonical artifact roots
- legacy compatibility paths
- source registry locations
- schema mapping template locations

### Reusable bootstrap/diagnostic layer
- `floodscraper/bootstrap_state.py`
- `floodscraper/state_diagnostics.py`

This layer provides:
- new-state config/bootstrap scaffolding
- per-state artifact directory creation
- placeholder parcel/tax/linkage registries
- state diagnostics for row counts, county coverage, schema mapping, geometry quality, marketability, review eligibility, and reviewed-pilot metrics

### First state-aware workflow
- `floodscraper/vacancy_ai_reviewed_pilot_ms.py`

This still uses Mississippi review/training logic, but its default input/output resolution is now driven by `state_code` and `run_name` through the state registry instead of hardcoded file constants alone.

## Target Layering

### Canonical layer
Responsibilities:
- parcel identity contract
- parcel schema mapping
- canonical parcel artifact path resolution

State-owned through config:
- parcel master path
- schema mapping file
- source registry files

### Feature layers
Responsibilities:
- geometry quality
- marketability
- vacancy imagery/review/training artifacts
- tax linkage/source coverage

Rule:
- feature layers append fields and artifacts
- they do not redefine state identity or artifact layout

### Output builders
Responsibilities:
- runtime parquet
- API payloads
- frontend fallback assets
- review/training manifests

Rule:
- output builders map from canonical + feature-enriched data
- output naming and root directories should be state-configured

## Compatibility Strategy

Mississippi remains the reference implementation.

For now:
- legacy Mississippi artifact paths remain valid through `config/states/ms.json`
- Mississippi scripts keep their existing filenames
- state-aware helpers resolve to legacy paths for Mississippi where needed

This keeps the live Mississippi pipeline reproducible while making future states use the same registry/bootstrap pattern from the start.

## Immediate Next Refactor Targets

1. Lift artifact-path constants out of:
- `floodscraper/vacancy_ai_common.py`
- `floodscraper/build_backend_parcel_runtime_ms.py`
- `backend/app/services/mississippi_leads_service.py`

2. Add generic wrappers around Mississippi-specific builders before renaming the underlying scripts.

3. Move backend/frontend runtime asset resolution behind a state registry rather than `mississippi_*` file constants.

4. Keep compatibility shims until a second state is live and the generic flow is proven.
