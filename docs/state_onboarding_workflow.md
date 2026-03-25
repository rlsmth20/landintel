# State Onboarding Workflow

## Purpose

This workflow bootstraps a new state without breaking the working Mississippi reference pipeline.

## Bootstrap

Run:

```powershell
python floodscraper\bootstrap_state.py <state_code> "<State Name>"
```

Example:

```powershell
python floodscraper\bootstrap_state.py al "Alabama"
```

This creates:
- `config/states/<state_code>.json`
- `config/states/<state_code>_parcel_schema_mapping.json`
- `floodscraper/state_configs/parcel_source_<state_code>.json`
- `floodscraper/state_configs/tax_source_<state_code>.json`
- `floodscraper/state_configs/tax_linkage_<state_code>.json`
- `data/parcels/<state_code>/`
- `data/runtime/<state_code>/`
- `data/review/<state_code>/`
- `data/training/<state_code>/`

## Minimum State Setup

Populate the new state config with:
- parcel master artifact path
- app-ready/runtime input paths if they already exist
- runtime summary path
- review sample path
- reviewed-pilot input/output paths if available
- geometry-quality artifact path

Populate the parcel schema mapping template with:
- `parcel_row_id`
- `parcel_id`
- `state_code`
- `county_name`
- `county_fips`
- `geometry`

## Source Registration

Use the per-state placeholder registries to define:
- parcel source discovery inputs
- tax source discovery inputs
- linkage rules/config

Keep these state-owned:
- source URLs
- source formats
- identifier mapping rules
- county/state coverage metadata

## Canonical Parcel Readiness Checklist

Before feature work starts, confirm:
- canonical parcel identity fields are mapped
- county coverage exists
- geometry field is valid
- schema mapping file reflects source reality
- output roots exist for runtime/review/training artifacts

## Feature Workflow Checklist

For each new state:

1. Canonical parcel artifact
- parcel master readable
- row count known
- county coverage summarized

2. Geometry quality artifact
- build reusable geometry-quality parquet
- confirm exclusion counts and county distribution

3. Marketability layer
- validate marketability/action counts
- confirm default-lead exclusion diagnostics

4. Vacancy review/training path
- review sample path defined
- reviewed labels path defined if available
- reviewed pilot outputs routed through the state registry

5. Runtime/output layer
- runtime summary path defined
- frontend fallback targets defined if applicable
- backend asset resolution uses configured state paths

## Diagnostics

Generate a state summary with:

```powershell
python floodscraper\state_diagnostics.py <state_code> --output data\training\<state_code>\state_diagnostics.json
```

The diagnostics artifact reports:
- parcel/app-ready row counts
- county coverage summary
- schema mapping summary
- geometry-quality overview
- marketability summary
- review eligibility summary
- reviewed-pilot metrics summary

## Recommended Rollout Order

1. Bootstrap config and folders
2. Map canonical parcel schema
3. Build state diagnostics
4. Wire canonical/runtime artifact paths
5. Enable review/training outputs
6. Only then add state-specific ingestion or UI/runtime endpoints

## Guardrails

- Do not rename canonical identifiers per state.
- Do not let feature layers overwrite canonical fields.
- Keep Mississippi legacy paths intact until a second state reproduces the same outputs through config-driven resolution.
- Prefer generic wrappers and config routing before renaming Mississippi modules.
