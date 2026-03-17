# Mississippi Parcel Detail Contract

This is the authoritative contract for the parcel detail paths used by the leads explorer.

## Identity
- `parcel_row_id`: internal stable row key
- `parcel_id`: canonical external parcel identifier for display/export

## Field ownership
- Parcel identity, ownership, tax, physical, and score fields:
  - source of truth: parcel master + owner/signals/tax runtime build
  - shipped in: runtime parcel index, app-ready leads, fallback detail JSON
- Tax interpretation fields:
  - source of truth: backend/service normalization
  - shipped in: runtime detail metrics, live detail payload, fallback detail JSON
- Vacancy assessment fields:
  - source of truth: backend/service derivation from parcel evidence plus AI when available
  - shipped in: live detail payload, fallback detail JSON

## AI vacancy fields
- `ai_building_present_probability`
- `ai_building_present_flag`
- `building_present_confidence`
- `building_presence_reason`
- `ai_vacancy_available_flag`
- `ai_vacancy_source`
- `ai_vacancy_status_note`
- `vacancy_model_version`

## AI behavior
- Preferred path: precomputed Mississippi AI predictions merged into runtime artifacts.
- Current stable behavior:
  - live detail: may add AI on demand when precomputed AI is missing
  - fallback/static detail: explicitly marks AI as unavailable unless precomputed AI has been shipped
- `ai_vacancy_source` values:
  - `precomputed`
  - `on_demand`
  - `unavailable`

## Rule
Null AI fields must not be presented as a model result. If no prediction is present, detail payloads must carry explicit AI availability/source/status fields.
