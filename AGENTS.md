Good. Do not spend more time forcing Hinds linkage.

Next step: identify and onboard the next best free county-hosted structured Mississippi tax-sale or delinquent-tax source that is most likely to link successfully to parcel master.

Selection criteria, in priority order:
1. source is genuinely free and downloadable
2. structured format preferred: CSV, XLSX, ZIP, GeoJSON, GPKG
3. source includes PPIN, parcel ID, or another identifier likely to align to parcel master
4. source has lower risk of source-side duplicate identifiers than Hinds/SOS
5. source fits the existing free-tax ingestion architecture

Tasks:

1. Search the current discovery outputs/registry and identify the best next candidate county source.
2. Rank candidate counties by expected linkage potential, not just by availability.
3. Explain why the chosen county is the best next onboarding target.
4. Ingest that source through the same free-tax pipeline architecture:
   - raw snapshot
   - manifest
   - standardization
   - linkage
   - diagnostics
   - QA
   - registry update
   - statewide summary update
5. Preserve exact vs heuristic match separation.
6. Produce county-level outputs:
   - standardized records
   - linked records
   - unmatched records
   - ambiguous records
   - linkage summary
   - unmatched reason summary
   - ambiguity reason summary
   - identifier diagnostics
7. Compare the new county’s linkage quality against Pike, SOS, and Hinds.

Important:
- Do not create one-off logic outside the shared pipeline.
- Do not use unsafe fuzzy matching to inflate linkage rate.
- Prefer a county that can become a strong linkage template for future onboarding.
- If no good candidate exists in current discovery results, expand discovery for additional genuinely free Mississippi county structured tax-sale sources and rank those candidates.

After implementation, report:
- selected county and why
- total ingested
- linked / unmatched / ambiguous counts
- exact vs heuristic match rates
- whether this county is a good linkage template
- what source characteristics best predict successful parcel linkage