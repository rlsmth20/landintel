# Utilities Layer

The Mississippi utilities layer combines official utility territory polygons, official service-area polygons, and utility distance metrics for parcel-level analysis.

Utility territory datasets represent the official service area assigned to the utility provider by the state regulator. Being inside the territory means the provider is responsible for offering service in the area, but it does not guarantee an existing connection at the parcel.

## Territory and Service-Area Fields

`electric_in_service_territory`
TRUE when a parcel intersects the official PSC-certified electric utility territory polygon.

`gas_in_service_territory`
TRUE when a parcel intersects the official PSC-certified gas utility territory polygon.

`water_service_area`
TRUE when a parcel intersects an official municipal or public water service boundary.

`sewer_service_area`
TRUE when a parcel intersects an official municipal or PSC-certified sewer service boundary.

These fields indicate territory or service-area membership only. They do not confirm that a parcel has an active meter, lateral, tap, or live utility hookup.

## Provider Fields

`electric_provider_name`
Provider name from the intersecting official electric territory polygon, when present.

`gas_provider_name`
Provider name from the intersecting official gas territory polygon, when present.

`water_provider_name`
Provider name from the intersecting official water service-area polygon, when present.

`sewer_provider_name`
Provider name from the intersecting official sewer service-area polygon, when present.

## Distance Metrics

Distance metrics are retained as the primary parcel-level indicators of connection feasibility:

- `distance_to_powerline`
- `distance_to_substation`
- `distance_to_pipeline`

These metrics are proximity measures only and should be interpreted separately from official service-territory fields.

## Mississippi Sources

- Electric territory: Mississippi PSC `PSC_CurrentCAs` electric layer
- Gas territory: Mississippi PSC `PSC_CurrentCAs` gas layer
- Sewer territory: Mississippi PSC `PSC_CurrentCAs` sewer layer
- Water service boundaries: public water system boundary polygons
- Utility distance metrics: Mississippi transmission lines, substations, and pipelines

## Interpretation

Use territory or service-area fields to determine whether a parcel falls inside an official utility responsibility area.

Use distance metrics to evaluate probable connection difficulty or extension feasibility.

Do not interpret territory membership as proof of an existing parcel hookup.
