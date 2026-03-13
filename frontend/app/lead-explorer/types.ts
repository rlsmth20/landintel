export type GeometryPoint = {
  type: "Point";
  coordinates: [number, number];
};

export type GeometryPayload = {
  type: string;
  centroid?: GeometryPoint | null;
  bounds?: [number, number, number, number] | null;
};

export type GeometryBounds = [number, number, number, number];

export type GeoJsonGeometry = {
  type: string;
  coordinates?: unknown;
};

export type GeometryFeature = {
  type: "Feature";
  geometry: GeoJsonGeometry | null;
  properties: {
    parcel_row_id: string;
    parcel_id?: string | null;
    county_name?: string | null;
    lead_score_total?: number | null;
    lead_score_tier?: string | null;
    parcel_vacant_flag?: boolean | null;
    wetland_flag?: boolean | null;
    flood_risk_score?: number | null;
    road_access_tier?: string | null;
    county_hosted_flag?: boolean | null;
    best_source_type?: string | null;
    selected?: boolean;
  };
};

export type FeatureCollectionPayload = {
  type: "FeatureCollection";
  features: GeometryFeature[];
};

export type LeadRecord = {
  parcel_row_id: string;
  parcel_id: string | null;
  county_name: string | null;
  county_fips: string | null;
  state_code: string | null;
  geometry?: GeometryPayload | null;
  acreage: number | null;
  acreage_bucket: string | null;
  land_use: string | null;
  parcel_vacant_flag: boolean | null;
  building_count: number | null;
  building_area_total: number | null;
  nearby_building_count_1km: number | null;
  nearby_building_density: number | null;
  growth_pressure_bucket: string | null;
  road_distance_ft: number | null;
  road_access_tier: string | null;
  wetland_flag: boolean | null;
  flood_risk_score: number | null;
  buildability_score: number | null;
  environment_score: number | null;
  investment_score: number | null;
  electric_provider_name: string | null;
  owner_name: string | null;
  owner_type: string | null;
  corporate_owner_flag: boolean | null;
  absentee_owner_flag: boolean | null;
  out_of_state_owner_flag: boolean | null;
  owner_parcel_count: number | null;
  owner_total_acres: number | null;
  mailer_target_score: number | null;
  delinquent_amount: number | null;
  delinquent_amount_bucket: string | null;
  delinquent_flag: boolean | null;
  forfeited_flag: boolean | null;
  best_source_type: string | null;
  best_source_name: string | null;
  source_confidence_tier: string | null;
  county_source_coverage_tier: string | null;
  amount_trust_tier: string | null;
  high_confidence_link_flag: boolean | null;
  county_hosted_flag: boolean | null;
  lead_score_total: number | null;
  lead_score_tier: string | null;
  lead_score_driver_1: string | null;
  lead_score_driver_2: string | null;
  lead_score_driver_3: string | null;
  lead_score_explanation: string | null;
  size_score: number | null;
  access_score: number | null;
  buildability_component: number | null;
  environmental_component: number | null;
  owner_targeting_component: number | null;
  delinquency_component: number | null;
  source_confidence_component: number | null;
  vacant_land_component: number | null;
  growth_pressure_component: number | null;
  recommended_sort_reason: string | null;
  top_score_driver: string | null;
  caution_flags: string | null;
  vacant_reason: string | null;
  growth_pressure_reason: string | null;
  recommended_use_case: string | null;
  recommended_view_bucket: string | null;
};

export type ExplorerMeta = {
  row_count?: number;
  source?: string;
  geometry_mode?: string;
  geometry_bounds?: number[];
  geometry_view_box?: number[];
  sections?: Record<string, Record<string, string>[]>;
};

export type PresetItem = {
  view_name: string;
  description?: string;
  filter_expression?: string;
  row_count?: string;
  average_lead_score?: string;
};

export type GeometryItem = {
  parcel_row_id: string;
  path: string | null;
  lead_score_total: number | null;
};

export type GeometryResponse = {
  geometry_mode?: string;
  render_mode?: "none" | "points" | "centroids" | "polygons";
  geometry_bounds?: number[];
  geometry_view_box?: number[];
  requested_bounds?: number[];
  zoom?: number | null;
  feature_count?: number;
  feature_collection?: FeatureCollectionPayload;
  items: GeometryItem[];
};

export type LeadsResponse = {
  total_count: number;
  limit: number;
  offset: number;
  items: LeadRecord[];
};

export type Filters = {
  countyName: string;
  leadScoreTier: string[];
  minLeadScore: number;
  acreageMin: string;
  acreageMax: string;
  parcelVacantOnly: boolean;
  countyHostedOnly: boolean;
  highConfidenceOnly: boolean;
  wetlandMode: "any" | "exclude" | "only";
  amountTrustTiers: string[];
  corporateOnly: boolean;
  absenteeOnly: boolean;
  outOfStateOnly: boolean;
  growthPressureBuckets: string[];
  recommendedViewBucket: string;
  roadDistanceMax: string;
  roadAccessTiers: string[];
};

export type SortField = "lead_score_total" | "acreage" | "delinquent_amount" | "road_distance_ft";

export type MapOverlayId =
  | "parcels"
  | "fema_flood"
  | "wetlands"
  | "utilities"
  | "slope"
  | "road_access"
  | "zoning";

export type MapOverlayDefinition = {
  id: MapOverlayId;
  label: string;
  description: string;
  enabled: boolean;
};

export type MapViewportState = {
  center: [number, number];
  zoom: number;
  bounds: GeometryBounds | null;
};
