import type { ExplorerMeta, GeometryResponse, LeadRecord, LeadsResponse, PresetItem, SortField, Filters } from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";

async function fetchJson<T>(path: string, searchParams?: URLSearchParams): Promise<T> {
  const url = `${API_BASE_URL}${path}${searchParams && searchParams.toString() ? `?${searchParams.toString()}` : ""}`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchSummary(): Promise<ExplorerMeta> {
  return fetchJson<ExplorerMeta>("/api/summary");
}

export async function fetchPresets(): Promise<PresetItem[]> {
  const response = await fetchJson<{ items: PresetItem[] }>("/api/presets");
  return response.items;
}

function appendList(searchParams: URLSearchParams, key: string, values: string[]) {
  values.forEach((value) => searchParams.append(key, value));
}

export function buildLeadQuery(
  filters: Filters,
  sortField: SortField,
  sortDirection: "asc" | "desc",
  limit: number,
  offset: number,
) {
  const searchParams = new URLSearchParams();
  if (filters.countyName !== "all") searchParams.set("county_name", filters.countyName);
  appendList(searchParams, "lead_score_tier", filters.leadScoreTier);
  searchParams.set("min_lead_score_total", String(filters.minLeadScore));
  if (filters.acreageMin !== "") searchParams.set("acreage_min", filters.acreageMin);
  if (filters.acreageMax !== "") searchParams.set("acreage_max", filters.acreageMax);
  if (filters.parcelVacantOnly) searchParams.set("parcel_vacant_flag", "true");
  if (filters.countyHostedOnly) searchParams.set("county_hosted_flag", "true");
  if (filters.highConfidenceOnly) searchParams.set("high_confidence_link_flag", "true");
  if (filters.wetlandMode === "exclude") searchParams.set("wetland_flag", "false");
  if (filters.wetlandMode === "only") searchParams.set("wetland_flag", "true");
  appendList(searchParams, "amount_trust_tier", filters.amountTrustTiers);
  if (filters.corporateOnly) searchParams.set("corporate_owner_flag", "true");
  if (filters.absenteeOnly) searchParams.set("absentee_owner_flag", "true");
  if (filters.outOfStateOnly) searchParams.set("out_of_state_owner_flag", "true");
  appendList(searchParams, "growth_pressure_bucket", filters.growthPressureBuckets);
  if (filters.recommendedViewBucket !== "all") searchParams.append("recommended_view_bucket", filters.recommendedViewBucket);
  appendList(searchParams, "road_access_tier", filters.roadAccessTiers);
  if (filters.roadDistanceMax !== "") searchParams.set("road_distance_ft_max", filters.roadDistanceMax);
  searchParams.set("sort_by", sortField);
  searchParams.set("sort_direction", sortDirection);
  searchParams.set("limit", String(limit));
  searchParams.set("offset", String(offset));
  return searchParams;
}

export async function fetchLeads(
  filters: Filters,
  sortField: SortField,
  sortDirection: "asc" | "desc",
  limit: number,
  offset: number,
): Promise<LeadsResponse> {
  return fetchJson<LeadsResponse>("/api/leads", buildLeadQuery(filters, sortField, sortDirection, limit, offset));
}

export async function fetchLeadDetail(parcelRowId: string): Promise<LeadRecord> {
  return fetchJson<LeadRecord>(`/api/leads/${parcelRowId}`);
}

export async function fetchGeometry(parcelRowIds: string[]): Promise<GeometryResponse> {
  const searchParams = new URLSearchParams();
  parcelRowIds.forEach((id) => searchParams.append("parcel_row_id", id));
  return fetchJson<GeometryResponse>("/api/leads/geometry", searchParams);
}
