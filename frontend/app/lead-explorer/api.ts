import type { ExplorerMeta, GeometryPoint, GeometryResponse, LeadRecord, LeadsResponse, PresetItem, SortField, Filters } from "./types";

const DEFAULT_PRODUCTION_API_BASE_URL = "https://landintel-production.up.railway.app";
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  (process.env.NODE_ENV === "production" ? DEFAULT_PRODUCTION_API_BASE_URL : "");

async function fetchJson<T>(path: string, searchParams?: URLSearchParams, options?: { timeoutMs?: number }): Promise<T> {
  const url = `${API_BASE_URL}${path}${searchParams && searchParams.toString() ? `?${searchParams.toString()}` : ""}`;
  if (process.env.NODE_ENV !== "production") {
    console.debug("[lead-explorer] request", url);
  }
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options?.timeoutMs ?? 10000);
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

let staticMetaCache: Record<string, unknown> | null = null;
let staticLeadCache: LeadRecord[] | null = null;
let staticLeadDetailCache: LeadRecord[] | null = null;

async function fetchStaticJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Static request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function fetchStaticMetaSource() {
  if (!staticMetaCache) {
    staticMetaCache = await fetchStaticJson<Record<string, unknown>>("/data/mississippi_lead_explorer_meta.json");
  }
  return staticMetaCache;
}

async function fetchStaticLeadSource() {
  if (!staticLeadCache) {
    staticLeadCache = await fetchStaticLeadDetailSource();
  }
  return staticLeadCache;
}

async function fetchStaticLeadDetailSource() {
  if (!staticLeadDetailCache) {
    staticLeadDetailCache = await fetchStaticJson<LeadRecord[]>("/data/mississippi_lead_detail_fallback.json");
  }
  return staticLeadDetailCache;
}

export async function fetchStaticLeadDetail(parcelRowId: string): Promise<LeadRecord | null> {
  const rows = await fetchStaticLeadDetailSource();
  return rows.find((row) => row.parcel_row_id === parcelRowId) ?? null;
}

export async function fetchSummary(): Promise<ExplorerMeta> {
  try {
    return await fetchJson<ExplorerMeta>("/api/summary");
  } catch {
    const source = await fetchStaticMetaSource();
    return {
      row_count: Number(source.rowCount ?? 0),
      source: typeof source.source === "string" ? source.source : "static explorer fallback",
      geometry_mode: typeof source.geometryMode === "string" ? source.geometryMode : undefined,
      geometry_bounds: Array.isArray(source.geometryBounds) ? (source.geometryBounds as number[]) : undefined,
      geometry_view_box: Array.isArray(source.geometryViewBox) ? (source.geometryViewBox as number[]) : undefined,
      sections: {
        statewide: [],
        top_counties: [],
        recommended_view_bucket: [],
      },
    };
  }
}

export async function fetchPresets(): Promise<PresetItem[]> {
  try {
    const response = await fetchJson<{ items: PresetItem[] }>("/api/presets");
    return response.items;
  } catch {
    const source = await fetchStaticMetaSource();
    const defaultViews = Array.isArray(source.defaultViews) ? (source.defaultViews as Array<Record<string, string>>) : [];
    const grouped = new Map<string, PresetItem>();
    defaultViews.forEach((item) => {
      const key = item.view_name;
      if (!key) return;
      const current = grouped.get(key) ?? {
        view_name: key,
        description: item.description,
        filter_expression: item.filter_expression,
      };
      if (item.metric === "row_count") current.row_count = item.value;
      if (item.metric === "average_lead_score") current.average_lead_score = item.value;
      grouped.set(key, current);
    });
    return [...grouped.values()];
  }
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
  try {
    return await fetchJson<LeadsResponse>("/api/leads", buildLeadQuery(filters, sortField, sortDirection, limit, offset));
  } catch {
    const rows = await fetchStaticLeadSource();
    const sorted = [...rows].sort((left, right) => {
      const leftValue = (left as Record<string, unknown>)[sortField] as number | null | undefined;
      const rightValue = (right as Record<string, unknown>)[sortField] as number | null | undefined;
      const a = leftValue ?? (sortDirection === "asc" ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY);
      const b = rightValue ?? (sortDirection === "asc" ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY);
      return sortDirection === "asc" ? a - b : b - a;
    });
    const paged = sorted.slice(offset, offset + limit);
    return {
      total_count: rows.length,
      limit,
      offset,
      items: paged,
    };
  }
}

export async function fetchLeadDetail(parcelRowId: string): Promise<LeadRecord> {
  return fetchJson<LeadRecord>(`/api/leads/${parcelRowId}`);
}

export async function fetchParcelGeometryById(parcelRowId: string, zoom = 14): Promise<GeometryResponse> {
  const searchParams = new URLSearchParams();
  searchParams.set("zoom", String(zoom));
  if (process.env.NODE_ENV !== "production") {
    console.debug("[lead-explorer] parcel geometry request", { parcelRowId, zoom });
  }
  return fetchJson<GeometryResponse>(`/api/parcels/${parcelRowId}/geometry`, searchParams, { timeoutMs: 6000 });
}
