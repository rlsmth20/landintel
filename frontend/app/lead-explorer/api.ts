import type { ExplorerMeta, GeometryPoint, GeometryResponse, LeadRecord, LeadsResponse, NearbyCompsResponse, PresetItem, SearchResponse, SearchResultRecord, SortField, Filters } from "./types";
import stateConfig from "./stateConfig";

const DEFAULT_PRODUCTION_API_BASE_URL = "https://landintel-production.up.railway.app";
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  (process.env.NODE_ENV === "production" ? DEFAULT_PRODUCTION_API_BASE_URL : "");

function requireStateCode(stateCode: string, callSite: string): string {
  const normalized = typeof stateCode === "string" ? stateCode.trim().toLowerCase() : "";
  if (!normalized) {
    const message = `[lead-explorer] missing stateCode for ${callSite}`;
    console.error(message);
    throw new Error(message);
  }
  return normalized;
}

function buildStateApiPath(stateCode: string, suffix: string) {
  return stateConfig.buildStateApiPath(requireStateCode(stateCode, "buildStateApiPath"), suffix);
}

async function fetchJson<T>(
  path: string,
  searchParams?: URLSearchParams,
  options?: { timeoutMs?: number; signal?: AbortSignal; stateCode?: string },
): Promise<T> {
  const url = `${API_BASE_URL}${path}${searchParams && searchParams.toString() ? `?${searchParams.toString()}` : ""}`;
  console.info("[lead-explorer] request", {
    stateCode: options?.stateCode ?? null,
    url,
  });
  const controller = new AbortController();
  const timeoutMs = options?.timeoutMs ?? 10000;
  const abortSignal = options?.signal;
  const abortListener = () => controller.abort();
  if (abortSignal) {
    if (abortSignal.aborted) {
      controller.abort();
    } else {
      abortSignal.addEventListener("abort", abortListener, { once: true });
    }
  }
  const timeout = timeoutMs > 0 ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", signal: controller.signal });
  } finally {
    if (timeout !== null) {
      window.clearTimeout(timeout);
    }
    if (abortSignal) {
      abortSignal.removeEventListener("abort", abortListener);
    }
  }
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

const staticMetaCache = new Map<string, Record<string, unknown>>();
const staticLeadCache = new Map<string, LeadRecord[]>();
const staticLeadDetailCache = new Map<string, LeadRecord[]>();

const PARCEL_ID_CANDIDATE_FIELDS = [
  "apn",
  "source_parcel_number",
  "source_alt_parcel_number",
  "source_ppin",
  "legacy_parcel_id",
  "source_parcel_id_normalized",
  "source_parcel_id_raw",
] as const;

const CANONICAL_DETAIL_NULL_FIELDS = [
  "ai_building_present_probability",
  "ai_building_present_flag",
  "ai_vacancy_available_flag",
  "ai_vacancy_source",
  "ai_vacancy_status_note",
  "assessed_total_value",
  "building_signal_conflict_flag",
  "building_presence_reason",
  "building_present_confidence",
  "county_tax_coverage_reason",
  "county_tax_coverage_status",
  "county_tax_source_configured_flag",
  "county_tax_source_loaded_flag",
  "county_vacant_flag",
  "delinquency_last_verified",
  "delinquent_year",
  "elevation_mean_ft",
  "flood_area_sqft",
  "flood_pct",
  "max_slope_pct",
  "mean_slope_pct",
  "overall_vacancy_assessment",
  "parcel_improvement_status",
  "parcel_improvement_confidence",
  "parcel_improvement_reason",
  "parcel_improvement_evidence_summary",
  "parcel_frontage_ft_estimate",
  "parcel_tax_status",
  "parcel_tax_status_label",
  "parcel_tax_status_category",
  "parcel_tax_actionability",
  "parcel_tax_data_warning",
  "parcel_tax_freshness_bucket",
  "parcel_tax_freshness_reason",
  "parcel_width_ft_estimate",
  "primary_fema_zone",
  "shape_compactness",
  "slope_class",
  "slope_score",
  "tax_data_available_flag",
  "tax_data_source",
  "tax_data_upload_date",
  "tax_data_year",
  "vacancy_confidence_score",
  "vacancy_model_version",
  "wetland_area_sqft",
  "wetland_pct",
] as const;

function normalizeParcelIdentifier(record: LeadRecord): LeadRecord {
  const normalized = { ...record } as LeadRecord & Record<string, unknown>;
  const rowId = typeof normalized.parcel_row_id === "string" ? normalized.parcel_row_id.trim() : "";
  const currentParcelId = typeof normalized.parcel_id === "string" ? normalized.parcel_id.trim() : "";
  const isInternalRowId = (value: string) => value === rowId || /^row_\d+$/i.test(value);
  const alternateParcelId = PARCEL_ID_CANDIDATE_FIELDS
    .map((field) => normalized[field])
    .find((value): value is string => typeof value === "string" && value.trim().length > 0 && !isInternalRowId(value.trim()));

  if (alternateParcelId) {
    normalized.parcel_id = alternateParcelId.trim();
    return normalized;
  }

  normalized.parcel_id = currentParcelId && !isInternalRowId(currentParcelId) ? currentParcelId : null;
  return normalized;
}

export function normalizeDetailLeadRecord(record: LeadRecord): LeadRecord {
  const normalized = normalizeParcelIdentifier(record) as LeadRecord & Record<string, unknown>;
  for (const field of CANONICAL_DETAIL_NULL_FIELDS) {
    if (!(field in normalized)) {
      normalized[field] = null;
    }
  }
  const hasAiPrediction =
    normalized.ai_building_present_flag !== null ||
    normalized.ai_building_present_probability !== null ||
    normalized.building_present_confidence !== null ||
    normalized.building_presence_reason !== null;
  if (normalized.ai_vacancy_source == null) {
    normalized.ai_vacancy_source = hasAiPrediction ? "precomputed" : "unavailable";
  }
  if (normalized.ai_vacancy_available_flag == null) {
    normalized.ai_vacancy_available_flag = hasAiPrediction;
  }
  if (normalized.ai_vacancy_status_note == null) {
    normalized.ai_vacancy_status_note = hasAiPrediction
      ? "Precomputed AI vacancy prediction is available for this parcel."
      : "AI vacancy prediction is unavailable in this parcel detail source.";
  }
  return normalized;
}

async function fetchStaticJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(`Static request failed: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function buildEmptyMetaSource(stateCode: string): Record<string, unknown> {
  return {
    defaultViews: [],
    fieldReadiness: [],
    summary: [],
    rowCount: 0,
    source: `missing static explorer fallback for ${stateCode}`,
    geometryMode: null,
    geometryBounds: null,
    geometryViewBox: null,
    geometrySimplifyTolerance: null,
    warnings: [`Static explorer metadata is unavailable for ${stateCode}.`],
  };
}

function logStaticFallbackFailure(stateCode: string, path: string, error: unknown) {
  console.error("[lead-explorer] static fallback load failed", {
    stateCode,
    path,
    error: error instanceof Error ? error.message : String(error),
  });
}

async function fetchStaticMetaSource(stateCode: string) {
  const normalizedStateCode = requireStateCode(stateCode, "fetchStaticMetaSource");
  const cached = staticMetaCache.get(normalizedStateCode);
  if (cached) {
    return cached;
  }
  const config = stateConfig.getStateConfig(normalizedStateCode);
  console.info("[lead-explorer] static meta fallback", {
    stateCode: normalizedStateCode,
    path: config.staticMetaPath,
  });
  let source: Record<string, unknown>;
  try {
    source = await fetchStaticJson<Record<string, unknown>>(config.staticMetaPath);
  } catch (error) {
    logStaticFallbackFailure(normalizedStateCode, config.staticMetaPath, error);
    source = buildEmptyMetaSource(normalizedStateCode);
  }
  staticMetaCache.set(normalizedStateCode, source);
  return source;
}

async function fetchStaticLeadSource(stateCode: string) {
  const normalizedStateCode = requireStateCode(stateCode, "fetchStaticLeadSource");
  const cached = staticLeadCache.get(normalizedStateCode);
  if (cached) {
    return cached;
  }
  const source = await fetchStaticLeadDetailSource(normalizedStateCode);
  staticLeadCache.set(normalizedStateCode, source);
  return source;
}

async function fetchStaticLeadDetailSource(stateCode: string) {
  const normalizedStateCode = requireStateCode(stateCode, "fetchStaticLeadDetailSource");
  const cached = staticLeadDetailCache.get(normalizedStateCode);
  if (cached) {
    return cached;
  }
  const config = stateConfig.getStateConfig(normalizedStateCode);
  console.info("[lead-explorer] static detail fallback", {
    stateCode: normalizedStateCode,
    path: config.staticLeadDetailPath,
  });
  let source: LeadRecord[];
  try {
    source = await fetchStaticJson<LeadRecord[]>(config.staticLeadDetailPath);
  } catch (error) {
    logStaticFallbackFailure(normalizedStateCode, config.staticLeadDetailPath, error);
    source = [];
  }
  staticLeadDetailCache.set(normalizedStateCode, source);
  return source;
}

export async function fetchStaticLeadDetail(stateCode: string, parcelRowId: string): Promise<LeadRecord | null> {
  const rows = await fetchStaticLeadDetailSource(stateCode);
  const row = rows.find((item) => item.parcel_row_id === parcelRowId) ?? null;
  return row ? normalizeDetailLeadRecord(row) : null;
}

export async function fetchSummary(stateCode: string): Promise<ExplorerMeta> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchSummary");
  try {
    return await fetchJson<ExplorerMeta>(buildStateApiPath(normalizedStateCode, "/summary"), undefined, {
      stateCode: normalizedStateCode,
    });
  } catch {
    const source = await fetchStaticMetaSource(normalizedStateCode);
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

export async function fetchPresets(stateCode: string): Promise<PresetItem[]> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchPresets");
  try {
    const response = await fetchJson<{ items: PresetItem[] }>(buildStateApiPath(normalizedStateCode, "/presets"), undefined, {
      stateCode: normalizedStateCode,
    });
    return response.items;
  } catch {
    const source = await fetchStaticMetaSource(normalizedStateCode);
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
  stateCode: string,
  filters: Filters,
  sortField: SortField,
  sortDirection: "asc" | "desc",
  limit: number,
  offset: number,
): Promise<LeadsResponse> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchLeads");
  try {
    const response = await fetchJson<LeadsResponse>(
      buildStateApiPath(normalizedStateCode, "/leads"),
      buildLeadQuery(filters, sortField, sortDirection, limit, offset),
      { stateCode: normalizedStateCode },
    );
    return {
      ...response,
      items: response.items.map((item) => normalizeParcelIdentifier(item)),
    };
  } catch {
    const rows = await fetchStaticLeadSource(normalizedStateCode);
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
      items: paged.map((item) => normalizeParcelIdentifier(item)),
    };
  }
}

export async function fetchLeadDetail(stateCode: string, parcelRowId: string): Promise<LeadRecord> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchLeadDetail");
  console.info("[lead-explorer] detail request", { stateCode: normalizedStateCode, parcelRowId });
  return fetchJson<LeadRecord>(buildStateApiPath(normalizedStateCode, `/leads/${encodeURIComponent(parcelRowId)}`), undefined, {
    stateCode: normalizedStateCode,
  });
}

function scoreSearchRecord(record: LeadRecord, rawQuery: string) {
  const query = rawQuery.trim().toLowerCase();
  const rowId = (record.parcel_row_id ?? "").trim().toLowerCase();
  const parcelId = (record.parcel_id ?? "").trim().toLowerCase();
  const ownerName = (record.owner_name ?? "").trim().toLowerCase();

  if (!query) {
    return null;
  }
  if (rowId === query) return { rank: 0, matchField: "parcel_row_id_exact" };
  if (parcelId === query) return { rank: 1, matchField: "parcel_id_exact" };
  if (rowId.startsWith(query)) return { rank: 2, matchField: "parcel_row_id_prefix" };
  if (parcelId.startsWith(query)) return { rank: 3, matchField: "parcel_id_prefix" };
  if (rowId.includes(query)) return { rank: 4, matchField: "parcel_row_id_partial" };
  if (parcelId.includes(query)) return { rank: 5, matchField: "parcel_id_partial" };
  if (query.length >= 3 && ownerName === query) return { rank: 6, matchField: "owner_name_exact" };
  if (query.length >= 3 && ownerName.startsWith(query)) return { rank: 7, matchField: "owner_name_prefix" };
  if (query.length >= 3 && ownerName.includes(query)) return { rank: 8, matchField: "owner_name_partial" };
  return null;
}

function toSearchResultRecord(record: LeadRecord, matchField?: string | null): SearchResultRecord {
  return {
    parcel_row_id: record.parcel_row_id,
    parcel_id: record.parcel_id,
    county_name: record.county_name,
    acreage: record.acreage,
    owner_name: record.owner_name,
    centroid:
      record.geometry?.centroid && Array.isArray(record.geometry.centroid.coordinates)
        ? ({ type: "Point", coordinates: record.geometry.centroid.coordinates as [number, number] } satisfies GeometryPoint)
        : null,
    match_field: matchField ?? null,
  };
}

export async function fetchLeadSearch(stateCode: string, q: string, limit = 10): Promise<SearchResponse> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchLeadSearch");
  const searchParams = new URLSearchParams();
  searchParams.set("q", q);
  searchParams.set("limit", String(limit));
  try {
    return await fetchJson<SearchResponse>(buildStateApiPath(normalizedStateCode, "/leads/search"), searchParams, {
      timeoutMs: 8000,
      stateCode: normalizedStateCode,
    });
  } catch {
    const rows = await fetchStaticLeadDetailSource(normalizedStateCode);
    const normalizedRows = rows.map((row) => normalizeDetailLeadRecord(row));
    const scored = normalizedRows
      .map((row) => {
        const match = scoreSearchRecord(row, q);
        return match ? { row, ...match } : null;
      })
      .filter((value): value is { row: LeadRecord; rank: number; matchField: string } => value !== null)
      .sort((left, right) => {
        if (left.rank !== right.rank) return left.rank - right.rank;
        return left.row.parcel_row_id.localeCompare(right.row.parcel_row_id);
      })
      .slice(0, limit)
      .map(({ row, matchField }) => toSearchResultRecord(row, matchField));
    return { query: q.trim(), items: scored };
  }
}

export async function fetchNearbyComps(stateCode: string, parcelRowId: string, limit = 8): Promise<NearbyCompsResponse> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchNearbyComps");
  const searchParams = new URLSearchParams();
  searchParams.set("limit", String(limit));
  console.info("[lead-explorer] nearby comps request", { stateCode: normalizedStateCode, parcelRowId, limit });
  return fetchJson<NearbyCompsResponse>(
    buildStateApiPath(normalizedStateCode, `/leads/${encodeURIComponent(parcelRowId)}/nearby-comps`),
    searchParams,
    { timeoutMs: 10000, stateCode: normalizedStateCode },
  );
}

export async function fetchParcelGeometryById(
  stateCode: string,
  parcelRowId: string,
  zoom = 14,
  options?: { signal?: AbortSignal; timeoutMs?: number },
): Promise<GeometryResponse> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchParcelGeometryById");
  const searchParams = new URLSearchParams();
  searchParams.set("zoom", String(zoom));
  console.info("[lead-explorer] parcel geometry request", { stateCode: normalizedStateCode, parcelRowId, zoom });
  return fetchJson<GeometryResponse>(buildStateApiPath(normalizedStateCode, `/parcels/${encodeURIComponent(parcelRowId)}/geometry`), searchParams, {
    timeoutMs: options?.timeoutMs ?? 120000,
    signal: options?.signal,
    stateCode: normalizedStateCode,
  });
}
