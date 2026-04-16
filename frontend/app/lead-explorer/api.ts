import type { ExplorerMeta, GeometryPoint, GeometryResponse, LeadRecord, LeadsResponse, NearbyCompsResponse, PresetItem, SearchResponse, SearchResultRecord, SortField, Filters } from "./types";
import requestLifecycle from "./requestLifecycle";
import stateConfig from "./stateConfig";
import { INITIAL_FILTERS } from "./utils";

const DEFAULT_PRODUCTION_API_BASE_URL = "https://landintel-production.up.railway.app";
const CONFIGURED_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ?? "";
const API_BASE_URL =
  CONFIGURED_API_BASE_URL ||
  (process.env.NODE_ENV === "production" ? DEFAULT_PRODUCTION_API_BASE_URL : "");
const { isAbortLikeError } = requestLifecycle;
const SUMMARY_REQUEST_TIMEOUT_MS = 4000;
const PRESETS_REQUEST_TIMEOUT_MS = 4000;

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

function normalizeResponseNumber(value: unknown, fallback: number) {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function buildPayloadShapeError(context: string, detail: string) {
  return new Error(`[lead-explorer] invalid ${context} payload: ${detail}`);
}

function assertObjectPayload(value: unknown, context: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw buildPayloadShapeError(context, "expected an object payload");
  }
}

function normalizeSummarySections(payload: unknown): ExplorerMeta["sections"] {
  const sections: ExplorerMeta["sections"] = {
    statewide: [],
    top_counties: [],
    recommended_view_bucket: [],
  };
  if (!Array.isArray(payload)) {
    return sections;
  }
  payload.forEach((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return;
    const section = typeof entry.section === "string" && entry.section.trim().length > 0 ? entry.section.trim() : null;
    const metric = typeof entry.metric === "string" && entry.metric.trim().length > 0 ? entry.metric.trim() : null;
    if (!section || !metric) return;
    const nextEntry: Record<string, string> = {
      metric,
      key: typeof entry.key === "string" ? entry.key : "",
      value: entry.value == null ? "" : String(entry.value),
    };
    const currentSection = sections[section] ?? [];
    currentSection.push(nextEntry);
    sections[section] = currentSection;
  });
  return sections;
}

function normalizeExplorerMetaResponse(payload: ExplorerMeta, context: string): ExplorerMeta {
  assertObjectPayload(payload, context);
  if (payload.sections !== undefined && (typeof payload.sections !== "object" || payload.sections === null || Array.isArray(payload.sections))) {
    throw buildPayloadShapeError(context, "expected sections to be an object when present");
  }
  return payload;
}

function normalizeApiLeadsResponse(payload: LeadsResponse, context: string): LeadsResponse {
  assertObjectPayload(payload, context);
  if (!Array.isArray(payload.items)) {
    throw buildPayloadShapeError(context, "expected items to be an array");
  }
  return {
    total_count: normalizeResponseNumber(payload.total_count, payload.items.length),
    limit: normalizeResponseNumber(payload.limit, payload.items.length),
    offset: normalizeResponseNumber(payload.offset, 0),
    items: payload.items.map((item) => normalizeParcelIdentifier(item)),
    fallback_notice: payload.fallback_notice ?? null,
  };
}

function buildSummaryFallbackNotice() {
  return "Summary is using packaged metadata because the live runtime summary is unavailable.";
}

function buildDefaultLeadsFallbackNotice() {
  return "Parcel results are using the packaged statewide fallback because the live API request failed.";
}

function buildDetailLeadsFallbackNotice() {
  return "Parcel results are using the packaged detail fallback because the live API request failed.";
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
    response = await fetch(url, {
      cache: "no-store",
      mode: API_BASE_URL ? "cors" : "same-origin",
      signal: controller.signal,
    });
  } catch (error) {
    if (!controller.signal.aborted) {
      console.error("[lead-explorer] request failed before response", {
        stateCode: options?.stateCode ?? null,
        url,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    throw error;
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
const staticLeadCache = new Map<string, LeadsResponse>();
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

function logStaticFallbackFailure(stateCode: string, path: string, error: unknown) {
  console.error("[lead-explorer] static fallback load failed", {
    stateCode,
    path,
    error: error instanceof Error ? error.message : String(error),
  });
}

function buildStaticFallbackUnavailableError(kind: string, stateCode: string, path: string, error: unknown) {
  const detail = error instanceof Error ? error.message : String(error);
  return new Error(`Static ${kind} fallback unavailable for ${stateCode} at ${path}: ${detail}`);
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
    throw buildStaticFallbackUnavailableError("meta", normalizedStateCode, config.staticMetaPath, error);
  }
  staticMetaCache.set(normalizedStateCode, source);
  return source;
}

function summaryPayloadNeedsStaticFallback(summary: ExplorerMeta) {
  const source = typeof summary.source === "string" ? summary.source.toLowerCase() : "";
  const rowCount = normalizeResponseNumber(summary.row_count, 0);
  const sections = summary.sections;
  const sectionCount =
    sections && typeof sections === "object"
      ? Object.values(sections).reduce((count, value) => count + (Array.isArray(value) ? value.length : 0), 0)
      : 0;
  return rowCount === 0 && sectionCount === 0 && source.includes("runtime artifacts unavailable");
}

function normalizePresetsResponse(payload: { items?: PresetItem[] }, context: string) {
  assertObjectPayload(payload, context);
  if (!Array.isArray(payload.items)) {
    throw buildPayloadShapeError(context, "expected items to be an array");
  }
  return payload.items;
}

function buildStaticSummaryResponse(source: Record<string, unknown>): ExplorerMeta {
  return {
    row_count: Number(source.rowCount ?? 0),
    source: typeof source.source === "string" ? source.source : "static explorer fallback",
    fallback_notice: buildSummaryFallbackNotice(),
    geometry_mode: typeof source.geometryMode === "string" ? source.geometryMode : undefined,
    geometry_bounds: Array.isArray(source.geometryBounds) ? (source.geometryBounds as number[]) : undefined,
    geometry_view_box: Array.isArray(source.geometryViewBox) ? (source.geometryViewBox as number[]) : undefined,
    sections: normalizeSummarySections(source.summary),
  };
}

function buildStaticPresetsResponse(source: Record<string, unknown>): PresetItem[] {
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

function normalizeStaticLeadResponse(payload: LeadRecord[] | LeadsResponse): LeadsResponse {
  if (Array.isArray(payload)) {
    const items = payload.map((item) => normalizeParcelIdentifier(item));
    return {
      total_count: items.length,
      limit: items.length,
      offset: 0,
      items,
    };
  }
  const items = Array.isArray(payload.items) ? payload.items.map((item) => normalizeParcelIdentifier(item)) : [];
  return {
    total_count: normalizeResponseNumber(payload.total_count, items.length),
    limit: normalizeResponseNumber(payload.limit, items.length),
    offset: normalizeResponseNumber(payload.offset, 0),
    items,
  };
}

async function fetchStaticLeadSource(stateCode: string) {
  const normalizedStateCode = requireStateCode(stateCode, "fetchStaticLeadSource");
  const cached = staticLeadCache.get(normalizedStateCode);
  if (cached) {
    return cached;
  }
  const config = stateConfig.getStateConfig(normalizedStateCode);
  console.info("[lead-explorer] static leads fallback", {
    stateCode: normalizedStateCode,
    path: config.staticLeadPath,
  });
  let source: LeadsResponse;
  try {
    source = normalizeStaticLeadResponse(await fetchStaticJson<LeadRecord[] | LeadsResponse>(config.staticLeadPath));
  } catch (error) {
    logStaticFallbackFailure(normalizedStateCode, config.staticLeadPath, error);
    throw buildStaticFallbackUnavailableError("leads", normalizedStateCode, config.staticLeadPath, error);
  }
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
    throw buildStaticFallbackUnavailableError("detail", normalizedStateCode, config.staticLeadDetailPath, error);
  }
  staticLeadDetailCache.set(normalizedStateCode, source);
  return source;
}

export async function fetchStaticLeadDetail(stateCode: string, parcelRowId: string): Promise<LeadRecord | null> {
  const rows = await fetchStaticLeadDetailSource(stateCode);
  const row = rows.find((item) => item.parcel_row_id === parcelRowId) ?? null;
  return row ? normalizeDetailLeadRecord(row) : null;
}

export async function fetchSummary(stateCode: string, options?: { signal?: AbortSignal }): Promise<ExplorerMeta> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchSummary");
  let fallbackReason: unknown = null;
  try {
    const response = normalizeExplorerMetaResponse(
      await fetchJson<ExplorerMeta>(buildStateApiPath(normalizedStateCode, "/summary"), undefined, {
        timeoutMs: SUMMARY_REQUEST_TIMEOUT_MS,
        signal: options?.signal,
        stateCode: normalizedStateCode,
      }),
      `summary response for ${normalizedStateCode}`,
    );
    console.info("[lead-explorer] summary loaded", {
      stateCode: normalizedStateCode,
      source: response.source ?? "api",
      rowCount: response.row_count ?? null,
    });
    if (!summaryPayloadNeedsStaticFallback(response)) {
      return response;
    }
    fallbackReason = new Error(
      `runtime summary unavailable: source=${response.source ?? "unknown"} row_count=${response.row_count ?? "unknown"}`,
    );
  } catch (error) {
    if (isAbortLikeError(error)) {
      throw error;
    }
    fallbackReason = error;
  }
  console.warn("[lead-explorer] summary unavailable, using packaged metadata", {
    stateCode: normalizedStateCode,
    error: fallbackReason instanceof Error ? fallbackReason.message : String(fallbackReason),
  });
  const source = await fetchStaticMetaSource(normalizedStateCode);
  return buildStaticSummaryResponse(source);
}

export async function fetchPresets(stateCode: string, options?: { signal?: AbortSignal }): Promise<PresetItem[]> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchPresets");
  let fallbackReason: unknown = null;
  try {
    const items = normalizePresetsResponse(
      await fetchJson<{ items: PresetItem[] }>(buildStateApiPath(normalizedStateCode, "/presets"), undefined, {
        timeoutMs: PRESETS_REQUEST_TIMEOUT_MS,
        signal: options?.signal,
        stateCode: normalizedStateCode,
      }),
      `presets response for ${normalizedStateCode}`,
    );
    if (items.length > 0) {
      return items;
    }
    fallbackReason = new Error("runtime presets unavailable: empty items payload");
  } catch (error) {
    if (isAbortLikeError(error)) {
      throw error;
    }
    fallbackReason = error;
  }
  console.warn("[lead-explorer] presets unavailable, using packaged metadata", {
    stateCode: normalizedStateCode,
    error: fallbackReason instanceof Error ? fallbackReason.message : String(fallbackReason),
  });
  const source = await fetchStaticMetaSource(normalizedStateCode);
  return buildStaticPresetsResponse(source);
}

function appendList(searchParams: URLSearchParams, key: string, values: string[]) {
  values.forEach((value) => searchParams.append(key, value));
}

function sameStringSelections(left: string[], right: string[]) {
  if (left.length !== right.length) return false;
  const leftSet = new Set(left.map((value) => value.toLowerCase()));
  return right.every((value) => leftSet.has(value.toLowerCase()));
}

function isDefaultLeadRequest(
  filters: Filters,
  sortField: SortField,
  sortDirection: "asc" | "desc",
  offset: number,
) {
  return (
    filters.countyName === INITIAL_FILTERS.countyName &&
    filters.leadScoreTier.length === INITIAL_FILTERS.leadScoreTier.length &&
    filters.minLeadScore === INITIAL_FILTERS.minLeadScore &&
    filters.acreageMin === INITIAL_FILTERS.acreageMin &&
    filters.acreageMax === INITIAL_FILTERS.acreageMax &&
    filters.parcelVacantOnly === INITIAL_FILTERS.parcelVacantOnly &&
    filters.countyHostedOnly === INITIAL_FILTERS.countyHostedOnly &&
    filters.highConfidenceOnly === INITIAL_FILTERS.highConfidenceOnly &&
    filters.wetlandMode === INITIAL_FILTERS.wetlandMode &&
    sameStringSelections(filters.amountTrustTiers, INITIAL_FILTERS.amountTrustTiers) &&
    filters.corporateOnly === INITIAL_FILTERS.corporateOnly &&
    filters.absenteeOnly === INITIAL_FILTERS.absenteeOnly &&
    filters.outOfStateOnly === INITIAL_FILTERS.outOfStateOnly &&
    filters.growthPressureBuckets.length === INITIAL_FILTERS.growthPressureBuckets.length &&
    filters.recommendedViewBucket === INITIAL_FILTERS.recommendedViewBucket &&
    filters.roadDistanceMax === INITIAL_FILTERS.roadDistanceMax &&
    filters.roadAccessTiers.length === INITIAL_FILTERS.roadAccessTiers.length &&
    sortField === "lead_score_total" &&
    sortDirection === "desc" &&
    offset === 0
  );
}

function normalizeStringValue(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const normalized = value.trim().toLowerCase();
  return normalized.length > 0 ? normalized : null;
}

function normalizeBooleanValue(value: unknown): boolean | null {
  if (typeof value === "boolean") return value;
  return null;
}

function normalizeNumericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function matchesStringSelection(value: unknown, allowed: string[]) {
  if (allowed.length === 0) return true;
  const normalizedValue = normalizeStringValue(value);
  if (!normalizedValue) return false;
  return allowed.some((candidate) => normalizeStringValue(candidate) === normalizedValue);
}

function recordMatchesLeadFilters(record: LeadRecord, filters: Filters) {
  if (filters.countyName !== "all" && normalizeStringValue(record.county_name) !== normalizeStringValue(filters.countyName)) {
    return false;
  }
  if (!matchesStringSelection(record.lead_score_tier, filters.leadScoreTier)) return false;
  const leadScore = normalizeNumericValue(record.lead_score_total);
  if (leadScore === null || leadScore < filters.minLeadScore) return false;

  const acreage = normalizeNumericValue(record.acreage);
  if (filters.acreageMin !== "" && (acreage === null || acreage < Number(filters.acreageMin))) return false;
  if (filters.acreageMax !== "" && (acreage === null || acreage > Number(filters.acreageMax))) return false;

  const parcelVacant = normalizeBooleanValue(record.parcel_vacant_flag) ?? normalizeStringValue(record.parcel_improvement_status) === "likely_vacant";
  if (filters.parcelVacantOnly && !parcelVacant) return false;
  if (filters.countyHostedOnly && normalizeBooleanValue(record.county_hosted_flag) !== true) return false;
  if (filters.highConfidenceOnly && normalizeBooleanValue(record.high_confidence_link_flag) !== true) return false;

  const wetlandFlag = normalizeBooleanValue(record.wetland_flag);
  if (filters.wetlandMode === "exclude" && wetlandFlag === true) return false;
  if (filters.wetlandMode === "only" && wetlandFlag !== true) return false;

  if (!matchesStringSelection(record.amount_trust_tier, filters.amountTrustTiers)) return false;
  if (filters.corporateOnly && normalizeBooleanValue(record.corporate_owner_flag) !== true) return false;
  if (filters.absenteeOnly && normalizeBooleanValue(record.absentee_owner_flag) !== true) return false;
  if (filters.outOfStateOnly && normalizeBooleanValue(record.out_of_state_owner_flag) !== true) return false;
  if (!matchesStringSelection(record.growth_pressure_bucket, filters.growthPressureBuckets)) return false;
  if (filters.recommendedViewBucket !== "all" && normalizeStringValue(record.recommended_view_bucket) !== normalizeStringValue(filters.recommendedViewBucket)) {
    return false;
  }
  if (!matchesStringSelection(record.road_access_tier, filters.roadAccessTiers)) return false;

  const roadDistance = normalizeNumericValue(record.road_distance_ft);
  if (filters.roadDistanceMax !== "" && (roadDistance === null || roadDistance > Number(filters.roadDistanceMax))) {
    return false;
  }
  return true;
}

function compareNullableNumber(left: number | null, right: number | null, ascending: boolean) {
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  return ascending ? left - right : right - left;
}

function sortStaticLeadRows(rows: LeadRecord[], sortField: SortField, sortDirection: "asc" | "desc") {
  const ascending = sortDirection === "asc";
  return [...rows].sort((left, right) => {
    const comparison = compareNullableNumber(
      normalizeNumericValue(left[sortField]),
      normalizeNumericValue(right[sortField]),
      ascending,
    );
    if (comparison !== 0) return comparison;
    return left.parcel_row_id.localeCompare(right.parcel_row_id);
  });
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
  if (filters.minLeadScore > INITIAL_FILTERS.minLeadScore) {
    searchParams.set("min_lead_score_total", String(filters.minLeadScore));
  }
  if (filters.acreageMin !== "") searchParams.set("acreage_min", filters.acreageMin);
  if (filters.acreageMax !== "") searchParams.set("acreage_max", filters.acreageMax);
  if (filters.parcelVacantOnly) searchParams.set("parcel_vacant_flag", "true");
  if (filters.countyHostedOnly) searchParams.set("county_hosted_flag", "true");
  if (filters.highConfidenceOnly) searchParams.set("high_confidence_link_flag", "true");
  if (filters.wetlandMode === "exclude") searchParams.set("wetland_flag", "false");
  if (filters.wetlandMode === "only") searchParams.set("wetland_flag", "true");
  if (!sameStringSelections(filters.amountTrustTiers, INITIAL_FILTERS.amountTrustTiers)) {
    appendList(searchParams, "amount_trust_tier", filters.amountTrustTiers);
  }
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
  options?: { signal?: AbortSignal },
): Promise<LeadsResponse> {
  const normalizedStateCode = requireStateCode(stateCode, "fetchLeads");
  try {
    const response = normalizeApiLeadsResponse(
      await fetchJson<LeadsResponse>(
        buildStateApiPath(normalizedStateCode, "/leads"),
        buildLeadQuery(filters, sortField, sortDirection, limit, offset),
        { timeoutMs: 10000, signal: options?.signal, stateCode: normalizedStateCode },
      ),
      `leads response for ${normalizedStateCode}`,
    );
    return response;
  } catch (error) {
    if (isAbortLikeError(error)) {
      throw error;
    }
    if (isDefaultLeadRequest(filters, sortField, sortDirection, offset)) {
      try {
        const staticLeadSource = await fetchStaticLeadSource(normalizedStateCode);
        const pagedDefaultItems = staticLeadSource.items.slice(offset, offset + limit).map((item) => normalizeParcelIdentifier(item));
        console.warn("[lead-explorer] leads request failed, using packaged default leads fallback", {
          stateCode: normalizedStateCode,
          totalCount: staticLeadSource.total_count,
          itemCount: pagedDefaultItems.length,
          error: error instanceof Error ? error.message : String(error),
        });
        return {
          total_count: staticLeadSource.total_count,
          limit,
          offset,
          items: pagedDefaultItems,
          fallback_notice: buildDefaultLeadsFallbackNotice(),
        };
      } catch (staticLeadError) {
        console.warn("[lead-explorer] packaged default leads fallback unavailable, trying detail fallback", {
          stateCode: normalizedStateCode,
          error: staticLeadError instanceof Error ? staticLeadError.message : String(staticLeadError),
        });
      }
    }

    try {
      const detailRows = await fetchStaticLeadDetailSource(normalizedStateCode);
      const filteredRows = detailRows.filter((item) => recordMatchesLeadFilters(item, filters));
      const paged = sortStaticLeadRows(filteredRows, sortField, sortDirection).slice(offset, offset + limit);
      console.warn("[lead-explorer] leads request failed, using packaged detail fallback", {
        stateCode: normalizedStateCode,
        totalCount: filteredRows.length,
        itemCount: paged.length,
        error: error instanceof Error ? error.message : String(error),
      });
      return {
        total_count: filteredRows.length,
        limit,
        offset,
        items: paged.map((item) => normalizeParcelIdentifier(item)),
        fallback_notice: buildDetailLeadsFallbackNotice(),
      };
    } catch (detailError) {
      throw new Error(
        `Parcel results request failed for ${normalizedStateCode}: ${
          error instanceof Error ? error.message : String(error)
        }. Packaged detail fallback also failed: ${detailError instanceof Error ? detailError.message : String(detailError)}`,
      );
    }
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
