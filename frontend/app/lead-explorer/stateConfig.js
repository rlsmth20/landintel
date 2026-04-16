function normalizeStateCode(value) {
  if (typeof value !== "string") return "ms";
  const normalized = value.trim().toLowerCase();
  return normalized.length > 0 ? normalized : "ms";
}

const DEFAULT_STATE_CODE = normalizeStateCode(process.env.NEXT_PUBLIC_DEFAULT_STATE_CODE ?? "ms");
const DEFAULT_CONUS_BOUNDS = [-125.0, 24.0, -66.0, 49.5];
const DEFAULT_PARCEL_PMTILES_PUBLIC_BASE_URL = "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev";
const DEFAULT_PARCEL_PMTILES_MIN_ZOOM = 10;

function normalizeOptionalUrl(value) {
  if (typeof value !== "string") return null;
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
}

function shouldUseLocalParcelPmtiles() {
  const flag = (process.env.NEXT_PUBLIC_USE_LOCAL_PARCEL_PMTILES ?? "").trim().toLowerCase();
  if (flag === "1" || flag === "true" || flag === "yes") return true;
  return process.env.NODE_ENV !== "production";
}

function statePmtilesEnvOverride(stateCode) {
  const normalized = normalizeStateCode(stateCode);
  const overrides = {
    ar: normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_AR),
    ct: normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_CT),
    ms: normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_MS),
    mt: normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_MT),
    ny: normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_NY),
    vt: normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_VT),
    wi: normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_WI),
  };
  return overrides[normalized] ?? null;
}

function buildDefaultParcelPmtilesPublicUrl(stateCode) {
  const normalized = normalizeStateCode(stateCode);
  return `${DEFAULT_PARCEL_PMTILES_PUBLIC_BASE_URL}/tiles/${normalized === "ms" ? "mississippi" : normalized}_parcels.pmtiles`;
}

function resolveParcelPmtilesUrl(config) {
  const genericOverride = normalizeOptionalUrl(process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL);
  const stateOverride = statePmtilesEnvOverride(config.stateCode);
  if (stateOverride) return stateOverride;
  if (genericOverride) return genericOverride;
  if (shouldUseLocalParcelPmtiles()) return config.parcelPmtilesLocalUrl;
  return config.parcelPmtilesPublicUrl;
}

function finalizeStateConfig(config) {
  return {
    ...config,
    parcelPmtilesUrl: resolveParcelPmtilesUrl(config),
  };
}

const STATE_CONFIG_OVERRIDES = {
  ar: {
    stateCode: "ar",
    displayName: "Arkansas",
    countyDivisionLabel: "county",
    apiPrefix: "/api/states/ar",
    staticMetaPath: "/data/ar_lead_explorer_meta.json",
    staticLeadPath: "/data/ar_lead_explorer.json",
    staticLeadDetailPath: "/data/ar_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: "/tiles/ar_parcels.pmtiles",
    parcelPmtilesPublicUrl: "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/ar_parcels.pmtiles",
    parcelPmtilesMinZoom: 6,
    defaultBounds: [-94.62, 33.0, -89.55, 36.55],
  },
  ct: {
    stateCode: "ct",
    displayName: "Connecticut",
    countyDivisionLabel: "town",
    apiPrefix: "/api/states/ct",
    staticMetaPath: "/data/ct_lead_explorer_meta.json",
    staticLeadPath: "/data/ct_lead_explorer.json",
    staticLeadDetailPath: "/data/ct_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: "/tiles/ct_parcels.pmtiles",
    parcelPmtilesPublicUrl: "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/ct_parcels.pmtiles",
    parcelPmtilesMinZoom: 7,
    defaultBounds: [-73.75, 40.95, -71.78, 42.05],
  },
  mt: {
    stateCode: "mt",
    displayName: "Montana",
    countyDivisionLabel: "county",
    apiPrefix: "/api/states/mt",
    staticMetaPath: "/data/mt_lead_explorer_meta.json",
    staticLeadPath: "/data/mt_lead_explorer.json",
    staticLeadDetailPath: "/data/mt_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: "/tiles/mt_parcels.pmtiles",
    parcelPmtilesPublicUrl: "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/mt_parcels.pmtiles",
    parcelPmtilesMinZoom: 6,
    defaultBounds: [-116.15, 44.35, -104.0, 49.05],
  },
  ms: {
    stateCode: "ms",
    displayName: "Mississippi",
    countyDivisionLabel: "county",
    apiPrefix: "/api/states/ms",
    staticMetaPath: "/data/mississippi_lead_explorer_meta.json",
    staticLeadPath: "/data/mississippi_lead_explorer.json",
    staticLeadDetailPath: "/data/mississippi_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: "/tiles/mississippi_parcels.pmtiles",
    parcelPmtilesPublicUrl: "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/mississippi_parcels.pmtiles",
    parcelPmtilesMinZoom: 6,
    defaultBounds: [-91.65, 30.15, -88.0, 35.1],
  },
  ny: {
    stateCode: "ny",
    displayName: "New York",
    countyDivisionLabel: "county",
    apiPrefix: "/api/states/ny",
    staticMetaPath: "/data/ny_lead_explorer_meta.json",
    staticLeadPath: "/data/ny_lead_explorer.json",
    staticLeadDetailPath: "/data/ny_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: "/tiles/ny_parcels.pmtiles",
    parcelPmtilesPublicUrl: "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/ny_parcels.pmtiles",
    parcelPmtilesMinZoom: 7,
    defaultBounds: [-79.85, 40.45, -71.75, 45.1],
  },
  ut: {
    stateCode: "ut",
    displayName: "Utah",
    countyDivisionLabel: "county",
    apiPrefix: "/api/states/ut",
    staticMetaPath: "/data/ut_lead_explorer_meta.json",
    staticLeadPath: "/data/ut_lead_explorer.json",
    staticLeadDetailPath: "/data/ut_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: null,
    parcelPmtilesPublicUrl: null,
    selectionEnabled: false,
    defaultBounds: [-114.08, 36.95, -109.0, 42.05],
  },
  vt: {
    stateCode: "vt",
    displayName: "Vermont",
    countyDivisionLabel: "town",
    apiPrefix: "/api/states/vt",
    staticMetaPath: "/data/vt_lead_explorer_meta.json",
    staticLeadPath: "/data/vt_lead_explorer.json",
    staticLeadDetailPath: "/data/vt_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: "/tiles/vt_parcels.pmtiles",
    parcelPmtilesPublicUrl: "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/vt_parcels.pmtiles",
    parcelPmtilesMinZoom: 7,
    defaultBounds: [-73.45, 42.72, -71.47, 45.05],
  },
  wi: {
    stateCode: "wi",
    displayName: "Wisconsin",
    countyDivisionLabel: "county",
    apiPrefix: "/api/states/wi",
    staticMetaPath: "/data/wi_lead_explorer_meta.json",
    staticLeadPath: "/data/wi_lead_explorer.json",
    staticLeadDetailPath: "/data/wi_lead_detail_fallback.json",
    parcelPmtilesLocalUrl: "/tiles/wi_parcels.pmtiles",
    parcelPmtilesPublicUrl: "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/wi_parcels.pmtiles",
    parcelPmtilesMinZoom: 6,
    defaultBounds: [-92.95, 42.4, -86.23, 47.35],
  },
};

function defaultDisplayName(stateCode) {
  const normalized = normalizeStateCode(stateCode);
  return normalized.toUpperCase();
}

function buildDefaultStateConfig(stateCode) {
  const normalized = normalizeStateCode(stateCode);
  return {
    stateCode: normalized,
    displayName: defaultDisplayName(normalized),
    countyDivisionLabel: "county",
    apiPrefix: `/api/states/${normalized}`,
    staticMetaPath: `/data/${normalized}_lead_explorer_meta.json`,
    staticLeadPath: `/data/${normalized}_lead_explorer.json`,
    staticLeadDetailPath: `/data/${normalized}_lead_detail_fallback.json`,
    parcelPmtilesLocalUrl: `/tiles/${normalized}_parcels.pmtiles`,
    parcelPmtilesPublicUrl: buildDefaultParcelPmtilesPublicUrl(normalized),
    parcelPmtilesMinZoom: DEFAULT_PARCEL_PMTILES_MIN_ZOOM,
    selectionEnabled: true,
    defaultBounds: DEFAULT_CONUS_BOUNDS,
  };
}

function getStateConfig(stateCode) {
  const normalized = normalizeStateCode(stateCode);
  return finalizeStateConfig({
    ...buildDefaultStateConfig(normalized),
    ...(STATE_CONFIG_OVERRIDES[normalized] ?? {}),
  });
}

function getKnownStateConfigs() {
  return Object.keys(STATE_CONFIG_OVERRIDES)
    .sort()
    .map((stateCode) => getStateConfig(stateCode));
}

function getActiveStateConfig() {
  return getStateConfig(DEFAULT_STATE_CODE);
}

function isStateSelectable(stateCode) {
  return getStateConfig(stateCode).selectionEnabled !== false;
}

function resolveSelectableStateCode(stateCode, fallbackStateCode = DEFAULT_STATE_CODE) {
  const normalized = normalizeStateCode(stateCode);
  if (isStateSelectable(normalized)) return normalized;
  const fallback = normalizeStateCode(fallbackStateCode);
  return isStateSelectable(fallback) ? fallback : DEFAULT_STATE_CODE;
}

function readStateCodeFromSearch(search, fallbackStateCode = DEFAULT_STATE_CODE) {
  const params = new URLSearchParams(search ?? "");
  const stateCode =
    params.get("state_code") ??
    params.get("state") ??
    fallbackStateCode;
  return normalizeStateCode(stateCode);
}

function buildStateApiPath(stateCode, suffix) {
  const config = getStateConfig(stateCode);
  const normalizedSuffix = typeof suffix === "string" && suffix.startsWith("/") ? suffix : `/${suffix ?? ""}`;
  return `${config.apiPrefix}${normalizedSuffix}`;
}

module.exports = {
  DEFAULT_STATE_CODE,
  DEFAULT_PARCEL_PMTILES_MIN_ZOOM,
  DEFAULT_PARCEL_PMTILES_PUBLIC_BASE_URL,
  buildStateApiPath,
  getActiveStateConfig,
  getKnownStateConfigs,
  getStateConfig,
  isStateSelectable,
  normalizeStateCode,
  readStateCodeFromSearch,
  resolveSelectableStateCode,
  resolveParcelPmtilesUrl,
};
