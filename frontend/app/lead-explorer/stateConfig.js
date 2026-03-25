function normalizeStateCode(value) {
  if (typeof value !== "string") return "ms";
  const normalized = value.trim().toLowerCase();
  return normalized.length > 0 ? normalized : "ms";
}

const DEFAULT_STATE_CODE = normalizeStateCode(process.env.NEXT_PUBLIC_DEFAULT_STATE_CODE ?? "ms");
const DEFAULT_CONUS_BOUNDS = [-125.0, 24.0, -66.0, 49.5];

const STATE_CONFIG_OVERRIDES = {
  ar: {
    stateCode: "ar",
    displayName: "Arkansas",
    apiPrefix: "/api/states/ar",
    staticMetaPath: "/data/ar_lead_explorer_meta.json",
    staticLeadDetailPath: "/data/ar_lead_detail_fallback.json",
    parcelPmtilesUrl: "/tiles/ar_parcels.pmtiles",
    defaultBounds: [-94.62, 33.0, -89.55, 36.55],
  },
  ms: {
    stateCode: "ms",
    displayName: "Mississippi",
    apiPrefix: "/api/states/ms",
    staticMetaPath: "/data/mississippi_lead_explorer_meta.json",
    staticLeadDetailPath: "/data/mississippi_lead_detail_fallback.json",
    parcelPmtilesUrl: "/tiles/mississippi_parcels.pmtiles",
    defaultBounds: [-91.65, 30.15, -88.0, 35.1],
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
    apiPrefix: `/api/states/${normalized}`,
    staticMetaPath: `/data/${normalized}_lead_explorer_meta.json`,
    staticLeadDetailPath: `/data/${normalized}_lead_detail_fallback.json`,
    parcelPmtilesUrl: `/tiles/${normalized}_parcels.pmtiles`,
    defaultBounds: DEFAULT_CONUS_BOUNDS,
  };
}

function getStateConfig(stateCode) {
  const normalized = normalizeStateCode(stateCode);
  return STATE_CONFIG_OVERRIDES[normalized] ?? buildDefaultStateConfig(normalized);
}

function getKnownStateConfigs() {
  return Object.keys(STATE_CONFIG_OVERRIDES)
    .sort()
    .map((stateCode) => getStateConfig(stateCode));
}

function getActiveStateConfig() {
  return getStateConfig(DEFAULT_STATE_CODE);
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
  buildStateApiPath,
  getActiveStateConfig,
  getKnownStateConfigs,
  getStateConfig,
  normalizeStateCode,
  readStateCodeFromSearch,
};
