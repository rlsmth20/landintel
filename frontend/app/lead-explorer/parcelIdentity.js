function normalizeString(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

const stateConfig = require("./stateConfig.js");

const DISPLAY_PARCEL_ID_FIELDS = [
  "parcel_id",
  "apn",
  "source_parcel_number",
  "source_alt_parcel_number",
  "source_ppin",
  "legacy_parcel_id",
  "source_parcel_id_normalized",
  "source_parcel_id_raw",
];

function isInternalParcelRowId(value) {
  const normalized = normalizeString(value);
  return normalized ? /^row_/i.test(normalized) : false;
}

function extractDisplayedParcelId(recordOrParcelId, parcelRowId) {
  if (recordOrParcelId && typeof recordOrParcelId === "object") {
    const rowId = normalizeString(recordOrParcelId.parcel_row_id);
    for (const field of DISPLAY_PARCEL_ID_FIELDS) {
      const candidate = normalizeString(recordOrParcelId[field]);
      if (!candidate) continue;
      if (candidate === rowId) continue;
      if (isInternalParcelRowId(candidate)) continue;
      return candidate;
    }
    return null;
  }

  const candidate = normalizeString(recordOrParcelId);
  const normalizedRowId = normalizeString(parcelRowId);
  if (!candidate) return null;
  if (candidate === normalizedRowId) return null;
  if (isInternalParcelRowId(candidate)) return null;
  return candidate;
}

function getDisplayedParcelId(recordOrParcelId, parcelRowId) {
  return extractDisplayedParcelId(recordOrParcelId, parcelRowId) ?? "Parcel ID unavailable";
}

function getLeadSelectionParcelRowId(lead) {
  return normalizeString(lead?.parcel_row_id);
}

function getSearchSelectionParcelRowId(result) {
  return normalizeString(result?.parcel_row_id);
}

function getMapFeatureSelectionParcelRowId(feature) {
  return normalizeString(feature?.properties?.parcel_row_id);
}

function normalizePathStateCode(stateCode) {
  return stateConfig.normalizeStateCode(stateCode ?? stateConfig.DEFAULT_STATE_CODE);
}

function buildLeadDetailPath(parcelRowId, stateCode) {
  return stateConfig.buildStateApiPath(normalizePathStateCode(stateCode), `/leads/${encodeURIComponent(parcelRowId)}`);
}

function buildNearbyCompsPath(parcelRowId, stateCode) {
  return stateConfig.buildStateApiPath(
    normalizePathStateCode(stateCode),
    `/leads/${encodeURIComponent(parcelRowId)}/nearby-comps`,
  );
}

function buildParcelGeometryPath(parcelRowId, stateCode) {
  return stateConfig.buildStateApiPath(
    normalizePathStateCode(stateCode),
    `/parcels/${encodeURIComponent(parcelRowId)}/geometry`,
  );
}

function readDeepLinkedParcelRowId(search) {
  const params = new URLSearchParams(search ?? "");
  const parcelRowId = normalizeString(params.get("parcel_row_id"));
  if (parcelRowId) {
    return parcelRowId;
  }
  const legacyParcelParam = normalizeString(params.get("parcel"));
  return isInternalParcelRowId(legacyParcelParam) ? legacyParcelParam : null;
}

function featureBounds(feature) {
  const coordinates = feature?.geometry?.coordinates;
  if (!coordinates) return null;

  let minLng = Number.POSITIVE_INFINITY;
  let minLat = Number.POSITIVE_INFINITY;
  let maxLng = Number.NEGATIVE_INFINITY;
  let maxLat = Number.NEGATIVE_INFINITY;

  function walk(value) {
    if (!Array.isArray(value)) return;
    if (value.length >= 2 && typeof value[0] === "number" && typeof value[1] === "number") {
      const lng = value[0];
      const lat = value[1];
      if (!Number.isFinite(lng) || !Number.isFinite(lat)) return;
      minLng = Math.min(minLng, lng);
      minLat = Math.min(minLat, lat);
      maxLng = Math.max(maxLng, lng);
      maxLat = Math.max(maxLat, lat);
      return;
    }
    value.forEach(walk);
  }

  walk(coordinates);
  if (![minLng, minLat, maxLng, maxLat].every(Number.isFinite)) return null;
  return [minLng, minLat, maxLng, maxLat];
}

function mergeBounds(boundsList) {
  if (!Array.isArray(boundsList) || boundsList.length === 0) return null;
  return boundsList.reduce(
    (accumulator, bounds) => [
      Math.min(accumulator[0], bounds[0]),
      Math.min(accumulator[1], bounds[1]),
      Math.max(accumulator[2], bounds[2]),
      Math.max(accumulator[3], bounds[3]),
    ],
    boundsList[0],
  );
}

function selectedFeatureBounds(featureCollection, selectedParcelRowId) {
  if (!featureCollection || !selectedParcelRowId) return null;
  const features = Array.isArray(featureCollection.features) ? featureCollection.features : [];
  const matchedFeature =
    features.find((feature) => getMapFeatureSelectionParcelRowId(feature) === selectedParcelRowId) ??
    features.find((feature) => feature?.properties?.selected === true) ??
    (features.length === 1 ? features[0] : null);
  return matchedFeature ? featureBounds(matchedFeature) : null;
}

module.exports = {
  buildLeadDetailPath,
  extractDisplayedParcelId,
  buildNearbyCompsPath,
  buildParcelGeometryPath,
  featureBounds,
  getDisplayedParcelId,
  getLeadSelectionParcelRowId,
  getMapFeatureSelectionParcelRowId,
  getSearchSelectionParcelRowId,
  isInternalParcelRowId,
  mergeBounds,
  readDeepLinkedParcelRowId,
  selectedFeatureBounds,
};
