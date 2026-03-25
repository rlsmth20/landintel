const assert = require("node:assert/strict");

const parcelIdentity = require("../app/lead-explorer/parcelIdentity.js");

function run(name, fn) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

run("list and search selection stay on parcel_row_id while display uses parcel_id", () => {
  assert.equal(
    parcelIdentity.getLeadSelectionParcelRowId({ parcel_row_id: "row_123", parcel_id: "41-115A-38" }),
    "row_123",
  );
  assert.equal(
    parcelIdentity.getSearchSelectionParcelRowId({ parcel_row_id: "row_456", parcel_id: "42-001-1" }),
    "row_456",
  );
  assert.equal(parcelIdentity.getDisplayedParcelId({ parcel_row_id: "row_123", parcel_id: "41-115A-38" }), "41-115A-38");
  assert.equal(parcelIdentity.getDisplayedParcelId({ parcel_row_id: "row_123", parcel_id: "row_123" }), "Parcel ID unavailable");
  assert.equal(
    parcelIdentity.getDisplayedParcelId({
      parcel_row_id: "row_123",
      parcel_id: "row_123",
      source_parcel_id_raw: "41-115A-38",
    }),
    "41-115A-38",
  );
  assert.equal(parcelIdentity.getDisplayedParcelId(null), "Parcel ID unavailable");
});

run("internal API request paths use parcel_row_id", () => {
  assert.equal(parcelIdentity.buildLeadDetailPath("row_123"), "/api/states/ms/leads/row_123");
  assert.equal(parcelIdentity.buildNearbyCompsPath("row_123"), "/api/states/ms/leads/row_123/nearby-comps");
  assert.equal(parcelIdentity.buildParcelGeometryPath("row_123"), "/api/states/ms/parcels/row_123/geometry");
  assert.equal(parcelIdentity.buildLeadDetailPath("row_123", "ar"), "/api/states/ar/leads/row_123");
  assert.equal(parcelIdentity.buildNearbyCompsPath("row_123", "ar"), "/api/states/ar/leads/row_123/nearby-comps");
  assert.equal(parcelIdentity.buildParcelGeometryPath("row_123", "ar"), "/api/states/ar/parcels/row_123/geometry");
});

run("map feature clicks resolve parcel_row_id for selection/highlight", () => {
  assert.equal(
    parcelIdentity.getMapFeatureSelectionParcelRowId({
      properties: { parcel_row_id: "row_789", parcel_id: "41-999-1" },
    }),
    "row_789",
  );
});

run("selected parcel bounds match parcel_row_id and fall back to selected geometry markers", () => {
  const featureCollection = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [[[-91.4, 31.5], [-91.3, 31.5], [-91.3, 31.6], [-91.4, 31.6], [-91.4, 31.5]]],
        },
        properties: { parcel_row_id: "row_123", parcel_id: "41-115A-38", selected: false },
      },
      {
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [[[-91.2, 31.2], [-91.1, 31.2], [-91.1, 31.3], [-91.2, 31.3], [-91.2, 31.2]]],
        },
        properties: { parcel_row_id: "row_999", parcel_id: "41-200-1", selected: true },
      },
    ],
  };

  assert.deepEqual(parcelIdentity.selectedFeatureBounds(featureCollection, "row_123"), [-91.4, 31.5, -91.3, 31.6]);
  assert.deepEqual(parcelIdentity.selectedFeatureBounds(featureCollection, "row_missing"), [-91.2, 31.2, -91.1, 31.3]);
  assert.deepEqual(
    parcelIdentity.selectedFeatureBounds(
      {
        type: "FeatureCollection",
        features: [featureCollection.features[0]],
      },
      "row_missing",
    ),
    [-91.4, 31.5, -91.3, 31.6],
  );
});

run("deep links only seed internal selection from parcel_row_id values", () => {
  assert.equal(parcelIdentity.readDeepLinkedParcelRowId("?parcel_row_id=row_123"), "row_123");
  assert.equal(parcelIdentity.readDeepLinkedParcelRowId("?parcel=row_456"), "row_456");
  assert.equal(parcelIdentity.readDeepLinkedParcelRowId("?parcel=41-115A-38"), null);
});
