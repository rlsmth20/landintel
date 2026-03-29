const assert = require("node:assert/strict");

const stateConfig = require("../app/lead-explorer/stateConfig.js");

function run(name, fn) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

run("active state defaults to Mississippi", () => {
  const active = stateConfig.getActiveStateConfig();
  assert.equal(active.stateCode, "ms");
  assert.equal(active.displayName, "Mississippi");
  assert.equal(active.countyDivisionLabel, "county");
  assert.equal(active.apiPrefix, "/api/states/ms");
});

run("state API paths are built from state code", () => {
  assert.equal(stateConfig.buildStateApiPath("ms", "/summary"), "/api/states/ms/summary");
  assert.equal(stateConfig.buildStateApiPath("ms", "leads"), "/api/states/ms/leads");
});

run("unknown configured states use generic state-aware asset paths instead of falling back to Mississippi", () => {
  const arkansas = stateConfig.getStateConfig("ar");
  assert.equal(arkansas.stateCode, "ar");
  assert.equal(arkansas.displayName, "Arkansas");
  assert.equal(arkansas.apiPrefix, "/api/states/ar");
  assert.equal(arkansas.staticMetaPath, "/data/ar_lead_explorer_meta.json");
  assert.equal(arkansas.staticLeadDetailPath, "/data/ar_lead_detail_fallback.json");
  assert.equal(arkansas.parcelPmtilesLocalUrl, "/tiles/ar_parcels.pmtiles");
  assert.equal(arkansas.parcelPmtilesPublicUrl, "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/ar_parcels.pmtiles");
  assert.equal(arkansas.parcelPmtilesMinZoom, 10);
  assert.equal(arkansas.parcelPmtilesUrl, "/tiles/ar_parcels.pmtiles");
});

run("state code can be read from URL search params", () => {
  assert.equal(stateConfig.readStateCodeFromSearch("?state_code=ar"), "ar");
  assert.equal(stateConfig.readStateCodeFromSearch("?state=ar"), "ar");
  assert.equal(stateConfig.readStateCodeFromSearch("?state_code=ms"), "ms");
  assert.equal(stateConfig.readStateCodeFromSearch("", "ar"), "ar");
});

run("known states are exposed for the dataset selector", () => {
  const states = stateConfig.getKnownStateConfigs().map((item) => item.stateCode);
  assert.deepEqual(states, ["ar", "ct", "ms", "mt", "ny", "ut", "vt", "wi"]);
});

run("point-only states can expose parcel PMTiles without falling back to Mississippi", () => {
  const ny = stateConfig.getStateConfig("ny");
  assert.equal(ny.stateCode, "ny");
  assert.equal(ny.displayName, "New York");
  assert.equal(ny.apiPrefix, "/api/states/ny");
  assert.equal(ny.parcelPmtilesPublicUrl, "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/ny_parcels.pmtiles");
  assert.equal(ny.parcelPmtilesUrl, "/tiles/ny_parcels.pmtiles");
});

run("town-based states expose their local division label", () => {
  const ct = stateConfig.getStateConfig("ct");
  const vt = stateConfig.getStateConfig("vt");
  assert.equal(ct.countyDivisionLabel, "town");
  assert.equal(vt.countyDivisionLabel, "town");
  assert.equal(ct.parcelPmtilesMinZoom, 9);
  assert.equal(vt.parcelPmtilesMinZoom, 9);
  assert.equal(ct.parcelPmtilesUrl, "/tiles/ct_parcels.pmtiles");
  assert.equal(vt.parcelPmtilesUrl, "/tiles/vt_parcels.pmtiles");
});

run("production parcel PMTiles resolution prefers checked-in public URLs", () => {
  const previousNodeEnv = process.env.NODE_ENV;
  const previousForceLocal = process.env.NEXT_PUBLIC_USE_LOCAL_PARCEL_PMTILES;
  process.env.NODE_ENV = "production";
  delete process.env.NEXT_PUBLIC_USE_LOCAL_PARCEL_PMTILES;
  try {
    const mississippi = stateConfig.getStateConfig("ms");
    assert.equal(mississippi.parcelPmtilesUrl, "https://pub-f5f866f9a229419696c3066b960daae4.r2.dev/tiles/mississippi_parcels.pmtiles");
  } finally {
    if (previousNodeEnv === undefined) {
      delete process.env.NODE_ENV;
    } else {
      process.env.NODE_ENV = previousNodeEnv;
    }
    if (previousForceLocal === undefined) {
      delete process.env.NEXT_PUBLIC_USE_LOCAL_PARCEL_PMTILES;
    } else {
      process.env.NEXT_PUBLIC_USE_LOCAL_PARCEL_PMTILES = previousForceLocal;
    }
  }
});

run("states without explicit overrides still default to county division labels", () => {
  const unknown = stateConfig.getStateConfig("zz");
  assert.equal(unknown.stateCode, "zz");
  assert.equal(unknown.countyDivisionLabel, "county");
  assert.equal(unknown.parcelPmtilesMinZoom, 10);
  assert.equal(unknown.selectionEnabled, true);
  assert.equal(unknown.parcelPmtilesUrl, "/tiles/zz_parcels.pmtiles");
});

run("blocked loaded states remain visible but not selectable", () => {
  const utah = stateConfig.getStateConfig("ut");
  assert.equal(utah.stateCode, "ut");
  assert.equal(utah.displayName, "Utah");
  assert.equal(utah.selectionEnabled, false);
  assert.equal(utah.parcelPmtilesLocalUrl, null);
  assert.equal(utah.parcelPmtilesPublicUrl, null);
  assert.equal(stateConfig.isStateSelectable("ut"), false);
  assert.equal(stateConfig.resolveSelectableStateCode("ut", "ms"), "ms");
});
