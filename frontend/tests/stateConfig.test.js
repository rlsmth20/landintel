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
  assert.deepEqual(states, ["ar", "ms"]);
});
