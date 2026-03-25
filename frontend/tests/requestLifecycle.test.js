const assert = require("node:assert/strict");

const requestLifecycle = require("../app/lead-explorer/requestLifecycle.js");

function run(name, fn) {
  try {
    fn();
    console.log(`PASS ${name}`);
  } catch (error) {
    console.error(`FAIL ${name}`);
    throw error;
  }
}

run("abort-like errors are suppressed", () => {
  assert.equal(requestLifecycle.isAbortLikeError({ name: "AbortError", message: "The operation was aborted." }), true);
  assert.equal(requestLifecycle.isAbortLikeError(new Error("signal is aborted without reason")), true);
});

run("non-abort errors still surface", () => {
  assert.equal(requestLifecycle.isAbortLikeError(new Error("Request failed: 500 Internal Server Error")), false);
});
