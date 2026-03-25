import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const rootDir = resolve(process.cwd());
const nextDir = resolve(rootDir, ".next");
const buildManifestPath = resolve(nextDir, "build-manifest.json");
const routesManifestPath = resolve(nextDir, "routes-manifest.json");
const reportPath = resolve(nextDir, "cache-strategy-report.json");
const port = Number(process.env.CACHE_CHECK_PORT ?? 3101);
const previousReportArgIndex = process.argv.indexOf("--previous-report");
const previousReportPath =
  previousReportArgIndex >= 0 ? resolve(rootDir, process.argv[previousReportArgIndex + 1]) : null;

function fail(message) {
  throw new Error(message);
}

function readJson(path) {
  if (!existsSync(path)) {
    fail(`Missing file: ${path}. Run npm run build first.`);
  }
  return JSON.parse(readFileSync(path, "utf8"));
}

function collectStrings(value, output = new Set()) {
  if (typeof value === "string") {
    output.add(value);
    return output;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectStrings(item, output));
    return output;
  }
  if (value && typeof value === "object") {
    Object.values(value).forEach((item) => collectStrings(item, output));
  }
  return output;
}

function normalizeAssetPath(assetPath) {
  if (assetPath.startsWith("/")) {
    return assetPath;
  }
  if (assetPath.startsWith("_next/")) {
    return `/${assetPath}`;
  }
  if (assetPath.startsWith("static/")) {
    return `/_next/${assetPath}`;
  }
  return `/${assetPath}`;
}

function hasHashedFilename(assetPath) {
  return /[A-Za-z0-9]{8,}\.(?:js|css)$/.test(assetPath);
}

async function waitForServer(url, timeoutMs = 30000) {
  const start = Date.now();
  for (;;) {
    try {
      const response = await fetch(url, { redirect: "manual" });
      if (response.ok) {
        return;
      }
    } catch {
      // Keep polling until the server is ready.
    }
    if (Date.now() - start > timeoutMs) {
      fail(`Timed out waiting for ${url}`);
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 500));
  }
}

const buildManifest = readJson(buildManifestPath);
const routesManifest = readJson(routesManifestPath);
const assetPaths = Array.from(collectStrings(buildManifest))
  .filter((assetPath) => /\.(?:js|css)$/.test(assetPath))
  .map(normalizeAssetPath)
  .filter((assetPath) => assetPath.startsWith("/_next/static/"));

if (assetPaths.length === 0) {
  fail("No built static assets found in .next/build-manifest.json.");
}
if (!assetPaths.every(hasHashedFilename)) {
  fail("Expected all built JS/CSS assets to use hashed filenames.");
}

const staticHeaderRule = routesManifest.headers?.find((rule) => rule.source === "/_next/static/:path*");
if (!staticHeaderRule) {
  fail("Missing explicit /_next/static cache header rule in routes manifest.");
}
const staticCacheHeader = staticHeaderRule.headers?.find((header) => header.key.toLowerCase() === "cache-control")?.value ?? "";
if (!staticCacheHeader.includes("immutable") || !staticCacheHeader.includes("max-age=31536000")) {
  fail("Static asset cache header rule is missing immutable long-term caching.");
}

const startProcess = spawn(
  process.execPath,
  [resolve(rootDir, "node_modules", "next", "dist", "bin", "next"), "start", "-p", String(port)],
  {
    cwd: rootDir,
    env: {
      ...process.env,
      PORT: String(port),
    },
    stdio: ["ignore", "pipe", "pipe"],
  },
);

let serverOutput = "";
startProcess.stdout.on("data", (chunk) => {
  serverOutput += chunk.toString();
});
startProcess.stderr.on("data", (chunk) => {
  serverOutput += chunk.toString();
});

const origin = `http://127.0.0.1:${port}`;

try {
  await waitForServer(`${origin}/build-info`);

  const htmlResponse = await fetch(`${origin}/`, { redirect: "manual" });
  const buildInfoResponse = await fetch(`${origin}/build-info`, { redirect: "manual", cache: "no-store" });
  const assetResponse = await fetch(`${origin}${assetPaths[0]}`, { redirect: "manual" });

  const htmlCacheControl = htmlResponse.headers.get("cache-control") ?? "";
  const buildInfoCacheControl = buildInfoResponse.headers.get("cache-control") ?? "";
  const assetCacheControl = assetResponse.headers.get("cache-control") ?? "";
  const buildInfoPayload = await buildInfoResponse.json();

  if (!htmlCacheControl.includes("no-cache")) {
    fail(`Expected HTML Cache-Control to include no-cache, got: ${htmlCacheControl || "<missing>"}`);
  }
  if (!(buildInfoCacheControl.includes("no-store") || buildInfoCacheControl.includes("no-cache"))) {
    fail(`Expected build-info Cache-Control to include no-store or no-cache, got: ${buildInfoCacheControl || "<missing>"}`);
  }
  if (
    !assetCacheControl.includes("public") ||
    !assetCacheControl.includes("max-age=31536000") ||
    !assetCacheControl.includes("immutable")
  ) {
    fail(`Expected static asset Cache-Control to be immutable, got: ${assetCacheControl || "<missing>"}`);
  }

  const report = {
    assetCount: assetPaths.length,
    assetSample: assetPaths.slice(0, 10),
    buildTimestamp: buildInfoPayload.buildTimestamp ?? null,
    buildVersion: buildInfoPayload.buildVersion ?? null,
    cacheHeaders: {
      asset: assetCacheControl,
      buildInfo: buildInfoCacheControl,
      html: htmlCacheControl,
    },
    verifiedAt: new Date().toISOString(),
  };

  if (previousReportPath) {
    const previousReport = readJson(previousReportPath);
    report.previousBuildVersion = previousReport.buildVersion ?? null;
    report.assetListChanged = JSON.stringify(previousReport.assetSample ?? []) !== JSON.stringify(report.assetSample);
    if (!report.assetListChanged) {
      fail("Current asset sample matches the previous report. Compare against a prior build to confirm bundle hash changes.");
    }
  }

  writeFileSync(reportPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  startProcess.kill("SIGTERM");
}
