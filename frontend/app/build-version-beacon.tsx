"use client";

import { useEffect } from "react";

const BUILD_INFO_URL = "/build-info";
const RELOAD_MARKER_KEY = "landintel-build-version-reload";
const bundledBuildTimestamp = process.env.NEXT_PUBLIC_BUILD_TIMESTAMP ?? null;
const bundledBuildVersion = process.env.NEXT_PUBLIC_BUILD_VERSION ?? "development";

export default function BuildVersionBeacon() {
  useEffect(() => {
    console.info("[landintel] frontend build", {
      buildTimestamp: bundledBuildTimestamp,
      buildVersion: bundledBuildVersion,
    });

    let cancelled = false;

    async function checkBuildVersion() {
      try {
        const response = await fetch(BUILD_INFO_URL, { cache: "no-store" });
        if (!response.ok) {
          return;
        }

        const payload = (await response.json()) as {
          buildTimestamp?: string | null;
          buildVersion?: string | null;
        };
        if (cancelled) {
          return;
        }

        const liveBuildVersion = payload.buildVersion ?? null;
        if (!liveBuildVersion || liveBuildVersion === bundledBuildVersion) {
          sessionStorage.removeItem(RELOAD_MARKER_KEY);
          return;
        }

        const reloadMarker = `${bundledBuildVersion}->${liveBuildVersion}`;
        if (sessionStorage.getItem(RELOAD_MARKER_KEY) === reloadMarker) {
          return;
        }

        console.warn("[landintel] stale frontend bundle detected, forcing reload", {
          bundledBuildVersion,
          liveBuildVersion,
        });
        sessionStorage.setItem(RELOAD_MARKER_KEY, reloadMarker);
        window.location.reload();
      } catch {
        // Keep the page usable if the build-info check fails.
      }
    }

    void checkBuildVersion();

    return () => {
      cancelled = true;
    };
  }, []);

  return null;
}
