import type { NextConfig } from "next";

const buildTimestamp = new Date().toISOString();
const buildVersion =
  process.env.NEXT_PUBLIC_BUILD_VERSION ??
  process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 12) ??
  process.env.RAILWAY_GIT_COMMIT_SHA?.slice(0, 12) ??
  process.env.SOURCE_VERSION?.slice(0, 12) ??
  buildTimestamp.replace(/[-:.TZ]/g, "");

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_BUILD_TIMESTAMP: buildTimestamp,
    NEXT_PUBLIC_BUILD_VERSION: buildVersion,
  },
  generateBuildId: async () =>
    process.env.NEXT_BUILD_ID ??
    process.env.VERCEL_GIT_COMMIT_SHA?.slice(0, 20) ??
    process.env.RAILWAY_GIT_COMMIT_SHA?.slice(0, 20) ??
    process.env.SOURCE_VERSION?.slice(0, 20) ??
    buildVersion,
  async headers() {
    return [
      {
        source: "/_next/static/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
      {
        source: "/build-info",
        headers: [
          {
            key: "Cache-Control",
            value: "no-store, no-cache, must-revalidate",
          },
        ],
      },
      {
        source: "/((?!_next/static|build-info).*)",
        headers: [
          {
            key: "Cache-Control",
            value: "no-cache",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
