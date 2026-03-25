import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json(
    {
      buildTimestamp: process.env.NEXT_PUBLIC_BUILD_TIMESTAMP ?? null,
      buildVersion: process.env.NEXT_PUBLIC_BUILD_VERSION ?? "development",
    },
    {
      headers: {
        "Cache-Control": "no-store, no-cache, must-revalidate",
      },
    },
  );
}
