import LeadExplorerClient from "./lead-explorer-client";
import stateConfig from "./lead-explorer/stateConfig";

export const dynamic = "force-dynamic";
export const revalidate = 0;

type PageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

export default async function Home({ searchParams }: PageProps) {
  const resolvedSearchParams = searchParams ? await searchParams : {};
  const rawStateCode = resolvedSearchParams.state_code ?? resolvedSearchParams.state;
  const rawParcelRowId = resolvedSearchParams.parcel_row_id ?? resolvedSearchParams.parcel;
  const initialParcelRowIdValue = Array.isArray(rawParcelRowId) ? rawParcelRowId[0] : rawParcelRowId;
  const initialParcelRowId =
    typeof initialParcelRowIdValue === "string" && /^row_/i.test(initialParcelRowIdValue.trim())
      ? initialParcelRowIdValue.trim()
      : null;
  const initialStateCode = stateConfig.normalizeStateCode(
    Array.isArray(rawStateCode) ? rawStateCode[0] : rawStateCode,
  );
  return <LeadExplorerClient initialStateCode={initialStateCode} initialParcelRowId={initialParcelRowId} />;
}
