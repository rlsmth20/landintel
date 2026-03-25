import type { LeadRecord, SearchResultRecord } from "./types";

type Bounds = [number, number, number, number];

type BoundsFeature = {
  geometry?: unknown;
  properties?: Record<string, unknown> | null;
} | null | undefined;

type BoundsFeatureCollection = {
  features?: BoundsFeature[] | null;
} | null | undefined;

declare const parcelIdentity: {
  buildLeadDetailPath(parcelRowId: string, stateCode?: string): string;
  buildNearbyCompsPath(parcelRowId: string, stateCode?: string): string;
  buildParcelGeometryPath(parcelRowId: string, stateCode?: string): string;
  extractDisplayedParcelId(
    recordOrParcelId: Record<string, unknown> | string | null | undefined,
    parcelRowId?: string | null | undefined,
  ): string | null;
  featureBounds(feature: BoundsFeature): Bounds | null;
  getDisplayedParcelId(
    recordOrParcelId: Record<string, unknown> | string | null | undefined,
    parcelRowId?: string | null | undefined,
  ): string;
  getLeadSelectionParcelRowId(lead: Pick<LeadRecord, "parcel_row_id"> | null | undefined): string | null;
  getMapFeatureSelectionParcelRowId(feature: BoundsFeature): string | null;
  getSearchSelectionParcelRowId(
    result: Pick<SearchResultRecord, "parcel_row_id"> | null | undefined,
  ): string | null;
  isInternalParcelRowId(value: string | null | undefined): boolean;
  mergeBounds(boundsList: Bounds[]): Bounds | null;
  readDeepLinkedParcelRowId(search: string): string | null;
  selectedFeatureBounds(featureCollection: BoundsFeatureCollection, selectedParcelRowId: string | null): Bounds | null;
};

export default parcelIdentity;
