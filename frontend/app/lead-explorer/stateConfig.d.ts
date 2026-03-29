type StateConfig = {
  stateCode: string;
  displayName: string;
  countyDivisionLabel: string;
  apiPrefix: string;
  staticMetaPath: string;
  staticLeadDetailPath: string;
  parcelPmtilesLocalUrl: string | null;
  parcelPmtilesPublicUrl: string | null;
  parcelPmtilesUrl: string | null;
  parcelPmtilesMinZoom: number;
  selectionEnabled: boolean;
  defaultBounds: [number, number, number, number];
};

declare const stateConfig: {
  DEFAULT_STATE_CODE: string;
  DEFAULT_PARCEL_PMTILES_MIN_ZOOM: number;
  normalizeStateCode(value: unknown): string;
  getStateConfig(stateCode?: string): StateConfig;
  getActiveStateConfig(): StateConfig;
  getKnownStateConfigs(): StateConfig[];
  isStateSelectable(stateCode?: string): boolean;
  readStateCodeFromSearch(search?: string | null, fallbackStateCode?: string): string;
  resolveSelectableStateCode(stateCode?: string, fallbackStateCode?: string): string;
  buildStateApiPath(stateCode: string, suffix: string): string;
  resolveParcelPmtilesUrl(config: {
    stateCode: string;
    parcelPmtilesLocalUrl: string | null;
    parcelPmtilesPublicUrl: string | null;
  }): string | null;
};

export = stateConfig;
