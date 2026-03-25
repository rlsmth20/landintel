declare const stateConfig: {
  DEFAULT_STATE_CODE: string;
  normalizeStateCode(value: unknown): string;
  getStateConfig(stateCode?: string): {
    stateCode: string;
    displayName: string;
    apiPrefix: string;
    staticMetaPath: string;
    staticLeadDetailPath: string;
    parcelPmtilesUrl: string | null;
    defaultBounds: [number, number, number, number];
  };
  getActiveStateConfig(): {
    stateCode: string;
    displayName: string;
    apiPrefix: string;
    staticMetaPath: string;
    staticLeadDetailPath: string;
    parcelPmtilesUrl: string | null;
    defaultBounds: [number, number, number, number];
  };
  getKnownStateConfigs(): Array<{
    stateCode: string;
    displayName: string;
    apiPrefix: string;
    staticMetaPath: string;
    staticLeadDetailPath: string;
    parcelPmtilesUrl: string | null;
    defaultBounds: [number, number, number, number];
  }>;
  readStateCodeFromSearch(search?: string | null, fallbackStateCode?: string): string;
  buildStateApiPath(stateCode: string, suffix: string): string;
};

export = stateConfig;
