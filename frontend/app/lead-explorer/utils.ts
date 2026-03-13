import type { Filters, MapOverlayDefinition } from "./types";

export const PRESET_LABELS: Record<string, string> = {
  safest_early_investor_use: "Safest Outreach",
  vacant_land_targeting: "Vacant Buildable",
  larger_acreage_land_targeting: "Larger Land Target",
  growth_edge_targeting: "Growth Edge",
};

export const MAP_OVERLAYS: MapOverlayDefinition[] = [
  {
    id: "parcels",
    label: "Parcels",
    description: "Core parcel geometry layer for the current result set.",
    enabled: true,
  },
  {
    id: "fema_flood",
    label: "FEMA Flood",
    description: "Prepared for parcel flood overlay and future flood coverage metrics.",
    enabled: false,
  },
  {
    id: "wetlands",
    label: "Wetlands",
    description: "Parcel wetland signal today; swap to dedicated overlay geometry later.",
    enabled: true,
  },
  {
    id: "utilities",
    label: "Utilities",
    description: "Utility context architecture for future proximity and service overlays.",
    enabled: false,
  },
  {
    id: "slope",
    label: "Slope",
    description: "Prepared for future terrain and buildability overlays.",
    enabled: false,
  },
  {
    id: "road_access",
    label: "Road Access",
    description: "Highlights parcel access quality using current road-distance intelligence.",
    enabled: true,
  },
  {
    id: "zoning",
    label: "Zoning",
    description: "Reserved for future zoning and entitlement overlays.",
    enabled: false,
  },
];

export const INITIAL_FILTERS: Filters = {
  countyName: "all",
  leadScoreTier: ["high", "very_high"],
  minLeadScore: 65,
  acreageMin: "",
  acreageMax: "",
  parcelVacantOnly: false,
  countyHostedOnly: false,
  highConfidenceOnly: false,
  wetlandMode: "any",
  amountTrustTiers: ["trusted", "use_with_caution", "not_trusted_for_prominent_display"],
  corporateOnly: false,
  absenteeOnly: false,
  outOfStateOnly: false,
  growthPressureBuckets: [],
  recommendedViewBucket: "all",
  roadDistanceMax: "",
  roadAccessTiers: [],
};

export function formatNumber(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

export function formatCurrency(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatBoolean(value: boolean | null | undefined) {
  if (value === null || value === undefined) return "-";
  return value ? "Yes" : "No";
}

export function toggleSelection(values: string[], next: string) {
  return values.includes(next) ? values.filter((value) => value !== next) : [...values, next];
}

export function badgeTone(value: string | null | undefined) {
  if (!value) return "neutral";
  if (value.includes("trusted") && !value.includes("not_")) return "good";
  if (value.includes("caution") || value.includes("sos")) return "warn";
  if (value.includes("not_") || value.includes("low")) return "bad";
  if (value.includes("direct") || value.includes("county")) return "good";
  return "neutral";
}

export function applyPreset(name: string): Filters {
  if (name === "safest_early_investor_use") {
    return {
      ...INITIAL_FILTERS,
      parcelVacantOnly: true,
      countyHostedOnly: true,
      highConfidenceOnly: true,
      wetlandMode: "exclude",
      amountTrustTiers: ["trusted", "use_with_caution"],
    };
  }
  if (name === "vacant_land_targeting") {
    return {
      ...INITIAL_FILTERS,
      parcelVacantOnly: true,
      wetlandMode: "exclude",
      roadAccessTiers: ["direct", "near"],
    };
  }
  if (name === "larger_acreage_land_targeting") {
    return {
      ...INITIAL_FILTERS,
      parcelVacantOnly: true,
      countyHostedOnly: true,
      acreageMin: "5",
    };
  }
  if (name === "growth_edge_targeting") {
    return {
      ...INITIAL_FILTERS,
      growthPressureBuckets: ["moderate", "high"],
      roadAccessTiers: ["direct", "near", "moderate"],
    };
  }
  return INITIAL_FILTERS;
}
