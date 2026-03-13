import type { Filters, MapOverlayDefinition } from "./types";

type TableColumnDefinition = {
  key: string;
  label: string;
  tooltip: string;
  sortField?: "lead_score_total" | "acreage" | "delinquent_amount" | "road_distance_ft";
};

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
  leadScoreTier: [],
  minLeadScore: 0,
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

export const TABLE_COLUMNS: TableColumnDefinition[] = [
  {
    key: "county",
    label: "County",
    tooltip: "The Mississippi county where the parcel is located.",
  },
  {
    key: "parcel_id",
    label: "Parcel ID",
    tooltip: "The county parcel identifier used for parcel-level lookup and review.",
  },
  {
    key: "acreage",
    label: "Acreage",
    tooltip: "Estimated parcel acreage from the statewide parcel master.",
    sortField: "acreage",
  },
  {
    key: "owner",
    label: "Owner",
    tooltip: "Current owner name from parcel ownership records.",
  },
  {
    key: "lead_score_total",
    label: "Acquisition Score",
    tooltip: "A composite score estimating how attractive a parcel may be for acquisition based on ownership signals, physical constraints, and market indicators.",
    sortField: "lead_score_total",
  },
  {
    key: "lead_score_tier",
    label: "Lead Tier",
    tooltip: "A bucketed view of the acquisition score from low to very high.",
  },
  {
    key: "parcel_vacant_flag",
    label: "Likely Vacant",
    tooltip: "Indicates whether building footprints suggest the parcel is likely vacant land.",
  },
  {
    key: "road_access_tier",
    label: "Road Access",
    tooltip: "An access tier derived from parcel-to-road distance.",
    sortField: "road_distance_ft",
  },
  {
    key: "growth_pressure_bucket",
    label: "Area Growth Potential",
    tooltip: "A nearby-building-density bucket that approximates local growth pressure.",
  },
  {
    key: "best_source_type",
    label: "Data Source",
    tooltip: "The strongest available motivation-source context for the parcel, such as county-hosted delinquent data or statewide inventory.",
  },
  {
    key: "source_confidence_tier",
    label: "Data Confidence",
    tooltip: "An overall confidence tier for the parcel’s motivation and signal data.",
  },
  {
    key: "delinquent_amount",
    label: "Delinquent Tax Amount",
    tooltip: "The best available reported delinquent tax amount when a tax signal exists.",
    sortField: "delinquent_amount",
  },
  {
    key: "amount_trust_tier",
    label: "Amount Reliability",
    tooltip: "Indicates whether the displayed delinquent tax amount is trusted, cautionary, or not ready for prominent display.",
  },
  {
    key: "recommended_sort_reason",
    label: "Lead Reason",
    tooltip: "The main factor currently pushing the parcel toward the top of the ranked view.",
  },
] as const;

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
