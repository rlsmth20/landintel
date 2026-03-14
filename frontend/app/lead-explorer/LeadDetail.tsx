"use client";

import type { LeadRecord } from "./types";
import { badgeTone, formatBoolean, formatCurrency, formatNumber } from "./utils";

function LeadBadge({ label, tone }: { label: string; tone?: string }) {
  return <span className={`badge badge-${tone ?? "neutral"}`}>{label}</span>;
}

function DetailSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="detail-section">
      <h4>{title}</h4>
      <div className="detail-grid">{children}</div>
    </section>
  );
}

function DetailRow({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <>
      <span className="detail-label">{label}</span>
      <span className="detail-value">{value ?? "-"}</span>
    </>
  );
}

export function LeadDetail({ lead }: { lead: LeadRecord }) {
  return (
    <div className="detail-scroll">
      <div className="detail-header">
        <p className="eyebrow">{lead.county_name} parcel intelligence record</p>
        <h3>{lead.parcel_id ?? lead.parcel_row_id}</h3>
        <div className="inline-badges">
          <LeadBadge label={lead.lead_score_tier ?? "-"} tone="good" />
          <LeadBadge label={lead.county_hosted_flag ? "County-hosted" : "SOS"} tone={lead.county_hosted_flag ? "good" : "warn"} />
          <LeadBadge label={lead.amount_trust_tier ?? "-"} tone={badgeTone(lead.amount_trust_tier)} />
        </div>
      </div>

      <DetailSection title="Parcel Basics">
        <DetailRow label="Parcel row ID" value={lead.parcel_row_id} />
        <DetailRow label="Parcel ID" value={lead.parcel_id} />
        <DetailRow label="County" value={lead.county_name} />
        <DetailRow label="County FIPS" value={lead.county_fips} />
        <DetailRow label="Acreage" value={formatNumber(lead.acreage, 2)} />
        <DetailRow label="Acreage bucket" value={lead.acreage_bucket} />
        <DetailRow label="Land use" value={lead.land_use} />
        <DetailRow label="Assessed total value" value={formatCurrency(lead.assessed_total_value)} />
      </DetailSection>

      <DetailSection title="Motivation Signals">
        <DetailRow label="Raw vacant flag" value={formatBoolean(lead.parcel_vacant_flag)} />
        <DetailRow label="Delinquent amount" value={formatCurrency(lead.delinquent_amount)} />
        <DetailRow label="Delinquent amount bucket" value={lead.delinquent_amount_bucket} />
        <DetailRow label="Delinquent flag" value={formatBoolean(lead.delinquent_flag)} />
        <DetailRow label="Forfeited flag" value={formatBoolean(lead.forfeited_flag)} />
      </DetailSection>

      <DetailSection title="Vacancy Intelligence">
        <DetailRow label="Footprint vacant flag" value={formatBoolean(lead.parcel_vacant_flag)} />
        <DetailRow label="County vacant flag" value={formatBoolean(lead.county_vacant_flag)} />
        <DetailRow label="AI building-present signal" value={formatBoolean(lead.ai_building_present_flag)} />
        <DetailRow label="Overall vacancy assessment" value={lead.overall_vacancy_assessment} />
        <DetailRow label="Vacancy likelihood score" value={formatNumber(lead.vacancy_confidence_score, 1)} />
        <DetailRow label="Vacant reason" value={lead.vacant_reason} />
      </DetailSection>

      <DetailSection title="Ownership">
        <DetailRow label="Owner name" value={lead.owner_name} />
        <DetailRow label="Owner type" value={lead.owner_type} />
        <DetailRow label="Corporate owner" value={formatBoolean(lead.corporate_owner_flag)} />
        <DetailRow label="Absentee owner" value={formatBoolean(lead.absentee_owner_flag)} />
        <DetailRow label="Out-of-state owner" value={formatBoolean(lead.out_of_state_owner_flag)} />
        <DetailRow label="Owner parcel count" value={formatNumber(lead.owner_parcel_count)} />
        <DetailRow label="Owner total acres" value={formatNumber(lead.owner_total_acres, 2)} />
        <DetailRow label="Mailer target score" value={formatNumber(lead.mailer_target_score, 1)} />
      </DetailSection>

      <DetailSection title="Physical Constraints">
        <DetailRow label="Building count" value={formatNumber(lead.building_count)} />
        <DetailRow label="Building area total" value={formatNumber(lead.building_area_total, 0)} />
        <DetailRow label="Road distance ft" value={formatNumber(lead.road_distance_ft, 0)} />
        <DetailRow label="Road access tier" value={lead.road_access_tier} />
        <DetailRow label="Buildability score" value={formatNumber(lead.buildability_score, 1)} />
        <DetailRow label="Environment score" value={formatNumber(lead.environment_score, 1)} />
      </DetailSection>

      <DetailSection title="Terrain & Shape">
        <DetailRow label="Slope mean %" value={formatNumber(lead.mean_slope_pct, 2)} />
        <DetailRow label="Slope max %" value={formatNumber(lead.max_slope_pct, 2)} />
        <DetailRow label="Slope class" value={lead.slope_class} />
        <DetailRow label="Slope score" value={formatNumber(lead.slope_score, 1)} />
        <DetailRow label="Elevation mean ft" value={formatNumber(lead.elevation_mean_ft, 0)} />
        <DetailRow label="Shape compactness" value={formatNumber(lead.shape_compactness, 3)} />
        <DetailRow label="Frontage estimate ft" value={formatNumber(lead.parcel_frontage_ft_estimate, 0)} />
        <DetailRow label="Width estimate ft" value={formatNumber(lead.parcel_width_ft_estimate, 0)} />
      </DetailSection>

      <DetailSection title="Utilities">
        <DetailRow label="Electric provider" value={lead.electric_provider_name} />
      </DetailSection>

      <DetailSection title="Flood / Wetlands">
        <DetailRow label="Wetland" value={formatBoolean(lead.wetland_flag)} />
        <DetailRow label="Wetland coverage %" value={formatNumber(lead.wetland_pct, 1)} />
        <DetailRow label="Wetland area sqft" value={formatNumber(lead.wetland_area_sqft, 0)} />
        <DetailRow label="Flood risk score" value={formatNumber(lead.flood_risk_score, 1)} />
        <DetailRow label="Flood coverage %" value={formatNumber(lead.flood_pct, 1)} />
        <DetailRow label="Flood area sqft" value={formatNumber(lead.flood_area_sqft, 0)} />
        <DetailRow label="Primary FEMA zone" value={lead.primary_fema_zone} />
      </DetailSection>

      <DetailSection title="Market Context">
        <DetailRow label="Nearby building count 1km" value={formatNumber(lead.nearby_building_count_1km)} />
        <DetailRow label="Nearby building density" value={formatNumber(lead.nearby_building_density, 2)} />
        <DetailRow label="Growth pressure bucket" value={lead.growth_pressure_bucket} />
        <DetailRow label="Investment score" value={formatNumber(lead.investment_score, 1)} />
      </DetailSection>

      <DetailSection title="Risk">
        <DetailRow label="Best source type" value={lead.best_source_type} />
        <DetailRow label="Best source name" value={lead.best_source_name} />
        <DetailRow label="Source confidence tier" value={lead.source_confidence_tier} />
        <DetailRow label="County coverage tier" value={lead.county_source_coverage_tier} />
        <DetailRow label="Amount trust tier" value={lead.amount_trust_tier} />
        <DetailRow label="High-confidence link" value={formatBoolean(lead.high_confidence_link_flag)} />
      </DetailSection>

      <DetailSection title="Acquisition Score">
        <DetailRow label="Acquisition score" value={formatNumber(lead.lead_score_total, 2)} />
        <DetailRow label="Acquisition tier" value={lead.lead_score_tier} />
        <DetailRow label="Driver 1" value={lead.lead_score_driver_1} />
        <DetailRow label="Driver 2" value={lead.lead_score_driver_2} />
        <DetailRow label="Driver 3" value={lead.lead_score_driver_3} />
        <DetailRow label="Explanation" value={lead.lead_score_explanation} />
        <DetailRow label="Size score" value={formatNumber(lead.size_score, 1)} />
        <DetailRow label="Access score" value={formatNumber(lead.access_score, 1)} />
        <DetailRow label="Buildability component" value={formatNumber(lead.buildability_component, 1)} />
        <DetailRow label="Environmental component" value={formatNumber(lead.environmental_component, 1)} />
        <DetailRow label="Owner targeting component" value={formatNumber(lead.owner_targeting_component, 1)} />
        <DetailRow label="Delinquency component" value={formatNumber(lead.delinquency_component, 1)} />
        <DetailRow label="Source confidence component" value={formatNumber(lead.source_confidence_component, 1)} />
        <DetailRow label="Vacant land component" value={formatNumber(lead.vacant_land_component, 1)} />
        <DetailRow label="Growth pressure component" value={formatNumber(lead.growth_pressure_component, 1)} />
      </DetailSection>

      <DetailSection title="Product Explanations">
        <DetailRow label="Recommended sort reason" value={lead.recommended_sort_reason} />
        <DetailRow label="Top score driver" value={lead.top_score_driver} />
        <DetailRow label="Caution flags" value={lead.caution_flags} />
        <DetailRow label="Growth pressure reason" value={lead.growth_pressure_reason} />
        <DetailRow label="Recommended use case" value={lead.recommended_use_case} />
      </DetailSection>
    </div>
  );
}
