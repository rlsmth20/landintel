"use client";

import type { LeadRecord } from "./types";
import { badgeTone, formatBoolean, formatCurrency, formatNumber } from "./utils";

function formatDateValue(value: string | null | undefined) {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleDateString();
}

function humanizeValue(value: string | null | undefined) {
  if (!value) {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  if (!/[_-]/.test(trimmed)) {
    return trimmed;
  }
  return trimmed
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

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

function DetailDisclosure({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="detail-disclosure">
      <summary className="detail-disclosure-summary">{title}</summary>
      <div className="detail-disclosure-content">{children}</div>
    </details>
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

function formatRoadDistance(value: number | null | undefined) {
  const formatted = formatNumber(value, 0);
  return formatted ? `${formatted} ft` : null;
}

function formatTaxDelinquency(lead: LeadRecord) {
  return humanizeValue(lead.parcel_tax_status_label ?? lead.parcel_tax_status) ?? formatBoolean(lead.delinquent_flag);
}

function formatTaxFreshness(lead: LeadRecord) {
  const freshness = humanizeValue(lead.parcel_tax_freshness_bucket);
  const coverage = humanizeValue(lead.county_tax_coverage_status);
  const year = lead.tax_data_year ? `tax year ${formatNumber(lead.tax_data_year)}` : null;
  const parts = [freshness, coverage && coverage !== freshness ? coverage : null, year].filter(Boolean);
  return parts.length ? parts.join(" · ") : null;
}

function taxFreshnessTone(lead: LeadRecord) {
  const bucket = lead.parcel_tax_freshness_bucket ?? "";
  if (bucket === "current" || bucket === "actionable_recent") {
    return "good";
  }
  if (bucket === "historical_only" || bucket === "non_actionable_historical") {
    return "neutral";
  }
  if (bucket === "stale_caution") {
    return "warn";
  }
  return badgeTone(lead.county_tax_coverage_status);
}

function formatVacancyNote(lead: LeadRecord) {
  return lead.vacant_reason ?? lead.building_presence_reason ?? null;
}

function aiUnavailable(lead: LeadRecord) {
  return lead.ai_vacancy_available_flag === false || lead.ai_vacancy_source === "unavailable";
}

function formatAiVacancySignal(lead: LeadRecord) {
  if (aiUnavailable(lead)) {
    return "Unavailable";
  }
  const confidence = lead.building_present_confidence;
  if (confidence !== null && confidence !== undefined) {
    if (confidence >= 65) {
      return "Possibly improved";
    }
    if (confidence <= 35) {
      return "Likely vacant";
    }
    return "Unclear";
  }
  if (lead.ai_building_present_flag === true) {
    return "Possibly improved";
  }
  if (lead.ai_building_present_flag === false) {
    return "Likely vacant";
  }
  return "Unclear";
}

function formatAiConfidence(lead: LeadRecord) {
  if (aiUnavailable(lead)) {
    return null;
  }
  const confidence = lead.building_present_confidence;
  if (confidence === null || confidence === undefined) {
    return null;
  }
  const certainty = Math.abs(confidence - 50);
  if (certainty >= 30) {
    return "High";
  }
  if (certainty >= 15) {
    return "Medium";
  }
  return "Low";
}

function formatAiWhy(lead: LeadRecord) {
  if (aiUnavailable(lead)) {
    return lead.ai_vacancy_status_note ?? "AI vacancy prediction is unavailable in this parcel detail source.";
  }
  const confidence = lead.building_present_confidence;
  if (confidence !== null && confidence !== undefined) {
    if (confidence >= 65) {
      return "Imagery suggests a structure or improvement is present on the parcel.";
    }
    if (confidence <= 35) {
      return "Imagery does not clearly show a structure on the parcel.";
    }
  }
  return lead.vacant_reason ?? lead.building_presence_reason ?? lead.overall_vacancy_assessment ?? null;
}

function formatWetland(lead: LeadRecord) {
  if ((lead.wetland_pct ?? 0) > 0) {
    return `${formatNumber(lead.wetland_pct, 1)}% coverage`;
  }
  return formatBoolean(lead.wetland_flag);
}

function formatFloodRisk(lead: LeadRecord) {
  if (lead.primary_fema_zone && (lead.flood_pct ?? 0) > 0) {
    return `${lead.primary_fema_zone} · ${formatNumber(lead.flood_pct, 1)}% coverage`;
  }
  if ((lead.flood_pct ?? 0) > 0) {
    return `${formatNumber(lead.flood_pct, 1)}% coverage`;
  }
  return formatNumber(lead.flood_risk_score, 1);
}

export function LeadDetail({ lead }: { lead: LeadRecord }) {
  const parcelIdValue = lead.parcel_id ?? "Not available";
  const leadTier = humanizeValue(lead.lead_score_tier) ?? "-";
  const taxFreshness = formatTaxFreshness(lead);

  return (
    <div className="detail-scroll">
      <div className="detail-header">
        <p className="eyebrow">{lead.county_name} parcel intelligence record</p>
        <h3>{parcelIdValue}</h3>
        <div className="inline-badges">
          <LeadBadge label={leadTier} tone="good" />
          {taxFreshness ? <LeadBadge label={taxFreshness} tone={taxFreshnessTone(lead)} /> : null}
        </div>
      </div>

      <DetailSection title="Overview">
        <DetailRow label="Parcel ID" value={parcelIdValue} />
        <DetailRow label="County" value={lead.county_name} />
        <DetailRow label="Acreage" value={formatNumber(lead.acreage, 2)} />
        <DetailRow label="Land use" value={lead.land_use} />
        <DetailRow label="Lead score" value={formatNumber(lead.lead_score_total, 2)} />
        <DetailRow label="Lead tier" value={leadTier} />
        <DetailRow label="Best use" value={humanizeValue(lead.recommended_use_case)} />
      </DetailSection>

      <DetailSection title="Motivation">
        <DetailRow label="Tax delinquency" value={formatTaxDelinquency(lead)} />
        <DetailRow label="Delinquent amount" value={formatCurrency(lead.delinquent_amount)} />
        <DetailRow label="Delinquent year" value={formatNumber(lead.delinquent_year)} />
        <DetailRow label="Tax data freshness" value={taxFreshness} />
        <DetailRow label="Vacancy assessment" value={lead.overall_vacancy_assessment} />
        <DetailRow label="Vacancy note" value={formatVacancyNote(lead)} />
      </DetailSection>

      <DetailSection title="Vacancy Intelligence">
        <DetailRow label="AI vacancy signal" value={formatAiVacancySignal(lead)} />
        <DetailRow label="Confidence" value={formatAiConfidence(lead)} />
        <DetailRow label="Why" value={formatAiWhy(lead)} />
      </DetailSection>

      <DetailSection title="Ownership">
        <DetailRow label="Owner name" value={lead.owner_name} />
        <DetailRow label="Owner type" value={lead.owner_type} />
        <DetailRow label="Business-owned" value={formatBoolean(lead.corporate_owner_flag)} />
        <DetailRow label="Mailing address differs" value={formatBoolean(lead.absentee_owner_flag)} />
        <DetailRow label="Out-of-state owner" value={formatBoolean(lead.out_of_state_owner_flag)} />
      </DetailSection>

      <DetailSection title="Physical">
        <DetailRow label="Building count" value={formatNumber(lead.building_count)} />
        <DetailRow label="Road access" value={lead.road_access_tier} />
        <DetailRow label="Distance to road" value={formatRoadDistance(lead.road_distance_ft)} />
        <DetailRow label="Buildability" value={formatNumber(lead.buildability_score, 1)} />
        <DetailRow label="Slope class" value={lead.slope_class} />
        <DetailRow label="Flood risk" value={formatFloodRisk(lead)} />
        <DetailRow label="Wetland" value={formatWetland(lead)} />
      </DetailSection>

      <DetailSection title="Market">
        <DetailRow label="Investment score" value={formatNumber(lead.investment_score, 1)} />
        <DetailRow label="Development pressure" value={humanizeValue(lead.growth_pressure_bucket)} />
      </DetailSection>

      <DetailDisclosure title="Advanced details">
        <DetailSection title="Identifiers & Tax Data">
          <DetailRow label="Parcel row ID" value={lead.parcel_row_id} />
          <DetailRow label="County FIPS" value={lead.county_fips} />
          <DetailRow label="Acreage bucket" value={lead.acreage_bucket} />
          <DetailRow label="Assessed total value" value={formatCurrency(lead.assessed_total_value)} />
          <DetailRow label="Parcel tax status" value={lead.parcel_tax_status} />
          <DetailRow label="Tax actionability" value={humanizeValue(lead.parcel_tax_actionability)} />
          <DetailRow label="Tax warning" value={lead.parcel_tax_data_warning} />
          <DetailRow label="Tax data year" value={formatNumber(lead.tax_data_year)} />
          <DetailRow label="Tax data upload date" value={formatDateValue(lead.tax_data_upload_date)} />
          <DetailRow label="Tax data source" value={lead.tax_data_source} />
          <DetailRow label="Delinquency last verified" value={formatDateValue(lead.delinquency_last_verified)} />
          <DetailRow label="County tax coverage status" value={humanizeValue(lead.county_tax_coverage_status)} />
          <DetailRow label="County tax coverage note" value={lead.county_tax_coverage_reason} />
          <DetailRow label="County tax source configured" value={formatBoolean(lead.county_tax_source_configured_flag)} />
          <DetailRow label="County tax source loaded" value={formatBoolean(lead.county_tax_source_loaded_flag)} />
          <DetailRow label="Tax data available" value={formatBoolean(lead.tax_data_available_flag)} />
          <DetailRow label="Delinquent amount bucket" value={lead.delinquent_amount_bucket} />
          <DetailRow label="Delinquent flag" value={formatBoolean(lead.delinquent_flag)} />
          <DetailRow label="Forfeited flag" value={formatBoolean(lead.forfeited_flag)} />
        </DetailSection>

        <DetailSection title="Vacancy Model Details">
          <DetailRow label="Footprint vacancy signal" value={formatBoolean(lead.parcel_vacant_flag)} />
          <DetailRow label="County vacant flag" value={formatBoolean(lead.county_vacant_flag)} />
          <DetailRow label="AI availability" value={formatBoolean(lead.ai_vacancy_available_flag)} />
          <DetailRow label="AI source" value={humanizeValue(lead.ai_vacancy_source)} />
          <DetailRow label="AI status note" value={lead.ai_vacancy_status_note} />
          <DetailRow label="AI building-present signal" value={formatBoolean(lead.ai_building_present_flag)} />
          <DetailRow label="AI building probability" value={formatNumber(lead.ai_building_present_probability, 2)} />
          <DetailRow label="Building-present confidence" value={formatNumber(lead.building_present_confidence, 1)} />
          <DetailRow label="Building-presence reason" value={lead.building_presence_reason} />
          <DetailRow label="Vacancy likelihood score" value={formatNumber(lead.vacancy_confidence_score, 1)} />
          <DetailRow label="Vacancy model version" value={lead.vacancy_model_version} />
        </DetailSection>

        <DetailSection title="Ownership Signals">
          <DetailRow label="Owner parcel count" value={formatNumber(lead.owner_parcel_count)} />
          <DetailRow label="Owner total acres" value={formatNumber(lead.owner_total_acres, 2)} />
          <DetailRow label="Mailer target score" value={formatNumber(lead.mailer_target_score, 1)} />
        </DetailSection>

        <DetailSection title="Additional Physical Metrics">
          <DetailRow label="Building area total" value={formatNumber(lead.building_area_total, 0)} />
          <DetailRow label="Environment score" value={formatNumber(lead.environment_score, 1)} />
          <DetailRow label="Slope mean %" value={formatNumber(lead.mean_slope_pct, 2)} />
          <DetailRow label="Slope max %" value={formatNumber(lead.max_slope_pct, 2)} />
          <DetailRow label="Slope score" value={formatNumber(lead.slope_score, 1)} />
          <DetailRow label="Elevation mean ft" value={formatNumber(lead.elevation_mean_ft, 0)} />
          <DetailRow label="Shape compactness" value={formatNumber(lead.shape_compactness, 3)} />
          <DetailRow label="Frontage estimate ft" value={formatNumber(lead.parcel_frontage_ft_estimate, 0)} />
          <DetailRow label="Width estimate ft" value={formatNumber(lead.parcel_width_ft_estimate, 0)} />
          <DetailRow label="Electric provider" value={lead.electric_provider_name} />
          <DetailRow label="Wetland area sqft" value={formatNumber(lead.wetland_area_sqft, 0)} />
          <DetailRow label="Flood area sqft" value={formatNumber(lead.flood_area_sqft, 0)} />
          <DetailRow label="Primary FEMA zone" value={lead.primary_fema_zone} />
          <DetailRow label="Nearby building count 1km" value={formatNumber(lead.nearby_building_count_1km)} />
          <DetailRow label="Nearby building density" value={formatNumber(lead.nearby_building_density, 2)} />
        </DetailSection>

        <DetailSection title="Source & Scoring">
          <DetailRow label="Best source type" value={lead.best_source_type} />
          <DetailRow label="Best source name" value={lead.best_source_name} />
          <DetailRow label="Source confidence tier" value={lead.source_confidence_tier} />
          <DetailRow label="County coverage tier" value={lead.county_source_coverage_tier} />
          <DetailRow label="Amount trust tier" value={lead.amount_trust_tier} />
          <DetailRow label="High-confidence link" value={formatBoolean(lead.high_confidence_link_flag)} />
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
          <DetailRow label="Recommended sort reason" value={lead.recommended_sort_reason} />
          <DetailRow label="Top score driver" value={lead.top_score_driver} />
          <DetailRow label="Caution flags" value={lead.caution_flags} />
          <DetailRow label="Growth pressure reason" value={lead.growth_pressure_reason} />
          <DetailRow label="Recommended view bucket" value={lead.recommended_view_bucket} />
        </DetailSection>
      </DetailDisclosure>
    </div>
  );
}
