"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchGeometryViewport, fetchLeadDetail, fetchLeads, fetchParcelGeometryById, fetchPresets, fetchStaticLeadDetail, fetchSummary } from "./lead-explorer/api";
import { LeadDetail } from "./lead-explorer/LeadDetail";
import { LeadMap } from "./lead-explorer/LeadMap";
import type { ExplorerMeta, Filters, GeometryResponse, LeadRecord, MapOverlayId, MapViewportState, PresetItem, SortField } from "./lead-explorer/types";
import {
  INITIAL_FILTERS,
  MAP_OVERLAYS,
  PRESET_LABELS,
  TABLE_COLUMNS,
  applyPreset,
  badgeTone,
  formatCurrency,
  formatNumber,
  toggleSelection,
} from "./lead-explorer/utils";

const DEFAULT_LIMIT = 200;
const MAX_LIMIT = 250;
const GEOMETRY_LIMIT = 800;
const FILTER_DEBOUNCE_MS = 250;
const DEFAULT_VIEWPORT: MapViewportState = {
  center: [-89.6787, 32.7416],
  zoom: 6.1,
  bounds: [-91.65, 30.15, -88.0, 35.1],
};

function LeadBadge({ label, tone }: { label: string; tone?: string }) {
  return <span className={`badge badge-${tone ?? "neutral"}`}>{label}</span>;
}

function HeaderCell({
  label,
  tooltip,
  onSort,
  indicator,
}: {
  label: string;
  tooltip: string;
  onSort?: () => void;
  indicator?: string;
}) {
  return (
    <th title={tooltip}>
      {onSort ? (
        <button type="button" className="table-header-button" onClick={onSort} title={tooltip}>
          <span>{label}{indicator}</span>
          <span className="header-help" aria-hidden="true">?</span>
        </button>
      ) : (
        <div className="table-header-static">
          <span>{label}</span>
          <span className="header-help" aria-hidden="true">?</span>
        </div>
      )}
    </th>
  );
}

function SummaryValue({
  summary,
  section,
  metric,
  keyValue = "",
}: {
  summary: ExplorerMeta | null;
  section: string;
  metric: string;
  keyValue?: string;
}) {
  const entry =
    summary?.sections?.[section]?.find((item) => item.metric === metric && (item.key ?? "") === keyValue) ?? null;
  return <>{entry?.value ?? "-"}</>;
}

export default function LeadExplorerClient() {
  const [summary, setSummary] = useState<ExplorerMeta | null>(null);
  const [presets, setPresets] = useState<PresetItem[]>([]);
  const [filters, setFilters] = useState<Filters>(INITIAL_FILTERS);
  const [debouncedFilters, setDebouncedFilters] = useState<Filters>(INITIAL_FILTERS);
  const [sortField, setSortField] = useState<SortField>("lead_score_total");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const [offset, setOffset] = useState(0);
  const [limit] = useState(DEFAULT_LIMIT);
  const [activePreset, setActivePreset] = useState<string | null>(null);

  const [leads, setLeads] = useState<LeadRecord[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedLead, setSelectedLead] = useState<LeadRecord | null>(null);
  const [geometryResponse, setGeometryResponse] = useState<GeometryResponse | null>(null);
  const [selectedGeometryResponse, setSelectedGeometryResponse] = useState<GeometryResponse | null>(null);
  const [fitNonce, setFitNonce] = useState(0);
  const [locateSelectedNonce, setLocateSelectedNonce] = useState(0);
  const [activeOverlays, setActiveOverlays] = useState<MapOverlayId[]>(["parcels", "road_access"]);
  const [viewport, setViewport] = useState<MapViewportState>(DEFAULT_VIEWPORT);

  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [leadsLoading, setLeadsLoading] = useState(true);
  const [leadsError, setLeadsError] = useState<string | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [geometryLoading, setGeometryLoading] = useState(false);
  const [geometryError, setGeometryError] = useState<string | null>(null);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setDebouncedFilters(filters);
      setOffset(0);
    }, FILTER_DEBOUNCE_MS);
    return () => window.clearTimeout(timeout);
  }, [filters]);

  useEffect(() => {
    let cancelled = false;
    async function loadBootstrap() {
      setSummaryLoading(true);
      setSummaryError(null);
      try {
        const [summaryResponse, presetsResponse] = await Promise.all([fetchSummary(), fetchPresets()]);
        if (cancelled) return;
        setSummary(summaryResponse);
        setPresets(presetsResponse);
      } catch (error) {
        if (cancelled) return;
        setSummaryError(error instanceof Error ? error.message : "Failed to load summary");
      } finally {
        if (!cancelled) setSummaryLoading(false);
      }
    }
    void loadBootstrap();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadLeads() {
      setLeadsLoading(true);
      setLeadsError(null);
      try {
        const response = await fetchLeads(
          debouncedFilters,
          sortField,
          sortDirection,
          Math.min(limit, MAX_LIMIT),
          offset,
        );
        if (cancelled) return;
        setLeads(response.items);
        setTotalCount(response.total_count);
        if (response.items.length === 0) {
          setGeometryResponse(null);
          return;
        }
        setSelectedId((current) => {
          if (current) return current;
          return response.items[0].parcel_row_id;
        });
      } catch (error) {
        if (cancelled) return;
        setLeadsError(error instanceof Error ? error.message : "Failed to load leads");
      } finally {
        if (!cancelled) setLeadsLoading(false);
      }
    }
    void loadLeads();
    return () => {
      cancelled = true;
    };
  }, [debouncedFilters, limit, offset, sortDirection, sortField]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedGeometryResponse(null);
      return;
    }
    const selectedParcelId = selectedId;
    let cancelled = false;
    async function loadSelectedGeometry() {
      try {
        const response = await fetchParcelGeometryById(selectedParcelId, Math.max(viewport.zoom, 13));
        if (cancelled) return;
        setSelectedGeometryResponse(response);
      } catch (error) {
        if (cancelled) return;
        if (process.env.NODE_ENV !== "production") {
          console.debug("[lead-explorer] selected geometry failed", { selectedParcelId, error });
        }
        setSelectedGeometryResponse(null);
      }
    }
    void loadSelectedGeometry();
    return () => {
      cancelled = true;
    };
  }, [selectedId, viewport.zoom]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedLead(null);
      return;
    }
    const detailId = selectedId;
    const summaryFallback = leads.find((item) => item.parcel_row_id === detailId) ?? null;
    let cancelled = false;
    async function loadDetail() {
      setDetailLoading(true);
      setDetailError(null);
      if (summaryFallback) {
        setSelectedLead(summaryFallback);
      }
      try {
        const response = await fetchLeadDetail(detailId);
        if (cancelled) return;
        setSelectedLead(response);
      } catch (error) {
        if (cancelled) return;
        let staticFallback: LeadRecord | null = null;
        try {
          staticFallback = await fetchStaticLeadDetail(detailId);
        } catch {
          staticFallback = null;
        }
        const fallbackLead = staticFallback ?? summaryFallback;
        if (fallbackLead) {
          setSelectedLead(fallbackLead);
          setDetailError(null);
        } else {
          setDetailError(error instanceof Error ? error.message : "Failed to load parcel detail");
        }
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    }
    void loadDetail();
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!viewport.bounds) {
      setGeometryResponse(null);
      return;
    }

    let cancelled = false;
    async function loadGeometry() {
      setGeometryLoading(true);
      setGeometryError(null);
      try {
        const response = await fetchGeometryViewport(
          debouncedFilters,
          viewport.bounds,
          viewport.zoom,
          selectedId,
          GEOMETRY_LIMIT,
        );
        if (cancelled) return;
        setGeometryResponse(response);
      } catch (error) {
        if (cancelled) return;
        setGeometryError(error instanceof Error ? error.message : "Failed to load parcel geometry");
      } finally {
        if (!cancelled) setGeometryLoading(false);
      }
    }
    void loadGeometry();
    return () => {
      cancelled = true;
    };
  }, [debouncedFilters, selectedId, viewport.bounds, viewport.zoom]);

  const countySuggestions = useMemo(
    () => summary?.sections?.top_counties?.map((item) => item.key).filter(Boolean) ?? [],
    [summary],
  );

  const visibleLeads = leads;
  const mergedGeometryResponse = useMemo(() => {
    if (!selectedGeometryResponse?.feature_collection?.features?.length) return geometryResponse;
    const viewportFeatures = geometryResponse?.feature_collection?.features ?? [];
    const selectedFeatures = selectedGeometryResponse.feature_collection.features;
    const selectedIds = new Set(selectedFeatures.map((feature) => feature.properties.parcel_row_id));
    const mergedFeatures = [
      ...selectedFeatures,
      ...viewportFeatures.filter((feature) => !selectedIds.has(feature.properties.parcel_row_id)),
    ];
    return {
      ...(geometryResponse ?? selectedGeometryResponse),
      render_mode:
        mergedFeatures.some((feature) => feature.geometry?.type === "Polygon" || feature.geometry?.type === "MultiPolygon")
          ? "polygons"
          : (geometryResponse?.render_mode ?? selectedGeometryResponse.render_mode),
      feature_count: mergedFeatures.length,
      feature_collection: {
        type: "FeatureCollection" as const,
        features: mergedFeatures,
      },
      items: [
        ...(selectedGeometryResponse.items ?? []),
        ...((geometryResponse?.items ?? []).filter((item) => !selectedIds.has(item.parcel_row_id))),
      ],
    } satisfies GeometryResponse;
  }, [geometryResponse, selectedGeometryResponse]);
  const visibleVacantCount = useMemo(
    () => visibleLeads.reduce((count, lead) => count + (lead.parcel_vacant_flag ? 1 : 0), 0),
    [visibleLeads],
  );
  const currentPage = Math.floor(offset / limit) + 1;
  const pageCount = Math.max(1, Math.ceil(totalCount / limit));

  function updateFilter<K extends keyof Filters>(key: K, value: Filters[K]) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  function handlePreset(name: string) {
    setActivePreset(name);
    setFilters(applyPreset(name));
    setOffset(0);
    setFitNonce((current) => current + 1);
  }

  function handleSelectParcel(parcelRowId: string) {
    if (process.env.NODE_ENV !== "production") {
      console.debug("[lead-explorer] row-or-map selection", { parcelRowId });
    }
    setSelectedId(parcelRowId);
    setFitNonce((current) => current + 1);
  }

  function toggleOverlay(overlayId: MapOverlayId, enabled: boolean) {
    if (!enabled) return;
    setActiveOverlays((current) =>
      current.includes(overlayId) ? current.filter((value) => value !== overlayId) : [...current, overlayId],
    );
  }

  function handleSort(nextField: SortField) {
    if (sortField === nextField) {
      setSortDirection((current) => (current === "desc" ? "asc" : "desc"));
      return;
    }
    setSortField(nextField);
    setSortDirection("desc");
  }

  function sortIndicator(field: SortField) {
    if (sortField !== field) return "";
    return sortDirection === "desc" ? " ↓" : " ↑";
  }

  return (
    <div className="explorer-shell">
      <aside className="filters-panel">
        <div className="panel-header">
          <p className="eyebrow">LandIntel</p>
          <h1>Parcel Intelligence Platform</h1>
          <p className="muted">
            Evaluate parcel characteristics, owner motivation, development potential, physical constraints, risk, and acquisition attractiveness.
          </p>
        </div>

        <section className="panel-section">
          <h2>Default Views</h2>
          <div className="preset-grid">
            {presets.map((preset) => (
              <button
                key={preset.view_name}
                className={`preset-card ${activePreset === preset.view_name ? "is-active" : ""}`}
                onClick={() => handlePreset(preset.view_name)}
                type="button"
              >
                <strong>{PRESET_LABELS[preset.view_name] ?? preset.view_name}</strong>
                <p className="field-note">{preset.description}</p>
                <div className="inline-badges">
                  <LeadBadge label={`${preset.row_count ?? "-"} leads`} tone="good" />
                  <LeadBadge label={`avg ${preset.average_lead_score ?? "-"}`} tone="neutral" />
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="panel-section">
          <h2>Dataset</h2>
          <div className="inline-badges">
            <LeadBadge label="Dataset: Mississippi" tone="neutral" />
            <LeadBadge label="Platform-ready" tone="good" />
          </div>
          <label>
            County name
            <input
              list="county-suggestions"
              placeholder="all counties"
              value={filters.countyName === "all" ? "" : filters.countyName}
              onChange={(event) => {
                setActivePreset(null);
                updateFilter("countyName", event.target.value.trim() === "" ? "all" : event.target.value.trim());
              }}
            />
            <datalist id="county-suggestions">
              {countySuggestions.map((county) => (
                <option key={county} value={county} />
              ))}
            </datalist>
          </label>

          <label>
            Recommended view bucket
            <select
              value={filters.recommendedViewBucket}
              onChange={(event) => {
                setActivePreset(null);
                updateFilter("recommendedViewBucket", event.target.value);
              }}
            >
              <option value="all">All</option>
              {summary?.sections?.recommended_view_bucket?.map((item) => (
                <option key={item.key} value={item.key}>
                  {item.key}
                </option>
              ))}
            </select>
          </label>
        </section>

        <section className="panel-section">
          <h2>Acquisition Score</h2>
          <label>
            Minimum lead score
            <input
              type="range"
              min="40"
              max="100"
              step="1"
              value={filters.minLeadScore}
              onChange={(event) => {
                setActivePreset(null);
                updateFilter("minLeadScore", Number(event.target.value));
              }}
            />
            <span className="range-value">{filters.minLeadScore}</span>
          </label>

          <fieldset>
            <legend>Lead score tier</legend>
            <div className="chip-grid">
              {["very_high", "high", "medium", "low"].map((tier) => (
                <button
                  key={tier}
                  type="button"
                  className={`chip ${filters.leadScoreTier.includes(tier) ? "is-selected" : ""}`}
                  onClick={() => {
                    setActivePreset(null);
                    updateFilter("leadScoreTier", toggleSelection(filters.leadScoreTier, tier));
                  }}
                >
                  {tier}
                </button>
              ))}
            </div>
          </fieldset>
        </section>

        <section className="panel-section">
          <h2>Parcel Basics</h2>
          <div className="field-row">
            <label>
              Acreage min
              <input
                type="number"
                min="0"
                value={filters.acreageMin}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("acreageMin", event.target.value);
                }}
              />
            </label>
            <label>
              Acreage max
              <input
                type="number"
                min="0"
                value={filters.acreageMax}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("acreageMax", event.target.value);
                }}
              />
            </label>
          </div>
        </section>

        <section className="panel-section">
          <h2>Ownership</h2>
          <div className="checkbox-grid">
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={filters.corporateOnly}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("corporateOnly", event.target.checked);
                }}
              />
              Corporate only
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={filters.absenteeOnly}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("absenteeOnly", event.target.checked);
                }}
              />
              Absentee only
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={filters.outOfStateOnly}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("outOfStateOnly", event.target.checked);
                }}
              />
              Out-of-state only
            </label>
          </div>
        </section>

        <section className="panel-section">
          <h2>Motivation Signals</h2>
          <div className="checkbox-grid">
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={filters.parcelVacantOnly}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("parcelVacantOnly", event.target.checked);
                }}
              />
              Vacant parcels only
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={filters.countyHostedOnly}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("countyHostedOnly", event.target.checked);
                }}
              />
              County-hosted motivation source
            </label>
            <label className="checkbox-item">
              <input
                type="checkbox"
                checked={filters.highConfidenceOnly}
                onChange={(event) => {
                  setActivePreset(null);
                  updateFilter("highConfidenceOnly", event.target.checked);
                }}
              />
              High-confidence parcel linkage
            </label>
          </div>
        </section>

        <section className="panel-section">
          <h2>Flood / Wetlands</h2>
          <label>
            Wetland filter
            <select
              value={filters.wetlandMode}
              onChange={(event) => {
                setActivePreset(null);
                updateFilter("wetlandMode", event.target.value as Filters["wetlandMode"]);
              }}
            >
              <option value="any">Any</option>
              <option value="exclude">Exclude wetlands</option>
              <option value="only">Wetlands only</option>
            </select>
          </label>
        </section>

        <section className="panel-section">
          <h2>Physical Constraints</h2>
          <label>
            Road distance max (ft)
            <input
              type="number"
              min="0"
              value={filters.roadDistanceMax}
              onChange={(event) => {
                setActivePreset(null);
                updateFilter("roadDistanceMax", event.target.value);
              }}
            />
          </label>

          <fieldset>
            <legend>Road access tier</legend>
            <div className="chip-grid">
              {["direct", "near", "moderate", "limited", "remote"].map((tier) => (
                <button
                  key={tier}
                  type="button"
                  className={`chip ${filters.roadAccessTiers.includes(tier) ? "is-selected" : ""}`}
                  onClick={() => {
                    setActivePreset(null);
                    updateFilter("roadAccessTiers", toggleSelection(filters.roadAccessTiers, tier));
                  }}
                >
                  {tier}
                </button>
              ))}
            </div>
          </fieldset>
        </section>

        <section className="panel-section">
          <h2>Market Context</h2>
          <fieldset>
            <legend>Growth pressure</legend>
            <div className="chip-grid">
              {["very_low", "low", "moderate", "high"].map((bucket) => (
                <button
                  key={bucket}
                  type="button"
                  className={`chip ${filters.growthPressureBuckets.includes(bucket) ? "is-selected" : ""}`}
                  onClick={() => {
                    setActivePreset(null);
                    updateFilter("growthPressureBuckets", toggleSelection(filters.growthPressureBuckets, bucket));
                  }}
                >
                  {bucket}
                </button>
              ))}
            </div>
          </fieldset>
        </section>

        <section className="panel-section">
          <h2>Utilities</h2>
          <div className="field-note">
            Electric provider is available in the parcel detail panel as partial context only. No numeric power-distance
            filter exists in the current API.
          </div>
        </section>

        <section className="panel-section">
          <h2>Source / Amount Quality</h2>
          <fieldset>
            <legend>Amount trust tier</legend>
            <div className="chip-grid">
              {["trusted", "use_with_caution", "not_trusted_for_prominent_display"].map((tier) => (
                <button
                  key={tier}
                  type="button"
                  className={`chip ${filters.amountTrustTiers.includes(tier) ? "is-selected" : ""}`}
                  onClick={() => {
                    setActivePreset(null);
                    updateFilter("amountTrustTiers", toggleSelection(filters.amountTrustTiers, tier));
                  }}
                >
                  {tier}
                </button>
              ))}
            </div>
          </fieldset>
        </section>

        <section className="panel-section">
          <h2>Map Layers</h2>
          <div className="overlay-list">
            {MAP_OVERLAYS.map((overlay) => (
              <button
                key={overlay.id}
                type="button"
                className={`overlay-toggle ${activeOverlays.includes(overlay.id) ? "is-selected" : ""} ${!overlay.enabled ? "is-disabled" : ""}`}
                onClick={() => toggleOverlay(overlay.id, overlay.enabled)}
                disabled={!overlay.enabled}
              >
                <strong>{overlay.label}</strong>
                <span>{overlay.description}</span>
              </button>
            ))}
          </div>
        </section>

        <section className="panel-section">
          <h2>Summary</h2>
          {summaryLoading ? <p className="muted">Loading summary...</p> : null}
          {summaryError ? <p className="error-text">{summaryError}</p> : null}
          <div className="stats-grid">
            <div>
              <span className="stat-label">Statewide parcels</span>
              <strong><SummaryValue summary={summary} section="statewide" metric="lead_count" /></strong>
            </div>
            <div>
              <span className="stat-label">Likely vacant parcels</span>
              <strong>
                <SummaryValue summary={summary} section="statewide" metric="likely_vacant_count" />
              </strong>
            </div>
            <div>
              <span className="stat-label">Average score</span>
              <strong>
                <SummaryValue summary={summary} section="statewide" metric="average_lead_score" />
              </strong>
            </div>
            <div>
              <span className="stat-label">Vacant share</span>
              <strong>
                <SummaryValue summary={summary} section="statewide" metric="vacant_share_pct" />%
              </strong>
            </div>
            <div>
              <span className="stat-label">County-hosted share</span>
              <strong>
                <SummaryValue summary={summary} section="statewide" metric="county_hosted_share_pct" />%
              </strong>
            </div>
          </div>
        </section>
      </aside>

      <main className="explorer-main">
        <section className="hero-strip">
          <div className="hero-copy">
            <p className="eyebrow">LandIntel</p>
            <h2>Parcel Intelligence Platform</h2>
            <p className="muted">
              LandIntel helps evaluate parcel characteristics, ownership, motivation signals, physical constraints, utilities, flood and wetland context, market pressure, risk, and acquisition score.
            </p>
          </div>
          <div className="hero-badges">
            <LeadBadge label={`Dataset: Mississippi`} tone="neutral" />
            <LeadBadge label={`${totalCount.toLocaleString()} parcels`} tone="good" />
            <LeadBadge label={`${visibleLeads.length} loaded`} tone="neutral" />
            <LeadBadge label={`${visibleVacantCount.toLocaleString()} likely vacant in view`} tone="warn" />
            <LeadBadge label={geometryLoading ? "Geometry loading" : "Geometry ready"} tone={geometryLoading ? "warn" : "good"} />
          </div>
        </section>

        <div className="content-grid">
          <section className="map-card">
            <div className="card-header">
              <div className="card-header-copy">
                <h3>Parcel Map</h3>
                <p>Statewide parcel exploration with overlay-ready GIS layers. Use filters to narrow parcels, then inspect geometry and details in place.</p>
              </div>
              <div className="card-header-actions">
                <button type="button" className="chip" onClick={() => setFitNonce((current) => current + 1)}>
                  Zoom to results
                </button>
                <button type="button" className="chip" disabled={!selectedId} onClick={() => setLocateSelectedNonce((current) => current + 1)}>
                  Locate selected parcel
                </button>
                {mergedGeometryResponse?.feature_count ? <LeadBadge label={`${mergedGeometryResponse.feature_count} map features`} tone="neutral" /> : null}
              </div>
            </div>
            {geometryError ? <p className="error-text">{geometryError}</p> : null}
            <LeadMap
              geometryResponse={mergedGeometryResponse}
              selectedId={selectedId}
              onSelect={handleSelectParcel}
              fitNonce={fitNonce}
              locateSelectedNonce={locateSelectedNonce}
              activeOverlays={activeOverlays}
              viewport={viewport}
              onViewportChange={setViewport}
              loading={geometryLoading}
              error={geometryError}
              totalCount={totalCount}
            />
            <div className="map-legend">
              <span><i className="legend-swatch legend-base" /> Parcel</span>
              <span><i className="legend-swatch legend-selected" /> Selected</span>
              <span><i className="legend-swatch legend-wetland" /> Wetland overlay</span>
              <span><i className="legend-swatch legend-road" /> Strong road access</span>
              <span><i className="legend-swatch legend-flood" /> Flood signal</span>
            </div>
          </section>

          <section className="list-card">
            <div className="card-header">
              <div className="card-header-copy">
                <h3>Parcel Results</h3>
                <p>Page {currentPage} of {pageCount}. Filters narrow the full statewide parcel base; motivation and risk signals layer on top.</p>
              </div>
              <div className="inline-badges">
                <button
                  type="button"
                  className="chip"
                  disabled={offset === 0}
                  onClick={() => setOffset((current) => Math.max(0, current - limit))}
                >
                  Previous
                </button>
                <button
                  type="button"
                  className="chip"
                  disabled={offset + limit >= totalCount}
                  onClick={() => setOffset((current) => current + limit)}
                >
                  Next
                </button>
              </div>
            </div>
            {leadsLoading ? <p className="muted">Loading leads...</p> : null}
            {leadsError ? <p className="error-text">{leadsError}</p> : null}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {TABLE_COLUMNS.map((column) => (
                      <HeaderCell
                        key={column.key}
                        label={column.label}
                        tooltip={column.tooltip}
                        onSort={column.sortField ? (() => handleSort(column.sortField!)) : undefined}
                        indicator={column.sortField ? sortIndicator(column.sortField!) : ""}
                      />
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {visibleLeads.map((lead) => (
                    <tr
                      key={lead.parcel_row_id}
                      className={selectedId === lead.parcel_row_id ? "is-selected" : ""}
                      onClick={() => handleSelectParcel(lead.parcel_row_id)}
                    >
                      <td>{lead.county_name ?? "-"}</td>
                      <td>{lead.parcel_id ?? lead.parcel_row_id}</td>
                      <td>{formatNumber(lead.acreage, 2)}</td>
                      <td>{lead.owner_name ?? "-"}</td>
                      <td>{formatNumber(lead.lead_score_total, 1)}</td>
                      <td>
                        <LeadBadge label={lead.lead_score_tier ?? "-"} tone="good" />
                      </td>
                      <td>{lead.parcel_vacant_flag ? "Yes" : "No"}</td>
                      <td>{lead.road_access_tier ?? "-"}</td>
                      <td>{lead.growth_pressure_bucket ?? "-"}</td>
                      <td>
                        <LeadBadge label={lead.county_hosted_flag ? "County" : "SOS"} tone={lead.county_hosted_flag ? "good" : "warn"} />
                      </td>
                      <td>
                        <LeadBadge label={lead.source_confidence_tier ?? "-"} tone={badgeTone(lead.source_confidence_tier)} />
                      </td>
                      <td>{formatCurrency(lead.delinquent_amount)}</td>
                      <td>
                        <LeadBadge label={lead.amount_trust_tier ?? "-"} tone={badgeTone(lead.amount_trust_tier)} />
                      </td>
                      <td>{lead.recommended_sort_reason ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>

      <aside className="detail-panel">
        <div className="panel-header">
          <p className="eyebrow">Parcel detail</p>
          <h1>Selection</h1>
          <p className="muted">Parcel details load from the API when available and fall back to the packaged explorer detail dataset if the backend detail request fails.</p>
        </div>
        {detailLoading ? <p className="muted">Loading parcel detail...</p> : null}
        {detailError ? <p className="error-text">{detailError}</p> : null}
        {!detailLoading && !selectedLead ? <p className="empty-detail">Select a parcel to inspect the full lead payload.</p> : null}
        {selectedLead ? <LeadDetail lead={selectedLead} /> : null}
      </aside>
    </div>
  );
}
