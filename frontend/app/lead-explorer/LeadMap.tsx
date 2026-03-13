"use client";

import { useEffect, useMemo, useState } from "react";

import type { LeadRecord, MapOverlayId } from "./types";

type Bounds = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

const DEFAULT_VIEW_BOX: [number, number, number, number] = [0, 0, 1000, 700];

function parsePathBounds(path: string): Bounds | null {
  const numbers = path.match(/-?\d*\.?\d+/g)?.map(Number) ?? [];
  if (numbers.length < 2) return null;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < numbers.length - 1; index += 2) {
    const x = numbers[index];
    const y = numbers[index + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  if (![minX, minY, maxX, maxY].every(Number.isFinite)) return null;
  return { minX, minY, maxX, maxY };
}

function combineBounds(boundsList: Bounds[]): Bounds | null {
  if (boundsList.length === 0) return null;
  return boundsList.reduce(
    (combined, bounds) => ({
      minX: Math.min(combined.minX, bounds.minX),
      minY: Math.min(combined.minY, bounds.minY),
      maxX: Math.max(combined.maxX, bounds.maxX),
      maxY: Math.max(combined.maxY, bounds.maxY),
    }),
    boundsList[0],
  );
}

function paddedViewBox(bounds: Bounds, paddingFactor: number): [number, number, number, number] | null {
  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null;

  const safeWidth = Math.max(width, 1.5);
  const safeHeight = Math.max(height, 1.5);
  const paddingX = Math.max(safeWidth * paddingFactor, 1.25);
  const paddingY = Math.max(safeHeight * paddingFactor, 1.25);

  return [
    bounds.minX - paddingX,
    bounds.minY - paddingY,
    safeWidth + paddingX * 2,
    safeHeight + paddingY * 2,
  ];
}

export function LeadMap({
  leads,
  geometryMap,
  selectedId,
  onSelect,
  zoomNonce,
  activeOverlays,
  onZoomStateChange,
}: {
  leads: LeadRecord[];
  geometryMap: Record<string, string>;
  selectedId: string | null;
  onSelect: (value: string) => void;
  zoomNonce: number;
  activeOverlays: MapOverlayId[];
  onZoomStateChange?: (value: { featureCount: number; bounds: [number, number, number, number] | null }) => void;
}) {
  const [currentViewBox, setCurrentViewBox] = useState<[number, number, number, number]>(DEFAULT_VIEW_BOX);

  const pathEntries = useMemo(() => {
    return leads
      .map((lead) => ({
        lead,
        path: geometryMap[lead.parcel_row_id],
      }))
      .filter((entry) => Boolean(entry.path));
  }, [geometryMap, leads]);

  const pathBounds = useMemo(() => {
    return Object.fromEntries(
      pathEntries
        .map((entry) => {
          const bounds = parsePathBounds(entry.path as string);
          return bounds ? [entry.lead.parcel_row_id, bounds] : null;
        })
        .filter((entry): entry is [string, Bounds] => entry !== null),
    );
  }, [pathEntries]);

  const resultsBounds = useMemo(() => combineBounds(Object.values(pathBounds)), [pathBounds]);
  const selectedBounds = selectedId ? pathBounds[selectedId] ?? null : null;

  const overlaySet = useMemo(() => new Set(activeOverlays), [activeOverlays]);

  useEffect(() => {
    const featureCount = Object.keys(pathBounds).length;
    const targetBounds = selectedBounds ?? resultsBounds;
    const nextViewBox = targetBounds
      ? paddedViewBox(targetBounds, selectedBounds ? 0.24 : 0.12)
      : null;

    if (process.env.NODE_ENV !== "production") {
      console.debug("[lead-map] feature_count", featureCount);
      console.debug("[lead-map] selected_parcel_id", selectedId);
      console.debug("[lead-map] computed_bounds", nextViewBox);
      if (!nextViewBox && featureCount > 0) {
        const firstEntry = pathEntries[0];
        console.debug("[lead-map] invalid_bounds_first_path", firstEntry?.lead.parcel_row_id, firstEntry?.path?.slice(0, 160));
      }
    }

    if (nextViewBox) {
      setCurrentViewBox(nextViewBox);
      onZoomStateChange?.({ featureCount, bounds: nextViewBox });
      return;
    }

    if (featureCount === 0) {
      setCurrentViewBox(DEFAULT_VIEW_BOX);
      onZoomStateChange?.({ featureCount: 0, bounds: null });
    }
  }, [onZoomStateChange, pathBounds, pathEntries, resultsBounds, selectedBounds, selectedId, zoomNonce]);

  if (pathEntries.length === 0) {
    return (
      <div className="map-empty-state">
        <strong>No parcel geometry loaded</strong>
        <p>Adjust filters or paging to load a result subset with parcel polygons.</p>
      </div>
    );
  }

  return (
    <svg
      className="lead-map"
      viewBox={currentViewBox.join(" ")}
      role="img"
      aria-label="Mississippi parcel lead map"
      preserveAspectRatio="xMidYMid meet"
    >
      <rect x={currentViewBox[0]} y={currentViewBox[1]} width={currentViewBox[2]} height={currentViewBox[3]} rx="26" className="map-bg" />
      {Array.from({ length: 5 }).map((_, index) => {
        const x = currentViewBox[0] + (currentViewBox[2] / 4) * index;
        return <line key={`v-${index}`} x1={x} x2={x} y1={currentViewBox[1]} y2={currentViewBox[1] + currentViewBox[3]} className="map-grid" />;
      })}
      {Array.from({ length: 5 }).map((_, index) => {
        const y = currentViewBox[1] + (currentViewBox[3] / 4) * index;
        return <line key={`h-${index}`} x1={currentViewBox[0]} x2={currentViewBox[0] + currentViewBox[2]} y1={y} y2={y} className="map-grid" />;
      })}
      {pathEntries.map(({ lead, path }) => {
        const selected = selectedId === lead.parcel_row_id;
        const overlayClasses = [
          overlaySet.has("wetlands") && lead.wetland_flag ? "has-wetland-overlay" : "",
          overlaySet.has("road_access") && lead.road_access_tier === "direct" ? "has-road-overlay" : "",
          overlaySet.has("fema_flood") && (lead.flood_risk_score ?? 0) > 0 ? "has-flood-overlay" : "",
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <path
            key={lead.parcel_row_id}
            d={path}
            className={selected ? `parcel-shape is-selected ${overlayClasses}`.trim() : `parcel-shape ${overlayClasses}`.trim()}
            onClick={() => onSelect(lead.parcel_row_id)}
          />
        );
      })}
    </svg>
  );
}
