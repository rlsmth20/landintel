"use client";

import type { LeadRecord } from "./types";

export function LeadMap({
  leads,
  geometryMap,
  geometryViewBox,
  selectedId,
  onSelect,
}: {
  leads: LeadRecord[];
  geometryMap: Record<string, string>;
  geometryViewBox?: number[];
  selectedId: string | null;
  onSelect: (value: string) => void;
}) {
  const viewBox = geometryViewBox?.length === 4 ? geometryViewBox.join(" ") : "0 0 1000 700";

  return (
    <svg className="lead-map" viewBox={viewBox} role="img" aria-label="Mississippi parcel lead map">
      <rect x="0" y="0" width="100%" height="100%" rx="26" className="map-bg" />
      {Array.from({ length: 6 }).map((_, index) => (
        <line key={`v-${index}`} x1={30 + index * 155} x2={30 + index * 155} y1={24} y2={676} className="map-grid" />
      ))}
      {Array.from({ length: 4 }).map((_, index) => (
        <line key={`h-${index}`} x1={24} x2={976} y1={32 + index * 160} y2={32 + index * 160} className="map-grid" />
      ))}
      {leads.map((lead) => {
        const path = geometryMap[lead.parcel_row_id];
        const selected = selectedId === lead.parcel_row_id;
        if (!path) return null;
        return (
          <path
            key={lead.parcel_row_id}
            d={path}
            className={selected ? "parcel-shape is-selected" : "parcel-shape"}
            onClick={() => onSelect(lead.parcel_row_id)}
          />
        );
      })}
    </svg>
  );
}
