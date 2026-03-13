"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { useEffect, useMemo, useRef } from "react";
import maplibregl, { GeoJSONSource, LngLatBoundsLike, Map } from "maplibre-gl";
import { PMTiles, Protocol } from "pmtiles";

import type { FeatureCollectionPayload, GeometryFeature, GeometryResponse, MapOverlayId, MapViewportState } from "./types";

const DEFAULT_CENTER: [number, number] = [-98.5795, 39.8283];
const DEFAULT_ZOOM = 3.4;
const MISSISSIPPI_BOUNDS: [[number, number], [number, number]] = [
  [-91.65, 30.15],
  [-88.0, 35.1],
];
const PARCEL_TILE_SOURCE_ID = "landintel-parcel-tiles";
const PARCEL_TILE_LAYER = "parcels";
const SELECTED_PARCEL_SOURCE_ID = "landintel-selected-parcel";
const PARCEL_TILE_MIN_ZOOM = 6;
const DEFAULT_PARCEL_PMTILES_URL = "/tiles/mississippi_parcels.pmtiles";
let pmtilesProtocol: Protocol | null = null;
let pmtilesArchiveUrl: string | null = null;

function getParcelPmtilesUrl() {
  const configured = process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL?.trim();
  const candidate = configured && configured.length > 0 ? configured : DEFAULT_PARCEL_PMTILES_URL;
  if (/^https?:\/\//i.test(candidate)) return candidate;
  if (typeof window !== "undefined") {
    return new URL(candidate, window.location.origin).toString();
  }
  return candidate;
}

function ensurePmtilesProtocol() {
  if (typeof window === "undefined") return;
  if (!pmtilesProtocol) {
    pmtilesProtocol = new Protocol();
    maplibregl.addProtocol("pmtiles", pmtilesProtocol.tile);
  }
  const nextUrl = getParcelPmtilesUrl();
  if (pmtilesArchiveUrl !== nextUrl) {
    pmtilesProtocol.add(new PMTiles(nextUrl));
    pmtilesArchiveUrl = nextUrl;
  }
}

const BASE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "&copy; OpenStreetMap contributors",
    },
  },
  glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
  layers: [
    {
      id: "osm",
      type: "raster",
      source: "osm",
    },
  ],
};

function featureBounds(feature: GeometryFeature): [number, number, number, number] | null {
  const coordinates = feature.geometry?.coordinates;
  if (!coordinates) return null;

  let minLng = Number.POSITIVE_INFINITY;
  let minLat = Number.POSITIVE_INFINITY;
  let maxLng = Number.NEGATIVE_INFINITY;
  let maxLat = Number.NEGATIVE_INFINITY;

  function walk(value: unknown) {
    if (!Array.isArray(value)) return;
    if (value.length >= 2 && typeof value[0] === "number" && typeof value[1] === "number") {
      const lng = value[0];
      const lat = value[1];
      if (!Number.isFinite(lng) || !Number.isFinite(lat)) return;
      minLng = Math.min(minLng, lng);
      minLat = Math.min(minLat, lat);
      maxLng = Math.max(maxLng, lng);
      maxLat = Math.max(maxLat, lat);
      return;
    }
    value.forEach(walk);
  }

  walk(coordinates);
  if (![minLng, minLat, maxLng, maxLat].every(Number.isFinite)) return null;
  return [minLng, minLat, maxLng, maxLat];
}

function mergeBounds(boundsList: Array<[number, number, number, number]>): [number, number, number, number] | null {
  if (boundsList.length === 0) return null;
  return boundsList.reduce(
    (accumulator, bounds) => [
      Math.min(accumulator[0], bounds[0]),
      Math.min(accumulator[1], bounds[1]),
      Math.max(accumulator[2], bounds[2]),
      Math.max(accumulator[3], bounds[3]),
    ],
    boundsList[0],
  );
}

function toMapBounds(bounds: [number, number, number, number]): LngLatBoundsLike {
  return [
    [bounds[0], bounds[1]],
    [bounds[2], bounds[3]],
  ];
}

function selectedFeatureBounds(featureCollection: FeatureCollectionPayload | undefined, selectedId: string | null) {
  if (!featureCollection || !selectedId) return null;
  const match = featureCollection.features.find((feature) => feature.properties.parcel_row_id === selectedId);
  return match ? featureBounds(match) : null;
}

function updateLayerVisibility(map: Map, layerId: string, visible: boolean) {
  if (!map.getLayer(layerId)) return;
  map.setLayoutProperty(layerId, "visibility", visible ? "visible" : "none");
}

function initializeParcelLayers(map: Map) {
  if (map.getSource(PARCEL_TILE_SOURCE_ID)) return;
  ensurePmtilesProtocol();

  map.addSource(PARCEL_TILE_SOURCE_ID, {
    type: "vector",
    url: `pmtiles://${getParcelPmtilesUrl()}`,
    minzoom: PARCEL_TILE_MIN_ZOOM,
    maxzoom: 15,
    promoteId: { [PARCEL_TILE_LAYER]: "parcel_row_id" },
  });

  map.addSource(SELECTED_PARCEL_SOURCE_ID, {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: [],
    },
  });

  map.addLayer({
    id: "parcel-fills",
    type: "fill",
    source: PARCEL_TILE_SOURCE_ID,
    "source-layer": PARCEL_TILE_LAYER,
    paint: {
      "fill-color": "#2f6b6d",
      "fill-opacity": 0.18,
    },
  });

  map.addLayer({
    id: "parcel-lines",
    type: "line",
    source: PARCEL_TILE_SOURCE_ID,
    "source-layer": PARCEL_TILE_LAYER,
    paint: {
      "line-color": "#17393a",
      "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.4, 10, 0.8, 13, 1.2, 15, 1.6],
      "line-opacity": 0.88,
    },
  });

  map.addLayer({
    id: "parcel-wetlands-overlay",
    type: "line",
    source: PARCEL_TILE_SOURCE_ID,
    "source-layer": PARCEL_TILE_LAYER,
    filter: ["==", ["get", "wetland_flag"], true],
    paint: {
      "line-color": "#617f56",
      "line-width": 2.2,
      "line-opacity": 0.95,
    },
  });

  map.addLayer({
    id: "parcel-road-overlay",
    type: "line",
    source: PARCEL_TILE_SOURCE_ID,
    "source-layer": PARCEL_TILE_LAYER,
    filter: ["==", ["get", "road_access_tier"], "direct"],
    paint: {
      "line-color": "#1f7f80",
      "line-width": 2.4,
      "line-opacity": 0.95,
    },
  });

  map.addLayer({
    id: "parcel-flood-overlay",
    type: "line",
    source: PARCEL_TILE_SOURCE_ID,
    "source-layer": PARCEL_TILE_LAYER,
    filter: [">", ["coalesce", ["get", "flood_risk_score"], 0], 0],
    paint: {
      "line-color": "#5f8db8",
      "line-width": 2.4,
      "line-opacity": 0.95,
    },
  });

  map.addLayer({
    id: "selected-parcel-fill",
    type: "fill",
    source: SELECTED_PARCEL_SOURCE_ID,
    paint: {
      "fill-color": "#d9472f",
      "fill-opacity": 0.44,
    },
  });

  map.addLayer({
    id: "selected-parcel-line",
    type: "line",
    source: SELECTED_PARCEL_SOURCE_ID,
    paint: {
      "line-color": "#fff8ee",
      "line-width": 3.2,
      "line-opacity": 1,
    },
  });

  map.addLayer({
    id: "parcel-hover",
    type: "line",
    source: PARCEL_TILE_SOURCE_ID,
    "source-layer": PARCEL_TILE_LAYER,
    paint: {
      "line-color": "#ffe5cf",
      "line-width": ["case", ["boolean", ["feature-state", "hover"], false], 3.2, 0],
      "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 1, 0],
    },
  });
}

export function LeadMap({
  geometryResponse,
  selectedId,
  onSelect,
  fitNonce,
  locateSelectedNonce,
  activeOverlays,
  viewport,
  onViewportChange,
  resultsLoading,
  loading,
  error,
  totalCount,
}: {
  geometryResponse: GeometryResponse | null;
  selectedId: string | null;
  onSelect: (value: string) => void;
  fitNonce: number;
  locateSelectedNonce: number;
  activeOverlays: MapOverlayId[];
  viewport: MapViewportState;
  onViewportChange: (value: MapViewportState) => void;
  resultsLoading: boolean;
  loading: boolean;
  error: string | null;
  totalCount: number;
}) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const hoveredFeatureIdRef = useRef<string | null>(null);
  const hasInitializedViewportRef = useRef(false);
  const lastAppliedFitNonceRef = useRef<number>(-1);
  const lastLocateSelectedNonceRef = useRef<number>(-1);
  const lastSelectedIdRef = useRef<string | null>(null);

  const featureCollection = geometryResponse?.feature_collection;
  const featureCount = featureCollection?.features.length ?? 0;
  const boundsList = useMemo(
    () => (featureCollection?.features ?? []).map(featureBounds).filter((value): value is [number, number, number, number] => value !== null),
    [featureCollection],
  );
  const resultBounds = useMemo(() => mergeBounds(boundsList), [boundsList]);
  const selectedBounds = useMemo(() => selectedFeatureBounds(featureCollection, selectedId), [featureCollection, selectedId]);

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: BASE_STYLE,
      center: viewport.center ?? DEFAULT_CENTER,
      zoom: viewport.zoom ?? DEFAULT_ZOOM,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    mapRef.current = map;

    map.on("load", () => {
      initializeParcelLayers(map);

      map.on("click", "parcel-fills", (event) => {
        const feature = event.features?.[0];
        const parcelRowId = feature?.properties?.parcel_row_id;
        if (typeof parcelRowId === "string") {
          if (process.env.NODE_ENV !== "production") {
            console.debug("[landintel-map] parcel click", { parcelRowId });
          }
          onSelect(parcelRowId);
        }
      });

      map.on("mouseleave", "parcel-fills", () => {
        if (hoveredFeatureIdRef.current) {
          map.setFeatureState({ source: PARCEL_TILE_SOURCE_ID, sourceLayer: PARCEL_TILE_LAYER, id: hoveredFeatureIdRef.current }, { hover: false });
          hoveredFeatureIdRef.current = null;
        }
        map.getCanvas().style.cursor = "";
      });

      const currentBounds = map.getBounds();
      onViewportChange({
        center: [map.getCenter().lng, map.getCenter().lat],
        zoom: map.getZoom(),
        bounds: [currentBounds.getWest(), currentBounds.getSouth(), currentBounds.getEast(), currentBounds.getNorth()],
      });
    });

    map.fitBounds(MISSISSIPPI_BOUNDS, { padding: 28, duration: 0, maxZoom: 7.2 });

    map.on("moveend", () => {
      const currentBounds = map.getBounds();
      onViewportChange({
        center: [map.getCenter().lng, map.getCenter().lat],
        zoom: map.getZoom(),
        bounds: [currentBounds.getWest(), currentBounds.getSouth(), currentBounds.getEast(), currentBounds.getNorth()],
      });
    });

    map.on("mousemove", (event) => {
      if (!map.getLayer("parcel-fills")) return;
      const parcelFeature = map.queryRenderedFeatures(event.point, {
        layers: ["parcel-fills"],
      })[0];
      const nextId = parcelFeature?.properties?.parcel_row_id;
      if (hoveredFeatureIdRef.current && hoveredFeatureIdRef.current !== nextId) {
        map.setFeatureState({ source: PARCEL_TILE_SOURCE_ID, sourceLayer: PARCEL_TILE_LAYER, id: hoveredFeatureIdRef.current }, { hover: false });
      }
      if (typeof nextId === "string") {
        hoveredFeatureIdRef.current = nextId;
        map.setFeatureState({ source: PARCEL_TILE_SOURCE_ID, sourceLayer: PARCEL_TILE_LAYER, id: nextId }, { hover: true });
        map.getCanvas().style.cursor = "pointer";
      } else {
        hoveredFeatureIdRef.current = null;
        map.getCanvas().style.cursor = "";
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [onSelect, onViewportChange]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const source = map.getSource(SELECTED_PARCEL_SOURCE_ID) as GeoJSONSource | undefined;
    if (!source) return;

    const nextCollection: FeatureCollectionPayload = {
      type: "FeatureCollection",
      features:
        (featureCollection?.features ?? []).map((feature) => ({
          ...feature,
          id: feature.properties.parcel_row_id,
          properties: {
            ...feature.properties,
            selected: feature.properties.parcel_row_id === selectedId,
          },
        })) ?? [],
    };
    source.setData(nextCollection as never);

    if (process.env.NODE_ENV !== "production") {
      console.debug("[landintel-map] feature_count_loaded", nextCollection.features.length);
      console.debug("[landintel-map] computed_map_bounds", geometryResponse?.geometry_bounds ?? resultBounds);
      console.debug("[landintel-map] selected_parcel_id", selectedId);
      if (!resultBounds && nextCollection.features.length > 0) {
        console.debug("[landintel-map] invalid_geometry_first_feature", nextCollection.features[0]);
      }
    }
  }, [featureCollection, geometryResponse?.geometry_bounds, resultBounds, selectedId]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    updateLayerVisibility(map, "parcel-fills", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-lines", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-hover", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "selected-parcel-fill", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "selected-parcel-line", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-wetlands-overlay", activeOverlays.includes("wetlands"));
    updateLayerVisibility(map, "parcel-road-overlay", activeOverlays.includes("road_access"));
    updateLayerVisibility(map, "parcel-flood-overlay", activeOverlays.includes("fema_flood"));
  }, [activeOverlays]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || featureCount === 0) return;

    const hasNewFitRequest = fitNonce !== lastAppliedFitNonceRef.current;
    const hasSelectionChange = selectedId !== lastSelectedIdRef.current && Boolean(selectedId);
    const hasLocateSelectedRequest = locateSelectedNonce !== lastLocateSelectedNonceRef.current && Boolean(selectedId);
    if (!hasNewFitRequest && !hasSelectionChange && !hasLocateSelectedRequest) {
      return;
    }

    const targetBounds = selectedBounds ?? resultBounds;
    if (!targetBounds) {
      if (hasNewFitRequest && !selectedId) {
        map.fitBounds(MISSISSIPPI_BOUNDS, {
          padding: 28,
          duration: hasInitializedViewportRef.current ? 600 : 0,
          maxZoom: 7.2,
        });
        hasInitializedViewportRef.current = true;
        lastAppliedFitNonceRef.current = fitNonce;
        lastLocateSelectedNonceRef.current = locateSelectedNonce;
        lastSelectedIdRef.current = selectedId;
        return;
      }
      if (process.env.NODE_ENV !== "production") {
        console.debug("[landintel-map] fit_failed_no_bounds", {
          featureCount,
          geometryResponse,
          firstFeature: featureCollection?.features?.[0],
        });
      }
      return;
    }

    const padding = selectedBounds ? 72 : 48;
    try {
      if (process.env.NODE_ENV !== "production") {
        console.debug("[landintel-map] fit_to_bounds", { selectedId, targetBounds, featureCount, zoom: map.getZoom() });
      }
      map.fitBounds(toMapBounds(targetBounds), {
        padding,
        duration: hasInitializedViewportRef.current ? 600 : 0,
        maxZoom: selectedBounds ? 16.5 : 14.5,
      });
      hasInitializedViewportRef.current = true;
      lastAppliedFitNonceRef.current = fitNonce;
      lastLocateSelectedNonceRef.current = locateSelectedNonce;
      lastSelectedIdRef.current = selectedId;
    } catch (error) {
      if (process.env.NODE_ENV !== "production") {
        console.debug("[landintel-map] fit_failed", {
          error,
          featureCount,
          targetBounds,
          firstFeature: featureCollection?.features?.[0],
        });
      }
    }
  }, [featureCount, fitNonce, geometryResponse, locateSelectedNonce, resultBounds, selectedBounds, selectedId]);

  let emptyTitle: string | null = null;
  let emptyBody: string | null = null;
  if (resultsLoading) {
    emptyTitle = "Loading parcel results";
    emptyBody = "Fetching parcel records and filters for the current dataset.";
  } else if (totalCount === 0) {
    emptyTitle = "No parcels match current filters";
    emptyBody = "Try broadening the current filter set or clearing preset constraints.";
  } else if (viewport.zoom < PARCEL_TILE_MIN_ZOOM) {
    emptyTitle = "Zoom in to inspect parcel boundaries";
    emptyBody = "The base parcel layer uses PMTiles and becomes legible once you zoom further into Mississippi.";
  } else if (loading && selectedId) {
    emptyTitle = "Loading selected parcel";
    emptyBody = "Fetching detailed geometry for the current selection.";
  } else if (error && selectedId) {
    emptyTitle = "Selected parcel geometry failed to load";
    emptyBody = error;
  }

  return (
    <div className="lead-map-shell">
      <div className="lead-map-canvas" ref={mapContainerRef} />
      {emptyTitle ? (
        <div className="map-empty-state map-overlay-empty">
          <strong>{emptyTitle}</strong>
          <p>{emptyBody}</p>
        </div>
      ) : null}
    </div>
  );
}
