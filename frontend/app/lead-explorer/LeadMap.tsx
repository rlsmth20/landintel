"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { useEffect, useMemo, useRef } from "react";
import maplibregl, { GeoJSONSource, LngLatBoundsLike, Map } from "maplibre-gl";
import { PMTiles, Protocol } from "pmtiles";

import parcelIdentity from "./parcelIdentity";
import stateConfig from "./stateConfig";
import type { BasemapMode, FeatureCollectionPayload, GeometryFeature, GeometryResponse, LeadRecord, MapOverlayId, MapViewportState } from "./types";

const {
  featureBounds,
  getMapFeatureSelectionParcelRowId,
  mergeBounds,
  selectedFeatureBounds,
} = parcelIdentity;

const DEFAULT_CENTER: [number, number] = [-98.5795, 39.8283];
const DEFAULT_ZOOM = 3.4;
const PARCEL_TILE_SOURCE_ID = "landintel-parcel-tiles";
const PARCEL_TILE_LAYER = "parcels";
const SELECTED_PARCEL_SOURCE_ID = "landintel-selected-parcel";
const PARCEL_TILE_MIN_ZOOM = 6;
let pmtilesProtocol: Protocol | null = null;
let pmtilesArchiveUrl: string | null = null;
const STATE_PM_TILES_ENV_OVERRIDES: Record<string, string | null> = {
  ar: process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_AR?.trim() || null,
  ms: process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL_MS?.trim() || null,
};

function toStateBounds(defaultBounds: [number, number, number, number]): [[number, number], [number, number]] {
  return [
    [defaultBounds[0], defaultBounds[1]],
    [defaultBounds[2], defaultBounds[3]],
  ];
}

function getParcelPmtilesUrl(stateCode: string, configuredPmtilesUrl: string | null | undefined): string | null {
  const stateOverride = STATE_PM_TILES_ENV_OVERRIDES[stateCode]?.trim() || null;
  const configured = process.env.NEXT_PUBLIC_PARCEL_PMTILES_URL?.trim();
  const candidate =
    stateOverride && stateOverride.length > 0
      ? stateOverride
      : configured && configured.length > 0
        ? configured
        : typeof configuredPmtilesUrl === "string" && configuredPmtilesUrl.trim().length > 0
          ? configuredPmtilesUrl
          : null;
  if (!candidate) return null;
  if (/^https?:\/\//i.test(candidate)) return candidate;
  if (typeof window !== "undefined") {
    return new URL(candidate, window.location.origin).toString();
  }
  return candidate;
}

function ensurePmtilesProtocol(stateCode: string, parcelPmtilesUrl: string | null | undefined) {
  const nextUrl = getParcelPmtilesUrl(stateCode, parcelPmtilesUrl);
  if (typeof window === "undefined" || !nextUrl) return null;
  if (!pmtilesProtocol) {
    pmtilesProtocol = new Protocol();
    maplibregl.addProtocol("pmtiles", pmtilesProtocol.tile);
  }
  if (pmtilesArchiveUrl !== nextUrl) {
    pmtilesProtocol.add(new PMTiles(nextUrl));
    pmtilesArchiveUrl = nextUrl;
  }
  return nextUrl;
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
    satellite: {
      type: "raster",
      tiles: ["https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    },
  },
  glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
  layers: [
    {
      id: "street-base",
      type: "raster",
      source: "osm",
    },
    {
      id: "satellite-base",
      type: "raster",
      source: "satellite",
      layout: {
        visibility: "none",
      },
    },
  ],
};

function toMapBounds(bounds: [number, number, number, number]): LngLatBoundsLike {
  return [
    [bounds[0], bounds[1]],
    [bounds[2], bounds[3]],
  ];
}

function updateLayerVisibility(map: Map, layerId: string, visible: boolean) {
  if (!map.getLayer(layerId)) return;
  map.setLayoutProperty(layerId, "visibility", visible ? "visible" : "none");
}

function initializeParcelLayers(map: Map, stateCode: string, parcelPmtilesUrl: string | null | undefined) {
  map.addSource(SELECTED_PARCEL_SOURCE_ID, {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: [],
    },
  });

  const resolvedParcelPmtilesUrl = ensurePmtilesProtocol(stateCode, parcelPmtilesUrl);
  if (resolvedParcelPmtilesUrl && !map.getSource(PARCEL_TILE_SOURCE_ID)) {
    map.addSource(PARCEL_TILE_SOURCE_ID, {
      type: "vector",
      url: `pmtiles://${resolvedParcelPmtilesUrl}`,
      minzoom: PARCEL_TILE_MIN_ZOOM,
      maxzoom: 15,
      promoteId: { [PARCEL_TILE_LAYER]: "parcel_row_id" },
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
  }

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

  if (parcelPmtilesUrl) {
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
}

export function LeadMap({
  stateCode,
  geometryResponse,
  selectedParcelRowId,
  onSelect,
  fitNonce,
  locateSelectedNonce,
  activeOverlays,
  basemapMode,
  viewport,
  onViewportChange,
  resultsLoading,
  loading,
  error,
  totalCount,
}: {
  stateCode: string;
  geometryResponse: GeometryResponse | null;
  selectedParcelRowId: string | null;
  onSelect: (value: string, lead?: LeadRecord | null) => void;
  fitNonce: number;
  locateSelectedNonce: number;
  activeOverlays: MapOverlayId[];
  basemapMode: BasemapMode;
  viewport: MapViewportState;
  onViewportChange: (value: MapViewportState) => void;
  resultsLoading: boolean;
  loading: boolean;
  error: string | null;
  totalCount: number;
}) {
  const activeState = useMemo(() => stateConfig.getStateConfig(stateCode), [stateCode]);
  const defaultStateBounds = useMemo(() => toStateBounds(activeState.defaultBounds), [activeState.defaultBounds]);
  const hasParcelTileArchive = Boolean(getParcelPmtilesUrl(stateCode, activeState.parcelPmtilesUrl));
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
  const selectedBounds = useMemo(
    () => selectedFeatureBounds(featureCollection, selectedParcelRowId),
    [featureCollection, selectedParcelRowId],
  );

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
      initializeParcelLayers(map, stateCode, activeState.parcelPmtilesUrl);

      if (map.getLayer("parcel-fills")) {
        map.on("click", "parcel-fills", (event) => {
          const feature = event.features?.[0];
          const parcelRowId = getMapFeatureSelectionParcelRowId(feature);
          if (parcelRowId) {
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
      }

      const currentBounds = map.getBounds();
      onViewportChange({
        center: [map.getCenter().lng, map.getCenter().lat],
        zoom: map.getZoom(),
        bounds: [currentBounds.getWest(), currentBounds.getSouth(), currentBounds.getEast(), currentBounds.getNorth()],
      });
    });

    map.fitBounds(defaultStateBounds, { padding: 28, duration: 0, maxZoom: 7.2 });

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
      const nextId = getMapFeatureSelectionParcelRowId(parcelFeature);
      if (hoveredFeatureIdRef.current && hoveredFeatureIdRef.current !== nextId) {
        map.setFeatureState({ source: PARCEL_TILE_SOURCE_ID, sourceLayer: PARCEL_TILE_LAYER, id: hoveredFeatureIdRef.current }, { hover: false });
      }
      if (nextId) {
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
  }, [activeState.parcelPmtilesUrl, defaultStateBounds, onSelect, onViewportChange, stateCode]);

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
          id: getMapFeatureSelectionParcelRowId(feature) ?? feature.properties.parcel_row_id,
          properties: {
            ...feature.properties,
            selected: getMapFeatureSelectionParcelRowId(feature) === selectedParcelRowId,
          },
        })) ?? [],
    };
    source.setData(nextCollection as never);

    if (process.env.NODE_ENV !== "production") {
      console.debug("[landintel-map] feature_count_loaded", nextCollection.features.length);
      console.debug("[landintel-map] computed_map_bounds", geometryResponse?.geometry_bounds ?? resultBounds);
      console.debug("[landintel-map] selected_parcel_row_id", selectedParcelRowId);
      if (!resultBounds && nextCollection.features.length > 0) {
        console.debug("[landintel-map] invalid_geometry_first_feature", nextCollection.features[0]);
      }
    }
  }, [featureCollection, geometryResponse?.geometry_bounds, resultBounds, selectedParcelRowId]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    updateLayerVisibility(map, "street-base", basemapMode === "street");
    updateLayerVisibility(map, "satellite-base", basemapMode === "satellite");
    updateLayerVisibility(map, "parcel-fills", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-lines", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-hover", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "selected-parcel-fill", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "selected-parcel-line", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-wetlands-overlay", activeOverlays.includes("wetlands"));
    updateLayerVisibility(map, "parcel-road-overlay", activeOverlays.includes("road_access"));
    updateLayerVisibility(map, "parcel-flood-overlay", activeOverlays.includes("fema_flood"));
  }, [activeOverlays, basemapMode]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || featureCount === 0) return;

    const hasNewFitRequest = fitNonce !== lastAppliedFitNonceRef.current;
    const hasSelectionChange = selectedParcelRowId !== lastSelectedIdRef.current && Boolean(selectedParcelRowId);
    const hasLocateSelectedRequest = locateSelectedNonce !== lastLocateSelectedNonceRef.current && Boolean(selectedParcelRowId);
    if (!hasNewFitRequest && !hasSelectionChange && !hasLocateSelectedRequest) {
      return;
    }

    if (selectedParcelRowId && !selectedBounds) {
      if (process.env.NODE_ENV !== "production") {
        console.debug("[landintel-map] waiting_for_selected_geometry", { selectedParcelRowId, featureCount });
      }
      return;
    }

    const targetBounds = selectedBounds ?? resultBounds;
    if (!targetBounds) {
      if (hasNewFitRequest && !selectedParcelRowId) {
        map.fitBounds(defaultStateBounds, {
          padding: 28,
          duration: hasInitializedViewportRef.current ? 600 : 0,
          maxZoom: 7.2,
        });
        hasInitializedViewportRef.current = true;
        lastAppliedFitNonceRef.current = fitNonce;
        lastLocateSelectedNonceRef.current = locateSelectedNonce;
        lastSelectedIdRef.current = selectedParcelRowId;
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
        console.debug("[landintel-map] fit_to_bounds", { selectedParcelRowId, targetBounds, featureCount, zoom: map.getZoom() });
      }
      map.fitBounds(toMapBounds(targetBounds), {
        padding,
        duration: hasInitializedViewportRef.current ? 600 : 0,
        maxZoom: selectedBounds ? 16.5 : 14.5,
      });
      hasInitializedViewportRef.current = true;
      lastAppliedFitNonceRef.current = fitNonce;
      lastLocateSelectedNonceRef.current = locateSelectedNonce;
      lastSelectedIdRef.current = selectedParcelRowId;
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
  }, [defaultStateBounds, featureCount, fitNonce, geometryResponse, locateSelectedNonce, resultBounds, selectedBounds, selectedParcelRowId]);

  let emptyTitle: string | null = null;
  let emptyBody: string | null = null;
  if (resultsLoading) {
    emptyTitle = "Loading parcel results";
    emptyBody = "Fetching parcel records and filters for the current dataset.";
  } else if (totalCount === 0) {
    emptyTitle = "No parcels match current filters";
    emptyBody = "Try broadening the current filter set or clearing preset constraints.";
  } else if (hasParcelTileArchive && viewport.zoom < PARCEL_TILE_MIN_ZOOM) {
    emptyTitle = "Zoom in to inspect parcel boundaries";
    emptyBody = `The base parcel layer uses PMTiles and becomes legible once you zoom further into ${activeState.displayName}.`;
  } else if (loading && selectedParcelRowId) {
    emptyTitle = "Loading selected parcel";
    emptyBody = "Fetching detailed geometry for the current selection.";
  } else if (error && selectedParcelRowId) {
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
