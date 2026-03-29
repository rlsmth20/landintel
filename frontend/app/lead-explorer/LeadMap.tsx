"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { useEffect, useEffectEvent, useMemo, useRef } from "react";
import maplibregl, { GeoJSONSource, LngLatBoundsLike, Map } from "maplibre-gl";
import { PMTiles, Protocol } from "pmtiles";

import parcelIdentity from "./parcelIdentity";
import stateConfig from "./stateConfig";
import type { BasemapMode, FeatureCollectionPayload, GeometryResponse, LeadRecord, MapOverlayId, MapViewportState } from "./types";

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
const DEFAULT_PARCEL_TILE_MIN_ZOOM = 10;
const POINT_GEOMETRY_FILTER: maplibregl.FilterSpecification = ["any", ["==", ["geometry-type"], "Point"], ["==", ["geometry-type"], "MultiPoint"]];
let pmtilesProtocol: Protocol | null = null;
let pmtilesArchiveUrl: string | null = null;
const PARCEL_INTERACTIVE_LAYER_IDS = ["parcel-points", "parcel-fills"] as const;
const PARCEL_LAYER_IDS = [
  "parcel-hover",
  "parcel-point-hover",
  "parcel-flood-overlay",
  "parcel-road-overlay",
  "parcel-wetlands-overlay",
  "parcel-points",
  "parcel-lines",
  "parcel-fills",
] as const;
const SELECTED_LAYER_IDS = ["selected-parcel-point", "selected-parcel-line", "selected-parcel-fill"] as const;

function toStateBounds(defaultBounds: [number, number, number, number]): [[number, number], [number, number]] {
  return [
    [defaultBounds[0], defaultBounds[1]],
    [defaultBounds[2], defaultBounds[3]],
  ];
}

function getParcelPmtilesUrl(configuredPmtilesUrl: string | null | undefined): string | null {
  const candidate = typeof configuredPmtilesUrl === "string" && configuredPmtilesUrl.trim().length > 0
    ? configuredPmtilesUrl.trim()
    : null;
  if (!candidate) return null;
  if (/^https?:\/\//i.test(candidate)) return candidate;
  if (typeof window !== "undefined") {
    return new URL(candidate, window.location.origin).toString();
  }
  return candidate;
}

function ensurePmtilesProtocol(parcelPmtilesUrl: string | null | undefined) {
  const nextUrl = getParcelPmtilesUrl(parcelPmtilesUrl);
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

function removeLayerIfPresent(map: Map, layerId: string) {
  if (map.getLayer(layerId)) {
    map.removeLayer(layerId);
  }
}

function removeSourceIfPresent(map: Map, sourceId: string) {
  if (map.getSource(sourceId)) {
    map.removeSource(sourceId);
  }
}

function emptyFeatureCollection(): FeatureCollectionPayload {
  return {
    type: "FeatureCollection",
    features: [],
  };
}

function applyLayerVisibility(map: Map, basemapMode: BasemapMode, activeOverlays: MapOverlayId[]) {
  updateLayerVisibility(map, "street-base", basemapMode === "street");
  updateLayerVisibility(map, "satellite-base", basemapMode === "satellite");
  updateLayerVisibility(map, "parcel-fills", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "parcel-lines", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "parcel-points", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "parcel-hover", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "parcel-point-hover", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "selected-parcel-fill", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "selected-parcel-line", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "selected-parcel-point", activeOverlays.includes("parcels"));
  updateLayerVisibility(map, "parcel-wetlands-overlay", activeOverlays.includes("wetlands"));
  updateLayerVisibility(map, "parcel-road-overlay", activeOverlays.includes("road_access"));
  updateLayerVisibility(map, "parcel-flood-overlay", activeOverlays.includes("fema_flood"));
}

function teardownParcelLayers(map: Map) {
  [...PARCEL_LAYER_IDS, ...SELECTED_LAYER_IDS].forEach((layerId) => removeLayerIfPresent(map, layerId));
  removeSourceIfPresent(map, PARCEL_TILE_SOURCE_ID);
  removeSourceIfPresent(map, SELECTED_PARCEL_SOURCE_ID);
}

function initializeParcelLayers(
  map: Map,
  parcelPmtilesUrl: string | null | undefined,
  parcelTileMinZoom: number,
) {
  map.addSource(SELECTED_PARCEL_SOURCE_ID, {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: [],
    },
  });

  const resolvedParcelPmtilesUrl = ensurePmtilesProtocol(parcelPmtilesUrl);
  if (resolvedParcelPmtilesUrl && !map.getSource(PARCEL_TILE_SOURCE_ID)) {
    map.addSource(PARCEL_TILE_SOURCE_ID, {
      type: "vector",
      url: `pmtiles://${resolvedParcelPmtilesUrl}`,
      minzoom: parcelTileMinZoom,
      maxzoom: 15,
      promoteId: { [PARCEL_TILE_LAYER]: "parcel_row_id" },
    });

    map.addLayer({
      id: "parcel-fills",
      type: "fill",
      source: PARCEL_TILE_SOURCE_ID,
      "source-layer": PARCEL_TILE_LAYER,
      minzoom: parcelTileMinZoom,
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
      minzoom: parcelTileMinZoom,
      paint: {
        "line-color": "#17393a",
        "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.4, 10, 0.8, 13, 1.2, 15, 1.6],
        "line-opacity": 0.88,
      },
    });

    map.addLayer({
      id: "parcel-points",
      type: "circle",
      source: PARCEL_TILE_SOURCE_ID,
      "source-layer": PARCEL_TILE_LAYER,
      minzoom: parcelTileMinZoom,
      filter: POINT_GEOMETRY_FILTER,
      paint: {
        "circle-color": "#2f6b6d",
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 1.2, 10, 1.8, 13, 2.6, 15, 3.4],
        "circle-opacity": 0.82,
        "circle-stroke-color": "#17393a",
        "circle-stroke-width": 0.8,
      },
    });

    map.addLayer({
      id: "parcel-wetlands-overlay",
      type: "line",
      source: PARCEL_TILE_SOURCE_ID,
      "source-layer": PARCEL_TILE_LAYER,
      minzoom: parcelTileMinZoom,
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
      minzoom: parcelTileMinZoom,
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
      minzoom: parcelTileMinZoom,
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

  map.addLayer({
    id: "selected-parcel-point",
    type: "circle",
    source: SELECTED_PARCEL_SOURCE_ID,
    filter: POINT_GEOMETRY_FILTER,
    paint: {
      "circle-color": "#d9472f",
      "circle-radius": 7.5,
      "circle-opacity": 0.94,
      "circle-stroke-color": "#fff8ee",
      "circle-stroke-width": 2.2,
    },
  });

  if (parcelPmtilesUrl) {
    map.addLayer({
      id: "parcel-hover",
      type: "line",
      source: PARCEL_TILE_SOURCE_ID,
      "source-layer": PARCEL_TILE_LAYER,
      minzoom: parcelTileMinZoom,
      paint: {
        "line-color": "#ffe5cf",
        "line-width": ["case", ["boolean", ["feature-state", "hover"], false], 3.2, 0],
        "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 1, 0],
      },
    });

    map.addLayer({
      id: "parcel-point-hover",
      type: "circle",
      source: PARCEL_TILE_SOURCE_ID,
      "source-layer": PARCEL_TILE_LAYER,
      minzoom: parcelTileMinZoom,
      filter: POINT_GEOMETRY_FILTER,
      paint: {
        "circle-color": "#ffe5cf",
        "circle-radius": ["case", ["boolean", ["feature-state", "hover"], false], 6, 0],
        "circle-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.45, 0],
        "circle-stroke-color": "#fff8ee",
        "circle-stroke-width": ["case", ["boolean", ["feature-state", "hover"], false], 1.4, 0],
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
  const parcelTileMinZoom = useMemo(
    () => activeState.parcelPmtilesMinZoom ?? stateConfig.DEFAULT_PARCEL_PMTILES_MIN_ZOOM ?? DEFAULT_PARCEL_TILE_MIN_ZOOM,
    [activeState.parcelPmtilesMinZoom],
  );
  const hasParcelTileArchive = Boolean(getParcelPmtilesUrl(activeState.parcelPmtilesUrl));
  const initialViewportRef = useRef(viewport);
  const initialMapConfigRef = useRef({
    stateCode,
    parcelPmtilesUrl: activeState.parcelPmtilesUrl,
    parcelTileMinZoom,
    basemapMode,
    activeOverlays,
    defaultStateBounds,
  });
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const hoveredFeatureIdRef = useRef<string | null>(null);
  const configuredStateCodeRef = useRef<string | null>(null);
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
  const emitViewportChange = useEffectEvent((map: Map) => {
    const currentBounds = map.getBounds();
    onViewportChange({
      center: [map.getCenter().lng, map.getCenter().lat],
      zoom: map.getZoom(),
      bounds: [currentBounds.getWest(), currentBounds.getSouth(), currentBounds.getEast(), currentBounds.getNorth()],
    });
  });
  const handleParcelSelect = useEffectEvent((parcelRowId: string) => {
    onSelect(parcelRowId);
  });

  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;
    const initialViewport = initialViewportRef.current;
    const initialMapConfig = initialMapConfigRef.current;
    if (process.env.NODE_ENV !== "production") {
      console.debug("[landintel-map] mount", { stateCode: initialMapConfig.stateCode });
    }

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: BASE_STYLE,
      center: initialViewport.center ?? DEFAULT_CENTER,
      zoom: initialViewport.zoom ?? DEFAULT_ZOOM,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    mapRef.current = map;

    const clearHoveredFeature = () => {
      if (hoveredFeatureIdRef.current) {
        try {
          map.setFeatureState({ source: PARCEL_TILE_SOURCE_ID, sourceLayer: PARCEL_TILE_LAYER, id: hoveredFeatureIdRef.current }, { hover: false });
        } catch {
          // Layer/source can disappear during state reconfiguration.
        }
        hoveredFeatureIdRef.current = null;
      }
      map.getCanvas().style.cursor = "";
    };

    const handleMapLoad = () => {
      initializeParcelLayers(map, initialMapConfig.parcelPmtilesUrl, initialMapConfig.parcelTileMinZoom);
      configuredStateCodeRef.current = initialMapConfig.stateCode;
      applyLayerVisibility(map, initialMapConfig.basemapMode, initialMapConfig.activeOverlays);
      emitViewportChange(map);
    };

    const handleMapClick = (event: maplibregl.MapMouseEvent & maplibregl.EventData) => {
      const interactiveLayers = PARCEL_INTERACTIVE_LAYER_IDS.filter((layerId) => map.getLayer(layerId));
      if (interactiveLayers.length === 0) return;
      const feature = map.queryRenderedFeatures(event.point, { layers: [...interactiveLayers] })[0];
      const parcelRowId = getMapFeatureSelectionParcelRowId(feature);
      if (!parcelRowId) return;
      if (process.env.NODE_ENV !== "production") {
        console.debug("[landintel-map] parcel click", { parcelRowId });
      }
      handleParcelSelect(parcelRowId);
    };

    map.fitBounds(initialMapConfig.defaultStateBounds, { padding: 28, duration: 0, maxZoom: 7.2 });

    const handleMapMoveEnd = () => emitViewportChange(map);

    const handleMapMouseMove = (event: maplibregl.MapMouseEvent & maplibregl.EventData) => {
      const interactiveLayers = PARCEL_INTERACTIVE_LAYER_IDS.filter((layerId) => map.getLayer(layerId));
      if (interactiveLayers.length === 0) {
        clearHoveredFeature();
        return;
      }
      const parcelFeature = map.queryRenderedFeatures(event.point, {
        layers: [...interactiveLayers],
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
    };

    map.on("load", handleMapLoad);
    map.on("click", handleMapClick);
    map.on("moveend", handleMapMoveEnd);
    map.on("mousemove", handleMapMouseMove);

    return () => {
      if (process.env.NODE_ENV !== "production") {
        console.debug("[landintel-map] unmount", { stateCode: configuredStateCodeRef.current ?? initialMapConfig.stateCode });
      }
      map.off("load", handleMapLoad);
      map.off("click", handleMapClick);
      map.off("moveend", handleMapMoveEnd);
      map.off("mousemove", handleMapMouseMove);
      map.remove();
      mapRef.current = null;
      configuredStateCodeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    if (configuredStateCodeRef.current === stateCode) return;
    if (process.env.NODE_ENV !== "production") {
      console.debug("[landintel-map] reconfigure", {
        fromStateCode: configuredStateCodeRef.current,
        toStateCode: stateCode,
      });
    }
    hoveredFeatureIdRef.current = null;
    teardownParcelLayers(map);
    initializeParcelLayers(map, activeState.parcelPmtilesUrl, parcelTileMinZoom);
    const selectedSource = map.getSource(SELECTED_PARCEL_SOURCE_ID) as GeoJSONSource | undefined;
    if (selectedSource) {
      selectedSource.setData(emptyFeatureCollection() as never);
    }
    configuredStateCodeRef.current = stateCode;
    lastAppliedFitNonceRef.current = -1;
    lastLocateSelectedNonceRef.current = -1;
    lastSelectedIdRef.current = null;
    applyLayerVisibility(map, basemapMode, activeOverlays);
    map.fitBounds(defaultStateBounds, { padding: 28, duration: 0, maxZoom: 7.2 });
    hasInitializedViewportRef.current = true;
  }, [activeOverlays, activeState.parcelPmtilesUrl, basemapMode, defaultStateBounds, parcelTileMinZoom, stateCode]);

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
    applyLayerVisibility(map, basemapMode, activeOverlays);
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
  }, [defaultStateBounds, featureCollection?.features, featureCount, fitNonce, geometryResponse, locateSelectedNonce, resultBounds, selectedBounds, selectedParcelRowId]);

  const showParcelLayerHint =
    hasParcelTileArchive &&
    activeOverlays.includes("parcels") &&
    (viewport.zoom ?? DEFAULT_ZOOM) < parcelTileMinZoom;

  const statusMessages: Array<{ key: string; tone: "neutral" | "warn"; title: string; body?: string | null }> = [];
  if (resultsLoading) {
    statusMessages.push({
      key: "results-loading",
      tone: "neutral",
      title: "Refreshing parcel results",
    });
  } else if (totalCount === 0) {
    statusMessages.push({
      key: "no-results",
      tone: "neutral",
      title: "No parcels match current filters",
      body: "Broaden the current filters to repopulate the statewide parcel base.",
    });
  }

  if (showParcelLayerHint) {
    statusMessages.push({
      key: "parcel-minzoom",
      tone: "neutral",
      title: `Parcel layer appears at zoom ${parcelTileMinZoom}+`,
      body: `Base parcel geometry stays hidden until it is legible in ${activeState.displayName}.`,
    });
  }

  if (loading && selectedParcelRowId) {
    statusMessages.push({
      key: "selected-loading",
      tone: "neutral",
      title: "Loading selected parcel",
    });
  } else if (error && selectedParcelRowId) {
    statusMessages.push({
      key: "selected-error",
      tone: "warn",
      title: "Selected parcel detail unavailable",
      body: error,
    });
  }

  return (
    <div className="lead-map-shell">
      <div className="lead-map-canvas" ref={mapContainerRef} />
      {statusMessages.length > 0 ? (
        <div className="map-status-stack" aria-live="polite">
          {statusMessages.map((message) => (
            <div key={message.key} className={`map-status-note is-${message.tone}`}>
              <strong>{message.title}</strong>
              {message.body ? <p>{message.body}</p> : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
