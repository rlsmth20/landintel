"use client";

import "maplibre-gl/dist/maplibre-gl.css";

import { useEffect, useMemo, useRef } from "react";
import maplibregl, { GeoJSONSource, LngLatBoundsLike, Map } from "maplibre-gl";

import type { FeatureCollectionPayload, GeometryFeature, GeometryResponse, MapOverlayId, MapViewportState } from "./types";

const DEFAULT_CENTER: [number, number] = [-98.5795, 39.8283];
const DEFAULT_ZOOM = 3.4;
const PARCEL_SOURCE_ID = "landintel-parcels";

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
  if (map.getSource(PARCEL_SOURCE_ID)) return;

  map.addSource(PARCEL_SOURCE_ID, {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: [],
    },
  });

  map.addLayer({
    id: "parcel-fills",
    type: "fill",
    source: PARCEL_SOURCE_ID,
    filter: ["==", ["geometry-type"], "Polygon"],
    paint: {
      "fill-color": [
        "case",
        ["boolean", ["get", "selected"], false],
        "#d9472f",
        "#2f6b6d",
      ],
      "fill-opacity": [
        "case",
        ["boolean", ["get", "selected"], false],
        0.44,
        0.26,
      ],
    },
  });

  map.addLayer({
    id: "parcel-lines",
    type: "line",
    source: PARCEL_SOURCE_ID,
    filter: ["==", ["geometry-type"], "Polygon"],
    paint: {
      "line-color": [
        "case",
        ["boolean", ["get", "selected"], false],
        "#fff8ee",
        "#17393a",
      ],
      "line-width": [
        "case",
        ["boolean", ["get", "selected"], false],
        3.2,
        1.6,
      ],
      "line-opacity": 0.96,
    },
  });

  map.addLayer({
    id: "parcel-wetlands-overlay",
    type: "line",
    source: PARCEL_SOURCE_ID,
    filter: ["all", ["==", ["geometry-type"], "Polygon"], ["==", ["get", "wetland_flag"], true]],
    paint: {
      "line-color": "#617f56",
      "line-width": 2.2,
      "line-opacity": 0.95,
    },
  });

  map.addLayer({
    id: "parcel-road-overlay",
    type: "line",
    source: PARCEL_SOURCE_ID,
    filter: ["all", ["==", ["geometry-type"], "Polygon"], ["==", ["get", "road_access_tier"], "direct"]],
    paint: {
      "line-color": "#1f7f80",
      "line-width": 2.4,
      "line-opacity": 0.95,
    },
  });

  map.addLayer({
    id: "parcel-flood-overlay",
    type: "line",
    source: PARCEL_SOURCE_ID,
    filter: ["all", ["==", ["geometry-type"], "Polygon"], [">", ["coalesce", ["get", "flood_risk_score"], 0], 0]],
    paint: {
      "line-color": "#5f8db8",
      "line-width": 2.4,
      "line-opacity": 0.95,
    },
  });

  map.addLayer({
    id: "parcel-points",
    type: "circle",
    source: PARCEL_SOURCE_ID,
    filter: ["==", ["geometry-type"], "Point"],
    paint: {
      "circle-radius": [
        "interpolate",
        ["linear"],
        ["zoom"],
        3,
        3.5,
        8,
        6,
        12,
        8,
      ],
      "circle-color": [
        "case",
        ["boolean", ["get", "selected"], false],
        "#d9472f",
        ["boolean", ["get", "county_hosted_flag"], false],
        "#1f7f80",
        "#f18f01",
      ],
      "circle-stroke-color": "#fff8ee",
      "circle-stroke-width": 1.4,
      "circle-opacity": 0.9,
    },
  });

  map.addLayer({
    id: "parcel-hover",
    type: "line",
    source: PARCEL_SOURCE_ID,
    filter: ["all", ["==", ["geometry-type"], "Polygon"], ["boolean", ["feature-state", "hover"], false]],
    paint: {
      "line-color": "#ffe5cf",
      "line-width": 3.6,
      "line-opacity": 1,
    },
  });
}

export function LeadMap({
  geometryResponse,
  selectedId,
  hoveredId,
  onSelect,
  onHoverChange,
  fitNonce,
  activeOverlays,
  viewport,
  onViewportChange,
}: {
  geometryResponse: GeometryResponse | null;
  selectedId: string | null;
  hoveredId: string | null;
  onSelect: (value: string) => void;
  onHoverChange: (value: string | null) => void;
  fitNonce: number;
  activeOverlays: MapOverlayId[];
  viewport: MapViewportState;
  onViewportChange: (value: MapViewportState) => void;
}) {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<Map | null>(null);
  const hoveredFeatureIdRef = useRef<string | null>(null);
  const hasInitializedViewportRef = useRef(false);
  const lastAppliedFitNonceRef = useRef<number>(-1);
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
        if (typeof parcelRowId === "string") onSelect(parcelRowId);
      });

      map.on("click", "parcel-points", (event) => {
        const feature = event.features?.[0];
        const parcelRowId = feature?.properties?.parcel_row_id;
        if (typeof parcelRowId === "string") onSelect(parcelRowId);
      });

      map.on("mouseleave", "parcel-fills", () => {
        if (hoveredFeatureIdRef.current) {
          map.setFeatureState({ source: PARCEL_SOURCE_ID, id: hoveredFeatureIdRef.current }, { hover: false });
          hoveredFeatureIdRef.current = null;
        }
        map.getCanvas().style.cursor = "";
        onHoverChange(null);
      });

      const currentBounds = map.getBounds();
      onViewportChange({
        center: [map.getCenter().lng, map.getCenter().lat],
        zoom: map.getZoom(),
        bounds: [currentBounds.getWest(), currentBounds.getSouth(), currentBounds.getEast(), currentBounds.getNorth()],
      });
    });

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
        layers: ["parcel-fills", "parcel-points"],
      })[0];
      const nextId = parcelFeature?.properties?.parcel_row_id;
      if (hoveredFeatureIdRef.current && hoveredFeatureIdRef.current !== nextId) {
        map.setFeatureState({ source: PARCEL_SOURCE_ID, id: hoveredFeatureIdRef.current }, { hover: false });
      }
      if (typeof nextId === "string") {
        hoveredFeatureIdRef.current = nextId;
        map.setFeatureState({ source: PARCEL_SOURCE_ID, id: nextId }, { hover: true });
        map.getCanvas().style.cursor = "pointer";
        onHoverChange(nextId);
      } else {
        hoveredFeatureIdRef.current = null;
        map.getCanvas().style.cursor = "";
        onHoverChange(null);
      }
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [onHoverChange, onSelect, onViewportChange, viewport.center, viewport.zoom]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const source = map.getSource(PARCEL_SOURCE_ID) as GeoJSONSource | undefined;
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
            hovered: feature.properties.parcel_row_id === hoveredId,
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
  }, [featureCollection, geometryResponse?.geometry_bounds, hoveredId, resultBounds, selectedId]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    updateLayerVisibility(map, "parcel-fills", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-lines", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-points", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-hover", activeOverlays.includes("parcels"));
    updateLayerVisibility(map, "parcel-wetlands-overlay", activeOverlays.includes("wetlands"));
    updateLayerVisibility(map, "parcel-road-overlay", activeOverlays.includes("road_access"));
    updateLayerVisibility(map, "parcel-flood-overlay", activeOverlays.includes("fema_flood"));
  }, [activeOverlays]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || featureCount === 0) return;

    const isFirstFit = !hasInitializedViewportRef.current;
    const hasNewFitRequest = fitNonce !== lastAppliedFitNonceRef.current;
    const hasSelectionChange = selectedId !== lastSelectedIdRef.current && Boolean(selectedId);
    if (!isFirstFit && !hasNewFitRequest && !hasSelectionChange) {
      return;
    }

    const targetBounds = selectedBounds ?? resultBounds;
    if (!targetBounds) {
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
      map.fitBounds(toMapBounds(targetBounds), {
        padding,
        duration: hasInitializedViewportRef.current ? 600 : 0,
        maxZoom: selectedBounds ? 16.5 : 14.5,
      });
      hasInitializedViewportRef.current = true;
      lastAppliedFitNonceRef.current = fitNonce;
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
  }, [featureCount, fitNonce, geometryResponse, resultBounds, selectedBounds, selectedId]);

  return (
    <div className="lead-map-shell">
      <div className="lead-map-canvas" ref={mapContainerRef} />
      {featureCount === 0 ? (
        <div className="map-empty-state map-overlay-empty">
          <strong>No parcel geometry loaded for this view</strong>
          <p>Pan, zoom, or adjust filters to load parcel geometry in the visible map area.</p>
        </div>
      ) : null}
    </div>
  );
}
