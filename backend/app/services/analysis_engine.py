from sqlalchemy import text


def analyze_land(polygon, db):
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 points")

    if polygon[0] != polygon[-1]:
        polygon = polygon + [polygon[0]]

    coords = ",".join([f"{p[0]} {p[1]}" for p in polygon])
    wkt = f"POLYGON(({coords}))"

    area_query = text("""
        SELECT
            ABS(ST_Area(ST_GeomFromText(:wkt, 4326)::geography) / 4046.86) AS acres
    """)

    road_query = text("""
        WITH parcel AS (
            SELECT ST_Centroid(ST_GeomFromText(:wkt, 4326)) AS geom
        )
        SELECT
            ST_Distance(
                ST_SetSRID(r.geom, 4326)::geography,
                p.geom::geography
            ) * 3.28084 AS road_distance_ft
        FROM roads r
        CROSS JOIN parcel p
        ORDER BY ST_SetSRID(r.geom, 4326) <-> p.geom
        LIMIT 1
    """)

    area_result = db.execute(area_query, {"wkt": wkt}).fetchone()
    road_result = db.execute(road_query, {"wkt": wkt}).fetchone()

    acres = float(area_result[0]) if area_result and area_result[0] is not None else 0.0
    road_distance = float(road_result[0]) if road_result and road_result[0] is not None else 0.0

    return {
        "area_acres": round(acres, 2),
        "road_distance_ft": round(road_distance, 2),
        "power_distance_mi": None,
        "flood_overlap_pct": 0.0,
        "slope_mean_pct": 4.2,
        "risk_score": 22
    }