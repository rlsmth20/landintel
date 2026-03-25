from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


FEET_PER_METER = 3.28084

GEOMETRY_MARKETABILITY_COLUMNS = [
    "geometry_estimated_frontage_feet",
    "geometry_estimated_width_feet",
    "geometry_min_dimension_feet",
    "geometry_max_dimension_feet",
    "geometry_frontage_to_width_ratio",
    "geometry_effective_buildable_flag",
    "geometry_marketability_base_flag",
    "geometry_marketability_flag",
    "geometry_marketability_context",
    "geometry_marketability_action",
    "geometry_penalty_points",
    "geometry_penalty_reason",
    "geometry_marketability_default_leads_excluded_flag",
]


@dataclass(frozen=True)
class GeometryMarketabilityConfig:
    suburban_density_threshold: float = 45.0
    unbuildable_width_feet: float = 25.0
    unbuildable_min_dimension_feet: float = 20.0
    unbuildable_small_area_acres: float = 0.5
    unbuildable_small_area_compactness: float = 0.25
    unbuildable_wedge_frontage_to_width_ratio: float = 7.0
    unbuildable_wedge_max_width_feet: float = 35.0
    unbuildable_aspect_ratio: float = 6.0
    unbuildable_aspect_compactness: float = 0.18
    poor_geometry_small_area_acres: float = 1.0
    poor_geometry_small_area_compactness: float = 0.28
    poor_geometry_frontage_to_width_ratio: float = 4.5
    poor_geometry_max_width_feet: float = 60.0
    poor_geometry_irregular_small_area_acres: float = 2.0
    remnant_small_area_acres: float = 0.75
    remnant_compactness: float = 0.24
    remnant_width_feet: float = 40.0
    remnant_min_dimension_feet: float = 30.0
    remnant_frontage_to_width_ratio: float = 5.0
    rural_meaningful_area_acres: float = 5.0
    rural_meaningful_width_feet: float = 60.0
    rural_meaningful_min_dimension_feet: float = 35.0
    constrained_width_feet: float = 50.0
    constrained_min_dimension_feet: float = 35.0
    constrained_compactness: float = 0.35
    constrained_frontage_to_width_ratio: float = 3.5
    constrained_aspect_ratio: float = 4.5
    urban_density_threshold: float = 120.0
    marketable_penalty_points: float = 0.0
    constrained_penalty_points: float = -8.0
    poor_geometry_penalty_points: float = -28.0
    poor_geometry_exclude_penalty_points: float = -45.0
    unbuildable_penalty_points: float = -60.0


DEFAULT_GEOMETRY_MARKETABILITY_CONFIG = GeometryMarketabilityConfig()


def _float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _string_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="string")
    return frame[column].astype("string").fillna(default)


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="bool")
    return frame[column].fillna(default).astype(bool)


def _first_non_null(*series_list: pd.Series) -> pd.Series:
    if not series_list:
        raise ValueError("parcel_marketability_ms._first_non_null requires at least one series")
    result = series_list[0].copy()
    for series in series_list[1:]:
        result = result.fillna(series)
    return result


def add_geometry_marketability_fields(
    frame: pd.DataFrame,
    *,
    config: GeometryMarketabilityConfig | None = None,
) -> pd.DataFrame:
    config = config or DEFAULT_GEOMETRY_MARKETABILITY_CONFIG
    enriched = frame.copy()

    frontage_feet = _first_non_null(
        _float_series(enriched, "parcel_frontage_ft_estimate"),
        np.maximum(_float_series(enriched, "bounding_box_width_meters"), _float_series(enriched, "bounding_box_height_meters")) * FEET_PER_METER,
    )
    width_feet = _first_non_null(
        _float_series(enriched, "parcel_width_ft_estimate"),
        np.minimum(_float_series(enriched, "bounding_box_width_meters"), _float_series(enriched, "bounding_box_height_meters")) * FEET_PER_METER,
    )
    bbox_width_feet = _float_series(enriched, "bounding_box_width_meters") * FEET_PER_METER
    bbox_height_feet = _float_series(enriched, "bounding_box_height_meters") * FEET_PER_METER
    bbox_min_feet = np.minimum(bbox_width_feet, bbox_height_feet)
    bbox_max_feet = np.maximum(bbox_width_feet, bbox_height_feet)

    min_dimension_feet = _first_non_null(
        pd.Series(np.minimum(frontage_feet, width_feet), index=enriched.index, dtype="float64"),
        bbox_min_feet,
    )
    max_dimension_feet = _first_non_null(
        pd.Series(np.maximum(frontage_feet, width_feet), index=enriched.index, dtype="float64"),
        bbox_max_feet,
    )
    aspect_ratio = _first_non_null(
        _float_series(enriched, "aspect_ratio"),
        _float_series(enriched, "parcel_aspect_ratio_estimate"),
        pd.Series(
            np.where(min_dimension_feet.gt(0), max_dimension_feet / min_dimension_feet, np.nan),
            index=enriched.index,
            dtype="float64",
        ),
    )
    compactness = _first_non_null(
        _float_series(enriched, "compactness"),
        _float_series(enriched, "shape_compactness"),
    )
    area_acres = _first_non_null(
        _float_series(enriched, "area_acres"),
        _float_series(enriched, "acreage"),
    )
    frontage_to_width_ratio = pd.Series(
        np.where(width_feet.gt(0), frontage_feet / width_feet.clip(lower=1.0), np.nan),
        index=enriched.index,
        dtype="float64",
    )
    nearby_density = _float_series(enriched, "nearby_building_density").fillna(0.0)
    growth_bucket = _string_series(enriched, "growth_pressure_bucket", default="unknown").str.lower()
    land_use = _first_non_null(
        _string_series(enriched, "land_use", default=""),
        _string_series(enriched, "land_use_raw", default=""),
    ).str.lower()
    geometry_quality_flag = _string_series(enriched, "geometry_quality_flag", default="good").str.lower()
    is_multipart = _bool_series(enriched, "is_multipart", default=False)
    part_count = _float_series(enriched, "part_count").fillna(0).astype(int)
    building_count = _float_series(enriched, "building_count").fillna(0.0)
    building_area_total = _float_series(enriched, "building_area_total").fillna(0.0)
    assessed_total_value = _first_non_null(
        _float_series(enriched, "assessed_total_value"),
        _float_series(enriched, "total_value"),
    ).fillna(0.0)
    parcel_improvement_status = _string_series(enriched, "parcel_improvement_status", default="").str.lower()

    developed_land_use = land_use.str.contains(
        r"residential|commercial|industrial|mobile|manufactured|subdivision|lot|urban|town|city",
        case=False,
        regex=True,
        na=False,
    )
    rural_land_use = land_use.str.contains(
        r"ag|agric|farm|forest|timber|pasture|crop|rural|acreage|wood",
        case=False,
        regex=True,
        na=False,
    )
    high_density = nearby_density.ge(float(config.urban_density_threshold))
    moderate_density = nearby_density.ge(float(config.suburban_density_threshold))
    urban_or_suburban = high_density | moderate_density | growth_bucket.isin(["moderate", "high"]) | developed_land_use
    rural_context = ~urban_or_suburban | (rural_land_use & ~developed_land_use)
    width_too_small = width_feet.lt(float(config.unbuildable_width_feet))
    min_dimension_too_small = min_dimension_feet.lt(float(config.unbuildable_min_dimension_feet))
    tiny_low_compactness = area_acres.lt(float(config.unbuildable_small_area_acres)) & compactness.lt(float(config.unbuildable_small_area_compactness))
    wedge_like = frontage_to_width_ratio.gt(float(config.unbuildable_wedge_frontage_to_width_ratio)) & width_feet.lt(
        float(config.unbuildable_wedge_max_width_feet)
    )
    slender_scrap = aspect_ratio.gt(float(config.unbuildable_aspect_ratio)) & compactness.lt(float(config.unbuildable_aspect_compactness))
    geometry_access_strip = geometry_quality_flag.eq("access_strip")
    urban_tiny_irregular = urban_or_suburban & area_acres.lt(0.35) & compactness.lt(0.35)

    poor_small_irregular = area_acres.lt(float(config.poor_geometry_small_area_acres)) & compactness.lt(
        float(config.poor_geometry_small_area_compactness)
    )
    poor_frontage_remnant = frontage_to_width_ratio.gt(float(config.poor_geometry_frontage_to_width_ratio)) & width_feet.lt(
        float(config.poor_geometry_max_width_feet)
    )
    poor_irregular = geometry_quality_flag.eq("multipart_complex") | (
        geometry_quality_flag.eq("irregular") & (area_acres.lt(float(config.poor_geometry_irregular_small_area_acres)) | urban_or_suburban)
    )
    narrow_remnant_like = width_feet.lt(float(config.remnant_width_feet)) | min_dimension_feet.lt(float(config.remnant_min_dimension_feet))
    compact_small_remnant = area_acres.lt(float(config.remnant_small_area_acres)) & compactness.lt(float(config.remnant_compactness))
    frontage_remnant_like = frontage_to_width_ratio.gt(float(config.remnant_frontage_to_width_ratio))
    developed_remnant_like = urban_or_suburban & (
        compact_small_remnant
        | (narrow_remnant_like & frontage_remnant_like)
        | (narrow_remnant_like & compactness.lt(float(config.poor_geometry_small_area_compactness)))
    )
    dense_side_scrap = high_density & narrow_remnant_like & (frontage_remnant_like | compactness.lt(float(config.remnant_compactness)))
    suburban_frontage_scrap = urban_or_suburban & area_acres.lt(1.0) & frontage_remnant_like & width_feet.lt(float(config.poor_geometry_max_width_feet))
    rural_meaningful = (
        rural_context
        & area_acres.ge(float(config.rural_meaningful_area_acres))
        & width_feet.ge(float(config.rural_meaningful_width_feet))
        & min_dimension_feet.ge(float(config.rural_meaningful_min_dimension_feet))
    )
    improvement_signal = (
        building_count.ge(1)
        | building_area_total.ge(400)
        | assessed_total_value.ge(25000)
        | parcel_improvement_status.eq("likely_improved")
    )

    constrained_shape = (
        width_feet.lt(float(config.constrained_width_feet))
        | min_dimension_feet.lt(float(config.constrained_min_dimension_feet))
        | compactness.lt(float(config.constrained_compactness))
        | frontage_to_width_ratio.gt(float(config.constrained_frontage_to_width_ratio))
        | (aspect_ratio.gt(float(config.constrained_aspect_ratio)) & compactness.lt(float(config.poor_geometry_small_area_compactness)))
        | is_multipart
        | part_count.gt(1)
    )

    unbuildable = width_too_small | min_dimension_too_small | tiny_low_compactness | wedge_like | slender_scrap | geometry_access_strip
    poor_geometry = (~unbuildable) & (poor_small_irregular | poor_frontage_remnant | poor_irregular | urban_tiny_irregular)
    constrained = (~unbuildable) & (~poor_geometry) & constrained_shape

    base_marketability_flag = pd.Series("marketable", index=enriched.index, dtype="string")
    base_marketability_flag = base_marketability_flag.mask(constrained, "constrained")
    base_marketability_flag = base_marketability_flag.mask(poor_geometry, "poor_geometry")
    base_marketability_flag = base_marketability_flag.mask(unbuildable, "unbuildable_candidate")

    marketability_flag = base_marketability_flag.copy()
    marketability_flag = marketability_flag.mask(base_marketability_flag.eq("constrained") & developed_remnant_like, "poor_geometry")
    marketability_flag = marketability_flag.mask(
        base_marketability_flag.eq("poor_geometry") & rural_meaningful & ~dense_side_scrap & ~suburban_frontage_scrap,
        "constrained",
    )

    marketability_context = pd.Series("rural", index=enriched.index, dtype="string")
    marketability_context = marketability_context.mask(urban_or_suburban, "urban_suburban")

    marketability_action = pd.Series("keep", index=enriched.index, dtype="string")
    exclude = marketability_flag.eq("unbuildable_candidate") | (
        marketability_flag.eq("poor_geometry") & (developed_remnant_like | dense_side_scrap | suburban_frontage_scrap) & ~improvement_signal
    )
    penalize = (~exclude) & (
        marketability_flag.eq("poor_geometry")
        | (marketability_flag.eq("constrained") & (urban_or_suburban | ~rural_meaningful))
    )
    marketability_action = marketability_action.mask(penalize, "penalize")
    marketability_action = marketability_action.mask(exclude, "exclude")

    penalty_points = pd.Series(float(config.marketable_penalty_points), index=enriched.index, dtype="float64")
    penalty_points = penalty_points.mask(penalize & marketability_flag.eq("constrained"), float(config.constrained_penalty_points))
    penalty_points = penalty_points.mask(penalize & marketability_flag.eq("poor_geometry"), float(config.poor_geometry_penalty_points))
    penalty_points = penalty_points.mask(exclude & marketability_flag.eq("poor_geometry"), float(config.poor_geometry_exclude_penalty_points))
    penalty_points = penalty_points.mask(exclude & marketability_flag.eq("unbuildable_candidate"), float(config.unbuildable_penalty_points))

    penalty_reason = pd.Series("Geometry appears broadly marketable.", index=enriched.index, dtype="string")
    penalty_reason = penalty_reason.mask(
        marketability_flag.eq("constrained") & marketability_action.eq("keep"),
        "Geometry is somewhat constrained, but acreage and dimensions remain usable in a lower-density context.",
    )
    penalty_reason = penalty_reason.mask(
        marketability_flag.eq("constrained") & marketability_action.eq("penalize"),
        "Geometry is constrained enough to reduce marketability, especially in a developed context.",
    )
    penalty_reason = penalty_reason.mask(
        marketability_flag.eq("poor_geometry") & marketability_action.eq("penalize"),
        "Geometry appears poor, but the parcel may remain usable given its rural context or acreage.",
    )
    penalty_reason = penalty_reason.mask(
        marketability_flag.eq("poor_geometry") & marketability_action.eq("exclude"),
        "Poor geometry in a developed context suggests a remnant parcel rather than a practical acquisition target.",
    )
    penalty_reason = penalty_reason.mask(unbuildable, "Geometry appears unbuildable or highly unmarketable due to extreme narrowness or wedge-like shape.")
    penalty_reason = penalty_reason.mask(width_too_small, "Estimated parcel width is below 25 feet.")
    penalty_reason = penalty_reason.mask(min_dimension_too_small & ~width_too_small, "Minimum parcel dimension is below 20 feet.")
    penalty_reason = penalty_reason.mask(tiny_low_compactness & ~(width_too_small | min_dimension_too_small), "Small parcel area combined with very low compactness suggests a wedge or remnant.")
    penalty_reason = penalty_reason.mask(wedge_like & ~(width_too_small | min_dimension_too_small | tiny_low_compactness), "Frontage-to-width ratio is extreme for a narrow wedge-like parcel.")
    penalty_reason = penalty_reason.mask(slender_scrap & ~(width_too_small | min_dimension_too_small | tiny_low_compactness | wedge_like), "Aspect ratio and compactness indicate a narrow remnant or sliver parcel.")
    penalty_reason = penalty_reason.mask(
        poor_frontage_remnant & ~unbuildable,
        "Frontage greatly exceeds usable width, which suggests a frontage remnant rather than a practical building lot.",
    )
    penalty_reason = penalty_reason.mask(
        urban_tiny_irregular & ~(unbuildable | poor_frontage_remnant),
        "Small irregular parcel in a denser area appears unlikely to be meaningfully buildable or marketable.",
    )
    penalty_reason = penalty_reason.mask(
        poor_irregular & ~(unbuildable | poor_frontage_remnant | urban_tiny_irregular),
        "Multipart or irregular geometry reduces practical marketability.",
    )
    penalty_reason = penalty_reason.mask(
        developed_remnant_like & ~unbuildable,
        "Urban or suburban context makes this remnant-like shape unlikely to be a realistic acquisition target.",
    )
    penalty_reason = penalty_reason.mask(
        rural_meaningful & marketability_action.eq("keep"),
        "Geometry is irregular, but parcel size and width remain meaningful enough to keep in rural/default review.",
    )

    enriched["geometry_estimated_frontage_feet"] = frontage_feet.round(1)
    enriched["geometry_estimated_width_feet"] = width_feet.round(1)
    enriched["geometry_min_dimension_feet"] = min_dimension_feet.round(1)
    enriched["geometry_max_dimension_feet"] = max_dimension_feet.round(1)
    enriched["geometry_frontage_to_width_ratio"] = frontage_to_width_ratio.round(2)
    enriched["geometry_effective_buildable_flag"] = marketability_flag.isin(["marketable", "constrained"]) & marketability_action.ne("exclude")
    enriched["geometry_marketability_base_flag"] = base_marketability_flag
    enriched["geometry_marketability_flag"] = marketability_flag
    enriched["geometry_marketability_context"] = marketability_context
    enriched["geometry_marketability_action"] = marketability_action
    enriched["geometry_penalty_points"] = penalty_points.round(1)
    enriched["geometry_penalty_reason"] = penalty_reason
    enriched["geometry_marketability_default_leads_excluded_flag"] = marketability_action.eq("exclude")
    return enriched


def apply_geometry_marketability_score_adjustment(
    frame: pd.DataFrame,
    *,
    effective_score_column: str = "lead_score_total_effective",
) -> pd.DataFrame:
    adjusted = frame.copy()
    penalty_points = _float_series(adjusted, "geometry_penalty_points").fillna(0.0)
    base_effective = _float_series(adjusted, effective_score_column).fillna(_float_series(adjusted, "lead_score_total").fillna(0.0))
    adjusted[effective_score_column] = (base_effective + penalty_points).round(2)
    return adjusted


def filter_default_marketability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "geometry_marketability_default_leads_excluded_flag" in frame.columns:
        return frame.loc[~_bool_series(frame, "geometry_marketability_default_leads_excluded_flag")].copy()
    if "geometry_marketability_action" in frame.columns:
        return frame.loc[_string_series(frame, "geometry_marketability_action", default="keep").ne("exclude")].copy()
    if "geometry_marketability_flag" in frame.columns:
        return frame.loc[_string_series(frame, "geometry_marketability_flag", default="marketable").ne("unbuildable_candidate")].copy()
    return frame.copy()


def geometry_marketability_diagnostics(
    frame: pd.DataFrame,
    *,
    config: GeometryMarketabilityConfig | None = None,
) -> dict[str, Any]:
    config = config or DEFAULT_GEOMETRY_MARKETABILITY_CONFIG
    row_count = int(len(frame))
    base_flag_counts = _string_series(frame, "geometry_marketability_base_flag", default="unknown").value_counts(dropna=False).sort_index().to_dict()
    flag_counts = _string_series(frame, "geometry_marketability_flag", default="unknown").value_counts(dropna=False).sort_index().to_dict()
    action_counts = _string_series(frame, "geometry_marketability_action", default="unknown").value_counts(dropna=False).sort_index().to_dict()
    excluded = int(_bool_series(frame, "geometry_marketability_default_leads_excluded_flag").sum())
    base_flag_series = _string_series(frame, "geometry_marketability_base_flag", default="unknown")
    final_flag_series = _string_series(frame, "geometry_marketability_flag", default="unknown")
    action_series = _string_series(frame, "geometry_marketability_action", default="unknown")
    transition_counts = (
        (base_flag_series + "->" + final_flag_series)
        .loc[base_flag_series.ne(final_flag_series)]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    affected_frame = frame.loc[action_series.ne("keep")].copy()
    top_counties_affected: dict[str, int] = {}
    if "county_name" in affected_frame.columns and not affected_frame.empty:
        top_counties_affected = {
            str(key): int(value)
            for key, value in _string_series(affected_frame, "county_name", default="unknown").value_counts(dropna=False).head(10).to_dict().items()
        }
    example_columns = [
        column
        for column in [
            "parcel_row_id",
            "parcel_id",
            "county_name",
            "geometry_marketability_base_flag",
            "geometry_marketability_flag",
            "geometry_marketability_action",
            "geometry_marketability_context",
            "geometry_estimated_width_feet",
            "geometry_estimated_frontage_feet",
            "geometry_frontage_to_width_ratio",
            "area_acres",
            "compactness",
            "nearby_building_density",
            "lead_score_total_effective",
        ]
        if column in frame.columns
    ]

    def _example_rows(mask: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
        if not example_columns:
            return []
        subset = frame.loc[mask, example_columns].copy().head(limit)
        if subset.empty:
            return []
        records = subset.to_dict(orient="records")
        normalized: list[dict[str, Any]] = []
        for record in records:
            normalized.append(
                {
                    str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                    for key, value in record.items()
                }
            )
        return normalized

    return {
        "config": asdict(config),
        "row_count": row_count,
        "geometry_marketability_base_flag_counts": {str(key): int(value) for key, value in base_flag_counts.items()},
        "geometry_marketability_flag_counts": {str(key): int(value) for key, value in flag_counts.items()},
        "geometry_marketability_action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "geometry_marketability_flag_transition_counts": {str(key): int(value) for key, value in transition_counts.items()},
        "default_leads_excluded_count": excluded,
        "default_leads_excluded_pct": round((excluded / row_count) * 100.0, 2) if row_count else 0.0,
        "default_leads_excluded_by_flag": {
            str(key): int(value)
            for key, value in _string_series(
                frame.loc[_bool_series(frame, "geometry_marketability_default_leads_excluded_flag")],
                "geometry_marketability_flag",
                default="unknown",
            ).value_counts(dropna=False).sort_index().to_dict().items()
        },
        "top_counties_affected": top_counties_affected,
        "example_refined_flag_rows": _example_rows(base_flag_series.ne(final_flag_series)),
        "example_penalized_rows": _example_rows(action_series.eq("penalize")),
        "example_excluded_rows": _example_rows(action_series.eq("exclude")),
    }
