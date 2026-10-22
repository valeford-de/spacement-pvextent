# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""CAD-grade geospatial vectorization and rectilinear regularization module."""

from .elevation import (
    DEFAULT_CO2_EMISSION_FACTOR_KG_PER_KWH,
    DEFAULT_GHI_KWH_M2,
    DEFAULT_SYSTEM_PERFORMANCE_RATIO,
    azimuth_to_cardinal,
    calculate_3d_surface_area,
    calculate_annual_solar_yield,
    fit_roof_plane_from_dsm,
)
from .georef import extract_georeference, extract_geotiff_transform, parse_worldfile
from .packing import fit_module_grid, point_in_polygon
from .metrics import (
    DEFAULT_GSD_M,
    DEFAULT_MODULE_EFFICIENCY,
    DEFAULT_MODULE_WATTAGE_KW,
    calculate_pv_metrics,
    polygon_area_pixels,
    polygon_centroid,
    polygon_perimeter_pixels,
)
from .regularize import (
    detect_dominant_orientation,
    minimum_bounding_box,
    regularize_manhattan,
    simplify_douglas_peucker,
)
from .temporal import (
    CHANGE_TYPE_DECOMMISSIONED,
    CHANGE_TYPE_EXPANSION,
    CHANGE_TYPE_NEW,
    CHANGE_TYPE_STABLE,
    calculate_temporal_deltas,
    detect_temporal_changes,
    polygon_iou_raster,
    render_temporal_cad_overlay,
)
from .vectorize import (
    marching_squares_contours,
    render_cad_overlay,
    transform_pixel_to_geo,
    vectorize_mask,
)

__all__ = [
    "CHANGE_TYPE_DECOMMISSIONED",
    "CHANGE_TYPE_EXPANSION",
    "CHANGE_TYPE_NEW",
    "CHANGE_TYPE_STABLE",
    "DEFAULT_CO2_EMISSION_FACTOR_KG_PER_KWH",
    "DEFAULT_GHI_KWH_M2",
    "DEFAULT_GSD_M",
    "DEFAULT_MODULE_EFFICIENCY",
    "DEFAULT_MODULE_WATTAGE_KW",
    "DEFAULT_SYSTEM_PERFORMANCE_RATIO",
    "azimuth_to_cardinal",
    "calculate_3d_surface_area",
    "calculate_annual_solar_yield",
    "calculate_pv_metrics",
    "calculate_temporal_deltas",
    "detect_dominant_orientation",
    "detect_temporal_changes",
    "extract_georeference",
    "extract_geotiff_transform",
    "fit_module_grid",
    "fit_roof_plane_from_dsm",
    "marching_squares_contours",
    "minimum_bounding_box",
    "polygon_area_pixels",
    "polygon_centroid",
    "polygon_perimeter_pixels",
    "polygon_iou_raster",
    "regularize_manhattan",
    "render_cad_overlay",
    "render_temporal_cad_overlay",
    "simplify_douglas_peucker",
    "transform_pixel_to_geo",
    "vectorize_mask",
]
