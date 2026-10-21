# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""CAD-grade geospatial vectorization and rectilinear regularization module."""

from .georef import extract_georeference, extract_geotiff_transform, parse_worldfile
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
from .vectorize import (
    marching_squares_contours,
    render_cad_overlay,
    transform_pixel_to_geo,
    vectorize_mask,
)

__all__ = [
    "DEFAULT_GSD_M",
    "DEFAULT_MODULE_EFFICIENCY",
    "DEFAULT_MODULE_WATTAGE_KW",
    "calculate_pv_metrics",
    "detect_dominant_orientation",
    "extract_georeference",
    "extract_geotiff_transform",
    "marching_squares_contours",
    "minimum_bounding_box",
    "polygon_area_pixels",
    "polygon_centroid",
    "polygon_perimeter_pixels",
    "regularize_manhattan",
    "render_cad_overlay",
    "simplify_douglas_peucker",
    "transform_pixel_to_geo",
    "vectorize_mask",
]
