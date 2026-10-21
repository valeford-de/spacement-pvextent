# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Georeferencing extraction from GeoTIFF tags, Worldfiles (.tfw/.pgw), and metadata sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
from PIL import Image


def parse_worldfile(path: Path | str) -> list[float] | None:
    """Parse a standard 6-line ESRI / GDAL Worldfile (.tfw, .pgw, .wld).

    Format:
        Line 1: A (pixel size in x direction)
        Line 2: D (rotation about y axis)
        Line 3: B (rotation about x axis)
        Line 4: E (pixel size in y direction, typically negative)
        Line 5: C (x coordinate of upper left pixel center)
        Line 6: F (y coordinate of upper left pixel center)

    Returns:
        GDAL 6-parameter affine transform [A, B, C, D, E, F]
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) >= 6:
            a, d, b, e, c, f = [float(x) for x in lines[:6]]
            return [round(a, 6), round(b, 6), round(c, 3), round(d, 6), round(e, 6), round(f, 3)]
    except Exception:
        return None
    return None


def extract_geotiff_transform(img: Image.Image) -> list[float] | None:
    """Extract GDAL affine transform [a, b, c, d, e, f] from GeoTIFF ModelPixelScaleTag and ModelTiepointTag."""
    try:
        tags = getattr(img, "tag_v2", getattr(img, "tag", None))
        if not tags:
            return None

        # Tag 33550: ModelPixelScaleTag (scale_x, scale_y, scale_z)
        # Tag 33922: ModelTiepointTag (i, j, k, x, y, z)
        if 33550 in tags and 33922 in tags:
            scale = tags[33550]
            tiepoint = tags[33922]
            sx, sy = float(scale[0]), float(scale[1])
            px, py = float(tiepoint[0]), float(tiepoint[1])
            gx, gy = float(tiepoint[3]), float(tiepoint[4])

            # Affine: X_geo = c + x * a + y * b, Y_geo = f + x * d + y * e
            c = gx - px * sx
            f = gy + py * sy
            return [round(sx, 6), 0.0, round(c, 3), 0.0, round(-sy, 6), round(f, 3)]
    except Exception:
        return None
    return None


def extract_georeference(
    image_path: Path | str,
    img: Image.Image | None = None,
    default_crs: str = "EPSG:25832",
) -> dict[str, Any] | None:
    """Automatically resolve geospatial georeference metadata from GeoTIFF headers, worldfiles, or JSON sidecars.

    Search priority:
    1. Embedded GeoTIFF tags (ModelPixelScaleTag 33550, ModelTiepointTag 33922).
    2. Companion Worldfile (.tfw, .pgw, .wld).
    3. Companion JSON sidecar (<basename>.json, <basename>.georef.json, or metadata.json).

    Returns:
        dict: {'crs': str, 'transform': [a, b, c, d, e, f]} or None
    """
    path = Path(image_path).resolve()

    # 1. Check embedded GeoTIFF tags
    if img is not None:
        transform = extract_geotiff_transform(img)
        if transform:
            return {"crs": default_crs, "transform": transform}
    elif path.suffix.lower() in (".tif", ".tiff") and path.is_file():
        try:
            with Image.open(path) as opened_img:
                transform = extract_geotiff_transform(opened_img)
                if transform:
                    return {"crs": default_crs, "transform": transform}
        except Exception:
            pass

    # 2. Check Companion Worldfile
    worldfile_candidates = [
        path.with_suffix(".tfw"),
        path.with_suffix(".pgw"),
        path.with_suffix(".wld"),
        path.parent / (path.stem + ".wld"),
    ]
    for wf in worldfile_candidates:
        if wf.is_file():
            tf = parse_worldfile(wf)
            if tf:
                return {"crs": default_crs, "transform": tf}

    # 3. Check Companion JSON sidecar
    json_candidates = [
        path.with_suffix(".json"),
        path.with_suffix(".georef.json"),
        path.parent / "metadata.json",
    ]
    for jf in json_candidates:
        if jf.is_file():
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "georeference" in data and isinstance(data["georeference"], dict):
                        return {
                            "crs": data["georeference"].get("crs", default_crs),
                            "transform": data["georeference"].get("transform"),
                        }
                    if "transform" in data and isinstance(data["transform"], list):
                        return {
                            "crs": data.get("crs", default_crs),
                            "transform": data["transform"],
                        }
            except Exception:
                pass

    return None
