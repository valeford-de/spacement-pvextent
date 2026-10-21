# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""CAD-grade polygon simplification, dominant angle detection, and rectilinear regularization."""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def _point_line_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    """Perpendicular distance from a point to a line segment."""
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    line_len_sq = dx * dx + dy * dy
    if line_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    # Project point onto line: cross product divided by length
    return abs(dy * px - dx * py + x2 * y1 - y2 * x1) / math.sqrt(line_len_sq)


def simplify_douglas_peucker(
    vertices: Sequence[tuple[float, float]],
    tolerance: float = 1.0,
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker algorithm for polygonal boundary simplification.

    Preserves closed polygon ring topology while pruning collinear and near-collinear points.
    """
    if len(vertices) <= 3:
        return list(vertices)

    # Ensure closed ring
    is_closed = math.hypot(vertices[0][0] - vertices[-1][0], vertices[0][1] - vertices[-1][1]) < 1e-5
    ring = list(vertices) if is_closed else list(vertices) + [vertices[0]]

    def _rdp(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(pts) <= 2:
            return pts
        dmax = 0.0
        index = 0
        start, end = pts[0], pts[-1]
        for i in range(1, len(pts) - 1):
            d = _point_line_distance(pts[i], start, end)
            if d > dmax:
                index = i
                dmax = d
        if dmax > tolerance:
            left = _rdp(pts[:index + 1])
            right = _rdp(pts[index:])
            return left[:-1] + right
        return [start, end]

    # Find the two furthest points on the ring to split into two paths
    n = len(ring) - 1
    max_d = 0.0
    split_idx = n // 2
    for i in range(1, n):
        d = math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1])
        if d > max_d:
            max_d = d
            split_idx = i

    path1 = _rdp(ring[:split_idx + 1])
    path2 = _rdp(ring[split_idx:])

    simplified = path1[:-1] + path2
    if len(simplified) >= 4 and math.hypot(simplified[0][0] - simplified[-1][0], simplified[0][1] - simplified[-1][1]) < 1e-5:
        simplified = simplified[:-1]

    return simplified


def detect_dominant_orientation(vertices: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Calculate the dominant orientation of a polygon using 4-fold length-weighted circular mean.

    Returns:
        tuple[float, float]: (dominant_angle_rad in [0, pi/2), dominant_azimuth_deg in [0, 360))
    """
    n = len(vertices)
    if n < 2:
        return (0.0, 0.0)

    # Accumulate length-weighted vectors in 4-fold folded space
    sum_cos4 = 0.0
    sum_sin4 = 0.0
    total_len = 0.0

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-4:
            continue
        theta = math.atan2(dy, dx)
        # 4*theta maps 90-degree orthogonal edges (0, 90, 180, 270) to 0
        sum_cos4 += length * math.cos(4.0 * theta)
        sum_sin4 += length * math.sin(4.0 * theta)
        total_len += length

    if total_len < 1e-4 or math.hypot(sum_cos4, sum_sin4) < 1e-4:
        return (0.0, 0.0)

    # Folded orientation in [0, pi/2)
    angle_4 = math.atan2(sum_sin4, sum_cos4)
    dominant_rad = (angle_4 / 4.0) % (math.pi / 2.0)
    if abs(dominant_rad - math.pi / 2.0) < 1e-4 or abs(dominant_rad) < 1e-4:
        dominant_rad = 0.0

    # Geographic azimuth: 0 deg = North (up, -y in image), 90 deg = East (+x)
    # Standard math theta is from +x axis.
    # In aerial images, y increases downward.
    compass_deg = (90.0 - math.degrees(dominant_rad)) % 180.0
    # For solar panels, preferred tilt azimuth faces south (around 180)
    azimuth_deg = compass_deg if compass_deg >= 90.0 else (compass_deg + 180.0) % 360.0

    return (dominant_rad, azimuth_deg)


def minimum_bounding_box(vertices: Sequence[tuple[float, float]]) -> dict:
    """Compute the minimum-area oriented bounding box (OBB) for the polygon.

    Returns:
        dict: {
            'corners': list of 4 (x, y) points,
            'width': float,
            'height': float,
            'aspect_ratio': float,
            'angle_rad': float,
            'area': float,
        }
    """
    pts = np.array(vertices, dtype=np.float64)
    if len(pts) < 3:
        return {"corners": list(vertices), "width": 0.0, "height": 0.0, "aspect_ratio": 1.0, "angle_rad": 0.0, "area": 0.0}

    # Test candidate angles aligned with polygon edges
    best_area = math.inf
    best_box = None
    best_dims = (0.0, 0.0)
    best_angle = 0.0

    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1e-4:
            continue
        angle = math.atan2(dy, dx)
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)

        # Rotate points
        rx = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        ry = pts[:, 0] * sin_a + pts[:, 1] * cos_a

        min_x, max_x = rx.min(), rx.max()
        min_y, max_y = ry.min(), ry.max()
        w = max_x - min_x
        h = max_y - min_y
        area = w * h

        if area < best_area:
            best_area = area
            best_dims = (w, h)
            best_angle = angle
            # Unrotate box corners
            box_r = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ])
            cos_b = math.cos(angle)
            sin_b = math.sin(angle)
            corners_x = box_r[:, 0] * cos_b - box_r[:, 1] * sin_b
            corners_y = box_r[:, 0] * sin_b + box_r[:, 1] * cos_b
            best_box = [(float(corners_x[k]), float(corners_y[k])) for k in range(4)]

    w, h = best_dims
    aspect = max(w, h) / max(min(w, h), 1e-6)
    return {
        "corners": best_box or list(vertices),
        "width": float(min(w, h)),
        "length": float(max(w, h)),
        "aspect_ratio": float(aspect),
        "angle_rad": float(best_angle),
        "area": float(best_area),
    }


def regularize_manhattan(
    vertices: Sequence[tuple[float, float]],
    tolerance_deg: float = 20.0,
    force_box: bool = False,
) -> list[tuple[float, float]]:
    """Regularize polygon edges into clean orthogonal 90-degree CAD lines aligned with dominant roof azimuth.

    Args:
        vertices: Sequence of 2D polygon vertices.
        tolerance_deg: Maximum angular deviation from 90°/180° allowed for snapping.
        force_box: If True, returns the minimum-area oriented bounding box directly.

    Returns:
        list[tuple[float, float]]: Clean CAD-grade rectilinear polygon vertices.
    """
    if len(vertices) < 3:
        return list(vertices)

    if force_box:
        obb = minimum_bounding_box(vertices)
        return obb["corners"]

    # 1. Simplify first with modest tolerance to eliminate pixel staircase
    simplified = simplify_douglas_peucker(vertices, tolerance=1.5)
    if len(simplified) < 4:
        return minimum_bounding_box(vertices)["corners"]

    # 2. Detect dominant orientation
    theta_dom, _ = detect_dominant_orientation(simplified)

    # 3. Rotate polygon into canonical frame aligned with dominant angle
    cos_t = math.cos(-theta_dom)
    sin_t = math.sin(-theta_dom)

    rot_pts = []
    for x, y in simplified:
        rx = x * cos_t - y * sin_t
        ry = x * sin_t + y * cos_t
        rot_pts.append((rx, ry))

    # 4. Snap near-orthogonal edges to strict horizontal/vertical lines
    n = len(rot_pts)
    tol_rad = math.radians(tolerance_deg)
    
    # Classify each segment: 'H' (horizontal), 'V' (vertical), or 'D' (diagonal/free)
    seg_types = []
    for i in range(n):
        x1, y1 = rot_pts[i]
        x2, y2 = rot_pts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        angle = math.atan2(abs(dy), abs(dx))
        if angle <= tol_rad or angle >= (math.pi - tol_rad):
            seg_types.append("H")
        elif abs(angle - math.pi / 2.0) <= tol_rad:
            seg_types.append("V")
        else:
            seg_types.append("D")

    # If all or most segments are H and V, regularize vertices by alternating intersections
    h_count = seg_types.count("H")
    v_count = seg_types.count("V")
    if (h_count + v_count) / max(n, 1) < 0.6:
        # Fall back to minimum bounding box if shape is not naturally rectilinear
        return minimum_bounding_box(vertices)["corners"]

    regularized_rot = []
    for i in range(n):
        prev_i = (i - 1) % n
        prev_type = seg_types[prev_i]
        curr_type = seg_types[i]
        p_prev = rot_pts[prev_i]
        p_curr = rot_pts[i]
        p_next = rot_pts[(i + 1) % n]

        if prev_type == "H" and curr_type == "V":
            # Intersection of horizontal line (y = p_curr[1]) and vertical line (x = p_curr[0])
            regularized_rot.append((p_curr[0], p_prev[1]))
        elif prev_type == "V" and curr_type == "H":
            regularized_rot.append((p_prev[0], p_curr[1]))
        else:
            regularized_rot.append(p_curr)

    # 5. Rotate back to original frame
    cos_back = math.cos(theta_dom)
    sin_back = math.sin(theta_dom)

    out = []
    for rx, ry in regularized_rot:
        ox = rx * cos_back - ry * sin_back
        oy = rx * sin_back + ry * cos_back
        out.append((round(ox, 2), round(oy, 2)))

    # Deduplicate consecutive identical vertices
    clean = []
    for p in out:
        if not clean or math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 0.5:
            clean.append(p)

    if len(clean) >= 3:
        return clean
    return minimum_bounding_box(vertices)["corners"]
