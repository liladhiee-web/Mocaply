"""Robust ground-plane fitting for recovered 3D motion.

The HQ worker samples the lowest sole vertices of the SMPL-X body mesh for a
subset of frames and passes them here. RANSAC finds the plane the subject
actually stood on, in the same world frame as the motion, so the client can
align gravity and place the character on the floor.

Plane convention: ``normal . point + d = 0`` with a unit normal whose Y
component is positive (pointing up). Pure standard library on purpose: it is
cheap, deterministic and unit-testable without numpy or torch.
"""

from __future__ import annotations

import math
import random

UP = (0.0, 1.0, 0.0)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(_dot(a, a))


def _normalize(a):
    n = _norm(a)
    if n < 1e-12:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def plane_distance(plane, point):
    """Signed distance of ``point`` to a ``(normal, d)`` plane."""
    n, d = plane
    return _dot(n, point) + d


def _plane_through(a, b, c):
    n = _normalize(_cross(_sub(b, a), _sub(c, a)))
    if n is None:
        return None
    if n[1] < 0:  # always point the same way as world up
        n = (-n[0], -n[1], -n[2])
    return n, -_dot(n, a)


def _fit_least_squares(points, seed):
    """Total-least-squares plane through ``points``; returns ``(normal, d)``.

    The smallest eigenvector of the covariance matrix is found by power
    iteration on ``trace(C)*I - C`` (whose largest eigenvector is that one),
    which avoids pulling in numpy for a 3x3 problem.
    """
    count = len(points)
    if count < 3:
        return None
    mx = sum(p[0] for p in points) / count
    my = sum(p[1] for p in points) / count
    mz = sum(p[2] for p in points) / count
    cxx = cyy = czz = cxy = cxz = cyz = 0.0
    for x, y, z in points:
        dx, dy, dz = x - mx, y - my, z - mz
        cxx += dx * dx
        cyy += dy * dy
        czz += dz * dz
        cxy += dx * dy
        cxz += dx * dz
        cyz += dy * dz
    trace = cxx + cyy + czz
    # B = trace*I - C
    b = (
        (trace - cxx, -cxy, -cxz),
        (-cxy, trace - cyy, -cyz),
        (-cxz, -cyz, trace - czz),
    )
    v = seed if seed and _norm(seed) > 1e-9 else (0.0, 1.0, 0.0)
    v = _normalize(v)
    if v is None:
        return None
    for _ in range(64):
        w = (
            b[0][0] * v[0] + b[0][1] * v[1] + b[0][2] * v[2],
            b[1][0] * v[0] + b[1][1] * v[1] + b[1][2] * v[2],
            b[2][0] * v[0] + b[2][1] * v[1] + b[2][2] * v[2],
        )
        nv = _normalize(w)
        if nv is None:
            return None
        if abs(nv[0] - v[0]) + abs(nv[1] - v[1]) + abs(nv[2] - v[2]) < 1e-10:
            v = nv
            break
        v = nv
    n = v
    if n[1] < 0:
        n = (-n[0], -n[1], -n[2])
    return n, -(n[0] * mx + n[1] * my + n[2] * mz)


def _clean(points):
    out = []
    for p in points or ():
        try:
            x, y, z = float(p[0]), float(p[1]), float(p[2])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            out.append((x, y, z))
    return out


def fit_ground_plane(
    points,
    threshold=0.02,
    iterations=256,
    seed=7,
    up_min=0.5,
    min_inliers=12,
    max_tilt_deg=35.0,
    min_ratio=0.2,
):
    """Fit the floor plane to sampled sole/contact points.

    Args:
        points: iterable of ``[x, y, z]`` candidate contact points (world).
        threshold: inlier distance in metres.
        iterations: RANSAC iterations.
        up_min: minimum ``normal.y`` for a candidate plane (rejects walls).
        min_inliers: minimum support for a valid plane.
        max_tilt_deg: reject fits further than this from world up.
        min_ratio: minimum inlier fraction before a fit is trusted.

    Returns:
        ``{"normal": [x, y, z], "d": float, "tilt_deg": float,
        "inlier_ratio": float, "rms": float, "threshold": float,
        "samples": int}`` or ``None`` when the points do not describe a floor.
    """
    pts = _clean(points)
    if len(pts) < min_inliers:
        return None

    rng = random.Random(seed)
    best = None  # (score tuple, normal, d, inliers)
    sample_n = len(pts)
    for _ in range(max(1, iterations)):
        a, b, c = rng.sample(range(sample_n), 3)
        plane = _plane_through(pts[a], pts[b], pts[c])
        if plane is None:
            continue
        n, d = plane
        if n[1] < up_min:
            continue
        inliers = [p for p in pts if abs(_dot(n, p) + d) <= threshold]
        if len(inliers) < min_inliers:
            continue
        refined = _fit_least_squares(inliers, n)
        if refined is None:
            continue
        n2, d2 = refined
        if n2[1] < up_min:
            continue
        inliers2 = [p for p in pts if abs(_dot(n2, p) + d2) <= threshold]
        if len(inliers2) < min_inliers:
            continue
        # Prefer more support, then the lower plane (larger d for an up normal).
        score = (len(inliers2), d2)
        if best is None or score > best[0]:
            best = (score, n2, d2, inliers2)

    if best is None:
        return None
    _, n, d, inliers = best
    ratio = len(inliers) / len(pts)
    if ratio < min_ratio:
        return None
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, n[1]))))
    if tilt > max_tilt_deg:
        return None
    rms = math.sqrt(sum((_dot(n, p) + d) ** 2 for p in inliers) / len(inliers))
    return {
        "normal": [round(n[0], 6), round(n[1], 6), round(n[2], 6)],
        "d": round(d, 6),
        "tilt_deg": round(tilt, 3),
        "inlier_ratio": round(ratio, 4),
        "rms": round(rms, 5),
        "threshold": threshold,
        "samples": len(pts),
    }
