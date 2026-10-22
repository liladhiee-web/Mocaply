"""Unit tests for hq/ground.py (pure stdlib; run with `python hq/test_ground.py`)."""

from __future__ import annotations

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ground import fit_ground_plane, plane_distance  # noqa: E402


def normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def basis(n):
    helper = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 0.0, 1.0)
    u = normalize(
        (
            helper[1] * n[2] - helper[2] * n[1],
            helper[2] * n[0] - helper[0] * n[2],
            helper[0] * n[1] - helper[1] * n[0],
        )
    )
    v = (
        n[1] * u[2] - n[2] * u[1],
        n[2] * u[0] - n[0] * u[2],
        n[0] * u[1] - n[1] * u[0],
    )
    return u, v


def plane_points(normal, d, count=200, noise=0.004, seed=3):
    n = normalize(normal)
    u, v = basis(n)
    origin = tuple(-d * c for c in n)
    rng = random.Random(seed)
    pts = []
    for _ in range(count):
        a = rng.uniform(-0.6, 0.6)
        b = rng.uniform(-0.6, 0.6)
        pts.append(
            tuple(
                origin[i] + u[i] * a + v[i] * b + rng.gauss(0, noise)
                for i in range(3)
            )
        )
    return pts, n


class FitGroundPlaneTests(unittest.TestCase):
    def test_flat_floor(self):
        pts, _ = plane_points((0, 1, 0), -0.02)
        fit = fit_ground_plane(pts, iterations=200, seed=11)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["normal"][1], 1.0, places=2)
        self.assertAlmostEqual(fit["d"], -0.02, delta=0.01)
        self.assertGreater(fit["inlier_ratio"], 0.9)
        self.assertLess(fit["tilt_deg"], 2.0)

    def test_tilted_floor(self):
        normal = normalize((0.12, 1.0, -0.06))
        pts, _ = plane_points(normal, 0.35)
        fit = fit_ground_plane(pts, iterations=300, seed=5)
        self.assertIsNotNone(fit)
        got = fit["normal"]
        dot = sum(a * b for a, b in zip(got, normal))
        self.assertGreater(dot, 0.995)
        for p in pts[:5]:
            self.assertLess(abs(plane_distance((tuple(got), fit["d"]), p)), 0.01)

    def test_outliers_do_not_win(self):
        pts, _ = plane_points((0, 1, 0), 0.0, count=160)
        rng = random.Random(9)
        pts += [(-0.2 + rng.random() * 0.4, 0.9 + rng.random() * 0.4, rng.random() * 0.4) for _ in range(60)]
        fit = fit_ground_plane(pts, iterations=300, seed=2)
        self.assertIsNotNone(fit)
        self.assertLess(fit["tilt_deg"], 2.0)
        self.assertLess(abs(fit["d"]), 0.01)

    def test_too_few_points(self):
        self.assertIsNone(fit_ground_plane([(0, 0, 0), (1, 0, 0), (0, 0, 1)]))

    def test_collinear_points(self):
        pts = [(i * 0.1, 0.0, 0.0) for i in range(30)]
        self.assertIsNone(fit_ground_plane(pts, iterations=64, seed=1))

    def test_steep_fit_is_rejected(self):
        normal = normalize((0.0, math.cos(math.radians(55)), math.sin(math.radians(55))))
        pts, _ = plane_points(normal, 0.0, count=120)
        self.assertIsNone(fit_ground_plane(pts, iterations=128, seed=4))

    def test_vertical_wall_is_rejected(self):
        pts = [(0.0, y, z) for y in range(20) for z in range(20)]
        self.assertIsNone(fit_ground_plane(pts, iterations=128, seed=6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
