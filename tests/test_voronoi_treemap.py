"""Checks that the code does what the README proves the mathematics does."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import check_grad
from scipy.spatial import Voronoi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voronoi_treemap as vt

DISC = vt.circle((0.0, 0.0), 1.0)
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def is_convex(poly):
    """Every turn is a left turn (allowing straight runs)."""
    a = np.roll(poly, -1, axis=0) - poly
    b = np.roll(a, -1, axis=0)
    return np.all(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] > -1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_areas_match_weights_and_cells_tile_the_boundary(seed):
    rng = np.random.default_rng(seed)
    weights = rng.lognormal(0.0, 1.0, size=20)          # skewed, like donations
    cells, info = vt.voronoi_treemap(weights, DISC, seed=seed, rounds=10, return_info=True)
    areas = np.array([vt.polygon_area(c) for c in cells])
    share = weights / weights.sum()
    assert np.max(np.abs(areas / areas.sum() - share) / share) < 1e-3
    assert areas.sum() == pytest.approx(vt.polygon_area(DISC), rel=1e-9)   # no gaps, no overlap
    assert all(is_convex(c) for c in cells)


def test_same_seed_gives_identical_layout():
    a = vt.voronoi_treemap([5, 3, 2, 1], SQUARE, seed=7, rounds=5)
    b = vt.voronoi_treemap([5, 3, 2, 1], SQUARE, seed=7, rounds=5)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))


def test_gradient_is_area_minus_target():
    """Analytic gradient of -F against finite differences."""
    rng = np.random.default_rng(1)
    sites = vt.random_points_in(SQUARE, 8, rng)
    targets = np.full(8, 1 / 8)
    w = rng.normal(0, 0.01, 8)
    f = lambda x: vt.objective(x, sites, targets, SQUARE)[0]
    g = lambda x: vt.objective(x, sites, targets, SQUARE)[1]
    # Forward differences are only accurate to about epsilon * curvature, so the
    # bound is loose; a sign or indexing mistake would give an error near 0.1.
    assert check_grad(f, g, w, epsilon=1e-6) < 1e-4


def test_equal_weights_reproduce_scipy_voronoi():
    """With all weights equal, a power diagram is the ordinary Voronoi diagram."""
    rng = np.random.default_rng(3)
    sites = rng.uniform(0.0, 1.0, size=(30, 2))
    big = (SQUARE - 0.5) * 20 + 0.5                       # far outside every finite cell
    ours = vt.power_cells(sites, np.zeros(30), big)
    ref = Voronoi(sites)
    compared = 0
    for i, region_index in enumerate(ref.point_region):
        region = ref.regions[region_index]
        if -1 in region:                                  # unbounded cell: skip
            continue
        theirs = ref.vertices[region]
        if np.abs(theirs - 0.5).max() > 9:                # reaches our clip box: skip
            continue
        key = lambda p: sorted(map(tuple, np.round(p, 6)))
        assert key(ours[i]) == key(theirs)
        compared += 1
    assert compared >= 10


def test_second_moment_of_unit_square_about_its_corner():
    # integral of x^2 + y^2 over [0,1]^2 = 2/3
    assert vt.second_moment(SQUARE, np.zeros(2)) == pytest.approx(2 / 3)
