"""Voronoi treemaps: split a convex region into convex cells of prescribed areas.

The cells are those of a *power diagram* (a Voronoi diagram in which each site
also carries a weight). Raising a site's weight grows its cell, so the layout
problem is: find the weights that make every cell's area hit its target.
That is a concave maximisation problem, which SciPy solves for us.

README.md walks through the mathematics and proves each step; function names
here match the section names there.

Conventions: a polygon is an (n, 2) float array of vertices in counter-clockwise
order, y pointing up. An empty cell is an array of shape (0, 2).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EMPTY = np.zeros((0, 2))


# --------------------------------------------------------------------------
# Polygon measurements (shoelace formulas)
# --------------------------------------------------------------------------

def polygon_area(poly: np.ndarray) -> float:
    """Area of a counter-clockwise polygon: 1/2 * sum of cross(p_k, p_k+1)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Centre of mass of a polygon of uniform density."""
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = 0.5 * cross.sum()
    return np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6.0 * area)


def second_moment(poly: np.ndarray, point: np.ndarray) -> float:
    """Integral of |x - point|^2 over the polygon.

    Split the polygon into a fan of triangles (point, p_k, p_k+1). With the fan
    apex at the origin, a triangle with other corners p and q and signed area T
    contributes  T/6 * (|p|^2 + |q|^2 + p.q).  Signed areas make the fan correct
    even when `point` lies outside the polygon.
    """
    if len(poly) < 3:
        return 0.0
    p = poly - point
    q = np.roll(p, -1, axis=0)
    signed_area = 0.5 * (p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1])
    return float(np.sum(signed_area / 6.0 * ((p * p).sum(1) + (q * q).sum(1) + (p * q).sum(1))))


def circle(center, radius: float, n: int = 180) -> np.ndarray:
    """A regular n-gon standing in for a disc (counter-clockwise)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


# --------------------------------------------------------------------------
# Power diagram by half-plane clipping
# --------------------------------------------------------------------------

def clip_halfplane(poly: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Part of a convex polygon where  normal . x <= offset  (Sutherland-Hodgman).

    Walk the edges once. Keep every vertex on the allowed side; wherever an
    edge crosses the line, add the crossing point. For a convex polygon the
    result is exactly the intersection, and is again convex.
    """
    if len(poly) == 0:
        return poly
    side = poly @ normal - offset          # <= 0 means "allowed side"
    inside = side <= 0.0
    if inside.all():
        return poly
    if not inside.any():
        return EMPTY
    out = []
    n = len(poly)
    for k in range(n):
        m = (k + 1) % n
        if inside[k]:
            out.append(poly[k])
        if inside[k] != inside[m]:
            # Point on edge k->m where side == 0, by linear interpolation.
            t = side[k] / (side[k] - side[m])
            out.append(poly[k] + t * (poly[m] - poly[k]))
    return np.array(out)


def power_cells(sites: np.ndarray, weights: np.ndarray, boundary: np.ndarray) -> list[np.ndarray]:
    """Power diagram of weighted sites, restricted to a convex boundary.

    Cell i is every x in the boundary that is closer to site i than to any
    other site in *power distance*  |x - s_i|^2 - w_i.  Comparing sites i and j,
    the |x|^2 terms cancel and the condition becomes linear in x:

        2 (s_j - s_i) . x  <=  |s_j|^2 - |s_i|^2 + w_i - w_j

    so each cell is the boundary clipped by one half-plane per other site.
    """
    sq = (sites * sites).sum(axis=1)
    cells = []
    for i in range(len(sites)):
        cell = boundary
        # Clip against near neighbours first: the cell shrinks fast and most of
        # the remaining clips return immediately.
        order = np.argsort(((sites - sites[i]) ** 2).sum(axis=1))
        for j in order:
            if j == i:
                continue
            cell = clip_halfplane(cell, 2.0 * (sites[j] - sites[i]),
                                  sq[j] - sq[i] + weights[i] - weights[j])
            if len(cell) == 0:
                break
        cells.append(cell)
    return cells


# --------------------------------------------------------------------------
# Weight solve: make the cell areas equal the targets
# --------------------------------------------------------------------------

def objective(weights, sites, targets, boundary):
    """Return (-F, -grad F) for the concave function F described in the README.

        F(w) = sum_i  integral over C_i(w) of (|x - s_i|^2 - w_i) dx  +  sum_i w_i * target_i
        dF/dw_i = target_i - area(C_i(w))

    The gradient vanishes exactly when every cell has its target area, and F is
    concave, so that point is the global maximum. SciPy minimises, hence the
    minus signs.
    """
    cells = power_cells(sites, weights, boundary)
    areas = np.array([polygon_area(c) for c in cells])
    transport = sum(second_moment(c, s) for c, s in zip(cells, sites))
    F = transport - weights @ areas + weights @ targets
    return -F, areas - targets


def solve_weights(sites, targets, boundary, w0=None, history=None):
    """Weights whose power cells have the target areas, for fixed sites.

    `history`, if given, is a list that receives the worst relative area error
    after each optimiser iteration (used for the convergence plot in the primer).
    """
    w0 = np.zeros(len(sites)) if w0 is None else w0
    # The cells always fill the boundary, so the targets must add up to its
    # area. If they do not, no weights can satisfy them and F has no maximum:
    # the optimiser would push every weight towards infinity.
    if not np.isclose(np.sum(targets), polygon_area(boundary), rtol=1e-9):
        raise ValueError("targets must sum to the area of the boundary")

    def record(w):
        if history is not None:
            areas = np.array([polygon_area(c) for c in power_cells(sites, w, boundary)])
            history.append(float(np.max(np.abs(areas - targets) / targets)))

    record(w0)
    result = minimize(objective, w0, args=(sites, targets, boundary), jac=True,
                      method="L-BFGS-B", callback=record,
                      options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-9})
    # Adding one constant to every weight changes nothing (it cancels in the
    # inequality above), so pin the mean at zero to keep numbers small.
    return result.x - result.x.mean()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def random_points_in(boundary: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniform random points inside a convex polygon (rejection sampling)."""
    lo, hi = boundary.min(axis=0), boundary.max(axis=0)
    edges = np.roll(boundary, -1, axis=0) - boundary
    points = []
    while len(points) < n:
        p = rng.uniform(lo, hi)
        rel = p - boundary
        # Inside a counter-clockwise convex polygon = left of every edge.
        if np.all(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0] > 0):
            points.append(p)
    return np.array(points)


def voronoi_treemap(weights, boundary, seed: int = 0, sites=None,
                    rounds: int = 40, tol: float = 1e-3, return_info: bool = False):
    """Split `boundary` into convex cells with areas proportional to `weights`.

    weights   positive numbers, one per cell
    boundary  convex counter-clockwise polygon, (n, 2)
    seed      fixes the random starting sites, so the layout is reproducible
    sites     optional starting sites, (len(weights), 2); overrides `seed`
    rounds    number of "move each site to its cell's centroid" passes. They
              only make cells rounder; areas are solved afresh after each move
    tol       largest relative area error allowed; exceeding it raises

    Returns a list of polygons, in the order of `weights`. With
    return_info=True also returns a dict (sites, cell weights, max_error).
    """
    weights = np.asarray(weights, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")

    # Work in normalised coordinates (boundary centred, area 1) so tolerances
    # mean the same thing whatever the caller's units are.
    origin = polygon_centroid(boundary)
    scale = np.sqrt(polygon_area(boundary))
    unit = (boundary - origin) / scale
    targets = weights / weights.sum()            # areas sum to the unit area

    if sites is None:
        s = random_points_in(unit, len(weights), np.random.default_rng(seed))
    else:
        s = (np.asarray(sites, dtype=float) - origin) / scale

    w = np.zeros(len(weights))
    for r in range(rounds + 1):
        w = solve_weights(s, targets, unit, w0=w)
        cells = power_cells(s, w, unit)
        if r < rounds:
            s = np.array([polygon_centroid(c) for c in cells])

    areas = np.array([polygon_area(c) for c in cells])
    max_error = float(np.max(np.abs(areas - targets) / targets))
    if max_error > tol:
        raise RuntimeError(f"area error {max_error:.2%} exceeds tolerance {tol:.2%}")

    polygons = [c * scale + origin for c in cells]
    if return_info:
        return polygons, {"sites": s * scale + origin, "cell_weights": w * scale ** 2,
                          "max_error": max_error}
    return polygons
