"""Build docs/scipy_primer.pdf: a two-page introduction to the parts of SciPy this project uses.

Usage (from the repository root):  python docs/make_primer.py

Every figure in the primer is produced by running the code it illustrates,
so the pictures cannot drift from what SciPy actually does.
"""
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy.optimize import minimize, rosen, rosen_der
from scipy.spatial import Voronoi, voronoi_plot_2d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import voronoi_treemap as vt

PAGE = (8.5, 11.0)                      # US Letter, inches
LEFT, RIGHT = 0.09, 0.91                # page margins as figure fractions
BODY, MONO = "Inter", "DejaVu Sans Mono"
INK, MUTED, ACCENT, BLUE = "#1f1f1f", "#666666", "#b3122d", "#4f7d92"
LINE = 0.0158                           # body line height, figure fraction


# ---- small typesetting helpers -------------------------------------------

def heading(fig, y, text, size=13):
    fig.text(LEFT, y, text, fontsize=size, fontfamily=BODY, fontweight="bold", color=ACCENT, va="top")
    return y - 0.028


def para(fig, y, text, x=LEFT, chars=108, size=9.6, color=INK):
    """Wrapped paragraph starting at height y; returns the height below it."""
    for line in textwrap.wrap(" ".join(text.split()), chars):
        fig.text(x, y, line, fontsize=size, fontfamily=BODY, color=color, va="top")
        y -= LINE
    return y - 0.008


def code(fig, y, source, x=LEFT, width=0.44):
    """Monospace block on a grey panel; returns the height below it."""
    lines = textwrap.dedent(source).strip("\n").split("\n")
    height = len(lines) * 0.0148 + 0.014
    fig.add_artist(FancyBboxPatch((x, y - height), width, height, boxstyle="round,pad=0,rounding_size=0.004",
                                  facecolor="#f2f2f2", edgecolor="#dddddd", linewidth=0.6,
                                  transform=fig.transFigure))
    for i, line in enumerate(lines):
        fig.text(x + 0.012, y - 0.009 - i * 0.0148, line, fontsize=7.6, fontfamily=MONO, color=INK, va="top")
    return y - height - 0.016


def caption(ax, text):
    ax.set_title(text, fontsize=8.2, fontfamily=BODY, color=MUTED, loc="left", pad=6)


def tidy(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#cccccc")


def footer(fig, n):
    fig.text(LEFT, 0.028, "SciPy primer for the midterm donor chart", fontsize=7.5, fontfamily=BODY, color=MUTED)
    fig.text(RIGHT, 0.028, f"{n} / 2", fontsize=7.5, fontfamily=BODY, color=MUTED, ha="right")


# ---- page 1: what SciPy is ------------------------------------------------

def page_one():
    fig = plt.figure(figsize=PAGE)
    fig.text(LEFT, 0.962, "SciPy in ten minutes", fontsize=24, fontfamily=BODY, fontweight="bold", color=INK, va="top")
    fig.text(LEFT, 0.924, "What it is, and the two pieces this project uses", fontsize=11.5, fontfamily=BODY,
             color=MUTED, va="top")

    y = heading(fig, 0.885, "NumPy holds the numbers. SciPy does things with them.")
    y = para(fig, y, """NumPy gives you arrays and fast arithmetic on them. SciPy is a library of ready-made
        numerical algorithms that take NumPy arrays in and hand NumPy arrays back: optimisation, geometry,
        integration, statistics, signal and image processing. You import the one submodule you need. There is
        nothing to set up and no new data type to learn. The habit to build is: before writing a numerical
        routine yourself, check whether SciPy already has a tested one.""")

    # --- minimize ---
    y = heading(fig, y - 0.01, "1.  scipy.optimize.minimize: find the lowest point of a function")
    top = y
    y = para(fig, y, """You give it a function of a vector, a starting guess, and ideally the gradient (the vector
        of slopes). It walks downhill and returns the lowest point it finds. With jac=True your function returns
        the value and the gradient together.""", chars=52)
    y = code(fig, y, """
        from scipy.optimize import minimize

        def f(v):                       # value, gradient
            x, y = v
            value = (1-x)**2 + 100*(y-x*x)**2
            grad  = [-2*(1-x) - 400*x*(y-x*x),
                     200*(y-x*x)]
            return value, grad

        res = minimize(f, x0=[-1.2, 1.0],
                       jac=True, method="L-BFGS-B")
        res.x        # -> [1.0, 1.0]
        """)
    y = para(fig, y, """The method "L-BFGS-B" is a good default when you have a gradient: it learns the
        curvature of the function from the last few steps, so it needs far fewer steps than plain
        gradient descent.""", chars=52)

    # Figure: the optimiser's actual path on that function (the Rosenbrock "banana").
    path = [np.array([-1.2, 1.0])]
    res = minimize(rosen, path[0], jac=rosen_der, method="L-BFGS-B", callback=lambda v: path.append(v.copy()))
    path = np.array(path)
    ax = fig.add_axes([0.58, top - 0.29, 0.33, 0.25])
    gx, gy = np.meshgrid(np.linspace(-1.6, 1.4, 300), np.linspace(-0.5, 1.6, 300))
    ax.contour(gx, gy, (1 - gx) ** 2 + 100 * (gy - gx ** 2) ** 2, levels=np.logspace(-0.5, 3, 12),
               colors="#b9c7cf", linewidths=0.7)
    ax.plot(path[:, 0], path[:, 1], "-o", color=ACCENT, markersize=2.6, linewidth=1.0)
    ax.plot(*path[0], "s", color=INK, markersize=5), ax.plot(1, 1, "*", color=INK, markersize=11)
    ax.annotate("start", path[0], xytext=(-6, 8), textcoords="offset points", fontsize=8, fontfamily=BODY, ha="right")
    ax.annotate("minimum", (1, 1), xytext=(-46, 6), textcoords="offset points", fontsize=8, fontfamily=BODY)
    tidy(ax)
    caption(ax, f"The path minimize took: {res.nit} steps along a curved valley.\nGrey lines join points of equal height.")

    # --- Voronoi ---
    y = heading(fig, min(y, top - 0.30) - 0.006, "2.  scipy.spatial.Voronoi: who is nearest?")
    top = y
    y = para(fig, y, """Give it points ("sites"). It returns the map in which every location belongs to its
        nearest site. Each site's territory is a convex polygon called its cell.""", chars=52)
    y = code(fig, y, """
        from scipy.spatial import Voronoi

        sites = rng.uniform(0, 1, size=(12, 2))
        vor = Voronoi(sites)
        vor.vertices      # corner coordinates
        vor.regions       # corners of each cell
        vor.point_region  # which cell is site i's
        """)
    y = para(fig, y, """Cells on the outside run off to infinity; SciPy marks those with a -1 in the corner
        list. This project needs cells of chosen sizes, which plain Voronoi cannot give, so we use SciPy's
        version only as an independent check on our own code (page 2).""", chars=52)

    rng = np.random.default_rng(5)
    sites = rng.uniform(0, 1, size=(12, 2))
    ax = fig.add_axes([0.58, top - 0.275, 0.33, 0.235])
    voronoi_plot_2d(Voronoi(sites), ax=ax, show_vertices=False, line_colors=BLUE, line_width=1.1, point_size=0)
    ax.plot(sites[:, 0], sites[:, 1], "o", color=INK, markersize=3.2)
    ax.set_xlim(-0.05, 1.05), ax.set_ylim(-0.05, 1.05)
    tidy(ax)
    caption(ax, "Twelve sites and their Voronoi cells.\nDashed edges continue to infinity.")
    footer(fig, 1)
    return fig


# ---- page 2: how the project uses it ---------------------------------------

def page_two():
    fig = plt.figure(figsize=PAGE)
    y = heading(fig, 0.962, "3.  How this project uses SciPy", size=15)
    y = para(fig, y, """The chart needs each donor's cell to have an area proportional to the money given. The
        trick is to give every site a weight. A point x belongs to the site with the smallest value of
        (distance to the site) squared minus (the site's weight). Raise a weight and that cell grows at its
        neighbours' expense; the edges stay straight and the cells stay convex. This weighted version is called
        a power diagram.""")

    # Figure: same sites, equal weights vs one raised weight.
    rng = np.random.default_rng(5)
    sites = rng.uniform(0.08, 0.92, size=(12, 2))
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    grown = int(np.argmin(((sites - 0.5) ** 2).sum(1)))          # the most central site
    panels = [("All weights equal: the ordinary Voronoi diagram.", np.zeros(12))]
    w = np.zeros(12)
    w[grown] = 0.06
    panels.append(("One weight raised: that cell grows, the rest give way.", w))
    for k, (title, weights) in enumerate(panels):
        ax = fig.add_axes([LEFT + k * 0.43, y - 0.208, 0.25, 0.19])
        for i, cell in enumerate(vt.power_cells(sites, weights, square)):
            ax.add_patch(Polygon(cell, facecolor="#ce6767" if i == grown else "#dfe7ea",
                                 edgecolor="white", linewidth=1.6))
        ax.plot(sites[:, 0], sites[:, 1], "o", color=INK, markersize=3)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
        tidy(ax)
        caption(ax, title)
    y -= 0.224

    # Convergence on the real data.
    top = y
    y = para(fig, y, """So the job is: find the weights that give every cell its target area. That is a
        "find the lowest point" problem, and minimize solves it. Our function's gradient is simply (cell
        area minus target area), so the lowest point is exactly where every area is right. The function is
        bowl-shaped (convex), which means there is one answer and downhill always leads to it.""", chars=50)
    y = code(fig, y, """
        def objective(w, sites, targets, boundary):
            cells = power_cells(sites, w, boundary)
            areas = [polygon_area(c) for c in cells]
            return value, areas - targets   # gradient
        res = minimize(objective, w0, jac=True,
                       args=(sites, targets, boundary),
                       method="L-BFGS-B")
        """, width=0.43)

    ax = fig.add_axes([0.61, top - 0.232, 0.30, 0.2])
    with (ROOT / "data/donors_2026-10-26.csv").open() as f:
        donors = list(csv.DictReader(f))
    disc = vt.circle((0.0, 0.0), np.sqrt(1 / np.pi))             # area 1
    for group, color in (("Republican", "#ce6767"), ("Democratic", BLUE), ("Bipartisan", "#8c8c8c")):
        amounts = np.array([float(d["amount_musd"]) for d in donors if d["group"] == group])
        history = []
        start = vt.random_points_in(disc, len(amounts), np.random.default_rng(4))
        # Targets must add up to the region's area, or there is nothing to converge to.
        targets = amounts / amounts.sum() * vt.polygon_area(disc)
        vt.solve_weights(start, targets, disc, history=history)
        ax.semilogy(np.maximum(history, 1e-12), color=color, linewidth=1.4, label=f"{group} ({len(amounts)} cells)")
    ax.axhline(1e-3, color=MUTED, linewidth=0.7, linestyle=":")
    ax.text(0.3, 1.5e-3, "tolerance 0.1%", fontsize=7, fontfamily=BODY, color=MUTED)
    ax.set_xlabel("optimiser step", fontsize=8, fontfamily=BODY)
    ax.set_ylabel("worst area error (fraction of target)", fontsize=8, fontfamily=BODY)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, prop={"family": BODY, "size": 7}, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    caption(ax, "Real donor data, random starting sites:\nworst error in any cell's area, step by step.")

    y = heading(fig, min(y, top - 0.262) - 0.002, "4.  Reading what minimize returns")
    y = code(fig, y, """
        res.x        # the answer: here, one weight per cell      res.fun      # the function's value there
        res.success  # True if a stopping test was met            res.message  # why it stopped, in words
        res.nit      # number of steps taken
        """, width=0.82)
    y = para(fig, y, """res.success only says the optimiser's own stopping rule fired. This project measures what
        it cares about (the worst area error) after every solve and raises an error above 0.1%.""")

    y = heading(fig, y - 0.004, "5.  Three mistakes worth knowing in advance")
    y = para(fig, y, """It only minimises. To maximise a function F, hand it -F and also negate the gradient.
        Our F is concave (dome-shaped) and we want its top, so objective() returns -F and (areas - targets).""")
    y = para(fig, y, """A wrong gradient fails quietly. If the sign or an index is off, minimize still runs
        and returns something, slowly or wrongly. scipy.optimize.check_grad compares your gradient with a
        numerical estimate. The tests do this, and also check that with equal weights our cells match
        scipy.spatial.Voronoi corner for corner.""")
    y = para(fig, y, """An impossible problem does not announce itself. Cells always fill the region, so target
        areas must add up to its area. Building this page, targets summing to 1 were passed for a disc of area
        0.9998: minimize ran, reported nothing wrong, and sent the weights to 10^23. The code now refuses that.""")
    footer(fig, 2)
    return fig


def main():
    out = ROOT / "docs/scipy_primer.pdf"
    with PdfPages(out) as pdf:
        for n, build in enumerate((page_one, page_two), start=1):
            fig = build()
            pdf.savefig(fig)
            if len(sys.argv) > 1:                   # optional: also dump PNG previews to a folder
                fig.savefig(Path(sys.argv[1]) / f"primer_{n}.png", dpi=110)
            plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
