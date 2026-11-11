"""Draw the top-50 midterm donor chart as a Voronoi treemap.

Usage:  python make_chart.py
Reads:  data/donors_2026-10-26.csv   baseline (Al Jazeera graphic, FEC as of 26 Oct)
        data/updates_2026-11-03.csv  cited changes up to election day
        data/final_2026-12-03.csv    final figures; empty until they exist
Writes: out/midterm_donors_<as-of date>.png

Later files override earlier ones row by row (matched on name), so replacing
the numbers on 3 December means filling in the final CSV and rerunning this.

All drawing happens in a 770 x 962 "unit" canvas (the source graphic's pixel
size, y pointing up), rendered at 2x.
"""
from __future__ import annotations

import csv
import textwrap
from datetime import date
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Polygon
from matplotlib.textpath import TextPath

from voronoi_treemap import circle, polygon_area, polygon_centroid, voronoi_treemap

# ---- canvas --------------------------------------------------------------
W, H = 770, 962
PT_PER_UNIT = 7.7 * 72 / W             # figure is 7.7 in wide, so 1 unit = 0.72 pt
CENTER, RADIUS = (385.0, 440.0), 318.0
FONT = "Inter"

# ---- groups: colours sampled from the source graphic ----------------------
GROUPS = {
    "Republican": {"color": "#ce6767", "label": "Republican-aligned", "start": (400, 330)},
    "Democratic": {"color": "#7896a3", "label": "Democratic-aligned", "start": (530, 610)},
    "Bipartisan": {"color": "#b4b4b4", "label": "Special interest or bipartisan", "start": (160, 540)},
}
# Totals printed on the source graphic ($m); the baseline must reproduce them.
PRINTED_TOTALS = {"Republican": 1670, "Democratic": 832, "Bipartisan": 264}

BASELINE = Path("data/donors_2026-10-26.csv")
OVERRIDES = [Path("data/updates_2026-11-03.csv"), Path("data/final_2026-12-03.csv")]
BASELINE_DATE = date(2026, 10, 26)
SEED = 4                                # fixes the layout; change it for a different arrangement


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_donors():
    """Baseline rows with overrides applied. Returns (donors, as_of date)."""
    with BASELINE.open() as f:
        donors = {r["name"]: {**r, "amount": float(r["amount_musd"]), "baseline": float(r["amount_musd"])}
                  for r in csv.DictReader(f)}

    # Transcription check: group sums must round to the totals printed on the graphic.
    for group, printed in PRINTED_TOTALS.items():
        total = sum(d["amount"] for d in donors.values() if d["group"] == group)
        tolerance = 5 if printed >= 1000 else 0.5          # $1.67bn is printed to the nearest $10m
        assert abs(total - printed) <= tolerance, f"{group}: {total:.1f} != printed {printed}"

    as_of = BASELINE_DATE
    for path in OVERRIDES:
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        as_of = date.fromisoformat(path.stem.split("_")[-1])
        for r in rows:
            if not r["amount_musd"]:                       # blank amount = dropped out of the top 50
                donors.pop(r["name"], None)
                continue
            old = donors.get(r["name"], {"kind": "org", "face": "", "baseline": None})
            donors[r["name"]] = {**old, "name": r["name"], "group": r["group"] or old["group"],
                                 "amount": float(r["amount_musd"])}
    return list(donors.values()), as_of


def money(musd: float, whole: bool = False) -> str:
    """446 -> $446m, 92.2 -> $92.2m, 1674.7 -> $1.67bn. `whole` rounds millions to integers."""
    if musd >= 1000:
        return f"${musd / 1000:.2f}bn"
    if whole:
        return f"${musd:.0f}m"
    return f"${musd:.1f}".rstrip("0").rstrip(".") + "m"


def long_date(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


# --------------------------------------------------------------------------
# Fitting a label inside a convex cell
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def text_width(text: str, weight: str) -> float:
    """Width of `text` per unit of font size (text width scales linearly with size)."""
    return TextPath((0, 0), text, size=1, prop=FontProperties(family=FONT, weight=weight)).get_extents().width


def inside(poly: np.ndarray, points: np.ndarray, margin: float) -> bool:
    """True if every point is at least `margin` inside the convex CCW polygon."""
    a, b = poly, np.roll(poly, -1, axis=0)
    edge = b - a
    length = np.hypot(edge[:, 0], edge[:, 1])
    for p in points:
        # Signed distance from p to each edge line; positive = on the inner side.
        dist = (edge[:, 0] * (p[1] - a[:, 1]) - edge[:, 1] * (p[0] - a[:, 0])) / length
        if dist.min() < margin:
            return False
    return True


def wrappings(name: str):
    """The distinct ways of breaking a name over 1 to 4 lines."""
    seen = []
    for width in range(max(len(name), 6), 5, -1):
        lines = textwrap.wrap(name, width, break_long_words=False)
        if len(lines) <= 4 and lines not in seen:
            seen.append(lines)
    return seen


def fit_label(poly, name, amount, note, face_shape):
    """Largest text block (optional face, name lines, amount, optional note) that fits in `poly`.

    Returns a list of draw instructions. The block is tried at a few positions
    around the cell's centroid and at decreasing font sizes; the first size at
    which every line's rectangle lies inside the cell wins.
    """
    centroid = polygon_centroid(poly)
    reach = np.sqrt(polygon_area(poly))
    shifts = [(0, 0)] + [(dx * reach, dy * reach) for dx in (-.08, 0, .08) for dy in (-.1, -.05, .05, .1)]
    sizes = np.arange(min(15.0, 0.2 * reach), 3.9, -0.5)

    for size in sizes:
        for lines in wrappings(name):
            # Rows from top to bottom: (kind, text, font size, weight, row height)
            rows = [("name", ln, size, "medium", 1.15 * size) for ln in lines]
            rows.append(("amount", amount, 1.3 * size, "bold", 1.3 * 1.2 * size))
            if note:
                rows.append(("note", note, 0.72 * size, "medium", 0.72 * 1.3 * size))
            face_h = face_w = 0.0
            if face_shape:
                # Portraits are small in the source; never enlarge beyond 1.3x.
                k = float(np.clip(size / 8.5, 0.6, 1.3))
                face_h, face_w = face_shape[0] * k, face_shape[1] * k
            total_h = face_h + sum(r[4] for r in rows) + (0.25 * size if face_shape else 0)

            for dx, dy in shifts:
                cx, top = centroid[0] + dx, centroid[1] + dy + total_h / 2
                corners, draw = [], []
                y = top
                if face_shape:
                    # Heads are narrower than shoulders: test the top edge at 60% width.
                    corners += [(cx - .3 * face_w, y), (cx + .3 * face_w, y),
                                (cx - .5 * face_w, y - face_h), (cx + .5 * face_w, y - face_h)]
                    draw.append(("face", (cx - face_w / 2, cx + face_w / 2, y - face_h, y)))
                    y -= face_h + 0.25 * size
                for kind, text, fs, weight, row_h in rows:
                    half = text_width(text, weight) * fs / 2
                    corners += [(cx - half, y), (cx + half, y), (cx - half, y - row_h), (cx + half, y - row_h)]
                    draw.append((kind, text, fs, weight, cx, y - row_h / 2))
                    y -= row_h
                if inside(poly, np.array(corners), margin=0.5 * size):
                    return draw
    raise RuntimeError(f"no label size fits the cell for {name!r}; try another SEED")


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def text(ax, x, y, s, size, weight="medium", color="#222222", **kw):
    ax.text(x, y, s, fontsize=size * PT_PER_UNIT, fontfamily=FONT, fontweight=weight,
            color=color, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), **kw)


def main() -> None:
    donors, as_of = load_donors()
    group_total = {g: sum(d["amount"] for d in donors if d["group"] == g) for g in GROUPS}
    grand_total = sum(group_total.values())

    # Level 1: split the disc between the three groups.
    disc = circle(CENTER, RADIUS)
    group_cells = voronoi_treemap([group_total[g] for g in GROUPS], disc,
                                  sites=[GROUPS[g]["start"] for g in GROUPS])

    fig = plt.figure(figsize=(7.7, 9.62), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")
    ax.add_patch(Circle(CENTER, RADIUS + 9, fill=False, edgecolor="#a9a9a9", linewidth=1.6))

    # Level 2: split each group's cell between its donors.
    for (group, style), group_cell in zip(GROUPS.items(), group_cells):
        members = sorted((d for d in donors if d["group"] == group), key=lambda d: -d["amount"])
        cells, info = voronoi_treemap([d["amount"] for d in members], group_cell, seed=SEED, return_info=True)
        print(f"{group}: {len(members)} cells, worst area error {info['max_error']:.1e}")

        for donor, cell in zip(members, cells):
            ax.add_patch(Polygon(cell, facecolor=style["color"], edgecolor="white", linewidth=1.3, joinstyle="round"))
            face = plt.imread(f"assets/faces/{donor['face']}") if donor["face"] else None
            changed = donor["baseline"] is not None and donor["amount"] != donor["baseline"]
            note = None
            if donor["baseline"] is None:
                note = "new since 26 Oct"
            elif changed:
                note = f"{donor['amount'] - donor['baseline']:+.1f}".rstrip("0").rstrip(".").replace("+", "+$").replace("-", "-$") + "m since 26 Oct"
            for item in fit_label(cell, donor["name"], money(donor["amount"]), note,
                                  face.shape[:2] if face is not None else None):
                if item[0] == "face":
                    ax.imshow(face, extent=item[1], interpolation="lanczos", zorder=3)
                else:
                    kind, s, fs, weight, x, y = item
                    text(ax, x, y, s, fs, weight, color="#000000" if kind == "amount" else "#262626", zorder=4)
        # Thick white outline separates the groups.
        ax.add_patch(Polygon(group_cell, fill=False, edgecolor="white", linewidth=4, joinstyle="round", zorder=2))
    ax.set_xlim(0, W), ax.set_ylim(0, H)      # imshow resets the limits

    # Header
    text(ax, W / 2, 925, "Who funded the 2026 US midterm elections?", 30, "bold", "#b3122d")
    text(ax, W / 2, 878, f"The 50 largest donors and organisations gave a combined {money(grand_total)}\n"
                         f"during the 2026 midterm election cycle, as reported by {long_date(as_of)}.",
         15, color="#333333", linespacing=1.35)
    for x, (group, style) in zip((150, 385, 620), GROUPS.items()):
        ax.add_patch(Circle((x - 62, 812), 9, facecolor=style["color"], edgecolor="none"))
        text(ax, x - 46, 818, money(group_total[group], whole=True), 17, "bold", "#000000", ha="left")
        text(ax, x - 46, 801, style["label"], 10.5, color="#333333", ha="left")

    # Footer
    if as_of.month == 12:
        basis = f"Figures are final Federal Election Commission data as of {long_date(as_of)}."
    else:
        basis = (f"Figures are Federal Election Commission data as of {long_date(BASELINE_DATE)}, with later changes applied\n"
                 "only where a dated report could be cited (marked “since 26 Oct”). Final filings are due about 3 December.")
    text(ax, 28, 84, "Note: " + basis + "\nThey cover the 50 largest donors and organisations, not total election funding.",
         9.6, color="#444444", ha="left", va="top", linespacing=1.4)
    text(ax, 28, 20, "This chart: CC BY-NC-SA", 8.2, color="#777777", ha="left")
    text(ax, W - 28, 20, "Portraits and 26 Oct figures: Al Jazeera / AJ Labs, CC BY-NC-SA · Data: FEC",
         8.2, color="#777777", ha="right")

    out = Path("out") / f"midterm_donors_{as_of.isoformat()}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}  ({money(grand_total)} total, as of {as_of})")


if __name__ == "__main__":
    main()
