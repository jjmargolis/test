"""Cut the eight portraits out of the Al Jazeera graphic.

Each portrait in the source is a greyscale cut-out sitting on a coloured cell
(red or blue). That makes separating it from the background easy without any
face detection: grey pixels are portrait, coloured pixels are background.

Usage:  python crop_faces.py
Writes: assets/faces/<slug>.png (RGBA, transparent background)
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SOURCE = Path("assets/source/aljazeera_2026-10-26.webp")
OUT_DIR = Path("assets/faces")

# (left, top, right, bottom) in source pixels, set by hand against a pixel grid.
# The slug is the donor name as captioned in the source graphic.
BOXES = {
    "elon-musk": (477, 605, 592, 697),
    "elizabeth-and-richard-uihlein": (355, 751, 446, 810),
    "jeff-and-janine-yass": (391, 491, 472, 541),
    "ben-horowitz-and-marc-andreessen": (635, 561, 702, 598),
    "miriam-adelson": (316, 459, 365, 505),
    "paul-singer": (305, 629, 354, 665),
    "mike-bloomberg": (575, 353, 627, 402),
    "george-soros": (606, 451, 679, 501),
}

# Chroma (max channel minus min channel) below this is treated as pure grey.
# It absorbs the colour noise that WebP compression adds to grey pixels.
GREY_NOISE = 10.0


def cut_out(crop: np.ndarray) -> np.ndarray:
    """Turn an RGB crop into RGBA with the coloured background removed.

    Model: every pixel is a blend  pixel = a * portrait + (1 - a) * background,
    where the portrait is grey (zero chroma) and the background is one flat
    colour. Chroma is linear under blending, so

        chroma(pixel) = (1 - a) * chroma(background)   =>   a = 1 - chroma / chroma_bg

    Having `a`, we solve the blend equation for the portrait colour so that
    soft edges do not carry a red or blue fringe.
    """
    rgb = crop.astype(float)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)

    # Background colour = median of the most colourful pixels in the crop.
    is_background = chroma >= np.percentile(chroma, 90) * 0.9
    background = np.median(rgb[is_background], axis=0)
    chroma_bg = background.max() - background.min()

    alpha = 1.0 - np.clip((chroma - GREY_NOISE) / (chroma_bg - 2 * GREY_NOISE), 0.0, 1.0)

    # Un-blend: portrait = (pixel - (1 - a) * background) / a
    safe = np.maximum(alpha, 1e-3)[..., None]
    portrait = (rgb - (1.0 - alpha[..., None]) * background) / safe
    # The portrait is grey by assumption; force it so no colour cast survives.
    grey = np.clip(portrait.mean(axis=2), 0, 255)

    # White cell borders are grey too, so slivers of them survive in the corners.
    # They are not attached to the portrait: label connected opaque blobs and
    # keep only those at least a tenth the size of the largest (the people).
    blobs, count = ndimage.label(alpha > 0.5)
    sizes = ndimage.sum(alpha > 0.5, blobs, index=range(1, count + 1))
    keep = 1 + np.flatnonzero(sizes >= 0.1 * sizes.max())
    # Grow the kept region by 2 px so the soft anti-aliased edge is not shaved off.
    alpha = alpha * ndimage.binary_dilation(np.isin(blobs, keep), iterations=2)

    out = np.dstack([grey, grey, grey, alpha * 255.0])
    return out.round().astype(np.uint8)


def main() -> None:
    source = Image.open(SOURCE).convert("RGB")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, box in BOXES.items():
        rgba = cut_out(np.asarray(source.crop(box)))
        Image.fromarray(rgba, "RGBA").save(OUT_DIR / f"{slug}.png")
        print(f"{slug}: {rgba.shape[1]}x{rgba.shape[0]} px")


if __name__ == "__main__":
    main()
