"""
Contour and ink-mass extraction.

Contours are taken from the *tonal band boundaries*, not from an edge detector.
That is the important choice in this module. Edge detection finds every change
in the photograph, including texture and noise; band boundaries find the places
where one flat tone meets another, which is precisely what a block cutter
carves. The result reads as a woodcut rather than as a traced photo.

Line weight follows Ray's convention: the outer silhouette is heavy, interior
tonal divisions are fine. That single contrast does more for the look than any
amount of hatching tuning.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .stroke import FilledShape, Stroke, resample, wobble

Array = np.ndarray


# --------------------------------------------------------------------------
# polyline conditioning
# --------------------------------------------------------------------------


def smooth_polyline(pts: Array, sigma: float = 2.0, closed: bool = True) -> Array:
    """Remove the pixel staircase from a traced contour.

    Mask contours are axis-aligned at the pixel level. Drawn as-is they look
    like a screenshot of a selection, so every contour gets smoothed before it
    becomes a stroke.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 5 or sigma <= 0:
        return pts
    mode = "wrap" if closed else "nearest"
    out = pts.copy()
    out[:, 0] = gaussian_filter1d(pts[:, 0], sigma, mode=mode)
    out[:, 1] = gaussian_filter1d(pts[:, 1], sigma, mode=mode)
    return out


def simplify(pts: Array, epsilon: float = 1.0) -> Array:
    """Douglas-Peucker, to drop redundant vertices before smoothing."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 4 or epsilon <= 0:
        return pts.reshape(-1, 2)
    out = cv2.approxPolyDP(pts, epsilon, True)
    return out.reshape(-1, 2).astype(np.float32)


def trace(
    mask: Array,
    min_area: float = 40.0,
    epsilon: float = 1.0,
    smooth: float = 2.0,
    with_holes: bool = False,
    hole_min_area: float | None = None,
):
    """Trace a boolean mask into conditioned polylines.

    Returns a list of (outer, holes) when `with_holes`, else a flat list of
    polylines including hole boundaries.
    """
    m = (mask.astype(np.uint8)) * 255
    mode = cv2.RETR_CCOMP if with_holes else cv2.RETR_LIST
    found = cv2.findContours(m, mode, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = found[-2], found[-1]
    if contours is None or len(contours) == 0:
        return []

    hole_floor = min_area if hole_min_area is None else hole_min_area

    def cond(c, floor: float = min_area) -> Array | None:
        c = c.reshape(-1, 2).astype(np.float32)
        if len(c) < 4 or cv2.contourArea(c.reshape(-1, 1, 2)) < floor:
            return None
        c = smooth_polyline(c, sigma=smooth, closed=True)
        c = simplify(c, epsilon=epsilon)
        if len(c) < 3:
            return None
        return c

    if not with_holes:
        return [c for c in (cond(c) for c in contours) if c is not None]

    hier = hierarchy.reshape(-1, 4) if hierarchy is not None else None
    groups: list[tuple[Array, list[Array]]] = []
    for i, c in enumerate(contours):
        if hier is not None and hier[i][3] != -1:
            continue  # this is a hole; collected below
        outer = cond(c)
        if outer is None:
            continue
        holes: list[Array] = []
        if hier is not None:
            child = hier[i][2]
            while child != -1:
                hc = cond(contours[child], floor=hole_floor)
                if hc is not None:
                    holes.append(hc)
                child = hier[child][0]
        groups.append((outer, holes))
    return groups


# --------------------------------------------------------------------------
# strokes with swelling weight
# --------------------------------------------------------------------------


def variable_width_stroke(
    pts: Array,
    rng: np.random.Generator,
    width: float = 2.6,
    variation: float = 0.42,
    chunk: float = 55.0,
    closed: bool = True,
    wobble_amp: float = 0.7,
    wobble_wavelength: float = 40.0,
) -> list[Stroke]:
    """Split a contour into overlapping chunks of differing weight.

    A brush or gouge swells and thins as it travels; a constant-width outline
    looks like vector art. Chunking is a cheap way to get that swelling while
    keeping each piece a plain Stroke.
    """
    pts = wobble(pts, rng, amp=wobble_amp, wavelength=wobble_wavelength, step=2.0)
    n = len(pts)
    if n < 3:
        return []

    if closed:
        pts = np.vstack([pts, pts[:1]])
        n += 1

    seg = np.hypot(*np.diff(pts, axis=0).T)
    total = float(seg.sum())
    if total < 1e-3:
        return []

    n_chunks = max(1, int(round(total / max(chunk, 8.0))))
    bounds = np.linspace(0, n - 1, n_chunks + 1).astype(int)

    out: list[Stroke] = []
    for i in range(n_chunks):
        a = bounds[i]
        b = min(n - 1, bounds[i + 1] + 2)  # overlap hides the joins
        if b - a < 2:
            continue
        w = width * float(rng.uniform(1.0 - variation, 1.0 + variation))
        out.append(
            Stroke(
                pts[a : b + 1],
                width=max(0.4, w),
                taper=0.18 if n_chunks > 1 else 0.0,
                end_width=0.72,
                closed=False,
            )
        )
    return out


def contour_strokes(
    mask: Array,
    rng: np.random.Generator,
    width: float = 1.5,
    min_area: float = 40.0,
    epsilon: float = 1.0,
    smooth: float = 2.0,
    variation: float = 0.3,
    wobble_amp: float = 0.7,
) -> list[Stroke]:
    """Outline every region in a mask with a swelling line."""
    out: list[Stroke] = []
    for c in trace(mask, min_area=min_area, epsilon=epsilon, smooth=smooth):
        out += variable_width_stroke(
            c, rng, width=width, variation=variation, wobble_amp=wobble_amp
        )
    return out


def silhouette_strokes(
    subject: Array,
    rng: np.random.Generator,
    width: float = 3.4,
    min_area: float = 400.0,
    smooth: float = 3.0,
    variation: float = 0.34,
) -> list[Stroke]:
    """The heavy outer line. Deliberately smoother and bolder than interiors."""
    return contour_strokes(
        subject,
        rng,
        width=width,
        min_area=min_area,
        epsilon=1.4,
        smooth=smooth,
        variation=variation,
        wobble_amp=0.9,
    )


# --------------------------------------------------------------------------
# ink masses
# --------------------------------------------------------------------------


def ink_masses(
    mask: Array,
    rng: np.random.Generator,
    min_area: float = 60.0,
    epsilon: float = 0.9,
    smooth: float = 2.2,
    wobble_amp: float = 0.8,
    hole_min_area: float | None = None,
) -> list[FilledShape]:
    """Solid blacks, as filled shapes, keeping only substantial holes.

    The outline is wobbled before filling so the edge of a black mass has the
    same hand as the drawn lines. A geometrically perfect fill edge next to
    wobbly hatching is an immediate tell.

    Small holes are deliberately filled in rather than reproduced. Specular
    highlights inside a dark mass - light catching hair, a fold in black cloth -
    survive quantisation as a scatter of little islands, and reproducing them
    faithfully makes a black mass look moth-eaten. An engraver would simply
    leave the mass solid and let one or two larger lights carry it.
    """
    out: list[FilledShape] = []
    for outer, holes in trace(
        mask, min_area=min_area, epsilon=epsilon, smooth=smooth, with_holes=True,
        hole_min_area=hole_min_area,
    ):
        o = wobble(outer, rng, amp=wobble_amp, wavelength=46.0, step=2.5)
        hs = [wobble(h, rng, amp=wobble_amp * 0.8, wavelength=38.0, step=2.5) for h in holes]
        out.append(FilledShape(o, hs))
    return out


def detail_strokes(
    edge_map: Array,
    rng: np.random.Generator,
    width: float = 1.0,
    min_len: float = 9.0,
    smooth: float = 1.4,
) -> list[Stroke]:
    """Fine interior accents recovered from an edge map.

    Quantisation loses small features - the line of a mouth, a spectacle rim,
    the edge of a lapel. This puts a light pass of them back.
    """
    m = (edge_map.astype(np.uint8)) * 255
    found = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = found[-2]
    out: list[Stroke] = []
    for c in contours or []:
        p = c.reshape(-1, 2).astype(np.float32)
        if len(p) < 6:
            continue
        length = float(np.hypot(*np.diff(p, axis=0).T).sum())
        if length < min_len:
            continue
        p = smooth_polyline(p, sigma=smooth, closed=False)
        p = resample(p, step=2.0)
        p = wobble(p, rng, amp=0.45, wavelength=22.0, step=2.0)
        out.append(
            Stroke(
                p,
                width=max(0.4, width * float(rng.uniform(0.85, 1.15))),
                taper=0.3,
                end_width=0.2,
            )
        )
    return out


# --------------------------------------------------------------------------
# spot colour plate
# --------------------------------------------------------------------------


def spot_shapes(
    mask: Array,
    rng: np.random.Generator,
    simplify_px: float = 14.0,
    min_area: float = 900.0,
    grow: int = 6,
) -> list[FilledShape]:
    """Blobby flat shapes for the second ink.

    Heavily blurred and re-thresholded so the colour reads as a loose flat
    area behind the drawing, not as a coloured-in version of it. On Ray's
    two-ink covers the colour block never lines up with the black exactly.
    """
    m = mask.astype(np.uint8) * 255
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1))
        m = cv2.dilate(m, k)
    m = cv2.GaussianBlur(m, (0, 0), max(2.0, simplify_px * 0.6))
    m = (m > 110).astype(np.uint8) * 255

    out: list[FilledShape] = []
    for outer, holes in trace(
        m > 0, min_area=min_area, epsilon=simplify_px * 0.35, smooth=simplify_px * 0.5,
        with_holes=True,
    ):
        o = wobble(outer, rng, amp=1.8, wavelength=90.0, step=4.0)
        hs = [wobble(h, rng, amp=1.4, wavelength=70.0, step=4.0) for h in holes]
        out.append(FilledShape(o, hs))
    return out
