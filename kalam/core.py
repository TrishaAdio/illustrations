"""
Image pipeline: photograph in, discrete tonal bands out.

Ray's ink work has essentially no continuous grey. Tone is *implied* by the
density of marks, never by a soft gradient. So the whole job of this module is
to throw away smooth tone as early and as deliberately as possible, leaving a
small number of flat regions that the hatching engine can then represent with
marks.

The order matters: denoise while preserving edges, fix local contrast so
detail survives quantisation, *then* quantise. Quantising first destroys the
information the later stages need.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

Array = np.ndarray


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load(path: str, long_edge: int = 1600) -> Image.Image:
    """Load an image, fix EXIF rotation, scale so the long edge is `long_edge`."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = long_edge / float(max(w, h))
    if abs(scale - 1.0) > 0.01:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


def to_luma(img: Image.Image) -> Array:
    """Perceptual luminance in [0, 1]; 0 is black."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    return (0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]).astype(
        np.float32
    )


# --------------------------------------------------------------------------
# preparation
# --------------------------------------------------------------------------


@dataclass
class PrepConfig:
    # --- flattening: the most important stage for real photographs ---------
    meanshift: int = 18  # mean-shift spatial radius; 0 disables
    meanshift_colour: float = 34.0
    smooth_passes: int = 2  # bilateral iterations
    smooth_space: int = 9
    smooth_colour: float = 55.0

    # --- grading -----------------------------------------------------------
    clahe_clip: float = 0.0  # 0 disables local contrast
    clahe_grid: int = 8
    gamma: float = 1.0  # <1 lightens, >1 darkens
    exposure: float = 0.0  # additive, in luma units
    autolevel: bool = True
    autolevel_clip: float = 0.5  # percent clipped at each end

    # --- banding ------------------------------------------------------------
    band_scale: float = 0.42  # quantise at this fraction of full resolution
    min_area_frac: float = 0.0016  # drop regions smaller than this share of area
    median: int = 5  # median passes on the band map, merges speckle
    dither: float = 0.032  # threshold dither amplitude; 0 gives hard staircases
    dither_scale: float = 45.0  # px wavelength of the dither field


def flatten(img: Image.Image, cfg: PrepConfig) -> Image.Image:
    """Collapse a photograph into broad areas of near-constant tone.

    Mean-shift is doing the heavy lifting here, and it is the single change
    that decides whether a real photograph works at all. Bilateral filtering
    smooths *within* a neighbourhood but leaves a continuous tonal drift, so
    quantising afterwards still fragments a face into dozens of slivers. Mean
    shift instead pulls pixels toward modes in joint colour-space, producing
    genuinely flat plateaux with abrupt edges - which is what a block cutter
    sees when they look at a photograph and decide where one tone ends.
    """
    if cfg.meanshift <= 0:
        return img
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    flat = cv2.pyrMeanShiftFiltering(
        bgr, sp=float(cfg.meanshift), sr=float(cfg.meanshift_colour), maxLevel=2
    )
    return Image.fromarray(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB), mode="RGB")


def prepare(luma: Array, cfg: PrepConfig) -> Array:
    """Denoise, normalise and grade luma ahead of quantisation."""
    x = np.clip(luma, 0.0, 1.0)

    if cfg.autolevel:
        lo = np.percentile(x, cfg.autolevel_clip)
        hi = np.percentile(x, 100.0 - cfg.autolevel_clip)
        if hi - lo > 1e-3:
            x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    u8 = (x * 255.0).astype(np.uint8)

    for _ in range(max(0, cfg.smooth_passes)):
        u8 = cv2.bilateralFilter(u8, cfg.smooth_space, cfg.smooth_colour, cfg.smooth_space)

    if cfg.clahe_clip > 0:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)
        )
        u8 = clahe.apply(u8)

    x = u8.astype(np.float32) / 255.0

    if cfg.exposure:
        x = x + cfg.exposure
    if abs(cfg.gamma - 1.0) > 1e-3:
        x = np.clip(x, 0.0, 1.0) ** cfg.gamma

    return np.clip(x, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# tonal banding
# --------------------------------------------------------------------------


def dither_field(
    shape: tuple[int, int], amp: float = 0.03, scale: float = 45.0, seed: int = 0
) -> Array:
    """Smooth low-frequency noise, added to luma before banding.

    This exists to kill a specific ugly artefact. Quantising a smooth gradient
    puts the band boundary wherever luma crosses the threshold - which, on a
    gradient, is a near-straight line, and after morphological cleanup becomes
    a regular sawtooth staircase that instantly reads as digital.

    Perturbing the threshold spatially fixes it, and the way it fails is the
    useful part: boundary displacement is roughly amp/|grad luma|, so soft
    gradients wander a long way while genuine edges barely move. Gentle tone
    gets an organic, hand-judged boundary; real detail stays put.
    """
    if amp <= 0:
        return np.zeros(shape, dtype=np.float32)
    h, w = shape
    rng = np.random.default_rng(seed)
    lh, lw = max(2, int(h / scale)), max(2, int(w / scale))
    low = rng.normal(0.0, 1.0, (lh, lw)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    low = cv2.GaussianBlur(low, (0, 0), scale * 0.35)
    peak = float(np.abs(low).max()) or 1.0
    return (low / peak * amp).astype(np.float32)


def band_masks(luma: Array, thresholds: list[float]) -> list[Array]:
    """Split luma into flat regions. Index 0 is the darkest band.

    `thresholds` are ascending luma cuts. N thresholds yield N+1 bands; the
    last band is the paper-white one and normally receives no ink at all.
    """
    ts = sorted(float(t) for t in thresholds)
    masks: list[Array] = []
    lo = -1e-6
    for t in ts:
        masks.append((luma > lo) & (luma <= t))
        lo = t
    masks.append(luma > lo)
    return masks


def clean_mask(mask: Array, min_area: int = 24, close: int = 3) -> Array:
    """Remove speckle and close pinholes so hatching lands on real shapes.

    Without this, sensor noise and JPEG artefacts become thousands of one-pixel
    islands, each of which would sprout its own tiny hatch dash.
    """
    m = (mask.astype(np.uint8)) * 255
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        keep = np.zeros(n, dtype=bool)
        for i in range(1, n):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
        m = np.where(keep[labels], 255, 0).astype(np.uint8)
    return m > 0


def smooth_mask(mask: Array, sigma: float = 2.5, thresh: float = 0.5, inset: float = 0.0) -> Array:
    """Round off a mask boundary, and optionally pull it inward.

    Two jobs. Blur-and-rethreshold removes pixel-scale raggedness, which
    otherwise shows up as a regular sawtooth wherever hatching is clipped to a
    boundary running roughly parallel to it.

    The inset matters just as much: a real pen stops hatching a little short of
    the outline rather than running exactly into it, so the contour reads as the
    edge and the hatching sits inside. Butting hatch ends right up against the
    boundary is what produces a mechanical, filled-selection look.
    """
    m = (mask.astype(np.uint8)) * 255
    if sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigma)
        m = (m > int(255 * thresh)).astype(np.uint8) * 255
    if inset > 0:
        k = max(1, int(round(inset)) * 2 + 1)
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m > 0


def quantise(
    luma: Array,
    thresholds: list[float],
    cfg: PrepConfig | None = None,
    boundary_smooth: float = 2.0,
) -> list[Array]:
    """Tonal bands as full-resolution masks, darkest first.

    Works on a band *index* map rather than on each band independently, and
    does so at reduced resolution. Both choices matter:

    - An index map is a partition, so simplifying it can never open gaps or
      overlaps between bands. Cleaning each band separately can do both.
    - Downscaling before quantising destroys fine detail deliberately. A pen
      drawing of a face is not a per-pixel decision about every pore; it is a
      handful of decisive regions. Deciding those regions at ~40% resolution
      and drawing them at full resolution is what keeps plates legible.

    Median passes then merge leftover speckle into whichever band surrounds it,
    which a per-band area filter cannot do without leaving holes.
    """
    cfg = cfg or PrepConfig()
    h, w = luma.shape
    n_bands = len(thresholds) + 1

    scale = float(np.clip(cfg.band_scale, 0.05, 1.0))
    if scale < 0.99:
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        small = cv2.resize(luma, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh = w, h
        small = luma

    # index map: 0 = darkest band
    idx = np.digitize(small, np.sort(np.asarray(thresholds, dtype=np.float32))).astype(
        np.uint8
    )
    idx = np.clip(idx, 0, n_bands - 1)

    # merge speckle into surrounding tone
    for k in range(max(0, cfg.median)):
        ks = 3 if k % 2 == 0 else 5
        idx = cv2.medianBlur(idx, ks)

    # drop any remaining islands below the area floor, then let a wider median
    # pass hand those pixels to a neighbouring band
    min_area = max(6, int(sw * sh * cfg.min_area_frac))
    for b in range(n_bands):
        m = (idx == b).astype(np.uint8)
        if not m.any():
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        small_lbls = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < min_area]
        if not small_lbls:
            continue
        kill = np.isin(labels, small_lbls)
        # temporarily mark as an impossible value, then median-fill
        idx[kill] = 255
    if (idx == 255).any():
        filled = idx.copy()
        for _ in range(6):
            blurred = cv2.medianBlur(np.where(filled == 255, 0, filled), 5)
            filled = np.where(filled == 255, blurred, filled)
            if not (filled == 255).any():
                break
        idx = np.where(idx == 255, filled, idx).astype(np.uint8)
        idx[idx == 255] = n_bands - 1

    masks: list[Array] = []
    for b in range(n_bands):
        m = (idx == b).astype(np.uint8) * 255
        if scale < 0.99:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        if boundary_smooth > 0:
            m = cv2.GaussianBlur(m, (0, 0), boundary_smooth)
        masks.append(m > 127)
    return masks


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def flow_field(luma: Array, smooth: float = 9.0) -> Array:
    """Per-pixel angle (radians) that follows form rather than cutting across it.

    Derived from the smoothed structure tensor, so hatching can be made to run
    *along* a cheekbone or a fold instead of ignoring it. Ray does both; this
    enables the form-following variety.
    """
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=5)

    # tensor components, blurred so the field is coherent rather than noisy
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), smooth)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), smooth)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), smooth)

    # dominant gradient orientation, then rotate 90 deg to run along the form
    angle = 0.5 * np.arctan2(2.0 * jxy, (jxx - jyy) + 1e-9)
    return (angle + np.pi / 2.0).astype(np.float32)


def edges(luma: Array, low: float = 40.0, high: float = 110.0, blur: float = 1.2) -> Array:
    """Canny edge map, used to sharpen interior detail the bands may have lost."""
    u8 = (np.clip(luma, 0, 1) * 255).astype(np.uint8)
    if blur > 0:
        u8 = cv2.GaussianBlur(u8, (0, 0), blur)
    return cv2.Canny(u8, low, high) > 0


def close_holes(mask: Array, size: int = 15) -> Array:
    """Close pinholes in a mask with a disc of radius ~size/2.

    More reliable than filtering holes by area after tracing, because it does
    not depend on the contour hierarchy resolving each speck as a child of the
    right parent.
    """
    if size < 3:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size) | 1, int(size) | 1))
    m = cv2.morphologyEx((mask.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, k)
    return m > 0


def fill_holes(mask: Array) -> Array:
    """Fill interior holes, keeping only outer boundaries."""
    m = (mask.astype(np.uint8)) * 255
    found = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, found[-2], -1, 255, thickness=-1)
    return out > 0


def background_by_flood(
    img: Image.Image, tol: int = 16, from_bottom: bool = False
) -> Array:
    """Background mask by flooding inward from the frame edge.

    For studio portraits and old plate photography this beats GrabCut
    comfortably. The backdrop is a smooth, continuous region that touches the
    frame edge, which is exactly what a tolerance flood is good at, whereas
    GrabCut needs the subject and background to be separable in colour - and in
    a monochrome portrait, skin and a grey wall frequently are not.

    Bottom-edge seeds are off by default: in a portrait the sitter's clothing
    usually runs off the bottom of the frame, and flooding from there eats it.
    """
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    flood = np.zeros((h + 2, w + 2), np.uint8)

    # Seed densely along the top and both sides. FIXED_RANGE compares every
    # candidate against its own seed rather than against its neighbour, which
    # is essential: with neighbour comparison a smoothly graded backdrop lets
    # the flood drift without limit and walk straight into the subject. Fixed
    # range keeps each flood local in tone, and many seeds then cover a
    # gradient in overlapping steps.
    n_side = 9
    seeds: list[tuple[int, int]] = []
    for i in range(n_side):
        fx = int((i + 0.5) / n_side * (w - 1))
        seeds.append((fx, 2))
    for i in range(n_side):
        fy = int((i + 0.5) / n_side * (h - 1))
        seeds.append((2, fy))
        seeds.append((w - 3, fy))
    if from_bottom:
        for i in range(n_side):
            fx = int((i + 0.5) / n_side * (w - 1))
            seeds.append((fx, h - 3))

    lo = (tol, tol, tol)
    work = arr.copy()
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    for sx, sy in seeds:
        if not (0 <= sx < w and 0 <= sy < h):
            continue
        if flood[sy + 1, sx + 1]:
            continue  # already covered by an earlier flood
        try:
            cv2.floodFill(work, flood, (int(sx), int(sy)), 255, lo, lo, flags)
        except cv2.error:
            continue

    return flood[1:-1, 1:-1] > 0


def isolate_subject(
    img: Image.Image,
    margin: float = 0.06,
    iters: int = 6,
    largest_only: bool = True,
    tol: int = 26,
) -> Array:
    """Rough foreground mask via GrabCut, so a plate can drop its background
    to bare paper - which is what most character plates actually do.

    Seeded with an explicit mask rather than a plain rectangle: the outer frame
    is marked definite background and a centre ellipse probable foreground.
    Rectangle seeding alone tends to keep a slab of wall whose colour resembles
    the subject, because nothing ever told it the wall was background.
    """
    h, w = np.asarray(img).shape[:2]

    # 1. flood from the frame edge - cheap, and usually right for portraits
    bg = background_by_flood(img, tol=tol, from_bottom=False)
    fg = fill_holes(clean_mask(~bg, min_area=(h * w) // 300, close=9))
    if largest_only and fg.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
        if n > 2:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fg = labels == best
    cover = float(fg.mean())
    if 0.10 <= cover <= 0.88:
        m = cv2.GaussianBlur((fg.astype(np.uint8)) * 255, (0, 0), 2.0)
        return m > 110

    # 2. fall back to GrabCut only if flooding produced nothing plausible
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mx, my = max(1, int(w * margin)), max(1, int(h * margin))
    # hard background: a strip right around the edge
    band = max(2, int(min(h, w) * margin * 0.5))
    mask[:band, :] = cv2.GC_BGD
    mask[-band:, :] = cv2.GC_BGD
    mask[:, :band] = cv2.GC_BGD
    mask[:, -band:] = cv2.GC_BGD
    # probable foreground: a generous centre ellipse
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.55)),
        (int(w * 0.5) - mx, int(h * 0.45) - my),
        0, 0, 360,
        int(cv2.GC_PR_FGD),
        -1,
    )
    cv2.ellipse(
        mask,
        (w // 2, int(h * 0.55)),
        (int(w * 0.30), int(h * 0.28)),
        0, 0, 360,
        int(cv2.GC_FGD),
        -1,
    )

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.ones((h, w), dtype=bool)

    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    fg = clean_mask(fg, min_area=(h * w) // 400, close=7)

    if largest_only and fg.any():
        # a figure is one connected thing; stray slabs of matching wall are not
        n, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
        if n > 2:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fg = labels == best

    # Fail safe rather than fail destructive. When GrabCut cannot separate the
    # subject it returns its own seed shape almost unchanged, and using that
    # blanks the plate to a featureless blob. Better to decline to isolate and
    # let the whole frame be drawn.
    cover = float(fg.mean())
    if not (0.10 <= cover <= 0.92):
        return np.ones((h, w), dtype=bool)

    m = (fg.astype(np.uint8)) * 255
    m = cv2.GaussianBlur(m, (0, 0), 2.5)
    return m > 110


def vignette_falloff(shape: tuple[int, int], strength: float = 0.0) -> Array:
    """Radial weight in [0,1], used to thin marks toward the edges of a plate.

    Ray's plates very often dissolve at the margins rather than filling the
    frame edge to edge; this drives that.
    """
    if strength <= 0:
        return np.ones(shape, dtype=np.float32)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.hypot((xx - cx) / cx, (yy - cy) / cy) / np.sqrt(2.0)
    return np.clip(1.0 - strength * r**1.6, 0.0, 1.0).astype(np.float32)
