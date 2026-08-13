"""
Plate assembly.

Runs the whole pipeline: photograph -> prepared luma -> tonal bands -> marks ->
composited plate, as both raster and SVG. Also simulates the physical side of
the thing: laid paper, and a second ink deliberately out of register.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import contour as ct
from . import core
from .hatch import HatchStyle, hatch_band
from .stroke import Layer, RasterRenderer, Stroke, SvgRenderer, wobble

Array = np.ndarray


# --------------------------------------------------------------------------
# inks and papers
# --------------------------------------------------------------------------

SPOT_COLOURS: dict[str, tuple[int, int, int]] = {
    "brick": (196, 68, 47),
    "red": (188, 44, 38),
    "ochre": (214, 154, 58),
    "mustard": (223, 178, 62),
    "indigo": (43, 76, 126),
    "blue": (38, 95, 148),
    "green": (58, 118, 88),
    "plum": (128, 58, 92),
    "sepia": (146, 100, 62),
}

PAPERS: dict[str, tuple[int, int, int]] = {
    "cream": (244, 239, 228),
    "white": (252, 251, 248),
    "newsprint": (234, 228, 210),
    "aged": (228, 216, 192),
    "grey": (223, 222, 217),
}

INKS: dict[str, tuple[int, int, int]] = {
    "black": (26, 24, 22),
    "soot": (18, 17, 16),
    "warmblack": (38, 31, 26),
    "indigo": (28, 38, 62),
}


def make_paper(
    size: tuple[int, int],
    colour: tuple[int, int, int],
    grain: float = 0.014,
    mottle: float = 0.022,
    seed: int = 0,
) -> Image.Image:
    """Paper stock: base colour, fine grain, and slow blotching.

    Ink on perfectly flat white is the fastest way to make a plate look
    digital. The amplitudes here are deliberately small - just enough to break
    the flatness without looking like a filter.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    base = np.array(colour, dtype=np.float32) / 255.0
    field_ = np.ones((h, w), dtype=np.float32)

    if mottle > 0:
        low = rng.normal(0.0, 1.0, (max(2, h // 24), max(2, w // 24))).astype(np.float32)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        low = cv2.GaussianBlur(low, (0, 0), 12.0)
        peak = float(np.abs(low).max()) or 1.0
        field_ += (low / peak) * mottle

    if grain > 0:
        field_ += rng.normal(0.0, grain, (h, w)).astype(np.float32)

    img = np.clip(base[None, None, :] * field_[:, :, None], 0.0, 1.0)
    return Image.fromarray((img * 255.0).astype(np.uint8), mode="RGB")


# --------------------------------------------------------------------------
# preset description
# --------------------------------------------------------------------------


@dataclass
class BandSpec:
    """One tonal band and how it should be marked.

    `threshold` is the upper luma bound. Bands are listed darkest first.
    `mode` is solid | cross | parallel | flow | stipple | none.
    """

    threshold: float
    mode: str
    style: HatchStyle = field(default_factory=HatchStyle)


@dataclass
class Preset:
    name: str = "custom"
    description: str = ""
    long_edge: int = 1600
    prep: core.PrepConfig = field(default_factory=core.PrepConfig)
    bands: list[BandSpec] = field(default_factory=list)

    silhouette: bool = True
    silhouette_width: float = 3.4
    interior_contours: bool = True
    interior_width: float = 1.4
    interior_min_area_frac: float = 0.004  # ignore regions smaller than this
    detail: bool = True
    detail_width: float = 0.95
    detail_thresholds: tuple[float, float] = (45.0, 120.0)

    isolate: bool = False
    vignette: float = 0.0

    # Holes smaller than this share of the plate get filled in, so solid masses
    # read as brush rather than moth-eaten.
    hole_min_area_frac: float = 0.0012
    hole_close_frac: float = 0.022  # closing disc, as a share of the short edge
    hatch_smooth: float = 2.5  # rounds band boundaries before hatching
    hatch_inset: float = 1.0  # px the hatching stops short of the outline

    # Cumulative passes (how hatching actually works) rather than disjoint
    # bands. Disjoint banding leaves a visible outline at every tonal step.
    layered: bool = True

    spot: str | None = None
    spot_band: int = 1  # which band seeds the colour block
    spot_offset: tuple[float, float] = (5.0, -4.0)
    spot_opacity: float = 0.9

    ink: str = "black"
    paper: str = "cream"
    grain: float = 0.014
    mottle: float = 0.022

    border: str | None = None
    border_inset: float = 0.035

    aspect: float | None = None  # crop to width/height before drawing
    crop_bias: float = 0.5  # 0 keeps the top edge, 1 the bottom

    def thresholds(self) -> list[float]:
        return [b.threshold for b in self.bands]


# --------------------------------------------------------------------------
# border
# --------------------------------------------------------------------------


def border_strokes(
    size: tuple[int, int], style: str, rng: np.random.Generator, inset: float = 0.035
) -> list[Stroke]:
    """Hand-ruled frames, as used on the Sandesh covers."""
    w, h = size
    m = inset * min(w, h)

    def rect(pad: float, width: float) -> list[Stroke]:
        pts = np.array(
            [[pad, pad], [w - pad, pad], [w - pad, h - pad], [pad, h - pad], [pad, pad]],
            dtype=np.float32,
        )
        p = wobble(pts, rng, amp=1.1, wavelength=180.0, step=6.0)
        return ct.variable_width_stroke(
            p, rng, width=width, variation=0.22, chunk=200.0, closed=False, wobble_amp=0.0
        )

    if style == "rule":
        return rect(m, 2.2)
    if style == "double":
        return rect(m, 2.8) + rect(m + max(5.0, m * 0.22), 1.1)
    if style == "sandesh":
        out = rect(m, 3.6) + rect(m + max(6.0, m * 0.28), 1.2)
        return out
    return []


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------


def crop_to_aspect(img: Image.Image, aspect: float, bias: float = 0.5) -> Image.Image:
    """Crop to an aspect ratio. `bias` picks where the kept slice sits.

    0 keeps the top (or left) edge, 1 the bottom (or right), 0.5 centres. A
    centred banner crop of a portrait cuts straight through the face, so
    banner-shaped presets bias upward.
    """
    w, h = img.size
    cur = w / h
    b = float(np.clip(bias, 0.0, 1.0))
    if abs(cur - aspect) < 0.01:
        return img
    if cur > aspect:  # too wide, trim sides
        nw = int(round(h * aspect))
        x0 = int(round((w - nw) * b))
        return img.crop((x0, 0, x0 + nw, h))
    nh = int(round(w / aspect))
    y0 = int(round((h - nh) * b))
    return img.crop((0, y0, w, y0 + nh))


@dataclass
class PlateResult:
    image: Image.Image
    svg: str
    stats: dict


def render_plate(
    img: Image.Image,
    preset: Preset,
    seed: int = 7,
    supersample: int = 3,
    make_svg: bool = True,
) -> PlateResult:
    """Photograph in, finished plate out."""
    rng = np.random.default_rng(seed)

    if preset.aspect:
        img = crop_to_aspect(img, preset.aspect, preset.crop_bias)
    w, h = img.size

    flat = core.flatten(img, preset.prep)
    luma = core.to_luma(flat)
    prepared = core.prepare(luma, preset.prep)

    subject = None
    if preset.isolate:
        subject = core.isolate_subject(img)
        # push the background to paper white so no marks are generated there
        prepared = np.where(subject, prepared, 1.0).astype(np.float32)
        if not subject.any():
            subject = None

    # Dither only the copy used for banding. The flow field and edge detector
    # want the clean signal - perturbing those would smear real structure.
    banding = prepared
    if preset.prep.dither > 0:
        banding = np.clip(
            prepared
            + core.dither_field(
                (h, w), amp=preset.prep.dither, scale=preset.prep.dither_scale, seed=seed
            ),
            0.0,
            1.0,
        ).astype(np.float32)

    masks = core.quantise(banding, preset.thresholds(), preset.prep)
    flow = core.flow_field(prepared)
    weight = core.vignette_falloff((h, w), preset.vignette)

    ink_colour = INKS.get(preset.ink, INKS["black"])
    black = Layer("ink", colour=ink_colour)

    # --- marks ----------------------------------------------------------
    # Passes are cumulative, not disjoint: each one covers everything at or
    # below its threshold, so darker areas simply receive more passes. This is
    # how a hand actually builds tone, and it matters for two reasons.
    #
    # Disjoint bands give every band a visible boundary, and the plate reads as
    # a contour map of blobs. Cumulative passes bury each boundary under the
    # next pass, so tone gradates smoothly and no coastlines survive.
    density = np.clip(1.0 - prepared, 0.0, 1.0)
    passes = list(reversed(preset.bands)) if preset.layered else list(preset.bands)

    for order, spec in enumerate(passes):
        if preset.layered:
            if spec.mode in ("none", "blank", "paper"):
                continue
            mask = banding <= spec.threshold
            mask = core.clean_mask(mask, min_area=max(24, int(w * h * 0.0004)), close=5)
        else:
            i = preset.bands.index(spec)
            if i >= len(masks):
                continue
            mask = masks[i]

        if not mask.any():
            continue

        if spec.mode == "solid":
            solid = core.close_holes(
                core.smooth_mask(mask, sigma=3.0),
                size=max(3, int(round(min(w, h) * preset.hole_close_frac))),
            )
            black.extend(
                ct.ink_masses(
                    solid,
                    rng,
                    min_area=max(60.0, w * h * 0.0004),
                    hole_min_area=max(180.0, w * h * preset.hole_min_area_frac),
                )
            )
            continue

        # Hatch a rounded, slightly inset copy so runs stop just short of the
        # boundary rather than butting into a ragged edge.
        hm = core.smooth_mask(mask, sigma=preset.hatch_smooth, inset=preset.hatch_inset)
        if not hm.any():
            continue

        black.extend(
            hatch_band(
                spec.mode,
                hm,
                spec.style,
                rng,
                flow=flow,
                weight=weight,
                density=density,
            )
        )

    # --- interior tonal boundaries --------------------------------------
    # Only substantial regions get an outline. Contouring every small region is
    # what turns a portrait into a contour map, so the area floor is a fraction
    # of the plate rather than a fixed pixel count.
    if preset.interior_contours and len(masks) > 2:
        interior_min = max(220.0, (w * h) * preset.interior_min_area_frac)
        cumulative = np.zeros((h, w), dtype=bool)
        for i in range(len(masks) - 1):
            cumulative = cumulative | masks[i]
            if i == 0:
                continue  # the darkest core reads via its own fill/hatch
            black.extend(
                ct.contour_strokes(
                    core.smooth_mask(cumulative, sigma=3.0),
                    rng,
                    width=preset.interior_width,
                    min_area=interior_min,
                    epsilon=1.4,
                    smooth=3.0,
                    variation=0.26,
                )
            )

    # --- heavy outer silhouette -----------------------------------------
    if preset.silhouette:
        if subject is not None:
            sil = subject
        else:
            sil = np.zeros((h, w), dtype=bool)
            for m in masks[:-1]:
                sil = sil | m
            sil = core.clean_mask(sil, min_area=(w * h) // 500, close=9)
        black.extend(
            ct.silhouette_strokes(sil, rng, width=preset.silhouette_width, min_area=(w * h) // 900)
        )

    # --- recovered fine detail ------------------------------------------
    if preset.detail:
        lo, hi = preset.detail_thresholds
        em = core.edges(prepared, low=lo, high=hi)
        if subject is not None:
            em = em & subject
        black.extend(ct.detail_strokes(em, rng, width=preset.detail_width))

    # --- border ---------------------------------------------------------
    if preset.border:
        black.extend(border_strokes((w, h), preset.border, rng, preset.border_inset))

    # --- second ink -----------------------------------------------------
    layers: list[Layer] = []
    if preset.spot:
        spot_rgb = SPOT_COLOURS.get(preset.spot)
        if spot_rgb is not None:
            idx = max(0, min(preset.spot_band, len(masks) - 1))
            seed_mask = np.zeros((h, w), dtype=bool)
            for m in masks[: idx + 1]:
                seed_mask = seed_mask | m
            spot = Layer(
                "spot",
                colour=spot_rgb,
                offset=preset.spot_offset,
                opacity=preset.spot_opacity,
            )
            spot.extend(ct.spot_shapes(seed_mask, rng, min_area=(w * h) // 220))
            if len(spot):
                layers.append(spot)  # under the black

    layers.append(black)

    paper = make_paper((w, h), PAPERS.get(preset.paper, PAPERS["cream"]),
                       grain=preset.grain, mottle=preset.mottle, seed=seed)

    image = RasterRenderer((w, h), supersample=supersample).render(layers, paper)

    svg = ""
    if make_svg:
        svg = SvgRenderer((w, h)).render(
            layers,
            PAPERS.get(preset.paper, PAPERS["cream"]),
            title=f"kalam / {preset.name}",
        )

    stats = {
        "size": (w, h),
        "layers": {l.name: {"strokes": len(l.strokes), "shapes": len(l.shapes)} for l in layers},
        "total_marks": sum(len(l) for l in layers),
        "bands": len(masks),
        "seed": seed,
    }
    return PlateResult(image=image, svg=svg, stats=stats)
