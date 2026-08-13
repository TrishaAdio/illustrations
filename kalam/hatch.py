"""
Hatching engines.

Given a flat region, produce the marks that stand in for its tone. Four modes:

    parallel  - straight rule hatching at a fixed angle
    cross     - two or three passes at differing angles, for deep shadow
    flow      - streamlines that follow the form (cheekbones, folds, drapery)
    stipple   - dots, for soft or dusty passages

The details that matter are unglamorous: every run of hatching is clipped to
the region, given tapered ends so it reads as a pen lifting, wobbled off true,
and its endpoints nudged so consecutive runs don't line up into a visible seam.
Skip any one of those and the output looks like a halftone screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stroke import Stroke, jitter_endpoints, resample, wobble

Array = np.ndarray


@dataclass
class HatchStyle:
    """How one band of tone should be marked."""

    angle: float = 45.0  # degrees
    spacing: float = 6.0  # perpendicular gap between lines, px
    width: float = 1.3  # nib width
    taper: float = 0.4
    end_width: float = 0.12
    wobble_amp: float = 0.85
    wobble_wavelength: float = 30.0
    drift: float = 0.6  # slow bow across the whole run
    min_len: float = 3.5  # discard shorter runs, they read as dirt
    end_jitter: float = 1.5
    step: float = 2.0  # vertex spacing along a run
    cross_angles: list[float] = field(default_factory=list)  # extra passes
    spacing_jitter: float = 0.18  # fraction of spacing, breaks up regularity
    skip_chance: float = 0.0  # randomly drop runs, for a looser feel
    flow_rotate: float = 0.0  # degrees to rotate the flow field, for flow mode


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _runs(inside: Array, min_run: int = 1) -> list[tuple[int, int]]:
    """Contiguous True spans in a boolean array, as (start, end) inclusive."""
    if inside.size == 0 or not inside.any():
        return []
    padded = np.concatenate([[False], inside, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_run]


def _sample_mask(mask: Array, xs: Array, ys: Array) -> Array:
    """Nearest-neighbour mask lookup with bounds handling."""
    h, w = mask.shape
    xi = np.rint(xs).astype(np.int32)
    yi = np.rint(ys).astype(np.int32)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(xs.shape, dtype=bool)
    if ok.any():
        out[ok] = mask[yi[ok], xi[ok]]
    return out


def _sample_weight(weight: Array | None, xs: Array, ys: Array) -> Array:
    if weight is None:
        return np.ones(xs.shape, dtype=np.float32)
    h, w = weight.shape
    xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return weight[yi, xi]


def _finish(
    p0: Array,
    p1: Array,
    style: HatchStyle,
    rng: np.random.Generator,
    width_scale: float = 1.0,
) -> Stroke | None:
    """Turn a clipped run into a finished, hand-looking stroke."""
    seg = np.array([p0, p1], dtype=np.float32)
    seg = jitter_endpoints(seg, rng, style.end_jitter)
    pts = wobble(
        seg,
        rng,
        amp=style.wobble_amp,
        wavelength=style.wobble_wavelength,
        step=style.step,
        drift=style.drift,
    )
    if len(pts) < 2:
        return None
    w = style.width * width_scale * float(rng.uniform(0.88, 1.14))
    return Stroke(
        pts,
        width=max(0.35, w),
        taper=style.taper,
        end_width=style.end_width,
    )


# --------------------------------------------------------------------------
# parallel / cross
# --------------------------------------------------------------------------


def parallel_hatch(
    mask: Array,
    style: HatchStyle,
    rng: np.random.Generator,
    weight: Array | None = None,
    angle: float | None = None,
    width_scale: float = 1.0,
) -> list[Stroke]:
    """Rule hatching at one angle, clipped to `mask`.

    Lines are generated in a rotated frame that spans the image diagonal, so
    any angle is covered without special-casing orientation.
    """
    if not mask.any():
        return []

    h, w = mask.shape
    theta = np.deg2rad(style.angle if angle is None else angle)
    ct, st = float(np.cos(theta)), float(np.sin(theta))

    # unit vector along the lines, and its normal (the stepping direction)
    ux, uy = ct, st
    nx, ny = -st, ct

    cx, cy = w / 2.0, h / 2.0
    diag = float(np.hypot(w, h)) / 2.0 + 4.0
    spacing = max(0.8, style.spacing)
    n_lines = int(np.ceil(2 * diag / spacing)) + 1

    strokes: list[Stroke] = []
    sample_step = 1.0
    n_samples = int(np.ceil(2 * diag / sample_step)) + 1
    t = np.linspace(-diag, diag, n_samples)

    for i in range(n_lines):
        off = -diag + i * spacing
        if style.spacing_jitter:
            off += float(rng.uniform(-1, 1)) * spacing * style.spacing_jitter

        ox, oy = cx + nx * off, cy + ny * off
        xs = ox + ux * t
        ys = oy + uy * t

        inside = _sample_mask(mask, xs, ys)
        if not inside.any():
            continue

        wts = _sample_weight(weight, xs, ys)
        min_run = max(1, int(style.min_len / sample_step))

        for s, e in _runs(inside, min_run=min_run):
            if style.skip_chance and rng.random() < style.skip_chance:
                continue
            # regions weighted below 1 thin out probabilistically
            wm = float(wts[s : e + 1].mean())
            if wm < 1.0 and rng.random() > wm:
                continue
            p0 = np.array([xs[s], ys[s]], dtype=np.float32)
            p1 = np.array([xs[e], ys[e]], dtype=np.float32)
            if float(np.hypot(*(p1 - p0))) < style.min_len:
                continue
            stk = _finish(p0, p1, style, rng, width_scale=width_scale)
            if stk is not None:
                strokes.append(stk)

    return strokes


def cross_hatch(
    mask: Array,
    style: HatchStyle,
    rng: np.random.Generator,
    weight: Array | None = None,
    width_scale: float = 1.0,
) -> list[Stroke]:
    """Primary pass plus every angle in `style.cross_angles`.

    Later passes are drawn slightly finer and slightly sparser: a real hand
    presses less on the second crossing, and matching densities exactly
    produces an obvious mechanical grid.
    """
    out = parallel_hatch(mask, style, rng, weight=weight, width_scale=width_scale)
    for k, ang in enumerate(style.cross_angles, start=1):
        sub = HatchStyle(**{**style.__dict__, "cross_angles": []})
        sub.spacing = style.spacing * (1.0 + 0.12 * k)
        out += parallel_hatch(
            mask,
            sub,
            rng,
            weight=weight,
            angle=ang,
            width_scale=width_scale * (0.92**k),
        )
    return out


# --------------------------------------------------------------------------
# flow (form-following)
# --------------------------------------------------------------------------


def flow_hatch(
    mask: Array,
    flow: Array,
    style: HatchStyle,
    rng: np.random.Generator,
    weight: Array | None = None,
    max_len: float = 90.0,
    width_scale: float = 1.0,
) -> list[Stroke]:
    """Evenly spaced streamlines through a direction field.

    Uses the Jobard-Lefebvre idea: integrate a streamline, then refuse to start
    another one closer than `spacing` to any existing line. The occupancy grid
    is what keeps the marks evenly dense instead of clumping.
    """
    if not mask.any():
        return []

    # Rotating the field lets successive cumulative passes cross each other
    # while still following the form, instead of retracing the same streamlines.
    if style.flow_rotate:
        flow = flow + np.deg2rad(style.flow_rotate)

    h, w = mask.shape
    spacing = max(1.0, style.spacing)
    cell = spacing / np.sqrt(2.0)
    gh, gw = int(np.ceil(h / cell)) + 1, int(np.ceil(w / cell)) + 1
    occupied = np.zeros((gh, gw), dtype=bool)

    def too_close(x: float, y: float) -> bool:
        gx, gy = int(x / cell), int(y / cell)
        if not (0 <= gx < gw and 0 <= gy < gh):
            return True
        x0, x1 = max(0, gx - 1), min(gw, gx + 2)
        y0, y1 = max(0, gy - 1), min(gh, gy + 2)
        return bool(occupied[y0:y1, x0:x1].any())

    def mark(x: float, y: float) -> None:
        gx, gy = int(x / cell), int(y / cell)
        if 0 <= gx < gw and 0 <= gy < gh:
            occupied[gy, gx] = True

    # seed order is randomised so the field doesn't fill in scanline order
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    idx = rng.permutation(len(xs))
    seed_stride = max(1, int(len(xs) / max(1, (h * w) / (spacing * spacing * 2))))
    idx = idx[::seed_stride]

    step = 1.4
    max_steps = int(max_len / step)
    strokes: list[Stroke] = []

    for i in idx:
        sx, sy = float(xs[i]), float(ys[i])
        if too_close(sx, sy):
            continue

        pts_f: list[tuple[float, float]] = []
        pts_b: list[tuple[float, float]] = []

        for direction, acc in ((1.0, pts_f), (-1.0, pts_b)):
            x, y = sx, sy
            for _ in range(max_steps // 2):
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < w and 0 <= yi < h) or not mask[yi, xi]:
                    break
                a = float(flow[yi, xi])
                dx, dy = np.cos(a) * step * direction, np.sin(a) * step * direction
                # midpoint integration keeps streamlines smooth through curvature
                mxi, myi = int(round(x + dx * 0.5)), int(round(y + dy * 0.5))
                if 0 <= mxi < w and 0 <= myi < h:
                    am = float(flow[myi, mxi])
                    # resolve the 180-degree ambiguity of an orientation field
                    if np.cos(am - a) < 0:
                        am += np.pi
                    dx, dy = np.cos(am) * step * direction, np.sin(am) * step * direction
                x, y = x + dx, y + dy
                acc.append((x, y))

        line = list(reversed(pts_b)) + [(sx, sy)] + pts_f
        if len(line) < 3:
            continue
        arr = np.array(line, dtype=np.float32)
        seg_len = float(np.hypot(*np.diff(arr, axis=0).T).sum())
        if seg_len < max(style.min_len, spacing):
            continue

        for x, y in arr:
            mark(float(x), float(y))

        wm = float(_sample_weight(weight, arr[:, 0], arr[:, 1]).mean())
        if wm < 1.0 and rng.random() > wm:
            continue
        if style.skip_chance and rng.random() < style.skip_chance:
            continue

        pts = wobble(
            arr,
            rng,
            amp=style.wobble_amp * 0.6,
            wavelength=style.wobble_wavelength,
            step=style.step,
            drift=0.0,
        )
        strokes.append(
            Stroke(
                pts,
                width=max(0.35, style.width * width_scale * float(rng.uniform(0.88, 1.12))),
                taper=style.taper,
                end_width=style.end_width,
            )
        )

    return strokes


# --------------------------------------------------------------------------
# stipple
# --------------------------------------------------------------------------


def stipple(
    mask: Array,
    style: HatchStyle,
    rng: np.random.Generator,
    density: Array | None = None,
    dot: float = 1.5,
    weight: Array | None = None,
) -> list[Stroke]:
    """Dots on a jittered grid, thinned by `density` (1 = keep all).

    Emitted as degenerate two-point strokes so round caps render them as dots
    in both the raster and the SVG path.
    """
    if not mask.any():
        return []
    h, w = mask.shape
    spacing = max(1.2, style.spacing * 0.55)

    gy, gx = np.mgrid[0 : h : spacing, 0 : w : spacing].astype(np.float32)
    gx = gx.ravel() + rng.uniform(-spacing * 0.5, spacing * 0.5, gx.size).astype(np.float32)
    gy = gy.ravel() + rng.uniform(-spacing * 0.5, spacing * 0.5, gy.size).astype(np.float32)

    keep = _sample_mask(mask, gx, gy)
    gx, gy = gx[keep], gy[keep]
    if gx.size == 0:
        return []

    prob = _sample_weight(density, gx, gy) * _sample_weight(weight, gx, gy)
    keep2 = rng.random(gx.size) < np.clip(prob, 0.0, 1.0)
    gx, gy = gx[keep2], gy[keep2]

    out: list[Stroke] = []
    for x, y in zip(gx, gy):
        r = float(rng.uniform(0.75, 1.3)) * dot
        pts = np.array([[x, y], [x + 0.35, y + 0.2]], dtype=np.float32)
        out.append(Stroke(pts, width=r, taper=0.0, end_width=1.0))
    return out


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def hatch_band(
    mode: str,
    mask: Array,
    style: HatchStyle,
    rng: np.random.Generator,
    flow: Array | None = None,
    weight: Array | None = None,
    density: Array | None = None,
    width_scale: float = 1.0,
) -> list[Stroke]:
    """Route a band to the requested engine."""
    if mode in ("none", "blank", "paper"):
        return []
    if mode == "parallel":
        return parallel_hatch(mask, style, rng, weight=weight, width_scale=width_scale)
    if mode == "cross":
        return cross_hatch(mask, style, rng, weight=weight, width_scale=width_scale)
    if mode == "flow":
        if flow is None:
            return parallel_hatch(mask, style, rng, weight=weight, width_scale=width_scale)
        return flow_hatch(mask, flow, style, rng, weight=weight, width_scale=width_scale)
    if mode == "stipple":
        return stipple(mask, style, rng, density=density, weight=weight)
    raise ValueError(f"unknown hatch mode: {mode!r}")
