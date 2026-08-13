"""
Character plate: Akhilesh Bose.

A hand-authored drawing rather than a photograph conversion, but built out of
kalam's own primitives and hatched by its own engines. That is the point: an
invented character has no reference image, yet his plates have to sit beside
photograph-derived ones without looking like they came from a different hand.
Routing the geometry through the same stroke, hatch and paper code guarantees
they match.

Everything is defined as control points and smoothed with Catmull-Rom, so the
shapes are editable by moving a handful of numbers rather than by fighting a
long list of bezier coefficients.

    python plates/akhilesh_bose.py -o out/akhilesh --preview 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalam import contour as ct  # noqa: E402
from kalam.hatch import HatchStyle, cross_hatch, parallel_hatch  # noqa: E402
from kalam.plate import INKS, PAPERS, SPOT_COLOURS, make_paper  # noqa: E402
from kalam.stroke import (  # noqa: E402
    FilledShape,
    Layer,
    RasterRenderer,
    Stroke,
    SvgRenderer,
    wobble,
)

W, H = 900, 1200


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def catmull(pts, closed: bool = False, n: int = 18) -> np.ndarray:
    """Smooth a few control points into a drawable polyline."""
    P = np.asarray(pts, dtype=np.float64)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append(
            0.5
            * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    arr = np.vstack(out)
    if not closed:
        arr = np.vstack([arr, P[-2]])
    return arr.astype(np.float32)


def mask_of(*polys: np.ndarray, subtract: tuple[np.ndarray, ...] = ()) -> np.ndarray:
    """Rasterise polygons to a boolean mask, so hatch engines can fill them."""
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for p in polys:
        d.polygon([(float(x), float(y)) for x, y in p], fill=255)
    for p in subtract:
        d.polygon([(float(x), float(y)) for x, y in p], fill=0)
    return np.asarray(im) > 127


def circle(cx: float, cy: float, rx: float, ry: float | None = None, n: int = 96):
    ry = rx if ry is None else ry
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# the drawing
# --------------------------------------------------------------------------

# Profile facing left, from the crown down the face, round the jaw and back up
# the skull. A heavy brow, a large nose and a firm chin are what make him read
# as sixty-one and stubborn; earlier passes with a smooth egg-shaped skull and a
# small nose read as a hunched bird.
PROFILE = [
    (440, 232), (376, 252), (340, 300),
    (326, 356), (322, 378), (340, 388),          # heavy brow ridge
    (330, 400), (310, 428), (286, 458),          # the nose, well out
    (322, 474), (338, 490),                      # underside, philtrum
    (344, 502), (334, 512), (348, 524),          # lip, mouth, lower lip
    (358, 546), (392, 572), (446, 582),          # firm chin, jaw
    (502, 562), (534, 520), (556, 456),
    (558, 386), (534, 294), (486, 240),
]

# Hair sits above and behind the ear only. He is bald across the crown.
HAIR_BAND = [
    (492, 282), (532, 330), (554, 400), (552, 458),
    (532, 508), (504, 532), (496, 502), (524, 454),
    (528, 392), (512, 336), (482, 296),
]

# Light from the left, so the shadow is the back third of the head.
FACE_SHADOW = [
    (442, 246), (458, 330), (452, 440), (462, 520),
    (446, 578), (500, 560), (532, 520), (554, 456),
    (556, 388), (532, 296), (486, 242),
]

NECK = [(446, 580), (500, 562), (524, 596), (520, 676), (452, 682), (438, 610)]

# Narrower than a dome, with a distinct shoulder point and a break for the arm.
SHOULDERS = [
    (170, 1200), (180, 978), (202, 860), (238, 782), (288, 724),
    (354, 690), (422, 676), (474, 670), (538, 684), (612, 710),
    (678, 752), (724, 828), (750, 948), (758, 1200),
]

# Chador over the left shoulder: the plate's one true solid mass.
CHADOR = [
    (178, 978), (200, 860), (240, 784), (300, 726), (368, 692),
    (394, 750), (354, 852), (314, 980), (294, 1120), (290, 1200),
    (172, 1200),
]

# A narrow shirt opening between the lapels - not the wide oval of the first
# pass, which read as an egg sitting on his chest.
SHIRT = [(452, 628), (492, 626), (516, 700), (502, 772), (452, 776), (432, 700)]

COLLAR_L = [(402, 692), (426, 642), (452, 626)]
COLLAR_R = [(538, 694), (518, 642), (492, 624)]
LAPEL_L = [(402, 692), (358, 792), (344, 906)]
LAPEL_R = [(538, 694), (580, 782), (602, 884)]
SLEEVE_L = [(240, 786), (252, 902), (264, 1034)]
SLEEVE_R = [(724, 830), (708, 962), (702, 1086)]

LENS_C = (378, 396)
LENS_R = 48.0


def build(seed: int = 11) -> tuple[list[Layer], str]:
    rng = np.random.default_rng(seed)

    ink = Layer("ink", colour=INKS["black"])
    spot = Layer(
        "spot",
        colour=SPOT_COLOURS["ochre"],
        offset=(8.0, -7.0),
        opacity=0.88,
    )

    profile = catmull(PROFILE, closed=True)
    hair = catmull(HAIR_BAND, closed=True)
    shadow = catmull(FACE_SHADOW, closed=True)
    neck = catmull(NECK, closed=True)
    coat = catmull(SHOULDERS, closed=False)
    chador = catmull(CHADOR, closed=True)
    shirt = catmull(SHIRT, closed=True)

    # ---- second ink: a loose field behind the head ---------------------
    halo = wobble(circle(430, 402, 244, 252), rng, amp=3.4, wavelength=140.0, step=6.0)
    spot.add(FilledShape(halo))

    # ---- coat: dense cross-hatch, not solid ---------------------------
    coat_poly = np.vstack([coat, [[824, 1200], [120, 1200]]])
    coat_mask = mask_of(coat_poly, subtract=(chador, shirt, neck))
    ink.extend(
        cross_hatch(
            coat_mask,
            HatchStyle(
                angle=62, spacing=5.0, width=1.25, cross_angles=[128],
                wobble_amp=0.8, wobble_wavelength=34, min_len=4.0, drift=0.7,
            ),
            rng,
        )
    )

    # ---- chador: the solid ---------------------------------------------
    ink.add(FilledShape(wobble(chador, rng, amp=1.6, wavelength=70.0, step=3.0)))

    # ---- under-jaw shadow, dense, to lift the head off the coat -------
    ink.extend(
        cross_hatch(
            mask_of(neck),
            HatchStyle(
                angle=20, spacing=3.6, width=1.15, cross_angles=[104],
                wobble_amp=0.6, wobble_wavelength=24, min_len=3.0,
            ),
            rng,
        )
    )

    # ---- face shadow: light, following the turn of the head -----------
    face_shadow_mask = mask_of(shadow, subtract=(hair,))
    ink.extend(
        parallel_hatch(
            face_shadow_mask,
            HatchStyle(
                angle=74, spacing=9.5, width=0.95, wobble_amp=0.7,
                wobble_wavelength=40, min_len=6.0, taper=0.5, skip_chance=0.08,
            ),
            rng,
        )
    )

    # ---- white hair: directional strokes, never a hatch fill ----------
    # He is white-haired, so on paper the hair is mostly bare. Rule hatching
    # across the band reads as a striped helmet; short strokes laid along the
    # curve of the skull read as coarse hair standing out above the ear.
    for i in range(16):
        t = i / 15.0
        ang = np.deg2rad(-64 + t * 128)          # sweep round the back of the skull
        r0 = 96.0 + rng.uniform(-4, 4)
        cx, cy = 458.0, 404.0
        x0 = cx + r0 * np.sin(ang) * 1.02
        y0 = cy - r0 * np.cos(ang) * 1.18
        ln = rng.uniform(26, 44)
        x1 = cx + (r0 + ln) * np.sin(ang) * 1.02
        y1 = cy - (r0 + ln) * np.cos(ang) * 1.18
        mx = (x0 + x1) / 2 + rng.uniform(-5, 5)
        my = (y0 + y1) / 2 + rng.uniform(-5, 5)
        ink.add(
            Stroke(
                wobble(catmull([(x0, y0), (mx, my), (x1, y1)]), rng, amp=0.8,
                       wavelength=18.0, step=1.6),
                width=float(rng.uniform(0.9, 1.5)),
                taper=0.55,
                end_width=0.1,
            )
        )

    # ---- outlines ------------------------------------------------------
    ink.extend(ct.variable_width_stroke(profile, rng, width=3.2, variation=0.34,
                                        chunk=90.0, closed=True, wobble_amp=0.7))
    ink.extend(ct.variable_width_stroke(coat, rng, width=3.6, variation=0.32,
                                        chunk=120.0, closed=False, wobble_amp=0.8))
    ink.extend(ct.variable_width_stroke(shirt, rng, width=1.6, variation=0.28,
                                        chunk=80.0, closed=True, wobble_amp=0.7))

    # ---- coat anatomy: collar, lapels, sleeve break --------------------
    # Without these the shoulders are one smooth dome and read as a mountain
    # rather than a man in a coat. They cost four strokes and do most of the
    # work of making the figure legible.
    for pts, wid in (
        (COLLAR_L, 2.8), (COLLAR_R, 2.8),
        (LAPEL_L, 2.4), (LAPEL_R, 2.4),
        (SLEEVE_L, 2.0), (SLEEVE_R, 2.0),
    ):
        ink.extend(
            ct.variable_width_stroke(
                catmull(pts), rng, width=wid, variation=0.28, chunk=70.0,
                closed=False, wobble_amp=0.7,
            )
        )

    # ---- spectacles: the focal point, heaviest line on the plate ------
    lens = circle(*LENS_C, LENS_R, LENS_R * 1.02)
    ink.extend(ct.variable_width_stroke(lens, rng, width=3.8, variation=0.22,
                                        chunk=60.0, closed=True, wobble_amp=0.5))
    # bridge, forward to the nose
    ink.add(Stroke(wobble(catmull([(330, 394), (318, 390)]), rng, amp=0.4,
                          wavelength=12.0, step=1.5), width=2.6, taper=0.15))
    # temple arm, back to the ear
    ink.add(Stroke(wobble(catmull([(424, 386), (468, 390), (508, 410)]), rng,
                          amp=0.6, wavelength=26.0, step=2.0),
                   width=2.4, taper=0.2, end_width=0.5))
    # a crack in the left lens, refused since 1959
    ink.add(Stroke(wobble(catmull([(352, 366), (370, 396), (362, 430)]), rng,
                          amp=0.4, wavelength=18.0, step=1.5),
                   width=0.9, taper=0.4))

    # ---- features ------------------------------------------------------
    brow = catmull([(322, 352), (354, 338), (396, 336), (416, 346)])
    ink.extend(ct.variable_width_stroke(brow, rng, width=4.2, variation=0.3,
                                        chunk=40.0, closed=False, wobble_amp=0.5))
    # nostril
    ink.add(Stroke(wobble(catmull([(300, 460), (318, 466)]), rng, amp=0.3,
                          wavelength=10.0, step=1.2), width=2.2, taper=0.3))
    # moustache: white, so outline plus a few strokes
    must = catmull([(320, 478), (350, 494), (392, 496), (410, 486)], closed=False)
    ink.extend(ct.variable_width_stroke(must, rng, width=1.8, variation=0.35,
                                        chunk=30.0, closed=False, wobble_amp=0.8))
    for i in range(7):
        x0 = 326 + i * 12
        ink.add(Stroke(wobble(catmull([(x0, 482), (x0 + 5, 498)]), rng, amp=0.5,
                              wavelength=10.0, step=1.5), width=0.85, taper=0.5))
    # mouth
    ink.add(Stroke(wobble(catmull([(340, 512), (362, 515), (384, 511)]), rng,
                          amp=0.4, wavelength=14.0, step=1.5),
                   width=1.6, taper=0.45))
    # ear
    ear = catmull([(500, 428), (518, 442), (520, 470), (504, 484), (496, 464)],
                  closed=False)
    ink.extend(ct.variable_width_stroke(ear, rng, width=1.5, variation=0.3,
                                        chunk=40.0, closed=False, wobble_amp=0.6))

    # ---- umbrella, held as a stick ------------------------------------
    shaft = catmull([(806, 1200), (802, 1020), (800, 862)])
    ink.extend(ct.variable_width_stroke(shaft, rng, width=4.0, variation=0.22,
                                        chunk=110.0, closed=False, wobble_amp=0.7))
    crook = catmull([(800, 862), (796, 820), (776, 800), (754, 812), (752, 836)])
    ink.extend(ct.variable_width_stroke(crook, rng, width=3.6, variation=0.24,
                                        chunk=50.0, closed=False, wobble_amp=0.5))

    # ---- watch chain across the coat ----------------------------------
    chain = catmull([(392, 852), (430, 892), (476, 900), (508, 878)])
    ink.add(Stroke(wobble(chain, rng, amp=0.8, wavelength=30.0, step=2.0),
                   width=1.5, taper=0.3))
    ink.extend(ct.variable_width_stroke(circle(516, 866, 15, 15), rng, width=1.8,
                                        variation=0.25, chunk=40.0, closed=True,
                                        wobble_amp=0.5))

    # ---- background: sparse marks upper right, to push the head forward
    bg = mask_of(
        np.array([[560, 40], [860, 40], [860, 560], [600, 470]], dtype=np.float32),
        subtract=(profile, halo),
    )
    ink.extend(
        parallel_hatch(
            bg,
            HatchStyle(
                angle=52, spacing=15.0, width=1.0, wobble_amp=1.2,
                wobble_wavelength=48, min_len=14.0, taper=0.6, skip_chance=0.3,
                drift=1.4,
            ),
            rng,
        )
    )

    layers = [spot, ink]
    paper = make_paper((W, H), PAPERS["cream"], grain=0.014, mottle=0.024, seed=seed)
    image = RasterRenderer((W, H), supersample=3).render(layers, paper)
    svg = SvgRenderer((W, H)).render(layers, PAPERS["cream"],
                                     title="Akhilesh Bose / kalam")
    return image, svg


def main() -> int:
    ap = argparse.ArgumentParser(description="Render the Akhilesh Bose character plate.")
    ap.add_argument("-o", "--output", default="out/akhilesh/akhilesh-bose")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image, svg = build(a.seed)
    image.save(out.with_suffix(".png"))
    out.with_suffix(".svg").write_text(svg, encoding="utf-8")
    print(f"  wrote {out.with_suffix('.png')}")
    print(f"  wrote {out.with_suffix('.svg')}")
    if a.preview:
        p = image.copy()
        p.thumbnail((a.preview, a.preview), Image.LANCZOS)
        p.convert("RGB").save(out.with_suffix(".preview.jpg"), quality=84, optimize=True)
        print(f"  wrote {out.with_suffix('.preview.jpg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
