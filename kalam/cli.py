"""
Command line interface.

    kalam presets                       list the available looks
    kalam render IN -o OUT.png          make one plate
    kalam variants IN -o DIR            same plate, several seeds
    kalam sheet IN -o OUT.png           every preset side by side
    kalam demo -o DIR                   run on a synthetic subject, no input needed

The global overrides (--spacing-scale, --width-scale, --angle, --gamma ...) are
applied on top of whichever preset was chosen, so a preset is a starting point
rather than a fixed recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import core, presets
from .plate import INKS, PAPERS, SPOT_COLOURS, Preset, render_plate


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------


def apply_overrides(p: Preset, a: argparse.Namespace) -> Preset:
    if a.size:
        p.long_edge = a.size
    if a.gamma is not None:
        p.prep.gamma = a.gamma
    if a.exposure is not None:
        p.prep.exposure = a.exposure
    if a.contrast is not None:
        p.prep.clahe_clip = a.contrast
    if a.smooth is not None:
        p.prep.smooth_passes = a.smooth

    if a.spacing_scale and a.spacing_scale != 1.0:
        for b in p.bands:
            b.style.spacing *= a.spacing_scale
    if a.width_scale and a.width_scale != 1.0:
        for b in p.bands:
            b.style.width *= a.width_scale
        p.silhouette_width *= a.width_scale
        p.interior_width *= a.width_scale
        p.detail_width *= a.width_scale
    if a.wobble_scale and a.wobble_scale != 1.0:
        for b in p.bands:
            b.style.wobble_amp *= a.wobble_scale
    if a.angle is not None:
        delta = a.angle
        for b in p.bands:
            b.style.angle = delta
            b.style.cross_angles = [(c + delta - 45.0) for c in b.style.cross_angles]

    if a.spot is not None:
        p.spot = None if a.spot in ("none", "") else a.spot
    if a.paper:
        p.paper = a.paper
    if a.ink:
        p.ink = a.ink
    if a.border is not None:
        p.border = None if a.border in ("none", "") else a.border
    if a.aspect is not None:
        p.aspect = None if a.aspect <= 0 else a.aspect
    if a.vignette is not None:
        p.vignette = a.vignette
    if a.isolate is not None:
        p.isolate = a.isolate
    if a.no_detail:
        p.detail = False
    if a.no_silhouette:
        p.silhouette = False
    return p


def add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("-p", "--preset", default="feluda", help="preset name")
    sp.add_argument("--seed", type=int, default=7, help="controls all randomness")
    sp.add_argument("--size", type=int, default=0, help="long edge in px")
    sp.add_argument("--supersample", type=int, default=3, help="raster AA factor")

    g = sp.add_argument_group("tone")
    g.add_argument("--gamma", type=float, default=None)
    g.add_argument("--exposure", type=float, default=None)
    g.add_argument("--contrast", type=float, default=None, help="CLAHE clip, 0 disables")
    g.add_argument("--smooth", type=int, default=None, help="bilateral passes")

    g = sp.add_argument_group("marks")
    g.add_argument("--spacing-scale", type=float, default=1.0)
    g.add_argument("--width-scale", type=float, default=1.0)
    g.add_argument("--wobble-scale", type=float, default=1.0)
    g.add_argument("--angle", type=float, default=None, help="primary hatch angle, deg")
    g.add_argument("--no-detail", action="store_true")
    g.add_argument("--no-silhouette", action="store_true")

    g = sp.add_argument_group("plate")
    g.add_argument("--spot", default=None, choices=sorted(SPOT_COLOURS) + ["none"])
    g.add_argument("--paper", default=None, choices=sorted(PAPERS))
    g.add_argument("--ink", default=None, choices=sorted(INKS))
    g.add_argument("--border", default=None, choices=["none", "rule", "double", "sandesh"])
    g.add_argument("--aspect", type=float, default=None, help="crop to w/h, 0 disables")
    g.add_argument("--vignette", type=float, default=None)
    g.add_argument("--isolate", dest="isolate", action="store_true", default=None)
    g.add_argument("--no-isolate", dest="isolate", action="store_false", default=None)
    g.add_argument("--no-svg", action="store_true")
    g.add_argument("--preview", type=int, default=0,
                   help="also write a downscaled JPEG preview of this long edge")


# --------------------------------------------------------------------------
# synthetic subject, so the tool is testable with no photograph to hand
# --------------------------------------------------------------------------


def synthetic_subject(size: tuple[int, int] = (900, 1200)) -> Image.Image:
    """A crude lit figure against a wall. Enough tonal structure to exercise
    every band, silhouette detection and the flow field."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # wall with a light gradient from an off-frame window
    img = 0.62 + 0.3 * (1.0 - xx / w) * (1.0 - 0.4 * yy / h)

    # floor
    img = np.where(yy > h * 0.80, 0.34 + 0.1 * (xx / w), img)

    def blob(cx, cy, rx, ry, val, soft=0.06):
        d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        m = np.clip((1.0 - d) / soft, 0.0, 1.0)
        return m, val

    parts = [
        blob(w * 0.5, h * 0.72, w * 0.30, h * 0.26, 0.30),   # torso
        blob(w * 0.5, h * 0.34, w * 0.155, h * 0.115, 0.66),  # head
        blob(w * 0.5, h * 0.475, w * 0.075, h * 0.055, 0.44),  # neck
        blob(w * 0.30, h * 0.74, w * 0.085, h * 0.20, 0.24),  # left arm
        blob(w * 0.70, h * 0.74, w * 0.085, h * 0.20, 0.24),  # right arm
    ]
    for m, val in parts:
        img = img * (1 - m) + val * m

    # light falling from the left: brighten the lit side of head and torso
    lit = np.clip(1.0 - (xx - w * 0.34) / (w * 0.34), 0.0, 1.0) * 0.30
    body = (parts[0][0] + parts[1][0] + parts[2][0]) > 0.4
    img = np.where(body, np.clip(img + lit, 0, 1), img)

    # hair mass, and features
    hm, _ = blob(w * 0.5, h * 0.275, w * 0.16, h * 0.075, 0.0, soft=0.10)
    img = img * (1 - hm) + 0.07 * hm
    for cx, cy, rx, ry, v in [
        (0.445, 0.335, 0.022, 0.011, 0.10),  # eye
        (0.555, 0.335, 0.022, 0.011, 0.10),
        (0.50, 0.375, 0.014, 0.028, 0.42),   # nose
        (0.50, 0.415, 0.045, 0.010, 0.20),   # mouth
    ]:
        m, _ = blob(w * cx, h * cy, w * rx, h * ry, 0.0, soft=0.5)
        img = img * (1 - m) + v * m

    img = np.clip(img, 0.02, 0.98)
    rgb = (np.stack([img, img * 0.985, img * 0.95], axis=2) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_presets(a: argparse.Namespace) -> int:
    width = max(len(n) for n in presets.names())
    print()
    for name, desc in presets.describe():
        print(f"  {name.ljust(width)}   {desc}")
    print()
    print(f"  spots:  {', '.join(sorted(SPOT_COLOURS))}")
    print(f"  papers: {', '.join(sorted(PAPERS))}")
    print(f"  inks:   {', '.join(sorted(INKS))}")
    print()
    return 0


def _load_input(path: str, long_edge: int) -> Image.Image:
    if path == ":synthetic:":
        return synthetic_subject()
    return core.load(path, long_edge=long_edge)


def _render_one(a: argparse.Namespace, src: str, out: Path, seed: int) -> dict:
    preset = apply_overrides(presets.get(a.preset), a)
    img = _load_input(src, preset.long_edge)
    res = render_plate(
        img, preset, seed=seed, supersample=a.supersample, make_svg=not a.no_svg
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    res.image.save(out)
    if res.svg:
        svg_path = out.with_suffix(".svg")
        svg_path.write_text(res.svg, encoding="utf-8")
    if getattr(a, "preview", 0):
        prev = res.image.copy()
        prev.thumbnail((a.preview, a.preview), Image.LANCZOS)
        prev.convert("RGB").save(out.with_suffix(".preview.jpg"), quality=82, optimize=True)
    return res.stats


def cmd_render(a: argparse.Namespace) -> int:
    out = Path(a.output)
    stats = _render_one(a, a.input, out, a.seed)
    print(f"  wrote {out}  ({stats['size'][0]}x{stats['size'][1]}, "
          f"{stats['total_marks']} marks)")
    if not a.no_svg:
        print(f"  wrote {out.with_suffix('.svg')}")
    if a.stats:
        print(json.dumps(stats, indent=2))
    return 0


def cmd_variants(a: argparse.Namespace) -> int:
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(a.count):
        seed = a.seed + i * 101
        out = outdir / f"{a.preset}-seed{seed}.png"
        stats = _render_one(a, a.input, out, seed)
        print(f"  wrote {out}  ({stats['total_marks']} marks)")
    return 0


def cmd_sheet(a: argparse.Namespace) -> int:
    """Every preset on one sheet. The fastest way to choose a look."""
    from PIL import ImageFont

    chosen = presets.names() if a.all else [n for n in presets.names() if n != "stipple"]
    thumbs: list[tuple[str, Image.Image]] = []

    for name in chosen:
        p = presets.get(name)
        p.long_edge = a.size or 700
        p.aspect = None  # keep the sheet uniform
        img = _load_input(a.input, p.long_edge)
        res = render_plate(img, p, seed=a.seed, supersample=2, make_svg=False)
        thumbs.append((name, res.image))
        print(f"  rendered {name}")

    cols = a.columns
    rows = (len(thumbs) + cols - 1) // cols
    cw = max(t.width for _, t in thumbs)
    ch = max(t.height for _, t in thumbs)
    pad, label = 16, 26

    sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad + label) + pad),
                      (250, 248, 243))
    drw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(16)
    except TypeError:
        font = ImageFont.load_default()

    for i, (name, t) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = pad + r * (ch + pad + label)
        sheet.paste(t, (x + (cw - t.width) // 2, y))
        drw.text((x + 2, y + ch + 5), name, fill=(60, 55, 50), font=font)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"  wrote {out}  ({sheet.width}x{sheet.height})")
    return 0


def cmd_stages(a: argparse.Namespace) -> int:
    from .diagnose import dump_stages

    preset = apply_overrides(presets.get(a.preset), a)
    img = _load_input(a.input, preset.long_edge)
    stats = dump_stages(img, preset, a.output, seed=a.seed)
    print(f"  wrote stage images to {a.output}/")
    print(json.dumps(stats, indent=2))
    return 0


def cmd_demo(a: argparse.Namespace) -> int:
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = outdir / "00-reference.png"
    synthetic_subject().save(ref)
    print(f"  wrote {ref}  (synthetic subject)")

    a.input = ":synthetic:"
    for name in presets.names():
        a.preset = name
        out = outdir / f"{name}.png"
        stats = _render_one(a, a.input, out, a.seed)
        print(f"  wrote {out}  ({stats['total_marks']} marks)")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="kalam",
        description="Turn photographs into Satyajit Ray-style pen-and-ink plates.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("presets", help="list presets, inks and papers")
    sp.set_defaults(func=cmd_presets)

    sp = sub.add_parser("render", help="render a single plate")
    sp.add_argument("input")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--stats", action="store_true")
    add_common(sp)
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("variants", help="render the same plate at several seeds")
    sp.add_argument("input")
    sp.add_argument("-o", "--output", required=True, help="output directory")
    sp.add_argument("-n", "--count", type=int, default=4)
    add_common(sp)
    sp.set_defaults(func=cmd_variants)

    sp = sub.add_parser("sheet", help="contact sheet of every preset")
    sp.add_argument("input")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--columns", type=int, default=4)
    sp.add_argument("--all", action="store_true")
    add_common(sp)
    sp.set_defaults(func=cmd_sheet)

    sp = sub.add_parser("stages", help="dump intermediate stages for tuning")
    sp.add_argument("input")
    sp.add_argument("-o", "--output", required=True, help="output directory")
    add_common(sp)
    sp.set_defaults(func=cmd_stages)

    sp = sub.add_parser("demo", help="render every preset on a synthetic subject")
    sp.add_argument("-o", "--output", default="out/demo")
    add_common(sp)
    sp.set_defaults(func=cmd_demo)

    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
