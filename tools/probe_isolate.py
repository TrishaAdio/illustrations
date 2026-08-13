"""Probe background flooding across tolerances, to pick a sane default."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kalam import core, presets  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "samples/tagore.jpg"
outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "out/probe")
outdir.mkdir(parents=True, exist_ok=True)

p = presets.get("portrait")
img = core.load(src, p.long_edge)
flat = core.flatten(img, p.prep)

for tol in (6, 10, 14, 18, 24, 32, 44):
    for name, base in (("raw", img), ("flat", flat)):
        bg = core.background_by_flood(base, tol=tol)
        fg = core.fill_holes(core.clean_mask(~bg, min_area=bg.size // 300, close=9))
        print(f"tol={tol:<3} {name:<5} background={100*bg.mean():5.1f}%  "
              f"subject={100*fg.mean():5.1f}%")
        Image.fromarray((fg * 255).astype(np.uint8)).save(
            outdir / f"fg-{name}-tol{tol:02d}.png"
        )
