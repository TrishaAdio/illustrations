"""
Stage dumps for tuning.

Plates fail in ways that are hard to read from the finished image - mush at the
end could be bad flattening, bad thresholds, a bad subject mask or too many
marks. This writes out each intermediate so the failing stage is obvious, and
prints the tonal histogram that the band thresholds are being compared against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import core
from .plate import Preset


def dump_stages(img: Image.Image, preset: Preset, outdir: str | Path, seed: int = 7) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    img.save(out / "0-input.png")

    flat = core.flatten(img, preset.prep)
    flat.save(out / "1-flattened.png")

    luma = core.to_luma(flat)
    prepared = core.prepare(luma, preset.prep)
    Image.fromarray((prepared * 255).astype(np.uint8)).save(out / "2-prepared.png")

    h, w = prepared.shape
    banding = prepared
    if preset.prep.dither > 0:
        banding = np.clip(
            prepared
            + core.dither_field(
                (h, w), amp=preset.prep.dither, scale=preset.prep.dither_scale, seed=seed
            ),
            0, 1,
        ).astype(np.float32)

    masks = core.quantise(banding, preset.thresholds(), preset.prep)

    # bands as a stepped grey image, darkest band darkest
    idx = np.zeros((h, w), np.uint8)
    n = max(1, len(masks) - 1)
    for i, m in enumerate(masks):
        idx[m] = int(255 * i / n)
    Image.fromarray(idx).save(out / "3-bands.png")

    subject = None
    if preset.isolate:
        subject = core.isolate_subject(img)
        Image.fromarray((subject * 255).astype(np.uint8)).save(out / "4-subject.png")

    lo, hi = preset.detail_thresholds
    em = core.edges(prepared, low=lo, high=hi)
    Image.fromarray((em * 255).astype(np.uint8)).save(out / "5-edges.png")

    stats = {
        "size": (w, h),
        "thresholds": preset.thresholds(),
        "luma": {
            f"p{q}": round(float(np.percentile(prepared, q)), 3)
            for q in (1, 5, 25, 50, 75, 95, 99)
        },
        "band_coverage": [round(float(m.mean()) * 100, 1) for m in masks],
        "subject_coverage": (
            round(float(subject.mean()) * 100, 1) if subject is not None else None
        ),
        "edge_density": round(float(em.mean()) * 100, 2),
    }
    return stats
