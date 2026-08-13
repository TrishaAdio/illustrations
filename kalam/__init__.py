"""
kalam - Satyajit Ray-style pen-and-ink plates from photographs.

    from kalam import presets, render_plate, load

    img = load("reference.jpg", long_edge=1600)
    res = render_plate(img, presets.get("feluda"), seed=7)
    res.image.save("plate.png")
    open("plate.svg", "w").write(res.svg)

The pipeline is: photograph -> denoise -> local contrast -> quantise to a few
flat tonal bands -> represent each band with marks (hatch, cross-hatch,
streamlines, stipple or solid fill) -> outline with a heavy silhouette and fine
interiors -> composite on paper with an optional second ink out of register.
"""

from .core import PrepConfig, load, to_luma
from .hatch import HatchStyle
from .plate import BandSpec, INKS, PAPERS, Preset, SPOT_COLOURS, render_plate
from .stroke import FilledShape, Layer, Stroke

from . import presets  # noqa: E402  (after Preset, which presets imports)

__version__ = "0.1.0"

__all__ = [
    "load",
    "to_luma",
    "PrepConfig",
    "HatchStyle",
    "BandSpec",
    "Preset",
    "render_plate",
    "presets",
    "Stroke",
    "FilledShape",
    "Layer",
    "SPOT_COLOURS",
    "PAPERS",
    "INKS",
    "__version__",
]
