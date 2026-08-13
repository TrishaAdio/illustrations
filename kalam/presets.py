"""
Presets.

Each preset is a complete opinion about how to translate tone into marks. They
are built by functions rather than declared as constants so that every call
returns fresh, independently mutable objects - a preset sharing a HatchStyle
between bands would let CLI overrides leak from one pass into another.

Reading a preset: `bands` is listed darkest-first, but in layered mode (the
default) it is *applied* lightest-first and cumulatively. A pass with threshold
0.62 covers everything at or below 0.62, not just the slice above the next
threshold down. So the darkest areas collect every pass and the highlights
collect none, and tone comes out of how many passes overlap.

Because passes stack, each one needs its own angle. Two passes at the same
angle just thicken the same lines; passes at 55, 112 and 18 degrees build a
genuine cross-hatch.
"""

from __future__ import annotations

from .core import PrepConfig
from .hatch import HatchStyle
from .plate import BandSpec, Preset

__all__ = ["PRESETS", "get", "names", "describe"]


def _feluda() -> Preset:
    """Interior line plate: figure work, one ink on cream.

    The Feluda and Shonku interiors are line blocks - single ink, high
    contrast, heavy silhouette, shadow carried by crossing hatch. Deep tone
    goes solid rather than being hatched into mud.
    """
    return Preset(
        name="feluda",
        description="Interior line plate. One ink, heavy silhouette, crossing hatch.",
        long_edge=1600,
        prep=PrepConfig(meanshift=18, clahe_clip=0.0, gamma=1.04),
        bands=[
            BandSpec(0.21, "solid"),
            BandSpec(0.43, "parallel", HatchStyle(
                angle=18, spacing=6.5, width=1.15, wobble_amp=0.8,
                wobble_wavelength=28, min_len=4.0, drift=0.5)),
            BandSpec(0.64, "parallel", HatchStyle(
                angle=112, spacing=6.5, width=1.1, wobble_amp=0.85,
                wobble_wavelength=30, min_len=4.5, drift=0.6)),
            BandSpec(0.85, "parallel", HatchStyle(
                angle=55, spacing=7.5, width=1.05, wobble_amp=0.9,
                wobble_wavelength=34, min_len=5.0, drift=0.7, skip_chance=0.05)),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=3.6,
        interior_contours=False,
        detail=True,
        detail_width=0.95,
        ink="black",
        paper="cream",
    )


def _portrait() -> Preset:
    """Face and figure, hatching that follows the form.

    Rule hatching flattens a face; streamlines through the structure tensor
    wrap around the cheek and jaw. Successive passes rotate the field so they
    cross each other instead of retracing the same lines.
    """
    return Preset(
        name="portrait",
        description="Form-following hatch for faces. Streamlines wrap the structure.",
        long_edge=1500,
        prep=PrepConfig(meanshift=22, meanshift_colour=38, clahe_clip=0.0,
                        band_scale=0.4),
        bands=[
            BandSpec(0.22, "solid"),
            BandSpec(0.46, "flow", HatchStyle(
                spacing=6.0, width=1.1, wobble_amp=0.5, wobble_wavelength=40,
                min_len=7.0, taper=0.42, flow_rotate=90.0)),
            BandSpec(0.68, "flow", HatchStyle(
                spacing=6.5, width=1.0, wobble_amp=0.55, wobble_wavelength=44,
                min_len=9.0, taper=0.45, flow_rotate=0.0)),
            BandSpec(0.88, "none"),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=3.0,
        interior_contours=False,
        detail=True,
        detail_width=0.9,
        detail_thresholds=(40.0, 115.0),
        isolate=False,  # opt in with --isolate; unreliable on busy backgrounds
        ink="black",
        paper="cream",
    )


def _sandesh() -> Preset:
    """Two-ink cover: bold, flat, bordered.

    Fewer passes and larger solid masses than an interior plate, and the spot
    colour deliberately out of register - that misalignment is most of the
    charm of the letterpress covers.
    """
    return Preset(
        name="sandesh",
        description="Two-ink cover. Bold flat masses, ruled border, spot out of register.",
        long_edge=1400,
        prep=PrepConfig(meanshift=26, meanshift_colour=44, clahe_clip=0.0, gamma=1.10,
                        band_scale=0.32, min_area_frac=0.004),
        bands=[
            BandSpec(0.32, "solid"),
            BandSpec(0.58, "parallel", HatchStyle(
                angle=45, spacing=8.0, width=1.5, wobble_amp=0.9,
                wobble_wavelength=28, min_len=4.5)),
            BandSpec(0.80, "parallel", HatchStyle(
                angle=125, spacing=9.0, width=1.35, wobble_amp=1.0,
                wobble_wavelength=32, min_len=5.5)),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=4.4,
        interior_contours=False,
        detail=False,
        spot="brick",
        spot_band=0,
        spot_offset=(6.0, -5.0),
        spot_opacity=0.92,
        ink="soot",
        paper="newsprint",
        border="sandesh",
        border_inset=0.04,
    )


def _shonku() -> Preset:
    """Scratchier and finer - the Professor Shonku register.

    Tighter spacing, thinner nib, four crossing passes, indigo second ink.
    Reads as pen-and-nib rather than brush.
    """
    return Preset(
        name="shonku",
        description="Fine scratchy nib work, four crossing passes, indigo second ink.",
        long_edge=1600,
        prep=PrepConfig(meanshift=16, clahe_clip=0.0, band_scale=0.5),
        bands=[
            BandSpec(0.18, "solid"),
            BandSpec(0.38, "parallel", HatchStyle(
                angle=35, spacing=4.6, width=0.95, wobble_amp=0.55,
                wobble_wavelength=20, min_len=3.0, taper=0.3)),
            BandSpec(0.56, "parallel", HatchStyle(
                angle=100, spacing=4.6, width=0.9, wobble_amp=0.6,
                wobble_wavelength=22, min_len=3.0, taper=0.32)),
            BandSpec(0.74, "parallel", HatchStyle(
                angle=160, spacing=5.0, width=0.9, wobble_amp=0.6,
                wobble_wavelength=24, min_len=3.5, taper=0.32)),
            BandSpec(0.90, "parallel", HatchStyle(
                angle=70, spacing=6.0, width=0.85, wobble_amp=0.65,
                wobble_wavelength=26, min_len=4.0, skip_chance=0.1)),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=2.6,
        interior_contours=False,
        detail=True,
        detail_width=0.8,
        detail_thresholds=(35.0, 100.0),
        spot="indigo",
        spot_band=0,
        spot_offset=(4.0, 3.0),
        spot_opacity=0.85,
        ink="black",
        paper="white",
    )


def _chapter_head() -> Preset:
    """Wide banner for the head of a chapter.

    Light and airy, dissolving at the margins so it can sit above text without
    competing with it. No border, no second ink.
    """
    return Preset(
        name="chapter-head",
        description="Wide, light banner that dissolves at the margins. Sits above text.",
        long_edge=1800,
        prep=PrepConfig(meanshift=18, clahe_clip=0.0, gamma=0.96),
        bands=[
            BandSpec(0.18, "solid"),
            BandSpec(0.44, "parallel", HatchStyle(
                angle=50, spacing=8.0, width=1.1, wobble_amp=0.85,
                wobble_wavelength=30, min_len=4.5)),
            BandSpec(0.70, "parallel", HatchStyle(
                angle=120, spacing=9.5, width=1.0, wobble_amp=0.95,
                wobble_wavelength=34, min_len=5.5, skip_chance=0.12)),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=2.4,
        interior_contours=False,
        detail=True,
        detail_width=0.85,
        vignette=0.55,
        aspect=2.6,
        crop_bias=0.22,  # keep the upper part; a centred banner crop bisects faces
        ink="black",
        paper="cream",
    )


def _linocut() -> Preset:
    """Maximum reduction: two tones, big flat masses, gouge-like outline.

    Closest to the woodcut end of Ray's range, and the most forgiving preset
    for a difficult or low-contrast source. Interior contours stay on here,
    because the carved edge between masses *is* the drawing.
    """
    return Preset(
        name="linocut",
        description="Two tones, large flat masses, gouge-like swelling outline.",
        long_edge=1400,
        prep=PrepConfig(meanshift=30, meanshift_colour=50, clahe_clip=0.0, gamma=1.12,
                        band_scale=0.28, min_area_frac=0.006),
        bands=[
            BandSpec(0.44, "solid"),
            BandSpec(0.74, "parallel", HatchStyle(
                angle=44, spacing=9.0, width=2.0, wobble_amp=1.2,
                wobble_wavelength=36, min_len=6.0, taper=0.25)),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=5.0,
        interior_contours=True,
        interior_width=2.0,
        interior_min_area_frac=0.01,
        detail=False,
        ink="soot",
        paper="aged",
    )


def _scene() -> Preset:
    """Streets, interiors, landscape.

    Architectural subjects want long uninterrupted runs - skies, walls,
    shutters - so spacing is wider and runs are allowed to be long.
    """
    return Preset(
        name="scene",
        description="Architecture and landscape. Long rule hatching across broad areas.",
        long_edge=1800,
        prep=PrepConfig(meanshift=20, clahe_clip=0.0, gamma=1.02),
        bands=[
            BandSpec(0.20, "solid"),
            BandSpec(0.42, "parallel", HatchStyle(
                angle=58, spacing=6.5, width=1.2, wobble_amp=0.8,
                wobble_wavelength=34, min_len=4.5, drift=0.9)),
            BandSpec(0.64, "parallel", HatchStyle(
                angle=122, spacing=7.0, width=1.1, wobble_amp=0.9,
                wobble_wavelength=40, min_len=5.5, drift=1.1)),
            BandSpec(0.86, "parallel", HatchStyle(
                angle=15, spacing=9.0, width=1.05, wobble_amp=1.0,
                wobble_wavelength=44, min_len=7.0, drift=1.2, skip_chance=0.1)),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=2.8,
        interior_contours=False,
        detail=True,
        detail_width=0.9,
        ink="black",
        paper="cream",
    )


def _stipple() -> Preset:
    """Dotted tone, for soft or dusty passages - mist, dusk, smoke."""
    return Preset(
        name="stipple",
        description="Dotted tone for mist, dusk and smoke. Soft, no rule hatching.",
        long_edge=1500,
        prep=PrepConfig(meanshift=18, clahe_clip=0.0),
        bands=[
            BandSpec(0.20, "solid"),
            BandSpec(0.44, "stipple", HatchStyle(spacing=4.0)),
            BandSpec(0.68, "stipple", HatchStyle(spacing=5.0)),
            BandSpec(0.88, "stipple", HatchStyle(spacing=6.5)),
            BandSpec(1.01, "none"),
        ],
        silhouette=True,
        silhouette_width=2.6,
        interior_contours=False,
        detail=True,
        detail_width=0.85,
        ink="black",
        paper="cream",
    )


PRESETS: dict[str, callable] = {
    "feluda": _feluda,
    "portrait": _portrait,
    "sandesh": _sandesh,
    "shonku": _shonku,
    "chapter-head": _chapter_head,
    "linocut": _linocut,
    "scene": _scene,
    "stipple": _stipple,
}


def get(name: str) -> Preset:
    key = name.strip().lower()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}")
    return PRESETS[key]()


def names() -> list[str]:
    return sorted(PRESETS)


def describe() -> list[tuple[str, str]]:
    return [(n, PRESETS[n]().description) for n in names()]
