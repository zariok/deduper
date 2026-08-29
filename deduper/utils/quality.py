"""Image quality measurement, modification detection and best-file ranking.

"Best" used to mean the most pixels. That is wrong in both directions: an
upscaled copy has more pixels and less picture than its source, and two files of
the same size can differ by a heavily compressed JPEG, a colour filter or a
burned-in watermark. This module measures what a file actually carries:

* **Detail bandwidth** - the fraction of the image's own Nyquist frequency that
  still holds picture information, read off the radial power spectrum. Natural
  images carry roughly equal energy per octave; an upscale or a blur has none
  above the resolution it was scaled from, so the spectrum falls off a cliff at
  that point. ``pixel_count * bandwidth**2`` is the resolution the file
  effectively has. JPEG artifacts are deliberately *not* captured here (block
  edges put energy back at high frequency); compression is read from the
  quantisation tables instead.
* **Codec quality** - lossless formats score 100, JPEGs are estimated from the
  luminance quantisation table's scale against the standard table.
* **Modifications** - group-relative: an overlay (watermark, caption, logo) is
  a small dense region where one copy differs from the others and carries more
  edges than they do; a filter is a shift in warmth, brightness, saturation or
  contrast against the others; a different aspect ratio is a crop or padding.
  These need the other copies to be visible at all, so they are computed per
  group rather than per file.

Everything here is pure: paths in, numbers out. Caching and orchestration live
in ``deduper.services.quality_analysis``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from .logging_config import get_logger

logger = get_logger(__name__)

# Bump when the measurements change meaning, so cached rows are recomputed.
QUALITY_VERSION = 1

# --- Detail bandwidth ---------------------------------------------------------
# Spectrum rings are log-spaced, this many per octave, from 1/16 to 1.0 of the
# Nyquist frequency. The lowest octave is the reference every other ring is
# measured against; bandwidth ends at the first ring whose energy falls under
# BANDWIDTH_CUTOFF of it.
_RINGS_PER_OCTAVE = 4
_RING_COUNT = 4 * _RINGS_PER_OCTAVE
_REFERENCE_RINGS = _RINGS_PER_OCTAVE  # rings in [1/16, 1/8)
_FIRST_JUDGED_RING = _RINGS_PER_OCTAVE  # bandwidth can fall no lower than 1/8
BANDWIDTH_CUTOFF = 0.2
# Native pixels are sampled in tiles rather than decoded into one huge FFT.
_TILE = 512
# Below this the spectrum has too few bins per ring to say anything.
_MIN_JUDGEABLE_SIDE = 128

# pixel_count * bandwidth**2 is the information a file carries; the extra half
# power is a thumb on the scale so an original beats its own upscale, whose
# effective pixels are otherwise equal to the source's plus interpolation blur.
DETAIL_EXPONENT = 2.5

# --- Codec ------------------------------------------------------------------
# IJG standard luminance quantisation table, natural order. Pillow returns
# tables in this order, so the ratio against it gives the encoder's scale.
_STD_LUMA_QUANT = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)
_LOSSLESS_FORMATS = frozenset({"PNG", "BMP", "TIFF", "PPM", "PBM", "PGM"})
# Quality factor applied for a codec: lossless 1.0, unknown lossy 0.95, JPEG on
# a gentle curve. Compression blur already dents the bandwidth; this is the
# tie-breaker between equally detailed copies.
_UNKNOWN_CODEC_FACTOR = 0.95

# --- Modification detection -------------------------------------------------
COMPARISON_SIZE = 320
# Aspect ratios further apart than this cannot be aligned pixel to pixel.
ASPECT_TOLERANCE = 0.02
# Normalised absolute difference (in standard deviations, box-filtered) above
# which a pixel counts as changed.
_DIFF_THRESHOLD = 0.6
_DIFF_BOX = 5
# A changed area is an overlay when it is neither negligible nor most of the
# frame, and the changed pixels are packed into their bounding box rather than
# scattered - a reframe or re-encode spreads change everywhere thinly.
_OVERLAY_MIN_FRACTION = 0.0005
_OVERLAY_MAX_FRACTION = 0.25
_OVERLAY_MAX_BOX_FRACTION = 0.6
_OVERLAY_MIN_DENSITY = 0.25
# Rows/columns count as part of the region when they hold at least this share
# of the busiest row's/column's changed pixels; the box must then hold most of
# the changed mass, or the change is really global.
_PROFILE_FRACTION = 0.2
_OVERLAY_MIN_CONTAINED = 0.5
# Edge energy in the region, relative to the rest of the image, must exceed the
# reference's by this factor to attribute the overlay to a copy.
_OVERLAY_EDGE_RATIO = 1.3
# ...and, for text, must itself be denser than the copy's own surroundings; a
# solid overlay is one far flatter than its surroundings.
_TEXT_EDGE_RATIO = 1.3
_SOLID_EDGE_RATIO = 0.3
# A boxed overlay's border steps at least this many typical pixel steps, and
# this many times harder than the same border in the other copy.
_BORDER_STEP_RATIO = 3.0
_BORDER_MARGIN = 2.0
# Colour statistics are on the 0-255 scale.
_WARMTH_THRESHOLD = 10.0
_LUMINANCE_THRESHOLD = 12.0
_SATURATION_THRESHOLD = 8.0
_CONTRAST_RATIO = 1.2

# Score multipliers for attributed modifications.
OVERLAY_PENALTY = 0.25
COLOR_PENALTY = 0.5


@dataclass(frozen=True)
class ImageQuality:
    """Per-file measurements. ``version`` lets a cache spot stale rows."""

    width: int
    height: int
    bandwidth: float
    codec_quality: int | None  # 0-100; None when the format does not say
    format: str
    version: int = QUALITY_VERSION

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def effective_pixels(self) -> float:
        """Pixels that carry picture: what a *sharp* image this size would need."""
        return self.pixel_count * self.bandwidth ** 2

    @property
    def codec_factor(self) -> float:
        if self.format in _LOSSLESS_FORMATS:
            return 1.0
        if self.codec_quality is None:
            return _UNKNOWN_CODEC_FACTOR
        return 0.75 + 0.25 * self.codec_quality / 100.0

    def detail_label(self) -> str:
        if self.bandwidth >= 0.84:
            return "full detail"
        if self.bandwidth >= 0.6:
            return "slightly soft"
        if self.bandwidth >= 0.4:
            return "soft - likely upscaled ~2x or blurred"
        return "very soft - likely upscaled or heavily blurred"

    def codec_label(self) -> str:
        if self.format in _LOSSLESS_FORMATS:
            return f"{self.format} (lossless)"
        if self.codec_quality is not None:
            return f"{self.format} q≈{self.codec_quality}"
        return self.format or "unknown format"

    def summary(self) -> str:
        return " · ".join(
            (
                f"{self.width}×{self.height}",
                f"{self.effective_pixels / 1e6:.2f} MP effective",
                self.codec_label(),
                self.detail_label(),
            )
        )


@dataclass(frozen=True)
class Modification:
    """One detected difference between a copy and the rest of its group."""

    kind: str  # 'overlay' | 'color' | 'framing'
    label: str
    detail: str
    penalty: float = 1.0  # score multiplier; 1.0 when not attributed

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "penalised": self.penalty < 1.0,
        }


@dataclass
class ComparisonImage:
    """A copy decoded small enough to compare against its group."""

    path: str
    rgb: NDArray[np.float32]  # (h, w, 3)
    gray: NDArray[np.float32]  # (h, w) luminance, 0-255

    @property
    def aspect(self) -> float:
        h, w = self.gray.shape
        return w / h if h else 0.0


# --- Per-file measurement -----------------------------------------------------


def estimate_jpeg_quality(image: Image.Image) -> int | None:
    """Estimate the IJG quality a JPEG was saved with, or None for other formats.

    The encoder scales the standard table by (200-2q)/100 above q=50 and 50/q
    below it; the median ratio against the standard table recovers the scale.
    Custom tables (some cameras, Photoshop) give an approximation, hence "q≈".
    """
    tables = getattr(image, "quantization", None)
    if not tables:
        return None
    luma = tables.get(0)
    if not luma or len(luma) != len(_STD_LUMA_QUANT):
        return None
    ratios = np.asarray(luma, dtype=np.float64) / np.asarray(_STD_LUMA_QUANT, dtype=np.float64)
    scale = float(np.median(ratios))
    if scale <= 0:
        return None
    quality = 100.0 - 50.0 * scale if scale <= 1.0 else 50.0 / scale
    return int(round(min(100.0, max(1.0, quality))))


def _tiles(gray: NDArray[np.float32]) -> list[NDArray[np.float32]]:
    h, w = gray.shape
    if h <= _TILE and w <= _TILE:
        return [gray]
    th, tw = min(_TILE, h), min(_TILE, w)
    tiles = []
    for fy in (0.25, 0.75):
        for fx in (0.25, 0.75):
            y0 = min(max(int(fy * h) - th // 2, 0), h - th)
            x0 = min(max(int(fx * w) - tw // 2, 0), w - tw)
            tiles.append(gray[y0 : y0 + th, x0 : x0 + tw])
    return tiles


def spectral_bandwidth(gray: NDArray[np.float32]) -> float:
    """Fraction of Nyquist (1/8 .. 1.0) below which the image holds real detail.

    Returns 1.0 for images too small to judge.
    """
    h, w = gray.shape
    if min(h, w) < _MIN_JUDGEABLE_SIDE:
        return 1.0

    tiles = _tiles(gray)
    th, tw = tiles[0].shape
    window = np.outer(np.hanning(th), np.hanning(tw)).astype(np.float32)
    # Frequencies as a fraction of Nyquist on each axis, so a non-square tile
    # still maps its corners to r > 1 and its axes to r = 1.
    fy = np.fft.fftfreq(th)[:, None] * 2.0
    fx = np.fft.rfftfreq(tw)[None, :] * 2.0
    radius = np.sqrt(fx ** 2 + fy ** 2)
    valid = (radius >= 2.0 ** (-_RING_COUNT / _RINGS_PER_OCTAVE)) & (radius < 1.0)
    ring_index = (
        np.floor(np.log2(radius[valid]) * _RINGS_PER_OCTAVE).astype(np.int64) + _RING_COUNT
    )

    energy = np.zeros(_RING_COUNT, dtype=np.float64)
    for tile in tiles:
        centred = (tile - float(tile.mean())) * window
        power = np.abs(np.fft.rfft2(centred)) ** 2
        energy += np.bincount(ring_index, weights=power[valid], minlength=_RING_COUNT)[:_RING_COUNT]

    reference = float(np.median(energy[:_REFERENCE_RINGS]))
    if reference <= 0.0:
        return 1.0
    relative = energy / reference
    for index in range(_FIRST_JUDGED_RING, _RING_COUNT):
        if relative[index] < BANDWIDTH_CUTOFF:
            return float(2.0 ** ((index - _RING_COUNT) / _RINGS_PER_OCTAVE))
    return 1.0


_ORIENTATION_TAG = 0x0112


def _oriented(image: Image.Image) -> Image.Image:
    """Apply the EXIF orientation only when there is one to apply.

    ImageOps.exif_transpose copies the raster even when no rotation is needed,
    which on a 20-megapixel photo is a copy for nothing.
    """
    try:
        orientation = image.getexif().get(_ORIENTATION_TAG, 1)
    except Exception:  # pragma: no cover - exotic EXIF blobs
        orientation = 1
    if orientation == 1:
        return image
    return ImageOps.exif_transpose(image) or image


def decode_for_analysis(
    path: str | Path, measure: bool, size: int = COMPARISON_SIZE
) -> tuple[ImageQuality | None, ComparisonImage | None]:
    """Open *path* once and return (quality, comparison image).

    With *measure* the full raster is decoded, measured, and then reduced to
    the comparison copy - one decode serves both. Without it the quality is
    already known, so the JPEG decoder's draft mode reads only what the
    comparison needs. Draft mode shifts fine detail - fatal for hashing,
    irrelevant for a colour and region comparison at this size.

    An unreadable file measures as unknown (0x0, bandwidth 1.0) and yields no
    comparison image.
    """
    quality: ImageQuality | None = None
    try:
        with Image.open(path) as image:
            fmt = image.format or ""
            if measure:
                codec_quality = estimate_jpeg_quality(image) if fmt == "JPEG" else None
                width, height = image.size
                gray = np.asarray(image.convert("L"), dtype=np.float32)
                quality = ImageQuality(width, height, spectral_bandwidth(gray), codec_quality, fmt)
                del gray
            else:
                image.draft("RGB", (size * 2, size * 2))
            oriented = _oriented(image)
            rgb_image = oriented.convert("RGB")
            rgb_image.thumbnail((size, size), Image.Resampling.LANCZOS)
            rgb = np.asarray(rgb_image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("Unable to decode %s: %s", path, exc)
        return (ImageQuality(0, 0, 1.0, None, "") if measure else None), None
    return quality, ComparisonImage(str(path), rgb, _luminance(rgb))


def measure_image_quality(path: str | Path) -> ImageQuality:
    """Decode *path* once and measure it. Unreadable files measure as unknown."""
    quality, _ = decode_for_analysis(path, measure=True)
    assert quality is not None
    return quality


# --- Group-relative modification detection ---------------------------------------


def load_comparison_image(path: str | Path, size: int = COMPARISON_SIZE) -> ComparisonImage | None:
    """Decode *path* small, orientation applied, for comparison with its group."""
    return decode_for_analysis(path, measure=False, size=size)[1]


def _luminance(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    result: NDArray[np.float32] = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    return result


def _normalise(gray: NDArray[np.float32], mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Zero-centre and scale by median/MAD over the pixels in *mask*."""
    sample = gray[mask] if mask.any() else gray.reshape(-1)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = mad * 1.4826 if mad > 0 else float(sample.std()) or 1.0
    return ((gray - median) / scale).astype(np.float32)


def _box_filter(a: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    """Mean over a k x k window, edge-padded, via summed-area table."""
    pad = k // 2
    padded = np.pad(a, pad, mode="edge").astype(np.float64)
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = padded.cumsum(0).cumsum(1)
    h, w = a.shape
    total = (
        integral[k : k + h, k : k + w]
        - integral[:h, k : k + w]
        - integral[k : k + h, :w]
        + integral[:h, :w]
    )
    return (total / (k * k)).astype(np.float32)


def _edge_energy(gray: NDArray[np.float32]) -> NDArray[np.float32]:
    """Absolute 4-neighbour Laplacian, zero on the border."""
    out = np.zeros_like(gray)
    out[1:-1, 1:-1] = np.abs(
        4.0 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return out


def _resize_to(image: ComparisonImage, shape: tuple[int, int]) -> ComparisonImage:
    if image.gray.shape == shape:
        return image
    h, w = shape
    rgb = Image.fromarray(np.clip(image.rgb, 0, 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.LANCZOS
    )
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    return ComparisonImage(image.path, rgb_arr, _luminance(rgb_arr))


def _region_label(y0: int, y1: int, x0: int, x1: int, h: int, w: int) -> str:
    cy = (y0 + y1) / 2.0 / h
    cx = (x0 + x1) / 2.0 / w
    vertical = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    if vertical == "middle" and horizontal == "centre":
        return "centre"
    if (y1 - y0) > 0.8 * h:
        return f"{horizontal} edge"
    if (x1 - x0) > 0.8 * w:
        return f"{vertical} edge"
    return f"{vertical}-{horizontal}"


@dataclass
class _RegionDiff:
    y0: int
    y1: int
    x0: int
    x1: int
    fraction: float


def _changed_region(a: NDArray[np.float32], b: NDArray[np.float32]) -> _RegionDiff | None:
    """Bounding box of where *a* and *b* differ, if the change is localised.

    Tone and contrast are equalised before differencing, and equalised again
    over only the pixels that agreed the first time: a large banner or caption
    drags a whole-image median far enough that the *rest* of the frame reads as
    faintly changed, and the region is no longer localised.
    """
    agree = np.ones(a.shape, dtype=bool)
    hot = ~agree
    for _ in range(2):
        diff = _box_filter(np.abs(_normalise(a, agree) - _normalise(b, agree)), _DIFF_BOX)
        hot = diff > _DIFF_THRESHOLD
        if hot.all() or not hot.any():
            break
        agree = ~hot
    fraction = float(hot.mean())
    if not (_OVERLAY_MIN_FRACTION < fraction < _OVERLAY_MAX_FRACTION):
        return None
    total = float(hot.sum())
    # Bound the box by the rows and columns that are *dense* with change, not
    # by the outermost changed pixel: a re-encode or a shifted normalisation
    # sprinkles change thinly across the whole frame, and taking the extremes
    # would stretch every box to the full image.
    row_profile = hot.sum(axis=1)
    col_profile = hot.sum(axis=0)
    dense_rows = np.flatnonzero(row_profile >= _PROFILE_FRACTION * row_profile.max())
    dense_cols = np.flatnonzero(col_profile >= _PROFILE_FRACTION * col_profile.max())
    y0, y1 = int(dense_rows[0]), int(dense_rows[-1])
    x0, x1 = int(dense_cols[0]), int(dense_cols[-1])
    box = hot[y0 : y1 + 1, x0 : x1 + 1]
    if box.size / hot.size > _OVERLAY_MAX_BOX_FRACTION:
        return None
    if float(box.sum()) / total < _OVERLAY_MIN_CONTAINED:
        return None
    if float(box.mean()) < _OVERLAY_MIN_DENSITY:
        return None
    return _RegionDiff(y0, y1, x0, x1, fraction)


def _relative_edge_energy(gray: NDArray[np.float32], region: _RegionDiff) -> float:
    edges = _edge_energy(gray)
    mask = np.zeros(gray.shape, dtype=bool)
    mask[region.y0 : region.y1 + 1, region.x0 : region.x1 + 1] = True
    inside = float(edges[mask].mean()) if mask.any() else 0.0
    outside = float(edges[~mask].mean()) if (~mask).any() else 0.0
    return inside / (outside + 1e-6)


def _outside(shape: tuple[int, int], region: _RegionDiff | None) -> NDArray[np.bool_]:
    mask = np.ones(shape, dtype=bool)
    if region is not None:
        mask[region.y0 : region.y1 + 1, region.x0 : region.x1 + 1] = False
    return mask


def _boundary_contrast(gray: NDArray[np.float32], region: _RegionDiff) -> float:
    """Step across the region's border, relative to a typical pixel step.

    A caption box or pasted logo has a hard rectangular edge where it meets the
    picture; the same rectangle in the clean copy is just more picture. The
    box filter that found the region pads it by a couple of pixels, so the
    strongest step within a few pixels inside the border is taken.
    """
    h, w = gray.shape
    y0, y1, x0, x1 = region.y0, region.y1, region.x0, region.x1
    steps = []
    for inset in range(_DIFF_BOX // 2 + 2):
        ty, by, lx, rx = y0 + inset, y1 - inset, x0 + inset, x1 - inset
        if ty >= by or lx >= rx:
            break
        edges = []
        if ty > 0:
            edges.append(np.abs(gray[ty, lx : rx + 1] - gray[ty - 1, lx : rx + 1]))
        if by < h - 1:
            edges.append(np.abs(gray[by, lx : rx + 1] - gray[by + 1, lx : rx + 1]))
        if lx > 0:
            edges.append(np.abs(gray[ty : by + 1, lx] - gray[ty : by + 1, lx - 1]))
        if rx < w - 1:
            edges.append(np.abs(gray[ty : by + 1, rx] - gray[ty : by + 1, rx + 1]))
        if edges:
            steps.append(float(np.concatenate(edges).mean()))
    if not steps:
        return 0.0
    typical = (
        float(np.abs(np.diff(gray, axis=0)).mean()) + float(np.abs(np.diff(gray, axis=1)).mean())
    ) / 2.0
    return max(steps) / (typical + 1e-6)


def _attribute_overlay(own: float, theirs: float, own_border: float, theirs_border: float) -> str:
    """Decide who carries the overlay in the changed region.

    ``own``/``theirs`` are edge energy inside the region relative to each copy's
    own surroundings; ``*_border`` the step across the region's border relative
    to a typical pixel step. Three shapes of overlay, in order:

    * text or a logo is *denser* in edges than the picture around it, so the
      copy that is both denser than its surroundings and denser than the other
      copy carries it;
    * a solid bar, box or blur is far *flatter* than its surroundings;
    * a caption box or pasted panel has a hard rectangular border that the
      same rectangle of plain picture does not.

    Natural texture sits near 1.0 on every measure, so a textured original is
    not mistaken for text just because a banner on the other copy erased the
    texture there.
    """
    if own >= _TEXT_EDGE_RATIO and own >= theirs * _OVERLAY_EDGE_RATIO:
        return "text"
    if theirs >= _TEXT_EDGE_RATIO and theirs >= own * _OVERLAY_EDGE_RATIO:
        return "clean"
    if own <= _SOLID_EDGE_RATIO < theirs:
        return "solid"
    if theirs <= _SOLID_EDGE_RATIO < own:
        return "clean"
    if own_border >= _BORDER_STEP_RATIO and own_border >= theirs_border * _BORDER_MARGIN:
        return "boxed"
    if theirs_border >= _BORDER_STEP_RATIO and theirs_border >= own_border * _BORDER_MARGIN:
        return "clean"
    return "undecided"


def _colour_stats(
    rgb: NDArray[np.float32], mask: NDArray[np.bool_] | None = None
) -> tuple[float, float, float, float]:
    """(warmth, luminance, saturation, contrast) on the 0-255 scale."""
    flat = rgb[mask] if mask is not None and mask.any() else rgb.reshape(-1, 3)
    means = flat.mean(axis=0)
    luminance = flat @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    saturation = float((flat.max(axis=1) - flat.min(axis=1)).mean())
    return (
        float(means[0] - means[2]),
        float(luminance.mean()),
        saturation,
        float(luminance.std()),
    )


def _colour_differences(
    stats: tuple[float, float, float, float], reference: tuple[float, float, float, float]
) -> list[str]:
    warmth, luminance, saturation, contrast = stats
    ref_warmth, ref_luminance, ref_saturation, ref_contrast = reference
    found = []
    if abs(warmth - ref_warmth) >= _WARMTH_THRESHOLD:
        found.append("warmer" if warmth > ref_warmth else "cooler")
    if abs(luminance - ref_luminance) >= _LUMINANCE_THRESHOLD:
        found.append("brighter" if luminance > ref_luminance else "darker")
    if abs(saturation - ref_saturation) >= _SATURATION_THRESHOLD:
        found.append("more saturated" if saturation > ref_saturation else "less saturated")
    if ref_contrast > 0 and (
        contrast / ref_contrast >= _CONTRAST_RATIO or ref_contrast / contrast >= _CONTRAST_RATIO
    ):
        found.append("higher contrast" if contrast > ref_contrast else "lower contrast")
    return found


def detect_modifications(images: Sequence[ComparisonImage]) -> dict[str, list[Modification]]:
    """Compare each copy against the rest of its group.

    With two copies nothing says which is the original, so colour differences
    are reported on both, relative to each other, and carry no penalty. With
    three or more the odd one out is the modified copy. An overlay is
    attributed by edge energy regardless of group size: text and logos add
    edges wherever they land.
    """
    result: dict[str, list[Modification]] = {img.path: [] for img in images}
    if len(images) < 2:
        return result

    # Aspect: majority wins; with two, both differ from each other
    aspects = sorted(img.aspect for img in images)
    reference_aspect = aspects[len(aspects) // 2]
    aligned: list[ComparisonImage] = []
    for img in images:
        if reference_aspect and abs(img.aspect / reference_aspect - 1.0) > ASPECT_TOLERANCE:
            others_word = "its duplicate" if len(images) == 2 else "the other copies"
            result[img.path].append(
                Modification(
                    "framing",
                    "Different framing",
                    f"Aspect ratio {img.aspect:.3f} differs from {others_word} - cropped, padded or rotated",
                )
            )
        else:
            aligned.append(img)
    if len(images) == 2 and len(aligned) < 2:
        # Both differ from each other; nothing else can be aligned
        for img in images:
            if not result[img.path]:
                result[img.path].append(
                    Modification(
                        "framing",
                        "Different framing",
                        f"Aspect ratio {img.aspect:.3f} differs from its duplicate - cropped, padded or rotated",
                    )
                )
        return result
    if len(aligned) < 2:
        return result

    # Bring the aligned copies to one size: the smallest, so nobody is upscaled
    target = min((img.gray.shape for img in aligned), key=lambda s: s[0] * s[1])
    resized = [_resize_to(img, target) for img in aligned]
    pair = len(resized) == 2
    gray_stack = np.stack([img.gray for img in resized])
    # The median over every copy, self included: with three copies and one
    # filtered, the median is the original. A median over the *others* would be
    # the mean of two values and flag all three.
    median_gray = np.median(gray_stack, axis=0).astype(np.float32) if not pair else None

    regions: dict[str, _RegionDiff | None] = {}
    for index, img in enumerate(resized):
        ref_gray = resized[1 - index].gray if pair else median_gray
        assert ref_gray is not None
        regions[img.path] = _changed_region(img.gray, ref_gray)

    # Colour is judged outside any changed region, so a white banner is
    # reported as the overlay it is rather than as a brightness shift.
    colour_stats = {
        img.path: _colour_stats(img.rgb, _outside(img.gray.shape, regions[img.path]))
        for img in resized
    }
    if not pair:
        medians = [
            float(np.median([colour_stats[img.path][i] for img in resized])) for i in range(4)
        ]
        median_colour = (medians[0], medians[1], medians[2], medians[3])

    for index, img in enumerate(resized):
        if pair:
            other = resized[1 - index]
            ref_gray = other.gray
            ref_colour = colour_stats[other.path]
            others_label = "its duplicate"
        else:
            assert median_gray is not None
            ref_gray = median_gray
            ref_colour = median_colour
            others_label = f"the other {len(resized) - 1} copies"

        differences = _colour_differences(colour_stats[img.path], ref_colour)
        if differences:
            described = ", ".join(differences)
            if pair:
                result[img.path].append(
                    Modification(
                        "color",
                        "Colour differs",
                        f"{described.capitalize()} than {others_label} - one of the two "
                        "may have a filter applied",
                    )
                )
            else:
                result[img.path].append(
                    Modification(
                        "color",
                        "Filter likely applied",
                        f"{described.capitalize()} than {others_label}",
                        COLOR_PENALTY,
                    )
                )

        region = regions[img.path]
        if region is None:
            continue
        h, w = img.gray.shape
        where = _region_label(region.y0, region.y1, region.x0, region.x1, h, w)
        verdict = _attribute_overlay(
            _relative_edge_energy(img.gray, region),
            _relative_edge_energy(ref_gray, region),
            _boundary_contrast(img.gray, region),
            _boundary_contrast(ref_gray, region),
        )
        if verdict == "text":
            result[img.path].append(
                Modification(
                    "overlay",
                    f"Overlay detected ({where})",
                    f"Text, logo or watermark burned in at the {where}: "
                    f"{region.fraction * 100:.1f}% of the frame differs from {others_label} "
                    "and carries extra edges",
                    OVERLAY_PENALTY,
                )
            )
        elif verdict == "solid":
            result[img.path].append(
                Modification(
                    "overlay",
                    f"Overlay detected ({where})",
                    f"Solid bar, box or blur covering the {where}: "
                    f"{region.fraction * 100:.1f}% of the frame differs from {others_label} "
                    "and the picture there has been flattened",
                    OVERLAY_PENALTY,
                )
            )
        elif verdict == "boxed":
            result[img.path].append(
                Modification(
                    "overlay",
                    f"Overlay detected ({where})",
                    f"Caption box or pasted panel at the {where}: "
                    f"{region.fraction * 100:.1f}% of the frame differs from {others_label} "
                    "behind a hard rectangular border",
                    OVERLAY_PENALTY,
                )
            )
        elif verdict == "clean":
            if not pair:
                result[img.path].append(
                    Modification(
                        "overlay",
                        f"Cleaner here ({where})",
                        f"Differs from {others_label} at the {where}, which is where they were altered",
                    )
                )
            # In a pair the other copy is flagged on its own turn
        else:
            result[img.path].append(
                Modification(
                    "overlay",
                    f"Differs at the {where}",
                    f"{region.fraction * 100:.1f}% of the frame differs from {others_label} "
                    f"at the {where}; neither side is clearly the overlay",
                )
            )
    return result


# --- Ranking ---------------------------------------------------------------------


@dataclass
class ImageCandidate:
    path: str
    quality: ImageQuality
    size: int
    modifications: list[Modification] = field(default_factory=list)

    @property
    def penalty(self) -> float:
        return float(math.prod(m.penalty for m in self.modifications))

    @property
    def score(self) -> float:
        return float(
            self.quality.pixel_count
            * self.quality.bandwidth ** DETAIL_EXPONENT
            * self.quality.codec_factor
            * self.penalty
        )


def rank_images(candidates: Iterable[ImageCandidate]) -> list[ImageCandidate]:
    """Best first. Ties go to the larger file, then the lexically first path."""
    return sorted(candidates, key=lambda c: (-c.score, -c.size, c.path))


def explain_choice(best: ImageCandidate, others: Sequence[ImageCandidate]) -> str:
    """One line on why *best* won, for the UI."""
    reasons = [best.quality.summary()]
    if others:
        runner = max(others, key=lambda c: c.score)
        if runner.quality.effective_pixels < best.quality.effective_pixels * 0.9:
            reasons.append("most picture detail")
        elif best.quality.codec_factor > runner.quality.codec_factor:
            reasons.append("least compressed")
        elif best.penalty > runner.penalty:
            reasons.append("no detected modifications")
        elif best.size > runner.size:
            reasons.append("largest file at equal quality")
    return "; ".join(reasons)
