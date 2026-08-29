"""Quality measurement, modification detection and best-file ranking.

The images are synthetic but textured across every spatial frequency, the way a
photograph is: a smooth base, pink noise and a scatter of hard-edged shapes. A
plain gradient would have no fine detail for the bandwidth estimate to find.
"""

import io
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from deduper.utils.quality import (
    ImageCandidate,
    detect_modifications,
    estimate_jpeg_quality,
    load_comparison_image,
    measure_image_quality,
    rank_images,
    spectral_bandwidth,
)


def textured_image(width: int = 640, height: int = 480, seed: int = 1) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = 128 + 60 * np.sin(xx / 120) + 50 * np.cos(yy / 90)
    spectrum = np.fft.fft2(rng.normal(0, 1, (height, width)))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    radius[0, 0] = 1
    pink = np.real(np.fft.ifft2(spectrum / radius))
    pink = pink / pink.std() * 35
    image = Image.fromarray(np.clip(base + pink, 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(30):
        x0, y0 = int(rng.integers(0, width - 80)), int(rng.integers(0, height - 80))
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        draw.rectangle(
            [x0, y0, x0 + int(rng.integers(10, 80)), y0 + int(rng.integers(10, 80))],
            outline=colour, width=2,
        )
        draw.ellipse([x0, y0, x0 + int(rng.integers(10, 60)), y0 + int(rng.integers(10, 60))], fill=colour)
    return image


def as_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def watermark(image: Image.Image, text: str = "SAMPLE © SITE") -> Image.Image:
    copy = image.copy()
    draw = ImageDraw.Draw(copy)
    width, height = copy.size
    # Big, blocky, high-contrast: what a burned-in caption looks like
    draw.rectangle([width - 190, height - 46, width - 20, height - 16], fill=(0, 0, 0))
    draw.text((width - 180, height - 40), text, fill=(255, 255, 255))
    for offset in range(3):
        draw.text((width - 180 + offset, height - 40), text, fill=(255, 255, 255))
    return copy


def warm(image: Image.Image) -> Image.Image:
    r, g, b = image.split()
    return Image.merge(
        "RGB",
        (
            ImageEnhance.Brightness(r).enhance(1.12),
            g,
            ImageEnhance.Brightness(b).enhance(0.88),
        ),
    )


@pytest.fixture(scope="module")
def original() -> Image.Image:
    return textured_image()


class TestBandwidth:
    def test_an_original_is_full_detail(self, original):
        assert spectral_bandwidth(gray(original)) >= 0.84

    def test_a_downscale_is_still_full_detail(self, original):
        small = original.resize((320, 240), Image.Resampling.LANCZOS)
        assert spectral_bandwidth(gray(small)) >= 0.84

    def test_a_two_times_upscale_stops_at_half(self, original):
        small = original.resize((320, 240), Image.Resampling.LANCZOS)
        up = small.resize((640, 480), Image.Resampling.BICUBIC)
        assert 0.4 <= spectral_bandwidth(gray(up)) <= 0.6

    def test_a_blur_loses_detail(self, original):
        assert spectral_bandwidth(gray(original.filter(ImageFilter.GaussianBlur(2)))) < 0.6

    def test_a_tiny_image_is_not_judged(self):
        assert spectral_bandwidth(np.zeros((64, 64), dtype=np.float32)) == 1.0


class TestJpegQuality:
    @pytest.mark.parametrize("quality", [95, 75, 40])
    def test_recovers_the_saved_quality(self, original, quality):
        assert abs(estimate_jpeg_quality(as_jpeg(original, quality)) - quality) <= 3

    def test_png_has_no_jpeg_quality(self, original):
        assert estimate_jpeg_quality(original) is None


class TestMeasure:
    def test_png_is_lossless_full_detail(self, original, tmp_path):
        path = tmp_path / "o.png"
        original.save(path)
        q = measure_image_quality(path)
        assert (q.width, q.height) == (640, 480)
        assert q.format == "PNG"
        assert q.codec_factor == 1.0
        assert q.bandwidth >= 0.84
        assert "lossless" in q.summary()

    def test_upscale_has_fewer_effective_pixels_than_its_source(self, original, tmp_path):
        small = original.resize((320, 240), Image.Resampling.LANCZOS)
        small.save(tmp_path / "small.png")
        small.resize((640, 480), Image.Resampling.BICUBIC).save(tmp_path / "up.png")
        assert (
            measure_image_quality(tmp_path / "up.png").effective_pixels
            <= measure_image_quality(tmp_path / "small.png").effective_pixels * 1.05
        )

    def test_unreadable_file_measures_as_unknown(self, tmp_path):
        path = tmp_path / "junk.png"
        path.write_bytes(b"not an image")
        q = measure_image_quality(path)
        assert q.pixel_count == 0


def candidates(tmp_path, **images) -> dict[str, ImageCandidate]:
    """Save each image, measure it, and compare the set for modifications."""
    paths = {}
    for name, image in images.items():
        path = str(tmp_path / name)
        if name.endswith(".jpg"):
            image.save(path, quality=92)
        else:
            image.save(path)
        paths[name] = path
    comparison = [load_comparison_image(p) for p in paths.values()]
    modifications = detect_modifications([c for c in comparison if c is not None])
    return {
        name: ImageCandidate(
            path, measure_image_quality(path), os.path.getsize(path), modifications[path]
        )
        for name, path in paths.items()
    }


def best_of(cands: dict[str, ImageCandidate]) -> str:
    return os.path.basename(rank_images(cands.values())[0].path)


def kinds(cand: ImageCandidate) -> list[str]:
    return [m.kind for m in cand.modifications]


class TestRanking:
    def test_original_beats_its_own_upscale(self, original, tmp_path):
        small = original.resize((320, 240), Image.Resampling.LANCZOS)
        c = candidates(
            tmp_path,
            **{"small.png": small, "up.png": small.resize((640, 480), Image.Resampling.BICUBIC)},
        )
        assert best_of(c) == "small.png"

    def test_higher_resolution_original_still_wins(self, original, tmp_path):
        c = candidates(
            tmp_path,
            **{"full.png": original, "half.png": original.resize((320, 240), Image.Resampling.LANCZOS)},
        )
        assert best_of(c) == "full.png"

    def test_less_compressed_wins_at_equal_resolution(self, original, tmp_path):
        original.save(tmp_path / "q92.jpg", quality=92)
        original.save(tmp_path / "q35.jpg", quality=35)
        c = {
            name: ImageCandidate(str(tmp_path / name), measure_image_quality(tmp_path / name), 0)
            for name in ("q92.jpg", "q35.jpg")
        }
        assert best_of(c) == "q92.jpg"

    def test_lossless_beats_jpeg_at_equal_resolution(self, original, tmp_path):
        c = candidates(tmp_path, **{"a.png": original, "b.jpg": original})
        assert best_of(c) == "a.png"


class TestOverlayDetection:
    def test_text_overlay_is_attributed_and_penalised(self, original, tmp_path):
        c = candidates(tmp_path, **{"clean.jpg": original, "wm.jpg": watermark(original)})
        assert kinds(c["clean.jpg"]) == []
        (mod,) = c["wm.jpg"].modifications
        assert mod.kind == "overlay"
        assert mod.penalty < 1.0
        assert "bottom-right" in mod.label
        assert best_of(c) == "clean.jpg"

    def test_a_clean_smaller_copy_beats_a_watermarked_larger_one(self, original, tmp_path):
        c = candidates(
            tmp_path,
            **{
                "clean_small.png": original.resize((480, 360), Image.Resampling.LANCZOS),
                "wm_large.png": watermark(original),
            },
        )
        assert best_of(c) == "clean_small.png"

    def test_a_solid_banner_is_attributed_to_the_copy_carrying_it(self, original, tmp_path):
        banner = original.copy()
        ImageDraw.Draw(banner).rectangle([0, 400, 640, 480], fill=(255, 255, 255))
        c = candidates(tmp_path, **{"clean.png": original, "banner.png": banner})
        assert kinds(c["clean.png"]) == []
        (mod,) = c["banner.png"].modifications
        assert mod.kind == "overlay" and mod.penalty < 1.0
        assert "bottom edge" in mod.label

    def test_a_re_encode_is_not_a_modification(self, original, tmp_path):
        c = candidates(tmp_path, **{"a.png": original, "b.jpg": original})
        assert kinds(c["a.png"]) == [] and kinds(c["b.jpg"]) == []

    def test_a_rescale_is_not_a_modification(self, original, tmp_path):
        c = candidates(
            tmp_path,
            **{"a.png": original, "b.png": original.resize((400, 300), Image.Resampling.LANCZOS)},
        )
        assert kinds(c["a.png"]) == [] and kinds(c["b.png"]) == []

    def test_a_slight_reframe_is_not_an_overlay(self, original, tmp_path):
        # Same aspect, shifted content: change everywhere, concentrated nowhere
        shifted = original.crop((8, 6, 632, 474)).resize((640, 480), Image.Resampling.LANCZOS)
        c = candidates(tmp_path, **{"a.png": original, "b.png": shifted})
        assert "overlay" not in kinds(c["a.png"]) + kinds(c["b.png"])


class TestColourDetection:
    def test_a_pair_reports_both_sides_without_penalty(self, original, tmp_path):
        c = candidates(tmp_path, **{"a.png": original, "warm.png": warm(original)})
        for cand in c.values():
            (mod,) = cand.modifications
            assert mod.kind == "color" and mod.penalty == 1.0
        assert "warmer" in c["warm.png"].modifications[0].detail.lower()
        assert "cooler" in c["a.png"].modifications[0].detail.lower()

    def test_the_odd_one_out_of_three_is_penalised(self, original, tmp_path):
        c = candidates(
            tmp_path, **{"a.png": original, "b.jpg": original, "warm.png": warm(original)}
        )
        assert kinds(c["a.png"]) == [] and kinds(c["b.jpg"]) == []
        (mod,) = c["warm.png"].modifications
        assert mod.kind == "color" and mod.penalty < 1.0
        assert best_of(c) == "a.png"

    def test_brightness_is_reported(self, original, tmp_path):
        c = candidates(
            tmp_path,
            **{"a.png": original, "bright.png": ImageEnhance.Brightness(original).enhance(1.25)},
        )
        assert "brighter" in c["bright.png"].modifications[0].detail.lower()


class TestFraming:
    def test_a_crop_to_another_aspect_is_flagged_on_both(self, original, tmp_path):
        c = candidates(tmp_path, **{"a.png": original, "square.png": original.crop((80, 0, 560, 480))})
        assert kinds(c["a.png"]) == ["framing"]
        assert kinds(c["square.png"]) == ["framing"]

    def test_the_majority_aspect_is_the_reference(self, original, tmp_path):
        c = candidates(
            tmp_path,
            **{"a.png": original, "b.jpg": original, "square.png": original.crop((80, 0, 560, 480))},
        )
        assert kinds(c["a.png"]) == [] and kinds(c["b.jpg"]) == []
        assert kinds(c["square.png"]) == ["framing"]
