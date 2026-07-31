"""Tests for media probing and hashing.

Hashing upstream is dual (pHash + dHash) via ``MultiHash``; distance is the max of
the two, exposed through ``__sub__``, so distance assertions read the same as they
would for a single hash.
"""

import os

import pytest
from PIL import Image

from deduper.utils.media import (
    MultiHash,
    Resolution,
    _normalize_extensions,
    _resolve_media_resolution_cached,
    get_image_hash,
    get_video_duration,
    resolve_media_resolution,
)

from .conftest import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, requires_ffmpeg


class TestNormalizeExtensions:
    def test_adds_leading_dot_and_lowercases(self):
        assert _normalize_extensions(["JPG", ".PNG", "gif"]) == (".jpg", ".png", ".gif")

    def test_drops_empty_values_and_duplicates(self):
        assert _normalize_extensions(["", ".jpg", "jpg"]) == (".jpg",)

    def test_accepts_a_set(self):
        assert set(_normalize_extensions({".mp4", ".mov"})) == {".mp4", ".mov"}


class TestResolution:
    def test_unknown_resolution_reports_label(self):
        assert Resolution().label() == "Unknown"
        assert Resolution().is_known() is False

    def test_pixel_count(self):
        assert Resolution(1920, 1080).pixel_count() == 1920 * 1080
        assert Resolution(1920, 1080).label() == "1920x1080"


class TestMultiHash:
    def test_distance_is_the_worse_of_the_two_hashes(self, images_only_dir):
        """__sub__ must not be more permissive than either component alone.

        Grouping keys off this value, so a distance below both components would
        silently widen what counts as a duplicate.
        """
        a = get_image_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        b = get_image_hash(os.path.join(images_only_dir, "img_b.png"), VIDEO_EXTENSIONS)
        assert int(a - b) == max(a.phash_distance(b), a.dhash_distance(b))

    def test_serialises_and_round_trips(self, images_only_dir):
        original = get_image_hash(
            os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS
        )
        restored = MultiHash.from_str(str(original))
        assert int(original - restored) == 0

    def test_string_form_carries_both_hashes(self, images_only_dir):
        hash_obj = get_image_hash(
            os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS
        )
        assert "|" in str(hash_obj)
        assert str(hash_obj).split("|") == [str(hash_obj.phash), str(hash_obj.dhash)]


class TestImageHashing:
    def test_identical_files_hash_identically(self, images_only_dir):
        a = get_image_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        b = get_image_hash(os.path.join(images_only_dir, "img_a_copy.png"), VIDEO_EXTENSIONS)
        assert a is not None and b is not None
        assert str(a) == str(b)

    def test_downscaled_copy_hashes_within_threshold(self, images_only_dir):
        a = get_image_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        small = get_image_hash(
            os.path.join(images_only_dir, "img_a_small.png"), VIDEO_EXTENSIONS
        )
        assert a is not None and small is not None
        assert int(a - small) < 5

    def test_unrelated_image_hashes_outside_threshold(self, images_only_dir):
        a = get_image_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        b = get_image_hash(os.path.join(images_only_dir, "img_b.png"), VIDEO_EXTENSIONS)
        assert a is not None and b is not None
        assert int(a - b) >= 5

    def test_corrupt_image_returns_none(self, tmp_path):
        broken = tmp_path / "broken.png"
        broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 8)
        assert get_image_hash(str(broken), VIDEO_EXTENSIONS) is None

    def test_truncated_image_returns_none(self, tmp_path, images_only_dir):
        source = os.path.join(images_only_dir, "img_a.png")
        data = open(source, "rb").read()
        truncated = tmp_path / "truncated.png"
        truncated.write_bytes(data[: len(data) // 3])
        assert get_image_hash(str(truncated), VIDEO_EXTENSIONS) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert get_image_hash(str(tmp_path / "nope.png"), VIDEO_EXTENSIONS) is None

    def test_hash_components_match_plain_imagehash(self, images_only_dir):
        """Pin both components to imagehash's own output for a full decode.

        If this drifts, hashes already stored in .deduper stop being comparable with
        freshly computed ones and CACHE_VERSION must be bumped.
        """
        import imagehash

        path = os.path.join(images_only_dir, "img_a.png")
        with Image.open(path) as image:
            expected_phash = imagehash.phash(image)
        with Image.open(path) as image:
            expected_dhash = imagehash.dhash(image)

        actual = get_image_hash(path, VIDEO_EXTENSIONS)
        assert str(actual.phash) == str(expected_phash)
        assert str(actual.dhash) == str(expected_dhash)


class TestImageResolution:
    def test_reads_image_dimensions(self, images_only_dir):
        resolution = resolve_media_resolution(
            os.path.join(images_only_dir, "img_a.png"), IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
        )
        assert (resolution.width, resolution.height) == (240, 240)

    def test_missing_file_yields_unknown(self, tmp_path):
        resolution = resolve_media_resolution(
            str(tmp_path / "absent.png"), IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
        )
        assert resolution == Resolution()

    def test_unknown_extension_yields_unknown(self, tmp_path):
        other = tmp_path / "notes.txt"
        other.write_text("hello")
        resolution = resolve_media_resolution(
            str(other), IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
        )
        assert resolution == Resolution()

    @pytest.mark.skip(reason="port: stat-keyed memo so in-place edits invalidate")
    def test_editing_a_file_in_place_invalidates_the_memoised_resolution(
        self, images_only_dir
    ):
        """The memo is keyed on path alone, so a file replaced in place keeps
        reporting its previous dimensions for the rest of the process."""
        import time

        from .conftest import gradient_image

        path = os.path.join(images_only_dir, "img_a.png")
        assert resolve_media_resolution(
            path, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
        ) == Resolution(240, 240)

        time.sleep(0.01)
        gradient_image(100, 100, 11).save(path)

        assert resolve_media_resolution(
            path, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
        ) == Resolution(100, 100)

    def test_unchanged_file_is_still_memoised(self, images_only_dir):
        path = os.path.join(images_only_dir, "img_a.png")
        before = _resolve_media_resolution_cached.cache_info()
        for _ in range(5):
            resolve_media_resolution(path, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        after = _resolve_media_resolution_cached.cache_info()
        assert after.hits - before.hits == 4


@requires_ffmpeg
class TestVideoProbing:
    def test_reads_video_resolution(self, media_dir):
        resolution = resolve_media_resolution(
            os.path.join(media_dir, "vid_hi.mp4"), IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
        )
        assert (resolution.width, resolution.height) == (640, 480)

    def test_reads_video_duration(self, media_dir):
        duration = get_video_duration(os.path.join(media_dir, "vid_hi.mp4"))
        assert duration == pytest.approx(3.0, abs=0.5)

    @pytest.mark.skip(reason="port: fix #3 single ffprobe for resolution + duration")
    def test_resolution_and_duration_share_one_probe(self, media_dir, count_ffprobe):
        """Upstream probes twice: once for dimensions, once for duration."""
        path = os.path.join(media_dir, "vid_hi.mp4")
        resolve_media_resolution(path, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        get_video_duration(path)
        assert len(count_ffprobe) == 1

    def test_missing_video_probes_nothing(self, tmp_path, count_ffprobe):
        assert get_video_duration(str(tmp_path / "absent.mp4")) == 0.0
        assert count_ffprobe == []

    def test_unprobeable_video_degrades_gracefully(self, tmp_path):
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"not a video at all")
        assert get_video_duration(str(junk)) == 0.0
        assert resolve_media_resolution(
            str(junk), IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
        ) == Resolution()

    def test_video_hash_uses_thumbnail(self, media_dir):
        hi = get_image_hash(os.path.join(media_dir, "vid_hi.mp4"), VIDEO_EXTENSIONS)
        lo = get_image_hash(os.path.join(media_dir, "vid_lo.mp4"), VIDEO_EXTENSIONS)
        assert hi is not None and lo is not None
        assert int(hi - lo) < 5
