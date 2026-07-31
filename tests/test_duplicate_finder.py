"""Tests for the duplicate detection pipeline.

These exercise the destructive path: exact duplicates really are deleted and
replaced with symlinks, so every test scans a throwaway copy of the fixtures.
"""

import os

import pytest

from deduper.services.duplicate_finder import DuplicateFinder
from deduper.utils.hash_cache import HashCache

from .conftest import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    group_basenames,
    gradient_image,
    requires_ffmpeg,
)


def finder() -> DuplicateFinder:
    return DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)


def scan(folder: str):
    return finder().find_duplicates(folder)


def find_group(groups: list[dict], member: str) -> dict | None:
    for group in groups:
        if member in group_basenames(group):
            return group
    return None


class TestColdScan:
    def test_groups_similar_images_together(self, images_only_dir):
        images, _ = scan(images_only_dir)
        group = find_group(images, "img_a.png")
        assert group is not None
        assert "img_a_small.png" in group_basenames(group)

    def test_unrelated_image_is_not_grouped(self, images_only_dir):
        images, _ = scan(images_only_dir)
        assert find_group(images, "img_b.png") is None

    def test_exact_duplicate_becomes_a_symlink(self, images_only_dir):
        scan(images_only_dir)
        assert os.path.islink(os.path.join(images_only_dir, "img_a_copy.png"))

    def test_symlink_resolves_to_the_kept_file(self, images_only_dir):
        scan(images_only_dir)
        link = os.path.join(images_only_dir, "img_a_copy.png")
        assert os.path.realpath(link) == os.path.join(images_only_dir, "img_a.png")

    def test_similar_file_is_left_as_a_real_file(self, images_only_dir):
        """Only exact matches may be removed automatically."""
        scan(images_only_dir)
        path = os.path.join(images_only_dir, "img_a_small.png")
        assert os.path.isfile(path) and not os.path.islink(path)

    def test_best_file_is_the_highest_resolution(self, images_only_dir):
        images, _ = scan(images_only_dir)
        group = find_group(images, "img_a.png")
        assert os.path.basename(group["best_file"]["path"]) == "img_a.png"
        assert group["best_file"]["resolution"]["label"] == "240x240"

    def test_duplicate_entries_carry_metadata(self, images_only_dir):
        images, _ = scan(images_only_dir)
        group = find_group(images, "img_a.png")
        duplicate = group["duplicate_files"][0]
        assert duplicate["resolution"]["label"] == "120x120"
        assert duplicate["size"] > 0
        assert duplicate["size_formatted"]
        assert duplicate["is_exact_match"] is False

    @pytest.mark.skip(reason="port: resolve folder_path (cache keys) + SQLite API")

    def test_writes_relative_cache_keys(self, images_only_dir):
        """Cache keys must be plain relative names.

        HashCache resolves its directory, so a scan of an unresolved path used to
        store '../../..'-prefixed keys that never matched on the next run.
        """
        scan(images_only_dir)
        cache = HashCache(images_only_dir)
        assert cache.cache_data["hashes"]
        for key in cache.cache_data["hashes"]:
            assert not key.startswith(".."), f"non-relative cache key: {key}"
            assert key == os.path.basename(key)

    @pytest.mark.skip(reason="port: resolve folder_path (cache keys)")

    def test_scanning_an_unresolved_path_still_writes_clean_keys(self, images_only_dir):
        """Passing a path through a symlinked parent must not corrupt cache keys."""
        parent = os.path.dirname(images_only_dir)
        alias = os.path.join(parent, "alias_link")
        os.symlink(images_only_dir, alias)

        scan(alias)

        cache = HashCache(images_only_dir)
        assert cache.cache_data["hashes"]
        for key in cache.cache_data["hashes"]:
            assert not key.startswith("..")


class TestWarmRescan:
    def test_results_are_stable(self, images_only_dir):
        first, _ = scan(images_only_dir)
        second, _ = scan(images_only_dir)
        assert [group_basenames(g) for g in first] == [group_basenames(g) for g in second]

    def test_group_ids_are_stable(self, images_only_dir):
        first, _ = scan(images_only_dir)
        second, _ = scan(images_only_dir)
        assert {g["group_id"] for g in first} == {g["group_id"] for g in second}

    @pytest.mark.skip(reason="port: fix #1 single exact-match pass")

    def test_skips_exact_match_pass_when_nothing_changed(
        self, images_only_dir, count_exact_match_passes
    ):
        """An unchanged folder must not re-run exact-match elimination.

        The pass probes the resolution of every file in every group, so repeating it
        on a cold cache and again on every rescan was the dominant cost of a no-op scan.
        """
        scan(images_only_dir)
        count_exact_match_passes.clear()

        scan(images_only_dir)
        assert count_exact_match_passes == []

    def test_runs_exact_match_pass_once_on_a_cold_scan(
        self, images_only_dir, count_exact_match_passes
    ):
        """One pass per media kind, never two passes over the same groups."""
        scan(images_only_dir)
        assert len(count_exact_match_passes) <= 2

    def test_runs_exact_match_pass_when_a_file_arrives(
        self, images_only_dir, count_exact_match_passes
    ):
        scan(images_only_dir)
        count_exact_match_passes.clear()

        gradient_image(300, 300, 77).save(os.path.join(images_only_dir, "img_c.png"))
        scan(images_only_dir)
        assert count_exact_match_passes, "a new file must trigger exact-match handling"


class TestIncrementalGrouping:
    def test_new_similar_file_joins_the_existing_group(self, images_only_dir):
        """Regression: the group lookup compared absolute against relative paths.

        The guard was effectively always false, so every new file became a singleton
        and newly arrived duplicates were never reported.
        """
        scan(images_only_dir)

        gradient_image(240, 240, 11).resize((180, 180)).save(
            os.path.join(images_only_dir, "img_a_mid.png")
        )
        images, _ = scan(images_only_dir)

        group = find_group(images, "img_a_mid.png")
        assert group is not None
        assert "img_a.png" in group_basenames(group)

    def test_new_duplicate_of_a_previously_unique_file_is_found(self, images_only_dir):
        """A file that was unique is not a group representative.

        Indexing only representatives left such files unmatchable, so a duplicate
        arriving later was silently missed.
        """
        scan(images_only_dir)

        # img_b was unique on the first scan; add a near-copy of it
        gradient_image(200, 200, 200).resize((150, 150)).save(
            os.path.join(images_only_dir, "img_b_small.png")
        )
        images, _ = scan(images_only_dir)

        group = find_group(images, "img_b_small.png")
        assert group is not None
        assert "img_b.png" in group_basenames(group)

    def test_unrelated_new_file_stays_ungrouped(self, images_only_dir):
        scan(images_only_dir)
        gradient_image(210, 210, 133).save(os.path.join(images_only_dir, "img_d.png"))
        images, _ = scan(images_only_dir)
        assert find_group(images, "img_d.png") is None

    @pytest.mark.skip(reason="port: get_cached_groups must drop missing files")

    def test_removing_a_file_does_not_break_the_next_scan(self, images_only_dir):
        scan(images_only_dir)
        os.remove(os.path.join(images_only_dir, "img_a_small.png"))
        images, _ = scan(images_only_dir)
        assert all("img_a_small.png" not in group_basenames(g) for g in images)


class TestCachedHashDetection:
    @pytest.mark.skip(reason="port: _has_valid_cached_hash helper")
    def test_reports_known_and_unknown_files(self, images_only_dir):
        path = os.path.join(images_only_dir, "img_a.png")
        cache = HashCache(images_only_dir)
        assert DuplicateFinder._has_valid_cached_hash(path, cache) is False

        cache.get_hash(path, VIDEO_EXTENSIONS)
        assert DuplicateFinder._has_valid_cached_hash(path, cache) is True

    @pytest.mark.skip(reason="port: _has_valid_cached_hash helper")

    def test_modified_file_is_no_longer_valid(self, images_only_dir):
        import time

        path = os.path.join(images_only_dir, "img_a.png")
        cache = HashCache(images_only_dir)
        cache.get_hash(path, VIDEO_EXTENSIONS)

        time.sleep(0.01)
        gradient_image(90, 90, 3).save(path)
        assert DuplicateFinder._has_valid_cached_hash(path, cache) is False


class TestEmptyAndOddInputs:
    def test_empty_folder_returns_no_groups(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        images, videos = scan(str(folder))
        assert images == [] and videos == []

    def test_non_media_files_are_ignored(self, tmp_path):
        folder = tmp_path / "docs"
        folder.mkdir()
        (folder / "notes.txt").write_text("nothing to see")
        (folder / "data.json").write_text("{}")
        images, videos = scan(str(folder))
        assert images == [] and videos == []

    def test_single_image_is_not_a_duplicate(self, tmp_path):
        folder = tmp_path / "one"
        folder.mkdir()
        gradient_image(120, 120, 9).save(folder / "only.png")
        images, videos = scan(str(folder))
        assert images == [] and videos == []


@requires_ffmpeg
class TestVideoScanning:
    def test_groups_rescaled_video_with_its_source(self, media_dir):
        _, videos = scan(media_dir)
        group = find_group(videos, "vid_hi.mp4")
        assert group is not None
        assert "vid_lo.mp4" in group_basenames(group)

    def test_best_video_is_the_higher_resolution(self, media_dir):
        _, videos = scan(media_dir)
        group = find_group(videos, "vid_hi.mp4")
        assert os.path.basename(group["best_file"]["path"]) == "vid_hi.mp4"
        assert group["best_file"]["resolution"]["label"] == "640x480"

    def test_video_groups_report_duration(self, media_dir):
        _, videos = scan(media_dir)
        group = find_group(videos, "vid_hi.mp4")
        assert group["best_file"]["duration"] > 0
        assert group["best_file"]["duration_formatted"].endswith("s")

    @pytest.mark.skip(reason="port: fix #3 combined ffprobe")

    def test_cold_scan_probes_each_video_once(self, media_dir, count_ffprobe):
        """Resolution and duration share a probe, and the probe result is cached."""
        scan(media_dir)
        probed = {cmd[-1] if cmd[-1].endswith(".mp4") else None for cmd in count_ffprobe}
        assert len(count_ffprobe) <= 2, f"expected <=2 probes for 2 videos, got {len(count_ffprobe)}"
        assert len(probed - {None}) <= 2

    @pytest.mark.skip(reason="port: fix #3 persisted media metadata")

    def test_warm_rescan_runs_no_ffprobe(self, media_dir, count_ffprobe):
        """Video metadata is persisted, so an unchanged rescan spawns no subprocess."""
        from deduper.utils import media as media_module

        scan(media_dir)

        # Simulate a fresh process: in-memory caches gone, .deduper still on disk
        media_module._probe_video_cached.cache_clear()
        media_module._resolve_media_resolution_cached.cache_clear()
        count_ffprobe.clear()

        scan(media_dir)
        assert count_ffprobe == [], f"unexpected ffprobe calls: {count_ffprobe}"

    def test_thumbnails_are_not_treated_as_media(self, media_dir):
        scan(media_dir)
        images, _ = scan(media_dir)
        for group in images:
            for name in group_basenames(group):
                assert not name.startswith("thumb-deduper.")
