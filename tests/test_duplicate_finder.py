"""Tests for the duplicate detection pipeline.

These exercise the destructive path: exact duplicates really are deleted and
replaced with symlinks, so every test scans a throwaway copy of the fixtures.
"""

import os
import shutil

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


    def test_writes_relative_cache_keys(self, images_only_dir):
        """Cache keys must be plain relative names.

        HashCache resolves its directory, so a scan of an unresolved path used to
        store '../../..'-prefixed keys that never matched on the next run.
        """
        scan(images_only_dir)
        cache = HashCache(images_only_dir)
        for name in ("img_a.png", "img_a_small.png", "img_b.png"):
            assert cache.has_cached_hash(name), f"expected a plain relative key for {name}"

    def test_scanning_an_unresolved_path_still_writes_clean_keys(self, images_only_dir):
        """Reaching the folder through a symlinked parent must not corrupt keys."""
        parent = os.path.dirname(images_only_dir)
        alias = os.path.join(parent, "alias_link")
        os.symlink(images_only_dir, alias)

        scan(alias)

        cache = HashCache(images_only_dir)
        assert cache.has_cached_hash("img_a.png"), (
            "scanning via a symlinked path stored keys the resolved cache cannot find"
        )


def _pad_to_match(smaller: str, larger: str) -> None:
    """Append nulls so two files share a byte count. PIL ignores data past IEND."""
    gap = os.path.getsize(larger) - os.path.getsize(smaller)
    assert gap > 0, "expected the first file to be the smaller one"
    with open(smaller, "ab") as fh:
        fh.write(b"\0" * gap)


@pytest.fixture
def lookalikes(tmp_path) -> str:
    """A folder holding two images that every cheap test says are the same file.

    Identical perceptual hash, identical resolution, identical byte count - and
    visibly different content. This is the collision the old rule deleted on.
    """
    folder = tmp_path / "lookalikes"
    folder.mkdir()

    base = gradient_image(240, 240, 11)
    a = folder / "twin_a.png"
    base.save(a)

    marked = base.copy()
    pixels = marked.load()
    for x in range(8):
        for y in range(8):
            pixels[x, y] = (255, 0, 0)  # a red corner block: plainly not the same image
    b = folder / "twin_b.png"
    marked.save(b)

    _pad_to_match(str(a), str(b))
    assert os.path.getsize(a) == os.path.getsize(b)
    assert a.read_bytes() != b.read_bytes()
    return str(folder)


class TestOnlyIdenticalFilesAreRemoved:
    """Auto-elimination deletes files, so it must prove identity, not infer it."""

    def test_lookalikes_are_grouped_but_both_survive(self, lookalikes):
        images, _ = scan(lookalikes)

        # Guard the premise: if these stopped grouping, the test proves nothing.
        group = find_group(images, "twin_a.png")
        assert group is not None, "fixture no longer collides; the test is vacuous"
        assert sorted(group_basenames(group)) == ["twin_a.png", "twin_b.png"]

        for name in ("twin_a.png", "twin_b.png"):
            path = os.path.join(lookalikes, name)
            assert os.path.isfile(path) and not os.path.islink(path), (
                f"{name} shares a hash, resolution and size with its pair but not its "
                f"bytes - deleting it destroys a distinct image"
            )

    def test_lookalikes_are_not_flagged_as_exact_matches(self, lookalikes):
        images, _ = scan(lookalikes)
        group = find_group(images, "twin_a.png")
        assert all(d["is_exact_match"] is False for d in group["duplicate_files"])

    def test_one_of_each_identical_pair_is_kept(self, tmp_path):
        """Identical files that are not the group's best still collapse to one.

        The old pass compared every file against the best file alone, so a pair
        that was identical to each other but smaller than the best survived intact.
        """
        folder = tmp_path / "nested"
        folder.mkdir()
        base = gradient_image(240, 240, 11)
        base.save(folder / "big.png")
        small = base.resize((120, 120))
        small.save(folder / "small_one.png")
        shutil.copy(folder / "small_one.png", folder / "small_two.png")

        scan(str(folder))

        assert os.path.isfile(folder / "big.png") and not os.path.islink(folder / "big.png")
        survivors = [
            n for n in ("small_one.png", "small_two.png")
            if not os.path.islink(folder / n)
        ]
        links = [n for n in ("small_one.png", "small_two.png") if os.path.islink(folder / n)]
        assert len(survivors) == 1, f"expected exactly one of the identical pair to remain, kept {survivors}"
        assert len(links) == 1
        assert os.path.realpath(folder / links[0]) == str(folder / survivors[0])

    def test_the_kept_file_is_stable_across_scans(self, tmp_path):
        folder = tmp_path / "stable"
        folder.mkdir()
        gradient_image(240, 240, 11).save(folder / "b_copy.png")
        shutil.copy(folder / "b_copy.png", folder / "a_copy.png")

        scan(str(folder))
        kept = [n for n in ("a_copy.png", "b_copy.png") if not os.path.islink(folder / n)]
        assert len(kept) == 1

        scan(str(folder))
        assert not os.path.islink(folder / kept[0]), "the survivor moved between scans"


@requires_ffmpeg
class TestVideoSignatureMatching:
    """The signature exists so a video is judged on its whole length.

    Every case here is one a single frame at a fixed 1s offset gets wrong:
    it cannot see past a shared opening, cannot follow a start trim, and
    cannot reach into a clip shorter than the offset itself.
    """

    def _links(self, folder):
        return {n for n in os.listdir(folder) if os.path.islink(os.path.join(folder, n))}

    def test_a_gif_rendered_from_a_video_matches_it(self, signature_dir):
        _, videos = scan(signature_dir)
        group = find_group(videos, "full.mp4")
        assert group is not None
        assert "full.gif" in group_basenames(group) or os.path.islink(
            os.path.join(signature_dir, "full.gif")
        ), "the gif was neither grouped with its source nor tombstoned to it"

    def test_a_clip_trimmed_at_the_end_still_matches(self, signature_dir):
        _, videos = scan(signature_dir)
        group = find_group(videos, "full.mp4")
        assert "trim_end.mp4" in group_basenames(group)

    def test_a_clip_trimmed_at_the_start_still_matches(self, signature_dir):
        """A fixed-offset single frame misses this outright: t=1s in the trim is
        t=11s in the source, so the one sampled instant shows different content."""
        _, videos = scan(signature_dir)
        group = find_group(videos, "full.mp4")
        assert "trim_start.mp4" in group_basenames(group)

    def test_clips_sharing_an_opening_are_not_merged(self, signature_dir):
        """Both start with the same 8s navy card, then diverge completely."""
        _, videos = scan(signature_dir)
        group = find_group(videos, "share_a.mp4")
        if group is not None:
            assert "share_b.mp4" not in group_basenames(group), (
                "a shared opening was treated as the whole clip"
            )
        assert not os.path.islink(os.path.join(signature_dir, "share_b.mp4"))

    def test_a_sub_second_clip_is_detected_at_all(self, signature_dir):
        """0.6s: the old fixed 1s seek was past the end, so ffmpeg produced no
        frame and the file was dropped from detection entirely."""
        scan(signature_dir)
        brief_gif = os.path.join(signature_dir, "brief.gif")
        assert os.path.islink(brief_gif), (
            "the sub-second gif was never matched to its source video"
        )
        assert os.path.realpath(brief_gif) == os.path.join(signature_dir, "brief.mp4")

    def test_a_sub_second_clip_gets_a_thumbnail(self, signature_dir):
        scan(signature_dir)
        assert os.path.isfile(os.path.join(signature_dir, "thumb-deduper.brief.mp4.jpg"))

    def test_the_video_keeps_its_thumbnail_after_its_gif_is_tombstoned(self, signature_dir):
        """clip.mp4 and clip.gif share a stem, so a stem-named thumbnail was one
        shared file: tombstoning the gif deleted the thumbnail the video needed,
        and concurrent extraction raced on the same path before that."""
        scan(signature_dir)
        assert os.path.islink(os.path.join(signature_dir, "full.gif")), "premise: gif was tombstoned"
        assert os.path.isfile(os.path.join(signature_dir, "thumb-deduper.full.mp4.jpg")), (
            "the video lost its thumbnail when its gif was tombstoned"
        )

    def test_results_are_stable_across_rescans(self, signature_dir):
        first, _ = scan(signature_dir), None
        second, _ = scan(signature_dir), None
        assert [group_basenames(g) for g in first[1]] == [group_basenames(g) for g in second[1]]


@requires_ffmpeg
class TestGifTombstoning:
    def test_the_gif_becomes_a_symlink_to_the_video(self, signature_dir):
        scan(signature_dir)
        gif = os.path.join(signature_dir, "full.gif")
        assert os.path.islink(gif), "gif was not tombstoned"
        assert os.path.realpath(gif) == os.path.join(signature_dir, "full.mp4")

    def test_the_video_itself_is_untouched(self, signature_dir):
        scan(signature_dir)
        mp4 = os.path.join(signature_dir, "full.mp4")
        assert os.path.isfile(mp4) and not os.path.islink(mp4)

    def test_the_tombstone_keeps_the_path_visible(self, signature_dir):
        """The symlink is what a scraper checks, so the path must still exist."""
        scan(signature_dir)
        assert os.path.exists(os.path.join(signature_dir, "full.gif"))

    def test_a_gif_is_not_tombstoned_against_unmatched_footage(self, signature_dir):
        """Tombstoning runs only on groups alignment has already confirmed."""
        scan(signature_dir)
        gif = os.path.join(signature_dir, "full.gif")
        assert os.path.realpath(gif) != os.path.join(signature_dir, "share_a.mp4")
        assert os.path.realpath(gif) != os.path.join(signature_dir, "share_b.mp4")

    def test_tombstoned_gifs_do_not_return_on_rescan(self, signature_dir):
        scan(signature_dir)
        links = self._links = {
            n for n in os.listdir(signature_dir)
            if os.path.islink(os.path.join(signature_dir, n))
        }
        scan(signature_dir)
        still = {
            n for n in os.listdir(signature_dir)
            if os.path.islink(os.path.join(signature_dir, n))
        }
        assert links == still, "a rescan disturbed the tombstones"


class TestWarmRescan:
    def test_results_are_stable(self, images_only_dir):
        first, _ = scan(images_only_dir)
        second, _ = scan(images_only_dir)
        assert [group_basenames(g) for g in first] == [group_basenames(g) for g in second]

    def test_group_ids_are_stable(self, images_only_dir):
        first, _ = scan(images_only_dir)
        second, _ = scan(images_only_dir)
        assert {g["group_id"] for g in first} == {g["group_id"] for g in second}

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

    def test_removing_a_file_does_not_break_the_next_scan(self, images_only_dir):
        scan(images_only_dir)
        os.remove(os.path.join(images_only_dir, "img_a_small.png"))
        images, _ = scan(images_only_dir)
        assert all("img_a_small.png" not in group_basenames(g) for g in images)


class TestCachedHashDetection:
    def test_reports_known_and_unknown_files(self, images_only_dir):
        path = os.path.join(images_only_dir, "img_a.png")
        cache = HashCache(images_only_dir)
        assert DuplicateFinder._has_valid_cached_hash(path, cache) is False

        cache.get_hash(path, VIDEO_EXTENSIONS)
        assert DuplicateFinder._has_valid_cached_hash(path, cache) is True

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


    def test_cold_scan_probes_each_video_once(self, media_dir, count_ffprobe):
        """Resolution and duration share a probe, and the probe result is cached."""
        scan(media_dir)
        probed = {cmd[-1] if cmd[-1].endswith(".mp4") else None for cmd in count_ffprobe}
        assert len(count_ffprobe) <= 2, f"expected <=2 probes for 2 videos, got {len(count_ffprobe)}"
        assert len(probed - {None}) <= 2


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
