"""Guards for the warm rescan path: no per-file queries, no repeated stats.

A rescan of an unchanged folder should read the hash table once and take each
file's (mtime, size) from the walk, not ask SQLite and the filesystem again per
file. Getting this wrong costs time rather than correctness, so nothing else in
the suite would notice. See docs/OPTIMIZATION_AND_NATIVE_CORE_PLAN.md 2.4/2.5.
"""

import os
import threading
from collections import Counter

from deduper.services.duplicate_finder import DuplicateFinder
from deduper.utils.hash_cache import HashCache, get_hash_cache

from .conftest import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def scan(folder: str):
    return DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)


def real_media_files(folder: str) -> set[str]:
    """What the scan's walk should see: media files, symlinks excluded."""
    exts = tuple(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    found = set()
    for root, _, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            if name.lower().endswith(exts) and not os.path.islink(path):
                found.add(path)
    return found


def capture_scan_stats(monkeypatch) -> list[dict[str, tuple[float, int]]]:
    """Record a copy of every stat snapshot a scan installs."""
    captured: list[dict[str, tuple[float, int]]] = []
    original = HashCache.scan_stats

    def recording(self, stats):
        captured.append(dict(stats))
        return original(self, stats)

    monkeypatch.setattr(HashCache, "scan_stats", recording)
    return captured


class TestCachedGroupsMembership:
    def _groups_with_link_and_missing(self, folder: str) -> tuple[HashCache, str]:
        cache = HashCache(folder)
        kept = os.path.join(folder, "img_a.png")
        linked = os.path.join(folder, "img_a_copy.png")
        removed = os.path.join(folder, "img_a_small.png")
        os.remove(linked)
        os.symlink("img_a.png", linked)
        cache.set_cached_groups({kept: [kept, linked, removed]})
        os.remove(removed)
        return cache, kept

    def test_existing_set_matches_the_filesystem_filter(self, images_only_dir):
        cache, kept = self._groups_with_link_and_missing(images_only_dir)
        from_fs = cache.get_cached_groups()
        from_set = cache.get_cached_groups(existing=real_media_files(images_only_dir))
        assert from_fs == from_set == {kept: [kept]}

    def test_existing_set_is_used_instead_of_the_filesystem(self, images_only_dir, monkeypatch):
        cache = HashCache(images_only_dir)
        a = os.path.join(images_only_dir, "img_a.png")
        b = os.path.join(images_only_dir, "img_b.png")
        cache.set_cached_groups({a: [a, b]})

        def forbidden(path):
            raise AssertionError(f"touched the filesystem for {path}")

        monkeypatch.setattr(os.path, "islink", forbidden)
        monkeypatch.setattr(os.path, "exists", forbidden)
        # Membership decides alone: b exists on disk but is not in the set
        assert cache.get_cached_groups(existing={a}) == {a: [a]}

    def test_without_a_set_the_filesystem_decides(self, images_only_dir):
        cache = HashCache(images_only_dir)
        a = os.path.join(images_only_dir, "img_a.png")
        b = os.path.join(images_only_dir, "img_b.png")
        cache.set_cached_groups({a: [a, b]})
        assert cache.get_cached_groups() == {a: [a, b]}

    def test_walk_paths_match_stored_group_paths(self, tmp_path, monkeypatch):
        """The walk set only works as a filter if its paths are spelled exactly as
        get_cached_groups rebuilds them - including in nested folders."""
        from .conftest import gradient_image

        root = os.path.realpath(tmp_path / "nested")
        os.makedirs(os.path.join(root, "sub"))
        image = gradient_image(120, 120, 5)
        image.save(os.path.join(root, "top.png"))
        image.resize((60, 60)).save(os.path.join(root, "sub", "small.png"))
        scan(root)

        captured = capture_scan_stats(monkeypatch)
        scan(root)
        walked = set(captured[0])
        groups = get_hash_cache(root).get_cached_groups()
        members = {m for files in groups.values() for m in files}
        assert members, "expected the two copies to group"
        assert members <= walked


class TestWarmRescan:
    def test_no_per_file_hash_queries(self, images_only_dir, monkeypatch):
        scan(images_only_dir)

        calls: Counter[str] = Counter()
        for name in ("has_cached_hash", "_is_file_unchanged", "get_cached_hash_str",
                     "get_file_hash_rows"):
            original = getattr(HashCache, name)

            def counting(*args, _name=name, _original=original, **kwargs):
                calls[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(HashCache, name, counting)

        scan(images_only_dir)
        assert calls["has_cached_hash"] == 0
        assert calls["_is_file_unchanged"] == 0
        assert calls["get_cached_hash_str"] == 0
        assert calls["get_file_hash_rows"] == 1

    def test_no_stat_syscalls_on_media_files(self, images_only_dir, monkeypatch):
        """Everything after the walk reads (mtime, size) from its snapshot."""
        scan(images_only_dir)
        media = real_media_files(images_only_dir)
        assert media

        # Count only once the walk is done and its snapshot is installed; the
        # walk's own DirEntry.stat() is the one stat per file the scan keeps
        walked = threading.Event()
        original_scan_stats = HashCache.scan_stats

        def marking(self, stats):
            walked.set()
            return original_scan_stats(self, stats)

        monkeypatch.setattr(HashCache, "scan_stats", marking)

        touched: Counter[str] = Counter()
        real_stat, real_lstat = os.stat, os.lstat

        def counting_stat(path, *args, **kwargs):
            if walked.is_set() and os.fspath(path) in media:
                touched[os.fspath(path)] += 1
            return real_stat(path, *args, **kwargs)

        def counting_lstat(path, *args, **kwargs):
            if walked.is_set() and os.fspath(path) in media:
                touched[os.fspath(path)] += 1
            return real_lstat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", counting_stat)
        monkeypatch.setattr(os, "lstat", counting_lstat)
        images, _ = scan(images_only_dir)
        assert walked.is_set()
        assert images, "the warm scan should still report the similar pair"
        assert sum(touched.values()) == 0, dict(touched)

    def test_auto_pass_leaves_symlinked_copy_out_of_stored_groups(
        self, images_only_dir, monkeypatch
    ):
        """Dropping the per-file remove_file_from_groups must not change the table:
        the trimmed groups are written once after the pass."""
        rewrites = []
        monkeypatch.setattr(
            HashCache, "remove_file_from_groups", lambda self, path: rewrites.append(path)
        )
        scan(images_only_dir)
        assert rewrites == []
        assert os.path.islink(os.path.join(images_only_dir, "img_a_copy.png"))
        rows = get_hash_cache(images_only_dir)._db.execute(
            "SELECT files FROM grouping_results"
        ).fetchall()
        stored = " ".join(r[0] for r in rows)
        assert "img_a_copy.png" not in stored
        assert "img_a_small.png" in stored


class TestScanStats:
    def test_snapshot_equals_get_file_stats(self, images_only_dir, monkeypatch):
        captured = capture_scan_stats(monkeypatch)
        scan(images_only_dir)
        assert len(captured) == 1
        snapshot = captured[0]

        # Outside any scan, _get_file_stats does a real os.stat
        cache = HashCache(images_only_dir)
        # The first scan symlinked the exact copy, and the walk ran before that
        stats = {p: cache._get_file_stats(p) for p in real_media_files(images_only_dir)}
        assert stats
        for path, value in stats.items():
            assert snapshot[path] == value
            assert type(snapshot[path][0]) is type(value[0])
            assert type(snapshot[path][1]) is type(value[1])

    def test_snapshot_is_scoped_to_the_block(self, images_only_dir):
        cache = HashCache(images_only_dir)
        path = os.path.join(images_only_dir, "img_a.png")
        real = cache._get_file_stats(path)
        stale = {path: (1.0, 1)}

        with cache.scan_stats(stale):
            assert cache._get_file_stats(path) == (1.0, 1)
            assert cache.get_file_size(path) == 1
        assert cache._get_file_stats(path) == real

    def test_snapshot_is_not_seen_by_other_threads(self, images_only_dir):
        cache = HashCache(images_only_dir)
        path = os.path.join(images_only_dir, "img_a.png")
        real = cache._get_file_stats(path)
        seen: list[tuple[float, int]] = []

        with cache.scan_stats({path: (1.0, 1)}):
            worker = threading.Thread(target=lambda: seen.append(cache._get_file_stats(path)))
            worker.start()
            worker.join()
        assert seen == [real]

    def test_update_file_stats_forgets_the_snapshot(self, images_only_dir):
        cache = HashCache(images_only_dir)
        path = os.path.join(images_only_dir, "img_a.png")
        cache.get_hash(path, VIDEO_EXTENSIONS)
        real = cache._get_file_stats(path)

        with cache.scan_stats({path: (1.0, 1)}):
            cache.update_file_stats(path)
            assert cache._get_file_stats(path) == real
        row = cache._db.execute(
            "SELECT mtime, size FROM file_hashes WHERE relative_path='img_a.png'"
        ).fetchone()
        assert tuple(row) == real
