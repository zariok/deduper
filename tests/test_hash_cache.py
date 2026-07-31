"""Tests for the SQLite-backed hash cache."""

import os
import sqlite3
import time

import pytest

from deduper.utils.hash_cache import (
    HashCache,
    close_all_caches,
    close_hash_cache,
    get_hash_cache,
)

from .conftest import VIDEO_EXTENSIONS, gradient_image


class TestHashRoundTrip:
    def test_hash_is_persisted_and_reused(self, images_only_dir):
        path = os.path.join(images_only_dir, "img_a.png")
        cache = HashCache(images_only_dir)
        original = cache.get_hash(path, VIDEO_EXTENSIONS)
        cache.save()
        close_hash_cache(images_only_dir)

        reopened = HashCache(images_only_dir)
        assert str(reopened.get_hash(path, VIDEO_EXTENSIONS)) == str(original)

    def test_cache_keys_are_relative(self, images_only_dir):
        cache = HashCache(images_only_dir)
        cache.get_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        assert cache.has_cached_hash("img_a.png")
        assert cache.get_cached_hash_str("img_a.png")

    def test_modified_file_invalidates_the_cached_hash(self, images_only_dir):
        path = os.path.join(images_only_dir, "img_a.png")
        cache = HashCache(images_only_dir)
        cache.get_hash(path, VIDEO_EXTENSIONS)
        assert cache._is_file_unchanged(path) is True

        time.sleep(0.01)
        gradient_image(100, 100, 42).save(path)
        assert cache._is_file_unchanged(path) is False

    def test_missing_file_yields_no_hash(self, images_only_dir):
        cache = HashCache(images_only_dir)
        assert cache.get_hash(os.path.join(images_only_dir, "gone.png"), VIDEO_EXTENSIONS) is None

    def test_creates_a_sqlite_database(self, images_only_dir):
        """The SQLite store is .deduper.db; .deduper remains the legacy JSON name."""
        cache = HashCache(images_only_dir)
        cache.get_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        cache.save()

        db = os.path.join(images_only_dir, HashCache.DB_FILENAME)
        assert os.path.exists(db)
        with sqlite3.connect(db) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert tables, "expected at least one table"


class TestConnectionPooling:
    def test_same_directory_returns_the_same_instance(self, images_only_dir):
        assert get_hash_cache(images_only_dir) is get_hash_cache(images_only_dir)

    def test_pool_is_keyed_on_the_resolved_path(self, images_only_dir):
        trailing = images_only_dir + os.sep
        assert get_hash_cache(images_only_dir) is get_hash_cache(trailing)

    def test_closing_drops_the_pooled_instance(self, images_only_dir):
        first = get_hash_cache(images_only_dir)
        close_hash_cache(images_only_dir)
        assert get_hash_cache(images_only_dir) is not first

    def test_close_all_is_safe_to_repeat(self, images_only_dir):
        get_hash_cache(images_only_dir)
        close_all_caches()
        close_all_caches()


class TestBestFileSelection:
    def test_round_trips_a_selection(self, images_only_dir):
        cache = HashCache(images_only_dir)
        target = os.path.join(images_only_dir, "img_a_small.png")
        cache.set_best_file("group1", target)
        assert cache.get_best_file("group1") == target

    def test_unknown_group_has_no_selection(self, images_only_dir):
        assert HashCache(images_only_dir).get_best_file("nope") is None

    def test_selection_survives_a_reopen(self, images_only_dir):
        target = os.path.join(images_only_dir, "img_a_small.png")
        cache = HashCache(images_only_dir)
        cache.set_best_file("group1", target)
        cache.save()
        close_hash_cache(images_only_dir)

        assert HashCache(images_only_dir).get_best_file("group1") == target


class TestGroups:
    def test_round_trips_groups(self, images_only_dir):
        cache = HashCache(images_only_dir)
        members = [
            os.path.join(images_only_dir, "img_a.png"),
            os.path.join(images_only_dir, "img_a_small.png"),
        ]
        cache.set_cached_groups({members[0]: members})
        assert cache.get_cached_groups() == {members[0]: members}

    def test_symlinked_members_are_dropped(self, images_only_dir):
        cache = HashCache(images_only_dir)
        kept = os.path.join(images_only_dir, "img_a.png")
        linked = os.path.join(images_only_dir, "img_a_copy.png")
        os.remove(linked)
        os.symlink("img_a.png", linked)
        cache.set_cached_groups({kept: [kept, linked]})

        assert cache.get_cached_groups() == {kept: [kept]}

    @pytest.mark.skip(reason="port: get_cached_groups must drop missing files too")
    def test_deleted_members_are_dropped(self, images_only_dir):
        """os.path.islink is False for a missing path, so a deleted file otherwise
        stays in its group on every later scan, reported with zeroed metadata."""
        cache = HashCache(images_only_dir)
        kept = os.path.join(images_only_dir, "img_a.png")
        removed = os.path.join(images_only_dir, "img_a_small.png")
        cache.set_cached_groups({kept: [kept, removed]})

        os.remove(removed)
        assert cache.get_cached_groups() == {kept: [kept]}

    def test_group_files_round_trip(self, images_only_dir):
        cache = HashCache(images_only_dir)
        members = [
            os.path.join(images_only_dir, "img_a.png"),
            os.path.join(images_only_dir, "img_a_small.png"),
        ]
        cache.set_group_files("g1", members)
        assert sorted(cache.get_group_files("g1")) == sorted(members)

    def test_removing_a_file_updates_groups(self, images_only_dir):
        cache = HashCache(images_only_dir)
        kept = os.path.join(images_only_dir, "img_a.png")
        other = os.path.join(images_only_dir, "img_a_small.png")
        cache.set_cached_groups({kept: [kept, other]})

        cache.remove_file_from_groups(other)
        assert cache.get_cached_groups() == {kept: [kept]}


class TestCleanup:
    def test_removes_entries_for_deleted_files(self, images_only_dir):
        path = os.path.join(images_only_dir, "img_b.png")
        cache = HashCache(images_only_dir)
        cache.get_hash(path, VIDEO_EXTENSIONS)
        assert cache.has_cached_hash("img_b.png")

        os.remove(path)
        remaining = {
            os.path.join(images_only_dir, name)
            for name in ("img_a.png", "img_a_copy.png", "img_a_small.png")
        }
        cache.cleanup_deleted_files(remaining)

        assert not cache.has_cached_hash("img_b.png")

    def test_invalidate_clears_everything(self, images_only_dir):
        cache = HashCache(images_only_dir)
        cache.get_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        cache.invalidate_cache()
        assert not cache.has_cached_hash("img_a.png")


class TestCacheStats:
    def test_reports_counts(self, images_only_dir):
        cache = HashCache(images_only_dir)
        cache.get_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        cache.save()

        stats = cache.get_cache_stats()
        assert stats["total_cached_files"] >= 1
        assert stats["cache_size_mb"] >= 0


class TestCorruptDatabase:
    def test_unreadable_database_is_replaced(self, images_only_dir):
        db = os.path.join(images_only_dir, HashCache.DB_FILENAME)
        with open(db, "wb") as handle:
            handle.write(b"this is not a sqlite database")
        close_all_caches()

        cache = HashCache(images_only_dir)
        # Must recover rather than raise, and be usable afterwards
        cache.get_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        assert cache.has_cached_hash("img_a.png")
