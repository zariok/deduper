"""Tests for the auto-resolve ledger and the automatic resolution tiers.

The ledger records what the scanner did, or in 'ledger' mode would do, on its
own. Precision is measured from it before 'apply' is ever switched on, so the
rows have to describe exactly the files a tier would touch - no more, no fewer.
"""

import logging
import os

import pytest

from deduper import config as config_module
from deduper.config import AUTO_RESOLVE_POLICIES, env_choice
from deduper.utils.hash_cache import HashCache

from .conftest import requires_ffmpeg

VAR = "DEDUPER_TEST_AUTO_RESOLVE"


class TestEnvChoice:
    def test_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv(VAR, raising=False)
        assert env_choice(VAR, AUTO_RESOLVE_POLICIES, "ledger") == "ledger"

    def test_blank_uses_default(self, monkeypatch):
        monkeypatch.setenv(VAR, "  ")
        assert env_choice(VAR, AUTO_RESOLVE_POLICIES, "ledger") == "ledger"

    @pytest.mark.parametrize("raw,expected", [("off", "off"), ("apply", "apply"),
                                              (" Ledger\n", "ledger"), ("APPLY", "apply")])
    def test_valid_values(self, monkeypatch, raw, expected):
        monkeypatch.setenv(VAR, raw)
        assert env_choice(VAR, AUTO_RESOLVE_POLICIES, "ledger") == expected

    @pytest.mark.parametrize("raw", ["yes", "on", "dry-run", "1"])
    def test_invalid_falls_back_with_warning(self, monkeypatch, caplog, raw):
        monkeypatch.setenv(VAR, raw)
        with caplog.at_level(logging.WARNING, logger=config_module.__name__):
            assert env_choice(VAR, AUTO_RESOLVE_POLICIES, "ledger") == "ledger"
        assert any(VAR in record.getMessage() for record in caplog.records)

    def test_config_default_is_ledger(self):
        # The suite never sets DEDUPER_AUTO_RESOLVE, so this is the shipped default
        assert config_module.Config.AUTO_RESOLVE_POLICY == "ledger"


@pytest.fixture
def cache(tmp_path) -> HashCache:
    folder = tmp_path / "ledger"
    folder.mkdir()
    return HashCache(os.path.realpath(folder))


def _abs(cache: HashCache, name: str) -> str:
    return os.path.join(str(cache.directory_path), name)


class TestLedgerApi:
    def test_record_stores_relative_and_returns_absolute(self, cache):
        dup, keeper = _abs(cache, "sub/a.jpg"), _abs(cache, "sub/a.png")
        row_id = cache.record_auto_action("g1", "identical", dup, keeper, {"size": 3}, True)
        assert isinstance(row_id, int)

        stored = cache._db.execute(
            "SELECT duplicate, keeper FROM auto_actions WHERE id=?", (row_id,)
        ).fetchone()
        assert stored == (os.path.join("sub", "a.jpg"), os.path.join("sub", "a.png"))

        (row,) = cache.get_auto_actions()
        assert row["id"] == row_id
        assert row["group_id"] == "g1"
        assert row["tier"] == "identical"
        assert row["duplicate"] == dup and row["keeper"] == keeper
        assert row["evidence"] == {"size": 3}
        assert row["applied"] is True
        assert row["created_at"] > 0

    def test_mark_merges_evidence_and_flips_applied(self, cache):
        row_id = cache.record_auto_action(
            "g1", "identical", _abs(cache, "a"), _abs(cache, "b"), {"size": 3}, True
        )
        cache.mark_auto_action(row_id, False, {"error": "symlink failed"})
        (row,) = cache.get_auto_actions()
        assert row["applied"] is False
        assert row["evidence"] == {"size": 3, "error": "symlink failed"}

    def test_mark_unknown_row_is_a_no_op(self, cache):
        cache.mark_auto_action(12345, True, {"x": 1})
        assert cache.count_auto_actions() == 0

    def test_replace_pending_is_idempotent_and_spares_other_rows(self, cache):
        applied_id = cache.record_auto_action(
            "g0", "reencode", _abs(cache, "x.jpg"), _abs(cache, "x.png"), {}, True
        )
        cache.replace_pending_auto_actions(
            "other", [("g9", _abs(cache, "o.jpg"), _abs(cache, "o.png"), {})]
        )
        rows = [
            ("g1", _abs(cache, "a.jpg"), _abs(cache, "a.png"), {"phash_max": 0}),
            ("g2", _abs(cache, "b.jpg"), _abs(cache, "b.png"), {"phash_max": 1}),
        ]
        cache.replace_pending_auto_actions("reencode", rows)
        cache.replace_pending_auto_actions("reencode", rows)

        assert cache.count_auto_actions(applied=False, tier="reencode") == 2
        # Applied history and other tiers are untouched
        assert cache.count_auto_actions(applied=True) == 1
        assert cache.get_auto_actions(applied=True)[0]["id"] == applied_id
        assert cache.count_auto_actions(tier="other") == 1

        cache.replace_pending_auto_actions("reencode", rows[:1])
        pending = cache.get_auto_actions(applied=False, tier="reencode")
        assert [r["group_id"] for r in pending] == ["g1"]
        assert pending[0]["duplicate"] == _abs(cache, "a.jpg")
        assert pending[0]["evidence"] == {"phash_max": 0}

        cache.replace_pending_auto_actions("reencode", [])
        assert cache.count_auto_actions(applied=False, tier="reencode") == 0

    def test_filters_counts_and_paging(self, cache):
        for i in range(3):
            cache.record_auto_action(
                f"g{i}", "identical", _abs(cache, f"{i}.a"), _abs(cache, f"{i}.b"), {}, True
            )
        cache.record_auto_action("gg", "gif", _abs(cache, "c.gif"), _abs(cache, "c.mp4"), {}, True)
        cache.replace_pending_auto_actions(
            "reencode", [("gr", _abs(cache, "r.jpg"), _abs(cache, "r.png"), {})]
        )

        assert cache.count_auto_actions() == 5
        assert cache.count_auto_actions(applied=True) == 4
        assert cache.count_auto_actions(applied=False) == 1
        assert cache.count_auto_actions(tier="identical") == 3
        assert cache.count_auto_actions(applied=False, tier="identical") == 0
        assert cache.count_auto_actions_by_tier(True) == {"identical": 3, "gif": 1}
        assert cache.count_auto_actions_by_tier(False) == {"reencode": 1}
        assert cache.count_auto_actions_by_tier(None) == {"identical": 3, "gif": 1, "reencode": 1}

        assert [r["tier"] for r in cache.get_auto_actions(tier="gif")] == ["gif"]
        newest_first = cache.get_auto_actions(tier="identical")
        assert [r["group_id"] for r in newest_first] == ["g2", "g1", "g0"]
        page = cache.get_auto_actions(tier="identical", limit=2, offset=1)
        assert [r["group_id"] for r in page] == ["g1", "g0"]

    def test_ledger_survives_reopen(self, cache):
        cache.record_auto_action("g1", "identical", _abs(cache, "a"), _abs(cache, "b"), {}, True)
        directory = str(cache.directory_path)
        cache._close_connection()
        assert HashCache(directory).count_auto_actions(applied=True) == 1


def _scan(folder: str):
    from deduper.services.duplicate_finder import DuplicateFinder

    from .conftest import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

    return DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)


class TestProvenPassesWriteLedger:
    def test_identical_pass_records_an_applied_row(self, images_only_dir):
        _scan(images_only_dir)
        cache = HashCache(images_only_dir)
        rows = cache.get_auto_actions(tier="identical")
        assert len(rows) == 1
        (row,) = rows
        assert row["applied"] is True
        assert row["duplicate"] == os.path.join(images_only_dir, "img_a_copy.png")
        assert row["keeper"] == os.path.join(images_only_dir, "img_a.png")
        assert row["evidence"]["size"] == os.path.getsize(row["keeper"])
        assert len(row["evidence"]["digest"]) == 64
        assert len(row["group_id"]) == 8
        assert os.path.islink(row["duplicate"])

    def test_failed_symlink_is_marked_not_applied(self, images_only_dir, monkeypatch):
        from deduper.services import duplicate_finder

        monkeypatch.setattr(
            duplicate_finder, "create_symlink_and_remove_duplicate", lambda d, k: False
        )
        _scan(images_only_dir)
        (row,) = HashCache(images_only_dir).get_auto_actions(tier="identical")
        assert row["applied"] is False
        assert row["evidence"]["error"] == "symlink failed"
        assert not os.path.islink(row["duplicate"])

    @requires_ffmpeg
    def test_gif_pass_records_an_applied_row(self, signature_dir):
        _scan(signature_dir)
        rows = HashCache(signature_dir).get_auto_actions(tier="gif", applied=True)
        by_dup = {os.path.basename(r["duplicate"]): r for r in rows}
        assert "full.gif" in by_dup
        row = by_dup["full.gif"]
        assert os.path.basename(row["keeper"]) == "full.mp4"
        assert row["evidence"]["keeper_width"] == 640
        assert os.path.islink(row["duplicate"])
