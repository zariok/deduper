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


@pytest.fixture
def reencode_dir(tmp_path) -> str:
    """One picture saved twice at the same size - PNG and a high-quality JPEG -
    plus an unrelated image. The JPEG is 4:4:4: Pillow's default 4:2:0 shifts
    the colour of this saturated synthetic gradient enough for the analysis to
    flag a colour difference, which (correctly) disqualifies the group."""
    from .conftest import gradient_image

    folder = tmp_path / "reencode"
    folder.mkdir()
    picture = gradient_image(240, 240, 11)
    picture.save(folder / "pic.png")
    picture.save(folder / "pic.jpg", quality=95, subsampling=0)
    gradient_image(200, 200, 200).save(folder / "other.png")
    return os.path.realpath(folder)


def _policy(monkeypatch, value: str) -> None:
    monkeypatch.setattr(config_module.Config, "AUTO_RESOLVE_POLICY", value)


def _reencode_rows(folder: str, applied: bool | None = None) -> list[dict]:
    return HashCache(folder).get_auto_actions(applied=applied, tier="reencode")


class TestReencodeTier:
    def test_fixture_is_within_the_tier_bound(self, reencode_dir):
        """Premise: if the JPEG drifts past 2 bits the other tests mean nothing."""
        from deduper.utils.media import MultiHash

        _scan(reencode_dir)
        rows = HashCache(reencode_dir).get_file_hash_rows()
        png = MultiHash.from_str(rows["pic.png"][0])
        jpg = MultiHash.from_str(rows["pic.jpg"][0])
        assert png.phash_distance(jpg) <= 2
        assert png.dhash_distance(jpg) <= 2

    def test_ledger_records_without_touching_files(self, reencode_dir, monkeypatch):
        _policy(monkeypatch, "ledger")
        png = os.path.join(reencode_dir, "pic.png")
        jpg = os.path.join(reencode_dir, "pic.jpg")
        before = {p: os.stat(p).st_mtime_ns for p in (png, jpg)}

        images, _ = _scan(reencode_dir)

        (row,) = _reencode_rows(reencode_dir)
        assert row["applied"] is False
        assert row["duplicate"] == jpg and row["keeper"] == png
        evidence = row["evidence"]
        assert evidence["width"] == 240 and evidence["height"] == 240
        assert evidence["phash_max"] <= 2 and evidence["dhash_max"] <= 2
        assert evidence["members"] == 2
        assert evidence["keeper_score"] >= evidence["duplicate_score"] > 0
        assert evidence["keeper_size"] == os.path.getsize(png)
        assert evidence["duplicate_size"] == os.path.getsize(jpg)
        assert evidence["keeper_reason"]
        assert row["group_id"] == images[0]["group_id"]

        for path, mtime in before.items():
            assert not os.path.islink(path) and os.stat(path).st_mtime_ns == mtime
        assert len(images) == 1, "ledger mode must leave the group for the user"

    def test_second_ledger_scan_does_not_duplicate_rows(self, reencode_dir, monkeypatch):
        _policy(monkeypatch, "ledger")
        _scan(reencode_dir)
        _scan(reencode_dir)  # warm: no new files, exact-match pass skipped
        assert len(_reencode_rows(reencode_dir)) == 1

    def test_warm_ledger_scan_reads_hash_rows_once(self, reencode_dir, monkeypatch):
        """The tier reuses the warm path's read of file_hashes."""
        _policy(monkeypatch, "ledger")
        _scan(reencode_dir)
        calls = []
        original = HashCache.get_file_hash_rows

        def counting(self):
            calls.append(1)
            return original(self)

        monkeypatch.setattr(HashCache, "get_file_hash_rows", counting)
        _scan(reencode_dir)
        assert len(calls) == 1
        assert len(_reencode_rows(reencode_dir)) == 1

    def test_apply_replaces_the_duplicate(self, reencode_dir, monkeypatch):
        _policy(monkeypatch, "ledger")
        _scan(reencode_dir)
        assert len(_reencode_rows(reencode_dir, applied=False)) == 1

        _policy(monkeypatch, "apply")
        images, _ = _scan(reencode_dir)

        jpg = os.path.join(reencode_dir, "pic.jpg")
        assert os.path.islink(jpg)
        assert os.path.realpath(jpg) == os.path.join(reencode_dir, "pic.png")
        (row,) = _reencode_rows(reencode_dir, applied=True)
        assert row["duplicate"] == jpg
        assert _reencode_rows(reencode_dir, applied=False) == []
        assert images == [], "the resolved group must drop out of the results"

        # And it stays resolved: a rescan finds nothing further to do
        images, _ = _scan(reencode_dir)
        assert images == []
        assert len(_reencode_rows(reencode_dir)) == 1

    def test_policy_off_writes_nothing(self, reencode_dir, monkeypatch):
        _policy(monkeypatch, "off")
        images, _ = _scan(reencode_dir)
        assert _reencode_rows(reencode_dir) == []
        assert len(images) == 1
        assert not os.path.islink(os.path.join(reencode_dir, "pic.jpg"))

    def test_differing_dimensions_do_not_qualify(self, images_only_dir, monkeypatch):
        """img_a and img_a_small group together, but one is a downscale."""
        _policy(monkeypatch, "apply")
        images, _ = _scan(images_only_dir)
        assert images, "premise: img_a and img_a_small must still group"
        assert _reencode_rows(images_only_dir) == []
        assert not os.path.islink(os.path.join(images_only_dir, "img_a_small.png"))

    def test_manual_pick_that_left_the_group_disqualifies(self, reencode_dir, monkeypatch):
        _policy(monkeypatch, "ledger")
        images, _ = _scan(reencode_dir)
        cache = HashCache(reencode_dir)
        cache.set_best_file(images[0]["group_id"], os.path.join(reencode_dir, "other.png"))

        _scan(reencode_dir)
        assert _reencode_rows(reencode_dir) == []

    def test_manual_pick_still_in_the_group_is_the_keeper(self, reencode_dir, monkeypatch):
        _policy(monkeypatch, "ledger")
        images, _ = _scan(reencode_dir)
        jpg = os.path.join(reencode_dir, "pic.jpg")
        HashCache(reencode_dir).set_best_file(images[0]["group_id"], jpg)

        _scan(reencode_dir)
        (row,) = _reencode_rows(reencode_dir)
        assert row["keeper"] == jpg
        assert row["duplicate"] == os.path.join(reencode_dir, "pic.png")
        assert row["evidence"]["keeper_reason"] == "Chosen manually"


class TestReencodeGuards:
    """Guards a real scan cannot easily reach, checked on the tier directly."""

    @pytest.fixture
    def scanned(self, reencode_dir, monkeypatch):
        from deduper.services.duplicate_finder import DuplicateFinder

        from .conftest import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

        _policy(monkeypatch, "ledger")
        _scan(reencode_dir)
        assert len(_reencode_rows(reencode_dir)) == 1, "premise: the group qualifies"
        cache = HashCache(reencode_dir)
        groups = {k: v for k, v in cache.get_cached_groups().items() if len(v) > 1}
        finder = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        return finder, cache, groups

    def test_guards_pass_unmodified(self, scanned):
        finder, cache, groups = scanned
        assert finder._auto_resolve_reencodes(groups, cache) is groups
        assert cache.count_auto_actions(applied=False, tier="reencode") == 1

    def test_video_member_disqualifies(self, scanned, tmp_path):
        finder, cache, groups = scanned
        (key, members), = groups.items()
        clip = os.path.join(str(cache.directory_path), "pic.mp4")
        with open(clip, "wb") as fh:
            fh.write(b"\0")
        finder._auto_resolve_reencodes({key: members + [clip]}, cache)
        assert cache.count_auto_actions(tier="reencode") == 0

    def test_missing_analysis_disqualifies(self, scanned):
        finder, cache, groups = scanned
        cache._db.execute("DELETE FROM group_analysis")
        cache._db.commit()
        finder._auto_resolve_reencodes(groups, cache)
        assert cache.count_auto_actions(tier="reencode") == 0

    def test_stale_analysis_version_disqualifies(self, scanned):
        import json

        finder, cache, groups = scanned
        for group_id, payload in cache._db.execute(
            "SELECT group_id, payload FROM group_analysis"
        ).fetchall():
            data = json.loads(payload)
            data["version"] = -1
            cache._db.execute(
                "UPDATE group_analysis SET payload=? WHERE group_id=?",
                (json.dumps(data), group_id),
            )
        cache._db.commit()
        finder._auto_resolve_reencodes(groups, cache)
        assert cache.count_auto_actions(tier="reencode") == 0

    def test_changed_member_disqualifies(self, scanned):
        """A member edited since it was hashed and analysed: neither its stored
        hash nor the group's analysis fingerprint describes it any more."""
        finder, cache, groups = scanned
        jpg = os.path.join(str(cache.directory_path), "pic.jpg")
        os.utime(jpg, (1_000_000_000, 1_000_000_000))
        finder._auto_resolve_reencodes(groups, cache)
        assert cache.count_auto_actions(tier="reencode") == 0

    def test_modification_flag_disqualifies(self, scanned):
        import json

        finder, cache, groups = scanned
        for group_id, payload in cache._db.execute(
            "SELECT group_id, payload FROM group_analysis"
        ).fetchall():
            data = json.loads(payload)
            first = next(iter(data["members"].values()))
            first["modifications"] = [{"kind": "overlay", "label": "Overlay"}]
            cache._db.execute(
                "UPDATE group_analysis SET payload=? WHERE group_id=?",
                (json.dumps(data), group_id),
            )
        cache._db.commit()
        finder._auto_resolve_reencodes(groups, cache)
        assert cache.count_auto_actions(tier="reencode") == 0

    def test_symlinked_member_disqualifies(self, scanned):
        finder, cache, groups = scanned
        jpg = os.path.join(str(cache.directory_path), "pic.jpg")
        os.remove(jpg)
        os.symlink("pic.png", jpg)
        finder._auto_resolve_reencodes(groups, cache)
        assert cache.count_auto_actions(tier="reencode") == 0
