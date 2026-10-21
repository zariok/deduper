"""Tests for symlink replacement safety in utils.helpers."""
import os

from deduper.utils.helpers import create_symlink_and_remove_duplicate


def _write(path, data=b"x"):
    path.write_bytes(data)
    return str(path)


def test_replaces_duplicate_with_relative_link(tmp_path):
    best = _write(tmp_path / "best.gif")
    dup = _write(tmp_path / "dup.gif")
    assert create_symlink_and_remove_duplicate(dup, best)
    assert os.path.islink(dup)
    assert os.readlink(dup) == "best.gif"
    assert os.path.exists(dup)


def test_links_across_directories(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    best = _write(tmp_path / "a" / "best.jpg")
    dup = _write(tmp_path / "b" / "dup.jpg")
    assert create_symlink_and_remove_duplicate(dup, best)
    assert os.path.exists(dup)
    assert os.path.realpath(dup) == os.path.realpath(best)


def test_refuses_same_path(tmp_path):
    f = _write(tmp_path / "a.gif")
    assert not create_symlink_and_remove_duplicate(f, f)
    assert os.path.isfile(f) and not os.path.islink(f)


def test_refuses_when_keeper_is_link_to_duplicate(tmp_path):
    dup = _write(tmp_path / "real.gif")
    best = str(tmp_path / "alias.gif")
    os.symlink("real.gif", best)
    assert not create_symlink_and_remove_duplicate(dup, best)
    assert os.path.isfile(dup) and not os.path.islink(dup)


def test_keeper_symlink_is_resolved_not_chained(tmp_path):
    real = _write(tmp_path / "real.gif")
    best = str(tmp_path / "alias.gif")
    os.symlink("real.gif", best)
    dup = _write(tmp_path / "dup.gif")
    assert create_symlink_and_remove_duplicate(dup, best)
    assert os.readlink(dup) == "real.gif"
    assert os.path.exists(dup)


def test_dangling_keeper_is_rejected_and_duplicate_kept(tmp_path):
    best = str(tmp_path / "best.gif")
    os.symlink("best.gif.tmp", best)
    dup = _write(tmp_path / "dup.gif", b"keep me")
    assert not create_symlink_and_remove_duplicate(dup, best)
    assert (tmp_path / "dup.gif").read_bytes() == b"keep me"


def test_failure_leaves_duplicate_intact(tmp_path, monkeypatch):
    best = _write(tmp_path / "best.gif")
    dup = _write(tmp_path / "dup.gif", b"keep me")

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(os, "symlink", boom)
    assert not create_symlink_and_remove_duplicate(dup, best)
    assert (tmp_path / "dup.gif").read_bytes() == b"keep me"
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".lnk")] == []
