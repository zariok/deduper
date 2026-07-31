"""Shared fixtures for the deduper test suite.

The environment is prepared before ``deduper`` is imported: Config reads several
values at class-definition time, so setting them later has no effect.
"""

import os
import shutil
import subprocess
import tempfile

import pytest

os.environ.setdefault("DEDUPER_DEV", "true")
# Point the data directory somewhere disposable before deduper is imported. Otherwise
# Config.DATA_DIR defaults to ./data and the background scanner's file logger writes
# into the working copy - and into the developer's real media folder.
os.environ.setdefault(
    "DEDUPER_DATA_DIR", tempfile.mkdtemp(prefix="deduper_test_data_")
)

from PIL import Image  # noqa: E402

from deduper.utils import media  # noqa: E402
from deduper.utils.hash_cache import close_all_caches  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _ffmpeg_available() -> bool:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            return False
    return True


requires_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg and ffprobe are required for video fixtures",
)


def gradient_image(width: int, height: int, seed: int) -> Image.Image:
    """Build a deterministic gradient, distinct per seed, for stable perceptual hashes."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                (x * 7 + seed) % 256,
                (y * 5 + seed) % 256,
                ((x + y) * 3 + seed) % 256,
            )
    return image


@pytest.fixture(scope="session")
def sample_media(tmp_path_factory) -> dict[str, str]:
    """Build the media fixtures once per session; tests copy them per-test.

    Generating video files with ffmpeg is slow, so this is deliberately shared.
    """
    source = tmp_path_factory.mktemp("sample_media_source")

    base = gradient_image(240, 240, 11)
    base_path = source / "img_a.png"
    base.save(base_path)

    # Byte-identical copy -> same hash, resolution and size, so an exact match
    shutil.copy(base_path, source / "img_a_copy.png")
    # Downscaled -> near-identical hash but a different resolution, so a similar match
    base.resize((120, 120)).save(source / "img_a_small.png")
    # Unrelated content -> must never join the group above
    gradient_image(200, 200, 200).save(source / "img_b.png")

    files = {
        "img_a": str(base_path),
        "img_a_copy": str(source / "img_a_copy.png"),
        "img_a_small": str(source / "img_a_small.png"),
        "img_b": str(source / "img_b.png"),
    }

    if _ffmpeg_available():
        hi = source / "vid_hi.mp4"
        lo = source / "vid_lo.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=duration=3:size=640x480:rate=10",
             "-pix_fmt", "yuv420p", str(hi)],
            check=True,
        )
        # Re-encode smaller so the 1s thumbnail hashes close to the original's
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(hi),
             "-vf", "scale=320:240", "-pix_fmt", "yuv420p", str(lo)],
            check=True,
        )
        files["vid_hi"] = str(hi)
        files["vid_lo"] = str(lo)

    return files


@pytest.fixture
def media_dir(tmp_path, sample_media) -> str:
    """A fresh scan folder per test, since scanning mutates it destructively.

    ``realpath`` matters: HashCache resolves its directory, and on macOS the temp
    root is reached through a symlink (/var -> /private/var).
    """
    folder = tmp_path / "photos"
    folder.mkdir()
    for path in sample_media.values():
        shutil.copy(path, folder / os.path.basename(path))
    return os.path.realpath(folder)


@pytest.fixture
def images_only_dir(tmp_path, sample_media) -> str:
    """A scan folder holding only images, for tests that must not touch ffmpeg."""
    folder = tmp_path / "images"
    folder.mkdir()
    for key in ("img_a", "img_a_copy", "img_a_small", "img_b"):
        path = sample_media[key]
        shutil.copy(path, folder / os.path.basename(path))
    return os.path.realpath(folder)


@pytest.fixture(autouse=True)
def clear_media_caches():
    """Drop the module-level lru_caches around every test.

    They are process-global, so without this a later test would silently read
    values probed by an earlier one and metadata assertions would not mean anything.
    """
    media._resolve_media_resolution_cached.cache_clear()
    yield
    media._resolve_media_resolution_cached.cache_clear()


@pytest.fixture(autouse=True)
def close_sqlite_caches():
    """Close pooled SQLite connections after every test.

    HashCache instances are pooled per directory by get_hash_cache(), so without
    this the pool hands a later test a connection onto a deleted tmp directory,
    and open file descriptors accumulate across the run.
    """
    yield
    close_all_caches()


@pytest.fixture
def count_ffprobe(monkeypatch):
    """Count ffprobe invocations made through the media module."""
    calls: list[list[str]] = []
    real_run = subprocess.run

    def counting_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "ffprobe" and "-version" not in cmd:
            calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(media.subprocess, "run", counting_run)
    return calls


@pytest.fixture
def count_exact_match_passes(monkeypatch):
    """Count calls to the exact-match elimination pass."""
    from deduper.services.duplicate_finder import DuplicateFinder

    calls: list[int] = []
    real = DuplicateFinder._process_exact_matches_automatically

    def counting(self, groups, cache, progress_callback=None):
        calls.append(len(groups))
        return real(self, groups, cache, progress_callback)

    monkeypatch.setattr(
        DuplicateFinder, "_process_exact_matches_automatically", counting
    )
    return calls


def group_basenames(group: dict) -> list[str]:
    """Flatten a result group to sorted basenames for readable assertions."""
    members = [group["best_file"]["path"]]
    members += [d["path"] for d in group["duplicate_files"]]
    return sorted(os.path.basename(m) for m in members)
