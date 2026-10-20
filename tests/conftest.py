"""Shared fixtures for the deduper test suite.

The environment is prepared before ``deduper`` is imported: Config reads several
values at class-definition time, so setting them later has no effect.
"""

import gc
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

# Mirror deduper/config.py: .gif belongs to VIDEO_EXTENSIONS there, because an
# animated gif is a clip and goes through the ffmpeg path. Classifying it as an
# image here meant the suite exercised a rule the app does not have.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".gif"}


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


@pytest.fixture(scope="session")
def signature_media(tmp_path_factory) -> dict[str, str]:
    """Clips exercising the multi-frame signature, built once per session.

    ``testsrc`` carries a burned-in timecode, so every second looks different -
    which is what makes a trim or a shared opening detectable at all.
    """
    if not _ffmpeg_available():
        pytest.skip("ffmpeg is required for video signature fixtures")

    src = tmp_path_factory.mktemp("signature_media")

    def ff(*args: str) -> None:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)

    full = src / "full.mp4"
    ff("-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=15", "-pix_fmt", "yuv420p", str(full))

    # Same footage as a gif: lower resolution, fewer frames, 256 colours
    ff("-i", str(full), "-vf",
       "fps=8,scale=240:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
       str(src / "full.gif"))
    # Trims: the end trim keeps t=0, the start trim moves every absolute offset
    ff("-i", str(full), "-t", "20", "-c", "copy", str(src / "trim_end.mp4"))
    ff("-ss", "10", "-i", str(full), "-c", "copy", str(src / "trim_start.mp4"))
    # Under a second — unreachable by the old fixed 1s seek
    ff("-f", "lavfi", "-i", "testsrc=duration=0.6:size=320x180:rate=15", "-pix_fmt", "yuv420p",
       str(src / "brief.mp4"))
    ff("-i", str(src / "brief.mp4"), "-vf", "scale=160:-1", str(src / "brief.gif"))
    # Two clips sharing an opening then diverging: distinct footage that a
    # single frame at t=1s cannot tell apart
    intro = src / "intro.mp4"
    ff("-f", "lavfi", "-i", "color=c=navy:size=640x360:duration=8:rate=15", "-pix_fmt", "yuv420p", str(intro))
    for tag, pattern in (("share_a", "smptebars"), ("share_b", "rgbtestsrc")):
        body = src / f"{tag}_body.mp4"
        ff("-f", "lavfi", "-i", f"{pattern}=duration=22:size=640x360:rate=15", "-pix_fmt", "yuv420p", str(body))
        listing = src / f"{tag}.txt"
        listing.write_text(f"file '{intro}'\nfile '{body}'\n")
        ff("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(src / f"{tag}.mp4"))

    # Keyed by full name, not stem: full.mp4 and full.gif share a stem, and the
    # pair being present together is the whole point of the fixture.
    return {
        p.name: str(p)
        for p in src.iterdir()
        if p.suffix in {".mp4", ".gif"} and not p.stem.endswith("_body") and p.stem != "intro"
    }


@pytest.fixture
def signature_dir(tmp_path, signature_media) -> str:
    """A throwaway copy of the signature clips; scanning mutates them."""
    folder = tmp_path / "clips"
    folder.mkdir()
    for path in signature_media.values():
        shutil.copy(path, folder / os.path.basename(path))
    return os.path.realpath(folder)


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
    """Close SQLite connections after every test.

    Two separate leaks to contain. Pooled instances (get_hash_cache) outlive the
    test and would hand a later one a connection onto a deleted tmp directory.
    Instances built directly with HashCache() are not in the pool at all, so they
    are only closed by __del__ - collecting deterministically keeps a genuine
    descriptor leak visible instead of drowning it in ResourceWarnings.
    """
    yield
    close_all_caches()
    gc.collect()


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
