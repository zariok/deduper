"""Tests for the HTTP layer, focused on the cached-results fast path."""

import json
import os

import pytest

from deduper.config import Config, TestingConfig
from deduper.services.duplicate_finder import DuplicateFinder

from .conftest import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, requires_ffmpeg


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A test client whose DATA_DIR is an empty temp directory.

    views.py reads ``Config.DATA_DIR`` directly rather than ``app.config``, so the
    class attribute is what has to be patched.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(Config, "DATA_DIR", data_dir)
    monkeypatch.setattr(TestingConfig, "DATA_DIR", data_dir)

    from deduper.app import create_app

    app = create_app("testing")  # 'testing' skips the background scanner
    app.config.update(TESTING=True)
    return app.test_client(), str(data_dir)


def populate(data_dir: str, media_dir: str, name: str = "photos") -> str:
    """Move a prepared fixture folder into the app's data directory."""
    import shutil

    target = os.path.join(data_dir, name)
    shutil.copytree(media_dir, target)
    return os.path.realpath(target)


class TestCachedResults:
    def test_reports_no_cache_before_a_scan(self, client, images_only_dir):
        http, data_dir = client
        populate(data_dir, images_only_dir)

        payload = http.get("/cached-results/photos").get_json()
        assert payload["cached"] is False

    def test_missing_folder_returns_404(self, client):
        http, _ = client
        assert http.get("/cached-results/does-not-exist").status_code == 404

    def test_returns_groups_after_a_scan(self, client, images_only_dir):
        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        response = http.get("/cached-results/photos")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["cached"] is True
        assert len(payload["duplicate_images"]) >= 1

    def test_groups_include_urls_and_metadata(self, client, images_only_dir):
        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        payload = http.get("/cached-results/photos").get_json()
        group = payload["duplicate_images"][0]
        best = group["best_file"]
        assert best["full"].startswith("/data/")
        assert best["thumb"].startswith("/thumb/")
        assert best["resolution"]["label"] != "Unknown"
        assert best["size"] > 0

    def test_symlinked_duplicates_are_filtered_out(self, client, images_only_dir):
        """The exact duplicate was replaced by a symlink and must not resurface."""
        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        payload = http.get("/cached-results/photos").get_json()
        for group in payload["duplicate_images"]:
            names = [os.path.basename(group["best_file"]["path"])]
            names += [os.path.basename(d["path"]) for d in group["duplicate_files"]]
            assert "img_a_copy.png" not in names


@requires_ffmpeg
class TestCachedResultsVideoMetadata:
    def test_reports_video_duration(self, client, media_dir):
        http, data_dir = client
        folder = populate(data_dir, media_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        payload = http.get("/cached-results/photos").get_json()
        assert payload["duplicate_videos"]
        best = payload["duplicate_videos"][0]["best_file"]
        assert best["duration"] > 0
        assert best["duration_formatted"].endswith("s")


    def test_page_load_runs_no_ffprobe(self, client, media_dir, count_ffprobe):
        """A page load must serve metadata from the cache, not re-probe every video.

        Re-probing here is what made loading a video-heavy folder stall.
        """
        from deduper.utils import media as media_module

        http, data_dir = client
        folder = populate(data_dir, media_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        # Simulate a fresh process serving the request
        media_module._probe_video_cached.cache_clear()
        media_module._resolve_media_resolution_cached.cache_clear()
        count_ffprobe.clear()

        payload = http.get("/cached-results/photos").get_json()
        assert payload["cached"] is True
        assert count_ffprobe == [], f"unexpected ffprobe calls: {count_ffprobe}"


    def test_metadata_probed_on_request_is_persisted(self, client, media_dir):
        """If a request does have to probe, the result must be written back.

        Otherwise every page load re-probes and the cache never warms.
        """
        import sqlite3

        from deduper.utils.hash_cache import HashCache, close_all_caches
        from deduper.utils import media as media_module

        http, data_dir = client
        folder = populate(data_dir, media_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        # Drop the persisted metadata so the request is forced to probe again
        close_all_caches()
        db = os.path.join(folder, HashCache.DB_FILENAME)
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM media_metadata")
            conn.commit()
        media_module._probe_video_cached.cache_clear()
        media_module._resolve_media_resolution_cached.cache_clear()

        http.get("/cached-results/photos")

        close_all_caches()
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM media_metadata").fetchone()[0]
        assert rows > 0, "probed metadata should be written back to the cache"


class TestLiveCacheStats:
    def test_reports_zero_for_an_unscanned_folder(self, client, images_only_dir):
        http, data_dir = client
        populate(data_dir, images_only_dir)
        payload = http.get("/cache/photos/live-stats").get_json()
        assert payload["total_cached_files"] == 0

    def test_reports_counts_after_a_scan(self, client, images_only_dir):
        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        payload = http.get("/cache/photos/live-stats").get_json()
        assert payload["total_cached_files"] > 0
        assert payload["duplicate_groups"] >= 1


class TestIndex:
    def test_renders_for_a_populated_data_dir(self, client, images_only_dir):
        """The folder names themselves are rendered client-side, so only the
        successful render is asserted here."""
        http, data_dir = client
        populate(data_dir, images_only_dir)
        response = http.get("/")
        assert response.status_code == 200
        assert b"<html" in response.data

    def test_creates_an_example_folder_when_empty(self, client):
        http, data_dir = client
        assert os.listdir(data_dir) == []
        assert http.get("/").status_code == 200
        assert os.listdir(data_dir), "an example folder should be created"
