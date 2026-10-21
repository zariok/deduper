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


@pytest.fixture
def idle_scanner(monkeypatch):
    """Install a BackgroundScanner singleton without starting its threads.

    'testing' config skips the scanner entirely, but the dropdown count lives on
    its FolderState, so the wiring is untestable without one.
    """
    from deduper.services import background_scanner as bg

    def _install(data_dir: str) -> "bg.BackgroundScanner":
        scanner = bg.BackgroundScanner(
            data_dir=data_dir,
            image_extensions=IMAGE_EXTENSIONS,
            video_extensions=VIDEO_EXTENSIONS,
        )
        scanner._discover_folders()
        monkeypatch.setattr(bg, "_background_scanner", scanner)
        return scanner

    return _install


class TestPendingAutoCount:
    """The ledger's pending count rides along with duplicate_count, so the UI
    can show "would auto-resolve: N" per folder from the status endpoints."""

    def _scanned(self, client, reencode_dir, idle_scanner, monkeypatch):
        monkeypatch.setattr(Config, "AUTO_RESOLVE_POLICY", "ledger")
        http, data_dir = client
        folder = populate(data_dir, reencode_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)
        return http, folder, idle_scanner(data_dir)

    def test_discovery_seeds_it_from_the_cache(self, client, reencode_dir, idle_scanner, monkeypatch):
        _, _, scanner = self._scanned(client, reencode_dir, idle_scanner, monkeypatch)
        assert scanner.get_folder_status("photos").pending_auto_count == 1

    def test_status_endpoints_report_it(self, client, reencode_dir, idle_scanner, monkeypatch):
        http, _, _ = self._scanned(client, reencode_dir, idle_scanner, monkeypatch)
        folders = http.get("/scanner/status").get_json()["folders"]
        assert folders["photos"]["pending_auto_count"] == 1
        status = http.get("/scanner/folder/photos/status").get_json()
        assert status["pending_auto_count"] == 1

    def test_refresh_after_management_recounts_it(self, client, reencode_dir, idle_scanner, monkeypatch):
        http, folder, scanner = self._scanned(client, reencode_dir, idle_scanner, monkeypatch)
        state = scanner.get_folder_status("photos")
        state.pending_auto_count = 99

        from deduper.routes.views import _refresh_folder_duplicate_count
        from deduper.utils.hash_cache import get_hash_cache

        _refresh_folder_duplicate_count("photos", get_hash_cache(folder))
        assert state.pending_auto_count == 1

    def test_a_folder_scanned_before_the_ledger_reads_zero(self, client, images_only_dir, idle_scanner):
        """A database without the auto_actions table must not break discovery."""
        import sqlite3

        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)
        from deduper.utils.hash_cache import close_all_caches

        close_all_caches()
        with sqlite3.connect(os.path.join(folder, ".deduper.db")) as conn:
            conn.execute("DROP TABLE auto_actions")
        scanner = idle_scanner(data_dir)
        state = scanner.get_folder_status("photos")
        assert state.pending_auto_count == 0
        assert state.duplicate_count >= 1


class TestDuplicateCountAfterManagement:
    """Removing duplicates has to move the dropdown count immediately.

    duplicate_count is otherwise only written by a full background scan, which is
    STABILITY_WAIT_SECONDS + MIN_RESCAN_INTERVAL_SECONDS away at best.
    """

    def _first_duplicate(self, http):
        payload = http.get("/cached-results/photos").get_json()
        for group in payload["duplicate_images"]:
            if group["duplicate_files"]:
                return group
        pytest.skip("fixture produced no group with a removable duplicate")

    def test_delete_updates_the_scanner_count(self, client, images_only_dir, idle_scanner):
        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)
        scanner = idle_scanner(data_dir)

        state = scanner.get_folder_status("photos")
        state.duplicate_count = 99  # stand in for a count from an older scan
        state.duplicate_count_time = 0

        group = self._first_duplicate(http)
        response = http.post(
            "/manage-duplicate",
            json={
                "action": "delete",
                "folder": "photos",
                "file_path": group["duplicate_files"][0]["path"],
                "best_file_path": group["best_file"]["path"],
            },
        )
        assert response.status_code == 200

        from deduper.utils.hash_cache import get_hash_cache

        expected = get_hash_cache(folder).count_duplicate_groups()
        assert state.duplicate_count == expected
        assert state.duplicate_count != 99
        assert state.duplicate_count_time > 0

    def test_recounting_restores_freshness_on_a_stale_folder(self, client, images_only_dir, idle_scanner):
        """A stale folder shows '(rescan needed)' rather than a count from before
        the change - but recounting after the user's edits must earn the number
        back, or the fix above would be erased 30s later by the change check."""
        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)
        scanner = idle_scanner(data_dir)

        from deduper.routes.socketio_events import _serialize_folder_state
        from deduper.services.background_scanner import ScanStatus

        # Backdate the scan so the folder's real mtimes read as a later change,
        # which is what the 30s check reacts to.
        state = scanner.get_folder_status("photos")
        state.last_scan_time = 0
        state.last_modified_time = 0
        state.duplicate_count_time = 0
        scanner._check_folders_for_changes()

        assert state.status is ScanStatus.STALE
        assert _serialize_folder_state(state)["duplicate_count_fresh"] is False

        group = self._first_duplicate(http)
        http.post(
            "/manage-duplicate",
            json={
                "action": "delete",
                "folder": "photos",
                "file_path": group["duplicate_files"][0]["path"],
                "best_file_path": group["best_file"]["path"],
            },
        )

        # Still stale — but the count was verified after the change, so it shows.
        assert state.status is ScanStatus.STALE
        assert _serialize_folder_state(state)["duplicate_count_fresh"] is True

    def test_a_count_predating_a_change_is_not_fresh(self, client, images_only_dir, idle_scanner):
        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)
        scanner = idle_scanner(data_dir)

        state = scanner.get_folder_status("photos")
        state.duplicate_count = 4
        state.duplicate_count_time = 100.0
        state.last_modified_time = 200.0  # folder changed after we last counted

        from deduper.routes.socketio_events import _serialize_folder_state

        assert _serialize_folder_state(state)["duplicate_count_fresh"] is False


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


class TestCachedResultsAgreeWithTheScan:
    def test_best_file_matches_the_scan_and_carries_quality(self, client, tmp_path):
        """The page-load path once took the first member as best. It must rank."""
        from PIL import Image

        from .test_quality import textured_image

        http, data_dir = client
        source = tmp_path / "src"
        source.mkdir()
        original = textured_image(320, 240)
        # Named so the upscale sorts first: that is the copy the old code showed
        original.resize((960, 720), Image.Resampling.BICUBIC).save(source / "a_big.png")
        original.save(source / "b_orig.png")
        folder = populate(data_dir, str(source))

        (scanned,) = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)[0]
        payload = http.get("/cached-results/photos").get_json()
        (cached,) = payload["duplicate_images"]

        assert os.path.basename(cached["best_file"]["path"]) == "b_orig.png"
        assert cached["best_file"]["path"] == scanned["best_file"]["path"]
        assert cached["best_reason"] == scanned["best_reason"]
        assert cached["best_file"]["quality"]["summary"]
        (dup,) = cached["duplicate_files"]
        assert dup["quality"]["detail"] < 0.6
        assert dup["modifications"] == []

    def test_exact_match_means_identical_bytes(self, client, images_only_dir, monkeypatch):
        """Same resolution and size is not enough; the scan proves identity, so must the page."""
        from deduper.services import duplicate_finder

        http, data_dir = client
        folder = populate(data_dir, images_only_dir)
        # Keep the twins on disk so the page has a same-size pair to judge
        monkeypatch.setattr(
            duplicate_finder.DuplicateFinder,
            "_process_exact_matches_automatically",
            lambda self, groups, cache, progress_callback=None: groups,
        )
        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(folder)

        payload = http.get("/cached-results/photos").get_json()
        (group,) = payload["duplicate_images"]
        flags = {os.path.basename(d["path"]): d["is_exact_match"] for d in group["duplicate_files"]}
        assert flags["img_a_small.png"] is False
        copy_flags = [v for k, v in flags.items() if k in ("img_a.png", "img_a_copy.png")]
        assert copy_flags == [True]


class TestImageThumbnails:
    def test_thumbnail_is_reduced_cached_and_invisible_to_the_scanner(self, client, tmp_path):
        """A results page must not pull full originals into 200px cards."""
        import io

        from PIL import Image

        from .test_quality import textured_image

        http, data_dir = client
        source = tmp_path / "src"
        source.mkdir()
        textured_image(1600, 1200).save(source / "big.jpg", quality=92)
        folder = populate(data_dir, str(source))

        response = http.get("/thumb/photos/big.jpg")
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.data)) as thumb:
            assert max(thumb.size) == 320
        assert len(response.data) < os.path.getsize(os.path.join(folder, "big.jpg")) / 10

        cached = os.path.join(folder, "thumb-deduper.big.jpg.jpg")
        assert os.path.exists(cached)
        stamp = os.path.getmtime(cached)
        assert http.get("/thumb/photos/big.jpg").status_code == 200
        assert os.path.getmtime(cached) == stamp, "second request must reuse the file"

        # The thumbnail beside the original is not itself a scan candidate
        finder = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        images, _ = finder.find_duplicates(folder)
        assert images == []
        from deduper.utils.hash_cache import HashCache

        assert HashCache(folder).get_file_hash_rows().keys() == {"big.jpg"}

    def test_unreadable_image_falls_back_to_the_original(self, client, tmp_path):
        http, data_dir = client
        source = tmp_path / "src"
        source.mkdir()
        (source / "broken.jpg").write_bytes(b"not a jpeg")
        populate(data_dir, str(source))
        response = http.get("/thumb/photos/broken.jpg")
        assert response.status_code == 200
        assert response.data == b"not a jpeg"

    def test_the_scrapers_own_thumbnail_is_preferred(self, client, tmp_path):
        """saviorpy writes thumb.<name> beside every download; serve it, build nothing."""
        from .test_quality import textured_image

        http, data_dir = client
        source = tmp_path / "src"
        source.mkdir()
        textured_image(1600, 1200).save(source / "big.jpg", quality=92)
        textured_image(200, 150, seed=7).save(source / "thumb.big.jpg", quality=80)
        folder = populate(data_dir, str(source))

        response = http.get("/thumb/photos/big.jpg")
        assert response.status_code == 200
        with open(os.path.join(folder, "thumb.big.jpg"), "rb") as handle:
            assert response.data == handle.read()
        assert not os.path.exists(os.path.join(folder, "thumb-deduper.big.jpg.jpg"))


class TestAutoResolveReview:
    """The review page for the auto-resolve ledger and its JSON twin."""

    def _ledger(self, data_dir, tmp_path, rows=1, tier="reencode"):
        """A folder with two real images and *rows* pending ledger rows."""
        from deduper.utils.hash_cache import get_hash_cache

        from .test_quality import textured_image

        source = tmp_path / "src"
        source.mkdir()
        textured_image(320, 240).save(source / "keep.png")
        textured_image(320, 240).save(source / "dup.jpg", quality=95)
        folder = populate(data_dir, str(source))
        cache = get_hash_cache(folder)
        evidence = {
            "phash_max": 1, "dhash_max": 0, "width": 64, "height": 48,
            "keeper_score": 2.0, "duplicate_score": 1.5,
            "keeper_reason": "Highest measured quality", "members": 2,
            "duplicate_size": 1500, "keeper_size": 2_200_000,
        }
        ids = [
            cache.record_auto_action(
                "g1", tier,
                os.path.join(folder, "dup.jpg"), os.path.join(folder, "keep.png"),
                evidence, applied=False,
            )
            for _ in range(rows)
        ]
        return folder, cache, ids

    def test_renders_with_no_rows(self, client, tmp_path):
        http, data_dir = client
        os.mkdir(os.path.join(data_dir, "photos"))
        response = http.get("/auto-resolve/photos")
        assert response.status_code == 200
        assert b"Auto-resolve: photos" in response.data
        assert b"Nothing is pending" in response.data
        # An unscanned folder must not get a database just for being viewed
        assert not os.path.exists(os.path.join(data_dir, "photos", ".deduper.db"))

    def test_renders_a_pending_reencode_row(self, client, tmp_path):
        http, data_dir = client
        self._ledger(data_dir, tmp_path)
        response = http.get("/auto-resolve/photos")
        assert response.status_code == 200
        page = response.get_data(as_text=True)
        assert "dup.jpg" in page and "keep.png" in page
        assert "tier-reencode" in page
        assert "64×48 · pHash ≤1 · dHash ≤0" in page
        assert "/data/photos/dup.jpg" in page
        assert "Highest measured quality" in page
        assert "Apply</button>" in page and "Dismiss</button>" in page
        assert "already resolved" not in page

    def test_a_row_whose_duplicate_is_gone_still_renders(self, client, tmp_path):
        http, data_dir = client
        folder, _, _ = self._ledger(data_dir, tmp_path)
        os.remove(os.path.join(folder, "dup.jpg"))
        response = http.get("/auto-resolve/photos")
        assert response.status_code == 200
        assert "already resolved" in response.get_data(as_text=True)

    def test_dismiss_removes_the_row_once(self, client, tmp_path):
        http, data_dir = client
        _, cache, (row_id,) = self._ledger(data_dir, tmp_path)
        url = "/auto-resolve/photos/dismiss"
        assert http.post(url, json={"id": row_id}).status_code == 200
        assert cache.count_auto_actions(applied=False) == 0
        assert http.post(url, json={"id": row_id}).status_code == 404
        assert http.post(url, json={"id": "nope"}).status_code == 400

    def test_dismiss_never_deletes_an_applied_row(self, client, tmp_path):
        http, data_dir = client
        folder, cache, _ = self._ledger(data_dir, tmp_path)
        applied_id = cache.record_auto_action(
            "g1", "identical", os.path.join(folder, "dup.jpg"),
            os.path.join(folder, "keep.png"), {}, applied=True,
        )
        response = http.post("/auto-resolve/photos/dismiss", json={"id": applied_id})
        assert response.status_code == 404
        assert cache.count_auto_actions(applied=True) == 1

    def test_json_paginates(self, client, tmp_path):
        http, data_dir = client
        self._ledger(data_dir, tmp_path, rows=150)
        first = http.get("/auto-resolve/photos.json").get_json()
        assert first["total"] == 150 and first["pages"] == 2
        assert len(first["rows"]) == 100
        second = http.get("/auto-resolve/photos.json?page=2").get_json()
        assert len(second["rows"]) == 50
        assert not {r["id"] for r in first["rows"]} & {r["id"] for r in second["rows"]}
        row = second["rows"][0]
        assert row["duplicate"]["path"] == "dup.jpg"
        assert row["keeper"]["path"] == "keep.png"
        assert first["pending_by_tier"] == {"reencode": 150}
        applied = http.get("/auto-resolve/photos.json?applied=1").get_json()
        assert applied["total"] == 0 and applied["rows"] == []

    def test_apply_records_a_manual_row(self, client, tmp_path):
        """The page's Apply button is a plain /manage-duplicate delete; that
        endpoint has to write the ledger itself."""
        http, data_dir = client
        folder, cache, _ = self._ledger(data_dir, tmp_path)
        response = http.post(
            "/manage-duplicate",
            json={
                "action": "delete", "folder": "photos", "group_id": "g1",
                "file_path": "dup.jpg", "best_file_path": "keep.png",
            },
        )
        assert response.status_code == 200
        assert os.path.islink(os.path.join(folder, "dup.jpg"))
        (row,) = cache.get_auto_actions(applied=True)
        assert row["tier"] == "manual" and row["group_id"] == "g1"
        assert os.path.basename(row["duplicate"]) == "dup.jpg"
        assert os.path.basename(row["keeper"]) == "keep.png"

    def test_a_failed_manual_replacement_leaves_no_row(self, client, tmp_path):
        http, data_dir = client
        _, cache, _ = self._ledger(data_dir, tmp_path)
        response = http.post(
            "/manage-duplicate",
            json={
                "action": "delete", "folder": "photos",
                "file_path": "dup.jpg", "best_file_path": "no-such-keeper.png",
            },
        )
        assert response.status_code == 500
        assert cache.count_auto_actions(applied=True) == 0

    @pytest.mark.parametrize("name", ["..", "..%2F..", "a..b", "%2E%2E"])
    def test_path_traversal_is_rejected(self, client, name):
        http, _ = client
        assert http.get(f"/auto-resolve/{name}").status_code == 400
        assert http.get(f"/auto-resolve/{name}.json").status_code == 400
        assert http.post(f"/auto-resolve/{name}/dismiss", json={"id": 1}).status_code == 400

    def test_missing_folder_is_404(self, client):
        http, _ = client
        assert http.get("/auto-resolve/nope").status_code == 404

    def test_index_labels_use_the_pending_count(self, client):
        http, _ = client
        page = http.get("/").get_data(as_text=True)
        assert "pending_auto_count" in page
        assert "Would auto-resolve: " in page
        assert 'id="autoResolveLink"' in page
