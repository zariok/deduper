# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deduper is a Python Flask web application for finding and managing duplicate images and videos. It uses perceptual hashing to detect similar media files and provides a web interface for managing duplicates by creating symlinks to the "best" version of each file.

**IMPORTANT**: This tool is DESTRUCTIVE - it deletes duplicate files and replaces them with symlinks. Operations should be carefully tested.

## Development Commands

### Setup and Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt   # for the test suite
```

### Running the Application
```bash
# A SECRET_KEY is required; without one the app prints an error and exits
SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')" python3 -m deduper

# Or, for local work only, fall back to an insecure dev key
DEDUPER_DEV=true python3 -m deduper

# Access at http://localhost:5000
```

### Testing
```bash
pytest                    # full suite, with coverage (see addopts in pyproject.toml)
pytest --no-cov           # faster, no coverage report
pytest -k "not Video"     # skip everything requiring ffmpeg
```

The suite generates real images and ffmpeg videos and scans them end to end.
Video tests skip automatically when `ffmpeg`/`ffprobe` are absent. Fixtures are
built once per session; the whole run takes a few seconds.

### Code Quality Tools
```bash
isort deduper/
mypy deduper/
```

**Do not run `black deduper/` as a drive-by.** The codebase does not conform -
Black reformats most files - so running it buries whatever change you are making
in hundreds of unrelated lines. Match the surrounding style, or reformat
deliberately as its own commit.

### Configuration
Environment variables (see README.md for the full list):
- `SECRET_KEY`: **required** unless `DEDUPER_DEV=true` or `FLASK_ENV=development`
- `DEDUPER_PORT` (default 5000), `DEDUPER_HOST` (default 127.0.0.1)
- `DEDUPER_DATA_DIR` (default ./data)
- `DEDUPER_LOG_LEVEL` (default INFO)

## Architecture

### Core Components

1. **Duplicate Detection Pipeline** (`deduper/services/duplicate_finder.py`)
   - Main class: `DuplicateFinder`
   - Hashes with `MultiHash` (pHash + dHash); videos are hashed via a thumbnail
     extracted at the 1s mark
   - Groups files with a BK-tree, clustered by disjoint-set union
   - Incremental grouping so unchanged files are not reprocessed
   - Auto-eliminates exact matches (same hash, resolution and file size) in a
     single pass, skipped entirely when nothing is new or changed
   - Hashing runs on a module-level `ProcessPoolExecutor` using the **spawn**
     context - forking a process whose threads hold locks deadlocks the children.
     `shutdown_process_pool()` is registered for exit
   - Walks with `os.scandir`, so the symlink test reuses the directory read
   - Resolves `folder_path` on entry, because `HashCache` resolves its own
     directory and mismatched paths produce unusable `../../..` cache keys
   - Yields the GIL periodically during clustering so HTTP threads stay responsive

2. **Hash Cache** (`deduper/utils/hash_cache.py`)
   - Class: `HashCache`, **SQLite-backed**
   - `.deduper.db` per scanned directory. `.deduper` is the *legacy JSON* name,
     kept only so `scripts/migrate_deduper_to_sqlite.py` can find files to convert
   - Tables: `meta`, `file_hashes`, `grouping_results`, `best_files`, `groups`,
     `media_metadata`
   - Instances are **pooled** by resolved path via `get_hash_cache()`. Use that
     rather than constructing `HashCache` directly, and call `close_hash_cache()`
     / `close_all_caches()` to release file descriptors
   - `CACHE_VERSION` mismatch wipes hashes, groupings, best files and groups
   - `read_metadata()` is a static, connection-free reader for the scanner

3. **Media Handling** (`deduper/utils/media.py`)
   - `MultiHash.__sub__` is `max(phash_distance, dhash_distance)`: a pair counts
     as similar only when **both** hashes agree
   - One `ffprobe` call returns width, height and duration together
   - Probe memos are keyed on `(path, mtime, size)` so an in-place edit invalidates

4. **Flask Web Interface** (`deduper/routes/views.py`, `deduper/routes/socketio_events.py`)
   - Blueprint `main`, plus Socket.IO events for live progress and scanner status
   - `socketio` is initialised in `app.py` with `async_mode="threading"`
   - Key routes: `/`, `/scan/<folder>`, `/cached-results/<folder>`,
     `/manage-duplicate`, `/data/<filename>`, `/thumb/<filename>`,
     `/scanner/status`, `/scanner/folder/<folder>/status`

5. **Background Scanner** (`deduper/services/background_scanner.py`)
   - Class: `BackgroundScanner`, daemon thread started at app startup
   - Pre-scans folders so results are cached before the user opens them
   - Waits 5 minutes after a folder stops changing before rescanning; minimum
     10 minutes between rescans of the same folder
   - Global singleton via `get_background_scanner()`

### Key Technical Details

- **Symlink Strategy**: duplicates are replaced with relative symlinks to the group's best file
- **Best File Selection**: images by highest resolution; videos by resolution, then duration, then size
- **Thumbnails**: `thumb-deduper.<name>.jpg`, extracted at the 1s mark
- **Media metadata**: resolution and duration live in the `media_metadata` table
  and are reused while mtime and size are unchanged. A warm rescan or page load of
  a video folder should spawn **zero** ffprobe processes
- **Group IDs**: `md5` of the group's sorted relative paths - derived from paths
  only, never hashes
- **File Paths**: relative internally, absolute for operations

### Package Structure

```
deduper/
├── __main__.py          # Entry point
├── app.py               # Flask app factory, Socket.IO, scanner, pool shutdown
├── config.py            # Config classes (Development, Production, Testing)
├── routes/
│   ├── views.py             # HTTP routes
│   └── socketio_events.py   # WebSocket events
├── services/
│   ├── duplicate_finder.py  # Detection pipeline
│   └── background_scanner.py
└── utils/
    ├── bktree.py, hash_cache.py, helpers.py
    ├── logging_config.py, media.py, metrics.py, setup.py

scripts/migrate_deduper_to_sqlite.py   # one-time JSON -> SQLite migration

tests/
├── conftest.py                # Fixtures: media corpora, ffprobe counters
├── test_duplicate_finder.py   # Scan pipeline, symlinking, incremental grouping
├── test_hash_cache.py         # SQLite round-trip, pooling, groups
├── test_media.py              # Hashing, probing, MultiHash
├── test_performance_paths.py  # Guards for the optimised paths
└── test_views.py              # HTTP endpoints
```

### Dependencies

Flask, flask-socketio, Pillow, imagehash, and FFmpeg (external).

## Development Guidelines

### Working with the Cache

- Obtain instances with `get_hash_cache(path)`, not `HashCache(path)` - the pool
  avoids reconnecting and re-validating the schema per request
- Close connections in tests and long-lived processes (`close_all_caches()`),
  otherwise descriptors accumulate and a pooled instance can outlive its directory
- `get_cached_groups()` drops both symlinks **and** paths that no longer exist.
  `islink` alone is False for a missing path, which once kept deleted files in
  their group forever
- `media_metadata` rows carry their own mtime/size rather than reusing
  `file_hashes`. Keep it that way: sharing the stamp would let a metadata read on
  a read-only path mark a stale *hash* as current
- Schema additions are `CREATE TABLE IF NOT EXISTS`, so they apply to existing
  databases without a migration

### File Operations Safety

- Update the cache immediately after any deletion
- Prefer relative symlinks
- `os.path.exists()` follows symlinks; use `os.path.islink()` to tell them apart,
  and test both when you mean "a real, present file"

### Testing Duplicate Detection

Scanning is destructive, so fixtures hand every test a throwaway copy. The
behaviours worth re-checking are already encoded - run them rather than reasoning
about them:
- Scanning twice produces identical groups and group IDs
- An unchanged rescan runs zero exact-match passes and zero ffprobe calls
- Exact matches become symlinks; similar-but-not-identical files do not
- A new near-duplicate joins an existing group, including one whose only member
  was previously unique

### Changing the Hash Algorithm

Nothing in a stored hash reveals which algorithm produced it, so `CACHE_VERSION`
is the only signal. Bump it if you change how hashes are produced; `_init_db`
then clears the stale tables, and `read_metadata` reports affected folders as
unscanned so the scanner re-queues them.

Note: `Image.draft("L", (32, 32))` was evaluated as a ~2.4x speedup for JPEG
hashing and **rejected**. It shifts hashes of low-detail images (screenshots,
gradients, flat graphics) by up to 12 bits against a matching threshold of 5,
flipping grouping verdicts on half of such pairs. Raising the draft target does
not fix it.

### Performance Invariants

Guarded by `tests/test_performance_paths.py`; breaking these costs speed or
correctness quietly:
- Hashes are compared as packed ints via `_packed_distance`, never
  `MultiHash.__sub__`, in clustering. The packed form must stay **exactly**
  equivalent - taking only the pHash distance would widen what counts as a
  duplicate and cause wrong deletions
- Warm rescans and page loads spawn no ffprobe processes
- Images are decoded once per hash, not twice
- The background scanner never holds `self._lock` across filesystem I/O; every
  status endpoint contends on that lock

### Code Style

- Line length 88, isort with the black profile
- **The codebase is not Black-clean.** Match surrounding style; do not reformat
  as a side effect of another change
- Type hints required (mypy strict mode)
- Requires Python 3.10+: runtime `X | Y` annotations and `int.bit_count()`
- Comprehensive logging at DEBUG, INFO, WARNING, ERROR levels
