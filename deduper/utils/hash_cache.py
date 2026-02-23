"""SQLite-based hash cache for duplicate detection. Replaces the legacy JSON .deduper format."""

import os
import json
import time
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .logging_config import get_logger
from .media import get_image_hash

logger = get_logger(__name__)

# ---- Connection pool (singleton registry) ------------------------------------
# Keyed by resolved directory path so the same HashCache is reused across
# requests, avoiding repeated sqlite3.connect() and schema validation overhead.
_cache_registry: dict[str, "HashCache"] = {}
_registry_lock = threading.Lock()


def get_hash_cache(directory_path: str) -> "HashCache":
    """Return a shared HashCache instance for *directory_path*, creating one if needed."""
    key = str(Path(directory_path).resolve())
    if key not in _cache_registry:
        with _registry_lock:
            if key not in _cache_registry:
                _cache_registry[key] = HashCache(directory_path)
    return _cache_registry[key]


def close_hash_cache(directory_path: str) -> None:
    """Close and remove a single cached connection so it doesn't hold file descriptors."""
    key = str(Path(directory_path).resolve())
    with _registry_lock:
        cache = _cache_registry.pop(key, None)
        if cache:
            try:
                if cache._conn:
                    cache._conn.close()
                    cache._conn = None
            except Exception:
                pass


def close_all_caches() -> None:
    """Close every pooled connection (call at shutdown)."""
    with _registry_lock:
        for cache in _cache_registry.values():
            try:
                if cache._conn:
                    cache._conn.close()
            except Exception:
                pass
        _cache_registry.clear()

# Legacy JSON filename (used by migration script to find files to migrate)
CACHE_FILENAME = ".deduper"
# SQLite database filename
DB_FILENAME = ".deduper.db"

CACHE_VERSION = "2.2"  # 2.2: Multi-hash (pHash+dHash); old single-hash values invalid


def _is_old_cache_format(data: dict[str, Any]) -> bool:
    """Check if cache uses old tuple format for file_stats."""
    for stats in data.get("file_stats", {}).values():
        if isinstance(stats, (list, tuple)) and len(stats) == 2:
            return True
    return False


class HashCache:
    """SQLite-based persistent hash cache for duplicate detection."""

    CACHE_FILENAME = CACHE_FILENAME
    CACHE_VERSION = CACHE_VERSION
    DB_FILENAME = DB_FILENAME

    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path).resolve()
        self.db_file = self.directory_path / self.DB_FILENAME
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        try:
            self._init_db()

        except sqlite3.DatabaseError as e:
            # Corrupt database file — delete and recreate from scratch
            logger.warning(
                f"Corrupt SQLite cache detected at {self.db_file}: {e}. "
                f"Deleting and recreating."
            )
            try:
                if self._conn:
                    self._conn.close()
                    self._conn = None
                if self.db_file.exists():
                    self.db_file.unlink()
            except OSError as remove_err:
                logger.error(f"Failed to remove corrupt DB {self.db_file}: {remove_err}")
            self._init_db()  # Retry with fresh file

    def __del__(self):
        """Close the SQLite connection when this object is garbage-collected.

        This is a safety net for HashCache instances evicted from the registry
        by _release_cache() — once all request threads drop their references
        the GC will reclaim the object and free the file descriptors.
        """
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass

    def _init_db(self) -> None:
        """Create or open SQLite database and ensure schema."""
        self._conn = sqlite3.connect(str(self.db_file), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS file_hashes (
                relative_path TEXT PRIMARY KEY,
                hash_value TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_file_hashes_hash ON file_hashes(hash_value);
            CREATE TABLE IF NOT EXISTS grouping_results (
                rep_path TEXT PRIMARY KEY,
                files TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS best_files (
                group_id TEXT PRIMARY KEY,
                best_file TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS groups (
                group_id TEXT PRIMARY KEY,
                files TEXT NOT NULL,
                processed INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            );
        """)
        self._conn.commit()
        # Ensure meta exists and check version for auto-invalidation
        cur = self._conn.execute("SELECT value FROM meta WHERE key='version'")
        row = cur.fetchone()
        if row is None:
            now = time.time()
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [("version", CACHE_VERSION), ("created", str(now)), ("last_updated", str(now))],
            )
            self._conn.commit()
        elif row[0] != CACHE_VERSION:
            # Version mismatch — cache was built with different extension mappings
            # or algorithm. Invalidate hashes and grouping but keep schema.
            old_version = row[0]
            logger.info(
                f"Cache version mismatch ({old_version} -> {CACHE_VERSION}) "
                f"in {self.directory_path}, invalidating stale data"
            )
            self._conn.execute("DELETE FROM file_hashes")
            self._conn.execute("DELETE FROM grouping_results")
            self._conn.execute("DELETE FROM best_files")
            self._conn.execute("DELETE FROM groups")
            now = time.time()
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [("version", CACHE_VERSION), ("last_updated", str(now))],
            )
            self._conn.commit()

    def _get_relative_path(self, file_path: str) -> str:
        try:
            return os.path.relpath(file_path, self.directory_path)
        except ValueError:
            return os.path.basename(file_path)

    def _get_absolute_path(self, relative_path: str) -> str:
        return os.path.join(self.directory_path, relative_path)

    def _get_file_stats(self, file_path: str) -> tuple[float, int]:
        try:
            st = os.stat(file_path)
            return st.st_mtime, st.st_size
        except OSError:
            return 0.0, 0

    def _is_file_unchanged(self, file_path: str) -> bool:
        mtime, size = self._get_file_stats(file_path)
        rel = self._get_relative_path(file_path)
        row = self._conn.execute(
            "SELECT mtime, size FROM file_hashes WHERE relative_path = ?", (rel,)
        ).fetchone()
        if not row:
            return False
        return row[0] == mtime and row[1] == size

    def has_cached_hash(self, relative_path: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM file_hashes WHERE relative_path = ? LIMIT 1", (relative_path,)
        ).fetchone()
        return row is not None

    def get_cached_hash_str(self, relative_path: str) -> str | None:
        row = self._conn.execute(
            "SELECT hash_value FROM file_hashes WHERE relative_path = ?", (relative_path,)
        ).fetchone()
        return row[0] if row else None

    # Maximum rows to write per lock acquisition.  Keeps the lock held for
    # only a few milliseconds so web-request threads can interleave.
    _BATCH_CHUNK_SIZE = 200

    def batch_update_hashes(self, updates: dict[str, dict[str, Any]]) -> None:
        """Batch insert/update hashes. updates: {relative_path: {hash: str, stats: {mtime, size}}}."""
        if not updates:
            return
        now = time.time()
        rows = []
        for rel, d in updates.items():
            h = d.get("hash", "")
            s = d.get("stats", {})
            mtime = s.get("mtime", 0.0)
            size = int(s.get("size", 0))
            rows.append((rel, h, mtime, size, now))

        # Write in chunks so the lock is released between each chunk, giving
        # web-request threads a chance to acquire it instead of blocking for
        # the entire (potentially thousands-of-rows) batch.
        for i in range(0, len(rows), self._BATCH_CHUNK_SIZE):
            chunk = rows[i : i + self._BATCH_CHUNK_SIZE]
            with self._lock:
                self._conn.executemany(
                    """INSERT OR REPLACE INTO file_hashes
                       (relative_path, hash_value, mtime, size, updated_at) VALUES (?,?,?,?,?)""",
                    chunk,
                )
                self._conn.commit()

        # Final metadata update
        with self._lock:
            self._conn.execute("UPDATE meta SET value=? WHERE key='last_updated'", (str(now),))
            self._conn.commit()

    def get_hash(self, file_path: str, video_extensions: set) -> Any | None:
        from .media import MultiHash
        if not os.path.exists(file_path):
            return None
        rel = self._get_relative_path(file_path)
        row = self._conn.execute(
            "SELECT hash_value, mtime, size FROM file_hashes WHERE relative_path = ?", (rel,)
        ).fetchone()
        if row:
            mtime, size = self._get_file_stats(file_path)
            if row[1] == mtime and row[2] == size:
                try:
                    return MultiHash.from_str(row[0])
                except Exception as e:
                    logger.warning(f"Error converting cached hash for {file_path}: {e}")
                    return None
        h = get_image_hash(file_path, tuple(video_extensions))
        if h is not None:
            mtime, size = self._get_file_stats(file_path)
            with self._lock:
                self._conn.execute(
                    """INSERT OR REPLACE INTO file_hashes
                       (relative_path, hash_value, mtime, size, updated_at) VALUES (?,?,?,?,?)""",
                    (rel, str(h), mtime, size, time.time()),
                )
                self._conn.execute("UPDATE meta SET value=? WHERE key='last_updated'", (str(time.time()),))
                self._conn.commit()
        return h

    def update_file_stats(self, file_path: str) -> None:
        rel = self._get_relative_path(file_path)
        if os.path.exists(file_path):
            mtime, size = self._get_file_stats(file_path)
            with self._lock:
                self._conn.execute(
                    "UPDATE file_hashes SET mtime=?, size=?, updated_at=? WHERE relative_path=?",
                    (mtime, size, time.time(), rel),
                )
                self._conn.commit()
        else:
            with self._lock:
                self._conn.execute("DELETE FROM file_hashes WHERE relative_path=?", (rel,))
                self._conn.commit()

    def cleanup_deleted_files(self, existing_files: set) -> None:
        existing_rel = {self._get_relative_path(p) for p in existing_files}
        cur = self._conn.execute("SELECT relative_path FROM file_hashes")
        to_del = [r[0] for r in cur.fetchall() if r[0] not in existing_rel]
        for rel in to_del:
            abs_path = self.directory_path / rel
            if abs_path.exists():
                continue
            thumb = abs_path.with_name(f"thumb-deduper.{abs_path.stem}.jpg")
            if thumb.exists():
                try:
                    thumb.unlink()
                except OSError as e:
                    logger.warning(f"Failed to remove thumbnail {thumb}: {e}")
        if to_del:
            del_rows = [(r,) for r in to_del]
            for i in range(0, len(del_rows), self._BATCH_CHUNK_SIZE):
                chunk = del_rows[i : i + self._BATCH_CHUNK_SIZE]
                with self._lock:
                    self._conn.executemany("DELETE FROM file_hashes WHERE relative_path=?", chunk)
                    self._conn.commit()
            logger.info(f"Cleaned up {len(to_del)} deleted files from cache")

    def get_cache_stats(self) -> dict[str, Any]:
        total = 0
        try:
            total = self._conn.execute("SELECT COUNT(*) FROM file_hashes").fetchone()[0]
        except Exception:
            pass
        size_mb = 0.0
        if self.db_file.exists():
            size_mb = self.db_file.stat().st_size / (1024 * 1024)
        created = 0.0
        last = 0.0
        for row in self._conn.execute("SELECT key, value FROM meta WHERE key IN ('created','last_updated')"):
            if row[0] == "created":
                created = float(row[1])
            else:
                last = float(row[1])
        total_groups = 0
        duplicate_groups = 0
        processed_groups = 0
        remaining = 0
        for row in self._conn.execute("SELECT rep_path, files FROM grouping_results"):
            total_groups += 1
            try:
                files = json.loads(row[1])
            except Exception:
                files = []
            if len(files) > 1:
                duplicate_groups += 1
            for r in files:
                p = self._get_absolute_path(r)
                if os.path.exists(p) and not os.path.islink(p):
                    remaining += 1
        for _ in self._conn.execute("SELECT 1 FROM groups WHERE processed=1"):
            processed_groups += 1
        return {
            "total_cached_files": total,
            "cache_size_mb": size_mb,
            "created": created,
            "last_updated": last,
            "total_groups": total_groups,
            "duplicate_groups": duplicate_groups,
            "processed_groups": processed_groups,
            "remaining_duplicates": remaining,
        }

    def save(self) -> None:
        with self._lock:
            self._conn.execute("UPDATE meta SET value=? WHERE key='last_updated'", (str(time.time()),))
            self._conn.commit()

    def force_save(self) -> None:
        self.save()

    def invalidate_cache(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM file_hashes")
            self._conn.execute("DELETE FROM grouping_results")
            self._conn.execute("DELETE FROM best_files")
            self._conn.execute("DELETE FROM groups")
            now = time.time()
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                [("version", CACHE_VERSION), ("created", str(now)), ("last_updated", str(now))],
            )
            self._conn.commit()
        logger.info("Cache invalidated")

    def clear_corrupted_cache(self) -> None:
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
            if self.db_file.exists():
                self.db_file.unlink()
        except OSError as e:
            logger.error(f"Error removing DB: {e}")
        self._init_db()
        logger.info("Created new cache")

    def set_best_file(self, group_id: str, best_file_path: str) -> None:
        rel = self._get_relative_path(best_file_path)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO best_files (group_id, best_file, updated_at) VALUES (?,?,?)",
                (group_id, rel, time.time()),
            )
            self._conn.commit()
        logger.debug(f"Set best file for group {group_id}: {rel}")

    def get_best_file(self, group_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT best_file FROM best_files WHERE group_id=?", (group_id,)
        ).fetchone()
        if row:
            return self._get_absolute_path(row[0])
        return None

    def get_group_files(self, group_id: str) -> list:
        row = self._conn.execute("SELECT files FROM groups WHERE group_id=?", (group_id,)).fetchone()
        if row:
            try:
                return [self._get_absolute_path(r) for r in json.loads(row[0])]
            except Exception:
                pass
        # Fallback: search grouping_results by building group_id from each group
        for r in self._conn.execute("SELECT rep_path, files FROM grouping_results"):
            try:
                files = json.loads(r[1])
            except Exception:
                continue
            if not files:
                continue
            abs_files = [self._get_absolute_path(f) for f in files]
            if self._generate_group_id_from_files(abs_files) == group_id:
                self.set_group_files(group_id, abs_files)
                return abs_files
        return []

    def set_group_files(self, group_id: str, file_paths: list) -> None:
        rels = [self._get_relative_path(p) for p in file_paths]
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO groups (group_id, files, processed, updated_at) VALUES (?,?,0,?)",
                (group_id, json.dumps(rels), time.time()),
            )
            self._conn.commit()

    def mark_group_processed(self, group_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE groups SET processed=1, updated_at=? WHERE group_id=?",
                (time.time(), group_id),
            )
            self._conn.commit()
        logger.debug(f"Marked group {group_id} as processed")

    def get_group_ids(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT group_id FROM groups").fetchall()]

    def get_cached_groups(self) -> dict:
        out = {}
        for row in self._conn.execute("SELECT rep_path, files FROM grouping_results"):
            try:
                files = json.loads(row[1])
            except Exception:
                continue
            abs_rep = self._get_absolute_path(row[0])
            abs_files = []
            for r in files:
                p = self._get_absolute_path(r)
                if not os.path.islink(p):
                    abs_files.append(p)
            if abs_files:
                out[abs_rep] = abs_files
        return out

    def set_cached_groups(self, groups: dict) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute("DELETE FROM grouping_results")
            for rep, paths in groups.items():
                rel_rep = self._get_relative_path(rep)
                rels = [self._get_relative_path(p) for p in paths]
                self._conn.execute(
                    "INSERT INTO grouping_results (rep_path, files, updated_at) VALUES (?,?,?)",
                    (rel_rep, json.dumps(rels), now),
                )
            self._conn.execute("UPDATE meta SET value=? WHERE key='last_updated'", (str(now),))
            self._conn.commit()
        logger.debug(f"Cached {len(groups)} groups")

    def get_grouping_timestamp(self) -> float:
        row = self._conn.execute("SELECT value FROM meta WHERE key='grouping_timestamp'").fetchone()
        return float(row[0]) if row else 0.0

    def set_grouping_timestamp(self, timestamp: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('grouping_timestamp', ?)",
                (str(timestamp),),
            )
            self._conn.commit()

    def is_grouping_stale(self, file_paths: set) -> bool:
        cached = set()
        for row in self._conn.execute("SELECT files FROM grouping_results"):
            try:
                for r in json.loads(row[0]):
                    p = self._get_absolute_path(r)
                    if not os.path.islink(p):
                        cached.add(p)
            except Exception:
                continue
        return cached != file_paths

    def invalidate_grouping_cache(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM grouping_results")
            self._conn.execute("DELETE FROM meta WHERE key='grouping_timestamp'")
            self._conn.commit()
        logger.debug("Invalidated grouping cache")

    def remove_file_from_groups(self, file_path: str) -> None:
        rel = self._get_relative_path(file_path)
        with self._lock:
            updated = {}
            for row in self._conn.execute("SELECT rep_path, files FROM grouping_results"):
                try:
                    files = [f for f in json.loads(row[1]) if f != rel]
                except Exception:
                    continue
                if files:
                    updated[row[0]] = files
            self._conn.execute("DELETE FROM grouping_results")
            for rep, files in updated.items():
                self._conn.execute(
                    "INSERT INTO grouping_results (rep_path, files, updated_at) VALUES (?,?,?)",
                    (rep, json.dumps(files), time.time()),
                )
            self._conn.commit()
        logger.debug(f"Removed {rel} from cached groups")

    def _generate_group_id_from_files(self, file_paths: list) -> str:
        rels = sorted({self._get_relative_path(p) for p in file_paths})
        return hashlib.md5("|".join(rels).encode()).hexdigest()[:8]

    @staticmethod
    def read_metadata(folder_path: str, folder_mtime: float | None = None) -> dict[str, Any]:
        """
        Read lightweight metadata from .deduper.db without a full HashCache.
        Returns: status ('pending'|'complete'|'stale'), last_scan_time, duplicate_count, file_count.
        """
        default: dict[str, Any] = {
            "status": "pending",
            "last_scan_time": 0.0,
            "duplicate_count": -1,
            "file_count": 0,
        }
        db = Path(folder_path) / DB_FILENAME
        if not db.exists():
            return default
        conn = None
        try:
            conn = sqlite3.connect(str(db))
            # Check cache version — if it doesn't match, the data will be
            # invalidated when HashCache is instantiated, so report as pending.
            ver_row = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
            if ver_row and ver_row[0] != CACHE_VERSION:
                return default
            row = conn.execute("SELECT value FROM meta WHERE key='last_updated'").fetchone()
            last_scan_time = float(row[0]) if row else 0.0
            duplicate_count = 0
            file_count = 0
            has_any = False
            for r in conn.execute("SELECT files FROM grouping_results"):
                has_any = True
                try:
                    files = json.loads(r[0])
                    file_count += len(files)
                    if len(files) > 1:
                        duplicate_count += 1
                except Exception:
                    pass
            if not has_any:
                return default
            if folder_mtime is None:
                try:
                    folder_mtime = os.path.getmtime(folder_path)
                except OSError:
                    folder_mtime = 0
            status = "stale" if folder_mtime > last_scan_time else "complete"
            return {
                "status": status,
                "last_scan_time": last_scan_time,
                "duplicate_count": duplicate_count,
                "file_count": file_count,
            }
        except Exception as e:
            logger.debug(f"Could not read cache metadata for {folder_path}: {e}")
            return default
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    # --- Migration: one-time JSON -> SQLite ---

    @classmethod
    def migrate_from_json(cls, json_path: Path) -> bool:
        """
        Migrate a single .deduper JSON file to .deduper.db.
        Call this BEFORE running the upgraded app. Removes json_path on success.
        Returns True if migration succeeded.
        """
        if not json_path.is_file() or json_path.name != CACHE_FILENAME:
            return False
        dir_path = json_path.parent
        db_path = dir_path / DB_FILENAME
        if db_path.exists():
            logger.warning(f"SQLite DB already exists, skipping migration: {db_path}")
            return False
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load {json_path}: {e}")
            return False
        if data.get("version") != "1.0":
            logger.warning(f"Unknown cache version in {json_path}, skipping")
            return False
        if _is_old_cache_format(data):
            logger.warning(f"Old cache format in {json_path}, skipping")
            return False
        conn = None
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS file_hashes (
                    relative_path TEXT PRIMARY KEY, hash_value TEXT NOT NULL,
                    mtime REAL NOT NULL, size INTEGER NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_file_hashes_hash ON file_hashes(hash_value);
                CREATE TABLE IF NOT EXISTS grouping_results (rep_path TEXT PRIMARY KEY, files TEXT NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS best_files (group_id TEXT PRIMARY KEY, best_file TEXT NOT NULL, updated_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS groups (group_id TEXT PRIMARY KEY, files TEXT NOT NULL, processed INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL);
            """)
            now = str(time.time())
            created = data.get("created", time.time())
            conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [("version", CACHE_VERSION), ("created", str(created)), ("last_updated", data.get("last_updated", now))],
            )
            hashes = data.get("hashes", {})
            file_stats = data.get("file_stats", {})
            for rel, h in hashes.items():
                st = file_stats.get(rel)
                if isinstance(st, dict):
                    mtime, size = st.get("mtime", 0), int(st.get("size", 0))
                else:
                    mtime, size = 0, 0
                conn.execute(
                    "INSERT INTO file_hashes (relative_path, hash_value, mtime, size, updated_at) VALUES (?,?,?,?,?)",
                    (rel, h, mtime, size, now),
                )
            for rep, files in data.get("grouping_results", {}).items():
                conn.execute(
                    "INSERT INTO grouping_results (rep_path, files, updated_at) VALUES (?,?,?)",
                    (rep, json.dumps(files if isinstance(files, list) else []), now),
                )
            for gid, best in data.get("best_files", {}).items():
                conn.execute(
                    "INSERT INTO best_files (group_id, best_file, updated_at) VALUES (?,?,?)",
                    (gid, best, now),
                )
            for gid, g in data.get("groups", {}).items():
                files = g.get("files", []) if isinstance(g, dict) else []
                proc = 1 if (isinstance(g, dict) and g.get("processed")) else 0
                conn.execute(
                    "INSERT INTO groups (group_id, files, processed, updated_at) VALUES (?,?,?,?)",
                    (gid, json.dumps(files), proc, now),
                )
            ts = data.get("grouping_timestamp")
            if ts is not None:
                conn.execute("INSERT INTO meta (key, value) VALUES ('grouping_timestamp', ?)", (str(ts),))
            conn.commit()
        except Exception as e:
            logger.error(f"Migration failed for {json_path}: {e}")
            if db_path.exists():
                try:
                    db_path.unlink()
                except OSError:
                    pass
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        logger.info(f"Migrated {json_path} -> {db_path}")
        return True
