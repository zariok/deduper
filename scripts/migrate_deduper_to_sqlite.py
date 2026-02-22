#!/usr/bin/env python3
"""
One-time migration: move existing .deduper (JSON) files to .deduper.db (SQLite)
and remove the .deduper files.

Run this BEFORE starting the upgraded deduper that uses SQLite.

Usage:
  python scripts/migrate_deduper_to_sqlite.py [--root PATH] [--dry-run]

  --root   Root directory to search for .deduper files (default: DEDUPER_DATA_DIR or ./data)
  --dry-run  Only report what would be migrated, do not write or delete.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path so deduper can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deduper.utils.hash_cache import HashCache, CACHE_FILENAME


def find_deduper_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if CACHE_FILENAME in filenames:
            p = Path(dirpath) / CACHE_FILENAME
            if p.is_file():
                out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate .deduper JSON to .deduper.db SQLite and remove .deduper")
    ap.add_argument("--root", type=Path, default=None, help="Root to search (default: DEDUPER_DATA_DIR or ./data)")
    ap.add_argument("--dry-run", action="store_true", help="Only report, do not migrate or delete")
    args = ap.parse_args()

    root = args.root
    if root is None:
        root = Path(os.environ.get("DEDUPER_DATA_DIR", _PROJECT_ROOT / "data"))
    root = root.resolve()
    if not root.is_dir():
        print(f"Root does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    candidates = find_deduper_files(root)
    if not candidates:
        print(f"No .deduper files found under {root}")
        return 0

    print(f"Found {len(candidates)} .deduper file(s) under {root}")
    if args.dry_run:
        for p in candidates:
            print(f"  would migrate: {p}")
        return 0

    ok = 0
    for p in candidates:
        if HashCache.migrate_from_json(p):
            try:
                p.unlink()
                print(f"  migrated and removed: {p}")
                ok += 1
            except OSError as e:
                print(f"  migrated {p} but could not remove: {e}", file=sys.stderr)
        else:
            print(f"  skipped (already migrated or invalid): {p}", file=sys.stderr)

    print(f"Done. Migrated and removed {ok} of {len(candidates)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
