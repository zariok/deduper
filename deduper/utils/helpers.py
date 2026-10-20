from __future__ import annotations

import os
from pathlib import Path
# No typing imports needed - using modern Python 3.10+ syntax

from .logging_config import get_logger

logger = get_logger(__name__)

# Python 3.10+ supports | for union types
PathLike = str | os.PathLike[str]


def get_file_size(file_path: PathLike) -> int:
    """
    Return the size of the supplied path in bytes.

    If the file cannot be read, returns 0 instead of raising.
    """
    try:
        return Path(file_path).stat().st_size
    except OSError as exc:
        logger.debug("Unable to read size for %s: %s", file_path, exc)
        return 0


def format_file_size(num_bytes: int) -> str:
    """
    Produce a human-readable representation of a byte count.

    Examples:
        0 -> "0 B"
        1024 -> "1.0 KB"
    """
    if num_bytes <= 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    unit_index = 0

    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1

    return f"{value:.1f} {units[unit_index]}"


def find_symlinks_pointing_to(target_file_path: str, search_directory: str) -> list[str]:
    """
    Find all symlinks in a directory that point to the target file.
    
    Args:
        target_file_path: The absolute path to the target file
        search_directory: The directory to search for symlinks
        
    Returns:
        List of absolute paths to symlinks that point to the target file
    """
    symlinks = []
    target_file_path = os.path.abspath(target_file_path)
    
    try:
        for root, dirs, files in os.walk(search_directory):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.islink(file_path):
                    try:
                        link_target = os.readlink(file_path)
                        # Convert to absolute path for comparison
                        if not os.path.isabs(link_target):
                            link_target = os.path.join(os.path.dirname(file_path), link_target)
                        link_target = os.path.abspath(link_target)
                        
                        if link_target == target_file_path:
                            symlinks.append(file_path)
                    except OSError:
                        # Skip broken symlinks
                        continue
    except OSError as e:
        logger.warning(f"Error searching for symlinks in {search_directory}: {e}")
    
    return symlinks


def create_symlink_and_remove_duplicate(duplicate_path: str, best_file_path: str) -> bool:
    """
    Replace a duplicate file with a relative symlink to the best file.

    The swap is atomic (a temporary symlink is renamed over the duplicate), so a
    failure never leaves the duplicate deleted without a link in its place. The
    link always targets the keeper's real path rather than another symlink, and
    the call refuses when both paths resolve to the same file, so it cannot
    produce dangling links, chains, or loops.

    Args:
        duplicate_path: Path to the duplicate file to be replaced
        best_file_path: Path to the best file to link to

    Returns:
        True if successful, False otherwise
    """
    tmp_link = None
    try:
        if not os.path.exists(duplicate_path):
            logger.warning(f"Duplicate file not found: {duplicate_path}")
            return False

        if not os.path.exists(best_file_path):
            logger.warning(f"Best file not found: {best_file_path}")
            return False

        if os.path.isdir(duplicate_path) and not os.path.islink(duplicate_path):
            logger.warning(f"Refusing to replace a directory: {duplicate_path}")
            return False

        # Resolve the keeper so we never link to a link. The duplicate is resolved
        # only for the identity check: replacing a link is fine, but replacing
        # the real file the keeper points at would leave a loop.
        real_best = os.path.realpath(best_file_path)
        if real_best == os.path.realpath(duplicate_path):
            logger.warning(f"Duplicate and best file are the same file: {duplicate_path}")
            return False
        if not os.path.isfile(real_best):
            logger.warning(f"Best file does not resolve to a regular file: {best_file_path}")
            return False

        duplicate_dir = os.path.dirname(os.path.abspath(duplicate_path))
        try:
            link_target = os.path.relpath(real_best, os.path.realpath(duplicate_dir))
        except ValueError:
            # Different drives on Windows: relative paths are impossible
            link_target = real_best

        # Build the link beside the duplicate, then rename it into place
        tmp_link = os.path.join(duplicate_dir, f".{os.path.basename(duplicate_path)}.{os.getpid()}.lnk")
        if os.path.lexists(tmp_link):
            os.remove(tmp_link)
        os.symlink(link_target, tmp_link)
        os.replace(tmp_link, duplicate_path)
        tmp_link = None

        # Also remove deduper thumbnails if they exist
        from .media import legacy_thumbnail_path_for, thumbnail_path_for

        for thumb in (thumbnail_path_for(duplicate_path), legacy_thumbnail_path_for(duplicate_path)):
            if thumb.exists():
                thumb.unlink()

        logger.info(f"Successfully replaced {duplicate_path} with symlink to {best_file_path}")
        return True

    except OSError as e:
        logger.error(f"Failed to create symlink for {duplicate_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating symlink for {duplicate_path}: {e}")
        return False
    finally:
        if tmp_link is not None:
            try:
                os.remove(tmp_link)
            except OSError:
                pass


__all__ = ["format_file_size", "get_file_size", "create_symlink_and_remove_duplicate", "find_symlinks_pointing_to"]