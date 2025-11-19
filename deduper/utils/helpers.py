from __future__ import annotations

import os
from pathlib import Path
from typing import Union, List

from .logging_config import get_logger

logger = get_logger(__name__)

PathLike = Union[str, os.PathLike]


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


def find_symlinks_pointing_to(target_file_path: str, search_directory: str) -> List[str]:
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
    Create a symlink to the best file and remove the duplicate file.
    
    Args:
        duplicate_path: Path to the duplicate file to be replaced
        best_file_path: Path to the best file to link to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(duplicate_path):
            logger.warning(f"Duplicate file not found: {duplicate_path}")
            return False
            
        if not os.path.exists(best_file_path):
            logger.warning(f"Best file not found: {best_file_path}")
            return False
        
        # Calculate relative path for symlink
        duplicate_dir = os.path.dirname(duplicate_path)
        best_file_dir = os.path.dirname(best_file_path)
        best_file_name = os.path.basename(best_file_path)
        
        # If they're in the same directory, just use the filename
        if duplicate_dir == best_file_dir:
            relative_path = best_file_name
        else:
            # Calculate relative path between directories
            try:
                relative_path = os.path.relpath(best_file_path, duplicate_dir)
            except ValueError:
                # If we can't calculate relative path (different drives on Windows), use absolute
                relative_path = best_file_path
        
        # Remove the original duplicate file
        os.remove(duplicate_path)
        
        # Create relative symlink to the best file
        os.symlink(relative_path, duplicate_path)
        
        # Also remove deduper thumbnail if it exists
        directory = os.path.dirname(duplicate_path)
        basename = os.path.basename(duplicate_path)
        basename_stem = Path(basename).stem
        deduper_thumb_path = os.path.join(directory, f"thumb-deduper.{basename_stem}.jpg")
        if os.path.exists(deduper_thumb_path):
            os.remove(deduper_thumb_path)
        
        logger.info(f"Successfully replaced {duplicate_path} with symlink to {best_file_path}")
        return True
        
    except OSError as e:
        logger.error(f"Failed to create symlink for {duplicate_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error creating symlink for {duplicate_path}: {e}")
        return False


__all__ = ["format_file_size", "get_file_size", "create_symlink_and_remove_duplicate", "find_symlinks_pointing_to"]