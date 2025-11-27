import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List, Set
from .logging_config import get_logger
from .media import get_image_hash

logger = get_logger(__name__)


class HashCache:
    """Manages persistent hash cache for duplicate detection."""

    CACHE_VERSION = "1.0"
    CACHE_FILENAME = ".deduper"

    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path).resolve()
        self.cache_file = self.directory_path / self.CACHE_FILENAME
        # Instance-level lock to prevent contention between different folder operations
        self._file_lock = threading.Lock()
        self.cache_data = self._load_cache()
        self._last_save_time = 0
        self._save_debounce_seconds = 2  # Minimum time between saves
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from .deduper file."""
        if not self.cache_file.exists():
            logger.debug(f"No cache file exists at {self.cache_file}, creating new cache")
            return {
                "version": self.CACHE_VERSION,
                "created": time.time(),
                "last_updated": time.time(),
                "hashes": {},
                "file_stats": {}
            }

        logger.debug(f"Loading cache file: {self.cache_file}")
        file_size = self.cache_file.stat().st_size if self.cache_file.exists() else 0
        logger.debug(f"Cache file size: {file_size / 1024 / 1024:.2f} MB")

        with self._file_lock:
            try:
                load_start = time.time()
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                load_time = time.time() - load_start
                logger.debug(f"Cache JSON parsed in {load_time:.2f}s, {len(data.get('hashes', {}))} hashes")

                # Validate cache version
                if data.get("version") != self.CACHE_VERSION:
                    logger.info(f"Cache version mismatch, creating new cache")
                    return self._create_empty_cache()

                # Migrate old cache format if needed
                self._migrate_cache_format(data)

                return data
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Error loading cache: {e}, creating new cache")
                # Try to remove corrupted cache file
                try:
                    if self.cache_file.exists():
                        self.cache_file.unlink()
                        logger.info("Removed corrupted cache file")
                except OSError:
                    pass
                return self._create_empty_cache()
    
    def _migrate_cache_format(self, data: Dict[str, Any]):
        """Migrate old cache format to new format."""
        migrated = False
        for relative_path, stats in data.get("file_stats", {}).items():
            if isinstance(stats, (list, tuple)) and len(stats) == 2:
                # Old format: (mtime, size) -> New format: {"mtime": mtime, "size": size}
                mtime, size = stats
                data["file_stats"][relative_path] = {"mtime": mtime, "size": size}
                migrated = True
        
        if migrated:
            logger.info("Migrated cache format from tuple to dictionary format")
            # Save the migrated cache
            self.cache_data = data
            self._save_cache()
    
    def _get_relative_path(self, file_path: str) -> str:
        """Convert absolute path to relative path within the directory."""
        try:
            return os.path.relpath(file_path, self.directory_path)
        except ValueError:
            # If we can't make it relative (different drives on Windows), use the filename
            return os.path.basename(file_path)
    
    def _get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path back to absolute path."""
        return os.path.join(self.directory_path, relative_path)
    
    def _create_empty_cache(self) -> Dict[str, Any]:
        """Create a new empty cache structure."""
        return {
            "version": self.CACHE_VERSION,
            "created": time.time(),
            "last_updated": time.time(),
            "hashes": {},
            "file_stats": {}
        }
    
    def _save_cache(self):
        """Save cache to .deduper file."""
        with self._file_lock:
            try:
                # Validate cache data before saving
                if not self._validate_cache_data():
                    logger.error("Cache data validation failed, skipping save")
                    return
                
                self.cache_data["last_updated"] = time.time()
                # Write to a temporary file first, then rename to avoid corruption
                temp_file = self.cache_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename to prevent corruption
                temp_file.replace(self.cache_file)
                self._last_save_time = time.time()
                logger.debug(f"Cache saved successfully to {self.cache_file}")
            except OSError as e:
                logger.error(f"Error saving cache: {e}")
                # Clean up temp file if it exists
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except OSError:
                    pass
            except Exception as e:
                logger.error(f"Unexpected error saving cache: {e}")
                # Clean up temp file if it exists
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except OSError:
                    pass
    
    def _save_cache_debounced(self):
        """Save cache with debouncing to prevent too frequent saves."""
        current_time = time.time()
        if current_time - self._last_save_time >= self._save_debounce_seconds:
            self._save_cache()
        else:
            logger.debug(f"Skipping cache save, too soon since last save ({current_time - self._last_save_time:.1f}s)")
    
    def _validate_cache_data(self) -> bool:
        """Validate that cache data is in a good state before saving."""
        try:
            # Check that required keys exist
            required_keys = ["version", "created", "last_updated", "hashes", "file_stats"]
            for key in required_keys:
                if key not in self.cache_data:
                    logger.warning(f"Missing required cache key: {key}")
                    return False
            
            # Check that hashes and file_stats are dictionaries
            if not isinstance(self.cache_data["hashes"], dict):
                logger.warning("Cache hashes is not a dictionary")
                return False
            
            if not isinstance(self.cache_data["file_stats"], dict):
                logger.warning("Cache file_stats is not a dictionary")
                return False
            
            # Try to serialize to JSON to catch any serialization issues
            json.dumps(self.cache_data, ensure_ascii=False)
            
            return True
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False
    
    def _get_file_stats(self, file_path: str) -> Tuple[float, int]:
        """Get file modification time and size."""
        try:
            stat = os.stat(file_path)
            return stat.st_mtime, stat.st_size
        except OSError:
            return 0, 0
    
    def _is_file_unchanged(self, file_path: str) -> bool:
        """Check if file has been modified since last cache."""
        current_mtime, current_size = self._get_file_stats(file_path)
        relative_path = self._get_relative_path(file_path)
        cached_stats = self.cache_data["file_stats"].get(relative_path)
        
        if not cached_stats:
            return False
        
        # Handle both old tuple format and new dict format for backward compatibility
        if isinstance(cached_stats, (list, tuple)):
            # Old format: (mtime, size)
            cached_mtime, cached_size = cached_stats
        else:
            # New format: {"mtime": mtime, "size": size}
            cached_mtime = cached_stats["mtime"]
            cached_size = cached_stats["size"]
        
        return (cached_mtime == current_mtime and cached_size == current_size)
    
    def get_hash(self, file_path: str, video_extensions: set) -> Optional[Any]:
        """Get hash for file, using cache if available and file unchanged."""
        # Check if file exists and is unchanged
        if not os.path.exists(file_path):
            return None
        
        relative_path = self._get_relative_path(file_path)
        
        if self._is_file_unchanged(file_path):
            cached_hash_str = self.cache_data["hashes"].get(relative_path)
            if cached_hash_str:
                # Convert string back to ImageHash object
                try:
                    import imagehash
                    return imagehash.hex_to_hash(cached_hash_str)
                except Exception as e:
                    logger.warning(f"Error converting cached hash for {file_path}: {e}")
                    return None
        
        # File is new or changed, calculate hash
        file_hash = get_image_hash(file_path, tuple(video_extensions))
        
        if file_hash is not None:
            # Update cache with relative path (convert ImageHash to string for JSON serialization)
            self.cache_data["hashes"][relative_path] = str(file_hash)
            mtime, size = self._get_file_stats(file_path)
            self.cache_data["file_stats"][relative_path] = {"mtime": mtime, "size": size}
            # Use debounced save to reduce I/O operations
            self._save_cache_debounced()
        
        return file_hash
    
    def update_file_stats(self, file_path: str):
        """Update file statistics in cache."""
        relative_path = self._get_relative_path(file_path)
        if os.path.exists(file_path):
            mtime, size = self._get_file_stats(file_path)
            self.cache_data["file_stats"][relative_path] = {"mtime": mtime, "size": size}
        else:
            # File was deleted, remove from cache
            self.cache_data["hashes"].pop(relative_path, None)
            self.cache_data["file_stats"].pop(relative_path, None)
    
    def cleanup_deleted_files(self, existing_files: set):
        """Remove entries for files that no longer exist and clean up associated thumbnails."""
        # Convert existing absolute paths to relative paths for comparison
        existing_relative_files = {self._get_relative_path(f) for f in existing_files}
        cached_files = set(self.cache_data["hashes"].keys())
        deleted_files = cached_files - existing_relative_files
        
        for relative_path in deleted_files:
            # Clean up cache entries
            self.cache_data["hashes"].pop(relative_path, None)
            self.cache_data["file_stats"].pop(relative_path, None)
            
            # Clean up associated thumb-deduper file if it exists
            absolute_path = self.directory_path / relative_path
            if absolute_path.exists():
                # File still exists, skip thumbnail cleanup
                continue
                
            # File was deleted, check for and remove associated thumbnail
            thumb_path = absolute_path.with_name(f"thumb-deduper.{absolute_path.stem}.jpg")
            if thumb_path.exists():
                try:
                    thumb_path.unlink()
                    logger.debug(f"Cleaned up thumbnail: {thumb_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove thumbnail {thumb_path}: {e}")
        
        if deleted_files:
            logger.info(f"Cleaned up {len(deleted_files)} deleted files from cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_cached_files": len(self.cache_data["hashes"]),
            "cache_size_mb": self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0,
            "created": self.cache_data["created"],
            "last_updated": self.cache_data["last_updated"]
        }
    
    def save(self):
        """Save cache to disk."""
        self._save_cache()
    
    def force_save(self):
        """Force save cache immediately, bypassing debouncing."""
        self._save_cache()
    
    def invalidate_cache(self):
        """Invalidate entire cache."""
        self.cache_data = self._create_empty_cache()
        self._save_cache()
        logger.info("Cache invalidated")
    
    def clear_corrupted_cache(self):
        """Clear cache if it's corrupted and create a new one."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info("Removed corrupted cache file")
        except OSError as e:
            logger.error(f"Error removing corrupted cache: {e}")
        
        self.cache_data = self._create_empty_cache()
        self._save_cache()
        logger.info("Created new cache")
    
    def set_best_file(self, group_id: str, best_file_path: str):
        """Set the best file for a duplicate group."""
        if "best_files" not in self.cache_data:
            self.cache_data["best_files"] = {}
        
        relative_path = self._get_relative_path(best_file_path)
        self.cache_data["best_files"][group_id] = relative_path
        self._save_cache_debounced()
        logger.debug(f"Set best file for group {group_id}: {relative_path}")
    
    def get_best_file(self, group_id: str) -> Optional[str]:
        """Get the best file for a duplicate group."""
        if "best_files" not in self.cache_data:
            return None
        
        relative_path = self.cache_data["best_files"].get(group_id)
        if relative_path:
            return self._get_absolute_path(relative_path)
        return None
    
    def get_group_files(self, group_id: str) -> list:
        """Get all files in a duplicate group."""
        if "groups" not in self.cache_data:
            self.cache_data["groups"] = {}
        
        group_data = self.cache_data["groups"].get(group_id, {})
        file_paths = group_data.get("files", [])
        
        # If no files found in groups, try to find the group in grouping_results
        if not file_paths and "grouping_results" in self.cache_data:
            logger.debug(f"Group {group_id} not found in groups, searching in grouping_results")
            # Search through grouping_results to find files for this group
            for rep_path, files in self.cache_data["grouping_results"].items():
                # Check if any file in this group matches the group_id pattern
                # Group ID is based on sorted relative paths, so we need to reconstruct it
                if files:  # Only process non-empty groups
                    # Convert relative paths to absolute for comparison
                    abs_files = [self._get_absolute_path(rel_path) for rel_path in files]
                    # Generate group ID for this group
                    temp_group_id = self._generate_group_id_from_files(abs_files)
                    if temp_group_id == group_id:
                        logger.debug(f"Found group {group_id} in grouping_results with {len(files)} files")
                        # Store the group files for future use
                        self.set_group_files(group_id, abs_files)
                        return abs_files
        
        # If still no files found, try a more aggressive search in grouping_results
        # This handles cases where the group might be newly created but not yet stored in groups
        if not file_paths and "grouping_results" in self.cache_data:
            # Try to find the group by checking all possible combinations
            for rep_path, files in self.cache_data["grouping_results"].items():
                if files:  # Only process non-empty groups
                    # Convert relative paths to absolute for comparison
                    abs_files = [self._get_absolute_path(rel_path) for rel_path in files]
                    # Generate group ID for this group
                    temp_group_id = self._generate_group_id_from_files(abs_files)
                    if temp_group_id == group_id:
                        # Store the group files for future use
                        self.set_group_files(group_id, abs_files)
                        return abs_files
        
        # Convert relative paths back to absolute paths
        return [self._get_absolute_path(rel_path) for rel_path in file_paths]
    
    def set_group_files(self, group_id: str, file_paths: list):
        """Set the files in a duplicate group."""
        if "groups" not in self.cache_data:
            self.cache_data["groups"] = {}
        
        # Convert absolute paths to relative paths for storage
        relative_paths = [self._get_relative_path(path) for path in file_paths]
        self.cache_data["groups"][group_id] = {"files": relative_paths}
        self._save_cache_debounced()
    
    def mark_group_processed(self, group_id: str):
        """Mark a group as processed (all duplicates converted to symlinks)."""
        if "groups" not in self.cache_data:
            self.cache_data["groups"] = {}
        
        if group_id not in self.cache_data["groups"]:
            self.cache_data["groups"][group_id] = {}
        
        self.cache_data["groups"][group_id]["processed"] = True
        self._save_cache_debounced()
        logger.debug(f"Marked group {group_id} as processed")
    
    def get_cached_groups(self) -> dict:
        """Get cached grouping results, filtering out symlinks."""
        if "grouping_results" not in self.cache_data:
            logger.debug("No grouping_results in cache_data")
            return {}
        
        logger.debug(f"Found {len(self.cache_data['grouping_results'])} groups in cache_data")
        
        # Convert relative paths back to absolute paths and filter out symlinks
        cached_groups = {}
        for rep_path, files in self.cache_data["grouping_results"].items():
            abs_rep_path = self._get_absolute_path(rep_path)
            abs_files = []
            
            for rel_path in files:
                abs_path = self._get_absolute_path(rel_path)
                # Skip symlinks - they point to the "best image"
                if not os.path.islink(abs_path):
                    abs_files.append(abs_path)
            
            # Only include groups that still have non-symlink files
            if abs_files:
                cached_groups[abs_rep_path] = abs_files
        
        logger.debug(f"Converted to {len(cached_groups)} absolute path groups (filtered symlinks)")
        return cached_groups
    
    def set_cached_groups(self, groups: dict):
        """Cache grouping results."""
        logger.debug(f"set_cached_groups called with {len(groups)} groups")
        
        # Convert absolute paths to relative paths for storage
        relative_groups = {}
        for rep_path, files in groups.items():
            rel_rep_path = self._get_relative_path(rep_path)
            rel_files = [self._get_relative_path(file_path) for file_path in files]
            relative_groups[rel_rep_path] = rel_files
        
        self.cache_data["grouping_results"] = relative_groups
        logger.debug(f"Stored {len(relative_groups)} groups in cache_data")
        self._save_cache_debounced()
        logger.debug(f"Cached {len(groups)} groups")
    
    def get_grouping_timestamp(self) -> float:
        """Get timestamp of last grouping operation."""
        return self.cache_data.get("grouping_timestamp", 0)
    
    def set_grouping_timestamp(self, timestamp: float):
        """Set timestamp of grouping operation."""
        self.cache_data["grouping_timestamp"] = timestamp
        self._save_cache_debounced()
    
    def is_grouping_stale(self, file_paths: set) -> bool:
        """Check if cached grouping is stale (files added/removed)."""
        if "grouping_results" not in self.cache_data:
            logger.debug("No grouping_results in cache, returning stale=True")
            return True
        
        # Get all non-symlink files that were in the last grouping
        cached_files = set()
        for files in self.cache_data["grouping_results"].values():
            for rel_path in files:
                abs_path = self._get_absolute_path(rel_path)
                # Only include non-symlink files in comparison
                if not os.path.islink(abs_path):
                    cached_files.add(abs_path)
        
        logger.debug(f"Cached non-symlink files count: {len(cached_files)}")
        logger.debug(f"Current files count: {len(file_paths)}")
        logger.debug(f"Files match: {cached_files == file_paths}")
        
        # Check if current file set matches cached file set
        is_stale = cached_files != file_paths
        logger.debug(f"is_grouping_stale returning: {is_stale}")
        return is_stale
    
    def invalidate_grouping_cache(self):
        """Invalidate the grouping cache to force regeneration."""
        if "grouping_results" in self.cache_data:
            del self.cache_data["grouping_results"]
        if "grouping_timestamp" in self.cache_data:
            del self.cache_data["grouping_timestamp"]
        self._save_cache_debounced()
        logger.debug("Invalidated grouping cache")
    
    def remove_file_from_groups(self, file_path: str):
        """Remove a specific file from all cached groups."""
        if "grouping_results" not in self.cache_data:
            return
        
        relative_path = self._get_relative_path(file_path)
        updated_groups = {}
        
        for rep_path, files in self.cache_data["grouping_results"].items():
            # Filter out the deleted file from this group
            filtered_files = [f for f in files if f != relative_path]
            
            # Only keep groups that still have files (not empty)
            if filtered_files:
                updated_groups[rep_path] = filtered_files
        
        self.cache_data["grouping_results"] = updated_groups
        self._save_cache_debounced()
        logger.debug(f"Removed {relative_path} from cached groups")
    
    def _generate_group_id_from_files(self, file_paths: list) -> str:
        """Generate a group ID from a list of file paths (same logic as in duplicate_finder.py)."""
        # Convert absolute paths to relative paths for consistent group IDs
        relative_files = []
        for file_path in file_paths:
            relative_path = self._get_relative_path(file_path)
            relative_files.append(relative_path)
        
        # Sort the relative file paths to ensure consistent group ID regardless of order
        sorted_files = sorted(relative_files)
        # Create a hash of the sorted relative file paths
        group_string = "|".join(sorted_files)
        return hashlib.md5(group_string.encode()).hexdigest()[:8]