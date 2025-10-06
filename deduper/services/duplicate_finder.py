import os
import time
import multiprocessing as mp
from collections import defaultdict
from functools import partial
from typing import List, Dict, Tuple, Optional, Callable, Any
import imagehash
from ..utils.bktree import BKTree
from ..utils.helpers import get_file_size, format_file_size, create_symlink_and_remove_duplicate
from ..utils.media import get_detailed_resolution, get_file_resolution, get_image_hash, resolve_media_resolution, select_best_video_from_group, get_video_duration
from ..utils.hash_cache import HashCache
from ..utils.logging_config import get_logger
from ..utils.metrics import metrics, timer, increment_counter, set_gauge

logger = get_logger(__name__)

class DuplicateFinder:
    def __init__(self, image_extensions, video_extensions):
        self.image_extensions = image_extensions
        self.video_extensions = video_extensions

    @timer('duplicate_detection_total')
    def find_duplicates(self, folder_path: str, progress_callback: Optional[Callable] = None) -> Tuple[List[Dict], List[Dict]]:
        """Find duplicate images and videos in the given folder."""
        try:
            logger.info(f"Starting duplicate detection in: {folder_path}")
            increment_counter('duplicate_detection_started')
            
            if progress_callback:
                progress_callback('initializing_cache', 0, 0, 'Initializing cache...')
            
            # Initialize hash cache
            cache = HashCache(folder_path)
            cache_stats = cache.get_cache_stats()
            logger.info(f"Cache stats: {cache_stats['total_cached_files']} files cached, {cache_stats['cache_size_mb']:.2f} MB")
            
            # Record cache metrics
            set_gauge('cache_files_cached', cache_stats['total_cached_files'])
            set_gauge('cache_size_mb', cache_stats['cache_size_mb'])
            
            # First pass: collect all files and their hashes
            image_files = []
            video_files = []
            all_files = set()
            
            if progress_callback:
                progress_callback('scanning', 0, 0, 'Scanning directory structure...')
            
            logger.debug("Scanning directory structure...")
            file_count = 0
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip thumbnails and symlinks
                    if file.startswith('thumb.') or file.startswith('thumb-deduper.') or os.path.islink(file_path):
                        continue
                        
                    all_files.add(file_path)
                    file_count += 1
                    
                    # Categorize files
                    if any(file.lower().endswith(ext) for ext in self.image_extensions):
                        image_files.append(file_path)
                    elif any(file.lower().endswith(ext) for ext in self.video_extensions):
                        video_files.append(file_path)
                    
                    # Update progress every 100 files
                    if file_count % 100 == 0 and progress_callback:
                        progress_callback('scanning', file_count, 0, f'Found {file_count} files...')
            
            # Clean up deleted files from cache
            cache.cleanup_deleted_files(all_files)
            
            logger.info(f"Found {len(image_files)} images and {len(video_files)} videos")
            logger.info(f"Note: Videos without valid thumbnails will be skipped from duplicate detection")
            
            # Record file counts
            set_gauge('files_found_images', len(image_files))
            set_gauge('files_found_videos', len(video_files))
            set_gauge('files_found_total', len(image_files) + len(video_files))
            
            if progress_callback:
                progress_callback('hashing', 0, len(image_files) + len(video_files), f'Found {len(image_files)} images and {len(video_files)} videos')
            
            # Always try to use cached groups first, then do incremental grouping for new files
            all_media_files = set(image_files + video_files)
            logger.debug(f"Checking for cached groups for {len(all_media_files)} files")
            
            # Load any existing cached groups
            cached_groups = cache.get_cached_groups()
            
            if cached_groups:
                logger.info("Found cached groups, performing incremental grouping...")
                logger.debug(f"Cached groups count: {len(cached_groups)}")
                logger.debug(f"First few cached group keys: {list(cached_groups.keys())[:3]}")
                if progress_callback:
                    progress_callback('grouping', 0, 1, 'Loading cached groups and processing new files...')
                
                # Separate cached groups into image and video groups
                cached_image_groups = {}
                cached_video_groups = {}
                
                for rep_path, files in cached_groups.items():
                    # Determine if this is an image or video group
                    is_image_group = any(any(file.lower().endswith(ext) for ext in self.image_extensions) for file in files)
                    
                    if is_image_group:
                        cached_image_groups[rep_path] = files
                    else:
                        cached_video_groups[rep_path] = files
                
                logger.info(f"Loaded {len(cached_image_groups)} cached image groups and {len(cached_video_groups)} cached video groups")
                
                # Perform incremental grouping
                image_groups, video_groups = self._incremental_grouping(
                    cached_image_groups, cached_video_groups, image_files, video_files, cache, progress_callback
                )
                
                if progress_callback:
                    progress_callback('grouping', 1, 1, f'Incremental grouping complete: {len(image_groups)} image groups, {len(video_groups)} video groups')
                
                # Check if there are actually new files that need processing
                all_current_files = set(image_files + video_files)
                existing_files = set()
                for files in cached_image_groups.values():
                    existing_files.update(files)
                for files in cached_video_groups.values():
                    existing_files.update(files)
                
                # Normalize paths for consistent comparison
                normalized_existing_files = {os.path.normpath(f) for f in existing_files}
                normalized_current_files = {os.path.normpath(f) for f in all_current_files}
                new_files = normalized_current_files - normalized_existing_files
                
                logger.debug(f"Main logic - Existing files count: {len(existing_files)}")
                logger.debug(f"Main logic - Current files count: {len(all_current_files)}")
                logger.debug(f"Main logic - New files count: {len(new_files)}")
                if new_files:
                    logger.debug(f"Main logic - First few new files: {list(new_files)[:3]}")
                
                if new_files:
                    # Only process exact matches if there are new files
                    if progress_callback:
                        progress_callback('auto_eliminating', 0, 1, 'Auto-eliminating exact matches...')
                    
                    # Process exact matches for image groups
                    if image_groups:
                        logger.info("Processing exact matches for image groups (incremental)...")
                        image_groups = self._process_exact_matches_automatically(image_groups, cache, progress_callback)
                    
                    # Process exact matches for video groups
                    if video_groups:
                        logger.info("Processing exact matches for video groups (incremental)...")
                        video_groups = self._process_exact_matches_automatically(video_groups, cache, progress_callback)
                else:
                    logger.info("No new files found, skipping auto-elimination phase")
                    if progress_callback:
                        progress_callback('processing', 0, 1, 'No new files found, using cached results...')
                
                # Update cache with processed groups
                all_processed_groups = {**image_groups, **video_groups}
                cache.set_cached_groups(all_processed_groups)
            else:
                logger.info("No cached groups found, performing full grouping...")
                
                # Process images with optimized grouping using cache-aware processing
                if image_files:
                    logger.info("Processing images...")
                    if progress_callback:
                        progress_callback('hashing', 0, len(image_files), 'Hashing images...')
                    image_groups = self._group_files_by_hash_with_cache(image_files, self.video_extensions, cache, progress_callback)
                else:
                    image_groups = {}
                
                # Process videos with optimized grouping using cache-aware processing
                if video_files:
                    logger.info("Processing videos...")
                    if progress_callback:
                        progress_callback('hashing', len(image_files), len(image_files) + len(video_files), 'Hashing videos...')
                    video_groups = self._group_files_by_hash_with_cache(video_files, self.video_extensions, cache, progress_callback)
                else:
                    video_groups = {}
                
                # Cache the grouping results
                all_groups = {**image_groups, **video_groups}
                cache.set_cached_groups(all_groups)
                cache.set_grouping_timestamp(time.time())
            
            # Save cache
            cache.save()
            logger.info("Cache saved successfully")
            
            # Process exact matches automatically after hashing phase
            if progress_callback:
                progress_callback('auto_eliminating', 0, 1, 'Auto-eliminating exact matches...')
            
            # Process exact matches for image groups
            if image_groups:
                logger.info("Processing exact matches for image groups...")
                image_groups = self._process_exact_matches_automatically(image_groups, cache, progress_callback)
            
            # Process exact matches for video groups
            if video_groups:
                logger.info("Processing exact matches for video groups...")
                video_groups = self._process_exact_matches_automatically(video_groups, cache, progress_callback)
            
            # Update cache with processed groups
            all_processed_groups = {**image_groups, **video_groups}
            cache.set_cached_groups(all_processed_groups)
            cache.save()
            
            # Add progress callback for final processing phase
            if progress_callback:
                progress_callback('processing', 0, 1, 'Building final duplicate list...')
            
            # Process groups to find best files with resolution caching
            duplicate_images = []
            duplicate_videos = []
            
            # Cache for resolution calculations to avoid repeated work
            resolution_cache = {}
            
            def get_cached_resolution(file_path):
                """Get resolution with caching to avoid repeated calculations."""
                if file_path not in resolution_cache:
                    resolution_cache[file_path] = resolve_media_resolution(
                        file_path, tuple(self.image_extensions), tuple(self.video_extensions)
                    )
                return resolution_cache[file_path]
            
            # Process image groups
            total_image_groups = len([g for g in image_groups.values() if len(g) > 1])
            processed_image_groups = 0
            
            for group in image_groups.values():
                if len(group) > 1:
                    # Check if there's a cached best file selection for this group
                    group_id = self._get_group_id(group, cache)
                    cached_best_file = cache.get_best_file(group_id)
                    
                    if cached_best_file and cached_best_file in group:
                        best_file = cached_best_file
                        logger.debug(f"Using cached best file for group {group_id}: {best_file}")
                    else:
                        best_file = max(group, key=lambda x: get_file_resolution(x, tuple(self.image_extensions), tuple(self.video_extensions)))
                    
                    # Get metadata for best file using cache
                    best_resolution_obj = get_cached_resolution(best_file)
                    best_size = get_file_size(best_file)
                    
                    # Get metadata for duplicate files using cache
                    duplicate_files_with_metadata = []
                    for f in group:
                        if f != best_file:
                            resolution_obj = get_cached_resolution(f)
                            size = get_file_size(f)
                            
                            # Check if this is an exact match (same hash, resolution, file size)
                            is_exact_match = (
                                resolution_obj.width == best_resolution_obj.width and
                                resolution_obj.height == best_resolution_obj.height and
                                size == best_size
                            )
                            
                            duplicate_files_with_metadata.append({
                                'path': cache._get_relative_path(f),
                                'resolution': {
                                    'width': resolution_obj.width,
                                    'height': resolution_obj.height,
                                    'label': resolution_obj.label()
                                },
                                'size': size,
                                'size_formatted': format_file_size(size),
                                'is_exact_match': is_exact_match
                            })
                    
                    # Store group information in cache
                    cache.set_group_files(group_id, group)
                    
                    duplicate_images.append({
                        'group_id': group_id,
                        'best_file': {
                            'path': cache._get_relative_path(best_file),
                            'resolution': {
                                'width': best_resolution_obj.width,
                                'height': best_resolution_obj.height,
                                'label': best_resolution_obj.label()
                            },
                            'size': best_size,
                            'size_formatted': format_file_size(best_size)
                        },
                        'duplicate_files': duplicate_files_with_metadata
                    })
                    
                    processed_image_groups += 1
                    if progress_callback and processed_image_groups % 10 == 0:  # Update every 10 groups
                        progress_callback('processing', processed_image_groups, total_image_groups, f'Processing image groups... {processed_image_groups}/{total_image_groups}')
                    
            # Process video groups
            total_video_groups = len([g for g in video_groups.values() if len(g) > 1])
            processed_video_groups = 0
            
            for group in video_groups.values():
                if len(group) > 1:
                    # Check if there's a cached best file selection for this group
                    group_id = self._get_group_id(group, cache)
                    cached_best_file = cache.get_best_file(group_id)
                    
                    if cached_best_file and cached_best_file in group:
                        best_file = cached_best_file
                        logger.debug(f"Using cached best file for group {group_id}: {best_file}")
                    else:
                        # Use enhanced video selection logic
                        best_file = select_best_video_from_group(group, tuple(self.video_extensions))
                        logger.debug(f"Selected best video using enhanced criteria: {best_file}")
                    
                    # Get metadata for best file using cache
                    best_resolution_obj = get_cached_resolution(best_file)
                    best_size = get_file_size(best_file)
                    best_duration = get_video_duration(best_file)
                    
                    # Get metadata for duplicate files using cache
                    duplicate_files_with_metadata = []
                    for f in group:
                        if f != best_file:
                            resolution_obj = get_cached_resolution(f)
                            size = get_file_size(f)
                            duration = get_video_duration(f)
                            
                            # Check if this is an exact match (same hash, resolution, file size)
                            is_exact_match = (
                                resolution_obj.width == best_resolution_obj.width and
                                resolution_obj.height == best_resolution_obj.height and
                                size == best_size
                            )
                            
                            duplicate_files_with_metadata.append({
                                'path': cache._get_relative_path(f),
                                'resolution': {
                                    'width': resolution_obj.width,
                                    'height': resolution_obj.height,
                                    'label': resolution_obj.label()
                                },
                                'size': size,
                                'size_formatted': format_file_size(size),
                                'duration': duration,
                                'duration_formatted': f"{duration:.1f}s" if duration > 0 else "Unknown",
                                'is_exact_match': is_exact_match
                            })
                    
                    # Store group information in cache
                    cache.set_group_files(group_id, group)
                    
                    duplicate_videos.append({
                        'group_id': group_id,
                        'best_file': {
                            'path': cache._get_relative_path(best_file),
                            'resolution': {
                                'width': best_resolution_obj.width,
                                'height': best_resolution_obj.height,
                                'label': best_resolution_obj.label()
                            },
                            'size': best_size,
                            'size_formatted': format_file_size(best_size),
                            'duration': best_duration,
                            'duration_formatted': f"{best_duration:.1f}s" if best_duration > 0 else "Unknown"
                        },
                        'duplicate_files': duplicate_files_with_metadata
                    })
                    
                    processed_video_groups += 1
                    if progress_callback and processed_video_groups % 10 == 0:  # Update every 10 groups
                        progress_callback('processing', processed_video_groups, total_video_groups, f'Processing video groups... {processed_video_groups}/{total_video_groups}')
                    
            # Record final metrics
            total_duplicates = len(duplicate_images) + len(duplicate_videos)
            set_gauge('duplicate_groups_found', total_duplicates)
            set_gauge('duplicate_groups_images', len(duplicate_images))
            set_gauge('duplicate_groups_videos', len(duplicate_videos))
            increment_counter('duplicate_detection_completed')
            
            # Final progress update
            if progress_callback:
                progress_callback('processing', 1, 1, f'Finalizing results... {total_duplicates} duplicate groups found')
            
            logger.info(f"Duplicate detection completed: {total_duplicates} groups found")
            return duplicate_images, duplicate_videos
            
        except Exception as e:
            logger.error(f"Error in find_duplicates: {e}", exc_info=True)
            increment_counter('duplicate_detection_errors')
            return [], []
    
    def _group_files_by_hash_parallel(self, file_paths, video_extensions, threshold=5):
        """Group files by perceptual hash using parallel processing."""
        if not file_paths:
            return {}
        
        # Use multiprocessing for hash calculation
        num_processes = min(mp.cpu_count(), len(file_paths))
        logger.info(f"Using {num_processes} processes for parallel processing")
        
        with mp.Pool(processes=num_processes) as pool:
            # Process files in parallel to get hashes
            hash_func = partial(self._get_file_hash, video_extensions=video_extensions)
            hash_results = pool.map(hash_func, file_paths)
        
        logger.debug("Grouping similar files...")
        # Group files by similar hashes
        groups = {}
        hash_to_groups = defaultdict(list)
        
        for file_path, file_hash in zip(file_paths, hash_results):
            if file_hash is None:
                continue
                
            # Convert hash to integer for easier comparison
            hash_int = int(str(file_hash), 16)
            
            # Find existing group with similar hash
            found_group = None
            for existing_hash in hash_to_groups:
                if abs(hash_int - existing_hash) < threshold:
                    found_group = hash_to_groups[existing_hash][0]
                    break
            
            if found_group is not None:
                groups[found_group].append(file_path)
            else:
                # Create new group
                groups[file_path] = [file_path]
                hash_to_groups[hash_int].append(file_path)
        
        duplicate_count = len([g for g in groups.values() if len(g) > 1])
        logger.info(f"Found {duplicate_count} duplicate groups")
        return groups
    
    def _group_files_by_hash_with_cache(self, file_paths, video_extensions, cache, progress_callback=None, threshold=5):
        """Group files by perceptual hash using cache-aware parallel processing."""
        if not file_paths:
            return {}
        
        logger.debug("Calculating hashes (using cache where possible)...")
        
        if progress_callback:
            progress_callback('hashing', 0, len(file_paths), 'Calculating hashes (using cache where possible)...')
        
        # First, check cache for all files to identify which ones need processing
        files_to_process = []
        cached_hashes = {}
        cached_files = 0
        
        for file_path in file_paths:
            relative_path = cache._get_relative_path(file_path)
            has_cached_hash = relative_path in cache.cache_data["hashes"]
            is_unchanged = cache._is_file_unchanged(file_path)
            
            # Debug logging for first few files
            if len(files_to_process) < 3:
                logger.debug(f"File: {file_path}")
                logger.debug(f"Relative path: {relative_path}")
                logger.debug(f"Has cached hash: {has_cached_hash}")
                logger.debug(f"Is unchanged: {is_unchanged}")
                logger.debug(f"Cache has {len(cache.cache_data['hashes'])} total hashes")
            
            if has_cached_hash and is_unchanged:
                # Use cached hash (convert string back to ImageHash object)
                try:
                    cached_hash_str = cache.cache_data["hashes"][relative_path]
                    cached_hashes[file_path] = imagehash.hex_to_hash(cached_hash_str)
                    cached_files += 1
                except Exception as e:
                    logger.warning(f"Error loading cached hash for {file_path}: {e}")
                    files_to_process.append(file_path)
            else:
                # Need to process this file
                files_to_process.append(file_path)
        
        logger.info(f"Found {cached_files} cached hashes, need to process {len(files_to_process)} files")
        
        # Process new/changed files in parallel
        file_hashes = {}
        if files_to_process:
            num_processes = min(mp.cpu_count(), len(files_to_process))
            logger.info(f"Using {num_processes} processes for parallel hash calculation")
            
            if progress_callback:
                progress_callback('hashing', cached_files, len(file_paths), f'Hashing {len(files_to_process)} files in parallel...')
            
            with mp.Pool(processes=num_processes) as pool:
                # Process files in parallel - use regular hash function, not cache-aware
                hash_func = partial(self._get_file_hash, video_extensions=tuple(video_extensions))
                hash_results = pool.map(hash_func, files_to_process)
            
            # Store results and update cache (batch update for better performance)
            cache_updates = {}
            for i, (file_path, file_hash) in enumerate(zip(files_to_process, hash_results)):
                if file_hash is not None:
                    file_hashes[file_path] = file_hash
                    # Prepare cache updates for batch processing
                    relative_path = cache._get_relative_path(file_path)
                    mtime, size = cache._get_file_stats(file_path)
                    cache_updates[relative_path] = {
                        "hash": str(file_hash),
                        "stats": {"mtime": mtime, "size": size}
                    }
                
                # Update progress every 100 files to reduce overhead
                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback('hashing', cached_files + i + 1, len(file_paths), f'Hashed {cached_files + i + 1}/{len(file_paths)} files...')
            
            # Batch update cache to reduce I/O operations
            if cache_updates:
                logger.debug(f"Batch updating cache with {len(cache_updates)} entries")
                for relative_path, data in cache_updates.items():
                    cache.cache_data["hashes"][relative_path] = data["hash"]
                    cache.cache_data["file_stats"][relative_path] = data["stats"]
                
                # Save cache once after all updates
                cache._save_cache()
        
        # Combine cached and newly calculated hashes
        all_hashes = {**cached_hashes, **file_hashes}
        
        logger.info(f"Used {cached_files} cached hashes, calculated {len(files_to_process)} new hashes")
        
        # Group files by similar hashes using BK-tree clustering
        logger.debug("Grouping similar files with BK-tree...")
        groups, duplicate_stats = self._cluster_with_bktree(all_hashes, threshold, progress_callback)
        
        exact_duplicate_count = duplicate_stats.get("exact_groups", 0)
        similar_duplicate_groups = duplicate_stats.get("similar_groups", 0)
        total_duplicate_groups = duplicate_stats.get("total_groups", 0)
        
        logger.info(f"Found {exact_duplicate_count} exact duplicate groups")
        logger.info(f"Found {similar_duplicate_groups} similar duplicate groups")
        logger.info(f"Found {total_duplicate_groups} total duplicate groups")
        
        return groups
    
    def _incremental_grouping(self, existing_image_groups: Dict, existing_video_groups: Dict, 
                            image_files: List[str], video_files: List[str], cache, progress_callback: Optional[Callable] = None) -> Tuple[Dict, Dict]:
        """Perform incremental grouping for new files only."""
        logger.info("Performing incremental grouping...")
        
        # Get all existing files from cached groups
        existing_files = set()
        for files in existing_image_groups.values():
            existing_files.update(files)
        for files in existing_video_groups.values():
            existing_files.update(files)
        
        # Debug logging for file comparison
        logger.debug(f"Existing files count: {len(existing_files)}")
        logger.debug(f"Current files count: {len(image_files + video_files)}")
        logger.debug(f"First few existing files: {list(existing_files)[:3]}")
        logger.debug(f"First few current files: {list(set(image_files + video_files))[:3]}")
        
        # Check for path mismatches
        if existing_files and image_files + video_files:
            sample_existing = list(existing_files)[0]
            sample_current = list(set(image_files + video_files))[0]
            logger.debug(f"Sample existing file: {sample_existing}")
            logger.debug(f"Sample current file: {sample_current}")
            logger.debug(f"Paths match: {sample_existing == sample_current}")
        
        # Find new files
        all_current_files = set(image_files + video_files)
        
        # Normalize paths for consistent comparison
        normalized_existing_files = {os.path.normpath(f) for f in existing_files}
        normalized_current_files = {os.path.normpath(f) for f in all_current_files}
        
        new_files = normalized_current_files - normalized_existing_files
        
        logger.debug(f"New files count: {len(new_files)}")
        if new_files:
            logger.debug(f"First few new files: {list(new_files)[:3]}")
        
        if not new_files:
            logger.info("No new files found, using cached groups")
            return existing_image_groups, existing_video_groups
        
        logger.info(f"Found {len(new_files)} new files to process")
        
        # Separate new files by type
        new_image_files = [f for f in new_files if any(f.lower().endswith(ext) for ext in self.image_extensions)]
        new_video_files = [f for f in new_files if any(f.lower().endswith(ext) for ext in self.video_extensions)]
        
        logger.info(f"New files: {len(new_image_files)} images, {len(new_video_files)} videos")
        
        # Get hashes for new files (this will use cache for unchanged files)
        new_file_hashes = {}
        for file_path in new_files:
            hash_result = cache.get_hash(file_path, set(self.video_extensions))
            if hash_result is not None:
                new_file_hashes[file_path] = hash_result
        
        # Create copies of existing groups
        image_groups = existing_image_groups.copy()
        video_groups = existing_video_groups.copy()
        
        # Group new files against existing groups
        for file_path, file_hash in new_file_hashes.items():
            found_similar = False
            
            # Check against existing image groups
            if any(file_path.lower().endswith(ext) for ext in self.image_extensions):
                for rep_path, files in image_groups.items():
                    if rep_path in cache.cache_data["hashes"]:
                        existing_hash_str = cache.cache_data["hashes"][cache._get_relative_path(rep_path)]
                        existing_hash = imagehash.hex_to_hash(existing_hash_str)
                        if file_hash - existing_hash < 5:  # threshold
                            image_groups[rep_path].append(file_path)
                            found_similar = True
                            logger.debug(f"Added {file_path} to existing image group {rep_path}")
                            break
                
                if not found_similar:
                    image_groups[file_path] = [file_path]
                    logger.debug(f"Created new image group for {file_path}")
            
            # Check against existing video groups
            elif any(file_path.lower().endswith(ext) for ext in self.video_extensions):
                for rep_path, files in video_groups.items():
                    if rep_path in cache.cache_data["hashes"]:
                        existing_hash_str = cache.cache_data["hashes"][cache._get_relative_path(rep_path)]
                        existing_hash = imagehash.hex_to_hash(existing_hash_str)
                        if file_hash - existing_hash < 5:  # threshold
                            video_groups[rep_path].append(file_path)
                            found_similar = True
                            logger.debug(f"Added {file_path} to existing video group {rep_path}")
                            break
                
                if not found_similar:
                    video_groups[file_path] = [file_path]
                    logger.debug(f"Created new video group for {file_path}")
        
        # Update cache with new groups
        all_groups = {**image_groups, **video_groups}
        cache.set_cached_groups(all_groups)
        
        logger.info(f"Incremental grouping complete: {len(image_groups)} image groups, {len(video_groups)} video groups")
        return image_groups, video_groups
    
    @staticmethod
    def _get_file_hash(file_path: str, video_extensions: Tuple[str, ...]) -> Optional[Any]:
        """Get hash for a single file - used for multiprocessing."""
        try:
            return get_image_hash(file_path, video_extensions)
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None
    
    
    def _cluster_with_bktree(self, all_hashes: Dict[str, Any], threshold: int, progress_callback: Optional[Callable]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """Cluster files using a BK-tree to avoid O(n²) comparisons."""
        # Filter out files without hashes
        valid_items = [(path, hash_obj) for path, hash_obj in all_hashes.items() if hash_obj is not None]
        if not valid_items:
            return {}, {"exact_groups": 0, "similar_groups": 0, "total_groups": 0}
        
        distance_func = lambda a, b: int(a - b)
        bk_tree: BKTree[Any, str] = BKTree(distance_func)
        
        # Build BK-tree more efficiently
        logger.debug(f"Building BK-tree with {len(valid_items)} items")
        for hash_value, path in ((hash_obj, path) for path, hash_obj in valid_items):
            bk_tree.add(hash_value, path)
        
        # Disjoint-set union structure
        parent: Dict[str, str] = {}
        rank: Dict[str, int] = {}
        
        def find(item: str) -> str:
            root = parent.setdefault(item, item)
            if root != item:
                parent[item] = find(root)
            return parent[item]
        
        def union(a: str, b: str) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a == root_b:
                return
            rank.setdefault(root_a, 0)
            rank.setdefault(root_b, 0)
            if rank[root_a] < rank[root_b]:
                parent[root_a] = root_b
            elif rank[root_a] > rank[root_b]:
                parent[root_b] = root_a
            else:
                parent[root_b] = root_a
                rank[root_a] += 1
        
        # Optimize clustering by processing in batches and using set operations
        total = len(valid_items)
        processed_pairs = set()  # Track processed pairs to avoid duplicate work
        
        logger.debug(f"Starting clustering with threshold {threshold}")
        for index, (path, hash_obj) in enumerate(valid_items, start=1):
            matches = bk_tree.search(hash_obj, threshold)
            
            # Process matches more efficiently
            for match_path, distance in matches:
                if match_path == path:
                    continue
                
                # Create a canonical pair to avoid duplicate processing
                pair = tuple(sorted([path, match_path]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                union(path, match_path)
            
            # Update progress less frequently to reduce overhead
            if progress_callback and (index % max(1, total // 20) == 0 or index == total):
                progress_callback('grouping', index, total, f'Clustering hashes {index}/{total}')
        
        logger.debug(f"Clustering complete, processed {len(processed_pairs)} unique pairs")
        
        # Build groups more efficiently
        grouped_paths: Dict[str, List[str]] = defaultdict(list)
        for path in parent.keys():
            root = find(path)
            grouped_paths[root].append(path)
        
        # Ensure singletons exist
        seen_paths = set(parent.keys())
        for path, _ in valid_items:
            if path not in seen_paths:
                grouped_paths[path].append(path)
        
        # Normalize groups and count types
        normalized_groups: Dict[str, List[str]] = {}
        exact_groups = 0
        similar_groups = 0
        
        for group_paths in grouped_paths.values():
            if len(group_paths) > 1:
                group_paths.sort()
                representative = group_paths[0]
                normalized_groups[representative] = group_paths
                
                # Determine if this is an exact duplicate group more efficiently
                root_hash = all_hashes.get(representative)
                if root_hash is not None:
                    # Check if all hashes in the group are identical
                    all_identical = True
                    for other_path in group_paths[1:]:
                        other_hash = all_hashes.get(other_path)
                        if other_hash is None or int(root_hash - other_hash) != 0:
                            all_identical = False
                            break
                    
                    if all_identical:
                        exact_groups += 1
                    else:
                        similar_groups += 1
                else:
                    similar_groups += 1
        
        total_groups = len([g for g in normalized_groups.values() if len(g) > 1])
        logger.debug(f"Clustering results: {exact_groups} exact groups, {similar_groups} similar groups, {total_groups} total groups")
        
        return normalized_groups, {
            "exact_groups": exact_groups,
            "similar_groups": similar_groups,
            "total_groups": total_groups
        }
    
    
    def _get_group_id(self, group, cache):
        """Generate a consistent group ID based on the relative paths of files in the group."""
        # Use a more efficient approach with frozenset for consistent ordering
        # and avoid repeated string operations
        relative_files = frozenset(cache._get_relative_path(file_path) for file_path in group)
        
        # Create a hash of the sorted relative file paths more efficiently
        import hashlib
        group_string = "|".join(sorted(relative_files))
        return hashlib.md5(group_string.encode()).hexdigest()[:8]
    
    def _process_exact_matches_automatically(self, groups: Dict[str, List[str]], cache, progress_callback: Optional[Callable] = None) -> Dict[str, List[str]]:
        """
        Automatically process exact matches by creating symlinks and removing them from groups.
        
        Args:
            groups: Dictionary of file groups
            cache: HashCache instance
            progress_callback: Optional progress callback
            
        Returns:
            Updated groups with exact matches removed
        """
        logger.info("Processing exact matches automatically...")
        
        if progress_callback:
            progress_callback('auto_eliminating', 0, len(groups), 'Auto-eliminating exact matches...')
        
        processed_groups = {}
        exact_matches_processed = 0
        total_files_processed = 0
        
        for idx, (group_id, group_files) in enumerate(groups.items(), start=1):
            if progress_callback:
                progress_callback('auto_eliminating', idx - 1, len(groups), f'Auto-eliminating exact matches in group {idx}/{len(groups)}')
            if len(group_files) <= 1:
                # Single file groups don't need processing
                processed_groups[group_id] = group_files
                continue
            
            # Find the best file (highest resolution)
            best_file = max(group_files, key=lambda x: get_file_resolution(x, tuple(self.image_extensions), tuple(self.video_extensions)))
            best_resolution = resolve_media_resolution(best_file, tuple(self.image_extensions), tuple(self.video_extensions))
            best_size = get_file_size(best_file)
            
            # Separate exact matches from similar matches
            exact_matches = []
            similar_matches = []
            
            for file_path in group_files:
                if file_path == best_file:
                    continue
                    
                file_resolution = resolve_media_resolution(file_path, tuple(self.image_extensions), tuple(self.video_extensions))
                file_size = get_file_size(file_path)
                
                # Check if this is an exact match (same hash, resolution, file size)
                is_exact_match = (
                    file_resolution.width == best_resolution.width and
                    file_resolution.height == best_resolution.height and
                    file_size == best_size
                )
                
                if is_exact_match:
                    exact_matches.append(file_path)
                else:
                    similar_matches.append(file_path)
            
            # Process exact matches automatically
            if exact_matches:
                logger.info(f"Found {len(exact_matches)} exact matches for group {group_id}")
                
                for duplicate_file in exact_matches:
                    try:
                        success = create_symlink_and_remove_duplicate(duplicate_file, best_file)
                        if success:
                            exact_matches_processed += 1
                            total_files_processed += 1
                            
                            # Update cache to reflect the file deletion
                            cache.update_file_stats(duplicate_file)
                            cache.remove_file_from_groups(duplicate_file)
                            
                            logger.debug(f"Processed exact match: {duplicate_file} -> {best_file}")
                        else:
                            logger.warning(f"Failed to process exact match: {duplicate_file}")
                    except Exception as e:
                        logger.error(f"Error processing exact match {duplicate_file}: {e}")
            
            # Update the group - only keep the best file and similar matches
            remaining_files = [best_file] + similar_matches
            if len(remaining_files) > 1:
                processed_groups[group_id] = remaining_files
            else:
                # If only the best file remains, this group no longer has duplicates
                logger.debug(f"Group {group_id} no longer has duplicates after processing exact matches")
        
        logger.info(f"Processed {exact_matches_processed} exact matches, {total_files_processed} total files")
        
        if progress_callback:
            progress_callback('auto_eliminating', len(groups), len(groups), f'Processed {exact_matches_processed} exact matches')
        
        return processed_groups