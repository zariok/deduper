import os
import time
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Callable, Any
import imagehash
from ..utils.bktree import BKTree
from ..utils.helpers import get_file_size, format_file_size, create_symlink_and_remove_duplicate
from ..utils.media import MultiHash, batch_extract_video_thumbnails, get_detailed_resolution, get_file_resolution, get_image_hash, resolve_media_resolution, select_best_video_from_group, get_video_duration
from ..utils.hash_cache import HashCache, get_hash_cache
from ..utils.logging_config import get_logger
from ..utils.metrics import metrics, timer, increment_counter, set_gauge

logger = get_logger(__name__)


def _pack_multihash(hash_obj: "MultiHash") -> tuple[int, int]:
    """Pack a MultiHash into a (phash, dhash) pair of ints for fast comparison.

    imagehash renders as hex whose digits map straight onto the bits of the
    underlying bool array, so XOR over the packed ints differs in exactly the
    positions the arrays do.
    """
    return int(str(hash_obj.phash), 16), int(str(hash_obj.dhash), 16)


def _packed_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Distance between two packed hashes: the larger of the two Hamming distances.

    Identical in result to MultiHash.__sub__ - a pair is similar only when *both*
    hashes agree, so the effective distance is the max - but ~15x cheaper, since
    imagehash counts differing entries of two numpy bool arrays. This runs at the
    innermost level of every BK-tree query and dominates clustering time.
    """
    return max((a[0] ^ b[0]).bit_count(), (a[1] ^ b[1]).bit_count())

# Module-level persistent process pool — avoids fork/spawn overhead per scan.
# Uses 'spawn' context to avoid deadlocks when forking a multi-threaded process
# (the background scanner runs worker threads that hold locks which would be
# copied in a deadlocked state by fork()).
_process_pool: ProcessPoolExecutor | None = None
_pool_lock = mp.Lock()
_spawn_ctx = mp.get_context("spawn")


def _get_process_pool() -> ProcessPoolExecutor:
    """Return (and lazily create) the shared process pool."""
    global _process_pool
    if _process_pool is None:
        with _pool_lock:
            if _process_pool is None:
                workers = min(mp.cpu_count(), 8)  # cap at 8 to avoid memory pressure
                _process_pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=_spawn_ctx,
                )
                logger.info(f"Created persistent process pool with {workers} workers (spawn)")
    return _process_pool


def shutdown_process_pool():
    """Shut down the shared process pool, killing child processes."""
    global _process_pool
    if _process_pool is not None:
        logger.info("Shutting down process pool...")
        _process_pool.shutdown(wait=False, cancel_futures=True)
        _process_pool = None

class DuplicateFinder:
    _BKTREE_CHUNK_SIZE = 10_000

    def __init__(self, image_extensions, video_extensions):
        self.image_extensions = image_extensions
        self.video_extensions = video_extensions

    @timer('duplicate_detection_total')
    def find_duplicates(self, folder_path: str, progress_callback: Callable | None = None) -> tuple[list[dict], list[dict]]:
        """Find duplicate images and videos in the given folder."""
        try:
            # HashCache resolves its own directory, so walk the resolved path too.
            # Otherwise a symlinked parent (/var -> /private/var) makes every relative
            # cache key a long '../../..' path that never matches on the next run.
            folder_path = str(Path(folder_path).resolve())
            logger.info(f"Starting duplicate detection in: {folder_path}")
            increment_counter('duplicate_detection_started')

            if progress_callback:
                progress_callback('initializing_cache', 0, 0, 'Initializing cache...')

            # Initialize hash cache (uses connection pool)
            logger.debug(f"Getting HashCache for: {folder_path}")
            cache = get_hash_cache(folder_path)
            logger.debug(f"HashCache ready, getting stats...")

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

            logger.debug("Scanning directory structure (os.scandir)...")
            file_count = 0

            def _scan_recursive(path: str) -> None:
                nonlocal file_count
                try:
                    with os.scandir(path) as entries:
                        for entry in entries:
                            if entry.name.startswith('thumb.') or entry.name.startswith('thumb-deduper.'):
                                continue
                            if entry.is_symlink():
                                continue
                            if entry.is_file():
                                file_path = entry.path
                                all_files.add(file_path)
                                file_count += 1
                                if any(file_path.lower().endswith(ext) for ext in self.image_extensions):
                                    image_files.append(file_path)
                                elif any(file_path.lower().endswith(ext) for ext in self.video_extensions):
                                    video_files.append(file_path)
                                if file_count % 100 == 0 and progress_callback:
                                    progress_callback('scanning', file_count, 0, f'Found {file_count} files...')
                            elif entry.is_dir():
                                _scan_recursive(entry.path)
                except PermissionError:
                    logger.warning(f"Permission denied: {path}")
                except OSError as e:
                    logger.warning(f"Error scanning {path}: {e}")

            _scan_recursive(folder_path)
            
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
                
                # A file is new when the cache holds no valid hash for it. Group
                # membership is not a usable test: a file that hashed to a group of one
                # is absent from grouping_results, so every unique file would look new
                # on every scan and the work below could never be skipped.
                new_files = {
                    os.path.normpath(f)
                    for f in image_files + video_files
                    if not self._has_valid_cached_hash(f, cache)
                }
                logger.info(f"Found {len(new_files)} new or changed files since the cached scan")
                if new_files:
                    logger.debug(f"First few new files: {list(new_files)[:3]}")

                # Perform incremental grouping
                image_groups, video_groups = self._incremental_grouping(
                    cached_image_groups, cached_video_groups, image_files, video_files,
                    new_files, cache, progress_callback
                )

                if progress_callback:
                    progress_callback('grouping', 1, 1, f'Incremental grouping complete: {len(image_groups)} image groups, {len(video_groups)} video groups')

                # Exact-match elimination runs once, after this branch. Skip it entirely
                # when nothing arrived: re-running it over unchanged cached groups probes
                # the resolution of every file in every group and finds nothing.
                needs_exact_match_pass = bool(new_files)
                if not needs_exact_match_pass:
                    logger.info("No new files found, skipping auto-elimination phase")
                    if progress_callback:
                        progress_callback('processing', 0, 1, 'No new files found, using cached results...')

                # Update cache with the current grouping
                cache.set_cached_groups({**image_groups, **video_groups})
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
                needs_exact_match_pass = True

            # Save cache
            cache.save()
            logger.info("Cache saved successfully")

            # Process exact matches automatically after hashing phase
            if needs_exact_match_pass:
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
                cache.set_cached_groups({**image_groups, **video_groups})
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
                    if processed_image_groups % 10 == 0:
                        time.sleep(0)  # release GIL for HTTP threads
                        if progress_callback:
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
                    if processed_video_groups % 10 == 0:
                        time.sleep(0)  # release GIL for HTTP threads
                        if progress_callback:
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
        """Group files by perceptual hash using parallel processing.

        Uses MultiHash comparison: two files match only when both pHash
        and dHash distances are within *threshold*.
        """
        if not file_paths:
            return {}

        pool = _get_process_pool()
        logger.info("Using persistent process pool for parallel processing")

        hash_func = partial(self._get_file_hash, video_extensions=video_extensions)
        futures = {pool.submit(hash_func, fp): fp for fp in file_paths}
        hash_by_path = {}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                hash_by_path[fp] = future.result()
            except Exception:
                hash_by_path[fp] = None

        logger.debug("Grouping similar files (multi-hash)...")
        groups: dict[str, list[str]] = {}
        # Store (representative_path, representative_hash) for comparison
        rep_hashes: list[tuple[str, Any]] = []

        for file_path in file_paths:
            file_hash = hash_by_path.get(file_path)
            if file_hash is None:
                continue

            found_group = None
            for rep_path, rep_hash in rep_hashes:
                # MultiHash.__sub__ returns max(pHash_dist, dHash_dist)
                if (file_hash - rep_hash) < threshold:
                    found_group = rep_path
                    break

            if found_group is not None:
                groups[found_group].append(file_path)
            else:
                groups[file_path] = [file_path]
                rep_hashes.append((file_path, file_hash))

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
        
        for cache_idx, file_path in enumerate(file_paths):
            relative_path = cache._get_relative_path(file_path)
            has_cached = cache.has_cached_hash(relative_path) and cache._is_file_unchanged(file_path)

            if has_cached:
                cached_hash_str = cache.get_cached_hash_str(relative_path)
                if cached_hash_str:
                    try:
                        cached_hashes[file_path] = MultiHash.from_str(cached_hash_str)
                        cached_files += 1
                    except Exception as e:
                        logger.warning(f"Error loading cached hash for {file_path}: {e}")
                        files_to_process.append(file_path)
                else:
                    files_to_process.append(file_path)
            else:
                files_to_process.append(file_path)
            if cache_idx % 200 == 0:
                time.sleep(0)  # release GIL for HTTP threads
        
        logger.info(f"Found {cached_files} cached hashes, need to process {len(files_to_process)} files")
        
        # Process new/changed files in parallel
        file_hashes = {}
        if files_to_process:
            # Pre-extract video thumbnails concurrently before hashing.
            # This avoids each hash worker spawning its own FFmpeg sequentially.
            video_files_to_process = [
                f for f in files_to_process
                if any(f.lower().endswith(ext) for ext in video_extensions)
            ]
            if video_files_to_process:
                logger.info(f"Pre-extracting {len(video_files_to_process)} video thumbnails concurrently...")
                if progress_callback:
                    progress_callback('hashing', cached_files, len(file_paths),
                                      f'Extracting {len(video_files_to_process)} video thumbnails...')
                batch_extract_video_thumbnails(video_files_to_process, progress_callback=progress_callback)

            pool = _get_process_pool()
            logger.info(f"Using persistent process pool for parallel hash calculation ({len(files_to_process)} files)")

            if progress_callback:
                progress_callback('hashing', cached_files, len(file_paths), f'Hashing {len(files_to_process)} files...')

            hash_func = partial(self._get_file_hash, video_extensions=tuple(video_extensions))

            # Submit all tasks and collect results as they complete so we can
            # report real-time progress instead of blocking on pool.map().
            futures = {
                pool.submit(hash_func, fp): fp
                for fp in files_to_process
            }

            cache_updates = {}
            completed = 0
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    file_hash = future.result()
                except Exception as e:
                    logger.error(f"Error hashing file {file_path}: {e}")
                    file_hash = None

                if file_hash is not None:
                    file_hashes[file_path] = file_hash
                    relative_path = cache._get_relative_path(file_path)
                    mtime, size = cache._get_file_stats(file_path)
                    cache_updates[relative_path] = {
                        "hash": str(file_hash),
                        "stats": {"mtime": mtime, "size": size}
                    }

                completed += 1
                if progress_callback and completed % 50 == 0:
                    progress_callback(
                        'hashing', cached_files + completed, len(file_paths),
                        f'Hashed {cached_files + completed}/{len(file_paths)} files...',
                    )

            # Final progress update
            if progress_callback:
                progress_callback(
                    'hashing', cached_files + completed, len(file_paths),
                    f'Hashed {cached_files + completed}/{len(file_paths)} files...',
                )

            # Batch update cache to reduce I/O operations
            if cache_updates:
                logger.debug(f"Batch updating cache with {len(cache_updates)} entries")
                cache.batch_update_hashes(cache_updates)
        
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
    
    def _incremental_grouping(self, existing_image_groups: dict, existing_video_groups: dict,
                            image_files: list[str], video_files: list[str], new_files: set[str],
                            cache, progress_callback: Callable | None = None) -> tuple[dict, dict]:
        """Group newly seen files against the files already known to the cache.

        Builds a BK-tree over every known file - not only group representatives -
        then searches it once per new file. Complexity: O(new_files * log(known)).

        Indexing non-representatives matters. A file that was unique on an earlier
        scan represents no group, so a tree of representatives alone could never
        match a duplicate of it arriving later.
        """
        logger.info("Performing incremental grouping (BK-tree accelerated)...")

        if not new_files:
            logger.info("No new files found, using cached groups")
            return existing_image_groups, existing_video_groups

        logger.info(f"Found {len(new_files)} new files to process")

        new_image_files = [f for f in new_files if any(f.lower().endswith(ext) for ext in self.image_extensions)]
        new_video_files = [f for f in new_files if any(f.lower().endswith(ext) for ext in self.video_extensions)]
        logger.info(f"New files: {len(new_image_files)} images, {len(new_video_files)} videos")

        # Hash all new files
        new_file_hashes: dict[str, Any] = {}
        for file_path in new_files:
            hash_result = cache.get_hash(file_path, set(self.video_extensions))
            if hash_result is not None:
                new_file_hashes[file_path] = _pack_multihash(hash_result)

        # Create copies of existing groups
        image_groups = {k: list(v) for k, v in existing_image_groups.items()}
        video_groups = {k: list(v) for k, v in existing_video_groups.items()}

        threshold = 5

        # Map every already-grouped file to its group representative
        file_to_group: dict[str, str] = {}
        for groups in (image_groups, video_groups):
            for rep_path, members in groups.items():
                for member in members:
                    file_to_group[os.path.normpath(member)] = rep_path

        # --- Build BK-trees over every known file, grouped or not ---
        def _build_tree(known_files: list[str]) -> tuple[BKTree, int]:
            tree: BKTree = BKTree(_packed_distance)
            indexed = 0
            for file_path in known_files:
                hash_str = cache.get_cached_hash_str(cache._get_relative_path(file_path))
                if not hash_str:
                    continue
                try:
                    tree.add(_pack_multihash(MultiHash.from_str(hash_str)), os.path.normpath(file_path))
                    indexed += 1
                except Exception:
                    pass
            return tree, indexed

        known_image_files = [f for f in image_files if os.path.normpath(f) not in new_files]
        known_video_files = [f for f in video_files if os.path.normpath(f) not in new_files]
        image_tree, indexed_images = _build_tree(known_image_files)
        video_tree, indexed_videos = _build_tree(known_video_files)

        logger.debug(f"Built BK-trees: {indexed_images} known images, {indexed_videos} known videos")

        # --- Search the trees for each new file ---
        for inc_idx, (file_path, file_hash) in enumerate(new_file_hashes.items()):
            is_image = any(file_path.lower().endswith(ext) for ext in self.image_extensions)
            groups, tree, kind = (
                (image_groups, image_tree, 'image') if is_image
                else (video_groups, video_tree, 'video')
            )

            normalized = os.path.normpath(file_path)
            matches = tree.search(file_hash, threshold)

            if matches:
                nearest = min(matches, key=lambda m: m[1])[0]  # closest match
                rep_path = file_to_group.get(nearest)
                if rep_path is None:
                    # Nearest match was an ungrouped file: start a group around it
                    rep_path = nearest
                    groups[rep_path] = [nearest]
                    file_to_group[nearest] = rep_path
                groups[rep_path].append(normalized)
                file_to_group[normalized] = rep_path
                logger.debug(f"Added {os.path.basename(normalized)} to {kind} group {os.path.basename(rep_path)}")
            else:
                groups[normalized] = [normalized]
                file_to_group[normalized] = normalized
                logger.debug(f"Created new {kind} group for {os.path.basename(normalized)}")

            # Index the new file so later new files can match against it
            tree.add(file_hash, normalized)

            if inc_idx % 50 == 0:
                time.sleep(0)  # release GIL for HTTP threads

        # Update cache with new groups
        all_groups = {**image_groups, **video_groups}
        cache.set_cached_groups(all_groups)

        logger.info(f"Incremental grouping complete: {len(image_groups)} image groups, {len(video_groups)} video groups")
        return image_groups, video_groups
    
    @staticmethod
    def _has_valid_cached_hash(file_path: str, cache) -> bool:
        """Report whether the cache already holds a usable hash for this file."""
        relative_path = cache._get_relative_path(file_path)
        return cache.has_cached_hash(relative_path) and cache._is_file_unchanged(file_path)

    @staticmethod
    def _get_file_hash(file_path: str, video_extensions: tuple[str, ...]) -> Any | None:
        """Get hash for a single file - used for multiprocessing."""
        try:
            return get_image_hash(file_path, video_extensions)
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None
    
    
    def _cluster_with_bktree(self, all_hashes: dict[str, Any], threshold: int, progress_callback: Callable | None) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Cluster files using a BK-tree with chunked construction to limit memory use."""
        # Pack the hashes up front; see _packed_distance for why
        valid_items = [
            (path, _pack_multihash(hash_obj))
            for path, hash_obj in all_hashes.items()
            if hash_obj is not None
        ]
        if not valid_items:
            return {}, {"exact_groups": 0, "similar_groups": 0, "total_groups": 0}

        packed_hashes = dict(valid_items)
        bk_tree: BKTree[Any, str] = BKTree(_packed_distance)

        parent: dict[str, str] = {}
        rank: dict[str, int] = {}

        def find(item: str) -> str:
            root = parent.setdefault(item, item)
            if root != item:
                parent[item] = find(root)
            return parent[item]

        def union(a: str, b: str) -> None:
            root_a, root_b = find(a), find(b)
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

        total = len(valid_items)
        chunk_size = self._BKTREE_CHUNK_SIZE

        logger.debug(f"Clustering with BK-tree (chunked, {total} items, threshold={threshold})")
        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk = valid_items[chunk_start:chunk_end]
            for item_idx, (path, hash_obj) in enumerate(chunk):
                matches = bk_tree.search(hash_obj, threshold)
                # No pair-deduplication set here: union() already returns immediately
                # when both paths share a root, so tracking seen pairs only added a
                # sorted tuple of two full path strings per match - O(matches) memory
                # and hashing - to skip a call that is a no-op anyway.
                for match_path, _ in matches:
                    if match_path != path:
                        union(path, match_path)
                bk_tree.add(hash_obj, path)
                if item_idx % 100 == 0:
                    time.sleep(0)  # release GIL for HTTP threads
            if progress_callback:
                progress_callback('grouping', chunk_end, total, f'Clustering hashes {chunk_end}/{total}')
            time.sleep(0)  # release GIL between chunks
        logger.debug(f"Clustering complete over {total} items")
        
        # Build groups more efficiently
        grouped_paths: dict[str, list[str]] = defaultdict(list)
        for path in parent.keys():
            root = find(path)
            grouped_paths[root].append(path)
        
        # Ensure singletons exist
        seen_paths = set(parent.keys())
        for path, _ in valid_items:
            if path not in seen_paths:
                grouped_paths[path].append(path)
        
        # Normalize groups and count types
        normalized_groups: dict[str, list[str]] = {}
        exact_groups = 0
        similar_groups = 0
        
        for grp_idx, group_paths in enumerate(grouped_paths.values()):
            if len(group_paths) > 1:
                group_paths.sort()
                representative = group_paths[0]
                normalized_groups[representative] = group_paths

                # Determine if this is an exact duplicate group more efficiently
                root_hash = packed_hashes.get(representative)
                if root_hash is not None:
                    # Check if all hashes in the group are identical
                    all_identical = True
                    for other_path in group_paths[1:]:
                        other_hash = packed_hashes.get(other_path)
                        if other_hash is None or other_hash != root_hash:
                            all_identical = False
                            break

                    if all_identical:
                        exact_groups += 1
                    else:
                        similar_groups += 1
                else:
                    similar_groups += 1
            if grp_idx % 50 == 0:
                time.sleep(0)  # release GIL for HTTP threads
        
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
    
    def _process_exact_matches_automatically(self, groups: dict[str, list[str]], cache, progress_callback: Callable | None = None) -> dict[str, list[str]]:
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
            time.sleep(0)  # release GIL for HTTP threads between groups
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