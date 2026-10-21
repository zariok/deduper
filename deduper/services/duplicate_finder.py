import os
import time
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable
import imagehash
from ..utils.bktree import BKTree
from ..utils.helpers import format_file_size, create_symlink_and_remove_duplicate
from ..utils.media import (
    SIGNATURE_MATCH_DISTANCE,
    SIGNATURE_MIN_OVERLAP,
    MultiHash,
    Resolution,
    VideoSignature,
    batch_extract_video_thumbnails,
    get_image_hash,
    get_video_signature,
    normalize_extensions,
    select_best_video_from_group,
)
from ..utils.hash_cache import HashCache, get_hash_cache
from ..utils.logging_config import get_logger
from ..utils.metrics import metrics, timer, increment_counter, set_gauge
from .quality_analysis import analyze_group, exact_match_checker, prime_group_analyses

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

# A GIF rendered from a video is redundant with it: same footage, far larger,
# no audio, 256 colours. When the two are confirmed to be the same footage the
# GIF is replaced by a symlink to the video - which reclaims the space while
# leaving the path in place, so a scraper still sees the file and will not
# re-download it. Set False to leave GIFs for manual review instead.
TOMBSTONE_GIFS = True
GIF_SUFFIX = ".gif"


class _UnionFind:
    """Disjoint-set over paths, with path compression and union by rank.

    Iterative on purpose: the inline version in _cluster_with_bktree recurses,
    which is fine there but would risk a deep chain when every frame of every
    video is unioned into the same structure.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        parent = self._parent
        parent.setdefault(item, item)
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:  # compress
            parent[item], item = root, parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        rank_a = self._rank.setdefault(root_a, 0)
        rank_b = self._rank.setdefault(root_b, 0)
        if rank_a < rank_b:
            self._parent[root_a] = root_b
        elif rank_a > rank_b:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] = rank_a + 1


def _packed_matches(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
    limit: float,
    min_overlap: int,
) -> bool:
    """Packed-int equivalent of ``VideoSignature.matches``.

    Must stay **exactly** equivalent to the reference implementation in media.py;
    tests/test_performance_paths.py pins the two together. Three things make this
    the version clustering uses:

    - frames are compared with _packed_distance, ~15x cheaper than
      MultiHash.__sub__ - the same reason image clustering packs its hashes
    - an alignment is abandoned the moment its running total exceeds
      ``limit * overlap``, since the mean can only rise from there
    - it answers "is there an alignment under the limit", so it returns on the
      first one and never finishes scoring the rest

    Together these matter most in the case that used to be worst: a shared intro
    unions many unrelated videos, and every one of those pairs has to be refuted.
    """
    if not a or not b:
        return False
    need = min(min_overlap, len(a), len(b))
    for offset in range(-(len(b) - 1), len(a)):
        start = max(0, offset)
        stop = min(len(a), offset + len(b))
        overlap = stop - start
        if overlap < need:
            continue
        budget = limit * overlap
        total = 0
        for i in range(start, stop):
            total += _packed_distance(a[i], b[i - offset])
            if total > budget:
                break  # mean is already above the limit and cannot come back down
        else:
            return True
    return False


def _identical_sets(file_paths: list[str], cache: "HashCache") -> list[list[str]]:
    """Partition *file_paths* into sets of byte-identical files, size >= 2.

    Files of different sizes cannot be identical, so sizes bucket the work first
    and only buckets with a real collision get their contents read. That keeps a
    scan from hashing every file's bytes while still proving identity before
    anything is deleted.

    Files that cannot be read are skipped rather than assumed identical.
    """
    by_size: dict[int, list[str]] = defaultdict(list)
    for path in file_paths:
        by_size[cache.get_file_size(path)].append(path)

    identical: list[list[str]] = []
    for candidates in by_size.values():
        if len(candidates) < 2:
            continue
        by_digest: dict[str, list[str]] = defaultdict(list)
        for path in candidates:
            digest = cache.get_content_hash(path)
            if digest is not None:
                by_digest[digest].append(path)
        identical.extend(members for members in by_digest.values() if len(members) > 1)
    return identical


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
                # Imported here, not at module level: spawned workers import this
                # module, and config's import-time SECRET_KEY check does not
                # belong in a hashing child. DEDUPER_HASH_WORKERS sizes the pool.
                from ..config import Config

                workers = Config.HASH_WORKERS
                _process_pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=_spawn_ctx,
                )
                logger.info(f"Created persistent process pool with {workers} workers (spawn)")
    return _process_pool


def shutdown_process_pool() -> None:
    """Shut down the shared process pool, killing child processes."""
    global _process_pool
    if _process_pool is not None:
        logger.info("Shutting down process pool...")
        _process_pool.shutdown(wait=False, cancel_futures=True)
        _process_pool = None

class DuplicateFinder:
    _BKTREE_CHUNK_SIZE = 10_000

    def __init__(self, image_extensions: Iterable[str], video_extensions: Iterable[str]):
        self.image_extensions = image_extensions
        self.video_extensions = video_extensions

    @timer('duplicate_detection_total')
    def find_duplicates(self, folder_path: str, progress_callback: Callable | None = None) -> tuple[list[dict], list[dict]]:
        """Find duplicate images and videos in the given folder."""
        # Holds the walk's stat snapshot (HashCache.scan_stats) open for exactly
        # this scan, and closes it however the scan ends
        scan_scope = ExitStack()
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
            # path -> (mtime, size) for every media file, from the walk's own
            # DirEntry.stat(); see HashCache.scan_stats. Built for and used by
            # this scan only.
            stat_cache: dict[str, tuple[float, int]] = {}
            
            if progress_callback:
                progress_callback('scanning', 0, 0, 'Scanning directory structure...')

            logger.debug("Scanning directory structure (os.scandir)...")
            file_count = 0

            def _remember_stat(entry: os.DirEntry[str]) -> None:
                # One syscall, cached on the entry. On failure the path is simply
                # absent, and _get_file_stats falls back to a real stat.
                try:
                    st = entry.stat()
                except OSError:
                    return
                stat_cache[entry.path] = (st.st_mtime, st.st_size)

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
                                    _remember_stat(entry)
                                elif any(file_path.lower().endswith(ext) for ext in self.video_extensions):
                                    video_files.append(file_path)
                                    _remember_stat(entry)
                                if file_count % 100 == 0 and progress_callback:
                                    progress_callback('scanning', file_count, 0, f'Found {file_count} files...')
                            elif entry.is_dir():
                                _scan_recursive(entry.path)
                except PermissionError:
                    logger.warning(f"Permission denied: {path}")
                except OSError as e:
                    logger.warning(f"Error scanning {path}: {e}")

            _scan_recursive(folder_path)
            scan_scope.enter_context(cache.scan_stats(stat_cache))
            
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
            
            # Load any existing cached groups. The walk has just listed every real
            # file and skipped symlinks, so filter members against it instead of
            # asking the filesystem twice per member.
            cached_groups = cache.get_cached_groups(existing=all_files)
            
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
                # One SELECT for the table, checked against the walk's stats.
                stored = cache.get_file_hash_rows()
                new_files = {
                    os.path.normpath(f)
                    for f in image_files + video_files
                    if not self._cached_hash_is_current(f, cache, stored)
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
                    video_groups = self._group_files_by_hash_with_cache(video_files, self.video_extensions, cache, progress_callback, is_video=True)
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

                # The pass keeps the group's best twin, so it needs each
                # group ranked; rank them all on the pool first rather than
                # one at a time inline
                self._prime_analyses(list(image_groups.values()) + list(video_groups.values()), cache, progress_callback)

                # Process exact matches for image groups
                if image_groups:
                    logger.info("Processing exact matches for image groups...")
                    image_groups = self._process_exact_matches_automatically(image_groups, cache, progress_callback)

                # Process exact matches for video groups
                if video_groups:
                    logger.info("Processing exact matches for video groups...")
                    video_groups = self._process_exact_matches_automatically(video_groups, cache, progress_callback)

                # A GIF rendered from a video is never byte-identical to it, so
                # the pass above can never reclaim one; this does.
                if video_groups and TOMBSTONE_GIFS:
                    video_groups = self._tombstone_gifs_automatically(video_groups, cache, progress_callback)

                # Update cache with processed groups
                cache.set_cached_groups({**image_groups, **video_groups})
                cache.save()


            # Add progress callback for final processing phase
            if progress_callback:
                progress_callback('processing', 0, 1, 'Building final duplicate list...')

            # Membership changed for any group the pass above collapsed, and a
            # cache written before analysis existed has none at all
            self._prime_analyses(list(image_groups.values()) + list(video_groups.values()), cache, progress_callback)
            
            # Process groups to find best files with resolution caching
            duplicate_images: list[dict[str, Any]] = []
            duplicate_videos: list[dict[str, Any]] = []
            
            # Cache for resolution calculations to avoid repeated work
            def get_media_meta(file_path: str) -> tuple[Resolution, float]:
                """(resolution, duration) via the persistent cache - no re-probing."""
                return cache.get_media_metadata(file_path, self.image_extensions, self.video_extensions)

            # Byte identity, reported exactly as the auto-elimination pass acts
            # on it, so the UI cannot label a file an exact match that the
            # scanner declined to remove
            is_byte_identical = exact_match_checker(cache)

            # Process image groups
            total_image_groups = len([g for g in image_groups.values() if len(g) > 1])
            processed_image_groups = 0
            
            for group in image_groups.values():
                if len(group) > 1:
                    # Ranked by measured quality, not pixel count - see
                    # quality_analysis. A best file chosen by hand still wins.
                    group_id = self._get_group_id(group, cache)
                    analysis = analyze_group(
                        group, cache, self.image_extensions, self.video_extensions, group_id
                    )
                    best_file = analysis.best_file
                    logger.debug(f"Best image for group {group_id}: {best_file} ({analysis.best_reason})")
                    # Get metadata for best file using cache
                    best_resolution_obj, _ = get_media_meta(best_file)
                    best_size = cache.get_file_size(best_file)
                    
                    # Get metadata for duplicate files using cache
                    duplicate_files_with_metadata = []
                    for f in group:
                        if f != best_file:
                            resolution_obj, _ = get_media_meta(f)
                            size = cache.get_file_size(f)
                            
                            # Byte-identical, not merely same-resolution-and-size
                            is_exact_match = is_byte_identical(f, best_file, size, best_size)

                            duplicate_files_with_metadata.append({
                                'path': cache._get_relative_path(f),
                                'resolution': {
                                    'width': resolution_obj.width,
                                    'height': resolution_obj.height,
                                    'label': resolution_obj.label()
                                },
                                'size': size,
                                'size_formatted': format_file_size(size),
                                'is_exact_match': is_exact_match,
                                **analysis.report_for(f)
                            })
                    
                    # Store group information in cache
                    cache.set_group_files(group_id, group)
                    
                    duplicate_images.append({
                        'group_id': group_id,
                        'best_reason': analysis.best_reason,
                        'best_source': analysis.best_source,
                        'best_file': {
                            'path': cache._get_relative_path(best_file),
                            'resolution': {
                                'width': best_resolution_obj.width,
                                'height': best_resolution_obj.height,
                                'label': best_resolution_obj.label()
                            },
                            'size': best_size,
                            'size_formatted': format_file_size(best_size),
                            **analysis.report_for(best_file)
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
                    group_id = self._get_group_id(group, cache)
                    analysis = analyze_group(
                        group, cache, self.image_extensions, self.video_extensions, group_id
                    )
                    best_file = analysis.best_file
                    logger.debug(f"Best video for group {group_id}: {best_file} ({analysis.best_reason})")
                    # Get metadata for best file using cache
                    best_resolution_obj, best_duration = get_media_meta(best_file)
                    best_size = cache.get_file_size(best_file)
                    
                    # Get metadata for duplicate files using cache
                    duplicate_files_with_metadata = []
                    for f in group:
                        if f != best_file:
                            resolution_obj, duration = get_media_meta(f)
                            size = cache.get_file_size(f)
                            
                            # Byte-identical, not merely same-resolution-and-size
                            is_exact_match = is_byte_identical(f, best_file, size, best_size)

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
                                'is_exact_match': is_exact_match,
                                **analysis.report_for(f)
                            })
                    
                    # Store group information in cache
                    cache.set_group_files(group_id, group)
                    
                    duplicate_videos.append({
                        'group_id': group_id,
                        'best_reason': analysis.best_reason,
                        'best_source': analysis.best_source,
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
                            'duration_formatted': f"{best_duration:.1f}s" if best_duration > 0 else "Unknown",
                            **analysis.report_for(best_file)
                        },
                        'duplicate_files': duplicate_files_with_metadata
                    })
                    
                    processed_video_groups += 1
                    if processed_video_groups % 10 == 0:
                        time.sleep(0)  # release GIL for HTTP threads
                        if progress_callback:
                            progress_callback('processing', processed_video_groups, total_video_groups, f'Processing video groups... {processed_video_groups}/{total_video_groups}')

            # Persist metadata probed while building the results above, so the
            # next scan and the next page load do not repeat the work
            cache.flush_media_metadata()

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
        finally:
            scan_scope.close()
    
    def _group_files_by_hash_with_cache(
        self,
        file_paths: list[str],
        video_extensions: Iterable[str],
        cache: HashCache,
        progress_callback: Callable | None = None,
        threshold: int = 5,
        is_video: bool = False,
    ) -> dict[str, list[str]]:
        """Group files by perceptual hash using cache-aware parallel processing.

        *is_video* selects the clustering path: videos carry a multi-frame
        VideoSignature and are clustered with alignment verification, images a
        single MultiHash through the BK-tree as before.
        """
        if not file_paths:
            return {}
        decode = VideoSignature.from_str if is_video else MultiHash.from_str
        
        logger.debug("Calculating hashes (using cache where possible)...")
        
        if progress_callback:
            progress_callback('hashing', 0, len(file_paths), 'Calculating hashes (using cache where possible)...')
        
        # First, check cache for all files to identify which ones need processing.
        # One SELECT for the table rather than three queries per file.
        files_to_process = []
        cached_hashes = {}
        cached_files = 0
        stored = cache.get_file_hash_rows()
        
        for cache_idx, file_path in enumerate(file_paths):
            row = stored.get(cache._get_relative_path(file_path))
            if row is not None:
                cached_hash_str, mtime, size = row
                if (mtime, size) != cache._get_file_stats(file_path):
                    row = None

            if row is not None and cached_hash_str:
                try:
                    cached_hashes[file_path] = decode(cached_hash_str)
                    cached_files += 1
                except Exception as e:
                    logger.warning(f"Error loading cached hash for {file_path}: {e}")
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
        
        # Group files by similar hashes
        if is_video:
            logger.debug("Grouping videos by frame signature...")
            groups, duplicate_stats = self._cluster_videos(all_hashes, threshold, progress_callback)
        else:
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
                            cache: HashCache, progress_callback: Callable | None = None) -> tuple[dict, dict]:
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

        # Hash all new files. Videos are handled separately below: a signature is
        # a sequence, so it cannot be packed into the single-hash BK-tree that
        # makes the incremental image path fast.
        new_file_hashes: dict[str, Any] = {}
        for file_path in new_files:
            hash_result = cache.get_hash(file_path, set(self.video_extensions))
            if hash_result is None:
                continue
            if isinstance(hash_result, VideoSignature):
                continue
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
        stored = cache.get_file_hash_rows()

        def _build_tree(known_files: list[str]) -> tuple[BKTree, int]:
            tree: BKTree = BKTree(_packed_distance)
            indexed = 0
            for file_path in known_files:
                row = stored.get(cache._get_relative_path(file_path))
                hash_str = row[0] if row else None
                if not hash_str:
                    continue
                try:
                    tree.add(_pack_multihash(MultiHash.from_str(hash_str)), os.path.normpath(file_path))
                    indexed += 1
                except Exception:
                    pass
            return tree, indexed

        known_image_files = [f for f in image_files if os.path.normpath(f) not in new_files]
        image_tree, indexed_images = _build_tree(known_image_files)

        logger.debug(f"Built BK-tree over {indexed_images} known images")

        # --- Search the tree for each new image ---
        for inc_idx, (file_path, file_hash) in enumerate(new_file_hashes.items()):
            groups, tree, kind = image_groups, image_tree, 'image'

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

        # --- Videos: re-cluster from cached signatures when any video is new ---
        # Alignment compares whole sequences, so a new video cannot simply be
        # dropped next to its nearest neighbour the way an image can. Re-running
        # the clustering is cheap regardless: only the new files need hashing,
        # and every other signature is read straight from the cache.
        if new_video_files:
            logger.info(f"{len(new_video_files)} new video(s); re-clustering videos from cached signatures")
            signatures: dict[str, Any] = {}
            for file_path in video_files:
                try:
                    signature = cache.get_hash(file_path, set(self.video_extensions))
                except Exception as e:
                    logger.warning(f"Could not load signature for {file_path}: {e}")
                    continue
                if isinstance(signature, VideoSignature):
                    signatures[os.path.normpath(file_path)] = signature
            video_groups, _ = self._cluster_videos(signatures, threshold, None)

        # Update cache with new groups
        all_groups = {**image_groups, **video_groups}
        cache.set_cached_groups(all_groups)

        logger.info(f"Incremental grouping complete: {len(image_groups)} image groups, {len(video_groups)} video groups")
        return image_groups, video_groups
    
    @staticmethod
    def _has_valid_cached_hash(file_path: str, cache: HashCache) -> bool:
        """Report whether the cache already holds a usable hash for this file.

        Two SELECTs and a stat per call; the scan uses _cached_hash_is_current
        over one get_file_hash_rows() instead.
        """
        relative_path = cache._get_relative_path(file_path)
        return bool(cache.has_cached_hash(relative_path) and cache._is_file_unchanged(file_path))

    @staticmethod
    def _cached_hash_is_current(
        file_path: str, cache: HashCache, stored: dict[str, tuple[str, float, int]]
    ) -> bool:
        """_has_valid_cached_hash against prefetched rows (get_file_hash_rows)."""
        row = stored.get(cache._get_relative_path(file_path))
        return row is not None and (row[1], row[2]) == cache._get_file_stats(file_path)

    @staticmethod
    def _get_file_hash(file_path: str, video_extensions: tuple[str, ...]) -> Any | None:
        """Get hash for a single file - used for multiprocessing.

        Videos return a VideoSignature (many frames), images a MultiHash. Both
        serialize to a single string, so the cache column is unchanged.
        """
        try:
            if Path(file_path).suffix.lower() in normalize_extensions(video_extensions):
                return get_video_signature(file_path)
            return get_image_hash(file_path, video_extensions)
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return None

    def _cluster_videos(
        self,
        signatures: dict[str, Any],
        threshold: int,
        progress_callback: Callable | None,
    ) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Cluster videos by frame signature, in two stages.

        Stage 1 indexes *every* frame of every video in the BK-tree and unions
        two videos when any single frame matches. Being over-inclusive here is
        deliberate: a copy whose start was trimmed shares no frame *position*
        with its source, so retrieval has to work off any frame, not a chosen one.

        Stage 2 pays for that by re-checking each candidate group with the
        aligned distance over the whole sequence, and splitting apart members
        that do not hold up. A shared title card unions two unrelated videos in
        stage 1 and is rejected here, which one frame alone could never do.
        """
        valid = {p: s for p, s in signatures.items() if s is not None and len(s) > 0}
        empty_stats = {"exact_groups": 0, "similar_groups": 0, "total_groups": 0}
        if not valid:
            return {}, empty_stats

        # Pack every frame once. Both stages compare hashes, and doing it here
        # keeps stage 2 off the numpy path entirely.
        packed_frames: dict[str, list[tuple[int, int]]] = {
            path: [_pack_multihash(frame) for frame in signature.frames]
            for path, signature in valid.items()
        }

        bk_tree: BKTree[Any, str] = BKTree(_packed_distance)
        candidates = _UnionFind()

        total = len(valid)
        logger.debug(f"Clustering {total} videos by frame signature (threshold={threshold})")
        for idx, path in enumerate(valid):
            for packed in packed_frames[path]:
                for match_path, _ in bk_tree.search(packed, threshold):
                    if match_path != path:
                        candidates.union(path, match_path)
                bk_tree.add(packed, path)
            if idx % 25 == 0:
                time.sleep(0)  # release GIL for HTTP threads
                if progress_callback:
                    progress_callback('grouping', idx, total, f'Matching video frames {idx}/{total}')

        # --- Stage 2: verify each candidate group by full-sequence alignment ---
        grouped: dict[str, list[str]] = defaultdict(list)
        for path in valid:
            grouped[candidates.find(path)].append(path)

        normalized_groups: dict[str, list[str]] = {}
        exact_groups = 0
        similar_groups = 0
        rejected = 0

        for grp_idx, members in enumerate(grouped.values()):
            if len(members) < 2:
                continue
            members.sort()
            verified = _UnionFind()
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    # Already proven equivalent through some other member, and
                    # equivalence is transitive - skip the comparison entirely.
                    if verified.find(left) == verified.find(right):
                        continue
                    if _packed_matches(
                        packed_frames[left], packed_frames[right],
                        SIGNATURE_MATCH_DISTANCE, SIGNATURE_MIN_OVERLAP,
                    ):
                        verified.union(left, right)
                if i % 20 == 0:
                    time.sleep(0)  # release GIL: a large candidate group is O(n^2)

            subgroups: dict[str, list[str]] = defaultdict(list)
            for member in members:
                subgroups[verified.find(member)].append(member)
            if len(subgroups) > 1:
                rejected += 1
                logger.debug(
                    f"Split a {len(members)}-video candidate group into "
                    f"{len(subgroups)} after alignment: shared footage was not the whole clip"
                )

            for sub in subgroups.values():
                if len(sub) < 2:
                    continue
                sub.sort()
                normalized_groups[sub[0]] = sub
                first = valid[sub[0]]
                if all(valid[m].frames == first.frames for m in sub[1:]):
                    exact_groups += 1
                else:
                    similar_groups += 1

            if grp_idx % 25 == 0:
                time.sleep(0)  # release GIL for HTTP threads

        if rejected:
            logger.info(f"Alignment rejected {rejected} candidate video group(s) as false matches")
        logger.debug(
            f"Video clustering: {exact_groups} exact, {similar_groups} similar, "
            f"{len(normalized_groups)} total groups"
        )
        return normalized_groups, {
            "exact_groups": exact_groups,
            "similar_groups": similar_groups,
            "total_groups": len(normalized_groups),
        }


    
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
    
    
    def _prime_analyses(self, groups: list[list[str]], cache: HashCache, progress_callback: Callable | None) -> None:
        """Rank every group lacking a current analysis, fanned out on the pool."""
        pending = [g for g in groups if len(g) > 1]
        if not pending:
            return
        if progress_callback:
            progress_callback('processing', 0, len(pending), f'Ranking {len(pending)} groups by quality...')
        started = time.time()
        analysed = prime_group_analyses(
            pending, cache, self.image_extensions, self.video_extensions, executor=_get_process_pool()
        )
        if analysed:
            logger.info(f"Analysed {analysed} group(s) for quality in {time.time() - started:.2f}s")

    def _get_group_id(self, group: list[str], cache: HashCache) -> str:
        """Generate a consistent group ID based on the relative paths of files in the group."""
        # Use a more efficient approach with frozenset for consistent ordering
        # and avoid repeated string operations
        relative_files = frozenset(cache._get_relative_path(file_path) for file_path in group)
        
        # Create a hash of the sorted relative file paths more efficiently
        import hashlib
        group_string = "|".join(sorted(relative_files))
        return hashlib.md5(group_string.encode()).hexdigest()[:8]
    
    def _tombstone_gifs_automatically(self, groups: dict[str, list[str]], cache: "HashCache", progress_callback: Callable | None = None) -> dict[str, list[str]]:
        """Replace GIFs with symlinks to the video they were rendered from.

        Runs only on video groups, which by this point have survived alignment
        verification over the whole clip - so a shared title card cannot get a
        GIF tombstoned against footage it does not actually come from. That
        ordering is the safety property: a single-frame match was never
        sufficient evidence to delete anything automatically.

        A group holding only GIFs is left alone; there is no better artifact to
        point at, and choosing between equals is the user's call.
        """
        logger.info("Tombstoning GIFs that duplicate a video...")

        def media_meta(path: str) -> tuple[Resolution, float]:
            return cache.get_media_metadata(path, self.image_extensions, self.video_extensions)

        processed_groups: dict[str, list[str]] = {}
        tombstoned = 0

        for idx, (group_id, group_files) in enumerate(groups.items(), start=1):
            time.sleep(0)  # release GIL for HTTP threads between groups
            gifs = [f for f in group_files if Path(f).suffix.lower() == GIF_SUFFIX]
            others = [f for f in group_files if Path(f).suffix.lower() != GIF_SUFFIX]

            if not gifs or not others:
                processed_groups[group_id] = group_files
                continue

            if progress_callback:
                progress_callback('auto_eliminating', idx - 1, len(groups),
                                  f'Tombstoning GIFs in group {idx}/{len(groups)}')

            keeper = select_best_video_from_group(
                others, self.video_extensions, metadata_provider=media_meta
            )
            ledger_group_id = self._get_group_id(group_files, cache)

            removed: set[str] = set()
            for gif in gifs:
                try:
                    gif_res, gif_duration = media_meta(gif)
                    keeper_res, keeper_duration = media_meta(keeper)
                    # Worth surfacing: a GIF larger than the video it came from is
                    # unusual enough that a wrong match would most likely look
                    # like this, and the log is the only trace afterwards.
                    if gif_res.pixel_count() > keeper_res.pixel_count():
                        logger.info(
                            f"Tombstoning {os.path.basename(gif)} despite it out-resolving "
                            f"{os.path.basename(keeper)}; the video is still the better copy"
                        )

                    # Recorded before the symlink so a crash leaves a record.
                    # Group membership here already survived whole-clip
                    # alignment in _cluster_videos; that is the evidence.
                    row_id = cache.record_auto_action(
                        ledger_group_id, "gif", gif, keeper,
                        {
                            "verified": "aligned_distance",
                            "match_distance": SIGNATURE_MATCH_DISTANCE,
                            "members": len(group_files),
                            "gif_width": gif_res.width,
                            "gif_height": gif_res.height,
                            "gif_duration": gif_duration,
                            "keeper_width": keeper_res.width,
                            "keeper_height": keeper_res.height,
                            "keeper_duration": keeper_duration,
                            "duplicate_size": cache.get_file_size(gif),
                            "keeper_size": cache.get_file_size(keeper),
                        },
                        applied=True,
                    )
                    if create_symlink_and_remove_duplicate(gif, keeper):
                        removed.add(gif)
                        tombstoned += 1
                        # No remove_file_from_groups here: find_duplicates writes
                        # the trimmed groups with set_cached_groups after this pass
                        cache.update_file_stats(gif)
                        logger.info(f"Tombstoned GIF: {os.path.basename(gif)} -> {os.path.basename(keeper)}")
                    else:
                        cache.mark_auto_action(row_id, False, {"error": "symlink failed"})
                        logger.warning(f"Failed to tombstone GIF: {gif}")
                except Exception as e:
                    logger.error(f"Error tombstoning GIF {gif}: {e}")

            remaining = [f for f in group_files if f not in removed]
            if len(remaining) > 1:
                processed_groups[group_id] = remaining
            else:
                logger.debug(f"Group {group_id} has no duplicates left after tombstoning GIFs")

        logger.info(f"Tombstoned {tombstoned} GIF(s)")
        return processed_groups

    def _process_exact_matches_automatically(self, groups: dict[str, list[str]], cache: HashCache, progress_callback: Callable | None = None) -> dict[str, list[str]]:
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
            
            # Only byte-identical files may be removed automatically, so bucket
            # the group by actual content rather than by resolution and size -
            # distinct images do collide on those, and this branch deletes.
            identical_sets = _identical_sets(group_files, cache)
            if not identical_sets:
                # Nothing to remove, and no reason to resolve a best file
                processed_groups[group_id] = group_files
                continue

            # The group's best by measured quality (or the user's pick), so the
            # surviving twin is the file the results will show as best
            # Keyed by the md5 group id, not by this dict's representative
            # path, so the verdict primed on the pool is found rather than
            # recomputed inline
            ledger_group_id = self._get_group_id(group_files, cache)
            best_file = analyze_group(
                group_files, cache, self.image_extensions, self.video_extensions,
                ledger_group_id,
            ).best_file

            removable: list[tuple[str, str]] = []  # (duplicate, keeper) pairs
            for identical in identical_sets:
                # Any member is as good as any other, they are the same bytes.
                # Prefer the group's best file so the surviving path does not
                # move between scans, and fall back to a deterministic pick.
                keeper = best_file if best_file in identical else min(identical)
                removable.extend((f, keeper) for f in identical if f != keeper)

            logger.info(f"Found {len(removable)} byte-identical duplicates for group {group_id}")

            removed = set()
            for duplicate_file, keeper in removable:
                try:
                    # Recorded before the symlink so a crash leaves a record.
                    # The digest was read (and cached) by _identical_sets; it
                    # is fetched before update_file_stats drops the dup's row.
                    evidence: dict[str, Any] = {"size": cache.get_file_size(keeper)}
                    digest = cache.get_content_hash(keeper)
                    if digest is not None:
                        evidence["digest"] = digest
                    row_id = cache.record_auto_action(
                        ledger_group_id, "identical", duplicate_file, keeper,
                        evidence, applied=True,
                    )
                    success = create_symlink_and_remove_duplicate(duplicate_file, keeper)
                    if success:
                        exact_matches_processed += 1
                        total_files_processed += 1
                        removed.add(duplicate_file)

                        # Update cache to reflect the file deletion. Group
                        # membership is not rewritten per file: find_duplicates
                        # writes the trimmed groups with set_cached_groups after
                        # this pass, replacing the whole table either way.
                        cache.update_file_stats(duplicate_file)

                        logger.debug(f"Processed exact match: {duplicate_file} -> {keeper}")
                    else:
                        cache.mark_auto_action(row_id, False, {"error": "symlink failed"})
                        logger.warning(f"Failed to process exact match: {duplicate_file}")
                except Exception as e:
                    logger.error(f"Error processing exact match {duplicate_file}: {e}")

            # Keep everything that was not removed, original order preserved.
            # A file that merely resembles another survives, and so does the one
            # representative of each identical set.
            remaining_files = [f for f in group_files if f not in removed]
            if len(remaining_files) > 1:
                processed_groups[group_id] = remaining_files
            else:
                # If only one file remains, this group no longer has duplicates
                logger.debug(f"Group {group_id} no longer has duplicates after processing exact matches")
        
        logger.info(f"Processed {exact_matches_processed} exact matches, {total_files_processed} total files")
        
        if progress_callback:
            progress_callback('auto_eliminating', len(groups), len(groups), f'Processed {exact_matches_processed} exact matches')
        
        return processed_groups