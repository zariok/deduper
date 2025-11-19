from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

import imagehash
from PIL import Image, UnidentifiedImageError

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Resolution:
    """Represents pixel dimensions for media files."""
    width: int = 0
    height: int = 0

    def is_known(self) -> bool:
        return self.width > 0 and self.height > 0

    def pixel_count(self) -> int:
        return self.width * self.height

    def label(self) -> str:
        return f"{self.width}x{self.height}" if self.is_known() else "Unknown"


def _normalize_extensions(extensions: Iterable[str]) -> Tuple[str, ...]:
    normalized = []
    for ext in extensions:
        if not ext:
            continue
        ext = ext.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in normalized:
            normalized.append(ext)
    return tuple(normalized)


@lru_cache(maxsize=2048)
def _resolve_media_resolution_cached(
    path_str: str,
    image_exts: Tuple[str, ...],
    video_exts: Tuple[str, ...],
) -> Resolution:
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix in image_exts:
        return _read_image_resolution(path)
    if suffix in video_exts:
        return _read_video_resolution(path)
    return Resolution()


def resolve_media_resolution(
    file_path: str,
    image_extensions: Iterable[str],
    video_extensions: Iterable[str],
) -> Resolution:
    """
    Return the pixel dimensions for a media file.

    Falls back to Resolution() when the path cannot be probed or
    does not match the supplied extension lists.
    """
    path = Path(file_path)
    if not path.exists():
        logger.debug("Resolution requested for missing file: %s", file_path)
        return Resolution()

    image_exts = _normalize_extensions(image_extensions)
    video_exts = _normalize_extensions(video_extensions)
    return _resolve_media_resolution_cached(str(path), image_exts, video_exts)


def get_file_resolution(
    file_path: str,
    image_extensions: Iterable[str],
    video_extensions: Iterable[str],
) -> int:
    """Return the total pixel count for the media file."""
    return resolve_media_resolution(file_path, image_extensions, video_extensions).pixel_count()


def get_detailed_resolution(
    file_path: str,
    image_extensions: Iterable[str],
    video_extensions: Iterable[str],
) -> str:
    """Return a human-readable resolution label (e.g. 1920x1080)."""
    return resolve_media_resolution(file_path, image_extensions, video_extensions).label()


def check_ffmpeg() -> bool:
    """Return True when FFmpeg is available on PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        logger.warning("FFmpeg binary not found on PATH")
        return False
    except subprocess.SubprocessError as exc:
        logger.warning("FFmpeg check failed: %s", exc)
        return False
    return True


def check_ffprobe() -> bool:
    """Return True when FFprobe is available on PATH."""
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        logger.warning("FFprobe binary not found on PATH")
        return False
    except subprocess.SubprocessError as exc:
        logger.warning("FFprobe check failed: %s", exc)
        return False
    return True


def check_python_dependencies() -> Tuple[bool, str]:
    """
    Check if required Python dependencies are available.
    
    Returns:
        Tuple of (all_dependencies_available, error_message)
    """
    missing_deps = []
    
    try:
        import imagehash
    except ImportError:
        missing_deps.append("imagehash")
    
    try:
        from PIL import Image
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import flask
    except ImportError:
        missing_deps.append("Flask")
    
    if missing_deps:
        error_msg = f"""
ERROR: Required Python dependencies are missing!

Missing packages: {', '.join(missing_deps)}

Please install them using:
  pip install {' '.join(missing_deps)}

Or install all requirements:
  pip install -r requirements.txt
"""
        return False, error_msg
    
    return True, ""


def check_video_tools() -> Tuple[bool, str]:
    """
    Check if required video processing tools are available.
    
    Returns:
        Tuple of (all_tools_available, error_message)
    """
    ffmpeg_available = check_ffmpeg()
    ffprobe_available = check_ffprobe()
    
    if not ffmpeg_available and not ffprobe_available:
        error_msg = """
ERROR: Required video processing tools are missing!

This application requires FFmpeg and FFprobe for video processing.
Please install them using one of the following methods:

Windows:
  - Download from https://ffmpeg.org/download.html
  - Or use chocolatey: choco install ffmpeg
  - Or use winget: winget install ffmpeg

macOS:
  - brew install ffmpeg

Linux (Ubuntu/Debian):
  - sudo apt-get update && sudo apt-get install ffmpeg

Linux (CentOS/RHEL):
  - sudo yum install ffmpeg

After installation, make sure ffmpeg and ffprobe are available in your PATH.
You can verify this by running: ffmpeg -version && ffprobe -version
"""
        return False, error_msg
    elif not ffmpeg_available:
        error_msg = """
ERROR: FFmpeg is missing!

Please install FFmpeg using one of the following methods:

Windows:
  - Download from https://ffmpeg.org/download.html
  - Or use chocolatey: choco install ffmpeg
  - Or use winget: winget install ffmpeg

macOS:
  - brew install ffmpeg

Linux (Ubuntu/Debian):
  - sudo apt-get update && sudo apt-get install ffmpeg

Linux (CentOS/RHEL):
  - sudo yum install ffmpeg
"""
        return False, error_msg
    elif not ffprobe_available:
        error_msg = """
ERROR: FFprobe is missing!

FFprobe is usually included with FFmpeg installations.
Please reinstall FFmpeg to ensure FFprobe is included:

Windows:
  - Download from https://ffmpeg.org/download.html
  - Or use chocolatey: choco install ffmpeg
  - Or use winget: winget install ffmpeg

macOS:
  - brew install ffmpeg

Linux (Ubuntu/Debian):
  - sudo apt-get update && sudo apt-get install ffmpeg

Linux (CentOS/RHEL):
  - sudo yum install ffmpeg
"""
        return False, error_msg
    
    return True, ""


def check_all_requirements() -> Tuple[bool, str]:
    """
    Check all application requirements (Python dependencies and video tools).
    
    Returns:
        Tuple of (all_requirements_met, error_message)
    """
    # Check Python dependencies first
    deps_ok, deps_error = check_python_dependencies()
    if not deps_ok:
        return False, deps_error
    
    # Check video tools
    tools_ok, tools_error = check_video_tools()
    if not tools_ok:
        return False, tools_error
    
    return True, ""


def extract_video_thumbnail(
    video_path: str,
    *,
    timestamp_seconds: float = 1.0,
    width: int = 320,
) -> Optional[str]:
    """
    Extract a single-frame thumbnail for the supplied video.

    Returns the thumbnail path when successful, otherwise None.
    """
    source = Path(video_path)
    if not source.exists():
        logger.debug("Cannot create thumbnail, video missing: %s", video_path)
        return None

    target = source.with_name(f"thumb-deduper.{source.stem}.jpg")
    try:
        if target.exists() and target.stat().st_size > 0:
            # Validate that the existing thumbnail is a valid image
            if _is_valid_image(target):
                return str(target)
            else:
                logger.debug("Existing thumbnail is invalid, regenerating: %s", target)
                target.unlink(missing_ok=True)
    except OSError:
        # If the existing file is inaccessible, attempt to recreate it.
        pass

    target.parent.mkdir(parents=True, exist_ok=True)

    # Use consistent timestamp for all videos to ensure duplicate detection works
    # Try with different FFmpeg options for better compatibility
    ffmpeg_options = [
        # Standard approach
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _format_timestamp(timestamp_seconds),
            "-i",
            str(source),
            "-vframes",
            "1",
            "-vf",
            f"scale={width}:-1",
            "-f",
            "image2",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(target),
        ],
        # Alternative approach for problematic videos
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            _format_timestamp(timestamp_seconds),
            "-i",
            str(source),
            "-vframes",
            "1",
            "-vf",
            f"scale={width}:-1",
            "-f",
            "image2",
            "-avoid_negative_ts",
            "make_zero",
            "-max_muxing_queue_size",
            "1024",
            "-y",
            str(target),
        ]
    ]
    
    for attempt, cmd in enumerate(ffmpeg_options):
        try:
            # Increase timeout for problematic videos
            timeout = 15 if attempt > 0 else 10
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout, text=True)
            
            # Validate the generated thumbnail
            if target.exists() and target.stat().st_size > 0 and _is_valid_image(target):
                logger.debug("Successfully created thumbnail for %s using approach %d", source, attempt + 1)
                return str(target)
            else:
                logger.debug("FFmpeg did not produce a valid thumbnail for %s using approach %d", source, attempt + 1)
                target.unlink(missing_ok=True)
                
        except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
            logger.debug("Thumbnail attempt %d failed for %s: %s", attempt + 1, source, exc)
            if hasattr(exc, 'stderr') and exc.stderr:
                logger.debug("FFmpeg stderr: %s", exc.stderr)
            target.unlink(missing_ok=True)
            
            # If this was the last attempt, log the final failure
            if attempt == len(ffmpeg_options) - 1:
                logger.warning("Unable to create thumbnail for %s after %d attempts: %s", source, len(ffmpeg_options), exc)
                logger.warning("This video will be SKIPPED from duplicate detection due to thumbnail generation failure")

    return None


def get_image_hash(
    file_path: str,
    video_extensions: Iterable[str],
) -> Optional[imagehash.ImageHash]:
    """
    Compute a perceptual hash for an image or video.

    Videos are hashed based on the extracted thumbnail frame.
    """
    path = Path(file_path)
    video_exts = _normalize_extensions(video_extensions)
    try:
        if path.suffix.lower() in video_exts:
            thumbnail_path = extract_video_thumbnail(str(path))
            if not thumbnail_path:
                logger.warning("No valid thumbnail available for video: %s - SKIPPING from duplicate detection", path)
                return None
            
            # Double-check that the thumbnail is valid before hashing
            thumbnail_path_obj = Path(thumbnail_path)
            if not _is_valid_image(thumbnail_path_obj):
                logger.debug("Thumbnail is not a valid image: %s", thumbnail_path)
                return None
                
            with Image.open(thumbnail_path) as frame:
                return imagehash.phash(frame)
        else:
            # For images, validate before hashing
            if not _is_valid_image(path):
                logger.debug("File is not a valid image: %s", path)
                return None
            with Image.open(path) as image:
                return imagehash.phash(image)
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Unable to hash %s: %s", path, exc)
        return None


def _read_image_resolution(path: Path) -> Resolution:
    try:
        with Image.open(path) as image:
            return Resolution(image.width, image.height)
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Unable to read image %s: %s", path, exc)
        return Resolution()


def _read_video_resolution(path: Path) -> Resolution:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-i",
        str(path),
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        logger.warning("Unable to probe video %s: %s", path, exc)
        if hasattr(exc, 'stderr') and exc.stderr:
            logger.debug("FFprobe stderr: %s", exc.stderr)
        return Resolution()

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid ffprobe output for %s: %s", path, exc)
        return Resolution()

    for stream in payload.get("streams", []):
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width > 0 and height > 0:
            return Resolution(width, height)

    return Resolution()


def get_video_duration(file_path: str) -> float:
    """
    Extract video duration in seconds using ffprobe.
    
    Returns 0.0 if duration cannot be determined.
    """
    path = Path(file_path)
    if not path.exists():
        logger.debug("Duration requested for missing file: %s", file_path)
        return 0.0

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-i",
        str(path),
        "-show_entries",
        "format=duration",
        "-of",
        "json",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
        logger.warning("Unable to probe video duration %s: %s", path, exc)
        if hasattr(exc, 'stderr') and exc.stderr:
            logger.debug("FFprobe stderr: %s", exc.stderr)
        return 0.0

    try:
        payload = json.loads(result.stdout)
        duration_str = payload.get("format", {}).get("duration")
        if duration_str:
            return float(duration_str)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Invalid ffprobe duration output for %s: %s", path, exc)
    
    return 0.0


def compare_thumbnail_similarity(video_path1: str, video_path2: str, video_extensions: Iterable[str]) -> float:
    """
    Compare thumbnail similarity between two videos.
    
    Returns a similarity score between 0.0 (completely different) and 1.0 (identical).
    Returns 0.0 if thumbnails cannot be compared.
    """
    try:
        # Get hashes for both videos
        hash1 = get_image_hash(video_path1, video_extensions)
        hash2 = get_image_hash(video_path2, video_extensions)
        
        if hash1 is None or hash2 is None:
            logger.debug("Cannot compare thumbnails - one or both hashes are None")
            return 0.0
        
        # Calculate Hamming distance (0 = identical, higher = more different)
        hamming_distance = hash1 - hash2
        
        # Convert to similarity score (0-1 scale)
        # Using a threshold of 10 for "completely different" (adjustable)
        max_distance = 10
        similarity = max(0.0, 1.0 - (hamming_distance / max_distance))
        
        return similarity
        
    except Exception as exc:
        logger.debug("Error comparing thumbnails: %s", exc)
        return 0.0


def get_enhanced_video_score(
    file_path: str, 
    video_extensions: Iterable[str],
    reference_resolution: Optional[Resolution] = None,
    reference_duration: Optional[float] = None,
    reference_size: Optional[int] = None
) -> float:
    """
    Calculate an enhanced score for video selection based on multiple criteria.
    
    Higher scores indicate better video quality. Considers:
    - Resolution (pixel count)
    - Duration (longer is better if similar resolution)
    - File size (larger is better if similar resolution)
    - Thumbnail similarity (if reference provided)
    
    Returns a score between 0.0 and 1.0.
    """
    try:
        # Get basic metadata
        resolution = resolve_media_resolution(file_path, [], video_extensions)
        duration = get_video_duration(file_path)
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        
        # Base score from resolution (normalized)
        resolution_score = min(1.0, resolution.pixel_count() / (1920 * 1080))  # Normalize to 1080p
        
        # Duration bonus (if we have a reference duration)
        duration_score = 0.0
        if reference_duration and reference_duration > 0:
            # Prefer longer videos if they have similar resolution
            if reference_resolution and resolution.width == reference_resolution.width and resolution.height == reference_resolution.height:
                duration_ratio = duration / reference_duration
                if duration_ratio > 1.0:  # This video is longer
                    duration_score = min(0.3, (duration_ratio - 1.0) * 0.1)  # Up to 30% bonus for longer duration
                elif duration_ratio < 0.5:  # This video is much shorter
                    duration_score = -0.2  # Penalty for much shorter duration
        
        # File size bonus (if we have a reference size and similar resolution)
        size_score = 0.0
        if reference_size and reference_size > 0 and reference_resolution:
            if resolution.width == reference_resolution.width and resolution.height == reference_resolution.height:
                size_ratio = file_size / reference_size
                if size_ratio > 1.0:  # This video is larger (better quality)
                    size_score = min(0.2, (size_ratio - 1.0) * 0.1)  # Up to 20% bonus for larger size
                elif size_ratio < 0.5:  # This video is much smaller
                    size_score = -0.1  # Penalty for much smaller size
        
        # Combine scores
        total_score = resolution_score + duration_score + size_score
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, total_score))
        
    except Exception as exc:
        logger.debug("Error calculating enhanced video score for %s: %s", file_path, exc)
        return 0.0


def select_best_video_from_group(
    video_group: list, 
    video_extensions: Iterable[str]
) -> str:
    """
    Select the best video from a group of duplicate videos using enhanced criteria.
    
    Considers resolution, duration, file size, and thumbnail similarity.
    For videos with same resolution and similar thumbnails, prefers longer duration.
    
    Returns the path to the best video file.
    """
    if not video_group:
        raise ValueError("Video group cannot be empty")
    
    if len(video_group) == 1:
        return video_group[0]
    
    logger.debug(f"Selecting best video from group of {len(video_group)} videos")
    
    # First, get basic metadata for all videos
    video_metadata = []
    for video_path in video_group:
        try:
            resolution = resolve_media_resolution(video_path, [], video_extensions)
            duration = get_video_duration(video_path)
            file_size = Path(video_path).stat().st_size if Path(video_path).exists() else 0
            
            video_metadata.append({
                'path': video_path,
                'resolution': resolution,
                'duration': duration,
                'file_size': file_size,
                'pixel_count': resolution.pixel_count()
            })
        except Exception as exc:
            logger.debug(f"Error getting metadata for {video_path}: {exc}")
            # Add with default values
            video_metadata.append({
                'path': video_path,
                'resolution': Resolution(),
                'duration': 0.0,
                'file_size': 0,
                'pixel_count': 0
            })
    
    # Sort by resolution first (highest pixel count)
    video_metadata.sort(key=lambda x: x['pixel_count'], reverse=True)
    
    # Group by resolution
    resolution_groups = {}
    for video in video_metadata:
        res_key = (video['resolution'].width, video['resolution'].height)
        if res_key not in resolution_groups:
            resolution_groups[res_key] = []
        resolution_groups[res_key].append(video)
    
    # Find the highest resolution group
    best_resolution = max(resolution_groups.keys(), key=lambda x: x[0] * x[1])
    candidates = resolution_groups[best_resolution]
    
    logger.debug(f"Found {len(candidates)} videos with resolution {best_resolution[0]}x{best_resolution[1]}")
    
    if len(candidates) == 1:
        return candidates[0]['path']
    
    # Among videos with the same resolution, prefer longer duration
    # and larger file size (indicating better quality)
    best_video = max(candidates, key=lambda x: (
        x['duration'],  # Longer duration is better
        x['file_size']  # Larger file size is better (tiebreaker)
    ))
    
    logger.debug(f"Selected best video: {best_video['path']} "
                f"(duration: {best_video['duration']:.2f}s, "
                f"size: {best_video['file_size']} bytes)")
    
    return best_video['path']


def _is_valid_image(image_path: Path) -> bool:
    """
    Check if a file is a valid image that can be opened and processed.
    
    Returns True if the file is a valid image, False otherwise.
    """
    try:
        with Image.open(image_path) as img:
            # Try to load the image to ensure it's not corrupted
            img.load()
            # Check that the image has valid dimensions
            return img.width > 0 and img.height > 0
    except (UnidentifiedImageError, OSError, Exception) as exc:
        logger.debug("Invalid image file %s: %s", image_path, exc)
        return False


def _format_timestamp(seconds: float) -> str:
    safe_seconds = max(0, int(seconds))
    delta = timedelta(seconds=safe_seconds)
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


__all__ = [
    "Resolution",
    "check_ffmpeg",
    "check_ffprobe",
    "check_python_dependencies",
    "check_video_tools",
    "check_all_requirements",
    "extract_video_thumbnail",
    "get_detailed_resolution",
    "get_file_resolution",
    "get_image_hash",
    "get_video_duration",
    "compare_thumbnail_similarity",
    "get_enhanced_video_score",
    "select_best_video_from_group",
    "resolve_media_resolution",
]