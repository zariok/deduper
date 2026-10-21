"""Choose the best file in a duplicate group and describe every member.

Both places that present a group - the scan pipeline and the cached-results
page - go through ``analyze_group`` so they cannot disagree about which file is
best. The cached-results route used to take the first member of the group,
which is what showed a lower-resolution copy as "best" after a background scan.

Images are ranked by measured quality (``deduper.utils.quality``); videos keep
the resolution / duration / size rule in ``select_best_video_from_group``, and
are not checked for overlays or filters - their thumbnails are taken at
duration-dependent offsets, so two copies of different length would be
compared at different moments.

Results are cached per group in the ``group_analysis`` table under a fingerprint
of the members' paths, mtimes and sizes, so a warm rescan or page load decodes
nothing. ``prime_group_analyses`` fills that cache for many groups at once on
the process pool; ``analyze_group`` then reads it, and only analyses inline
when a group was missed.
"""

from __future__ import annotations

import time
from concurrent.futures import Executor, Future, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..utils.hash_cache import HashCache
from ..utils.helpers import get_file_size
from ..utils.logging_config import get_logger
from ..utils.media import Resolution, normalize_extensions, select_best_video_from_group
from ..utils.quality import (
    ComparisonImage,
    ImageCandidate,
    ImageQuality,
    Modification,
    decode_for_analysis,
    detect_modifications,
    explain_choice,
    rank_images,
)

logger = get_logger(__name__)

# Payload layout version, so an older stored analysis is rebuilt rather than read
ANALYSIS_VERSION = 1


@dataclass
class MemberReport:
    """What the UI shows for one file: a quality summary and its flags."""

    quality: dict[str, Any]
    modifications: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"quality": self.quality, "modifications": self.modifications}


@dataclass
class GroupAnalysis:
    best_file: str  # absolute path
    best_reason: str
    best_source: str  # 'auto' | 'manual'
    members: dict[str, MemberReport]  # absolute path -> report

    def report_for(self, path: str) -> dict[str, Any]:
        report = self.members.get(path)
        if report is None:
            return {"quality": {}, "modifications": []}
        return report.to_dict()


def _is_video(path: str, video_extensions: Iterable[str]) -> bool:
    return Path(path).suffix.lower() in normalize_extensions(video_extensions)


def _image_payload(candidates: list[ImageCandidate], ranked: list[ImageCandidate]) -> dict:
    best = ranked[0]
    members = {}
    for c in candidates:
        q = c.quality
        members[c.path] = {
            "quality": {
                "score": c.score,
                "effective_megapixels": round(q.effective_pixels / 1e6, 2),
                "detail": round(q.bandwidth, 3),
                "detail_label": q.detail_label(),
                "codec": q.codec_label(),
                "summary": q.summary(),
            },
            "modifications": [m.to_dict() for m in c.modifications],
        }
    return {
        "version": ANALYSIS_VERSION,
        "best_file": best.path,
        "best_reason": explain_choice(best, ranked[1:]),
        "members": members,
    }


def analyze_image_files(
    group: list[str], known: dict[str, ImageQuality]
) -> tuple[dict[str, Any], dict[str, ImageQuality]]:
    """Measure, compare and rank an image group. Pure: no cache, no locks.

    *known* holds measurements already on record; only the rest are decoded
    at full size, and the comparison copy is reduced from that same decode.
    Returns the analysis payload and the measurements that were new, so the
    caller can persist them. Picklable in and out, so this is what runs in a
    pool worker.
    """
    measured: dict[str, ImageQuality] = {}
    qualities: dict[str, ImageQuality] = {}
    comparison: list[ComparisonImage] = []
    for path in group:
        cached = known.get(path)
        quality, image = decode_for_analysis(path, measure=cached is None)
        if cached is None:
            assert quality is not None
            measured[path] = quality
            qualities[path] = quality
        else:
            qualities[path] = cached
        if image is not None:
            comparison.append(image)

    modifications: dict[str, list[Modification]] = detect_modifications(comparison)
    candidates = [
        ImageCandidate(path, qualities[path], get_file_size(path), modifications.get(path, []))
        for path in group
    ]
    return _image_payload(candidates, rank_images(candidates)), measured


def _known_qualities(group: list[str], cache: HashCache) -> dict[str, ImageQuality]:
    known = {}
    for path in group:
        quality = cache.get_cached_image_quality(path)
        if quality is not None:
            known[path] = quality
    return known


def _analyze_images(group: list[str], cache: HashCache) -> dict:
    payload, measured = analyze_image_files(group, _known_qualities(group, cache))
    for path, quality in measured.items():
        cache.set_image_quality(path, quality)
    return payload


def prime_group_analyses(
    groups: Iterable[list[str]],
    cache: HashCache,
    image_extensions: Iterable[str],
    video_extensions: Iterable[str],
    executor: Executor | None = None,
) -> int:
    """Analyse every group that has no current analysis, in parallel.

    Image analysis is CPU-bound - a full decode and an FFT per file - and was
    running one group at a time on the scan thread while the hashing pool sat
    idle; on a cold scan it cost ten times the hashing itself. Hand it an
    executor and the groups fan out across the workers; the cache is only
    touched here, on the calling thread. Video groups are analysed inline:
    they need the cache's ffprobe metadata and cost nothing to rank.

    Returns the number of groups analysed. A group whose analysis is current
    costs one SELECT, plus one stat per member outside a scan (inside one the
    fingerprint reads the walk's snapshot; see HashCache.scan_stats).
    """
    futures: dict[Future[tuple[dict[str, Any], dict[str, ImageQuality]]], tuple[str, str]] = {}
    analysed = 0
    for group in groups:
        if len(group) < 2:
            continue
        group_id = cache._generate_group_id_from_files(group)
        fingerprint = cache.group_fingerprint(group)
        payload = cache.get_group_analysis(group_id, fingerprint)
        if payload is not None and payload.get("version") == ANALYSIS_VERSION:
            continue
        if executor is None or any(_is_video(p, video_extensions) for p in group):
            analyze_group(group, cache, image_extensions, video_extensions, group_id)
            analysed += 1
            continue
        future = executor.submit(analyze_image_files, list(group), _known_qualities(group, cache))
        futures[future] = (group_id, fingerprint)

    for future in as_completed(futures):
        group_id, fingerprint = futures[future]
        try:
            payload, measured = future.result()
        except Exception as exc:
            # The results loop falls back to analysing this one inline
            logger.warning("Group analysis failed in worker for %s: %s", group_id, exc)
            continue
        for path, quality in measured.items():
            cache.set_image_quality(path, quality)
        cache.set_group_analysis(group_id, fingerprint, payload)
        analysed += 1
        time.sleep(0)  # release GIL for HTTP threads between groups
    return analysed


def _analyze_videos(group: list[str], cache: HashCache, image_extensions: Iterable[str],
                    video_extensions: Iterable[str]) -> dict:
    def media_meta(path: str) -> tuple[Resolution, float]:
        return cache.get_media_metadata(path, image_extensions, video_extensions)

    best = select_best_video_from_group(group, video_extensions, metadata_provider=media_meta)
    members = {}
    summaries = {}
    for path in group:
        resolution, duration = media_meta(path)
        size = cache.get_file_size(path)
        bitrate = (size * 8 / duration / 1e6) if duration > 0 else 0.0
        parts = [resolution.label()]
        if duration > 0:
            parts.append(f"{duration:.1f}s")
        if bitrate > 0:
            parts.append(f"{bitrate:.1f} Mbps")
        summaries[path] = " · ".join(parts)
        members[path] = {
            "quality": {
                "score": float(resolution.pixel_count()),
                "summary": summaries[path],
                "codec": Path(path).suffix.lstrip(".").upper(),
            },
            "modifications": [],
        }
    return {
        "version": ANALYSIS_VERSION,
        "best_file": best,
        "best_reason": f"{summaries[best]}; highest resolution, then longest, then largest",
        "members": members,
    }


def analyze_group(
    group: list[str],
    cache: HashCache,
    image_extensions: Iterable[str],
    video_extensions: Iterable[str],
    group_id: str | None = None,
    honour_manual_choice: bool = True,
) -> GroupAnalysis:
    """Rank *group* (absolute paths) and describe each member.

    A best file the user picked by hand (``best_files`` table) wins over the
    measured choice while it is still a member of the group.
    """
    if not group:
        raise ValueError("group cannot be empty")
    if group_id is None:
        group_id = cache._generate_group_id_from_files(group)

    fingerprint = cache.group_fingerprint(group)
    payload = cache.get_group_analysis(group_id, fingerprint)
    if payload is None or payload.get("version") != ANALYSIS_VERSION:
        started = time.monotonic()
        if any(_is_video(p, video_extensions) for p in group):
            payload = _analyze_videos(group, cache, image_extensions, video_extensions)
        else:
            payload = _analyze_images(group, cache)
        cache.set_group_analysis(group_id, fingerprint, payload)
        logger.debug(
            "Analysed group %s (%d files) in %.2fs", group_id, len(group), time.monotonic() - started
        )

    best_file = str(payload["best_file"])
    best_reason = str(payload.get("best_reason", ""))
    best_source = "auto"
    if honour_manual_choice:
        chosen = cache.get_best_file(group_id)
        if chosen and chosen in group and chosen != best_file:
            best_file, best_source = chosen, "manual"
            best_reason = "Chosen manually"
        elif chosen and chosen == best_file:
            best_source = "manual"

    members = {
        path: MemberReport(
            quality=dict(report.get("quality", {})),
            modifications=list(report.get("modifications", [])),
        )
        for path, report in payload.get("members", {}).items()
    }
    return GroupAnalysis(best_file, best_reason, best_source, members)


def exact_match_checker(cache: HashCache) -> Callable[[str, str, int, int], bool]:
    """Return ``is_exact(path, other, size, other_size)`` proving byte identity.

    Guarded on size so a plain near-duplicate never has its contents read, and
    memoised per keeper so a group of N reads the keeper's digest once.
    The stat each digest lookup validates against comes from the scan's
    snapshot when one is active (HashCache.scan_stats).
    """
    digests: dict[str, str | None] = {}

    def is_exact(file_path: str, other: str, size: int, other_size: int) -> bool:
        if file_path == other or size != other_size:
            return False
        if other not in digests:
            digests[other] = cache.get_content_hash(other)
        digest = digests[other]
        return digest is not None and cache.get_content_hash(file_path) == digest

    return is_exact
