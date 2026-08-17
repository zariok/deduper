"""Guards for the performance-motivated code paths.

These protect behaviour that is easy to regress silently, because getting it
wrong costs speed or memory rather than producing an obviously wrong answer.
The exception is the packed-distance equivalence below: that one decides which
files are treated as duplicates, so a mismatch would change what gets deleted.
"""

import os
import random

import imagehash
import pytest

from deduper.services.duplicate_finder import (
    DuplicateFinder,
    _pack_multihash,
    _packed_distance,
    _packed_matches,
)
from deduper.utils.media import (
    SIGNATURE_MATCH_DISTANCE,
    SIGNATURE_MIN_OVERLAP,
    MultiHash,
    VideoSignature,
    get_image_hash,
    get_video_signature,
)

from .conftest import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    gradient_image,
)


def _random_multihash(rng):
    hex_of = lambda: "".join(rng.choice("0123456789abcdef") for _ in range(16))
    return MultiHash(
        phash=imagehash.hex_to_hash(hex_of()),
        dhash=imagehash.hex_to_hash(hex_of()),
    )


def _random_signature(rng, length, step=5.0, shared_prefix=None):
    """A signature of *length* frames, optionally opening with shared frames."""
    frames = list(shared_prefix or [])
    frames += [_random_multihash(rng) for _ in range(length - len(frames))]
    return VideoSignature(frames=tuple(frames), step=step)


class TestPackedSignatureMatching:
    """_packed_matches decides which videos get grouped - and, via the GIF
    tombstone, which files get deleted. It must agree with the reference
    implementation on media.VideoSignature exactly, never merely usually.
    """

    def _agree(self, a, b):
        packed_a = [_pack_multihash(f) for f in a.frames]
        packed_b = [_pack_multihash(f) for f in b.frames]
        fast = _packed_matches(
            packed_a, packed_b, SIGNATURE_MATCH_DISTANCE, SIGNATURE_MIN_OVERLAP
        )
        return fast, a.matches(b)

    def test_agrees_on_unrelated_signatures(self):
        rng = random.Random(3)
        for _ in range(60):
            a = _random_signature(rng, 13)
            b = _random_signature(rng, 13)
            fast, reference = self._agree(a, b)
            assert fast == reference

    def test_agrees_when_one_is_a_trimmed_copy(self):
        rng = random.Random(7)
        for cut in (0, 1, 3, 5):
            full = _random_signature(rng, 14)
            trimmed = VideoSignature(frames=full.frames[cut:], step=full.step)
            fast, reference = self._agree(full, trimmed)
            assert fast == reference, f"disagreed on a {cut}-frame trim"

    def test_agrees_when_only_an_opening_is_shared(self):
        """The case the early-abandon prune is built for: a long shared prefix
        followed by entirely different frames."""
        rng = random.Random(13)
        for prefix_len in (1, 3, 6, 9):
            shared = [_random_multihash(rng) for _ in range(prefix_len)]
            a = _random_signature(rng, 13, shared_prefix=shared)
            b = _random_signature(rng, 13, shared_prefix=shared)
            fast, reference = self._agree(a, b)
            assert fast == reference, f"disagreed on a {prefix_len}-frame shared opening"

    def test_agrees_on_identical_and_on_length_mismatch(self):
        rng = random.Random(21)
        a = _random_signature(rng, 12)
        assert self._agree(a, a) == (True, True)

        short = VideoSignature(frames=a.frames[:2], step=a.step)
        fast, reference = self._agree(a, short)
        assert fast == reference

    def test_agrees_on_real_video_signatures(self, signature_dir):
        sigs = [
            get_video_signature(os.path.join(signature_dir, n))
            for n in ("full.mp4", "full.gif", "trim_start.mp4", "share_a.mp4", "share_b.mp4")
        ]
        sigs = [s for s in sigs if s]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                fast, reference = self._agree(sigs[i], sigs[j])
                assert fast == reference


class TestPackedDistance:
    """The packed comparison must agree exactly with MultiHash's own.

    Clustering keys off this value. If the packed form were ever more permissive,
    unrelated files would be grouped and the exact-match pass would symlink one
    over the other.
    """

    def test_matches_multihash_subtraction(self):
        rng = random.Random(11)
        hashes = [_random_multihash(rng) for _ in range(120)]
        packed = [_pack_multihash(h) for h in hashes]

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                assert _packed_distance(packed[i], packed[j]) == int(hashes[i] - hashes[j])

    def test_matches_on_real_image_hashes(self, images_only_dir):
        names = ["img_a.png", "img_a_copy.png", "img_a_small.png", "img_b.png"]
        hashes = [
            get_image_hash(os.path.join(images_only_dir, n), VIDEO_EXTENSIONS)
            for n in names
        ]
        packed = [_pack_multihash(h) for h in hashes]

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                assert _packed_distance(packed[i], packed[j]) == int(hashes[i] - hashes[j])

    def test_is_the_max_of_both_components(self):
        rng = random.Random(5)
        for _ in range(50):
            a, b = _random_multihash(rng), _random_multihash(rng)
            assert _packed_distance(_pack_multihash(a), _pack_multihash(b)) == max(
                a.phash_distance(b), a.dhash_distance(b)
            )

    def test_identical_hashes_are_zero(self, images_only_dir):
        h = get_image_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        assert _packed_distance(_pack_multihash(h), _pack_multihash(h)) == 0

    def test_differing_only_in_dhash_is_still_counted(self):
        """A pair identical in pHash but far apart in dHash must not read as similar.

        This is the whole point of the dual hash; taking only the pHash distance
        would collapse them together.
        """
        same = imagehash.hex_to_hash("ffffffffffffffff")
        a = MultiHash(phash=same, dhash=imagehash.hex_to_hash("0000000000000000"))
        b = MultiHash(phash=same, dhash=imagehash.hex_to_hash("ffffffffffffffff"))
        assert _packed_distance(_pack_multihash(a), _pack_multihash(b)) == 64
        assert int(a - b) == 64

    def test_leading_zeros_do_not_affect_the_result(self):
        a = MultiHash(
            phash=imagehash.hex_to_hash("000000000000000f"),
            dhash=imagehash.hex_to_hash("0000000000000000"),
        )
        b = MultiHash(
            phash=imagehash.hex_to_hash("000000000000000e"),
            dhash=imagehash.hex_to_hash("0000000000000000"),
        )
        assert _packed_distance(_pack_multihash(a), _pack_multihash(b)) == int(a - b) == 1

    def test_pack_round_trips_through_the_string_form(self, images_only_dir):
        h = get_image_hash(os.path.join(images_only_dir, "img_a.png"), VIDEO_EXTENSIONS)
        assert _pack_multihash(MultiHash.from_str(str(h))) == _pack_multihash(h)


class TestClustering:
    def test_groups_identical_and_similar_files(self, images_only_dir):
        """End-to-end check that clustering still forms the expected groups."""
        finder = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        hashes = {}
        for name in ("img_a.png", "img_a_copy.png", "img_a_small.png", "img_b.png"):
            hashes[os.path.join(images_only_dir, name)] = get_image_hash(
                os.path.join(images_only_dir, name), VIDEO_EXTENSIONS
            )

        groups, stats = finder._cluster_with_bktree(hashes, 5, None)
        members = sorted(
            sorted(os.path.basename(p) for p in g) for g in groups.values()
        )
        assert members == [["img_a.png", "img_a_copy.png", "img_a_small.png"]]
        assert stats["total_groups"] == 1

    def test_unrelated_files_do_not_cluster(self, images_only_dir):
        finder = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        hashes = {}
        for i, name in enumerate(("img_a.png", "img_b.png")):
            hashes[os.path.join(images_only_dir, name)] = get_image_hash(
                os.path.join(images_only_dir, name), VIDEO_EXTENSIONS
            )
        groups, stats = finder._cluster_with_bktree(hashes, 5, None)
        assert groups == {}
        assert stats["total_groups"] == 0

    def test_files_without_hashes_are_skipped(self, images_only_dir):
        finder = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        groups, stats = finder._cluster_with_bktree({"/nope.png": None}, 5, None)
        assert groups == {}

    def test_exact_and_similar_groups_are_counted_separately(self, images_only_dir):
        finder = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        hashes = {}
        for name in ("img_a.png", "img_a_copy.png"):
            hashes[os.path.join(images_only_dir, name)] = get_image_hash(
                os.path.join(images_only_dir, name), VIDEO_EXTENSIONS
            )
        groups, stats = finder._cluster_with_bktree(hashes, 5, None)
        # Byte-identical copies hash identically, so this is an exact group
        assert stats["exact_groups"] == 1
        assert stats["similar_groups"] == 0


class TestFileWalk:
    def test_finds_nested_files(self, tmp_path):
        """Every level of the tree must be hashed, with keys relative to the root."""
        from deduper.utils.hash_cache import HashCache

        root = os.path.realpath(tmp_path / "nested")
        os.makedirs(os.path.join(root, "a", "b"))
        gradient_image(50, 50, 1).save(os.path.join(root, "top.png"))
        gradient_image(90, 90, 40).save(os.path.join(root, "a", "mid.png"))
        gradient_image(130, 130, 90).save(os.path.join(root, "a", "b", "deep.png"))

        DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS).find_duplicates(root)

        cache = HashCache(root)
        for relative in ("top.png", "a/mid.png", "a/b/deep.png"):
            assert cache.has_cached_hash(relative), f"walk missed {relative}"

    def test_symlinked_files_are_skipped(self, images_only_dir):
        """Symlinks point at a kept original and must not be rescanned as media."""
        finder = DuplicateFinder(IMAGE_EXTENSIONS, VIDEO_EXTENSIONS)
        finder.find_duplicates(images_only_dir)
        # The exact duplicate became a symlink on the first scan
        assert os.path.islink(os.path.join(images_only_dir, "img_a_copy.png"))
        images, _ = finder.find_duplicates(images_only_dir)
        for group in images:
            names = [os.path.basename(group["best_file"]["path"])]
            names += [os.path.basename(d["path"]) for d in group["duplicate_files"]]
            assert "img_a_copy.png" not in names
