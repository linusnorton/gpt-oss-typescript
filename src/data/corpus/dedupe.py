"""
Corpus deduplication using MinHash LSH.

Removes near-duplicate files based on content similarity
to improve training data quality.
"""

import hashlib
import re
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


class MinHash:
    """MinHash signature for similarity estimation."""

    def __init__(self, num_perm: int = 128, seed: int = 1):
        """
        Initialize MinHash.

        Args:
            num_perm: Number of permutations (hash functions)
            seed: Random seed for reproducibility
        """
        self.num_perm = num_perm
        self.seed = seed
        # Large prime for hash computation
        self._mersenne_prime = (1 << 61) - 1
        self._max_hash = (1 << 32) - 1

        # Generate random coefficients for hash functions
        import random
        random.seed(seed)
        self._a = [random.randint(1, self._mersenne_prime - 1) for _ in range(num_perm)]
        self._b = [random.randint(0, self._mersenne_prime - 1) for _ in range(num_perm)]

    def _hash_token(self, token: str) -> int:
        """Hash a token to an integer."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)

    def compute(self, tokens: set[str]) -> list[int]:
        """
        Compute MinHash signature for a set of tokens.

        Args:
            tokens: Set of tokens (shingles)

        Returns:
            MinHash signature as list of integers
        """
        if not tokens:
            return [self._max_hash] * self.num_perm

        signature = [self._max_hash] * self.num_perm
        token_hashes = [self._hash_token(t) for t in tokens]

        for i in range(self.num_perm):
            for h in token_hashes:
                # Apply hash function: (a * h + b) mod prime
                hash_val = ((self._a[i] * h + self._b[i]) % self._mersenne_prime) & self._max_hash
                if hash_val < signature[i]:
                    signature[i] = hash_val

        return signature


def jaccard_similarity(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must have same length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


class Deduplicator:
    """Remove near-duplicate files from corpus."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        """
        Initialize deduplicator.

        Args:
            threshold: Jaccard similarity threshold for duplicates
            num_perm: Number of MinHash permutations
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.minhash = MinHash(num_perm=num_perm)

        # Shingle size (n-gram of characters)
        self.shingle_size = 5

    def _tokenize(self, content: str) -> set[str]:
        """
        Convert content to shingles (character n-grams).

        Args:
            content: Text content

        Returns:
            Set of shingles
        """
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content.strip().lower())

        if len(content) < self.shingle_size:
            return {content} if content else set()

        # Generate shingles
        shingles = set()
        for i in range(len(content) - self.shingle_size + 1):
            shingles.add(content[i:i + self.shingle_size])

        return shingles

    def deduplicate(self, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Remove near-duplicate files from corpus.

        Uses MinHash signatures with a simple pairwise comparison
        approach. For very large corpora, consider using LSH bands.

        Args:
            corpus: List of file dictionaries

        Returns:
            Deduplicated corpus
        """
        if not corpus:
            return []

        logger.info(f"Computing MinHash signatures for {len(corpus)} files...")

        # Compute signatures for all files
        signatures = []
        for item in corpus:
            content = item.get("content", "")
            tokens = self._tokenize(content)
            sig = self.minhash.compute(tokens)
            signatures.append(sig)

        # Track which items to keep (not duplicates)
        keep = [True] * len(corpus)
        duplicates_found = 0

        logger.info("Finding duplicates...")

        # Use LSH-style banding for efficiency
        # Split signature into bands and hash each band
        num_bands = 20
        rows_per_band = self.num_perm // num_bands

        # Build band hash tables
        band_buckets: list[dict[int, list[int]]] = [
            {} for _ in range(num_bands)
        ]

        for idx, sig in enumerate(signatures):
            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band = tuple(sig[start:end])
                band_hash = hash(band)

                if band_hash not in band_buckets[band_idx]:
                    band_buckets[band_idx][band_hash] = []
                band_buckets[band_idx][band_hash].append(idx)

        # Find candidate pairs (items that share at least one band)
        candidate_pairs: set[tuple[int, int]] = set()
        for bucket_dict in band_buckets:
            for indices in bucket_dict.values():
                if len(indices) > 1:
                    for i, idx1 in enumerate(indices):
                        for idx2 in indices[i + 1:]:
                            if idx1 < idx2:
                                candidate_pairs.add((idx1, idx2))
                            else:
                                candidate_pairs.add((idx2, idx1))

        logger.info(f"Found {len(candidate_pairs)} candidate pairs to check")

        # Check candidate pairs for actual similarity
        for idx1, idx2 in candidate_pairs:
            if not keep[idx2]:  # Already marked as duplicate
                continue

            sim = jaccard_similarity(signatures[idx1], signatures[idx2])
            if sim >= self.threshold:
                # Mark the later one as duplicate (keep earlier/first seen)
                keep[idx2] = False
                duplicates_found += 1

        # Filter corpus
        deduped = [item for item, k in zip(corpus, keep) if k]

        logger.info(f"Removed {duplicates_found} near-duplicates")
        return deduped
