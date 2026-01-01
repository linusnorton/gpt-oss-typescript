"""
Corpus chunking for training data preparation.

Packs entire repositories into single training samples,
preserving cross-file context for large context models.
"""

from collections import defaultdict
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


# File path header template
FILE_HEADER = "// FILE: {path}\n"
FILE_SEPARATOR = "\n\n"


class Chunker:
    """Pack repository files into training samples."""

    def __init__(
        self,
        chunk_size: int = 500000,  # 500K tokens default for large context
        overlap: int = 0,  # No overlap for repo packing
        min_chunk_size: int = 1000,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Maximum tokens per sample (for splitting large repos)
            overlap: Unused, kept for API compatibility
            min_chunk_size: Minimum tokens for a sample to be included
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        # Approximate chars per token (rough estimate for code)
        self.chars_per_token = 3.5

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) / self.chars_per_token)

    def _format_file(self, item: dict[str, Any]) -> str:
        """Format a file with its path header."""
        path = item.get("path", "unknown")
        content = item.get("content", "")
        return FILE_HEADER.format(path=path) + content

    def _sort_files(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sort files in logical order for training.

        Priority:
        1. Package.json / config files first (project context)
        2. Type definitions
        3. Source files by path depth (shallower first)
        4. Test files last
        """
        def sort_key(item: dict[str, Any]) -> tuple:
            path = item.get("path", "").lower()

            # Config files first
            if path == "package.json":
                return (0, 0, path)
            if path in ("tsconfig.json", "jsconfig.json"):
                return (0, 1, path)
            if path.endswith((".json", ".yaml", ".yml")) and "/" not in path:
                return (0, 2, path)

            # Type definition files
            if path.endswith(".d.ts"):
                return (1, 0, path)
            if "/types/" in path or "/typings/" in path:
                return (1, 1, path)

            # Test files last
            if any(x in path for x in ["/test/", "/tests/", "/__tests__/", ".test.", ".spec."]):
                return (4, path.count("/"), path)

            # Regular source files by depth
            return (2, path.count("/"), path)

        return sorted(files, key=sort_key)

    def _pack_repo(self, repo_name: str, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Pack all files from a repo into training samples.

        Args:
            repo_name: Name of the repository
            files: List of file dictionaries from this repo

        Returns:
            List of training samples (usually 1 per repo)
        """
        if not files:
            return []

        # Sort files in logical order
        sorted_files = self._sort_files(files)

        # Calculate max chars per sample
        max_chars = int(self.chunk_size * self.chars_per_token)
        min_chars = int(self.min_chunk_size * self.chars_per_token)

        samples = []
        current_parts = []
        current_size = 0
        current_file_count = 0

        for item in sorted_files:
            formatted = self._format_file(item)
            file_size = len(formatted) + len(FILE_SEPARATOR)

            # If this single file exceeds max, truncate it
            if file_size > max_chars:
                # Save current sample if any
                if current_parts:
                    text = FILE_SEPARATOR.join(current_parts)
                    if len(text) >= min_chars:
                        samples.append({
                            "text": text,
                            "repo": repo_name,
                            "file_count": current_file_count,
                            "sample_index": len(samples),
                        })
                    current_parts = []
                    current_size = 0
                    current_file_count = 0

                # Add truncated file as its own sample
                truncated = formatted[:max_chars]
                if len(truncated) >= min_chars:
                    samples.append({
                        "text": truncated,
                        "repo": repo_name,
                        "file_count": 1,
                        "sample_index": len(samples),
                        "truncated": True,
                    })
                continue

            # If adding this file would exceed max, start new sample
            if current_size + file_size > max_chars and current_parts:
                text = FILE_SEPARATOR.join(current_parts)
                if len(text) >= min_chars:
                    samples.append({
                        "text": text,
                        "repo": repo_name,
                        "file_count": current_file_count,
                        "sample_index": len(samples),
                    })
                current_parts = []
                current_size = 0
                current_file_count = 0

            current_parts.append(formatted)
            current_size += file_size
            current_file_count += 1

        # Don't forget the last sample
        if current_parts:
            text = FILE_SEPARATOR.join(current_parts)
            if len(text) >= min_chars:
                samples.append({
                    "text": text,
                    "repo": repo_name,
                    "file_count": current_file_count,
                    "sample_index": len(samples),
                })

        # Update total_samples count
        for sample in samples:
            sample["total_samples"] = len(samples)

        return samples

    def chunk_corpus(self, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Pack corpus files into repo-level training samples.

        Args:
            corpus: List of file dictionaries

        Returns:
            List of training samples (one or more per repo)
        """
        # Group files by repository
        repos: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in corpus:
            repo = item.get("repo", "unknown")
            repos[repo].append(item)

        logger.info(f"Packing {len(corpus)} files from {len(repos)} repositories")

        all_samples = []
        single_sample_repos = 0
        multi_sample_repos = 0

        for repo_name, files in repos.items():
            samples = self._pack_repo(repo_name, files)
            all_samples.extend(samples)

            if len(samples) == 1:
                single_sample_repos += 1
            elif len(samples) > 1:
                multi_sample_repos += 1

        logger.info(
            f"Created {len(all_samples)} training samples from {len(repos)} repos "
            f"({single_sample_repos} single-sample, {multi_sample_repos} multi-sample)"
        )

        # Log some stats
        if all_samples:
            avg_files = sum(s.get("file_count", 0) for s in all_samples) / len(all_samples)
            avg_tokens = sum(self._estimate_tokens(s["text"]) for s in all_samples) / len(all_samples)
            logger.info(f"Average: {avg_files:.1f} files/sample, {avg_tokens:.0f} tokens/sample")

        return all_samples
