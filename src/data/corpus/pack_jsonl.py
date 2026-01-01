"""
Corpus packer for extracting source files from repositories.

Walks through cloned repositories and extracts source files into
a JSONL format suitable for training.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SourceFile:
    """A single source file extracted from a repository."""

    path: str
    content: str
    repo: str
    language: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "repo": self.repo,
            "language": self.language,
            "size": self.size,
        }


# Map file extensions to language names
EXTENSION_LANGUAGE_MAP = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".json": "json",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _extract_from_repo(args: tuple[Path, set[str], int]) -> list[dict[str, Any]]:
    """
    Extract files from a single repository.

    This is a standalone function for multiprocessing.
    """
    repo_dir, extensions, max_file_size = args
    files = []
    repo_name = repo_dir.name

    # Directories to skip
    skip_dirs = {
        "node_modules", ".git", "dist", "build", "out", ".next",
        "coverage", ".nyc_output", "__pycache__", ".cache",
        "vendor", "bower_components", ".svn", ".hg",
    }

    for root, dirs, filenames in os.walk(repo_dir):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for filename in filenames:
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()

            if ext not in extensions:
                continue

            # Check file size
            try:
                size = filepath.stat().st_size
                if size > max_file_size or size == 0:
                    continue
            except OSError:
                continue

            # Read file content
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            # Skip empty or binary files
            if not content or "\x00" in content:
                continue

            # Get relative path within repo
            rel_path = str(filepath.relative_to(repo_dir))

            files.append({
                "path": rel_path,
                "content": content,
                "repo": repo_name,
                "language": EXTENSION_LANGUAGE_MAP.get(ext, "unknown"),
                "size": size,
            })

    return files


class CorpusPacker:
    """Extract and pack source files from repositories."""

    def __init__(
        self,
        extensions: set[str] | None = None,
        max_file_size: int = 1024 * 1024,
        workers: int = 4,
    ):
        """
        Initialize the packer.

        Args:
            extensions: Set of file extensions to include
            max_file_size: Maximum file size in bytes
            workers: Number of worker processes
        """
        self.extensions = extensions or {
            ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
            ".json", ".md", ".yaml", ".yml"
        }
        self.max_file_size = max_file_size
        self.workers = workers

    def extract_files(self, input_dir: Path) -> list[dict[str, Any]]:
        """
        Extract source files from all repositories.

        Args:
            input_dir: Directory containing cloned repositories

        Returns:
            List of file dictionaries
        """
        input_dir = Path(input_dir)

        # Find all repository directories
        repo_dirs = [
            d for d in input_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        logger.info(f"Found {len(repo_dirs)} repositories")

        all_files = []

        # Process repos in parallel
        args_list = [
            (repo_dir, self.extensions, self.max_file_size)
            for repo_dir in repo_dirs
        ]

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(_extract_from_repo, args): args[0]
                for args in args_list
            }

            for i, future in enumerate(as_completed(futures)):
                repo_dir = futures[future]
                try:
                    files = future.result()
                    all_files.extend(files)
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(repo_dirs)} repos")
                except Exception as e:
                    logger.warning(f"Failed to process {repo_dir.name}: {e}")

        logger.info(f"Extracted {len(all_files)} files from {len(repo_dirs)} repos")
        return all_files

    def save_jsonl(self, corpus: list[dict[str, Any]], output_path: Path) -> None:
        """
        Save corpus to JSONL file.

        Args:
            corpus: List of file dictionaries
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in corpus:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(corpus)} items to {output_path}")
