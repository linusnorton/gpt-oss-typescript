"""
GitHub repository collector for DAPT corpus building.

Searches GitHub for TypeScript/JavaScript repositories matching
quality and licensing criteria, then clones them locally.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from github import Github, GithubException, RateLimitExceededException
from github.Repository import Repository

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CollectionStats:
    """Statistics from a collection run."""

    repos_found: int = 0
    repos_cloned: int = 0
    repos_skipped: int = 0
    repos_failed: int = 0
    total_size_mb: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repos_found": self.repos_found,
            "repos_cloned": self.repos_cloned,
            "repos_skipped": self.repos_skipped,
            "repos_failed": self.repos_failed,
            "total_size_mb": round(self.total_size_mb, 2),
            "errors": self.errors[:10],  # Limit errors in output
        }

    def __str__(self) -> str:
        return (
            f"found={self.repos_found}, cloned={self.repos_cloned}, "
            f"skipped={self.repos_skipped}, failed={self.repos_failed}"
        )


class GitHubCollector:
    """Collect repositories from GitHub for training data."""

    def __init__(self, config_path: Path | str | None = None):
        """
        Initialize the collector.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)

        # Get GitHub token from environment
        token = os.environ.get("GITHUB_PAT")
        if not token:
            raise ValueError(
                "GITHUB_PAT environment variable not set. "
                "Create a token at https://github.com/settings/tokens"
            )

        self.github = Github(token)
        self._verify_token()

        # Load settings from config
        self.languages = self.config.get("languages", ["TypeScript", "JavaScript"])
        self.min_stars = self.config.get("min_stars", 100)
        self.max_repos = self.config.get("max_repos", 1000)
        self.allowed_licenses = self.config.get(
            "licenses", ["mit", "apache-2.0", "bsd-3-clause"]
        )

        # Filters
        filters = self.config.get("filters", {})
        self.skip_archived = filters.get("skip_archived", True)
        self.skip_forks = filters.get("skip_forks", True)
        self.max_size_mb = filters.get("max_size_mb", 100)
        self.max_days_since_push = filters.get("max_days_since_push", 365)
        self.required_files = filters.get("required_files", ["package.json"])

        # Rate limiting
        rate_limit = self.config.get("rate_limit", {})
        self.delay_seconds = rate_limit.get("delay_seconds", 1)

        # Output settings
        output = self.config.get("output", {})
        self.flatten_structure = output.get("flatten_structure", True)
        self.clone_depth = output.get("clone_depth", 1)
        self.save_metadata = output.get("save_metadata", True)

        # Search queries
        self.queries = self.config.get("queries", [])

    def _load_config(self, config_path: Path | str | None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            return {}

        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    def _verify_token(self) -> None:
        """Verify the GitHub token is valid."""
        try:
            user = self.github.get_user()
            logger.info(f"Authenticated as: {user.login}")
            rate_limit = self.github.get_rate_limit()
            # Handle both old and new PyGithub API
            if hasattr(rate_limit, 'core'):
                remaining = rate_limit.core.remaining
                limit = rate_limit.core.limit
            else:
                remaining = rate_limit.rate.remaining
                limit = rate_limit.rate.limit
            logger.info(f"Rate limit: {remaining}/{limit}")
        except GithubException as e:
            raise ValueError(f"Invalid GitHub token: {e}")

    def _wait_for_rate_limit(self) -> None:
        """Wait if rate limited."""
        rate_limit = self.github.get_rate_limit()
        # Handle both old and new PyGithub API
        if hasattr(rate_limit, 'core'):
            rate_info = rate_limit.core
        else:
            rate_info = rate_limit.rate

        if rate_info.remaining < 10:
            reset_time = rate_info.reset
            wait_seconds = (reset_time - datetime.now(timezone.utc)).total_seconds()
            if wait_seconds > 0:
                logger.warning(f"Rate limited, waiting {wait_seconds:.0f} seconds...")
                time.sleep(wait_seconds + 1)

    def _matches_criteria(self, repo: Repository) -> tuple[bool, str]:
        """
        Check if a repository matches collection criteria.

        Returns:
            Tuple of (matches, reason_if_not)
        """
        # Check if archived
        if self.skip_archived and repo.archived:
            return False, "archived"

        # Check if fork
        if self.skip_forks and repo.fork:
            return False, "fork"

        # Check size
        size_mb = repo.size / 1024  # GitHub reports size in KB
        if size_mb > self.max_size_mb:
            return False, f"too large ({size_mb:.1f}MB > {self.max_size_mb}MB)"

        # Check license
        if repo.license:
            license_key = repo.license.key.lower()
            if license_key not in [l.lower() for l in self.allowed_licenses]:
                return False, f"license not allowed ({license_key})"
        else:
            return False, "no license"

        # Check recent activity
        if repo.pushed_at:
            days_since_push = (
                datetime.now(timezone.utc) - repo.pushed_at.replace(tzinfo=timezone.utc)
            ).days
            if days_since_push > self.max_days_since_push:
                return False, f"inactive ({days_since_push} days since push)"

        return True, ""

    def search_repos(self) -> list[Repository]:
        """
        Search for repositories matching criteria.

        Returns:
            List of matching repositories
        """
        seen_repos: set[str] = set()
        repos: list[Repository] = []

        # Use configured queries or default language-based search
        queries = self.queries or [
            f"stars:>={self.min_stars} language:{lang}"
            for lang in self.languages
        ]

        for query in queries:
            if len(repos) >= self.max_repos:
                break

            logger.info(f"Searching: {query}")
            self._wait_for_rate_limit()

            try:
                results = self.github.search_repositories(
                    query=query,
                    sort="stars",
                    order="desc",
                )

                for repo in results:
                    if len(repos) >= self.max_repos:
                        break

                    # Skip duplicates
                    if repo.full_name in seen_repos:
                        continue
                    seen_repos.add(repo.full_name)

                    # Check criteria
                    matches, reason = self._matches_criteria(repo)
                    if matches:
                        repos.append(repo)
                        logger.debug(f"  Found: {repo.full_name} ({repo.stargazers_count} stars)")
                    else:
                        logger.debug(f"  Skipped: {repo.full_name} ({reason})")

                    # Rate limit delay
                    time.sleep(self.delay_seconds)

            except RateLimitExceededException:
                logger.warning("Rate limit exceeded, waiting...")
                self._wait_for_rate_limit()
            except GithubException as e:
                logger.error(f"Search error: {e}")

        logger.info(f"Found {len(repos)} repositories matching criteria")
        return repos

    def _clone_repo(
        self, repo: Repository, output_dir: Path
    ) -> tuple[bool, str | None]:
        """
        Clone a single repository.

        Returns:
            Tuple of (success, error_message)
        """
        # Determine target directory
        if self.flatten_structure:
            repo_dir = output_dir / f"{repo.owner.login}_{repo.name}"
        else:
            repo_dir = output_dir / repo.owner.login / repo.name

        # Skip if already cloned
        if repo_dir.exists():
            logger.debug(f"Already cloned: {repo.full_name}")
            return True, None

        # Clone with depth
        clone_url = repo.clone_url
        cmd = ["git", "clone", "--depth", str(self.clone_depth), clone_url, str(repo_dir)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                return False, result.stderr.strip()

            # Save metadata
            if self.save_metadata:
                metadata = {
                    "full_name": repo.full_name,
                    "url": repo.html_url,
                    "clone_url": repo.clone_url,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "language": repo.language,
                    "license": repo.license.key if repo.license else None,
                    "description": repo.description,
                    "topics": repo.topics,
                    "created_at": repo.created_at.isoformat() if repo.created_at else None,
                    "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(repo_dir / "repo_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

            return True, None

        except subprocess.TimeoutExpired:
            return False, "clone timeout"
        except Exception as e:
            return False, str(e)

    def collect(self, output_dir: Path | str) -> CollectionStats:
        """
        Search and clone repositories.

        Args:
            output_dir: Directory to clone repositories into

        Returns:
            Collection statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = CollectionStats()

        # Search for repos
        repos = self.search_repos()
        stats.repos_found = len(repos)

        # Clone each repo
        for i, repo in enumerate(repos):
            logger.info(f"[{i + 1}/{len(repos)}] Cloning {repo.full_name}...")

            success, error = self._clone_repo(repo, output_dir)

            if success:
                stats.repos_cloned += 1
                stats.total_size_mb += repo.size / 1024
            elif error:
                stats.repos_failed += 1
                stats.errors.append(f"{repo.full_name}: {error}")
                logger.error(f"  Failed: {error}")

            # Rate limit delay
            time.sleep(self.delay_seconds)

        return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect GitHub repositories")
    parser.add_argument("--config", type=Path, help="Config file path")
    parser.add_argument("--output", type=Path, default=Path("data/repos"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    collector = GitHubCollector(config_path=args.config)

    if args.dry_run:
        repos = collector.search_repos()
        for repo in repos:
            print(f"{repo.full_name} ({repo.stargazers_count} stars)")
    else:
        stats = collector.collect(args.output)
        print(f"Collection complete: {stats}")


if __name__ == "__main__":
    main()
