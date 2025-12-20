#!/usr/bin/env python3
"""
collect_github_repos.py - Collect TypeScript/JavaScript repositories from GitHub

This script searches for and clones permissively licensed repositories
for use in domain-adaptive pre-training (DAPT).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.github.github_client import GitHubCollector
from src.common.logging import setup_logging, get_logger
from src.common.paths import get_data_dir, get_output_dir

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TypeScript/JavaScript repositories from GitHub"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/github_collection.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for cloned repos",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Maximum number of repos to collect (overrides config)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Languages to collect (default: TypeScript JavaScript)",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=None,
        help="Minimum stars filter",
    )
    parser.add_argument(
        "--licenses",
        type=str,
        nargs="+",
        default=None,
        help="Allowed licenses (e.g., mit apache-2.0 bsd-3-clause)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search but don't clone repositories",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger.info("Starting GitHub repository collection")

    # Determine output directory
    output_dir = args.output or get_data_dir() / "repos"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize collector
    try:
        collector = GitHubCollector(config_path=args.config)
    except Exception as e:
        logger.error(f"Failed to initialize collector: {e}")
        return 1

    # Override config with CLI args
    if args.max_repos is not None:
        collector.max_repos = args.max_repos
    if args.languages is not None:
        collector.languages = args.languages
    if args.min_stars is not None:
        collector.min_stars = args.min_stars
    if args.licenses is not None:
        collector.allowed_licenses = args.licenses

    # Run collection
    try:
        if args.dry_run:
            logger.info("Dry run mode - searching without cloning")
            repos = collector.search_repos()
            logger.info(f"Found {len(repos)} repositories matching criteria")

            # Save search results
            results_file = output_dir / "search_results.json"
            with open(results_file, "w") as f:
                json.dump(
                    [
                        {
                            "name": r.full_name,
                            "url": r.html_url,
                            "stars": r.stargazers_count,
                            "language": r.language,
                            "license": r.license.key if r.license else None,
                        }
                        for r in repos
                    ],
                    f,
                    indent=2,
                )
            logger.info(f"Search results saved to {results_file}")
        else:
            stats = collector.collect(output_dir=output_dir)
            logger.info(f"Collection complete: {stats}")

            # Save collection stats
            stats_file = output_dir / "collection_stats.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

    except KeyboardInterrupt:
        logger.warning("Collection interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return 1

    logger.info("Repository collection complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
