#!/usr/bin/env python3
"""
build_dapt_corpus.py - Build training corpus from collected repositories

Processes collected repositories into a deduplicated, sanitized JSONL corpus
suitable for domain-adaptive pre-training.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.corpus.pack_jsonl import CorpusPacker
from src.data.corpus.dedupe import Deduplicator
from src.data.corpus.sanitize import Sanitizer
from src.data.corpus.chunking import Chunker
from src.common.logging import setup_logging, get_logger
from src.common.paths import get_data_dir, get_output_dir

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DAPT corpus from collected repositories"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dapt_train.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory containing cloned repos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for processed corpus",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1024 * 1024,  # 1MB
        help="Maximum file size in bytes",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Target chunk size in tokens",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=256,
        help="Overlap between chunks in tokens",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Skip deduplication step",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Skip sanitization step",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes",
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

    logger.info("Starting corpus build")

    # Determine paths
    input_dir = args.input or get_data_dir() / "repos"
    output_dir = args.output or get_data_dir() / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")

    # File extensions to include
    extensions = {
        ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
        ".json", ".md", ".yaml", ".yml"
    }

    try:
        # Step 1: Extract and filter files
        logger.info("Step 1: Extracting source files...")
        packer = CorpusPacker(
            extensions=extensions,
            max_file_size=args.max_file_size,
            workers=args.workers,
        )
        raw_corpus = packer.extract_files(input_dir)
        logger.info(f"Extracted {len(raw_corpus)} files")

        # Save raw corpus
        raw_file = output_dir / "raw_corpus.jsonl"
        packer.save_jsonl(raw_corpus, raw_file)
        logger.info(f"Raw corpus saved to {raw_file}")

        # Step 2: Sanitization (remove secrets, PII, etc.)
        if not args.no_sanitize:
            logger.info("Step 2: Sanitizing corpus...")
            sanitizer = Sanitizer()
            sanitized_corpus = sanitizer.process(raw_corpus)
            logger.info(f"Sanitized: {len(sanitized_corpus)} files remain")
        else:
            sanitized_corpus = raw_corpus

        # Step 3: Deduplication
        if not args.no_dedupe:
            logger.info("Step 3: Deduplicating corpus...")
            deduper = Deduplicator(
                threshold=0.8,
                num_perm=128,
            )
            deduped_corpus = deduper.deduplicate(sanitized_corpus)
            logger.info(f"Deduplicated: {len(deduped_corpus)} files remain")
        else:
            deduped_corpus = sanitized_corpus

        # Step 4: Chunking
        logger.info("Step 4: Chunking into training samples...")
        chunker = Chunker(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        chunks = chunker.chunk_corpus(deduped_corpus)
        logger.info(f"Created {len(chunks)} training chunks")

        # Save final corpus
        final_file = output_dir / "train_corpus.jsonl"
        with open(final_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        logger.info(f"Training corpus saved to {final_file}")

        # Save stats
        stats = {
            "raw_files": len(raw_corpus),
            "sanitized_files": len(sanitized_corpus),
            "deduped_files": len(deduped_corpus),
            "training_chunks": len(chunks),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
        }
        stats_file = output_dir / "corpus_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats saved to {stats_file}")

    except KeyboardInterrupt:
        logger.warning("Build interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Corpus build failed: {e}")
        return 1

    logger.info("Corpus build complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
