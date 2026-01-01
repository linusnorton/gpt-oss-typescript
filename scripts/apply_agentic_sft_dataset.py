#!/usr/bin/env python3
"""
apply_agentic_sft_dataset.py - Prepare agentic SFT dataset

Downloads and processes the Nemotron-Agentic-v1 dataset for supervised
fine-tuning on tool use and instruction following.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hf_download import download_dataset
from src.train.sft.format_agentic import AgenticFormatter
from src.common.logging import setup_logging, get_logger
from src.common.paths import get_data_dir

logger = get_logger(__name__)


# Default datasets for agentic SFT
AGENTIC_DATASETS = [
    "nvidia/Nemotron-Agentic-v1",
    # Add other relevant datasets here
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare agentic SFT dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sft_data.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=AGENTIC_DATASETS,
        help="Dataset IDs to download",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--filter-typescript",
        action="store_true",
        help="Filter to TypeScript/JavaScript related examples",
    )
    parser.add_argument(
        "--format",
        choices=["chatml", "alpaca", "sharegpt", "gptoss"],
        default="chatml",
        help="Output format for conversations",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.95,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger.info("Starting agentic SFT data preparation")

    # Load config
    config = load_config(args.config)

    # Determine output directory
    output_dir = args.output or get_data_dir() / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    all_examples = []

    for dataset_id in args.datasets:
        logger.info(f"Processing dataset: {dataset_id}")

        try:
            # Download dataset
            dataset = download_dataset(
                dataset_id,
                cache_dir=get_data_dir() / "hf_cache",
            )

            if dataset is None:
                logger.warning(f"Failed to download {dataset_id}")
                continue

            logger.info(f"Downloaded {len(dataset)} examples from {dataset_id}")

            # Format examples
            formatter = AgenticFormatter(
                format_type=args.format,
                filter_typescript=args.filter_typescript,
            )

            formatted = formatter.format_dataset(dataset)
            logger.info(f"Formatted {len(formatted)} examples")

            all_examples.extend(formatted)

        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            continue

    if not all_examples:
        logger.error("No examples collected")
        return 1

    logger.info(f"Total examples: {len(all_examples)}")

    # Apply max samples limit
    if args.max_samples and len(all_examples) > args.max_samples:
        import random
        random.shuffle(all_examples)
        all_examples = all_examples[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    # Split train/val
    import random
    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * args.train_split)
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")

    # Save datasets
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "val.jsonl"

    with open(train_file, "w") as f:
        for example in train_data:
            f.write(json.dumps(example) + "\n")

    with open(val_file, "w") as f:
        for example in val_data:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Training data saved to {train_file}")
    logger.info(f"Validation data saved to {val_file}")

    # Save stats
    stats = {
        "datasets": args.datasets,
        "total_examples": len(all_examples),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "format": args.format,
        "filter_typescript": args.filter_typescript,
    }
    stats_file = output_dir / "sft_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("SFT data preparation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
