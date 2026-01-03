#!/usr/bin/env python3
"""
run_dapt_train.py - Domain-Adaptive Pre-Training with QLoRA

Runs continued pre-training on TypeScript/JavaScript corpus using
QLoRA for memory-efficient single-GPU training.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train.dapt.train_hf_peft import DAPTTrainer
from src.common.logging import setup_logging, get_logger
from src.common.paths import get_data_dir, get_output_dir

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run domain-adaptive pre-training with QLoRA"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dapt_train.yaml"),
        help="Path to training configuration",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Base model ID (overrides config)",
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Path to training data JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Model ID for HuggingFace Hub",
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
        logger.warning(f"Config file not found: {config_path}")
        return {}

    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    args = parse_args()
    setup_logging(verbose=args.verbose)

    logger.info("Starting DAPT training")

    # Load base config
    config = load_config(args.config)
    training_config = config.get("training", {})

    # Override config with CLI args
    model_id = args.model_id or config.get("model_id") or os.getenv(
        "MODEL_ID", "unsloth/gpt-oss-20b"
    )

    train_data = args.train_data or Path(
        config.get("train_data", str(get_data_dir() / "corpus" / "train_corpus.jsonl"))
    )

    output_dir = args.output_dir or Path(
        config.get("output_dir", str(get_output_dir() / "checkpoints" / "dapt"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training hyperparameters (from CLI, then config.training, then env)
    batch_size = (
        args.batch_size
        or training_config.get("batch_size")
        or int(os.getenv("TRAIN_BATCH_SIZE", "1"))
    )
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        or training_config.get("gradient_accumulation_steps")
        or int(os.getenv("TRAIN_GRADIENT_ACCUMULATION_STEPS", "8"))
    )
    learning_rate = (
        args.learning_rate
        or training_config.get("learning_rate")
        or float(os.getenv("TRAIN_LEARNING_RATE", "2e-4"))
    )
    num_epochs = (
        args.num_epochs
        or training_config.get("num_epochs")
        or int(os.getenv("TRAIN_NUM_EPOCHS", "3"))
    )

    # LoRA config (nested under 'lora' in YAML)
    lora_config = config.get("lora", {})
    lora_r = (
        args.lora_r
        or lora_config.get("r")
        or int(os.getenv("QLORA_R", "64"))
    )
    lora_alpha = (
        args.lora_alpha
        or lora_config.get("alpha")
        or int(os.getenv("QLORA_ALPHA", "16"))
    )
    lora_dropout = lora_config.get("dropout", float(os.getenv("QLORA_DROPOUT", "0.0")))

    logger.info(f"Model: {model_id}")
    logger.info(f"Training data: {train_data}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"LoRA rank: {lora_r}, alpha: {lora_alpha}")

    # Validate training data exists
    if not train_data.exists():
        logger.error(f"Training data not found: {train_data}")
        logger.info("Run 'make build-corpus' first to create training data")
        return 1

    # Initialize trainer
    try:
        trainer = DAPTTrainer(
            model_id=model_id,
            output_dir=output_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_steps=args.max_steps,
            use_wandb=args.wandb,
            resume_from=args.resume_from,
        )
    except Exception as e:
        logger.exception(f"Failed to initialize trainer: {e}")
        return 1

    # Run training
    try:
        trainer.train(train_data)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint()
        return 130
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1

    # Push to hub if requested
    if args.push_to_hub and args.hub_model_id:
        try:
            trainer.push_to_hub(args.hub_model_id)
            logger.info(f"Model pushed to {args.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")

    logger.info("DAPT training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
