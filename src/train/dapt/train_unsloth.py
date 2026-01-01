"""
Training with Unsloth for fast QLoRA fine-tuning.

Uses Unsloth's FastLanguageModel for 2x faster training and
significantly reduced memory usage.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from unsloth import FastLanguageModel

from ...common.logging import get_logger
from ...common.paths import get_checkpoint_dir, get_hf_token

logger = get_logger(__name__)


@dataclass
class DAPTConfig:
    """Configuration for DAPT training with Unsloth."""

    # Model
    model_id: str = "unsloth/gpt-oss-20b"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # Unsloth optimizes with 0 dropout
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    # Training
    batch_size: int = 2  # Can use larger batches with Unsloth
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4  # Slightly higher LR works with Unsloth
    num_epochs: int = 3
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence
    max_seq_length: int = 4096

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10

    # Hardware
    bf16: bool = True


class DAPTTrainer:
    """
    Domain-Adaptive Pre-Training trainer using Unsloth.

    Enables fast QLoRA fine-tuning with significantly reduced
    memory usage (~14GB for gpt-oss-20b).
    """

    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        lora_r: int = 64,
        lora_alpha: int = 16,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        max_steps: Optional[int] = None,
        max_seq_length: int = 4096,
        use_wandb: bool = False,
        resume_from: Optional[Path] = None,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create config
        self.config = DAPTConfig(
            model_id=model_id,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_steps=max_steps or -1,
            max_seq_length=max_seq_length,
        )

        self.use_wandb = use_wandb
        self.resume_from = resume_from

        # Will be initialized in train()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _load_model(self) -> None:
        """Load and prepare model for training with Unsloth."""
        logger.info(f"Loading model with Unsloth: {self.model_id}")

        # Load model and tokenizer together via Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect (bfloat16 for Ampere+)
            load_in_4bit=True,  # QLoRA quantization
            token=get_hf_token(),
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA via Unsloth's optimized method
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth optimized
            random_state=42,
        )

        # Print trainable parameters
        self.model.print_trainable_parameters()

    def _load_dataset(self, data_path: Path) -> Dataset:
        """Load training dataset from JSONL."""
        logger.info(f"Loading dataset from {data_path}")

        # Load JSONL file
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))

        dataset = Dataset.from_list(data)
        logger.info(f"Loaded {len(dataset)} examples")

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )

        tokenized = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        return tokenized

    def train(self, data_path: Path) -> None:
        """
        Run training.

        Args:
            data_path: Path to training data JSONL
        """
        # Load model if not already loaded
        if self.model is None:
            self._load_model()

        # Load dataset
        train_dataset = self._load_dataset(data_path)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            bf16=self.config.bf16,
            optim="adamw_8bit",  # Unsloth-compatible optimizer
            report_to="wandb" if self.use_wandb else "none",
            run_name=f"dapt-{self.model_id.split('/')[-1]}",
            remove_unused_columns=True,
            dataloader_num_workers=4,
            seed=42,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training with Unsloth...")

        if self.resume_from:
            self.trainer.train(resume_from_checkpoint=str(self.resume_from))
        else:
            self.trainer.train()

        # Save final checkpoint
        self.save_checkpoint()

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """Save model checkpoint."""
        save_path = path or self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to {save_path}")

        # Save LoRA adapter (PEFT-compatible format)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save config
        with open(save_path / "training_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

    def save_merged(self, path: Optional[Path] = None, quantization: Optional[str] = None) -> None:
        """
        Save merged model (base + LoRA) for inference.

        Args:
            path: Output path
            quantization: Optional quantization method for export
                - None: Save in 16-bit (for vLLM)
                - "q4_k_m": GGUF Q4_K_M (for llama.cpp)
        """
        save_path = path or self.output_dir / "merged"
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving merged model to {save_path}")

        if quantization:
            # Save as GGUF for llama.cpp
            self.model.save_pretrained_gguf(
                save_path,
                self.tokenizer,
                quantization_method=quantization,
            )
        else:
            # Save merged 16-bit model for vLLM
            self.model.save_pretrained_merged(
                save_path,
                self.tokenizer,
                save_method="merged_16bit",
            )

    def push_to_hub(self, repo_id: str, merged: bool = False) -> None:
        """Push model to HuggingFace Hub."""
        logger.info(f"Pushing to {repo_id}")

        if merged:
            self.model.push_to_hub_merged(
                repo_id,
                self.tokenizer,
                save_method="merged_16bit",
                token=get_hf_token(),
            )
        else:
            self.model.push_to_hub(repo_id, token=get_hf_token())
            self.tokenizer.push_to_hub(repo_id, token=get_hf_token())


def main():
    """CLI entry point."""
    import argparse
    from ...common.logging import setup_logging

    parser = argparse.ArgumentParser(description="Run DAPT training with Unsloth")
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/gpt-oss-20b",
        help="Model ID (default: unsloth/gpt-oss-20b)"
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--save-merged", action="store_true", help="Save merged model for inference")
    args = parser.parse_args()

    setup_logging(verbose=True)

    trainer = DAPTTrainer(
        model_id=args.model_id,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        use_wandb=args.wandb,
    )

    trainer.train(args.train_data)

    if args.save_merged:
        trainer.save_merged()


if __name__ == "__main__":
    main()
