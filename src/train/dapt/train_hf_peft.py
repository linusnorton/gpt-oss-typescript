"""
Training with HuggingFace Transformers + PEFT (QLoRA).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from ...common.logging import get_logger
from ...common.paths import get_checkpoint_dir, get_hf_token

logger = get_logger(__name__)


@dataclass
class DAPTConfig:
    """Configuration for DAPT training."""

    # Model
    model_id: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    trust_remote_code: bool = True

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
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
    gradient_checkpointing: bool = True
    bf16: bool = True
    tf32: bool = True


class DAPTTrainer:
    """
    Domain-Adaptive Pre-Training trainer using QLoRA.

    Enables fine-tuning of large models on a single GPU.
    """

    def __init__(
        self,
        model_id: str,
        output_dir: Path,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        max_steps: Optional[int] = None,
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
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            max_steps=max_steps or -1,
        )

        self.use_wandb = use_wandb
        self.resume_from = resume_from

        # Will be initialized in train()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _setup_quantization(self) -> BitsAndBytesConfig:
        """Configure 4-bit quantization."""
        compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)

        return BitsAndBytesConfig(
            load_in_4bit=self.config.use_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config.use_double_quant,
        )

    def _setup_lora(self) -> LoraConfig:
        """Configure LoRA adapter."""
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

    def _load_model(self) -> None:
        """Load and prepare model for training."""
        logger.info(f"Loading model: {self.model_id}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.config.trust_remote_code,
            token=get_hf_token(),
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        bnb_config = self._setup_quantization()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
            token=get_hf_token(),
            torch_dtype=torch.bfloat16,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
        )

        # Add LoRA adapter
        lora_config = self._setup_lora()
        self.model = get_peft_model(self.model, lora_config)

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
            tf32=self.config.tf32,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim="paged_adamw_32bit",
            report_to="wandb" if self.use_wandb else "none",
            run_name=f"dapt-{self.model_id.split('/')[-1]}",
            remove_unused_columns=True,
            dataloader_num_workers=4,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")

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

        # Save LoRA adapter
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save config
        with open(save_path / "training_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

    def push_to_hub(self, repo_id: str) -> None:
        """Push model to HuggingFace Hub."""
        logger.info(f"Pushing to {repo_id}")
        self.model.push_to_hub(repo_id, token=get_hf_token())
        self.tokenizer.push_to_hub(repo_id, token=get_hf_token())


def main():
    """CLI entry point."""
    import argparse
    from ...common.logging import setup_logging

    parser = argparse.ArgumentParser(description="Run DAPT training")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=True)

    trainer = DAPTTrainer(
        model_id=args.model_id,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_wandb=args.wandb,
    )

    trainer.train(args.train_data)


if __name__ == "__main__":
    main()
