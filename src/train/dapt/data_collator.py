"""
Data collator for language modeling training.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator for causal language modeling (next token prediction).

    Handles padding, attention masks, and label creation for autoregressive training.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"
    mlm: bool = False  # Causal LM, not masked LM

    def __call__(
        self, examples: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.

        Args:
            examples: List of example dictionaries with 'input_ids' or 'text'

        Returns:
            Batch dictionary with input_ids, attention_mask, labels
        """
        # Handle different input formats
        if "input_ids" in examples[0]:
            input_ids = [e["input_ids"] for e in examples]
        elif "text" in examples[0]:
            # Tokenize on the fly
            texts = [e["text"] for e in examples]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=False,
                return_tensors=None,
            )
            input_ids = tokenized["input_ids"]
        else:
            raise ValueError("Examples must have 'input_ids' or 'text' field")

        # Pad sequences
        batch = self._pad_sequences(input_ids)

        # Create labels (same as input_ids for causal LM)
        # Shift happens inside the model
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding tokens in labels
        batch["labels"][batch["attention_mask"] == 0] = -100

        return batch

    def _pad_sequences(
        self, sequences: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """Pad sequences to the same length."""
        max_len = max(len(seq) for seq in sequences)

        # Round up to multiple
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        max_len = min(max_len, self.max_length)

        # Pad sequences
        padded_ids = []
        attention_masks = []

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for seq in sequences:
            # Truncate if needed
            if len(seq) > max_len:
                seq = seq[:max_len]

            # Calculate padding
            padding_length = max_len - len(seq)

            # Pad on right
            padded = seq + [pad_token_id] * padding_length
            mask = [1] * len(seq) + [0] * padding_length

            padded_ids.append(padded)
            attention_masks.append(mask)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


@dataclass
class DataCollatorForPackedSequences:
    """
    Data collator for packed sequences (multiple documents per sequence).

    Used for more efficient training by packing short documents together.
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 4096
    return_tensors: str = "pt"

    def __call__(
        self, examples: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Pack multiple examples into sequences."""
        all_input_ids = []
        all_labels = []

        current_ids = []
        current_labels = []

        # Separator token
        sep_token = self.tokenizer.eos_token_id

        for example in examples:
            if "input_ids" in example:
                ids = example["input_ids"]
            else:
                ids = self.tokenizer.encode(
                    example["text"],
                    add_special_tokens=False,
                )

            # Add separator
            ids = ids + [sep_token]

            # Check if fits in current sequence
            if len(current_ids) + len(ids) <= self.max_length:
                current_ids.extend(ids)
                current_labels.extend(ids)
            else:
                # Save current sequence if not empty
                if current_ids:
                    all_input_ids.append(current_ids)
                    all_labels.append(current_labels)

                # Start new sequence
                current_ids = ids[:self.max_length]
                current_labels = ids[:self.max_length]

        # Don't forget last sequence
        if current_ids:
            all_input_ids.append(current_ids)
            all_labels.append(current_labels)

        # Pad to same length
        max_len = max(len(seq) for seq in all_input_ids)

        padded_ids = []
        padded_labels = []
        attention_masks = []

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for ids, labels in zip(all_input_ids, all_labels):
            padding = max_len - len(ids)
            padded_ids.append(ids + [pad_token_id] * padding)
            padded_labels.append(labels + [-100] * padding)
            attention_masks.append([1] * len(ids) + [0] * padding)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
