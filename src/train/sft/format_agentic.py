"""
Format agentic datasets for SFT training.
"""

import json
from typing import Any, Iterator

from ...common.logging import get_logger

logger = get_logger(__name__)


# ChatML format template
CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>"""

# Alpaca format template
ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""

# Nemotron format (based on their chat template)
NEMOTRON_TEMPLATE = """<extra_id_0>System
{system}

<extra_id_1>User
{user}

<extra_id_1>Assistant
{assistant}"""


class AgenticFormatter:
    """
    Formats agentic datasets for supervised fine-tuning.

    Supports multiple output formats and filtering.
    """

    def __init__(
        self,
        format_type: str = "chatml",
        filter_typescript: bool = False,
        include_tool_calls: bool = True,
        max_turns: int = 10,
    ):
        """
        Initialize formatter.

        Args:
            format_type: Output format (chatml, alpaca, sharegpt, nemotron)
            filter_typescript: Only include TypeScript/JavaScript examples
            include_tool_calls: Include tool/function call examples
            max_turns: Maximum conversation turns to include
        """
        self.format_type = format_type
        self.filter_typescript = filter_typescript
        self.include_tool_calls = include_tool_calls
        self.max_turns = max_turns

        self.templates = {
            "chatml": CHATML_TEMPLATE,
            "alpaca": ALPACA_TEMPLATE,
            "nemotron": NEMOTRON_TEMPLATE,
        }

    def format_dataset(self, dataset: Any) -> list[dict]:
        """
        Format entire dataset.

        Args:
            dataset: HuggingFace dataset object

        Returns:
            List of formatted examples
        """
        formatted = []

        for item in dataset:
            try:
                result = self.format_example(item)
                if result:
                    formatted.append(result)
            except Exception as e:
                logger.debug(f"Failed to format example: {e}")

        logger.info(f"Formatted {len(formatted)} examples")
        return formatted

    def format_example(self, example: dict) -> dict | None:
        """
        Format a single example.

        Args:
            example: Raw example dictionary

        Returns:
            Formatted example or None if filtered
        """
        # Extract conversation
        conversation = self._extract_conversation(example)
        if not conversation:
            return None

        # Filter if needed
        if self.filter_typescript and not self._is_typescript_related(example):
            return None

        # Format based on type
        if self.format_type == "sharegpt":
            return self._format_sharegpt(conversation, example)
        else:
            return self._format_template(conversation, example)

    def _extract_conversation(self, example: dict) -> list[dict] | None:
        """Extract conversation turns from example."""
        # Handle different dataset formats
        if "conversations" in example:
            return example["conversations"]
        elif "messages" in example:
            return example["messages"]
        elif "instruction" in example and "response" in example:
            return [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["response"]},
            ]
        elif "prompt" in example and "completion" in example:
            return [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]
        else:
            return None

    def _is_typescript_related(self, example: dict) -> bool:
        """Check if example is TypeScript/JavaScript related."""
        keywords = [
            "typescript", "javascript", "ts", "js",
            "node", "npm", "react", "vue", "angular",
            "express", "next.js", "nestjs", "deno",
        ]

        # Check all text fields
        text = json.dumps(example).lower()
        return any(kw in text for kw in keywords)

    def _format_template(self, conversation: list[dict], example: dict) -> dict:
        """Format using template."""
        template = self.templates.get(self.format_type, CHATML_TEMPLATE)

        # Extract system message
        system = ""
        messages = []

        for msg in conversation[:self.max_turns * 2]:
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))

            if role in ("system",):
                system = content
            else:
                messages.append({"role": role, "content": content})

        if not messages:
            return None

        # Build formatted text
        formatted_parts = []

        for i in range(0, len(messages) - 1, 2):
            if i + 1 >= len(messages):
                break

            user_msg = messages[i]
            assistant_msg = messages[i + 1]

            if self.format_type == "chatml":
                part = CHATML_TEMPLATE.format(
                    system=system or "You are a helpful assistant.",
                    user=user_msg["content"],
                    assistant=assistant_msg["content"],
                )
            elif self.format_type == "alpaca":
                part = ALPACA_TEMPLATE.format(
                    instruction=user_msg["content"],
                    input="",
                    response=assistant_msg["content"],
                )
            elif self.format_type == "nemotron":
                part = NEMOTRON_TEMPLATE.format(
                    system=system or "You are a helpful assistant.",
                    user=user_msg["content"],
                    assistant=assistant_msg["content"],
                )
            else:
                part = f"User: {user_msg['content']}\nAssistant: {assistant_msg['content']}"

            formatted_parts.append(part)

        text = "\n\n".join(formatted_parts)

        return {
            "text": text,
            "format": self.format_type,
            "num_turns": len(messages) // 2,
        }

    def _format_sharegpt(self, conversation: list[dict], example: dict) -> dict:
        """Format in ShareGPT format."""
        formatted_conv = []

        for msg in conversation[:self.max_turns * 2]:
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))

            # Normalize role names
            if role in ("human", "user"):
                role = "human"
            elif role in ("gpt", "assistant", "bot"):
                role = "gpt"
            elif role == "system":
                role = "system"
            else:
                continue

            formatted_conv.append({
                "from": role,
                "value": content,
            })

        return {
            "conversations": formatted_conv,
            "format": "sharegpt",
        }


def convert_to_training_format(
    examples: list[dict],
    tokenizer: Any,
    max_length: int = 4096,
) -> list[dict]:
    """
    Convert formatted examples to tokenized training format.

    Args:
        examples: List of formatted examples
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        List of tokenized examples
    """
    tokenized = []

    for ex in examples:
        text = ex.get("text", "")
        if not text:
            continue

        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        tokenized.append({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        })

    return tokenized
