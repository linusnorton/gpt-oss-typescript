"""
Path utilities and project directory management.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def get_project_root() -> Path:
    """Get the project root directory."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


def get_data_dir() -> Path:
    """
    Get the data directory path.

    Uses DATA_DIR environment variable if set, otherwise defaults to ./data
    """
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        path = Path(data_dir)
    else:
        path = get_project_root() / "data"

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir() -> Path:
    """
    Get the output directory path.

    Uses OUTPUT_DIR environment variable if set, otherwise defaults to ./outputs
    """
    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        path = Path(output_dir)
    else:
        path = get_project_root() / "outputs"

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir() -> Path:
    """Get the cache directory for downloaded models and datasets."""
    cache_dir = get_data_dir() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_checkpoint_dir(name: Optional[str] = None) -> Path:
    """
    Get a checkpoint directory.

    Args:
        name: Optional subdirectory name (e.g., 'dapt', 'sft')
    """
    base = get_output_dir() / "checkpoints"
    if name:
        base = base / name
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_log_dir() -> Path:
    """Get the logs directory."""
    log_dir = get_output_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_model_id() -> str:
    """Get the default model ID from environment."""
    return os.getenv(
        "MODEL_ID",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    )


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    return os.getenv("HF_TOKEN")


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment."""
    return os.getenv("GITHUB_PAT")
