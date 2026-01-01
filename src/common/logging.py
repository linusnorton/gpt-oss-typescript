"""
Logging configuration for the project.
"""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


_logger: Optional[logging.Logger] = None


def setup_logging(
    level: int = logging.INFO,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Set up logging configuration with rich formatting.

    Args:
        level: Base logging level
        verbose: If True, set level to DEBUG
        log_file: Optional file path for logging
    """
    global _logger

    if verbose:
        level = logging.DEBUG

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                tracebacks_show_locals=verbose,
            )
        ],
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logging.getLogger().addHandler(file_handler)

    _logger = logging.getLogger("gptoss-ts")
    _logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    if _logger is None:
        setup_logging()

    return logging.getLogger(name)
