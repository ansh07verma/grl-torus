"""
Structured logging setup for GRL-Torus.

Provides consistent log formatting across all modules with support
for both console and file output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    module_name: str = "grl_torus",
) -> logging.Logger:
    """Configure structured logging for the project.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a log file.
        module_name: Name for the logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(module_name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format: timestamp | level | module | message
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the grl_torus namespace.

    Args:
        name: Sub-module name (e.g., 'sim.torus_graph').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"grl_torus.{name}")
