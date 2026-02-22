"""
Logging utilities for the multi-modal-transformer project.

This module provides a small helper to set up a logger that writes to both:

  - stdout (so you see logs in the console),
  - an optional log file under a given output directory (e.g. runs/.../out.log).

Typical usage in a script:

    from mmt.utils.logging import setup_logging
    from mmt.utils.config_loader import load_experiment_config

    cfg = load_experiment_config("mmt/configs/task_2-1/finetune_default.yaml")
    logger = setup_logging(cfg.paths["output_root"], logger_name="mmt.finetune")

    logger.info("Starting finetuning")
    logger.info(f"Task: {cfg.task}, phase: {cfg.phase}")

"""

from __future__ import annotations

import sys
from pathlib import Path

import logging as py_logging


class _BlankLineFormatter(py_logging.Formatter):
    """Formatter that emits true blank lines when message is empty."""

    def format(self, record: py_logging.LogRecord) -> str:
        if record.getMessage() == "":
            return ""
        return super().format(record)


def setup_logging(
    run_dir: Path,
    *,
    logger_name: str = "mmt",
    level: str = "INFO",
    log_to_file: bool = True,
    filename: str | None = "out.log",
    console: bool = True,
) -> py_logging.Logger:
    """
    Create (or retrieve) a logger configured for console + optional file logging.

    Parameters
    ----------
    run_dir :
        Directory where the log file should be created. The directory will be
        created if it does not exist.
    logger_name :
        Name of the logger to create/retrieve. Using different names lets you
        separate logs from different entrypoints, if needed.
    level :
        Logging level as a string, e.g. "INFO", "DEBUG", "WARNING".
    log_to_file :
        If True, add a FileHandler pointing to `output_dir / filename`.
    filename :
        Name of the log file. If None, file logging is disabled.
    console:
        True to print logs
    Returns
    -------
    logger :
        Configured logger instance.
    """
    logger = py_logging.getLogger(logger_name)

    # If the logger already has handlers, assume we've configured it before.
    if logger.handlers:
        return logger

    level_value = getattr(py_logging, level.upper(), py_logging.INFO)
    logger.setLevel(level_value)

    # Common formatter for all handlers
    formatter = _BlankLineFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = py_logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file and filename is not None:
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename

        file_handler = py_logging.FileHandler(file_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Do not propagate to the root logger to avoid duplicate logs
    logger.propagate = False

    return logger
