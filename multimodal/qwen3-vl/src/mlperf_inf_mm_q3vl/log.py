"""Logging utilities for the Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .schema import Settings, Verbosity


def get_log_file_path(key: str, settings: Settings) -> Path:
    """Get the log file path for a given key based on MLPerf LoadGen's convention."""
    datetime_str_in_log_filename = (
        datetime.now(tz=UTC).astimezone().strftime("%FT%TZ_")
        if settings.logging.log_output.prefix_with_datetime
        else ""
    )
    return Path(
        settings.logging.log_output.outdir
        / (
            f"{settings.logging.log_output.prefix}"
            f"{datetime_str_in_log_filename}"
            f"{key}"
            f"{settings.logging.log_output.suffix}"
            ".txt"
        ),
    )


def setup_loguru_for_benchmark(
        settings: Settings, verbosity: Verbosity) -> None:
    """Setup the loguru logger for running the benchmark."""
    logger.remove()
    logger.add(sys.stdout, level=verbosity.value.upper())
    logger.add(
        get_log_file_path(key="benchmark", settings=settings),
        level=verbosity.value.upper(),
    )
