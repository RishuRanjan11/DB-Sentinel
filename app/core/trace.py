from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path


_LOGGER_NAME = "db_sentinel"


def _configure_external_logging() -> None:
    """
    Keep third-party SDK diagnostics out of the normal terminal output.

    Detailed DB-Sentinel diagnostics continue to be written to the
    DB-Sentinel log file.
    """

    for name in (
        "google.genai",
        "google.genai.models",
        "google.genai._interactions",
    ):

        logging.getLogger(
            name
        ).setLevel(
            logging.ERROR
        )

    warnings.filterwarnings(
        "ignore",
        message=(
            r".*automatic function calling.*"
            r".*AFC.*not recommended.*"
        ),
    )


def get_logger() -> logging.Logger:

    _configure_external_logging()

    logger = logging.getLogger(
        _LOGGER_NAME
    )

    level = os.getenv(
        "DB_SENTINEL_LOG_LEVEL",
        "DEBUG",
    ).upper()

    logger.setLevel(
        getattr(
            logging,
            level,
            logging.DEBUG,
        )
    )

    logger.propagate = False

    log_dir = Path(
        os.getenv(
            "DB_SENTINEL_LOG_DIR",
            "logs",
        )
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        log_dir
        / "db-sentinel.log"
    )

    # Avoid duplicate handlers when the module is imported/reloaded.
    for handler in logger.handlers:

        if (
            isinstance(
                handler,
                logging.FileHandler,
            )
            and Path(
                handler.baseFilename
            ).resolve()
            == log_file.resolve()
        ):

            return logger

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.DEBUG
    )

    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )

    logger.addHandler(
        file_handler
    )

    return logger


logger = get_logger()