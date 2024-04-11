from __future__ import annotations

import logging

# create logger object
logger = logging.getLogger(__name__)


def configure_logger(level: int | str = logging.INFO) -> None:
    """Get console logger by name.

    Args:
        level (int | str, optional): Logger Level. Defaults to logging.INFO.

    Returns:
        Logger: The expected logger.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=level)

    # Set Pytorch Lightning logs to have a the consistent formatting with anomalib.
    for handler in logging.getLogger("SCM-4.0").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)


def attach_file_to_logger(log_file):
    """This function attaches file stream to the logger to consume all logged information and saved it"""
    global logger
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt=format_string)
    # Creates a new logs every minute if restarted.
    # Otherwise, it will use the same file
    handler = logging.FileHandler(log_file, "a")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return handler
