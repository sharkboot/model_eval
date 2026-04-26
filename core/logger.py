import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "model_eval",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path to also log to file
        format_string: Optional custom format string
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(message)s"

    formatter = logging.Formatter(format_string)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


_default_logger: Optional[logging.Logger] = None


def get_logger(name: str = "model_eval") -> logging.Logger:
    """Get or create the default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger(name)
    return logging.getLogger(name)
