"""Logging configuration for command-line analysis."""

import logging
import sys


def configure_logging(verbose: bool = False) -> None:
    """Send diagnostics to stderr, keeping machine-readable stdout clean."""
    logger = logging.getLogger("satellite_analyzer")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    # Each CLI invocation gets the current stderr (including CLI test captures).
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
