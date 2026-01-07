"""Logging utilities with colored output support"""

import datetime
import logging
import sys

from termcolor import colored


class ColoredFormatter(logging.Formatter):
    """Logging formatter with colors for terminal output"""

    COLORS = {
        logging.DEBUG: "dark_grey",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "red",
    }

    LEVEL_NAMES = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARN",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRIT",
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # ISO format timestamp
        timestamp = datetime.datetime.fromtimestamp(record.created).isoformat(
            timespec="seconds"
        )
        level_name = self.LEVEL_NAMES.get(record.levelno, record.levelname[:5])
        message = record.getMessage()

        if self.use_color:
            color = self.COLORS.get(record.levelno, "white")
            level_str = colored(f"{level_name:5}", color, attrs=["bold"])
            name_str = colored(record.name, "cyan")
            return f"{timestamp} {level_str} {name_str}: {message}"
        else:
            return f"{timestamp} {level_name:5} {record.name}: {message}"


def setup_logging(debug: bool = False, force_color: bool = False):
    """Set up logging with optional colors for terminal output

    Args:
        debug: Enable debug level logging
        force_color: Force colored output even if not a TTY
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Check if stderr is a TTY (terminal) for colored output
    use_color = force_color or sys.stderr.isatty()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColoredFormatter(use_color=use_color))
    root_logger.addHandler(handler)

    # Set specific loggers to INFO to reduce noise
    logging.getLogger("xpm.hash").setLevel(logging.INFO)
