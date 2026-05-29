import logging

from app.config import settings
from zns_logging.ZnsLogger import ZnsLogger


def setup_logging() -> None:
    """Initialize and standardize application logging levels, streaming exclusively to stderr."""
    level_str = (settings.log_level or "info").upper()
    level = getattr(logging, level_str, logging.INFO)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    logger = ZnsLogger(__name__, level_str)
    handlers = logger.handlers

    core_logger = logging.getLogger("core")
    core_logger.setLevel(level)
    core_logger.handlers = handlers
    core_logger.propagate = False

    translator_logger = logging.getLogger("translator")
    translator_logger.setLevel(level)
    translator_logger.handlers = handlers
    translator_logger.propagate = False


def get_core_logger(name: str) -> logging.Logger:
    """Retrieve a child core logger with standard configuration."""
    return logging.getLogger(f"core.{name}")

def get_translator_logger(name: str) -> logging.Logger:
  """Retrieve a child translator logger with standard configuration."""
  return logging.getLogger(f"translator.{name}")
