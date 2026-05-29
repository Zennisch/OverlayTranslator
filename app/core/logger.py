import logging
import sys

from app.config import settings


def setup_logging() -> None:
    """Initialize and standardize application logging levels, streaming exclusively to stderr."""
    level_str = (settings.log_level or "info").upper()
    level = getattr(logging, level_str, logging.INFO)

    # Clear root handlers to avoid double logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Use sys.stderr so sys.stdout is reserved strictly for clean JSON outputs
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Configure our primary CLI logger
    sidecar_logger = logging.getLogger("sidecar")
    sidecar_logger.setLevel(level)
    sidecar_logger.handlers = [handler]
    sidecar_logger.propagate = False

    # Route the 'manga-translator' engine logs through the same stderr handler
    manga_logger = logging.getLogger("manga-translator")
    manga_logger.setLevel(level)
    manga_logger.handlers = [handler]
    manga_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child sidecar logger with standard configuration."""
    return logging.getLogger(f"sidecar.{name}")
