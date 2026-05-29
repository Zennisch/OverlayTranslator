import logging


def get_logger(name: str) -> logging.Logger:
    """Retrieve a child logger under the 'manga-translator' namespace."""
    return logging.getLogger(f"manga-translator.{name}")
