"""
Watty Structured Logging
=========================
Configure once, use everywhere. File gets DEBUG+, stderr gets INFO+.
Log rotation: 5MB x 3 files.
"""

import logging
import logging.handlers
import os
import time

from watty.config import WATTY_HOME

log = logging.getLogger("watty")
log.addHandler(logging.NullHandler())  # prevent "No handlers" warning

_configured = False


def setup():
    """Configure logging. Safe to call multiple times."""
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("WATTY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stderr: INFO+ (or env override)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(fmt)
    log.addHandler(stderr_handler)

    # File: DEBUG+ with rotation
    try:
        WATTY_HOME.mkdir(parents=True, exist_ok=True)
        log_path = WATTY_HOME / "watty.log"
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_path), maxBytes=5 * 1024 * 1024, backupCount=3,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        log.addHandler(file_handler)
    except Exception:
        pass  # Can't write log file — stderr only


def timed(operation: str):
    """Context manager that logs operation duration."""
    class _Timer:
        def __init__(self):
            self.start = 0
            self.elapsed_ms = 0
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *exc):
            self.elapsed_ms = (time.perf_counter() - self.start) * 1000
            return False
    return _Timer()
