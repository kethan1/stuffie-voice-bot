"""Logging set-up: compact when attached to journald, timestamped on a terminal."""

import logging
import os
import sys

_SYSLOG_PRIORITY = {
    logging.CRITICAL: 2,
    logging.ERROR: 3,
    logging.WARNING: 4,
    logging.INFO: 6,
    logging.DEBUG: 7,
}


class _JournalFormatter(logging.Formatter):
    """Prefix each line with ``<N>`` so journald stores the right priority."""

    def format(self, record: logging.LogRecord) -> str:
        return f"<{_SYSLOG_PRIORITY.get(record.levelno, 6)}>{super().format(record)}"


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if "JOURNAL_STREAM" in os.environ:  # journald adds timestamps itself
        handler.setFormatter(_JournalFormatter("%(name)s: %(message)s"))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "hpack", "h2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
