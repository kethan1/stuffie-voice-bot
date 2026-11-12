"""Writing files so that a power cut never leaves half of one behind."""

import contextlib
import os
from pathlib import Path


def write_atomically(path: Path, text: str, mode: int) -> None:
    """Replace ``path`` with ``text`` in one step: the old file or the new one, never a mix.

    The file gets exactly ``mode`` (not reduced by the umask), and is on the card before
    this returns.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fchmod(file.fileno(), mode)
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
