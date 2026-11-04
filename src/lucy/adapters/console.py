"""Stand-ins for the hardware button and LED, for development and bring-up.

``lucy run --keyboard`` turns the Enter key into the push-to-talk button, so
the whole voice loop can be tested over SSH (or on a laptop) before any wiring
is done.
"""

import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator

from lucy.core.types import ButtonEvent, Status

log = logging.getLogger(__name__)


class KeyboardButton:
    """Enter starts talking, Enter again stops (a toggle stands in for press-and-hold)."""

    async def events(self) -> AsyncIterator[ButtonEvent]:
        loop = asyncio.get_running_loop()
        lines: asyncio.Queue[str | None] = asyncio.Queue()

        def on_readable() -> None:
            line = sys.stdin.readline()
            lines.put_nowait(line if line else None)  # "" means end of input

        loop.add_reader(sys.stdin.fileno(), on_readable)
        print("Press Enter to start talking to Lucy, and Enter again when you're done.")
        pressed = False
        try:
            while await lines.get() is not None:
                pressed = not pressed
                print("🎙  listening… (Enter to stop)" if pressed else "⏳ thinking…")
                yield ButtonEvent(pressed, time.monotonic())
        finally:
            loop.remove_reader(sys.stdin.fileno())


class LogIndicator:
    def show(self, status: Status) -> None:
        log.debug("status: %s", status.value)


class NullIndicator:
    def show(self, status: Status) -> None:
        pass
