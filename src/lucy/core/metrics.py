"""Per-turn latency measurement.

Every turn logs one line such as::

    stt=0.38s first_token=0.71s first_synth=1.02s first_audio=1.04s total=6.90s

with every value measured from the moment the button was released. That line
is the primary tool for tuning latency on real hardware.
"""

import time
from collections.abc import Callable


class TurnTimer:
    def __init__(
        self, *, origin: float | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock = clock
        self._origin = clock() if origin is None else origin
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        """Record the first time ``name`` happens; later calls are ignored."""
        self._marks.setdefault(name, self._clock() - self._origin)

    def get(self, name: str) -> float | None:
        return self._marks.get(name)

    def report(self) -> str:
        parts = [f"{name}={seconds:.2f}s" for name, seconds in self._marks.items()]
        parts.append(f"total={self._clock() - self._origin:.2f}s")
        return " ".join(parts)


class RateBudget:
    """Sliding-window cap on turns, so a stuck or hammered button cannot drain API quota."""

    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._stamps: list[float] = []

    def try_acquire(self) -> bool:
        if self._limit <= 0:
            return True
        now = self._clock()
        self._stamps = [t for t in self._stamps if now - t < self._window_s]
        if len(self._stamps) >= self._limit:
            return False
        self._stamps.append(now)
        return True
