"""A sound card's own volume (its ALSA mixer), set with ``amixer`` from alsa-utils.

Lucy changes only the sound card's own volume through this adapter. Levels are in decibels as
``alsamixer`` and ``amixer`` show them, which is what the card reports, so they mean the same
on any card that reports decibels. ``amixer`` keeps a level inside the card's range, and
:meth:`Mixer.set_db` says where it ended up.

``amixer`` is run rather than linked against: Lucy only changes the volume when she opens
her speaker or a child asks, and a missing control, card, or program means she cannot change
that speaker's volume (an I2S amplifier has no mixer).
"""

import logging
import re
import subprocess
from collections.abc import Callable
from typing import Any, Protocol

log = logging.getLogger(__name__)

Run = Callable[..., Any]

# The usual names for a speaker's volume, most likely first; any other playback volume after.
_PREFERRED = ("PCM", "Speaker", "Master", "Headphone", "Digital")
_CONTROL = re.compile(r"^Simple mixer control '(?P<name>.*)',(?P<index>\d+)$")
_DB = re.compile(r"Playback.*?\[(?P<db>-?\d+(?:\.\d+)?)dB\]")
_TIMEOUT_S = 3.0


class CardVolume(Protocol):
    def set_db(self, db: float) -> float | None:
        """Set the card's own volume. The level it took, or None if it could not."""
        ...


class Mixer:
    """The playback volume of one sound card, named by its ALSA id (``aplay -l``, in brackets)."""

    def __init__(self, card: str, *, run: Run = subprocess.run) -> None:
        self.card = card
        self._run = run
        self._control: str | None = None
        self._looked = False

    def control(self) -> str | None:
        """The card's playback volume control (looked up once), or None if it has none."""
        if not self._looked:
            output = self._amixer("scontents")
            self._control = _playback_volume(output) if output is not None else None
            self._looked = output is not None  # a card that was not there: look again next time
            if self._looked and self._control is None:
                log.info("the %s card has no volume control", self.card)
        return self._control

    def get_db(self) -> float | None:
        """The level the card's volume is at now, in dB, if it can say."""
        control = self.control()
        if control is None:
            return None
        return _level(self._amixer("sget", control))

    def set_db(self, db: float) -> float | None:
        """Set the card's volume (and unmute it). The level it took, or None if it could not."""
        control = self.control()
        if control is None:
            return None
        output = self._amixer("--", "sset", control, f"{db:.2f}dB", "unmute")
        return _level(output)

    def _amixer(self, *args: str) -> str | None:
        try:
            done = self._run(
                ["amixer", "-c", self.card, *args],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("cannot change the %s card's volume: %s", self.card, error)
            return None
        if done.returncode != 0:
            log.debug("amixer -c %s %s: %s", self.card, " ".join(args), done.stderr.strip())
            return None
        return str(done.stdout)


def _playback_volume(scontents: str) -> str | None:
    """The best playback volume control in ``amixer scontents`` output, as amixer names it."""
    found: list[str] = []
    name = None
    for line in scontents.splitlines():
        if match := _CONTROL.match(line.strip()):
            index = match["index"]
            name = match["name"] if index == "0" else f"{match['name']},{index}"
        elif name is not None and line.strip().startswith("Capabilities:"):
            capabilities = line.split(":", 1)[1].split()
            if "pvolume" in capabilities or "volume" in capabilities:
                found.append(name)
    for preferred in _PREFERRED:
        if preferred in found:
            return preferred
    return found[0] if found else None


def _level(output: str | None) -> float | None:
    """The first channel's level in ``amixer sget``/``sset`` output, in dB."""
    if output is None:
        return None
    for line in output.splitlines():
        if match := _DB.search(line):
            return float(match["db"])
    return None
