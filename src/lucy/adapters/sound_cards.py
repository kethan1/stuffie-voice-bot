"""Which sound card is Lucy's speaker, and which her microphone: looked up as each one opens.

Lucy plays and records through ALSA's default device, which ``/etc/asound.conf`` (from
``deploy/alsa/asound.conf``) splits between a playback card and a capture card. It names
them through two environment variables, read by libasound each time a stream is opened::

    card { @func getenv vars [ LUCY_SPEAKER_CARD ] default "LucySpeaker" }

Just before a stream opens, :func:`route` finds the card it should use and sets that
variable, so a speaker that is swapped for another, plugged into another port, or numbered
differently after a reconnect is used without a restart or a new ``asound.conf``:

1. The card the configuration names (``audio.speaker_card``, ``audio.microphone_card``,
   which the installer writes), if it is plugged in.
2. Otherwise a USB card that can play (or record), one that does only that first (so a
   headset's speaker is not chosen over a plain speaker), then the lowest-numbered.
3. Otherwise nothing: the variable is cleared and ``asound.conf``'s own card is used. That is
   how an I2S amplifier (always present, not USB) keeps working, and when it is missing too
   the stream fails, with an error that says no speaker was found.

The onboard outputs (HDMI and the headphone jack) are never chosen in step 2: they are
always present, so choosing one would play to nothing and hide that the speaker is gone.

Cards are read from ``/proc/asound``, the way ``aplay -l`` lists them.
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

Kind = Literal["output", "input"]

PROC_ASOUND = Path("/proc/asound")
VARIABLES: dict[Kind, str] = {"output": "LUCY_SPEAKER_CARD", "input": "LUCY_MICROPHONE_CARD"}
_NAMES: dict[Kind, str] = {"output": "speaker", "input": "microphone"}


@dataclass(frozen=True, slots=True)
class Card:
    number: int
    id: str
    """ALSA's id for the card: the name in brackets in ``aplay -l``."""
    usb: bool
    playback: bool
    capture: bool

    def can(self, kind: Kind) -> bool:
        return self.playback if kind == "output" else self.capture

    def only(self, kind: Kind) -> bool:
        """True if the card does just this (a plain speaker, not a headset)."""
        return self.can(kind) and not (self.capture if kind == "output" else self.playback)

    def __str__(self) -> str:
        return f"{self.id} (card {self.number}{', USB' if self.usb else ''})"


@dataclass(frozen=True, slots=True)
class Route:
    """Where ALSA's default device sends a kind of stream."""

    kind: Kind
    card: Card | None
    """None: no card was found, and ``asound.conf``'s own card is used."""
    wanted: str
    """The card the configuration names ("" for none)."""

    @property
    def stand_in(self) -> bool:
        """True if a card other than the configured one is used, because that one is gone."""
        return bool(self.wanted) and self.card is not None and self.card.id != self.wanted

    def __str__(self) -> str:
        name = _NAMES[self.kind]
        if self.card is None:
            if self.wanted:
                return f"no {name} found: {self.wanted} and every USB {name} are unplugged"
            return f"no USB {name} found, so the one in /etc/asound.conf"
        if self.stand_in:
            return f"{self.card}, standing in for {self.wanted}, which is unplugged"
        return str(self.card)


def cards(root: Path | None = None) -> list[Card]:
    """The sound cards plugged in now, in ALSA's order."""
    found = []
    try:
        folders = list((root or PROC_ASOUND).glob("card[0-9]*"))
    except OSError:
        return []
    for folder in folders:
        try:
            number = int(folder.name.removeprefix("card"))
            card_id = (folder / "id").read_text(encoding="ascii").strip()
            found.append(
                Card(
                    number=number,
                    id=card_id,
                    usb=(folder / "usbid").exists(),
                    playback=any(folder.glob("pcm*p")),
                    capture=any(folder.glob("pcm*c")),
                )
            )
        except (OSError, ValueError, UnicodeDecodeError):
            continue  # unplugged while it was being read
    return sorted(found, key=lambda card: card.number)


def choose(kind: Kind, wanted: str, found: list[Card]) -> Card | None:
    """The card to use for ``kind``: the one named ``wanted``, or else the best USB card."""
    for card in found:
        if wanted and card.id == wanted and card.can(kind):
            return card
    usable = [card for card in found if card.usb and card.can(kind)]
    usable.sort(key=lambda card: (not card.only(kind), card.number))
    return usable[0] if usable else None


class _Router:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._said: dict[Kind, str] = {}

    def route(self, kind: Kind, wanted: str, root: Path | None) -> Route:
        with self._lock:
            route = Route(kind, choose(kind, wanted, cards(root)), wanted)
            variable = VARIABLES[kind]
            value = route.card.id if route.card is not None else None
            if os.environ.get(variable) != value:  # setenv is rare: only when the card changes
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value
            said = str(route)
            if self._said.get(kind) != said:
                self._said[kind] = said
                trouble = route.stand_in or (route.card is None and bool(wanted))
                level = logging.WARNING if trouble else logging.INFO
                log.log(level, "%s: %s", _NAMES[kind], said)
            return route


_router = _Router()


def route(kind: Kind, wanted: str = "", *, root: Path | None = None) -> Route:
    """Point ALSA's default ``kind`` device at the card to use now, and say which it is."""
    return _router.route(kind, wanted.strip(), root)


def user(route: Route, *, root: Path | None = None) -> int | None:
    """The process using the routed card's first stream of that kind, if one is (Lucy, often)."""
    if route.card is None:
        return None
    letter = "p" if route.kind == "output" else "c"
    status = (
        (root or PROC_ASOUND) / f"card{route.card.number}" / f"pcm0{letter}" / "sub0" / "status"
    )
    try:
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("owner_pid"):
                return int(line.split(":", 1)[1])
    except (OSError, ValueError):
        pass
    return None
