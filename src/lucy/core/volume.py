"""How loud Lucy is, which a child can change just by asking ("louder!", "too loud!").

A request is recognised on the Pi, like the safety screens (:mod:`lucy.core.safety`), so
Lucy answers at once, already at her new loudness, and the request never reaches the
model. Only requests aimed at Lucy count. A question about loudness ("which is louder, a
lion or a whale?") is not a request, and goes to the model as usual.

The change is a master volume on the speaker, so everything she plays follows it: her
voice, her honks, and her chimes. It lasts until she is turned off. When she wakes up she
is as loud as her settings say, so a grown-up's choice always comes back.
"""

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from lucy.core.phrases import Phrase
from lucy.core.ports import Speaker


class VolumeRequest(StrEnum):
    LOUDER = "louder"
    QUIETER = "quieter"
    LOUDEST = "loudest"
    QUIETEST = "quietest"


@dataclass(frozen=True, slots=True)
class VolumeSettings:
    commands: bool = True
    """Children can ask Lucy to talk louder or quieter."""
    step_db: float = 3.0
    """How much louder or quieter each request makes her."""
    max_db: float = 9.0
    """The loudest a child can make her, above her usual loudness."""
    min_db: float = -15.0
    """The quietest a child can make her, below her usual loudness."""
    speaker_mixer: bool = True
    """Change the speaker's own volume (its ALSA mixer) first, with software gain on top
    once it is as loud as ``speaker_max_db``. Off, or on a card with no volume of its own
    (an I2S amplifier), the volume is software alone."""
    speaker_db: float = -9.0
    """The speaker's own volume at her usual loudness, in dB as ``alsamixer`` shows it. Lucy
    sets it each time she opens her speaker, so a speaker swapped or plugged back in gets it."""
    speaker_max_db: float = -3.0
    """The loudest the speaker's own volume goes. A small USB speaker driven harder can draw
    more power than its USB port gives, and drop off the bus (Lucy's did, at 0 dB)."""

    @property
    def software_headroom_db(self) -> float:
        """How far below full scale her usual loudness plays, so that the software part of the
        loudest volume reaches full scale and no further: louder in software only by playing
        closer to full scale, never by rounding off peaks, which distorts."""
        hardware = self.speaker_max_db - self.speaker_db if self.speaker_mixer else 0.0
        return max(0.0, self.max_db - max(0.0, hardware))


_CURLY = "\N{RIGHT SINGLE QUOTATION MARK}"  # speech-to-text writes "can\u2019t" for "can't"
_MAX_WORDS = 14  # a request is short; a long sentence that mentions loudness is a question
_NOT_A_REQUEST = re.compile(
    r"^(?:(?:hey|hi|ok|okay|so|um+|uh+|lucy|goose)\s+)*"
    r"(?:what|why|how|which|who|whose|when|where|is|are|was|were|does|do|did|has|have|"
    r"would you rather)\b"
)
_UP_TO_SOMETHING = r"(?:it|that|yourself|you|your (?:voice|volume|sound)|the (?:volume|sound))"
_A_BIT = r"(?:(?:a|little|bit|lot|tiny|way|much|even|more|teeny|wee) )*"
# "Talk louder" must be said to Lucy: first ("could you please talk louder"), after "you"
# ("you need to be louder"), or after "to" ("I want you to talk louder"). Not "can a whale
# be louder than a jet?".
_TO_LUCY = (
    r"(?:^(?:(?:hey|hi|ok|okay|lucy|goose|please|just|now|so|and|um+|uh+|"
    r"(?:can|could|would|will) you) )*|"
    r"\byou (?:(?:should|need to|have to|gotta|must|can|could|will|would) )?|\bto )"
)
_SPEAK = rf"{_TO_LUCY}(?:talk|speak|say (?:it|that)|sing|honk|be|get|go|sound)"
_YOU_ARE = r"(?:you'?re|you are|your voice is|it'?s|it is|that'?s|that is)"
_VERY = r"(?:too|so|really|very|way too|super)"

_LOUDEST = re.compile(
    r"\b(?:as loud as (?:you can|possible|you go)|(?:max|maximum|full|loudest|top) volume|"
    r"your loudest|loudest (?:you can|possible))\b"
)
_QUIETEST = re.compile(
    r"\b(?:as (?:quiet|soft)(?:ly)? as (?:you can|possible|you go)|"
    r"(?:min|minimum|lowest|quietest) volume|your quietest|quietest (?:you can|possible))\b"
)
_LOUDER = re.compile(
    rf"\b(?:(?:turn|crank|pump|bump|jack) {_UP_TO_SOMETHING} up|"
    r"(?:turn|crank|pump) up (?:your|the) (?:volume|voice|sound)|volume up|"
    r"(?:increase|raise|boost) (?:your|the) (?:volume|voice|sound)|"
    rf"{_SPEAK} {_A_BIT}(?:louder|more loudly)|speak up|"
    r"make (?:it|yourself|you|your voice) (?:a little |a bit )?louder|"
    r"(?:can'?t|cannot|can not|couldn'?t|could not|barely) hear you|"
    rf"{_YOU_ARE} {_VERY} (?:quiet|soft))\b"
)
_QUIETER = re.compile(
    rf"\b(?:(?:turn|bring) {_UP_TO_SOMETHING} down|"
    r"turn down (?:your|the) (?:volume|voice|sound)|volume down|"
    r"(?:decrease|lower|reduce) (?:your|the) (?:volume|voice|sound)|"
    rf"{_SPEAK} {_A_BIT}(?:quieter|softer|more (?:quietly|softly))|"
    r"make (?:it|yourself|you|your voice) (?:a little |a bit )?(?:quieter|softer)|"
    r"use (?:your |a )?(?:quiet|quieter|soft|softer|indoor|inside) voice|(?:indoor|inside) voice|"
    r"not so loud|"
    rf"{_YOU_ARE} {_VERY} loud)\b"
)
# The whole request in a word or two: "Louder!", "a bit quieter please", "too loud".
_JUST_LOUDER = re.compile(rf"^{_A_BIT}(?:louder|too quiet|volume up)(?: louder)*$")
_JUST_QUIETER = re.compile(rf"^{_A_BIT}(?:quieter|softer|too loud|volume down)(?: quieter)*$")
_POLITE = re.compile(r"^(?:(?:hey|ok|okay|lucy|goose|please)\s+)+|(?:\s+(?:please|lucy|goose))+$")


def volume_request(text: str) -> VolumeRequest | None:
    """The loudness change ``text`` asks Lucy for, if it is such a request."""
    words = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace(_CURLY, "'")).split()
    said = " ".join(words)
    if not said or len(words) > _MAX_WORDS or _NOT_A_REQUEST.match(said):
        return None
    for pattern, request in (
        (_LOUDEST, VolumeRequest.LOUDEST),
        (_QUIETEST, VolumeRequest.QUIETEST),
    ):
        if pattern.search(said):
            return request
    core = _POLITE.sub("", said).strip()
    louder = bool(_LOUDER.search(said) or _JUST_LOUDER.match(core))
    quieter = bool(_QUIETER.search(said) or _JUST_QUIETER.match(core))
    if louder == quieter:
        return None  # neither, or both ("not louder, quieter"): let the model sort it out
    return VolumeRequest.LOUDER if louder else VolumeRequest.QUIETER


class Volume:
    """Lucy's master volume in decibels: 0 is the loudness her settings give her."""

    def __init__(self, speaker: Speaker, settings: VolumeSettings) -> None:
        self._speaker = speaker
        self._settings = settings
        self._db = 0.0

    @property
    def db(self) -> float:
        return self._db

    def request(self, text: str) -> VolumeRequest | None:
        """The loudness change a child asked for, if they did and children may."""
        return volume_request(text) if self._settings.commands else None

    def change(self, request: VolumeRequest) -> Phrase:
        """Make Lucy louder or quieter, and pick what she says about it."""
        settings = self._settings
        louder = request in (VolumeRequest.LOUDER, VolumeRequest.LOUDEST)
        target = {
            VolumeRequest.LOUDER: min(settings.max_db, self._db + settings.step_db),
            VolumeRequest.QUIETER: max(settings.min_db, self._db - settings.step_db),
            VolumeRequest.LOUDEST: settings.max_db,
            VolumeRequest.QUIETEST: settings.min_db,
        }[request]
        if not math.isclose(target, self._db):
            self._db = target
            self._speaker.set_volume(target)
        if louder:
            return Phrase.LOUDEST if math.isclose(target, settings.max_db) else Phrase.LOUDER
        return Phrase.QUIETEST if math.isclose(target, settings.min_db) else Phrase.QUIETER
