"""Fixed things Lucy says, rendered once and replayed from memory.

Error and safety replies must work when the network does not, and should not
cost synthesis time when a child is already waiting, so each line is rendered
by Lucy's voice on first use (or ahead of time by :meth:`preload`) and cached
as PCM. Several variants per phrase keep Lucy from sounding robotic.
"""

import asyncio
import logging
import random
from collections.abc import Mapping, Sequence
from enum import StrEnum
from functools import partial

from lucy.core.audio import Pcm
from lucy.core.ports import TextToSpeech
from lucy.core.speech import render

log = logging.getLogger(__name__)


class Phrase(StrEnum):
    GREETING = "greeting"
    DIDNT_HEAR = "didnt_hear"
    HOLD_TO_TALK = "hold_to_talk"
    OFFLINE = "offline"
    OWN_BRAIN = "own_brain"
    TIRED = "tired"
    TANGLED = "tangled"
    NEEDS_HELP = "needs_help"
    CHANGE_SUBJECT = "change_subject"
    BE_KIND = "be_kind"
    NICE_TRY = "nice_try"
    TELL_A_GROWN_UP = "tell_a_grown_up"
    NOT_SAFE = "not_safe"
    KEEP_PRIVATE = "keep_private"
    SETUP = "setup"
    NO_SETUP = "no_setup"
    LOUDER = "louder"
    QUIETER = "quieter"
    LOUDEST = "loudest"
    QUIETEST = "quietest"


class PhraseBook:
    def __init__(
        self,
        lines: Mapping[Phrase, Sequence[str]],
        tts: TextToSpeech,
        *,
        rng: random.Random | None = None,
    ) -> None:
        missing = [phrase.value for phrase in Phrase if not lines.get(phrase)]
        if missing:
            raise ValueError(f"persona is missing lines for: {', '.join(missing)}")
        self._lines = {phrase: tuple(lines[phrase]) for phrase in Phrase}
        self._tts = tts
        self._rng = rng or random.Random()
        self._cache: dict[str, asyncio.Future[Pcm]] = {}

    async def render(self, phrase: Phrase) -> Pcm:
        return (await self.line(phrase))[1]

    async def line(self, phrase: Phrase) -> tuple[str, Pcm]:
        """One of the phrase's lines, picked at random, and its audio."""
        text = self._rng.choice(self._lines[phrase])
        # Shielded: if the turn is interrupted, the render still finishes and is cached.
        return text, await asyncio.shield(self._render(text))

    async def preload(self) -> None:
        """Render every line, one at a time so a real turn is never stuck behind a batch."""
        for lines in self._lines.values():
            for text in lines:
                try:
                    await asyncio.shield(self._render(text))
                except Exception:
                    log.warning("could not pre-render phrase %r", text, exc_info=True)

    def forget(self) -> None:
        """Drop every finished render, so each line is rendered again the next time it is used."""
        for text, future in list(self._cache.items()):
            if future.done():
                del self._cache[text]

    def _render(self, text: str) -> asyncio.Future[Pcm]:
        future = self._cache.get(text)
        if future is None:
            future = asyncio.ensure_future(render(self._tts, text))
            future.add_done_callback(partial(self._discard_failure, text))
            self._cache[text] = future
        return future

    def _discard_failure(self, text: str, future: asyncio.Future[Pcm]) -> None:
        if future.cancelled() or future.exception() is not None:
            self._cache.pop(text, None)
