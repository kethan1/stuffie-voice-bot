"""Ports: the interfaces the core depends on.

The core (controller, pipeline, conversation) only ever talks to these
protocols. Concrete adapters (Groq over HTTP, Piper, PortAudio, libgpiod, ...)
live in :mod:`lucy.adapters` and are wired together in :mod:`lucy.app`. Tests
substitute in-memory fakes, so the whole conversation flow is exercised
without a network, a sound card, or a Raspberry Pi.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Protocol

from lucy.core.audio import Pcm
from lucy.core.types import Assessment, ButtonEvent, Message, Motor, PowerMode, Status


class SpeechToText(Protocol):
    async def transcribe(self, speech: Pcm) -> str:
        """Return what was said (empty if nothing intelligible)."""
        ...


class ChatModel(Protocol):
    def stream(self, messages: Sequence[Message]) -> AsyncGenerator[str, None]:
        """Stream the reply as text deltas, as soon as the model produces them."""
        ...


class Moderator(Protocol):
    async def assess(self, text: str) -> Assessment:
        """Decide whether a child's message may be answered by the language model."""
        ...


class ReplyScreen(Protocol):
    def screen(self, sentence: str) -> Assessment:
        """Whether Lucy may say this sentence of her reply aloud, and if not, why not."""
        ...


class TextToSpeech(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        """Render one sentence, yielding audio in order as soon as each part is ready.

        Joined, the parts are the whole sentence. :func:`lucy.core.speech.render`
        collects them into one clip.
        """
        ...


class Microphone(Protocol):
    async def start(self) -> None:
        """Open the capture device and start recording."""
        ...

    async def stop(self) -> Pcm:
        """Stop recording, close the device, and return everything captured."""
        ...


class Speaker(Protocol):
    def enqueue(self, audio: Pcm) -> asyncio.Future[None]:
        """Queue audio for gapless playback.

        The future resolves once the clip has been handed to the sound card.
        Cancelling the future removes the clip if it has not started playing.
        """
        ...

    def stop(self) -> bool:
        """Silence immediately and drop everything queued. True if anything was playing."""
        ...

    async def open(self) -> None:
        """Open the output device ahead of time so the first word is not delayed."""
        ...

    async def close(self) -> None:
        """Release the output device so the amplifier and sound card can idle."""
        ...


class PlaybackListener(Protocol):
    """Told what the speaker plays, and when it is heard (see :mod:`lucy.core.motion`)."""

    def playing(self, audio: Pcm, at: float) -> None:
        """``audio`` has started to play: its first sample is heard at loop time ``at``."""
        ...

    def stopped(self) -> None:
        """Playback was cut short: nothing already reported will be heard any more."""
        ...


class Motors(Protocol):
    def drive(self, motor: Motor, power: float) -> None:
        """Run ``motor`` at ``power``, from -1 to 1: the sign picks the way, 0 lets it coast.

        Wakes the motor driver if it is asleep.
        """
        ...

    def sleep(self) -> None:
        """Let every motor coast and put the driver to sleep, drawing next to nothing."""
        ...


class Button(Protocol):
    def events(self) -> AsyncIterator[ButtonEvent]:
        """Debounced press/release events, for as long as the iterator is consumed."""
        ...


class Indicator(Protocol):
    def show(self, status: Status) -> None: ...


class PowerControl(Protocol):
    async def apply(self, mode: PowerMode) -> None: ...

    async def restore(self) -> None:
        """Put the system back the way it was before Lucy started."""
        ...


class Prewarmable(Protocol):
    def prewarm(self) -> None:
        """Start getting ready (DNS, TCP, TLS) for a request that is about to come."""
        ...
