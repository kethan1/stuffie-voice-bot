"""In-memory stand-ins for every port, so the core runs without hardware or network."""

import asyncio
import math
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence

import numpy as np

from lucy.core.audio import Pcm
from lucy.core.metrics import TurnTimer
from lucy.core.phrases import Phrase
from lucy.core.types import ButtonEvent, Message, Motor, PowerMode, Status

RATE = 1_000  # tiny sample rate keeps fake audio cheap


def tone(seconds: float, rate: int = 16_000, dbfs: float = -20.0) -> Pcm:
    t = np.arange(round(seconds * rate)) / rate
    amplitude = 10 ** (dbfs / 20) * math.sqrt(2)
    return Pcm.from_float(np.sin(2 * np.pi * 440 * t) * amplitude, rate)


async def until(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.001)


class FakeSTT:
    def __init__(self, text: str = "why is the sky blue", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[Pcm] = []

    async def transcribe(self, speech: Pcm) -> str:
        self.calls.append(speech)
        if self.error:
            raise self.error
        return self.text


class FakeChat:
    def __init__(
        self,
        deltas: Sequence[str] = ("Honk! The sky ", "is blue because ", "of sunlight. ", "Neat?"),
        error: Exception | None = None,
        error_after: int | None = None,
        delay: float = 0.0,
    ) -> None:
        self.deltas = list(deltas)
        self.error = error
        self.error_after = error_after
        self.delay = delay
        self.calls: list[list[Message]] = []
        self.closed = False

    async def stream(self, messages: Sequence[Message]) -> AsyncGenerator[str, None]:
        self.calls.append(list(messages))
        try:
            if self.error and self.error_after is None:
                raise self.error
            for index, delta in enumerate(self.deltas):
                if self.error and index == self.error_after:
                    raise self.error
                await asyncio.sleep(self.delay)
                yield delta
        finally:
            self.closed = True


class FakeTTS:
    """Renders each character as one sample, in ``parts`` equal parts per sentence."""

    sample_rate = RATE

    def __init__(
        self,
        error: Exception | None = None,
        *,
        parts: int = 1,
        error_after: int | None = None,
        sample: bytes = b"\x01\x00",
    ) -> None:
        self.texts: list[str] = []
        self.error = error
        self.error_after = error_after
        self.parts = parts
        self.sample = sample

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        self.texts.append(text)
        await asyncio.sleep(0)
        if self.error and self.error_after is None:
            raise self.error
        for index in range(self.parts):
            if self.error and index == self.error_after:
                raise self.error
            yield Pcm(self.sample * len(text), self.sample_rate)
            await asyncio.sleep(0)


class FakeSpeaker:
    """Plays each clip for ``seconds_per_clip`` of wall time, strictly in order."""

    def __init__(self, seconds_per_clip: float = 0.001) -> None:
        self.seconds_per_clip = seconds_per_clip
        self.played: list[Pcm] = []
        self.queued: list[tuple[Pcm, asyncio.Future[None]]] = []
        self.opened = 0
        self.closed = 0
        self._worker: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()

    def enqueue(self, audio: Pcm) -> asyncio.Future[None]:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.queued.append((audio, future))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._play())
        return future

    async def _play(self) -> None:
        while self.queued:
            audio, future = self.queued[0]
            if future.done():
                self.queued.pop(0)
                continue
            await asyncio.sleep(self.seconds_per_clip)
            if self.queued and self.queued[0][1] is future and not future.done():
                self.queued.pop(0)
                self.played.append(audio)
                future.set_result(None)

    def stop(self) -> bool:
        pending = [f for _, f in self.queued if not f.done()]
        for future in pending:
            future.cancel()
        self.queued.clear()
        return bool(pending)

    async def open(self) -> None:
        self.opened += 1

    async def close(self) -> None:
        self.closed += 1
        self.stop()


class FakeMic:
    def __init__(self, recording: Pcm | None = None) -> None:
        self.recording = recording or tone(1.0)
        self.recording_now = False
        self.starts = 0

    async def start(self) -> None:
        self.recording_now = True
        self.starts += 1

    async def stop(self) -> Pcm:
        self.recording_now = False
        return self.recording


class FakeIndicator:
    def __init__(self) -> None:
        self.history: list[Status] = []

    def show(self, status: Status) -> None:
        self.history.append(status)

    @property
    def current(self) -> Status | None:
        return self.history[-1] if self.history else None


class FakePower:
    def __init__(self) -> None:
        self.modes: list[PowerMode] = []
        self.restored = False

    async def apply(self, mode: PowerMode) -> None:
        self.modes.append(mode)

    async def restore(self) -> None:
        self.restored = True


class FakeMotors:
    """Records every motor command with the event-loop time it was given."""

    def __init__(self) -> None:
        self.moves: list[tuple[float, Motor, float]] = []
        self.awake = False
        self.sleeps = 0

    def drive(self, motor: Motor, power: float) -> None:
        self.moves.append((asyncio.get_running_loop().time(), motor, power))
        self.awake = self.awake or bool(power)

    def sleep(self) -> None:
        self.awake = False
        self.sleeps += 1

    def of(self, motor: Motor) -> list[tuple[float, float]]:
        """(time, power) of each command ``motor`` was given."""
        return [(at, power) for at, which, power in self.moves if which is motor]

    def powers(self, motor: Motor) -> list[float]:
        return [power for _, power in self.of(motor)]


class FakeWarmer:
    def __init__(self) -> None:
        self.count = 0

    def prewarm(self) -> None:
        self.count += 1


class FakeButton:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[ButtonEvent] = asyncio.Queue()
        self.now = 100.0

    def press(self, at: float | None = None) -> None:
        self.now = self.now if at is None else at
        self.queue.put_nowait(ButtonEvent(True, self.now))

    def release(self, after: float = 1.0) -> None:
        self.now += after
        self.queue.put_nowait(ButtonEvent(False, self.now))

    async def events(self) -> AsyncIterator[ButtonEvent]:
        while True:
            yield await self.queue.get()


class FakeResponder:
    """Stands in for TurnPipeline when testing the controller."""

    def __init__(self, respond_s: float = 0.0, say_s: float = 0.0) -> None:
        self.respond_s = respond_s
        self.say_s = say_s
        self.recordings: list[Pcm] = []
        self.said: list[Phrase] = []
        self.cancelled = 0

    async def respond(self, recording: Pcm, timer: TurnTimer) -> None:
        self.recordings.append(recording)
        try:
            await asyncio.sleep(self.respond_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    async def say(self, phrase: Phrase, *, status: Status = Status.SPEAKING) -> None:
        self.said.append(phrase)
        try:
            await asyncio.sleep(self.say_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
