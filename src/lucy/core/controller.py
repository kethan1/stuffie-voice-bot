"""The push-to-talk state machine.

::

            press                     release (held ≥ tap)
    IDLE ──────────► LISTENING ─────────────────────────► RESPONDING ──┐
     ▲  ▲               │ release (tap)                       │ press   │ done
     │  │               ▼                                     ▼ (barge  │
     │  └──── "keep squeezing" hint / stop talking      LISTENING  in)  │
     └──────────────────────────────────────────────────────────────────┘

    tap, tap, hold        → the knock (lucy.core.knock): that turn may ask for
                            setup mode. So the hint waits tap_gap_s after a tap,
                            in case the tap is the start of one.
    after awake_s idle    → DORMANT power profile (Wi-Fi power save on, CPU
                            powersave, speaker closed so the amp sleeps)
    after forget_after_s  → conversation memory is cleared

Every handler runs under one lock, so button edges, the recording time limit,
and the dormant timer can never interleave half-way through each other.
"""

import asyncio
import contextlib
import logging
import math
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

from lucy.core.audio import Pcm, chime
from lucy.core.conversation import Conversation
from lucy.core.errors import AudioDeviceError
from lucy.core.knock import KNOCK_TAPS
from lucy.core.metrics import RateBudget, TurnTimer
from lucy.core.phrases import Phrase
from lucy.core.ports import Button, Indicator, Microphone, PowerControl, Prewarmable, Speaker
from lucy.core.tasks import BackgroundTasks, detach
from lucy.core.types import PowerMode, Status

log = logging.getLogger(__name__)


class Responder(Protocol):
    """What the controller needs from the turn pipeline (see :class:`TurnPipeline`)."""

    async def respond(self, recording: Pcm, timer: TurnTimer, *, knocked: bool = False) -> None: ...

    async def say(self, phrase: Phrase, *, status: Status = Status.SPEAKING) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionSettings:
    tap_max_s: float = 0.3
    """A press shorter than this is a tap: it stops Lucy talking, or prompts a hint."""
    tap_gap_s: float = 0.7
    """Taps this close together, and a squeeze this soon after them, are one gesture: the
    knock (lucy.core.knock). The hint after a tap waits this long, in case more follows."""
    max_record_s: float = 15.0
    """Recording stops on its own after this long (protects against a stuck button)."""
    awake_s: float = 45.0
    """Stay in the fast STANDBY profile this long after a turn before going DORMANT."""
    forget_after_s: float = 300.0
    """Clear the conversation after this much quiet; the next child starts fresh."""
    max_turns_per_hour: int = 120
    """0 disables the cap."""
    greet_on_start: bool = True
    earcons: bool = True
    earcon_volume: float = 1.0
    """The chimes when Lucy starts and stops listening: 1 is a little louder than her voice,
    so they are heard over a busy room. They also follow her volume (lucy.core.volume)."""


@dataclass(frozen=True, slots=True)
class Earcons:
    listening: Pcm
    thinking: Pcm

    @classmethod
    def generate(cls, sample_rate: int, *, volume: float = 1.0) -> Self:
        """Two short rising and falling chimes. The octave overtone makes them bright, which a
        small speaker plays far louder than a pure tone of the same peak."""
        return cls(
            listening=chime(
                [(587.0, 0.06), (880.0, 0.08)], sample_rate, volume=0.6 * volume, overtone=0.35
            ),
            thinking=chime(
                [(880.0, 0.06), (698.0, 0.08)], sample_rate, volume=0.5 * volume, overtone=0.35
            ),
        )


@dataclass(frozen=True, slots=True)
class _Press:
    at: float
    interrupted: bool


class Controller:
    def __init__(
        self,
        *,
        pipeline: Responder,
        mic: Microphone,
        speaker: Speaker,
        indicator: Indicator,
        power: PowerControl,
        conversation: Conversation,
        warmers: Sequence[Prewarmable],
        earcons: Earcons | None,
        settings: SessionSettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pipeline = pipeline
        self._mic = mic
        self._speaker = speaker
        self._indicator = indicator
        self._power = power
        self._conversation = conversation
        self._warmers = tuple(warmers)
        self._earcons = earcons if settings.earcons else None
        self._settings = settings
        self._clock = clock
        self._budget = RateBudget(settings.max_turns_per_hour, 3600.0, clock)
        self._lock = asyncio.Lock()
        self._tasks = BackgroundTasks()
        self._press: _Press | None = None
        self._turn: asyncio.Task[None] | None = None
        self._taps = 0  # taps in a row, each soon after the last
        self._tapped_at = -math.inf  # when the last of them ended
        self._record_limit: asyncio.TimerHandle | None = None
        self._hint_timer: asyncio.TimerHandle | None = None
        self._dormant_timer: asyncio.TimerHandle | None = None
        self._forget_timer: asyncio.TimerHandle | None = None

    async def run(self, button: Button) -> None:
        """Handle button events until the button iterator ends or the task is cancelled."""
        if self._settings.greet_on_start:
            self._start_turn(self._pipeline.say(Phrase.GREETING))
        else:
            self._settle()
        async for event in button.events():
            if event.pressed:
                await self._pressed(event.at)
            else:
                await self._released(event.at)

    async def close(self) -> None:
        async with self._lock:
            self._cancel_timers()
            await self._interrupt()
            if self._press is not None:
                self._press = None
                with contextlib.suppress(AudioDeviceError):
                    await self._mic.stop()
        await self._tasks.cancel_all()
        await self._speaker.close()

    @property
    def busy(self) -> bool:
        return self._press is not None or self._turn is not None

    # -- button handlers --------------------------------------------------------------

    async def _pressed(self, at: float) -> None:
        async with self._lock:
            if self._press is not None:
                return
            if at - self._tapped_at > self._settings.tap_gap_s:
                self._taps = 0  # too long after the last tap to belong with it
            self._cancel_timers()
            interrupted = await self._interrupt()
            self._tasks.spawn(self._power.apply(PowerMode.ACTIVE), "power-active")
            for warmer in self._warmers:
                warmer.prewarm()
            try:
                await self._mic.start()
            except AudioDeviceError:
                log.exception("could not start the microphone")
                self._start_turn(self._pipeline.say(Phrase.NEEDS_HELP, status=Status.ERROR))
                return
            self._press = _Press(at, interrupted)
            self._indicator.show(Status.LISTENING)
            if self._earcons:
                detach(self._speaker.enqueue(self._earcons.listening))
            else:
                self._tasks.spawn(self._speaker.open(), "speaker-open")
            loop = asyncio.get_running_loop()
            self._record_limit = loop.call_later(
                self._settings.max_record_s, self._record_limit_reached
            )

    async def _released(self, at: float) -> None:
        async with self._lock:
            press, self._press = self._press, None
            if press is None:
                return
            _cancel(self._record_limit)
            try:
                recording = await self._mic.stop()
            except AudioDeviceError:
                log.exception("could not read the microphone")
                self._start_turn(self._pipeline.say(Phrase.NEEDS_HELP, status=Status.ERROR))
                return

            if at - press.at < self._settings.tap_max_s:
                self._taps += 1
                self._tapped_at = at
                if press.interrupted:
                    log.info("tap: stopped talking")
                    self._settle()
                else:
                    # She explains how to talk to her, unless this tap starts a knock.
                    self._indicator.show(Status.IDLE)
                    self._hint_timer = asyncio.get_running_loop().call_later(
                        self._settings.tap_gap_s, self._hint
                    )
                return
            knocked, self._taps = self._taps >= KNOCK_TAPS, 0
            if knocked:
                log.info("the knock: taps, then this squeeze")
            if not self._budget.try_acquire():
                log.warning("hourly turn budget used up; asking the child to wait")
                self._start_turn(self._pipeline.say(Phrase.TIRED))
                return

            self._indicator.show(Status.THINKING)
            if self._earcons:
                detach(self._speaker.enqueue(self._earcons.thinking))
            timer = TurnTimer(origin=at, clock=self._clock)
            self._start_turn(self._pipeline.respond(recording, timer, knocked=knocked))

    def _hint(self) -> None:
        """A tap was not the start of a knock after all: explain how to talk to her."""
        if not self.busy:
            self._start_turn(self._pipeline.say(Phrase.HOLD_TO_TALK))

    def _record_limit_reached(self) -> None:
        log.info("recording limit of %.0fs reached", self._settings.max_record_s)
        self._tasks.spawn(self._released(self._clock()), "record-limit")

    # -- turn lifecycle ---------------------------------------------------------------

    def _start_turn(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work, name="turn")
        self._turn = task
        task.add_done_callback(self._turn_finished)

    def _turn_finished(self, task: asyncio.Task[None]) -> None:
        if task is not self._turn:
            return  # superseded by a barge-in; the press handler owns the state now
        self._turn = None
        if not task.cancelled() and (error := task.exception()) is not None:
            log.error("turn crashed", exc_info=error)
        self._settle()

    async def _interrupt(self) -> bool:
        """Stop talking and cancel any turn in flight. True if anything was interrupted."""
        was_playing = self._speaker.stop()
        task, self._turn = self._turn, None
        if task is None or task.done():
            return was_playing
        task.cancel()
        await asyncio.wait({task})
        return True

    def _settle(self) -> None:
        """A turn ended: show idle, stay responsive for a while, then doze."""
        if self._press is not None:
            return
        self._indicator.show(Status.IDLE)
        self._tasks.spawn(self._power.apply(PowerMode.STANDBY), "power-standby")
        self._cancel_timers()
        loop = asyncio.get_running_loop()
        self._dormant_timer = loop.call_later(self._settings.awake_s, self._doze)
        self._forget_timer = loop.call_later(self._settings.forget_after_s, self._forget)

    def _doze(self) -> None:
        self._tasks.spawn(self._go_dormant(), "power-dormant")

    async def _go_dormant(self) -> None:
        async with self._lock:
            if self.busy:
                return
            log.info("dozing: switching to the dormant power profile")
            await self._speaker.close()
            await self._power.apply(PowerMode.DORMANT)

    def _forget(self) -> None:
        if self.busy:
            return
        if len(self._conversation):
            log.info("conversation forgotten after %.0fs of quiet", self._settings.forget_after_s)
        # Even with nothing remembered: whatever is said next belongs to a new conversation.
        self._conversation.clear()

    def _cancel_timers(self) -> None:
        timers = (self._record_limit, self._hint_timer, self._dormant_timer, self._forget_timer)
        for handle in timers:
            _cancel(handle)


def _cancel(handle: asyncio.TimerHandle | None) -> None:
    if handle is not None:
        handle.cancel()
