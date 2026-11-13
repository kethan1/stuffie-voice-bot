"""Lucy's beak and body, moving in time with every sound she makes.

Lucy has two DC motors on an L293D driver (docs/HARDWARE.md). The beak motor
pushes her beak open against a spring, and the spring snaps it shut when the
motor lets go. The body motor wiggles her, one way or both. :class:`Puppeteer`
decides what they do:

* **The beak follows her voice.** Each clip is measured in 25 ms frames, and the
  beak opens as wide as the frame is loud: wide on a stressed syllable, shut in
  the gaps between words. It opens at once and closes over a few frames, the
  way a spring-loaded beak moves.
* **A honk is a flap.** Each goose call carries a honk cue
  (:class:`~lucy.core.audio.Cue`), so her body flaps through it, even inside a
  pre-rendered line such as the greeting.
* **She bobs as she talks.** When she starts a phrase after a pause, her body
  bobs, at most every ``bob_every_s``, alternating ways (if it can) like a sway.
* **She ponders.** When a child lets go of her wing, she tilts while she thinks.
* **She keeps still while listening.** Nothing moves while the microphone is
  open, so no motor noise gets into what she hears.
* **Chimes are not her voice,** so her beak stays shut through the earcons.

Timing: the speaker reports when each clip's first sample will reach the
loudspeaker (:class:`~lucy.core.ports.PlaybackListener`), so every move is timed
against the sound itself, not against when it was queued. Moves start
``lead_s`` early, because a motor takes a moment to get going. Barge-in stops
everything at once.

Safety and power: motor power is capped, no motor runs longer than
``max_hold_s`` without a short rest (so a stalled motor cannot overheat inside
the plush), and whenever nothing is moving both motors coast, with the
driver's outputs off.
"""

import asyncio
import heapq
import itertools
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lucy.core.audio import CueKind, Pcm
from lucy.core.ports import Motors
from lucy.core.types import Motor, Status

log = logging.getLogger(__name__)

Levels = npt.NDArray[np.float64]

FRAME_S = 0.025
"""How often the beak can change: 40 times a second, quicker than a motor can follow."""
_QUIET_DBFS = -30.0  # a frame this quiet (the gap between words) leaves the beak shut
_LOUD_DBFS = -12.0  # a frame this loud (a stressed syllable) opens it wide
_POSITIONS = 10  # openings are rounded to tenths: finer steps are invisible, and cost writes
_CLOSING_PER_FRAME = 0.34  # the beak takes about three frames (75 ms) to shut fully
_PAUSE_S = 0.25  # this much quiet before a sound makes it the start of a phrase
_PHRASE_LEVEL = 0.3  # how loud (as a beak opening) the start of a phrase must be
_CONTIGUOUS_S = 0.03  # a clip starting this close to the last one's end continues its sound
_SHUT_AFTER_S = 0.1  # if no clip continues the sound by then, the beak shuts
_HOLD_REST_S = 0.2  # a motor rests this long once it has run for max_hold_s
_SLACK_S = 0.002  # moves due this close together are made together


@dataclass(frozen=True, slots=True)
class MotionSettings:
    lead_s: float = 0.05
    """Start each move this much before its sound, since a motor takes a moment to respond."""
    beak_min_power: float = 0.35
    """The least power that moves the beak at all (it has to overcome the spring)."""
    beak_max_power: float = 0.8
    """Power for a wide-open beak: enough to open it fully, not enough to slam it."""
    body_power: float = 0.6
    body_two_way: bool = True
    """False if the body motor may only run forward (for example, against a return spring)."""
    bob_every_s: float = 2.5
    """Bob at the start of a phrase at most this often; 0 turns bobbing off (honks still flap)."""
    max_hold_s: float = 2.0
    """No motor runs longer than this without a short rest, so a stalled one cannot overheat."""


@dataclass(frozen=True, slots=True)
class Step:
    """Part of a body gesture: run for ``seconds`` at ``power`` (a fraction of ``body_power``)."""

    seconds: float
    power: float


Gesture = tuple[Step, ...]

PONDER: Gesture = (Step(0.25, 0.5), Step(0.25, -0.5))
"""A thoughtful tilt and back, while Lucy waits for her answer."""
BOB: Gesture = (Step(0.16, 0.7),)
"""A small bob at the start of a phrase."""


def flap(seconds: float) -> Gesture:
    """A flap that lasts as long as the honk it goes with: out, back, and still again."""
    return (Step(0.55 * seconds, 1.0), Step(0.35 * seconds, -1.0))


# -- Measuring the sound ----------------------------------------------------------------------


def loudness(audio: Pcm) -> Levels:
    """How far open the beak should be (0 to 1) for each frame of ``audio``, unsmoothed.

    Frames are :data:`FRAME_S` long from the start of the clip. The last one may be
    shorter, and is measured over the samples it has, so a word split across two
    streamed parts does not dip at the seam. Chimes count as silence.
    """
    samples = audio.samples().astype(np.float64) / 32768.0
    if samples.size == 0:
        return np.zeros(0)
    size = max(1, round(FRAME_S * audio.sample_rate))
    count = -(-samples.size // size)
    squares = np.zeros(count * size)
    squares[: samples.size] = samples * samples
    lengths = np.full(count, size)
    lengths[-1] = samples.size - (count - 1) * size
    rms = np.sqrt(squares.reshape(count, size).sum(axis=1) / lengths)
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-9))
    levels: Levels = np.clip((dbfs - _QUIET_DBFS) / (_LOUD_DBFS - _QUIET_DBFS), 0.0, 1.0)
    for cue in audio.cues:
        if cue.kind is CueKind.CHIME:
            levels[round(cue.start_s / FRAME_S) : math.ceil(cue.end_s / FRAME_S - 1e-9)] = 0.0
    return levels


def beak_openings(levels: Levels, start: float = 0.0) -> Levels:
    """Smooth ``levels`` the way the beak moves: open at once, shut over a few frames.

    ``start`` is the opening the beak already has. The result is rounded to tenths.
    """
    openings = np.empty_like(levels)
    opening = start
    for index, level in enumerate(levels):
        opening = level if level >= opening else max(level, opening - _CLOSING_PER_FRAME)
        openings[index] = opening
    rounded: Levels = np.round(openings * _POSITIONS) / _POSITIONS
    return rounded


def phrase_starts(levels: Levels, quiet_before_s: float) -> tuple[list[float], float]:
    """When phrases start after a pause (seconds into the clip), and the quiet left at its end.

    ``quiet_before_s`` is how long it had already been quiet when the clip started.
    """
    starts: list[float] = []
    quiet = quiet_before_s
    for index, level in enumerate(levels):
        if level < _PHRASE_LEVEL:
            quiet += FRAME_S
            continue
        if quiet >= _PAUSE_S:
            starts.append(index * FRAME_S)
        quiet = 0.0
    return starts, quiet


# -- Moving -----------------------------------------------------------------------------------


class Puppeteer:
    """Drives Lucy's motors in time with what she plays.

    It is both an :class:`~lucy.core.ports.Indicator` (what Lucy is doing) and a
    :class:`~lucy.core.ports.PlaybackListener` (what the speaker plays, and when).
    """

    def __init__(self, motors: Motors, settings: MotionSettings) -> None:
        self._motors = motors
        self._settings = settings
        # (when, tie-breaker, motor, power): a heap of moves, earliest first
        self._moves: list[tuple[float, int, Motor, float]] = []
        self._order = itertools.count()
        self._timer: asyncio.TimerHandle | None = None
        self._shut: asyncio.TimerHandle | None = None
        self._power = dict.fromkeys(Motor, 0.0)
        self._held: dict[Motor, asyncio.TimerHandle] = {}
        self._resting_until = dict.fromkeys(Motor, -math.inf)
        self._awake = False
        self._listening = False
        self._forget_sound()

    def _forget_sound(self) -> None:
        """Start afresh: nothing has played, and it has been quiet for a long time."""
        self._sound_end = -math.inf  # when the last clip reported finishes playing
        self._beak_end = 0.0  # the beak's opening at that moment
        self._quiet_end = math.inf  # how long it had been quiet by then
        self._body_free_at = -math.inf
        self._last_bob = -math.inf
        self._bob_way = 1.0

    # -- Indicator ---------------------------------------------------------------------------

    def show(self, status: Status) -> None:
        if status is Status.LISTENING:
            self._listening = True
            self._halt()
            return
        self._listening = False
        if status is Status.THINKING:
            self._perform(PONDER, _now(), take_over=False)
            self._schedule()

    # -- PlaybackListener --------------------------------------------------------------------

    def playing(self, audio: Pcm, at: float) -> None:
        if self._listening or audio.frames == 0:
            return  # the microphone is open (keep still and quiet), or there is no sound
        settings = self._settings
        gap = at - self._sound_end
        contiguous = gap < _CONTIGUOUS_S
        levels = loudness(audio)
        openings = beak_openings(levels, self._beak_end if contiguous else 0.0)
        starts, quiet_end = phrase_starts(levels, self._quiet_end + max(0.0, gap))
        begin = at - settings.lead_s

        last = self._beak_end if contiguous else math.nan  # nan: always set the first frame
        for index, opening in enumerate(openings):
            if opening != last:
                self._push(begin + index * FRAME_S, Motor.BEAK, self._beak_power(opening))
                last = opening

        honks = [
            (begin + cue.start_s, flap(cue.duration_s))
            for cue in audio.cues
            if cue.kind is CueKind.HONK
        ]
        planned: list[tuple[float, Gesture | None]] = [*honks]  # None: a bob, if there is room
        if settings.bob_every_s > 0:
            planned += [(begin + offset, None) for offset in starts]
        # In time order, so a bob between two honks is judged against the body as it is then.
        for when, gesture in sorted(planned, key=lambda move: move[0]):
            if gesture is not None:
                self._perform(gesture, when, take_over=True)  # a honk always flaps
                continue
            if when - self._last_bob < settings.bob_every_s:
                continue
            bob = (Step(BOB[0].seconds, BOB[0].power * self._bob_way),)
            if any(_overlap(when, bob, start, honk) for start, honk in honks):
                continue  # a honk is coming: the flap is enough
            if self._perform(bob, when, take_over=False):
                self._last_bob = when
                if settings.body_two_way:
                    self._bob_way = -self._bob_way

        self._sound_end = at + audio.duration
        self._beak_end = float(openings[-1])
        self._quiet_end = quiet_end
        self._watch_for_silence()
        self._schedule()

    def stopped(self) -> None:
        self._halt()

    def close(self) -> None:
        """Stop and sleep, for shutdown."""
        self._halt()
        self._motors.sleep()

    # -- Planning ----------------------------------------------------------------------------

    def _beak_power(self, opening: float) -> float:
        if opening <= 0:
            return 0.0
        low, high = self._settings.beak_min_power, self._settings.beak_max_power
        return low + opening * (high - low)

    def _perform(self, gesture: Gesture, when: float, *, take_over: bool) -> bool:
        """Plan a body gesture from ``when``. False if the body is busy and it may not take over.

        Taking over drops whatever the body had planned from ``when`` on.
        """
        if when < self._body_free_at:
            if not take_over:
                return False
            self._moves = [
                move for move in self._moves if move[2] is not Motor.BODY or move[0] < when
            ]
            heapq.heapify(self._moves)
        settings = self._settings
        for step in gesture:
            power = step.power * settings.body_power
            if power < 0 and not settings.body_two_way:
                power = 0.0
            self._push(when, Motor.BODY, power)
            when += step.seconds
        self._push(when, Motor.BODY, 0.0)
        self._body_free_at = when
        return True

    def _push(self, when: float, motor: Motor, power: float) -> None:
        heapq.heappush(self._moves, (when, next(self._order), motor, power))

    def _watch_for_silence(self) -> None:
        """Shut the beak shortly after the sound ends, unless another clip carries it on."""
        if self._shut is not None:
            self._shut.cancel()
            self._shut = None
        if self._beak_end > 0:
            when = self._sound_end - self._settings.lead_s + _SHUT_AFTER_S
            self._shut = asyncio.get_running_loop().call_at(when, self._shut_beak)

    def _shut_beak(self) -> None:
        self._shut = None
        self._beak_end = 0.0
        self._push(_now(), Motor.BEAK, 0.0)
        self._schedule()

    # -- Moving ------------------------------------------------------------------------------

    def _schedule(self) -> None:
        """Wake for the next move, or let both motors coast if there is none."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._moves:
            loop = asyncio.get_running_loop()
            self._timer = loop.call_at(self._moves[0][0], self._move_due)
        elif self._awake and not any(self._power.values()):
            self._awake = False
            self._motors.sleep()

    def _move_due(self) -> None:
        self._timer = None
        due = _now() + _SLACK_S
        while self._moves and self._moves[0][0] <= due:
            _, _, motor, power = heapq.heappop(self._moves)
            self._drive(motor, power)
        self._schedule()

    def _drive(self, motor: Motor, power: float) -> None:
        if power and _now() < self._resting_until[motor]:
            power = 0.0
        was = self._power[motor]
        if power == was:
            return
        self._power[motor] = power
        self._motors.drive(motor, power)
        if power:
            self._awake = True
            if not was:
                self._held[motor] = asyncio.get_running_loop().call_later(
                    self._settings.max_hold_s, self._overheld, motor
                )
        elif (held := self._held.pop(motor, None)) is not None:
            held.cancel()

    def _overheld(self, motor: Motor) -> None:
        self._held.pop(motor, None)
        log.debug("%s motor ran for %.1fs; resting it", motor, self._settings.max_hold_s)
        self._resting_until[motor] = _now() + _HOLD_REST_S
        self._drive(motor, 0.0)
        self._schedule()

    def _halt(self) -> None:
        """Drop every planned move and let both motors coast, at once."""
        for handle in (self._timer, self._shut):
            if handle is not None:
                handle.cancel()
        self._timer = self._shut = None
        self._moves.clear()
        for motor in Motor:
            self._drive(motor, 0.0)
        self._forget_sound()
        self._schedule()


def _now() -> float:
    return asyncio.get_running_loop().time()


def _overlap(when: float, gesture: Sequence[Step], start: float, other: Sequence[Step]) -> bool:
    end = when + sum(step.seconds for step in gesture)
    other_end = start + sum(step.seconds for step in other)
    return when < other_end and start < end
