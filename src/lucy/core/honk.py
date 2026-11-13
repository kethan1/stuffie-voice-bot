"""Lucy's honk: a goose call in place of the spoken word "honk".

Read aloud by a speech synthesizer, "Honk!" is a person saying a word. Lucy is
a goose, so when "honk" is an interjection ("Honk!", "Honk honk!", "..., honk!")
she makes a goose call instead. Used as an ordinary word ("geese honk to stay
together") it is still spoken.

The calls are synthesised at start-up, like the earcons, so no audio files are
shipped. They are built from measurements of real Canada geese (two free
recordings on Wikimedia Commons, "Branta canadensis.ogg" and "Low flyover by
five Canadian Geese.flac", from different birds, which agree). A honk lasts
about 0.3 s and is not one sound but three:

* **The growl.** A honk starts with about 150 ms of quiet, low pulses, 130 to
  200 a second and getting quicker: the "ha" of "ha-RONK". The windpipe rings
  at about 1.3 kHz after each pulse. It is 15 to 25 dB quieter than what follows.
* **The break.** In the growl's last 35 ms its pulses speed up to half the
  honk's pitch, and then the voice breaks up the last octave in a few
  milliseconds. All of it is one train of pulses: the honk is the same train
  with every other partial gone (the opposite of period doubling).
* **The honk.** Short, loud, and much higher than a person's "honk": 600 to
  850 Hz, with its second partial the loudest, a dip near 3 kHz and a brassy
  second peak around 4 kHz. It starts a little flat, holds for about 40 ms, and
  falls by a fifth as it dies away over the next 60 ms, growing duller as the
  beak closes. In moments it drops back into period doubling: the rasp.

Every number above is in the synthesis below, and ``tests/test_honk.py``
measures them back off Lucy's calls (the growl's pulse rate, the honk's pitch
and its fall, the spectrum, the loudness of each part).

Punctuation sets the mood: "Honk!" is bright, "Honk?" ends on a rise like a
question (the one thing here a goose does not do), and "Honk." or "Honk," is
lower and longer. Each mood has a few slightly different calls, picked at
random, so a honk never sounds like the same clip on repeat. Calls are levelled
to sound about as loud as the loud parts of Lucy's speech, and each one is cued
as a honk (:class:`~lucy.core.audio.Cue`) so her body flaps in time with it.
"""

import math
import random
import re
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from lucy.core.audio import CueKind, Pcm, soft_limit
from lucy.core.ports import TextToSpeech

Samples = npt.NDArray[np.float64]

_HONK = r"h+o+n+k+"
_RUN = re.compile(rf"\b{_HONK}(?:[\s,\u2013\u2014-]+{_HONK})*\b(?P<punct>[!?.\u2026,;:]*)", re.I)
_OPENERS = frozenset(".,!?;:\u2026\"'(\u201c\u2018\u2013\u2014-")  # what may come before a honk
_MAX_HONKS = 3
_TARGET_RMS = 10 ** (-18.5 / 20)  # to the ear, about as loud as the loud parts of her speech


class Mood(StrEnum):
    EXCITED = "!"
    CURIOUS = "?"
    CALM = "."


@dataclass(frozen=True, slots=True)
class Call:
    """One or more honks said as an interjection."""

    count: int
    mood: Mood


def split_honks(text: str) -> list[str | Call]:
    """Split ``text`` into the words to speak and the honks to call, in order."""
    segments: list[str | Call] = []
    position = 0
    for match in _RUN.finditer(text):
        before = text[: match.start()].rstrip()
        after = text[match.end() :].strip()
        if before and before[-1] not in _OPENERS:
            continue  # "geese honk ...": an ordinary word
        if not match["punct"] and after:
            continue  # "honk if you love geese"
        _add_words(segments, text[position : match.start()])
        punct = match["punct"]
        mood = Mood.CURIOUS if "?" in punct else Mood.EXCITED if "!" in punct else Mood.CALM
        segments.append(Call(len(re.findall(_HONK, match[0], re.I)), mood))
        position = match.end()
    _add_words(segments, text[position:])
    return segments


def starts_with_honk(text: str) -> bool:
    """True if ``text`` opens with an interjected honk ("Honk! Did you know...")."""
    segments = split_honks(text)
    return bool(segments) and isinstance(segments[0], Call)


def _add_words(segments: list[str | Call], words: str) -> None:
    words = words.strip()
    if any(ch.isalnum() for ch in words):
        segments.append(words)


@dataclass(frozen=True, slots=True)
class Shape:
    """How one call goes. The defaults are a Canada goose's honk, as measured."""

    pitch_hz: float = 740.0
    """The honk's pitch. Geese differ, from about 580 Hz to 850 Hz."""
    end: float = 0.82
    """Where the honk's pitch ends, over ``pitch_hz``. A goose's falls; above 1 it rises."""
    growl_s: float = 0.15
    """The quiet, low growl that leads in to the honk."""
    honk_s: float = 0.11
    rasp: float = 0.2
    """How strongly the honk drops back into period doubling for a moment (0: a clean honk)."""


_PLAIN_CALL = Shape()
_MOOD_SHAPES = {
    Mood.EXCITED: Shape(pitch_hz=760.0, rasp=0.25),
    Mood.CURIOUS: Shape(pitch_hz=710.0, end=1.22, growl_s=0.12, honk_s=0.15, rasp=0.15),
    Mood.CALM: Shape(pitch_hz=680.0, end=0.8, growl_s=0.13, honk_s=0.18, rasp=0.15),
}

# The measurements (see the module docstring). Spectra are (Hz, dB below the strongest partial).
_GROWL_PITCH = (1 / 5.3, 1 / 4.5)  # the growl's pulse rate over the honk's pitch: start, end
_SWEEP_S = 0.035  # the growl's last moments, when its pulses speed up to half the honk's pitch
_BREAK_S = 0.008  # ... and how quickly the voice then breaks up the last octave
_GROWL_SPECTRUM = (
    (0, -40),
    (250, -20),
    (400, -8),
    (1050, -7),
    (1200, -5),
    (1330, 0),  # the windpipe rings here after every pulse
    (1480, -7),
    (1650, -17),
    (1900, -19),
    (2050, -13),
    (2300, -17),
    (2800, -20),
    (3200, -28),
    (3800, -19),
    (4400, -21),
    (5000, -30),
    (9000, -55),
)
_HONK_SPECTRUM = (
    (0, -40),
    (300, -25),
    (600, -5),
    (750, -3),
    (1100, -1),
    (1450, 0),  # the second partial of a honk is its loudest
    (1700, 0),
    (2150, -5),
    (2500, -15),
    (2900, -18),
    (3300, -16),
    (3600, -6),  # the brassy edge: a second peak, above a dip
    (4300, -7),
    (5000, -12),
    (5800, -15),
    (6800, -24),
    (9000, -45),
)
# As the beak closes, each octave above 600 Hz loses this much more.
_CLOSING_SPECTRUM = ((0, 0), (600, 0), (1200, -8), (2400, -16), (4800, -24), (9000, -31))
_TOP_HZ = 8000.0  # too little of a honk is above this to matter


class _Resonance:
    """What a spectrum does to each partial: how loud it comes out (dB) and how late (phase).

    The phase is the one a real resonance with that spectrum has (minimum phase, found
    with the cepstrum). With it, each pulse is followed by its ringing, as in a goose's
    windpipe. Without it, the ringing would come as much before each pulse as after.
    """

    def __init__(self, spectrum: tuple[tuple[float, float], ...], points: int = 2048) -> None:
        self._hz = np.linspace(0.0, spectrum[-1][0], points + 1)
        self._db = np.interp(self._hz, *np.array(spectrum, dtype=np.float64).T)
        cepstrum = np.fft.irfft(self._db * (np.log(10) / 20))
        cepstrum[1:points] *= 2
        cepstrum[points + 1 :] = 0
        self._phase = np.unwrap(np.fft.rfft(cepstrum).imag)

    def db(self, hz: Samples) -> Samples:
        db: Samples = np.interp(hz, self._hz, self._db)
        return db

    def phase(self, hz: Samples) -> Samples:
        phase: Samples = np.interp(hz, self._hz, self._phase)
        return phase


_GROWL = _Resonance(_GROWL_SPECTRUM)
_OPEN = _Resonance(_HONK_SPECTRUM)
_CLOSING = _Resonance(_CLOSING_SPECTRUM)


def goose_call(sample_rate: int, shape: Shape = _PLAIN_CALL, *, seed: int = 0) -> Samples:
    """One call as float samples peaking at 1.0: the growl, the break, and the honk."""
    if sample_rate < 8_000:
        raise ValueError(f"a goose call needs at least 8 kHz audio, not {sample_rate} Hz")
    if shape.growl_s < 2 * _SWEEP_S or shape.honk_s < 2 * _BREAK_S:
        raise ValueError(f"a goose call needs a longer growl and honk than {shape}")
    rng = np.random.default_rng(seed)
    t = np.arange(round((shape.growl_s + shape.honk_s) * sample_rate)) / sample_rate
    growling = np.clip(t / (shape.growl_s - _SWEEP_S), 0.0, 1.0)  # how far into the growl
    sweep = np.clip((t - shape.growl_s) / _SWEEP_S + 1, 0.0, 1.0)  # 0 growl, 1 from the break
    into = np.clip((t - shape.growl_s) / shape.honk_s, 0.0, 1.0)  # how far into the honk
    broken = _smoothstep((t - shape.growl_s) / _BREAK_S)  # 0 until the voice breaks, then 1

    # Pulses: the growl's rate, sweeping up to half the honk's pitch. The honk keeps that
    # rate with its odd partials gone, which is the octave above: its pitch.
    growl_hz = shape.pitch_hz * (_GROWL_PITCH[0] + (_GROWL_PITCH[1] - _GROWL_PITCH[0]) * growling)
    arch = 1 - 0.05 * (1 - _smoothstep(into / 0.15))  # a honk starts a little flat
    glide = 1 + (shape.end - 1) * _smoothstep((into - 0.4) / 0.5)
    half_hz = 0.5 * shape.pitch_hz * arch * glide
    pulse_hz = growl_hz * (half_hz / growl_hz) ** sweep
    pulse_hz *= 1 + 0.012 * _wobble(t, rng, every_s=0.008)  # jitter
    phase = 2 * np.pi * np.cumsum(pulse_hz) / sample_rate
    # Period doubling: all of it before the break, then only in moments of rasp and as
    # the honk dies away.
    doubled = 0.03 + shape.rasp * _stretches(t, rng, every_s=0.02, share=0.35)
    doubled += 0.3 * _smoothstep((into - 0.75) / 0.25)
    voice = _pulses(
        sample_rate,
        pulse_hz,
        phase,
        opened=_smoothstep(sweep),
        closing=_smoothstep((into - 0.6) / 0.4),
        odd=1 - broken * (1 - np.clip(doubled, 0.0, 1.0)),
    )

    # Loudness, in dB below the peak: the growl swells from -24 to -15, the sweep carries
    # it up to the break, and the honk dies away faster and faster.
    level_db = -24.0 + 9.0 * growling + 13.0 * sweep**1.5 + 2.0 * broken
    level_db -= 30.0 * np.clip((into - 0.1) / 0.9, 0.0, 1.0) ** 1.5
    shimmer = 1 + (0.12 - 0.09 * broken) * _wobble(t, rng, every_s=0.007)
    breath = 0.06 * _shaped_noise(rng, t.size, sample_rate)
    edges = _smoothstep(t / 0.01) * _smoothstep((t[-1] - t) / 0.012)
    out = (voice * shimmer + breath) * 10 ** (level_db / 20) * edges
    call: Samples = out / np.max(np.abs(out))
    return call


def _pulses(
    rate: int,
    pulse_hz: Samples,
    phase: Samples,
    *,
    opened: Samples,
    closing: Samples,
    odd: Samples,
) -> Samples:
    """A train of pulses through the goose's resonances, at a steady loudness (RMS 1).

    Every partial is a cosine, in step with the rest, which is what makes them pulses and
    not a hum. ``opened`` goes from the growl's spectrum (0) to the honk's (1), ``closing``
    dulls it again as the beak closes, and ``odd`` scales the odd partials: at 0 the pitch
    is an octave up. The spectrum changes slowly, so each partial's loudness and lateness
    are worked out every 32 samples and interpolated between: the same sound, for a
    fraction of the work.
    """
    index = np.arange(pulse_hz.size)
    coarse = index[::32]
    top = min(0.45 * rate, _TOP_HZ)
    orders = np.arange(1, int(top / float(np.min(pulse_hz))) + 1)
    hz = orders[:, None] * pulse_hz[coarse]
    # Decibels and phases both add, so the resonances mix as simply as this.
    mix = ((_GROWL, 1 - opened[coarse]), (_OPEN, opened[coarse]), (_CLOSING, closing[coarse]))
    db, lag = np.zeros(hz.shape), np.zeros(hz.shape)
    for resonance, share in mix:
        db += share * resonance.db(hz)
        lag += share * resonance.phase(hz)
    gain = 10 ** (db / 20) * (hz < top)
    gain[0::2] *= odd[coarse]  # orders 1, 3, 5, ...
    gain /= np.sqrt(0.5 * np.sum(gain**2, axis=0))
    voice = np.zeros(pulse_hz.size)
    for order, loudness, late in zip(orders, gain, lag, strict=True):
        if loudness.any():  # (the highest partials are only there in the growl)
            voice += np.interp(index, coarse, loudness) * np.cos(
                order * phase + np.interp(index, coarse, late)
            )
    return voice


def _shaped_noise(rng: np.random.Generator, frames: int, rate: int) -> Samples:
    """Breath: noise with the growl's spectrum, at RMS 1."""
    spectrum = np.fft.rfft(rng.standard_normal(frames))
    spectrum *= 10 ** (_GROWL.db(np.fft.rfftfreq(frames, 1 / rate)) / 20)
    noise: Samples = np.fft.irfft(spectrum, frames)
    noise /= np.sqrt(np.mean(noise**2)) + 1e-12
    return noise


def _wobble(t: Samples, rng: np.random.Generator, *, every_s: float) -> Samples:
    """A random wander between -1 and 1, with a new value every ``every_s``."""
    points = np.arange(0.0, float(t[-1]) + every_s, every_s)
    wobble: Samples = np.interp(t, points, rng.uniform(-1.0, 1.0, points.size))
    return wobble


def _stretches(t: Samples, rng: np.random.Generator, *, every_s: float, share: float) -> Samples:
    """1 in random stretches (about ``share`` of the time, each a multiple of ``every_s``)."""
    points = np.arange(0.0, float(t[-1]) + every_s, every_s)
    on: Samples = np.interp(t, points, (rng.uniform(0.0, 1.0, points.size) < share) * 1.0)
    return on


def _smoothstep(x: Samples) -> Samples:
    x = np.clip(x, 0.0, 1.0)
    smooth: Samples = x * x * (3 - 2 * x)
    return smooth


class GooseCalls:
    """Pre-rendered honks for every mood, at the loudness of Lucy's voice."""

    def __init__(
        self,
        sample_rate: int,
        *,
        volume: float = 1.0,
        variants: int = 3,
        rng: random.Random | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._rng = rng or random.Random()
        self._calls: dict[Mood, list[Pcm]] = {}
        for mood, shape in _MOOD_SHAPES.items():
            pitches = [
                shape.pitch_hz * (0.95 + 0.1 * index / max(1, variants - 1))
                for index in range(variants)
            ]
            self._calls[mood] = [
                self._level(
                    goose_call(sample_rate, replace(shape, pitch_hz=pitch), seed=index), volume
                )
                for index, pitch in enumerate(pitches)
            ]
        self._lead = Pcm.silence(0.04, sample_rate)
        self._gap = Pcm.silence(0.07, sample_rate)

    def render(self, call: Call) -> Pcm:
        count = max(1, min(call.count, _MAX_HONKS))
        # Earlier honks in "honk honk!" are bright; the last one carries the mood.
        moods = [Mood.EXCITED] * (count - 1) + [call.mood]
        audio = self._lead
        for index, mood in enumerate(moods):
            if index:
                audio += self._gap
            audio += self._rng.choice(self._calls[mood])
        return audio + self._gap

    def _level(self, samples: Samples, volume: float) -> Pcm:
        rms = math.sqrt(float(np.mean(samples * samples)))
        scaled = samples * (_TARGET_RMS * volume / rms)
        call = Pcm.from_float(0.95 * soft_limit(scaled / 0.95), self.sample_rate)  # no clipping
        return call.cued(CueKind.HONK)


class HonkingVoice:
    """A voice that calls each interjected "honk" instead of saying it."""

    def __init__(self, tts: TextToSpeech, calls: GooseCalls) -> None:
        if calls.sample_rate != tts.sample_rate:
            raise ValueError(
                f"honks at {calls.sample_rate} Hz cannot join a {tts.sample_rate} Hz voice"
            )
        self._tts = tts
        self._calls = calls

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        for segment in split_honks(text):
            if isinstance(segment, Call):
                yield self._calls.render(segment)
                continue
            async with aclosing(self._tts.stream(segment)) as parts:
                async for part in parts:
                    yield part
