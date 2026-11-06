"""Lucy's honk: a goose call in place of the spoken word "honk".

Read aloud by a speech synthesizer, "Honk!" is a person saying a word. Lucy is
a goose, so when "honk" is an interjection ("Honk!", "Honk honk!", "..., honk!")
she makes a goose call instead. Used as an ordinary word ("geese honk to stay
together") it is still spoken.

The calls are synthesised at start-up, like the earcons, so no audio files are
shipped. Each one is about a third of a second:

* **h:** a short breath, shaped like the vowel that follows;
* **aw:** a buzzy, harmonic-rich voice whose pitch jumps up and then glides
  down, with a little period-doubling rasp (the throaty edge of a real goose);
* **ng:** the formants close into a quieter nasal hum;
* **k:** a brief closure and a soft burst.

Punctuation sets the mood: "Honk!" is bright, "Honk?" rises like a question,
and "Honk." or "Honk," is calmer. Each mood has a few slightly different
calls, picked at random, so a honk never sounds like the same clip on repeat.
Calls are normalised to the loudness of Lucy's voice, and each one is cued as a
honk (:class:`~lucy.core.audio.Cue`) so her body flaps in time with it.
"""

import math
import random
import re
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from lucy.core.audio import CueKind, Pcm
from lucy.core.ports import TextToSpeech

Samples = npt.NDArray[np.float64]

_HONK = r"h+o+n+k+"
_RUN = re.compile(rf"\b{_HONK}(?:[\s,\u2013\u2014-]+{_HONK})*\b(?P<punct>[!?.\u2026,;:]*)", re.I)
_OPENERS = frozenset(".,!?;:\u2026\"'(\u201c\u2018\u2013\u2014-")  # what may come before a honk
_MAX_HONKS = 3
_TARGET_RMS = 10 ** (-17 / 20)  # about as loud as Lucy's voice (Piper and Speechify alike)


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


def _add_words(segments: list[str | Call], words: str) -> None:
    words = words.strip()
    if any(ch.isalnum() for ch in words):
        segments.append(words)


# (pitch in Hz, how far it jumps up, where it ends relative to the start, rasp)
_MOOD_SHAPES = {
    Mood.EXCITED: (440.0, 1.18, 0.86, 0.3),
    Mood.CURIOUS: (420.0, 1.0, 1.3, 0.25),
    Mood.CALM: (380.0, 1.08, 0.85, 0.25),
}
# Formants (Hz, bandwidth Hz, gain) of the open "aw" vowel and the nasal "ng" it closes into.
_AW = ((730, 110, 1.0), (1090, 120, 0.85), (2440, 200, 0.35), (3400, 300, 0.12))
_NG = ((300, 80, 1.0), (1650, 160, 0.18), (1950, 160, 0.14), (3200, 300, 0.03))


def goose_call(
    sample_rate: int,
    *,
    pitch_hz: float = 440.0,
    rise: float = 1.18,
    fall: float = 0.86,
    rasp: float = 0.3,
    seed: int = 0,
) -> Samples:
    """One honk as float samples peaking at 1.0. ``fall`` is the final pitch over ``pitch_hz``."""
    if sample_rate < 8_000:
        raise ValueError(f"a goose call needs at least 8 kHz audio, not {sample_rate} Hz")
    rng = np.random.default_rng(seed)
    vowel_s, nasal_s = 0.17, 0.07
    voiced = _voiced(
        sample_rate,
        rng,
        pitch_hz=pitch_hz,
        rise=rise,
        fall=fall,
        rasp=rasp,
        vowel_s=vowel_s,
        nasal_s=nasal_s,
    )

    h_frames = round(0.07 * sample_rate)
    breath = _shaped_noise(rng, h_frames, sample_rate, _AW)
    breath *= 0.5 * np.sin(np.linspace(0.0, np.pi, h_frames)) ** 0.7
    burst_frames = round(0.025 * sample_rate)
    burst = _shaped_noise(rng, burst_frames, sample_rate, ((1500, 400, 1.0),))
    burst *= 0.12 * np.exp(-np.linspace(0.0, 5.0, burst_frames))

    start = h_frames - h_frames // 4  # the voice starts as the breath fades
    closure = round(0.045 * sample_rate)
    out = np.zeros(start + voiced.size + closure + burst_frames)
    out[:h_frames] += breath
    out[start : start + voiced.size] += voiced
    out[-burst_frames:] += burst
    honk: Samples = out / np.max(np.abs(out))
    return honk


def _voiced(
    rate: int,
    rng: np.random.Generator,
    *,
    pitch_hz: float,
    rise: float,
    fall: float,
    rasp: float,
    vowel_s: float,
    nasal_s: float,
) -> Samples:
    seconds = vowel_s + nasal_s
    t = np.arange(round(seconds * rate)) / rate
    jump = np.sin(np.clip(t / 0.04, 0.0, 1.0) * np.pi / 2)
    f0 = pitch_hz * (0.88 + (rise - 0.88) * jump)
    f0 *= 1 + (fall / rise - 1) * np.clip((t - 0.04) / (seconds - 0.04), 0.0, 1.0) ** 1.3
    f0 *= 1 + 0.012 * np.interp(t, np.linspace(0.0, seconds, 12), rng.uniform(-1, 1, 12))
    phase = 2 * np.pi * np.cumsum(f0) / rate
    closing = np.clip((t - 0.8 * vowel_s) / (nasal_s + 0.2 * vowel_s), 0.0, 1.0)
    closing = closing * closing * (3 - 2 * closing)  # smooth "aw" -> "ng"

    voice = np.zeros(t.size)
    for k in range(1, max(1, int(0.45 * rate / (pitch_hz * max(rise, fall)))) + 1):
        harmonic = k * f0
        gain = np.full(t.size, 0.02 / k)  # a little buzz between the formants
        for (fa, ba, ga), (fb, bb, gb) in zip(_AW, _NG, strict=True):
            centre = fa + (fb - fa) * closing
            width = ba + (bb - ba) * closing
            gain += (ga + (gb - ga) * closing) / (1 + ((harmonic - centre) / width) ** 2)
        voice += gain * np.sin(k * phase)
    voice *= 1 + rasp * (1 - closing) * np.cos(phase / 2)
    envelope = np.clip(t / 0.012, 0.0, 1.0) * (1 - 0.55 * closing)
    envelope *= np.clip((seconds - t) / 0.02, 0.0, 1.0)
    shaped: Samples = voice / np.max(np.abs(voice)) * envelope
    return shaped


def _shaped_noise(
    rng: np.random.Generator, frames: int, rate: int, formants: Sequence[tuple[float, float, float]]
) -> Samples:
    spectrum = np.fft.rfft(rng.standard_normal(frames))
    freqs = np.fft.rfftfreq(frames, 1 / rate)
    spectrum *= sum(g / (1 + ((freqs - fc) / bw) ** 2) for fc, bw, g in formants)
    noise: Samples = np.fft.irfft(spectrum, frames)
    noise /= np.max(np.abs(noise)) + 1e-12
    return noise


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
        for mood, (pitch_hz, rise, fall, rasp) in _MOOD_SHAPES.items():
            self._calls[mood] = [
                self._level(
                    goose_call(
                        sample_rate,
                        pitch_hz=pitch_hz * (0.96 + 0.09 * index / max(1, variants - 1)),
                        rise=rise,
                        fall=fall,
                        rasp=rasp,
                        seed=index,
                    ),
                    volume,
                )
                for index in range(variants)
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
        call = Pcm.from_float(np.clip(scaled, -0.95, 0.95), self.sample_rate)
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
