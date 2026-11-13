"""Lucy's goose voice: her speech, made more like a goose's, as it streams.

Lucy speaks with a natural human voice (Speechify, or Piper on the Pi), because children
have to understand every word. This filter gives that voice what makes a goose sound like a
goose, taken from the same measurements of real geese as her honk (:mod:`lucy.core.honk`):

* **Smaller.** A goose is a small animal with a high voice. The speech is resampled, which
  raises its pitch and its resonances together: the sound of a smaller creature. That
  speeds it up as well, so each voice is asked to speak that much more slowly in the first
  place (:func:`pitch_ratio`) and comes out at its usual pace.
* **Honky.** A goose's windpipe rings at about 1.35 kHz, the nasal note in every honk. A
  honk also has a dip near 3 kHz and a brassy second peak near 4 kHz. Her voice is given
  the same three, and loses the chest tones below 300 Hz that a goose does not have.
* **Raspy.** A goose's voice starts as a low growl and breaks upward into the honk, and
  drops back into period doubling in between: every other cycle of the voice comes out
  weaker, which is heard as a rasp an octave below the pitch. :class:`Rasp` does that to
  her voice, in step with its pitch, most strongly as she starts a phrase and lightly in
  between. It follows her pitch, so it sounds like her voice breaking, not like a buzz
  added on top.

``strength`` (``tts.goose_voice``) scales all three together: 0 leaves her voice untouched,
and 1 is as much of a goose as she can be while staying easy to understand. Honks are not
filtered, because they are goose calls already: this voice goes inside
:class:`~lucy.core.honk.HonkingVoice`.

Everything is done in numpy on each streamed part, with each stage's state carried into the
next part. The parts therefore join without a seam, and the work is small: about 0.4 ms per
100 ms of speech on a laptop.
"""

from collections.abc import AsyncGenerator
from contextlib import aclosing

import numpy as np
import numpy.typing as npt

from lucy.core.audio import Pcm, soft_limit
from lucy.core.ports import TextToSpeech

Samples = npt.NDArray[np.float64]

_PITCH_UP = 0.25  # at full strength: about four semitones higher
# The goose in the filter, at full strength: (Hz as she is heard, dB, width in octaves).
_BANDS = (
    (1350.0, 12.0, 0.33),  # the windpipe's ring: honky
    (2900.0, -4.0, 0.25),  # the dip above it
    (4000.0, 6.0, 0.3),  # the brassy edge
)
_LOW_CUT_HZ = 300.0
_TAPS = 255  # FIR length: fine enough for these shapes, and 6 ms of delay at 22 kHz

# The rasp, at full strength: how much weaker every other cycle of her voice is.
_RASP = 0.45  # all the time (it comes and goes: see _RASP_SWAY_HZ)
_RASP_ONSET = 0.5  # more, as she starts a phrase
_RASP_SWAY_HZ = 0.8
_ONSET_S = 0.15  # how long the start of a phrase lasts
_PITCH_TOP_HZ = 330.0  # her pitch is below this (as she is heard); her overtones are above
_QUIET = 10 ** (-50 / 20)  # nothing quieter than this is given a rasp


def pitch_ratio(strength: float) -> float:
    """How much higher the goose voice makes speech, and how much quicker, which is how
    much more slowly a voice has to speak to come out of it at its usual pace."""
    return 1.0 + _PITCH_UP * max(0.0, strength)


class GooseVoice:
    """A voice (:class:`~lucy.core.ports.TextToSpeech`) that sounds more like a goose."""

    def __init__(self, tts: TextToSpeech, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"the goose voice's strength must be between 0 and 1, not {strength}")
        self._tts = tts
        self._strength = strength
        self._taps = design_filter(tts.sample_rate, strength)

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        if self._strength == 0:
            async with aclosing(self._tts.stream(text)) as parts:
                async for part in parts:
                    yield part
            return
        goose = Goosifier(self._taps, self._strength, self.sample_rate)
        async with aclosing(self._tts.stream(text)) as parts:
            async for part in parts:
                audio = goose.process(part.samples() / 32768.0)
                if audio.size:
                    yield Pcm.from_float(audio, self.sample_rate)
        tail = goose.finish()
        if tail.size:
            yield Pcm.from_float(tail, self.sample_rate)


def design_filter(rate: int, strength: float) -> Samples:
    """The goose voice's FIR filter: a goose's resonances, no chest tones, and nothing
    above what resampling up by :func:`pitch_ratio` would fold back down (aliasing).

    It runs before the resampler, so it is designed that much lower: the resonances land
    where a goose's are once the pitch has gone up. Linear phase, and normalised so speech
    comes out at the level it went in (:func:`_speech_power`). Her beak opens as wide as
    before, then, and her volume settings mean what they did. To the ear she is a few
    decibels louder, because the sound has moved to where ears are keenest, as a honk's is.
    """
    n = 4096
    freqs = np.fft.rfftfreq(n, 1 / rate)
    heard = np.maximum(freqs * pitch_ratio(strength), 1.0)  # where each frequency ends up
    goose_db = np.zeros(freqs.size)
    for hz, db, octaves in _BANDS:
        goose_db += strength * db * np.exp(-0.5 * (np.log2(heard / hz) / octaves) ** 2)
    high_pass = heard**2 / np.sqrt(heard**4 + _LOW_CUT_HZ**4)
    low_cut = (1 - strength) + strength * high_pass
    anti_alias = 1.0 / np.sqrt(1.0 + (heard / (0.45 * rate)) ** 16)
    magnitude = 10 ** (goose_db / 20) * low_cut * anti_alias
    speech = _speech_power(freqs)
    magnitude *= np.sqrt(np.sum(speech) / np.sum(speech * magnitude**2))
    impulse = np.roll(np.fft.irfft(magnitude, n), _TAPS // 2)[:_TAPS]
    taps: Samples = impulse * np.hanning(_TAPS + 2)[1:-1]
    return taps


def _speech_power(hz: Samples) -> Samples:
    """How the power of speech is spread over frequency, on average (per Hz).

    About the same power in each octave up to 400 Hz, and 6.5 dB less per octave above
    that: the long-term average of speech (Byrne et al., 1994) made a little darker, to sit
    between Lucy's two voices. There is nothing below the lowest voices.
    """
    hz = np.maximum(hz, 1.0)
    per_octave = np.where(hz < 400.0, 1.0, (400.0 / hz) ** (6.5 / 3.01))
    power: Samples = per_octave / hz * (hz > 100.0)
    return power


class Goosifier:
    """The goose voice's signal chain for one utterance, fed one streamed part at a time."""

    def __init__(self, taps: Samples, strength: float, rate: int) -> None:
        self._filter = _Fir(taps)
        self._ratio = pitch_ratio(strength)
        self._pending = np.zeros(0)  # filtered samples the resampler has not finished with
        self._position = 0.0  # where the next output sample is read, within ``_pending``
        self._rasp = Rasp(rate, strength)
        self._tail = taps.size // 2

    def process(self, samples: npt.NDArray[np.floating]) -> Samples:
        """The goose version of the next part of the speech (a little shorter than it)."""
        return soft_limit(self._rasp.process(self._resample(self._filter(samples))))

    def finish(self) -> Samples:
        """The end of the utterance, still inside the filter (half its length)."""
        return self.process(np.zeros(self._tail))

    def _resample(self, filtered: Samples) -> Samples:
        """Read ``filtered`` ``ratio`` times faster, which raises its pitch by ``ratio``."""
        buffer = np.concatenate((self._pending, filtered))
        last = buffer.size - 1
        if last < 0 or self._position > last:
            self._pending = buffer
            return np.zeros(0)
        count = int((last - self._position) // self._ratio) + 1
        positions = self._position + self._ratio * np.arange(count)
        out: Samples = np.interp(positions, np.arange(buffer.size), buffer)
        following = self._position + self._ratio * count
        keep = int(following)
        self._pending = buffer[keep:]
        self._position = following - keep
        return out


class Rasp:
    """A goose's rasp: every other cycle of the voice made weaker (period doubling).

    How the cycles are found is how an octave pedal for a guitar finds them. The voice is
    low-passed until only its fundamental is left, and each time that swings up through
    zero is one cycle. (A trigger with hysteresis does the counting, so a ripple of the
    overtones is not taken for a cycle.) A switch flips on every cycle, and the voice is
    turned up while it is one way and down while it is the other. Where it loses count,
    on a consonant or a breath, the rasp turns irregular, as a goose's is.

    ``depth`` of that is highest for the first moments after a silence, as a honk starts
    with a growl, and sways slowly the rest of the time. Silence gets none.
    """

    def __init__(self, rate: int, strength: float) -> None:
        self._rate = rate
        self._steady = _RASP * strength
        self._onset = _RASP_ONSET * strength
        self._fundamental = _Fir(_low_pass(rate, _PITCH_TOP_HZ, round(0.006 * rate) | 1))
        self._swing = _Mean(max(1, round(0.008 * rate)))  # how far the fundamental is swinging
        self._soften = _Fir(_window(round(0.0025 * rate) | 1))  # takes the click off each flip
        self._now = _Mean(max(1, round(0.015 * rate)))  # her loudness (power) just now
        self._lately = _Mean(max(1, round(2 * _ONSET_S * rate)))  # ... and over the last while
        self._high = False  # the trigger: the fundamental was last seen above zero
        self._weak = 0  # the switch: 1 during the cycles that are turned down
        self._produced = 0  # samples so far, for the sway's phase

    def process(self, voice: Samples) -> Samples:
        if not voice.size:
            return voice
        return voice * (1.0 + self._depth(voice) * self._soften(self._cycles(voice)))

    def _cycles(self, voice: Samples) -> Samples:
        """+1 and -1 on alternate cycles of the voice."""
        low = self._fundamental(voice)
        margin = 0.5 * self._swing(np.abs(low)) + 1e-5
        # The trigger's state after each sample: the last of "above" and "below" to be seen.
        seen = np.where(low > margin, 1, np.where(low < -margin, -1, 0))
        latest = np.maximum.accumulate(np.where(seen != 0, np.arange(seen.size), -1))
        high = np.where(latest >= 0, seen[latest] > 0, self._high)
        rose = high & ~np.concatenate(([self._high], high[:-1]))
        weak = (self._weak + np.cumsum(rose)) % 2
        self._high = bool(high[-1])
        self._weak = int(weak[-1])
        cycles: Samples = 1.0 - 2.0 * weak.astype(np.float64)
        return cycles

    def _depth(self, voice: Samples) -> Samples:
        power = voice * voice
        now = np.maximum(self._now(power), 0.0)
        lately = np.maximum(self._lately(power), 0.0)
        # 1 as she starts after a silence, down to 0 over _ONSET_S of talking.
        starting = np.clip(1.0 - 2.0 * lately / (now + 1e-12), 0.0, 1.0)
        seconds = (self._produced + np.arange(voice.size)) / self._rate
        self._produced += voice.size
        sway = 0.6 + 0.4 * np.sin(2 * np.pi * _RASP_SWAY_HZ * seconds)
        audible = now / (now + _QUIET**2)
        depth: Samples = (self._steady * sway + self._onset * starting) * audible
        return depth


class _Fir:
    """An FIR filter that carries its history from one block of a stream to the next."""

    def __init__(self, taps: Samples) -> None:
        self._taps = taps
        self._history = np.zeros(taps.size - 1)

    def __call__(self, block: npt.NDArray[np.floating]) -> Samples:
        stream = np.concatenate((self._history, block))
        self._history = stream[stream.size - self._history.size :]
        out: Samples = np.convolve(stream, self._taps, mode="valid")
        return out


class _Mean:
    """The mean of the last ``width`` samples of a stream, block by block."""

    def __init__(self, width: int) -> None:
        self._width = width
        self._history = np.zeros(width)

    def __call__(self, block: Samples) -> Samples:
        stream = np.concatenate((self._history, block))
        self._history = stream[block.size :]
        total = np.cumsum(stream)
        mean: Samples = (total[self._width :] - total[: -self._width]) / self._width
        return mean


def _window(taps: int) -> Samples:
    """A smoothing window (Hann) that adds up to 1."""
    window: Samples = np.hanning(taps + 2)[1:-1]
    window /= window.sum()
    return window


def _low_pass(rate: int, cutoff_hz: float, taps: int) -> Samples:
    """A windowed-sinc low-pass filter with ``taps`` taps (an odd number)."""
    n = np.arange(taps) - taps // 2
    kernel: Samples = np.sinc(2 * cutoff_hz / rate * n) * np.hamming(taps)
    kernel /= kernel.sum()
    return kernel
