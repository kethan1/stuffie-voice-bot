"""Lucy's goose voice: smaller, honky, raspy, at her usual level, and seamless as it streams."""

from collections.abc import AsyncGenerator

import numpy as np
import pytest

from lucy.core.audio import Pcm
from lucy.core.goose_voice import GooseVoice, Goosifier, Rasp, design_filter, pitch_ratio
from lucy.core.speech import render

RATE = 22_050


class Recording:
    """A voice that "says" a fixed recording, in parts of ``part_s``."""

    def __init__(self, samples: np.ndarray, part_s: float = 0.1) -> None:
        self.sample_rate = RATE
        self._audio = Pcm.from_float(samples, RATE)
        self._part = round(part_s * RATE) * 2

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        data = self._audio.data
        for start in range(0, len(data), self._part):
            yield Pcm(data[start : start + self._part], RATE)


def speech_like_noise(seconds: float = 2.0, dbfs: float = -17.0) -> np.ndarray:
    """Noise with the long-term spectrum of speech: the same power in each octave from
    100 Hz to 400 Hz, and 6.5 dB less per octave above that."""
    rng = np.random.default_rng(1)
    frames = round(seconds * RATE)
    freqs = np.maximum(np.fft.rfftfreq(frames, 1 / RATE), 1.0)
    per_octave = np.where(freqs < 400, 1.0, (400 / freqs) ** (6.5 / 3.01))
    spectrum = np.fft.rfft(rng.standard_normal(frames)) * np.sqrt(per_octave / freqs)
    noise = np.fft.irfft(np.where(freqs > 100, spectrum, 0), frames)
    return noise * (10 ** (dbfs / 20) / np.sqrt(np.mean(noise**2)))


def voice_like_tone(pitch_hz: float, seconds: float = 1.0, level: float = 0.2) -> np.ndarray:
    """A steady voiced sound: a fundamental and its overtones, each weaker than the last."""
    t = np.arange(round(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * k * pitch_hz * t + k) / k for k in range(1, 9))
    return level * tone / np.abs(tone).max()


def goosify(samples: np.ndarray, strength: float, sizes: list[int] | None = None) -> np.ndarray:
    goose = Goosifier(design_filter(RATE, strength), strength, RATE)
    if sizes is None:
        return np.concatenate((goose.process(samples), goose.finish()))
    parts, start = [], 0
    for size in sizes:
        parts.append(goose.process(samples[start : start + size]))
        start += size
    return np.concatenate((*parts, goose.process(samples[start:]), goose.finish()))


def band_power(samples: np.ndarray, low_hz: float, high_hz: float) -> float:
    power = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(samples.size, 1 / RATE)
    return float(power[(freqs > low_hz) & (freqs < high_hz)].sum() / samples.size)


def test_raises_her_pitch() -> None:
    tone = 0.3 * np.sin(2 * np.pi * 200 * np.arange(RATE) / RATE)
    out = goosify(tone, 1.0)
    spectrum = np.abs(np.fft.rfft(out * np.hanning(out.size)))
    peak_hz = np.fft.rfftfreq(out.size, 1 / RATE)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(200 * pitch_ratio(1.0), abs=1.5)
    assert out.size == pytest.approx(tone.size / pitch_ratio(1.0), rel=0.01)
    assert pitch_ratio(0.7) == pytest.approx(1.175)  # nearly three semitones, as she ships


def test_gives_her_a_gooses_resonances_and_drops_the_chest_tones() -> None:
    speech = speech_like_noise()
    out = goosify(speech, 0.7)

    def gain_db(hz: float) -> float:
        # What came in around ``hz`` comes out around ``hz`` times the pitch ratio.
        low, high = hz * 0.88, hz * 1.12
        ratio = pitch_ratio(0.7)
        before = band_power(speech, low / ratio, high / ratio)
        return float(10 * np.log10(band_power(out, low, high) / before))

    assert gain_db(1350) > gain_db(2900) + 6  # honky: the windpipe's ring, over the dip
    assert gain_db(4000) > gain_db(2900) + 3  # brassy: the second peak
    assert gain_db(1350) > gain_db(200) + 8  # and no deep chest tones


@pytest.mark.parametrize("strength", [0.3, 0.7, 1.0])
def test_keeps_her_level(strength: float) -> None:
    speech = speech_like_noise()
    out = goosify(speech, strength)
    assert 10 * np.log10(np.mean(out**2) / np.mean(speech**2)) == pytest.approx(0.0, abs=1.0)


def test_loud_speech_is_rounded_off_not_clipped() -> None:
    out = goosify(speech_like_noise(dbfs=-6.0), 1.0)
    assert np.abs(out).max() < 1.0


def test_parts_join_without_a_seam() -> None:
    speech = np.concatenate((speech_like_noise(0.5), voice_like_tone(210, 0.5)))
    whole = goosify(speech, 0.7)
    rng = np.random.default_rng(3)
    parted = goosify(speech, 0.7, sizes=[int(n) for n in rng.integers(1, 2_000, 12)])
    assert parted.size == whole.size
    assert np.allclose(parted, whole, atol=1e-9)


class TestRasp:
    """Period doubling: every other cycle of the voice is weaker."""

    @staticmethod
    def voice(period: int, seconds: float = 1.0) -> np.ndarray:
        """A voiced sound whose every cycle is exactly ``period`` samples long."""
        return voice_like_tone(RATE / period, seconds)

    @staticmethod
    def alternation(samples: np.ndarray, period: int, start_s: float, end_s: float) -> float:
        """How unequal neighbouring cycles are, from 0 (all alike) towards 1: the rasp's depth.

        (Wherever in the cycle the rasp switches: the cut into cycles that shows it best.)
        """
        first = round(start_s * RATE / period) * period
        cycles = (round(end_s * RATE) - first) // (2 * period) * 2 - 2

        def unequal(offset: int) -> float:
            start = first + offset
            part = samples[start : start + cycles * period].reshape(cycles // 2, 2, period)
            loud = np.sqrt(np.mean(part**2, axis=2))
            return float(np.mean(np.abs(loud[:, 0] - loud[:, 1]) / (loud[:, 0] + loud[:, 1])))

        return max(unequal(offset) for offset in range(0, period, 4))

    @pytest.mark.parametrize("period", [120, 100, 76])  # 184 Hz, 220 Hz, 290 Hz
    def test_follows_the_pitch_of_her_voice(self, period: int) -> None:
        voice = self.voice(period)
        assert self.alternation(voice, period, 0.3, 1.0) < 0.001  # none to begin with
        assert self.alternation(Rasp(RATE, 1.0).process(voice), period, 0.3, 1.0) > 0.12

    def test_is_strongest_as_she_starts_a_phrase(self) -> None:
        phrase = np.concatenate((np.zeros(50 * 100), self.voice(100, 1.6)))
        out = Rasp(RATE, 0.7).process(phrase)
        starting = self.alternation(out, 100, 0.24, 0.32)  # her first 100 ms
        later = self.alternation(out, 100, 0.55, 1.8)  # a whole sway of the steady rasp
        assert starting > 0.3
        assert 0.08 < later < starting / 2

    def test_scales_with_strength_down_to_nothing(self) -> None:
        voice = self.voice(100)
        light = self.alternation(Rasp(RATE, 0.3).process(voice), 100, 0.3, 1.0)
        heavy = self.alternation(Rasp(RATE, 1.0).process(voice), 100, 0.3, 1.0)
        assert heavy > 2.5 * light > 0
        assert np.array_equal(Rasp(RATE, 0.0).process(voice), voice)

    def test_leaves_silence_silent_and_her_loudness_alone(self) -> None:
        assert not Rasp(RATE, 1.0).process(np.zeros(RATE)).any()
        voice = self.voice(100)
        out = Rasp(RATE, 1.0).process(voice)
        assert 10 * np.log10(np.mean(out**2) / np.mean(voice**2)) == pytest.approx(0.0, abs=1.0)


async def test_a_voice_through_it_streams_part_by_part() -> None:
    speech = speech_like_noise(1.0)
    voice = GooseVoice(Recording(speech), 0.7)
    parts = [part async for part in voice.stream("Hello!")]
    assert len(parts) >= 10  # streamed as it came, not held back
    joined = await render(voice, "Hello!")
    heard = Pcm.from_float(speech, RATE).samples() / 32768.0  # as the voice hands it over
    expected = Pcm.from_float(goosify(heard, 0.7), RATE)
    assert np.abs(joined.samples().astype(int) - expected.samples().astype(int)).max() <= 1


async def test_strength_zero_is_her_plain_voice() -> None:
    recording = Recording(speech_like_noise(0.3))
    plain = [part async for part in recording.stream("Hi")]
    assert [part async for part in GooseVoice(recording, 0.0).stream("Hi")] == plain
    assert pitch_ratio(0.0) == 1.0


def test_strength_is_checked() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        GooseVoice(Recording(np.zeros(10)), 1.5)
