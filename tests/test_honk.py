import math
import random

import numpy as np
import pytest

from lucy.core.audio import level_dbfs
from lucy.core.honk import Call, GooseCalls, HonkingVoice, Mood, goose_call, split_honks
from lucy.core.speech import render
from tests.fakes import FakeTTS

RATE = 16_000


@pytest.mark.parametrize(
    ("text", "segments"),
    [
        ("Honk!", [Call(1, Mood.EXCITED)]),
        ("Honk honk! Lucy is awake.", [Call(2, Mood.EXCITED), "Lucy is awake."]),
        ("Honk? I didn't hear that.", [Call(1, Mood.CURIOUS), "I didn't hear that."]),
        ("Honk, I need a nap.", [Call(1, Mood.CALM), "I need a nap."]),
        ("It stays up like a bird, honk!", ["It stays up like a bird,", Call(1, Mood.EXCITED)]),
        ("Isn't that neat? honk!", ["Isn't that neat?", Call(1, Mood.EXCITED)]),
        ("HOOONK!!", [Call(1, Mood.EXCITED)]),
        ("Honk-honk, honk.", [Call(3, Mood.CALM)]),
        ('He said, "Honk! Hi."', ['He said, "', Call(1, Mood.EXCITED), 'Hi."']),
        # An ordinary word is spoken.
        ("Geese honk to stay together.", ["Geese honk to stay together."]),
        ("Honk if you love geese.", ["Honk if you love geese."]),
        ("Lucy says honk", ["Lucy says honk"]),
        ("Geese are honking.", ["Geese are honking."]),
        ("", []),
    ],
)
def test_interjected_honks_are_calls(text: str, segments: list[str | Call]) -> None:
    assert split_honks(text) == segments


def pitch_hz(samples: np.ndarray, at_s: float, rate: int = RATE) -> float:
    """Autocorrelation pitch of the 30 ms window starting at ``at_s``, between 300 and 800 Hz.

    (Below that is the rasp's subharmonic, at half the pitch.)
    """
    window = samples[round(at_s * rate) : round((at_s + 0.03) * rate)]
    corr = np.correlate(window, window, "full")[window.size - 1 :]
    low, high = rate // 800, rate // 300
    return rate / (low + int(np.argmax(corr[low:high])))


class TestGooseCall:
    def test_a_question_rises_and_an_exclamation_falls(self) -> None:
        excited = goose_call(RATE, pitch_hz=440, rise=1.18, fall=0.86)
        curious = goose_call(RATE, pitch_hz=420, rise=1.0, fall=1.3)
        # The voice starts after the breath (~50 ms); compare its start with its end.
        assert pitch_hz(excited, 0.25) < pitch_hz(excited, 0.11)
        assert pitch_hz(curious, 0.25) > pitch_hz(curious, 0.11) * 1.15

    def test_about_a_third_of_a_second(self) -> None:
        assert 0.3 < goose_call(RATE).size / RATE < 0.42

    def test_needs_a_real_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="8 kHz"):
            goose_call(1_000)


class TestGooseCalls:
    def test_as_loud_as_the_voice_and_never_clipped(self) -> None:
        calls = GooseCalls(RATE, rng=random.Random(1))
        for mood in Mood:
            audio = calls.render(Call(1, mood))
            assert audio.sample_rate == RATE
            samples = audio.samples()
            assert int(np.abs(samples).max()) < 32767 * 0.96
            voiced = samples[np.abs(samples) > 0]
            rms = math.sqrt(float(np.mean((voiced / 32768.0) ** 2)))
            assert -20 < 20 * math.log10(rms) < -14  # near the voice's -17 dBFS

    def test_volume_and_count(self) -> None:
        loud, quiet = GooseCalls(RATE), GooseCalls(RATE, volume=0.5)
        one = loud.render(Call(1, Mood.EXCITED))
        assert level_dbfs(quiet.render(Call(1, Mood.EXCITED))) < level_dbfs(one) - 5
        assert loud.render(Call(2, Mood.EXCITED)).duration > one.duration * 1.5
        assert loud.render(Call(40, Mood.EXCITED)).duration < one.duration * 3.5  # at most 3


class TestHonkingVoice:
    def voice(self) -> tuple[HonkingVoice, FakeTTS]:
        tts = FakeTTS()
        tts.sample_rate = RATE
        return HonkingVoice(tts, GooseCalls(RATE)), tts

    async def test_calls_the_honk_and_speaks_the_rest(self) -> None:
        voice, tts = self.voice()
        parts = [part async for part in voice.stream("Honk! Hi there.")]
        assert tts.texts == ["Hi there."]
        assert parts[0].duration > 0.3  # the call comes first
        assert len(parts) == 2

    async def test_text_without_a_honk_goes_straight_to_the_voice(self) -> None:
        voice, tts = self.voice()
        await render(voice, "Geese honk to stay together.")
        assert tts.texts == ["Geese honk to stay together."]

    def test_the_voice_and_the_calls_share_a_sample_rate(self) -> None:
        with pytest.raises(ValueError, match="cannot join"):
            HonkingVoice(FakeTTS(), GooseCalls(RATE))
