import math
import random

import numpy as np
import pytest

from lucy.core.audio import level_dbfs
from lucy.core.honk import (
    Call,
    GooseCalls,
    HonkingVoice,
    Mood,
    Shape,
    goose_call,
    split_honks,
    starts_with_honk,
)
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


def test_tells_a_reply_that_already_opens_with_a_honk() -> None:
    assert starts_with_honk("Honk! Geese can sleep while floating.")
    assert starts_with_honk("Honk honk, that's a great one.")
    assert not starts_with_honk("Geese honk to stay together.")
    assert not starts_with_honk("")


def pitch_hz(
    samples: np.ndarray, at_s: float, low_hz: float, high_hz: float, rate: int = RATE
) -> float:
    """Autocorrelation pitch of the 40 ms window starting at ``at_s``, within a range."""
    window = samples[round(at_s * rate) : round((at_s + 0.04) * rate)]
    corr = np.correlate(window, window, "full")[window.size - 1 :]
    low, high = round(rate / high_hz), round(rate / low_hz)
    return rate / (low + int(np.argmax(corr[low:high])))


def level_db(samples: np.ndarray, start_s: float, end_s: float) -> float:
    part = samples[round(start_s * RATE) : round(end_s * RATE)]
    return 10 * math.log10(float(np.mean(part**2)))


def share_of_energy(samples: np.ndarray, low_hz: float, high_hz: float) -> float:
    """The fraction of the honk's energy (the 60 ms after the break) between two frequencies."""
    honk = samples[round(0.155 * RATE) : round(0.215 * RATE)]
    power = np.abs(np.fft.rfft(honk * np.hanning(honk.size))) ** 2
    freqs = np.fft.rfftfreq(honk.size, 1 / RATE)
    return float(power[(freqs > low_hz) & (freqs < high_hz)].sum() / power.sum())


class TestGooseCall:
    """The default call: a 150 ms growl, the break, then a 110 ms honk at 740 Hz."""

    def test_a_low_growl_breaks_upward_into_the_honk(self) -> None:
        call = goose_call(RATE)
        growl = pitch_hz(call, 0.04, 80, 300)
        honk = pitch_hz(call, 0.165, 400, 1000)
        assert 125 < growl < 175  # pulses, as a real goose's growl measures
        assert honk == pytest.approx(740, rel=0.05)
        assert honk > 4 * growl  # "ha-RONK": more than two octaves up, in a moment

    def test_the_growl_is_quiet_and_the_honk_is_short_and_loud(self) -> None:
        call = goose_call(RATE)
        honk = level_db(call, 0.155, 0.185)
        assert level_db(call, 0.02, 0.1) < honk - 12
        assert level_db(call, 0.23, 0.26) < honk - 10  # it dies away fast

    def test_a_question_rises_and_an_exclamation_falls(self) -> None:
        excited = goose_call(RATE, Shape(end=0.82, honk_s=0.15))
        curious = goose_call(RATE, Shape(end=1.22, honk_s=0.15))
        # Compare the start of the honk (after the break) with its end.
        assert pitch_hz(excited, 0.255, 400, 1000) < pitch_hz(excited, 0.165, 400, 1000) * 0.9
        assert pitch_hz(curious, 0.255, 400, 1000) > pitch_hz(curious, 0.165, 400, 1000) * 1.12

    def test_brassy_like_a_goose_not_deep_like_a_person(self) -> None:
        call = goose_call(RATE, Shape(rasp=0.0))
        assert share_of_energy(call, 600, 1800) > 0.6  # the pitch and its second partial
        assert share_of_energy(call, 3300, 5500) > 0.05  # and a brassy edge above the dip
        assert share_of_energy(call, 3300, 5500) > 3 * share_of_energy(call, 2500, 3300)
        assert share_of_energy(call, 0, 500) < 0.02

    def test_rasp_drops_the_honk_back_into_period_doubling(self) -> None:
        # Sound at one and a half times the pitch (1110 Hz), where no partial of the honk is.
        # The rasp comes in random moments, so look at a few calls.
        def doubled(rasp: float) -> float:
            calls = (goose_call(RATE, Shape(rasp=rasp), seed=seed) for seed in range(6))
            return sum(share_of_energy(call, 1040, 1180) for call in calls)

        assert doubled(0.6) > 5 * doubled(0.0)

    def test_each_seed_is_a_slightly_different_call(self) -> None:
        assert not np.allclose(goose_call(RATE, seed=0), goose_call(RATE, seed=1))
        assert np.array_equal(goose_call(RATE, seed=2), goose_call(RATE, seed=2))

    def test_about_three_tenths_of_a_second_and_never_over_full_scale(self) -> None:
        call = goose_call(RATE)
        assert 0.22 < call.size / RATE < 0.4
        assert np.abs(call).max() == pytest.approx(1.0)

    def test_nothing_above_what_the_sample_rate_can_carry(self) -> None:
        call = goose_call(8_000)
        assert np.isfinite(call).all()
        assert pitch_hz(call, 0.165, 400, 1000, rate=8_000) == pytest.approx(740, rel=0.05)

    def test_needs_a_real_sample_rate_and_a_real_shape(self) -> None:
        with pytest.raises(ValueError, match="8 kHz"):
            goose_call(1_000)
        with pytest.raises(ValueError, match="longer growl"):
            goose_call(RATE, Shape(growl_s=0.01))


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
            assert -21.5 < 20 * math.log10(rms) < -15.5

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
