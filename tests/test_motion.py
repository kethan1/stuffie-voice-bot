"""Lucy's moving beak and body (lucy.core.motion), with motors that record each command.

Most tests report a clip as having started in the past, so every move it plans is due at
once and the sequence can be checked without waiting. The few that check timing wait for
real, for a fraction of a second.
"""

import asyncio

import numpy as np
import pytest

from lucy.core.audio import CueKind, Pcm, chime
from lucy.core.honk import Call, GooseCalls, HonkingVoice, Mood
from lucy.core.motion import (
    MotionSettings,
    Puppeteer,
    beak_openings,
    loudness,
    phrase_starts,
)
from lucy.core.speech import render
from lucy.core.types import Motor, Status
from tests.fakes import FakeMotors, FakeTTS, tone

RATE = 8_000
STILL_BODY = MotionSettings(lead_s=0.0, bob_every_s=0.0)


def loud(seconds: float) -> Pcm:
    return tone(seconds, RATE, dbfs=-9.0)


def quiet(seconds: float) -> Pcm:
    return Pcm.silence(seconds, RATE)


def rig(settings: MotionSettings = STILL_BODY) -> tuple[Puppeteer, FakeMotors]:
    motors = FakeMotors()
    return Puppeteer(motors, settings), motors


def now() -> float:
    return asyncio.get_running_loop().time()


async def play_at_once(puppeteer: Puppeteer, audio: Pcm) -> None:
    """Report ``audio`` as having started long ago, and let every move it plans happen."""
    puppeteer.playing(audio, now() - 60.0)
    await asyncio.sleep(0.01)


class TestLoudness:
    def test_silence_keeps_the_beak_shut(self) -> None:
        assert not loudness(quiet(0.2)).any()

    def test_a_loud_syllable_opens_it_wide(self) -> None:
        levels = loudness(loud(0.2))
        assert len(levels) == 8  # 25 ms frames
        assert levels.min() > 0.95

    def test_the_gaps_between_words_shut_it(self) -> None:
        assert not loudness(tone(0.1, RATE, dbfs=-40.0)).any()

    def test_ordinary_speech_opens_it_part_way(self) -> None:
        assert 0.3 < loudness(tone(0.1, RATE, dbfs=-21.0)).mean() < 0.7

    def test_a_short_last_frame_is_measured_on_its_own(self) -> None:
        levels = loudness(loud(0.105))  # 4 whole frames and 5 ms
        assert len(levels) == 5
        assert levels[-1] == pytest.approx(levels[0], abs=0.05)

    def test_chimes_are_not_her_voice(self) -> None:
        ding = chime([(880.0, 0.1)], RATE, volume=0.9)
        levels = loudness(loud(0.1) + ding)
        assert levels[:4].min() > 0.9
        assert not levels[4:].any()


class TestShape:
    def test_the_beak_opens_at_once_and_shuts_over_a_few_frames(self) -> None:
        openings = beak_openings(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        assert openings.tolist() == [0.0, 1.0, 1.0, 0.7, 0.3, 0.0, 0.0]

    def test_it_carries_on_from_where_the_beak_already_is(self) -> None:
        assert beak_openings(np.zeros(3), start=1.0).tolist() == [0.7, 0.3, 0.0]

    def test_a_phrase_starts_after_a_pause(self) -> None:
        levels = np.array([0.0] * 12 + [0.8] * 4 + [0.0] * 4 + [0.8] * 2)
        starts, quiet_end = phrase_starts(levels, quiet_before_s=0.0)
        assert starts == [pytest.approx(0.3)]  # not after the short 0.1 s gap
        assert quiet_end == 0.0

    def test_the_pause_can_begin_in_an_earlier_clip(self) -> None:
        assert phrase_starts(np.array([0.8]), quiet_before_s=1.0)[0] == [0.0]
        assert phrase_starts(np.zeros(4), quiet_before_s=0.1)[1] == pytest.approx(0.2)


class TestBeak:
    async def test_follows_the_voice_and_starts_early(self) -> None:
        settings = MotionSettings(lead_s=0.05, bob_every_s=0.0)
        puppeteer, motors = rig(settings)
        at = now() + 0.1
        puppeteer.playing(quiet(0.05) + loud(0.1) + quiet(0.15), at)
        await asyncio.sleep(0.4)
        beak = motors.of(Motor.BEAK)
        opened = [when for when, power in beak if power > 0]
        assert opened[0] - at == pytest.approx(0.05 - settings.lead_s, abs=0.02)
        powers = [power for _, power in beak if power > 0]
        assert max(powers) == pytest.approx(settings.beak_max_power)
        assert min(powers) >= settings.beak_min_power
        assert beak[-1][1] == 0.0  # shut again
        assert not motors.awake  # the driver sleeps once Lucy is still

    async def test_parts_of_one_word_join_without_a_dip(self) -> None:
        puppeteer, motors = rig()
        start = now() - 60.0
        first = loud(0.11)  # ends part-way through a frame, as streamed parts do
        puppeteer.playing(first, start)
        puppeteer.playing(loud(0.1), start + first.duration)
        await asyncio.sleep(0.01)
        assert motors.powers(Motor.BEAK) == [STILL_BODY.beak_max_power, 0.0]

    async def test_shuts_when_the_sound_stops_mid_word(self) -> None:
        puppeteer, motors = rig()
        await play_at_once(puppeteer, loud(0.1))  # no trailing silence, and nothing after it
        assert motors.powers(Motor.BEAK)[-1] == 0.0

    async def test_stays_shut_for_chimes(self) -> None:
        puppeteer, motors = rig()
        await play_at_once(puppeteer, chime([(587.0, 0.05), (880.0, 0.07)], RATE, volume=0.9))
        assert motors.moves == []

    async def test_a_motor_that_runs_too_long_rests(self) -> None:
        settings = MotionSettings(lead_s=0.0, bob_every_s=0.0, max_hold_s=0.1)
        puppeteer, motors = rig(settings)
        at = now()
        puppeteer.playing(loud(0.3), at)
        await asyncio.sleep(0.2)
        (opened, power), (rested, rest) = motors.of(Motor.BEAK)[:2]
        assert power > 0
        assert rest == 0.0
        assert rested - opened == pytest.approx(0.1, abs=0.03)


class TestBody:
    async def test_a_honk_is_a_flap(self) -> None:
        puppeteer, motors = rig()
        honk = GooseCalls(RATE).render(Call(1, Mood.EXCITED))
        await play_at_once(puppeteer, honk)
        power = STILL_BODY.body_power
        assert motors.powers(Motor.BODY) == [power, -power, 0.0]
        assert max(motors.powers(Motor.BEAK)) > 0  # the beak opens for the honk too

    async def test_a_flap_lasts_as_long_as_its_honk(self) -> None:
        puppeteer, motors = rig()
        honk = GooseCalls(RATE).render(Call(1, Mood.CALM))
        (cue,) = honk.cues
        at = now()
        puppeteer.playing(honk, at)
        await asyncio.sleep(cue.start_s + 0.9 * cue.duration_s + 0.03)
        body = motors.of(Motor.BODY)
        assert body[0][0] - at == pytest.approx(cue.start_s, abs=0.02)
        assert body[-1][0] - at == pytest.approx(cue.start_s + 0.9 * cue.duration_s, abs=0.02)

    async def test_a_one_way_body_never_runs_backward(self) -> None:
        puppeteer, motors = rig(MotionSettings(lead_s=0.0, bob_every_s=0.0, body_two_way=False))
        await play_at_once(puppeteer, GooseCalls(RATE).render(Call(2, Mood.EXCITED)))
        assert motors.powers(Motor.BODY) == [0.6, 0.0, 0.6, 0.0]

    async def test_every_honk_in_a_rendered_line_flaps(self) -> None:
        tts = FakeTTS()
        tts.sample_rate = RATE
        line = await render(HonkingVoice(tts, GooseCalls(RATE)), "Honk! Hi there. Honk honk!")
        assert [cue.kind for cue in line.cues] == [CueKind.HONK] * 3
        puppeteer, motors = rig()
        await play_at_once(puppeteer, line)
        assert sum(power > 0 for power in motors.powers(Motor.BODY)) == 3

    async def test_she_bobs_at_the_start_of_phrases_but_not_too_often(self) -> None:
        settings = MotionSettings(lead_s=0.0, bob_every_s=0.8)
        puppeteer, motors = rig(settings)
        speech = loud(0.3) + quiet(0.4) + loud(0.3) + quiet(0.4) + loud(0.3)  # phrases 0.7 s apart
        await play_at_once(puppeteer, speech)
        bob = 0.7 * settings.body_power
        assert motors.powers(Motor.BODY) == [pytest.approx(bob), 0.0, pytest.approx(-bob), 0.0]

    async def test_she_bobs_between_the_honks_of_one_line(self) -> None:
        settings = MotionSettings(lead_s=0.0, bob_every_s=0.5)
        puppeteer, motors = rig(settings)
        honk = GooseCalls(RATE).render(Call(1, Mood.EXCITED))
        await play_at_once(puppeteer, honk + quiet(0.3) + loud(0.5) + quiet(0.3) + honk)
        flap, bob = settings.body_power, pytest.approx(0.7 * settings.body_power)
        assert motors.powers(Motor.BODY) == [flap, -flap, 0.0, bob, 0.0, flap, -flap, 0.0]

    async def test_a_honk_is_not_also_a_bob(self) -> None:
        puppeteer, motors = rig(MotionSettings(lead_s=0.0, bob_every_s=0.5))
        await play_at_once(puppeteer, GooseCalls(RATE).render(Call(1, Mood.EXCITED)))
        assert motors.powers(Motor.BODY) == [0.6, -0.6, 0.0]

    async def test_she_ponders_while_thinking(self) -> None:
        puppeteer, motors = rig()
        puppeteer.show(Status.THINKING)
        await asyncio.sleep(0.55)
        assert motors.powers(Motor.BODY) == [0.3, -0.3, 0.0]
        assert not motors.awake


class TestStillness:
    async def test_nothing_moves_while_she_listens(self) -> None:
        puppeteer, motors = rig()
        puppeteer.show(Status.LISTENING)
        await play_at_once(puppeteer, loud(0.2))  # the "I'm listening" chime, or anything else
        assert motors.moves == []

    async def test_listening_stops_her_at_once(self) -> None:
        puppeteer, motors = rig(MotionSettings(lead_s=0.0))
        puppeteer.playing(GooseCalls(RATE).render(Call(1, Mood.EXCITED)), now())
        await asyncio.sleep(0.15)
        assert motors.awake
        puppeteer.show(Status.LISTENING)
        assert motors.powers(Motor.BEAK)[-1] == motors.powers(Motor.BODY)[-1] == 0.0
        assert not motors.awake
        moved = len(motors.moves)
        await asyncio.sleep(0.25)  # past the end of the flap it had planned
        assert len(motors.moves) == moved  # nothing planned was left behind

    async def test_barge_in_stops_everything_planned(self) -> None:
        puppeteer, motors = rig()
        puppeteer.playing(loud(0.1) + quiet(0.1) + loud(0.1), now())
        await asyncio.sleep(0.05)
        puppeteer.stopped()
        moved = len(motors.moves)
        await asyncio.sleep(0.25)
        assert len(motors.moves) == moved
        assert motors.powers(Motor.BEAK)[-1] == 0.0
        assert not motors.awake

    async def test_she_moves_again_after_listening(self) -> None:
        puppeteer, motors = rig()
        puppeteer.show(Status.LISTENING)
        puppeteer.show(Status.SPEAKING)
        await play_at_once(puppeteer, loud(0.1))
        assert motors.powers(Motor.BEAK)[0] > 0

    async def test_closing_stops_and_sleeps(self) -> None:
        puppeteer, motors = rig()
        puppeteer.playing(loud(0.5), now())
        await asyncio.sleep(0.02)
        puppeteer.close()
        assert motors.powers(Motor.BEAK)[-1] == 0.0
        assert not motors.awake
