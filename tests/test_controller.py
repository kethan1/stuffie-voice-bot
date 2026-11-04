import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from lucy.core.audio import Pcm
from lucy.core.controller import Controller, Earcons, SessionSettings
from lucy.core.conversation import Conversation
from lucy.core.phrases import Phrase
from lucy.core.types import PowerMode, Status
from tests.fakes import (
    FakeButton,
    FakeIndicator,
    FakeMic,
    FakePower,
    FakeResponder,
    FakeSpeaker,
    FakeWarmer,
    until,
)

SETTINGS = SessionSettings(
    tap_max_s=0.3,
    max_record_s=10.0,
    awake_s=60.0,
    forget_after_s=120.0,
    greet_on_start=False,
)


class Rig:
    def __init__(self, settings: SessionSettings = SETTINGS, **responder: float) -> None:
        self.button = FakeButton()
        self.mic = FakeMic()
        self.speaker = FakeSpeaker()
        self.indicator = FakeIndicator()
        self.power = FakePower()
        self.warmer = FakeWarmer()
        self.conversation = Conversation(4)
        self.responder = FakeResponder(**responder)
        self.controller = Controller(
            pipeline=self.responder,
            mic=self.mic,
            speaker=self.speaker,
            indicator=self.indicator,
            power=self.power,
            conversation=self.conversation,
            warmers=[self.warmer],
            earcons=Earcons.generate(1_000),
            settings=settings,
        )

    async def turn(self, held: float = 1.0) -> None:
        self.button.press()
        await until(lambda: self.mic.recording_now)
        self.button.release(after=held)
        await until(lambda: not self.mic.recording_now)


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    rig = Rig(respond_s=0.0)
    task = asyncio.create_task(rig.controller.run(rig.button))
    yield rig
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await rig.controller.close()


async def start(rig: Rig) -> asyncio.Task[None]:
    return asyncio.create_task(rig.controller.run(rig.button))


async def test_press_listens_and_release_responds(rig: Rig) -> None:
    rig.button.press()
    await until(lambda: rig.indicator.current is Status.LISTENING)
    assert rig.mic.recording_now
    assert rig.warmer.count == 1  # connection warmed while the child is still talking
    rig.button.release(after=1.5)
    await until(lambda: len(rig.responder.recordings) == 1)
    await until(lambda: rig.indicator.current is Status.IDLE)
    # Start-up settles into STANDBY; the turn goes ACTIVE, then back to STANDBY.
    await until(lambda: len(rig.power.modes) == 3)
    assert rig.power.modes == [PowerMode.STANDBY, PowerMode.ACTIVE, PowerMode.STANDBY]
    assert Status.THINKING in rig.indicator.history


async def test_earcons_mark_listening_and_thinking(rig: Rig) -> None:
    await rig.turn()
    await until(lambda: len(rig.speaker.played) == 2)
    earcons = Earcons.generate(1_000)
    assert rig.speaker.played == [earcons.listening, earcons.thinking]


async def test_tap_when_idle_explains_how_to_talk(rig: Rig) -> None:
    await rig.turn(held=0.1)
    await until(lambda: rig.responder.said == [Phrase.HOLD_TO_TALK])
    assert rig.responder.recordings == []


async def test_tap_while_talking_just_stops() -> None:
    rig = Rig(respond_s=5.0)
    task = await start(rig)
    await rig.turn()
    await until(lambda: len(rig.responder.recordings) == 1)
    await rig.turn(held=0.1)
    await until(lambda: rig.responder.cancelled == 1)
    await until(lambda: rig.indicator.current is Status.IDLE)
    assert rig.responder.said == []
    assert len(rig.responder.recordings) == 1
    task.cancel()
    await rig.controller.close()


async def test_long_press_while_talking_interrupts_and_asks_again() -> None:
    rig = Rig(respond_s=5.0)
    task = await start(rig)
    await rig.turn()
    await until(lambda: len(rig.responder.recordings) == 1)
    await rig.turn(held=1.0)
    await until(lambda: len(rig.responder.recordings) == 2)
    assert rig.responder.cancelled == 1
    task.cancel()
    await rig.controller.close()


async def test_recording_stops_at_the_limit() -> None:
    rig = Rig(replace(SETTINGS, max_record_s=0.05, tap_max_s=0.01))
    task = await start(rig)
    rig.button.press()
    await until(lambda: len(rig.responder.recordings) == 1)  # no release needed
    assert not rig.mic.recording_now
    rig.button.release(after=30.0)  # the late real release is ignored
    await asyncio.sleep(0.01)
    assert len(rig.responder.recordings) == 1
    task.cancel()
    await rig.controller.close()


async def test_dozes_then_forgets() -> None:
    rig = Rig(replace(SETTINGS, awake_s=0.05, forget_after_s=0.1))
    task = await start(rig)
    rig.conversation.record("q", "a")
    await rig.turn()
    await until(lambda: PowerMode.DORMANT in rig.power.modes)
    assert rig.speaker.closed >= 1
    await until(lambda: len(rig.conversation) == 0)
    task.cancel()
    await rig.controller.close()


async def test_turn_budget_protects_the_quota() -> None:
    rig = Rig(replace(SETTINGS, max_turns_per_hour=1))
    task = await start(rig)
    await rig.turn()
    await until(lambda: len(rig.responder.recordings) == 1)
    await rig.turn()
    await until(lambda: rig.responder.said == [Phrase.TIRED])
    assert len(rig.responder.recordings) == 1
    task.cancel()
    await rig.controller.close()


async def test_greets_on_start() -> None:
    rig = Rig(replace(SETTINGS, greet_on_start=True))
    task = await start(rig)
    await until(lambda: rig.responder.said == [Phrase.GREETING])
    await until(lambda: rig.indicator.current is Status.IDLE)
    task.cancel()
    await rig.controller.close()


async def test_close_releases_the_microphone_mid_press() -> None:
    rig = Rig()
    task = await start(rig)
    rig.button.press()
    await until(lambda: rig.mic.recording_now)
    task.cancel()
    await rig.controller.close()
    assert not rig.mic.recording_now
    assert rig.speaker.closed == 1
    assert isinstance(rig.mic.recording, Pcm)
