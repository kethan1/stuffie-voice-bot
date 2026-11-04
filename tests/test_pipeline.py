import asyncio

import pytest

from lucy.core.audio import Pcm
from lucy.core.conversation import Conversation
from lucy.core.errors import RateLimited, ServiceError, ServiceUnreachable
from lucy.core.metrics import TurnTimer
from lucy.core.phrases import Phrase, PhraseBook
from lucy.core.pipeline import TurnPipeline, TurnSettings, phrase_for
from lucy.core.safety import DistressScreen
from lucy.core.types import Role, Status
from lucy.persona import LUCY
from tests.fakes import FakeChat, FakeIndicator, FakeSpeaker, FakeSTT, FakeTTS, tone, until


class Rig:
    def __init__(self, *, stt: FakeSTT | None = None, chat: FakeChat | None = None) -> None:
        self.stt = stt or FakeSTT()
        self.chat = chat or FakeChat()
        self.tts = FakeTTS()
        self.speaker = FakeSpeaker()
        self.indicator = FakeIndicator()
        self.conversation = Conversation(4)
        self.pipeline = TurnPipeline(
            stt=self.stt,
            chat=self.chat,
            tts=self.tts,
            moderator=DistressScreen(),
            speaker=self.speaker,
            indicator=self.indicator,
            phrases=PhraseBook(LUCY.lines, self.tts),
            conversation=self.conversation,
            system_prompt=LUCY.system_prompt,
            settings=TurnSettings(sentence_pause_s=0.0, clause_pause_s=0.0),
        )
        self.timer = TurnTimer()

    async def respond(self, recording: Pcm | None = None) -> None:
        await self.pipeline.respond(recording or tone(1.0), self.timer)

    def spoken_phrase(self, phrase: Phrase) -> bool:
        return any(text in LUCY.lines[phrase] for text in self.tts.texts)


async def test_streams_reply_sentence_by_sentence() -> None:
    rig = Rig()
    await rig.respond()
    assert rig.tts.texts == ["Honk!", "The sky is blue because of sunlight.", "Neat?"]
    assert len(rig.speaker.played) == 3
    assert Status.SPEAKING in rig.indicator.history
    for stage in ("stt", "first_token", "first_synth", "first_audio"):
        assert rig.timer.get(stage) is not None


async def test_sends_system_prompt_history_and_question() -> None:
    rig = Rig()
    rig.conversation.record("hi", "Honk, hi!")
    await rig.respond()
    sent = rig.chat.calls[0]
    assert sent[0].role is Role.SYSTEM
    assert "Lucy" in sent[0].content
    assert [m.content for m in sent[1:]] == ["hi", "Honk, hi!", "why is the sky blue"]


async def test_remembers_the_exchange() -> None:
    rig = Rig()
    await rig.respond()
    last = rig.conversation.prompt("s", "next")[-2]
    assert last.role is Role.ASSISTANT
    assert last.content.startswith("Honk! The sky is blue")


async def test_silence_is_not_sent_to_the_cloud() -> None:
    rig = Rig()
    await rig.respond(Pcm.silence(2.0, 16_000))
    assert rig.stt.calls == []
    assert rig.spoken_phrase(Phrase.DIDNT_HEAR)


async def test_empty_transcript_asks_again() -> None:
    rig = Rig(stt=FakeSTT(text="  "))
    await rig.respond()
    assert rig.chat.calls == []
    assert rig.spoken_phrase(Phrase.DIDNT_HEAR)


async def test_distress_is_answered_locally() -> None:
    rig = Rig(stt=FakeSTT(text="my brother hits me"))
    await rig.respond()
    assert rig.chat.calls == []  # never forwarded to the language model
    assert rig.spoken_phrase(Phrase.TELL_A_GROWN_UP)
    assert len(rig.conversation) == 0


@pytest.mark.parametrize(
    ("error", "phrase"),
    [
        (ServiceUnreachable("no wifi"), Phrase.OFFLINE),
        (RateLimited("slow down", 30.0), Phrase.TIRED),
        (ServiceError("500"), Phrase.TANGLED),
        (RuntimeError("bug"), Phrase.TANGLED),
    ],
)
async def test_failures_are_explained(error: Exception, phrase: Phrase) -> None:
    rig = Rig(chat=FakeChat(error=error))
    await rig.respond()
    assert rig.spoken_phrase(phrase)
    assert rig.indicator.current is Status.ERROR
    assert phrase_for(error) is phrase


async def test_failure_after_speaking_stays_quiet() -> None:
    rig = Rig(chat=FakeChat(error=ServiceError("dropped"), error_after=3))
    await rig.respond()
    assert rig.tts.texts == ["Honk!", "The sky is blue because of sunlight."]
    assert not rig.spoken_phrase(Phrase.TANGLED)


async def test_empty_reply_is_an_error() -> None:
    rig = Rig(chat=FakeChat(deltas=["🦢 ", "**"]))
    await rig.respond()
    assert rig.spoken_phrase(Phrase.TANGLED)


async def test_barge_in_cancels_cleanly_and_keeps_partial_reply() -> None:
    rig = Rig(chat=FakeChat(delay=0.01))
    rig.speaker.seconds_per_clip = 0.05
    task = asyncio.create_task(rig.respond())
    await until(lambda: len(rig.speaker.played) >= 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert rig.chat.closed  # the HTTP stream was closed, not left dangling
    assert all(future.done() for _, future in rig.speaker.queued)
    remembered = rig.conversation.prompt("s", "next")[-2].content
    assert remembered.startswith("Honk!")
