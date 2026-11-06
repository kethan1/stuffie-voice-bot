import asyncio
from collections.abc import AsyncGenerator

import pytest

from lucy.core.audio import Pcm
from lucy.core.conversation import Conversation
from lucy.core.errors import ContentRefused, RateLimited, ServiceError, ServiceUnreachable
from lucy.core.metrics import TurnTimer
from lucy.core.phrases import Phrase, PhraseBook
from lucy.core.pipeline import Brain, TurnPipeline, TurnSettings, phrase_for
from lucy.core.safety import BlockedWords, DistressScreen
from lucy.core.types import Role, Status
from lucy.persona import LUCY
from tests.fakes import FakeChat, FakeIndicator, FakeSpeaker, FakeSTT, FakeTTS, tone, until


class Rig:
    def __init__(
        self,
        *,
        stt: FakeSTT | None = None,
        chat: FakeChat | None = None,
        local: tuple[FakeSTT, FakeChat] | None = None,
        cloud_reachable: bool = True,
        tts: FakeTTS | None = None,
        reply_screen: BlockedWords | None = None,
    ) -> None:
        self.stt = stt or FakeSTT()
        self.chat = chat or FakeChat()
        self.cloud_reachable = cloud_reachable
        self.local_stt, self.local_chat = local or (FakeSTT(), FakeChat())
        backup = (
            Brain("local", self.local_stt, self.local_chat, history_exchanges=1) if local else None
        )
        self.tts = tts or FakeTTS()
        self.speaker = FakeSpeaker()
        self.indicator = FakeIndicator()
        self.conversation = Conversation(4)
        self.pipeline = TurnPipeline(
            brain=Brain("cloud", self.stt, self.chat, reachable=lambda: self.cloud_reachable),
            backup=backup,
            tts=self.tts,
            moderator=DistressScreen(),
            speaker=self.speaker,
            indicator=self.indicator,
            phrases=PhraseBook(LUCY.lines, self.tts),
            conversation=self.conversation,
            system_prompt=LUCY.system_prompt,
            settings=TurnSettings(sentence_pause_s=0.0, clause_pause_s=0.0),
            reply_screen=reply_screen,
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


async def test_streamed_sentences_play_part_by_part() -> None:
    rig = Rig(tts=FakeTTS(parts=3))
    await rig.respond()
    assert len(rig.speaker.played) == 9  # three sentences, three parts each, in order
    assert [clip.frames for clip in rig.speaker.played[:3]] == [5, 5, 5]  # "Honk!"
    remembered = rig.conversation.prompt("s", "next")[-2].content
    assert remembered == "Honk! The sky is blue because of sunlight. Neat?"  # each sentence once


async def test_pauses_follow_each_sentence() -> None:
    rig = Rig()
    rig.pipeline._settings = TurnSettings(sentence_pause_s=0.01)
    await rig.respond()
    assert [any(clip.data) for clip in rig.speaker.played] == [True, False] * 3


class RefusingTTS(FakeTTS):
    """The cloud voice's content screen refuses any sentence mentioning ``word``."""

    def __init__(self, word: str) -> None:
        super().__init__()
        self.word = word

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        if self.word in text:
            self.texts.append(text)
            raise ContentRefused("HTTP 400: refused by the content policy (violence)")
        async for part in super().stream(text):
            yield part


async def test_a_refused_sentence_ends_the_reply_and_changes_the_subject() -> None:
    rig = Rig(tts=RefusingTTS("sunlight"))
    await rig.respond()
    assert "Neat?" not in rig.tts.texts  # nothing after the refused sentence
    assert rig.spoken_phrase(Phrase.CHANGE_SUBJECT)
    assert phrase_for(ContentRefused("refused")) is Phrase.CHANGE_SUBJECT


async def test_a_blocked_word_ends_the_reply_before_any_voice_hears_it() -> None:
    chat = FakeChat(("Sure! ", "It is spelled S H I T. ", "Neat?"))
    rig = Rig(chat=chat, reply_screen=BlockedWords())
    await rig.respond()
    assert rig.tts.texts[0] == "Sure!"
    assert not any("S H I T" in text or text == "Neat?" for text in rig.tts.texts)
    assert rig.spoken_phrase(Phrase.CHANGE_SUBJECT)
    assert rig.conversation.prompt("s", "next")[-2].content == "Sure!"  # only what was said


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


UNREACHABLE = ServiceUnreachable("no route to api.groq.com")
LOCAL_REPLY = ("Honk! I think ", "it is the sun. ")


async def test_cloud_unreachable_at_speech_moves_to_the_local_brain() -> None:
    rig = Rig(
        stt=FakeSTT(error=UNREACHABLE),
        local=(FakeSTT(text="why is the sky blue"), FakeChat(deltas=LOCAL_REPLY)),
    )
    await rig.respond()
    assert rig.chat.calls == []  # the cloud was never asked for a reply
    assert len(rig.local_stt.calls) == 1
    assert rig.spoken_phrase(Phrase.OWN_BRAIN)
    assert rig.tts.texts[-2:] == ["Honk!", "I think it is the sun."]
    assert rig.timer.get("fallback") is not None


async def test_cloud_unreachable_at_reply_moves_to_the_local_brain() -> None:
    rig = Rig(chat=FakeChat(error=UNREACHABLE), local=(FakeSTT(), FakeChat(deltas=LOCAL_REPLY)))
    await rig.respond()
    assert len(rig.stt.calls) == 1  # the cloud transcript is kept
    assert rig.local_stt.calls == []
    assert len(rig.local_chat.calls) == 1
    assert rig.spoken_phrase(Phrase.OWN_BRAIN)
    assert rig.tts.texts[-1] == "I think it is the sun."


async def test_known_outage_skips_the_cloud_entirely() -> None:
    rig = Rig(local=(FakeSTT(), FakeChat(deltas=LOCAL_REPLY)), cloud_reachable=False)
    await rig.respond()
    assert rig.stt.calls == []
    assert rig.chat.calls == []
    assert len(rig.local_chat.calls) == 1
    # Lucy explains the wait before the slower local answer.
    own_brain = next(i for i, t in enumerate(rig.tts.texts) if t in LUCY.lines[Phrase.OWN_BRAIN])
    assert own_brain < rig.tts.texts.index("Honk!")


async def test_local_brain_gets_less_history() -> None:
    rig = Rig(local=(FakeSTT(), FakeChat(deltas=LOCAL_REPLY)), cloud_reachable=False)
    for i in range(3):
        rig.conversation.record(f"q{i}", f"a{i}")
    await rig.respond()
    sent = [m.content for m in rig.local_chat.calls[0][1:]]
    assert sent == ["q2", "a2", "why is the sky blue"]


async def test_other_cloud_failures_do_not_use_the_local_brain() -> None:
    rig = Rig(chat=FakeChat(error=RateLimited("busy")), local=(FakeSTT(), FakeChat()))
    await rig.respond()
    assert rig.local_chat.calls == []
    assert rig.spoken_phrase(Phrase.TIRED)


async def test_never_mixes_brains_mid_reply() -> None:
    cloud = FakeChat(deltas=["Honk! ", "The sky"], error=UNREACHABLE, error_after=1)
    rig = Rig(chat=cloud, local=(FakeSTT(), FakeChat(deltas=LOCAL_REPLY)))
    await rig.respond()
    assert rig.local_chat.calls == []  # the cloud had started answering: no switch
    assert "I think it is the sun." not in rig.tts.texts


async def test_offline_everywhere_says_so() -> None:
    rig = Rig(stt=FakeSTT(error=UNREACHABLE), local=(FakeSTT(error=UNREACHABLE), FakeChat()))
    await rig.respond()
    assert rig.spoken_phrase(Phrase.OFFLINE)


async def test_without_a_backup_an_outage_is_reported() -> None:
    rig = Rig(stt=FakeSTT(error=UNREACHABLE), cloud_reachable=False)
    await rig.respond()
    assert len(rig.stt.calls) == 1  # with no backup, the cloud is still tried
    assert rig.spoken_phrase(Phrase.OFFLINE)
    assert not rig.spoken_phrase(Phrase.OWN_BRAIN)
