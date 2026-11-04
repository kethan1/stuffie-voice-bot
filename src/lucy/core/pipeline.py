"""One conversational turn, streamed end to end.

::

    recording ─► trim ─► STT ─► screen ─► LLM stream ─► chunker ─► TTS ─► speaker
                                              think()        voice()     speak()

The three stages on the right run concurrently and are joined by queues, so
Lucy starts talking as soon as the first clause has been synthesised. That
clause plays while the next one is synthesised, and the next while the LLM is
still writing. The queue to the speaker is bounded, which keeps memory flat and
limits wasted work when a child interrupts.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Sequence
from contextlib import aclosing
from dataclasses import dataclass, field

from lucy.core.audio import Pcm, trim_silence
from lucy.core.chunker import Chunk, SentenceChunker
from lucy.core.conversation import Conversation
from lucy.core.errors import (
    AudioDeviceError,
    AuthenticationError,
    ConfigError,
    LucyError,
    RateLimited,
    ServiceError,
    ServiceUnreachable,
)
from lucy.core.metrics import TurnTimer
from lucy.core.phrases import Phrase, PhraseBook
from lucy.core.ports import ChatModel, Indicator, Moderator, Speaker, SpeechToText, TextToSpeech
from lucy.core.tasks import run_together
from lucy.core.textnorm import to_speakable
from lucy.core.types import Concern, Message, Status

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TurnSettings:
    min_speech_s: float = 0.3
    """Less sound than this (after trimming silence) means nothing was said."""
    speech_threshold_dbfs: float = -50.0
    """Frames quieter than this count as silence."""
    sentence_pause_s: float = 0.2
    clause_pause_s: float = 0.08
    first_clause_min_words: int = 4
    max_chunk_words: int = 30
    log_transcripts: bool = False
    """Privacy: children's words are kept out of the logs unless this is enabled."""


@dataclass(slots=True)
class _Turn:
    timer: TurnTimer
    spoken: list[str] = field(default_factory=list)


class TurnPipeline:
    def __init__(
        self,
        *,
        stt: SpeechToText,
        chat: ChatModel,
        tts: TextToSpeech,
        moderator: Moderator,
        speaker: Speaker,
        indicator: Indicator,
        phrases: PhraseBook,
        conversation: Conversation,
        system_prompt: str,
        settings: TurnSettings,
    ) -> None:
        self._stt = stt
        self._chat = chat
        self._tts = tts
        self._moderator = moderator
        self._speaker = speaker
        self._indicator = indicator
        self._phrases = phrases
        self._conversation = conversation
        self._system_prompt = system_prompt
        self._settings = settings

    async def respond(self, recording: Pcm, timer: TurnTimer) -> None:
        """Answer one recording. Never raises except for cancellation (barge-in)."""
        turn = _Turn(timer)
        try:
            await self._respond(recording, turn)
        except asyncio.CancelledError:
            log.info("turn interrupted: %s", timer.report())
            raise
        except Exception as error:
            if isinstance(error, LucyError):
                log.warning("turn failed: %s: %s (%s)", type(error).__name__, error, timer.report())
            else:
                log.exception("turn failed unexpectedly (%s)", timer.report())
            if not turn.spoken:
                await self.say(phrase_for(error), status=Status.ERROR)
        else:
            log.info("turn complete: %s", timer.report())

    async def say(self, phrase: Phrase, *, status: Status = Status.SPEAKING) -> None:
        """Speak one of Lucy's fixed lines. Never raises except for cancellation."""
        try:
            audio = await self._phrases.render(phrase)
            self._indicator.show(status)
            await self._speaker.enqueue(audio)
        except Exception:
            # Nothing left to say it with: the LED is the only way left to signal trouble.
            log.exception("could not speak phrase %s", phrase)
            self._indicator.show(Status.ERROR)

    async def _respond(self, recording: Pcm, turn: _Turn) -> None:
        settings = self._settings
        speech = trim_silence(recording, settings.speech_threshold_dbfs)
        if speech.duration < settings.min_speech_s:
            log.info("heard no speech (%.2fs recorded)", recording.duration)
            await self.say(Phrase.DIDNT_HEAR)
            return

        text = (await self._stt.transcribe(speech)).strip()
        turn.timer.mark("stt")
        if not text:
            log.info("transcript was empty (%.2fs of sound)", speech.duration)
            await self.say(Phrase.DIDNT_HEAR)
            return
        if settings.log_transcripts:
            log.info("child said: %r", text)

        assessment = await self._moderator.assess(text)
        if not assessment.allowed:
            log.info("screened: %s (%s)", assessment.concern.value, ", ".join(assessment.reasons))
            await self.say(
                Phrase.TELL_A_GROWN_UP
                if assessment.concern is Concern.NEEDS_GROWN_UP
                else Phrase.CHANGE_SUBJECT
            )
            return

        messages = self._conversation.prompt(self._system_prompt, text)
        try:
            await self._stream_reply(messages, turn)
        finally:
            # Remember what was actually said, even when interrupted, so "why?" still works.
            self._conversation.record(text, " ".join(turn.spoken))

    async def _stream_reply(self, messages: Sequence[Message], turn: _Turn) -> None:
        settings = self._settings
        texts: asyncio.Queue[Chunk | None] = asyncio.Queue()
        clips: asyncio.Queue[tuple[str, Pcm] | None] = asyncio.Queue(maxsize=1)

        async def think() -> None:
            chunker = SentenceChunker(
                first_clause_min_words=settings.first_clause_min_words,
                max_words=settings.max_chunk_words,
            )
            async with aclosing(self._chat.stream(messages)) as deltas:
                async for delta in deltas:
                    turn.timer.mark("first_token")
                    for chunk in chunker.feed(delta):
                        texts.put_nowait(chunk)
            for chunk in chunker.flush():
                texts.put_nowait(chunk)
            texts.put_nowait(None)

        async def voice() -> None:
            while (chunk := await texts.get()) is not None:
                words = to_speakable(chunk.text)
                if not words:
                    continue
                audio = await self._tts.synthesize(words)
                turn.timer.mark("first_synth")
                pause = (
                    settings.sentence_pause_s if chunk.ends_sentence else settings.clause_pause_s
                )
                await clips.put((words, audio + Pcm.silence(pause, audio.sample_rate)))
            await clips.put(None)

        async def speak() -> None:
            item = await clips.get()
            if item is None:
                raise ServiceError("the language model returned nothing speakable")
            self._indicator.show(Status.SPEAKING)
            turn.timer.mark("first_audio")
            # Keep one clip queued behind the one playing so sentences join without gaps.
            playing: deque[asyncio.Future[None]] = deque()
            try:
                while item is not None:
                    words, audio = item
                    playing.append(self._speaker.enqueue(audio))
                    turn.spoken.append(words)
                    if len(playing) > 1:
                        await playing.popleft()
                    item = await clips.get()
                while playing:
                    await playing.popleft()
            finally:
                for future in playing:
                    future.cancel()

        await run_together(think(), voice(), speak())


def phrase_for(error: BaseException) -> Phrase:
    """What Lucy should say after a failed turn."""
    match error:
        case ServiceUnreachable():
            return Phrase.OFFLINE
        case RateLimited():
            return Phrase.TIRED
        case AuthenticationError() | AudioDeviceError() | ConfigError():
            return Phrase.NEEDS_HELP
        case _:
            return Phrase.TANGLED
