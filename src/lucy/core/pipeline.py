"""One conversational turn, streamed end to end.

::

    recording ─► trim ─► STT ─► screen ─► LLM stream ─► chunker ─► TTS ─► speaker
                                              think()        voice()     speak()

The three stages on the right run concurrently and are joined by queues, so
Lucy starts talking as soon as the first part of the first sentence has been
synthesised. The cloud voice streams each sentence, so its first audio plays
while the rest of the sentence is still arriving, the next sentence is
synthesised while this one plays, and the LLM is still writing. The queue to
the speaker is bounded, which keeps memory flat and limits wasted work when a
child interrupts.

Speech recognition and the reply come from a :class:`Brain`: normally Groq in
the cloud. When the cloud cannot be reached and a backup brain is configured
(models running on the Pi), the turn moves to it, at the start if the cloud is
already known to be down, or the moment a request fails to connect. Lucy says
she is "thinking with her own goose brain" so the slower local answer is
expected rather than a mystery.

When a safety screen answers instead of the model, the conversation remembers
the exchange as :data:`SCREENED` and the fixed line Lucy said, never the child's
words. A follow-up such as "yes!" (to "Want a weird animal fact instead?") or
"but I'm scared to tell" then reaches a model that knows what Lucy just said.
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass, field

from lucy.core.audio import Pcm, trim_silence
from lucy.core.chunker import Chunk, SentenceChunker
from lucy.core.conversation import Conversation
from lucy.core.errors import (
    AudioDeviceError,
    AuthenticationError,
    ConfigError,
    ContentRefused,
    LucyError,
    RateLimited,
    ServiceError,
    ServiceUnreachable,
)
from lucy.core.metrics import TurnTimer
from lucy.core.phrases import Phrase, PhraseBook
from lucy.core.ports import (
    ChatModel,
    Indicator,
    Moderator,
    ReplyScreen,
    Speaker,
    SpeechToText,
    TextToSpeech,
)
from lucy.core.tasks import detach, run_together
from lucy.core.textnorm import to_speakable
from lucy.core.types import Concern, Status

log = logging.getLogger(__name__)

_AHEAD_S = 0.3
"""Keep at least this much audio queued behind the clip that is playing, so streamed parts
and sentences join without gaps even when the network hiccups."""

SCREENED = "(Lucy's safety screen stopped this part of the conversation.)"
"""What the conversation remembers a child saying when a safety screen answered for Lucy."""


@dataclass(frozen=True, slots=True)
class TurnSettings:
    min_speech_s: float = 0.3
    """Less sound than this (after trimming silence) means nothing was said."""
    speech_threshold_dbfs: float = -50.0
    """Frames quieter than this count as silence."""
    sentence_pause_s: float = 0.2
    clause_pause_s: float = 0.08
    first_clause_min_words: int = 0
    """0 sends whole sentences only, which is what sounds natural. Above 0, the first chunk may
    end at a comma once it has this many words: a sooner first word from the on-device voice,
    at the cost of a seam in the first sentence."""
    max_chunk_words: int = 30
    log_transcripts: bool = False
    """Privacy: what a child said and what the model replied (which can echo it) are kept out
    of the logs unless this is enabled."""


def _always() -> bool:
    return True


@dataclass(frozen=True, slots=True)
class Brain:
    """Where Lucy's understanding and answers come from: the cloud, or models on the Pi."""

    name: str
    stt: SpeechToText
    chat: ChatModel
    history_exchanges: int | None = None
    """Send at most this many past exchanges (a small local model is slow to read long prompts)."""
    reachable: Callable[[], bool] = _always
    """False while this brain is known to be unreachable, so turns skip it without waiting."""


@dataclass(slots=True)
class _Turn:
    timer: TurnTimer
    brain: Brain
    heard: str = ""
    """What the child said, once the input screen has let it through to the model."""
    reply: list[str] = field(default_factory=list)
    """The model's reply as it streamed in."""
    spoken: list[str] = field(default_factory=list)

    def received(self, delta: str) -> None:
        """The next piece of the model's reply arrived."""
        self.timer.mark("first_token")
        self.reply.append(delta)


class TurnPipeline:
    def __init__(
        self,
        *,
        brain: Brain,
        backup: Brain | None = None,
        tts: TextToSpeech,
        moderator: Moderator,
        speaker: Speaker,
        indicator: Indicator,
        phrases: PhraseBook,
        conversation: Conversation,
        system_prompt: str,
        settings: TurnSettings,
        reply_screen: ReplyScreen | None = None,
    ) -> None:
        self._brain = brain
        self._backup = backup
        self._tts = tts
        self._moderator = moderator
        self._reply_screen = reply_screen
        self._speaker = speaker
        self._indicator = indicator
        self._phrases = phrases
        self._conversation = conversation
        self._system_prompt = system_prompt
        self._settings = settings

    async def respond(self, recording: Pcm, timer: TurnTimer) -> None:
        """Answer one recording. Never raises except for cancellation (barge-in)."""
        turn = _Turn(timer, self._brain)
        try:
            await self._respond(recording, turn)
        except asyncio.CancelledError:
            log.info("turn interrupted (%s brain): %s", turn.brain.name, timer.report())
            self._remember(turn)  # what was said before the interruption, so "why?" works
            raise
        except Exception as error:
            if isinstance(error, LucyError):
                log.warning(
                    "turn failed (%s brain): %s: %s (%s)",
                    turn.brain.name,
                    type(error).__name__,
                    error,
                    timer.report(),
                )
            else:
                log.exception("turn failed unexpectedly (%s)", timer.report())
            if isinstance(error, ContentRefused):
                # Not even part of a refused sentence should be heard: cut it off, move on.
                self._speaker.stop()
                self._log_reply(turn)
                await self._redirect(phrase_for(error))
                return
            if not turn.spoken:
                await self.say(phrase_for(error), status=Status.ERROR)
            self._remember(turn)
        else:
            log.info("turn complete (%s brain): %s", turn.brain.name, timer.report())
            self._remember(turn)

    async def say(self, phrase: Phrase, *, status: Status = Status.SPEAKING) -> None:
        """Speak one of Lucy's fixed lines. Never raises except for cancellation."""
        await self._say(phrase, status)

    async def _say(self, phrase: Phrase, status: Status = Status.SPEAKING) -> str | None:
        """Speak one of Lucy's fixed lines, and return it (None if it could not be said)."""
        try:
            text, audio = await self._phrases.line(phrase)
            self._indicator.show(status)
            await self._speaker.enqueue(audio)
        except Exception:
            # Nothing left to say it with: the LED is the only way left to signal trouble.
            log.exception("could not speak phrase %s", phrase)
            self._indicator.show(Status.ERROR)
            return None
        return text

    async def _redirect(self, phrase: Phrase) -> None:
        """Say a safety line in place of an answer, and remember it, but not what led to it."""
        said = await self._say(phrase)
        if said:
            self._conversation.record(SCREENED, said)

    def _remember(self, turn: _Turn) -> None:
        """Remember what was actually said, even if interrupted, so "why?" still works."""
        self._conversation.record(turn.heard, " ".join(turn.spoken))
        self._log_reply(turn)

    def _log_reply(self, turn: _Turn) -> None:
        if self._settings.log_transcripts and turn.reply:
            # The model's own words, as far as they got, including any the reply screen
            # stopped: not always what Lucy said out loud.
            log.info("%s brain replied: %r", turn.brain.name, "".join(turn.reply))

    async def _respond(self, recording: Pcm, turn: _Turn) -> None:
        settings = self._settings
        speech = trim_silence(recording, settings.speech_threshold_dbfs)
        if speech.duration < settings.min_speech_s:
            log.info("heard no speech (%.2fs recorded)", recording.duration)
            await self.say(Phrase.DIDNT_HEAR)
            return

        if not self._brain.reachable():
            await self._fall_back(turn, "still unreachable")
        text = (await self._transcribe(speech, turn)).strip()
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
            await self._redirect(screened_phrase(assessment.concern))
            return

        turn.heard = text
        await self._stream_reply(text, turn)

    async def _transcribe(self, speech: Pcm, turn: _Turn) -> str:
        try:
            return await turn.brain.stt.transcribe(speech)
        except ServiceUnreachable:
            if not await self._fall_back(turn, "unreachable"):
                raise
        return await turn.brain.stt.transcribe(speech)

    async def _reply(self, text: str, turn: _Turn) -> AsyncGenerator[str, None]:
        """Stream the reply, moving to the backup brain if the cloud cannot be reached.

        The switch only happens before the first word: a reply is never stitched
        together from two brains.
        """
        produced = False
        try:
            async with aclosing(self._ask(turn.brain, text)) as deltas:
                async for delta in deltas:
                    produced = True
                    yield delta
        except ServiceUnreachable:
            if produced or not await self._fall_back(turn, "unreachable"):
                raise
        else:
            return
        async with aclosing(self._ask(turn.brain, text)) as deltas:
            async for delta in deltas:
                yield delta

    def _ask(self, brain: Brain, text: str) -> AsyncGenerator[str, None]:
        messages = self._conversation.prompt(
            self._system_prompt, text, max_exchanges=brain.history_exchanges
        )
        return brain.chat.stream(messages)

    async def _fall_back(self, turn: _Turn, why: str) -> bool:
        """Move this turn to the backup brain. False if there is none, or it is already in use."""
        if self._backup is None or turn.brain is self._backup:
            return False
        log.warning("%s brain %s; using the %s brain", turn.brain.name, why, self._backup.name)
        turn.brain = self._backup
        turn.timer.mark("fallback")
        try:
            # Queued, not awaited: the local brain starts work while Lucy explains the wait.
            detach(self._speaker.enqueue(await self._phrases.render(Phrase.OWN_BRAIN)))
        except Exception:
            log.warning("could not say that Lucy is thinking locally", exc_info=True)
        return True

    def _speakable(self, text: str) -> str:
        """The words of one chunk of the reply as Lucy will say them ("" if there are none).

        Raises :class:`ContentRefused` if the reply screen stops them, which is handled like
        the cloud voice refusing a sentence: Lucy stops and says one of her fixed lines.
        """
        words = to_speakable(text)
        if words and self._reply_screen is not None:
            # The chunk as the model wrote it: cleaning it for speech would hide "f***" or "sh!t".
            assessment = self._reply_screen.screen(text)
            if not assessment.allowed:
                log.info(
                    "reply screened: %s (%s)",
                    assessment.concern.value,
                    ", ".join(assessment.reasons),
                )
                raise ContentRefused("Lucy's reply screen stopped a sentence", assessment.concern)
        return words

    async def _stream_reply(self, text: str, turn: _Turn) -> None:
        settings = self._settings
        texts: asyncio.Queue[Chunk | None] = asyncio.Queue()
        # (the chunk's words, on its first part only; audio)
        clips: asyncio.Queue[tuple[str | None, Pcm] | None] = asyncio.Queue(maxsize=1)

        async def think() -> None:
            chunker = SentenceChunker(
                first_clause_min_words=settings.first_clause_min_words,
                max_words=settings.max_chunk_words,
            )
            async with aclosing(self._reply(text, turn)) as deltas:
                async for delta in deltas:
                    turn.received(delta)
                    for chunk in chunker.feed(delta):
                        texts.put_nowait(chunk)
            for chunk in chunker.flush():
                texts.put_nowait(chunk)
            texts.put_nowait(None)

        async def voice() -> None:
            while (chunk := await texts.get()) is not None:
                words = self._speakable(chunk.text)
                if not words:
                    continue
                label: str | None = words
                async with aclosing(self._tts.stream(words)) as parts:
                    async for audio in parts:
                        turn.timer.mark("first_synth")
                        await clips.put((label, audio))
                        label = None
                pause = (
                    settings.sentence_pause_s if chunk.ends_sentence else settings.clause_pause_s
                )
                if pause > 0:
                    await clips.put((None, Pcm.silence(pause, self._tts.sample_rate)))
            await clips.put(None)

        async def speak() -> None:
            item = await clips.get()
            if item is None:
                raise ServiceError("the language model returned nothing speakable")
            self._indicator.show(Status.SPEAKING)
            turn.timer.mark("first_audio")
            playing: deque[tuple[asyncio.Future[None], float]] = deque()
            queued_s = 0.0  # audio queued behind the oldest clip, which is (about to be) playing
            try:
                while item is not None:
                    words, audio = item
                    if playing:
                        queued_s += audio.duration
                    playing.append((self._speaker.enqueue(audio), audio.duration))
                    if words:
                        turn.spoken.append(words)
                    while len(playing) > 1 and queued_s >= _AHEAD_S:
                        await playing.popleft()[0]
                        queued_s -= playing[0][1]
                    item = await clips.get()
                while playing:
                    await playing.popleft()[0]
            finally:
                for future, _ in playing:
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
        case ContentRefused():
            return screened_phrase(error.concern)
        case _:
            return Phrase.TANGLED


_SCREENED: dict[Concern, Phrase] = {
    Concern.NEEDS_GROWN_UP: Phrase.TELL_A_GROWN_UP,
    Concern.UNSAFE: Phrase.NOT_SAFE,
    Concern.PRIVATE: Phrase.KEEP_PRIVATE,
    Concern.UNKIND: Phrase.BE_KIND,
    Concern.TRICK: Phrase.NICE_TRY,
}


def screened_phrase(concern: Concern) -> Phrase:
    """What Lucy says instead of answering, for a message or reply a safety screen stopped."""
    return _SCREENED.get(concern, Phrase.CHANGE_SUBJECT)
