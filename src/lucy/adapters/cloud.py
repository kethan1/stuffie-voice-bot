"""Speech-to-text and chat over an OpenAI-compatible API (GroqCloud by default)."""

import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from lucy.adapters.openai_api import ApiClient
from lucy.core.audio import Pcm
from lucy.core.errors import AuthenticationError, ServiceError, ServiceUnreachable
from lucy.core.ports import ChatModel
from lucy.core.types import Message

log = logging.getLogger(__name__)


class WhisperTranscriber:
    """``POST /audio/transcriptions`` with ``whisper-large-v3-turbo``.

    * Audio is sent as 16 kHz mono WAV: Whisper's native format, so the server
      does no resampling, and there is no encoder cost on the Pi.
    * ``language`` is pinned, which skips language detection (faster, and more
      accurate on short, high-pitched children's utterances).
    * ``verbose_json`` exposes each segment's ``no_speech_prob``, so segments
      that Whisper itself thinks are not speech (its classic "Thank you." on
      background noise) are dropped instead of being answered.
    """

    def __init__(
        self,
        api: ApiClient,
        *,
        model: str,
        language: str,
        prompt: str = "",
        timeout_s: float = 8.0,
        no_speech_threshold: float = 0.6,
    ) -> None:
        self._api = api
        self._form = {
            "model": model,
            "language": language,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        if prompt:
            self._form["prompt"] = prompt
        self._timeout_s = timeout_s
        self._no_speech_threshold = no_speech_threshold

    async def transcribe(self, speech: Pcm) -> str:
        body = await self._api.post_form(
            "audio/transcriptions",
            data=self._form,
            files={"file": ("speech.wav", speech.to_wav(), "audio/wav")},
            timeout_s=self._timeout_s,
        )
        segments = body.get("segments")
        if not isinstance(segments, list) or not segments:
            return str(body.get("text", "")).strip()
        kept = [
            str(segment.get("text", ""))
            for segment in segments
            if isinstance(segment, dict)
            and float(segment.get("no_speech_prob", 0.0)) < self._no_speech_threshold
        ]
        if len(kept) < len(segments):
            log.debug("dropped %d non-speech segment(s)", len(segments) - len(kept))
        return " ".join(part.strip() for part in kept).strip()


@dataclass(frozen=True, slots=True)
class ChatModelSpec:
    model: str
    params: Mapping[str, Any] = field(default_factory=dict)
    """Model-specific request fields, e.g. ``reasoning_effort`` for gpt-oss."""


class StreamingChat:
    """``POST /chat/completions`` with ``stream: true``, yielding content deltas."""

    def __init__(self, api: ApiClient, spec: ChatModelSpec, *, timeout_s: float = 10.0) -> None:
        self._api = api
        self._spec = spec
        self._timeout_s = timeout_s

    @property
    def model(self) -> str:
        return self._spec.model

    async def stream(self, messages: Sequence[Message]) -> AsyncGenerator[str, None]:
        payload = {
            **self._spec.params,
            "model": self._spec.model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": True,
        }
        async with self._api.stream_events(
            "chat/completions", payload, timeout_s=self._timeout_s
        ) as events:
            async for event in events:
                for choice in event.get("choices") or ():
                    delta = choice.get("delta") or {}
                    # Reasoning (if a model streams it) arrives in delta.reasoning; never speak it.
                    content = delta.get("content")
                    if content:
                        yield content


class FallbackChat:
    """Use a second model when the first is rate-limited or failing.

    Groq's free-tier limits are per model, so falling back from gpt-oss-20b to
    gpt-oss-120b effectively doubles the daily quota. Fallback only happens
    before the first word is produced; a reply is never stitched from two
    models. Being offline or having a bad API key is not something another
    model can fix, so those errors are raised straight away.
    """

    def __init__(self, primary: ChatModel, fallback: ChatModel) -> None:
        self._primary = primary
        self._fallback = fallback

    async def stream(self, messages: Sequence[Message]) -> AsyncGenerator[str, None]:
        produced = False
        try:
            async with aclosing(self._primary.stream(messages)) as deltas:
                async for delta in deltas:
                    produced = True
                    yield delta
        except ServiceError as error:
            if produced or isinstance(error, ServiceUnreachable | AuthenticationError):
                raise
            log.warning("primary model failed (%s); using the fallback model", error)
        else:
            return
        async with aclosing(self._fallback.stream(messages)) as deltas:
            async for delta in deltas:
                yield delta
