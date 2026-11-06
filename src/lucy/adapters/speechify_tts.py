"""Lucy's natural voice: Speechify's Simba model, streamed as raw PCM.

Why this voice service:

* **Natural.** ``simba-3.2`` ranked 11th of 103 voices in Artificial
  Analysis's blind text-to-speech arena (November 2026), far ahead of
  anything a Raspberry Pi 4 can synthesise in real time.
* **Built for streaming.** Speechify's first audio leaves its servers about
  60 ms after a request arrives. Lucy can therefore send whole sentences
  (splitting inside a sentence is what makes speech sound stitched together)
  and still start talking about as soon as Piper could render a clause.
* **A generous free plan.** 500,000 characters a month with no credit card.
  Lucy's replies are about 150 characters, so that covers roughly 3,000
  replies. When it runs out, Speechify refuses requests rather than billing
  for them, and Lucy carries on with her on-device voice
  (:class:`lucy.core.speech.FallbackVoice`).

Latency and power choices:

* Raw PCM at the speaker's own sample rate (``output_format=pcm_22050``), so
  the Pi neither decodes nor resamples anything. Audio is passed on as it
  arrives, in parts of about 100 ms.
* The same lean HTTP client as Groq: one kept-alive HTTP/2 connection,
  pre-warmed when the wing is squeezed, so no TLS handshake delays the first
  word.
* Synthesis runs in Speechify's data centre. The Pi's cores stay at low
  clocks while Lucy talks, instead of running Piper flat out for each
  sentence.
* The free plan allows bursts of 10 requests, then one a second. If a burst
  (pre-rendering Lucy's fixed lines at start-up) runs into that limit, a
  request waits the second or so Speechify asks for, once, rather than
  switching voices.

Only the text of Lucy's own replies is sent. The child's voice never is.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any
from urllib.parse import urlencode

from lucy.adapters.openai_api import ApiClient, JsonObject
from lucy.core.audio import SAMPLE_WIDTH, Pcm, amplify
from lucy.core.errors import ConfigError, RateLimited

log = logging.getLogger(__name__)

SAMPLE_RATES = (8_000, 16_000, 22_050, 24_000, 44_100, 48_000)
"""The rates Speechify streams raw PCM at (its ``pcm_*`` output formats)."""
_MAX_RATE_LIMIT_WAIT_S = 1.5


class SpeechifySpeech:
    def __init__(
        self,
        api: ApiClient,
        *,
        voice_id: str,
        model: str = "simba-3.2",
        sample_rate: int = 22_050,
        volume: float = 1.0,
        timeout_s: float = 2.0,
        part_s: float = 0.1,
    ) -> None:
        if sample_rate not in SAMPLE_RATES:
            raise ConfigError(
                f"the cloud voice cannot stream {sample_rate} Hz audio "
                f"(it offers {', '.join(map(str, SAMPLE_RATES))} Hz)"
            )
        self._api = api
        self._model = model
        self._request: dict[str, Any] = {
            "voice_id": voice_id,
            "model": model,
            "output_format": f"pcm_{sample_rate}",
        }
        self._sample_rate = sample_rate
        self._volume = volume
        self._timeout_s = timeout_s
        self._part_bytes = max(1, round(part_s * sample_rate)) * SAMPLE_WIDTH

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        payload = {**self._request, "input": text}
        if any(ch.isdigit() for ch in text):
            # Says "56" as "fifty-six". It costs a little latency, so only when it is needed.
            payload["options"] = {"text_normalization": True}
        try:
            async with aclosing(self._stream(payload)) as parts:
                async for part in parts:
                    yield part
            return
        except RateLimited as error:  # always raised before any audio
            wait_s = error.retry_after_s
            if wait_s is None or wait_s > _MAX_RATE_LIMIT_WAIT_S:
                raise
        log.debug("cloud voice rate-limited; retrying in %.1f s", wait_s)
        await asyncio.sleep(wait_s)
        async with aclosing(self._stream(payload)) as parts:
            async for part in parts:
                yield part

    async def _stream(self, payload: dict[str, Any]) -> AsyncGenerator[Pcm, None]:
        pending = bytearray()
        async with self._api.stream_bytes(
            "audio/stream", payload, timeout_s=self._timeout_s
        ) as body:
            async for data in body:
                pending += data
                if len(pending) >= self._part_bytes:
                    yield self._take(pending)
        if len(pending) >= SAMPLE_WIDTH:
            yield self._take(pending)

    def _take(self, pending: bytearray) -> Pcm:
        """Remove every whole sample from ``pending`` (a trailing odd byte waits for its pair)."""
        whole = len(pending) - len(pending) % SAMPLE_WIDTH
        audio = Pcm(bytes(pending[:whole]), self._sample_rate)
        del pending[:whole]
        return amplify(audio, self._volume)

    async def voices(self, locale: str = "en") -> list[JsonObject]:
        """The catalogue voices this model can speak with, best first (for ``lucy voices``)."""
        voices: list[JsonObject] = []
        query = {"type": "shared", "locale": locale, "model": self._model, "limit": "200"}
        while True:
            body = await self._api.get_json(f"voices?{urlencode(query)}")
            voices.extend(v for v in body.get("voices") or () if isinstance(v, dict))
            if not body.get("has_more") or not body.get("next_cursor"):
                break
            query["cursor"] = str(body["next_cursor"])
        return sorted(voices, key=lambda v: (v.get("staff_pick_rank") or 999, str(v.get("id"))))
