"""Lucy's voice: a natural cloud voice, with the on-device voice ready to stand in.

The cloud voice sounds far more natural than anything a Raspberry Pi 4 can
synthesise in real time, but it needs the network, an API key, and free
characters left this month. :class:`FallbackVoice` speaks with it whenever it
can and with the on-device voice whenever it cannot, one sentence at a time:

* A sentence moves to the on-device voice only before any of it has been
  heard. Once the cloud voice has started a sentence, a failure ends that
  sentence instead of repeating it in a different voice.
* After a failure the cloud voice rests for a while (longer if the service
  asks), so later sentences do not each wait for the same timeout. During a
  known network outage it is skipped at once, until the connection is back.
* A content refusal is never handed to the on-device voice. The cloud
  service's own screen refused to say those words, so Lucy must not say them
  in another voice either.

When the cloud voice works again after failing, :meth:`FallbackVoice.on_recovery`
callbacks run, so audio cached while it was away (Lucy's fixed lines) can be
rendered again in her usual voice.
"""

import logging
import math
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing

from lucy.core.audio import Pcm
from lucy.core.errors import ContentRefused, RateLimited, ServiceError, ServiceUnreachable
from lucy.core.ports import TextToSpeech

log = logging.getLogger(__name__)


async def render(tts: TextToSpeech, text: str) -> Pcm:
    """Synthesise ``text`` into one clip."""
    async with aclosing(tts.stream(text)) as parts:
        audio = b"".join([part.data async for part in parts])
    return Pcm(audio, tts.sample_rate)


def _always() -> bool:
    return True


class FallbackVoice:
    def __init__(
        self,
        primary: TextToSpeech,
        backup: TextToSpeech,
        *,
        available: Callable[[], bool] = _always,
        rest_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if primary.sample_rate != backup.sample_rate:
            raise ValueError(
                f"both voices must share one sample rate, not {primary.sample_rate} Hz "
                f"and {backup.sample_rate} Hz"
            )
        self._primary = primary
        self._backup = backup
        self._available = available  # False while the cloud voice is known to be unreachable
        self._rest_s = rest_s
        self._clock = clock
        self._resting_until = -math.inf
        self._failing = False
        self._recovery_callbacks: list[Callable[[], None]] = []

    @property
    def sample_rate(self) -> int:
        return self._backup.sample_rate

    def on_recovery(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` each time the cloud voice speaks again after failing."""
        self._recovery_callbacks.append(callback)

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        if self._available() and self._clock() >= self._resting_until:
            produced = False
            try:
                async with aclosing(self._primary.stream(text)) as parts:
                    async for part in parts:
                        produced = True
                        yield part
            except ContentRefused:
                raise
            except ServiceError as error:
                if produced:
                    raise
                self._rest(error)
            else:
                if self._failing:
                    self._failing = False
                    log.info("the cloud voice is back")
                    for callback in self._recovery_callbacks:
                        callback()
                return
        async with aclosing(self._backup.stream(text)) as parts:
            async for part in parts:
                yield part

    def _rest(self, error: ServiceError) -> None:
        if isinstance(error, ServiceUnreachable):
            rest_s = 0.0  # ``available`` keeps it skipped until the connection is back
        elif isinstance(error, RateLimited) and error.retry_after_s:
            rest_s = max(self._rest_s, error.retry_after_s)
        else:
            rest_s = self._rest_s
        self._resting_until = self._clock() + rest_s
        if self._failing:
            log.debug("cloud voice still failing: %s: %s", type(error).__name__, error)
            return
        self._failing = True
        log.warning(
            "cloud voice failed (%s: %s); using the on-device voice", type(error).__name__, error
        )
