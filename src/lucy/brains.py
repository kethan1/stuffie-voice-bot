"""Assembling Lucy's brains: Groq in the cloud, and an optional local backup on the Pi.

The cloud brain is Groq: Whisper for speech, gpt-oss for replies, with the
120B model as a rate-limit fallback. The local brain is faster-whisper
``tiny.en`` in-process plus a small LLM (Gemma 3 1B) served by Ollama on
localhost. Ollama speaks the same OpenAI-compatible protocol as Groq, so both
brains use the same chat adapter and differ only in configuration.
"""

import logging
import time
from dataclasses import dataclass

from lucy.adapters.cloud import ChatModelSpec, FallbackChat, StreamingChat, WhisperTranscriber
from lucy.adapters.local_stt import FasterWhisperTranscriber
from lucy.adapters.openai_api import ApiClient
from lucy.config import Config, read_api_key
from lucy.core.pipeline import Brain
from lucy.core.ports import ChatModel

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Cloud:
    api: ApiClient
    stt: WhisperTranscriber
    chat: ChatModel

    def brain(self) -> Brain:
        return Brain("cloud", self.stt, self.chat, reachable=self.api.reachable)

    async def aclose(self) -> None:
        await self.api.aclose()


@dataclass(frozen=True, slots=True)
class Local:
    api: ApiClient
    stt: FasterWhisperTranscriber
    chat: StreamingChat
    history_exchanges: int

    def brain(self) -> Brain:
        return Brain("local", self.stt, self.chat, history_exchanges=self.history_exchanges)

    async def preload(self, system_prompt: str) -> None:
        """Load both models into RAM and let Ollama cache the system prompt.

        Reading Gemma 3 1B (~0.8 GB) from an SD card takes tens of seconds, and
        evaluating the system prompt on a Pi 4 takes several more. Doing both at
        start-up keeps that wait away from a child. Ollama keeps the model
        resident (``OLLAMA_KEEP_ALIVE=-1`` in its unit) and reuses the cached
        system-prompt prefix for every later request.
        """
        started = time.monotonic()
        await self.stt.preload()
        await self.api.post_json(
            "chat/completions",
            {
                "model": self.chat.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Hello!"},
                ],
                "max_tokens": 1,
                "stream": False,
            },
            timeout_s=300.0,
        )
        log.info("local brain ready in %.1fs", time.monotonic() - started)

    async def aclose(self) -> None:
        await self.api.aclose()
        self.stt.close()


def build_cloud(config: Config) -> Cloud:
    api_key = read_api_key(config.api)
    if not api_key:
        log.error(
            "no API key: install it as the systemd credential %r or set %s; "
            "Lucy will ask for a grown-up's help until then",
            config.api.api_key_credential,
            config.api.api_key_env,
        )
    api = ApiClient(
        config.api.base_url,
        api_key,
        connect_timeout_s=config.api.connect_timeout_s,
        keepalive_s=config.api.keepalive_s,
    )
    stt = WhisperTranscriber(
        api,
        model=config.stt.model,
        language=config.stt.language,
        prompt=config.stt.prompt,
        timeout_s=config.stt.timeout_s,
        no_speech_threshold=config.stt.no_speech_threshold,
    )
    llm = config.llm
    chat: ChatModel = StreamingChat(
        api, ChatModelSpec(llm.model, llm.params), timeout_s=llm.timeout_s
    )
    if llm.fallback_model:
        fallback = StreamingChat(
            api, ChatModelSpec(llm.fallback_model, llm.fallback_params), timeout_s=llm.timeout_s
        )
        chat = FallbackChat(chat, fallback)
    return Cloud(api, stt, chat)


def build_local(config: Config) -> Local | None:
    """The local brain, or None when it is not enabled."""
    local = config.local
    if not local.enabled:
        return None
    # Ollama ignores the key; a localhost connection either succeeds at once or not at all.
    api = ApiClient(local.base_url, "ollama", connect_timeout_s=1.0)
    stt = FasterWhisperTranscriber(
        local.stt_model,
        language=config.stt.language,
        threads=local.stt_threads,
        no_speech_threshold=config.stt.no_speech_threshold,
    )
    chat = StreamingChat(api, ChatModelSpec(local.model, local.params), timeout_s=local.timeout_s)
    return Local(api, stt, chat, local.history_exchanges)
