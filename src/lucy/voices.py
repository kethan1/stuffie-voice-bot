"""Assembling Lucy's voice: Speechify in the cloud, with Piper on the Pi standing in.

Whichever voice speaks, an interjected "Honk!" becomes a goose call
(:mod:`lucy.core.honk`).

Piper is always loaded. It speaks whenever the cloud voice cannot (no network,
no free characters left, a failing service), and it is Lucy's only voice when
no Speechify key is set up. The cloud voice streams at Piper's sample rate, so
one speaker stream plays either voice without resampling (Speechify streams
every rate a Piper voice uses).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from lucy.adapters.openai_api import ApiClient
from lucy.adapters.piper_tts import PiperSpeech
from lucy.adapters.speechify_tts import SpeechifySpeech
from lucy.config import Config, read_api_key
from lucy.core.honk import GooseCalls, HonkingVoice
from lucy.core.ports import Prewarmable, TextToSpeech
from lucy.core.speech import FallbackVoice

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Voice:
    tts: TextToSpeech
    """What Lucy speaks (and honks) with: the cloud voice backed by the on-device one, or the
    on-device one alone."""
    on_device: PiperSpeech
    name: str
    cloud: SpeechifySpeech | None = None
    api: ApiClient | None = None
    fallback: FallbackVoice | None = None

    @property
    def warmers(self) -> tuple[Prewarmable, ...]:
        """The cloud voice's connection, to pre-warm while the child talks."""
        return () if self.api is None else (self.api,)

    def on_recovery(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` whenever the cloud voice speaks again after failing."""
        if self.fallback is not None:
            self.fallback.on_recovery(callback)

    async def aclose(self) -> None:
        if self.api is not None:
            await self.api.aclose()
        self.on_device.close()


def load_on_device_voice(config: Config) -> PiperSpeech:
    """Load Piper (blocking: reads the model and builds the onnxruntime session)."""
    return PiperSpeech(
        config.tts.voice,
        threads=config.tts.threads,
        length_scale=config.tts.length_scale,
        volume=config.tts.volume,
        speaker_id=config.tts.speaker_id,
    )


def with_honks(config: Config, tts: TextToSpeech) -> TextToSpeech:
    """``tts``, calling each interjected "Honk!" as a goose would (unless ``tts.honks`` is off)."""
    if not config.tts.honks:
        return tts
    return HonkingVoice(tts, GooseCalls(tts.sample_rate, volume=config.tts.honk_volume))


def connect_cloud_voice(config: Config) -> ApiClient | None:
    """A client for the cloud voice service, or None when it is turned off or has no key."""
    settings = config.tts.cloud
    if not settings.enabled:
        return None
    api_key = read_api_key(settings)
    if not api_key:
        log.warning(
            "no Speechify API key (systemd credential %r or %s), so Lucy speaks with her "
            "on-device voice; see README.md for her natural voice",
            settings.api_key_credential,
            settings.api_key_env,
        )
        return None
    return ApiClient(
        settings.base_url,
        api_key,
        connect_timeout_s=config.api.connect_timeout_s,
        keepalive_s=config.api.keepalive_s,
        warm_path="audio/models",
    )


def build_voice(config: Config, on_device: PiperSpeech) -> Voice:
    settings = config.tts.cloud
    on_device_name = config.tts.voice.stem
    api = connect_cloud_voice(config)
    if api is None:
        return Voice(with_honks(config, on_device), on_device, on_device_name)
    cloud = SpeechifySpeech(
        api,
        voice_id=settings.voice_id,
        model=settings.model,
        sample_rate=on_device.sample_rate,
        volume=settings.volume,
        timeout_s=settings.timeout_s,
    )
    fallback = FallbackVoice(cloud, on_device, available=api.reachable, rest_s=settings.rest_s)
    name = f"Speechify {settings.voice_id}, backed by {on_device_name}"
    return Voice(with_honks(config, fallback), on_device, name, cloud, api, fallback)
