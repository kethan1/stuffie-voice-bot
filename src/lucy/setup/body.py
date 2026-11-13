"""Lucy's voice, beak, and body in setup mode.

Setup mode has no internet (the Wi-Fi is Lucy's own hotspot), so she speaks with her
on-device voice, through her goose voice and with her honks, and her beak moves as she talks.
The page's "Hear her" and "Test motors" buttons use the numbers on the page before they are
saved, so a grown-up can try a loudness, a goose voice, or a beak power first.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Protocol

from lucy.adapters.motors import L293d
from lucy.adapters.piper_tts import PiperSpeech
from lucy.adapters.sound import SoundDeviceSpeaker
from lucy.config import Config, device
from lucy.core.audio import Pcm, amplify
from lucy.core.errors import ConfigError, LucyError
from lucy.core.goose_voice import GooseVoice, pitch_ratio
from lucy.core.honk import GooseCalls, HonkingVoice
from lucy.core.motion import MotionSettings, Puppeteer
from lucy.core.ports import TextToSpeech
from lucy.core.speech import render
from lucy.motors import exercise, open_motors
from lucy.voices import load_on_device_voice

log = logging.getLogger(__name__)

SAMPLE = "Honk! Hi, I'm Lucy. Is this loud enough? Honk honk!"


class Body(Protocol):
    async def say(self, text: str) -> None: ...

    async def try_voice(
        self, *, volume: float, goose_voice: float, honks: bool, honk_volume: float
    ) -> None:
        """Say a sample sentence as she would with these (unsaved) settings."""
        ...

    async def try_motors(self, motion: MotionSettings) -> None:
        """Move the beak and body with these (unsaved) settings. Raises ConfigError if there
        are no working motors."""
        ...


class _Louder:
    """A voice at another loudness (honks are added afterwards, at their own loudness)."""

    def __init__(self, tts: TextToSpeech, gain: float) -> None:
        self._tts = tts
        self._gain = gain

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        async for part in self._tts.stream(text):
            yield amplify(part, self._gain)


class LucysBody:
    def __init__(self, config: Config, stack: AsyncExitStack) -> None:
        self._config = config
        self._stack = stack
        self._piper: PiperSpeech | None = None
        self._speaker: SoundDeviceSpeaker | None = None
        self._motors: L293d | None = None
        self._lock = asyncio.Lock()  # one sound or movement at a time

    async def open(self) -> None:
        """Load the voice (a few seconds on a Pi) and claim the motors, if she has them."""
        config = self._config
        # At full loudness: each line is scaled to the loudness wanted (see _speak).
        self._piper = piper = await asyncio.to_thread(load_on_device_voice, config, volume=1.0)
        self._stack.callback(piper.close)
        puppeteer = None
        if config.motors.enabled:
            try:
                await self._open_motors()
            except ConfigError as error:
                log.warning("Lucy will keep still in setup mode: %s", error)
        if self._motors is not None:
            puppeteer = Puppeteer(self._motors, config.motors.motion)
            self._stack.callback(puppeteer.close)
        self._speaker = SoundDeviceSpeaker(
            sample_rate=piper.sample_rate,
            device=device(config.audio.output_device),
            latency_s=config.audio.output_latency_s,
            blocksize=config.audio.output_blocksize,
            listener=puppeteer,
        )
        self._stack.push_async_callback(self._speaker.close)

    async def say(self, text: str) -> None:
        tts = self._config.tts
        await self._speak(
            text,
            volume=tts.volume,
            goose_voice=tts.goose_voice,
            honks=tts.honks,
            honk_volume=tts.honk_volume,
        )

    async def try_voice(
        self, *, volume: float, goose_voice: float, honks: bool, honk_volume: float
    ) -> None:
        await self._speak(
            SAMPLE, volume=volume, goose_voice=goose_voice, honks=honks, honk_volume=honk_volume
        )

    async def try_motors(self, motion: MotionSettings) -> None:
        async with self._lock:
            if self._motors is None:
                await self._open_motors()  # turned on just now, on the page
            assert self._motors is not None
            await exercise(self._motors, motion, repeats=2, report=log.info)

    async def _open_motors(self) -> None:
        try:
            self._motors = await asyncio.to_thread(open_motors, self._config)
        except (OSError, ImportError, ConfigError) as error:
            raise ConfigError(f"Lucy's motors are not working: {error}") from error
        self._stack.callback(self._motors.close)

    async def _speak(
        self, text: str, *, volume: float, goose_voice: float, honks: bool, honk_volume: float
    ) -> None:
        if self._piper is None or self._speaker is None:
            raise LucyError("Lucy's voice is not loaded")
        # At the pace this goose voice needs, which may not be the saved one's.
        voice: TextToSpeech = _Louder(self._piper.slowed(pitch_ratio(goose_voice)), volume)
        if goose_voice > 0:
            voice = GooseVoice(voice, goose_voice)
        if honks:
            voice = HonkingVoice(voice, GooseCalls(voice.sample_rate, volume=honk_volume))
        async with self._lock:
            clip = await render(voice, text)
            await self._speaker.enqueue(clip)
            await asyncio.sleep(self._config.audio.output_latency_s + 0.2)  # let it drain


class SilentBody:
    """For when Lucy's voice cannot load: setup mode still works, without her talking."""

    async def say(self, text: str) -> None:
        log.info("(Lucy would say) %s", text)

    async def try_voice(
        self, *, volume: float, goose_voice: float, honks: bool, honk_volume: float
    ) -> None:
        raise ConfigError("Lucy's voice is not working in setup mode (see the journal)")

    async def try_motors(self, motion: MotionSettings) -> None:
        raise ConfigError("Lucy's motors are not available in setup mode")
