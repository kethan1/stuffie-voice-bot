"""Composition root: builds the adapters from configuration and runs Lucy.

This is the only module that knows which concrete classes implement each
port. Everything below it receives its collaborators through its constructor.
"""

import asyncio
import contextlib
import logging
import signal
from contextlib import AsyncExitStack
from dataclasses import dataclass

from lucy import __version__
from lucy.adapters.cloud import ChatModelSpec, FallbackChat, StreamingChat, WhisperTranscriber
from lucy.adapters.console import KeyboardButton, LogIndicator
from lucy.adapters.openai_api import ApiClient
from lucy.adapters.piper_tts import PiperSpeech
from lucy.adapters.power import CpuGovernor, NoPowerControl, PowerKnob, PowerManager, WifiPowerSave
from lucy.adapters.sound import SoundDeviceMicrophone, SoundDeviceSpeaker
from lucy.config import Config, device, read_api_key, read_system_prompt
from lucy.core.controller import Controller, Earcons
from lucy.core.conversation import Conversation
from lucy.core.errors import ConfigError
from lucy.core.phrases import Phrase, PhraseBook
from lucy.core.pipeline import TurnPipeline
from lucy.core.ports import Button, ChatModel, Indicator, PowerControl
from lucy.core.safety import AllowAll, DistressScreen
from lucy.core.tasks import BackgroundTasks
from lucy.core.types import PowerMode
from lucy.persona import LUCY
from lucy.platform.systemd import SystemdNotifier

log = logging.getLogger(__name__)

EXIT_CONFIG = 78  # EX_CONFIG: systemd is told not to restart on this (see lucy.service)


@dataclass(frozen=True, slots=True)
class Cloud:
    api: ApiClient
    stt: WhisperTranscriber
    chat: ChatModel


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


def load_voice(config: Config) -> PiperSpeech:
    return PiperSpeech(
        config.tts.voice,
        threads=config.tts.threads,
        length_scale=config.tts.length_scale,
        volume=config.tts.volume,
        speaker_id=config.tts.speaker_id,
    )


def build_power(config: Config) -> PowerControl:
    settings = config.power
    knobs: list[PowerKnob] = []
    if settings.manage_cpu_governor:
        governors = {
            PowerMode.ACTIVE: settings.active_governor,
            PowerMode.STANDBY: settings.standby_governor,
            PowerMode.DORMANT: settings.dormant_governor,
        }
        try:
            knobs.append(CpuGovernor(governors))
        except (OSError, RuntimeError) as error:
            log.warning("CPU governor control unavailable: %s", error)
    if settings.manage_wifi:
        try:
            knobs.append(WifiPowerSave(settings.wifi_interface))
        except RuntimeError as error:
            log.warning("Wi-Fi power-save control unavailable: %s", error)
    return PowerManager(knobs) if knobs else NoPowerControl()


def build_button(config: Config, keyboard: bool) -> Button:
    if keyboard or config.button.backend == "keyboard":
        return KeyboardButton()
    from lucy.adapters.gpio import GpioButton

    b = config.button
    return GpioButton(b.chip, b.line, active_low=b.active_low, debounce_ms=b.debounce_ms)


def build_indicator(config: Config, stack: AsyncExitStack) -> Indicator:
    if not config.led.enabled:
        return LogIndicator()
    from lucy.adapters.gpio import GpioOutput, LedIndicator

    try:
        led = LedIndicator(
            GpioOutput(config.led.chip, config.led.line, active_low=config.led.active_low)
        )
    except (ConfigError, ImportError) as error:
        log.warning("status LED disabled: %s", error)
        return LogIndicator()
    stack.callback(led.close)
    return led


async def run(config: Config, *, keyboard: bool = False) -> int:
    """Run Lucy until SIGTERM/SIGINT. Returns the process exit code."""
    notifier = SystemdNotifier()
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)

    async with AsyncExitStack() as stack:
        notifier.status("loading voice")
        system_prompt = read_system_prompt(config, LUCY.system_prompt)
        tts = await asyncio.to_thread(load_voice, config)
        stack.callback(tts.close)
        await asyncio.to_thread(tts.warm_up)

        cloud = build_cloud(config)
        stack.push_async_callback(cloud.api.aclose)
        speaker = SoundDeviceSpeaker(
            sample_rate=tts.sample_rate,
            device=device(config.audio.output_device),
            latency_s=config.audio.output_latency_s,
            blocksize=config.audio.output_blocksize,
        )
        mic = SoundDeviceMicrophone(
            sample_rate=config.audio.input_sample_rate,
            device=device(config.audio.input_device),
            max_seconds=config.session.max_record_s + 1.0,
        )
        indicator = build_indicator(config, stack)
        power = build_power(config)
        stack.push_async_callback(power.restore)

        conversation = Conversation(config.llm.history_exchanges)
        phrases = PhraseBook(LUCY.lines, tts)
        pipeline = TurnPipeline(
            stt=cloud.stt,
            chat=cloud.chat,
            tts=tts,
            moderator=DistressScreen(extra=config.safety.extra_patterns)
            if config.safety.screen
            else AllowAll(),
            speaker=speaker,
            indicator=indicator,
            phrases=phrases,
            conversation=conversation,
            system_prompt=system_prompt,
            settings=config.turn,
        )
        controller = Controller(
            pipeline=pipeline,
            mic=mic,
            speaker=speaker,
            indicator=indicator,
            power=power,
            conversation=conversation,
            warmers=[cloud.api],
            earcons=Earcons.generate(tts.sample_rate),
            settings=config.session,
        )
        stack.push_async_callback(controller.close)

        background = BackgroundTasks()
        stack.push_async_callback(background.cancel_all)
        background.spawn(notifier.watchdog(), "watchdog")
        background.spawn(phrases.preload(), "preload-phrases")
        button = build_button(config, keyboard)

        notifier.ready(f"Lucy {__version__} is awake")
        log.info("Lucy %s is awake (voice %s)", __version__, config.tts.voice.name)
        serving = asyncio.create_task(controller.run(button), name="controller")
        stopping = asyncio.create_task(stop.wait(), name="stop")
        await asyncio.wait({serving, stopping}, return_when=asyncio.FIRST_COMPLETED)
        stopping.cancel()
        exit_code = 0
        if serving.done():
            exit_code = await _explain_failure(serving, pipeline)
        else:
            serving.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await serving
        notifier.stopping()
        log.info("Lucy is going to sleep")
    return exit_code


async def _explain_failure(serving: asyncio.Task[None], pipeline: TurnPipeline) -> int:
    """The controller stopped by itself: say so out loud, then report the exit status."""
    error = serving.exception()
    if error is None:
        return 0
    log.error("Lucy stopped: %s", error, exc_info=error)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(pipeline.say(Phrase.NEEDS_HELP), timeout=10.0)
        await asyncio.sleep(0.3)  # let the sound card drain before the speaker closes
    return EXIT_CONFIG if isinstance(error, ConfigError) else 1
