"""Composition root: builds the adapters from configuration and runs Lucy.

This is the only module that knows which concrete classes implement each
port. Everything below it receives its collaborators through its constructor.
"""

import asyncio
import contextlib
import dataclasses
import logging
import os
import signal
from collections.abc import Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Protocol, runtime_checkable

from lucy import __version__
from lucy.adapters.console import KeyboardButton, LogIndicator
from lucy.adapters.logbook import Logbook
from lucy.adapters.power import CpuGovernor, NoPowerControl, PowerKnob, PowerManager, WifiPowerSave
from lucy.adapters.sound import SoundDeviceMicrophone, SoundDeviceSpeaker
from lucy.brains import Local, build_cloud, build_local
from lucy.config import Config, device, read_system_prompt
from lucy.core.controller import Controller, Earcons
from lucy.core.conversation import Conversation
from lucy.core.errors import ConfigError, LucyError
from lucy.core.notes import NoteTaker
from lucy.core.phrases import Phrase, PhraseBook
from lucy.core.pipeline import TurnPipeline
from lucy.core.ports import Button, Indicator, Moderator, PowerControl
from lucy.core.safety import AllowAll, NoteScreen, ReplyFilter, SafetyScreen
from lucy.core.tasks import BackgroundTasks
from lucy.core.types import PowerMode, Status
from lucy.core.volume import Volume
from lucy.motors import start_motion
from lucy.persona import LUCY
from lucy.platform.systemd import SystemdNotifier
from lucy.voices import Voice, build_voice, load_on_device_voice

log = logging.getLogger(__name__)

EXIT_CONFIG = 78  # EX_CONFIG: systemd is told not to restart on this (see lucy.service)


async def open_voice(config: Config, stack: AsyncExitStack) -> Voice:
    """Load and warm up the on-device voice, then put the cloud voice (if set up) in front."""
    on_device = await asyncio.to_thread(load_on_device_voice, config)
    stack.callback(on_device.close)
    await asyncio.to_thread(on_device.warm_up)
    voice = build_voice(config, on_device)
    if voice.api is not None:
        stack.push_async_callback(voice.api.aclose)
    return voice


def open_logbook(config: Config, stack: AsyncExitStack) -> Logbook | None:
    """Lucy's logbook, if she keeps one. Nothing is read or written until a turn ends."""
    if not config.logbook.enabled:
        return None
    logbook = Logbook(config.logbook.database, keep_days=config.logbook.keep_days)
    # Closed after the controller (the stack unwinds backwards), so the last turn's row is
    # on the card before Lucy stops.
    stack.callback(logbook.close)
    return logbook


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


class Indicators:
    """Shows each status on several indicators at once (the LED, and Lucy's motors)."""

    def __init__(self, *indicators: Indicator) -> None:
        self._indicators = indicators

    def show(self, status: Status) -> None:
        for indicator in self._indicators:
            indicator.show(status)


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
        puppeteer = await start_motion(config, stack)  # her prompt says whether she can move
        system_prompt = read_system_prompt(config, LUCY.prompt(moving=puppeteer is not None))
        voice = await open_voice(config, stack)
        tts = voice.tts

        cloud = build_cloud(config)
        stack.push_async_callback(cloud.aclose)
        local = build_local(config)
        if local is not None:
            stack.push_async_callback(local.aclose)
        speaker = SoundDeviceSpeaker(
            sample_rate=tts.sample_rate,
            device=device(config.audio.output_device),
            latency_s=config.audio.output_latency_s,
            blocksize=config.audio.output_blocksize,
            listener=puppeteer,
        )
        mic = SoundDeviceMicrophone(
            sample_rate=config.audio.input_sample_rate,
            device=device(config.audio.input_device),
            max_seconds=config.session.max_record_s + 1.0,
        )
        indicator = build_indicator(config, stack)
        if puppeteer is not None:
            indicator = Indicators(indicator, puppeteer)
        power = build_power(config)
        stack.push_async_callback(power.restore)

        moderator: Moderator = AllowAll()
        reply_screen = None
        if config.safety.screen:
            moderator = SafetyScreen(extra=config.safety.extra_patterns)
            reply_screen = ReplyFilter()
        conversation = Conversation(config.llm.history_exchanges, notes=cloud.notes is not None)
        notes = None
        if cloud.notes is not None:
            notes = NoteTaker(
                conversation,
                cloud.notes,
                max_words=config.memory.max_words,
                available=cloud.api.reachable,
                screen=NoteScreen(moderator, reply_screen) if reply_screen is not None else None,
                log_notes=config.turn.log_transcripts,
            )
            stack.push_async_callback(notes.aclose)
        handover = SetupHandover(stop)
        phrases = PhraseBook(LUCY.lines, tts)
        voice.on_recovery(phrases.forget)  # lines cached in the on-device voice: render again
        pipeline = TurnPipeline(
            brain=cloud.brain(),
            backup=local.brain() if local is not None else None,
            tts=tts,
            moderator=moderator,
            speaker=speaker,
            indicator=indicator,
            phrases=phrases,
            conversation=conversation,
            system_prompt=system_prompt,
            # With honks off she would say the word, so she leaves honking to the model.
            settings=config.turn
            if config.tts.honks
            else dataclasses.replace(config.turn, opening_honk_chance=0.0),
            reply_screen=reply_screen,
            volume=Volume(speaker, config.volume),
            setup=handover if config.setup.enabled and config.setup.voice_command else None,
            notes=notes,
            logbook=open_logbook(config, stack),
        )
        controller = Controller(
            pipeline=pipeline,
            mic=mic,
            speaker=speaker,
            indicator=indicator,
            power=power,
            conversation=conversation,
            warmers=[cloud.api, *voice.warmers],
            earcons=Earcons.generate(tts.sample_rate, volume=config.session.earcon_volume),
            settings=config.session,
        )
        stack.push_async_callback(controller.close)

        background = BackgroundTasks()
        stack.push_async_callback(background.cancel_all)
        background.spawn(notifier.watchdog(), "watchdog")
        background.spawn(_warm_up(phrases, local, config, system_prompt), "warm-up")
        button = build_button(config, keyboard)

        notifier.ready(f"Lucy {__version__} is awake")
        log.info(
            "Lucy %s is awake (voice: %s; %s)",
            __version__,
            voice.name,
            "moving" if puppeteer is not None else "keeping still",
        )
        if await _hand_over_to_setup(config, button, pipeline, handover, notifier):
            return 0
        exit_code = await _serve(controller, button, pipeline, stop)
        notifier.stopping()
        log.info("Lucy is going to sleep")
    return exit_code


async def _serve(
    controller: Controller, button: Button, pipeline: TurnPipeline, stop: asyncio.Event
) -> int:
    """Answer the wing button until Lucy is told to stop. Returns the exit code."""
    serving = asyncio.create_task(controller.run(button), name="controller")
    stopping = asyncio.create_task(stop.wait(), name="stop")
    await asyncio.wait({serving, stopping}, return_when=asyncio.FIRST_COMPLETED)
    stopping.cancel()
    if serving.done():
        return await _explain_failure(serving, pipeline)
    serving.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await serving
    return 0


@runtime_checkable
class HoldableButton(Protocol):
    """A button that can tell whether it is being held right now (the wing button on GPIO)."""

    async def held(self, seconds: float) -> bool: ...


SETUP_REQUEST = "setup-request"
"""Written into Lucy's runtime directory (/run/lucy) to ask for setup mode: lucy-setup.path
starts lucy-setup.service, which stops Lucy while it runs and wakes her afterwards."""
SETUP_HANDOVER_S = 30.0


def _setup_request(environ: Mapping[str, str]) -> Path | None:
    """The file that asks for setup mode. None if Lucy is not running as lucy.service."""
    runtime = environ.get("RUNTIME_DIRECTORY")  # set by RuntimeDirectory= in lucy.service
    return Path(runtime.split(":")[0]) / SETUP_REQUEST if runtime else None


def request_setup(environ: Mapping[str, str] = os.environ) -> bool:
    """Ask systemd for setup mode. False if Lucy is not running as lucy.service."""
    request = _setup_request(environ)
    if request is None:
        log.warning("setup mode can only start when Lucy runs as lucy.service")
        return False
    try:
        request.touch()
    except OSError as error:
        log.warning("could not ask for setup mode: %s", error)
        return False
    return True


class SetupHandover:
    """Lucy's way into setup mode while she is running: the pipeline's
    :class:`~lucy.core.ports.SetupStarter`.

    She asks systemd for it, and systemd stops her to make way: ``stop`` is set when that
    happens. Used when her wing is held as she wakes up, and when a grown-up asks with the
    knock (:mod:`lucy.core.knock`).
    """

    def __init__(
        self,
        stop: asyncio.Event,
        environ: Mapping[str, str] = os.environ,
        *,
        wait_s: float = SETUP_HANDOVER_S,
    ) -> None:
        self._stop = stop
        self._environ = environ
        self._wait_s = wait_s

    @property
    def available(self) -> bool:
        return _setup_request(self._environ) is not None

    async def start(self) -> bool:
        """Ask for setup mode and wait to be stopped for it. False if nothing came of it."""
        if not request_setup(self._environ):
            return False
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), self._wait_s)  # systemd stops Lucy
        if self._stop.is_set():
            return True
        log.warning("setup mode did not start (is lucy-setup.path installed and running?)")
        request = _setup_request(self._environ)
        if request is not None:
            # Taken back, so that it cannot start setup mode out of the blue later on.
            with contextlib.suppress(OSError):
                request.unlink(missing_ok=True)
        return False


async def _hand_over_to_setup(
    config: Config,
    button: Button,
    pipeline: TurnPipeline,
    handover: SetupHandover,
    notifier: SystemdNotifier,
) -> bool:
    """If her wing is held while she wakes up, hand over to setup mode. True once stopped."""
    hold_s = config.setup.hold_to_start_s
    if not config.setup.enabled or hold_s <= 0 or not isinstance(button, HoldableButton):
        return False
    try:
        if not await button.held(hold_s):
            return False
    except ConfigError as error:  # the button line is busy: Lucy reports it when she listens
        log.warning("could not check the wing for the setup gesture: %s", error)
        return False
    log.info("wing held while waking up: asking for setup mode")
    with contextlib.suppress(LucyError):
        await pipeline.say(Phrase.SETUP)
    if await handover.start():
        notifier.stopping()
        log.info("Lucy is resting while setup mode runs")
        return True
    log.warning("carrying on as usual, without setup mode")
    return False


async def _warm_up(
    phrases: PhraseBook, local: Local | None, config: Config, system_prompt: str
) -> None:
    """Pre-render Lucy's fixed lines, then load the local brain: one after the other, gently."""
    await phrases.preload()
    if local is not None and config.local.preload:
        try:
            await local.preload(system_prompt)
        except LucyError as error:
            log.warning("local brain is not ready, so Lucy needs the cloud for now: %s", error)


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
