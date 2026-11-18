"""Command-line entry point: ``lucy run | say | voices | motors | bench | doctor | setup``."""

import argparse
import asyncio
import logging
import sys
from contextlib import AsyncExitStack, aclosing
from pathlib import Path

from lucy import __version__
from lucy.config import Config, default_config_path, load_config
from lucy.core.errors import ConfigError, LucyError
from lucy.platform.logs import configure_logging

log = logging.getLogger("lucy")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lucy", description="Lucy the Goose voice companion")
    parser.add_argument("--version", action="version", version=f"lucy {__version__}")
    parser.add_argument(
        "--config", type=Path, help="TOML config (default: $LUCY_CONFIG or /etc/lucy/config.toml)"
    )
    parser.add_argument("--log-level", help="override log_level from the config")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run Lucy (what the systemd service does)")
    run.add_argument("--keyboard", action="store_true", help="use Enter instead of the GPIO button")

    say = commands.add_parser("say", help="speak a sentence through the speaker")
    say.add_argument("text")
    say.add_argument(
        "--on-device", action="store_true", help="use the on-device voice, not the cloud voice"
    )

    commands.add_parser("voices", help="list the cloud voices Lucy can speak with")

    motors = commands.add_parser("motors", help="move the beak and body to check the wiring")
    motors.add_argument(
        "--off", action="store_true", help="only stop both motors (run after the service stops)"
    )

    bench = commands.add_parser("bench", help="measure end-to-end latency without a microphone")
    bench.add_argument("--question", default="Why do geese fly in a V shape?")
    bench.add_argument("--rounds", type=int, default=4)
    bench.add_argument(
        "--brain", choices=("cloud", "local"), default="cloud", help="which brain to measure"
    )

    doctor = commands.add_parser("doctor", help="check hardware, voice, network, and permissions")
    doctor.add_argument("--listen", action="store_true", help="also record 3 s to check mic level")

    setup = commands.add_parser(
        "setup", help="setup mode: Lucy's own Wi-Fi and settings page (what lucy-setup runs)"
    )
    setup.add_argument(
        "--if-offline",
        action="store_true",
        help="start setup mode only if no network connects soon (at boot)",
    )
    setup.add_argument(
        "--demo",
        action="store_true",
        help="try the settings page on this computer with pretend Wi-Fi (http://127.0.0.1:8080)",
    )
    card = commands.add_parser(
        "setup-card", help="print the card for Lucy's setup Wi-Fi, making it first if needed"
    )
    card.add_argument("--output", type=Path, help="also save a printable copy (HTML) here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = args.config or default_config_path()
    try:
        config = load_config(config_path)
    except ConfigError as error:
        configure_logging("info")
        log.error("%s", error)
        return 78  # EX_CONFIG
    configure_logging(args.log_level or config.log_level)

    try:
        match args.command:
            case "run":
                from lucy.app import run as run_lucy

                return asyncio.run(run_lucy(config, keyboard=args.keyboard))
            case "say":
                return asyncio.run(_say(config, args.text, on_device=args.on_device))
            case "voices":
                return asyncio.run(_voices(config))
            case "motors":
                return asyncio.run(_motors(config, off=args.off))
            case "bench":
                from lucy.tools.bench import bench as run_bench

                return asyncio.run(
                    run_bench(config, args.question, max(1, args.rounds), brain_name=args.brain)
                )
            case "doctor":
                from lucy.tools.doctor import doctor as run_doctor

                return asyncio.run(run_doctor(config, config_path, listen=args.listen))
            case "setup":
                return asyncio.run(_setup(config, config_path, args.if_offline, args.demo))
            case "setup-card":
                from lucy.setup.mode import print_card

                return print_card(config, args.output)
    except ConfigError as error:
        log.error("%s", error)
        return 78
    except LucyError as error:  # e.g. bench while offline: one clear line, not a traceback
        log.error("%s: %s", type(error).__name__, error)
        return 1
    except KeyboardInterrupt:
        return 130
    return 2


async def _say(config: Config, text: str, *, on_device: bool) -> int:
    from lucy.adapters.sound import SoundDeviceSpeaker
    from lucy.config import device
    from lucy.motors import start_motion
    from lucy.voices import as_goose, build_voice, load_on_device_voice

    async with AsyncExitStack() as stack:
        puppeteer = await start_motion(config, stack)  # her beak and body move as she speaks
        voice = build_voice(config, await asyncio.to_thread(load_on_device_voice, config))
        stack.push_async_callback(voice.aclose)
        tts = as_goose(config, voice.on_device) if on_device else voice.tts
        speaker = SoundDeviceSpeaker(
            sample_rate=tts.sample_rate,
            device=device(config.audio.output_device),
            card=config.audio.speaker_card,
            latency_s=config.audio.output_latency_s,
            volume=config.volume,
            listener=puppeteer,
        )
        stack.push_async_callback(speaker.close)
        print(f"voice: {config.tts.voice.stem if on_device else voice.name}")
        played = None
        async with aclosing(tts.stream(text)) as parts:  # streamed, as Lucy speaks
            async for part in parts:
                played = speaker.enqueue(part)
        if played is not None:
            await played
        await asyncio.sleep(config.audio.output_latency_s + 0.2)  # let the last block drain
    return 0


async def _setup(config: Config, config_path: Path | None, if_offline: bool, demo: bool) -> int:
    from lucy.setup.mode import run_setup, start_if_offline
    from lucy.setup.network import NetworkManager

    if if_offline:
        return await start_if_offline(config, NetworkManager(config.setup.interface))
    if config_path is None:
        raise ConfigError("setup mode needs a config file to save to (--config)")
    return await run_setup(config, config_path, demo=demo)


async def _motors(config: Config, *, off: bool) -> int:
    from lucy.motors import exercise, open_motors

    if not config.motors.enabled:
        print("Lucy has no motors in this config ([motors] enabled = false)")
        return 0
    try:
        driver = await asyncio.to_thread(open_motors, config)  # both motors still, outputs off
    except OSError as error:
        raise ConfigError(f"cannot open the motors: {error}") from error
    try:
        if not off:
            await exercise(driver, config.motors.motion)
    finally:
        driver.close()
    return 0


async def _voices(config: Config) -> int:
    from lucy.adapters.speechify_tts import SpeechifySpeech
    from lucy.voices import connect_cloud_voice

    api = connect_cloud_voice(config)
    if api is None:
        log.error("the cloud voice is not set up: it needs a Speechify API key (see README.md)")
        return 1
    current = config.tts.cloud.voice_id
    try:
        voices = await SpeechifySpeech(api, voice_id=current, model=config.tts.cloud.model).voices()
    finally:
        await api.aclose()
    for entry in voices:
        mark = "*" if entry.get("id") == current else " "
        # What a voice sounds like ("style:playful"), not what it is marketed for ("use-case:…").
        traits = [
            tag.partition(":")[2]
            for tag in entry.get("tags") or ()
            if tag.partition(":")[0] in ("age", "style", "timbre", "pitch", "accent")
        ]
        print(
            f"{mark} {entry.get('id', '?'):<16} {entry.get('gender', ''):<7} "
            f"{entry.get('locale', ''):<6} {', '.join(traits)}"
        )
        if preview := entry.get("preview_audio"):
            print(f"    listen: {preview}")
    print(f"\n* = tts.cloud.voice_id ({current}). Set it in the config file to change voice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
