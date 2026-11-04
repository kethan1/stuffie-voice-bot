"""Command-line entry point: ``lucy run | say | bench | doctor``."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from lucy import __version__
from lucy.config import Config, default_config_path, load_config
from lucy.core.errors import ConfigError, LucyError
from lucy.platform.logs import configure_logging

log = logging.getLogger("lucy")


def main(argv: list[str] | None = None) -> int:
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

    bench = commands.add_parser("bench", help="measure end-to-end latency without a microphone")
    bench.add_argument("--question", default="Why do geese fly in a V shape?")
    bench.add_argument("--rounds", type=int, default=4)
    bench.add_argument(
        "--brain", choices=("cloud", "local"), default="cloud", help="which brain to measure"
    )

    doctor = commands.add_parser("doctor", help="check hardware, voice, network, and permissions")
    doctor.add_argument("--listen", action="store_true", help="also record 3 s to check mic level")

    args = parser.parse_args(argv)
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
                return asyncio.run(_say(config, args.text))
            case "bench":
                from lucy.tools.bench import bench as run_bench

                return asyncio.run(
                    run_bench(config, args.question, max(1, args.rounds), brain_name=args.brain)
                )
            case "doctor":
                from lucy.tools.doctor import doctor as run_doctor

                return asyncio.run(run_doctor(config, config_path, listen=args.listen))
    except ConfigError as error:
        log.error("%s", error)
        return 78
    except LucyError as error:  # e.g. bench while offline: one clear line, not a traceback
        log.error("%s: %s", type(error).__name__, error)
        return 1
    except KeyboardInterrupt:
        return 130
    return 2


async def _say(config: Config, text: str) -> int:
    from lucy.adapters.sound import SoundDeviceSpeaker
    from lucy.app import load_voice
    from lucy.config import device

    tts = await asyncio.to_thread(load_voice, config)
    speaker = SoundDeviceSpeaker(
        sample_rate=tts.sample_rate,
        device=device(config.audio.output_device),
        latency_s=config.audio.output_latency_s,
    )
    try:
        await speaker.enqueue(await tts.synthesize(text))
        await asyncio.sleep(config.audio.output_latency_s + 0.1)  # let the last block drain
    finally:
        await speaker.close()
        tts.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
