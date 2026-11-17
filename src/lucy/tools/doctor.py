"""``lucy doctor``: check everything Lucy needs, and say what to fix.

Run it after installing, after rewiring, or whenever Lucy is quiet. It checks
the configuration, API keys, both voices, sound devices, GPIO lines, the
motors' PWM channels, her logbook, network/API access, and power controls. With
``--listen`` it also records three seconds so microphone levels can be checked
against ``turn.speech_threshold_dbfs``.
"""

import asyncio
import inspect
import os
import shutil
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from lucy.adapters import sound_cards
from lucy.adapters.logbook import Logbook
from lucy.adapters.power import CPUFREQ_ROOT
from lucy.adapters.sound import import_sounddevice
from lucy.adapters.speechify_tts import SpeechifySpeech
from lucy.brains import build_cloud, build_local
from lucy.config import Config, device, read_api_key, read_system_prompt
from lucy.core.audio import Pcm, level_dbfs, trim_silence
from lucy.core.errors import LucyError
from lucy.core.speech import render
from lucy.persona import LUCY
from lucy.voices import connect_cloud_voice, load_on_device_voice

OK, WARN, FAIL = "✔", "!", "✘"

Probe = Callable[[Config], Awaitable[str] | str]


async def doctor(config: Config, config_path: Path | None, *, listen: bool) -> int:
    print(f"Lucy doctor (config: {config_path or 'built-in defaults'})")
    probes: list[tuple[str, Probe]] = [("API key", api_key)]
    if config.tts.cloud.enabled:
        probes.append(("cloud voice key", cloud_voice_key))
    probes += [
        ("on-device voice", voice),
        ("speaker", speaker),
        ("microphone", microphone),
    ]
    if listen:
        probes.append(("microphone level", microphone_level))
    if config.button.backend == "gpio":
        probes.append(("button GPIO", button_line))
    if config.led.enabled:
        probes.append(("LED GPIO", led_line))
    if config.motors.enabled:
        probes.append(("motors", motors))
    if config.logbook.enabled:
        probes.append(("logbook", logbook))
    probes.append(("AI service", service))
    if config.tts.cloud.enabled:
        probes.append(("cloud voice", cloud_voice))
    if config.local.enabled:
        probes.append(("local speech model", local_speech))
        probes.append(("local brain (Ollama)", local_brain))
    if config.power.manage_cpu_governor:
        probes.append(("CPU governor", cpu_governor))
    if config.power.manage_wifi:
        probes.append(("Wi-Fi power save", wifi_power_save))

    failures = 0
    for name, probe in probes:
        mark, detail = await _run(probe, config)
        failures += mark == FAIL
        print(f" {mark} {name}" + (f": {detail}" if detail else ""))
    print("\nall good, honk!" if not failures else f"\n{failures} problem(s) found")
    return 1 if failures else 0


async def _run(probe: Probe, config: Config) -> tuple[str, str]:
    try:
        result = probe(config)
        return OK, await result if inspect.isawaitable(result) else result
    except Warning as warning:
        return WARN, str(warning)
    except Exception as error:  # every failure is reported; none aborts the run
        return FAIL, f"{type(error).__name__}: {error}"


async def logbook(config: Config) -> str:
    settings = config.logbook
    book = Logbook(settings.database, keep_days=settings.keep_days)
    try:
        found = await asyncio.to_thread(book.check)  # LogbookError if it cannot be written
    finally:
        book.close()
    kept = f"kept {settings.keep_days} days" if settings.keep_days else "kept until deleted"
    return (
        f"{settings.database}: {found.conversations} conversation(s), {found.turns} turn(s), "
        f"{kept}; read them in setup mode"
    )


def api_key(config: Config) -> str:
    key = read_api_key(config.api)
    if not key:
        raise LucyError(
            f"missing: set {config.api.api_key_env} (or put it in .env) or install "
            f"/etc/lucy/credentials/{config.api.api_key_credential} (see README)"
        )
    return f"present ({len(key)} characters)"


def cloud_voice_key(config: Config) -> str:
    cloud = config.tts.cloud
    key = read_api_key(cloud)
    if not key:
        raise Warning(
            f"missing, so Lucy uses her on-device voice: set {cloud.api_key_env} (or put it in "
            f".env) or install /etc/lucy/credentials/{cloud.api_key_credential} (see README)"
        )
    return f"present ({len(key)} characters)"


async def voice(config: Config) -> str:
    tts = await asyncio.to_thread(load_on_device_voice, config)
    try:
        await asyncio.to_thread(tts.warm_up)
        started = time.perf_counter()
        audio = await render(tts, "Honk! Did you know that geese can fly very far?")
        rtf = (time.perf_counter() - started) / audio.duration
    finally:
        tts.close()
    detail = f"{config.tts.voice.name}, {tts.sample_rate} Hz, real-time factor {rtf:.2f}"
    if rtf > 0.5:
        raise Warning(f"{detail} (slow: try a '-low' voice or more tts.threads)")
    return detail


async def cloud_voice(config: Config) -> str:
    api = connect_cloud_voice(config)
    if api is None:
        raise Warning("not set up, so the on-device voice speaks instead")
    settings = config.tts.cloud
    speech = SpeechifySpeech(api, voice_id=settings.voice_id, model=settings.model)
    try:
        await api.get_json("audio/models")  # open the connection, as Lucy does while you talk
        started = time.perf_counter()
        first_ms = None
        seconds = 0.0
        async for part in speech.stream("Honk! Did you know that geese can fly very far?"):
            if first_ms is None:
                first_ms = (time.perf_counter() - started) * 1000
            seconds += part.duration
    finally:
        await api.aclose()
    if first_ms is None:
        raise LucyError(f"{settings.voice_id} returned no audio")
    detail = f"{settings.voice_id} ({settings.model}): first audio in {first_ms:.0f} ms"
    if first_ms > settings.timeout_s * 1000 / 2:  # halfway to giving the sentence to Piper
        raise Warning(f"{detail} (slow: check the Wi-Fi signal)")
    return f"{detail}, {seconds:.1f}s of speech"


def speaker(config: Config) -> str:
    out = device(config.audio.output_device)
    route = None if out is not None else sound_cards.route("output", config.audio.speaker_card)
    if route is not None and (busy := _in_use(route)):
        return busy
    try:
        sd = import_sounddevice()
        info = sd.query_devices(out, "output")
        sd.check_output_settings(device=out, samplerate=22_050, channels=1, dtype="int16")
    except Exception as error:
        raise LucyError(f"{error} ({route})" if route is not None else str(error)) from error
    return _routed(str(info["name"]), route, "speaker_card")


def microphone(config: Config) -> str:
    mic = device(config.audio.input_device)
    route = None if mic is not None else sound_cards.route("input", config.audio.microphone_card)
    if route is not None and (busy := _in_use(route)):
        return busy
    rate = config.audio.input_sample_rate
    try:
        sd = import_sounddevice()
        info = sd.query_devices(mic, "input")
        sd.check_input_settings(device=mic, samplerate=rate, channels=1, dtype="int16")
    except Exception as error:
        raise LucyError(f"{error} ({route})" if route is not None else str(error)) from error
    return _routed(f"{info['name']} at {rate} Hz", route, "microphone_card")


def _in_use(route: sound_cards.Route) -> str | None:
    """Say so if another process (Lucy's service, most likely) holds the card: then it can be
    neither opened nor checked from here, though it works."""
    pid = sound_cards.user(route)
    if pid is None:
        return None
    try:
        name = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        name = "a process"
    return f"{route}, in use by {name} (pid {pid})"


def _routed(detail: str, route: sound_cards.Route | None, setting: str) -> str:
    if route is None or route.card is None:
        return detail
    detail = f"{detail} → {route}"
    if route.stand_in:  # it works, but the configuration names a card that has gone
        raise Warning(f'{detail}; to keep it, set audio.{setting} = "{route.card.id}"')
    return detail


async def microphone_level(config: Config) -> str:
    from lucy.adapters.sound import SoundDeviceMicrophone

    mic = SoundDeviceMicrophone(
        sample_rate=config.audio.input_sample_rate,
        device=device(config.audio.input_device),
        card=config.audio.microphone_card,
    )
    print("   … speak normally for 3 seconds")
    await mic.start()
    await asyncio.sleep(3.0)
    audio = await mic.stop()
    threshold = config.turn.speech_threshold_dbfs
    speech = trim_silence(audio, threshold)
    detail = (
        f"level {level_dbfs(audio):.0f} dBFS, "
        f"{speech.duration:.1f}s above the {threshold:.0f} dBFS speech threshold"
    )
    if speech.duration < 1.0:
        raise Warning(f"{detail} (too quiet: raise the capture gain with `alsamixer`)")
    return detail


def button_line(config: Config) -> str:
    return _gpio_line(config.button.chip, config.button.line, "button")


def led_line(config: Config) -> str:
    return _gpio_line(config.led.chip, config.led.line, "LED")


def _gpio_line(chip: str, line: int, label: str) -> str:
    import gpiod

    if not os.path.exists(chip):
        raise LucyError(f"{chip} does not exist")
    if not os.access(chip, os.R_OK | os.W_OK):
        raise LucyError(f"no permission on {chip} (add the user to the 'gpio' group)")
    with gpiod.Chip(chip) as handle:
        info = handle.get_line_info(line)
    if info.used and not info.consumer.startswith("lucy-"):
        raise LucyError(f"line {line} is in use by {info.consumer!r}")
    held = f" (held by the running service as {info.consumer})" if info.used else ""
    return f"{label} on {chip} line {line}{held}"


def motors(config: Config) -> str:
    """The PWM channels and GPIO lines are there and free. Nothing moves (`lucy motors` does)."""
    settings = config.motors
    chip = settings.pwm_chip
    if not (chip / "npwm").exists():
        raise LucyError(
            f"{chip} is missing: add the PWM overlay (sudo deploy/install.sh --motors) and reboot"
        )
    channels = int((chip / "npwm").read_text())
    for name, channel in (("beak", settings.beak_pwm_channel), ("body", settings.body_pwm_channel)):
        if channel >= channels:
            raise LucyError(
                f"{chip} has {channels} channel(s), so there is no {name} channel {channel}"
            )
    for name, line in settings.lines().items():
        _gpio_line(settings.gpio_chip, line, name.removesuffix("_line").replace("_", " "))
    return (
        f"L293D: beak on PWM {settings.beak_pwm_channel} + GPIO{settings.beak_forward_line}"
        f"/GPIO{settings.beak_backward_line}, body on PWM {settings.body_pwm_channel} + "
        f"GPIO{settings.body_forward_line}/GPIO{settings.body_backward_line} "
        "(`lucy motors` moves them)"
    )


async def service(config: Config) -> str:
    cloud = build_cloud(config)
    try:
        started = time.perf_counter()
        body = await cloud.api.get_json("models")
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        await cloud.api.aclose()
    available = {model.get("id") for model in body.get("data", []) if isinstance(model, dict)}
    wanted = [config.stt.model, config.llm.model, config.llm.fallback_model]
    if config.memory.notes:
        wanted.append(config.memory.model)
    missing = [model for model in wanted if model and model not in available]
    if missing:
        raise LucyError(f"model(s) not offered by the API: {', '.join(missing)}")
    return f"{config.api.base_url} answered in {elapsed_ms:.0f} ms; all models available"


def cpu_governor(config: Config) -> str:
    files = sorted(CPUFREQ_ROOT.glob("policy*/scaling_governor"))
    if not files:
        raise Warning("no cpufreq support on this machine")
    current = files[0].read_text().strip()
    if not all(os.access(path, os.W_OK) for path in files):
        raise Warning(f"'{current}', not writable (lucy.service grants this when it starts)")
    return f"currently '{current}', writable"


async def wifi_power_save(config: Config) -> str:
    iw = shutil.which("iw") or shutil.which("iw", path="/usr/sbin:/sbin")
    if iw is None:
        raise Warning("`iw` is not installed (sudo apt install iw)")
    interface = config.power.wifi_interface
    process = await asyncio.create_subprocess_exec(
        iw,
        "dev",
        interface,
        "get",
        "power_save",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await process.communicate()
    if process.returncode != 0:
        raise Warning(err.decode().strip() or "cannot query power save")
    return f"{interface}: {out.decode().strip()}"


async def local_speech(config: Config) -> str:
    local = build_local(config)
    assert local is not None
    try:
        started = time.perf_counter()
        await local.stt.preload()
        loaded_s = time.perf_counter() - started
        started = time.perf_counter()
        await local.stt.transcribe(Pcm.silence(2.0, 16_000))
        run_s = time.perf_counter() - started
    finally:
        await local.aclose()
    return f"{config.local.stt_model.name} loaded in {loaded_s:.1f}s, 2 s of audio in {run_s:.1f}s"


async def local_brain(config: Config) -> str:
    local = build_local(config)
    assert local is not None
    try:
        body = await local.api.get_json("models")
        available = {str(m.get("id")) for m in body.get("data", []) if isinstance(m, dict)}
        if config.local.model not in available:
            raise LucyError(
                f"Ollama has no {config.local.model!r} (installed: {', '.join(sorted(available))})"
                f"; run: ollama pull {config.local.model}"
            )
        started = time.perf_counter()
        await local.preload(read_system_prompt(config, LUCY.system_prompt))
        ready_s = time.perf_counter() - started
    finally:
        await local.aclose()
    return f"{config.local.model} loaded and answered in {ready_s:.1f}s"
