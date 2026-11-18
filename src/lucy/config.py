"""Configuration: one TOML file, strictly validated, with secrets kept out of it.

* Every setting has a sensible default, so an empty file is a valid file.
* Unknown keys are errors. A typo such as ``max_recrod_s`` fails loudly at
  start-up instead of being silently ignored.
* Relative paths are relative to the config file, so a config such as the
  repository's ``dev.toml`` works from any working directory.
* API keys never live in the TOML file. Each is read from a systemd
  credential (``groq_api_key``, ``speechify_api_key``; ``LoadCredential=``, so
  only the service can read them) or, for development, from ``GROQ_API_KEY``
  and ``SPEECHIFY_API_KEY`` in the environment or in a ``.env`` file in the
  working directory.
"""

import dataclasses
import os
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from lucy.core.controller import SessionSettings
from lucy.core.errors import ConfigError
from lucy.core.motion import MotionSettings
from lucy.core.pipeline import TurnSettings
from lucy.core.volume import VolumeSettings

DEFAULT_CONFIG_PATH = Path("/etc/lucy/config.toml")


def _gpt_oss_params(max_completion_tokens: int) -> dict[str, Any]:
    return {
        "reasoning_effort": "low",  # a few dozen reasoning tokens, not hundreds
        "include_reasoning": False,  # do not stream reasoning we would discard
        "max_completion_tokens": max_completion_tokens,  # includes reasoning tokens
        "temperature": 0.8,
    }


@dataclass(frozen=True, slots=True)
class ApiSettings:
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_credential: str = "groq_api_key"
    api_key_env: str = "GROQ_API_KEY"
    connect_timeout_s: float = 3.0
    keepalive_s: float = 120.0


@dataclass(frozen=True, slots=True)
class SttSettings:
    model: str = "whisper-large-v3-turbo"
    language: str = "en"
    prompt: str = ""
    timeout_s: float = 8.0
    no_speech_threshold: float = 0.6


@dataclass(frozen=True, slots=True)
class LlmSettings:
    model: str = "openai/gpt-oss-20b"
    params: dict[str, Any] = field(default_factory=lambda: _gpt_oss_params(512))
    fallback_model: str = "openai/gpt-oss-120b"
    """Used when the primary model is rate-limited or failing. Empty disables fallback."""
    fallback_params: dict[str, Any] = field(default_factory=lambda: _gpt_oss_params(512))
    timeout_s: float = 10.0
    history_exchanges: int = 6
    """Lucy's short-term memory: the most exchanges sent word for word (see ``memory``)."""


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """What Lucy remembers of a conversation beyond its last few exchanges (lucy.core.notes)."""

    notes: bool = True
    """When ``llm.history_exchanges`` are held, fold the older half into a few short notes
    instead of forgetting the oldest one each turn. Written in the background by ``model``."""
    max_words: int = 60
    """The notes are never longer than this, however long the conversation runs."""
    model: str = "qwen/qwen3.8-27b"
    """Writes the notes. Not ``llm.model``: each model has its own rate limit, so notes
    never use up the one that answers children, and this model keeps to what may be noted
    (no school names, no "rules" a child made up) where the gpt-oss models did not."""
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "reasoning_effort": "none",  # rewriting notes needs no thinking: quick and cheap
            "max_completion_tokens": 160,
            "temperature": 0.2,
        }
    )
    timeout_s: float = 10.0


@dataclass(frozen=True, slots=True)
class LogbookSettings:
    """Lucy's logbook: every conversation, kept for grown-ups to read (lucy.adapters.logbook)."""

    enabled: bool = True
    """Write down what children say and what Lucy answers. Off: nothing is written, and
    what was written before stays until it is deleted on the Conversations page."""
    database: Path = Path("/var/lib/lucy/logbook.db")
    """The SQLite file (lucy.service's ``StateDirectory=``)."""
    keep_days: int = 30
    """Rows older than this are deleted. 0 keeps them until a grown-up deletes them."""


@dataclass(frozen=True, slots=True)
class LocalSettings:
    """Lucy's own brain on the Pi, used when the cloud cannot be reached."""

    enabled: bool = False
    stt_model: Path = Path("/opt/lucy/models/faster-whisper-tiny.en")
    stt_threads: int = 4
    base_url: str = "http://127.0.0.1:11434/v1"
    """Ollama's OpenAI-compatible endpoint (localhost only)."""
    model: str = "gemma3:1b"
    params: dict[str, Any] = field(default_factory=lambda: {"max_tokens": 100, "temperature": 0.7})
    history_exchanges: int = 2
    """A 1B model on a Pi 4 reads prompts slowly, so it gets less history than the cloud
    (and Lucy's notes, which it reads once)."""
    timeout_s: float = 60.0
    preload: bool = True
    """Load both models at start-up, so the first offline answer does not wait for them."""


@dataclass(frozen=True, slots=True)
class SafetySettings:
    screen: bool = True
    """On-device screens of what a child says and what Lucy replies (see lucy.core.safety)."""
    extra_patterns: tuple[str, ...] = ()
    """Additional case-insensitive regular expressions that route to a trusted grown-up."""


@dataclass(frozen=True, slots=True)
class CloudVoiceSettings:
    """Lucy's natural voice, streamed from Speechify. The on-device voice stands in for it."""

    enabled: bool = True
    """Without an API key Lucy uses her on-device voice, so this only needs turning off to
    keep a configured key unused."""
    base_url: str = "https://api.speechify.ai/v1"
    api_key_credential: str = "speechify_api_key"
    api_key_env: str = "SPEECHIFY_API_KEY"
    model: str = "simba-3.2"
    voice_id: str = "harper_32"
    """See the choices with `lucy voices`."""
    volume: float = 1.0
    timeout_s: float = 2.0
    """The first audio must arrive this soon (and no later part may stall this long), or the
    on-device voice says the sentence instead."""
    rest_s: float = 60.0
    """After a failure, use the on-device voice this long before trying the cloud voice again."""


@dataclass(frozen=True, slots=True)
class TtsSettings:
    voice: Path = Path("/opt/lucy/voices/en_US-lessac-medium.onnx")
    """The on-device Piper voice: always ready, and the only voice without a cloud voice."""
    threads: int = 3
    length_scale: float = 1.0
    """Above 1.0 speaks more slowly, which can help the youngest children."""
    volume: float = 0.9
    speaker_id: int | None = None
    goose_voice: float = 0.7
    """How much like a goose her speech sounds, from 0 (her plain voice) to 1: a higher,
    smaller, honky voice with a rasp (lucy.core.goose_voice)."""
    honks: bool = True
    """Make a goose call for each interjected "Honk!" instead of saying the word."""
    honk_volume: float = 1.0
    """1.0 is about as loud as Lucy's voice."""
    cloud: CloudVoiceSettings = field(default_factory=CloudVoiceSettings)


@dataclass(frozen=True, slots=True)
class AudioSettings:
    input_device: str = ""
    """PortAudio device name (substring) or index; empty means the ALSA default."""
    output_device: str = ""
    speaker_card: str = ""
    """With no ``output_device``: the ALSA card id (in brackets in ``aplay -l``) of her speaker.
    When it is not plugged in, or this is empty, any USB speaker that is plugged in is used
    instead, so a speaker can be swapped for another without changing anything."""
    microphone_card: str = ""
    """The same for her microphone (``arecord -l``)."""
    input_sample_rate: int = 16_000
    output_latency_s: float = 0.08
    output_blocksize: int = 1024


@dataclass(frozen=True, slots=True)
class ButtonSettings:
    backend: str = "gpio"
    """``gpio`` for the wing button, ``keyboard`` for Enter-to-talk during bring-up."""
    chip: str = "/dev/gpiochip0"
    line: int = 17
    active_low: bool = True
    debounce_ms: int = 15


@dataclass(frozen=True, slots=True)
class LedSettings:
    enabled: bool = True
    chip: str = "/dev/gpiochip0"
    line: int = 27
    active_low: bool = False


@dataclass(frozen=True, slots=True)
class MotorSettings:
    """Lucy's beak and body motors on an L293D driver (docs/HARDWARE.md)."""

    enabled: bool = False
    pwm_chip: Path = Path("/sys/class/pwm/pwmchip0")
    """Hardware PWM from ``dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4``."""
    pwm_frequency_hz: int = 20_000
    """Above hearing, so the motors do not whine."""
    gpio_chip: str = "/dev/gpiochip0"
    beak_pwm_channel: int = 0
    """PWM channel 0 is GPIO12 (pin 32), wired to 1,2EN."""
    beak_forward_line: int = 22
    """GPIO22 (pin 15), wired to 1A: high runs the beak motor forward (opening the beak)."""
    beak_backward_line: int = 23
    """GPIO23 (pin 16), wired to 2A: high runs the beak motor backward."""
    body_pwm_channel: int = 1
    """PWM channel 1 is GPIO13 (pin 33), wired to 3,4EN."""
    body_forward_line: int = 24
    """GPIO24 (pin 18), wired to 3A: high runs the body motor forward."""
    body_backward_line: int = 25
    """GPIO25 (pin 22), wired to 4A: high runs the body motor backward."""
    motion: MotionSettings = field(default_factory=MotionSettings)

    def lines(self) -> dict[str, int]:
        """The four input lines, by setting name (without ``motors.``)."""
        return {
            "beak_forward_line": self.beak_forward_line,
            "beak_backward_line": self.beak_backward_line,
            "body_forward_line": self.body_forward_line,
            "body_backward_line": self.body_backward_line,
        }


@dataclass(frozen=True, slots=True)
class PowerSettings:
    manage_cpu_governor: bool = True
    active_governor: str = "performance"
    standby_governor: str = "ondemand"
    dormant_governor: str = "powersave"
    manage_wifi: bool = True
    wifi_interface: str = "wlan0"


@dataclass(frozen=True, slots=True)
class SetupSettings:
    """Setup mode: Lucy's own Wi-Fi and a settings page for a grown-up's phone (lucy.setup)."""

    enabled: bool = True
    hold_to_start_s: float = 3.0
    """Holding her wing while she wakes up, this long after she is ready, starts setup mode.
    0 turns the gesture off."""
    voice_command: bool = True
    """While she is awake, the knock (two taps on her wing, then holding it) and the words
    "enter setup mode" start setup mode (lucy.core.knock)."""
    when_offline: bool = True
    """Start setup mode by herself when no network has connected ``offline_wait_s`` after boot."""
    offline_wait_s: float = 90.0
    idle_timeout_s: float = 600.0
    """Leave setup mode after this long with nothing happening on the page."""
    interface: str = "wlan0"
    hotspot_file: Path = Path("/etc/lucy/setup-hotspot.toml")
    """Her setup Wi-Fi's name and password (root-only), made by ``lucy setup-card``."""
    credentials_dir: Path = Path("/etc/lucy/credentials")
    """Where the settings page stores API keys (the files lucy.service loads as credentials)."""


@dataclass(frozen=True, slots=True)
class Config:
    log_level: str = "info"
    system_prompt_file: Path | None = None
    """Optional replacement for Lucy's built-in system prompt."""
    api: ApiSettings = field(default_factory=ApiSettings)
    stt: SttSettings = field(default_factory=SttSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    logbook: LogbookSettings = field(default_factory=LogbookSettings)
    local: LocalSettings = field(default_factory=LocalSettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    button: ButtonSettings = field(default_factory=ButtonSettings)
    led: LedSettings = field(default_factory=LedSettings)
    motors: MotorSettings = field(default_factory=MotorSettings)
    power: PowerSettings = field(default_factory=PowerSettings)
    turn: TurnSettings = field(default_factory=TurnSettings)
    session: SessionSettings = field(default_factory=SessionSettings)
    volume: VolumeSettings = field(default_factory=VolumeSettings)
    setup: SetupSettings = field(default_factory=SetupSettings)


def load_config(path: Path | None) -> Config:
    """Load ``path``, or the built-in defaults when ``path`` is None."""
    if path is None:
        return Config()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"config file not found: {path}") from error
    return parse_config(text, path)


def parse_config(text: str, path: Path) -> Config:
    """Check ``text`` as the contents of the config file at ``path``, without reading it.

    Relative paths in it are relative to ``path``'s directory, as in :func:`load_config`.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error
    config = _build(Config, data, "", path.parent)
    _validate(config)
    return config


def default_config_path(environ: Mapping[str, str] = os.environ) -> Path | None:
    if override := environ.get("LUCY_CONFIG"):
        return Path(override)
    return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None


class KeySource(Protocol):
    """Settings that name where an API key comes from (:class:`ApiSettings` and the like)."""

    @property
    def api_key_credential(self) -> str: ...

    @property
    def api_key_env(self) -> str: ...


def read_api_key(
    settings: KeySource,
    environ: Mapping[str, str] = os.environ,
    env_file: Path = Path(".env"),
) -> str:
    """Return the API key from a systemd credential, the environment, or ``env_file``.

    Returns "" if none has it. The environment wins over ``env_file``, as with any
    ``.env`` loader, and the key read from the file is not added to the environment.
    """
    if credentials := environ.get("CREDENTIALS_DIRECTORY"):
        path = Path(credentials) / settings.api_key_credential
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    if key := environ.get(settings.api_key_env, "").strip():
        return key
    return _read_env_file(env_file).get(settings.api_key_env, "")


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file: ``NAME=value`` lines, optionally quoted or after ``export``."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:  # usually absent; a key problem must never stop Lucy from starting
        return {}
    values = {}
    for line in lines:
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if not sep or name.startswith("#"):
            continue
        value = value.strip()
        quote = value[:1]
        if quote in ("'", '"'):
            value = value[1:].partition(quote)[0]
        else:
            value = value.partition(" #")[0].rstrip()  # drop a trailing comment
        values[name.strip()] = value
    return values


def read_system_prompt(config: Config, default: str) -> str:
    if config.system_prompt_file is None:
        return default
    try:
        prompt = config.system_prompt_file.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ConfigError(f"cannot read system_prompt_file: {error}") from error
    if not prompt:
        raise ConfigError(f"{config.system_prompt_file} is empty")
    return prompt


def device(name: str) -> int | str | None:
    """Translate a config device string into what PortAudio expects."""
    name = name.strip()
    if not name:
        return None
    return int(name) if name.isdigit() else name


def _validate(config: Config) -> None:
    problems = []
    if config.button.backend not in ("gpio", "keyboard"):
        problems.append("button.backend must be 'gpio' or 'keyboard'")
    if config.tts.threads < 1:
        problems.append("tts.threads must be at least 1")
    cloud = config.tts.cloud
    volumes = {
        "tts.volume": config.tts.volume,
        "tts.honk_volume": config.tts.honk_volume,
        "tts.cloud.volume": cloud.volume,
    }
    problems += [
        f"{name} must be between 0 and 2" for name, value in volumes.items() if not 0 <= value <= 2
    ]
    problems += _sound_problems(config)
    if cloud.timeout_s <= 0 or cloud.rest_s < 0:
        problems.append("tts.cloud.timeout_s must be positive and tts.cloud.rest_s >= 0")
    if config.turn.first_clause_min_words < 0:
        problems.append("turn.first_clause_min_words must be 0 (whole sentences) or more")
    if config.audio.input_sample_rate not in (8_000, 16_000, 22_050, 24_000, 44_100, 48_000):
        problems.append("audio.input_sample_rate must be a standard rate (16000 recommended)")
    if config.local.enabled and config.audio.input_sample_rate != 16_000:
        problems.append("local.enabled needs audio.input_sample_rate = 16000")
    if config.local.history_exchanges < 0 or config.local.stt_threads < 1:
        problems.append("local.history_exchanges must be >= 0 and local.stt_threads >= 1")
    problems += _memory_problems(config)
    if config.session.tap_max_s >= config.session.max_record_s:
        problems.append("session.tap_max_s must be shorter than session.max_record_s")
    if not 0 < config.session.tap_gap_s <= 3:
        problems.append("session.tap_gap_s must be above 0 and at most 3 (seconds)")
    if config.session.awake_s <= 0 or config.session.forget_after_s <= 0:
        problems.append("session.awake_s and session.forget_after_s must be positive")
    if config.motors.enabled:
        problems += _motor_problems(config)
    problems += _setup_problems(config.setup)
    if problems:
        raise ConfigError("invalid configuration: " + "; ".join(problems))


def _memory_problems(config: Config) -> list[str]:
    problems = []
    if config.llm.history_exchanges < 0:
        problems.append("llm.history_exchanges must be 0 or more")
    if not 10 <= config.memory.max_words <= 200:
        problems.append("memory.max_words must be between 10 and 200")
    if config.memory.notes and not config.memory.model.strip():
        problems.append("memory.model must name a model (or set memory.notes = false)")
    if config.memory.timeout_s <= 0:
        problems.append("memory.timeout_s must be positive")
    if not 0 <= config.logbook.keep_days <= 3650:
        problems.append("logbook.keep_days must be between 0 (keep everything) and 3650")
    return problems


def _sound_problems(config: Config) -> list[str]:
    """Her goose voice, honks, chimes, and the volume a child can change."""
    problems = []
    if not 0 <= config.tts.goose_voice <= 1:
        problems.append("tts.goose_voice must be between 0 (her plain voice) and 1")
    if not 0 <= config.session.earcon_volume <= 1.5:
        problems.append("session.earcon_volume must be between 0 and 1.5")
    if not 0 <= config.turn.opening_honk_chance <= 1:
        problems.append("turn.opening_honk_chance must be between 0 and 1")
    volume = config.volume
    if not 0 < volume.step_db <= 12:
        problems.append("volume.step_db must be above 0 and at most 12")
    if not 0 <= volume.max_db <= 12:
        problems.append("volume.max_db must be between 0 and 12")
    if not -40 <= volume.min_db <= 0:
        problems.append("volume.min_db must be between -40 and 0")
    if not -60 <= volume.speaker_db <= volume.speaker_max_db <= 12:
        problems.append(
            "volume.speaker_db must be between -60 and volume.speaker_max_db, "
            "and volume.speaker_max_db at most 12"
        )
    return problems


def _setup_problems(setup: SetupSettings) -> list[str]:
    problems = []
    if setup.hold_to_start_s < 0 or setup.offline_wait_s <= 0:
        problems.append("setup.hold_to_start_s must be >= 0 and setup.offline_wait_s positive")
    if setup.idle_timeout_s < 60:
        problems.append("setup.idle_timeout_s must be at least 60 (seconds)")
    return problems


def _motor_problems(config: Config) -> list[str]:
    motors, motion = config.motors, config.motors.motion
    problems = []
    if motors.beak_pwm_channel == motors.body_pwm_channel:
        problems.append("motors.beak_pwm_channel and motors.body_pwm_channel must differ")
    if not 1_000 <= motors.pwm_frequency_hz <= 50_000:
        problems.append("motors.pwm_frequency_hz must be between 1000 and 50000 (20000 is silent)")
    lines = {f"motors.{name}": line for name, line in motors.lines().items()}
    taken = {}
    if config.button.chip == motors.gpio_chip and config.button.backend == "gpio":
        taken[config.button.line] = "button.line"
    if config.led.chip == motors.gpio_chip and config.led.enabled:
        taken[config.led.line] = "led.line"
    for name, line in lines.items():
        if line < 0:
            problems.append(f"{name} must be a GPIO line number")
        elif line in taken:
            problems.append(f"{name} {line} is already used by {taken[line]}")
        taken[line] = name
    powers = {
        "beak_min_power": motion.beak_min_power,
        "beak_max_power": motion.beak_max_power,
        "body_power": motion.body_power,
    }
    problems += [
        f"motors.motion.{name} must be between 0 and 1"
        for name, value in powers.items()
        if not 0 <= value <= 1
    ]
    if motion.beak_min_power > motion.beak_max_power:
        problems.append("motors.motion.beak_min_power must not be above beak_max_power")
    if not 0 <= motion.lead_s <= 0.3:
        problems.append("motors.motion.lead_s must be between 0 and 0.3")
    if motion.max_hold_s <= 0 or motion.bob_every_s < 0:
        problems.append("motors.motion.max_hold_s must be positive and bob_every_s >= 0")
    return problems


# -- strict TOML -> dataclass conversion ---------------------------------------------------

T = TypeVar("T")


def _build(cls: type[T], table: Any, where: str, base: Path) -> T:
    label = f"[{where}]" if where else "the top level"
    if not isinstance(table, Mapping):
        raise ConfigError(f"{label} must be a table")
    fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(table) - fields)
    if unknown:
        raise ConfigError(f"unknown setting(s) in {label}: {', '.join(unknown)}")
    hints = typing.get_type_hints(cls)
    values = {
        name: _coerce(hints[name], value, f"{where}.{name}" if where else name, base)
        for name, value in table.items()
    }
    return cls(**values)


def _coerce(kind: Any, value: Any, where: str, base: Path) -> Any:
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if dataclasses.is_dataclass(kind):
        return _build(kind, value, where, base)  # type: ignore[arg-type]
    if origin in (typing.Union, types.UnionType):
        (inner,) = [arg for arg in args if arg is not type(None)]
        return _coerce(inner, value, where, base)
    if kind is Any:
        return value
    if kind is bool and isinstance(value, bool):
        return value
    if kind is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind is float and isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if kind is str and isinstance(value, str):
        return value
    if kind is Path and isinstance(value, str):
        return base / value  # an absolute value replaces base entirely
    if origin is tuple and isinstance(value, list):
        return tuple(_coerce(args[0], item, f"{where}[]", base) for item in value)
    if origin is dict and isinstance(value, Mapping):
        return dict(value)
    expected = getattr(kind, "__name__", str(kind))
    raise ConfigError(f"{where} should be {expected}, not {type(value).__name__} ({value!r})")
