"""Configuration: one TOML file, strictly validated, with secrets kept out of it.

* Every setting has a sensible default, so an empty file is a valid file.
* Unknown keys are errors. A typo such as ``max_recrod_s`` fails loudly at
  start-up instead of being silently ignored.
* Relative paths are relative to the config file, so a config such as the
  repository's ``dev.toml`` works from any working directory.
* The API key never lives in the TOML file. It is read from the systemd
  credential ``groq_api_key`` (``LoadCredential=``; only the service can read
  it) or, for development, from ``GROQ_API_KEY`` in the environment or in a
  ``.env`` file in the working directory.
"""

import dataclasses
import os
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from lucy.core.controller import SessionSettings
from lucy.core.errors import ConfigError
from lucy.core.pipeline import TurnSettings

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
    """A 1B model on a Pi 4 reads prompts slowly, so it gets less history than the cloud."""
    timeout_s: float = 60.0
    preload: bool = True
    """Load both models at start-up, so the first offline answer does not wait for them."""


@dataclass(frozen=True, slots=True)
class SafetySettings:
    screen: bool = True
    """On-device distress screen (see lucy.core.safety). Strongly recommended."""
    extra_patterns: tuple[str, ...] = ()
    """Additional case-insensitive regular expressions that route to a trusted grown-up."""


@dataclass(frozen=True, slots=True)
class TtsSettings:
    voice: Path = Path("/opt/lucy/voices/en_US-amy-medium.onnx")
    threads: int = 3
    length_scale: float = 1.0
    """Above 1.0 speaks more slowly, which can help the youngest children."""
    volume: float = 0.9
    speaker_id: int | None = None


@dataclass(frozen=True, slots=True)
class AudioSettings:
    input_device: str = ""
    """PortAudio device name (substring) or index; empty means the ALSA default."""
    output_device: str = ""
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
class PowerSettings:
    manage_cpu_governor: bool = True
    active_governor: str = "performance"
    standby_governor: str = "ondemand"
    dormant_governor: str = "powersave"
    manage_wifi: bool = True
    wifi_interface: str = "wlan0"


@dataclass(frozen=True, slots=True)
class Config:
    log_level: str = "info"
    system_prompt_file: Path | None = None
    """Optional replacement for Lucy's built-in system prompt."""
    api: ApiSettings = field(default_factory=ApiSettings)
    stt: SttSettings = field(default_factory=SttSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    local: LocalSettings = field(default_factory=LocalSettings)
    safety: SafetySettings = field(default_factory=SafetySettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    button: ButtonSettings = field(default_factory=ButtonSettings)
    led: LedSettings = field(default_factory=LedSettings)
    power: PowerSettings = field(default_factory=PowerSettings)
    turn: TurnSettings = field(default_factory=TurnSettings)
    session: SessionSettings = field(default_factory=SessionSettings)


def load_config(path: Path | None) -> Config:
    """Load ``path``, or the built-in defaults when ``path`` is None."""
    if path is None:
        return Config()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"config file not found: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error
    config = _build(Config, data, "", path.parent)
    _validate(config)
    return config


def default_config_path(environ: Mapping[str, str] = os.environ) -> Path | None:
    if override := environ.get("LUCY_CONFIG"):
        return Path(override)
    return DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.is_file() else None


def read_api_key(
    settings: ApiSettings,
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
    if not 0.0 <= config.tts.volume <= 2.0:
        problems.append("tts.volume must be between 0 and 2")
    if config.audio.input_sample_rate not in (8_000, 16_000, 22_050, 24_000, 44_100, 48_000):
        problems.append("audio.input_sample_rate must be a standard rate (16000 recommended)")
    if config.local.enabled and config.audio.input_sample_rate != 16_000:
        problems.append("local.enabled needs audio.input_sample_rate = 16000")
    if config.local.history_exchanges < 0 or config.local.stt_threads < 1:
        problems.append("local.history_exchanges must be >= 0 and local.stt_threads >= 1")
    if config.session.tap_max_s >= config.session.max_record_s:
        problems.append("session.tap_max_s must be shorter than session.max_record_s")
    if config.session.awake_s <= 0 or config.session.forget_after_s <= 0:
        problems.append("session.awake_s and session.forget_after_s must be positive")
    if problems:
        raise ConfigError("invalid configuration: " + "; ".join(problems))


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
