"""The settings on setup mode's page, and saving them to Lucy's config file and credentials.

The page offers a hand-picked set of settings (:data:`SECTIONS`), each tied to its key in
``config.toml``; a test checks every one against :class:`lucy.config.Config`, so the page
cannot drift from the real settings. Safety settings are deliberately not among them (they
are only in the "Advanced" view of the whole file). The logbook's settings sit beside the
link to the conversations it holds.

Saving edits ``config.toml`` in place with tomlkit, so its comments and layout survive, and
checks the result with the same loader Lucy starts with before writing anything: a change
that Lucy would refuse is never saved. The file before the last change is kept, for "Undo".
API keys are written to the root-only files lucy.service loads as systemd credentials.
"""

import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
import tomlkit.exceptions

from lucy.config import Config, parse_config
from lucy.core.errors import ConfigError
from lucy.setup.files import write_atomically


@dataclass(frozen=True, slots=True)
class Field:
    path: str
    """Dotted key in config.toml, such as ``tts.cloud.volume``."""
    label: str
    kind: type
    help: str = ""
    low: float = 0.0
    high: float = 0.0
    step: float = 0.0


@dataclass(frozen=True, slots=True)
class Section:
    key: str
    title: str
    intro: str
    fields: tuple[Field, ...]


SECTIONS = (
    Section(
        "lucy",
        "Lucy",
        "How she sounds and behaves.",
        (
            Field(
                "tts.cloud.voice_id",
                "Natural voice",
                str,
                "A Speechify voice, such as harper_32 (sudo lucy voices lists them).",
            ),
            Field("tts.cloud.volume", "Natural voice loudness", float, "1 is normal.", 0, 2, 0.05),
            Field(
                "tts.volume",
                "Backup voice loudness",
                float,
                "Her voice on the Pi, used when the natural voice can't speak.",
                0,
                2,
                0.05,
            ),
            Field(
                "tts.goose_voice",
                "Goose voice",
                float,
                "How much like a goose she sounds: 0 is her plain voice. Try it with Hear her.",
                0,
                1,
                0.05,
            ),
            Field("tts.honks", "Honk like a real goose", bool, "Off: she says the word."),
            Field(
                "tts.honk_volume", "Honk loudness", float, "1 is as loud as her voice.", 0, 2, 0.05
            ),
            Field("session.greet_on_start", "Say hello when she wakes up", bool),
            Field("session.earcons", "Chime when she starts and stops listening", bool),
            Field(
                "session.earcon_volume",
                "Chime loudness",
                float,
                "1 is a little louder than her voice.",
                0,
                1.5,
                0.05,
            ),
            Field(
                "volume.commands",
                "Children can ask her to talk louder or quieter",
                bool,
                "She goes back to her usual loudness when she is turned off.",
            ),
            Field(
                "memory.notes",
                "Remember more of each conversation",
                bool,
                "She keeps a few short notes on what a child told her earlier. They are never "
                "saved, and she forgets them once it has been quiet for a while.",
            ),
            Field(
                "setup.voice_command",
                "Setup mode can be started by voice",
                bool,
                'Tap her wing twice, then hold it and say "enter setup mode".',
            ),
            Field(
                "session.awake_s",
                "Stay quick to answer for (seconds)",
                float,
                "After this long with nobody talking, she saves power.",
                5,
                600,
                5,
            ),
            Field(
                "session.max_turns_per_hour",
                "Most questions per hour",
                int,
                "Protects the free API allowance. 0 means no limit.",
                0,
                1000,
                1,
            ),
            Field(
                "turn.speech_threshold_dbfs",
                "Quietest voice she hears (dBFS)",
                float,
                "Lower hears quieter voices, and more noise. -50 suits most rooms.",
                -80,
                -20,
                1,
            ),
        ),
    ),
    Section(
        "motors",
        "Motors",
        "Her beak and body. Try the numbers with Test motors before saving.",
        (
            Field("motors.enabled", "Beak and body motors are fitted", bool),
            Field(
                "motors.motion.beak_min_power",
                "Smallest beak power",
                float,
                "Just enough to start the beak moving against its spring.",
                0,
                1,
                0.05,
            ),
            Field(
                "motors.motion.beak_max_power",
                "Largest beak power",
                float,
                "Opens the beak wide without slamming it.",
                0,
                1,
                0.05,
            ),
            Field("motors.motion.body_power", "Body power", float, "", 0, 1, 0.05),
            Field(
                "motors.motion.body_two_way",
                "Body motor may run both ways",
                bool,
                "Off if it may only run forward, against a spring.",
            ),
            Field(
                "motors.motion.lead_s",
                "Move ahead of her voice by (seconds)",
                float,
                "Larger moves the beak sooner. Try 0.03 to 0.1.",
                0,
                0.3,
                0.01,
            ),
            Field(
                "motors.motion.bob_every_s",
                "Bob at most every (seconds)",
                float,
                "0: only flap on honks.",
                0,
                30,
                0.5,
            ),
        ),
    ),
    # Shown with a link to the conversations themselves (pages._logbook), not as the others are.
    Section(
        "logbook",
        "Conversations",
        "Lucy writes down what is said to her and what she answers, on her memory card, for "
        "grown-ups to read here. She never reads it back herself.",
        (
            Field(
                "logbook.enabled",
                "Write down every conversation",
                bool,
                "Off: nothing more is written. What is already there stays until you delete it.",
            ),
            Field(
                "logbook.keep_days",
                "Keep them for (days)",
                int,
                "Older ones are deleted by themselves. 0 keeps them until you delete them.",
                0,
                3650,
                1,
            ),
        ),
    ),
)

SECTION = {section.key: section for section in SECTIONS}
FIELD = {field.path: field for section in SECTIONS for field in section.fields}
MAX_TEXT = 200


def current(config: Any, path: str) -> Any:
    """The value at a dotted ``path`` in a :class:`Config` (or anything with attributes)."""
    for name in path.split("."):
        config = getattr(config, name)
    return config


def parse_section(section: Section, form: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    """The changes a posted section asks for, and anything wrong with them."""
    changes: dict[str, Any] = {}
    problems = []
    for field in section.fields:
        raw = form.get(field.path)
        if field.kind is bool:
            changes[field.path] = raw is not None  # an unticked box is not sent at all
            continue
        text = (raw or "").strip()
        if field.kind is str:
            if not text or len(text) > MAX_TEXT or not text.isprintable():
                problems.append(f"{field.label}: type a value (letters, numbers, and symbols).")
            else:
                changes[field.path] = text
            continue
        try:
            number = int(text) if field.kind is int else float(text)
        except ValueError:
            problems.append(f"{field.label}: type a number.")
            continue
        if not math.isfinite(number) or not field.low <= number <= field.high:
            problems.append(f"{field.label}: use a number from {field.low:g} to {field.high:g}.")
            continue
        changes[field.path] = number
    return changes, problems


def key_problem(key: str) -> str | None:
    """What is wrong with a pasted API key, if anything (keys are long, plain, and unbroken)."""
    key = key.strip()
    if 16 <= len(key) <= 512 and key.isascii() and key.isprintable() and " " not in key:
        return None
    return "That doesn't look like an API key: paste the whole key, and only the key."


@dataclass(frozen=True, slots=True)
class KeyStatus:
    present: bool
    ending: str = ""
    """The key's last four characters, so a grown-up can tell which key it is."""


class SettingsStore:
    """Lucy's config file and API keys, as setup mode changes them."""

    def __init__(self, path: Path, credentials_dir: Path) -> None:
        self.path = path
        self.credentials_dir = credentials_dir
        self.backup = path.with_name(f"{path.name}.previous")

    def text(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def load(self) -> Config:
        return parse_config(self.text(), self.path)

    @property
    def can_undo(self) -> bool:
        return self.backup.is_file()

    def update(self, changes: Mapping[str, Any]) -> None:
        """Set each dotted key to its value, keeping the rest of the file as it is."""
        try:
            document = tomlkit.parse(self.text())
        except tomlkit.exceptions.TOMLKitError as error:
            raise ConfigError(f"{self.path} cannot be edited: {error}") from error
        for path, value in changes.items():
            *tables, key = path.split(".")
            node: Any = document
            for name in tables:
                if name not in node:
                    node[name] = tomlkit.table()
                node = node[name]
            node[key] = value
        self.replace(tomlkit.dumps(document))

    def replace(self, text: str) -> None:
        """Save ``text`` as the whole config file, if Lucy would accept it. Raises ConfigError."""
        parse_config(text, self.path)
        previous = self.text()
        if text == previous:
            return
        mode = self._mode()
        if previous:
            write_atomically(self.backup, previous, mode)
        write_atomically(self.path, text, mode)

    def undo(self) -> None:
        """Put back the file as it was before the last change (once)."""
        previous = self.backup.read_text(encoding="utf-8")
        parse_config(previous, self.path)
        write_atomically(self.path, previous, self._mode())
        self.backup.unlink()

    def _mode(self) -> int:
        try:
            return stat.S_IMODE(self.path.stat().st_mode)
        except FileNotFoundError:
            return 0o644

    # -- API keys ------------------------------------------------------------------------

    def key_status(self, credential: str) -> KeyStatus:
        try:
            key = (self.credentials_dir / credential).read_text(encoding="utf-8").strip()
        except OSError:
            return KeyStatus(False)
        return KeyStatus(bool(key), key[-4:] if len(key) >= 12 else "")

    def set_key(self, credential: str, key: str) -> None:
        """Store an API key (empty: remove it). Raises ConfigError if it cannot be a key."""
        key = key.strip()
        if key and (problem := key_problem(key)):
            raise ConfigError(problem)
        self.credentials_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.credentials_dir, 0o700)
        # The file stays even when empty: lucy.service loads it as a credential.
        write_atomically(self.credentials_dir / credential, key, 0o600)
