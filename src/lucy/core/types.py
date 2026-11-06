"""Small value types shared across the core and its adapters."""

from dataclasses import dataclass, field
from enum import Enum, StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ButtonEvent:
    """A debounced change of the push-to-talk button.

    ``at`` is a ``time.monotonic()``-compatible timestamp taken as close to the
    hardware edge as possible (the kernel stamps GPIO edges itself).
    """

    pressed: bool
    at: float


class Status(Enum):
    """What Lucy is doing, as shown on the status LED."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class Motor(StrEnum):
    """Lucy's motors (docs/HARDWARE.md)."""

    BEAK = "beak"
    """Pushes the beak open against a spring; the spring closes it when the motor lets go."""
    BODY = "body"
    """Wiggles or flaps Lucy's body, one way or both."""


class PowerMode(Enum):
    """System power profile.

    ACTIVE   a child is talking or Lucy is answering: favour latency.
    STANDBY  between turns of one conversation: stay quick to respond.
    DORMANT  nobody has talked for a while: favour battery life.
    """

    ACTIVE = "active"
    STANDBY = "standby"
    DORMANT = "dormant"


class Concern(Enum):
    """Why a child's message should not go to the language model as-is."""

    NONE = "none"
    OFF_LIMITS = "off_limits"  # a topic Lucy steers away from
    UNKIND = "unkind"  # being mean about someone, or asking Lucy to be
    TRICK = "trick"  # trying to talk Lucy out of her rules
    PRIVATE = "private"  # personal details (address, phone number), kept off the network
    UNSAFE = "unsafe"  # something that could hurt someone: a dare, a weapon, a poison
    NEEDS_GROWN_UP = "needs_grown_up"  # the child may be hurt, unsafe, or in distress


@dataclass(frozen=True, slots=True)
class Assessment:
    concern: Concern = Concern.NONE
    reasons: tuple[str, ...] = field(default=())

    @property
    def allowed(self) -> bool:
        return self.concern is Concern.NONE


ALLOWED = Assessment()
