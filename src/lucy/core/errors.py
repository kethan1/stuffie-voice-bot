"""Domain errors.

Adapters translate library-specific failures (httpx, PortAudio, libgpiod, ...)
into these types so the core can decide what Lucy should *say* without knowing
which library broke.
"""


class LucyError(Exception):
    """Base class for every error Lucy knows how to explain."""


class ConfigError(LucyError):
    """The configuration file or environment is invalid."""


class AudioDeviceError(LucyError):
    """A microphone or speaker could not be opened or used."""


class ServiceError(LucyError):
    """A remote AI service failed: bad status, malformed reply, or timeout."""


class ServiceUnreachable(ServiceError):
    """The network or the service host could not be reached at all."""


class AuthenticationError(ServiceError):
    """The service rejected our credentials (missing, wrong, or revoked API key)."""


class RateLimited(ServiceError):
    """The service asked us to slow down."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
