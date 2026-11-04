"""Minimal systemd integration: readiness, status, and the watchdog (no dependencies).

With ``Type=notify``, systemd considers Lucy started only once the voice is
loaded and warmed up. With ``WatchdogSec=``, the event loop must ping
regularly or systemd restarts the service. A wedged loop (deadlock, stuck
driver call) therefore heals itself instead of leaving a silent toy.
"""

import asyncio
import logging
import os
import socket

log = logging.getLogger(__name__)


class SystemdNotifier:
    def __init__(self, environ: dict[str, str] | None = None) -> None:
        env = os.environ if environ is None else environ
        address = env.get("NOTIFY_SOCKET", "")
        if address.startswith("@"):  # abstract namespace
            address = "\0" + address[1:]
        self._address = address or None
        watchdog_usec = env.get("WATCHDOG_USEC")
        watchdog_pid = env.get("WATCHDOG_PID")
        enabled = watchdog_usec is not None and watchdog_pid in (None, str(os.getpid()))
        self.watchdog_interval_s = (
            int(watchdog_usec) / 1e6 / 2 if enabled and watchdog_usec else None
        )

    @property
    def enabled(self) -> bool:
        return self._address is not None

    def notify(self, *fields: str) -> None:
        if self._address is None:
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sock:
                sock.connect(self._address)
                sock.sendall("\n".join(fields).encode())
        except OSError as error:
            log.warning("sd_notify failed: %s", error)

    def ready(self, status: str) -> None:
        self.notify("READY=1", f"STATUS={status}")

    def status(self, status: str) -> None:
        self.notify(f"STATUS={status}")

    def stopping(self) -> None:
        self.notify("STOPPING=1")

    async def watchdog(self) -> None:
        """Ping the watchdog from the event loop, at half the configured timeout."""
        if self.watchdog_interval_s is None:
            return
        while True:
            self.notify("WATCHDOG=1")
            await asyncio.sleep(self.watchdog_interval_s)
