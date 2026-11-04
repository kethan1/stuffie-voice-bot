"""System power profiles: trade latency for battery life depending on what Lucy is doing.

A Raspberry Pi 4 cannot suspend, so savings come from tuning what is running:

CPU frequency governor
    ACTIVE uses ``performance``: the cores sit at full clock while Whisper's
    reply is parsed and Piper synthesises, so there is no ramp-up delay.
    STANDBY uses ``ondemand`` (the Raspberry Pi OS default). DORMANT uses
    ``powersave``, which pins the lowest clock (and voltage) while nobody is
    talking.

Wi-Fi power saving
    With power save on, the access point buffers packets until the radio's
    next scheduled wake-up, which adds roughly 100-300 ms of jitter to every
    round trip. It is switched off while a conversation is active and back on
    when Lucy goes dormant.

Each knob degrades gracefully. If the service lacks permission (the unit's
``ExecStartPre`` grants the governor files and ``AmbientCapabilities`` grants
``iw``), the knob logs once and disables itself, and Lucy keeps working.
"""

import asyncio
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from lucy.core.types import PowerMode

log = logging.getLogger(__name__)

CPUFREQ_ROOT = Path("/sys/devices/system/cpu/cpufreq")


class PowerKnob(Protocol):
    name: str

    async def apply(self, mode: PowerMode) -> None: ...

    async def restore(self) -> None: ...


class PowerManager:
    """Applies a power mode to every knob, in order, skipping repeats."""

    def __init__(self, knobs: Sequence[PowerKnob]) -> None:
        self._knobs = list(knobs)
        self._mode: PowerMode | None = None
        self._lock = asyncio.Lock()

    async def apply(self, mode: PowerMode) -> None:
        async with self._lock:
            if mode is self._mode:
                return
            self._mode = mode
            await self._each("apply", mode)
            log.debug("power mode: %s", mode.value)

    async def restore(self) -> None:
        async with self._lock:
            self._mode = None
            await self._each("restore")

    async def _each(self, action: str, *args: PowerMode) -> None:
        for knob in list(self._knobs):
            try:
                await getattr(knob, action)(*args)
            except (OSError, RuntimeError) as error:
                log.warning("disabling power knob %s: %s", knob.name, error)
                self._knobs.remove(knob)


class CpuGovernor:
    name = "cpu-governor"

    def __init__(self, governors: Mapping[PowerMode, str], root: Path = CPUFREQ_ROOT) -> None:
        self._governors = dict(governors)
        self._files = sorted(root.glob("policy*/scaling_governor"))
        if not self._files:
            raise RuntimeError(f"no cpufreq policies under {root}")
        self._original = {path: path.read_text().strip() for path in self._files}

    async def apply(self, mode: PowerMode) -> None:
        governor = self._governors.get(mode)
        if governor:
            self._write(governor)

    async def restore(self) -> None:
        for path, governor in self._original.items():
            path.write_text(governor)

    def _write(self, governor: str) -> None:
        # sysfs writes are a few microseconds, so they run inline on the event loop.
        for path in self._files:
            path.write_text(governor)


class WifiPowerSave:
    name = "wifi-power-save"

    def __init__(self, interface: str, iw: str | None = None) -> None:
        resolved = iw or shutil.which("iw") or shutil.which("iw", path="/usr/sbin:/sbin")
        if resolved is None:
            raise RuntimeError("the `iw` tool is not installed")
        self._iw = resolved
        self._interface = interface
        self._original: str | None = None

    async def apply(self, mode: PowerMode) -> None:
        if self._original is None:
            self._original = await self._current()
        await self._set("on" if mode is PowerMode.DORMANT else "off")

    async def restore(self) -> None:
        if self._original is not None:
            await self._set(self._original)

    async def _current(self) -> str:
        output = await self._iw_run("dev", self._interface, "get", "power_save")
        return "on" if output.strip().lower().endswith("on") else "off"

    async def _set(self, state: str) -> None:
        await self._iw_run("dev", self._interface, "set", "power_save", state)

    async def _iw_run(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            self._iw,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"iw {' '.join(args)}: {stderr.decode().strip() or process.returncode}"
            )
        return stdout.decode()


class NoPowerControl:
    async def apply(self, mode: PowerMode) -> None:
        pass

    async def restore(self) -> None:
        pass
