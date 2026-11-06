"""Lucy's beak and body motors on a DRV8833 dual H-bridge (docs/HARDWARE.md).

Each motor has two driver inputs. xIN1 is on a hardware PWM channel and xIN2 on
a plain GPIO line that picks the direction. With the DRV8833's truth table, one
PWM channel per motor is enough for both directions:

=============  =========  ======================  ===============================
power          xIN2       xIN1                    the motor
=============  =========  ======================  ===============================
above 0        low        PWM, duty = power       forward, coasting between pulses
below 0        high       PWM, duty = 1 - |power| reverse, braking between pulses
0              low        low                     coasts (the beak's spring shuts it)
=============  =========  ======================  ===============================

The PWM runs in hardware at 20 kHz, which is above hearing (so the motors do not
whine) and takes no CPU. Each change of speed is one sysfs write of a few
microseconds, made on the event loop: no threads and no polling. While nothing
moves, nSLEEP is held low and the driver draws next to nothing.

Every line Lucy drives is pulled low while the Pi boots (GPIO 9 to 27 default to
pull-down, and the DRV8833 pulls nSLEEP down), so the motors stay still until
Lucy starts. A sysfs PWM channel keeps running after the process that set it
exits, so ``close()`` stops everything, and the service's ``ExecStopPost`` runs
``lucy motors --off`` in case Lucy was killed before she could.
"""

import contextlib
import logging
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Protocol

from lucy.core.errors import ConfigError
from lucy.core.types import Motor

log = logging.getLogger(__name__)


class Line(Protocol):
    """A GPIO output line (:class:`lucy.adapters.gpio.GpioOutput`)."""

    def set(self, on: bool) -> None: ...

    def close(self) -> None: ...


class Pwm(Protocol):
    def set(self, duty: float) -> None:
        """Set the fraction of each period the output is high, from 0 to 1."""
        ...

    def close(self) -> None: ...


class SysfsPwm:
    """One hardware PWM channel, through the kernel's ``/sys/class/pwm`` interface.

    On a Pi 4, ``dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4`` puts channel 0 of
    ``pwmchip0`` on GPIO12 and channel 1 on GPIO13. Raspberry Pi OS's udev rules let the
    ``gpio`` group use it; :meth:`open` waits for udev to grant that after exporting.
    """

    def __init__(self, directory: Path, period_ns: int) -> None:
        self._dir = directory
        self._period = period_ns
        self._duty = 0

    @classmethod
    def open(
        cls, chip: Path, channel: int, frequency_hz: float, *, wait_s: float = 2.0
    ) -> "SysfsPwm":
        """Export ``channel`` if needed and start it at ``frequency_hz`` with the output low."""
        if not (chip / "npwm").exists():
            raise ConfigError(
                f"{chip} does not exist: add the PWM overlay to config.txt "
                "(sudo deploy/install.sh --motors) and reboot"
            )
        count = int((chip / "npwm").read_text().strip() or 0)
        if not 0 <= channel < count:
            raise ConfigError(f"{chip} has {count} PWM channel(s), so no channel {channel}")
        directory = chip / f"pwm{channel}"
        if not directory.is_dir():
            try:
                (chip / "export").write_text(str(channel))
            except OSError as error:
                raise ConfigError(
                    f"cannot export PWM channel {channel} of {chip}: {error}"
                ) from error
        pwm = cls(directory, round(1e9 / frequency_hz))
        deadline = time.monotonic() + wait_s
        while True:
            try:
                pwm._start()
                return pwm
            except (PermissionError, FileNotFoundError) as error:  # udev is not done with it yet
                if time.monotonic() > deadline:
                    raise ConfigError(
                        f"no permission on {directory} (is the user in the 'gpio' group?)"
                    ) from error
                time.sleep(0.02)

    def _start(self) -> None:
        if self._read("duty_cycle") > self._period:
            self._write("duty_cycle", 0)  # the kernel refuses a period shorter than the duty
        self._write("period", self._period)
        self._write("duty_cycle", 0)
        self._write("enable", 1)

    def set(self, duty: float) -> None:
        duty_ns = round(min(1.0, max(0.0, duty)) * self._period)
        if duty_ns != self._duty:
            self._write("duty_cycle", duty_ns)
            self._duty = duty_ns

    def close(self) -> None:
        self.set(0.0)
        self._write("enable", 0)

    def _read(self, name: str) -> int:
        text = (self._dir / name).read_text().strip()
        return int(text) if text.isdigit() else 0

    def _write(self, name: str, value: int) -> None:
        (self._dir / name).write_text(str(value))


class HBridge:
    """One DRV8833 channel: xIN1 on a PWM channel, xIN2 on a GPIO line."""

    def __init__(self, pwm: Pwm, direction: Line) -> None:
        self._pwm = pwm
        self._direction = direction
        self._reverse = False
        direction.set(False)
        pwm.set(0.0)

    def drive(self, power: float) -> None:
        power = min(1.0, max(-1.0, power))
        reverse = power < 0
        if reverse != self._reverse:
            self._direction.set(reverse)
            self._reverse = reverse
        self._pwm.set(1.0 - abs(power) if reverse else power)

    def close(self) -> None:
        self.drive(0.0)
        self._pwm.close()
        self._direction.close()


class Drv8833:
    """Lucy's motors (:class:`~lucy.core.ports.Motors`) on one DRV8833."""

    def __init__(self, bridges: Mapping[Motor, HBridge], sleep_line: Line | None) -> None:
        self._bridges = dict(bridges)
        self._sleep_line = sleep_line
        self._awake = False
        self.sleep()

    def drive(self, motor: Motor, power: float) -> None:
        if power and not self._awake:
            self._awake = True
            if self._sleep_line is not None:
                self._sleep_line.set(True)  # awake within about a millisecond
        self._bridges[motor].drive(power)

    def sleep(self) -> None:
        for bridge in self._bridges.values():
            bridge.drive(0.0)
        if self._sleep_line is not None:
            self._sleep_line.set(False)
        self._awake = False

    def close(self) -> None:
        """Stop both motors, put the driver to sleep, and release every line."""
        with contextlib.suppress(OSError):
            self.sleep()
        for part in (*self._bridges.values(), self._sleep_line):
            if part is None:
                continue
            try:
                part.close()
            except OSError as error:
                log.warning("could not release a motor line: %s", error)


def open_drv8833(
    *,
    pwm_chip: Path,
    frequency_hz: float,
    gpio_chip: str,
    channels: Mapping[Motor, tuple[int, int]],
    sleep_line: int | None,
) -> Drv8833:
    """Claim a PWM channel and a direction line per motor, and the sleep line, all off.

    ``channels`` maps each motor to (PWM channel, direction GPIO line). Blocking: it may
    wait a moment for udev to make a newly exported PWM channel writable.
    """
    from lucy.adapters.gpio import GpioOutput

    with ExitStack() as opened:
        bridges = {}
        for motor, (channel, line) in channels.items():
            pwm = SysfsPwm.open(pwm_chip, channel, frequency_hz)
            opened.callback(pwm.close)
            direction = GpioOutput(gpio_chip, line, consumer=f"lucy-{motor}")
            opened.callback(direction.close)
            bridges[motor] = HBridge(pwm, direction)
        sleep = None
        if sleep_line is not None:
            sleep = GpioOutput(gpio_chip, sleep_line, consumer="lucy-motor-sleep")
            opened.callback(sleep.close)
        driver = Drv8833(bridges, sleep)
        opened.pop_all()
    return driver
