"""Lucy's beak and body motors on an L293D dual H-bridge (docs/HARDWARE.md).

Each motor has one half of the L293D: an enable pin (1,2EN or 3,4EN) on a hardware PWM
channel, which sets its speed (the Pi 4 has exactly two), and two inputs (1A and 2A, or 3A
and 4A) on GPIO lines, which pick its direction. ``power`` runs from -1 to 1: the sign picks
the way, and 0 lets the motor **coast**, so the beak's spring can snap it shut.

=============  ======  ======  ==============  ================================
power          1A      2A      EN              the motor
=============  ======  ======  ==============  ================================
above 0        high    low     PWM, = power    forward, coasting between pulses
below 0        low     high    PWM, = |power|  reverse, coasting between pulses
0              (kept)  (kept)  low             coasts (the beak's spring shuts it)
=============  ======  ======  ==============  ================================

An L293D brakes a motor whenever its two inputs are at the same level and its enable pin is
high. A braked beak motor would fight the spring and hold the beak open, so the PWM goes on
the enable pin: with it low the outputs are off and the motor coasts. The inputs change only
when the direction does, with the enable pin low first, so turning around never brakes
either. The L293D has no sleep mode: with both enable pins low its outputs are off, but its
logic keeps drawing a little from the Pi's 5 V.

The PWM runs in hardware at 20 kHz, which is above hearing (so the motors do not whine) and
takes no CPU. Each change of speed is one sysfs write of a few microseconds, made on the
event loop: no threads and no polling.

Every line Lucy drives is pulled low while the Pi boots (GPIO 9 to 27 default to
pull-down), so both enable pins stay low and the motors stay still until Lucy starts. A
sysfs PWM channel keeps running after the process that set it exits, so ``close()`` stops
everything, and the service's ``ExecStopPost`` runs ``lucy motors --off`` in case Lucy was
killed before she could.
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
    """One half of the L293D: its enable pin on a PWM channel, its two inputs on GPIO lines."""

    def __init__(self, pwm: Pwm, forward: Line, backward: Line) -> None:
        self._pwm = pwm
        self._forward = forward
        self._backward = backward
        self._way = 0  # 1 forward, -1 backward, 0 both inputs low
        pwm.set(0.0)
        forward.set(False)
        backward.set(False)

    def drive(self, power: float) -> None:
        power = min(1.0, max(-1.0, power))
        way = (power > 0) - (power < 0)
        if way and way != self._way:
            self._pwm.set(0.0)  # outputs off while the inputs change, so it never brakes
            self._forward.set(way > 0)
            self._backward.set(way < 0)
            self._way = way
        self._pwm.set(abs(power))

    def close(self) -> None:
        """Let the motor coast with both inputs low, then release the channel and the lines."""
        self._pwm.set(0.0)
        self._forward.set(False)
        self._backward.set(False)
        self._way = 0
        self._pwm.close()
        self._forward.close()
        self._backward.close()


class L293d:
    """Lucy's motors (:class:`~lucy.core.ports.Motors`) on one L293D."""

    def __init__(self, bridges: Mapping[Motor, HBridge]) -> None:
        self._bridges = dict(bridges)
        self.sleep()

    def drive(self, motor: Motor, power: float) -> None:
        self._bridges[motor].drive(power)

    def sleep(self) -> None:
        for bridge in self._bridges.values():
            bridge.drive(0.0)

    def close(self) -> None:
        """Stop both motors and release every line."""
        with contextlib.suppress(OSError):
            self.sleep()
        for bridge in self._bridges.values():
            try:
                bridge.close()
            except OSError as error:
                log.warning("could not release a motor line: %s", error)


def open_l293d(
    *,
    pwm_chip: Path,
    frequency_hz: float,
    gpio_chip: str,
    channels: Mapping[Motor, tuple[int, int, int]],
) -> L293d:
    """Claim a PWM channel and two input lines per motor, with both motors still.

    ``channels`` maps each motor to (PWM channel on its enable pin, forward line on 1A or
    3A, backward line on 2A or 4A). Blocking: it may wait a moment for udev to make a newly
    exported PWM channel writable.
    """
    from lucy.adapters.gpio import GpioOutput

    with ExitStack() as opened:
        bridges = {}
        for motor, (channel, forward_line, backward_line) in channels.items():
            pwm = SysfsPwm.open(pwm_chip, channel, frequency_hz)
            opened.callback(pwm.close)
            forward = GpioOutput(gpio_chip, forward_line, consumer=f"lucy-{motor}-forward")
            opened.callback(forward.close)
            backward = GpioOutput(gpio_chip, backward_line, consumer=f"lucy-{motor}-backward")
            opened.callback(backward.close)
            bridges[motor] = HBridge(pwm, forward, backward)
        driver = L293d(bridges)
        opened.pop_all()
    return driver
