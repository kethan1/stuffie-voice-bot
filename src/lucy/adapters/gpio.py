"""Push-to-talk button and status LED on the GPIO character device (libgpiod v2).

There are no polling loops and no helper threads. The kernel timestamps
each button edge, applies the debounce period itself, and wakes the asyncio
event loop through the line request's file descriptor (``loop.add_reader``).
Between presses the process is fully asleep, which matters for battery life.

``RPi.GPIO`` and the sysfs GPIO interface are deprecated. libgpiod is the
supported interface on current Raspberry Pi OS kernels (and on the Pi 5).
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

from lucy.core.errors import ConfigError
from lucy.core.types import ButtonEvent, Status

log = logging.getLogger(__name__)


class GpioButton:
    def __init__(
        self, chip: str, line: int, *, active_low: bool = True, debounce_ms: int = 15
    ) -> None:
        self._chip = chip
        self._line = line
        self._active_low = active_low
        self._debounce = timedelta(milliseconds=debounce_ms)

    def _request(self) -> Any:
        import gpiod
        from gpiod.line import Bias, Direction, Edge

        settings = gpiod.LineSettings(
            direction=Direction.INPUT,
            edge_detection=Edge.BOTH,
            # Wire the button between the pin and ground; the internal pull-up does the rest.
            bias=Bias.PULL_UP if self._active_low else Bias.PULL_DOWN,
            active_low=self._active_low,
            debounce_period=self._debounce,
        )
        try:
            return gpiod.request_lines(
                self._chip, consumer="lucy-button", config={self._line: settings}
            )
        except OSError as error:
            raise ConfigError(
                f"cannot claim button GPIO {self._line} on {self._chip}: {error}"
            ) from error

    async def held(self, seconds: float) -> bool:
        """True if the button is pressed now and stays pressed for ``seconds`` more.

        The wake-up gesture for setup mode: a grown-up holds Lucy's wing while she starts.
        """
        import gpiod

        request = self._request()
        loop = asyncio.get_running_loop()
        released = asyncio.Event()

        def on_readable() -> None:
            for event in request.read_edge_events():
                if event.event_type is event.Type.FALLING_EDGE:
                    released.set()

        try:
            if request.get_value(self._line) != gpiod.line.Value.ACTIVE:
                return False
            loop.add_reader(request.fd, on_readable)
            try:
                async with asyncio.timeout(seconds):
                    await released.wait()
            except TimeoutError:
                return True
            finally:
                loop.remove_reader(request.fd)
            return False
        finally:
            request.release()

    async def events(self) -> AsyncIterator[ButtonEvent]:
        import gpiod

        request = self._request()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ButtonEvent] = asyncio.Queue()
        # Logical value (active_low already applied): True means pressed.
        state = {"pressed": request.get_value(self._line) == gpiod.line.Value.ACTIVE}
        if state["pressed"]:
            log.warning("button is held at start-up; waiting for it to be released")

        def on_readable() -> None:
            for event in request.read_edge_events():
                pressed = event.event_type is event.Type.RISING_EDGE
                if pressed != state["pressed"]:  # drop duplicate edges that survive debounce
                    state["pressed"] = pressed
                    queue.put_nowait(ButtonEvent(pressed, event.timestamp_ns / 1e9))

        loop.add_reader(request.fd, on_readable)
        log.info("listening for the button on %s line %d", self._chip, self._line)
        try:
            while True:
                yield await queue.get()
        finally:
            loop.remove_reader(request.fd)
            request.release()


class GpioOutput:
    """A single output line (an LED behind a resistor, a motor driver input), starting off."""

    def __init__(
        self, chip: str, line: int, *, active_low: bool = False, consumer: str = "lucy-led"
    ) -> None:
        import gpiod
        from gpiod.line import Direction, Value

        self._active, self._inactive = Value.ACTIVE, Value.INACTIVE
        self._line = line
        try:
            self._request: Any = gpiod.request_lines(
                chip,
                consumer=consumer,
                config={
                    line: gpiod.LineSettings(
                        direction=Direction.OUTPUT,
                        active_low=active_low,
                        output_value=Value.INACTIVE,
                    )
                },
            )
        except OSError as error:
            raise ConfigError(
                f"cannot claim GPIO {line} on {chip} ({consumer}): {error}"
            ) from error

    def set(self, on: bool) -> None:
        self._request.set_value(self._line, self._active if on else self._inactive)

    def close(self) -> None:
        self.set(False)
        self._request.release()


_BLINK_PERIOD_S = {Status.THINKING: 0.25, Status.ERROR: 0.1}


class LedIndicator:
    """Shows Lucy's status on one LED (e.g. hidden in an eye) using event-loop timers.

    off: idle  ·  solid: listening  ·  slow blink: thinking  ·  solid: speaking
    fast blink: something went wrong
    """

    def __init__(self, led: GpioOutput) -> None:
        self._led = led
        self._blinker: asyncio.Task[None] | None = None

    def show(self, status: Status) -> None:
        if self._blinker is not None:
            self._blinker.cancel()
            self._blinker = None
        period = _BLINK_PERIOD_S.get(status)
        if period is not None:
            self._blinker = asyncio.get_running_loop().create_task(self._blink(period))
        else:
            self._led.set(status in (Status.LISTENING, Status.SPEAKING))

    async def _blink(self, period: float) -> None:
        on = True
        while True:
            self._led.set(on)
            on = not on
            await asyncio.sleep(period)

    def close(self) -> None:
        if self._blinker is not None:
            self._blinker.cancel()
        self._led.close()
