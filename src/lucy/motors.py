"""Assembling Lucy's moving body: the DRV8833 motor driver, and the Puppeteer that moves it
in time with everything she says (:mod:`lucy.core.motion`).

The motors are optional. If they are turned off, or cannot be opened (no PWM overlay, a
line in use), Lucy keeps still and goes on talking: a toy that cannot move is far better
than a toy that cannot talk.
"""

import asyncio
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack

from lucy.adapters.motors import Drv8833, open_drv8833
from lucy.config import Config
from lucy.core.errors import ConfigError
from lucy.core.motion import MotionSettings, Puppeteer
from lucy.core.ports import Motors
from lucy.core.types import Motor

log = logging.getLogger(__name__)


def open_motors(config: Config) -> Drv8833:
    """Claim the motor driver's PWM channels and GPIO lines, with both motors still.

    Blocking: it may wait a moment for udev to make a newly exported PWM channel writable.
    """
    settings = config.motors
    return open_drv8833(
        pwm_chip=settings.pwm_chip,
        frequency_hz=settings.pwm_frequency_hz,
        gpio_chip=settings.gpio_chip,
        channels={
            Motor.BEAK: (settings.beak_pwm_channel, settings.beak_direction_line),
            Motor.BODY: (settings.body_pwm_channel, settings.body_direction_line),
        },
        sleep_line=None if settings.sleep_line == -1 else settings.sleep_line,
    )


async def start_motion(config: Config, stack: AsyncExitStack) -> Puppeteer | None:
    """Lucy's Puppeteer, or None if she has no working motors. Closed with ``stack``."""
    if not config.motors.enabled:
        return None
    try:
        driver = await asyncio.to_thread(open_motors, config)
    except (ConfigError, OSError, ImportError) as error:
        log.warning("Lucy will keep still: her motors are not working (%s)", error)
        return None
    stack.callback(driver.close)
    puppeteer = Puppeteer(driver, config.motors.motion)
    stack.callback(puppeteer.close)  # runs first: stop moving, then release the lines
    return puppeteer


async def exercise(
    driver: Motors,
    motion: MotionSettings,
    *,
    repeats: int = 3,
    report: Callable[[str], None] = print,
) -> None:
    """Move each motor on its own, slowly, so the wiring and directions can be checked.

    Also used by setup mode's "Test motors" button, with settings not saved yet.
    """

    async def run(motor: Motor, power: float, seconds: float, rest_s: float = 0.5) -> None:
        driver.drive(motor, power)
        await asyncio.sleep(seconds)
        driver.drive(motor, 0.0)
        await asyncio.sleep(rest_s)

    times = {1: "once", 2: "twice"}.get(repeats, f"{repeats} times")
    report(f"beak: opening wide {times} (the spring should shut it in between)")
    for _ in range(repeats):
        await run(Motor.BEAK, motion.beak_max_power, 0.25, 0.4)
    report(f"beak: opening a little {times} (beak_min_power: it should only just move)")
    for _ in range(repeats):
        await run(Motor.BEAK, motion.beak_min_power, 0.25, 0.4)
    report("body: forward")
    await run(Motor.BODY, motion.body_power, 0.3)
    if motion.body_two_way:
        report("body: backward")
        await run(Motor.BODY, -motion.body_power, 0.3)
    driver.sleep()
    report("done. Swap a motor's two wires at the driver if it moved the wrong way.")
