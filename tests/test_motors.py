"""The DRV8833 motor adapter, its configuration, and its assembly, on a fake sysfs tree."""

import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import ClassVar

import pytest

from lucy.adapters.motors import Drv8833, HBridge, SysfsPwm, open_drv8833
from lucy.config import Config, ConfigError, load_config
from lucy.core.motion import Puppeteer
from lucy.core.types import Motor
from lucy.motors import start_motion


class FakePwm:
    def __init__(self) -> None:
        self.duties: list[float] = []
        self.closed = False

    def set(self, duty: float) -> None:
        self.duties.append(duty)

    def close(self) -> None:
        self.closed = True


class FakeLine:
    """Stands in for GpioOutput; records each level it is set to."""

    claimed: ClassVar[list["FakeLine"]] = []

    def __init__(self, chip: str = "", line: int = 0, *, consumer: str = "") -> None:
        self.line = line
        self.consumer = consumer
        self.levels: list[bool] = []
        self.closed = False
        FakeLine.claimed.append(self)

    def set(self, on: bool) -> None:
        self.levels.append(on)

    def close(self) -> None:
        self.closed = True


def pwm_chip(root: Path, channels: int = 2, *, exported: bool = True) -> Path:
    """A sysfs PWM chip, with each channel's files as the kernel shows them after export."""
    chip = root / "pwmchip0"
    chip.mkdir()
    (chip / "export").write_text("")
    (chip / "npwm").write_text(f"{channels}\n")
    if exported:
        for channel in range(channels):
            export_channel(chip, channel)
    return chip


def export_channel(chip: Path, channel: int) -> None:
    directory = chip / f"pwm{channel}"
    directory.mkdir()
    for name in ("period", "duty_cycle", "enable"):
        (directory / name).write_text("0\n")


def read(chip: Path, channel: int, name: str) -> str:
    return (chip / f"pwm{channel}" / name).read_text().strip()


class TestSysfsPwm:
    def test_starts_silent_at_the_frequency_and_sets_duty(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path)
        pwm = SysfsPwm.open(chip, 1, 20_000)
        assert read(chip, 1, "period") == "50000"  # 20 kHz, in nanoseconds
        assert read(chip, 1, "duty_cycle") == "0"
        assert read(chip, 1, "enable") == "1"
        pwm.set(0.25)
        assert read(chip, 1, "duty_cycle") == "12500"
        pwm.set(7.0)
        assert read(chip, 1, "duty_cycle") == "50000"  # clamped
        pwm.close()
        assert read(chip, 1, "duty_cycle") == "0"
        assert read(chip, 1, "enable") == "0"

    def test_clears_a_duty_cycle_left_longer_than_the_new_period(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path)
        (chip / "pwm0" / "period").write_text("1000000\n")  # left by a crash at 1 kHz
        (chip / "pwm0" / "duty_cycle").write_text("800000\n")
        SysfsPwm.open(chip, 0, 20_000)
        assert read(chip, 0, "duty_cycle") == "0"
        assert read(chip, 0, "period") == "50000"

    def test_exports_the_channel_and_waits_for_udev(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path, exported=False)
        # The kernel makes pwm0 at once; udev makes it writable for the gpio group a bit later.
        later = threading.Timer(0.05, export_channel, (chip, 0))
        later.start()
        started = time.monotonic()
        SysfsPwm.open(chip, 0, 20_000)
        later.join()
        assert (chip / "export").read_text() == "0"
        assert time.monotonic() - started >= 0.04
        assert read(chip, 0, "enable") == "1"

    def test_gives_up_with_a_clear_message(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path, exported=False)
        with pytest.raises(ConfigError, match="gpio' group"):
            SysfsPwm.open(chip, 0, 20_000, wait_s=0.05)

    def test_says_how_to_add_the_overlay(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="--motors"):
            SysfsPwm.open(tmp_path / "pwmchip9", 0, 20_000)


class TestHBridge:
    def test_follows_the_drv8833_truth_table(self) -> None:
        pwm, direction = FakePwm(), FakeLine()
        bridge = HBridge(pwm, direction)
        assert (direction.levels, pwm.duties) == ([False], [0.0])  # starts coasting
        bridge.drive(0.5)  # forward: xIN2 low, xIN1 at the power
        bridge.drive(-0.25)  # reverse: xIN2 high, xIN1 low for a quarter of each period
        bridge.drive(0.0)  # coast: both low
        bridge.drive(-3.0)  # clamped to full reverse: xIN1 always low
        assert direction.levels == [False, True, False, True]
        assert pwm.duties == [0.0, 0.5, 0.75, 0.0, 0.0]

    def test_close_coasts_and_releases(self) -> None:
        pwm, direction = FakePwm(), FakeLine()
        bridge = HBridge(pwm, direction)
        bridge.drive(-0.5)
        bridge.close()
        assert direction.levels[-1] is False
        assert pwm.duties[-1] == 0.0
        assert pwm.closed
        assert direction.closed


class TestDrv8833:
    def driver(self, sleep: FakeLine | None) -> tuple[Drv8833, dict[Motor, FakePwm]]:
        pwms = {motor: FakePwm() for motor in Motor}
        bridges = {motor: HBridge(pwms[motor], FakeLine()) for motor in Motor}
        return Drv8833(bridges, sleep), pwms

    def test_wakes_to_move_and_sleeps_when_told(self) -> None:
        sleep = FakeLine()
        driver, pwms = self.driver(sleep)
        assert sleep.levels == [False]  # starts asleep
        driver.drive(Motor.BEAK, 0.6)
        driver.drive(Motor.BEAK, 0.4)
        driver.drive(Motor.BODY, 0.0)
        assert sleep.levels == [False, True]  # woken once
        assert pwms[Motor.BEAK].duties[-1] == 0.4
        driver.sleep()
        assert sleep.levels[-1] is False
        assert all(pwm.duties[-1] == 0.0 for pwm in pwms.values())

    def test_works_without_a_sleep_line(self) -> None:
        driver, pwms = self.driver(None)
        driver.drive(Motor.BODY, -1.0)
        driver.close()
        assert pwms[Motor.BODY].duties[-1] == 0.0
        assert all(pwm.closed for pwm in pwms.values())

    def test_close_releases_everything(self) -> None:
        sleep = FakeLine()
        driver, pwms = self.driver(sleep)
        driver.drive(Motor.BEAK, 1.0)
        driver.close()
        assert sleep.levels[-1] is False
        assert sleep.closed
        assert all(pwm.closed for pwm in pwms.values())


class TestOpen:
    @pytest.fixture(autouse=True)
    def fake_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import lucy.adapters.gpio

        FakeLine.claimed = []
        monkeypatch.setattr(lucy.adapters.gpio, "GpioOutput", FakeLine)

    def open(self, chip: Path, *, body_channel: int = 1) -> Drv8833:
        return open_drv8833(
            pwm_chip=chip,
            frequency_hz=20_000,
            gpio_chip="/dev/gpiochip0",
            channels={Motor.BEAK: (0, 22), Motor.BODY: (body_channel, 23)},
            sleep_line=24,
        )

    def test_claims_every_line_with_both_motors_still(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path)
        driver = self.open(chip)
        assert {(line.line, line.consumer) for line in FakeLine.claimed} == {
            (22, "lucy-beak"),
            (23, "lucy-body"),
            (24, "lucy-motor-sleep"),
        }
        assert all(line.levels[-1] is False for line in FakeLine.claimed)
        driver.drive(Motor.BODY, 0.5)
        assert read(chip, 1, "duty_cycle") == "25000"
        driver.close()
        assert read(chip, 1, "enable") == "0"
        assert all(line.closed for line in FakeLine.claimed)

    def test_a_failure_part_way_releases_what_was_claimed(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path, channels=1)  # no channel 1 for the body
        with pytest.raises(ConfigError, match="no channel 1"):
            open_drv8833(
                pwm_chip=chip,
                frequency_hz=20_000,
                gpio_chip="/dev/gpiochip0",
                channels={Motor.BEAK: (0, 22), Motor.BODY: (1, 23)},
                sleep_line=None,
            )
        assert [line.line for line in FakeLine.claimed] == [22]
        assert FakeLine.claimed[0].closed
        assert read(chip, 0, "enable") == "0"


class TestMotorConfig:
    def test_off_until_installed_with_the_documented_wiring(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        assert not load_config(None).motors.enabled
        assert not load_config(repo / "dev.toml").motors.enabled
        shipped = load_config(repo / "deploy" / "config.toml").motors
        assert not shipped.enabled  # `install.sh --motors` turns them on
        assert (shipped.beak_pwm_channel, shipped.body_pwm_channel) == (0, 1)
        assert (shipped.beak_direction_line, shipped.body_direction_line) == (22, 23)
        assert shipped.sleep_line == 24

    @pytest.mark.parametrize(
        ("toml", "message"),
        [
            ("[motors]\nbody_pwm_channel = 0\n", "must differ"),
            ("[motors]\nbeak_direction_line = 17\n", "already used by button.line"),
            ("[motors]\nsleep_line = 23\n", "already used by motors.body_direction_line"),
            ("[motors]\npwm_frequency_hz = 100\n", "pwm_frequency_hz"),
            ("[motors.motion]\nbeak_max_power = 1.5\n", "beak_max_power must be between"),
            ("[motors.motion]\nbeak_min_power = 0.9\n", "must not be above"),
            ("[motors.motion]\nlead_s = 1\n", "lead_s"),
            ("[motors.motion]\nmax_hold_s = 0\n", "max_hold_s"),
        ],
    )
    def test_rejects_bad_settings(self, tmp_path: Path, toml: str, message: str) -> None:
        path = tmp_path / "c.toml"
        path.write_text("[motors]\nenabled = true\n" + toml.replace("[motors]\n", ""))
        with pytest.raises(ConfigError, match=message):
            load_config(path)

    def test_settings_are_only_checked_when_enabled(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text("[motors]\nbody_pwm_channel = 0\n")
        assert load_config(path).motors.body_pwm_channel == 0


class TestStartMotion:
    @pytest.fixture(autouse=True)
    def fake_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import lucy.adapters.gpio

        FakeLine.claimed = []
        monkeypatch.setattr(lucy.adapters.gpio, "GpioOutput", FakeLine)

    def config(self, tmp_path: Path, chip: Path) -> Config:
        path = tmp_path / "c.toml"
        path.write_text(f'[motors]\nenabled = true\npwm_chip = "{chip}"\n')
        return load_config(path)

    async def test_lucy_moves_with_working_motors(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path)
        async with AsyncExitStack() as stack:
            puppeteer = await start_motion(self.config(tmp_path, chip), stack)
            assert isinstance(puppeteer, Puppeteer)
            assert read(chip, 0, "enable") == "1"
        assert read(chip, 0, "enable") == "0"  # closed with the stack
        assert all(line.closed for line in FakeLine.claimed)

    async def test_lucy_keeps_still_and_talking_without_them(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        async with AsyncExitStack() as stack:
            puppeteer = await start_motion(self.config(tmp_path, tmp_path / "missing"), stack)
        assert puppeteer is None
        assert "Lucy will keep still" in caplog.text

    async def test_nothing_is_claimed_when_turned_off(self) -> None:
        async with AsyncExitStack() as stack:
            assert await start_motion(load_config(None), stack) is None
        assert FakeLine.claimed == []
