"""The L293D motor driver, its configuration, and its assembly, on a fake sysfs tree."""

import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import ClassVar

import pytest

from lucy.adapters.motors import HBridge, L293d, SysfsPwm, open_l293d
from lucy.config import Config, ConfigError, load_config
from lucy.core.motion import Puppeteer
from lucy.core.types import Motor
from lucy.motors import start_motion


class FakePwm:
    def __init__(self, events: list[str] | None = None) -> None:
        self.duties: list[float] = []
        self.closed = False
        self._events = events

    def set(self, duty: float) -> None:
        self.duties.append(duty)
        if self._events is not None:
            self._events.append(f"EN {duty:g}")

    def close(self) -> None:
        self.closed = True


class FakeLine:
    """Stands in for GpioOutput; records each level it is set to."""

    claimed: ClassVar[list["FakeLine"]] = []

    def __init__(
        self,
        chip: str = "",
        line: int = 0,
        *,
        consumer: str = "",
        events: list[str] | None = None,
    ) -> None:
        self.line = line
        self.consumer = consumer
        self.levels: list[bool] = []
        self.closed = False
        self._events = events
        FakeLine.claimed.append(self)

    def set(self, on: bool) -> None:
        self.levels.append(on)
        if self._events is not None:
            self._events.append(f"{self.consumer} {'high' if on else 'low'}")

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
    def bridge(self) -> tuple[HBridge, list[str]]:
        events: list[str] = []
        bridge = HBridge(
            FakePwm(events),
            FakeLine(consumer="1A", events=events),
            FakeLine(consumer="2A", events=events),
        )
        return bridge, events

    def test_starts_with_the_outputs_off_and_both_inputs_low(self) -> None:
        _, events = self.bridge()
        assert events == ["EN 0", "1A low", "2A low"]

    def test_follows_the_l293d_truth_table(self) -> None:
        bridge, events = self.bridge()
        events.clear()
        bridge.drive(0.5)  # forward: 1A high, 2A low, the enable pin at the power
        assert events[-3:] == ["1A high", "2A low", "EN 0.5"]
        events.clear()
        bridge.drive(0.3)  # the same way: only the enable pin changes
        bridge.drive(0.0)  # coast: the enable pin low, the inputs kept
        assert events == ["EN 0.3", "EN 0"]
        events.clear()
        bridge.drive(-0.25)  # reverse: 1A low, 2A high, the enable pin at |power|
        assert events[-3:] == ["1A low", "2A high", "EN 0.25"]
        events.clear()
        bridge.drive(-3.0)  # clamped to full reverse
        assert events == ["EN 1"]

    def test_turns_the_outputs_off_before_changing_direction(self) -> None:
        # Both inputs at the same level with the enable pin high would brake the motor.
        bridge, events = self.bridge()
        bridge.drive(0.8)
        events.clear()
        bridge.drive(-0.8)
        assert events == ["EN 0", "1A low", "2A high", "EN 0.8"]

    def test_close_coasts_and_releases(self) -> None:
        pwm, forward, backward = FakePwm(), FakeLine(), FakeLine()
        bridge = HBridge(pwm, forward, backward)
        bridge.drive(-0.5)
        bridge.close()
        assert pwm.duties[-1] == 0.0
        assert (forward.levels[-1], backward.levels[-1]) == (False, False)
        assert pwm.closed
        assert forward.closed
        assert backward.closed


class TestL293d:
    def driver(self) -> tuple[L293d, dict[Motor, FakePwm]]:
        pwms = {motor: FakePwm() for motor in Motor}
        bridges = {motor: HBridge(pwms[motor], FakeLine(), FakeLine()) for motor in Motor}
        return L293d(bridges), pwms

    def test_moves_each_motor_and_lets_both_coast_when_told(self) -> None:
        driver, pwms = self.driver()
        driver.drive(Motor.BEAK, 0.6)
        driver.drive(Motor.BODY, -0.4)
        assert pwms[Motor.BEAK].duties[-1] == 0.6
        assert pwms[Motor.BODY].duties[-1] == 0.4
        driver.sleep()
        assert all(pwm.duties[-1] == 0.0 for pwm in pwms.values())

    def test_close_releases_everything(self) -> None:
        driver, pwms = self.driver()
        driver.drive(Motor.BEAK, 1.0)
        driver.close()
        assert all(pwm.duties[-1] == 0.0 and pwm.closed for pwm in pwms.values())


@pytest.fixture
def fake_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    import lucy.adapters.gpio

    FakeLine.claimed = []
    monkeypatch.setattr(lucy.adapters.gpio, "GpioOutput", FakeLine)


@pytest.mark.usefixtures("fake_lines")
class TestOpen:
    def open(self, chip: Path) -> L293d:
        return open_l293d(
            pwm_chip=chip,
            frequency_hz=20_000,
            gpio_chip="/dev/gpiochip0",
            channels={Motor.BEAK: (0, 22, 23), Motor.BODY: (1, 24, 25)},
        )

    def test_claims_both_inputs_of_each_motor_with_both_still(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path)
        driver = self.open(chip)
        assert {(line.line, line.consumer) for line in FakeLine.claimed} == {
            (22, "lucy-beak-forward"),
            (23, "lucy-beak-backward"),
            (24, "lucy-body-forward"),
            (25, "lucy-body-backward"),
        }
        assert all(line.levels[-1] is False for line in FakeLine.claimed)
        assert read(chip, 0, "duty_cycle") == read(chip, 1, "duty_cycle") == "0"
        driver.close()
        assert read(chip, 0, "enable") == read(chip, 1, "enable") == "0"
        assert all(line.closed for line in FakeLine.claimed)

    def test_drives_the_enable_pin_and_the_inputs(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path)
        driver = self.open(chip)
        lines = {line.consumer: line for line in FakeLine.claimed}
        driver.drive(Motor.BODY, -0.5)
        assert read(chip, 1, "duty_cycle") == "25000"
        assert lines["lucy-body-forward"].levels[-1] is False
        assert lines["lucy-body-backward"].levels[-1] is True
        driver.sleep()
        assert read(chip, 1, "duty_cycle") == "0"  # outputs off: the motor coasts
        driver.close()

    def test_a_failure_part_way_releases_what_was_claimed(self, tmp_path: Path) -> None:
        chip = pwm_chip(tmp_path, channels=1)  # no channel 1 for the body
        with pytest.raises(ConfigError, match="no channel 1"):
            self.open(chip)
        assert [line.line for line in FakeLine.claimed] == [22, 23]
        assert all(line.closed for line in FakeLine.claimed)
        assert read(chip, 0, "enable") == "0"


class TestMotorConfig:
    def test_off_until_installed_with_the_documented_wiring(self) -> None:
        repo = Path(__file__).resolve().parent.parent
        assert not load_config(None).motors.enabled
        assert not load_config(repo / "dev.toml").motors.enabled
        shipped = load_config(repo / "deploy" / "config.toml").motors
        assert not shipped.enabled  # `install.sh --motors` turns them on
        assert (shipped.beak_pwm_channel, shipped.body_pwm_channel) == (0, 1)
        assert shipped.lines() == {
            "beak_forward_line": 22,
            "beak_backward_line": 23,
            "body_forward_line": 24,
            "body_backward_line": 25,
        }

    @pytest.mark.parametrize(
        ("toml", "message"),
        [
            ("[motors]\nbody_pwm_channel = 0\n", "must differ"),
            ("[motors]\nbeak_forward_line = 17\n", "already used by button.line"),
            ("[motors]\nbody_backward_line = 22\n", "already used by motors.beak_forward_line"),
            ("[motors]\nbeak_backward_line = -1\n", "must be a GPIO line number"),
            ("[motors]\npwm_frequency_hz = 100\n", "pwm_frequency_hz"),
            ("[motors]\nsleep_line = 24\n", "unknown setting"),  # the old DRV8833's
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


class TestDoctor:
    def test_checks_all_four_inputs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from lucy.tools import doctor

        checked: list[tuple[int, str]] = []
        monkeypatch.setattr(
            doctor, "_gpio_line", lambda chip, line, label: checked.append((line, label)) or ""
        )
        path = tmp_path / "c.toml"
        path.write_text(f'[motors]\nenabled = true\npwm_chip = "{pwm_chip(tmp_path)}"\n')
        report = doctor.motors(load_config(path))
        assert report.startswith("L293D: beak on PWM 0 + GPIO22/GPIO23, body on PWM 1")
        assert checked == [
            (22, "beak forward"),
            (23, "beak backward"),
            (24, "body forward"),
            (25, "body backward"),
        ]


@pytest.mark.usefixtures("fake_lines")
class TestStartMotion:
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
        assert len(FakeLine.claimed) == 4
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
