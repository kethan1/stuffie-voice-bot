import asyncio
import socket
from pathlib import Path

import pytest

from lucy.adapters.power import CpuGovernor, PowerManager
from lucy.config import Config, ConfigError, load_config, read_api_key
from lucy.core.types import PowerMode
from lucy.platform.systemd import SystemdNotifier

REPO = Path(__file__).resolve().parent.parent
GOVERNORS = {
    PowerMode.ACTIVE: "performance",
    PowerMode.STANDBY: "ondemand",
    PowerMode.DORMANT: "powersave",
}


class TestConfig:
    def test_defaults_are_valid(self) -> None:
        config = load_config(None)
        assert config.llm.model == "openai/gpt-oss-20b"
        assert config.llm.params["reasoning_effort"] == "low"

    def test_shipped_config_is_valid(self) -> None:
        config = load_config(REPO / "deploy" / "config.toml")
        assert config.button.backend == "gpio"

    def test_overrides_and_types(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text(
            'log_level = "debug"\n'
            "[session]\nawake_s = 30\n"
            '[tts]\nvoice = "/v/x.onnx"\n'
            '[llm.params]\nreasoning_effort = "medium"\n'
            '[safety]\nextra_patterns = ["fire"]\n'
        )
        config = load_config(path)
        assert config.log_level == "debug"
        assert config.session.awake_s == 30.0
        assert isinstance(config.session.awake_s, float)
        assert config.tts.voice == Path("/v/x.onnx")
        assert config.llm.params == {"reasoning_effort": "medium"}
        assert config.safety.extra_patterns == ("fire",)

    @pytest.mark.parametrize(
        ("toml", "message"),
        [
            ("[session]\nmax_recrod_s = 3\n", "unknown setting"),
            ("[tts]\nthreads = 'four'\n", "tts.threads should be int"),
            ("[button]\nbackend = 'telepathy'\n", "button.backend"),
            ("[session]\ntap_max_s = 20\n", "tap_max_s"),
            ("session = 3\n", "must be a table"),
            ("[[oops", "not valid TOML"),
        ],
    )
    def test_rejects_bad_config(self, tmp_path: Path, toml: str, message: str) -> None:
        path = tmp_path / "c.toml"
        path.write_text(toml)
        with pytest.raises(ConfigError, match=message):
            load_config(path)

    def test_api_key_prefers_systemd_credential(self, tmp_path: Path) -> None:
        (tmp_path / "groq_api_key").write_text("from-credential\n")
        settings = Config().api
        environ = {"CREDENTIALS_DIRECTORY": str(tmp_path), "GROQ_API_KEY": "from-env"}
        assert read_api_key(settings, environ) == "from-credential"
        assert read_api_key(settings, {"GROQ_API_KEY": " from-env "}) == "from-env"
        assert read_api_key(settings, {}) == ""


def fake_sysfs(tmp_path: Path, governor: str = "ondemand") -> Path:
    for policy in ("policy0", "policy4"):
        (tmp_path / policy).mkdir()
        (tmp_path / policy / "scaling_governor").write_text(governor + "\n")
    return tmp_path


class TestPower:
    async def test_governor_follows_mode_and_restores(self, tmp_path: Path) -> None:
        root = fake_sysfs(tmp_path, "schedutil")
        governor = CpuGovernor(GOVERNORS, root)
        await governor.apply(PowerMode.ACTIVE)
        assert (root / "policy4" / "scaling_governor").read_text() == "performance"
        await governor.apply(PowerMode.DORMANT)
        assert (root / "policy0" / "scaling_governor").read_text() == "powersave"
        await governor.restore()
        assert (root / "policy0" / "scaling_governor").read_text() == "schedutil"

    def test_governor_needs_cpufreq(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            CpuGovernor(GOVERNORS, tmp_path)

    async def test_manager_skips_repeats_and_drops_broken_knobs(self, tmp_path: Path) -> None:
        applied: list[PowerMode] = []

        class Recorder:
            name = "recorder"

            async def apply(self, mode: PowerMode) -> None:
                applied.append(mode)

            async def restore(self) -> None:
                pass

        class Broken(Recorder):
            name = "broken"

            async def apply(self, mode: PowerMode) -> None:
                raise PermissionError("read-only")

        manager = PowerManager([Broken(), Recorder()])
        await manager.apply(PowerMode.ACTIVE)
        await manager.apply(PowerMode.ACTIVE)
        await manager.apply(PowerMode.STANDBY)
        assert applied == [PowerMode.ACTIVE, PowerMode.STANDBY]
        await manager.restore()


class TestSystemd:
    async def test_notifies_over_the_socket(self, tmp_path: Path) -> None:
        path = str(tmp_path / "notify")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
            server.bind(path)
            notifier = SystemdNotifier({"NOTIFY_SOCKET": path, "WATCHDOG_USEC": "30000000"})
            assert notifier.enabled
            assert notifier.watchdog_interval_s == 15.0
            notifier.ready("awake")
            assert server.recv(1024) == b"READY=1\nSTATUS=awake"
            task = asyncio.create_task(notifier.watchdog())
            await asyncio.sleep(0)
            assert server.recv(1024) == b"WATCHDOG=1"
            task.cancel()

    def test_silent_without_systemd(self) -> None:
        notifier = SystemdNotifier({})
        assert not notifier.enabled
        assert notifier.watchdog_interval_s is None
        notifier.ready("nobody is listening")  # must not raise
