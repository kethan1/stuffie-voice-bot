"""Setup mode: Wi-Fi through NetworkManager, the settings, the card, and the page end to end."""

import asyncio
import re
import socket
import stat
import typing
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from lucy.app import SETUP_REQUEST, _hand_over_to_setup, request_setup
from lucy.config import Config, ConfigError, load_config
from lucy.core.motion import MotionSettings
from lucy.core.phrases import Phrase
from lucy.core.safety import BLOCKED_WORDS, ReplyFilter
from lucy.persona import LUCY, SetupLine
from lucy.setup.hotspot import Hotspot, card_html, load_hotspot, new_hotspot
from lucy.setup.http import Request as HttpRequest
from lucy.setup.http import Response as HttpResponse
from lucy.setup.http import read_request, start_server
from lucy.setup.mode import SetupMode, speakable_network, start_if_offline
from lucy.setup.network import (
    HOTSPOT_UUID,
    Completed,
    DemoNetwork,
    JoinProblem,
    NetworkError,
    NetworkManager,
    Security,
    WifiLogin,
    hotspot_keyfile,
    join_problem,
    network_keyfile,
    parse_scan,
    security_of,
    split_terse,
)
from lucy.setup.settings import FIELD, SECTION, SettingsStore, parse_section
from tests.fakes import FakeButton, until

REPO = Path(__file__).resolve().parent.parent
HOTSPOT = Hotspot("Lucy-Setup-4821", "abcd-efgh-jkmn", address="127.0.0.1")


def read(path: Path) -> str:
    """Small files, read from async tests (blocking briefly is fine in a test)."""
    return path.read_text()


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def keyfiles(directory: Path) -> list[Path]:
    return sorted(directory.glob("lucy-*.nmconnection"))


# -- NetworkManager ----------------------------------------------------------------------------


class FakeNmcli:
    """Answers nmcli and nm-online like NetworkManager would, and records every command."""

    def __init__(self, answers: dict[str, Completed] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    async def __call__(self, argv: Sequence[str]) -> Completed:
        self.calls.append(list(argv))
        line = " ".join(argv)
        for fragment, answer in self.answers.items():
            if fragment in line:
                return answer
        if "connection show" in line and "-g" in argv:
            return Completed(0, "School Wi-Fi\n")
        if "connection show" in line:
            return Completed(
                0,
                "old-uuid:802-11-wireless:School Wi-Fi\n"
                f"{HOTSPOT_UUID}:802-11-wireless:Lucy setup hotspot\n"
                "eth-uuid:802-3-ethernet:Wired\n",
            )
        return Completed(0)

    def ran(self, fragment: str) -> bool:
        return any(fragment in " ".join(call) for call in self.calls)


class TestNmcliOutput:
    def test_terse_lines_unescape_colons_and_backslashes(self) -> None:
        assert split_terse(r"Cafe\: 2nd floor:72:WPA2") == ["Cafe: 2nd floor", "72", "WPA2"]
        assert split_terse(r"a\\b::") == ["a\\b", "", ""]

    def test_scan_keeps_each_name_once_strongest_first_and_skips_hidden(self) -> None:
        output = "Home:40:WPA2\nSchool:80:WPA2 802.1X\n:90:WPA2\nHome:65:WPA2\nCafe:65:\n"
        networks = parse_scan(output)
        assert [(n.ssid, n.signal) for n in networks] == [
            ("School", 80),
            ("Cafe", 65),
            ("Home", 65),
        ]
        assert networks[0].security is Security.ENTERPRISE
        assert networks[1].security is Security.OPEN

    @pytest.mark.parametrize(
        ("flags", "security"),
        [
            ("", Security.OPEN),
            ("--", Security.OPEN),
            ("OWE", Security.OPEN),
            ("WPA2", Security.PERSONAL),
            ("WPA1 WPA2", Security.PERSONAL),
            ("WPA2 WPA3", Security.PERSONAL),
            ("WPA3", Security.WPA3),
            ("WPA2 802.1X", Security.ENTERPRISE),
        ],
    )
    def test_security(self, flags: str, security: Security) -> None:
        assert security_of(flags) is security

    @pytest.mark.parametrize(
        ("message", "problem"),
        [
            (
                "Error: Connection activation failed: Secrets were required, but not provided",
                JoinProblem.WRONG_PASSWORD,
            ),
            (
                "Error: Connection activation failed: 802.1X supplicant failed",
                JoinProblem.WRONG_PASSWORD,
            ),
            (
                "Error: Connection activation failed: The Wi-Fi network could not be found",
                JoinProblem.NOT_FOUND,
            ),
            (
                "Error: Connection activation failed: IP configuration could not be reserved",
                JoinProblem.NO_ADDRESS,
            ),
            ("Error: Timeout expired (45 seconds)", JoinProblem.FAILED),
        ],
    )
    def test_why_a_network_failed(self, message: str, problem: JoinProblem) -> None:
        assert join_problem(message) is problem


class TestKeyfiles:
    def test_a_password_network(self) -> None:
        text = network_keyfile(WifiLogin("Home", Security.PERSONAL, "hunter22!"), "u-1")
        assert "uuid=u-1" in text
        assert "autoconnect=false" in text  # turned on only once it works
        assert "ssid=72;111;109;101;" in text  # "Home", byte by byte
        assert "key-mgmt=wpa-psk" in text
        assert "psk=hunter22!" in text
        assert "cloned-mac-address=permanent" in text
        assert "hidden" not in text
        assert "802-1x" not in text

    def test_wpa3_open_hidden_and_enterprise(self) -> None:
        assert "key-mgmt=sae" in network_keyfile(WifiLogin("Lab", Security.WPA3, "x" * 8), "u")
        open_text = network_keyfile(WifiLogin("Cafe", Security.OPEN), "u")
        assert "wifi-security" not in open_text
        assert "hidden=true" in network_keyfile(
            WifiLogin("Secret", Security.PERSONAL, "x" * 8, hidden=True), "u"
        )
        school = network_keyfile(WifiLogin("School", Security.ENTERPRISE, "pw", "kid.teacher"), "u")
        assert "key-mgmt=wpa-eap" in school
        assert "eap=peap;" in school
        assert "identity=kid.teacher" in school
        assert "password=pw" in school
        assert "phase2-auth=mschapv2" in school

    def test_any_name_and_password_survive_exactly(self) -> None:
        login = WifiLogin("Café;=[x]", Security.PERSONAL, "  back\\slash;# pw\n")
        text = network_keyfile(login, "u")
        assert "ssid=" + "".join(f"{b};" for b in "Café;=[x]".encode()) in text
        assert "psk=\\s\\sback\\\\slash;# pw\\n" in text
        assert "id=Café;=[x]" in text

    def test_hotspot(self) -> None:
        text = hotspot_keyfile(HOTSPOT, "wlan0")
        assert f"uuid={HOTSPOT_UUID}" in text
        assert "mode=ap" in text
        assert "interface-name=wlan0" in text
        assert "method=shared" in text
        assert "address1=127.0.0.1/24" in text
        assert "psk=abcd-efgh-jkmn" in text
        assert "pmf=1" in text
        assert "autoconnect=false" in text


class TestWifiLogin:
    @pytest.mark.parametrize(
        ("login", "problem"),
        [
            (WifiLogin("", Security.OPEN), "1 to 32"),
            (WifiLogin("x" * 33, Security.OPEN), "1 to 32"),
            (WifiLogin("Home", Security.PERSONAL, ""), "needs a password"),
            (WifiLogin("Home", Security.PERSONAL, "short"), "8 to 63"),
            (WifiLogin("Home", Security.PERSONAL, "x" * 64), "8 to 63"),
            (WifiLogin("School", Security.ENTERPRISE, "pw"), "username"),
            (WifiLogin("Home", Security.PERSONAL, "tab\there"), "letters"),
        ],
    )
    def test_problems(self, login: WifiLogin, problem: str) -> None:
        assert any(problem in p for p in login.problems())

    def test_good_logins(self) -> None:
        assert WifiLogin("Cafe", Security.OPEN).problems() == []
        assert WifiLogin("Home", Security.PERSONAL, "correct horse").problems() == []
        assert WifiLogin("Home", Security.PERSONAL, "ab" * 32).problems() == []  # a hex key
        assert WifiLogin("School", Security.ENTERPRISE, "pw", "me").problems() == []


class TestNetworkManager:
    def manager(self, tmp_path: Path, nmcli: FakeNmcli) -> NetworkManager:
        (tmp_path / "net" / "wlan0").mkdir(parents=True)
        (tmp_path / "net" / "wlan0" / "address").write_text("dc:a6:32:12:34:56\n")
        return NetworkManager("wlan0", connections=tmp_path, runner=nmcli, sysfs=tmp_path / "net")

    async def test_joining_keeps_the_network_and_drops_older_copies(self, tmp_path: Path) -> None:
        nmcli = FakeNmcli()
        manager = self.manager(tmp_path, nmcli)
        result = await manager.join(WifiLogin("School Wi-Fi", Security.PERSONAL, "s3cret-pass"))
        assert result.ok
        [keyfile] = keyfiles(tmp_path)
        assert mode_of(keyfile) == 0o600
        new = keyfile.stem.removeprefix("lucy-")
        assert nmcli.ran(f"connection load {keyfile}")
        assert nmcli.ran(f"--wait 45 connection up uuid {new}")
        assert nmcli.ran(f"connection modify uuid {new} connection.autoconnect yes")
        assert nmcli.ran("connection delete uuid old-uuid")  # same name, older password
        assert not nmcli.ran(f"delete uuid {HOTSPOT_UUID}")
        assert not any("s3cret-pass" in " ".join(call) for call in nmcli.calls)
        assert manager.mac_address == "dc:a6:32:12:34:56"

    async def test_a_failed_network_is_removed_again(self, tmp_path: Path) -> None:
        refused = Completed(4, "", "Error: Connection activation failed: Secrets were required")
        nmcli = FakeNmcli({"connection up uuid": refused})
        manager = self.manager(tmp_path, nmcli)
        result = await manager.join(WifiLogin("School Wi-Fi", Security.PERSONAL, "wrong-pass"))
        assert result.problem is JoinProblem.WRONG_PASSWORD
        new = nmcli.calls[1][-1]
        assert nmcli.ran(f"connection delete uuid {new}")
        assert not nmcli.ran("delete uuid old-uuid")  # what worked before is untouched
        assert not nmcli.ran("connection.autoconnect yes")

    async def test_hotspot(self, tmp_path: Path) -> None:
        nmcli = FakeNmcli()
        manager = self.manager(tmp_path, nmcli)
        await manager.start_hotspot(HOTSPOT)
        assert (tmp_path / "lucy-setup-hotspot.nmconnection").is_file()
        assert nmcli.ran(f"connection up uuid {HOTSPOT_UUID}")
        await manager.stop_hotspot()
        assert nmcli.ran(f"connection down uuid {HOTSPOT_UUID}")
        failing = self.manager(tmp_path / "b", FakeNmcli({"connection up": Completed(1, "", "no")}))
        with pytest.raises(NetworkError, match="no"):
            await failing.start_hotspot(HOTSPOT)

    async def test_saved_networks_leave_out_the_hotspot_and_wired(self, tmp_path: Path) -> None:
        saved = await self.manager(tmp_path, FakeNmcli()).saved()
        assert [(n.uuid, n.ssid) for n in saved] == [("old-uuid", "School Wi-Fi")]

    async def test_scan_falls_back_to_the_last_scan(self, tmp_path: Path) -> None:
        nmcli = FakeNmcli(
            {
                "--rescan yes": Completed(1, "", "Scanning not allowed"),
                "--rescan no": Completed(0, "Home:50:WPA2\n"),
            }
        )
        assert [n.ssid for n in await self.manager(tmp_path, nmcli).scan()] == ["Home"]

    async def test_the_hotspot_cannot_be_forgotten(self, tmp_path: Path) -> None:
        with pytest.raises(NetworkError):
            await self.manager(tmp_path, FakeNmcli()).forget(HOTSPOT_UUID)

    async def test_wait_online(self, tmp_path: Path) -> None:
        nmcli = FakeNmcli({"nm-online": Completed(1)})
        assert not await self.manager(tmp_path, nmcli).wait_online(90)
        assert nmcli.calls == [["nm-online", "--quiet", "--timeout", "90"]]


# -- the hotspot's name and password, and its card ---------------------------------------------


class TestHotspot:
    def test_made_once_root_only_then_read_back(self, tmp_path: Path) -> None:
        path = tmp_path / "setup-hotspot.toml"
        made = load_hotspot(path, create=True)
        assert re.fullmatch(r"Lucy-Setup-\d{4}", made.ssid)
        assert re.fullmatch(r"[a-z2-9]{4}-[a-z2-9]{4}-[a-z2-9]{4}", made.password)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert load_hotspot(path) == made
        assert load_hotspot(path, create=True) == made

    def test_missing_or_broken(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="setup-card"):
            load_hotspot(tmp_path / "missing.toml")
        bad = tmp_path / "bad.toml"
        bad.write_text('ssid = "x"\npassword = "short"\n')
        with pytest.raises(ConfigError, match="8-63"):
            load_hotspot(bad)

    def test_spoken_name_and_card(self) -> None:
        assert HOTSPOT.spoken_name == "Lucy Setup 4 8 2 1"
        hotspot = Hotspot("Lucy-<Setup>", "p&ssword-1")
        card = card_html(hotspot)
        assert "Lucy-&lt;Setup&gt;" in card
        assert "p&amp;ssword-1" in card
        assert "<svg" in card
        assert new_hotspot() != new_hotspot()


# -- settings ----------------------------------------------------------------------------------


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text((REPO / "deploy" / "config.toml").read_text())
    path.chmod(0o644)
    return path


class TestSettingsStore:
    def test_every_field_on_the_page_is_a_real_setting(self) -> None:
        config = Config()
        hints: dict[type, dict[str, typing.Any]] = {}
        for path, field in FIELD.items():
            *parents, name = path.split(".")
            owner: typing.Any = config
            for parent in parents:
                owner = getattr(owner, parent)
            kind = hints.setdefault(type(owner), typing.get_type_hints(type(owner)))[name]
            assert kind is field.kind, path
            if field.kind in (int, float):
                assert field.low <= getattr(owner, name) <= field.high, path

    def test_update_keeps_the_comments_and_checks_first(self, config_file: Path) -> None:
        store = SettingsStore(config_file, config_file.parent / "credentials")
        store.update({"tts.cloud.volume": 1.2, "motors.motion.beak_max_power": 0.7})
        text = config_file.read_text()
        assert "volume = 1.2                   # match it to the on-device voice's loudness" in text
        assert "# Lucy the Goose: runtime configuration" in text
        assert store.load().motors.motion.beak_max_power == 0.7
        assert stat.S_IMODE(config_file.stat().st_mode) == 0o644
        before = config_file.read_text()
        with pytest.raises(ConfigError, match="beak_min_power"):
            store.update({"motors.enabled": True, "motors.motion.beak_min_power": 0.9})
        assert config_file.read_text() == before

    def test_new_tables_are_added(self, tmp_path: Path) -> None:
        path = tmp_path / "config.toml"
        path.write_text('log_level = "info"\n')
        store = SettingsStore(path, tmp_path)
        store.update({"motors.motion.lead_s": 0.08, "session.greet_on_start": False})
        config = load_config(path)
        assert config.motors.motion.lead_s == 0.08
        assert not config.session.greet_on_start

    def test_undo_once(self, config_file: Path) -> None:
        original = config_file.read_text()
        store = SettingsStore(config_file, config_file.parent)
        assert not store.can_undo
        store.update({"session.awake_s": 30.0})
        assert store.can_undo
        store.undo()
        assert config_file.read_text() == original
        assert not store.can_undo

    def test_replace_refuses_what_lucy_would_refuse(self, config_file: Path) -> None:
        store = SettingsStore(config_file, config_file.parent)
        for text in ("[[oops", "[session]\nmax_recrod_s = 3\n"):
            with pytest.raises(ConfigError):
                store.replace(text)
        store.replace("[session]\nawake_s = 20.0\n")
        assert store.load().session.awake_s == 20.0

    def test_keys(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "config.toml", tmp_path / "credentials")
        assert not store.key_status("groq_api_key").present
        store.set_key("groq_api_key", "  gsk_abcdefghijklmnop1234 \n")
        path = tmp_path / "credentials" / "groq_api_key"
        assert path.read_text() == "gsk_abcdefghijklmnop1234"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert store.key_status("groq_api_key").ending == "1234"
        store.set_key("groq_api_key", "")
        assert path.exists()  # lucy.service still loads the (empty) credential
        assert not store.key_status("groq_api_key").present
        with pytest.raises(ConfigError, match="API key"):
            store.set_key("groq_api_key", "not a key")

    def test_parse_section(self) -> None:
        form = {"tts.cloud.voice_id": "harper_32", "tts.cloud.volume": "1", "tts.volume": "abc"}
        form |= {
            "tts.honk_volume": "5",
            "session.awake_s": "45",
            "session.max_turns_per_hour": "60",
        }
        form |= {"turn.speech_threshold_dbfs": "-50", "tts.honks": "on"}
        changes, problems = parse_section(SECTION["lucy"], form)
        assert changes["tts.honks"] is True
        assert changes["session.greet_on_start"] is False  # unticked boxes are not sent
        assert changes["session.max_turns_per_hour"] == 60
        assert any("Backup voice loudness: type a number" in p for p in problems)
        assert any("Honk loudness: use a number from 0 to 2" in p for p in problems)


# -- the page, end to end ---------------------------------------------------------------------


class FakeBody:
    def __init__(self) -> None:
        self.said: list[str] = []
        self.voice_tries: list[dict[str, typing.Any]] = []
        self.motor_tries: list[MotionSettings] = []

    async def say(self, text: str) -> None:
        self.said.append(text)

    async def try_voice(self, *, volume: float, honks: bool, honk_volume: float) -> None:
        self.voice_tries.append({"volume": volume, "honks": honks, "honk_volume": honk_volume})

    async def try_motors(self, motion: MotionSettings) -> None:
        self.motor_tries.append(motion)

    def heard(self, line: SetupLine) -> bool:
        pattern = re.escape(LUCY.setup_lines[line]).replace(r"\{network\}", ".*")
        pattern = pattern.replace(r"\{hotspot\}", ".*")
        return any(re.fullmatch(pattern, text) for text in self.said)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Rig:
    def __init__(
        self,
        config_file: Path,
        *,
        network: DemoNetwork | None = None,
        button: FakeButton | None = None,
    ) -> None:
        self.port = free_port()
        self.network = network or DemoNetwork()
        self.body = FakeBody()
        self.store = SettingsStore(config_file, config_file.parent / "credentials")
        self.now = 0.0
        self.mode = SetupMode(
            config=self.store.load(),
            store=self.store,
            network=self.network,
            body=self.body,
            hotspot=HOTSPOT,
            address=("127.0.0.1", self.port),
            button=button,
            clock=lambda: self.now,
            handoff_s=0.0,
            wing_hold_s=0.05,
            idle_check_s=0.01,
        )
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}")
        self.task: asyncio.Task[int] | None = None
        self.token = ""

    async def __aenter__(self) -> "Rig":
        self.task = asyncio.create_task(self.mode.run())
        await until(lambda: self.body.heard(SetupLine.READY))
        page = await self.client.get("/")
        self.token = re.search(r'name="token" value="([^"]+)"', page.text).group(1)  # type: ignore[union-attr]
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.client.aclose()
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def post(self, path: str, **form: str) -> httpx.Response:
        return await self.client.post(path, data={"token": self.token, **form})

    async def finished(self) -> int:
        assert self.task is not None
        return await asyncio.wait_for(self.task, 2.0)


LUCY_FORM = {
    "section": "lucy",
    "tts.cloud.voice_id": "harper_32",
    "tts.cloud.volume": "1.1",
    "tts.volume": "0.7",
    "tts.honks": "on",
    "tts.honk_volume": "0.6",
    "session.awake_s": "60",
    "session.max_turns_per_hour": "100",
    "turn.speech_threshold_dbfs": "-52",
}
MOTOR_FORM = {
    "section": "motors",
    "motors.enabled": "on",
    "motors.motion.beak_min_power": "0.4",
    "motors.motion.beak_max_power": "0.9",
    "motors.motion.body_power": "0.5",
    "motors.motion.lead_s": "0.06",
    "motors.motion.bob_every_s": "3",
}


class TestPage:
    async def test_phones_are_sent_to_the_page_which_greets_them(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            assert rig.network.hotspot == HOTSPOT
            probe = await rig.client.get(
                "/hotspot-detect.html", headers={"Host": "captive.apple.com"}
            )
            assert probe.status_code == 302
            assert probe.headers["location"] == rig.mode.url
            page = await rig.client.get("/")
            assert page.status_code == 200
            assert "School Wi-Fi" in page.text
            assert "dc:a6:32:00:00:01" in page.text
            assert "<script" not in page.text.lower()
            assert "default-src 'none'" in page.headers["content-security-policy"]
            await until(lambda: rig.body.heard(SetupLine.HELLO))
            await rig.client.get("/")
            await asyncio.sleep(0.01)
            assert len([t for t in rig.body.said if "I see your phone" in t]) == 1  # once
            card = await rig.client.get("/card")
            assert "abcd-efgh-jkmn" in card.text

    async def test_forms_need_the_page_token(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            before = read(config_file)
            reply = await rig.client.post("/save", data={"token": "stale", **LUCY_FORM})
            assert reply.status_code == 403
            assert read(config_file) == before

    async def test_saving_settings(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            reply = await rig.post("/save", **LUCY_FORM)
            assert reply.status_code == 303
            assert reply.headers["location"] == "/?done=saved"
            config = load_config(config_file)
            assert config.tts.volume == 0.7
            assert not config.session.greet_on_start
            assert "Saved." in (await rig.client.get("/?done=saved")).text

    async def test_a_setting_lucy_would_refuse_is_explained_not_saved(
        self, config_file: Path
    ) -> None:
        async with Rig(config_file) as rig:
            before = read(config_file)
            form = MOTOR_FORM | {"motors.motion.beak_min_power": "0.95"}
            reply = await rig.post("/save", **form)
            assert reply.status_code == 400
            assert "beak_min_power must not be above beak_max_power" in reply.text
            assert 'value="0.95"' in reply.text  # what they typed is still there
            assert read(config_file) == before

    async def test_trying_voice_and_motors_saves_nothing(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            before = read(config_file)
            assert (await rig.post("/try-voice", **LUCY_FORM)).status_code == 204
            assert rig.body.voice_tries == [{"volume": 0.7, "honks": True, "honk_volume": 0.6}]
            assert (await rig.post("/try-motors", **MOTOR_FORM)).status_code == 204
            assert rig.body.motor_tries[0].beak_max_power == 0.9
            bad = MOTOR_FORM | {"motors.motion.beak_min_power": "0.95"}
            assert (await rig.post("/try-motors", **bad)).status_code == 400
            assert read(config_file) == before

    async def test_keys(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            reply = await rig.post(
                "/keys", groq="gsk_" + "a" * 40, speechify="", remove_speechify="on"
            )
            assert reply.status_code == 303
            credentials = config_file.parent / "credentials"
            assert read(credentials / "groq_api_key") == "gsk_" + "a" * 40
            assert read(credentials / "speechify_api_key") == ""
            assert (await rig.post("/keys", groq="nope")).status_code == 400

    async def test_whole_file_and_undo(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            original = read(config_file)
            reply = await rig.post("/advanced", config="[session]\r\nawake_s = 20.0\r\n")
            assert reply.status_code == 303
            assert read(config_file) == "[session]\nawake_s = 20.0\n"
            assert (await rig.post("/advanced", config="[session]\nnope = 1\n")).status_code == 400
            assert (await rig.post("/undo")).status_code == 303
            assert read(config_file) == original

    async def test_joining_a_network_ends_setup(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            reply = await rig.post("/join", network="Home", password="correct-horse")
            assert "Trying Home" in reply.text
            assert await rig.finished() == 0
            assert rig.body.heard(SetupLine.TRYING)
            assert rig.body.heard(SetupLine.JOINED)
            assert rig.network.joined[0].security is Security.PERSONAL
            assert rig.network.hotspot is None

    async def test_a_wrong_password_brings_the_hotspot_back(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            await rig.post("/join", network="Home", password="wrong-password")
            await until(lambda: rig.body.heard(SetupLine.WRONG_PASSWORD))
            assert rig.network.hotspot == HOTSPOT
            assert not rig.mode.done
            assert (await rig.client.get("/")).status_code == 200  # the page is back too

    async def test_joined_but_no_internet(self, config_file: Path) -> None:
        async with Rig(config_file, network=DemoNetwork(online=False)) as rig:
            await rig.post("/join", network="Library Guest", password="ignored for open")
            await until(lambda: rig.body.heard(SetupLine.JOINED_OFFLINE))
            assert rig.network.joined[0].password == ""
            assert rig.network.hotspot == HOTSPOT

    async def test_a_hidden_network_and_bad_answers(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            reply = await rig.post("/join", network="School Wi-Fi", password="pw")
            assert reply.status_code == 400
            assert "username" in reply.text
            reply = await rig.post("/join", network="Gone", password="whatever1")
            assert reply.status_code == 400
            await rig.post(
                "/join",
                network="__other__",
                hidden_ssid="Teachers",
                security="personal",
                password="hidden-pass",
            )
            assert await rig.finished() == 0
            assert rig.network.joined[0].hidden

    async def test_finish(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            assert "All done" in (await rig.post("/finish")).text
            assert await rig.finished() == 0
            assert rig.body.heard(SetupLine.FINISHED)
            assert rig.network.hotspot is None

    async def test_leaves_when_nothing_happens(self, config_file: Path) -> None:
        async with Rig(config_file) as rig:
            rig.now += 601
            assert await rig.finished() == 0
            assert rig.body.heard(SetupLine.IDLE)

    async def test_holding_the_wing_leaves(self, config_file: Path) -> None:
        button = FakeButton()
        async with Rig(config_file, button=button) as rig:
            button.press()
            button.release(after=0.01)  # a squeeze is not enough
            await asyncio.sleep(0.1)
            assert not rig.mode.done
            button.press()
            assert await rig.finished() == 0
            assert rig.body.heard(SetupLine.CANCELLED)

    async def test_no_hotspot(self, config_file: Path) -> None:
        class Broken(DemoNetwork):
            async def start_hotspot(self, hotspot: Hotspot) -> None:
                raise NetworkError("no Wi-Fi country set")

        rig = Rig(config_file, network=Broken())
        assert await rig.mode.run() == 1
        assert rig.body.heard(SetupLine.NO_HOTSPOT)
        await rig.client.aclose()


class TestSpeech:
    def test_every_setup_line_is_safe_to_say(self) -> None:
        screen = ReplyFilter()
        for line in SetupLine:
            text = LUCY.setup_lines[line].format(hotspot="Lucy Setup 4 8 2 1", network="Home")
            assert screen.screen(text).allowed, line
        assert LUCY.lines[Phrase.SETUP]

    def test_network_names_she_says(self) -> None:
        screen = ReplyFilter()
        assert speakable_network("School_WiFi-5G", screen) == "School WiFi 5G"
        assert speakable_network(BLOCKED_WORDS[0], screen) == "that network"
        assert speakable_network("\x07", screen) == "that network"


class TestStarting:
    async def test_only_when_offline(self) -> None:
        started: list[bool] = []

        async def start() -> int:
            started.append(True)
            return 0

        config = Config()
        assert await start_if_offline(config, DemoNetwork(online=True), start) == 0
        assert not started
        await start_if_offline(config, DemoNetwork(online=False), start)
        assert started == [True]
        off = load_config(None).__class__(setup=type(config.setup)(when_offline=False))
        await start_if_offline(off, DemoNetwork(online=False), start)
        assert started == [True]

    def test_lucy_asks_for_setup_mode(self, tmp_path: Path) -> None:
        assert request_setup({"RUNTIME_DIRECTORY": str(tmp_path)})
        assert (tmp_path / SETUP_REQUEST).exists()
        assert not request_setup({})  # not running as lucy.service

    async def test_held_wing_hands_over(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Wing:
            def __init__(self, held: bool) -> None:
                self._held = held

            async def held(self, seconds: float) -> bool:
                return self._held

        class Pipeline:
            def __init__(self) -> None:
                self.said: list[Phrase] = []

            async def say(self, phrase: Phrase) -> None:
                self.said.append(phrase)

        class Notifier:
            def stopping(self) -> None:
                pass

        monkeypatch.setenv("RUNTIME_DIRECTORY", str(tmp_path))
        config, pipeline, stop = Config(), Pipeline(), asyncio.Event()
        args = (pipeline, stop, Notifier())
        assert not await _hand_over_to_setup(config, Wing(False), *args)  # type: ignore[arg-type]
        assert not await _hand_over_to_setup(config, object(), *args)  # type: ignore[arg-type]
        asyncio.get_running_loop().call_later(0.05, stop.set)  # systemd stops Lucy
        assert await _hand_over_to_setup(config, Wing(True), *args)  # type: ignore[arg-type]
        assert pipeline.said == [Phrase.SETUP]
        assert (tmp_path / SETUP_REQUEST).exists()


# -- the HTTP server ---------------------------------------------------------------------------


class TestHttp:
    async def exchange(self, raw: bytes) -> bytes:
        async def echo(request: HttpRequest) -> HttpResponse:
            return HttpResponse.html(f"{request.method} {request.path} {dict(request.form)}")

        server = await start_server(echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(raw)
            await writer.drain()
            reply = await reader.read()
            writer.close()
            return reply
        finally:
            server.close()

    async def test_form(self) -> None:
        body = b"a=1&b=x+y&c="
        reply = await self.exchange(
            b"POST /save HTTP/1.1\r\nHost: x\r\nContent-Type: application/x-www-form-urlencoded\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        assert reply.startswith(b"HTTP/1.1 200 OK")
        assert b"POST /save {'a': '1', 'b': 'x y', 'c': ''}" in reply
        assert b"Connection: close" in reply

    @pytest.mark.parametrize(
        ("raw", "status"),
        [
            (b"NONSENSE\r\n\r\n", b"400"),
            (b"POST / HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n", b"413"),
            (b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", b"501"),
            (b"GET / HTTP/1.1\r\n" + b"X: " + b"y" * 20_000 + b"\r\n\r\n", b"431"),
        ],
    )
    async def test_refuses_what_it_does_not_handle(self, raw: bytes, status: bytes) -> None:
        assert (await self.exchange(raw)).split(b" ")[1] == status

    async def test_head_has_no_body(self) -> None:
        reply = await self.exchange(b"HEAD / HTTP/1.1\r\nHost: x\r\n\r\n")
        assert reply.endswith(b"\r\n\r\n")
        assert b"Content-Length: " in reply

    async def test_reads_one_request(self) -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(b"GET /a?x=1 HTTP/1.1\r\nHost: Lucy\r\n\r\n")
        request = await read_request(reader)
        assert (request.path, dict(request.query), request.host) == ("/a", {"x": "1"}, "lucy")


class TestConfig:
    def test_setup_settings(self, tmp_path: Path) -> None:
        assert Config().setup.enabled
        path = tmp_path / "c.toml"
        path.write_text("[setup]\nidle_timeout_s = 10\n")
        with pytest.raises(ConfigError, match="idle_timeout_s"):
            load_config(path)
