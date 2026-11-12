"""Wi-Fi through NetworkManager: scanning, Lucy's setup hotspot, and joining a network safely.

Connections are written as NetworkManager keyfiles and loaded with ``nmcli connection
load``, so a password never appears on a command line, where any process could read it.

Joining is all or nothing. The new network is added with autoconnect off and tried. Only if
it works is it kept (autoconnect on), and any older profile for the same network removed.
If it fails, it is deleted again, and every network saved before is just as it was.

The Pi's Wi-Fi chip is either Lucy's hotspot or a client of another network, never both,
and it scans poorly while it is a hotspot, so setup mode scans first and keeps the list.
nmcli runs with ``LC_ALL=C``, so its messages (which say why a network failed) are in
English, whatever language the system uses.
"""

import asyncio
import contextlib
import logging
import os
import uuid as uuids
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from lucy.core.errors import LucyError
from lucy.setup.files import write_atomically
from lucy.setup.hotspot import Hotspot

log = logging.getLogger(__name__)

CONNECTIONS_DIR = Path("/etc/NetworkManager/system-connections")
HOTSPOT_UUID = str(uuids.uuid5(uuids.NAMESPACE_DNS, "setup-hotspot.lucy.invalid"))
JOIN_TIMEOUT_S = 45
HOTSPOT_TIMEOUT_S = 30
REACH_ATTEMPTS = 3
REACH_TIMEOUT_S = 5.0


class NetworkError(LucyError):
    """NetworkManager refused to do something (start the hotspot, forget a network)."""


class Security(StrEnum):
    OPEN = "open"
    PERSONAL = "personal"
    """A password: WPA2, or WPA2 and WPA3 together. Most homes, and many schools."""
    WPA3 = "wpa3"
    """A password, on a network that only allows WPA3."""
    ENTERPRISE = "enterprise"
    """A username and a password (802.1X with PEAP), as many school networks use."""


def security_of(flags: str) -> Security:
    """Read nmcli's SECURITY column: "", "WPA2", "WPA1 WPA2", "WPA2 WPA3", "WPA2 802.1X", ..."""
    words = set(flags.split()) - {"--"}
    if "802.1X" in words:
        return Security.ENTERPRISE
    if words <= {"OWE"}:  # nothing, or "enhanced open": no password
        return Security.OPEN
    if words == {"WPA3"}:
        return Security.WPA3
    return Security.PERSONAL


@dataclass(frozen=True, slots=True)
class WifiNetwork:
    """A network in range."""

    ssid: str
    signal: int
    """0 to 100."""
    security: Security


@dataclass(frozen=True, slots=True)
class SavedNetwork:
    uuid: str
    name: str
    ssid: str


@dataclass(frozen=True, slots=True)
class WifiLogin:
    """What a grown-up typed to join a network."""

    ssid: str
    security: Security
    password: str = field(default="", repr=False)
    username: str = ""
    hidden: bool = False

    def problems(self) -> list[str]:
        problems = []
        if not 1 <= len(self.ssid.encode()) <= 32:
            problems.append("A Wi-Fi network's name is 1 to 32 characters long.")
        if not all(text.isprintable() for text in (self.ssid, self.password, self.username)):
            problems.append("Use only letters, numbers, spaces, and symbols you can type.")
        if self.security is Security.OPEN:
            return problems
        if not self.password:
            problems.append("This network needs a password.")
        elif self.security is not Security.ENTERPRISE and not _wpa_passphrase(self.password):
            problems.append("A Wi-Fi password is 8 to 63 characters long.")
        if self.security is Security.ENTERPRISE and not self.username:
            problems.append("This network needs a username as well as a password.")
        return problems


def _wpa_passphrase(password: str) -> bool:
    if len(password) == 64:
        return all(c in "0123456789abcdefABCDEF" for c in password)  # a raw key, in hex
    return 8 <= len(password.encode()) <= 63


class JoinProblem(StrEnum):
    WRONG_PASSWORD = "wrong_password"
    NOT_FOUND = "not_found"
    NO_ADDRESS = "no_address"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JoinResult:
    problem: JoinProblem | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.problem is None


def join_problem(message: str) -> JoinProblem:
    """Why nmcli could not activate a connection, from its (English) error message."""
    text = message.lower()
    if "secrets were required" in text or "supplicant" in text:
        return JoinProblem.WRONG_PASSWORD  # a wrong WPA key ends in "no secrets"
    if "could not be found" in text or "no network with ssid" in text:
        return JoinProblem.NOT_FOUND
    if "ip configuration" in text or "dhcp" in text:
        return JoinProblem.NO_ADDRESS
    return JoinProblem.FAILED


class Network(Protocol):
    """What setup mode needs from the Wi-Fi."""

    @property
    def mac_address(self) -> str: ...

    async def wait_online(self, timeout_s: float) -> bool:
        """True as soon as any network connects; False if none has within ``timeout_s``."""
        ...

    async def scan(self) -> list[WifiNetwork]:
        """Networks in range, strongest first, each name once."""
        ...

    async def saved(self) -> list[SavedNetwork]: ...

    async def start_hotspot(self, hotspot: Hotspot) -> None:
        """Turn on Lucy's own Wi-Fi (leaving any other network). Raises NetworkError."""
        ...

    async def stop_hotspot(self) -> None:
        """Turn it off: NetworkManager then joins a saved network in range by itself."""
        ...

    async def join(self, login: WifiLogin) -> JoinResult:
        """Try a network, and keep it only if it works."""
        ...

    async def forget(self, uuid: str) -> None: ...

    async def reaches(self, host: str, port: int) -> bool:
        """Whether ``host`` answers over the network Lucy is on now."""
        ...


@dataclass(frozen=True, slots=True)
class Completed:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str]], Awaitable[Completed]]


async def run(argv: Sequence[str]) -> Completed:
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "LC_ALL": "C"},
        )
    except OSError as error:  # e.g. nmcli is not installed
        return Completed(127, "", str(error))
    stdout, stderr = await process.communicate()
    return Completed(
        process.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


class NetworkManager:
    def __init__(
        self,
        interface: str = "wlan0",
        *,
        connections: Path = CONNECTIONS_DIR,
        runner: Runner = run,
        sysfs: Path = Path("/sys/class/net"),
    ) -> None:
        self._interface = interface
        self._connections = connections
        self._run = runner
        self._sysfs = sysfs

    @property
    def mac_address(self) -> str:
        try:
            return (self._sysfs / self._interface / "address").read_text().strip()
        except OSError:
            return ""

    async def wait_online(self, timeout_s: float) -> bool:
        result = await self._run(
            ["nm-online", "--quiet", "--timeout", str(max(1, round(timeout_s)))]
        )
        return result.returncode == 0

    async def scan(self) -> list[WifiNetwork]:
        listing = ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"]
        listing += ["ifname", self._interface]
        result = await self._run([*listing, "--rescan", "yes"])
        if result.returncode != 0:  # e.g. a scan has only just finished: use what it found
            result = await self._run([*listing, "--rescan", "no"])
        if result.returncode != 0:
            log.warning("could not list Wi-Fi networks: %s", result.stderr.strip())
            return []
        return parse_scan(result.stdout)

    async def saved(self) -> list[SavedNetwork]:
        result = await self._nmcli("-t", "-f", "UUID,TYPE,NAME", "connection", "show")
        if result.returncode != 0:
            log.warning("could not list saved networks: %s", result.stderr.strip())
            return []
        saved = []
        for line in result.stdout.splitlines():
            uuid, kind, name = [*split_terse(line), "", ""][:3]
            if kind not in ("802-11-wireless", "wifi") or uuid == HOTSPOT_UUID:
                continue
            ssid = await self._nmcli(
                "-e", "no", "-g", "802-11-wireless.ssid", "connection", "show", "uuid", uuid
            )
            saved.append(SavedNetwork(uuid, name, ssid.stdout.rstrip("\n")))
        return saved

    async def start_hotspot(self, hotspot: Hotspot) -> None:
        path = self._connections / "lucy-setup-hotspot.nmconnection"
        write_atomically(path, hotspot_keyfile(hotspot, self._interface), 0o600)
        await self._must("connection", "load", str(path))
        await self._must("--wait", str(HOTSPOT_TIMEOUT_S), "connection", "up", "uuid", HOTSPOT_UUID)

    async def stop_hotspot(self) -> None:
        await self._nmcli("connection", "down", "uuid", HOTSPOT_UUID)  # fine if already down

    async def join(self, login: WifiLogin) -> JoinResult:
        new = str(uuids.uuid4())
        path = self._connections / f"lucy-{new}.nmconnection"
        write_atomically(path, network_keyfile(login, new), 0o600)
        loaded = await self._nmcli("connection", "load", str(path))
        if loaded.returncode != 0:
            path.unlink(missing_ok=True)
            return JoinResult(JoinProblem.FAILED, loaded.stderr.strip())
        up = await self._nmcli("--wait", str(JOIN_TIMEOUT_S), "connection", "up", "uuid", new)
        if up.returncode != 0:
            detail = (up.stderr or up.stdout).strip()
            log.info("could not join the network: %s", detail)
            if (await self._nmcli("connection", "delete", "uuid", new)).returncode != 0:
                path.unlink(missing_ok=True)
                await self._nmcli("connection", "reload")
            return JoinResult(join_problem(detail), detail)
        await self._nmcli("connection", "modify", "uuid", new, "connection.autoconnect", "yes")
        for old in await self.saved():  # an older profile may hold an out-of-date password
            if old.ssid == login.ssid and old.uuid != new:
                await self._nmcli("connection", "delete", "uuid", old.uuid)
        return JoinResult()

    async def forget(self, uuid: str) -> None:
        if uuid == HOTSPOT_UUID:
            raise NetworkError("Lucy's own setup Wi-Fi cannot be forgotten")
        await self._must("connection", "delete", "uuid", uuid)

    async def reaches(self, host: str, port: int) -> bool:
        for attempt in range(REACH_ATTEMPTS):
            if attempt:
                await asyncio.sleep(2.0)  # DNS may take a moment on a brand-new connection
            try:
                async with asyncio.timeout(REACH_TIMEOUT_S):
                    _, writer = await asyncio.open_connection(host, port)
            except (OSError, TimeoutError):
                continue
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            return True
        return False

    async def _nmcli(self, *args: str) -> Completed:
        return await self._run(["nmcli", *args])

    async def _must(self, *args: str) -> Completed:
        result = await self._nmcli(*args)
        if result.returncode != 0:
            raise NetworkError(
                (result.stderr or result.stdout).strip() or f"nmcli {args[0]} failed"
            )
        return result


# -- nmcli output and keyfiles -------------------------------------------------------------


def split_terse(line: str) -> list[str]:
    """Split one line of ``nmcli -t`` output, where ``\\:`` and ``\\\\`` are escaped."""
    fields, current, escaped = [], [], False
    for char in line:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
    fields.append("".join(current))
    return fields


def parse_scan(output: str) -> list[WifiNetwork]:
    best: dict[str, WifiNetwork] = {}
    for line in output.splitlines():
        fields = split_terse(line)
        if len(fields) < 3 or not fields[0]:  # a hidden network has no name to show
            continue
        ssid, signal, flags = fields[0], fields[1], fields[2]
        network = WifiNetwork(ssid, int(signal) if signal.isdigit() else 0, security_of(flags))
        if ssid not in best or network.signal > best[ssid].signal:
            best[ssid] = network
    return sorted(best.values(), key=lambda n: (-n.signal, n.ssid.lower()))


def _value(text: str) -> str:
    """``text`` as a keyfile (GKeyFile) string: backslashes, line breaks, and leading spaces
    escaped, so any password survives exactly."""
    out, leading = [], True
    for char in text:
        if leading and char == " ":
            out.append("\\s")
            continue
        leading = False
        out.append({"\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(char, char))
    return "".join(out)


def _ssid(ssid: str) -> str:
    """An SSID as a list of byte values ("76;117;99;121;"): NetworkManager reads any name,
    with any characters, exactly."""
    return "".join(f"{byte};" for byte in ssid.encode())


def _keyfile(groups: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> str:
    lines = ["# Written by Lucy's setup mode (lucy.setup.network)."]
    for name, entries in groups:
        lines += ["", f"[{name}]", *(f"{key}={value}" for key, value in entries)]
    return "\n".join(lines) + "\n"


def network_keyfile(login: WifiLogin, uuid: str) -> str:
    wifi = [("mode", "infrastructure"), ("ssid", _ssid(login.ssid))]
    if login.hidden:
        wifi.append(("hidden", "true"))
    # The Pi's own address, never a random one, so a school can allow Lucy by her MAC address.
    wifi.append(("cloned-mac-address", "permanent"))
    groups: list[tuple[str, Sequence[tuple[str, str]]]] = [
        (
            "connection",
            [
                ("id", _value(login.ssid)),
                ("uuid", uuid),
                ("type", "wifi"),
                ("autoconnect", "false"),
            ],
        ),
        ("wifi", wifi),
    ]
    match login.security:
        case Security.PERSONAL | Security.WPA3:
            key_mgmt = "sae" if login.security is Security.WPA3 else "wpa-psk"
            groups.append(
                ("wifi-security", [("key-mgmt", key_mgmt), ("psk", _value(login.password))])
            )
        case Security.ENTERPRISE:
            groups.append(("wifi-security", [("key-mgmt", "wpa-eap")]))
            groups.append(
                (
                    "802-1x",
                    [
                        ("eap", "peap;"),
                        ("identity", _value(login.username)),
                        ("password", _value(login.password)),
                        ("phase2-auth", "mschapv2"),
                    ],
                )
            )
        case Security.OPEN:
            pass
    groups += [("ipv4", [("method", "auto")]), ("ipv6", [("method", "auto")])]
    return _keyfile(groups)


def hotspot_keyfile(hotspot: Hotspot, interface: str) -> str:
    return _keyfile(
        [
            (
                "connection",
                [
                    ("id", "Lucy setup hotspot"),
                    ("uuid", HOTSPOT_UUID),
                    ("type", "wifi"),
                    ("interface-name", _value(interface)),
                    ("autoconnect", "false"),
                ],
            ),
            ("wifi", [("mode", "ap"), ("band", "bg"), ("ssid", _ssid(hotspot.ssid))]),
            # WPA2 with AES only. Protected management frames off (pmf=1): some phones cannot
            # join a Raspberry Pi's hotspot with them on.
            (
                "wifi-security",
                [
                    ("key-mgmt", "wpa-psk"),
                    ("proto", "rsn;"),
                    ("pairwise", "ccmp;"),
                    ("group", "ccmp;"),
                    ("pmf", "1"),
                    ("psk", _value(hotspot.password)),
                ],
            ),
            # "shared": NetworkManager hands out addresses and answers DNS, which
            # deploy/networkmanager/lucy-setup.conf points at the settings page.
            ("ipv4", [("method", "shared"), ("address1", f"{hotspot.address}/24")]),
            ("ipv6", [("method", "disabled")]),
        ]
    )


# -- pretend Wi-Fi, for trying setup mode on a laptop ----------------------------------------


class DemoNetwork:
    """Pretend Wi-Fi for ``lucy setup --demo`` (and tests). Nothing real changes.

    A password of ``wrong-password`` is refused, and a name that is not in range and not
    marked hidden cannot be found.
    """

    def __init__(self, *, online: bool = True) -> None:
        self.networks = [
            WifiNetwork("School Wi-Fi", 82, Security.ENTERPRISE),
            WifiNetwork("Library Guest", 64, Security.OPEN),
            WifiNetwork("Home", 55, Security.PERSONAL),
            WifiNetwork("Lab", 31, Security.WPA3),
        ]
        self.saved_networks = [SavedNetwork("demo-home", "Home", "Home")]
        self.online = online
        self.hotspot: Hotspot | None = None
        self.joined: list[WifiLogin] = []

    @property
    def mac_address(self) -> str:
        return "dc:a6:32:00:00:01"

    async def wait_online(self, timeout_s: float) -> bool:
        return self.online

    async def scan(self) -> list[WifiNetwork]:
        return list(self.networks)

    async def saved(self) -> list[SavedNetwork]:
        return list(self.saved_networks)

    async def start_hotspot(self, hotspot: Hotspot) -> None:
        self.hotspot = hotspot

    async def stop_hotspot(self) -> None:
        self.hotspot = None

    async def join(self, login: WifiLogin) -> JoinResult:
        self.hotspot = None
        if login.password == "wrong-password":
            return JoinResult(JoinProblem.WRONG_PASSWORD, "pretend: wrong password")
        if not login.hidden and login.ssid not in {n.ssid for n in self.networks}:
            return JoinResult(JoinProblem.NOT_FOUND, "pretend: not in range")
        self.joined.append(login)
        self.saved_networks = [n for n in self.saved_networks if n.ssid != login.ssid]
        self.saved_networks.append(SavedNetwork(f"demo-{len(self.joined)}", login.ssid, login.ssid))
        return JoinResult()

    async def forget(self, uuid: str) -> None:
        self.saved_networks = [n for n in self.saved_networks if n.uuid != uuid]

    async def reaches(self, host: str, port: int) -> bool:
        return self.online
