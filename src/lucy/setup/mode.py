"""Setup mode: Lucy's hotspot, her settings page, joining a network, and when to leave.

::

    scan for networks ─► hotspot on ─► "My setup Wi-Fi is on!" ─► serve the page
                            ▲                                          │ Join
                            │ failed: say why                          ▼
                            └──────────────────────────── try the network (the phone loses
                                                            the hotspot): worked and online?
                                                                │ yes: "I'm on …!" ─► leave
    leave: "Finish setup" on the page, her wing held for two seconds, or nothing happening
    for setup.idle_timeout_s. lucy-setup.service then wakes Lucy up again.

Lucy says each step out loud, because the phone loses touch with her while she tries a
network. Settings saved on the page take effect when she wakes up.
"""

import asyncio
import contextlib
import dataclasses
import logging
import secrets
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit

from lucy import __version__
from lucy.config import Config
from lucy.core.errors import ConfigError, LucyError
from lucy.core.ports import Button
from lucy.core.safety import ReplyFilter
from lucy.core.tasks import BackgroundTasks
from lucy.persona import LUCY, SetupLine
from lucy.setup import pages
from lucy.setup.body import Body, LucysBody, SilentBody
from lucy.setup.files import write_atomically
from lucy.setup.hotspot import Hotspot, card_html, card_text, load_hotspot, new_hotspot
from lucy.setup.http import Request, Response, start_server
from lucy.setup.network import (
    DemoNetwork,
    JoinProblem,
    Network,
    NetworkError,
    NetworkManager,
    Runner,
    SavedNetwork,
    Security,
    WifiLogin,
    WifiNetwork,
    run,
)
from lucy.setup.settings import SECTION, SettingsStore, key_problem, parse_section

log = logging.getLogger(__name__)

WING_HOLD_S = 2.0
HANDOFF_S = 1.5
"""The phone gets its page before Lucy's Wi-Fi goes away."""
DEMO_ADDRESS = ("127.0.0.1", 8080)

_JOIN_LINES = {
    JoinProblem.WRONG_PASSWORD: SetupLine.WRONG_PASSWORD,
    JoinProblem.NOT_FOUND: SetupLine.NOT_FOUND,
    JoinProblem.NO_ADDRESS: SetupLine.NO_ADDRESS,
    JoinProblem.FAILED: SetupLine.JOIN_FAILED,
}


def speakable_network(ssid: str, screen: ReplyFilter) -> str:
    """A network's name as Lucy may say it: plain, short, and never a word she must not say."""
    name = " ".join(ssid.replace("_", " ").replace("-", " ").split())
    if not name or len(name) > 32 or not name.isprintable() or not screen.screen(name).allowed:
        return "that network"
    return name


def _problems(error: Exception) -> list[str]:
    message = str(error).removeprefix("invalid configuration: ")
    return [part.strip() for part in message.split("; ") if part.strip()]


class SetupMode:
    def __init__(
        self,
        *,
        config: Config,
        store: SettingsStore,
        network: Network,
        body: Body,
        hotspot: Hotspot,
        address: tuple[str, int],
        button: Button | None = None,
        clock: Callable[[], float] = time.monotonic,
        handoff_s: float = HANDOFF_S,
        wing_hold_s: float = WING_HOLD_S,
        idle_check_s: float = 5.0,
    ) -> None:
        self._config = config
        self._store = store
        self._network = network
        self._body = body
        self._hotspot = hotspot
        self._host, self._port = address
        self._origin = self._host if self._port == 80 else f"{self._host}:{self._port}"
        self._button = button
        self._clock = clock
        self._handoff_s = handoff_s
        self._wing_hold_s = wing_hold_s
        self._idle_check_s = idle_check_s
        self._idle_timeout_s = config.setup.idle_timeout_s
        api = urlsplit(config.api.base_url)
        self._probe = (api.hostname or "api.groq.com", api.port or 443)
        self._token = secrets.token_urlsafe(18)
        self._screen = ReplyFilter()
        self._lines = LUCY.setup_lines
        self._tasks = BackgroundTasks()
        self._done = asyncio.Event()
        self._server: asyncio.Server | None = None
        self._hotspot_on = False
        self._joining: asyncio.Task[None] | None = None
        self._networks: list[WifiNetwork] = []
        self._saved: list[SavedNetwork] = []
        self._last_activity = clock()
        self._greeted = False

    @property
    def url(self) -> str:
        return f"http://{self._origin}/"

    @property
    def done(self) -> bool:
        return self._done.is_set()

    async def run(self) -> int:
        """Run setup mode until it is time to leave. Returns the process exit code."""
        try:
            await self._refresh_networks()  # before the hotspot: the radio can't scan well then
            if not await self._open():
                await self._say(SetupLine.NO_HOTSPOT)
                return 1
            await self._say(SetupLine.READY)
            self._tasks.spawn(self._watch_idle(), "idle")
            if self._button is not None:
                self._tasks.spawn(self._watch_wing(self._button), "wing")
            await self._done.wait()
            return 0
        finally:
            await self._tasks.cancel_all()
            if self._joining is not None:
                self._joining.cancel()
                await asyncio.gather(self._joining, return_exceptions=True)
            await self._close()

    # -- the hotspot and the page --------------------------------------------------------

    async def _open(self) -> bool:
        """Turn on the hotspot and serve the page on it."""
        try:
            await self._network.start_hotspot(self._hotspot)
        except NetworkError as error:
            log.error("could not start the setup Wi-Fi: %s", error)
            return False
        self._hotspot_on = True
        for attempt in range(10):  # the address may take a moment to appear
            try:
                self._server = await start_server(self.handle, self._host, self._port)
                break
            except OSError as error:
                if attempt == 9:
                    log.error("could not serve the settings page at %s: %s", self.url, error)
                    return False
                await asyncio.sleep(0.5)
        log.info("setup mode: Wi-Fi %r, settings page %s", self._hotspot.ssid, self.url)
        return True

    async def _close_page(self) -> None:
        if self._server is not None:
            server, self._server = self._server, None
            server.close()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(server.wait_closed(), 5.0)

    async def _close(self) -> None:
        await self._close_page()
        if self._hotspot_on:
            self._hotspot_on = False
            await self._network.stop_hotspot()

    async def _refresh_networks(self) -> None:
        self._networks = await self._network.scan()
        self._saved = await self._network.saved()

    # -- leaving -------------------------------------------------------------------------

    async def _leave(self, line: SetupLine, *, after_s: float = 0.0) -> None:
        await asyncio.sleep(after_s)
        await self._say(line)
        self._done.set()

    async def _watch_idle(self) -> None:
        while True:
            await asyncio.sleep(self._idle_check_s)
            quiet = self._clock() - self._last_activity
            if self._joining is None and quiet >= self._idle_timeout_s:
                log.info("setup mode: nothing happened for %.0f s", quiet)
                await self._leave(SetupLine.IDLE)
                return

    async def _watch_wing(self, button: Button) -> None:
        """Holding her wing for a couple of seconds leaves setup mode, with no phone needed."""
        holding: asyncio.Task[None] | None = None
        try:
            async for event in button.events():
                if holding is not None:
                    holding.cancel()
                    holding = None
                if event.pressed:
                    holding = asyncio.create_task(self._wing_held())
        except ConfigError as error:
            log.warning("her wing cannot end setup mode: %s", error)
        finally:
            if holding is not None:
                holding.cancel()

    async def _wing_held(self) -> None:
        await asyncio.sleep(self._wing_hold_s)
        if self._joining is None:  # not cancelled by letting go from here on
            self._tasks.spawn(self._leave(SetupLine.CANCELLED), "leave")

    # -- speaking ------------------------------------------------------------------------

    async def _say(self, line: SetupLine, network: str = "") -> None:
        text = self._lines[line].format(hotspot=self._hotspot.spoken_name, network=network)
        log.info("setup mode: %s", line.value)
        try:
            await self._body.say(text)
        except Exception:  # setup mode still works if Lucy cannot speak
            log.warning("Lucy could not say the %s line", line.value, exc_info=True)

    # -- the page ------------------------------------------------------------------------

    async def handle(self, request: Request) -> Response:
        if request.host != self._origin:
            # A phone checking for a sign-in page, or any other site: show it Lucy's page. This
            # is what makes the phone open the page by itself.
            return Response.redirect(self.url, HTTPStatus.FOUND)
        self._last_activity = self._clock()
        if request.method in ("GET", "HEAD"):
            if request.path == "/card":  # only reachable by whoever already has the card
                return Response.html(card_html(self._hotspot))
            if request.path != "/":
                return Response.redirect("/", HTTPStatus.FOUND)
            if not self._greeted:
                self._greeted = True
                self._tasks.spawn(self._say(SetupLine.HELLO), "hello")
            return self._page(notice=pages.NOTICES.get(request.query.get("done", ""), ""))
        if request.method != "POST":
            return Response(HTTPStatus.METHOD_NOT_ALLOWED, headers=(("Allow", "GET, POST"),))
        if not secrets.compare_digest(request.form.get("token", "").encode(), self._token.encode()):
            return Response.html(pages.stale_page(), HTTPStatus.FORBIDDEN)
        routes: Mapping[str, Callable[[Mapping[str, str]], Awaitable[Response]]] = {
            "/save": self._save,
            "/keys": self._keys,
            "/advanced": self._advanced,
            "/undo": self._undo,
            "/try-voice": self._try_voice,
            "/try-motors": self._try_motors,
            "/join": self._join,
            "/forget": self._forget,
            "/finish": self._finish,
        }
        route = routes.get(request.path)
        if route is None or self._joining is not None or self.done:
            return Response.redirect("/")
        return await route(request.form)

    def _page(
        self,
        *,
        notice: str = "",
        errors: list[str] | None = None,
        where: str = "",
        posted: Mapping[str, str] | None = None,
    ) -> Response:
        try:
            config = self._store.load()
        except ConfigError:  # e.g. edited by hand since: show what Lucy started with
            config = self._config
        credentials = (config.api.api_key_credential, config.tts.cloud.api_key_credential)
        state = pages.PageState(
            token=self._token,
            hotspot_name=self._hotspot.ssid,
            networks=self._networks,
            saved=self._saved,
            mac_address=self._network.mac_address,
            config=config,
            config_text=self._store.text(),
            keys={
                name: self._store.key_status(credential)
                for name, credential in zip(("groq", "speechify"), credentials, strict=True)
            },
            can_undo=self._store.can_undo,
            version=__version__,
            idle_minutes=round(self._idle_timeout_s / 60),
        )
        text = pages.page(state, notice=notice, errors=errors or [], where=where, posted=posted)
        return Response.html(text, HTTPStatus.BAD_REQUEST if errors else HTTPStatus.OK)

    async def _save(self, form: Mapping[str, str]) -> Response:
        section = SECTION.get(form.get("section", ""))
        if section is None:
            return Response.redirect("/")
        changes, problems = parse_section(section, form)
        if not problems:
            try:
                self._store.update(changes)
            except (ConfigError, OSError) as error:
                problems = _problems(error)
        if problems:
            return self._page(errors=problems, where=section.key, posted=form)
        log.info("setup mode: saved the %s settings", section.key)
        return Response.redirect("/?done=saved")

    async def _keys(self, form: Mapping[str, str]) -> Response:
        config = self._config
        changes = {}
        if groq := form.get("groq", "").strip():
            changes[config.api.api_key_credential] = groq
        if "remove_speechify" in form:
            changes[config.tts.cloud.api_key_credential] = ""
        elif speechify := form.get("speechify", "").strip():
            changes[config.tts.cloud.api_key_credential] = speechify
        problems = sorted({p for key in changes.values() if key and (p := key_problem(key))})
        if not problems:
            try:
                for credential, key in changes.items():
                    self._store.set_key(credential, key)
            except (ConfigError, OSError) as error:
                problems = [str(error)]
        if problems:
            return self._page(errors=problems, where="keys")
        log.info("setup mode: saved %d API key(s)", len(changes))
        return Response.redirect("/?done=keys")

    async def _advanced(self, form: Mapping[str, str]) -> Response:
        text = form.get("config", "").replace("\r\n", "\n")
        try:
            self._store.replace(text)
        except (ConfigError, OSError) as error:
            return self._page(errors=_problems(error), where="advanced", posted=form)
        log.info("setup mode: saved the whole config file")
        return Response.redirect("/?done=saved")

    async def _undo(self, form: Mapping[str, str]) -> Response:
        try:
            self._store.undo()
        except (ConfigError, OSError) as error:
            return self._page(errors=_problems(error), where="advanced")
        return Response.redirect("/?done=undone")

    async def _try_voice(self, form: Mapping[str, str]) -> Response:
        section = SECTION["lucy"]
        changes, problems = parse_section(section, form)
        if not problems:
            try:
                await self._body.try_voice(
                    volume=changes["tts.volume"],
                    honks=changes["tts.honks"],
                    honk_volume=changes["tts.honk_volume"],
                )
            except (LucyError, OSError) as error:
                problems = [str(error)]
        if problems:
            return self._page(errors=problems, where=section.key, posted=form)
        return Response.empty()

    async def _try_motors(self, form: Mapping[str, str]) -> Response:
        section = SECTION["motors"]
        changes, problems = parse_section(section, form)
        motion = dataclasses.replace(
            self._config.motors.motion,
            **{
                path.removeprefix("motors.motion."): value
                for path, value in changes.items()
                if path.startswith("motors.motion.")
            },
        )
        if not problems and motion.beak_min_power > motion.beak_max_power:
            problems = ["The smallest beak power must not be above the largest."]
        if not problems:
            try:
                await self._body.try_motors(motion)
            except (LucyError, OSError) as error:
                problems = [str(error)]
        if problems:
            return self._page(errors=problems, where=section.key, posted=form)
        return Response.empty()

    async def _forget(self, form: Mapping[str, str]) -> Response:
        try:
            await self._network.forget(form.get("uuid", ""))
        except NetworkError as error:
            return self._page(errors=[str(error)], where="wifi")
        self._saved = await self._network.saved()
        return Response.redirect("/?done=forgotten")

    async def _finish(self, form: Mapping[str, str]) -> Response:
        self._tasks.spawn(self._leave(SetupLine.FINISHED, after_s=self._handoff_s), "finish")
        return Response.html(pages.finished_page())

    async def _join(self, form: Mapping[str, str]) -> Response:
        login, problems = self._login(form)
        if login is None or problems:
            posted = {k: v for k, v in form.items() if k != "password"}
            return self._page(errors=problems, where="wifi", posted=posted)
        self._joining = asyncio.create_task(self._try_network(login), name="join")
        return Response.html(pages.joining_page(login.ssid))

    def _login(self, form: Mapping[str, str]) -> tuple[WifiLogin | None, list[str]]:
        choice = form.get("network", "")
        if choice == pages.OTHER_NETWORK:
            ssid, hidden = form.get("hidden_ssid", "").strip(), True
            try:
                security = Security(form.get("security", Security.PERSONAL.value))
            except ValueError:
                security = Security.PERSONAL
        else:
            found = next((n for n in self._networks if n.ssid == choice), None)
            if found is None:
                return None, ["Pick a network from the list (or Another network)."]
            ssid, hidden, security = found.ssid, False, found.security
        login = WifiLogin(
            ssid=ssid,
            security=security,
            password="" if security is Security.OPEN else form.get("password", ""),
            username=form.get("username", "").strip() if security is Security.ENTERPRISE else "",
            hidden=hidden,
        )
        return login, login.problems()

    async def _try_network(self, login: WifiLogin) -> None:
        name = speakable_network(login.ssid, self._screen)
        try:
            await asyncio.sleep(self._handoff_s)
            await self._say(SetupLine.TRYING, name)
            await self._close_page()
            self._hotspot_on = False  # the new connection takes the radio over from the hotspot
            log.info("setup mode: trying a network")
            result = await self._network.join(login)
            if result.ok and await self._network.reaches(*self._probe):
                await self._say(SetupLine.JOINED, name)
                self._done.set()
                return
            line = (
                SetupLine.JOINED_OFFLINE
                if result.ok
                else _JOIN_LINES[result.problem or JoinProblem.FAILED]
            )
            await self._refresh_networks()
            if not await self._open():
                await self._say(SetupLine.NO_HOTSPOT)
                self._done.set()
                return
            await self._say(line, name)
        finally:
            self._joining = None
            self._last_activity = self._clock()


# -- entry points (lucy setup, lucy setup --if-offline, lucy setup-card) ---------------------


async def start_setup_service(runner: Runner = run) -> int:
    result = await runner(["systemctl", "start", "--no-block", "lucy-setup.service"])
    if result.returncode != 0:
        log.error("could not start setup mode: %s", result.stderr.strip())
    return result.returncode


async def start_if_offline(
    config: Config,
    network: Network,
    start: Callable[[], Awaitable[int]] = start_setup_service,
) -> int:
    """At boot (lucy-setup-offline.service): setup mode, if no network connects in time."""
    if not (config.setup.enabled and config.setup.when_offline):
        return 0
    if await network.wait_online(config.setup.offline_wait_s):
        log.info("a network is connected, so setup mode is not needed")
        return 0
    log.info("no network %.0f s after boot: starting setup mode", config.setup.offline_wait_s)
    return await start()


async def run_setup(config: Config, config_path: Path, *, demo: bool = False) -> int:
    """``lucy setup``: what lucy-setup.service runs (``--demo``: on a laptop, pretend Wi-Fi)."""
    if not config.setup.enabled and not demo:
        log.error("setup mode is turned off ([setup] enabled = false)")
        return 0
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)

    network: Network
    if demo:
        network, hotspot, address = DemoNetwork(), new_hotspot(), DEMO_ADDRESS
        credentials = config_path.parent / ".lucy-setup-demo" / "credentials"
    else:
        network = NetworkManager(config.setup.interface)
        if not config.setup.hotspot_file.exists():
            log.warning("Lucy had no setup Wi-Fi yet; print its card with: sudo lucy setup-card")
        hotspot = load_hotspot(config.setup.hotspot_file, create=True)
        address, credentials = (hotspot.address, 80), config.setup.credentials_dir
    if demo:
        print(f"Lucy's settings page (pretend Wi-Fi): http://{address[0]}:{address[1]}/")

    async with AsyncExitStack() as stack:
        body: Body
        try:
            lucys_body = LucysBody(config, stack)
            await lucys_body.open()
            body = lucys_body
        except (LucyError, OSError, ImportError) as error:
            log.warning("Lucy cannot speak in setup mode: %s", error)
            body = SilentBody()
        button = None
        if config.button.backend == "gpio" and not demo:
            from lucy.adapters.gpio import GpioButton

            b = config.button
            button = GpioButton(b.chip, b.line, active_low=b.active_low, debounce_ms=b.debounce_ms)
        mode = SetupMode(
            config=config,
            store=SettingsStore(config_path, credentials),
            network=network,
            body=body,
            hotspot=hotspot,
            address=address,
            button=button,
        )
        running = asyncio.create_task(mode.run(), name="setup")
        stopping = asyncio.create_task(stop.wait(), name="stop")
        await asyncio.wait({running, stopping}, return_when=asyncio.FIRST_COMPLETED)
        stopping.cancel()
        if not running.done():
            running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            return await running
    return 0


def print_card(config: Config, output: Path | None) -> int:
    """``lucy setup-card``: make Lucy's setup Wi-Fi if needed, and print its card."""
    hotspot = load_hotspot(config.setup.hotspot_file, create=True)
    print(card_text(hotspot))
    if output is not None:
        write_atomically(output, card_html(hotspot), 0o600)  # it holds the password
        print(f"\nA printable copy is in {output}")
    return 0
