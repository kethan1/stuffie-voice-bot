"""Lucy's setup Wi-Fi: its name and password, and the card a grown-up scans to join it.

Each goose gets her own name (``Lucy-Setup-4821``) and password, made once by
``lucy setup-card`` (the installer runs it) and kept root-only in
``/etc/lucy/setup-hotspot.toml``. The card is printed and sewn inside the zipper, where only
grown-ups go: whoever has it can change Lucy's settings, so children cannot, even if one
holds her wing while she wakes up. The QR code on it joins the Wi-Fi in one scan, and the
phone then opens the settings page by itself.
"""

import html
import io
import secrets
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import segno
import segno.helpers

from lucy.core.errors import ConfigError
from lucy.setup.files import write_atomically

ADDRESS = "10.42.0.1"
"""Lucy's address on her own Wi-Fi, where the settings page is."""

# No 0/O, 1/l/I: easy to read off a card and type on a phone.
_PASSWORD_LETTERS = "abcdefghjkmnpqrstuvwxyz23456789"


@dataclass(frozen=True, slots=True)
class Hotspot:
    ssid: str
    password: str = field(repr=False)
    address: str = ADDRESS

    @property
    def url(self) -> str:
        return f"http://{self.address}/"

    @property
    def spoken_name(self) -> str:
        """``Lucy-Setup-4821`` as Lucy says it: "Lucy Setup 4 8 2 1"."""
        words = self.ssid.replace("-", " ").split()
        return " ".join(" ".join(word) if word.isdigit() else word for word in words)


def new_hotspot() -> Hotspot:
    number = 1000 + secrets.randbelow(9000)
    password = "-".join(
        "".join(secrets.choice(_PASSWORD_LETTERS) for _ in range(4)) for _ in range(3)
    )
    return Hotspot(f"Lucy-Setup-{number}", password)


def load_hotspot(path: Path, *, create: bool = False) -> Hotspot:
    """Read the hotspot file, making a new one (root-only) first if asked and it is missing."""
    if create and not path.exists():
        hotspot = new_hotspot()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(
            path,
            "# Lucy's setup Wi-Fi (lucy setup-card prints the card for it). Keep it secret.\n"
            f'ssid = "{hotspot.ssid}"\npassword = "{hotspot.password}"\n',
            0o600,
        )
        return hotspot
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        ssid, password = data["ssid"], data["password"]
    except FileNotFoundError as error:
        raise ConfigError(
            f"{path} is missing: make it, and Lucy's setup card, with `sudo lucy setup-card`"
        ) from error
    except (OSError, tomllib.TOMLDecodeError, KeyError) as error:
        raise ConfigError(f"cannot read Lucy's setup Wi-Fi from {path}: {error}") from error
    if not isinstance(ssid, str) or not isinstance(password, str):
        raise ConfigError(f"{path}: ssid and password must be strings")
    if not 1 <= len(ssid.encode()) <= 32 or not 8 <= len(password) <= 63:
        raise ConfigError(f"{path}: the name must be 1-32 bytes and the password 8-63 characters")
    return Hotspot(ssid, password)


def _qr(hotspot: Hotspot) -> segno.QRCode:
    return segno.helpers.make_wifi(ssid=hotspot.ssid, password=hotspot.password, security="WPA")


def card_text(hotspot: Hotspot) -> str:
    """The card for a terminal: the QR code in block characters, then the details."""
    qr = io.StringIO()
    _qr(hotspot).terminal(out=qr, compact=True, border=2)
    lines = [
        qr.getvalue().rstrip("\n"),
        "Lucy's setup card",
        f"  Wi-Fi:     {hotspot.ssid}",
        f"  Password:  {hotspot.password}",
        f"  Page:      {hotspot.url}",
    ]
    return "\n".join(lines)


def card_html(hotspot: Hotspot) -> str:
    """A printable card: cut it out, laminate it, and sew it inside Lucy's zipper."""
    svg = _qr(hotspot).svg_inline(scale=5, border=2, dark="#000", light="#fff")
    ssid, password, url = (
        html.escape(text) for text in (hotspot.ssid, hotspot.password, hotspot.url)
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lucy's setup card</title>
<style>
  body {{ font: 15px/1.4 system-ui, sans-serif; color: #111; background: #fff; margin: 24px; }}
  .card {{ border: 2px dashed #555; border-radius: 12px; padding: 16px 20px; max-width: 26em; }}
  h1 {{ font-size: 1.3em; margin: 0 0 8px; }}
  .qr {{ float: right; margin: 0 0 8px 12px; }}
  dt {{ font-weight: 600; }} dd {{ margin: 0 0 6px; font-family: ui-monospace, monospace; }}
  ol {{ padding-left: 1.2em; clear: both; }}
  .keep {{ font-size: 0.9em; color: #444; }}
</style></head>
<body><div class="card">
  <h1>&#x1FABF; Lucy's setup card (for grown-ups)</h1>
  <div class="qr">{svg}</div>
  <dl><dt>Wi-Fi</dt><dd>{ssid}</dd><dt>Password</dt><dd>{password}</dd>
  <dt>Settings page</dt><dd>{url}</dd></dl>
  <ol>
    <li>Hold Lucy's wing while she wakes up, until she says it's setup time. (If she can't
      find any Wi-Fi she knows, she starts setup by herself.)</li>
    <li>Scan the code with a phone camera to join her Wi-Fi.</li>
    <li>Her settings page opens by itself. If it doesn't, open {url} in the browser.</li>
    <li>Listen to Lucy: she says whether a network worked.</li>
  </ol>
  <p class="keep">Keep this card where only grown-ups can reach it, such as inside her zipper:
  whoever has it can change her settings.</p>
</div></body></html>
"""
