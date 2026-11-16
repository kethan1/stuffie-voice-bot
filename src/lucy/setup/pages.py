"""The HTML of setup mode's pages.

No JavaScript and no outside resources: a phone's captive-portal window (the sign-in sheet
that opens by itself) is a limited browser, and Lucy's hotspot has no internet. Plain forms
post back to Lucy. "Try it" buttons answer 204 No Content, which leaves the page as it is.
Everything shown is escaped, and nothing more so than the Conversations page: what a child
said to Lucy is shown to a grown-up exactly as it was heard.
"""

import html
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lucy.adapters.logbook import LoggedConversation, LoggedTurn, Summary
from lucy.config import Config
from lucy.setup.network import SavedNetwork, Security, WifiNetwork
from lucy.setup.settings import SECTION, SECTIONS, Field, KeyStatus, Section, current

OTHER_NETWORK = "__other__"

_SECURITY_LABELS = {
    Security.OPEN: "no password",
    Security.PERSONAL: "password",
    Security.WPA3: "password (WPA3)",
    Security.ENTERPRISE: "username and password",
}

NOTICES = {
    "saved": "Saved. Lucy uses her new settings as soon as setup ends.",
    "keys": "Keys saved.",
    "undone": "The last change is undone.",
    "forgotten": "Network forgotten.",
    "cleared": "Every conversation is deleted.",
}

CONVERSATIONS_PER_PAGE = 10

# Why a safety screen stepped in (lucy.core.types.Concern), in a grown-up's words.
_CONCERN_LABELS = {
    "needs_grown_up": "may need a grown-up",
    "unsafe": "something dangerous",
    "private": "personal details",
    "off_limits": "a grown-up topic or a bad word",
    "unkind": "being unkind",
    "trick": "a trick on Lucy",
}
# How a turn ended (lucy.core.types.Outcome), when that is worth a remark.
_OUTCOME_LABELS = {
    "interrupted": "interrupted",
    "screened": "a safety screen answered",
    "stopped": "a safety screen stopped her reply",
    "failed": "she could not answer",
    "volume": "asked for her volume",
    "setup": "asked for setup mode",
}

_STYLE = """
:root { --bg: #f6f3ec; --card: #fff; --ink: #1d1b16; --muted: #5d584d; --line: #d9d3c5;
  --accent: #b5651d; --accent-ink: #fff; --ok: #2f6b3a; --bad: #a12a1e; color-scheme: light; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #1b1a17; --card: #262420; --ink: #f1ede4; --muted: #b3ac9c; --line: #3d3a33;
    --accent: #e09245; --accent-ink: #1b1a17; --ok: #7cc68a; --bad: #f08a7e; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 34rem; margin: 0 auto; padding: 16px 16px 48px; }
h1 { font-size: 1.5rem; margin: 8px 0 4px; }
h2 { font-size: 1.15rem; margin: 0 0 4px; }
p { margin: 6px 0; }
.lead, .help, .meta { color: var(--muted); }
.help { display: block; font-size: 0.875rem; font-weight: 400; margin: 2px 0 0; }
.meta { font-size: 0.8125rem; }
section, details { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
  padding: 14px 16px; margin: 14px 0; }
summary { cursor: pointer; font-weight: 600; font-size: 1.05rem; }
details[open] summary { margin-bottom: 8px; }
label { display: block; margin: 12px 0 0; font-weight: 600; }
input[type=text], input[type=password], input[type=number], select, textarea {
  display: block; width: 100%; margin-top: 4px; padding: 10px 12px; font: inherit;
  color: var(--ink); background: var(--bg); border: 1px solid var(--line); border-radius: 10px; }
textarea { font-family: ui-monospace, Menlo, monospace; font-size: 0.8125rem; min-height: 22rem; }
label.check { display: flex; gap: 10px; align-items: flex-start; font-weight: 600; }
label.check input { width: 1.25rem; height: 1.25rem; margin: 2px 0 0; flex: none; }
.buttons { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }
button { font: inherit; font-weight: 600; padding: 10px 16px; border-radius: 10px;
  border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink); }
button.quiet { background: transparent; color: var(--accent); }
.notice, .error { border-radius: 10px; padding: 10px 12px; margin: 12px 0; font-weight: 600; }
.notice { color: var(--ok); border: 1px solid var(--ok); }
.error { color: var(--bad); border: 1px solid var(--bad); }
.error ul { margin: 4px 0 0; padding-left: 1.2rem; font-weight: 400; }
ul.saved { list-style: none; padding: 0; margin: 6px 0 0; }
ul.saved li { display: flex; justify-content: space-between; align-items: center; gap: 8px;
  padding: 6px 0; border-top: 1px solid var(--line); }
ul.saved form { margin: 0; }
ul.saved button { padding: 6px 12px; }
code { font-family: ui-monospace, Menlo, monospace; }
a { color: var(--accent); }
.turn { border-top: 1px solid var(--line); padding: 8px 0 2px; overflow-wrap: anywhere; }
.turn.flagged { border-left: 4px solid var(--bad); padding-left: 10px; }
.turn.flagged .meta { color: var(--bad); font-weight: 600; }
.who { font-weight: 600; }
.pager { display: flex; justify-content: space-between; gap: 10px; margin: 14px 0; }
"""


@dataclass(frozen=True, slots=True)
class PageState:
    token: str
    hotspot_name: str
    networks: Sequence[WifiNetwork]
    saved: Sequence[SavedNetwork]
    mac_address: str
    config: Config
    config_text: str
    keys: Mapping[str, KeyStatus]
    """``groq`` and ``speechify``."""
    can_undo: bool
    version: str
    idle_minutes: int
    logged: Summary | None = None
    """What her logbook holds. None if it could not be read."""


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _document(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )


def _messages(notice: str, errors: Sequence[str]) -> str:
    parts = []
    if notice:
        parts.append(f'<p class="notice" role="status">{_e(notice)}</p>')
    if errors:
        items = "".join(f"<li>{_e(error)}</li>" for error in errors)
        parts.append(f'<div class="error" role="alert">Not saved:<ul>{items}</ul></div>')
    return "".join(parts)


def _token(state: PageState) -> str:
    return f'<input type="hidden" name="token" value="{_e(state.token)}">'


def page(
    state: PageState,
    *,
    notice: str = "",
    errors: Sequence[str] = (),
    where: str = "",
    posted: Mapping[str, str] | None = None,
) -> str:
    """The settings page. ``errors`` belong to the form ``where``, shown with ``posted``."""
    posted = posted or {}

    def messages_for(name: str) -> str:
        return _messages("", errors) if where == name else ""

    body = [
        "<h1>&#x1FABF; Lucy setup</h1>",
        '<p class="lead">Changes take effect when you finish setup. Keep this page open while '
        "you set her up.</p>",
        _messages(notice, () if where else errors),
        _wifi(state, messages_for("wifi"), posted if where == "wifi" else {}),
        *(
            _section(
                state, section, messages_for(section.key), posted if where == section.key else {}
            )
            for section in SECTIONS
            if section.key != "logbook"
        ),
        _logbook(state, messages_for("logbook"), posted if where == "logbook" else {}),
        _keys(state, messages_for("keys")),
        _advanced(state, messages_for("advanced"), posted if where == "advanced" else {}),
        _finish(state),
    ]
    return _document("Lucy setup", "".join(body))


def _wifi(state: PageState, messages: str, posted: Mapping[str, str]) -> str:
    chosen = posted.get("network", "")
    options = []
    for network in state.networks:
        selected = " selected" if network.ssid == chosen else ""
        bars = "▂▄▆█"[: max(1, min(4, (network.signal + 24) // 25))]
        detail = f"{bars} · {_SECURITY_LABELS[network.security]}"
        options.append(
            f'<option value="{_e(network.ssid)}"{selected}>'
            f"{_e(network.ssid)} ({_e(detail)})</option>"
        )
    other_selected = " selected" if chosen == OTHER_NETWORK or not state.networks else ""
    options.append(
        f'<option value="{OTHER_NETWORK}"{other_selected}>Another network (hidden)…</option>'
    )
    chosen_security = posted.get("security", Security.PERSONAL.value)
    security_options = "".join(
        f'<option value="{s.value}"{" selected" if chosen_security == s.value else ""}>'
        f"{_e(label.capitalize())}</option>"
        for s, label in _SECURITY_LABELS.items()
    )
    saved = (
        '<ul class="saved">'
        + "".join(
            f"<li><span>{_e(network.ssid or network.name)}</span>"
            f'<form method="post" action="/forget">{_token(state)}'
            f'<input type="hidden" name="uuid" value="{_e(network.uuid)}">'
            '<button class="quiet">Forget</button></form></li>'
            for network in state.saved
        )
        + "</ul>"
        if state.saved
        else '<p class="help">None yet.</p>'
    )
    mac = (
        f'<p class="meta">Lucy\'s Wi-Fi address (MAC): <code>{_e(state.mac_address)}</code>. '
        "If the school only lets in approved devices, give this to its IT team.</p>"
        if state.mac_address
        else ""
    )
    return f"""<section id="wifi"><h2>Wi-Fi</h2>{messages}
<p class="help">Networks Lucy knows:</p>{saved}
<form method="post" action="/join">{_token(state)}
<label>Network<select name="network">{"".join(options)}</select></label>
<label>Hidden network's name<input type="text" name="hidden_ssid" maxlength="32"
 autocapitalize="none" autocorrect="off" value="{_e(posted.get("hidden_ssid", ""))}">
<span class="help">Only for "Another network".</span></label>
<label>Its security<select name="security">{security_options}</select>
<span class="help">Only for "Another network".</span></label>
<label>Username<input type="text" name="username" autocapitalize="none" autocorrect="off"
 autocomplete="username" value="{_e(posted.get("username", ""))}">
<span class="help">Only for networks that ask for one (many school networks).</span></label>
<label>Password<input type="password" name="password" autocomplete="current-password"></label>
<p class="help">When you tap Join, your phone loses Lucy's Wi-Fi. Listen to her: she says
whether it worked. If not, her setup Wi-Fi comes back within a minute; join it again.</p>
<div class="buttons"><button>Join</button></div></form>{mac}</section>"""


def _field(field: Field, value: Any, posted: Mapping[str, str]) -> str:
    name = _e(field.path)
    help_text = f'<span class="help">{_e(field.help)}</span>' if field.help else ""
    if field.kind is bool:
        checked = (field.path in posted) if posted else bool(value)
        return (
            f'<label class="check"><input type="checkbox" name="{name}"'
            f"{' checked' if checked else ''}><span>{_e(field.label)}{help_text}</span></label>"
        )
    shown = posted.get(field.path, f"{value:g}" if isinstance(value, float) else str(value))
    if field.kind is str:
        control = (
            f'<input type="text" name="{name}" value="{_e(shown)}" maxlength="200" '
            'autocapitalize="none" autocorrect="off">'
        )
    else:
        control = (
            f'<input type="number" name="{name}" value="{_e(shown)}" min="{field.low:g}" '
            f'max="{field.high:g}" step="{field.step:g}" inputmode="decimal">'
        )
    return f"<label>{_e(field.label)}{control}{help_text}</label>"


def _section(state: PageState, section: Section, messages: str, posted: Mapping[str, str]) -> str:
    fields = "".join(
        _field(field, current(state.config, field.path), posted) for field in section.fields
    )
    trial = {
        "lucy": '<button class="quiet" formaction="/try-voice">Hear her</button>',
        "motors": '<button class="quiet" formaction="/try-motors">Test motors</button>',
    }.get(section.key, "")
    opened = " open" if messages or posted else ""
    return f"""<details id="{_e(section.key)}"{opened}><summary>{_e(section.title)}</summary>
<p class="help">{_e(section.intro)}</p>{messages}
<form method="post" action="/save">{_token(state)}
<input type="hidden" name="section" value="{_e(section.key)}">{fields}
<div class="buttons"><button>Save</button>{trial}</div></form></details>"""


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _logbook(state: PageState, messages: str, posted: Mapping[str, str]) -> str:
    """Her logbook: how much it holds, the way to the conversations, and its two settings."""
    section = SECTION["logbook"]
    fields = "".join(
        _field(field, current(state.config, field.path), posted) for field in section.fields
    )
    found = state.logged
    if found is None:
        held = "Her logbook could not be read."
    elif not found.turns:
        held = "Nothing is written down yet."
    else:
        held = f"{_count(found.conversations, 'conversation')} written down"
        if found.flagged:
            held += f", {found.flagged} of them with a safety screen stepping in"
        held += "."
    return f"""<section id="logbook"><h2>{_e(section.title)}</h2>
<p class="help">{_e(section.intro)}</p>
<p>{_e(held)} <a href="/conversations">Read them</a></p>{messages}
<form method="post" action="/save">{_token(state)}
<input type="hidden" name="section" value="{_e(section.key)}">{fields}
<div class="buttons"><button>Save</button></div></form></section>"""


def _key_line(status: KeyStatus) -> str:
    if not status.present:
        return "Not set."
    return f"Set (ends in …{_e(status.ending)})." if status.ending else "Set."


def _keys(state: PageState, messages: str) -> str:
    groq, speechify = state.keys["groq"], state.keys["speechify"]
    return f"""<details id="keys"{" open" if messages else ""}><summary>API keys</summary>
<p class="help">A grown-up makes these (README.md, "API keys"). Leave a box empty to keep
the key Lucy has.</p>{messages}
<form method="post" action="/keys">{_token(state)}
<label>Groq key (her brain)<input type="password" name="groq" autocomplete="off">
<span class="help">{_key_line(groq)}</span></label>
<label>Speechify key (her natural voice)
<input type="password" name="speechify" autocomplete="off">
<span class="help">{_key_line(speechify)} Optional: without it she uses her backup voice.</span>
</label>
<label class="check"><input type="checkbox" name="remove_speechify">
<span>Remove the Speechify key</span></label>
<div class="buttons"><button>Save keys</button></div></form></details>"""


def _advanced(state: PageState, messages: str, posted: Mapping[str, str]) -> str:
    text = posted.get("config", state.config_text)
    undo = (
        f'<form method="post" action="/undo">{_token(state)}'
        '<div class="buttons"><button class="quiet">Undo the last change</button></div></form>'
        if state.can_undo
        else ""
    )
    opened = " open" if messages else ""
    return f"""<details id="advanced"{opened}><summary>Advanced: the whole settings file</summary>
<p class="help">Every setting, with notes. Lucy checks it before saving, and refuses anything
she couldn't start with.</p>{messages}
<form method="post" action="/advanced">{_token(state)}
<textarea name="config" spellcheck="false" autocapitalize="none" autocorrect="off">
{_e(text)}</textarea>
<div class="buttons"><button>Save file</button></div></form>{undo}</details>"""


def _finish(state: PageState) -> str:
    return f"""<section id="finish"><h2>Done?</h2>
<p class="help">Lucy leaves setup and wakes up with her new settings, on the Wi-Fi she knows.</p>
<form method="post" action="/finish">{_token(state)}<div class="buttons">
<button>Finish setup</button></div></form>
<p class="meta"><a href="/card">Her setup card</a>, to print and keep inside her zipper.</p>
<p class="meta">Lucy {_e(state.version)} · setup Wi-Fi {_e(state.hotspot_name)} · setup ends by
itself after {state.idle_minutes} minutes with nothing to do, or if you hold her wing for two
seconds.</p></section>"""


@dataclass(frozen=True, slots=True)
class LogView:
    """One page of Lucy's logbook, as the Conversations page shows it."""

    token: str
    conversations: Sequence[LoggedConversation]
    found: Summary
    page: int
    """From 1 (the latest conversations) to ``pages``."""
    pages: int
    flagged: bool
    """Only the conversations in which a safety screen stepped in."""
    recording: bool
    keep_days: int


def _clock_time(at: float) -> str:
    return time.strftime("%H:%M", time.localtime(at))


def _turn(turn: LoggedTurn) -> str:
    remarks = [_clock_time(turn.at)]
    if outcome := _OUTCOME_LABELS.get(turn.outcome, ""):
        remarks.append(outcome)
    if turn.flagged:
        concern = _CONCERN_LABELS.get(turn.concern, turn.concern)
        remarks.append(f"{concern} ({turn.reasons})" if turn.reasons else concern)
    if turn.brain == "local":
        remarks.append("answered offline")
    drafted = (
        f'<p class="help">The model had written: {_e(turn.drafted)}</p>' if turn.drafted else ""
    )
    said = _e(turn.said) if turn.said else '<span class="meta">(nothing)</span>'
    return f"""<div class="turn{" flagged" if turn.flagged else ""}">
<p class="meta">{_e(" · ".join(remarks))}</p>
<p><span class="who">Heard:</span> {_e(turn.heard)}</p>
<p><span class="who">Lucy:</span> {said}</p>{drafted}</div>"""


def _conversation(conversation: LoggedConversation) -> str:
    started = time.strftime("%a %d %b %Y, %H:%M", time.localtime(conversation.started))
    turns = "".join(_turn(turn) for turn in conversation.turns)
    return f"""<section><h2>{_e(started)}</h2>
<p class="meta">{_count(len(conversation.turns), "turn")}</p>{turns}</section>"""


def conversations_page(view: LogView, *, notice: str = "", error: str = "") -> str:
    """Lucy's logbook: what was said to her and what she answered, the latest first."""

    def link(page: int, flagged: bool, text: str, *, here: bool = False) -> str:
        if here:
            return f"<strong>{_e(text)}</strong>"
        parts = [f"page={page}"] if page > 1 else []
        if flagged:
            parts.append("flagged=1")
        query = "?" + "&amp;".join(parts) if parts else ""
        return f'<a href="/conversations{query}">{_e(text)}</a>'

    found = view.found
    recording = (
        (
            f"She deletes them after {_count(view.keep_days, 'day')}."
            if view.keep_days
            else "She keeps them until you delete them."
        )
        if view.recording
        else "Writing them down is turned off, so nothing new is added."
    )
    filters = " · ".join(
        (
            link(1, False, f"All ({found.conversations})", here=not view.flagged),
            link(
                1,
                True,
                f"Where a safety screen stepped in ({found.flagged})",
                here=view.flagged,
            ),
        )
    )
    if view.conversations:
        listed = "".join(_conversation(conversation) for conversation in view.conversations)
    elif view.flagged and found.turns:
        listed = "<section><p>No safety screen has stepped in.</p></section>"
    else:
        listed = "<section><p>Nothing is written down yet.</p></section>"
    newer = link(view.page - 1, view.flagged, "← Newer") if view.page > 1 else "<span></span>"
    older = link(view.page + 1, view.flagged, "Older →") if view.page < view.pages else ""
    pager = (
        f'<p class="pager">{newer}<span class="meta">Page {view.page} of {view.pages}</span>'
        f"{older or '<span></span>'}</p>"
        if view.pages > 1
        else ""
    )
    delete = (
        f"""<details><summary>Delete every conversation</summary>
<p class="help">For good: they cannot be brought back.</p>
<form method="post" action="/conversations/clear">
<input type="hidden" name="token" value="{_e(view.token)}">
<label class="check"><input type="checkbox" name="sure" required>
<span>Yes, delete all {_count(found.conversations, "conversation")}</span></label>
<div class="buttons"><button>Delete them</button></div></form></details>"""
        if found.turns
        else ""
    )
    messages = (f'<p class="notice" role="status">{_e(notice)}</p>' if notice else "") + (
        f'<p class="error" role="alert">{_e(error)}</p>' if error else ""
    )
    return _document(
        "Lucy's conversations",
        f"""<h1>&#x1FABF; Conversations</h1>
<p class="lead">What was said to Lucy and what she answered, the latest first. Times are by
her own clock. {_e(recording)}</p>
<p><a href="/">← Back to her settings</a></p>{messages}
<p>{filters}</p>{listed}{pager}{delete}""",
    )


def joining_page(network: str) -> str:
    return _document(
        "Lucy is trying the Wi-Fi",
        f"""<h1>&#x1FABF; Trying {_e(network)}…</h1>
<section><p>Your phone will lose Lucy's Wi-Fi in a moment. Listen to her:</p>
<ul><li>If it worked, she says so and goes back to being a goose. You're done.</li>
<li>If not, she says why, and her setup Wi-Fi comes back within a minute. Join it again to
try once more.</li></ul></section>""",
    )


def finished_page() -> str:
    return _document(
        "Lucy setup is done",
        """<h1>&#x1FABF; All done</h1><section><p>Lucy is waking up with her new settings. You
can close this page.</p></section>""",
    )


def stale_page() -> str:
    return _document(
        "Lucy setup",
        """<h1>&#x1FABF; Please try again</h1><section><p>This page was out of date (Lucy may
have restarted setup). <a href="/">Open it again</a>.</p></section>""",
    )
