"""The knock: how a grown-up asks Lucy for something a child must not start by accident.

Two quick taps on her wing, then the squeeze: tap, tap, hold. The controller
(:mod:`lucy.core.controller`) notices it, and what is said during that squeeze may then be
a request that Lucy answers herself, on the Pi, with no model involved. There is one:

* "Enter into setup mode" (or "enter setup mode", "go into setup mode", "start setup")
  starts setup mode (:mod:`lucy.setup`): her own Wi-Fi and a settings page for a phone.

It takes both halves. The words alone, in an ordinary squeeze, are a question like any
other and go to the model, so a child who says them, or a story that happens to, changes
nothing. The knock alone changes nothing either: without the words it is an ordinary turn,
so a child who drums on her wing before asking a question just gets the answer.

Like the volume requests (:mod:`lucy.core.volume`), the words are matched after the safety
screen, on whatever the speech recogniser wrote: "Enter Setup Mode", "enter into set up
mode, please", "Lucy, go into set-up mode."
"""

import re

KNOCK_TAPS = 2
"""This many taps (or more) just before a squeeze make it a knock."""

_CURLY = "\N{RIGHT SINGLE QUOTATION MARK}"  # speech-to-text writes "don\u2019t" for "don't"
_MAX_WORDS = 10  # the request is short; a long sentence that mentions setup is something else
_NOT_A_REQUEST = re.compile(
    r"^(?:(?:hey|hi|ok|okay|so|um+|uh+|lucy|goose)\s+)*"
    r"(?:what|why|how|which|who|whose|when|where|is|are|was|were|does|do|did)\b"
    r"|\b(?:don'?t|do not|never|not|no)\b"
)
_SETUP = r"set ?up"  # "setup", and "set up" or "set-up" once the punctuation is gone
_THE = r"(?: (?:your|the))?"
_ASKS_FOR_SETUP = re.compile(
    rf"\b{_SETUP} mode\b"
    rf"|\b(?:enter|open|launch)(?: (?:in|into|in to))?{_THE} {_SETUP}\b"
    rf"|\b(?:start|begin){_THE} {_SETUP}\b"
    rf"|\bgo (?:in|into|in to|to){_THE} {_SETUP}\b"
)


def asks_for_setup(text: str) -> bool:
    """True if ``text`` asks Lucy to start setup mode (to be acted on only after the knock)."""
    words = re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace(_CURLY, "'")).split()
    said = " ".join(words)
    if not said or len(words) > _MAX_WORDS or _NOT_A_REQUEST.search(said):
        return False
    return bool(_ASKS_FOR_SETUP.search(said))
