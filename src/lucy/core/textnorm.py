"""Turn model output into text that sounds right when spoken.

The system prompt asks for plain spoken sentences, but models still slip in
markdown, emoji, or links now and then. Anything a speech engine would read
out literally ("asterisk asterisk") or mangle is removed here.
"""

import re
import unicodedata

_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+|www\.\S+")
_LIST_MARKER = re.compile(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+")
_HEADING = re.compile(r"(?m)^\s*#{1,6}\s*")
_EMPHASIS = re.compile(r"[*_`~#>|]+")
_SPACE = re.compile(r"\s+")
_EMOJI_GLUE = frozenset({"\u200d", "\ufe0f"})
_REPLACEMENTS = {"&": " and ", "@": " at ", "+": " plus ", "=": " equals "}


def to_speakable(text: str) -> str:
    """Return ``text`` cleaned for speech, or an empty string if nothing is left to say."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = _HEADING.sub("", text)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = "".join(_speakable_char(ch) for ch in text)
    text = _SPACE.sub(" ", text).strip()
    return text if any(ch.isalnum() for ch in text) else ""


def _speakable_char(ch: str) -> str:
    if ch in _REPLACEMENTS:
        return _REPLACEMENTS[ch]
    category = unicodedata.category(ch)
    # So: emoji and pictographs. Cs/Co/Cn: surrogates, private use, unassigned.
    # U+200D (zero-width joiner) and U+FE0F (variation selector) glue emoji together.
    if category in {"So", "Cs", "Co", "Cn"} or ch in _EMOJI_GLUE:
        return " "
    return ch
