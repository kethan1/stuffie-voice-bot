"""On-device screening of what a child said, before anything reaches the language model.

Groq's hosted guard model (Llama Guard 4) was retired in 2026 and its
replacement is a preview model with a 3-requests-per-minute free tier, so a
cloud moderator cannot sit in front of every turn. Lucy layers safety instead:

1. This screen runs locally in microseconds. It looks for signs that a child
   may be hurt, unsafe, or in distress, and answers with a fixed, caring script
   that always points to a trusted grown-up. Those words never go to the LLM.
2. The system prompt (see :mod:`lucy.persona`) keeps every other answer short,
   gentle, and age-appropriate, and steers away from off-limits topics.
3. gpt-oss models are themselves safety-trained.

False positives are acceptable by design: the scripted reply is kind and
harmless even when the child was only talking about a game.
"""

import re
from collections.abc import Iterable

from lucy.core.types import ALLOWED, Assessment, Concern

# Each pattern names *why* it matched, for the (privacy-safe) log line.
DISTRESS_PATTERNS: dict[str, str] = {
    "self-harm": (
        r"\b(?:kill|hurt|cut|harm)(?:ing)? myself\b"
        r"|\bsuicid\w*|\bend my life\b|\bwant(?:s)? to die\b"
        r"|\bdon'?t want to (?:be alive|live|wake up)\b|\bi hate myself\b"
    ),
    "being-hurt": (
        r"\b(?:hits?|hitting|hurts?|hurting|beats?|beating|kicks?|kicking|chok\w+)"
        r" me\b"
        r"|\btouch(?:es|ed|ing)? me\b"
        r"|\bkeep(?:s)? (?:it )?a secret from (?:my )?(?:mom|dad|parents)\b"
    ),
    "bullying": r"\b(?:bull(?:y|ies|ied|ying)) me\b|\bi'?m (?:being )?bullied\b",
    "lonely": r"\b(?:nobody|no one) (?:loves|likes|cares about) me\b",
    "danger": r"\b(?:stranger|someone|somebody) (?:is )?(?:following|took|grabbed) me\b",
}


_CURLY_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'"})


class DistressScreen:
    """Moderator that flags messages suggesting a child needs a trusted grown-up."""

    def __init__(self, patterns: dict[str, str] | None = None, extra: Iterable[str] = ()) -> None:
        source = dict(DISTRESS_PATTERNS if patterns is None else patterns)
        for index, pattern in enumerate(extra):
            source[f"custom-{index + 1}"] = pattern
        self._patterns = {
            name: re.compile(pattern, re.IGNORECASE) for name, pattern in source.items()
        }

    async def assess(self, text: str) -> Assessment:
        normalized = text.translate(_CURLY_QUOTES)
        reasons = tuple(name for name, rx in self._patterns.items() if rx.search(normalized))
        return Assessment(Concern.NEEDS_GROWN_UP, reasons) if reasons else ALLOWED


class AllowAll:
    """Moderator used when screening is disabled in the configuration."""

    async def assess(self, text: str) -> Assessment:
        return ALLOWED
