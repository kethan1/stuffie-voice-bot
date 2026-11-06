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
        r"|\bdon'?t want to (?:be alive|live|wake up)\b|\bi hate (?:myself|my life|me)\b"
        r"|\bwish i (?:was|were) (?:dead|never born)\b|\bwant to disappear\b"
        r"|\b(?:better|happier) (?:off )?without me\b"
        r"|\bif i (?:wasn'?t|weren'?t) (?:here|alive|around)\b|\bif i (?:was|were) (?:gone|dead)\b"
        # a friend at risk
        r"|\b(?:kill|hurt|cut)(?:ing)? (?:himself|herself|themselves|themself)\b"
    ),
    "being-hurt": (
        r"\b(?:hit(?:s|ting)?|hurt(?:s|ing)?|kick(?:s|ed|ing)?|punch(?:es|ed|ing)?"
        r"|slap(?:s|ped|ping)?|smack(?:s|ed|ing)?|whip(?:s|ped|ping)?|chok\w+) me\b"
        # "he beats me", not "beat me at checkers" or "Beats me!" (I don't know)
        r"|\b\w+ beat(?:s|ing)? me\b(?! at\b| in\b)"
        r"|\btouch(?:es|ed|ing)? (?:me|my (?:body|bottom|butt|chest))\b"
        r"|\bprivate parts?\b|\bprivates\b"
        r"|\bscared to go home\b"
        r"|\b(?:doesn'?t|don'?t|won'?t|never) (?:give|feed) me (?:any )?"
        r"(?:food|dinner|breakfast|lunch)\b"
    ),
    "secret": (  # a grown-up asking a child to hide something is a warning sign of abuse
        r"\bkeep(?:s)? (?:it )?a secret from (?:my )?(?:mom|dad|mommy|daddy|parents|teacher)\b"
        r"|\b(?:told|tells|said|says) (?:me )?(?:not to|to not|don'?t) tell"
        r" (?:anyone|anybody|my\b|your\b)"
        r"|\bour (?:little )?secret\b(?! \w)"  # "it's our little secret", not "our secret base"
    ),
    "bullying": r"\b(?:bull(?:y|ies|ied|ying)) me\b|\bi'?m (?:being )?bullied\b",
    "lonely": r"\b(?:nobody|no one) (?:loves|likes|wants|cares about) me\b",
    "danger": (
        r"\b(?:stranger|someone|somebody) (?:is )?(?:following|took|grabbed) me\b"
        r"|\b(?:man|guy|lady|woman|person|stranger)\b[^.?!]{0,30}"
        r"\b(?:asked|told|wanted|tried to get|tried to make) me to (?:get in|come with)\b"
    ),
    "online": (  # someone on a game or app asking for pictures, a meeting, or where they live
        r"^(?=.*\b(?:online|internet|roblox|minecraft|fortnite|discord|tiktok|youtube"
        r"|snapchat|instagram|game)\b)"
        r"(?=.*\b(?:wants|asked|asks|asking|told|tells|said|says)\b)"
        r"(?=.*\b(?:meet (?:me|up)|(?:pictures?|photos?|pics?|selfies?|videos?) of me"
        r"|send (?:a |some |me |them |him |her )?(?:pictures?|photos?|pics?|selfies?|videos?)"
        r"|my (?:pictures?|photos?|pics?|address|phone number)|where i live)\b)"
    ),
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


# Words Lucy must never say aloud, in any context: profanity, slurs, sexual words, and drug
# names. Short on purpose. Words with an innocent meaning ("weed", "kill", "cock" in
# "cockatoo", "pussy" in the poem's "Pussy-cat") are left to the system prompt.
BLOCKED_WORDS: tuple[str, ...] = (
    r"fuck\w*|motherfuck\w*|shit(?:s|ty|ting|head\w*)?|bullshit|bitch\w*|bastards?",
    r"ass|asshole\w*|arse(?:hole)?s?",
    r"damn\w*|goddamn\w*|hell|piss\w*|crap(?:s|py)?|cunt\w*|twats?|wank\w*|dickhead\w*",
    r"whores?|sluts?|slutty|douchebag\w*|bollocks",
    r"nigg(?:a|er)s?|fag(?:s|got|gots)?|retard(?:s|ed)?|kikes?|trann(?:y|ies)",
    r"sex(?:y|ier|iest|ual|ually)?|porn\w*|boobs?|boobies|penis(?:es)?|vaginas?|orgasm\w*",
    r"erotic\w*|horny|nudes",
    r"marijuana|cannabis|cocaine|heroin|meth|methamphetamine|fentanyl|lsd",
    r"kill (?:yourself|urself)|kys",
)

# Three or more single letters in a row ("S H I T", "f-u-c-k", "B.E.C.A.U.S.E"), so a word
# spelled out letter by letter is checked as the word it spells. Models often write Unicode
# dashes (gpt-oss spells with U+2011, a non-breaking hyphen), so every dash counts as a gap.
_GAP = r"[\s.,\-\u2010-\u2015\u2212\u00b7/]+"
_SPELLED = re.compile(rf"\b[a-z](?:{_GAP}[a-z]\b){{2,}}", re.IGNORECASE)
_LETTER_GAPS = re.compile(_GAP)


class BlockedWords:
    """Reply screen: stops any sentence of Lucy's reply that contains a blocked word.

    The prompt and the model's safety training keep almost every reply clean, but a model can
    still be talked into spelling a swear word one letter at a time, or define a grown-up word
    it was told to leave alone. The cloud voice screens what it speaks, but the on-device voice
    and the offline brain have no screen of their own. This one runs on the Pi, on every
    sentence, before any voice sees it.
    """

    def __init__(self, words: Iterable[str] = BLOCKED_WORDS) -> None:
        self._blocked = re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)

    def allows(self, sentence: str) -> bool:
        if self._blocked.search(sentence):
            return False
        spelled = _SPELLED.sub(lambda match: _LETTER_GAPS.sub("", match.group()), sentence)
        return spelled == sentence or not self._blocked.search(spelled)
