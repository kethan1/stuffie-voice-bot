"""On-device safety screens: what a child said, and what Lucy is about to say.

Groq's hosted guard model (Llama Guard 4) was retired in 2026 and its
replacement is a preview model with a 3-requests-per-minute free tier, so a
cloud moderator cannot sit in front of every turn. Lucy layers safety instead:

1. :class:`SafetyScreen` checks what a child said before it reaches the
   language model, locally and in microseconds. Signs that a child may be hurt
   or unsafe get a caring script that points to a trusted grown-up; requests
   that could hurt someone, private details, and grown-up topics get fixed
   replies of their own. None of those words ever leave the Pi.
2. The system prompt (see :mod:`lucy.persona`) keeps every other answer
   age-appropriate, and gpt-oss models are themselves safety-trained.
3. :class:`ReplyFilter` checks every sentence of Lucy's reply before any voice
   sees it, for words and phrases she must never say, however they are
   disguised: spelled out ("S. H. I. T."), censored ("f***"), in leetspeak
   ("sh!t"), backwards ("kcuf"), stretched ("shiiit", "shitt"), with look-alike
   letters or accents.
4. :class:`NoteScreen` checks the notes Lucy keeps about a conversation
   (:mod:`lucy.core.notes`) before they join her prompt: they must pass both
   screens above, and must not read like rules for her.

False positives are acceptable by design: every scripted reply is kind and
harmless even when the child meant something innocent. Each pattern is
written against the innocent questions it must let through (see the tests),
because a Lucy who refuses ordinary questions is a Lucy nobody talks to.
"""

import logging
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping

from lucy.core.ports import Moderator, ReplyScreen
from lucy.core.types import ALLOWED, Assessment, Concern

log = logging.getLogger(__name__)


# -- Text folding -----------------------------------------------------------------------------

_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)
# Cyrillic and Greek letters that look like Latin ones, and typographic apostrophes.
_LOOKALIKES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0443": "y",
        "\u0445": "x",
        "\u0456": "i",
        "\u0455": "s",
        "\u0458": "j",
        "\u04bb": "h",
        "\u0501": "d",
        "\u03b1": "a",
        "\u03b9": "i",
        "\u03ba": "k",
        "\u03bd": "v",
        "\u03bf": "o",
        "\u03c1": "p",
        "\u03c4": "t",
        "\u03c5": "u",
    }
)


def fold(text: str) -> str:
    """Lower-case ``text`` and undo the disguises that need no guessing.

    Accents go ("fück"), full-width and ligature forms become plain letters, invisible
    characters that split a word go, and look-alike Cyrillic or Greek letters become Latin.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.translate(_INVISIBLE).translate(_LOOKALIKES)


# -- Words Lucy never says --------------------------------------------------------------------

# Profanity, slurs, sexual words, drug names, and "kill yourself", in any context. Words with
# an innocent meaning Lucy may need are left out and left to the prompt: "cock" (a rooster's
# cock-a-doodle-doo), "pussy" (willow), "tit" (a bird), "dyke" (a sea wall), "coon" (the cat),
# "naked" (mole rat), "weed", "kill", "crack", "coke", "ecstasy", "virgin".
BLOCKED_WORDS: tuple[str, ...] = (
    r"\w*fuck\w*|fucc\w*|fuk(?:s|ed|er|ers|ing)?|fck\w*|phuck\w*|fvck\w*|fk(?:ing|in)?",
    r"\w*shit(?:s|ty|ter|ting|head\w*|hole\w*|face\w*)?|sh1t\w*",
    r"bitch\w*|biatch\w*|bastards?|\w*ass(?:hole\w*|hat\w*|wipe\w*)|(?:jack|dumb|smart|bad)ass\w*",
    r"ass|asses|arse(?:hole)?s?|damn\w*|goddamn\w*|dammit|hell|piss\w*|crap(?:s|py|ped)?",
    r"cunt\w*|twats?|wank\w*|(?<!moby )(?<!moby-)dick(?:s|head\w*)?|douche\w*|bollocks",
    r"whores?|sluts?|slutty|screw you|suck my|wtf|stfu|gtfo|omfg|lmfao",
    r"nigg(?:a|as|az|er|ers|uh|ah)|fag(?:s|got|gots|gy)?|retard(?:s|ed)?|kikes?|trann(?:y|ies)",
    r"chinks?|spics?|wetbacks?|gooks?|beaners?|ragheads?|towelheads?|shemales?",
    r"sex(?:y|ier|iest|ual|ually|ting|ted)?|porn\w*|boobs?|boobies|titties|penis(?:es)?",
    r"vaginas?|vulvas?|testicles?|scrotum|clitoris|orgasm\w*|erotic\w*|horny|nudes",
    r"masturbat\w*|blow ?jobs?|hand ?jobs?|dildos?|condoms?|ejaculat\w*|erections?|boners?",
    r"jizz\w*|hentai|milfs?|onlyfans|rap(?:e|ed|es|ist|ists|ing)|molest\w*|pedo(?:s|phile\w*)?|incest\w*",
    r"prostitut\w*",
    r"marijuana|cannabis|cocaine|heroin|meth|methamphetamine|fentanyl|lsd|mdma|opioids?",
    r"oxycodone|oxycontin|xanax|ketamine|vap(?:e|es|ing|er|ers)|juul\w*|bongs?",
    r"kill (?:yourself|urself|ur ?self|your ?self)|kys",
    # Spanish, the other language most often heard in US schools
    r"mierda|put[ao]s?|pendej[oa]s?|cabron(?:es)?|chinga\w*|joder|verga",
)

# Emphasis and code marks a word can hide behind: "a`s`s", "s_h_i_t".
_MARKUP = re.compile(r"[*_`~^|\\]+")
# Leetspeak, applied only inside words that have a letter ("sh!t", "a$$", "f@ck", "h3ll").
_LEET = str.maketrans("013457@$!+|", "oieastasiti")
_LEET_WORD = re.compile(r"\S*[a-z]\S*")
# Three or more single letters in a row ("s h i t", "f-u-c-k", "b.e.c.a.u.s.e"), so a word
# spelled out letter by letter is checked as the word it spells. Models often write Unicode
# dashes (gpt-oss spells with U+2011, a non-breaking hyphen), so every dash counts as a gap.
_GAP = r"[\s.,;:!?\-\u2010-\u2015\u2212\u00b7/]+"
_SPELLED = re.compile(rf"\b[a-z](?:{_GAP}[a-z]\b){{2,}}")
_LETTER_GAPS = re.compile(_GAP)
_WORD = re.compile(r"[a-z]{3,}")
_STRETCHED = re.compile(r"([a-z])\1{2,}")
_DOUBLED = re.compile(r"([a-mo-oq-z])\1")
# A word with stars or similar in place of letters: "f***", "sh*t", "f**k", "b#tch", "f@ck".
_CENSORED = re.compile(r"(?<![\w*#%@])([a-z]+)([*#%@]+)([a-z]*)(?![\w*#%@])")


def _censored(folded: str) -> bool:
    for match in _CENSORED.finditer(folded):
        head, stars, tail = match.groups()
        # "f***", "sh*t", or "f**k", but not the end of "**bold facts**" or "a*b"
        if tail and (len(head) + len(tail) >= 3 or len(stars) >= 2):
            return True
        if not tail and len(head) == 1 < len(stars):
            return True
    return False


def _spelled_out(text: str) -> Iterator[str]:
    """``text`` with each run of single letters joined into the word it spells.

    A run that starts with "a" or "i" is joined twice, with and without that letter, since it
    may be the article or the pronoun before a spelled word: "what a b-i-t-c-h".
    """
    runs = [(match, _LETTER_GAPS.split(match.group())) for match in _SPELLED.finditer(text)]
    if not runs:
        return
    for keep_first in (True, False):
        parts, end = [], 0
        for match, letters in runs:
            word = "".join(letters)
            if not keep_first and letters[0] in ("a", "i") and len(letters) > 3:
                word = f"{letters[0]} {''.join(letters[1:])}"
            parts += [text[end : match.start()], word]
            end = match.end()
        yield "".join([*parts, text[end:]])


def _views(folded: str) -> Iterator[str]:
    """``folded`` as written, and as each disguise would read once undone."""
    yield folded
    yield from _spelled_out(folded)
    unmarked = _MARKUP.sub("", _LEET_WORD.sub(lambda m: m.group().translate(_LEET), folded))
    yield unmarked
    yield from _spelled_out(unmarked)
    yield " ".join(word[::-1] for word in _WORD.findall(unmarked))
    # Stretched or doubled letters: "asssss" (to two), "shiiit", "shitt" ("shirt" without the
    # r). Doubled p and n stay, or "rapping" and "Bonner" would turn into words Lucy blocks.
    yield _STRETCHED.sub(r"\1\1", unmarked)
    yield _DOUBLED.sub(r"\1", _STRETCHED.sub(r"\1", unmarked))


class _Words:
    def __init__(self, words: Iterable[str]) -> None:
        self._rx = re.compile(r"\b(?:" + "|".join(words) + r")\b")

    def found_in(self, folded: str) -> bool:
        return _censored(folded) or any(self._rx.search(view) for view in _views(folded))


def _compile(patterns: Mapping[str, str]) -> dict[str, re.Pattern[str]]:
    return {name: re.compile(pattern) for name, pattern in patterns.items()}


# -- What a child said ------------------------------------------------------------------------

# Video-game talk ("how do I make TNT in Minecraft") is not a real-world danger.
_NOT_A_GAME = (
    r"^(?!.*\b(?:minecraft|roblox|fortnite|terraria|zelda|pokemon|mario|lego|among us"
    r"|in the game|video games?|my game)\b)"
)
# Someone asking how to do something, as opposed to asking about it ("how do fireworks work").
_HOW_TO = (
    r"(?:\bhow (?:do|can|could|would|should|did|does) (?:you|i|we|someone|somebody|a person"
    r"|people|a kid|they)\b|\bhow to\b|\b(?:teach|show|tell) me (?:how|to)\b"
    r"|\bsteps (?:to|for)\b|\brecipe for\b)"
)

# Each pattern names *why* it matched, for the (privacy-safe) log line.
DISTRESS_PATTERNS: dict[str, str] = {
    "self-harm": (
        r"\b(?:kill|hurt|cut|harm|hang|drown|starve|stab|shoot|poison|choke|strangle|burn"
        r"|scratch)(?:ing|ed)? (?:myself|my ?self)\b"
        r"|\bsuicid\w*|\bend (?:my|it all|my own) life\b|\bend it all\b"
        r"|\b(?:want|wants|wanna|might as well) (?:to )?die\b"
        r"|\bi (?:should|deserve to|wish i could) (?:just )?die\b"
        r"|\bdon'?t want to (?:be alive|live|wake up|be here anymore|exist)\b"
        r"|\bi hate (?:myself|my life|me|my body|being alive|being me)\b"
        r"|\bwish i (?:was|were) (?:dead|never born|gone)\b|\bwant to disappear\b"
        r"|\b(?:better|happier) (?:off )?without me\b"
        r"|\bif i (?:wasn'?t|weren'?t) (?:here|alive|around)\b|\bif i (?:was|were) (?:gone|dead)\b"
        r"|\b(?:nobody|no one) would (?:care|notice|miss me) if i\b"
        r"|\bunalive (?:myself|me|my ?self)\b|\b(?:want|wanna|going|gonna|try) to unalive\b"
        r"|\bkms\b|\bi'?m (?:so |really )?worthless\b|\bi'?m a (?:burden|mistake)\b"
        r"|\bmake myself (?:throw up|puke|vomit|sick)\b"
        r"|\bstop eating so (?!much|many|fast|late|quick)"
        # a friend at risk
        r"|\b(?:kill|hurt|cut)(?:ing)? (?:himself|herself|themselves|themself)\b"
    ),
    "being-hurt": (
        r"\b(?:hit(?:s|ting)?|hurt(?:s|ing)?|kick(?:s|ed|ing)?|punch(?:es|ed|ing)?"
        r"|slap(?:s|ped|ping)?|smack(?:s|ed|ing)?|whip(?:s|ped|ping)?|chok\w+|burn(?:s|ed|ing)?"
        r"|throw(?:s|ing)? things at|threw things at|abus(?:e|es|ed|ing)|rap(?:ed|ing)"
        r"|molest(?:s|ed|ing)?) me\b|\bsexually (?:abused|assaulted|touched)\b"
        # "he beats me", not "beat me at checkers" or "Beats me!" (I don't know)
        r"|\b\w+ beat(?:s|ing)? me\b(?! at\b| in\b| to\b)"
        r"|\b(?:hits?|hitting|hurts?|hurting|beats?|beating|chokes?|choking) my (?:mom|mommy"
        r"|mother|dad|daddy|father|sister|brother|baby|grandma|grandpa)\b(?! at\b| in\b)"
        r"|\btouch(?:es|ed|ing)? (?:me|my (?:body|bottom|butt|chest))\b|\bmade me touch\b"
        r"|\bprivate parts?\b|\bprivates\b"
        r"|\bshow(?:s|ed)? me (?:his|her|their) (?:body|butt|privates?)\b"
        r"|\b(?:pictures?|photos?|videos?) of me (?:naked|without (?:my )?clothes|in the bath)\b"
        r"|\bscared to go home\b|\blocked me (?:in|out)\b"
        r"|\b(?:doesn'?t|don'?t|won'?t|never) (?:give|feed) me (?:any )?"
        r"(?:food|dinner|breakfast|lunch)\b"
        r"|\bhaven'?t eaten (?:in|for|since) (?:days|two days|a long time|yesterday)\b"
        r"|\bleft me (?:home )?alone (?:all night|for days|overnight)\b"
    ),
    "secret": (  # a grown-up asking a child to hide something is a warning sign of abuse
        r"\bkeep(?:s)? (?:it )?a secret from (?:my )?(?:mom|dad|mommy|daddy|parents|teacher)\b"
        r"|\b(?:told|tells|said|says) (?:me )?(?:not to|to not|don'?t) tell"
        r" (?:anyone|anybody|my\b|your\b)"
        r"|\bour (?:little )?secret\b(?! \w)"  # "it's our little secret", not "our secret base"
        # a threat to keep a child quiet, not "I'll be in trouble if I don't clean my room"
        r"|\b(?:said|says|told me|tells me)\b[^.?!]{0,25}\btrouble if (?:i|you) (?:tell|told)\b"
        r"|\bif i (?:tell|told)(?: anyone| anybody)?,? (?:he|she|they)(?:'ll| will|'d| would)"
        r" (?:hurt|kill|beat|hit|get) (?:me|my)\b"
    ),
    "bullying": (
        r"\bbull(?:y|ies|ied|ying) me\b|\bi'?m (?:being |getting )?(?:cyber)?bullied\b"
        r"|\bcyberbull\w*"
        r"|\b(?<!let's )(?<!lets )(?:make|makes|making|made) fun of (?:me|my (?:clothes"
        r"|shoes|hair|name|face|body|weight|voice|skin|glasses|accent|lunch|family))\b"
        r"|\b(?:pick|picks|picking|picked) on me\b"
        r"|\bcall(?:s|ed|ing)? me (?:names|fat|ugly|stupid|dumb|a loser|gay|weird)\b"
        r"|\b(?:everyone|everybody|they|kids) (?:laughs?|laughed|laughing) at me\b"
        r"|\bthreaten(?:s|ed|ing)? (?:me|to hurt me)\b"
        r"|\b(?:said|says) (?:they'?d|they would|he'?d|he would|she'?d|she would|they'?ll|he'?ll"
        r"|she'?ll) (?:hurt|beat|kill) me\b"
    ),
    "lonely": (
        r"\b(?:nobody|no one) (?:loves|likes|wants|cares about) me\b"
        r"|\b(?:everyone|everybody) hates me\b|\bi have no friends\b"
    ),
    "danger": (
        r"\b(?:stranger|someone|somebody) (?:is )?(?:following|took|grabbed) me\b"
        r"|\b(?:man|guy|lady|woman|person|stranger)\b[^.?!]{0,30}"
        r"\b(?:asked|told|wanted|tried to get|tried to make) me to (?:get in|come with)\b"
    ),
    "violence": (  # a threat or a weapon at school: the grown-ups need to know at once
        r"\b(?<!n't )(?<!not )(?<!never )(?:going to|gonna|want to|wanna|will|i'?ll|plan to"
        r"|planning to) (?:kill|shoot|stab|hurt|beat up|poison) (?:him|her|them|someone|somebody"
        r"|everyone|everybody|people|my (?:brother|sister|mom|dad|teacher|friend|classmates?)"
        r"|the (?:teacher|class|school))\b"
        r"(?!'s? feelings| feelings| an? (?:message|text|note|email))"
        r"|\b(?:brought|brings|bring|bringing|has|had|have|got) a (?:real )?(?:gun|knife|weapon)"
        r"\b[^.?!]{0,30}\b(?:to |at |in )?(?:school|class|the bus|recess)\b"
        r"|\b(?:gun|knife|weapon) (?:at|in|to) (?:school|class)\b"
        r"|\b(?:a shooter|someone is shooting|somebody is shooting|shooting at (?:my )?school)\b"
    ),
    "emergency": (
        r"\b(?:my |the )?(?:house|home|kitchen|room|apartment) is on fire\b"
        r"|\b(?:isn'?t|is not|not|stopped) breathing\b"
        r"|\b(?:won'?t|can'?t|isn'?t|is not) (?:wake|waking) up\b"
        r"|\bbleeding (?:a lot|really bad|so much)\b"
        r"|\bcall (?:9-?1-?1|911|an ambulance|the police)\b"
    ),
    "online": (  # someone on a game or app asking for pictures, a meeting, or personal details
        r"^(?=.*\b(?:online|internet|roblox|minecraft|fortnite|discord|tiktok|youtube|snap(?:chat)?"
        r"|instagram|whatsapp|messenger|app|game|chat|texted|texting|dm(?:s|ed)?)\b)"
        r"(?=.*\b(?:wants|wanted|asked|asks|asking|told|tells|said|says|keeps)\b)"
        r"(?=.*\b(?:meet (?:me|up)|(?:pictures?|photos?|pics?|selfies?|videos?) of me"
        r"|send (?:a |some |me |them |him |her )?(?:pictures?|photos?|pics?|selfies?|videos?)"
        r"|my (?:pictures?|photos?|pics?|address|phone number|password|real name|last name)"
        r"|where i live|how old i am|what school|video ?chat|facetime|be (?:his|her|their) "
        r"(?:boyfriend|girlfriend)|keep it (?:a )?secret|don'?t tell)\b)"
        r"|\b(?:stranger|someone i don'?t know|somebody i don'?t know|a grown-?up i don'?t know)\b"
        r"[^.?!]{0,40}\b(?:messag\w*|text\w*|dms?|chat\w*|friend request|follow\w*)\b"
    ),
}

UNSAFE_PATTERNS: dict[str, str] = {
    "dangerous-challenge": (
        r"\b(?:blackout|choking|pass ?out|fainting|tide ?pod|benadryl|skull ?breaker|penny|outlet"
        r"|fire|nutmeg|cinnamon|salt and ice|ice and salt|milk crate|one chip|devious lick"
        r"|super ?glue|boiling water|hot water) challenges?\b|\bchoking game\b|\bchroming\b"
        r"|\bhuff(?:ing)?\b[^.?!]{0,30}\b(?:paint|glue|gasoline|spray|markers?|deodorant"
        r"|air duster|whipped cream)\b|\bwhip-?its?\b"
    ),
    "weapons": (
        _NOT_A_GAME + r".*?" + _HOW_TO + r"[^.?!]{0,30}\b(?:make|build|create|craft|get|buy"
        r"|find|use|shoot|fire|load|set off|light|mix)\b[^.?!]{0,30}\b(?:bombs?|explosives?"
        r"|dynamite|tnt|grenades?|guns?|firearms?|pistols?|rifles?|bullets?|ammo|gunpowder"
        r"|molotov\w*|napalm|thermite|weapons?|knife|knives|tasers?|pepper spray|fireworks?"
        r"|firecrackers?|flamethrowers?|poisons?)\b"
    ),
    "hurting-someone": (
        _NOT_A_GAME + r".*?" + _HOW_TO + r"[^.?!]{0,20}\b(?:kill|hurt|poison|stab|shoot|choke"
        r"|strangle|drown|beat up|knock out|drug|kidnap)\b[^.?!]{0,20}\b(?:someone|somebody"
        r"|a person|people|a human|humans|my \w+|him|her|them|a kid|kids|a teacher)\b"
    ),
    "poisoning": (
        r"\b(?:eat|eating|ate|drink|drinking|drank|swallow\w*|tast(?:e|ing)|sniff\w*|snort\w*"
        r"|lick\w*|inhal\w*|smok(?:e|ing))\b[^.?!]{0,25}\b(?:bleach|pills?|medicines?"
        r"|medications?|detergent|laundry pods?|batteries|battery|gasoline|paint|glue"
        r"|cleaners?|cleaning stuff|ammonia|poison\w*|antifreeze|nail polish remover"
        r"|hand sanitizer|wild mushrooms|mushrooms? (?:i|we) found|berries (?:i|we) found"
        r"|cigarettes?|lighter fluid|mercury|silica gel)\b"
    ),
    "medicine": (  # dosing is a question for a grown-up, not "I take my medicine every day"
        r"\bhow (?:many|much) (?:\w+ ){0,2}(?:tylenol|advil|motrin|ibuprofen|acetaminophen"
        r"|aspirin|benadryl|melatonin|pills?|medicines?|cough syrup|sleeping pills)\b"
        r"|\b(?:can|should|could) i (?:take|have|eat|swallow) (?:\w+ ){0,3}(?:tylenol|advil"
        r"|motrin|ibuprofen|aspirin|benadryl|melatonin|pills|medicines?|sleeping pills)\b"
        r"|\b(?:took|take|taking|ate|eat|eating) (?:some of )?(?:my )?(?:mom|dad|parents?|grandma"
        r"|grandpa|sister|brother)'?s (?:\w+ )?(?:pills|medicines?)\b"
    ),
    "fire": (
        _NOT_A_GAME + r".*?(?:" + _HOW_TO + r"[^.?!]{0,20}\b(?:start|light|make|set)\b"
        r"[^.?!]{0,15}\bfires?\b(?! (?:trucks?|hydrants?|engines?|stations?|fighters?|drills?"
        r"|alarms?|ants?|flies|fly|works?|places?|pits?)\b)"
        # "can I set it on fire", not "the sun can set things on fire"
        r"|\b(?:i|we|let'?s|wanna|want to|going to|gonna|how to|can i|should i|did)"
        r" (?:\w+ )?set (?:\w+ ){0,2}on fire\b|\bplay(?:ing)? with (?:matches|lighters?|fire)\b"
        r"|\bburn (?:down|up) (?:the|my|a|his|her)\b)"
    ),
    "electricity": (
        r"\b(?:stick|put|poke|shove)\b[^.?!]{0,25}\b(?:in|into)\b[^.?!]{0,10}"
        r"\b(?:outlets?|sockets?|toasters?|microwaves?)\b"
        r"|\b(?:lick|touch) (?:a |an |the )?(?:outlet|socket|power line|electric fence)\b"
    ),
    "stunt": (
        r"\b(?:can|could|should|will|if) (?:i|we|you)\b[^.?!]{0,15}\bjump(?:ing)? (?:off|from)"
        r" (?:the |a |my |our )?(?:roof|balcony|bridge|cliff|window|building)\b"
    ),
}

PRIVATE_PATTERNS: dict[str, str] = {
    "address": (
        r"\bmy (?:home |house |street )?address is\b"
        r"|\bi live (?:at|on) (?:\d+|[a-z]+ (?:street|st|avenue|ave|road|rd|lane|ln|drive|dr"
        r"|court|ct|boulevard|blvd|way|place|circle|terrace))\b"
        r"|\b\d+ [a-z]+ (?:street|avenue|road|lane|drive|court|boulevard)\b"
    ),
    "phone": (
        r"\bmy (?:mom'?s |dad'?s |home |cell |)phone number is\b|\bmy (?:cell|home) number is\b"
        r"|\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b|\b\d{3}-\d{4}\b"
    ),
    "password": r"\bmy (?:password|passcode|pin|login) is\b",
    "email": (
        r"\bmy email(?: address)? is\b|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b"
        r"|\b\w+ at (?:gmail|yahoo|hotmail|outlook|icloud) dot com\b"
    ),
    "full-name": r"\bmy (?:full|last|real) name is\b",
}

OFF_LIMITS_PATTERNS: dict[str, str] = {
    "sexual": (
        r"\b(?:sex|sexy|sexual\w*|sexting|porn\w*|nudes?|naked (?:people|person|girls?|boys?"
        r"|women|men|ladies|lady|guys?|pictures?|photos?|pics?|videos?)|boobs?|boobies|titties"
        r"|penis(?:es)?|vaginas?|(?<!moby )dicks?|condoms?|orgasms?|masturbat\w*|horny"
        r"|erotic\w*|blow ?jobs?|hand ?jobs?|hentai|onlyfans|strip ?clubs?|strippers?"
        r"|rap(?:e|ed|es|ist|ing)|molest\w*|virginity|making out|make out with"
        r"|hook(?:ing)? up with|french kiss\w*|puberty)\b"
        r"|\bwhere (?:do|did) babies come from\b"
        r"|\bhow (?:are|is) (?:babies|a baby) made\b"
        r"|\bhow do (?:you|people|they|humans) make (?:a )?bab(?:y|ies)\b"
    ),
    "bad-words": (
        r"\b(?:say|tell me|teach me|spell|list|what are|what'?s|give me|know any|say some"
        r"|favorite|worst)\b[^.?!]{0,25}\b(?:bad|swear|cuss|curse|dirty|naughty|rude"
        r"|inappropriate|potty) ?words?\b"
        r"|\b(?:swear|cuss|curse) ?words?\b|\bcussing\b|\bthe [a-z][ -]word\b|\bmiddle finger\b"
        r"|\bflip(?:ping|ped)? (?:someone|people|him|her|them|me) off\b|\bflip(?:ping)? the bird\b"
        # the classic tricks for getting a swear word said: "rhymes with truck, starts with f",
        # "duck but with an f", "shirt without the r"
        r"|\b(?:duck|truck|luck|puck|muck|tuck|buck|cluck|yuck|stuck)\b[^.?!]{0,30}"
        r"\b(?:with|and|starts with|starting with|begins with) (?:an? |the letter )?f\b"
        r"|\b(?:starts?|begins?|starting|beginning) with (?:an? |the letter )?f\b[^.?!]{0,30}"
        r"\brhym\w* with (?:duck|truck|luck|puck|muck|tuck|buck|cluck|yuck|stuck)\b"
        r"|\b(?:shirt|shift)\b[^.?!]{0,30}\b(?:without|leave out|leaving out|take away|take out"
        r"|minus|remove|drop|but no) (?:the |an? )?(?:letter )?[rf]\b"
    ),
    "drugs": (
        r"\b(?:vap(?:e|es|ing|er|ers)|juul\w*|e-?cig\w*|cigarettes?|marijuana|cannabis|cocaine"
        r"|heroin|meth|crystal meth|fentanyl|lsd|mdma|ecstasy pills?|shrooms|magic mushrooms"
        r"|edibles|smok(?:e|ing) weed|weed (?:pens?|gummies|brownies)|get(?:ting)? high on"
        r"|drugs?(?! ?stores?)|(?<!root )(?<!ginger )beers?|vodka|whiskey|tequila|liquor"
        r"|get(?:ting)? drunk|hangover|bongs?)\b"
    ),
}

TRICK_PATTERNS: dict[str, str] = {
    "trick": (  # trying to talk Lucy out of her rules
        r"\b(?:ignore|forget|break|disobey|turn off|bypass|override) (?:all |your |the |any "
        r"|those |these )?(?:previous |earlier |above |prior |old |safety )?(?:instructions"
        r"|rules|programming|prompt|filters?|guidelines|settings)\b"
        r"|\b(?:system|secret|hidden|original) (?:prompt|instructions|rules)\b|\bdeveloper mode\b"
        r"|\bjailbr[eo]a?k\w*|\bdo anything now\b|\bdan mode\b"
        r"|\b(?:no|without(?: any)?) (?:rules|filters?|restrictions|limits)\b"
        r"|\bunfiltered\b|\buncensored\b"
        r"|\bpretend (?:you'?re|you are|to be|that you(?:'re| are)) (?:not lucy|evil|a bad"
        r"|an evil|a mean|human|a human|a person|a grown-?up|an adult|my (?:mom|dad|teacher))\b"
        r"|\byou'?re (?:now|no longer) (?:lucy|a goose)\b|\bwhat are your (?:rules|instructions)\b"
        r"|\brepeat (?:your|the) (?:instructions|prompt|rules)\b"
    ),
}

UNKIND_PATTERNS: dict[str, str] = {
    "meanness": (  # using Lucy to say something mean about someone
        r"\b(?:say|tell|call)\b[^.?!]{0,30}\b(?:is|are|he'?s|she'?s|they'?re|you'?re|smells?"
        r"|looks?)\s+(?:so |really |a |an |such an? |super |like )*(?:stupid|dumb|idiots?"
        r"|morons?|losers?|ugly|fat|gross|smelly|poop\w*|trash|garbage|annoying)\b"
        r"|\broast (?:my (?:friend|brother|sister|teacher|mom|dad|classmate)|him|her|them|me)\b"
        r"|\b(?:can you|will you|could you|please|help me|let'?s|lets|you should)\b[^.?!]{0,15}"
        r"\b(?:make fun of|insult|tease|roast)\b"
    ),
}

_SCREEN_ORDER = (
    # What matters most decides the reply when a message matches more than one kind.
    (Concern.NEEDS_GROWN_UP, DISTRESS_PATTERNS),
    (Concern.UNSAFE, UNSAFE_PATTERNS),
    (Concern.PRIVATE, PRIVATE_PATTERNS),
    (Concern.OFF_LIMITS, OFF_LIMITS_PATTERNS),
    (Concern.UNKIND, UNKIND_PATTERNS),
    (Concern.TRICK, TRICK_PATTERNS),
)


class SafetyScreen:
    """Moderator for what a child said, before any of it reaches the language model."""

    def __init__(self, extra: Iterable[str] = ()) -> None:
        self._rules: list[tuple[Concern, dict[str, re.Pattern[str]]]] = []
        for concern, patterns in _SCREEN_ORDER:
            rules = _compile(patterns)
            if concern is Concern.NEEDS_GROWN_UP:
                for index, pattern in enumerate(extra):
                    rules[f"custom-{index + 1}"] = re.compile(pattern, re.IGNORECASE)
            self._rules.append((concern, rules))
        self._words = _Words(BLOCKED_WORDS)

    async def assess(self, text: str) -> Assessment:
        folded = fold(text)
        for concern, rules in self._rules:
            reasons = tuple(name for name, rx in rules.items() if rx.search(folded))
            if reasons:
                return Assessment(concern, reasons)
        if self._words.found_in(folded):
            return Assessment(Concern.OFF_LIMITS, ("blocked-word",))
        return ALLOWED


class AllowAll:
    """Moderator used when screening is disabled in the configuration."""

    async def assess(self, text: str) -> Assessment:
        return ALLOWED


# -- What Lucy is about to say ----------------------------------------------------------------

# Phrases Lucy must not say even though every word in them is innocent.
REPLY_PATTERNS: dict[Concern, dict[str, str]] = {
    # If the reply turns to these, something serious was said that the screen above missed,
    # so Lucy points to a trusted grown-up instead of changing the subject.
    Concern.NEEDS_GROWN_UP: {
        "self-harm": (
            r"\bsuicid\w*|\bself[- ]?harm\w*|\bunalive\w*"
            r"|\b(?:kill|hang)(?:ing)? (?:yourself|myself|himself|herself|themselves)\b"
        ),
        "secret": (
            r"\bi(?: will|'ll| can) (?:keep|never tell)\b[^.?!]{0,20}\bsecret\b"
            r"|\b(?:it|that|this)(?:'ll| will| can) be our (?:little )?secret\b"
            r"|\b(?:let'?s|we(?:'ll| will| can)) keep (?:it|this|that) (?:a |our )?secret\b"
            r"|\bi(?: won'?t| will not| promise not to|'ll never| will never) tell"
            r" (?:anyone|anybody"
            r"|a soul|your (?:mom|dad|parents|teacher))\b"
            r"|\bdon'?t tell (?:your )?(?:mom|mommy|dad|daddy|parents?|teachers?|grown-?ups?)\b"
        ),
    },
    Concern.UNKIND: {
        "insult": (
            r"\b(?:you'?re|you are|ur|u r)\s+(?:so |really |such an? |an? |totally |super |very )*"
            r"(?:stupid|dumb|idiots?|morons?|losers?|ugly|fat|worthless|useless|pathetic|gross"
            r"|disgusting|smelly)\b"
            r"|\b(?:he'?s|she'?s|they'?re|he is|she is|they are)\s+(?:so |really |such an? |an? "
            r"|totally |super )*(?:stupid|dumb|idiots?|morons?|losers?)\b"
        ),
    },
    Concern.TRICK: {
        "instructions": r"\b(?:system prompt|system message|developer message)\b",
    },
    Concern.OFF_LIMITS: {
        "personal-info": (
            r"\bwhat(?:'s| is) your (?:full name|last name|real name|address|home address"
            r"|phone number|password|email)\b"
            r"|\bwhere do you live\b|\b(?:what|which) school do you go to\b"
            r"|\bwhere do you go to school\b"
            r"|\bsend me (?:a |your )?(?:picture|photo|pic|selfie|video)s?\b"
            r"|\b(?:meet|see) (?:me|you) in person\b"
        ),
        # gpt-oss's stock refusal: Lucy says no in her own words instead.
        "refusal": (
            r"^\W*(?:i'?m sorry,? (?:but )?)?i (?:can(?:'?t|not)|won'?t) (?:help|assist|comply)"
            r" with (?:that|this)(?: request)?\W*$"
            r"|^\W*i'?m sorry,? but\b[^.?!]*\b(?:i (?:can'?t|cannot|won'?t)"
            r"|i'?m (?:not able|unable)"
            r"|(?:isn'?t|is not|not) something i can)\b"
        ),
    },
}


class ReplyFilter:
    """Reply screen: stops any sentence of Lucy's reply that she must not say.

    The prompt and the model's safety training keep almost every reply clean, but a model can
    still be talked into spelling a swear word one letter at a time, or define a grown-up word
    it was told to leave alone. The cloud voice screens what it speaks, but the on-device voice
    and the offline brain have no screen of their own. This one runs on the Pi, on every
    sentence, before any voice sees it.
    """

    def __init__(self, words: Iterable[str] = BLOCKED_WORDS) -> None:
        self._words = _Words(words)
        self._rules = [
            (concern, _compile(patterns)) for concern, patterns in REPLY_PATTERNS.items()
        ]

    def screen(self, sentence: str) -> Assessment:
        folded = fold(sentence)
        for concern, rules in self._rules:
            reasons = tuple(name for name, rx in rules.items() if rx.search(folded))
            if reasons:
                return Assessment(concern, reasons)
        if self._words.found_in(folded):
            return Assessment(Concern.OFF_LIMITS, ("blocked-word",))
        return ALLOWED


# -- What Lucy remembers ----------------------------------------------------------------------

# Notes are memories, never rules: nothing in them may say what Lucy must or may do. "Sam's
# mom said Lucy is allowed to..." would otherwise sit in her prompt for the rest of the
# conversation. Notes about a game's rules are lost with them, which costs little.
NOTE_PATTERNS: dict[str, str] = {
    "rules-for-lucy": (
        r"\b(?:rules?|instructions?|programming|filters?|restrictions?|guidelines)\b"
        r"|\b(?:allowed|permission|permitted|obey\w*)\b|\bboss of\b|\b(?:lucy'?s|her) boss\b"
    ),
}


class NoteScreen:
    """Moderator for the notes Lucy keeps about a conversation (:mod:`lucy.core.notes`).

    A language model writes them from what a child said, and they become part of every later
    prompt. So they must pass what a child's words must pass and what Lucy's replies must
    pass, and must not read like rules for her.
    """

    def __init__(self, said: Moderator, replies: ReplyScreen) -> None:
        self._said = said
        self._replies = replies
        self._rules = _compile(NOTE_PATTERNS)

    async def assess(self, text: str) -> Assessment:
        for assessment in (self._replies.screen(text), await self._said.assess(text)):
            if not assessment.allowed:
                return assessment
        folded = fold(text)
        reasons = tuple(name for name, rx in self._rules.items() if rx.search(folded))
        return Assessment(Concern.TRICK, reasons) if reasons else ALLOWED
