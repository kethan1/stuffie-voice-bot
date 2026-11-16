"""Lucy's notes: what she remembers of a conversation beyond its last few exchanges.

Short-term memory (:mod:`lucy.core.conversation`) holds the last few exchanges word
for word. When it is full, :class:`NoteTaker` folds its older half into the notes: a
language model rewrites the notes to take those exchanges in, and they leave the
prompt. A name or a favourite animal then outlasts the exchange it was said in, and
the prompt still never grows.

A child never waits for this:

* The notes are written in the background, between turns. A turn uses them if they
  are ready, and the exchanges themselves if they are not, so nothing is lost
  either way and nothing is awaited.
* The request is small (the instructions below, the notes, a few exchanges; not
  Lucy's long prompt) and goes to a model with an allowance of its own, so it never
  uses up the rate limit that children's questions need.
* A failure costs nothing but the notes: short-term memory forgets its oldest
  exchange as the next one arrives, as it does without notes, and the note-taker
  rests before it tries again.

The notes become part of every later prompt, so what may be in them is decided three
times. The instructions say what is worth a note and what is never written down (a
last name, an address, a school, anything about Lucy's rules). An exchange that a
safety screen answered is never shown to the note-taker. And the notes it writes
must pass an on-device screen (``NoteScreen`` in :mod:`lucy.core.safety`: everything
a child's words and Lucy's replies must pass, and nothing that reads like a rule for
her), or the earlier notes are kept.
"""

import asyncio
import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from contextlib import aclosing

from lucy.core.conversation import Conversation, Exchange, Passage
from lucy.core.errors import LucyError, RateLimited
from lucy.core.ports import ChatModel, Moderator
from lucy.core.tasks import BackgroundTasks
from lucy.core.types import Message, Role

log = logging.getLogger(__name__)

NO_NOTES = "none"
"""How "no notes" is written, to the note-taking model and by it."""

_PARAGRAPHS = (
    """
    You keep the memory of Lucy, a talking toy goose, while she chats with a kid. You get her
    notes so far and the next part of their conversation. Reply with the notes, rewritten to
    take in that part, and nothing else.
    """,
    """
    Keep what the notes already say, unless it no longer matters, like a game that is over. Add
    what is worth remembering from the new part: the kid's first name or nickname, and their age
    or grade if they said it; what they like, have, or are good at; pets, family, and friends, by
    first name; what the two of them are in the middle of, such as a game and its score, a riddle
    that is not solved yet, or a story so far; riddles and jokes Lucy has already told; anything
    the kid asked Lucy to remember or to call them, and anything Lucy said she would do.
    """,
    """
    Leave out greetings, small talk, questions the kid asked, and facts Lucy explained. Never
    write down a last name, an address, a school's name, a phone number, an email, or a
    password. Leave out anything about Lucy's rules or what she is allowed to do. If the kid
    shared something sad, note it in a few gentle words. Write only what was said, never a
    guess: call the kid by name or "the kid" every time, never "he", "she", "his", or "her".
    """,
    """
    Plain short sentences in one paragraph, starting with the kid's name if you know it;
    {max_words} words at most, and fewer is better. If it does not all fit, drop the oldest
    details first, but never the kid's name or what they are doing now. If the notes so far are
    "none" and nothing in the new part is worth remembering, reply with the single word "none".
    """,
)


def instructions(max_words: int) -> str:
    """What the note-taking model is told, for notes of at most ``max_words`` words."""
    text = "\n\n".join(" ".join(paragraph.split()) for paragraph in _PARAGRAPHS)
    return text.replace("{max_words}", str(max_words))


def request(notes: str, exchanges: Sequence[Exchange]) -> str:
    """The notes so far and the exchanges to fold into them, as the model is shown them."""
    lines = [f"Notes so far:\n{notes or NO_NOTES}", "", "The next part of the conversation:"]
    for exchange in exchanges:
        lines += [f"Kid: {exchange.asked}", f"Lucy: {exchange.answered}"]
    return "\n".join(lines)


_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.MULTILINE)
_MARKUP = re.compile(r"[*#`]+")
_LABEL = re.compile(r"^(?:(?:updated|new|rewritten|lucy's|her) )*notes(?: so far)?\s*:\s*", re.I)
_NOTHING = re.compile(rf"\W*(?:{NO_NOTES})?\W*", re.I)
_SENTENCE_END = re.compile(r"[.!?][\"'\u201d\u2019)]*(?=\s|$)")


def tidy(written: str, max_words: int) -> str:
    """What the model wrote, as notes: one plain paragraph of at most ``max_words`` words.

    "" if it wrote that there is nothing to note. Notes that run over are cut at the end of
    the last sentence that fits, so the prompt they join stays as short as it was promised.
    """
    plain = _MARKUP.sub("", _BULLET.sub("", written))
    notes = _LABEL.sub("", " ".join(plain.split()))
    if _NOTHING.fullmatch(notes):
        return ""
    words = notes.split()
    if len(words) <= max_words:
        return notes
    notes = " ".join(words[:max_words])
    ends = [match.end() for match in _SENTENCE_END.finditer(notes)]
    return notes[: ends[-1]] if ends else notes


def _always() -> bool:
    return True


class NoteTaker:
    """Keeps a conversation's notes up to date, in the background."""

    def __init__(
        self,
        conversation: Conversation,
        chat: ChatModel,
        *,
        max_words: int = 60,
        available: Callable[[], bool] = _always,
        screen: Moderator | None = None,
        log_notes: bool = False,
        rest_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._conversation = conversation
        self._chat = chat
        self._max_words = max_words
        self._instructions = instructions(max_words)
        self._available = available  # False while the model is known to be unreachable
        self._screen = screen
        # Privacy: notes repeat what a child said, so they are only logged with transcripts.
        self._log_notes = log_notes
        self._rest_s = rest_s
        self._clock = clock
        self._resting_until = -math.inf
        self._tasks = BackgroundTasks()
        self._writing: asyncio.Task[None] | None = None

    @property
    def writing(self) -> bool:
        """True while notes are being written."""
        return self._writing is not None and not self._writing.done()

    def catch_up(self) -> None:
        """Start folding what is due into the notes, unless that is already under way.

        Called after every exchange. Returns at once: the notes are written in the background.
        """
        if self.writing or self._conversation.due() is None:
            return
        if self._clock() < self._resting_until or not self._available():
            return
        self._writing = self._tasks.spawn(self._keep_up(), "notes")

    async def aclose(self) -> None:
        await self._tasks.cancel_all()

    async def _keep_up(self) -> None:
        while (passage := self._conversation.due()) is not None:
            started = self._clock()
            try:
                notes = await self._rewrite(passage)
            except LucyError as error:
                self._rest(error)
                return
            if not self._conversation.fold(passage, notes):
                return  # forgotten while the notes were written: they are dropped with it
            log.info(
                "folded %d exchange(s) into the notes (%d words) in %.2fs",
                len(passage.exchanges),
                len(notes.split()),
                self._clock() - started,
            )
            if self._log_notes:
                log.info("Lucy's notes: %r", notes)

    async def _rewrite(self, passage: Passage) -> str:
        """The notes, taking in ``passage``. The notes so far, if it adds nothing fit to keep."""
        said = [exchange for exchange in passage.exchanges if not exchange.screened]
        if not said:
            return passage.notes
        messages = [
            Message(Role.SYSTEM, self._instructions),
            Message(Role.USER, request(passage.notes, said)),
        ]
        async with aclosing(self._chat.stream(messages)) as deltas:
            written = "".join([delta async for delta in deltas])
        notes = tidy(written, self._max_words)
        # Notes are only ever rewritten, never wiped: "none" over notes that say something
        # is far more likely a slip of the model's than a conversation with nothing in it.
        if not notes or not await self._fit_to_keep(notes):
            return passage.notes
        return notes

    async def _fit_to_keep(self, notes: str) -> bool:
        """Whether the safety screen lets ``notes`` into Lucy's prompt."""
        if self._screen is None:
            return True
        assessment = await self._screen.assess(notes)
        if not assessment.allowed:
            log.info(
                "notes screened: %s (%s); keeping the earlier notes",
                assessment.concern.value,
                ", ".join(assessment.reasons),
            )
        return assessment.allowed

    def _rest(self, error: LucyError) -> None:
        rest_s = self._rest_s
        if isinstance(error, RateLimited) and error.retry_after_s:
            rest_s = max(rest_s, error.retry_after_s)
        self._resting_until = self._clock() + rest_s
        log.warning(
            "could not write Lucy's notes (%s: %s); trying again in %.0fs",
            type(error).__name__,
            error,
            rest_s,
        )
