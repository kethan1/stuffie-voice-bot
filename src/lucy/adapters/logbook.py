"""Lucy's logbook: every conversation, kept in an SQLite file for the grown-ups who look after her.

What is in it
    One row per turn in which Lucy heard words: what the child said, as it was
    transcribed; what Lucy said out loud; how the turn ended; and, if a safety screen
    stepped in, why. That includes what a screen answered for her, so a grown-up can
    see that a child said something that needs one. No audio is ever kept, and turns
    without words (a tap, "I didn't hear that", her greeting) are not rows.

Who reads it
    Grown-ups, on the Conversations page of setup mode (:mod:`lucy.setup`). Lucy never
    does: her own memory (:mod:`lucy.core.conversation`) is still RAM only, and still
    forgets. Rows older than ``keep_days`` are deleted as new conversations begin.

Never in a child's way
    :meth:`Logbook.record` hands the row to one worker thread and returns, so the SD
    card is written after the turn and off the event loop. If the file cannot be
    written, that is logged once and costs nothing else.

Safe to unplug
    The file is opened for each operation and closed again, with SQLite's defaults
    (a rollback journal, ``synchronous=FULL``): a power cut leaves the row before or
    the row after, never a broken file, and nothing is held open while Lucy idles.
    Two processes use it, never at once: Lucy (as ``lucy``) writes, and setup mode
    (as root, while Lucy rests) reads and deletes. Setup mode never creates the file,
    so it always belongs to Lucy. Deleted rows are overwritten (``secure_delete``).

The file is mode 600 in a directory only Lucy and root can enter
(``StateDirectory=`` in lucy.service). It is not encrypted: whoever holds the SD card,
or Lucy's setup card, can read it.
"""

import contextlib
import logging
import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from lucy.core.errors import LucyError
from lucy.core.types import Concern, TurnRecord

log = logging.getLogger(__name__)

DAY_S = 86_400.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id           INTEGER PRIMARY KEY,
    conversation INTEGER NOT NULL,  -- turns with the same number are one conversation
    at           REAL NOT NULL,     -- Unix time, by the Pi's clock
    heard        TEXT NOT NULL,     -- what the child said, as transcribed
    said         TEXT NOT NULL,     -- what Lucy said out loud
    outcome      TEXT NOT NULL,     -- lucy.core.types.Outcome
    concern      TEXT NOT NULL DEFAULT '',  -- lucy.core.types.Concern, '' if no screen stepped in
    reasons      TEXT NOT NULL DEFAULT '',  -- the patterns that matched, comma-separated
    drafted      TEXT NOT NULL DEFAULT '',  -- what the model had written when a screen stopped it
    brain        TEXT NOT NULL DEFAULT ''   -- cloud or local
);
CREATE INDEX IF NOT EXISTS turns_by_conversation ON turns (conversation, id);
CREATE INDEX IF NOT EXISTS turns_by_time ON turns (at);
"""
_COLUMNS = "conversation, at, heard, said, outcome, concern, reasons, drafted, brain"


class LogbookError(LucyError):
    """The logbook could not be read or changed."""


@dataclass(frozen=True, slots=True)
class LoggedTurn:
    at: float
    """When the turn ended (Unix time, by the Pi's clock)."""
    heard: str
    said: str
    outcome: str
    concern: str
    """Why a safety screen stepped in ("" if none did)."""
    reasons: str
    drafted: str
    brain: str

    @property
    def flagged(self) -> bool:
        return bool(self.concern)


@dataclass(frozen=True, slots=True)
class LoggedConversation:
    number: int
    turns: tuple[LoggedTurn, ...]
    """Oldest first; never empty."""

    @property
    def started(self) -> float:
        return self.turns[0].at

    @property
    def flagged(self) -> bool:
        return any(turn.flagged for turn in self.turns)


@dataclass(frozen=True, slots=True)
class Summary:
    conversations: int = 0
    turns: int = 0
    flagged: int = 0
    """Conversations in which a safety screen stepped in."""


class Logbook:
    def __init__(
        self,
        path: Path,
        *,
        keep_days: int = 30,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """``keep_days``: rows older than this are deleted as conversations begin; 0 keeps all."""
        self.path = path
        self._keep_s = keep_days * DAY_S
        self._clock = clock
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="logbook")
        # The rest is only touched on the writer's thread.
        self._era: int | None = None  # the conversation being written, as the pipeline counts
        self._number = 0  # ... and as the file numbers it, which carries on across restarts
        self._failing = False

    # -- writing: the TurnLog port ---------------------------------------------------------

    def record(self, turn: TurnRecord) -> None:
        """Write ``turn`` down in the background. Returns at once."""
        self._writer.submit(self._write, turn, self._clock())

    def close(self) -> None:
        """Wait for whatever is still being written. Nothing can be recorded afterwards."""
        self._writer.shutdown(wait=True)

    def _write(self, turn: TurnRecord, at: float) -> None:
        try:
            with self._open(create=True) as db:
                number = self._number
                if turn.conversation != self._era:  # a new conversation, or Lucy's first row
                    if self._keep_s > 0:
                        db.execute("DELETE FROM turns WHERE at < ?", (at - self._keep_s,))
                    number = db.execute(
                        "SELECT COALESCE(MAX(conversation), 0) + 1 FROM turns"
                    ).fetchone()[0]
                db.execute(
                    f"INSERT INTO turns ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        number,
                        at,
                        turn.heard,
                        turn.said,
                        turn.outcome.value,
                        "" if turn.concern is Concern.NONE else turn.concern.value,
                        ",".join(turn.reasons),
                        turn.drafted,
                        turn.brain,
                    ),
                )
            self._era, self._number = turn.conversation, number
        except Exception as error:  # never the words themselves: only what went wrong
            if not self._failing:
                log.warning(
                    "cannot write Lucy's logbook at %s (%s: %s); conversations are not "
                    "being recorded",
                    self.path,
                    type(error).__name__,
                    error,
                )
            self._failing = True
            return
        if self._failing:
            self._failing = False
            log.info("Lucy's logbook can be written again")

    # -- reading and deleting: setup mode's Conversations page, and lucy doctor ------------
    # These block for a moment on the SD card, so call them through asyncio.to_thread.

    def summary(self) -> Summary:
        if not self.path.exists():
            return Summary()
        with self._reading() as db:
            conversations, turns = db.execute(
                "SELECT COUNT(DISTINCT conversation), COUNT(*) FROM turns"
            ).fetchone()
            (flagged,) = db.execute(
                "SELECT COUNT(DISTINCT conversation) FROM turns WHERE concern != ''"
            ).fetchone()
        return Summary(conversations, turns, flagged)

    def conversations(
        self, *, limit: int = 10, offset: int = 0, flagged: bool = False
    ) -> list[LoggedConversation]:
        """Conversations, the latest first, each with its turns in the order they were said.

        ``flagged``: only those in which a safety screen stepped in.
        """
        if not self.path.exists():
            return []
        only = "HAVING MAX(concern != '') " if flagged else ""
        with self._reading() as db:
            numbers = [
                number
                for (number,) in db.execute(
                    f"SELECT conversation FROM turns GROUP BY conversation {only}"
                    "ORDER BY MAX(id) DESC LIMIT ? OFFSET ?",
                    (max(0, limit), max(0, offset)),
                )
            ]
            turns: dict[int, list[LoggedTurn]] = {number: [] for number in numbers}
            marks = ", ".join("?" * len(numbers))
            for number, *columns in db.execute(
                f"SELECT {_COLUMNS} FROM turns WHERE conversation IN ({marks}) ORDER BY id",
                numbers,
            ):
                turns[number].append(LoggedTurn(*columns))
        return [LoggedConversation(number, tuple(turns[number])) for number in numbers]

    def clear(self) -> int:
        """Delete every row, for good. Returns how many turns there were."""
        if not self.path.exists():
            return 0
        with self._reading() as db:
            removed = db.execute("DELETE FROM turns").rowcount
            db.commit()
            with contextlib.suppress(sqlite3.Error):  # the rows are gone either way
                db.execute("VACUUM")  # hand the freed (and zeroed) pages back to the card
        return int(removed)

    def check(self) -> Summary:
        """Open the file as a write would, making it if need be, and count what is in it."""
        try:
            with self._open(create=True):
                pass
        except (sqlite3.Error, OSError) as error:
            raise LogbookError(f"cannot write {self.path}: {error}") from error
        return self.summary()

    # -- the file --------------------------------------------------------------------------

    @contextlib.contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._open() as db:
                yield db
        except (sqlite3.Error, OSError) as error:
            raise LogbookError(f"cannot read {self.path}: {error}") from error

    @contextlib.contextmanager
    def _open(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        """The database, inside one transaction, closed again afterwards."""
        path = self.path.resolve()
        fresh = create and not path.exists()
        if fresh:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Without ``create`` a missing file is an error: setup mode (root) must never be the
        # one to make it, or Lucy could not write to it afterwards.
        mode = "rwc" if create else "rw"
        db = sqlite3.connect(f"{path.as_uri()}?mode={mode}", uri=True, timeout=5.0)
        try:
            if fresh:
                os.chmod(path, 0o600)  # children's words: for Lucy and root only
            db.execute("PRAGMA secure_delete = ON")
            db.executescript(_SCHEMA)
            with db:
                yield db
        finally:
            db.close()
