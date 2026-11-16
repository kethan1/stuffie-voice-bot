"""Lucy's logbook: the SQLite file her conversations are written to, and read back from."""

import contextlib
import logging
import sqlite3
import stat
import threading
from pathlib import Path

import pytest

from lucy.adapters.logbook import DAY_S, Logbook, LogbookError
from lucy.core.types import Concern, Outcome, TurnRecord

REPO = Path(__file__).resolve().parent.parent


def said(conversation: int, heard: str, answer: str = "Honk!") -> TurnRecord:
    return TurnRecord(conversation, heard, answer, brain="cloud")


def screened(conversation: int, heard: str) -> TurnRecord:
    return TurnRecord(
        conversation,
        heard,
        "Please tell a grown-up you trust.",
        Outcome.SCREENED,
        Concern.NEEDS_GROWN_UP,
        ("being-hurt", "secret"),
    )


class Book:
    """A logbook in a temporary directory, with a clock the test moves."""

    def __init__(self, tmp_path: Path, *, keep_days: int = 30) -> None:
        self.path = tmp_path / "state" / "logbook.db"
        self.keep_days = keep_days
        self.now = 1_800_000_000.0
        self.book = self.open()

    def open(self) -> Logbook:
        return Logbook(self.path, keep_days=self.keep_days, clock=lambda: self.now)

    def write(self, *turns: TurnRecord, after_s: float = 1.0) -> None:
        """Record each turn ``after_s`` after the last, and wait until they are on disk."""
        for turn in turns:
            self.now += after_s
            self.book.record(turn)
        self.book._writer.submit(lambda: None).result(2.0)  # the writer has caught up

    def restart(self) -> None:
        """What a restart of Lucy does: the last rows are written, and a new Logbook opens."""
        self.book.close()
        self.book = self.open()


def test_turns_are_read_back_by_conversation_latest_first(tmp_path: Path) -> None:
    book = Book(tmp_path)
    book.write(said(0, "why is the sky blue", "Sunlight!"), said(0, "why?"), said(1, "hi lucy"))
    latest, first = book.book.conversations()
    assert [turn.heard for turn in first.turns] == ["why is the sky blue", "why?"]
    assert [turn.heard for turn in latest.turns] == ["hi lucy"]
    turn = first.turns[0]
    assert (turn.said, turn.outcome, turn.concern, turn.brain) == (
        "Sunlight!",
        "answered",
        "",
        "cloud",
    )
    assert first.started == turn.at == 1_800_000_001.0
    assert not first.flagged
    found = book.book.summary()
    assert (found.conversations, found.turns, found.flagged) == (2, 3, 0)


def test_a_restart_begins_a_new_conversation_and_keeps_the_old(tmp_path: Path) -> None:
    book = Book(tmp_path)
    book.write(said(0, "before"))
    book.restart()  # Lucy counts her conversations from 0 again
    book.write(said(0, "after"), said(0, "after too"))
    assert [[turn.heard for turn in c.turns] for c in book.book.conversations()] == [
        ["after", "after too"],
        ["before"],
    ]


def test_what_a_safety_screen_did_is_kept_and_can_be_picked_out(tmp_path: Path) -> None:
    book = Book(tmp_path)
    stopped = TurnRecord(
        2, "spell it", "Sure! Nope.", Outcome.STOPPED, Concern.OFF_LIMITS, ("blocked-word",), "S H"
    )
    book.write(said(0, "hello"), screened(1, "my brother hits me"), said(1, "okay"), stopped)
    found = book.book.summary()
    assert (found.conversations, found.turns, found.flagged) == (3, 4, 2)
    flagged = book.book.conversations(flagged=True)
    assert [c.turns[0].heard for c in flagged] == ["spell it", "my brother hits me"]
    assert all(c.flagged for c in flagged)
    disclosure = flagged[1].turns[0]  # the whole conversation comes with it
    assert [turn.heard for turn in flagged[1].turns] == ["my brother hits me", "okay"]
    assert (disclosure.outcome, disclosure.concern) == ("screened", "needs_grown_up")
    assert disclosure.reasons == "being-hurt,secret"
    assert disclosure.flagged
    assert not flagged[1].turns[1].flagged
    assert flagged[0].turns[0].drafted == "S H"


def test_pages(tmp_path: Path) -> None:
    book = Book(tmp_path)
    book.write(*(said(number, f"question {number}") for number in range(5)))
    heard = [
        [c.turns[0].heard for c in book.book.conversations(limit=2, offset=offset)]
        for offset in (0, 2, 4, 6)
    ]
    assert heard == [["question 4", "question 3"], ["question 2", "question 1"], ["question 0"], []]


def test_old_conversations_are_deleted_as_new_ones_begin(tmp_path: Path) -> None:
    book = Book(tmp_path, keep_days=7)
    book.write(said(0, "a week ago"))
    book.write(said(1, "yesterday"), after_s=6 * DAY_S)
    book.write(said(1, "still yesterday"), after_s=2 * DAY_S)  # same conversation: nothing goes
    assert book.book.summary().turns == 3
    book.write(said(2, "today"))
    assert [c.turns[0].heard for c in book.book.conversations()] == ["today", "yesterday"]


def test_keep_days_zero_keeps_everything(tmp_path: Path) -> None:
    book = Book(tmp_path, keep_days=0)
    book.write(said(0, "long ago"))
    book.write(said(1, "now"), after_s=4000 * DAY_S)
    assert book.book.summary().turns == 2


def test_deleting_everything_leaves_none_of_it_in_the_file(tmp_path: Path) -> None:
    book = Book(tmp_path)
    book.write(said(0, "my name is Zebedee Quux"), screened(0, "a secret about Zebedee"))
    assert b"Zebedee" in book.path.read_bytes()
    assert book.book.clear() == 2
    assert book.book.conversations() == []
    assert book.book.summary().turns == 0
    assert b"Zebedee" not in book.path.read_bytes()  # overwritten, not just unlisted
    book.write(said(0, "hello again"))
    assert book.book.summary().turns == 1


def test_reading_never_makes_the_file(tmp_path: Path) -> None:
    # Setup mode runs as root: a file it made could not be written by Lucy afterwards.
    book = Book(tmp_path)
    assert book.book.conversations() == []
    assert book.book.summary().turns == 0
    assert book.book.clear() == 0
    assert not book.path.exists()
    assert not book.path.parent.exists()


def test_the_file_is_for_lucy_alone(tmp_path: Path) -> None:
    book = Book(tmp_path)
    book.write(said(0, "hello"))
    assert stat.S_IMODE(book.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(book.path.parent.stat().st_mode) == 0o700
    assert [path.name for path in book.path.parent.iterdir()] == ["logbook.db"]  # no journal left


def test_recording_does_not_wait_for_the_card(tmp_path: Path) -> None:
    book = Book(tmp_path)
    writing, release = threading.Event(), threading.Event()
    original = book.book._write

    def slow(turn: TurnRecord, at: float) -> None:
        writing.set()
        release.wait(2.0)
        original(turn, at)

    book.book._write = slow  # type: ignore[method-assign]
    book.book.record(said(0, "hello"))  # returns while the write is still under way
    assert writing.wait(2.0)
    assert not book.path.exists()
    release.set()
    book.restart()  # ... and closing waits for it
    assert book.book.summary().turns == 1


def test_a_logbook_that_cannot_be_written_is_reported_once_and_recovers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    book = Book(tmp_path)
    book.path.parent.mkdir()
    book.path.mkdir()  # a directory where the file should be
    with caplog.at_level(logging.INFO, logger="lucy.adapters.logbook"):
        book.write(said(0, "my secret question"), said(0, "another"))
        assert caplog.text.count("cannot write Lucy's logbook") == 1
        assert "secret question" not in caplog.text  # what went wrong, never the words
        book.path.rmdir()
        book.write(said(0, "third time lucky"))
    assert "can be written again" in caplog.text
    assert [turn.heard for c in book.book.conversations() for turn in c.turns] == [
        "third time lucky"
    ]


def test_a_broken_file_is_an_error_to_read_and_check(tmp_path: Path) -> None:
    book = Book(tmp_path)
    book.path.parent.mkdir()
    book.path.write_bytes(b"this is not a database, " * 64)
    with pytest.raises(LogbookError, match="cannot read"):
        book.book.conversations()
    with pytest.raises(LogbookError, match="cannot write"):
        book.book.check()


def test_check_makes_the_file_and_counts(tmp_path: Path) -> None:
    book = Book(tmp_path)
    assert book.book.check().turns == 0
    assert book.path.exists()
    book.write(said(0, "hello"))
    assert book.book.check().conversations == 1


def test_the_file_is_plain_sqlite(tmp_path: Path) -> None:
    book = Book(tmp_path)
    book.write(screened(0, 'it\'s "quoted"; DROP TABLE turns; --'))
    with contextlib.closing(sqlite3.connect(book.path)) as db:
        row = db.execute("SELECT heard, outcome, concern, reasons FROM turns").fetchone()
    assert row == (
        'it\'s "quoted"; DROP TABLE turns; --',
        "screened",
        "needs_grown_up",
        "being-hurt,secret",
    )


class TestDeployment:
    """The service files give Lucy, and setup mode, the access the logbook needs."""

    def test_lucy_may_write_where_the_logbook_lives_by_default(self) -> None:
        from lucy.config import Config, load_config

        database = Config().logbook.database
        assert load_config(REPO / "deploy" / "config.toml").logbook.database == database
        unit = (REPO / "deploy" / "systemd" / "lucy.service").read_text()
        assert database.parent == Path("/var/lib/lucy")
        assert "\nStateDirectory=lucy\n" in unit  # /var/lib/lucy, despite ProtectSystem=strict
        assert "\nStateDirectoryMode=0700\n" in unit  # children's words: not for other users
        assert "\nUMask=0077\n" in unit

    def test_setup_mode_may_read_and_delete_it_and_keeps_it_on_a_read_only_root(self) -> None:
        unit = (REPO / "deploy" / "systemd" / "lucy-setup.service").read_text()
        (paths,) = [line for line in unit.splitlines() if line.startswith("ReadWritePaths=")]
        assert "-/var/lib/lucy" in paths.split()  # "-": absent until Lucy has run once
        assert "Conflicts=lucy.service" in unit  # so the two never have it open together
        persist = (REPO / "deploy" / "bin" / "lucy-persist").read_text()
        assert "/var/lib/lucy" in persist
        wrapper = (REPO / "deploy" / "bin" / "lucy").read_text()
        assert "StateDirectory=lucy" in wrapper  # sudo lucy doctor / run --keyboard
