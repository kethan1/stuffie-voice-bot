"""Split a streaming LLM reply into speakable chunks.

Each sentence is handed to the speech synthesizer as soon as it is complete:

* chunks end at sentence boundaries, because a voice gives a whole sentence
  its natural intonation, and a sentence synthesised in pieces sounds stitched
  together;
* optionally (``first_clause_min_words``), the first chunk may end at a clause
  boundary (",", ";", ":", a dash) once it holds that many words. A slow
  on-device voice then starts sooner. A streaming cloud voice does not need
  this, since its first audio arrives within a fraction of a second anyway;
* an unusually long run without punctuation is force-split so one sentence can
  never hold the speaker hostage.

Boundaries are only confirmed once the *next* character has arrived (or the
stream ends), so "3.14" and "Dr. Seuss" are not split. After a single letter,
they wait for the next word too: a run of single letters ("J. K. Rowling", or a
word spelled out as "S. H. I. T.") stays in one chunk, so the reply screen sees
the word it spells.
"""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"""[.!?…]+["')\]]*(?=\s)|\n\s*\n""")
_CLAUSE_END = re.compile(r"""(?:[,;:]|\s[-\u2013\u2014])(?=\s)""")
# A single letter just before a sentence end: "in a V." or the "S" of "S. H. I. T."
_SINGLE_LETTER = re.compile(r"(?:^|[^\w'])[A-Za-z]$")
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "mt", "no"}
)


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    ends_sentence: bool


class SentenceChunker:
    def __init__(self, *, first_clause_min_words: int = 0, max_words: int = 30) -> None:
        self._first_clause_min_words = first_clause_min_words
        self._max_words = max_words
        self._buffer = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[Chunk]:
        self._buffer += delta
        chunks: list[Chunk] = []
        while (chunk := self._next_chunk()) is not None:
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[Chunk]:
        text, self._buffer = self._buffer.strip(), ""
        if not text:
            return []
        self._emitted += 1
        return [Chunk(text, ends_sentence=True)]

    def _next_chunk(self) -> Chunk | None:
        cut = self._sentence_cut()
        ends_sentence = cut is not None
        if cut is None and self._emitted == 0 and self._first_clause_min_words > 0:
            cut = self._clause_cut(self._first_clause_min_words)
        if cut is None and _word_count(self._buffer) > self._max_words:
            cut = self._clause_cut(1) or _word_cut(self._buffer, self._max_words)
        if cut is None:
            return None
        text, self._buffer = self._buffer[:cut].strip(), self._buffer[cut:].lstrip()
        if not text:  # the boundary was blank lines; the buffer shrank, so look again
            return self._next_chunk()
        self._emitted += 1
        return Chunk(text, ends_sentence)

    def _sentence_cut(self) -> int | None:
        for match in _SENTENCE_END.finditer(self._buffer):
            if _is_abbreviation(self._buffer, match.start()):
                continue
            if _SINGLE_LETTER.search(self._buffer, 0, match.start()):
                following = self._buffer[match.end() :].lstrip()
                if len(following) < 2:
                    return None  # not yet known whether another single letter follows
                if following[0].isalpha() and not following[1].isalpha():
                    continue  # "S. H. ...": the run of letters goes on
            return match.end()
        return None

    def _clause_cut(self, min_words: int) -> int | None:
        for match in _CLAUSE_END.finditer(self._buffer):
            if _word_count(self._buffer[: match.start()]) >= min_words:
                return match.end()
        return None


def _is_abbreviation(text: str, dot: int) -> bool:
    if text[dot] != ".":
        return False
    word = text[:dot].rsplit(maxsplit=1)[-1] if text[:dot].strip() else ""
    # Single letters are not abbreviations: "geese fly in a V. Then..." ends a sentence (see
    # _sentence_cut for a run of them).
    return word.lower().lstrip("(\"'") in _ABBREVIATIONS


def _word_count(text: str) -> int:
    return len(text.split())


def _word_cut(text: str, words: int) -> int:
    matches = list(re.finditer(r"\S+", text))
    return matches[words - 1].end()
