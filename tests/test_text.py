import pytest

from lucy.core.chunker import Chunk, SentenceChunker
from lucy.core.conversation import Conversation
from lucy.core.textnorm import to_speakable
from lucy.core.types import Role


def chunk_stream(deltas: list[str], **kwargs: int) -> list[Chunk]:
    chunker = SentenceChunker(**kwargs)
    chunks = [chunk for delta in deltas for chunk in chunker.feed(delta)]
    return chunks + chunker.flush()


def texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


class TestSentenceChunker:
    def test_splits_sentences_across_deltas(self) -> None:
        chunks = chunk_stream(["Honk! Geese fly", " in a V. It saves", " energy."])
        assert texts(chunks) == ["Honk!", "Geese fly in a V.", "It saves energy."]
        assert all(chunk.ends_sentence for chunk in chunks)

    def test_character_by_character_matches_whole_text(self) -> None:
        text = "Honk! Did you know geese can fly far? They really can. Wow!"
        assert texts(chunk_stream(list(text))) == texts(chunk_stream([text]))

    def test_waits_for_the_character_after_a_period(self) -> None:
        chunker = SentenceChunker()
        assert chunker.feed("Pi is about 3.") == []
        assert texts(chunker.feed("14 and Dr. Seuss wrote books. ")) == [
            "Pi is about 3.14 and Dr. Seuss wrote books."
        ]

    def test_first_chunk_may_end_at_a_clause(self) -> None:
        chunks = chunk_stream(["Well, that is a great question, and the answer is yes."])
        assert texts(chunks) == ["Well, that is a great question,", "and the answer is yes."]
        assert chunks[0].ends_sentence is False

    def test_later_chunks_wait_for_the_sentence(self) -> None:
        chunks = chunk_stream(["Yes. The moon, which is far away, is made of rock. "])
        assert texts(chunks) == ["Yes.", "The moon, which is far away, is made of rock."]

    def test_long_runs_are_force_split(self) -> None:
        words = " ".join(f"word{i}" for i in range(50))
        chunks = chunk_stream([words + " "], max_words=20)
        assert all(len(chunk.text.split()) <= 20 for chunk in chunks)
        assert " ".join(texts(chunks)) == words

    def test_blank_lines_end_a_chunk(self) -> None:
        assert texts(chunk_stream(["First idea\n\nSecond idea"])) == ["First idea", "Second idea"]

    def test_flush_of_nothing_is_empty(self) -> None:
        assert SentenceChunker().flush() == []


class TestToSpeakable:
    @pytest.mark.parametrize(
        ("raw", "spoken"),
        [
            ("**Honk!** Geese are _great_.", "Honk! Geese are great."),
            ("- one\n- two", "one two"),
            ("1. First\n2) Second", "First Second"),
            ("Look at [this page](https://example.com) now", "Look at this page now"),
            ("See https://example.com for more", "See for more"),
            ("I love you 🦢❤️!", "I love you !"),
            ("2+2=4", "2 plus 2 equals 4"),
            ("### Fun fact", "Fun fact"),
        ],
    )
    def test_cleans_text(self, raw: str, spoken: str) -> None:
        assert to_speakable(raw) == spoken

    @pytest.mark.parametrize("raw", ["", "   ", "🦢🦢", "**", "..."])
    def test_nothing_to_say(self, raw: str) -> None:
        assert to_speakable(raw) == ""


class TestConversation:
    def test_prompt_orders_history_between_system_and_new_question(self) -> None:
        conversation = Conversation(max_exchanges=4)
        conversation.record("hi", "Honk, hello!")
        messages = conversation.prompt("be a goose", "why?")
        assert [m.role for m in messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
        assert messages[-1].content == "why?"

    def test_keeps_only_recent_exchanges(self) -> None:
        conversation = Conversation(max_exchanges=2)
        for i in range(5):
            conversation.record(f"q{i}", f"a{i}")
        contents = [m.content for m in conversation.prompt("s", "now")]
        assert contents == ["s", "q3", "a3", "q4", "a4", "now"]

    def test_ignores_unanswered_and_clears(self) -> None:
        conversation = Conversation(max_exchanges=2)
        conversation.record("q", "")
        assert len(conversation) == 0
        conversation.record("q", "a")
        conversation.clear()
        assert len(conversation) == 0
