import io
import wave

import pytest

from lucy.core.audio import SILENCE_DBFS, Pcm, chime, level_dbfs, trim_silence
from lucy.core.phrases import Phrase, PhraseBook
from lucy.core.safety import BlockedWords, DistressScreen
from lucy.core.types import Concern
from lucy.persona import LUCY
from tests.fakes import FakeTTS, tone


class TestPcm:
    def test_silence_and_duration(self) -> None:
        audio = Pcm.silence(0.5, 16_000)
        assert audio.frames == 8_000
        assert audio.duration == pytest.approx(0.5)
        assert level_dbfs(audio) == SILENCE_DBFS

    def test_join_requires_matching_rates(self) -> None:
        joined = Pcm.silence(0.1, 16_000) + Pcm.silence(0.2, 16_000)
        assert joined.duration == pytest.approx(0.3)
        with pytest.raises(ValueError, match="cannot join"):
            _ = Pcm.silence(0.1, 16_000) + Pcm.silence(0.1, 22_050)

    def test_wav_round_trip(self) -> None:
        audio = tone(0.25)
        with wave.open(io.BytesIO(audio.to_wav())) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16_000)
            assert wav.readframes(wav.getnframes()) == audio.data

    def test_level_of_a_tone(self) -> None:
        assert level_dbfs(tone(0.5, dbfs=-20.0)) == pytest.approx(-20.0, abs=0.2)

    def test_trim_keeps_speech_and_a_little_padding(self) -> None:
        audio = Pcm.silence(1.0, 16_000) + tone(0.5) + Pcm.silence(1.0, 16_000)
        trimmed = trim_silence(audio, -50.0, pad_s=0.1)
        assert trimmed.duration == pytest.approx(0.7, abs=0.05)

    def test_trim_of_silence_is_empty(self) -> None:
        assert trim_silence(Pcm.silence(1.0, 16_000), -50.0).frames == 0

    def test_chime_length(self) -> None:
        audio = chime([(440.0, 0.05), (880.0, 0.05)], 22_050)
        assert audio.duration == pytest.approx(0.1, abs=1e-3)


class TestDistressScreen:
    @pytest.mark.parametrize(
        "said",
        [
            "I want to hurt myself",
            "sometimes I don't want to be alive",
            "my brother hits me every day",
            "nobody likes me at school",
            "a stranger is following me",
            "I’m being bullied",
            "I wish I was dead",
            "everyone would be happier if I wasn't here",
            "my friend said she's going to kill herself",
            "my stepdad punches me when he's mad",
            "my mom slapped me really hard",
            "my uncle touched my private parts",
            "a grown-up told me not to tell anyone what he did",
            "he said it's our little secret.",
            "my mom doesn't give me food",
            "I'm scared to go home",
            "a man in a car asked me to get in",
            "someone on Roblox wants a picture of me",
            "a stranger online asked where I live",
            "a kid in a game asked me to send a photo",
            "I'm stupid and ugly and I hate me",
            "nobody wants me",
        ],
    )
    async def test_routes_to_a_grown_up(self, said: str) -> None:
        assessment = await DistressScreen().assess(said)
        assert assessment.concern is Concern.NEEDS_GROWN_UP
        assert assessment.reasons

    @pytest.mark.parametrize(
        "said",
        [
            "why is the sky blue",
            "tell me a joke about a goose",
            "how many legs does a spider have",
            "I hit a home run today",
            "my brother beat me at Uno",
            "in Minecraft I built a secret base",
            "my mom asked me to get in the car",
            "can you keep a secret? Don't tell my mom",
            "I want to make YouTube videos of me playing Minecraft",
            "if I was a bird I would fly to the moon",
            "this homework is killing me",
            "I hate meatballs",
            "Beats me! Is it a towel?",
        ],
    )
    async def test_ordinary_questions_pass(self, said: str) -> None:
        assert (await DistressScreen().assess(said)).allowed

    async def test_extra_patterns(self) -> None:
        screen = DistressScreen(extra=[r"\bfire alarm\b"])
        assert (await screen.assess("the fire alarm is going off")).reasons == ("custom-1",)


class TestBlockedWords:
    @pytest.mark.parametrize(
        "sentence",
        [
            "S H I T.",
            "It's spelled f-u-c-k.",
            "S\u2011H\u2011I\u2011T.",  # gpt-oss spells with non-breaking hyphens
            "It is S, H, I, T.",
            "Sexy means something that makes people feel nice.",
            "Marijuana is a plant some people smoke.",
            "Oh, what the hell!",
        ],
    )
    def test_stops_words_lucy_must_not_say(self, sentence: str) -> None:
        assert not BlockedWords().allows(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "Because is spelled B-E-C-A-U-S-E.",
            "Hello! Pass the ball to your classmate.",
            "A naked mole rat has wrinkly pink skin.",
            "A cockatoo can dance, and a cockroach can run fast!",
            "Arsenic is a poison, and an arsenal is full of armor.",
            "The Owl and the Pussy-cat went to sea.",
            "A weed is a plant growing where gardeners don't want it.",
            "A hermit crab lives in a shell. Is it a B or a C?",
        ],
    )
    def test_lets_ordinary_sentences_through(self, sentence: str) -> None:
        assert BlockedWords().allows(sentence)


class TestPhraseBook:
    def test_every_phrase_has_lines(self) -> None:
        PhraseBook(LUCY.lines, FakeTTS())  # raises if the persona is incomplete

    def test_missing_lines_are_rejected(self) -> None:
        lines = dict(LUCY.lines)
        del lines[Phrase.OFFLINE]
        with pytest.raises(ValueError, match="offline"):
            PhraseBook(lines, FakeTTS())

    async def test_renders_each_line_once(self) -> None:
        tts = FakeTTS()
        book = PhraseBook(LUCY.lines, tts)
        first = await book.render(Phrase.OFFLINE)
        second = await book.render(Phrase.OFFLINE)
        assert first == second
        assert len(tts.texts) == 1

    async def test_failed_renders_are_retried(self) -> None:
        tts = FakeTTS(error=RuntimeError("boom"))
        book = PhraseBook(LUCY.lines, tts)
        with pytest.raises(RuntimeError):
            await book.render(Phrase.OFFLINE)
        tts.error = None
        await book.render(Phrase.OFFLINE)
        assert len(tts.texts) == 2

    async def test_forgotten_lines_are_rendered_again(self) -> None:
        tts = FakeTTS()
        book = PhraseBook(LUCY.lines, tts)
        await book.render(Phrase.OFFLINE)
        tts.sample = b"\x02\x00"  # say, the cloud voice is back
        book.forget()
        assert (await book.render(Phrase.OFFLINE)).data[:2] == b"\x02\x00"
        assert len(tts.texts) == 2

    async def test_preload_renders_everything(self) -> None:
        tts = FakeTTS()
        await PhraseBook(LUCY.lines, tts).preload()
        assert len(tts.texts) == sum(len(lines) for lines in LUCY.lines.values())
