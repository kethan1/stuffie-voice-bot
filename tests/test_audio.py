import io
import wave

import numpy as np
import pytest

from lucy.core.audio import (
    SILENCE_DBFS,
    Cue,
    CueKind,
    Pcm,
    amplify,
    chime,
    join,
    level_dbfs,
    trim_silence,
)
from lucy.core.phrases import Phrase, PhraseBook
from lucy.persona import LUCY, MOVING_SYSTEM_PROMPT, SYSTEM_PROMPT
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
        assert audio.cues == (Cue(CueKind.CHIME, 0.0, audio.duration),)  # not Lucy's voice

    def test_chime_peaks_at_its_volume_with_a_brighter_overtone(self) -> None:
        plain = chime([(587.0, 0.06)], 22_050, volume=0.6)
        bright = chime([(587.0, 0.06)], 22_050, volume=0.6, overtone=0.35)
        for audio in (plain, bright):
            assert np.abs(audio.samples()).max() / 32767 == pytest.approx(0.6, abs=0.01)
        spectrum = np.abs(np.fft.rfft(bright.samples().astype(float)))
        freqs = np.fft.rfftfreq(bright.frames, 1 / 22_050)
        assert spectrum[np.abs(freqs - 1174) < 40].max() > 0.2 * spectrum.max()  # the octave


class TestCues:
    def test_cues_follow_the_audio_through_joins(self) -> None:
        honk = Pcm.silence(0.3, 1_000).cued(CueKind.HONK)
        line = Pcm.silence(0.5, 1_000) + honk + Pcm.silence(0.1, 1_000) + honk
        assert [cue.kind for cue in line.cues] == [CueKind.HONK, CueKind.HONK]
        assert [cue.start_s for cue in line.cues] == pytest.approx([0.5, 0.9])
        assert [cue.end_s for cue in line.cues] == pytest.approx([0.8, 1.2])
        assert join([], 1_000) == Pcm(b"", 1_000)

    def test_louder_audio_keeps_its_cues(self) -> None:
        honk = tone(0.1).cued(CueKind.HONK)
        assert amplify(honk, 0.5).cues == honk.cues


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


class TestPersona:
    def test_lucy_knows_whether_she_can_move(self) -> None:
        assert LUCY.prompt(moving=False) == SYSTEM_PROMPT
        assert LUCY.prompt(moving=True) == MOVING_SYSTEM_PROMPT
        assert "move at all by yourself" in SYSTEM_PROMPT
        assert "motors" not in SYSTEM_PROMPT
        assert "motors inside you" in MOVING_SYSTEM_PROMPT
        assert "move at all by yourself" not in MOVING_SYSTEM_PROMPT

    def test_the_two_prompts_differ_only_in_her_body(self) -> None:
        still = SYSTEM_PROMPT.split("\n\n")
        moving = MOVING_SYSTEM_PROMPT.split("\n\n")
        assert [a == b for a, b in zip(still, moving, strict=True)] == [
            True,
            False,
            True,
            True,
            True,
            True,
        ]
