"""Changing Lucy's volume by asking, and making audio louder without clipping it."""

import numpy as np
import pytest

from lucy.core.audio import CueKind, adjust_volume, level_dbfs, soft_limit
from lucy.core.phrases import Phrase
from lucy.core.volume import Volume, VolumeRequest, VolumeSettings, volume_request
from tests.fakes import FakeSpeaker, tone

LOUDER, QUIETER = VolumeRequest.LOUDER, VolumeRequest.QUIETER
LOUDEST, QUIETEST = VolumeRequest.LOUDEST, VolumeRequest.QUIETEST


@pytest.mark.parametrize(
    ("said", "wanted"),
    [
        ("Louder!", LOUDER),
        ("louder please", LOUDER),
        ("Lucy, louder.", LOUDER),
        ("A little louder", LOUDER),
        ("Can you talk louder?", LOUDER),
        ("Could you please speak a little bit louder", LOUDER),
        ("Speak up!", LOUDER),
        ("I can't hear you", LOUDER),
        ("I can\N{RIGHT SINGLE QUOTATION MARK}t hear you.", LOUDER),  # how Whisper writes it
        ("Turn it up!", LOUDER),
        ("Turn up the volume", LOUDER),
        ("Increase the volume", LOUDER),
        ("Turn your volume up", LOUDER),
        ("Volume up", LOUDER),
        ("Be louder", LOUDER),
        ("Honk louder!", LOUDER),
        ("You're too quiet", LOUDER),
        ("I want you to talk louder", LOUDER),
        ("Make yourself louder", LOUDER),
        ("Quieter!", QUIETER),
        ("Too loud!", QUIETER),
        ("You're way too loud", QUIETER),
        ("That's so loud", QUIETER),
        ("Turn it down please", QUIETER),
        ("Turn down the volume", QUIETER),
        ("Volume down", QUIETER),
        ("Talk softer", QUIETER),
        ("Can you be quieter?", QUIETER),
        ("Use your indoor voice", QUIETER),
        ("Not so loud, Lucy", QUIETER),
        ("A bit softer", QUIETER),
        ("Talk as loud as you can!", LOUDEST),
        ("Max volume", LOUDEST),
        ("As quiet as you can", QUIETEST),
        ("Minimum volume please", QUIETEST),
    ],
)
def test_hears_a_request_to_change_her_volume(said: str, wanted: VolumeRequest) -> None:
    assert volume_request(said) is wanted


@pytest.mark.parametrize(
    "said",
    [
        "Which is louder, a lion or a whale?",
        "Is a lion louder than a whale?",
        "Can a whale be louder than a jet?",
        "My dog can get louder than a car",
        "Why do geese honk so loud?",
        "How loud can a goose honk?",
        "My brother is so loud",
        "I like your quiet voice",
        "Tell me a story about a loud goose",
        "louder, no, quieter",  # both: the model can ask what they meant
        "Can you tell me a really long story about a goose who learned to talk louder than "
        "everyone else in the pond",
        "",
    ],
)
def test_other_things_go_to_the_model(said: str) -> None:
    assert volume_request(said) is None


class TestVolume:
    def volume(self, **settings: float) -> tuple[Volume, FakeSpeaker]:
        speaker = FakeSpeaker()
        return Volume(speaker, VolumeSettings(**settings)), speaker  # type: ignore[arg-type]

    def test_steps_up_to_the_loudest_she_goes_and_says_so(self) -> None:
        volume, speaker = self.volume(step_db=3, max_db=9)
        said = [volume.change(LOUDER) for _ in range(4)]
        assert said == [Phrase.LOUDER, Phrase.LOUDER, Phrase.LOUDEST, Phrase.LOUDEST]
        assert volume.db == speaker.volume_db == 9.0

    def test_steps_down_to_the_quietest(self) -> None:
        volume, speaker = self.volume(step_db=5, min_db=-10)
        said = [volume.change(QUIETER) for _ in range(3)]
        assert said == [Phrase.QUIETER, Phrase.QUIETEST, Phrase.QUIETEST]
        assert speaker.volume_db == -10.0

    def test_goes_straight_to_the_loudest_or_the_quietest(self) -> None:
        volume, speaker = self.volume(max_db=6, min_db=-12)
        assert volume.change(LOUDEST) is Phrase.LOUDEST
        assert speaker.volume_db == 6.0
        assert volume.change(QUIETEST) is Phrase.QUIETEST
        assert speaker.volume_db == -12.0

    def test_a_grown_up_can_keep_it_fixed(self) -> None:
        volume, _ = self.volume(commands=False)
        assert volume.request("louder please") is None
        assert self.volume()[0].request("louder please") is LOUDER


class TestAdjustVolume:
    def test_changes_the_level_by_that_many_decibels(self) -> None:
        quiet = tone(0.5, dbfs=-30.0)
        assert level_dbfs(adjust_volume(quiet, 6.0)) == pytest.approx(-24.0, abs=0.1)
        assert level_dbfs(adjust_volume(quiet, -6.0)) == pytest.approx(-36.0, abs=0.1)
        assert adjust_volume(quiet, 0.0) is quiet

    def test_her_voice_gets_much_louder_without_clipping(self) -> None:
        speech = tone(0.5, dbfs=-17.0)  # as loud as her voice
        louder = adjust_volume(speech, 9.0)
        assert level_dbfs(louder) > level_dbfs(speech) + 7  # nearly all of it
        loudest = adjust_volume(tone(0.5, dbfs=-3.0), 9.0).samples()  # already at full scale
        assert int(np.abs(loudest).max()) < 32767  # rounded off, never flattened at the top

    def test_keeps_the_cues(self) -> None:
        honk = tone(0.2).cued(CueKind.HONK)
        assert adjust_volume(honk, 6.0).cues == honk.cues

    def test_soft_limit_is_smooth_and_bounded(self) -> None:
        x = np.linspace(-4, 4, 8001)
        y = soft_limit(x)
        assert np.all(np.abs(y) < 1.0)
        assert np.array_equal(y[np.abs(x) <= 0.5], x[np.abs(x) <= 0.5])
        assert np.all(np.diff(y) > 0)  # never folds back on itself
