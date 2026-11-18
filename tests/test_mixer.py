"""A sound card's own volume, set through amixer (simulated here)."""

import subprocess
import types
from typing import Any

import pytest

from lucy.adapters.mixer import Mixer
from lucy.core.volume import VolumeSettings

# What amixer printed for Lucy's USB speaker (a Jieli "UACDemoV1.0").
SCONTENTS = """Simple mixer control 'PCM',0
  Capabilities: pvolume pswitch pswitch-joined
  Playback channels: Front Left - Front Right
  Limits: Playback 0 - 147
  Mono:
  Front Left: Playback 59 [40%] [-17.37dB] [on]
  Front Right: Playback 59 [40%] [-17.37dB] [on]
"""


class Amixer:
    """Stands in for ``subprocess.run(["amixer", ...])``."""

    def __init__(self, scontents: str = SCONTENTS, level: str = "-9.00", code: int = 0) -> None:
        self.scontents, self.level, self.code = scontents, level, code
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **_: Any) -> types.SimpleNamespace:
        self.calls.append(args)
        if "scontents" in args:
            out = self.scontents
        else:
            line = f"  Front Left: Playback 99 [67%] [{self.level}dB] [on]"
            out = f"Simple mixer control 'PCM',0\n{line}\n"
        return types.SimpleNamespace(returncode=self.code, stdout=out, stderr="no such card")


def test_sets_the_volume_in_decibels_and_unmutes() -> None:
    amixer = Amixer()
    assert Mixer("UACDemoV10", run=amixer).set_db(-9.0) == -9.0
    assert amixer.calls[-1] == [
        "amixer",
        "-c",
        "UACDemoV10",
        "--",
        "sset",
        "PCM",
        "-9.00dB",
        "unmute",
    ]


def test_says_where_the_card_stopped() -> None:
    assert Mixer("UACDemoV10", run=Amixer(level="-1.50")).set_db(3.0) == -1.5


def test_the_control_is_looked_up_once() -> None:
    amixer = Amixer()
    mixer = Mixer("UACDemoV10", run=amixer)
    mixer.set_db(-9.0)
    mixer.set_db(-6.0)
    assert sum("scontents" in call for call in amixer.calls) == 1


def test_the_usual_control_is_chosen_over_others() -> None:
    scontents = """Simple mixer control 'Mic',0
  Capabilities: cvolume cswitch
Simple mixer control 'Tone',0
  Capabilities: pvolume
Simple mixer control 'Speaker',0
  Capabilities: pvolume pswitch
Simple mixer control 'Auto Gain Control',0
  Capabilities: pswitch
"""
    assert Mixer("x", run=Amixer(scontents)).control() == "Speaker"


def test_a_second_control_of_a_name_keeps_its_index() -> None:
    scontents = "Simple mixer control 'Front',1\n  Capabilities: pvolume\n"
    assert Mixer("x", run=Amixer(scontents)).control() == "Front,1"


def test_a_card_with_no_playback_volume() -> None:
    scontents = "Simple mixer control 'Mic',0\n  Capabilities: cvolume\n"
    assert Mixer("MAX98357A", run=Amixer(scontents)).set_db(-9.0) is None


def test_a_card_that_is_not_there_is_looked_up_again_later() -> None:
    amixer = Amixer(code=1)
    mixer = Mixer("UACDemoV10", run=amixer)
    assert mixer.set_db(-9.0) is None
    amixer.code = 0  # plugged in
    assert mixer.set_db(-9.0) == -9.0


@pytest.mark.parametrize(
    "error", [FileNotFoundError("amixer"), subprocess.TimeoutExpired("amixer", 3)]
)
def test_no_amixer_or_a_stuck_one_is_no_volume(error: Exception) -> None:
    def run(*_: Any, **__: Any) -> None:
        raise error

    assert Mixer("UACDemoV10", run=run).set_db(-9.0) is None


def test_a_level_without_decibels_is_unknown() -> None:
    amixer = Amixer()
    amixer.level = "x"  # no "[...dB]" on the line
    assert Mixer("UACDemoV10", run=amixer).get_db() is None


def test_software_headroom_covers_what_the_speaker_cannot() -> None:
    settings = VolumeSettings(max_db=9.0, speaker_db=-9.0, speaker_max_db=-3.0)
    assert settings.software_headroom_db == 3.0
    assert (
        VolumeSettings(max_db=9.0, speaker_db=-12.0, speaker_max_db=0.0).software_headroom_db == 0
    )
    assert VolumeSettings(max_db=9.0, speaker_mixer=False).software_headroom_db == 9.0
