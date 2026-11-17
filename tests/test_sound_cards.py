"""Choosing Lucy's speaker and microphone from the sound cards that are plugged in."""

import logging
import os
from pathlib import Path

import pytest

from lucy.adapters import sound_cards
from lucy.adapters.sound_cards import Card, cards, choose, route, user


def plug(
    root: Path,
    number: int,
    card_id: str,
    *,
    usb: bool = True,
    play: bool = True,
    record: bool = False,
) -> None:
    """A card as /proc/asound shows it."""
    folder = root / f"card{number}"
    folder.mkdir(parents=True)
    (folder / "id").write_text(card_id + "\n")
    if usb:
        (folder / "usbid").write_text("4c4a:4155\n")
    if play:
        (folder / "pcm0p" / "sub0").mkdir(parents=True)
    if record:
        (folder / "pcm0c" / "sub0").mkdir(parents=True)


@pytest.fixture
def pi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lucy's Pi after her speaker was swapped: HDMI, her microphone, and a new USB speaker."""
    monkeypatch.delenv("LUCY_SPEAKER_CARD", raising=False)
    monkeypatch.delenv("LUCY_MICROPHONE_CARD", raising=False)
    monkeypatch.setattr(sound_cards, "_router", sound_cards._Router())
    plug(tmp_path, 0, "vc4hdmi0", usb=False)
    plug(tmp_path, 1, "vc4hdmi1", usb=False)
    plug(tmp_path, 2, "LucyMic", play=False, record=True)
    plug(tmp_path, 3, "UACDemoV10")
    return tmp_path


def test_cards_are_read_as_aplay_lists_them(pi: Path) -> None:
    found = cards(pi)
    assert [card.id for card in found] == ["vc4hdmi0", "vc4hdmi1", "LucyMic", "UACDemoV10"]
    assert found[2] == Card(2, "LucyMic", usb=True, playback=False, capture=True)
    assert found[0] == Card(0, "vc4hdmi0", usb=False, playback=True, capture=False)


def test_a_card_unplugged_while_it_is_read_is_left_out(pi: Path) -> None:
    (pi / "card4").mkdir()  # its id has already gone
    assert len(cards(pi)) == 4


def test_no_sound_cards_at_all(tmp_path: Path) -> None:
    assert cards(tmp_path / "missing") == []


def test_the_configured_speaker_is_used_when_it_is_plugged_in(pi: Path) -> None:
    plug(pi, 4, "LucySpeaker")
    assert route("output", "LucySpeaker", root=pi).card == cards(pi)[4]


def test_a_swapped_speaker_stands_in_for_the_configured_one(
    pi: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        chosen = route("output", "LucySpeaker", root=pi)
    assert chosen.card is not None
    assert chosen.card.id == "UACDemoV10"
    assert chosen.stand_in
    assert os.environ["LUCY_SPEAKER_CARD"] == "UACDemoV10"  # what asound.conf reads
    assert "standing in for LucySpeaker" in caplog.text
    assert caplog.records[-1].levelno == logging.WARNING


def test_the_speaker_is_never_the_microphone_or_hdmi(pi: Path) -> None:
    (pi / "card3" / "pcm0p").rename(pi / "card3" / "gone")  # the new speaker is unplugged too
    chosen = route("output", "LucySpeaker", root=pi)
    assert chosen.card is None
    assert "LUCY_SPEAKER_CARD" not in os.environ  # asound.conf's own card, which is missing too
    assert str(chosen) == "no speaker found: LucySpeaker and every USB speaker are unplugged"


def test_without_a_usb_speaker_the_one_in_asound_conf_is_used(tmp_path: Path) -> None:
    """An I2S amplifier is not USB, so it is reached through asound.conf's own card."""
    plug(tmp_path, 0, "Headphones", usb=False)
    chosen = route("output", "", root=tmp_path)
    assert chosen.card is None
    assert not chosen.stand_in


def test_a_configured_card_that_is_not_usb_is_used(tmp_path: Path) -> None:
    plug(tmp_path, 0, "MAX98357A", usb=False)
    plug(tmp_path, 1, "Device")
    assert route("output", "MAX98357A", root=tmp_path).card == cards(tmp_path)[0]


def test_a_plain_speaker_is_chosen_over_a_headset() -> None:
    headset = Card(1, "Headset", usb=True, playback=True, capture=True)
    speaker = Card(2, "Speaker", usb=True, playback=True, capture=False)
    other = Card(3, "Speaker_1", usb=True, playback=True, capture=False)
    assert choose("output", "", [headset, speaker, other]) == speaker
    assert choose("output", "", [headset]) == headset  # better than nothing
    assert choose("input", "", [speaker, headset]) == headset


def test_the_microphone_is_routed_too(pi: Path) -> None:
    chosen = route("input", "LucyMic", root=pi)
    assert chosen.card is not None
    assert chosen.card.id == "LucyMic"
    assert not chosen.stand_in
    assert os.environ["LUCY_MICROPHONE_CARD"] == "LucyMic"


def test_a_change_is_logged_once(pi: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            route("output", "LucySpeaker", root=pi)
    assert len(caplog.records) == 1
    plug(pi, 4, "LucySpeaker")  # the old speaker is back
    with caplog.at_level(logging.INFO):
        assert route("output", "LucySpeaker", root=pi).card == cards(pi)[4]
    assert len(caplog.records) == 2
    assert caplog.records[-1].levelno == logging.INFO


def test_the_process_using_a_card_is_found(pi: Path) -> None:
    (pi / "card3" / "pcm0p" / "sub0" / "status").write_text(
        "state: RUNNING\nowner_pid   : 1321\ntrigger_time: 1.0\n"
    )
    assert user(route("output", "", root=pi), root=pi) == 1321
    (pi / "card3" / "pcm0p" / "sub0" / "status").write_text("closed\n")
    assert user(route("output", "", root=pi), root=pi) is None
