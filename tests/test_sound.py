"""The PortAudio adapters, driven by a simulated audio thread instead of a sound card."""

import asyncio
import dataclasses
import importlib.abc
import logging
import os
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lucy.core.audio import Pcm
from lucy.core.errors import AudioDeviceError
from tests.fakes import until

BLOCK = 64  # frames per callback


class _Status:
    output_underflow = False
    input_overflow = False


class FakeStream:
    """Calls ``callback`` from its own thread, like PortAudio does."""

    def __init__(
        self,
        *,
        callback: Callable[..., None],
        blocksize: int,
        finished_callback: Callable[[], None] | None = None,
        **_: Any,
    ) -> None:
        self.callback = callback
        self.finished_callback = finished_callback
        self.blocksize = blocksize
        self.output = bytearray()
        self.feed = [bytes([i % 256, 0]) * blocksize for i in range(1000)]
        self.timing: Any = None  # PortAudio's time info, as the callback is handed it
        self.start_error: Exception | None = None
        self.unplug_at_start = False
        self.unplugged = False
        self.stalled = False
        self.let_go = threading.Event()  # set when PortAudio gives up on a stalled stream
        self.closed = False
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        if self.unplug_at_start:
            self.unplugged = True
            if self.finished_callback is not None:
                self.finished_callback()
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running.is_set():
            if not self.stalled:
                self.tick()
            time.sleep(0.0005)
        if self.finished_callback is not None:
            self.finished_callback()  # as PortAudio does, however the stream came to stop

    def tick(self) -> None:
        raise NotImplementedError

    def unplug(self) -> None:
        """The sound card goes away: PortAudio's audio thread gives up."""
        self.unplugged = True
        self._running.clear()

    def stall(self) -> None:
        """The sound card goes away, as ALSA shows it: nothing more is asked for, and
        PortAudio says nothing (and cannot stop the stream) until it gives up, much later."""
        self.stalled = True

    def stop(self) -> None:
        if self.stalled:
            self.let_go.wait(5.0)  # (a failing test must not leave this thread waiting for ever)
        self._running.clear()
        if self._thread:
            self._thread.join()
        if self.unplugged or self.stalled:
            raise OSError("no such device")

    def close(self) -> None:
        self.closed = True


class FakeOutput(FakeStream):
    def tick(self) -> None:
        buffer = bytearray(self.blocksize * 2)
        self.callback(buffer, self.blocksize, self.timing, _Status())
        self.output += buffer


class FakeInput(FakeStream):
    def tick(self) -> None:
        if self.feed:
            self.callback(self.feed.pop(0), self.blocksize, None, _Status())


@pytest.fixture
def sounddevice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.SimpleNamespace:
    """A stand-in for the ``sounddevice`` module.

    ``fail``: no stream can be opened. ``listed``: the sound cards are on PortAudio's list.
    ``plugged_in``: they are there to be found when it lists them again (``lists`` counts).
    ``default``: the name of PortAudio's default device on ALSA (None: not on ALSA).
    ``cards``: /proc/asound, empty until a test plugs cards in. ``card_at_open``: the
    card ALSA's default device named as each stream opened.
    """
    from lucy.adapters import sound, sound_cards

    monkeypatch.setattr(sound_cards, "PROC_ASOUND", tmp_path)
    monkeypatch.setattr(sound_cards, "_router", sound_cards._Router())
    monkeypatch.delenv("LUCY_SPEAKER_CARD", raising=False)
    monkeypatch.delenv("LUCY_MICROPHONE_CARD", raising=False)

    streams: list[FakeStream] = []
    module = types.SimpleNamespace(
        fail=False,
        fail_start=False,
        unplug_at_start=False,
        listed=True,
        plugged_in=False,
        lists=0,
        streams=streams,
        timing=None,
        default=None,
        cards=tmp_path,
        card_at_open=[],
    )

    def make(kind: type[FakeStream]) -> Callable[..., FakeStream]:
        def factory(**kwargs: Any) -> FakeStream:
            variable = "LUCY_SPEAKER_CARD" if kind is FakeOutput else "LUCY_MICROPHONE_CARD"
            module.card_at_open.append(os.environ.get(variable))
            if module.fail:
                raise OSError("no such device")
            stream = kind(**kwargs)
            stream.timing = module.timing
            if module.fail_start:
                stream.start_error = OSError("the device went away")
            stream.unplug_at_start = module.unplug_at_start
            streams.append(stream)
            return stream

        return factory

    def query_devices(device: Any = None, kind: str | None = None) -> dict[str, Any]:
        if not module.listed:
            raise OSError("Error querying device -1")
        if module.default is not None:
            return {"name": module.default, "hostapi": 0}
        return {"name": "fake"}

    def initialize() -> None:
        module.lists += 1
        if module.plugged_in:
            module.fail, module.listed = False, True

    module.RawOutputStream = make(FakeOutput)
    module.RawInputStream = make(FakeInput)
    module.query_devices = query_devices
    module.query_hostapis = lambda index: {"name": "ALSA"}
    module._terminate = lambda: None
    module._initialize = initialize
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    monkeypatch.setattr(sound, "_streams", sound._Streams())  # no streams left from other tests
    return module


class _NoPortAudio(importlib.abc.MetaPathFinder):
    """Fails ``import sounddevice`` the way it fails on Linux without libportaudio2."""

    def find_spec(self, fullname: str, *_: Any) -> None:
        if fullname == "sounddevice":
            raise OSError("PortAudio library not found")


def test_missing_portaudio_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NoPortAudio(), *sys.meta_path])
    with pytest.raises(AudioDeviceError, match="apt install libportaudio2"):
        SoundDeviceSpeaker(sample_rate=8_000)


def clip(value: int, frames: int) -> Pcm:
    return Pcm(bytes([value, 0]) * frames, 8_000)


async def test_speaker_plays_clips_back_to_back(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    first = speaker.enqueue(clip(1, 100))
    second = speaker.enqueue(clip(2, 50))
    await asyncio.wait_for(asyncio.gather(first, second), 2.0)
    await speaker.close()
    out = bytes(sounddevice.streams[0].output)
    start = out.index(b"\x01\x00")
    # 100 frames of 1s then 50 frames of 2s with no silence between, then silence.
    assert out[start : start + 300] == b"\x01\x00" * 100 + b"\x02\x00" * 50
    assert set(out[start + 300 :]) <= {0}


async def test_stop_silences_and_cancels(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    long = speaker.enqueue(clip(3, 8_000 * 60))
    await until(lambda: bool(sounddevice.streams) and b"\x03" in sounddevice.streams[0].output)
    assert speaker.stop() is True
    assert long.cancelled()
    assert speaker.stop() is False
    await speaker.close()


async def test_cancelled_clip_is_skipped(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await speaker.open()
    skipped = speaker.enqueue(clip(4, 10_000))
    skipped.cancel()
    await asyncio.wait_for(speaker.enqueue(clip(5, 10)), 2.0)
    await speaker.close()
    assert b"\x04" not in sounddevice.streams[0].output


class Listener:
    def __init__(self) -> None:
        self.played: list[tuple[Pcm, float]] = []
        self.stops = 0

    def playing(self, audio: Pcm, at: float) -> None:
        self.played.append((audio, at))

    def stopped(self) -> None:
        self.stops += 1


async def test_speaker_says_when_each_clip_will_be_heard(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    listener = Listener()
    speaker = SoundDeviceSpeaker(
        sample_rate=8_000, blocksize=BLOCK, latency_s=0.2, listener=listener
    )
    first, second = clip(1, 10), clip(2, 10)
    before = time.monotonic()
    done = asyncio.gather(speaker.enqueue(first), speaker.enqueue(second))
    await asyncio.wait_for(done, 2.0)
    await until(lambda: len(listener.played) == 2)
    await speaker.close()
    (heard_first, at_first), (heard_second, at_second) = listener.played
    assert heard_first is first  # the clips themselves, cues and all
    assert heard_second is second
    # No DAC time from this host API, so the output latency stands in for it.
    assert at_first >= before + 0.2
    # Both start in the same buffer: exactly the first clip's length apart.
    assert at_second - at_first == pytest.approx(10 / 8_000)


async def test_speaker_uses_the_dac_time_when_portaudio_has_it(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    sounddevice.timing = types.SimpleNamespace(currentTime=100.0, outputBufferDacTime=100.03)
    listener = Listener()
    speaker = SoundDeviceSpeaker(
        sample_rate=8_000, blocksize=BLOCK, latency_s=0.5, listener=listener
    )
    before = time.monotonic()
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    await until(lambda: len(listener.played) == 1)
    await speaker.close()
    assert listener.played[0][1] - before == pytest.approx(0.03, abs=0.02)


async def test_stop_tells_the_listener(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    listener = Listener()
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, listener=listener)
    speaker.enqueue(clip(3, 8_000 * 60))
    await until(lambda: len(listener.played) == 1)
    speaker.stop()
    assert listener.stops == 1
    await speaker.close()


async def test_a_report_overtaken_by_stop_is_dropped(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    listener = Listener()
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, listener=listener)
    stale = speaker._generation
    speaker.stop()  # barge-in, after the audio thread posted the report but before it ran
    speaker._report(stale, clip(1, 10), 0.0)
    assert listener.played == []


async def test_withdrawing_a_clip_part_way_tells_the_listener(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    listener = Listener()
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, listener=listener)
    playing = speaker.enqueue(clip(4, 8_000 * 60))
    await until(lambda: len(listener.played) == 1)
    playing.cancel()
    await until(lambda: listener.stops == 1)
    await speaker.close()


class FakeMixer:
    """A sound card's own volume, from ``low`` to ``high`` dB (None: it has none)."""

    def __init__(self, card: str, low: float = -60.0, high: float | None = 0.0) -> None:
        self.card = card
        self.low, self.high = low, high
        self.levels: list[float] = []

    def set_db(self, db: float) -> float | None:
        if self.high is None:
            return None
        level = min(self.high, max(self.low, db))
        self.levels.append(level)
        return level


LOUD = 30_000  # a clip near full scale, as her voice arrives


async def peak_of(speaker: Any, sounddevice: types.SimpleNamespace) -> int:
    """The loudest sample of a near-full-scale clip as the speaker plays it now."""
    import numpy as np

    await asyncio.sleep(0.02)  # let a volume change reach the card
    clip = Pcm(np.full(256, LOUD, dtype="<i2").tobytes(), 8_000)
    if sounddevice.streams:
        sounddevice.streams[-1].output.clear()
    await asyncio.wait_for(speaker.enqueue(clip), 2.0)
    return int(np.abs(np.frombuffer(bytes(sounddevice.streams[-1].output), dtype="<i2")).max())


def db_below(peak: int) -> float:
    import math

    return round(20 * math.log10(peak / LOUD), 1)


@pytest.fixture
def mixers() -> list[FakeMixer]:
    return []


def speaker_with_mixer(
    sounddevice: types.SimpleNamespace,
    mixers: list[FakeMixer],
    *,
    low: float = -60.0,
    high: float | None = 0.0,
    **settings: Any,
) -> Any:
    from lucy.adapters.sound import SoundDeviceSpeaker
    from lucy.core.volume import VolumeSettings

    def make(card: str) -> FakeMixer:
        mixers.append(FakeMixer(card, low, high))
        return mixers[-1]

    plug_in(sounddevice.cards, 3, "UACDemoV10")
    volume = VolumeSettings(max_db=9.0, min_db=-15.0, speaker_db=-9.0, speaker_max_db=-3.0)
    volume = dataclasses.replace(volume, **settings)
    return SoundDeviceSpeaker(
        sample_rate=8_000, blocksize=BLOCK, card="LucySpeaker", volume=volume, mixer=make
    )


async def test_louder_changes_only_the_system_mixer(
    sounddevice: types.SimpleNamespace, mixers: list[FakeMixer]
) -> None:
    speaker = speaker_with_mixer(sounddevice, mixers)
    try:
        original_peak = await peak_of(speaker, sounddevice)
        assert mixers[0].card == "UACDemoV10"
        assert mixers[0].levels == [-9.0]  # her usual loudness, set as the speaker opened

        speaker.set_volume(6.0)  # all of it from the speaker's own volume
        assert await peak_of(speaker, sounddevice) == original_peak
        assert mixers[0].levels[-1] == -3.0

        speaker.set_volume(9.0)  # at the system mixer ceiling; no software increase
        assert await peak_of(speaker, sounddevice) == original_peak
        assert mixers[0].levels[-1] == -3.0

        speaker.set_volume(-15.0)  # quieter: the speaker's own volume goes down
        assert await peak_of(speaker, sounddevice) == original_peak
        assert mixers[0].levels[-1] == -24.0
    finally:
        await speaker.close()


async def test_system_mixer_stops_at_its_limit_without_software_gain(
    sounddevice: types.SimpleNamespace, mixers: list[FakeMixer]
) -> None:
    speaker = speaker_with_mixer(sounddevice, mixers, low=-12.0)
    speaker.set_volume(-15.0)  # the card stops at -12 dB: the other 12 dB in software
    await peak_of(speaker, sounddevice)
    assert mixers[0].levels[-1] == -12.0
    peak = await peak_of(speaker, sounddevice)
    speaker.set_volume(9.0)
    assert await peak_of(speaker, sounddevice) == peak
    assert mixers[0].levels[-1] == -12.0
    await speaker.close()


async def test_a_card_with_no_volume_control_does_not_change_software_audio(
    sounddevice: types.SimpleNamespace, mixers: list[FakeMixer]
) -> None:
    speaker = speaker_with_mixer(sounddevice, mixers, high=None)  # an I2S amplifier
    await peak_of(speaker, sounddevice)
    original_peak = await peak_of(speaker, sounddevice)
    speaker.set_volume(9.0)
    assert await peak_of(speaker, sounddevice) == original_peak
    await speaker.close()


async def test_the_speaker_mixer_can_be_turned_off(
    sounddevice: types.SimpleNamespace, mixers: list[FakeMixer]
) -> None:
    speaker = speaker_with_mixer(sounddevice, mixers, speaker_mixer=False)
    original_peak = await peak_of(speaker, sounddevice)
    speaker.set_volume(9.0)
    assert await peak_of(speaker, sounddevice) == original_peak
    assert mixers == []
    await speaker.close()


async def test_a_speaker_plugged_back_in_gets_her_volume_again(
    sounddevice: types.SimpleNamespace, mixers: list[FakeMixer]
) -> None:
    speaker = speaker_with_mixer(sounddevice, mixers)
    speaker.set_volume(3.0)
    await peak_of(speaker, sounddevice)
    sounddevice.streams[0].unplug()  # the card comes back at its own default volume
    await until(lambda: sounddevice.streams[0].closed)
    await peak_of(speaker, sounddevice)
    assert mixers[0].levels == [-6.0, -6.0]  # set again as the new stream opened
    await speaker.close()


async def test_without_settings_the_speaker_plays_at_full_scale(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    assert await peak_of(speaker, sounddevice) == LOUD
    await speaker.close()


async def test_rejects_mismatched_sample_rate(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=22_050)
    with pytest.raises(ValueError, match="22050"):
        speaker.enqueue(clip(1, 10))


async def test_open_failure_fails_waiting_clips(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    sounddevice.fail = True
    speaker = SoundDeviceSpeaker(sample_rate=8_000)
    with pytest.raises(AudioDeviceError):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    with pytest.raises(AudioDeviceError):
        await speaker.open()


async def test_an_unplugged_speaker_fails_what_was_queued(
    sounddevice: types.SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    listener = Listener()
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, listener=listener)
    playing = speaker.enqueue(clip(1, 8_000 * 60))
    waiting = speaker.enqueue(clip(2, 10))
    await until(lambda: len(listener.played) == 1)
    sounddevice.streams[0].unplug()
    # Neither clip will ever be played: they fail now, instead of waiting for ever.
    for future in (playing, waiting):
        with pytest.raises(AudioDeviceError, match="stopped working"):
            await asyncio.wait_for(future, 2.0)
    assert listener.stops == 1  # her beak stops with the sound
    assert "the speaker stopped working" in caplog.text
    await until(lambda: sounddevice.streams[0].closed)  # the dead stream is not left open
    await speaker.close()


async def test_a_speaker_plugged_back_in_plays_the_next_clip(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    sounddevice.streams[0].unplug()  # between two sounds, while Lucy is still awake
    await until(lambda: sounddevice.streams[0].closed)
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    await speaker.close()
    assert len(sounddevice.streams) == 2  # a new stream, with no restart and no doze
    assert b"\x02\x00" * 10 in sounddevice.streams[1].output
    assert sounddevice.lists == 0  # it was on PortAudio's list all along


async def test_no_sound_while_the_speaker_is_unplugged_then_sound_again(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await speaker.open()
    sounddevice.streams[0].unplug()
    sounddevice.fail = True
    await until(lambda: sounddevice.streams[0].closed)
    with pytest.raises(AudioDeviceError, match="cannot open the speaker"):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    sounddevice.fail = False  # plugged back in
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    await speaker.close()


async def test_a_speaker_unplugged_as_it_opens_is_noticed(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    # PortAudio reports the stream stopped before the event loop has been handed it.
    sounddevice.unplug_at_start = True
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    with pytest.raises(AudioDeviceError, match="stopped working"):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    await until(lambda: sounddevice.streams[0].closed)
    sounddevice.unplug_at_start = False  # plugged back in: the next sound opens it again
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    await speaker.close()


@pytest.fixture
def quick_to_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count a stream as stalled after 50 ms without a callback, not a second."""
    from lucy.adapters import sound

    monkeypatch.setattr(sound, "_STALLED_S", 0.05)


@pytest.mark.usefixtures("quick_to_notice")
async def test_a_stream_that_stops_asking_for_audio_is_replaced(
    sounddevice: types.SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    listener = Listener()
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, listener=listener)
    cut_off = speaker.enqueue(clip(1, 8_000 * 60))
    await until(lambda: len(listener.played) == 1)
    dead = sounddevice.streams[0]
    dead.stall()  # unplugged in the middle of a sentence, and plugged straight back in
    # PortAudio has not noticed, and no new sound is queued: the watchdog gives up on it.
    with pytest.raises(AudioDeviceError, match="stopped working"):
        await asyncio.wait_for(cut_off, 2.0)
    assert listener.stops == 1
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)  # the next sound: a new stream
    await until(lambda: b"\x02\x00" * 10 in sounddevice.streams[1].output)
    assert not dead.closed  # PortAudio is still holding on to it, in the background
    dead.let_go.set()
    await until(lambda: dead.closed)
    assert "did not close cleanly" not in caplog.text  # expected of a dead stream: not a warning
    await speaker.close()


@pytest.mark.usefixtures("quick_to_notice")
async def test_closing_does_not_wait_for_a_stream_that_has_stalled(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await speaker.open()
    dead = sounddevice.streams[0]
    dead.stall()
    await asyncio.sleep(0.1)
    await asyncio.wait_for(speaker.close(), 1.0)  # Lucy dozes at once; PortAudio takes its time
    assert not dead.closed
    dead.let_go.set()
    await until(lambda: dead.closed)


async def test_a_quiet_speaker_is_not_taken_for_a_stalled_one(
    sounddevice: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lucy.adapters import sound

    monkeypatch.setattr(sound, "_STALLED_S", 0.2)
    speaker = sound.SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    await asyncio.sleep(0.4)  # awake with nothing to say: the stream still asks, for silence
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    await speaker.close()
    assert len(sounddevice.streams) == 1


async def test_closing_the_speaker_is_not_taken_for_unplugging(
    sounddevice: types.SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    await speaker.close()  # Lucy dozes: PortAudio reports this stream stopped, too
    waiting = speaker.enqueue(clip(2, 8_000 * 60))
    await until(lambda: len(sounddevice.streams) == 2 and b"\x02" in sounddevice.streams[1].output)
    await asyncio.sleep(0.01)
    assert not waiting.done()  # the new stream is not mistaken for the one that was closed
    assert "stopped working" not in caplog.text
    await speaker.close()


async def test_close_survives_a_speaker_that_cannot_be_stopped(
    sounddevice: types.SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await speaker.open()
    stream = sounddevice.streams[0]
    stream.finished_callback = None  # unplugged, and PortAudio never said
    stream.unplug()
    await speaker.close()  # Lucy still dozes, and still shuts down
    assert stream.closed
    assert "did not close cleanly" in caplog.text


async def test_a_speaker_plugged_in_after_start_up_is_found(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    sounddevice.fail, sounddevice.listed = True, False  # Lucy started without her speaker
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    with pytest.raises(AudioDeviceError):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.lists == 1  # looked for it, and it is still not there
    sounddevice.plugged_in = True
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    assert sounddevice.lists == 2
    await speaker.close()


async def test_the_sound_cards_are_not_listed_again_while_recording(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceMicrophone, SoundDeviceSpeaker

    mic = SoundDeviceMicrophone(sample_rate=8_000, block_s=BLOCK / 8_000)
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await mic.start()
    sounddevice.fail, sounddevice.listed, sounddevice.plugged_in = True, False, True
    with pytest.raises(AudioDeviceError):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.lists == 0  # a new list would have cut the recording off
    assert (await mic.stop()).frames > 0
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)  # the wing is let go: now it may
    assert sounddevice.lists == 1
    await speaker.close()


async def test_a_listed_speaker_that_will_not_open_is_not_looked_for(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    sounddevice.fail = True  # unplugged, but on the list: plugging it back in is enough
    speaker = SoundDeviceSpeaker(sample_rate=8_000)
    with pytest.raises(AudioDeviceError):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.lists == 0


async def test_a_stream_that_will_not_start_is_closed(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    sounddevice.fail_start = True
    speaker = SoundDeviceSpeaker(sample_rate=8_000)
    with pytest.raises(AudioDeviceError, match="the device went away"):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.streams[0].closed  # left open, it would keep the sound card busy
    sounddevice.fail_start = False
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    await speaker.close()


def plug_in(cards: Path, number: int, card_id: str, *, record: bool = False) -> None:
    folder = cards / f"card{number}"
    folder.mkdir()
    (folder / "id").write_text(card_id)
    (folder / "usbid").write_text("4c4a:4155")
    (folder / ("pcm0c" if record else "pcm0p")).mkdir()


async def test_a_different_speaker_is_used_when_hers_is_gone(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    plug_in(sounddevice.cards, 2, "LucyMic", record=True)
    plug_in(sounddevice.cards, 3, "UACDemoV10")  # LucySpeaker was swapped for this one
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, card="LucySpeaker")
    assert os.environ["LUCY_SPEAKER_CARD"] == "UACDemoV10"  # before PortAudio lists the cards
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.card_at_open == ["UACDemoV10"]
    await speaker.close()

    plug_in(sounddevice.cards, 4, "LucySpeaker")  # her own is back: the next stream uses it
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    assert sounddevice.card_at_open == ["UACDemoV10", "LucySpeaker"]
    await speaker.close()


async def test_a_speaker_swapped_while_lucy_is_awake_is_used_for_the_next_sound(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    plug_in(sounddevice.cards, 3, "LucySpeaker")
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, card="LucySpeaker")
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    sounddevice.streams[0].unplug()
    (sounddevice.cards / "card3" / "id").unlink()
    (sounddevice.cards / "card3").rename(sounddevice.cards / "gone")
    plug_in(sounddevice.cards, 4, "Device")  # a new speaker, in another port
    await until(lambda: sounddevice.streams[0].closed)
    await asyncio.wait_for(speaker.enqueue(clip(2, 10)), 2.0)
    assert sounddevice.card_at_open == ["LucySpeaker", "Device"]
    await speaker.close()


async def test_with_no_speaker_anywhere_the_error_says_so(
    sounddevice: types.SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    sounddevice.fail = True
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, card="LucySpeaker")
    for value in range(3):
        with pytest.raises(AudioDeviceError, match="no speaker found: LucySpeaker and every USB"):
            await asyncio.wait_for(speaker.enqueue(clip(value, 10)), 2.0)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1  # every sound tries again; it is said once
    sounddevice.fail = False
    plug_in(sounddevice.cards, 3, "UACDemoV10")
    await asyncio.wait_for(speaker.enqueue(clip(4, 10)), 2.0)
    await speaker.close()
    assert "the speaker works again" in caplog.text


async def test_a_gone_speaker_is_not_asked_for(sounddevice: types.SimpleNamespace) -> None:
    """PortAudio can abort the process when asked to open a card that has just gone."""
    from lucy.adapters.sound import SoundDeviceSpeaker

    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, card="LucySpeaker")
    with pytest.raises(AudioDeviceError, match="no speaker found"):
        await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.card_at_open == []
    assert sounddevice.lists == 0
    await speaker.close()


async def test_a_list_made_without_the_speaker_is_made_again(
    sounddevice: types.SimpleNamespace,
) -> None:
    """PortAudio listed the cards while ALSA's default had no speaker behind it, so its
    default output is a raw card that cannot convert sample rates. That list is stale."""
    from lucy.adapters.sound import SoundDeviceSpeaker

    sounddevice.fail, sounddevice.default = True, "UACDemoV1.0: USB Audio (hw:3,0)"

    def initialize() -> None:
        sounddevice.lists += 1
        sounddevice.fail, sounddevice.default = False, "default"

    sounddevice._initialize = initialize
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK)
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.lists == 1
    await speaker.close()


async def test_a_named_device_is_not_routed(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceSpeaker

    plug_in(sounddevice.cards, 3, "UACDemoV10")
    speaker = SoundDeviceSpeaker(sample_rate=8_000, blocksize=BLOCK, device="Jieli")
    await asyncio.wait_for(speaker.enqueue(clip(1, 10)), 2.0)
    assert sounddevice.card_at_open == [None]
    await speaker.close()


async def test_microphone_records_until_stopped(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceMicrophone

    mic = SoundDeviceMicrophone(sample_rate=8_000, block_s=BLOCK / 8_000, max_seconds=1.0)
    await mic.start()
    await until(lambda: len(sounddevice.streams[0].feed) < 900)
    audio = await mic.stop()
    assert audio.sample_rate == 8_000
    assert audio.frames > 0
    assert audio.frames <= 8_000 + BLOCK  # capped at max_seconds (plus one block)
    assert audio.data[:2] == b"\x00\x00"  # first fake block, in order


async def test_microphone_open_failure(sounddevice: types.SimpleNamespace) -> None:
    from lucy.adapters.sound import SoundDeviceMicrophone

    sounddevice.fail = True
    with pytest.raises(AudioDeviceError):
        await SoundDeviceMicrophone().start()


async def test_a_different_microphone_is_used_when_hers_is_gone(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceMicrophone

    plug_in(sounddevice.cards, 2, "Device", record=True)
    mic = SoundDeviceMicrophone(sample_rate=8_000, block_s=BLOCK / 8_000, card="LucyMic")
    await mic.start()
    await mic.stop()
    assert sounddevice.card_at_open == ["Device"]


async def test_a_microphone_plugged_in_after_start_up_is_found(
    sounddevice: types.SimpleNamespace,
) -> None:
    from lucy.adapters.sound import SoundDeviceMicrophone

    sounddevice.fail, sounddevice.listed, sounddevice.plugged_in = True, False, True
    mic = SoundDeviceMicrophone(sample_rate=8_000, block_s=BLOCK / 8_000)
    await mic.start()
    assert sounddevice.lists == 1
    await until(lambda: len(sounddevice.streams[0].feed) < 990)
    assert (await mic.stop()).frames > 0
