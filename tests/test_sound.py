"""The PortAudio adapters, driven by a simulated audio thread instead of a sound card."""

import asyncio
import importlib.abc
import sys
import threading
import time
import types
from collections.abc import Callable
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

    def __init__(self, *, callback: Callable[..., None], blocksize: int, **_: Any) -> None:
        self.callback = callback
        self.blocksize = blocksize
        self.output = bytearray()
        self.feed = [bytes([i % 256, 0]) * blocksize for i in range(1000)]
        self.timing: Any = None  # PortAudio's time info, as the callback is handed it
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running.is_set():
            self.tick()
            time.sleep(0.0005)

    def tick(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join()

    def close(self) -> None:
        pass


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
def sounddevice(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    streams: list[FakeStream] = []
    module = types.SimpleNamespace(fail=False, streams=streams, timing=None)

    def make(kind: type[FakeStream]) -> Callable[..., FakeStream]:
        def factory(**kwargs: Any) -> FakeStream:
            if module.fail:
                raise OSError("no such device")
            stream = kind(**kwargs)
            stream.timing = module.timing
            streams.append(stream)
            return stream

        return factory

    module.RawOutputStream = make(FakeOutput)
    module.RawInputStream = make(FakeInput)
    monkeypatch.setitem(sys.modules, "sounddevice", module)
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
