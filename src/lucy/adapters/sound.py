"""Microphone and speaker on PortAudio/ALSA (via ``sounddevice``).

Power and privacy:

* The microphone stream exists only while the button is held. When Lucy is
  not listening, nothing is reading the microphone at all, not even into a
  buffer.
* The speaker stream is opened when a conversation starts and closed when
  Lucy goes dormant. Closing it lets the I2S amplifier power down (the
  ``max98357a`` overlay drives its SD pin from the PCM state) and stops the
  audio callback from waking the CPU.

Playback runs from a PortAudio callback fed by a queue of clips. Consecutive
clips are spliced sample-accurately (no gaps between sentences), an empty
queue plays silence instead of underrunning, and ``stop()`` takes effect
within one audio block (about 46 ms) for barge-in.

The speaker can report to a :class:`~lucy.core.ports.PlaybackListener` (Lucy's
motors) when each clip will actually be heard. The callback knows when the
buffer it fills reaches the DAC (PortAudio's ``outputBufferDacTime``), so the
report is accurate to a sample, however much audio is queued ahead of it.
"""

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import Any

from lucy.core.audio import SAMPLE_WIDTH, Pcm
from lucy.core.errors import AudioDeviceError
from lucy.core.ports import PlaybackListener

log = logging.getLogger(__name__)

Device = int | str | None


def import_sounddevice() -> Any:
    """Import ``sounddevice``, or say how to install the PortAudio library it loads."""
    try:
        import sounddevice
    except OSError as error:  # its macOS and Windows wheels bundle PortAudio; Linux needs it
        raise AudioDeviceError(
            f"{error} (on Debian or Ubuntu: sudo apt install libportaudio2)"
        ) from error
    return sounddevice


class SoundDeviceMicrophone:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        device: Device = None,
        max_seconds: float = 20.0,
        block_s: float = 0.05,
    ) -> None:
        self._sd = import_sounddevice()
        self._sample_rate = sample_rate
        self._device = device
        self._max_bytes = round(max_seconds * sample_rate) * SAMPLE_WIDTH
        self._blocksize = round(block_s * sample_rate)
        self._stream: Any = None
        self._chunks: list[bytes] = []
        self._captured = 0
        self._overflows = 0

    async def start(self) -> None:
        if self._stream is not None:
            return
        self._chunks, self._captured, self._overflows = [], 0, 0
        try:
            self._stream = await asyncio.to_thread(self._open)
        except Exception as error:
            raise AudioDeviceError(f"cannot open the microphone: {error}") from error

    async def stop(self) -> Pcm:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                await asyncio.to_thread(_shut, stream)
            except Exception as error:
                raise AudioDeviceError(f"cannot stop the microphone: {error}") from error
        if self._overflows:
            log.warning("microphone dropped audio %d time(s)", self._overflows)
        data, self._chunks = b"".join(self._chunks), []
        return Pcm(data, self._sample_rate)

    def _open(self) -> Any:
        stream = self._sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            device=self._device,
            blocksize=self._blocksize,
            latency="low",
            callback=self._on_audio,
        )
        stream.start()
        return stream

    def _on_audio(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        if status.input_overflow:
            self._overflows += 1
        if self._captured < self._max_bytes:
            chunk = bytes(indata)
            self._chunks.append(chunk)
            self._captured += len(chunk)


@dataclass(eq=False, slots=True)
class _Clip:
    audio: Pcm
    data: memoryview
    future: asyncio.Future[None]
    offset: int = 0


class SoundDeviceSpeaker:
    def __init__(
        self,
        *,
        sample_rate: int,
        device: Device = None,
        latency_s: float = 0.08,
        blocksize: int = 1024,
        listener: PlaybackListener | None = None,
    ) -> None:
        self._sd = import_sounddevice()
        self._sample_rate = sample_rate
        self._device = device
        self._latency_s = latency_s
        self._delay_s = latency_s  # until the stream reports its real output latency
        self._blocksize = blocksize
        self._listener = listener
        self._silence = bytes(blocksize * SAMPLE_WIDTH)
        self._lock = threading.Lock()
        self._queue: deque[_Clip] = deque()
        self._generation = 0  # bumped by stop(): reports from before it are stale
        self._stream: Any = None
        self._opening: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def enqueue(self, audio: Pcm) -> asyncio.Future[None]:
        if audio.sample_rate != self._sample_rate:
            raise ValueError(f"speaker runs at {self._sample_rate} Hz, got {audio.sample_rate} Hz")
        self._loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = self._loop.create_future()
        clip = _Clip(audio, memoryview(audio.data), future)
        with self._lock:
            self._queue.append(clip)
        future.add_done_callback(partial(self._withdraw, clip))
        self._ensure_open()
        return future

    def stop(self) -> bool:
        with self._lock:
            clips = list(self._queue)
            self._queue.clear()
            self._generation += 1
        for clip in clips:
            clip.future.cancel()
        if self._listener is not None:
            self._listener.stopped()
        return bool(clips)

    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._ensure_open()
        if self._opening is not None:
            await asyncio.shield(self._opening)
        if self._stream is None:
            raise AudioDeviceError("the speaker could not be opened")

    async def close(self) -> None:
        if self._opening is not None and not self._opening.done():
            await asyncio.wait({self._opening})
        stream, self._stream = self._stream, None
        self.stop()
        if stream is not None:
            await asyncio.to_thread(_shut, stream)
            log.debug("speaker closed")

    def _ensure_open(self) -> None:
        if self._stream is None and (self._opening is None or self._opening.done()):
            self._opening = asyncio.create_task(self._open(), name="speaker-open")

    async def _open(self) -> None:
        try:
            self._stream = await asyncio.to_thread(self._create_stream)
            log.debug("speaker opened")
        except Exception as error:
            log.error("cannot open the speaker: %s", error)
            with self._lock:
                clips = list(self._queue)
                self._queue.clear()
            for clip in clips:
                if not clip.future.done():
                    clip.future.set_exception(AudioDeviceError(f"cannot open the speaker: {error}"))

    def _create_stream(self) -> Any:
        stream = self._sd.RawOutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            device=self._device,
            blocksize=self._blocksize,
            latency=self._latency_s,
            callback=self._fill,
        )
        self._delay_s = float(getattr(stream, "latency", None) or self._latency_s)
        stream.start()
        return stream

    def _withdraw(self, clip: _Clip, future: asyncio.Future[None]) -> None:
        if not future.cancelled():
            return
        with self._lock:
            try:
                self._queue.remove(clip)
            except ValueError:
                return
            cut_short = clip.offset > 0
            if cut_short:
                self._generation += 1
        if cut_short and self._listener is not None:
            self._listener.stopped()  # the rest of a clip that had started will not be heard

    def _fill(self, outdata: Any, frames: int, timing: Any, status: Any) -> None:
        """PortAudio callback (audio thread): copy queued clips, pad with silence."""
        need = len(outdata)
        filled = 0
        finished: list[asyncio.Future[None]] = []
        started: list[tuple[Pcm, int]] = []  # clips starting in this buffer, at which byte
        with self._lock:
            generation = self._generation
            while filled < need and self._queue:
                clip = self._queue[0]
                if clip.offset == 0 and self._listener is not None:
                    started.append((clip.audio, filled))
                take = min(need - filled, len(clip.data) - clip.offset)
                outdata[filled : filled + take] = clip.data[clip.offset : clip.offset + take]
                clip.offset += take
                filled += take
                if clip.offset >= len(clip.data):
                    self._queue.popleft()
                    finished.append(clip.future)
        if filled < need:
            outdata[filled:need] = self._silence[: need - filled]
        if self._loop is None:
            return
        for future in finished:
            self._loop.call_soon_threadsafe(_resolve, future)
        if started:
            heard_at = monotonic() + self._output_delay(timing)
            for audio, offset in started:
                at = heard_at + offset / SAMPLE_WIDTH / self._sample_rate
                self._loop.call_soon_threadsafe(self._report, generation, audio, at)

    def _output_delay(self, timing: Any) -> float:
        """Seconds until the first sample of the buffer being filled is heard."""
        dac = getattr(timing, "outputBufferDacTime", 0.0) or 0.0
        now = getattr(timing, "currentTime", 0.0) or 0.0
        if dac > 0 and now > 0 and 0 <= dac - now < 1.0:
            return float(dac - now)
        return self._delay_s  # some host APIs leave the times at zero

    def _report(self, generation: int, audio: Pcm, at: float) -> None:
        """Event loop: tell the listener a clip is playing, unless stop() came since."""
        if generation == self._generation and self._listener is not None:
            self._listener.playing(audio, at)


def _resolve(future: asyncio.Future[None]) -> None:
    if not future.done():
        future.set_result(None)


def _shut(stream: Any) -> None:
    try:
        stream.stop()
    finally:
        stream.close()
