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

Its master volume (:meth:`SoundDeviceSpeaker.set_volume`, which a child changes by
asking, :mod:`lucy.core.volume`) scales each clip as it is queued. The listener is
told about the clip as it was made, so Lucy's beak opens just as wide at any volume.

A speaker that comes unplugged (a USB speaker's cable works loose inside a stuffed goose)
is opened again for the next sound once it is back, with no restart:

* A stream whose sound card has gone stops asking for audio. PortAudio says so in the
  end (on ALSA it first polls the card some 2,000 times: 13 seconds on Lucy's Pi, during
  which it calls the stream active and cannot stop it). Sooner than that, the speaker
  notices that nothing has been asked for in the last second: a watchdog looks twice a
  second while clips are waiting, and the next clip to be queued looks too. Either way
  the speaker fails the clips that were waiting (they would never be played) and forgets
  the stream, so the next clip opens a new one.
* While no speaker is plugged in, a sound fails at once, without asking PortAudio, which
  on ALSA can abort the whole process when asked to open a card that has just gone.
* PortAudio lists the sound cards only once, when it is loaded. A card that was not
  there then (Lucy started with her speaker unplugged) is found by having PortAudio
  list them again, which :class:`_Streams` does when no stream is open.

A speaker swapped for a different one works the same way. With no device configured, Lucy
plays through ALSA's default device, and just before each stream opens
:mod:`lucy.adapters.sound_cards` points that at the card to use: the configured one
(``audio.speaker_card``) if it is plugged in, or else whichever USB speaker is.
"""

import asyncio
import contextlib
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from time import monotonic
from typing import Any

from lucy.adapters import sound_cards
from lucy.adapters.sound_cards import Kind
from lucy.core.audio import SAMPLE_WIDTH, Pcm, adjust_volume
from lucy.core.errors import AudioDeviceError
from lucy.core.ports import PlaybackListener

log = logging.getLogger(__name__)

Device = int | str | None

# A running stream asks for audio many times a second. One that has not asked for this long
# has lost its sound card, though PortAudio still calls it active.
_STALLED_S = 1.0


def import_sounddevice() -> Any:
    """Import ``sounddevice``, or say how to install the PortAudio library it loads."""
    try:
        import sounddevice
    except OSError as error:  # its macOS and Windows wheels bundle PortAudio; Linux needs it
        raise AudioDeviceError(
            f"{error} (on Debian or Ubuntu: sudo apt install libportaudio2)"
        ) from error
    return sounddevice


class _Streams:
    """Opens and closes PortAudio streams, and has it look for sound cards plugged in since.

    PortAudio lists the sound cards once, when it is loaded, and can only open what is on
    that list. Listing them again closes every stream that is open. So a stream whose card
    is not on the list gets a second try after a new list, but only while no other stream
    is open: looking for the speaker never cuts off what the microphone is recording.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()  # one stream is opened at a time, so the count is right
        self._open = 0

    def open(
        self, sd: Any, device: Device, kind: Kind, create: Callable[[], Any], card: str = ""
    ) -> Any:
        """The started stream ``create`` makes on ``device`` (``kind``: "input" or "output").

        With no ``device``, ALSA's default is pointed at the card to use first (``card`` if
        it is plugged in, see :mod:`lucy.adapters.sound_cards`).
        """
        with self._lock:
            route = sound_cards.route(kind, card) if device is None else None
            if route is not None and route.card is None and route.wanted:
                # Nothing to open. Asked to, PortAudio may even abort() the whole process on a
                # card that has just gone (seen on Lucy's Pi: an assertion in pa_linux_alsa.c).
                raise AudioDeviceError(str(route))
            try:
                try:
                    stream = create()
                except Exception:
                    if self._open or _listed(sd, device, kind):
                        raise  # a new list would close a stream in use, or would not help
                    with contextlib.suppress(Exception):
                        sd._terminate()
                    sd._initialize()
                    stream = create()
                    log.info("found the %s sound card that was plugged in", kind)
            except Exception as error:
                if route is not None and route.card is None:
                    raise AudioDeviceError(f"{error} ({route})") from error
                raise
            self._open += 1
            return stream

    def shut(self, stream: Any) -> None:
        try:
            try:
                stream.stop()
            finally:
                stream.close()
        finally:
            with self._lock:
                self._open -= 1


def _listed(sd: Any, device: Device, kind: str) -> bool:
    """True if ``device`` is on PortAudio's list of sound cards, as an input or an output.

    With no device named, PortAudio uses ALSA's "default" device. If that had no card behind
    it when PortAudio made its list, PortAudio falls back to the first raw card instead, which
    cannot convert sample rates ("Invalid sample rate"): the list is out of date then too.
    """
    try:
        info = sd.query_devices(device, kind)
        if device is None and info.get("hostapi") is not None:
            alsa = sd.query_hostapis(info["hostapi"]).get("name") == "ALSA"
            return not alsa or info.get("name") == "default"
    except Exception:
        return False
    return True


def _started(stream: Any) -> Any:
    """Start a new stream, or close it: one left open would keep its sound card busy."""
    try:
        stream.start()
    except BaseException:
        stream.close()
        raise
    return stream


_streams = _Streams()


class SoundDeviceMicrophone:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        device: Device = None,
        card: str = "",
        max_seconds: float = 20.0,
        block_s: float = 0.05,
    ) -> None:
        if device is None:
            sound_cards.route("input", card)  # before PortAudio first lists the sound cards
        self._sd = import_sounddevice()
        self._sample_rate = sample_rate
        self._device = device
        self._card = card
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
            self._stream = await asyncio.to_thread(
                _streams.open, self._sd, self._device, "input", self._create_stream, self._card
            )
        except Exception as error:
            raise AudioDeviceError(f"cannot open the microphone: {error}") from error

    async def stop(self) -> Pcm:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                await asyncio.to_thread(_streams.shut, stream)
            except Exception as error:
                raise AudioDeviceError(f"cannot stop the microphone: {error}") from error
        if self._overflows:
            log.warning("microphone dropped audio %d time(s)", self._overflows)
        data, self._chunks = b"".join(self._chunks), []
        return Pcm(data, self._sample_rate)

    def _create_stream(self) -> Any:
        return _started(
            self._sd.RawInputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                device=self._device,
                blocksize=self._blocksize,
                latency="low",
                callback=self._on_audio,
            )
        )

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
        card: str = "",
        latency_s: float = 0.08,
        blocksize: int = 1024,
        listener: PlaybackListener | None = None,
    ) -> None:
        if device is None:
            sound_cards.route("output", card)  # before PortAudio first lists the sound cards
        self._sd = import_sounddevice()
        self._sample_rate = sample_rate
        self._device = device
        self._card = card
        self._latency_s = latency_s
        self._delay_s = latency_s  # until the stream reports its real output latency
        self._blocksize = blocksize
        self._listener = listener
        self._silence = bytes(blocksize * SAMPLE_WIDTH)
        self._lock = threading.Lock()
        self._queue: deque[_Clip] = deque()
        self._generation = 0  # bumped by stop(): reports from before it are stale
        self._stream: Any = None
        self._ended: threading.Event | None = None  # set when PortAudio stops self._stream
        self._asked_at = 0.0  # when the stream last asked for audio (or was started)
        self._opening: asyncio.Task[None] | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._discarding: set[asyncio.Task[None]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._volume_db = 0.0
        self._failing = False  # the last try to open the speaker failed (said once, not again)

    def set_volume(self, db: float) -> None:
        self._volume_db = db

    def enqueue(self, audio: Pcm) -> asyncio.Future[None]:
        if audio.sample_rate != self._sample_rate:
            raise ValueError(f"speaker runs at {self._sample_rate} Hz, got {audio.sample_rate} Hz")
        self._loop = asyncio.get_running_loop()
        self._check_stream()  # before this clip is queued: it is for the speaker as it is now
        future: asyncio.Future[None] = self._loop.create_future()
        played = adjust_volume(audio, self._volume_db)
        clip = _Clip(audio, memoryview(played.data), future)
        with self._lock:
            self._queue.append(clip)
        future.add_done_callback(partial(self._withdraw, clip))
        self._ensure_open()
        if self._watchdog is None or self._watchdog.done():
            self._watchdog = asyncio.create_task(self._watch(), name="speaker-watchdog")
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
        self._check_stream()
        self._ensure_open()
        if self._opening is not None:
            await asyncio.shield(self._opening)
        if self._stream is None:
            raise AudioDeviceError("the speaker could not be opened")

    async def close(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
        if self._opening is not None and not self._opening.done():
            await asyncio.wait({self._opening})
        self._check_stream()  # stopping a stream whose card has gone can take PortAudio a while
        stream, self._stream, self._ended = self._stream, None, None
        self.stop()
        if stream is not None:
            await self._shut(stream, logging.WARNING)
            log.debug("speaker closed")

    def _ensure_open(self) -> None:
        if self._stream is None and (self._opening is None or self._opening.done()):
            self._opening = asyncio.create_task(self._open(), name="speaker-open")

    async def _open(self) -> None:
        ended = threading.Event()
        create = partial(self._create_stream, ended)
        try:
            stream = await asyncio.to_thread(
                _streams.open, self._sd, self._device, "output", create, self._card
            )
        except Exception as error:
            # Every sound tries again, so while the speaker is gone this is said only once.
            log.log(
                logging.DEBUG if self._failing else logging.ERROR,
                "cannot open the speaker: %s",
                error,
            )
            self._failing = True
            self._fail_queued(f"cannot open the speaker: {error}")
            return
        self._stream, self._ended = stream, ended
        if self._failing:
            self._failing = False
            log.warning("the speaker works again")
        log.debug("speaker opened")
        if ended.is_set():
            self._lost()  # it stopped before the event loop knew it had started

    def _create_stream(self, ended: threading.Event) -> Any:
        def finished() -> None:
            """PortAudio: the stream has stopped, because close() stopped it or its card went."""
            ended.set()
            if self._loop is not None:
                with contextlib.suppress(RuntimeError):  # the loop has closed: shutting down
                    self._loop.call_soon_threadsafe(self._stream_ended, ended)

        stream = self._sd.RawOutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            device=self._device,
            blocksize=self._blocksize,
            latency=self._latency_s,
            callback=self._fill,
            finished_callback=finished,
        )
        self._delay_s = float(getattr(stream, "latency", None) or self._latency_s)
        self._asked_at = monotonic()
        return _started(stream)

    def _stream_ended(self, ended: threading.Event) -> None:
        """Event loop: a stream has stopped. If close() did not stop it, the speaker is gone."""
        if ended is self._ended:
            self._lost()

    async def _watch(self) -> None:
        """While clips wait to be played, look for a stream that has stopped asking for them.

        Without this, a reply that was being spoken when the speaker was unplugged would wait
        for PortAudio to give up on the card (13 seconds), with Lucy stuck mid-sentence.
        """
        while True:
            await asyncio.sleep(_STALLED_S / 2)
            with self._lock:
                waiting = bool(self._queue)
            if not waiting:
                return  # the next clip starts another watch
            self._check_stream()

    def _check_stream(self) -> None:
        """Notice a stream that has stopped asking for audio before PortAudio says it has."""
        if self._stream is not None and monotonic() - self._asked_at > _STALLED_S:
            self._lost()

    def _lost(self) -> None:
        """The speaker went away (unplugged, most likely): be ready to open it again.

        Nothing queued will be played, so those clips fail now instead of waiting for ever,
        and the next clip finds no stream and opens a new one. The old stream is closed in
        the background: PortAudio may take a while to let go of a card that has gone.
        """
        stream, self._stream, self._ended = self._stream, None, None
        log.warning("the speaker stopped working (unplugged?); the next sound opens it again")
        self._fail_queued("the speaker stopped working")
        if self._listener is not None:
            self._listener.stopped()
        discard = asyncio.create_task(self._shut(stream, logging.DEBUG), name="speaker-discard")
        self._discarding.add(discard)
        discard.add_done_callback(self._discarding.discard)

    def _fail_queued(self, reason: str) -> None:
        with self._lock:
            clips = list(self._queue)
            self._queue.clear()
            self._generation += 1
        for clip in clips:
            if not clip.future.done():
                clip.future.set_exception(AudioDeviceError(reason))

    async def _shut(self, stream: Any, level: int) -> None:
        """Stop and close a stream. A speaker that has just been unplugged may refuse both."""
        try:
            await asyncio.to_thread(_streams.shut, stream)
        except Exception as error:
            log.log(level, "the speaker did not close cleanly: %s", error)

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
        self._asked_at = monotonic()
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
