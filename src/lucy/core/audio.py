"""PCM audio value type and the small amount of DSP Lucy needs.

Everything in Lucy moves audio around as :class:`Pcm`: mono, signed 16-bit,
little-endian samples plus a sample rate. Keeping one canonical format means
no adapter ever has to guess what it was handed.

A clip can also carry :class:`Cue` regions, like the cue regions of a WAV file:
labelled stretches that say what part of the sound is (a honk, a chime). They
follow the audio through joins, so Lucy's motors (:mod:`lucy.core.motion`) know
when she honks even inside a pre-rendered line.
"""

import io
import math
import wave
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Self

import numpy as np
import numpy.typing as npt

SAMPLE_WIDTH = 2  # bytes per int16 sample
_INT16_FULL_SCALE = 32768.0
SILENCE_DBFS = -120.0


class CueKind(StrEnum):
    HONK = "honk"
    """A goose call, which Lucy flaps through."""
    CHIME = "chime"
    """An earcon: a tone, not Lucy's voice, so her beak stays shut."""


@dataclass(frozen=True, slots=True)
class Cue:
    """A labelled stretch of a clip, in seconds from its start."""

    kind: CueKind
    start_s: float
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def shifted(self, seconds: float) -> "Cue":
        return replace(self, start_s=self.start_s + seconds)


@dataclass(frozen=True, slots=True)
class Pcm:
    """Mono signed 16-bit little-endian PCM audio."""

    data: bytes
    sample_rate: int
    cues: tuple[Cue, ...] = ()

    @classmethod
    def silence(cls, seconds: float, sample_rate: int) -> Self:
        frames = max(0, round(seconds * sample_rate))
        return cls(bytes(frames * SAMPLE_WIDTH), sample_rate)

    @classmethod
    def from_float(cls, samples: npt.NDArray[np.floating], sample_rate: int) -> Self:
        """Build from float samples in [-1.0, 1.0] (values outside are clipped)."""
        clipped = np.clip(samples, -1.0, 1.0) * (_INT16_FULL_SCALE - 1)
        return cls(clipped.astype("<i2").tobytes(), sample_rate)

    @property
    def frames(self) -> int:
        return len(self.data) // SAMPLE_WIDTH

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def samples(self) -> npt.NDArray[np.int16]:
        return np.frombuffer(self.data, dtype="<i2")

    def cued(self, kind: CueKind) -> "Pcm":
        """This clip, marked from start to end as ``kind``."""
        return replace(self, cues=(*self.cues, Cue(kind, 0.0, self.duration)))

    def __add__(self, other: "Pcm") -> "Pcm":
        return join((self, other), self.sample_rate)

    def to_wav(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.data)
        return buffer.getvalue()


def join(parts: Iterable[Pcm], sample_rate: int) -> Pcm:
    """``parts`` one after another as a single clip, with their cues moved to match."""
    data: list[bytes] = []
    cues: list[Cue] = []
    offset = 0.0
    for part in parts:
        if part.sample_rate != sample_rate:
            raise ValueError(f"cannot join {part.sample_rate} Hz audio to {sample_rate} Hz audio")
        data.append(part.data)
        cues += (cue.shifted(offset) for cue in part.cues)
        offset += part.duration
    return Pcm(b"".join(data), sample_rate, tuple(cues))


def amplify(pcm: Pcm, gain: float) -> Pcm:
    """Scale the volume by ``gain`` (1.0 leaves it unchanged), clipping at full scale."""
    if gain == 1.0:
        return pcm
    samples = pcm.samples().astype(np.float32) * gain
    clipped = np.clip(samples, -_INT16_FULL_SCALE, _INT16_FULL_SCALE - 1)
    return Pcm(clipped.astype("<i2").tobytes(), pcm.sample_rate, pcm.cues)


def level_dbfs(pcm: Pcm) -> float:
    """RMS level relative to full scale; :data:`SILENCE_DBFS` for empty or silent audio."""
    if pcm.frames == 0:
        return SILENCE_DBFS
    samples = pcm.samples().astype(np.float32) / _INT16_FULL_SCALE
    rms = math.sqrt(float(np.mean(samples * samples)))
    return SILENCE_DBFS if rms <= 1e-6 else max(SILENCE_DBFS, 20.0 * math.log10(rms))


def trim_silence(
    pcm: Pcm, threshold_dbfs: float, *, frame_s: float = 0.02, pad_s: float = 0.15
) -> Pcm:
    """Drop leading and trailing frames quieter than ``threshold_dbfs``.

    Shorter uploads transcribe faster, and Whisper is far less prone to
    hallucinating words into silence it never receives. ``pad_s`` of context is
    kept on each side so soft consonants at the edges survive.
    """
    frame = max(1, round(frame_s * pcm.sample_rate))
    count = pcm.frames // frame
    if count == 0:
        return pcm
    samples = pcm.samples()[: count * frame].astype(np.float32) / _INT16_FULL_SCALE
    rms = np.sqrt(np.mean(samples.reshape(count, frame) ** 2, axis=1))
    loud = np.flatnonzero(rms > 10.0 ** (threshold_dbfs / 20.0))
    if loud.size == 0:
        return Pcm(b"", pcm.sample_rate)
    pad = round(pad_s / frame_s)
    start = max(0, int(loud[0]) - pad) * frame
    end = min(pcm.frames, (int(loud[-1]) + 1 + pad) * frame)
    return Pcm(pcm.data[start * SAMPLE_WIDTH : end * SAMPLE_WIDTH], pcm.sample_rate)


def chime(notes: list[tuple[float, float]], sample_rate: int, *, volume: float = 0.25) -> Pcm:
    """Render a short sequence of ``(frequency_hz, seconds)`` sine notes.

    Each note gets a raised-cosine envelope so there are no clicks. Used for the
    tiny "I'm listening" / "got it" earcons, which are generated at start-up
    rather than shipped as binary assets. The result is cued as a chime, so
    Lucy's beak stays shut while it plays.
    """
    parts: list[npt.NDArray[np.float64]] = []
    for frequency, seconds in notes:
        n = max(1, round(seconds * sample_rate))
        t = np.arange(n) / sample_rate
        envelope = np.sin(np.pi * np.arange(n) / n) ** 2
        parts.append(np.sin(2 * np.pi * frequency * t) * envelope)
    return Pcm.from_float(np.concatenate(parts) * volume, sample_rate).cued(CueKind.CHIME)
