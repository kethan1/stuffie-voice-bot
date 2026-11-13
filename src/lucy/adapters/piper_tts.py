"""On-device text-to-speech with Piper (VITS models on onnxruntime).

Lucy normally speaks with her natural cloud voice
(:mod:`lucy.adapters.speechify_tts`). Piper is the voice that never fails: it
needs no network, key, or quota, so it speaks whenever the cloud voice cannot,
and it is Lucy's only voice when no cloud voice is set up. It is loaded and
warmed up at start-up so it is ready the moment it is needed.

Tuning for a Raspberry Pi 4 (4 x Cortex-A72):

* The onnxruntime session is built here rather than by ``PiperVoice.load`` so
  we control its options: ``threads`` intra-op threads (default 3, leaving a
  core for audio and the event loop) and, importantly for power,
  ``allow_spinning=0``. By default onnxruntime's worker threads busy-wait
  after each inference, which keeps every core hot and burns power between
  sentences.
* One dedicated worker thread serialises synthesis, so sentences never fight
  each other for cores. onnxruntime releases the GIL, so the event loop keeps
  running.
* :meth:`warm_up` runs one inference at start-up so the first real sentence
  does not pay for memory-arena allocation and kernel selection.

Pace. Lucy's goose voice (:mod:`lucy.core.goose_voice`) raises the pitch of whatever it
is given, which also speeds it up, so the voice has to speak that much more slowly first
(``slower``). Piper's ``length_scale`` does slow it down, but by less than the number
says, because the short gaps Piper puts between sounds do not stretch. :data:`_PACE` is
what it really does, so ``slower`` can be exact.
"""

import asyncio
import copy
import json
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from lucy.core.audio import Pcm
from lucy.core.errors import ConfigError

log = logging.getLogger(__name__)

# (length_scale, how many times longer the speech really is), measured over seven sentences
# with en_US-lessac-medium, the voice Lucy ships with. Other Piper voices differ by a few
# percent.
_PACE = (
    (0.7, 0.77),
    (0.8, 0.875),
    (0.9, 0.955),
    (1.0, 1.0),
    (1.1, 1.05),
    (1.2, 1.115),
    (1.3, 1.19),
    (1.4, 1.27),
    (1.5, 1.35),
    (1.75, 1.545),
    (2.0, 1.74),
)


def slowed_length_scale(length_scale: float, slower: float) -> float:
    """The length scale at which Piper speaks ``slower`` times as slowly as at ``length_scale``."""
    if slower == 1.0:
        return length_scale
    scales, lengths = zip(*_PACE, strict=True)
    return float(np.interp(np.interp(length_scale, scales, lengths) * slower, lengths, scales))


class PiperSpeech:
    def __init__(
        self,
        model_path: Path,
        *,
        threads: int = 3,
        length_scale: float | None = None,
        slower: float = 1.0,
        volume: float = 1.0,
        speaker_id: int | None = None,
    ) -> None:
        import onnxruntime
        from piper import PiperConfig, PiperVoice, SynthesisConfig

        config_path = Path(f"{model_path}.json")
        if not model_path.is_file() or not config_path.is_file():
            raise ConfigError(
                f"Piper voice not found: need {model_path} and {config_path} "
                "(deploy/install.sh installs it on the Pi, `make voice` on a laptop)"
            )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._voice = PiperVoice(
            session=session,
            config=PiperConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8"))),
        )
        self._length_scale = (
            float(self._voice.config.length_scale) if length_scale is None else length_scale
        )
        self._synthesis = SynthesisConfig(
            speaker_id=speaker_id,
            length_scale=slowed_length_scale(self._length_scale, slower),
            volume=volume,
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        log.info(
            "loaded Piper voice %s (%d Hz, %d threads)",
            model_path.name,
            self.sample_rate,
            options.intra_op_num_threads,
        )

    @property
    def sample_rate(self) -> int:
        return int(self._voice.config.sample_rate)

    async def stream(self, text: str) -> AsyncGenerator[Pcm, None]:
        """One part per call: Piper renders a sentence in a fraction of its spoken length."""
        loop = asyncio.get_running_loop()
        yield await loop.run_in_executor(self._executor, self.render, text)

    def render(self, text: str) -> Pcm:
        """Synthesise synchronously (used by the worker thread and by CLI tools)."""
        audio = b"".join(
            chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, self._synthesis)
        )
        return Pcm(audio, self.sample_rate)

    def slowed(self, slower: float) -> "PiperSpeech":
        """This voice speaking ``slower`` times as slowly: the same loaded model and worker
        thread, so it costs nothing, and closing either closes both."""
        voice = copy.copy(self)
        voice._synthesis = replace(
            self._synthesis, length_scale=slowed_length_scale(self._length_scale, slower)
        )
        return voice

    def warm_up(self) -> None:
        self.render("Honk! Hello there.")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
