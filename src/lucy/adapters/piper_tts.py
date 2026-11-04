"""On-device text-to-speech with Piper (VITS models on onnxruntime).

Speech is synthesised on the Pi, not in the cloud. That keeps every reply
working offline, avoids a second network round trip per sentence, and keeps
the cost at zero. Groq's hosted TTS is a preview model with a small daily
request limit.

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
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lucy.core.audio import Pcm
from lucy.core.errors import ConfigError

log = logging.getLogger(__name__)


class PiperSpeech:
    def __init__(
        self,
        model_path: Path,
        *,
        threads: int = 3,
        length_scale: float | None = None,
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
        self._synthesis = SynthesisConfig(
            speaker_id=speaker_id, length_scale=length_scale, volume=volume
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

    async def synthesize(self, text: str) -> Pcm:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.render, text)

    def render(self, text: str) -> Pcm:
        """Synthesise synchronously (used by the worker thread and by CLI tools)."""
        audio = b"".join(
            chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, self._synthesis)
        )
        return Pcm(audio, self.sample_rate)

    def warm_up(self) -> None:
        self.render("Honk! Hello there.")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
