"""On-device speech-to-text with faster-whisper (CTranslate2), for when the cloud is unreachable.

The default model is Whisper ``tiny.en`` quantised to int8: about 75 MB on
disk, a little over 100 MB in RAM, and fast enough on a Pi 4 for the short
questions children ask. ``base.en`` is more accurate but roughly twice as slow.

* The model is loaded from a local directory only (``local_files_only``), with
  files installed and checksum-verified by ``deploy/install.sh``. Nothing is
  downloaded at run time, which matters because this runs precisely when the
  network is down.
* Decoding is greedy (``beam_size=1``) without timestamps or conditioning on
  earlier text: the cheapest settings, and the right ones for one short
  utterance.
* As with the cloud transcriber, segments Whisper itself rates as non-speech
  are dropped.
* One dedicated worker thread loads and runs the model. CTranslate2 releases
  the GIL, so the event loop and audio keep running.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from lucy.core.audio import Pcm
from lucy.core.errors import ConfigError, ServiceError

log = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16_000


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_dir: Path,
        *,
        language: str = "en",
        threads: int = 4,
        no_speech_threshold: float = 0.6,
    ) -> None:
        self._model_dir = model_dir
        self._language = language
        self._threads = threads
        self._no_speech_threshold = no_speech_threshold
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local-stt")
        self._model: Any = None
        self._loading = asyncio.Lock()

    async def preload(self) -> None:
        """Load the model now, so the first offline turn does not wait for it."""
        await self._load()

    async def transcribe(self, speech: Pcm) -> str:
        if speech.sample_rate != WHISPER_SAMPLE_RATE:
            raise ConfigError(f"local speech recognition needs {WHISPER_SAMPLE_RATE} Hz audio")
        model = await self._load()
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._run, model, speech)
        except Exception as error:
            raise ServiceError(f"local speech recognition failed: {error}") from error

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _load(self) -> Any:
        async with self._loading:
            if self._model is None:
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(self._executor, self._load_blocking)
        return self._model

    def _load_blocking(self) -> Any:
        if not (self._model_dir / "model.bin").is_file():
            raise ConfigError(
                f"local speech model not found in {self._model_dir} "
                "(run deploy/install.sh --local-backup)"
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ConfigError(
                "faster-whisper is not installed (install the 'local' extra)"
            ) from error
        model = WhisperModel(
            str(self._model_dir),
            device="cpu",
            compute_type="int8",
            cpu_threads=self._threads,
            local_files_only=True,
        )
        log.info("loaded local speech model %s (%d threads)", self._model_dir.name, self._threads)
        return model

    def _run(self, model: Any, speech: Pcm) -> str:
        samples = speech.samples().astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            samples,
            language=self._language,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # silence is already trimmed before this point
        )
        kept = [s.text.strip() for s in segments if s.no_speech_prob < self._no_speech_threshold]
        return " ".join(kept).strip()
