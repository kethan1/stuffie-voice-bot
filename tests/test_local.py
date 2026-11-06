"""The offline backup: on-device speech recognition and the Ollama-served local brain."""

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from lucy.adapters.cloud import ChatModelSpec, StreamingChat
from lucy.adapters.local_stt import FasterWhisperTranscriber
from lucy.adapters.openai_api import ApiClient
from lucy.brains import Local, build_local
from lucy.config import Config, ConfigError, LocalSettings, load_config
from lucy.core.audio import Pcm
from lucy.core.errors import ServiceError
from tests.fakes import tone


class Segment:
    def __init__(self, text: str, no_speech_prob: float) -> None:
        self.text = text
        self.no_speech_prob = no_speech_prob


@pytest.fixture
def faster_whisper(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    module = types.SimpleNamespace(loads=[], calls=[], fail=False)

    class WhisperModel:
        def __init__(self, path: str, **kwargs: Any) -> None:
            module.loads.append((path, kwargs))

        def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, None]:
            if module.fail:
                raise RuntimeError("ctranslate2 exploded")
            module.calls.append((audio, kwargs))
            segments = [Segment(" Why do geese honk?", 0.05), Segment(" Thank you.", 0.9)]
            return iter(segments), None

    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return module


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    (tmp_path / "model.bin").write_bytes(b"weights")
    return tmp_path


class TestFasterWhisper:
    async def test_transcribes_offline_and_drops_non_speech(
        self, faster_whisper: types.SimpleNamespace, model_dir: Path
    ) -> None:
        stt = FasterWhisperTranscriber(model_dir, threads=2)
        assert await stt.transcribe(tone(0.5)) == "Why do geese honk?"
        assert await stt.transcribe(tone(0.5)) == "Why do geese honk?"
        assert len(faster_whisper.loads) == 1  # loaded once, then reused
        path, options = faster_whisper.loads[0]
        assert path == str(model_dir)
        assert options["local_files_only"] is True
        assert options["compute_type"] == "int8"
        audio, decode = faster_whisper.calls[0]
        assert str(audio.dtype) == "float32"
        assert decode["language"] == "en"
        assert decode["beam_size"] == 1
        stt.close()

    async def test_missing_model_is_a_configuration_problem(
        self, faster_whisper: types.SimpleNamespace, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigError, match="local speech model not found"):
            await FasterWhisperTranscriber(tmp_path).transcribe(tone(0.5))

    async def test_needs_16_khz(
        self, faster_whisper: types.SimpleNamespace, model_dir: Path
    ) -> None:
        with pytest.raises(ConfigError, match="16000 Hz"):
            await FasterWhisperTranscriber(model_dir).transcribe(Pcm.silence(1.0, 22_050))

    async def test_engine_failures_are_service_errors(
        self, faster_whisper: types.SimpleNamespace, model_dir: Path
    ) -> None:
        faster_whisper.fail = True
        with pytest.raises(ServiceError, match="exploded"):
            await FasterWhisperTranscriber(model_dir).transcribe(tone(0.5))


class FakeLocalStt:
    def __init__(self) -> None:
        self.preloaded = False

    async def preload(self) -> None:
        self.preloaded = True

    def close(self) -> None:
        pass


async def test_preload_loads_the_model_and_caches_the_system_prompt() -> None:
    seen: list[dict[str, Any]] = []

    def ollama(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        seen.append(json.loads(request.read()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi"}}]})

    api = ApiClient("http://127.0.0.1:11434/v1", "ollama", transport=httpx.MockTransport(ollama))
    stt = FakeLocalStt()
    local = Local(api, stt, StreamingChat(api, ChatModelSpec("gemma3:1b")), 2)  # type: ignore[arg-type]
    await local.preload("You are Lucy.")
    await local.aclose()
    assert stt.preloaded
    assert seen[0]["model"] == "gemma3:1b"
    assert seen[0]["max_tokens"] == 1
    assert seen[0]["messages"][0] == {"role": "system", "content": "You are Lucy."}
    assert local.brain().history_exchanges == 2


class TestLocalConfig:
    def test_disabled_by_default(self) -> None:
        assert build_local(Config()) is None

    def test_enabled_from_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text(
            '[local]\nenabled = true\nmodel = "gemma3:270m"\n[local.params]\nmax_tokens = 60\n'
        )
        config = load_config(path)
        assert config.local == LocalSettings(
            enabled=True, model="gemma3:270m", params={"max_tokens": 60}
        )

    def test_needs_16_khz_microphone(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text("[local]\nenabled = true\n[audio]\ninput_sample_rate = 48000\n")
        with pytest.raises(ConfigError, match="16000"):
            load_config(path)


STT_MODEL = os.environ.get("LUCY_TEST_STT_MODEL")
VOICE = os.environ.get("LUCY_TEST_VOICE")


@pytest.mark.skipif(not (STT_MODEL and VOICE), reason="set LUCY_TEST_STT_MODEL and LUCY_TEST_VOICE")
async def test_real_model_understands_lucys_own_voice() -> None:
    from lucy.adapters.piper_tts import PiperSpeech
    from lucy.core.speech import render
    from lucy.tools.bench import resample

    assert STT_MODEL is not None
    assert VOICE is not None
    tts = PiperSpeech(Path(VOICE), threads=2)
    stt = FasterWhisperTranscriber(Path(STT_MODEL), threads=2)
    try:
        speech = resample(await render(tts, "Why do geese fly south in the winter?"), 16_000)
        text = await stt.transcribe(speech)
    finally:
        tts.close()
        stt.close()
    assert "geese" in text.lower()
    assert "winter" in text.lower()
