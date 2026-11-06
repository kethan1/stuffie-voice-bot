import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import numpy as np
import pytest

from lucy.adapters.openai_api import ApiClient
from lucy.adapters.speechify_tts import SpeechifySpeech
from lucy.config import Config, ConfigError, load_config
from lucy.core.audio import Pcm
from lucy.core.errors import ContentRefused, RateLimited, ServiceError, ServiceUnreachable
from lucy.core.honk import HonkingVoice
from lucy.core.speech import FallbackVoice, render
from lucy.voices import build_voice
from tests.fakes import RATE, FakeTTS

Handler = Callable[[httpx.Request], httpx.Response]
PCM_RATE = 8_000  # the lowest rate Speechify streams


def client(handler: Handler) -> ApiClient:
    return ApiClient("https://api.test/v1", "sk-test", transport=httpx.MockTransport(handler))


async def chunked(*pieces: bytes) -> AsyncIterator[bytes]:
    for piece in pieces:
        yield piece


def speechify(handler: Handler, **options: float) -> SpeechifySpeech:
    # Parts of 10 samples (20 bytes), so a few bytes exercise the splitting.
    return SpeechifySpeech(
        client(handler), voice_id="harper_32", sample_rate=PCM_RATE, part_s=10 / PCM_RATE, **options
    )


async def parts_of(tts: SpeechifySpeech | FallbackVoice, text: str = "Honk!") -> list[Pcm]:
    return [part async for part in tts.stream(text)]


class TestSpeechify:
    async def test_streams_raw_pcm_in_whole_samples(self) -> None:
        seen: list[httpx.Request] = []
        audio = bytes(range(64))  # 32 samples

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            # Split mid-sample on purpose: a sample must never be cut in two.
            return httpx.Response(200, content=chunked(audio[:7], audio[7:21], audio[21:]))

        parts = await parts_of(speechify(handler), "Geese fly in a V.")
        assert b"".join(part.data for part in parts) == audio
        assert all(part.sample_rate == PCM_RATE for part in parts)
        # A part as soon as 20 bytes have arrived; an odd byte waits for its pair.
        assert [len(part.data) for part in parts] == [20, 44]
        (request,) = seen
        assert str(request.url) == "https://api.test/v1/audio/stream"
        assert request.headers["authorization"] == "Bearer sk-test"
        assert json.loads(request.content) == {
            "input": "Geese fly in a V.",
            "voice_id": "harper_32",
            "model": "simba-3.2",
            "output_format": "pcm_8000",
        }

    async def test_numbers_are_read_as_words(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, content=b"\x00\x00")

        await parts_of(speechify(handler), "Seven times eight is 56.")
        assert bodies[0]["options"] == {"text_normalization": True}

    async def test_volume(self) -> None:
        loud = Pcm.from_float(np.full(40, 0.5), PCM_RATE).data
        tts = speechify(lambda _: httpx.Response(200, content=loud), volume=0.5)
        (part,) = await parts_of(tts)
        assert abs(int(part.samples()[0]) - 8192) <= 1

    async def test_content_refusal_is_recognised(self) -> None:
        body = {"error": {"code": "content_policy_violation", "message": "refused (violence)"}}
        tts = speechify(lambda _: httpx.Response(400, json=body))
        with pytest.raises(ContentRefused, match="refused"):
            await parts_of(tts)

    async def test_out_of_free_characters_is_an_ordinary_failure(self) -> None:
        body = {"error": {"code": "payment_required", "message": "monthly balance used up"}}
        tts = speechify(lambda _: httpx.Response(402, json=body))
        with pytest.raises(ServiceError, match="balance") as caught:
            await parts_of(tts)
        assert not isinstance(caught.value, ContentRefused)

    async def test_a_short_rate_limit_is_waited_out_once(self) -> None:
        answers = [
            httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "1/s"}}),
            httpx.Response(200, content=b"\x01\x00"),
        ]
        tts = speechify(lambda _: answers.pop(0))
        assert [part.data for part in await parts_of(tts)] == [b"\x01\x00"]

    async def test_a_long_rate_limit_is_reported(self) -> None:
        tts = speechify(
            lambda _: httpx.Response(429, headers={"retry-after": "30"}, json={"error": {}})
        )
        with pytest.raises(RateLimited):
            await parts_of(tts)

    async def test_a_stream_that_breaks_midway_is_a_service_error(self) -> None:
        async def broken() -> AsyncIterator[bytes]:
            yield b"\x01\x00" * 20
            raise httpx.RemoteProtocolError("peer closed connection")

        tts = speechify(lambda _: httpx.Response(200, content=broken()))
        received: list[Pcm] = []

        async def listen() -> None:
            async for part in tts.stream("Honk!"):
                received.append(part)

        with pytest.raises(ServiceError):
            await listen()
        assert received  # the first part was already playable

    def test_only_rates_it_can_stream(self) -> None:
        with pytest.raises(ConfigError, match="11025"):
            SpeechifySpeech(client(lambda _: httpx.Response(200)), voice_id="v", sample_rate=11025)

    async def test_lists_voices_best_first_across_pages(self) -> None:
        pages = {
            None: {
                "voices": [{"id": "b"}, {"id": "a", "staff_pick_rank": 2}],
                "has_more": True,
                "next_cursor": "page 2",
            },
            "page 2": {"voices": [{"id": "c", "staff_pick_rank": 1}], "has_more": False},
        }
        queries: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            queries.append(dict(request.url.params))
            return httpx.Response(200, json=pages[request.url.params.get("cursor")])

        voices = await speechify(handler).voices()
        assert [voice["id"] for voice in voices] == ["c", "a", "b"]
        assert queries[0] == {
            "type": "shared",
            "locale": "en",
            "model": "simba-3.2",
            "limit": "200",
        }
        assert queries[1]["cursor"] == "page 2"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Rig:
    def __init__(self, primary: FakeTTS | None = None, *, available: bool = True) -> None:
        self.primary = primary or FakeTTS(sample=b"\x01\x00")
        self.backup = FakeTTS(sample=b"\x02\x00")
        self.available = available
        self.clock = Clock()
        self.recoveries = 0
        self.voice = FallbackVoice(
            self.primary,
            self.backup,
            available=lambda: self.available,
            rest_s=60.0,
            clock=self.clock,
        )
        self.voice.on_recovery(self.recovered)

    def recovered(self) -> None:
        self.recoveries += 1

    async def say(self, text: str = "Honk!") -> bytes:
        return (await render(self.voice, text)).data


class TestFallbackVoice:
    async def test_speaks_with_the_cloud_voice(self) -> None:
        rig = Rig(FakeTTS(parts=3, sample=b"\x01\x00"))
        assert await rig.say("Hi") == b"\x01\x00" * 6
        assert rig.backup.texts == []

    async def test_a_failure_before_any_audio_moves_the_sentence_to_the_backup(self) -> None:
        rig = Rig(FakeTTS(ServiceError("503")))
        assert await rig.say("Hi") == b"\x02\x00" * 2
        assert rig.backup.texts == ["Hi"]

    async def test_after_a_failure_the_cloud_voice_rests_then_recovers(self) -> None:
        rig = Rig(FakeTTS(ServiceError("timed out")))
        await rig.say()
        rig.primary.error = None
        await rig.say()
        assert len(rig.primary.texts) == 1  # resting: not even tried
        rig.clock.now = 61.0
        assert await rig.say() == b"\x01\x00" * 5
        assert rig.recoveries == 1
        await rig.say()
        assert rig.recoveries == 1  # only on the way back

    async def test_rests_as_long_as_the_service_asks(self) -> None:
        rig = Rig(FakeTTS(RateLimited("slow down", retry_after_s=120.0)))
        await rig.say()
        rig.primary.error = None
        rig.clock.now = 90.0
        await rig.say()
        assert len(rig.primary.texts) == 1

    async def test_a_known_outage_skips_the_cloud_until_it_is_back(self) -> None:
        rig = Rig(FakeTTS(ServiceUnreachable("no Wi-Fi")))
        await rig.say()
        rig.available = False
        await rig.say()
        assert len(rig.primary.texts) == 1
        rig.available = True
        rig.primary.error = None
        assert await rig.say() == b"\x01\x00" * 5  # no rest: back as soon as reachable
        assert rig.recoveries == 1

    async def test_never_finishes_a_started_sentence_in_another_voice(self) -> None:
        rig = Rig(FakeTTS(ServiceError("dropped"), parts=3, error_after=1))
        with pytest.raises(ServiceError):
            await rig.say()
        assert rig.backup.texts == []

    async def test_a_refused_sentence_is_never_said_by_the_backup(self) -> None:
        rig = Rig(FakeTTS(ContentRefused("refused")))
        with pytest.raises(ContentRefused):
            await rig.say()
        assert rig.backup.texts == []

    def test_both_voices_share_a_sample_rate(self) -> None:
        other = FakeTTS()
        other.sample_rate = RATE * 2
        with pytest.raises(ValueError, match="sample rate"):
            FallbackVoice(other, FakeTTS())


def piper_at(rate: int = 22_050) -> FakeTTS:
    piper = FakeTTS()
    piper.sample_rate = rate
    return piper


class TestBuildVoice:
    @pytest.fixture(autouse=True)
    def no_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)  # no .env
        monkeypatch.delenv("SPEECHIFY_API_KEY", raising=False)
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)

    @staticmethod
    def config(tmp_path: Path, toml: str = "") -> Config:
        path = tmp_path / "c.toml"
        path.write_text("[tts]\nhonks = false\n" + toml)
        return load_config(path)

    async def test_without_a_key_the_on_device_voice_speaks(self, tmp_path: Path) -> None:
        on_device = piper_at()
        voice = build_voice(self.config(tmp_path), on_device)  # type: ignore[arg-type]
        assert voice.tts is on_device
        assert voice.warmers == ()

    async def test_honks_whichever_voice_speaks(self) -> None:
        voice = build_voice(load_config(None), piper_at())  # type: ignore[arg-type]
        assert isinstance(voice.tts, HonkingVoice)

    async def test_with_a_key_the_cloud_voice_speaks_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPEECHIFY_API_KEY", "sk-test")
        voice = build_voice(self.config(tmp_path), piper_at())  # type: ignore[arg-type]
        try:
            assert isinstance(voice.tts, FallbackVoice)
            assert voice.cloud is not None
            assert voice.cloud.sample_rate == 22_050  # the speaker plays either voice
            assert voice.warmers == (voice.api,)
        finally:
            assert voice.api is not None
            await voice.api.aclose()

    async def test_can_be_turned_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SPEECHIFY_API_KEY", "sk-test")
        on_device = piper_at()
        config = self.config(tmp_path, "[tts.cloud]\nenabled = false\n")
        assert build_voice(config, on_device).tts is on_device  # type: ignore[arg-type]

    def test_cloud_settings_are_validated(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        path.write_text("[tts.cloud]\nvolume = 3.0\n")
        with pytest.raises(ConfigError, match=r"tts\.cloud\.volume"):
            load_config(path)
        path.write_text("[tts.cloud]\nvoice = 'harper_32'\n")
        with pytest.raises(ConfigError, match="unknown setting"):
            load_config(path)
