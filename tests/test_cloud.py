import json
from collections.abc import Callable

import httpx
import pytest

from lucy.adapters.cloud import ChatModelSpec, FallbackChat, StreamingChat, WhisperTranscriber
from lucy.adapters.openai_api import ApiClient
from lucy.core.errors import AuthenticationError, RateLimited, ServiceError, ServiceUnreachable
from lucy.core.types import Message, Role
from tests.fakes import FakeChat, tone

Handler = Callable[[httpx.Request], httpx.Response]
QUESTION = [Message(Role.USER, "why?")]


def client(handler: Handler) -> ApiClient:
    return ApiClient(
        "https://api.test/openai/v1", "sk-test", transport=httpx.MockTransport(handler)
    )


def sse(*events: object) -> bytes:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def delta(content: str | None = None, reasoning: str | None = None) -> dict[str, object]:
    d: dict[str, object] = {}
    if content is not None:
        d["content"] = content
    if reasoning is not None:
        d["reasoning"] = reasoning
    return {"choices": [{"index": 0, "delta": d}]}


async def collect(chat: StreamingChat | FallbackChat) -> str:
    return "".join([piece async for piece in chat.stream(QUESTION)])


class TestWhisper:
    async def test_sends_wav_and_drops_non_speech_segments(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = request.read()
            return httpx.Response(
                200,
                json={
                    "text": "Why is the sky blue? Thank you.",
                    "segments": [
                        {"text": " Why is the sky blue?", "no_speech_prob": 0.01},
                        {"text": " Thank you.", "no_speech_prob": 0.93},
                    ],
                },
            )

        stt = WhisperTranscriber(client(handler), model="whisper-large-v3-turbo", language="en")
        assert await stt.transcribe(tone(0.5)) == "Why is the sky blue?"
        assert seen["url"] == "https://api.test/openai/v1/audio/transcriptions"
        assert seen["auth"] == "Bearer sk-test"
        body = seen["body"]
        assert isinstance(body, bytes)
        for field in (b"whisper-large-v3-turbo", b"verbose_json", b'name="language"', b"RIFF"):
            assert field in body

    async def test_text_without_segments(self) -> None:
        body = {"text": " Why do geese fly in a V? ", "segments": []}
        stt = WhisperTranscriber(
            client(lambda _: httpx.Response(200, json=body)), model="m", language="en"
        )
        assert await stt.transcribe(tone(0.2)) == "Why do geese fly in a V?"

    async def test_plain_json_response(self) -> None:
        stt = WhisperTranscriber(
            client(lambda _: httpx.Response(200, json={"text": " hello "})),
            model="m",
            language="en",
        )
        assert await stt.transcribe(tone(0.2)) == "hello"


class TestStreamingChat:
    async def test_streams_content_and_ignores_reasoning(self) -> None:
        payload: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            payload.update(json.loads(request.read()))
            body = sse(delta(reasoning="thinking…"), delta("Honk! "), {"choices": []}, delta("Hi."))
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

        spec = ChatModelSpec("openai/gpt-oss-20b", {"reasoning_effort": "low"})
        assert await collect(StreamingChat(client(handler), spec)) == "Honk! Hi."
        assert payload["stream"] is True
        assert payload["reasoning_effort"] == "low"
        assert payload["messages"] == [{"role": "user", "content": "why?"}]

    async def test_error_event_mid_stream(self) -> None:
        body = sse(delta("Honk"), {"error": {"message": "overloaded"}})
        chat = StreamingChat(
            client(lambda _: httpx.Response(200, content=body)), ChatModelSpec("m")
        )
        with pytest.raises(ServiceError, match="overloaded"):
            await collect(chat)

    @pytest.mark.parametrize(
        ("status", "error"),
        [(401, AuthenticationError), (429, RateLimited), (500, ServiceError), (400, ServiceError)],
    )
    async def test_http_errors_are_translated(self, status: int, error: type[Exception]) -> None:
        response = httpx.Response(
            status, json={"error": {"message": "nope"}}, headers={"retry-after": "7"}
        )
        chat = StreamingChat(client(lambda _: response), ChatModelSpec("m"))
        with pytest.raises(error, match="nope") as caught:
            await collect(chat)
        if isinstance(caught.value, RateLimited):
            assert caught.value.retry_after_s == 7.0

    async def test_connection_failure_means_offline(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("name resolution failed", request=request)

        with pytest.raises(ServiceUnreachable):
            await collect(StreamingChat(client(handler), ChatModelSpec("m")))


class TestFallbackChat:
    async def test_falls_back_when_rate_limited(self) -> None:
        chat = FallbackChat(FakeChat(error=RateLimited("busy")), FakeChat(deltas=["backup"]))
        assert await collect(chat) == "backup"  # type: ignore[arg-type]

    async def test_never_mixes_models_mid_reply(self) -> None:
        primary = FakeChat(deltas=["Honk", "!"], error=ServiceError("dropped"), error_after=1)
        chat = FallbackChat(primary, FakeChat(deltas=["backup"]))
        with pytest.raises(ServiceError):
            await collect(chat)  # type: ignore[arg-type]

    @pytest.mark.parametrize("error", [ServiceUnreachable("offline"), AuthenticationError("key")])
    async def test_does_not_fall_back_when_it_cannot_help(self, error: ServiceError) -> None:
        fallback = FakeChat(deltas=["backup"])
        with pytest.raises(type(error)):
            await collect(FallbackChat(FakeChat(error=error), fallback))  # type: ignore[arg-type]
        assert fallback.calls == []


class TestPrewarm:
    async def test_warms_once_when_cold(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json={"data": []})

        api = client(handler)
        api.prewarm()
        api.prewarm()  # already warming: no second request
        assert api._warming is not None
        await api._warming
        api.prewarm()  # just used: still warm
        await api.aclose()
        assert paths == ["/openai/v1/models"]

    async def test_warm_up_failures_are_harmless(self) -> None:
        api = client(lambda _: httpx.Response(503))
        api.prewarm()
        assert api._warming is not None
        await api._warming
        await api.aclose()


class TestReachability:
    async def test_down_after_a_failed_connection_until_any_answer(self) -> None:
        online = False

        def handler(request: httpx.Request) -> httpx.Response:
            if not online:
                raise httpx.ConnectError("network is unreachable", request=request)
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        api = client(handler)
        assert api.reachable()
        with pytest.raises(ServiceUnreachable):
            await api.get_json("models")
        assert not api.reachable()

        online = True
        api.prewarm()  # while down, every press probes, even right after a request
        assert api._warming is not None
        await api._warming
        assert api.reachable()  # an error response still proves the service is reachable
        await api.aclose()

    async def test_slow_or_broken_answers_do_not_count_as_unreachable(self) -> None:
        api = client(lambda _: httpx.Response(503, json={"error": {"message": "overloaded"}}))
        with pytest.raises(ServiceError):
            await api.get_json("models")
        assert api.reachable()
        await api.aclose()
