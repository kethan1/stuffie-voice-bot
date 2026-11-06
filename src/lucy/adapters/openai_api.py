"""A lean client for OpenAI-compatible REST APIs (GroqCloud by default).

Groq, OpenAI, and a local Ollama server all speak this protocol, so swapping
providers is a configuration change (``api.base_url`` and model names). The
same client also carries Lucy's cloud voice (Speechify), which is plain JSON
over HTTPS with a Bearer key too.

Latency choices:

* One shared HTTP/2 connection carries every request (Groq negotiates h2), so
  speech-to-text and the chat stream never pay for a second TLS handshake.
* :meth:`ApiClient.prewarm` is called the moment the button goes *down*. DNS,
  TCP, and TLS (often 150-400 ms on Wi-Fi) are done while the child is still
  talking instead of after they let go.
* Keep-alive is long, so a conversation stays on one warm connection.

Reachability: once a connection attempt fails, the client reports itself as
unreachable (:meth:`ApiClient.reachable`) until some request gets *any* HTTP
response again. Turns therefore go straight to Lucy's local brain instead of
waiting on a connect timeout each time. While unreachable, every button press
sends a probe, so recovery is noticed as soon as the network returns.

Every httpx failure is translated into a :mod:`lucy.core.errors` type here,
so the rest of the program never sees a transport exception.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from lucy import __version__
from lucy.core.errors import (
    AuthenticationError,
    ContentRefused,
    RateLimited,
    ServiceError,
    ServiceUnreachable,
)

log = logging.getLogger(__name__)

JsonObject = dict[str, Any]


class ApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        connect_timeout_s: float = 3.0,
        keepalive_s: float = 120.0,
        prewarm_after_idle_s: float = 20.0,
        warm_path: str = "models",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._prewarm_after_idle_s = prewarm_after_idle_s
        self._warm_path = warm_path  # a cheap GET that any valid key may make
        self._last_used = -float("inf")
        self._reachable = True
        self._warming: asyncio.Task[None] | None = None
        self._connect_timeout_s = connect_timeout_s
        if transport is None:
            transport = httpx.AsyncHTTPTransport(
                http2=True,
                retries=1,  # one quick retry, for connection failures only
                limits=httpx.Limits(
                    max_connections=4, max_keepalive_connections=2, keepalive_expiry=keepalive_s
                ),
            )
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"lucy-goose/{__version__}",
            },
            timeout=httpx.Timeout(15.0, connect=connect_timeout_s),
        )

    async def aclose(self) -> None:
        if self._warming is not None:
            self._warming.cancel()
        await self._http.aclose()

    def reachable(self) -> bool:
        """False from a failed connection attempt until the service answers again."""
        return self._reachable

    def prewarm(self) -> None:
        """Open (or refresh) the connection in the background if it may have gone cold.

        While the service is unreachable this doubles as a recovery probe.
        """
        if self._warming is not None and not self._warming.done():
            return
        idle = time.monotonic() - self._last_used
        if self._reachable and idle < self._prewarm_after_idle_s:
            return
        self._warming = asyncio.create_task(self._warm(), name="api-prewarm")

    async def _warm(self) -> None:
        started = time.monotonic()
        try:
            async with self._request("GET", self._warm_path, timeout_s=5.0) as response:
                await response.aread()
        except Exception as error:  # a failed warm-up is harmless; the real request will report
            log.debug("connection prewarm failed: %s", error)
        else:
            log.debug("connection warmed in %.0f ms", (time.monotonic() - started) * 1000)

    async def get_json(self, path: str, *, timeout_s: float = 10.0) -> JsonObject:
        async with self._request("GET", path, timeout_s=timeout_s) as response:
            return _json_body(await response.aread())

    async def post_json(
        self, path: str, payload: Mapping[str, Any], *, timeout_s: float
    ) -> JsonObject:
        async with self._request("POST", path, json=payload, timeout_s=timeout_s) as response:
            return _json_body(await response.aread())

    async def post_form(
        self,
        path: str,
        *,
        data: Mapping[str, str],
        files: Mapping[str, tuple[str, bytes, str]],
        timeout_s: float,
    ) -> JsonObject:
        async with self._request(
            "POST", path, data=data, files=files, timeout_s=timeout_s
        ) as response:
            return _json_body(await response.aread())

    @asynccontextmanager
    async def stream_events(
        self, path: str, payload: Mapping[str, Any], *, timeout_s: float
    ) -> AsyncIterator[AsyncIterator[JsonObject]]:
        """POST and yield each server-sent ``data:`` event as parsed JSON until ``[DONE]``."""
        async with self._request("POST", path, json=payload, timeout_s=timeout_s) as response:
            yield self._events(response)

    @asynccontextmanager
    async def stream_bytes(
        self, path: str, payload: Mapping[str, Any], *, timeout_s: float
    ) -> AsyncIterator[AsyncIterator[bytes]]:
        """POST JSON and yield the response body as it arrives (``timeout_s`` per read)."""
        async with self._request("POST", path, json=payload, timeout_s=timeout_s) as response:
            yield self._bytes(response)

    async def _bytes(self, response: httpx.Response) -> AsyncIterator[bytes]:
        try:
            async for data in response.aiter_bytes():
                self._last_used = time.monotonic()
                yield data
        except httpx.HTTPError as error:
            raise _translate(error) from error

    async def _events(self, response: httpx.Response) -> AsyncIterator[JsonObject]:
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                event = _json_body(data.encode())
                if "error" in event:
                    raise ServiceError(f"stream error: {_error_message(event)}")
                self._last_used = time.monotonic()
                yield event
        except httpx.HTTPError as error:
            raise _translate(error) from error

    @asynccontextmanager
    async def _request(
        self, method: str, path: str, *, timeout_s: float, **kwargs: Any
    ) -> AsyncIterator[httpx.Response]:
        timeout = httpx.Timeout(timeout_s, connect=self._connect_timeout_s)
        try:
            async with self._http.stream(method, path, timeout=timeout, **kwargs) as response:
                self._last_used = time.monotonic()
                self._mark_reachable(True)
                if response.status_code >= 400:
                    await response.aread()
                    raise _status_error(response)
                yield response
        except httpx.HTTPError as error:
            translated = _translate(error)
            if isinstance(translated, ServiceUnreachable):
                self._mark_reachable(False)
            raise translated from error

    def _mark_reachable(self, reachable: bool) -> None:
        if reachable != self._reachable:
            self._reachable = reachable
            if reachable:
                log.info("%s is reachable again", self._http.base_url.host)
            else:
                log.warning("%s is unreachable", self._http.base_url.host)


def _translate(error: httpx.HTTPError) -> ServiceError:
    if isinstance(error, httpx.ConnectError | httpx.ConnectTimeout):
        return ServiceUnreachable(f"cannot reach the AI service: {error}")
    if isinstance(error, httpx.TimeoutException):
        return ServiceError(f"the AI service timed out: {type(error).__name__}")
    return ServiceError(f"connection to the AI service failed: {error!r}")


def _status_error(response: httpx.Response) -> ServiceError:
    status = response.status_code
    body = _safe_json(response.content)
    message = f"HTTP {status}: {_error_message(body)}"
    if status in (401, 403):
        return AuthenticationError(message)
    if status == 429:
        return RateLimited(message, _retry_after(response.headers.get("retry-after")))
    error = body.get("error")
    if isinstance(error, Mapping) and error.get("code") == "content_policy_violation":
        return ContentRefused(message)
    return ServiceError(message)


def _retry_after(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _json_body(raw: bytes) -> JsonObject:
    try:
        body = json.loads(raw)
    except ValueError as error:
        raise ServiceError(f"malformed JSON from the AI service: {raw[:120]!r}") from error
    if not isinstance(body, dict):
        raise ServiceError("unexpected JSON shape from the AI service")
    return body


def _safe_json(raw: bytes) -> JsonObject:
    try:
        return _json_body(raw)
    except ServiceError:
        return {"error": {"message": raw[:200].decode(errors="replace")}}


def _error_message(body: Mapping[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, Mapping):
        return str(error.get("message") or error)
    return str(error or body)[:200]
