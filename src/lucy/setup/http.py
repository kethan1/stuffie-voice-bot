"""A small HTTP/1.1 server on the event loop, for setup mode's settings page.

One phone talks to it at a time, over Lucy's own hotspot, so it stays deliberately small:
one request per connection, bodies only with Content-Length, URL-encoded forms, and hard
limits on size and time so a stuck or hostile client cannot hold it up. Everything runs on
the asyncio event loop, like the rest of Lucy (no helper threads).
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
from urllib.parse import parse_qsl, urlsplit

log = logging.getLogger(__name__)

MAX_HEAD_BYTES = 16 * 1024
MAX_BODY_BYTES = 256 * 1024  # the largest form is the whole config file
MAX_FIELDS = 200
READ_TIMEOUT_S = 15.0

# No scripts at all; inline styles only; forms may only post back here.
_SECURITY_HEADERS = (
    (
        "Content-Security-Policy",
        "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
        "form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Cache-Control", "no-store"),
)


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    path: str
    headers: Mapping[str, str]
    """Lower-case names."""
    query: Mapping[str, str] = field(default_factory=dict)
    form: Mapping[str, str] = field(default_factory=dict)

    @property
    def host(self) -> str:
        return self.headers.get("host", "").lower()


@dataclass(frozen=True, slots=True)
class Response:
    status: int
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"
    headers: tuple[tuple[str, str], ...] = ()

    @classmethod
    def html(cls, text: str, status: int = 200) -> "Response":
        return cls(status, text.encode(), "text/html; charset=utf-8")

    @classmethod
    def redirect(cls, location: str, status: int = HTTPStatus.SEE_OTHER) -> "Response":
        return cls(status, headers=(("Location", location),))

    @classmethod
    def empty(cls) -> "Response":
        """204: a form posted here leaves the page as it is (the "Try it" buttons)."""
        return cls(HTTPStatus.NO_CONTENT)

    def encode(self, *, head_only: bool = False) -> bytes:
        reason = HTTPStatus(self.status).phrase
        lines = [f"HTTP/1.1 {self.status} {reason}"]
        headers = [*self.headers, *_SECURITY_HEADERS, ("Connection", "close")]
        if self.status != HTTPStatus.NO_CONTENT:
            headers += [
                ("Content-Type", self.content_type),
                ("Content-Length", str(len(self.body))),
            ]
        lines += [f"{name}: {value}" for name, value in headers]
        head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
        return head if head_only or self.status == HTTPStatus.NO_CONTENT else head + self.body


Handler = Callable[[Request], Awaitable[Response]]


class BadRequest(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


async def start_server(handler: Handler, host: str, port: int) -> asyncio.Server:
    return await asyncio.start_server(
        partial(_serve, handler), host, port, limit=MAX_HEAD_BYTES, reuse_address=True
    )


async def _serve(
    handler: Handler, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        try:
            async with asyncio.timeout(READ_TIMEOUT_S):
                request = await read_request(reader)
        except (TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            return  # the client went away or stalled: nothing to answer
        except BadRequest as error:
            writer.write(Response(error.status, str(error).encode()).encode())
            return
        try:
            response = await handler(request)
        except Exception:
            log.exception("setup page: %s %s failed", request.method, request.path)
            response = Response(HTTPStatus.INTERNAL_SERVER_ERROR, b"Something went wrong.")
        writer.write(response.encode(head_only=request.method == "HEAD"))
        await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


async def read_request(reader: asyncio.StreamReader) -> Request:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError as error:
        raise BadRequest(HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE, "headers too large") from error
    request_line, *header_lines = head.decode("latin-1").split("\r\n")
    try:
        method, target, _version = request_line.split(" ")
    except ValueError as error:
        raise BadRequest(HTTPStatus.BAD_REQUEST, "malformed request line") from error
    headers: dict[str, str] = {}
    for line in header_lines:
        if not line:
            continue
        name, colon, value = line.partition(":")
        if not colon or not name.strip():
            raise BadRequest(HTTPStatus.BAD_REQUEST, "malformed header")
        headers[name.strip().lower()] = value.strip()
    if "transfer-encoding" in headers:
        raise BadRequest(HTTPStatus.NOT_IMPLEMENTED, "send a Content-Length instead")
    try:
        length = int(headers.get("content-length") or 0)
    except ValueError as error:
        raise BadRequest(HTTPStatus.BAD_REQUEST, "bad Content-Length") from error
    if not 0 <= length <= MAX_BODY_BYTES:
        raise BadRequest(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "too much to send")
    body = await reader.readexactly(length)
    try:
        url = urlsplit(target)
        query = dict(parse_qsl(url.query, max_num_fields=MAX_FIELDS))
        form: dict[str, str] = {}
        if headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
            text = body.decode("utf-8")
            form = dict(parse_qsl(text, keep_blank_values=True, max_num_fields=MAX_FIELDS))
    except ValueError as error:  # includes UnicodeDecodeError and too many fields
        raise BadRequest(HTTPStatus.BAD_REQUEST, "malformed request") from error
    return Request(
        method=method.upper(), path=url.path or "/", headers=headers, query=query, form=form
    )
