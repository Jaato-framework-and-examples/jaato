"""Remote MCP transports: streamable HTTP and SSE (#1580).

``.mcp.json`` names a server's transport with ``"type"``.  Until #1580 that key
was read by nothing and every server was spawned as a subprocess, so a hosted
server (Context7 at ``https://mcp.context7.com/mcp``) needed a local
``npx mcp-remote`` process in front of it.  This module is the remote half:

* :func:`server_transport` decides the transport of one entry, and REFUSES an
  unknown one rather than falling back to stdio (a typo must not spawn
  ``"unknown"``).
* :func:`remote_endpoint` validates the ``url`` / ``headers`` of an ``http`` /
  ``sse`` entry, before anything is resolved.
* :func:`resolve_headers` expands header values at CONNECT time through the
  same ``expand_variables`` every plugin config uses (``${VAR}``, then a
  whole-value ``pass://`` / ``vault://`` secret URI), and refuses a value that
  did not resolve.  The result lives only inside the HTTP client; the stored
  config keeps the templates, so no repr, log line or trace carries a secret.
* :func:`bind_headers_to_host` is the ``web_fetch`` ``secret_host_bindings``
  rule applied to one server: every configured header is sent ONLY to the
  configured url's host (:func:`host_matches`, the same matcher).  A request
  to any other host (a redirect the HTTP client follows) has them removed.
* :func:`open_remote_streams` opens the transport on whichever mcp SDK
  generation is installed (1.x ``streamablehttp_client(headers=...)``, 2.x
  ``streamable_http_client(http_client=...)``), imported lazily so a stdio-only
  deployment never imports an HTTP client.

A remote server is an outbound HTTPS connection FROM THE RUNNER: an egress
allowlist must include its host.  ``scrub_secret_env`` filters a subprocess's
inherited environment and has nothing to act on here.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse

TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"
TRANSPORT_SSE = "sse"

#: Transports that connect to a URL instead of spawning a subprocess.
REMOTE_TRANSPORTS = frozenset({TRANSPORT_HTTP, TRANSPORT_SSE})

#: Spellings other clients write for the streamable HTTP transport.
TRANSPORT_ALIASES: Dict[str, str] = {"streamable-http": TRANSPORT_HTTP}

#: ``.mcp.json`` ``"type"`` values and what each does.  Rendered by
#: ``jaato-scaffold explain plugin mcp`` so the page cannot list a transport
#: the plugin does not open.
TRANSPORT_DOCS: Tuple[Tuple[str, str], ...] = (
    (TRANSPORT_STDIO, "subprocess over stdin/stdout: command, args, env "
                      "(the default when \"type\" is absent)"),
    (TRANSPORT_HTTP, "streamable HTTP: url, headers (alias "
                     "\"streamable-http\")"),
    (TRANSPORT_SSE, "HTTP + server-sent events (the older remote "
                    "transport): url, headers"),
)

#: The rules a remote entry is held to, rendered beside :data:`TRANSPORT_DOCS`.
REMOTE_RULES: Tuple[str, ...] = (
    "any other \"type\" is refused for that server, never treated as stdio",
    "url must be http:// or https:// with a host, no user:password@ and no "
    "${VAR} (a secret belongs in a header, not a url)",
    "a header value may use ${VAR} and may be a whole pass:// / vault:// "
    "secret URI; resolved at connect time in the runner, never stored "
    "expanded, never logged",
    "configured headers are sent ONLY to the url's host; a request to "
    "another host (a redirect) has them removed",
    "a remote server is an outbound connection from the runner: an egress "
    "allowlist must include its host",
    "scrub_secret_env applies to stdio subprocesses only; a remote server "
    "inherits no environment",
    "a stale Mcp-Session-Id (404 'Session terminated') reconnects once on "
    "the next call; a server that cannot connect leaves the session up "
    "without its tools, reported once at WARNING",
    "a 401 reconnects once with headers re-resolved and retries the call "
    "once, so a rotated token in the session env is picked up; a server that "
    "keeps refusing fails the call",
    "session.reload_env reopens the remote servers whose headers read a "
    "${VAR} the reload changed",
)

# An RFC 7230 header field-name (token).
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
# A value that still names a secret scheme after resolution: the resolver for
# it is missing.  Network schemes are literal values.
_UNRESOLVED_URI_RE = re.compile(r"^([a-z][a-z0-9_+-]*)://\S+$")
_NETWORK_SCHEMES = frozenset({"http", "https", "ws", "wss", "ftp", "ftps"})


class MCPConfigError(ValueError):
    """An ``.mcp.json`` entry the plugin refuses to connect.

    The message names the server and the offending key, never a header value.
    """


def server_transport(name: str, spec: Mapping[str, Any]) -> str:
    """Return the transport of one ``.mcp.json`` entry.

    Absent ``"type"`` keeps the pre-#1580 meaning (stdio), except for an entry
    that names a ``url`` and no ``command``: spawning ``"unknown"`` for it
    could never work, so it is refused with a hint instead.

    Raises:
        MCPConfigError: unknown type, or a url-only entry with no type.
    """
    raw = spec.get("type")
    if raw is None:
        if spec.get("url") and not spec.get("command"):
            raise MCPConfigError(
                f"MCP server '{name}': has a url but no \"type\"; add "
                f"\"type\": \"http\" (or \"sse\")")
        return TRANSPORT_STDIO
    kind = TRANSPORT_ALIASES.get(str(raw).strip().lower(),
                                 str(raw).strip().lower())
    if kind != TRANSPORT_STDIO and kind not in REMOTE_TRANSPORTS:
        supported = ", ".join(k for k, _ in TRANSPORT_DOCS)
        raise MCPConfigError(
            f"MCP server '{name}': unknown \"type\" {raw!r} "
            f"(supported: {supported})")
    return kind


def _check_url(name: str, url: Any) -> str:
    """Validate a remote entry's ``url``; return it unchanged."""
    if not isinstance(url, str) or not url.strip():
        raise MCPConfigError(f"MCP server '{name}': a remote server needs a \"url\"")
    if "${" in url:
        raise MCPConfigError(
            f"MCP server '{name}': \"url\" may not contain ${{VAR}}; put a "
            f"secret in a header")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise MCPConfigError(
            f"MCP server '{name}': \"url\" must be http:// or https:// with "
            f"a host")
    if parsed.username or parsed.password:
        raise MCPConfigError(
            f"MCP server '{name}': \"url\" may not carry user:password@; "
            f"use a header")
    return url


def _check_headers(name: str, headers: Any) -> Dict[str, str]:
    """Validate a remote entry's ``headers`` (names and template types only)."""
    if headers is None:
        return {}
    if not isinstance(headers, dict):
        raise MCPConfigError(f"MCP server '{name}': \"headers\" must be an object")
    out: Dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not _HEADER_NAME_RE.match(key):
            raise MCPConfigError(f"MCP server '{name}': invalid header name {key!r}")
        if not isinstance(value, str):
            raise MCPConfigError(
                f"MCP server '{name}': header {key!r} must be a string")
        out[key] = value
    return out


def remote_endpoint(name: str, spec: Mapping[str, Any]) -> Tuple[str, Dict[str, str]]:
    """Return ``(url, header_templates)`` of a remote entry, validated.

    Nothing is resolved here: the templates are what the stored config keeps.

    Raises:
        MCPConfigError: missing / non-http(s) url, or malformed headers.
    """
    return _check_url(name, spec.get("url")), _check_headers(name, spec.get("headers"))


def _unresolved(value: str) -> bool:
    """True when a resolved header value still carries a placeholder."""
    if "${" in value:
        return True
    match = _UNRESOLVED_URI_RE.match(value.strip())
    return bool(match) and match.group(1) not in _NETWORK_SCHEMES


def resolve_headers(name: str, templates: Mapping[str, str],
                    expand: Optional[Callable[[str], Any]] = None) -> Dict[str, str]:
    """Resolve header templates at connect time.

    ``expand`` defaults to the framework's ``expand_variables`` (the resolver
    every plugin config goes through).  A value that still holds ``${VAR}``
    (the variable is not set) or a secret URI (no resolver for its scheme) is
    REFUSED, naming the header and never the value: sending the literal would
    be a broken credential at best.

    Raises:
        MCPConfigError: an unresolved value, a resolver failure, or a value
            carrying a line break.
    """
    if expand is None:
        from jaato_server.shared.plugins.subagent.config import expand_variables
        expand = expand_variables
    resolved: Dict[str, str] = {}
    for key, template in templates.items():
        try:
            value = expand(template)
        except Exception as exc:  # SecretResolutionError and friends
            raise MCPConfigError(
                f"MCP server '{name}': header {key!r} could not be resolved "
                f"({type(exc).__name__})") from None
        if not isinstance(value, str) or _unresolved(value):
            raise MCPConfigError(
                f"MCP server '{name}': header {key!r} did not resolve (an "
                f"unset ${{VAR}} or a secret URI with no resolver)")
        if "\r" in value or "\n" in value:
            raise MCPConfigError(
                f"MCP server '{name}': header {key!r} resolved to a value "
                f"with a line break")
        resolved[key] = value
    return resolved


def bind_headers_to_host(header_names, url: str):
    """Return an async httpx request hook that keeps headers on one host.

    The ``web_fetch`` ``secret_host_bindings`` rule with the binding implied:
    every configured header may go to the configured url's host and nowhere
    else.  The hook runs for every request the client sends, redirects
    included, and removes the configured headers when the target host does
    not match (``host_matches``, the matcher ``web_fetch`` uses).
    """
    from jaato_server.shared.plugins.web_fetch.security import host_matches
    bound_host = urlparse(url).hostname or ""
    names = tuple(header_names)

    async def _strip_off_host(request) -> None:
        if host_matches(request.url.host, bound_host):
            return
        for header in names:
            if header in request.headers:
                del request.headers[header]

    return _strip_off_host


class SessionWatch:
    """Records what the server said about the session or the credential.

    Two HTTP answers mean a reconnect can fix the call (#1580, #1609):

    * **404** to a request carrying an ``Mcp-Session-Id``: the server forgot
      the session (restart, idle expiry).  What the SDK then raises depends
      on the server's body (``Session terminated`` when the body is empty,
      the server's own JSON-RPC error otherwise), so the status is observed
      here rather than guessed from a message.  Sets :attr:`stale`.
    * **401**: the server refused the credential.  Headers are resolved once,
      when the connection opens, so a rotated token (a refreshed OAuth access
      token written into the session env) reaches the server only through a
      new connection.  Sets :attr:`unauthorized`.

    Both also set :attr:`tripped`, an event the call path races the call
    against: on SSE the refused POST kills the transport and the pending
    call never answers, so waiting for an exception would wait forever.
    """

    def __init__(self) -> None:
        self.stale = False
        self.unauthorized = False
        self.tripped = asyncio.Event()

    async def response_hook(self, response) -> None:
        if response.status_code == 404 and "mcp-session-id" in response.request.headers:
            self.stale = True
            self.tripped.set()
        elif response.status_code == 401:
            self.unauthorized = True
            self.tripped.set()


_HEADER_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def header_env_names(templates: Mapping[str, str]) -> frozenset:
    """The variable names the ``${VAR}`` references in ``templates`` read.

    Used after ``session.reload_env`` to decide which remote connections
    resolved a header from a variable that changed (#1609).  A whole-value
    secret URI (``pass://...``) reads no variable and is not named here.
    """
    names = set()
    for value in templates.values():
        if isinstance(value, str):
            names.update(m.strip() for m in _HEADER_VAR_RE.findall(value))
    return frozenset(names)


def _install_hook(client, kind: str, hook) -> None:
    """Append ``hook`` to an httpx/httpx2 client's ``kind`` event hooks."""
    hooks = dict(client.event_hooks)
    hooks[kind] = [*hooks.get(kind, []), hook]
    client.event_hooks = hooks


def bound_client_factory(sdk_factory, header_names, url: str,
                         watch: Optional[SessionWatch] = None):
    """Wrap the SDK's client factory so every client it builds is host-bound.

    With ``watch``, the client also reports a stale session to it.
    """
    hook = bind_headers_to_host(header_names, url)

    def factory(headers=None, timeout=None, auth=None):
        client = sdk_factory(headers=headers, timeout=timeout, auth=auth)
        _install_hook(client, "request", hook)
        if watch is not None:
            _install_hook(client, "response", watch.response_hook)
        return client

    return factory


@asynccontextmanager
async def _streamable_http(url: str, headers: Dict[str, str], factory) -> AsyncIterator[Tuple[Any, Any]]:
    """Open streamable HTTP on whichever SDK generation is installed."""
    module = importlib.import_module("mcp.client.streamable_http")
    modern = getattr(module, "streamable_http_client", None)
    if modern is not None and "http_client" in inspect.signature(modern).parameters:
        client = factory(headers=headers)
        async with client:
            async with modern(url, http_client=client) as streams:
                yield streams[0], streams[1]
        return
    async with module.streamablehttp_client(
            url, headers=headers, httpx_client_factory=factory) as streams:
        yield streams[0], streams[1]


@asynccontextmanager
async def open_remote_streams(transport: str, url: str,
                              headers: Dict[str, str],
                              watch: Optional[SessionWatch] = None,
                              ) -> AsyncIterator[Tuple[Any, Any]]:
    """Yield ``(read, write)`` streams for a remote MCP server.

    The SDK modules are imported here, so a stdio-only setup never needs them.
    """
    from mcp.shared._httpx_utils import create_mcp_http_client
    factory = bound_client_factory(create_mcp_http_client, headers.keys(), url, watch)
    if transport == TRANSPORT_SSE:
        from mcp.client.sse import sse_client
        async with sse_client(url, headers=headers,
                              httpx_client_factory=factory) as streams:
            yield streams[0], streams[1]
        return
    async with _streamable_http(url, headers, factory) as streams:
        yield streams


def is_stale_session(exc: BaseException) -> bool:
    """True when ``exc``'s message reports a session the server no longer has.

    The fallback beside :class:`SessionWatch`: both SDK generations spell a
    body-less 404 on a request carrying an ``Mcp-Session-Id`` as
    ``Session terminated``.  Exception groups and causes are walked.
    """
    seen = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        error = getattr(cur, "error", None)
        if getattr(error, "message", None) == "Session terminated":
            return True
        if "Session terminated" in str(cur):
            return True
        stack.extend(getattr(cur, "exceptions", ()) or ())
        stack.extend((cur.__cause__, cur.__context__))
    return False


def describe_failure(exc: BaseException) -> str:
    """One readable line for a connect failure (unwraps exception groups)."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    text = str(exc) or type(exc).__name__
    return text if isinstance(exc, MCPConfigError) else f"{type(exc).__name__}: {text}"
