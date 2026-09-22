"""Minimal Chrome DevTools Protocol (CDP) client.

Provider-neutral: no dependency on any model provider, so anything in the
tree that needs to drive a Chromium browser from Python can use it.  The
``chrome_ai`` provider (the built-in AI Prompt API) is its first consumer,
not its owner -- it lived under that provider until the WebMCP assessment
(``docs/design/webmcp.md``) found the layering backwards: reaching a
browser required importing from a model provider.

No new dependencies: the WebSocket transport reuses ``websockets`` (already
a core jaato dependency for the WS server) via its synchronous client, and
browser launch/discovery is stdlib (``subprocess`` + ``urllib``).

Deliberately small -- this is not a general CDP library, it is a transport:

- :func:`launch_browser`: spawn Chrome with ``--remote-debugging-port=0``
  and read the browser WebSocket URL from the profile's
  ``DevToolsActivePort`` file (the standard race-free discovery pattern).
- :func:`discover_ws_url`: resolve an ``http(s)://host:port`` DevTools
  endpoint to its browser WebSocket URL via ``GET /json/version``.
- :class:`CDPConnection`: a thread-safe request/response + event pump.
  ``send()`` blocks the calling thread until the matching response id
  arrives; a daemon reader thread dispatches protocol EVENTS (e.g.
  ``Runtime.bindingCalled``) to registered listeners.  Listeners run on
  the reader thread and must be fast (the bridge just routes payloads
  onto per-run queues).
"""

import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class CDPError(Exception):
    """Base class for every failure raised by this module."""


class CDPConnectionError(CDPError):
    """The browser process or its DevTools (CDP) connection failed.

    Covers launch failures, a DevTools WebSocket that drops mid-use, and
    per-request timeouts.  Consumers generally treat this as *transient
    infra* -- the browser can be relaunched -- so a provider mapping this
    onto its own hierarchy should classify it that way.  ``chrome_ai`` does
    exactly that: ``CDPConnectionError`` subclasses this, so
    ``except CDPConnectionError`` catches both.
    """

#: Generous frame cap — Prompt API payloads (history + chunks) are small,
#: but screenshots/tracing-scale CDP frames exist; err on the large side.
_MAX_FRAME = 64 * 1024 * 1024


class _Pending:
    """One in-flight CDP request awaiting its response frame.

    ``event`` is set by the reader thread once ``response`` (the full
    frame dict) is populated; ``send()`` waits on it.
    """

    __slots__ = ("event", "response")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: Optional[Dict[str, Any]] = None


class CDPConnection:
    """Thread-safe CDP request/response client over one WebSocket.

    Lifecycle: constructed connected (raises :class:`CDPConnectionError`
    on failure) → ``send()``/``add_event_listener()`` while ``is_open`` →
    ``close()``.  When the socket drops, all in-flight ``send()`` calls are
    woken with an error and ``is_open`` flips False; the owner (the bridge)
    decides whether to reconnect.
    """

    def __init__(self, ws_url: str, *, open_timeout: float = 30.0,
                 default_timeout: float = 30.0) -> None:
        try:
            from websockets.sync.client import connect as ws_connect
        except ImportError as exc:  # pragma: no cover - websockets is a core dep
            raise CDPConnectionError(
                "the 'websockets' package is required for the CDP "
                "provider (it is a core jaato-server dependency; reinstall "
                "jaato-server)"
            ) from exc
        self._default_timeout = default_timeout
        try:
            self._ws = ws_connect(ws_url, open_timeout=open_timeout,
                                  max_size=_MAX_FRAME)
        except Exception as exc:
            raise CDPConnectionError(
                f"could not open CDP WebSocket {ws_url}: {exc}"
            ) from exc
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pending: Dict[int, _Pending] = {}
        self._next_id = 0
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._closed = False
        self._reader = threading.Thread(
            target=self._read_loop, name="cdp-reader", daemon=True)
        self._reader.start()

    @property
    def is_open(self) -> bool:
        """False once the socket has closed (locally or remotely)."""
        return not self._closed

    def add_event_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for CDP EVENT frames (frames without ``id``).

        Called on the reader thread with the raw frame dict — keep it fast.
        """
        self._listeners.append(callback)

    def send(self, method: str, params: Optional[Dict[str, Any]] = None,
             session_id: Optional[str] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send one CDP command and block until its response arrives.

        Args:
            method: CDP method, e.g. ``"Runtime.evaluate"``.
            params: Method parameters.
            session_id: Target session to scope the command to (flat mode).
            timeout: Seconds to wait for the response (default: the
                connection's ``default_timeout``).

        Returns:
            The ``result`` dict of the response frame.

        Raises:
            CDPConnectionError: on socket loss, timeout, or a CDP
                ``error`` response.
        """
        with self._state_lock:
            if self._closed:
                raise CDPConnectionError("CDP connection is closed")
            self._next_id += 1
            msg_id = self._next_id
            pending = _Pending()
            self._pending[msg_id] = pending
        frame: Dict[str, Any] = {"id": msg_id, "method": method,
                                 "params": params or {}}
        if session_id:
            frame["sessionId"] = session_id
        try:
            with self._send_lock:
                self._ws.send(json.dumps(frame))
        except Exception as exc:
            with self._state_lock:
                self._pending.pop(msg_id, None)
            raise CDPConnectionError(
                f"CDP send failed for {method}: {exc}") from exc
        if not pending.event.wait(timeout or self._default_timeout):
            with self._state_lock:
                self._pending.pop(msg_id, None)
            raise CDPConnectionError(
                f"timed out waiting for CDP response to {method}")
        response = pending.response or {}
        if "error" in response:
            err = response["error"]
            raise CDPConnectionError(
                f"CDP {method} failed: {err.get('message', err)}")
        return response.get("result", {})

    def close(self) -> None:
        """Close the socket and wake every in-flight ``send()``."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------- internal

    def _read_loop(self) -> None:
        """Reader thread: route responses to waiters, events to listeners."""
        while True:
            try:
                raw = self._ws.recv()
            except Exception:
                break
            try:
                frame = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("cdp: unparseable CDP frame dropped")
                continue
            if "id" in frame:
                with self._state_lock:
                    pending = self._pending.pop(frame["id"], None)
                if pending is not None:
                    pending.response = frame
                    pending.event.set()
            else:
                for listener in self._listeners:
                    try:
                        listener(frame)
                    except Exception:
                        logger.exception("cdp: CDP event listener failed")
        # Socket gone: flip closed and wake all waiters with no response
        # (send() surfaces this as a timeout-free connection error).
        with self._state_lock:
            self._closed = True
            pending_now = list(self._pending.values())
            self._pending.clear()
        for pending in pending_now:
            pending.response = {"error": {"message": "CDP connection lost"}}
            pending.event.set()


def launch_browser(
    binary: str,
    user_data_dir: str,
    *,
    headless: bool = True,
    extra_args: Optional[List[str]] = None,
    timeout: float = 30.0,
) -> Tuple[subprocess.Popen, str]:
    """Launch Chrome/Edge with remote debugging and discover its WS URL.

    Uses ``--remote-debugging-port=0`` (ephemeral port) and waits for the
    browser to write ``<user_data_dir>/DevToolsActivePort`` — the
    race-free discovery pattern (no port guessing, no log scraping).
    stderr is redirected to ``<user_data_dir>/chrome-stderr.log`` for
    post-mortem diagnosis without pipe-deadlock risk.

    Returns:
        ``(process, browser_ws_url)``.

    Raises:
        CDPConnectionError: if the process exits or the port file
            does not appear within ``timeout``.
    """
    os.makedirs(user_data_dir, exist_ok=True)
    port_file = os.path.join(user_data_dir, "DevToolsActivePort")
    try:
        os.remove(port_file)
    except FileNotFoundError:
        pass
    args = [
        binary,
        "--remote-debugging-port=0",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        args.append("--headless=new")
    if extra_args:
        args.extend(extra_args)
    stderr_log = os.path.join(user_data_dir, "chrome-stderr.log")
    stderr_fh = open(stderr_log, "wb")
    try:
        process = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=stderr_fh,
            stdin=subprocess.DEVNULL)
    except OSError as exc:
        stderr_fh.close()
        raise CDPConnectionError(
            f"failed to launch browser {binary!r}: {exc}") from exc
    finally:
        # Popen holds its own reference (or launch failed); either way the
        # parent-side handle can close.
        stderr_fh.close()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = _tail(stderr_log)
            raise CDPConnectionError(
                f"browser exited during startup (code {process.returncode})."
                f" stderr tail:\n{tail}")
        try:
            with open(port_file, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            if len(lines) >= 2 and lines[0].strip().isdigit():
                port = int(lines[0].strip())
                path = lines[1].strip()
                return process, f"ws://127.0.0.1:{port}{path}"
        except FileNotFoundError:
            pass
        time.sleep(0.1)
    process.terminate()
    raise CDPConnectionError(
        f"browser did not expose DevTools within {timeout:.0f}s "
        f"(no {port_file}). stderr tail:\n{_tail(stderr_log)}")


def discover_ws_url(cdp_url: str, *, timeout: float = 10.0) -> str:
    """Resolve a DevTools endpoint to the browser WebSocket URL.

    ``ws://``/``wss://`` URLs pass through unchanged; ``http(s)://``
    endpoints are queried via ``GET /json/version`` for
    ``webSocketDebuggerUrl``.
    """
    if cdp_url.startswith(("ws://", "wss://")):
        return cdp_url
    version_url = cdp_url.rstrip("/") + "/json/version"
    try:
        with urllib.request.urlopen(version_url, timeout=timeout) as resp:
            info = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CDPConnectionError(
            f"could not query DevTools endpoint {version_url}: {exc}. "
            "Is the browser running with --remote-debugging-port?") from exc
    ws_url = info.get("webSocketDebuggerUrl")
    if not ws_url:
        raise CDPConnectionError(
            f"{version_url} returned no webSocketDebuggerUrl")
    return ws_url


def _tail(path: str, max_bytes: int = 2048) -> str:
    """Last ``max_bytes`` of a log file, or a placeholder when unreadable."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", "replace").strip()
    except OSError:
        return "(no stderr captured)"
