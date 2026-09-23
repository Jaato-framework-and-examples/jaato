"""WebMCP page session: harvest and invoke a page's declared tools over CDP.

WebMCP (https://github.com/webmachinelearning/webmcp) lets a web page declare
tools for an agent instead of making the agent scrape and click.  It is **not**
an MCP transport -- there is no server, no JSON-RPC, no socket.  It is an
in-page JavaScript API, ``document.modelContext``, reachable only by being (or
driving) a browser.  jaato is a Python daemon, so it drives one, over
:mod:`shared.cdp`.

THE SHIPPED SURFACE IS NOT THE EXPLAINER'S.  Every shape below was measured
against Google Chrome for Testing 153.0.8010.36, because the explainer and the
write-ups disagree with what Chrome actually exposes:

===========================  ==========================================
Claim                        Chrome 153
===========================  ==========================================
``navigator.modelContext``   absent -- it is on ``document``
``unregisterTool``           absent -- unregistration is via AbortSignal
``provideContext``           absent
``inputSchema`` is an object it is a **JSON string**
``executeTool(name, args)``  refuses a name: needs a live ``RegisteredTool``
args are an object           refused -- they are a **JSON string**
results are an object        a **JSON string** of ``{content: [...]}``
===========================  ==========================================

Two consequences shape this module:

* **A call must re-harvest.**  ``executeTool`` demands the live
  ``RegisteredTool`` object, and a CDP ``Runtime.evaluate`` returning by value
  cannot carry one across calls.  So :meth:`WebMCPPage.call` does the lookup
  and the invocation in ONE evaluation, finding the tool by name inside the
  page.  That also makes the call self-healing: it sees the toolset as it
  stands at call time, not as it stood at harvest.
* **The browser does not validate arguments.**  Measured: a call omitting a
  ``required`` property resolved successfully with the value ``undefined``
  reaching the page's ``execute()``.  ``inputSchema`` is therefore advisory
  documentation for the model, never an enforced contract, and an argument
  mistake surfaces as whatever the page does with it.
"""

from __future__ import annotations

import json
import logging
import tempfile
from typing import Any, Dict, List, Optional

from jaato_server.shared.cdp import (CDPConnection, CDPConnectionError, discover_ws_url,
                        launch_browser)

logger = logging.getLogger(__name__)

#: Projection of a RegisteredTool into something JSON-serialisable.  The live
#: object carries a ``window`` back-reference, so ``JSON.stringify`` on it
#: throws "Converting circular structure to JSON" -- the fields must be named.
_HARVEST_JS = """
(async () => {
  if (!document.modelContext) return JSON.stringify({supported: false});
  const ts = await document.modelContext.getTools();
  return JSON.stringify({supported: true, tools: ts.map(t => ({
    name: t.name, title: t.title, description: t.description,
    inputSchema: t.inputSchema, origin: t.origin,
  }))});
})()
"""

#: Lookup + invoke in one evaluation -- see the module docstring.
_CALL_JS = """
(async () => {
  if (!document.modelContext) return JSON.stringify({ok: false, error: "unsupported"});
  const ts = await document.modelContext.getTools();
  const tool = ts.find(t => t.name === %(name)s);
  if (!tool) return JSON.stringify({ok: false, error: "unknown_tool",
                                    available: ts.map(t => t.name)});
  try {
    const raw = await document.modelContext.executeTool(tool, %(args)s);
    return JSON.stringify({ok: true, raw: raw, origin: tool.origin});
  } catch (e) {
    return JSON.stringify({ok: false, error: e.name || "Error",
                           message: String(e && e.message), origin: tool.origin});
  }
})()
"""


class WebMCPUnsupportedError(RuntimeError):
    """The attached page exposes no ``document.modelContext``.

    Raised rather than returning an empty tool list, because "this browser is
    too old / the flag is off" and "this page declares no tools" are different
    problems with different fixes, and an empty list conflates them.
    """


class WebMCPPage:
    """One browser page, driven for its WebMCP tools.

    Lifecycle: constructed idle -> :meth:`connect` launches or attaches a
    browser and navigates to ``page_url`` -> :meth:`harvest` and :meth:`call`
    are usable -> :meth:`close` tears down.  ``connect`` is idempotent while
    the connection is live, so callers may treat it as "ensure connected".

    When ``cdp_url`` is set the browser is ATTACHED to and left running on
    close (it is the user's browser, not ours); when it is not, one is launched
    into a throwaway profile and terminated on close.
    """

    def __init__(self, *, page_url: str, cdp_url: Optional[str] = None,
                 binary: Optional[str] = None, user_data_dir: Optional[str] = None,
                 headless: bool = True, extra_args: Optional[List[str]] = None,
                 connect_timeout: float = 30.0, call_timeout: float = 60.0) -> None:
        self._page_url = page_url
        self._cdp_url = cdp_url
        self._binary = binary
        self._user_data_dir = user_data_dir
        self._headless = headless
        self._extra_args = list(extra_args or [])
        self._connect_timeout = connect_timeout
        self._call_timeout = call_timeout
        self._conn: Optional[CDPConnection] = None
        self._session_id: Optional[str] = None
        self._process = None

    @property
    def is_connected(self) -> bool:
        return self._conn is not None and self._conn.is_open

    @property
    def page_url(self) -> str:
        return self._page_url

    def connect(self) -> None:
        """Launch or attach a browser and open ``page_url``.

        Prefers an EXISTING tab already showing ``page_url`` -- driving the
        page the user is looking at is the point of WebMCP, and creating a
        second tab would harvest a fresh copy of the app whose state is not
        the one they can see.
        """
        if self.is_connected:
            return
        if self._cdp_url:
            ws_url = discover_ws_url(self._cdp_url, timeout=self._connect_timeout)
        else:
            self._process, ws_url = launch_browser(
                self._binary or "chrome",
                self._user_data_dir or tempfile.mkdtemp(prefix="jaato-webmcp-"),
                headless=self._headless,
                extra_args=self._extra_args,
                timeout=self._connect_timeout,
            )
        self._conn = CDPConnection(ws_url, open_timeout=self._connect_timeout)
        self._session_id = self._attach_page()

    def _attach_page(self) -> str:
        """Attach to the tab showing ``page_url``, else open one."""
        targets = self._conn.send("Target.getTargets").get("targetInfos", [])
        target_id = next((t.get("targetId") for t in targets
                          if t.get("type") == "page" and t.get("url") == self._page_url),
                         None)
        if target_id is None:
            target_id = self._conn.send(
                "Target.createTarget", {"url": self._page_url})["targetId"]
        return self._conn.send("Target.attachToTarget",
                               {"targetId": target_id, "flatten": True})["sessionId"]

    def _eval(self, expression: str, timeout: Optional[float] = None) -> Any:
        if self._conn is None or self._session_id is None:
            raise CDPConnectionError("WebMCP page is not connected")
        result = self._conn.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            session_id=self._session_id, timeout=timeout)
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            message = ((details.get("exception") or {}).get("description")
                       or details.get("text") or "page evaluation failed")
            raise CDPConnectionError(f"WebMCP evaluation failed: {message}")
        return result.get("result", {}).get("value")

    def harvest(self) -> List[Dict[str, Any]]:
        """Return the page's currently-declared tools.

        Each entry is ``{name, title, description, input_schema, origin}``.
        ``input_schema`` is PARSED from the JSON string Chrome hands back; a
        schema that will not parse is reported as ``None`` rather than raising,
        because one malformed tool must not hide every other tool on the page.

        Re-read on every call: a page adds and removes tools as its state
        changes, and there is no cache here to go stale.
        """
        self.connect()
        payload = json.loads(self._eval(_HARVEST_JS, timeout=self._call_timeout))
        if not payload.get("supported"):
            raise WebMCPUnsupportedError(
                f"{self._page_url} exposes no document.modelContext -- the browser "
                "predates WebMCP or the feature is not enabled")
        return [self._normalise(t) for t in payload.get("tools", [])]

    @staticmethod
    def _normalise(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Project one harvested descriptor, parsing its JSON-string schema."""
        raw_schema = tool.get("inputSchema")
        schema: Optional[Dict[str, Any]]
        try:
            schema = json.loads(raw_schema) if isinstance(raw_schema, str) else raw_schema
        except (TypeError, ValueError):
            schema = None
        return {
            "name": tool.get("name") or "",
            "title": tool.get("title") or "",
            "description": tool.get("description") or "",
            "input_schema": schema,
            "origin": tool.get("origin") or "",
        }

    def call(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke one page tool by name.

        Returns ``{ok, content, origin}`` on success, or ``{ok: False, error,
        ...}`` on failure.  Failures are RETURNED, not raised, because every
        one of them is something the model can act on: an unknown tool means
        re-list (the page's toolset changed under it), and a page-side
        exception means the call was wrong or the app rejected it.  A dead CDP
        connection still raises -- that is infrastructure, not a tool outcome.
        """
        self.connect()
        expression = _CALL_JS % {
            "name": json.dumps(name),
            "args": json.dumps(json.dumps(arguments or {})),
        }
        payload = json.loads(self._eval(expression, timeout=self._call_timeout))
        if not payload.get("ok"):
            return payload
        return {"ok": True, "origin": payload.get("origin", ""),
                "content": _parse_content(payload.get("raw"))}

    def close(self) -> None:
        """Tear down.  A browser we ATTACHED to is left running."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - teardown is best-effort
                logger.debug("webmcp: CDP close failed", exc_info=True)
        self._conn = None
        self._session_id = None
        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:  # pragma: no cover
                logger.debug("webmcp: browser terminate failed", exc_info=True)
            self._process = None


def _parse_content(raw: Any) -> Any:
    """Unwrap the ``{content: [...]}`` envelope Chrome returns as a JSON string.

    Falls back to the raw value when it is not the documented envelope: the
    shape is young (output schemas and streaming are still open questions in
    the spec), and a result the model can read beats an exception.
    """
    if not isinstance(raw, str):
        return raw
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(parsed, dict) and isinstance(parsed.get("content"), list):
        return parsed["content"]
    return parsed
