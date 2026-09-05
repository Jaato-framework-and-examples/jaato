"""Prompt API page driver for the Chrome built-in AI provider.

:class:`PromptApiBridge` owns the whole browser side of the provider:
the Chrome process (or an attachment to a user-managed one), the CDP
connection, one dedicated page, and a small helper script installed into
that page which fronts the ``LanguageModel`` API.

Data flow for one generation turn:

1. ``start_turn(payload)`` evaluates ``__jaato.run(<id>, <payload>)`` in
   the page (fire-and-forget — the promise is NOT awaited so streaming
   events can flow while it runs).
2. The helper creates a fresh ``LanguageModel`` session from
   ``payload.initialPrompts`` (the provider is stateless per the
   ``ModelProviderPlugin`` contract, so context is rebuilt every turn and
   the session destroyed afterwards), then iterates
   ``session.promptStreaming(payload.prompt)``.
3. Every chunk/completion/error is reported through the CDP binding
   ``__jaatoEmit`` → ``Runtime.bindingCalled`` → the per-run
   ``queue.Queue`` returned by ``events()``.
4. ``abort(run_id)`` triggers the run's ``AbortController`` — this is
   what makes ``cancellation=True`` honest.

The helper feature-detects the ``inputQuota``/``inputUsage`` →
``contextWindow``/``contextUsage`` spec rename so it works on both
Chrome generations.
"""

import json
import logging
import queue
import subprocess
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from shared import cdp
from shared.cdp import CDPConnectionError
from .errors import ChromeAIConnectionError, ChromeAIUnavailableError

logger = logging.getLogger(__name__)

#: CDP binding name the page-side helper reports events through.
BINDING_NAME = "__jaatoEmit"

#: Page-side helper installed once per page.  Idempotent (guards on
#: ``window.__jaato``).  All entry points report through the binding; no
#: promise results cross the CDP evaluate boundary except tiny metadata
#: calls (availability / probe).
HELPER_JS = r"""
(() => {
  if (window.__jaato) { return "ok"; }
  const pending = new Map();
  const quotaOf = (s) => {
    const q = (s.inputQuota !== undefined) ? s.inputQuota : s.contextWindow;
    return (typeof q === "number" && isFinite(q)) ? q : null;
  };
  const usageOf = (s) => {
    const u = (s.inputUsage !== undefined) ? s.inputUsage : s.contextUsage;
    return (typeof u === "number" && isFinite(u)) ? u : null;
  };
  window.__jaato = {
    async availability() {
      const LM = globalThis.LanguageModel;
      if (!LM || typeof LM.availability !== "function") { return "no-api"; }
      try { return await LM.availability(); }
      catch (e) { return "error:" + String((e && e.message) || e); }
    },
    async probe() {
      const session = await LanguageModel.create();
      try { return { quota: quotaOf(session) }; }
      finally { try { session.destroy(); } catch (e) {} }
    },
    async download(id) {
      const emit = (o) => __jaatoEmit(JSON.stringify(Object.assign({ id }, o)));
      try {
        const session = await LanguageModel.create({
          monitor(m) {
            m.addEventListener("downloadprogress", (e) => {
              emit({ type: "progress", loaded: e.loaded });
            });
          }
        });
        try { session.destroy(); } catch (e) {}
        emit({ type: "done" });
      } catch (e) {
        emit({ type: "error", message: String((e && e.message) || e) });
      }
    },
    async run(id, payload) {
      const emit = (o) => __jaatoEmit(JSON.stringify(Object.assign({ id }, o)));
      const controller = new AbortController();
      pending.set(id, controller);
      let session = null;
      try {
        const createOpts = { signal: controller.signal };
        if (payload.initialPrompts && payload.initialPrompts.length) {
          createOpts.initialPrompts = payload.initialPrompts;
        }
        if (payload.temperature != null) { createOpts.temperature = payload.temperature; }
        if (payload.topK != null) { createOpts.topK = payload.topK; }
        session = await LanguageModel.create(createOpts);
        const promptOpts = { signal: controller.signal };
        if (payload.responseConstraint) {
          promptOpts.responseConstraint = payload.responseConstraint;
        }
        const stream = session.promptStreaming(payload.prompt, promptOpts);
        let text = "";
        for await (const chunk of stream) {
          text += chunk;
          emit({ type: "chunk", text: chunk });
        }
        emit({ type: "done", text, usage: { inputUsage: usageOf(session),
                                            inputQuota: quotaOf(session) } });
      } catch (e) {
        if (e && e.name === "AbortError") {
          emit({ type: "cancelled" });
        } else {
          emit({ type: "error", message: String((e && e.message) || e),
                 name: String((e && e.name) || "") });
        }
      } finally {
        pending.delete(id);
        if (session) { try { session.destroy(); } catch (e) {} }
      }
    },
    abort(id) {
      const c = pending.get(id);
      if (c) { c.abort(); }
    }
  };
  return "ok";
})()
"""


class PromptApiBridge:
    """Browser-side driver: process + CDP + page + helper script.

    Lifecycle: construct with resolved settings (no I/O) → ``connect()``
    (launch/attach, create page, install helper) → any number of
    ``start_turn()``/``events()``/``abort()`` cycles → ``close()``.
    ``is_alive`` reports whether the CDP link (and owned process, if any)
    is still usable; the provider checks it per-turn and reconnects.
    """

    def __init__(
        self,
        *,
        binary: Optional[str] = None,
        cdp_url: Optional[str] = None,
        user_data_dir: Optional[str] = None,
        headless: bool = True,
        page_url: str = "about:blank",
        extra_args: Optional[List[str]] = None,
        connect_timeout: float = 30.0,
        reuse_page: bool = False,
    ) -> None:
        self._binary = binary
        self._cdp_url = cdp_url
        self._user_data_dir = user_data_dir
        self._headless = headless
        self._page_url = page_url
        self._extra_args = list(extra_args or [])
        self._connect_timeout = connect_timeout
        self._reuse_page = reuse_page

        self._process: Optional[subprocess.Popen] = None
        self._conn: Optional[cdp.CDPConnection] = None
        self._session_id: Optional[str] = None
        self._target_id: Optional[str] = None
        #: Whether WE created ``_target_id`` (→ close it on teardown) or
        #: merely attached to a pre-existing user tab (→ leave it open).
        self._owns_target: bool = False
        self._queues: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}
        self._queues_lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle

    @property
    def is_alive(self) -> bool:
        """Whether the CDP link (and owned browser process) is usable."""
        if self._conn is None or not self._conn.is_open:
            return False
        if self._process is not None and self._process.poll() is not None:
            return False
        return True

    def connect(self) -> None:
        """Launch (or attach to) the browser and prepare the API page.

        Attach mode (``cdp_url`` set) never spawns a process — the user
        owns the browser's lifetime.  Launch mode spawns Chrome on the
        configured persistent profile (the Gemini Nano download is
        profile-bound, so ephemeral profiles would re-download per run).

        Page acquisition: by default a fresh tab is created at
        ``page_url`` (and closed on teardown).  When ``reuse_page`` is set,
        the bridge first looks for an already-open page whose URL equals
        ``page_url`` and *attaches* to it instead — no new tab, and the tab
        is left open on ``close()`` because it belongs to the user.  Only
        if no such tab exists does it fall back to creating one.  This is
        how a caller anchors the Prompt API onto a real https tab the user
        already has open (see the ``reuse_page`` provider knob).
        """
        if self._cdp_url:
            ws_url = cdp.discover_ws_url(self._cdp_url,
                                         timeout=self._connect_timeout)
        else:
            if not self._binary:
                raise ChromeAIConnectionError(
                    "no browser binary resolved (bridge misconfigured)")
            if not self._user_data_dir:
                raise ChromeAIConnectionError(
                    "no user_data_dir resolved (bridge misconfigured)")
            self._process, ws_url = cdp.launch_browser(
                self._binary,
                self._user_data_dir,
                headless=self._headless,
                extra_args=self._extra_args,
                timeout=self._connect_timeout,
            )
        self._conn = cdp.CDPConnection(ws_url,
                                       open_timeout=self._connect_timeout)
        self._conn.add_event_listener(self._on_cdp_event)

        existing = self._find_existing_page() if self._reuse_page else None
        if existing:
            self._target_id = existing
            self._owns_target = False
        else:
            target = self._conn.send("Target.createTarget",
                                     {"url": self._page_url})
            self._target_id = target["targetId"]
            self._owns_target = True
        attach = self._conn.send("Target.attachToTarget",
                                 {"targetId": self._target_id,
                                  "flatten": True})
        self._session_id = attach["sessionId"]
        self._page_send("Runtime.enable")
        self._page_send("Runtime.addBinding", {"name": BINDING_NAME})
        self._wait_page_ready()
        self._ensure_helper()

    def close(self) -> None:
        """Tear down the page, the CDP link, and any owned process.

        A tab we *created* (``_owns_target``) is closed; a pre-existing
        tab we merely *attached* to (``reuse_page`` mode) is left open —
        it belongs to the user, so closing it would be destructive.
        """
        if self._conn is not None and self._conn.is_open:
            try:
                if self._target_id and self._owns_target:
                    self._conn.send("Target.closeTarget",
                                    {"targetId": self._target_id}, timeout=5)
            except CDPConnectionError:
                pass
            try:
                if self._process is not None:
                    # We own the browser: ask it to exit cleanly.
                    self._conn.send("Browser.close", timeout=5)
            except CDPConnectionError:
                pass
            self._conn.close()
        self._conn = None
        self._session_id = None
        self._target_id = None
        self._owns_target = False
        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    # ------------------------------------------------------- Prompt API ops

    def availability(self) -> str:
        """The model's availability state.

        Returns one of the Prompt API states (``"available"``,
        ``"downloadable"``, ``"downloading"``, ``"unavailable"``) or the
        bridge-specific ``"no-api"`` when the ``LanguageModel`` global is
        absent, or ``"error:<message>"`` if the query itself threw.
        """
        return str(self._eval("window.__jaato.availability()"))

    def probe_quota(self) -> Optional[int]:
        """Create a throwaway session and read its context quota (tokens).

        Feature-detects the ``inputQuota`` → ``contextWindow`` rename.
        Returns ``None`` when the API reports no numeric quota — the
        provider then falls back to the manual context-length tiers.
        Failure-tolerant per the ``resolve_context_window`` contract.
        """
        try:
            result = self._eval("window.__jaato.probe()")
        except (CDPConnectionError, ChromeAIUnavailableError):
            return None
        if isinstance(result, dict):
            quota = result.get("quota")
            if isinstance(quota, (int, float)) and quota > 0:
                return int(quota)
        return None

    def download(self, on_progress: Optional[Callable[[float], None]] = None,
                 timeout: float = 1800.0) -> None:
        """Trigger the on-device model component download and wait for it.

        The Prompt API downloads the model on first ``create()``; progress
        events (0.0-1.0) are forwarded to ``on_progress``.  The component
        is ~2-4 GB, hence the generous default timeout.

        Raises:
            ChromeAIUnavailableError: when the download fails.
            ChromeAIConnectionError: on timeout or CDP loss.
        """
        run_id = self._new_run_id()
        events = self.events(run_id)
        try:
            self._call_helper("download", run_id)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ChromeAIConnectionError(
                        f"model download did not finish within {timeout:.0f}s")
                try:
                    event = events.get(timeout=min(remaining, 30.0))
                except queue.Empty:
                    if not self.is_alive:
                        raise ChromeAIConnectionError(
                            "browser died during model download")
                    continue
                etype = event.get("type")
                if etype == "progress":
                    if on_progress is not None:
                        try:
                            on_progress(float(event.get("loaded") or 0.0))
                        except (TypeError, ValueError):
                            pass
                elif etype == "done":
                    return
                elif etype == "error":
                    raise ChromeAIUnavailableError(
                        f"model download failed: {event.get('message')}")
        finally:
            self.finish_turn(run_id)

    def warmup(self, timeout: float = 60.0) -> None:
        """Run one throwaway generation to absorb the model's cold-start.

        Drives the same ``run`` path as a real turn with a minimal prompt
        and no history, draining events until the turn concludes and
        discarding all output. The first post-provision inference pays the
        on-device compile/load cost (~11s to first token), so calling this
        once at connect() moves that off the first real turn. Returns on the
        first terminal event; raises :class:`ChromeAIConnectionError` on
        timeout or browser death (the provider treats a warmup failure as
        non-fatal).
        """
        run_id = self.start_turn({"initialPrompts": [], "prompt": "Hi"})
        events = self.events(run_id)
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.abort(run_id)
                    raise ChromeAIConnectionError(
                        f"model warmup produced no result within {timeout:.0f}s")
                try:
                    event = events.get(timeout=min(remaining, 5.0))
                except queue.Empty:
                    if not self.is_alive:
                        raise ChromeAIConnectionError(
                            "browser died during model warmup")
                    continue
                if event.get("type") in ("done", "error", "cancelled"):
                    return
        finally:
            self.finish_turn(run_id)

    def start_turn(self, payload: Dict[str, Any]) -> str:
        """Kick off one generation run; returns its ``run_id``.

        Fire-and-forget on the CDP side: the helper's promise is not
        awaited, so chunk events stream through the binding while the
        model generates.  Consume them via ``events(run_id)`` and always
        call ``finish_turn(run_id)`` when done.
        """
        run_id = self._new_run_id()
        with self._queues_lock:
            self._queues.setdefault(run_id, queue.Queue())
        # Re-assert the page-side helper before every turn: in reuse_page
        # mode the anchored tab may have navigated since connect() (a
        # browser tool, or the user), which wipes ``window.__jaato``.  The
        # helper is idempotent and the CDP binding survives navigation, so
        # this is a cheap no-op on a settled page and a self-heal otherwise.
        self._wait_page_ready()
        self._ensure_helper()
        self._call_helper("run", run_id, payload)
        return run_id

    def events(self, run_id: str) -> "queue.Queue[Dict[str, Any]]":
        """The event queue for ``run_id`` (created on first access)."""
        with self._queues_lock:
            return self._queues.setdefault(run_id, queue.Queue())

    def abort(self, run_id: str) -> None:
        """Fire the run's AbortController (best-effort, never raises)."""
        try:
            self._eval(f"window.__jaato.abort({json.dumps(run_id)})",
                       await_promise=False)
        except CDPConnectionError:
            logger.debug("chrome_ai: abort(%s) after connection loss", run_id)

    def finish_turn(self, run_id: str) -> None:
        """Drop the run's event queue (call after the turn concludes)."""
        with self._queues_lock:
            self._queues.pop(run_id, None)

    # -------------------------------------------------------------- internal

    @staticmethod
    def _new_run_id() -> str:
        return uuid.uuid4().hex

    def _find_existing_page(self) -> Optional[str]:
        """Return the targetId of an open page whose URL == ``page_url``.

        Used only in ``reuse_page`` mode to anchor onto a tab the user
        already has open instead of creating a new one.  Matches on the
        exact URL (``Target.getTargets`` reports each page's current URL);
        ``None`` when no page matches, so ``connect()`` falls back to
        creating a fresh tab.  Never raises — a CDP hiccup here just means
        "no match", and creation proceeds.
        """
        try:
            result = self._conn.send("Target.getTargets")
        except CDPConnectionError:
            return None
        for info in (result or {}).get("targetInfos", []):
            if info.get("type") == "page" and info.get("url") == self._page_url:
                return info.get("targetId")
        return None

    def _ensure_helper(self) -> None:
        """(Re)install the page-side ``window.__jaato`` helper.

        Idempotent — the helper guards on ``window.__jaato`` and returns
        ``"ok"`` when already present — so it is safe (and cheap) to call
        before every turn.  This is what makes ``reuse_page`` robust: a
        navigation on the anchored tab wipes ``window.__jaato`` but not the
        ``Runtime.addBinding`` binding, so re-evaluating the helper here
        restores the full page-side surface without a reconnect.
        """
        result = self._eval(HELPER_JS)
        if result != "ok":
            raise ChromeAIConnectionError(
                f"helper script installation returned {result!r}")

    def _page_send(self, method: str, params: Optional[Dict[str, Any]] = None,
                   timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send a CDP command scoped to the API page's session."""
        if self._conn is None or self._session_id is None:
            raise ChromeAIConnectionError("bridge is not connected")
        return self._conn.send(method, params, session_id=self._session_id,
                               timeout=timeout)

    def _eval(self, expression: str, *, await_promise: bool = True,
              timeout: Optional[float] = None) -> Any:
        """Evaluate JS in the page and return its by-value result.

        Raises :class:`ChromeAIUnavailableError` when the page-side code
        threw (surfaced via ``exceptionDetails``) — that is an API-level
        failure, not a transport one.
        """
        result = self._page_send("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": True,
        }, timeout=timeout)
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            exc = details.get("exception") or {}
            message = (exc.get("description") or details.get("text")
                       or "page evaluation failed")
            raise ChromeAIUnavailableError(f"Prompt API call failed: {message}")
        return result.get("result", {}).get("value")

    def _call_helper(self, fn: str, run_id: str,
                     payload: Optional[Dict[str, Any]] = None) -> None:
        """Invoke a fire-and-forget helper entry point.

        ``json.dumps`` with the default ``ensure_ascii=True`` produces a
        valid JS literal (no raw U+2028/U+2029 issues), so the payload can
        be embedded directly in the expression.
        """
        args = json.dumps(run_id)
        if payload is not None:
            args += ", " + json.dumps(payload)
        self._eval(f"void window.__jaato.{fn}({args})", await_promise=False)

    def _wait_page_ready(self, timeout: float = 15.0) -> None:
        """Poll until the page's document has finished loading.

        ``about:blank`` is instant; a real ``page_url`` needs a beat.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                state = self._eval("document.readyState", await_promise=False)
            except ChromeAIUnavailableError:
                state = None
            if state == "complete":
                return
            time.sleep(0.1)
        raise ChromeAIConnectionError(
            f"page {self._page_url!r} did not finish loading within "
            f"{timeout:.0f}s")

    def _on_cdp_event(self, frame: Dict[str, Any]) -> None:
        """Route ``Runtime.bindingCalled`` payloads to per-run queues.

        Runs on the CDP reader thread — parse and enqueue only.
        """
        if frame.get("method") != "Runtime.bindingCalled":
            return
        params = frame.get("params", {})
        if params.get("name") != BINDING_NAME:
            return
        try:
            event = json.loads(params.get("payload", ""))
        except (json.JSONDecodeError, TypeError):
            logger.warning("chrome_ai: unparseable binding payload dropped")
            return
        run_id = event.get("id")
        if not run_id:
            return
        with self._queues_lock:
            q = self._queues.get(run_id)
        if q is not None:
            q.put(event)
