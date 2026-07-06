"""In-process convenience facade — Shape 1 (the embedded ``InProcessClient``).

Run the ``jaato_sdk`` ``ask`` / ``complete`` / ``stream`` facade against an
*embedded* ``jaato.JaatoClient`` — no daemon, no runner, no socket. See
``docs/design/in-process-facade.md``.

``InProcessClient`` is the embedded analog of ``jaato_sdk``'s ``IPCClient``: it
implements the small client contract the facade's ``Session`` rides on
(``subscribe`` / ``subscribe_once`` / ``send_message`` /
``respond_to_permission`` / ``connect`` / ``create_session`` / ``disconnect``)
by wrapping the embedded ``JaatoClient``. The facade (``Session.ask`` /
``.complete`` / ``.stream``) is reused **unchanged** — the whole point of
Shape 1. Everything below the facade is the embedded client REPLICATING what
the daemon does for an IPC session, since there is no daemon in-process:

* ``AGENT_OUTPUT`` <- ``send_message(on_output=...)``; ``TURN_COMPLETED`` <- the
  ``send_message`` return.
* ``PERMISSION_REQUESTED`` / gating <- the ``InProcessChannel`` registered on the
  loaded permission plugin (``_in_process_permission``).
* ``AGENT_COMPLETED`` (the typed completion payload) <- the ``set_ui_hooks``
  ``on_agent_completed`` bridge.
* tool executors, profiles, plugin policy, agent personas <- ``connect`` builds
  the full runner-tier registry ONCE (plugin parity with a daemon session — see
  ``_build_registry``), reused by every ``create_session``; the profile / agent
  persona is resolved per session and ``create_session(plugins=...)`` gates the
  model's tool view.

Remaining: ``stream`` polish.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jaato_sdk.client.convenience import Session
from jaato_sdk.events import (
    AgentCompletedEvent,
    AgentOutputEvent,
    SessionTerminatedEvent,
    TurnCompletedEvent,
)

from jaato_embedded.permission import InProcessChannel, PendingPermissions


def _default_embedded_factory(
    provider: Optional[str],
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
) -> Any:
    """Build the real embedded runtime for ``provider``.

    ``workspace_path`` + ``config_root`` are LOAD-BEARING for isolation: the
    embedded ``JaatoClient``'s runtime composes the system prompt and discovers
    instructions / agents / the prompt library from ``config_root`` (default
    ``<workspace>/.jaato``). When unpinned, that discovery walks up to the
    operator's home-tier ``~/.jaato`` and bleeds the operator's personal config
    into every embedded session — the daemon pins it; the embedded path must
    too. (The registry's ``set_config_root`` does NOT cover this — prompt_library
    / instruction discovery keys off the JaatoClient's config_root, not the
    registry's.)

    Imported lazily so only embedded use pulls in the server runtime — callers
    that never embed pay nothing for importing this module.
    """
    from shared import JaatoClient
    return JaatoClient(
        provider_name=provider,
        workspace_path=workspace_path,
        config_root=config_root,
    )


def _resolve_plugin_config_secrets(
    plugin_configs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve credential secret URIs in ``plugin_configs`` before the provider
    is built.

    The daemon resolves these UPSTREAM during profile/spec resolution; the
    embedded path has no such step, so it must resolve ``pass://``-style
    credential URIs itself (``create_provider`` only validates, never resolves,
    per PR #415). No-op for plain keys / env configs. See
    ``docs/design/in-process-facade.md``.

    ``resolve_secret_uri`` is imported lazily (only embedded use pulls it) so
    importing ``jaato_embedded.client`` stays cheap.
    """
    from shared.config_resolver import resolve_secret_uri

    if not plugin_configs:
        return {}
    resolved: Dict[str, Any] = {}
    for provider, cfg in plugin_configs.items():
        if isinstance(cfg, dict):
            resolved[provider] = {
                k: (resolve_secret_uri(v) if isinstance(v, str) else v)
                for k, v in cfg.items()
            }
        else:
            resolved[provider] = cfg
    return resolved


def _resolve_named_profile(name: str, config_root: Optional[str]) -> Dict[str, Any]:
    """Resolve a named profile from disk into a spec dict.

    The embedded analog of the daemon's profile resolution (the IPC client
    sends ``profile=name`` and the daemon resolves it; an embedded session has
    no daemon, so it discovers the profile itself). Mirrors the daemon loader:
    ``discover_profiles`` -> ``SubagentProfile`` -> the spec fields the session
    needs.
    """
    if not config_root:
        raise ValueError(
            f"named profile {name!r} needs config_root (the directory whose "
            f"'profiles/' holds the profile YAML/JSON)"
        )
    from shared.plugins.subagent.config import discover_profiles

    profiles_dir = str(Path(config_root).expanduser() / "profiles")
    result = discover_profiles(profiles_dir, config_root=str(config_root))
    profile = result.profiles.get(name)
    if profile is None:
        raise ValueError(
            f"profile {name!r} not found under {profiles_dir} "
            f"(available: {sorted(result.profiles)})"
        )
    return {
        "model": profile.model,
        "provider": profile.provider,
        "plugins": profile.plugins,
        "plugin_configs": profile.plugin_configs,
        "system_instructions": profile.system_instructions,
        "completion_payload_schema": profile.completion_payload_schema,
        "suppress_base_instructions": profile.suppress_base_instructions,
    }


def _resolve_agent_persona(
    agent_name: str,
    agent_params: Optional[Dict[str, str]],
    workspace_path: Optional[str],
    config_root: Optional[str],
) -> str:
    """Resolve an agent persona (.jaato/agents/<name>.md) to its system
    instructions — the embedded analog of the daemon resolving ``--agent``.

    Reuses the daemon's exact loader (lifted to ``shared`` so the daemon-free
    embedded path doesn't import ``server``), so an embedded session renders the
    persona (frontmatter + ``{{param}}`` substitution) identically. Raises
    ``ValueError`` when the named agent isn't found (fail loud, no silent bare).
    """
    from shared.plugins.subagent.config import resolve_agent

    resolved = resolve_agent(agent_name, agent_params, workspace_path, config_root)
    if resolved is None:
        where = config_root or workspace_path or "~"
        raise ValueError(
            f"agent {agent_name!r} not found under {where}/agents (or .jaato/agents)"
        )
    return resolved["system_instructions"]


class InProcessEventEmitter:
    """Thread-safe in-process pub/sub matching the facade's subscribe contract.

    The no-socket analog of the IPC client's event fan-out. The facade rides on
    exactly three methods:

    * ``subscribe(event_type, cb) -> unsubscribe``
    * ``subscribe_once(event_type, cb) -> unsubscribe`` (auto-unsubscribes
      before dispatching, so a re-entrant ``emit`` cannot double-fire)
    * ``emit(event)`` — dispatch to subscribers keyed on ``event.type``
    """

    def __init__(self) -> None:
        self._subs: Dict[Any, List[Callable[[Any], None]]] = {}
        self._all_subs: List[Callable[[Any], None]] = []
        self._lock = threading.Lock()

    def subscribe(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        with self._lock:
            self._subs.setdefault(event_type, []).append(cb)

        def _unsub() -> None:
            with self._lock:
                lst = self._subs.get(event_type)
                if lst and cb in lst:
                    lst.remove(cb)

        return _unsub

    def subscribe_all(
        self, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        """Subscribe to EVERY event (type-agnostic), returning an unsubscribe.

        Backs :meth:`InProcessClient.open_event_stream` / ``events`` — the
        no-socket analog of the IPC client's all-events fan-out.  ``cb`` fires
        on every ``emit`` regardless of ``event.type``.
        """
        with self._lock:
            self._all_subs.append(cb)

        def _unsub() -> None:
            with self._lock:
                if cb in self._all_subs:
                    self._all_subs.remove(cb)

        return _unsub

    def subscribe_once(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        box: Dict[str, Callable[[], None]] = {}

        def _wrapper(ev: Any) -> None:
            unsub = box.get("unsub")
            if unsub is not None:
                unsub()
            cb(ev)

        unsub = self.subscribe(event_type, _wrapper)
        box["unsub"] = unsub
        return unsub

    def emit(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        with self._lock:
            subs = list(self._subs.get(event_type, [])) + list(self._all_subs)
        for cb in subs:
            cb(event)


class _InProcessUIHooks:
    """``AgentUIHooks`` impl that emits ``AGENT_COMPLETED`` (the completion-gate
    payload) onto the facade emitter — the in-process analog of the daemon's
    ``on_agent_completed`` notification -> AGENT_COMPLETED.

    Only ``on_agent_completed`` is mapped here: ``AGENT_OUTPUT`` already flows
    via ``send_message(on_output=...)`` (so ``on_agent_output`` stays a no-op to
    avoid double-emit), and the other lifecycle hooks are no-ops for now. The
    embedded runtime calls these on its worker thread, so ``emit_threadsafe``
    marshals onto the event loop.
    """

    def __init__(self, emit_threadsafe: Callable[[Any], None]) -> None:
        self._emit = emit_threadsafe

    def on_agent_completed(
        self,
        agent_id: str = "",
        completed_at: Any = None,
        success: bool = True,
        token_usage: Any = None,
        turns_used: Any = None,
        payload: Optional[Dict[str, Any]] = None,
        **_kw: Any,
    ) -> None:
        self._emit(
            AgentCompletedEvent(
                agent_id=agent_id or "main",
                success=bool(success),
                payload=payload,
            )
        )

    def on_session_quiescent(
        self, agent_id: str = "", reason: str = "natural", **_kw: Any,
    ) -> None:
        """Emit SESSION_TERMINATED when the session winds down quiescent — fired
        by ``JaatoSession`` after ``signal_completion`` + the turn wraps up
        (jaato_session.py:5204). The daemon's adapter emits it to attached
        clients; the embedded path emits it onto the facade emitter. Explicit
        (not the ``__getattr__`` no-op) so a delegation lead that signals
        completion actually surfaces SESSION_TERMINATED to the facade — the
        event ex08 awaits."""
        self._emit(
            SessionTerminatedEvent(agent_id=agent_id or "main", reason=reason)
        )

    def __getattr__(self, name: str) -> Callable[..., None]:
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


class InProcessClient:
    """Embedded analog of ``IPCClient`` — the facade's ``Session`` rides on it.

    Wraps an embedded ``jaato.JaatoClient`` and exposes the facade client
    contract over an in-process event emitter (no socket). Construct via
    :meth:`session` (the ``async with`` entry point), not directly.

    The embedded runtime runs the blocking agent loop, so ``send_message`` is
    dispatched to a worker thread via ``asyncio.to_thread``; its
    ``on_output`` callback fires on that worker thread and marshals
    ``AGENT_OUTPUT`` events back onto the event loop with
    ``loop.call_soon_threadsafe``.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        plugin_configs: Optional[Dict[str, Any]] = None,
        plugins: Optional[List[str]] = None,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
        system_instructions: Optional[str] = None,
        completion_payload_schema: Optional[Any] = None,
        suppress_base_instructions: bool = False,
        env_file: Optional[str] = ".env",
        project: Optional[str] = None,
        location: Optional[str] = None,
        embedded_factory: Optional[Callable[[Optional[str]], Any]] = None,
        registry_factory: Optional[Callable[[], Any]] = None,
        **_ignored: Any,
    ) -> None:
        self._model = model
        self._provider = provider
        self._plugin_configs = plugin_configs
        self._resolved_plugin_configs: Dict[str, Any] = {}
        # The session's tool plugins (the spec's ``plugins`` list) — the MODEL's
        # tool-visibility gate, NOT the set of plugins INITIALIZED (for plugin
        # parity the registry always initializes the full runner-tier set; see
        # _build_registry). Gates which tools the model sees, exactly like the
        # daemon's create_session(plugins=...). Full IPC parity on the three
        # cases — so None must be PRESERVED, never collapsed to []:
        #   * None  -> INHERIT. In a base (non-derived) profile this drags ALL
        #     plugins (core + discoverable); in a derived profile it inherits the
        #     base's plugin set. The downstream create_session(plugins=None)
        #     applies the inherit/drag-all semantics.
        #   * []    -> explicit OVERRIDE: drag NONE (minimal essential surface
        #     only — introspection/permission). NOT the same as None.
        #   * [..]  -> exactly those plugins.
        self._plugins = list(plugins) if plugins is not None else None
        # Default the workspace to the embedding process's cwd, and config_root
        # to ``<workspace>/.jaato`` — the in-process analog of a daemon session's
        # workspace (client working_dir) + ``.jaato`` config root. Needed so the
        # full runner-tier plugin set initializes (plugin parity): config-rooted
        # plugins like file_edit fail to init without a config_root.
        self._workspace_path = workspace_path or os.getcwd()
        self._config_root = config_root or str(Path(self._workspace_path) / ".jaato")
        # Profile-derived (or directly-supplied) session instructions + the
        # typed-completion gate schema — forwarded to create_session so the
        # embedded session matches the daemon's (ex03 persona, ex04 byte-exact).
        self._system_instructions = system_instructions
        self._completion_payload_schema = completion_payload_schema
        # The profile knob that controls base-layer composition:
        # ``include_base = not suppress_base_instructions`` in configure(). False
        # (default) -> persona/system_instructions composes ON TOP of the
        # framework base; True -> the base is dropped (the profile keeps only its
        # own agent/plugin/framework content). Threaded so an embedded session
        # honors the same knob the daemon does.
        self._suppress_base_instructions = suppress_base_instructions
        # ``env_file`` is a BOTH-modes kwarg (unlike ``socket_path``, which is
        # IPC-only): the embedded runtime reads the same env the daemon loads
        # from ``.env`` (``JAATO_PROVIDER`` / ``MODEL_NAME`` / provider creds),
        # so the in-process path loads it too. Defaults to ``.env`` for parity
        # with ``IPCClient``; loaded in :meth:`connect` if the file exists.
        self._env_file = env_file
        self._project = project
        self._location = location
        # Test seam: inject a fake embedded client factory ``(provider) ->
        # client``. Production uses the real ``jaato.JaatoClient``.
        self._embedded_factory = embedded_factory or _default_embedded_factory
        # Test seam: inject the connect-time registry builder. None ->
        # _build_registry for the real embedded runtime, empty registry for an
        # injected (test) embedded factory.
        self._registry_factory = registry_factory
        self._embedded: Any = None
        # The session registry — built ONCE at connect (runtime boundary), reused
        # by every create_session. The in-process analog of the daemon's pre-warm
        # template: init the runner-tier plugins once, share across sessions.
        self._registry: Any = None
        self._emitter = InProcessEventEmitter()
        # Live open_event_stream / events queues + their emitter unsubscribes,
        # so disconnect() can push the None sentinel to each (mirrors the IPC
        # drain loop signalling disconnect).  Keyed by id(queue).
        self._event_queues: List[Any] = []
        self._event_queue_unsubs: Dict[int, Callable[[], None]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_id: Optional[str] = None
        # Cross-thread permission rendezvous: the InProcessChannel (registered
        # on the session's permission plugin once plugin-loading lands) parks a
        # request here; respond_to_permission resolves it.
        self._pending = PendingPermissions()
        # Client-provided ("host") tool specs, registered via
        # register_client_tools() BEFORE create_session so they reach the
        # session's wire (turn 1). Each: {name, description, parameters,
        # handler} (+ optional auto_approve). The in-process analog of the
        # daemon's ToolsRegisterClientRequest — but with NO wire: the handler
        # runs in-process directly (registered on the embedded session's
        # executor), so there is no daemon round-trip.
        self._client_tools: Optional[List[Dict[str, Any]]] = None
        # Auto-continue (subagent delegation): the embedded analog of the
        # daemon's run-loop resume. When the IDLE lead receives an injected
        # child/parent message (a subagent posting its result), the session calls
        # its continuation callback; _schedule_lead_continuation runs a follow-up
        # lead turn. These two fields serialize + coalesce continuations on the
        # (single-threaded) facade loop so overlapping injects don't double-run.
        self._continuation_busy = False
        self._continuation_pending: Optional[str] = None
        # agent_ids of subagents that have posted a TERMINAL inject
        # ([SUBAGENT ... event=COMPLETED|ERROR|CANCELLED]). The completion-nudge
        # gate keys on this — NOT session.is_running, which flickers mid-turn and
        # reads False both pre-spawn (the subagent isn't in _active_sessions yet)
        # and between the subagent's turns, opening the gate prematurely so the
        # lead nudges + signals before the subagent delivers its result.
        self._completed_subagent_ids: set = set()

    async def register_client_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Register client-provided ("host") tools the embedded agent can call.

        Mirrors :meth:`jaato_sdk.IPCClient.register_client_tools` (same spec
        shape — ``{name, description, parameters, handler}`` plus optional
        ``auto_approve``) so the facade is transport-agnostic. Stored here and
        applied to the embedded session by :meth:`create_session` AFTER
        ``configure_tools`` builds it. Must be called before ``create_session``
        so the schema reaches the session's initial wire (the facade's
        ``_SessionContext`` enforces this ordering, exactly like IPC).

        Unlike IPC there is no daemon to wire the schema to and no
        ``TOOL_EXECUTE_REQUEST`` round-trip: the ``handler(args) -> Any`` is
        registered directly on the embedded session's ``ToolExecutor``, so it
        runs in the embedding process when the model invokes the tool.
        """
        self._client_tools = list(tools) if tools else None

    # ---- facade contract: events ----
    def subscribe(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        return self._emitter.subscribe(event_type, cb)

    def subscribe_once(
        self, event_type: Any, cb: Callable[[Any], None]
    ) -> Callable[[], None]:
        return self._emitter.subscribe_once(event_type, cb)

    # ---- facade contract: iterator event surface (parity with IPCClient) ----
    def _subscribe_events(self) -> "asyncio.Queue":
        """Subscribe a fresh queue to the emitter's all-events fan-out.

        The no-socket analog of ``IPCClient._subscribe_events``.  ``emit`` runs
        on the event loop thread (both the ``send_message`` output path and the
        UI-hook path marshal via ``call_soon_threadsafe`` BEFORE calling
        ``emit``), so ``queue.put_nowait`` off it is loop-safe.
        """
        q: "asyncio.Queue" = asyncio.Queue()
        unsub = self._emitter.subscribe_all(q.put_nowait)
        self._event_queues.append(q)
        self._event_queue_unsubs[id(q)] = unsub
        return q

    def _unsubscribe_events(self, q: "asyncio.Queue") -> None:
        """Remove a previously-subscribed queue.  Idempotent."""
        unsub = self._event_queue_unsubs.pop(id(q), None)
        if unsub is not None:
            unsub()
        try:
            self._event_queues.remove(q)
        except ValueError:
            pass

    def open_event_stream(self) -> "Any":
        """Subscribe SYNCHRONOUSLY (at call time) and return an event iterator.

        The in-process transport's parity implementation of
        :meth:`IPCClient.open_event_stream`: registers the subscriber queue NOW
        (before returning) so the facade surface is uniform across transports.
        Yields every emitted event; ``None``-sentinel + ``aclose()``/``async
        with`` unsubscribe semantics match the socket clients.  (In-process
        ``subscribe`` is already synchronous, so this exists for a UNIFORM
        surface rather than to close a lazy-subscribe race.)
        """
        from jaato_sdk.client._event_stream import _SyncSubscribedStream
        return _SyncSubscribedStream(self, self._subscribe_events())

    async def events(self):
        """Async iterator over all emitted events (parity with
        :meth:`IPCClient.events`).  Subscribes lazily on first ``__anext__``;
        prefer :meth:`open_event_stream` when subscribe-before-action ordering
        matters.  Exits cleanly on the ``None`` disconnect sentinel."""
        q = self._subscribe_events()
        try:
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
        finally:
            self._unsubscribe_events(q)

    # ---- facade contract: lifecycle ----
    async def connect(self, timeout: Optional[float] = None) -> bool:
        self._loop = asyncio.get_running_loop()
        # Load the session ``.env`` into the process env before the embedded
        # runtime reads it — the embedded analog of the daemon loading
        # ``env_file`` via dotenv. Graceful when the file is absent; does not
        # override env vars already set in the process.
        if self._env_file:
            env_path = Path(self._env_file).expanduser()
            if env_path.is_file():
                from dotenv import load_dotenv

                load_dotenv(str(env_path), override=False)
        # Populate the session-context workspace_root / config_root the SAME way
        # the daemon's runner does at bootstrap (server/runner/session.py): set
        # the ContextVar AND os.environ, BEFORE _build_registry's expose_all.
        # The subagent spawn path (and config-rooted plugins) read
        # ``get_workspace_root()``; in-process nothing populated it, so it
        # returned None and spawn_subagent failed. The ContextVar covers
        # context-inheriting threads (the turn's ``to_thread`` worker); os.environ
        # covers the subagent ``ThreadPoolExecutor``, which does NOT inherit the
        # ContextVar. Like the runner (one session per process) there is no reset.
        from shared.session_context import set_workspace_root, set_config_root
        if self._workspace_path:
            set_workspace_root(self._workspace_path)
            os.environ["JAATO_WORKSPACE_ROOT"] = self._workspace_path
            os.environ["workspaceRoot"] = self._workspace_path
        if self._config_root:
            set_config_root(self._config_root)
            os.environ["JAATO_CONFIG_ROOT"] = self._config_root
        # Resolve credential secret URIs before the provider is built — the
        # embedded analog of the daemon's upstream profile/spec resolution.
        self._resolved_plugin_configs = _resolve_plugin_config_secrets(
            self._plugin_configs
        )
        # Pass workspace_path + config_root so the embedded runtime pins its
        # framework-config root (default <workspace>/.jaato) — without this the
        # system-prompt / instruction / prompt-library discovery inherits the
        # operator's home-tier ~/.jaato and contaminates every session.
        self._embedded = self._embedded_factory(
            self._provider, self._workspace_path, self._config_root
        )
        await asyncio.to_thread(
            self._embedded.connect, self._project, self._location, self._model
        )
        # verify_auth AFTER connect, BEFORE configure_tools (jaato_client.py
        # contract). Forwards plugin_configs so providers reading the cred from
        # ``plugin_configs[provider]`` see it. Fail loud before the first turn.
        authed = await asyncio.to_thread(
            self._embedded.verify_auth,
            plugin_configs=self._resolved_plugin_configs or None,
        )
        if not authed:
            raise RuntimeError(
                f"Authentication not configured for provider {self._provider!r}"
            )
        # Build the registry ONCE here, at the runtime boundary — the in-process
        # analog of the daemon's pre-warm template initializing the runner-tier
        # plugins a single time and sharing them across every session via fork.
        # create_session reuses this; that amortizes the plugin init to a
        # one-time connect cost (not per-session) and gives async-lifecycle
        # plugins (mcp, webhook) a single stable init point instead of
        # per-session build/teardown churn.
        #
        # Only the REAL embedded runtime gets the full runner-tier build; an
        # injected embedded factory (test seam) gets an empty registry, which the
        # fake's configure_tools ignores — keeps facade tests fast + hermetic
        # (no 31-plugin init / no workspace-config reads). The real registry
        # build is covered directly by the _build_registry tests.
        if self._registry_factory is not None:
            self._registry = await asyncio.to_thread(self._registry_factory)
        elif self._embedded_factory is _default_embedded_factory:
            self._registry = await asyncio.to_thread(self._build_registry)
        else:
            from shared import PluginRegistry

            self._registry = PluginRegistry(model_name=self._model)
        return True

    async def create_session(self, **_kwargs: Any) -> str:
        if not self._session_id:
            self._session_id = f"in-process-{uuid.uuid4().hex[:12]}"
        # Reuse the connect-built registry (shared, daemon-template style); stamp
        # this session's id for plugins that key per-session state on it.
        registry = self._registry
        registry.set_session_id(self._session_id)
        permission_plugin = self._wire_permission_channel(registry)
        # Parse plugin entries exactly like the daemon (parse_plugin_list):
        # split the raw list into clean names + the PRELOAD set (force-eager,
        # bypassing lazy tool-discovery so the plugin's tools are in the initial
        # wire) + per-plugin tool allow-lists. Pre-fix the facade passed the raw
        # list, so "cli(preload)" matched no plugin (silently dropped) and both
        # preload and tool-scoping were ignored — an IPC-parity gap. None stays
        # None (inherit); [] -> ([], no preload, no scopes).
        plugins = self._plugins
        preloaded_plugins = None
        tool_scopes = None
        if plugins is not None:
            from shared.plugins.subagent.config import parse_plugin_list
            plugins, preloaded_plugins, tool_scopes = parse_plugin_list(plugins)
        session_kwargs: Dict[str, Any] = {
            "plugin_configs": self._resolved_plugin_configs,
            "plugins": plugins,
            # The base-composition knob (default False = compose persona on top
            # of the base; True = drop the base). create_session maps it to
            # include_base = not suppress_base_instructions.
            "suppress_base_instructions": self._suppress_base_instructions,
        }
        if preloaded_plugins:
            session_kwargs["preloaded_plugins"] = preloaded_plugins
        if tool_scopes:
            session_kwargs["tool_scopes"] = tool_scopes
        # Profile-derived session instructions (ex03 persona) + the typed
        # completion gate (ex04 byte-exact) — omit when unset so create_session
        # applies its own defaults.
        if self._system_instructions is not None:
            session_kwargs["system_instructions"] = self._system_instructions
        if self._completion_payload_schema is not None:
            session_kwargs["completion_payload_schema"] = (
                self._completion_payload_schema
            )
        await asyncio.to_thread(
            self._embedded.configure_tools,
            registry,
            # Pass the loaded permission plugin so the session GATES tool
            # execution with it. Without this (was None) the policy + channel
            # were wired but never consulted — tools ran un-gated.
            permission_plugin,
            None,  # ledger
            session_kwargs,
        )
        # Install lifecycle hooks AFTER configure_tools (which creates the
        # session) so set_ui_hooks reaches it — the on_agent_completed ->
        # AGENT_COMPLETED bridge the facade's complete() waits on (ex04 payload).
        if hasattr(self._embedded, "set_ui_hooks"):
            loop = self._loop
            emitter = self._emitter

            def emit_threadsafe(ev: Any) -> None:
                loop.call_soon_threadsafe(emitter.emit, ev)

            self._embedded.set_ui_hooks(_InProcessUIHooks(emit_threadsafe))
            # The subagent plugin emits the SUBAGENT's events (on_agent_output /
            # on_agent_completed) through its OWN _ui_hooks slot — the daemon
            # wires it in _setup_agent_hooks (server/core.py); the embedded path
            # must too, with the SAME cross-thread bridge. Without it a spawned
            # subagent runs ISOLATED in its thread pool: it executes model turns
            # but the facade loop sees zero subagent events (the gap #5 symptom).
            subagent_plugin = registry.get_plugin("subagent")
            if subagent_plugin is not None and hasattr(
                subagent_plugin, "set_ui_hooks"
            ):
                subagent_plugin.set_ui_hooks(_InProcessUIHooks(emit_threadsafe))
            # Auto-continue: drive the lead's continuation callback so an IDLE
            # lead resumes when a subagent injects its result — the embedded
            # analog of the daemon's "continuation_needed" handler
            # (server/core.py:4258), which the facade must drive itself (no daemon
            # run loop). The session calls this callback (from inject_prompt /
            # _drain_child_messages) with the drained text; it fires on whatever
            # thread called inject_prompt (a subagent worker), so marshal onto the
            # facade loop. Without it the lead never resumes after delegating, so
            # signal_completion (-> SESSION_TERMINATED) never fires.
            lead_session = (
                self._embedded.get_session()
                if hasattr(self._embedded, "get_session")
                else None
            )
            if lead_session is not None and hasattr(
                lead_session, "set_continuation_callback"
            ):
                lead_session.set_continuation_callback(
                    lambda text: loop.call_soon_threadsafe(
                        self._on_lead_inject, text
                    )
                )
        # Register client/host tools on the now-built session: append their
        # schemas to the wire (so the model SEES them turn 1) and their handlers
        # to the executor (so the model can CALL them in-process). Must run
        # AFTER configure_tools (which creates the session) and is the embedded
        # analog of IPC registering before create_session.
        if self._client_tools:
            self._register_client_tools_on_session(permission_plugin)
        return self._session_id

    def _register_client_tools_on_session(self, permission_plugin: Any) -> None:
        """Append client-tool schemas to the embedded session's wire + register
        their handlers on its executor (+ whitelist the auto-approved ones).

        Appending to ``session._tools`` after configure is the same pattern the
        session itself uses for discovered tools (``activate_discovered_tools``),
        so the schemas reach the provider wire; registering the handler on
        ``session._executor`` is the same contract as
        ``JaatoClient.configure_custom_tools``."""
        from jaato_sdk.plugins.model_provider.types import ToolSchema

        session = self._embedded.get_session()
        if session is None:
            return
        approved: List[str] = []
        for spec in self._client_tools:
            name = spec["name"]
            session._tools.append(ToolSchema(
                name=name,
                description=spec.get("description", ""),
                parameters=spec.get("parameters", {}),
            ))
            handler = spec.get("handler")
            if handler is not None:
                session._executor.register(name, handler)
            if spec.get("auto_approve"):
                approved.append(name)
        if approved and permission_plugin is not None:
            permission_plugin.add_whitelist_tools(approved)

    def _wire_permission_channel(self, registry: Any) -> Any:
        """Register the InProcessChannel on the loaded permission plugin and
        return it (for ``configure_tools`` to gate with).

        The plugin's POLICY is applied by ``expose_all`` (it reads
        ``plugin_configs['permission']['policy']`` at ``initialize``); this
        method swaps its channel so a gated tool's request routes to the facade
        (and back via ``respond_to_permission``). In-process there's no
        runner/thread-local channel, so ``_get_channel`` falls to the plugin's
        default ``_channel`` slot — which we replace here (after ``initialize``
        set it to ConsoleChannel). Returns ``None`` when no permission plugin is
        loaded (the model-only ex01-ex04 path)."""
        permission_plugin = registry.get_plugin("permission")
        if permission_plugin is None:
            return None
        loop = self._loop
        emitter = self._emitter

        def emit_threadsafe(ev: Any) -> None:
            loop.call_soon_threadsafe(emitter.emit, ev)

        permission_plugin._channel = InProcessChannel(emit_threadsafe, self._pending)
        return permission_plugin

    def _build_registry(self) -> Any:
        """Replicate the daemon's per-session registry setup in-process — for
        PLUGIN PARITY: the SAME runner-tier plugins are initialized as a daemon
        session, regardless of which tools the model ends up seeing.

        Mirrors ``server/runner/session.py`` exactly:
        ``discover(tier_filter="runner")`` -> set workspace / config_root /
        session context -> ``expose_all(plugin_configs)`` with **no**
        ``requested_plugins`` filter, so EVERY runner-tier plugin's
        ``initialize`` (+ auto-wiring, enrichment/lifecycle hooks) runs — not
        just the model-visible subset. Model tool-VISIBILITY is gated separately
        at ``create_session(plugins=...)`` (the ``session_kwargs['plugins']``
        list), identical to the daemon. This makes the initialized plugin set —
        and thus enrichment/lifecycle behavior — byte-for-byte the same in both
        transports; gating only at expose_all (the old behavior) skipped
        non-requested plugins' init, diverging on enrichment side-effects.
        """
        from shared import PluginRegistry

        registry = PluginRegistry(model_name=self._model)
        # tier_filter="runner": match the daemon SESSION's plugin tier (the
        # daemon-tier plugins — provider, GC, formatters — are runtime-level, not
        # session tools; the embedded JaatoClient wires the provider via connect).
        registry.discover(tier_filter="runner")
        if self._workspace_path:
            registry.set_workspace_path(self._workspace_path)
        if self._config_root:
            registry.set_config_root(self._config_root)
        # NOTE: set_session_id is NOT done here — the registry is runtime-level
        # (built once at connect, session-agnostic, like the daemon's template);
        # the per-session id is stamped in create_session before each session use.
        # NO requested_plugins -> initialize the FULL runner-tier set (plugin
        # parity). Model tool-visibility is gated at create_session, not here.
        registry.expose_all(self._resolved_plugin_configs or None)
        if self._workspace_path:
            # Post-init refresh fires set_workspace_path hooks on plugins that
            # rebuild derived state (mirrors session.py step 6-7).
            registry.set_workspace_path(self._workspace_path)
        return registry

    async def send_message(
        self,
        prompt: str,
        *,
        parallel_tools: Optional[bool] = None,
        attachments: Optional[list] = None,
        sources: Any = None,
    ) -> str:
        loop = self._loop
        emitter = self._emitter

        def on_output(source: str, text: str, mode: str) -> None:
            # Called on the worker thread — marshal onto the loop.
            loop.call_soon_threadsafe(
                emitter.emit,
                AgentOutputEvent(
                    agent_id="main", source=source, text=text, mode=mode
                ),
            )

        final_text = await asyncio.to_thread(
            self._embedded.send_message, prompt, on_output
        )
        # Drain the AGENT_OUTPUT callbacks queued from the worker thread before
        # the terminal, so the facade sees all output ahead of TURN_COMPLETED.
        await asyncio.sleep(0)
        emitter.emit(TurnCompletedEvent(agent_id="main"))
        return final_text

    def _on_lead_inject(self, text: str) -> None:
        """Continuation-callback entry (runs on the facade loop). Records any
        TERMINAL subagent inject so the completion-nudge gate has a stable
        'subagent done' signal, then schedules the follow-up lead turn.

        The subagent posts ``[SUBAGENT agent_id=<id> event=COMPLETED|ERROR|
        CANCELLED]`` when its task ends (vs ``event=IDLE`` / status mid-run).
        Marking the id here means a continuation triggered by a mid-run STATUS
        inject keeps the gate closed (subagent still outstanding), so the lead
        only completes after the subagent actually finishes."""
        import re

        for m in re.finditer(
            r"\[SUBAGENT agent_id=(\S+) event=(?:COMPLETED|ERROR|CANCELLED)\]", text
        ):
            self._completed_subagent_ids.add(m.group(1))
        self._schedule_lead_continuation(text)

    def _schedule_lead_continuation(self, text: str) -> None:
        """Schedule a follow-up lead turn for a drained subagent result.

        Runs on the facade loop (via call_soon_threadsafe from the continuation
        callback), so the busy/pending flags need no lock. Serializes
        continuations: while one runs, a newer arrival is coalesced into
        ``_continuation_pending`` and picked up when the current one settles —
        so overlapping injects never double-run a turn.
        """
        if self._continuation_busy:
            self._continuation_pending = text
            return
        self._continuation_busy = True
        asyncio.ensure_future(self._run_lead_continuation(text))

    async def _run_lead_continuation(self, text: str) -> None:
        """Run one continuation lead turn with the drained subagent result.

        Mirrors :meth:`send_message`'s output bridge (AGENT_OUTPUT marshalled
        onto the loop) + terminal TURN_COMPLETED. The embedded session's own
        ``send_message`` finally-drains any messages queued during this turn and
        re-fires the continuation callback, so the loop chains until the lead is
        idle with an empty queue (the lead's ``signal_completion`` ->
        AGENT_COMPLETED / SESSION_TERMINATED arrives via the wired ui_hooks).
        """
        emitter = self._emitter
        loop = self._loop

        def on_output(source: str, t: str, mode: str) -> None:
            loop.call_soon_threadsafe(
                emitter.emit,
                AgentOutputEvent(agent_id="main", source=source, text=t, mode=mode),
            )

        try:
            await asyncio.to_thread(self._embedded.send_message, text, on_output)
            await asyncio.sleep(0)
            emitter.emit(TurnCompletedEvent(agent_id="main"))
        finally:
            self._continuation_busy = False
            pending = self._continuation_pending
            self._continuation_pending = None
            if pending is None:
                # No queued inject left. If the lead has SETTLED (delegation
                # done, no subagent still running) but never called
                # signal_completion, drive the completion nudge — the embedded
                # analog of the daemon's lead nudge (server/core.py:4939) — so
                # the lead actually signals and SESSION_TERMINATED fires.
                pending = self._maybe_completion_nudge()
            if pending is not None:
                self._schedule_lead_continuation(pending)

    def _has_outstanding_subagents(self) -> bool:
        """True if any spawned subagent has NOT yet posted a terminal inject.

        Gates the lead completion nudge so it never fires while the lead is
        legitimately waiting for a delegated result. Keys on _active_sessions
        membership vs ``_completed_subagent_ids`` (terminal injects recorded in
        :meth:`_on_lead_inject`) — a STABLE 'still working' signal, unlike
        ``session.is_running`` which flickers mid-turn and reads False both
        pre-spawn and between the subagent's turns (the premature-completion
        bug). A subagent stays outstanding from spawn until its
        COMPLETED/ERROR/CANCELLED inject lands."""
        registry = self._registry
        sub = registry.get_plugin("subagent") if registry is not None else None
        active = getattr(sub, "_active_sessions", None) if sub is not None else None
        if not active:
            return False
        return any(aid not in self._completed_subagent_ids for aid in active)

    def _maybe_completion_nudge(self) -> Optional[str]:
        """Return the completion-nudge prompt when the lead has SETTLED without
        calling ``signal_completion`` — else None.

        The embedded analog of the daemon's lead completion nudge
        (server/core.py:4939). Nudge only when: signal_completion is on the
        lead's wire (a completion/delegation session — a plain chat session never
        has it, so this is a no-op there), the lead hasn't signaled, no subagent
        is still running, and the per-session nudge budget (bounded by
        ``try_completion_nudge``) isn't spent. Without this an embedded
        delegation lead resumes + produces its result but never calls
        signal_completion, so SESSION_TERMINATED never fires."""
        MAX_COMPLETION_NUDGES = 2
        lead = (
            self._embedded.get_session()
            if hasattr(self._embedded, "get_session")
            else None
        )
        if lead is None:
            return None
        if getattr(lead, "_signal_completion_called", False):
            return None
        tools = getattr(lead, "_tools", None) or []
        if not any(getattr(t, "name", None) == "signal_completion" for t in tools):
            return None
        if self._has_outstanding_subagents():
            return None
        try_nudge = getattr(lead, "try_completion_nudge", None)
        if try_nudge is None:
            return None
        should, _ = try_nudge(MAX_COMPLETION_NUDGES)
        if not should:
            return None
        return (
            "Your session is about to end without calling `signal_completion`. "
            "The loop cannot close cleanly until you either continue the work "
            "with another tool call, or call `signal_completion` per your "
            "profile's payload schema with the appropriate decision and "
            "evidence. Please proceed with one of those two paths."
        )

    async def respond_to_permission(self, request_id: str, response: str) -> None:
        # Resolve the parked permission request (set by the InProcessChannel on
        # the embedded runtime's worker thread). A no-op when nothing is pending
        # — e.g. before the channel is wired into the session (plugin-loading
        # slice) or for a stale/duplicate response.
        self._pending.resolve(request_id, response)

    async def disconnect(self) -> None:
        # Signal disconnect to any open event streams (mirrors the IPC drain
        # loop pushing the None sentinel), so their `async for` exits cleanly.
        for q in list(self._event_queues):
            try:
                q.put_nowait(None)
            except Exception:  # noqa: BLE001 — a full queue must not block teardown
                pass
        if self._embedded is not None and hasattr(self._embedded, "close_session"):
            await asyncio.to_thread(self._embedded.close_session)

    # ---- entry point ----
    @classmethod
    def session(cls, **kwargs: Any) -> "_InProcessSessionContext":
        """Open an embedded session.

        ``async with InProcessClient.session(model=..., profile=...) as s:``
        yields the same :class:`~jaato_sdk.client.convenience.Session` the IPC
        path yields, so ``await s.ask(...)`` / ``.complete`` / ``.stream`` work
        identically.
        """
        return _InProcessSessionContext(cls, kwargs)


class _InProcessSessionContext:
    """Async context manager mirroring ``IPCClient.session``'s ``_SessionContext``.

    Separates connection kwargs (forwarded to ``InProcessClient.__init__``)
    from ``create_session`` kwargs and ``on_permission`` (handed to the facade
    ``Session``), then connects, creates the session, and tears down on exit.
    """

    _CREATE_KEYS = ("agent", "agent_params", "cascade_driver_id")

    def __init__(self, client_cls: type, kwargs: Dict[str, Any]) -> None:
        self._client_cls = client_cls
        self._kwargs = dict(kwargs)
        self._client: Optional[InProcessClient] = None

    async def __aenter__(self) -> Session:
        on_permission = self._kwargs.pop("on_permission", None)
        # Host/client tools — popped from the spec (not an __init__ kwarg) and
        # registered after connect but BEFORE create_session, so the schema
        # reaches the session's initial wire (exactly like IPC's _SessionContext).
        client_tools = self._kwargs.pop("client_tools", None)
        # Resolve the session spec into connection kwargs so both clients take
        # an identical spec (the transport-agnostic entry forwards ``profile``
        # unchanged):
        #   * inline ``profile`` dict {model, provider, plugins, plugin_configs,
        #     system_instructions, completion_payload_schema} — the same shape
        #     ``IPCClient.session`` accepts;
        #   * named ``profile`` (str) — resolved from disk via
        #     ``discover_profiles`` (the embedded analog of the daemon's profile
        #     resolution, which an embedded session has no daemon to do).
        profile = self._kwargs.pop("profile", None)
        spec: Optional[Dict[str, Any]] = None
        if isinstance(profile, dict):
            spec = profile
        elif isinstance(profile, str):
            spec = _resolve_named_profile(profile, self._kwargs.get("config_root"))
        if spec:
            for key in (
                "model", "provider", "plugins", "plugin_configs",
                "system_instructions", "completion_payload_schema",
                "suppress_base_instructions",
            ):
                if spec.get(key) is not None and self._kwargs.get(key) is None:
                    self._kwargs[key] = spec[key]
        # Agent persona (.jaato/agents/<name>.md) — the embedded analog of the
        # daemon resolving ``--agent``. Layered as ``system_instructions`` (the
        # ``additional`` arg get_system_instructions composes onto the base, so
        # the persona sits on top of the framework baseline, exactly like the
        # daemon). Resolved here, before the client is built, so it reaches
        # create_session via __init__. A profile-supplied system_instructions
        # wins (the explicit override).
        agent_name = self._kwargs.get("agent")
        if agent_name and self._kwargs.get("system_instructions") is None:
            persona = _resolve_agent_persona(
                agent_name,
                self._kwargs.get("agent_params"),
                self._kwargs.get("workspace_path"),
                self._kwargs.get("config_root"),
            )
            self._kwargs["system_instructions"] = persona
        create_kwargs = {
            k: self._kwargs.pop(k) for k in self._CREATE_KEYS if k in self._kwargs
        }
        self._client = self._client_cls(**self._kwargs)
        await self._client.connect()
        if client_tools:
            await self._client.register_client_tools(client_tools)
        session_id = await self._client.create_session(**create_kwargs)
        return Session(self._client, session_id, on_permission)

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.disconnect()
