"""High-level convenience facade over the SDK session clients.

The facade is **client-agnostic**: the same :class:`Session` (``ask`` /
``complete`` / ``stream``) rides on every client that implements the small
session contract —

* :class:`IPCClient` and :class:`IPCRecoveryClient` — talk to a running daemon
  over IPC (the latter adds auto-reconnect);
* :class:`WSClient` and :class:`WSRecoveryClient` — talk to a (typically
  **remote**) daemon over WebSocket (``ws://`` / ``wss://``); the recovery
  variant adds auto-reconnect, mirroring the IPC pair. Needs the
  ``jaato-sdk[ws]`` extra;
* ``jaato.InProcessClient`` — runs the runtime **embedded** in your own process,
  no daemon / runner / socket.

The transport-agnostic ``jaato.session(mode="in_process"|"ipc"|"ws", ...)``
entry picks one, so the same facade code runs across transports with the mode
the only variable. The daemon transports (``ipc`` / ``ws``) also accept
``recovery=True`` to return the auto-reconnect client, plus ``ssl=`` / ``ca=``
on ``mode="ws"`` for self-signed / dev ``wss://`` certificates. The Python
``WSClient`` / ``WSRecoveryClient`` back the WS mode directly — Python WS access
is no longer raw frames.

The SDK's low-level surface is event-loop primitives (``subscribe`` /
``send_message`` / ``events``).  The common path — open a session, ask, get
the answer — costs ~10 lines of ``asyncio.Event`` + subscribe + ``done.wait()``
plumbing, and that recipe is subtle enough that the canonical scaffold template
once shipped an infinite hang (waiting on ``SESSION_TERMINATED`` only, which a
plain turn never emits — jaato PR #399).

This module owns that recipe so user code can't reproduce it (or its hangs):

    async with IPCClient.session(profile="researcher", agent="pirate") as s:
        print(await s.ask("Research tide pools."))

It is purely additive sugar over existing methods — every client method is
untouched, and both config styles are preserved because ``session(...)``
forwards ``create_session``'s parameters unchanged (``profile`` as a name =
declarative, as a dict = programmatic; ``agent`` composes with either).

See ``docs/design/sdk-convenience-layer.md`` and
``docs/design/in-process-facade.md``.
"""

from __future__ import annotations

import asyncio
from typing import (Any, AsyncIterator, Callable, Collection, Dict,
                    Optional)

from ..events import ClientType, EventType


#: Seconds ``Session.complete`` will wait for the ``AGENT_STATUS_CHANGED`` that
#: confirms or withdraws a turn's proposed terminus, before settling on the
#: turn anyway.
#:
#: The daemon emits that event from the same thread that emitted the turn
#: event, microseconds later, so this is never reached in a healthy session --
#: it exists for the one where the daemon stops talking mid-sequence (a lost
#: connection between a turn and its status event, or a daemon that dies
#: there).  Without it that session waits forever, and this module has shipped
#: an infinite hang before (PR #399).  Deliberately generous: tripping it early
#: costs the accuracy #767 bought back, so it should only ever fire when the
#: alternative is not returning at all.
SETTLE_GRACE = 10.0


class AgentError(Exception):
    """A turn ended in error (``SESSION_TERMINATED(reason="error")`` /
    ``AgentErrorEvent``).  Carries the daemon's ``error_type`` /
    ``error_summary`` so callers can branch without parsing strings."""

    def __init__(self, error_type: Optional[str], error_summary: Optional[str]):
        self.error_type = error_type
        self.error_summary = error_summary
        super().__init__(f"{error_type or 'AgentError'}: {error_summary or ''}".rstrip(": "))


class PermissionUnhandled(Exception):
    """A gated tool requested permission but no ``on_permission`` callback was
    supplied to :meth:`IPCClient.session`.  The facade auto-denied (to unstick
    the daemon) and raised this rather than hang or silently degrade.  Pass
    ``on_permission=`` to ``session(...)`` or drop to the low-level API."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(
            f"tool {tool_name!r} requested permission but no on_permission "
            f"callback was set — pass on_permission= to session(...) or use "
            f"the low-level subscribe(EventType.PERMISSION_REQUESTED) API"
        )


class Session:
    """High-level handle over an open ``IPCClient`` session.

    Owns the send-and-wait recipe (first-of ``{TURN_COMPLETED,
    SESSION_TERMINATED}``) so a turn can never hang in user code.  Construct
    via :meth:`IPCClient.session`, not directly.
    """

    def __init__(self, client: Any, session_id: str, on_permission=None):
        self._client = client
        self.session_id = session_id
        self._on_permission = on_permission
        self._unhandled_perm: Optional[str] = None
        self._perm_callback_error: Optional[Exception] = None
        # Wire permissions once for the session's lifetime — fail loud, never
        # hang (the second hang trap after #399).
        client.subscribe(
            EventType.PERMISSION_REQUESTED,
            lambda ev: asyncio.ensure_future(self._on_perm(ev)),
        )

    @property
    def client(self):
        """The underlying low-level client (``IPCClient`` / ``IPCRecoveryClient``).

        The facade is purely additive — drop to the full event API on the SAME
        connection when you need it (custom ``subscribe``/``events`` routing,
        ``respond_to_permission(edited_arguments=...)``, ``cascade_events``,
        ``attach_session``, ...) while still using ``ask``/``complete``/``stream``
        for the common turns::

            async with IPCClient.session(profile=...) as s:
                s.client.subscribe(EventType.TOOL_CALL_END, observer)  # low-level
                print(await s.ask("..."))                              # facade

        ``ask``/``complete``/``stream`` clean up only their own subscriptions, so
        listeners you add via ``s.client`` are independent and persist across
        turns.
        """
        return self._client

    async def _on_perm(self, ev: Any) -> None:
        request_id = getattr(ev, "request_id", "")
        resp: Any = "n"
        if self._on_permission is not None:
            try:
                resp = self._on_permission(ev)
                if asyncio.iscoroutine(resp):
                    resp = await resp
            except Exception as exc:  # never let a user callback hang the turn
                # A raising on_permission must NOT deadlock the turn — the
                # in-process channel blocks a worker thread on the response with
                # no transport timeout, so a response is mandatory. Deny to
                # unstick, and surface the callback's error on the in-flight
                # ask()/complete() instead of hanging.
                self._perm_callback_error = exc
                resp = "n"
        else:
            # No policy to apply — deny to unstick, flag so the in-flight
            # ask()/complete() raises PermissionUnhandled.
            self._unhandled_perm = getattr(ev, "tool_name", "?")
        # ALWAYS respond — guaranteed even on a raising callback (the unconditional
        # await below is the no-deadlock invariant).
        await self._client.respond_to_permission(request_id, resp or "n")

    def _raise_if_needed(self, box: Dict[str, Any]) -> None:
        if self._perm_callback_error is not None:
            exc = self._perm_callback_error
            self._perm_callback_error = None
            raise exc
        if self._unhandled_perm is not None:
            tool = self._unhandled_perm
            self._unhandled_perm = None
            raise PermissionUnhandled(tool)
        if box.get("reason") == "error":
            raise AgentError(box.get("error_type"), box.get("error_summary"))

    @staticmethod
    def _note_terminal(box: Dict[str, Any], ev: Any) -> None:
        # first-of {TURN_COMPLETED, SESSION_TERMINATED}: TURN_COMPLETED carries
        # no reason (-> "natural"); SESSION_TERMINATED carries the rich
        # reason/error.  setdefault lets a real error reason win on a race.
        box.setdefault("reason", getattr(ev, "reason", None) or "natural")
        if getattr(ev, "error_type", None):
            box["error_type"] = ev.error_type
        if getattr(ev, "error_summary", None):
            box["error_summary"] = ev.error_summary

    def _subscribe_media(self, on_media) -> Any:
        """Route this session's MODEL SPEECH to ``on_media``; return unsubscribe.

        Symmetric with the text these methods already hand back: ``ask``
        returns what the model wrote, ``on_media`` delivers what it said.
        Audio gets a sink rather than a return value because it streams --
        accumulating a whole utterance before returning would defeat
        playing it as it arrives, and would hold an unbounded buffer.

        MODEL speech only.  A tool's attachments ride the same event but
        belong to that tool call, and a caller after those is not asking
        about this session's answer; ``client.subscribe`` on
        ``TOOL_OUTPUT`` still reaches them.

        Returns a no-op unsubscribe when ``on_media`` is None, so callers
        wanting no media pay nothing and the three call sites stay flat.
        """
        if on_media is None:
            return lambda: None

        def _fan(ev: Any) -> None:
            if ev.is_model_speech():
                on_media(ev)

        return self._client.subscribe(EventType.TOOL_OUTPUT, _fan)

    async def ask(self, prompt: str, *,
                  sources: Optional[Collection[str]] = ("model",),
                  parallel_tools: Optional[bool] = None,
                  attachments: Optional[list] = None,
                  on_media: Optional[Callable[[Any], None]] = None) -> str:
        """Send ``prompt``, wait for the turn to finish, return collected text.

        Waits on first-of ``{TURN_COMPLETED, SESSION_TERMINATED}`` so a plain
        turn (only ``TURN_COMPLETED``) never hangs.  ``sources`` selects which
        ``AGENT_OUTPUT`` chunks to keep by ``.source`` — default ``("model",)``
        (a clean answer); ``None`` collects everything.  Raises
        :class:`AgentError` on an error terminal, :class:`PermissionUnhandled`
        if a gated tool went unanswered.  ``on_media`` receives the model's
        own SPEECH as it streams (see :meth:`_subscribe_media`) — the text
        comes back, the audio is handed over.
        """
        chunks: list[str] = []
        box: Dict[str, Any] = {}
        done = asyncio.Event()

        def on_output(ev: Any) -> None:
            if sources is None or getattr(ev, "source", None) in sources:
                text = getattr(ev, "text", "")
                if text:
                    chunks.append(text)

        def on_terminal(ev: Any) -> None:
            self._note_terminal(box, ev)
            done.set()

        unsub_out = self._client.subscribe(EventType.AGENT_OUTPUT, on_output)
        unsub_media = self._subscribe_media(on_media)
        unsub_term = self._client.subscribe_once(EventType.SESSION_TERMINATED, on_terminal)
        unsub_turn = self._client.subscribe_once(EventType.TURN_COMPLETED, on_terminal)
        try:
            await self._client.send_message(
                prompt, parallel_tools=parallel_tools, attachments=attachments)
            await done.wait()
        finally:
            unsub_out()
            unsub_media()
            unsub_term()
            unsub_turn()
        self._raise_if_needed(box)
        return "".join(chunks)

    async def complete(self, prompt: str, *,
                       parallel_tools: Optional[bool] = None,
                       attachments: Optional[list] = None,
                       on_media: Optional[Callable[[Any], None]] = None,
                       ) -> Optional[Dict[str, Any]]:
        """Send ``prompt`` and return the typed completion ``payload``.

        For completion-gated profiles: captures ``AGENT_COMPLETED.payload``
        (emitted before the terminal), and returns when the SESSION settles --
        not when its first turn ends.  Returns the payload (``None`` if the
        profile declared no schema or the model didn't complete).  Raises
        :class:`AgentError` on an error terminal.  ``on_media`` receives the
        model's own SPEECH as it streams — a completion-gated stage that
        answers OUT LOUD returns its payload here and its audio there.

        WHY A TURN BOUNDARY IS NOT THE SESSION'S TERMINUS.

        This waited on first-of ``{SESSION_TERMINATED, TURN_COMPLETED}``, and
        for a completion-gated profile that is the WRONG event.  Such a
        session ends at ``signal_completion``; when the model instead ends its
        loop in prose, the daemon RE-PROMPTS it (``COMPLETION_NUDGE``) and the
        session keeps working -- so ``TURN_COMPLETED`` fired while the agent
        had minutes of work left.  A caller that graded on the return value
        graded a half-finished tree, and did so silently and in both
        directions: it can read a FAIL from a defect the next turn fixes, or a
        PASS from a tree the next turn regresses (jaato #767, where an eval
        arm was graded 19s before the agent's first commit and recorded FAIL
        on a tree that compiles).

        THE RULE.  A ``TURN_COMPLETED`` is the terminus only if the agent
        SETTLES on it.  The daemon closes every turn of the main agent with
        exactly one ``AGENT_STATUS_CHANGED`` for that agent: ``"active"`` when
        it is starting another turn (a nudge, a stashed continuation, a
        drained user send), ``"idle"``/``"done"`` when it is not.  So the turn
        event proposes a terminus and the status event confirms or withdraws
        it.  ``SESSION_TERMINATED`` remains an unconditional terminus -- it is
        what arrives when the nudge budget is spent.

        WHICH AGENT.  Only the agent this send started, latched from the
        ``AGENT_STATUS_CHANGED(status="active")`` the daemon emits as
        ``send_message`` begins.  Subagent turns therefore no longer settle
        the parent's call, which they did before -- any agent's
        ``TURN_COMPLETED`` ended the wait.

        NO HANG, ON EITHER SIDE OF THE LATCH.  Waiting on a confirmation
        introduces two ways for one never to arrive, and both fall back to the
        pre-#767 behaviour -- settle on the turn:

        * *Never latched.*  The opening status event does not arrive (an older
          daemon, or a connection whose cascade registration filters the
          type), so nothing is latched and the first ``TURN_COMPLETED``
          settles the call immediately, exactly as it used to.
        * *Latched, then unconfirmed.*  A turn proposes a terminus and neither
          the status event nor a terminal follows -- the daemon stopped
          talking mid-sequence.  The proposal is settled after
          ``SETTLE_GRACE``.  The first version of this fix had no such bound
          and hung here forever, which is the defect class this module already
          carries a scar from (PR #399, and ``ask``'s own "so a turn can never
          hang in user code").

        So version skew or a lost daemon costs accuracy, never a wait with no
        end.

        ``ask``/``stream`` keep turn semantics deliberately: their contract is
        one turn's output, and an interactive session must hand back each turn
        as it lands.
        """
        box: Dict[str, Any] = {}
        done = asyncio.Event()
        # Anything a handler changes wakes the wait below, which re-reads
        # ``state`` rather than trusting the event that woke it.
        wake = asyncio.Event()
        # The agent whose turns can settle this call, and the turn of its that
        # is currently PROPOSING a terminus (awaiting that agent's next status
        # event).  ``None`` = nothing proposed, so the wait is unbounded.
        state: Dict[str, Any] = {"agent": None, "proposal": None}

        def on_completed(ev: Any) -> None:
            box["payload"] = getattr(ev, "payload", None)

        def settle(ev: Any) -> None:
            self._note_terminal(box, ev)
            done.set()
            wake.set()

        def on_turn(ev: Any) -> None:
            agent = getattr(ev, "agent_id", None)
            if state["agent"] is None:
                # Never announced active -> old-daemon fallback: this turn IS
                # the terminus, as it was before #767.
                settle(ev)
                return
            if agent == state["agent"]:
                state["proposal"] = ev
                wake.set()

        def on_status(ev: Any) -> None:
            agent = getattr(ev, "agent_id", None)
            status = getattr(ev, "status", None)
            if state["agent"] is None:
                if status == "active":
                    state["agent"] = agent
                return
            if agent != state["agent"] or state["proposal"] is None:
                return
            if status == "active":
                state["proposal"] = None    # another turn is starting
                wake.set()
            else:
                settle(ev)                  # the agent settled on that turn

        unsub_media = self._subscribe_media(on_media)
        unsub_comp = self._client.subscribe_once(EventType.AGENT_COMPLETED, on_completed)
        unsub_term = self._client.subscribe_once(EventType.SESSION_TERMINATED, settle)
        # NOT ``subscribe_once``: a nudged session runs several turns and the
        # call must see every one of them to know which is the last.
        unsub_turn = self._client.subscribe(EventType.TURN_COMPLETED, on_turn)
        unsub_status = self._client.subscribe(
            EventType.AGENT_STATUS_CHANGED, on_status)
        try:
            await self._client.send_message(
                prompt, parallel_tools=parallel_tools, attachments=attachments)
            while not done.is_set():
                # Clear BEFORE reading the state the handlers write.  Sync
                # handlers run inline on this loop, so they cannot interleave
                # between the clear and the await -- whatever they had already
                # recorded is visible here, and anything later re-sets ``wake``.
                wake.clear()
                proposal = state["proposal"]
                if proposal is None:
                    await wake.wait()       # nothing proposed: wait as before
                    continue
                try:
                    await asyncio.wait_for(wake.wait(), timeout=SETTLE_GRACE)
                except asyncio.TimeoutError:
                    # The confirmation never came.  Settle on what the turn
                    # proposed rather than wait on a daemon that has stopped
                    # talking.
                    if state["proposal"] is proposal and not done.is_set():
                        settle(proposal)
        finally:
            unsub_media()
            unsub_comp()
            unsub_term()
            unsub_turn()
            unsub_status()
        self._raise_if_needed(box)
        return box.get("payload")

    async def stream(self, prompt: str, *,
                     sources: Optional[Collection[str]] = ("model",),
                     parallel_tools: Optional[bool] = None,
                     attachments: Optional[list] = None,
                     on_media: Optional[Callable[[Any], None]] = None,
                     ) -> AsyncIterator[str]:
        """Send ``prompt`` and yield text chunks live as they arrive.

        Async-iterator counterpart to :meth:`ask` — yields each
        ``AGENT_OUTPUT`` chunk (filtered by ``sources``; ``None`` = all) the
        moment it streams in, then stops at the terminal (first-of
        ``{TURN_COMPLETED, SESSION_TERMINATED}``).  ``TURN_COMPLETED`` fires
        after all of the turn's output, so no chunk is dropped.  Raises
        :class:`AgentError` on an error terminal / :class:`PermissionUnhandled`
        on an unanswered gated tool, after the stream drains.  ``on_media``
        receives the model's own SPEECH as it streams, alongside the text
        this yields::

            async with IPCClient.session(profile=...) as s:
                async for chunk in s.stream("Tell me a story."):
                    print(chunk, end="", flush=True)
        """
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        box: Dict[str, Any] = {}
        sentinel = object()

        def on_output(ev: Any) -> None:
            if sources is None or getattr(ev, "source", None) in sources:
                text = getattr(ev, "text", "")
                if text:
                    queue.put_nowait(text)

        def on_terminal(ev: Any) -> None:
            self._note_terminal(box, ev)
            queue.put_nowait(sentinel)

        unsub_out = self._client.subscribe(EventType.AGENT_OUTPUT, on_output)
        unsub_media = self._subscribe_media(on_media)
        unsub_term = self._client.subscribe_once(EventType.SESSION_TERMINATED, on_terminal)
        unsub_turn = self._client.subscribe_once(EventType.TURN_COMPLETED, on_terminal)
        try:
            await self._client.send_message(
                prompt, parallel_tools=parallel_tools, attachments=attachments)
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            unsub_out()
            unsub_media()
            unsub_term()
            unsub_turn()
        self._raise_if_needed(box)


class _SessionContext:
    """Async context manager returned by :meth:`IPCClient.session`.

    Connects + creates the session on ``__aenter__`` (failing loud rather than
    yielding a dead session) and disconnects on ``__aexit__``.
    """

    def __init__(self, client: Any, create_kwargs: Dict[str, Any],
                 on_permission, connect_timeout: float, client_tools=None):
        self._client = client
        self._create_kwargs = create_kwargs
        self._on_permission = on_permission
        self._connect_timeout = connect_timeout
        self._client_tools = client_tools

    async def __aenter__(self) -> Session:
        if not await self._client.connect(timeout=self._connect_timeout):
            raise ConnectionError(
                "could not connect to / autostart the jaato daemon — "
                "run `python -m jaato_sdk.doctor`"
            )
        # Host/client tools MUST be registered AFTER connect but BEFORE
        # create_session — the runner-tier model only sees tools registered
        # before the session exists (mid-session registration isn't picked up
        # until a later turn).  This is the hook that lets host-tool clients use
        # the facade instead of dropping to the low-level connect/register/create
        # dance.  See register_client_tools().
        if self._client_tools:
            await self._client.register_client_tools(self._client_tools)
        # No ``if not sid`` guard: ``create_session`` raises now, and its
        # exception STATES the cause.  What stood here guessed one --
        # "check provider auth" -- which is one of five causes and wrong for
        # the other four: a failed socket write, a refusal for an unknown
        # profile, a timeout, a dropped connection.  A hardcoded likely-cause
        # sends the reader to the wrong place with full confidence, which is
        # worse than saying nothing.
        #
        # The disconnect stays, and moves into a ``finally``-shaped guard: it
        # used to run only on the ``not sid`` path, so an EXCEPTION out of
        # create_session (already possible today -- the recovery client raises
        # ReconnectingError / ConnectionClosedError from _check_can_send)
        # leaked the connection.
        try:
            sid = await self._client.create_session(**self._create_kwargs)
        except BaseException:
            await self._client.disconnect()
            raise
        return Session(self._client, sid, self._on_permission)

    async def __aexit__(self, *exc: Any) -> bool:
        await self._client.disconnect()
        return False


def open_session(client_cls, *, profile=None, agent=None, agent_params=None,
                 cascade_driver_id=None, on_permission=None, client_tools=None,
                 socket_path=None, env_file: str = ".env",
                 workspace_path=None, auto_start: bool = True,
                 client_type: ClientType = ClientType.API,
                 connect_timeout: float = 120.0,
                 config_root=None, apparmor=None,
                 on_status_change=None, presentation=None,
                 min_protocol_version=None) -> _SessionContext:
    """Build a client of ``client_cls`` and return a session context manager.

    Backs :meth:`IPCClient.session` and :meth:`IPCRecoveryClient.session`.
    ``profile`` / ``agent`` / ``agent_params`` / ``cascade_driver_id`` are
    forwarded to ``create_session`` unchanged, so both the declarative (named)
    and programmatic (inline-dict) styles work.  ``client_tools`` (a list of
    host-tool specs, same shape as :meth:`register_client_tools`) is registered
    after connect but before create_session, so host-tool clients can use the
    facade instead of the low-level connect/register/create dance.
    ``config_root`` (read-only-config root override) and ``apparmor`` (opt-in
    per-session AppArmor confinement) are accepted by BOTH client classes.
    They were ``IPCClient``-only until 2026-08-23, and this docstring used to
    record that as deliberate — it was not.  ``on_status_change`` is the only
    genuinely asymmetric arg: it is an ``IPCRecoveryClient``-only
    reconnection-status callback, meaningless on a plain ``IPCClient``.
    Each is forwarded to the constructor only when set — including
    ``min_protocol_version``, the wire protocol this caller requires, so
    adopting the facade never quietly drops a compatibility guarantee the
    plain constructor offers.
    """
    ctor_kwargs = dict(client_type=client_type, auto_start=auto_start,
                       env_file=env_file, workspace_path=workspace_path,
                       presentation=presentation)
    # Accepted by BOTH client classes since 2026-08-23; forwarded only when
    # set so the constructors keep their own defaults.  The previous comment
    # here claimed they were IPCClient-only "by design, same pattern as
    # on_status_change" — a false analogy that made an accident look
    # deliberate.  on_status_change IS recovery-only (a reconnection concept);
    # config_root/apparmor are ordinary session bootstrap config that goes
    # straight into ClientConfigRequest, and withholding them from recovery
    # clients left confinement SILENTLY incomplete for anything that had to
    # survive a daemon restart.
    if config_root is not None:
        ctor_kwargs["config_root"] = config_root
    if apparmor is not None:
        ctor_kwargs["apparmor"] = apparmor
    if on_status_change is not None:
        ctor_kwargs["on_status_change"] = on_status_change
    # The wire protocol this caller REQUIRES.  Without it the facade was a
    # silent downgrade: a client that adopted the sugar could no longer say
    # which protocol it depends on, so an older daemon simply never sent the
    # newer fields and the caller saw an empty result rather than a refusal
    # naming the version.  A convenience layer must not cost a guarantee the
    # low-level constructor offers.
    if min_protocol_version is not None:
        ctor_kwargs["min_protocol_version"] = min_protocol_version
    client = (client_cls(socket_path, **ctor_kwargs) if socket_path is not None
              else client_cls(**ctor_kwargs))
    create_kwargs = dict(profile=profile, agent=agent, agent_params=agent_params,
                         cascade_driver_id=cascade_driver_id)
    return _SessionContext(client, create_kwargs, on_permission, connect_timeout,
                           client_tools=client_tools)


async def ask(prompt: str, *, sources: Optional[Collection[str]] = ("model",),
              **session_kwargs: Any) -> str:
    """One-shot: open a session, ask, return the answer, tear down.

    Sugar over ``IPCClient.session`` — accepts the same kwargs (``profile``,
    ``agent``, connection knobs, ``on_permission``).  One daemon connect per
    call; for repeated calls use the ``IPCClient.session`` context manager.
    """
    from .ipc import IPCClient
    async with IPCClient.session(**session_kwargs) as s:
        return await s.ask(prompt, sources=sources)
