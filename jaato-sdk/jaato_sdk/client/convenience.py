"""High-level convenience facade over :class:`IPCClient`.

The SDK's low-level surface is event-loop primitives (``subscribe`` /
``send_message`` / ``events``).  The common path — open a session, ask, get
the answer — costs ~10 lines of ``asyncio.Event`` + subscribe + ``done.wait()``
plumbing, and that recipe is subtle enough that the canonical scaffold template
once shipped an infinite hang (waiting on ``SESSION_TERMINATED`` only, which a
plain turn never emits — jaato PR #399).

This module owns that recipe so user code can't reproduce it (or its hangs):

    async with IPCClient.session(profile="researcher", agent="pirate") as s:
        print(await s.ask("Research tide pools."))

It is purely additive sugar over existing methods — every ``IPCClient`` method
is untouched, and both config styles are preserved because ``session(...)``
forwards ``create_session``'s parameters unchanged (``profile`` as a name =
declarative, as a dict = programmatic; ``agent`` composes with either).

See ``docs/design/sdk-convenience-layer.md``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Collection, Dict, Optional

from ..events import ClientType, EventType


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
        # Wire permissions once for the session's lifetime — fail loud, never
        # hang (the second hang trap after #399).
        client.subscribe(
            EventType.PERMISSION_REQUESTED,
            lambda ev: asyncio.ensure_future(self._on_perm(ev)),
        )

    async def _on_perm(self, ev: Any) -> None:
        request_id = getattr(ev, "request_id", "")
        if self._on_permission is not None:
            resp = self._on_permission(ev)
            if asyncio.iscoroutine(resp):
                resp = await resp
            await self._client.respond_to_permission(request_id, resp or "n")
        else:
            # No policy to apply — deny to unstick the daemon, flag so the
            # in-flight ask()/complete() raises PermissionUnhandled.
            self._unhandled_perm = getattr(ev, "tool_name", "?")
            await self._client.respond_to_permission(request_id, "n")

    def _raise_if_needed(self, box: Dict[str, Any]) -> None:
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

    async def ask(self, prompt: str, *,
                  sources: Optional[Collection[str]] = ("model",)) -> str:
        """Send ``prompt``, wait for the turn to finish, return collected text.

        Waits on first-of ``{TURN_COMPLETED, SESSION_TERMINATED}`` so a plain
        turn (only ``TURN_COMPLETED``) never hangs.  ``sources`` selects which
        ``AGENT_OUTPUT`` chunks to keep by ``.source`` — default ``("model",)``
        (a clean answer); ``None`` collects everything.  Raises
        :class:`AgentError` on an error terminal, :class:`PermissionUnhandled`
        if a gated tool went unanswered.
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
        unsub_term = self._client.subscribe_once(EventType.SESSION_TERMINATED, on_terminal)
        unsub_turn = self._client.subscribe_once(EventType.TURN_COMPLETED, on_terminal)
        try:
            await self._client.send_message(prompt)
            await done.wait()
        finally:
            unsub_out()
            unsub_term()
            unsub_turn()
        self._raise_if_needed(box)
        return "".join(chunks)

    async def complete(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Send ``prompt`` and return the typed completion ``payload``.

        For completion-gated profiles: captures ``AGENT_COMPLETED.payload``
        (emitted before the terminal), waits on first-of ``{SESSION_TERMINATED,
        TURN_COMPLETED}``.  Returns the payload (``None`` if the profile
        declared no schema or the model didn't complete).  Raises
        :class:`AgentError` on an error terminal.
        """
        box: Dict[str, Any] = {}
        done = asyncio.Event()

        def on_completed(ev: Any) -> None:
            box["payload"] = getattr(ev, "payload", None)

        def on_terminal(ev: Any) -> None:
            self._note_terminal(box, ev)
            done.set()

        unsub_comp = self._client.subscribe_once(EventType.AGENT_COMPLETED, on_completed)
        unsub_term = self._client.subscribe_once(EventType.SESSION_TERMINATED, on_terminal)
        unsub_turn = self._client.subscribe_once(EventType.TURN_COMPLETED, on_terminal)
        try:
            await self._client.send_message(prompt)
            await done.wait()
        finally:
            unsub_comp()
            unsub_term()
            unsub_turn()
        self._raise_if_needed(box)
        return box.get("payload")


class _SessionContext:
    """Async context manager returned by :meth:`IPCClient.session`.

    Connects + creates the session on ``__aenter__`` (failing loud rather than
    yielding a dead session) and disconnects on ``__aexit__``.
    """

    def __init__(self, client: Any, create_kwargs: Dict[str, Any],
                 on_permission, connect_timeout: float):
        self._client = client
        self._create_kwargs = create_kwargs
        self._on_permission = on_permission
        self._connect_timeout = connect_timeout

    async def __aenter__(self) -> Session:
        if not await self._client.connect(timeout=self._connect_timeout):
            raise ConnectionError(
                "could not connect to / autostart the jaato daemon — "
                "run `python -m jaato_sdk.doctor`"
            )
        sid = await self._client.create_session(**self._create_kwargs)
        if not sid:
            await self._client.disconnect()
            raise RuntimeError(
                "session.new failed — check provider auth / the daemon log"
            )
        return Session(self._client, sid, self._on_permission)

    async def __aexit__(self, *exc: Any) -> bool:
        await self._client.disconnect()
        return False


def open_session(client_cls, *, profile=None, agent=None, agent_params=None,
                 cascade_driver_id=None, on_permission=None,
                 socket_path=None, env_file: str = ".env",
                 workspace_path=None, auto_start: bool = True,
                 client_type: ClientType = ClientType.API,
                 connect_timeout: float = 120.0) -> _SessionContext:
    """Build a client of ``client_cls`` and return a session context manager.

    Backs :meth:`IPCClient.session`.  ``profile`` / ``agent`` / ``agent_params``
    / ``cascade_driver_id`` are forwarded to ``create_session`` unchanged, so
    both the declarative (named) and programmatic (inline-dict) styles work.
    """
    ctor_kwargs = dict(client_type=client_type, auto_start=auto_start,
                       env_file=env_file, workspace_path=workspace_path)
    client = (client_cls(socket_path, **ctor_kwargs) if socket_path is not None
              else client_cls(**ctor_kwargs))
    create_kwargs = dict(profile=profile, agent=agent, agent_params=agent_params,
                         cascade_driver_id=cascade_driver_id)
    return _SessionContext(client, create_kwargs, on_permission, connect_timeout)


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
