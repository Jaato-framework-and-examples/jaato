"""``app://`` secret resolution at spawn, by the owning application (#1226).

A per-user credential — a WUI user's GitHub token — cannot reach the sessions
that run in that user's workspaces: cascade stages, ``session.wake``,
reactor-spawned and revived sessions run without a browser, and the workspace
``.env`` cannot hold a rotating 8-hour token in plaintext at rest.  So the
``.env`` (or profile ``env:``) holds a REFERENCE — ``GH_TOKEN=app://github`` —
and the daemon resolves it at every spawn by asking the application that OWNS
the workspace, over the same authenticated bind channel #1074's ticket verbs
ride (``ticket.bind`` / ``ticket.revoke``, application -> daemon).
``secret.resolve`` is the one verb that runs the other way.

This module is the daemon-side resolver.  It holds NO transport of its own:
the two things it needs — mapping a workspace path to its qualified owner, and
sending ``secret.resolve`` to a named application — are injected as callables
by the daemon wiring (``server/__main__.py``), because both live on
``JaatoWSServer``.  IPC has its own resolver keyed on the peer uid
(:mod:`server.ipc_app_secret`, #1609), chained after this one by
:class:`ChainedAppSecretResolver`.  On the embedded / standalone paths no
resolver is injected and an ``app://`` reference is dropped (or, strict,
refuses the bootstrap) exactly as an unreachable application would be.

The rules this holds to (design §6.1):

* **Resolve for the workspace OWNER, not the session creator.**  A headless
  cascade stage may carry no ``created_by``; a revived session's creator is a
  record, not a connection.  "That WUI user's workspaces" IS the owner
  relation, so the owner is what is asked about.  An UNOWNED workspace
  resolves nothing.
* **Ask only the application the owner is qualified under.**  ``acme:alice``
  is asked of ``acme`` and never of ``other-app``.  The application id comes
  from the qualified owner, and the transport picks the connection that
  authenticated as that application — so a request can never reach an
  application it does not name.
* **An unresolved reference is DROPPED, never forwarded as a literal.**  A
  literal ``app://github`` in a subprocess is a token that fails with a
  confusing 401; a dropped variable makes ``gh`` report "not logged in", the
  true state.  This module returns a non-``ok`` :class:`AppSecretAnswer`
  rather than the reference; the DROP itself is done by the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from jaato_server.shared.plugins.subagent.config import (
    APP_SECRET_SCHEME,
    AppSecretReference,
    SecretResolutionError,
    SecretResolveContext,
)

logger = logging.getLogger(__name__)


class AppSecretResolutionError(Exception):
    """A ``app://<name>?required`` reference could not be resolved (#1226).

    Raised only for the STRICT form.  A plain ``app://<name>`` that fails is
    dropped with a WARNING and the session starts without the variable; the
    strict form turns the same failure into a bootstrap refusal, so a session
    that *must* have the credential does not come up silently lacking it.
    """


#: Default deadline for one ``secret.resolve`` round trip, in seconds.  The
#: application mints (or reuses) a token and answers; a request that does not
#: come back inside this window is treated as unreachable and the reference is
#: dropped.  Held here rather than as an env knob because there is one
#: transport and one deadline for it.
DEFAULT_RESOLVE_TIMEOUT_SECONDS = 10.0


#: What each non-``ok`` / ``ok`` status means to a session, rendered by
#: ``jaato-scaffold explain secrets``.  Every status but ``ok`` DROPS the
#: reference (or, for ``app://<name>?required``, refuses the bootstrap).
ANSWER_STATUSES = (
    ("ok", "the value replaces the reference in this session's env only"),
    ("unowned", "the workspace has no owner any transport can name"),
    ("not_found", "the resolver has no binding for this (user, workspace, name)"),
    ("denied", "the resolver refuses to answer"),
    ("error", "the resolver tried and failed, or answered ok with no value"),
    ("unreachable", "no live resolver connection, or no answer within the deadline"),
    ("no_transport", "no resolver is wired on this daemon at all"),
)

#: The rules every transport's resolution holds to (#1226), rendered by
#: ``jaato-scaffold explain secrets``.
RESOLUTION_RULES = (
    "resolved at EVERY spawn (create, cascade stage, wake, revive) by the "
    "daemon, for the workspace OWNER, never the session creator",
    "the value reaches the session's env only; the .env, the record and the "
    "snapshot keep `app://<name>`, so the next spawn asks again",
    "an unresolved reference is DROPPED, never forwarded as the literal; "
    "`app://<name>?required` refuses the bootstrap instead",
    "an `expires_at` in the answer schedules a re-resolution a margin "
    "before it (JAATO_OAUTH_REFRESH_MARGIN)",
    "a resolved name is granted past the `cli` / `interactive_shell` secret "
    "scrub (#1228); MCP servers never receive it",
    "a workspace an application owns over WebSocket is resolved by that "
    "application; an IPC account answers only for workspaces no "
    "application owns",
)


@dataclass(frozen=True)
class AppSecretAnswer:
    """The outcome of resolving one ``app://`` reference (#1226).

    ``status`` is the machine-readable verdict; only ``ok`` carries a
    ``value``.  This is the resolver's richer return — the ``SecretResolver``
    protocol's ``resolve`` returns a bare string, which cannot carry
    ``expires_at`` (needed to schedule the pre-expiry reload), so the
    session-env pass calls :meth:`AppSecretResolver.resolve_reference` and
    reads this.

    Attributes:
        status: ``ok`` (``value`` present) / ``unowned`` (the workspace has no
            owner, or an owner that is not ``app:user``-qualified) /
            ``not_found`` / ``denied`` / ``error`` / ``unreachable`` /
            ``no_transport`` (no bind channel wired at all).  Every non-``ok``
            status drops the reference.
        value: The resolved secret, present only when ``status == "ok"``.
        expires_at: ISO-8601 UTC instant the value stops being valid, when the
            application reported one; ``None`` means no expiry known.
        detail: Human-readable elaboration for the WARNING the caller logs on a
            non-``ok`` status.  Never the secret.
    """

    status: str
    value: Optional[str] = None
    expires_at: Optional[str] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.value is not None


#: A transport that sends ``secret.resolve`` to one application and awaits its
#: answer.  ``(app_id, user, workspace, name, timeout) -> AppSecretAnswer``.
#: Never raises for a refusal or an unreachable application — those are
#: statuses; it may raise only for a programming error, which the resolver
#: wraps.
AppSecretTransport = Callable[[str, str, str, str, float], AppSecretAnswer]

#: Maps an absolute workspace path to its qualified owner ``"app:user"``, or
#: ``None`` for an unowned / unknown workspace.
WorkspaceOwnerLookup = Callable[[str], Optional[str]]


class AppSecretResolver:
    """Resolves ``app://<name>`` references against the owning application.

    In-tree (part of the ticket mechanism, not a premium backend) and injected
    into each :class:`JaatoServer` by the daemon wiring.  Implements the
    ``SecretResolver`` protocol so the interface change #1226 describes is real
    and testable, but it is NOT registered in the entry-point resolver registry
    — ``_resolve_secret_uri`` defers the ``app`` scheme, and
    ``JaatoServer._resolve_session_env`` calls this resolver directly with the
    owner context.
    """

    schemes = frozenset({APP_SECRET_SCHEME})

    def __init__(
        self,
        owner_of: WorkspaceOwnerLookup,
        transport: AppSecretTransport,
        *,
        timeout: float = DEFAULT_RESOLVE_TIMEOUT_SECONDS,
    ) -> None:
        self._owner_of = owner_of
        self._transport = transport
        self._timeout = timeout

    def owner_for(self, workspace_path: Optional[str]) -> Optional[str]:
        """The qualified owner of ``workspace_path``, or ``None``.

        Used by ``_resolve_session_env`` to build the
        :class:`SecretResolveContext` before resolving, so the owner is
        carried in the context exactly as the design describes.
        """
        if not workspace_path:
            return None
        try:
            return self._owner_of(workspace_path)
        except Exception:  # noqa: BLE001 -- a lookup failure is "unowned"
            logger.warning(
                "app:// owner lookup for workspace %r raised; treating as "
                "unowned", workspace_path, exc_info=True,
            )
            return None

    def resolve_reference(
        self,
        ref: AppSecretReference,
        context: Optional[SecretResolveContext],
        *,
        timeout: Optional[float] = None,
    ) -> AppSecretAnswer:
        """Resolve one parsed ``app://`` reference for ``context``'s owner.

        Refuses (raises :class:`SecretResolutionError`) when ``context`` is
        absent — an ``app://`` reference with no owner cannot be answered, and
        guessing is worse than refusing.  An unowned workspace, or an owner not
        of the form ``app:user``, is answered ``unowned`` (dropped, not
        raised).  Otherwise the request is sent to the application the owner is
        qualified under and its answer returned verbatim.
        """
        if context is None:
            raise SecretResolutionError(
                f"{APP_SECRET_SCHEME}://{ref.name}",
                "no resolve context (app:// is resolvable only at session-env "
                "resolution, where the workspace owner is known)",
            )
        owner = context.workspace_owner
        if not owner or ":" not in owner:
            return AppSecretAnswer(
                status="unowned",
                detail=(
                    "workspace is unowned"
                    if not owner
                    else f"owner {owner!r} is not app:user-qualified"
                ),
            )
        app_id, _, user = owner.partition(":")
        if not app_id or not user:
            return AppSecretAnswer(
                status="unowned",
                detail=f"owner {owner!r} is not app:user-qualified",
            )
        deadline = self._timeout if timeout is None else timeout
        try:
            return self._transport(
                app_id, user, context.workspace_path or "", ref.name, deadline,
            )
        except Exception as exc:  # noqa: BLE001 -- transport error is a drop
            logger.warning(
                "app://%s resolution for %s failed: %s",
                ref.name, owner, exc, exc_info=True,
            )
            return AppSecretAnswer(status="unreachable", detail=str(exc))

    # -- SecretResolver protocol conformance -----------------------------

    def resolve(
        self,
        scheme: str,
        path: str,
        key: Optional[str] = None,
        context: Optional[SecretResolveContext] = None,
    ) -> str:
        """Protocol-shaped entry: return the value or raise.

        The bare-string contract cannot carry ``expires_at``, so the
        session-env pass uses :meth:`resolve_reference` instead; this exists
        so the resolver is a genuine ``SecretResolver`` and can be exercised
        as one.  A non-``ok`` answer raises rather than returning the literal
        reference, honouring "never forward an unresolved app://".
        """
        ref = AppSecretReference(name=path.split("?", 1)[0].strip("/"),
                                 required="?required" in path)
        answer = self.resolve_reference(ref, context)
        if answer.ok:
            return answer.value  # type: ignore[return-value]
        raise SecretResolutionError(
            f"{scheme}://{path}",
            answer.detail or f"status={answer.status}",
        )


def answer_from_result(result: Any) -> AppSecretAnswer:
    """Translate an application's ``secret.resolve.result`` into an :class:`AppSecretAnswer`.

    An ``ok`` status with no ``value`` is downgraded to ``error`` — the
    application claimed success and delivered nothing, which must not read as a
    resolved empty secret (dropping is the honest outcome).  Shared by the
    WS bind channel (#1226) and the IPC resolver (#1609).
    """
    if result.status == "ok":
        if result.value is None:
            return AppSecretAnswer(
                status="error", detail="application answered ok with no value",
            )
        return AppSecretAnswer(
            status="ok", value=result.value, expires_at=result.expires_at,
        )
    return AppSecretAnswer(status=result.status or "error", detail=result.detail)


class ChainedAppSecretResolver:
    """Ask each resolver in turn for the workspace owner; resolve with the first.

    Built by ``SessionManager`` when both the WS resolver (#1226) and the IPC
    resolver (#1609) are wired.  The first resolver that names an owner for a
    workspace resolves its references, so a workspace an application owns over
    WS is never answered by a local account.  Only the two methods
    ``JaatoServer._apply_app_secret_references`` calls are provided: owner
    checks elsewhere (memory curation, ``secret.reload``) read the WS
    resolver alone.
    """

    def __init__(self, *resolvers: object) -> None:
        self._resolvers = [r for r in resolvers if r is not None]

    def _for_owner(self, workspace_path: Optional[str]):
        for resolver in self._resolvers:
            owner = resolver.owner_for(workspace_path)  # type: ignore[attr-defined]
            if owner is not None:
                return resolver, owner
        return None, None

    def owner_for(self, workspace_path: Optional[str]) -> Optional[str]:
        """The owner named by the first resolver that names one."""
        return self._for_owner(workspace_path)[1]

    def resolve_reference(
        self,
        ref: AppSecretReference,
        context: Optional[SecretResolveContext],
        *,
        timeout: Optional[float] = None,
    ) -> AppSecretAnswer:
        """Resolve with the resolver that named ``context``'s owner."""
        if context is not None:
            for resolver in self._resolvers:
                if resolver.owner_for(context.workspace_path) == context.workspace_owner:  # type: ignore[attr-defined]
                    return resolver.resolve_reference(ref, context, timeout=timeout)  # type: ignore[attr-defined]
        if self._resolvers:
            return self._resolvers[0].resolve_reference(ref, context, timeout=timeout)  # type: ignore[attr-defined]
        return AppSecretAnswer(status="no_transport", detail="no resolver wired")


def chain_app_secret_resolvers(ws: Optional[object], ipc: Optional[object]) -> Optional[object]:
    """The resolver a new session resolves its ``app://`` references with.

    The WS resolver (#1226), the IPC resolver (#1609), or both chained with
    WS first, so a workspace an application owns is never answered by a
    local account.  ``None`` when neither transport wired one.
    """
    if ws is None or ipc is None:
        return ws if ws is not None else ipc
    return ChainedAppSecretResolver(ws, ipc)
