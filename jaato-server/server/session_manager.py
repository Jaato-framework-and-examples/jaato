"""Session Manager for multi-session support.

This module manages multiple named sessions, each with its own
JaatoServer instance and conversation state.

Sessions are:
- Persisted to disk via the Session Plugin
- Loaded on-demand when clients attach
- Saved periodically and on shutdown
- Identified by consistent IDs across memory and disk

Integration with Session Plugin:
- SessionManager uses SessionPlugin for persistence
- SessionState from the plugin is used for save/load
- Session IDs are consistent between runtime and storage
"""

import json
import logging
import os
import re
import sys
import pathlib
import threading
import time
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.utils.errors import exc_message
from shared.plugins.session import (
    create_plugin as create_session_plugin,
    load_session_config,
    SessionPlugin,
    SessionState,
    SessionConfig,
    SessionInfo as PluginSessionInfo,
)

from shared.instruction_token_cache import InstructionTokenCache
from shared.runtime_limits import RuntimeLimits, apply_isolated_defaults
from shared.session_envelope import BootstrapEnvelope
from shared.instruction_suppression import normalize_suppression
from .core import JaatoServer
from .session_logging import set_logging_context, clear_logging_context, get_session_handler
from .session_workspace_index import SessionWorkspaceIndex
from .wake_binding_registry import WakeBindingRegistry, BindOutcome


class WakeOutcome(str, Enum):
    """Structured result of :meth:`SessionManager.wake_session`.

    Callers route on this enum, not a prose reason string — notably an HTTP
    wake shim mapping to a status code that drives a webhook sender's (e.g.
    GitHub's) retry behavior.  Permanence guidance for a retrying ingress:

    - ``OK`` / ``DUPLICATE`` → **success** (map to 2xx).  ``DUPLICATE`` is a
      benign redelivery no-op (at-least-once senders redeliver by design) — it
      is NOT an error and must not trigger a retry.
    - ``INVALID`` / ``UNRESOLVED`` → **permanent** failure (bad id, or a cold
      id that is unknown / ambiguous in the workspace index) — map to 4xx, do
      NOT retry.
    - ``REVIVE_FAILED`` / ``NOT_DRIVABLE`` → **transient** failure (revive or
      dispatch failed) — map to 5xx, safe to retry.
    """
    OK = "ok"
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    UNRESOLVED = "unresolved"
    REVIVE_FAILED = "revive_failed"
    NOT_DRIVABLE = "not_drivable"
    DEFERRED = "deferred"

    @property
    def is_success(self) -> bool:
        """True for outcomes a caller should treat as success (2xx): the turn
        was dispatched (``OK``), a duplicate was idempotently ignored
        (``DUPLICATE``), or the wake was accepted but the turn DEFERRED until a
        client re-attaches (``DEFERRED`` — the session was revived cold with no
        client; a SessionWokenEvent was emitted to its observers)."""
        return self in (
            WakeOutcome.OK, WakeOutcome.DUPLICATE, WakeOutcome.DEFERRED)


from jaato_sdk.events import (
    Event,
    EventType,
    SystemMessageEvent,
    ErrorEvent,
    SessionInfoEvent,
    SessionDescriptionUpdatedEvent,
    SessionProfilesEvent,
    SessionRestoredEvent,
    ContextUpdatedEvent,
    UsageBreakdown,
    AgentCreatedEvent,
    InstructionBudgetEvent,
    InterruptedTurnRecoveredEvent,
    ToolCallStartEvent,
    ToolCallEndEvent,
    TurnCompletedEvent,
    AgentStatusChangedEvent,
    WorkspaceFilesChangedEvent,
    WorkspaceFilesSnapshotEvent,
)
from .workspace_monitor import WorkspaceMonitor


logger = logging.getLogger(__name__)


@dataclass
class _PendingWake:
    """A wake deferred because the (cold-revived) session had no attached client.

    Held in ``SessionManager._pending_wakes`` keyed by ``session_id``; the turn
    is driven when a client re-attaches (``attach_session`` drains it).  Expires
    with the wake binding so a permanently-detached bot's pending wake doesn't
    linger forever."""
    text: str
    source: str
    wake_ref: str
    cascade_driver_id: Optional[str]
    expires_at: float


@dataclass
class RuntimeSessionInfo:
    """Metadata about a session (runtime + persisted)."""
    session_id: str
    name: str
    description: Optional[str]
    created_at: str
    last_activity: str
    model_provider: str
    model_name: str
    is_processing: bool
    is_loaded: bool  # True if currently in memory
    client_count: int
    turn_count: int
    workspace_path: Optional[str] = None
    created_by: Optional[str] = None  # Authenticated user who created the session


@dataclass
class Session:
    """A managed session with its JaatoServer."""
    session_id: str
    name: str
    server: JaatoServer
    created_at: str
    last_activity: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    attached_clients: Set[str] = field(default_factory=set)
    description: Optional[str] = None
    is_dirty: bool = False  # True if has unsaved changes
    #: Correlation id of the ``session.new`` that created this session, so an
    #: event answering that create can be matched to it by the CLIENT.
    #:
    #: The SDK filters the create-wait on ``request_id`` -- an event without
    #: one is filed as incidental and never satisfies the wait.  So a refusal
    #: emitted without it is INVISIBLE to the caller, which waits out its own
    #: timeout and reports a cause that is not the real one.  That is exactly
    #: what a cascade-exhausted refusal did: the daemon logged the real reason
    #: and emitted a well-typed ErrorEvent, and the caller got a 30s
    #: runner-not-ready timeout naming nothing.
    create_request_id: Optional[str] = None
    #: Serializes saves OF THIS SESSION.  Lives on the session because the
    #: thing it protects is per-session: ``_save_session`` writes
    #: ``<session_id>.json.tmp`` and renames it, so two concurrent saves of
    #: ONE session race — the first rename wins and the second gets ENOENT on
    #: a temp file that no longer exists.
    #:
    #: Per-session and NOT global on purpose.  A global lock would also
    #: prevent the collision, by serializing saves of sessions that never
    #: shared a path — making unrelated sessions wait for each other to fix a
    #: race they were not in.
    #:
    #: On the record rather than in a ``Dict[str, Lock]`` so it is created and
    #: destroyed with the session: no lifecycle, no eviction, and no
    #: possibility of a lock outliving what it guarded.
    #:
    #: ``compare=False``/``repr=False``: a lock is identity, not state.
    save_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False,
    )
    workspace_path: Optional[str] = None  # Client's working directory
    config_root: Optional[str] = None  # Read-only framework-config root override
    user_inputs: List[str] = field(default_factory=list)  # Command history for prompt restoration
    interrupted_turn: Optional[Dict[str, Any]] = None  # Turn interruption state for recovery
    provisioned: bool = False  # True if workspace was auto-provisioned by server
    created_by: Optional[str] = None  # Authenticated user who created the session
    sandbox_mode: Optional[str] = None  # "apparmor" or "soft" when workspace sandboxing is active
    # The UNRESOLVED inline-profile spec (dict), for sessions created from
    # an inline profile rather than a named one.  Carried so _save_session
    # can persist it (SessionState.profile_spec) → disk-restore reconstructs
    # the recipe by id alone (no named profile on disk).  None for
    # named-profile sessions.  Set at create (from the BootstrapEnvelope)
    # and at restore (from state.profile_spec) so it survives save cycles.
    inline_profile_spec: Optional[Dict[str, Any]] = None
    # The RESOLVED recipe and the RENDERED prompt this session ran under,
    # frozen at creation and re-persisted unchanged on every save (issue
    # #787).  Both are WRITE-ONCE by intent: ``_save_session`` fills them
    # in on the first save that finds them empty and never overwrites
    # them, and ``_load_session`` carries the persisted values straight
    # back onto the restored record.
    #
    # Write-once is the load-bearing part.  Reviving with
    # ``JAATO_REVIVE_PERSONA=disk`` deliberately re-renders the prompt, so
    # a save that recomputed from the live server would replace the
    # original artifact with the re-render -- and the original is the
    # thing an interrogation exists to ask about.  Testing an alternative
    # must not destroy the record it is being compared against.
    profile_snapshot: Optional[Dict[str, Any]] = None
    rendered_instructions: Optional[str] = None
    # The ``agent_params`` this session was created with, captured with the
    # two snapshots above and under the same write-once rule.  Persisted so
    # the OPT-IN re-render path can re-run the persona's prefetch against
    # the ORIGINAL inputs; handing it an empty dict is the #787 defect.
    # NEVER a place for a credential -- see SessionState.agent_params.
    agent_params: Optional[Dict[str, str]] = None
    # Server 0.6.164+ (Bug B real root cause): opaque cascade tenant
    # ID stamped at session creation.  Consumed by
    # :meth:`_dispatch_to_cascade_clients` (Phase 1 cascade-as-client
    # dispatch) and :meth:`_record_cid_session_activity` (Bug B GC
    # sweep cid-scoped activity tracker).  Pre-0.6.164 BOTH paths
    # called ``getattr(session, "cascade_driver_id", None)`` which
    # silently returned None because the Session dataclass didn't
    # have the field — dispatch never fired for cascade sessions and
    # PR-188's GC sweep cid-skip never engaged for downstream
    # reactor-spawned stages.  See peer 7:1's retry-45 diagnostic
    # (2026-05-28): cascade-stamped runner-pool slots showed cid
    # propagation working at the pool layer, but session-level cid
    # was never set, so the GC reaped the observer at 22:07:25
    # despite host_validator having spawned 85s earlier.
    cascade_driver_id: Optional[str] = None
    # Cascade-scoped ADDRESS for sibling messaging (design §4).  Distinct
    # from ``session_name`` (free-text display) and ``agent_name``
    # (persona): this is the string another session passes to
    # ``send_to_sibling``.  None = not addressable by peers.
    sibling_name: Optional[str] = None
    # False when this child declared its own budget_control: a delegation
    # to another department, accounted on its own books.  Such a child does
    # not deplete the parent's shared pot, is not clamped by it, and is not
    # degraded when it crosses.  True (default) = draws on the parent's
    # budget and is governed by it.
    draws_on_parent_budget: bool = True
    # Phase 3 §3.12 disk-restore + peer-review M5/N1: True when this
    # Session was loaded from disk and is awaiting its first
    # client-attach.  While True, ``check_permission`` ASK paths
    # **hold** tool calls (queue the prompt rather than deny) so a
    # session restored after a daemon restart doesn't silently fail
    # in-flight work.  Cleared on first client attach, at which
    # point the daemon emits ``SessionRestoredEvent`` so the client
    # surfaces a "review pending tool calls" prompt and drains the
    # queue.  False for fresh / never-restored sessions.
    restored_pending_attach: bool = False


@dataclass
class SubRunnerHandle:
    """Daemon-side bookkeeping for an isolated-subagent sub-runner
    (Phase 4 §4.3.6a).

    Holds the RPC client + spawned subprocess handle for a sub-runner
    spawned by the §4.3 isolated-subagent opt-in.  Distinct from
    ``Session`` (which tracks top-level user-visible sessions) — these
    handles live in their own dict keyed by isolated_session_id, off
    the parent session's lifecycle.

    Lifecycle (each §4.3.6 sub-commit advances):
    - §4.3.6a: created when the helper's spawn invocation succeeds.
    - §4.3.6b: cross-runner output forwarding subscribes by handle.
    - §4.3.6c: cross-runner prompt forwarding dispatches via ``rpc``.
    - §4.3.6d: parent-cascade teardown reads + cleans up.

    Fields:
        parent_session_id: The session that owns this isolated
            subagent.  Used for parent-cascade teardown lookups.
        subagent_id: Pre-generated id from the runner-side subagent
            plugin.  Combined with ``parent_session_id`` to derive
            the isolated_session_id, sub-AppArmor profile name, and
            sub-cgroup path.
        isolated_session_id: ``{parent}__sub_{subagent}`` — used as
            the session_id for daemon-side RPC + kernel-resource
            names.
        rpc: The :class:`RunnerRPCClient` started against the
            sub-runner's parent socket.  Daemon talks to the
            sub-runner via this handle.
        spawned: The :class:`SpawnedRunner` (pid + socket pair).
            Held for teardown.
        sub_apparmor_profile: The kernel-visible sub-profile name
            (``jaato-ws-{parent}//{subagent}``).  Empty when
            unconfined fallback was taken (not the normal path).
        cgroup_path: Absolute path to the sub-cgroup directory.
            Empty when no cgroup was created (no runtime_limits or
            cgroups unavailable).
        created_at: Timestamp for diagnostics + leak detection.
    """
    parent_session_id: str
    subagent_id: str
    isolated_session_id: str
    rpc: Any  # server.runner_rpc_client.RunnerRPCClient
    spawned: Any  # server.runner_spawner.SpawnedRunner
    sub_apparmor_profile: str
    cgroup_path: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


@dataclass
class CascadeClientEntry:
    """Phase 1 cascade-as-client: one cascade-client registration entry.

    See ``docs/design/cascade-as-client.md`` §4.1.  Multiple entries
    per ``cascade_driver_id`` are allowed — one ``owner`` + N
    ``observer`` per Decision 5 in the design doc.

    Fields:
        client_id: Namespaced identifier for this registration.
            Convention: ``_cascade:{cid}`` for the owner; observer
            client_ids may be anything (e.g., the IPC connection's
            UUID, an extension-supplied label).  Decision 1 locks the
            namespaced shape.
        role: ``"owner"`` (single per cid; has lifecycle authority)
            or ``"observer"`` (multiple per cid; read-only event
            subscription).  Decision 5.
        callback: In-process callable invoked when an event matching
            ``event_types`` arrives for a session stamped with this
            cid.  Signature: ``Callable[[Event], None]``.  Runs
            synchronously inside ``_emit_to_session`` — callback
            MUST NOT re-enter SessionManager methods that take
            ``_lock`` (avoid deadlock).  Phase 2 will add an
            IPC-RPC variant where callback dispatches to the
            connected client's event channel.
        event_types: Set of event-type names this entry subscribes
            to.  ``None`` = subscribe to all.  Decision 3 (subscriber-
            defined filter).  Type name comparison via
            ``type(event).__name__`` for cheap dispatch.
        registered_at: Wall-clock monotonic timestamp at registration.
        last_event_ts: Monotonic timestamp of the most recent event
            dispatched to this entry.  ``None`` until the first event.
            Used by the GC backstop sweep (Decision 6).
        delivery_target_id: Raw connection identifier this entry's
            ``callback`` delivers TO (server 0.6.178+).  Distinct
            from ``client_id`` which is the namespaced REGISTRATION
            identifier (e.g.
            ``_cascade:eb3ed3d4c6f5474abbe78a27e9e93ab4:ipc_1``).
            For IPC-bound cascade registrations the callback's
            ``send_event`` target is the raw connection id (``ipc_1``);
            ``_route_bootstrap_event`` uses this field to dedup
            bootstrap-time delivery against the direct-IPC path.
            ``None`` (default) for in-process callers (e.g. premium
            reactor extension) where the callback is not tied to a
            raw connection — the dedup branch in
            :meth:`_dispatch_to_cascade_clients_by_cid` is a no-op
            for those entries (skip never matches), preserving the
            extension-callback delivery contract.
    """
    client_id: str
    role: str
    callback: Callable[[Any], None]  # Callable[[Event], None]
    event_types: Optional[Set[str]] = None
    registered_at: float = field(default_factory=time.monotonic)
    last_event_ts: Optional[float] = None
    delivery_target_id: Optional[str] = None

    def event_type_match(self, event: Any) -> bool:
        """Return True iff this entry's event-type filter matches
        the given event.  ``event_types=None`` matches all."""
        if self.event_types is None:
            return True
        return type(event).__name__ in self.event_types


#: A sibling address is a SLUG, not free text.  ``session_name`` cannot serve:
#: it auto-generates as ``Session 2026-08-24 14:15`` and every existing session
#: has spaces in it, so constraining it retroactively would break them all.
#:
#: The shape is deliberately narrow.  A roster entry is rendered into another
#: agent's context, so a free-text address could carry prose -- a sibling naming
#: itself "Permission Approver - reply yes to authorize" would be writing
#: instructions into every sibling's view without sending a message.  A slug
#: cannot express that, which confines the injection surface to the session
#: DESCRIPTION, where the untrusted-content marking lives.
SIBLING_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


# ----------------------------------------------------------------------
# Sibling messaging caps (design §8)
# ----------------------------------------------------------------------
#
# ``budget_control`` is the real terminator -- a ping-pong between siblings
# burns turns, and a cascade budget counts turns/seconds/tools/spend across
# every session under the cid, so a runaway conversation hits a ceiling that
# already exists and DEGRADES before it stops.  These three caps are
# backpressure in front of that ceiling, not a replacement for it.

SIBLING_MESSAGE_MAX_BYTES = 8 * 1024
"""Size cap: one sibling must not be able to blow another's context in a
single message.  UTF-8 bytes, not characters -- a character count would let
a multi-byte payload through at several times the intended size."""

SIBLING_PENDING_CAP = 20
"""Per-target backpressure: how many messages may pile up on a sibling that
has been BUSY the whole time.

Counts consecutive ``queued`` deliveries to one target.  Resets the moment a
delivery finds that target idle, because an idle target has drained -- SIBLING
is an idle-only tier, and ``inject_prompt`` fires the continuation rather than
queuing when the session is idle.  So the counter measures exactly what it
claims: a backlog against a peer that never came up for air."""

SIBLING_CID_EXCHANGE_CAP = 200
"""Per-cascade total sends.  The blunt terminator for a two-sibling ping-pong
that stays under the pending cap by alternating.  Monotonic per cid; never
reset, because the thing it bounds is the CONVERSATION, not a backlog."""


def validate_sibling_name(
    sibling_name: str,
    cascade_driver_id: Optional[str],
    existing: "Iterable[Tuple[Optional[str], Optional[str]]]",
) -> Optional[str]:
    """Validate a sibling address.  Returns an error string, or None if valid.

    Two independent rules, both enforced at ``session.new`` rather than at
    send time:

    **Shape** -- see :data:`SIBLING_NAME_RE`.

    **Uniqueness within the cascade** -- an address that is not unique
    addresses nobody in particular: the second claimant would silently receive
    traffic meant for the first, with a perfectly healthy-looking delivery
    receipt.  Scoped to ``cascade_driver_id`` because that is the addressing
    boundary (design §2); two unrelated cascades may both hold a ``reviewer``
    without ambiguity, and forbidding that would make names a global namespace
    nobody asked for.

    A session with no ``cascade_driver_id`` is not addressable by siblings at
    all, so its name is checked for SHAPE but collides with nothing.

    Args:
        sibling_name: The proposed address.
        cascade_driver_id: The cascade this session will join, if any.
        existing: ``(sibling_name, cascade_driver_id)`` for every live session.

    Returns:
        A human-readable reason, or ``None`` when the name is acceptable.
    """
    if not SIBLING_NAME_RE.match(sibling_name):
        return (
            f"sibling_name {sibling_name!r} is not a valid address: expected "
            f"{SIBLING_NAME_RE.pattern} (lowercase, no spaces, max 32 chars). "
            f"A sibling address is rendered into other agents' context, so it "
            f"must not be able to carry prose."
        )
    if cascade_driver_id is None:
        return None
    for other_name, other_cid in existing:
        if other_name == sibling_name and other_cid == cascade_driver_id:
            return (
                f"sibling_name {sibling_name!r} is already taken in cascade "
                f"{cascade_driver_id!r}. Addresses must be unique within a "
                f"cascade, or traffic meant for one peer silently reaches "
                f"another."
            )
    return None


#: Roles a sibling row can carry, RELATIVE TO THE ASKER.
#:
#: ``role`` must describe the relationship to whoever called ``list_siblings``,
#: not a property of the row.  Computed from ``owner_id`` alone it would be a
#: property of the row, and that is a real defect: top-level siblings A and C
#: share a cascade, A spawns B, and C asks for the roster.  Row-computed, B
#: renders as "child" -- but B is A's child, not C's.  C reads "child" as "my
#: child", i.e. a TRUSTED field telling a sibling it has authority it does not
#: have.  That is worse than an untrusted field carrying a hostile string,
#: because nothing marks it as suspect.
#:
#: ``unrelated`` is the honest answer for B-seen-from-C: in your cascade, not
#: on your line.  Under-describing is recoverable; mis-describing authority is
#: not.
PEER_ROLES = ("self", "parent", "child", "sibling", "unrelated")


def compute_peer_role(
    viewer_session_id: str,
    viewer_owner_id: Optional[str],
    row_session_id: str,
    row_owner_id: Optional[str],
) -> str:
    """The asker's relationship to one roster row.

    Args:
        viewer_session_id: Session calling ``list_siblings``.
        viewer_owner_id: That session's owner, or None if top-level.
        row_session_id: The session being described.
        row_owner_id: Its owner, or None if top-level.

    Returns:
        One of :data:`PEER_ROLES`.
    """
    if row_session_id == viewer_session_id:
        return "self"
    if row_session_id == viewer_owner_id:
        return "parent"
    if row_owner_id == viewer_session_id:
        return "child"
    # Same owner = true siblings.  Two TOP-LEVEL sessions (both owners None)
    # are siblings of the cascade driver, which is the common case for
    # cascade stages, so None == None counts.
    if row_owner_id == viewer_owner_id:
        return "sibling"
    return "unrelated"


#: Why a delivery did not happen, in words the SENDING MODEL can act on.
#:
#: ``deliver_prompt_to_session`` distinguishes five outcomes; the refusals
#: that quote them used to collapse all of them into "no live runner
#: channel" -- a hardcoded likely-cause that was WRONG for four of the five.
#: Observed live: a peer that was alive, whose runner channel was fine, whose
#: delivery timed out because the daemon's event loop did not schedule the
#: coroutine within 7s.  The sender was told its sibling had no channel.
#:
#: The same collapse then happened one level down, inside ``unreachable``
#: itself.  FIVE producers shared the word -- no server attached, no runner
#: channel, runner too old for the offer verb, the offer raised, and the
#: DRIVE failing after the target answered ``needs_turn`` -- and the single
#: prose reason described only the fourth ("may still have been enqueued and
#: only the acknowledgement lost").  For the other four nothing was ever
#: offered, so that sentence was not vague, it was FALSE: it warned about a
#: duplicate that could not exist, and a careful sender therefore declined to
#: re-send a message that had definitely never arrived.
#:
#: They are split on the ONE axis a sender can act on -- was anything put in
#: flight? -- not on mechanism.  Mechanism is what the log names; a sender
#: cannot do anything differently for "no runner channel" than for "no server
#: attached", so those do not earn separate words.
_DELIVERY_FAILURE_REASON = {
    "no_session": (
        "is not loaded (no session with that id is in memory; it may have "
        "been unloaded or never existed)"
    ),
    "terminated": (
        "has terminated and will run no further turns (reported by the "
        "session itself, not inferred from silence)"
    ),
    "unreachable": (
        "could not be reached -- NOTHING WAS SENT. The session is loaded but "
        "has no delivery path right now (no server attached, no runner "
        "channel, a runner too old to accept the offer, or a turn that could "
        "not be started). Re-sending is SAFE -- nothing was enqueued, so it "
        "cannot duplicate -- but will keep failing until the path is "
        "restored. The daemon log names which of the four it was"
    ),
    "not_confirmed": (
        "was sent an offer whose answer was lost (the delivery call raised "
        "or timed out). The message may be in its queue right now, or may "
        "never have arrived -- from here those are indistinguishable. "
        "RE-SENDING MAY DELIVER IT TWICE. The daemon log names the exception "
        "and which layer it came from"
    ),
    "busy": (
        "is mid-turn and has too many messages already waiting; the target "
        "itself declined to take another"
    ),
}


def _delivery_failure_reason(status: str) -> str:
    """Render *status* for a sender.  Unknown statuses are NOT guessed at."""
    return _DELIVERY_FAILURE_REASON.get(
        status, f"could not be delivered (status={status!r})"
    )


def _stamp_session_id(event: Any, session_id: Optional[str]) -> None:
    """Attribute *event* to *session_id* — unless it already names one.

    NEVER OVERWRITES.  An emitter that set ``session_id`` explicitly is
    stating the event's SUBJECT, which is not always the session it was
    routed through: ``SlotSettledEvent`` means "the session that just
    ended", ``GateReleasedEvent`` means "the originating session".
    Relabelling those with whoever emitted them would replace a true
    fact with a plausible one — worse than leaving them blank, because
    a wrong attribution is indistinguishable from a right one.

    Tolerant of events that predate the base-class field (and of test
    doubles that aren't ``Event`` at all): a failure to stamp must never
    break event delivery.  The read side treats empty as "not routed",
    so an unstamped event degrades to exactly the pre-1.2 behaviour
    rather than to a wrong answer.
    """
    if not session_id:
        return
    try:
        if not getattr(event, "session_id", ""):
            event.session_id = session_id
    except (AttributeError, ValueError):
        # Pydantic raises ValueError for an undeclared field; a plain
        # object raises AttributeError.  Neither is worth failing a
        # delivery over.
        pass


class SessionManager:
    """Manages multiple named sessions with persistence.

    Integrates with the Session Plugin to provide:
    - Persistent storage of session history
    - Load sessions on-demand from disk
    - Save sessions periodically and on shutdown
    - Unified view of in-memory and on-disk sessions

    Each session has its own JaatoServer with isolated:
    - Conversation history
    - Agent state
    - Plugin state
    - Token accounting

    Clients can:
    - Create new sessions
    - Attach to existing sessions (loads from disk if needed)
    - List all sessions (memory + disk)
    - Save/checkpoint sessions
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
    ):
        """Initialize the session manager.

        Args:
            storage_path: Override for session storage path.
        """

        # Freeze the revive posture (issue #787) BEFORE any session exists.
        # ``JAATO_REVIVE_PROFILE`` / ``JAATO_REVIVE_PERSONA`` are ``host``-
        # scoped: one answer for this process, decided by whoever started it.
        # Read live per revive they would NOT be, because
        # ``JaatoServer._with_session_env`` copies every key of a session's
        # workspace ``.env`` into the daemon-global ``os.environ`` for that
        # session's turn, with no scope filter — so one workspace could set
        # the posture for every other session's revive, and
        # ``PERSONA=disk`` re-runs prefetch scripts.  Capturing here removes
        # the window instead of narrowing it; see ``server/revive_policy.py``.
        from server import revive_policy
        revive_policy.capture()

        # Initialize session plugin for persistence.
        # storage_path stays relative (e.g. ".jaato/sessions") — it is
        # resolved per-workspace via _session_storage_dir() at each call site.
        self._session_plugin: SessionPlugin = create_session_plugin()
        self._session_config: SessionConfig = load_session_config()

        if storage_path:
            self._session_config.storage_path = storage_path

        # Initialize with relative storage_path. The plugin's self._storage_path
        # acts as a fallback for standalone JaatoClient usage; SessionManager
        # always passes an explicit storage_dir resolved per-workspace.
        self._session_plugin.initialize({
            'storage_path': self._session_config.storage_path
        })

        # Daemon-wide reactor EventBus: the SINGLE bus reactors subscribe to so
        # they receive events from ALL sessions and survive session unloads.
        # Each per-session bus forwards into this via a "reactor_bus_sink"
        # subscription wired when that session's server is built (below).  Per-
        # session subscribers stay isolated on their own bus; only the sink
        # forward crosses into this daemon-wide one.  Exposed to daemon
        # extensions as ``_ExtensionContext.event_bus`` so the reactor engine
        # subscribes ONCE here instead of per-loaded-session.  See
        # docs/design/reactor-bus-session-scope.md.
        from shared.event_bus import EventBus
        self.reactor_event_bus = EventBus()

        # In-memory session storage
        self._sessions: Dict[str, Session] = {}
        # In-flight async unloads (``_do_session_unload``).  session_id → an
        # event set when that unload completes.  An entry exists ONLY between
        # the unload's commit point (it passed its attached-clients re-check)
        # and its final ``_sessions.pop`` — so ``attach_session`` can detect a
        # session being torn down and await+reload instead of attaching to a
        # session whose runner is mid-disposal (the attach-vs-unload race).
        self._unloading: Dict[str, threading.Event] = {}
        # Use RLock (reentrant) because initialize() may emit events during session load
        self._lock = threading.RLock()

        # Wake primitive (session.wake): daemon-owned session_id → workspace
        # index so a cold session can be revived by id alone WITHOUT the caller
        # supplying a path (which would let an untrusted wake caller point
        # revival at a weaker sandbox root).  See session_workspace_index.py.
        self._session_workspace_index = SessionWorkspaceIndex()
        # Bounded LRU of wake event_ids already actioned — external ingresses
        # (GitHub, etc.) redeliver; a duplicate event_id is dropped.
        self._wake_seen_event_ids: "OrderedDict[str, None]" = OrderedDict()
        # wake_ref → binding registry (the SESSION-owned half of the wake
        # contract: which key(s) a session trusts to wake it, per opaque ref).
        # Written by bind_wake/unbind_wake (owner = caller's session), read by
        # the mode-B verify shim.  See wake_binding_registry.py.
        self._wake_binding_registry = WakeBindingRegistry(
            owner_exists=self._owner_session_record_exists)
        # Operator-declared public wake endpoint (wake.json public_url), set at
        # daemon boot; surfaced on bind_wake so a session can advertise it with
        # no bot-side URL config.  Empty until the daemon wires it.
        self._wake_public_url: str = ""
        # Deferred wakes (Option 2): session_id → _PendingWake for a session
        # revived COLD with no attached client.  Driven when a client
        # re-attaches (attach_session drains it); re-emitted on observer
        # (re)register.  Guarded by _lock.
        self._pending_wakes: Dict[str, _PendingWake] = {}

        # Phase 1 cascade-as-client (server 0.6.154+): registry of
        # cascade-clients keyed by cascade_driver_id.  See
        # docs/design/cascade-as-client.md §4.1-§4.3.  Each entry is a
        # :class:`CascadeClientEntry`.  Multiple entries per cid are
        # allowed (one owner + N observers per Decision 5 in the
        # design doc).  Guarded by its own lock to avoid coupling
        # with ``_lock`` (which is held during _emit_to_session and
        # would deadlock if a cascade-client callback re-entered
        # SessionManager).
        self._cascade_clients: Dict[str, List["CascadeClientEntry"]] = {}
        self._cascade_clients_lock = threading.Lock()
        # Sibling-messaging caps (design §8).  Both are daemon-side because
        # the daemon is the only party that sees every send in a cascade --
        # a sender-side counter would be per-session and a ping-pong between
        # two siblings would never reach either one's limit.
        #
        # ``_sibling_pending``: target session_id -> consecutive queued sends.
        # ``_sibling_exchanges``: cid -> total sends, monotonic.
        self._sibling_pending: Dict[str, int] = {}
        self._sibling_exchanges: Dict[str, int] = {}
        # Per-cid AGGREGATE budget ceilings (design note §8/b).  Declared by
        # the cascade OWNER at launch — deliberately not a leaf-profile field,
        # because a cascade cap is a runtime aggregate over a live cid, not a
        # property of any one reusable template (§3.1).  Keyed by
        # cascade_driver_id; absent = that cascade is uncapped.
        # Session ids CLAIMED but not yet registered in ``_sessions``.
        # Allocation and registration are ~336 lines and a full runner spawn
        # apart, so without an atomic claim the check-then-act races: every
        # concurrent create inside that window sees the same id free.
        self._reserved_session_ids: Set[str] = set()
        self._cascade_budgets: Dict[str, "CascadeBudgetPool"] = {}
        self._cascade_budgets_lock = threading.Lock()
        # GC backstop: cascade-client entries with no event for this
        # many seconds + no active sessions in their cid get reaped
        # by the periodic sweep.  Matches the Phase 2 cascade-idle
        # slot teardown timeout for design coherence.
        self._cascade_client_idle_timeout = 300.0
        # Sweep thread for cascade-client GC; lazily started on first
        # registration.  Stopped on shutdown.
        self._cascade_client_sweep_stop: Optional[threading.Event] = None
        self._cascade_client_sweep_thread: Optional[threading.Thread] = None
        # Server 0.6.161+: monotonic timestamp of the most recent
        # session creation per cascade_driver_id.  Used by the
        # cascade-client GC sweep as an "is this cascade alive?"
        # signal alongside the "any currently-loaded session" check —
        # closes the mid-cascade-reap bug where the sweep landed in
        # the brief window between session N unloading (per PR-183
        # default policy) and session N+1 spawning, reaped the
        # observer registration even though the cascade was actively
        # progressing.  See ``_record_cid_session_activity``.
        self._cid_last_session_ts: Dict[str, float] = {}

        # cascade.cancel: set of cascade_driver_ids that have been
        # cancelled by an operator (typically kb-side ^C → IPC verb
        # ``cascade.cancel cid``).  Reactor extensions consult via
        # :meth:`is_cid_cancelled` before firing on AgentCompletedEvent
        # so a cancelled cascade stops spawning new sessions.
        #
        # Lifetime: entries persist until daemon shutdown.  Unbounded
        # growth is bounded in practice by the operator-driven nature
        # of cancellations (one entry per ^C, not per session).  No
        # eviction logic — adding TTL/cap would be defensive
        # programming with no concrete failure-mode evidence.
        self._cancelled_cids: Set[str] = set()
        self._cancelled_cids_lock = threading.Lock()

        # Phase 4 §4.3.6a: bookkeeping for isolated-subagent sub-runners.
        # Keyed by isolated_session_id (``{parent}__sub_{subagent}``).
        # Distinct from ``_sessions`` — these are off the top-level
        # session lifecycle and are owned by the parent session for
        # parent-cascade teardown.  Guarded by ``_lock`` (same as
        # ``_sessions``) — both dicts are mutated together at
        # parent-shutdown time.
        self._isolated_sub_runners: Dict[str, SubRunnerHandle] = {}

        # Path H (cycle 10): serialize concurrent async saves so
        # parallel ToolCallStartEvents (parallel tool execution)
        # don't trample each other.  atomic_write_json already
        # prevents file tear.  Ordering across concurrent saves of ONE
        # session is now given by ``Session.save_lock``, held inside
        # ``_save_session`` itself, so every caller inherits it.  The global
        # lock that used to live here guarded a single call site and made
        # unrelated sessions wait on each other; it is gone rather than left
        # as a decoy for the next reader to trust.

        # Client to session mapping
        self._client_to_session: Dict[str, str] = {}

        # Per-client configuration (presentation context, working_dir, etc.)
        self._client_config: Dict[str, Dict[str, Any]] = {}
        # Client-provided ("host") tool schema dicts buffered by the transport
        # when registered BEFORE session.new (client_id -> [tool_def, ...]).
        # Drained in the session.new flow to seed JaatoServer.client_tool_schemas
        # BEFORE spawn_session_runner reads it for envelope.client_tools — the
        # schemas must beat the spawn (PR #349 race fix).  Transport-agnostic.
        self._pending_client_tools: Dict[str, List[Dict[str, Any]]] = {}

        # Event routing callback
        self._event_callback: Optional[Callable[[str, Event], None]] = None
        # Broadcast callback — wired to CompositeEventSink.broadcast_event
        # by the daemon (see __main__.py).  Daemon-wide events (currently
        # HandoffGate transitions from jaato-premium) flow through this.
        self._broadcast_callback: Optional[Callable[[Event], None]] = None

        # Workspace file monitors keyed by session_id
        self._workspace_monitors: Dict[str, WorkspaceMonitor] = {}

        # Shared instruction token cache — survives across session
        # creates/restores within the same daemon process.
        self._instruction_token_cache = InstructionTokenCache()

        # Session hooks — callbacks invoked after each session is initialized.
        # Registered by daemon extensions via ``add_session_hook()``.
        self._session_hooks: List[Callable] = []
        # Registered by transports via ``add_pre_initialize_hook()``.
        # Fire BEFORE ``server.initialize()`` so kernel-level provisioning
        # (AppArmor profile, cgroup) is in place before prefetch scripts
        # run (server 0.6.49+).
        self._pre_initialize_hooks: List[Callable] = []

        logger.info(f"SessionManager initialized with storage template: {self._session_config.storage_path}")

    def buffer_client_tools(
        self, client_id: str, tools: List[Dict[str, Any]]
    ) -> None:
        """Buffer client-provided ("host") tool schema dicts a transport
        received BEFORE this client's session exists.

        Drained in the session.new flow to seed
        ``JaatoServer.client_tool_schemas`` BEFORE ``spawn_session_runner``
        reads it for ``envelope.client_tools`` (the schemas must beat the
        spawn — PR #349 race fix).  The transport still registers the proxy
        EXECUTORS itself, post-session.new (execution is transport-specific).
        """
        self._pending_client_tools[client_id] = list(tools or [])

    def _session_storage_dir(self, workspace_path: str) -> pathlib.Path:
        """Resolve session storage directory for a workspace.

        Combines the workspace path with the configured (relative) storage_path
        to produce an absolute directory, e.g.
        ``/home/user/project`` + ``.jaato/sessions`` → ``/home/user/project/.jaato/sessions``.

        Args:
            workspace_path: Absolute path to the client's workspace.

        Returns:
            Absolute Path to the session storage directory.

        Raises:
            ValueError: If workspace_path is empty/None.
        """
        if not workspace_path:
            raise ValueError("workspace_path required for session storage")
        return pathlib.Path(workspace_path) / self._session_config.storage_path

    @staticmethod
    def _resolve_agent(
        agent_name: str,
        params: Optional[Dict[str, str]],
        workspace_path: Optional[str],
        config_root: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve an agent by name from .jaato/agents/ and .jaato/prompts/.

        Scans agent directories (workspace then user-level), reads the
        markdown file, parses frontmatter, substitutes params, and returns
        the rendered system instructions.

        Args:
            agent_name: Agent name (filename stem).
            params: Parameter values for ``{{param}}`` placeholders.
            workspace_path: Workspace directory for agent resolution.
            config_root: Optional override for the workspace tier.  When
                set, scans ``<config_root>/agents/`` and
                ``<config_root>/prompts/`` instead of the
                workspace-anchored paths.  See
                :func:`shared.config_resolver.resolve_config_search_path`.

        Returns:
            Dict with ``system_instructions``, ``description``,
            ``default_profile``, ``missing_params``, or ``None`` if not found.
        """
        # Logic lives in the shared loader so the embedded in-process client
        # can reuse it without importing ``server`` (mirrors how
        # ``shared.config_resolver.resolve_secret_uri`` was lifted out of the
        # daemon). This thin delegate keeps the daemon's existing call sites.
        from shared.plugins.subagent.config import resolve_agent
        return resolve_agent(agent_name, params, workspace_path, config_root)

    def _resolve_profile(
        self,
        profile_name: str,
        workspace_path: str,
        config_root: Optional[str] = None,
        env_file: Optional[str] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Resolve an agent profile by name.

        Discovers profiles from the workspace tier (``<config_root>/profiles/``
        when set, else ``<workspace_path>/.jaato/profiles/``), the user tier
        (``~/.jaato/profiles/``), and premium entry points, then returns the
        matching ``SubagentProfile``.  When the profile is not found, the
        second element carries a user-friendly error message (e.g. a JSON
        parse error for a broken profile file).

        Args:
            profile_name: Name of the profile to look up.
            workspace_path: Workspace directory containing ``.jaato/profiles/``.
            config_root: Optional override for the workspace tier — see
                :func:`shared.config_resolver.resolve_config_search_path`.
            env_file: Optional path to the session's ``.env`` file.  When
                provided, the file is parsed and overlaid onto the
                session-scoped ``ContextVar`` BEFORE
                :func:`discover_profiles` runs — closes the wire-gap
                where ``JAATO_PROFILE_SET`` (and any other env-var
                read by profile discovery) is empty because the
                normal ``_with_session_env`` block hasn't fired yet.
                Discovered 2026-05-18 with the kb-enablement-2.0
                handoff_test profile-set refactor: ``profiles/<set>/``
                subdir was silently skipped because the contextvar
                wasn't populated at this point in
                ``_create_session_impl``.  Symmetric to PR #139
                (completion_validators wire-gap) and PR #140
                (plugin_configs pass:// wire-gap).  Server 0.6.124+.

        Returns:
            Tuple of ``(profile, None)`` on success, or
            ``(None, error_message)`` when the profile cannot be resolved.
        """
        from shared.plugins.subagent.config import discover_profiles

        # Overlay env_file onto the session-scoped ContextVar so
        # ``discover_profiles`` sees the workspace-declared
        # ``JAATO_PROFILE_SET`` (and any other env-affected reads it
        # may grow).  Same parsing pipeline as
        # ``JaatoServer._resolve_session_env`` — ``dotenv_values`` +
        # ``expand_variables`` so ``pass://`` / ``vault://`` / ``${VAR}``
        # cross-references resolve daemon-side, never reaching the
        # AppArmor-confined runner literal.  Saved/restored via
        # try/finally so concurrent callers don't clobber each other's
        # contextvar state.
        from shared.session_context import (
            _session_env as _session_env_var,
            set_session_env,
        )
        previous_env = _session_env_var.get()
        overlay_applied = False
        if env_file:
            try:
                from dotenv import dotenv_values
                from shared.plugins.subagent.config import expand_variables
                raw = dotenv_values(env_file)
                raw_filtered = {k: v for k, v in raw.items() if v is not None}
                resolved = expand_variables(raw_filtered, context=raw_filtered)
                set_session_env(resolved)
                overlay_applied = True
            except Exception as exc:  # noqa: BLE001
                # Failure to parse env_file is non-fatal for profile
                # resolution — discover_profiles still falls back to
                # the contextvar's existing value / ``os.environ``.
                # Log at INFO so operators see the source of any
                # profile-set-not-found surprises that follow.
                logger.info(
                    "_resolve_profile: env_file overlay skipped for "
                    "%r (parse failed: %s); discover_profiles will "
                    "fall back to contextvar / os.environ",
                    env_file, exc,
                )

        # Qualified path support: when the caller asks for
        # ``<set>/<name>``, route the request to the named profile set
        # regardless of the per-session ``JAATO_PROFILE_SET`` env var.
        # The set's subdirectory under ``<config_root>/profiles/`` (or
        # ``<workspace>/.jaato/profiles/`` when no config_root override
        # is in effect) is scanned and the bare name is looked up
        # within the resulting profile map.  Pre-fix the SDK would hand
        # back ``Profile not found`` for any qualified path whose set
        # didn't match the current ``JAATO_PROFILE_SET`` value (or
        # whose env var was unset), even though the underlying file
        # existed at ``profiles/<set>/<name>.yaml``.  Symmetric to the
        # SDK Bug-B fix landed on 2026-06-06.
        force_profile_set: Optional[str] = None
        lookup_name = profile_name
        if "/" in profile_name:
            head, _, tail = profile_name.partition("/")
            if head and tail and "/" not in tail:
                force_profile_set = head
                lookup_name = tail

        try:
            result = discover_profiles(
                ".jaato/profiles",
                base_path=workspace_path,
                config_root=config_root,
                force_profile_set=force_profile_set,
            )
        finally:
            if overlay_applied:
                _session_env_var.set(previous_env)
        profile = result.profiles.get(lookup_name)
        if profile is not None:
            return profile, None

        # Profile not in the successfully parsed set — check if there was
        # a parse error for a file matching the requested name.
        if lookup_name in result.errors:
            return None, (
                f"Profile '{profile_name}' exists but failed to parse: "
                f"{result.errors[lookup_name]}"
            )
        if force_profile_set is not None:
            return None, (
                f"Agent profile '{profile_name}' not found in "
                f".jaato/profiles/{force_profile_set}/"
            )
        return None, f"Agent profile '{profile_name}' not found in .jaato/profiles/"

    def add_session_hook(self, hook: Callable) -> None:
        """Register a callback invoked after each session is initialized.

        Session hooks are the primary mechanism for daemon extensions to
        inject per-session functionality (e.g., registering custom
        environment aspects, wiring remote spawn handlers).

        The hook is called with two arguments:

        1. ``server`` — the ``JaatoServer`` instance for the newly created
           or loaded session.  Fully initialized: plugin registry populated,
           provider connected, tools configured.
        2. ``session_id`` — the unique session identifier string
           (e.g., ``"20260321_132926"``).

        Hooks are called in registration order.  If a hook raises an
        exception, it is logged and subsequent hooks still run.

        Args:
            hook: A callable with signature
                ``(server: JaatoServer, session_id: str) -> None``.

        Example (from a daemon extension)::

            def _on_session_ready(self, server, session_id):
                env = server.registry.get_plugin("environment")
                if env and hasattr(env, 'register_aspect'):
                    env.register_aspect("my_aspect", self._handler)

            # In the extension's start():
            ctx.session_manager.add_session_hook(self._on_session_ready)
        """
        self._session_hooks.append(hook)

    # ------------------------------------------------------------------
    # IPC AppArmor + runner spawn (Phase 3 §3.13)
    # ------------------------------------------------------------------
    #
    # Phase 2 §2.3 wired this as a pre-initialize hook in
    # server/__main__.py.  Phase 3 §3.13 relocates the logic into the
    # SessionManager itself so it lives next to the bootstrap helper
    # rather than reaching across module boundaries via a hook
    # registration: the hook indirection was a transitional step,
    # not the design endpoint.  The helper is now invoked inline from
    # ``_bootstrap_session`` AFTER ``JaatoServer`` construction and
    # BEFORE ``_run_pre_initialize_hooks`` (which still fires for the
    # WS-side pre-init hook + any third-party hooks).
    #
    # The AppArmorManager + daemon_loop dependencies the relocated
    # method needs are wired at daemon startup via
    # :meth:`set_apparmor_dependencies` (called from
    # ``server/__main__.py``).  Tests construct a SessionManager
    # without these dependencies; the method is a clean no-op when
    # they're unset.

    def set_apparmor_dependencies(
        self,
        ws_server: Any = None,
        daemon_loop: Any = None,
        pool_manager: Any = None,
    ) -> None:
        """Wire the IPC AppArmor + runner-spawn dependencies (§3.13).

        Phase 3 §3.13.  Called once at daemon startup by
        ``server/__main__.py:JaatoDaemon.start``.  The relocated IPC
        apparmor logic in :meth:`_provision_ipc_apparmor_and_spawn_runner`
        reads these to:

        - Skip provisioning for sessions whose workspace lives under a
          running WS server's root (those are handled by the WS hook,
          not by us — avoids double-provisioning).
        - Pass the daemon's main asyncio loop to the
          :class:`AppArmorManager` so confined-worker mutations
          dispatch back to the unconfined main loop, AND to the
          :func:`spawn_session_runner` helper which runs
          ``RunnerRPCClient.start()`` async.

        Args:
            ws_server: Optional reference to the running WS server.
                ``None`` for IPC-only daemons.
            daemon_loop: The daemon's main asyncio loop.  ``None`` is
                accepted but disables apparmor provisioning (no
                AppArmor manager can be constructed without it).
            pool_manager: Pool PR 4 — the daemon's
                :class:`server.runner_pool.PoolManager`, threaded
                into :func:`spawn_session_runner` so IPC sessions can
                claim a pre-warm pool slot instead of cold-spawning
                a runner.  ``None`` disables the pool path for IPC
                sessions (cold-spawn fallback applies).
        """
        self._ws_server_ref = ws_server
        self._daemon_loop = daemon_loop
        self._pool_manager_ref = pool_manager

    def _provision_ipc_apparmor_and_spawn_runner(
        self,
        server: 'JaatoServer',
        session_id: str,
        workspace_path: Optional[str],
        client_id: Optional[str],
        *,
        apparmor_override: Optional[bool] = None,
        config_root_override: Optional[str] = None,
        env_file_override: Optional[str] = None,
        apparmor_fragments_override: Optional[List[str]] = None,
        cascade_driver_id: Optional[str] = None,
    ) -> Optional[str]:
        """Provision IPC AppArmor (opt-in) + spawn the per-session runner.

        Phase 3 §3.13 + §7a.  §3.13 relocated this from the legacy
        pre-init hook in ``server/__main__.py``.  §7a refactored the
        body into two composed helpers — apparmor provisioning is
        opt-in, but the runner spawn is unconditional (every IPC
        session with a workspace gets a runner; the runner-RPC
        dispatch surface is always available for the seat-flip's
        ``self._jaato.X`` migrations).

        Lifecycle:
        1. Skip when ``client_id`` is ``None`` (non-client paths
           supply their own bootstrap; this method is IPC-only).
        2. Skip when the session has no ``workspace_path`` — the
           runner needs a cwd; sessions without one don't get one
           (matches pre-§7a behavior).
        3. Skip when the session's workspace lives under a running
           WS server's workspace_root — the WS hook owns this
           lane; don't double-spawn.
        4. **Apparmor (opt-in)**: if ``client_config["apparmor"]``
           is set, call :meth:`_provision_apparmor_for_session` to
           load the profile.  Returns the resolved profile_name +
           sandbox_mode (``"apparmor"`` on success, ``"soft"`` on
           provisioning failure).
        5. **Spawn (unconditional)**: call
           :meth:`_spawn_session_runner_unconditional` with the
           profile_name from step 4 (or empty + disable_confine=True
           when no opt-in).  Spawn failure downgrades the return
           value to ``"soft"`` (or leaves it ``None`` for the
           no-opt-in case).

        Returns:
            The planned ``sandbox_mode`` for the Session record:
            ``"apparmor"`` (kernel-confined, runner spawned),
            ``"soft"`` (apparmor downgrade due to provisioning /
            spawn failure), or ``None`` (no apparmor opt-in or
            spawn was unconditional but unconfined — sandbox_mode
            stays None for the runner-without-confinement case
            since the field semantically tracks confinement, not
            runner presence).
        """
        # Phase 7a: only IPC client-driven sessions.
        # Non-client-driven bootstrap paths (loaded-from-disk,
        # ephemeral, standalone WS) supply their own bootstrap.
        if client_id is None:
            return None

        # PR-A (2026-05-14): resolve AppArmor opt-in across three
        # sources, in precedence order:
        #   1. ``apparmor_override`` kwarg — explicit caller intent
        #      from ``SessionManager.create_headless_session`` /
        #      reactor surface.  Wins outright when set.
        #   2. ``client_config["apparmor"]`` — IPC ClientConfigRequest
        #      opt-in (the existing pre-PR-A signal).  Used by the
        #      TUI / direct-IPC clients.
        #   3. ``server._profile.apparmor`` — profile-declared baseline.
        #      Default ``False`` today; PR-B will flip the default to
        #      ``True`` after kb-enablement-2.0 has validated the field.
        #
        # The two earlier sources are "explicit yes/no"; the profile
        # field is the "what does this profile prefer" baseline.  A
        # TUI user setting ``--apparmor`` on the wire still wins; a
        # reactor passing ``apparmor=True`` via kwarg still wins.
        client_config = self._client_config.get(client_id, {})
        client_config_signal = client_config.get("apparmor")
        profile = getattr(server, "_profile", None)
        profile_apparmor = bool(getattr(profile, "apparmor", False)) if profile else False
        if apparmor_override is not None:
            opt_in_apparmor = bool(apparmor_override)
        elif client_config_signal is not None:
            opt_in_apparmor = bool(client_config_signal)
        else:
            opt_in_apparmor = profile_apparmor

        # 2026-05-14 unification: resolve ``config_root`` + ``env_file``
        # for the AppArmor policy generator across all entry points.
        # Caller-supplied envelope override wins (used by headless +
        # disk-restore + ephemeral subagent paths that don't populate
        # ``client_config``); otherwise fall back to the IPC
        # ``client_config`` (populated by ``ClientConfigRequest``);
        # otherwise ``None`` (policy template's ``{config_root_rules}``
        # renders empty).  Pre-unification, this helper read ONLY from
        # ``client_config[client_id]`` — closes the v76 reactor-spawned
        # cascade crash class where headless callers couldn't supply
        # config_root to the policy generator.  See
        # ``project_backlog_apparmor_config_root_threading_for_headless``.
        if config_root_override is not None:
            config_root = config_root_override
        else:
            config_root = client_config.get("config_root")
        if env_file_override is not None:
            env_file = env_file_override
        else:
            env_file = client_config.get("env_file")

        # Piece 1 (2026-05-14): per-profile apparmor fragment scoping.
        # Resolution: explicit kwarg override wins; otherwise read
        # from the resolved profile's ``apparmor_fragments`` field
        # (already inheritance-merged with child-replaces semantics
        # at profile-load time).  ``None`` (default) preserves the
        # pre-Piece-1 "compose all fragments" behaviour;
        # non-``None`` (including ``[]``) restricts the compose set.
        if apparmor_fragments_override is not None:
            requested_fragments: Optional[List[str]] = list(apparmor_fragments_override)
        else:
            profile_fragments = getattr(profile, "apparmor_fragments", None) if profile else None
            requested_fragments = list(profile_fragments) if profile_fragments is not None else None

        # Phase 0 (template v20, 2026-05-16): plugin-contribution hook.
        # Walk ``profile.plugins`` to resolve each plugin's class via the
        # server registry, then call its optional
        # ``get_apparmor_rules`` classmethod with session context.
        # ``None`` when no profile is set OR no plugin contributes —
        # ``_render_profile`` treats both as "no contributions".  See
        # docs/design/plugin-apparmor-contribution.md.
        from server.apparmor import resolve_plugin_apparmor_rules
        plugin_rules = resolve_plugin_apparmor_rules(
            server=server,
            profile=profile,
            session_id=session_id,
            workspace_path=workspace_path,
            config_root=config_root,
        )

        # Spawn requires a workspace (cwd target).  Sessions without
        # one don't get a runner; pre-§7a behavior preserved.
        if not workspace_path:
            if opt_in_apparmor:
                # Surface the apparmor downgrade if confinement was
                # requested.  Silent skip otherwise — most IPC sessions
                # without workspace are short-lived / headless and don't
                # expect a runner.
                self._notify_apparmor(
                    client_id, session_id,
                    "requested but session has no workspace_path — "
                    "running unconfined",
                    style="warning",
                )
            return None

        # WS-overlap precedence: WS hook handles confinement +
        # runner spawn for sessions whose workspace is under the
        # WS server's root.  Skip both apparmor and spawn here so
        # we don't duplicate.
        if self._workspace_under_ws_root(workspace_path):
            return None

        # ----- Step 4: apparmor (opt-in) -----
        profile_name = ""
        sandbox_mode: Optional[str] = None
        if opt_in_apparmor:
            profile_name, sandbox_mode = self._provision_apparmor_for_session(
                session_id=session_id,
                workspace_path=workspace_path,
                client_id=client_id,
                config_root=config_root,
                env_file=env_file,
                requested_fragments=requested_fragments,
                plugin_rules=plugin_rules,
            )
            if profile_name == "" and sandbox_mode == "soft":
                # Apparmor unavailable / provisioning failed.
                # Continue to the unconditional spawn but the
                # runner is unconfined — that's the §7a intent
                # (always have a runner; confinement is layered).
                pass

        # Seed client-provided ("host") tool SCHEMAS the transport buffered for
        # this client BEFORE session.new, so spawn_session_runner's
        # envelope.client_tools sees them.  The transport registers the proxy
        # EXECUTORS post-session.new (execution-side, before the model's first
        # turn) — only the schemas must beat the spawn.  Fixes the
        # spawn-vs-buffered-apply race in PR #349 (peer e2e 2026-06-21).
        for _ct in self._pending_client_tools.pop(client_id, []):
            _ctn = _ct.get("name")
            if _ctn:
                server.client_tool_schemas[_ctn] = {
                    "name": _ctn,
                    "description": _ct.get("description", ""),
                    "parameters": _ct.get("parameters", {}),
                    "category": _ct.get("category", ""),
                }

        # ----- Step 5: spawn (unconditional) -----
        spawn_ok = self._spawn_session_runner_unconditional(
            server=server,
            session_id=session_id,
            workspace_path=workspace_path,
            client_id=client_id,
            profile_name=profile_name,
            cascade_driver_id=cascade_driver_id,
        )
        if not spawn_ok:
            # Spawn failed.  If apparmor was opted-in, downgrade
            # to "soft"; otherwise the session continues with
            # in-process tool execution (no sandbox_mode change).
            if opt_in_apparmor:
                return "soft"
            return None

        # ----- Step 5b: success notification + return -----
        if opt_in_apparmor and sandbox_mode == "apparmor":
            # ``config_root`` is the resolved value from above (envelope
            # override first, then client_config); the notification
            # reflects what the policy was actually generated with.
            self._notify_apparmor(
                client_id, session_id,
                f"profile provisioned (workspace={workspace_path}, "
                f"config_root={config_root or '(none)'}); runner spawned",
                style="info",
            )
            return "apparmor"
        if opt_in_apparmor and sandbox_mode == "soft":
            # Apparmor opt-in but provisioning failed; runner
            # spawned anyway (always-spawn).
            return "soft"
        # No apparmor opt-in: runner spawned unconfined.
        # sandbox_mode stays None (semantically tracks confinement
        # — there's none here, even though there IS a runner).
        return None

    def _notify_apparmor(
        self,
        client_id: str,
        session_id: str,
        message: str,
        style: str,
    ) -> None:
        """Surface an apparmor-status line to the client.

        Helper extracted for reuse between the apparmor-provision
        path and the no-workspace warning path.  Routes via
        ``_emit_to_client`` directly because at the call point the
        Session record doesn't exist in ``_sessions`` yet (we're
        pre-init).  Failures are swallowed — emit must not break
        session creation.
        """
        from jaato_sdk.events import SystemMessageEvent
        logger.info("[apparmor] %s", message)
        try:
            self._emit_to_client(client_id, SystemMessageEvent(
                message=f"[apparmor] {message}",
                style=style,
            ))
        except Exception:
            logger.warning(
                "Failed to emit apparmor status event for %s",
                session_id, exc_info=True,
            )

    def _workspace_under_ws_root(self, workspace_path: str) -> bool:
        """Return True iff *workspace_path* is under a running WS
        server's ``_workspace_root``.

        Used by ``_provision_ipc_apparmor_and_spawn_runner`` to
        skip its work for sessions the WS hook owns — preventing
        double-provision + double-spawn.
        """
        ws_server = getattr(self, "_ws_server_ref", None)
        if ws_server is None:
            return False
        ws_root = getattr(ws_server, "_workspace_root", None)
        if not ws_root:
            return False
        try:
            ws_root_real = os.path.realpath(ws_root)
            sess_real = os.path.realpath(workspace_path)
            return (
                sess_real == ws_root_real
                or sess_real.startswith(ws_root_real + os.sep)
            )
        except OSError:
            return False

    def _provision_apparmor_for_session(
        self,
        session_id: str,
        workspace_path: str,
        client_id: str,
        *,
        config_root: Optional[str],
        env_file: Optional[str],
        requested_fragments: Optional[List[str]] = None,
        plugin_rules: Optional[List[str]] = None,
    ) -> "Tuple[str, Optional[str]]":
        """Provision the AppArmor profile for a session
        (Phase 3 §7a — opt-in only).

        Caller has already checked the apparmor opt-in flag and
        the workspace gates.  This method does only the apparmor
        lifecycle: lazy-init manager + provision_profile.

        2026-05-14: signature unified to take ``config_root`` +
        ``env_file`` as explicit kwargs.  Pre-fix the helper read
        these from ``client_config[client_id]``, which made the
        reactor-spawned headless path (no client_config) silently
        omit the ``{config_root_rules}`` placeholder and produce a
        profile that didn't grant reads on the repo-root
        ``.jaato/`` for the kb-enablement-2.0 "framework config at
        parent, sandbox inside" layout.  Now every entry point's
        envelope carries these fields explicitly; the helper reads
        the envelope's values uniformly.  See
        ``project_backlog_apparmor_config_root_threading_for_headless``.

        Returns:
            ``(profile_name, sandbox_mode)``:
            - ``("<name>", "apparmor")`` on success.
            - ``("", "soft")`` when AppArmor is unavailable on the
              host or provisioning failed.  Caller should still
              spawn the runner (with disable_confine=True) — that's
              the §7a always-spawn intent.
        """
        # Lazy-init the AppArmor manager.
        if getattr(self, "_apparmor_manager", None) is None:
            from server.apparmor import AppArmorManager
            daemon_loop = getattr(self, "_daemon_loop", None)
            self._apparmor_manager = AppArmorManager(
                workspace_root=workspace_path,
                loop=daemon_loop,
            )

        apparmor = self._apparmor_manager
        if not apparmor.is_available():
            self._notify_apparmor(
                client_id, session_id,
                "requested but AppArmor is unavailable on this "
                "host (non-Linux, kernel module not loaded, or "
                "apparmor_parser missing) — running unconfined",
                style="warning",
            )
            return "", "soft"

        if not apparmor.provision_profile(
            session_id,
            workspace_path,
            config_root=config_root,
            env_file=env_file,
            requested_fragments=requested_fragments,
            plugin_rules=plugin_rules,
        ):
            self._notify_apparmor(
                client_id, session_id,
                "profile provisioning failed (see daemon log) — "
                "running unconfined",
                style="warning",
            )
            return "", "soft"

        return apparmor.get_profile_name(session_id), "apparmor"

    def _teardown_prior_apparmor_profile_after_transition(
        self,
        *,
        server: 'JaatoServer',
        current_session_id: str,
        current_profile_name: str,
    ) -> None:
        """Phase 3 cascade-sharing: unload the prior session's
        apparmor profile after the runner has transitioned to the
        current session's profile.

        Called from :meth:`_spawn_session_runner_unconditional` right
        after :func:`dispatch_bootstrap_envelope` returns success.  By
        the time this fires, the runner's main thread has already
        called ``aa_change_profile(current_profile_name)`` (bootstrap
        step 1c, re-entry path) — so the prior session's profile is
        no longer the runner's active profile and is safe to unload.

        No-op when:
          - ``current_profile_name`` is empty (operator opted out of
            apparmor; no transition occurred; nothing to unload).
          - The session was NOT pool-served (no ``pool_slot`` on the
            SpawnedRunner; this is the first session in the runner's
            life, no prior profile exists).
          - ``slot.last_session_id`` is unset (first cascade session
            on this slot — no prior profile to unload).
          - ``slot.last_session_id == current_session_id`` (defensive
            self-check; shouldn't happen because session_ids are
            unique).

        Best-effort: any unload failure (EBUSY because of lingering
        references, apparmor unavailable, etc.) is logged at WARNING
        and tolerated.  The cascade-idle teardown sweep will reap
        leftover profiles when the slot itself is torn down.
        """
        if not current_profile_name:
            return
        spawned = getattr(server, "_spawned_runner", None)
        pool_slot = getattr(spawned, "pool_slot", None) if spawned else None
        if pool_slot is None:
            return
        prior_session_id = pool_slot.last_session_id
        if not prior_session_id or prior_session_id == current_session_id:
            return

        apparmor = getattr(self, "_apparmor_manager", None)
        if apparmor is None or not apparmor.is_available():
            return

        try:
            ok = apparmor.teardown_profile(prior_session_id)
            if ok:
                logger.info(
                    "AppArmor: cascade-sharing transition complete — "
                    "unloaded prior profile for session=%s after slot "
                    "transitioned to session=%s",
                    prior_session_id, current_session_id,
                )
            else:
                logger.warning(
                    "AppArmor: cascade-sharing transition — "
                    "teardown_profile returned False for prior "
                    "session=%s (current=%s); kernel may EBUSY or "
                    "the profile file is already removed",
                    prior_session_id, current_session_id,
                )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "AppArmor: cascade-sharing teardown_profile raised "
                "for prior session=%s (current=%s): %s — leaving the "
                "kernel profile loaded; cascade-idle sweep will reap",
                prior_session_id, current_session_id, exc,
            )

    def _spawn_session_runner_unconditional(
        self,
        server: 'JaatoServer',
        session_id: str,
        workspace_path: str,
        client_id: str,
        profile_name: str,
        cascade_driver_id: Optional[str] = None,
    ) -> bool:
        """Spawn the per-session runner subprocess (Phase 3 §7a —
        always-called for IPC sessions with a workspace).

        Args:
            server: The session's JaatoServer instance.
            session_id: Session identifier.
            workspace_path: Session workspace (the runner's cwd).
            client_id: For warning notifications on failure.
            profile_name: AppArmor profile to self-confine to.
                Empty string ``""`` means run unconfined (no
                apparmor opt-in or provisioning failed) — the
                spawn helper passes ``disable_confine=True`` to
                ``RunnerSpawner``.

        Returns:
            True on successful spawn; False on failure.  Caller
            decides whether to downgrade ``sandbox_mode`` based on
            the apparmor opt-in flag.
        """
        try:
            from server.runner_spawn import (
                spawn_session_runner,
                dispatch_bootstrap_envelope,
            )

            # Profile-driven cgroup confinement (parity with the WS path,
            # websocket.py:661).  Historically only the WS + isolated-runner
            # paths provisioned a per-session cgroup, so a main IPC session got
            # AppArmor confinement (opt-in) but no cgroup — and the cgroup-nft
            # egress layer (§5.11d-v2) rides on the cgroup.  Here we make
            # confinement follow the PROFILE, not the transport: if the
            # session's profile declares runtime_limits and cgroups are
            # available, provision a per-session cgroup and pass the attach
            # callback so the runner migrates into it at fork().
            #
            # Tradeoff: passing a non-None cgroup_attach bypasses the pre-warm
            # pool (spawn_session_runner routes to cold-spawn when
            # cgroup_attach is not None — pool slots are forked from a shared
            # template and can't be migrated mid-life yet).  So only profiles
            # that declare runtime_limits pay the cold-spawn cost; profiles
            # without limits are unchanged (cgroup_attach stays None -> pool).
            cgroup_attach = None
            cgroup_profile = getattr(server, "_profile", None)
            cgroup_limits = (
                getattr(cgroup_profile, "runtime_limits", None)
                if cgroup_profile else None
            )
            if cgroup_limits is not None:
                cgroups_manager = self._resolve_cgroups_manager()
                if cgroups_manager is not None and cgroups_manager.is_available():
                    try:
                        if cgroups_manager.provision_cgroup(session_id, cgroup_limits):
                            cgroup_attach = cgroups_manager.make_attach_callback(
                                session_id)
                            if cgroup_limits.has_kernel_limits():
                                logger.info(
                                    "Cgroup limits applied to IPC session %s "
                                    "(memory=%s pids=%s cpu_weight=%s)",
                                    session_id, cgroup_limits.memory_max_mb,
                                    cgroup_limits.pids_max, cgroup_limits.cpu_weight,
                                )
                    except Exception as exc:  # noqa: BLE001 — best-effort
                        logger.warning(
                            "Cgroup provisioning failed for IPC session %s "
                            "(%s: %s) — runner spawns in the daemon's cgroup",
                            session_id, type(exc).__name__, exc,
                        )

            # Cascade budget: hand the pool to the envelope builder the same
            # way cascade_driver_id already reaches it — via the server
            # object — so no signature threads through three helpers.  The
            # builder does the min(profile, cascade_remaining) clamp and
            # raises CascadeExhaustedError when there is no headroom.
            server._cascade_budget_pool = self.get_cascade_budget(
                cascade_driver_id)
            # Reconcile the pool against every live sibling's TRACKER before
            # the clamp is computed.  Incremental accumulation from
            # TurnCompletedEvent is a best-effort live view, but the event
            # stream has proven both duplicable and droppable, so the number
            # the clamp is computed FROM is refreshed from the authoritative
            # per-session tracker at the one moment it has to be right.
            self._reconcile_cascade_pool(cascade_driver_id)

            # Pre-flight the cascade clamp HERE rather than letting it raise
            # inside the envelope builder.  Raising there produced a correct
            # daemon-side log and a correct SessionTerminatedEvent, but the
            # driver observed only a session id, silence, and its own
            # 120s turn timeout — indistinguishable from a hung daemon, which
            # wants the opposite response to a budget refusal.  Refusing at
            # the spawn boundary lets the requesting client be told
            # synchronously, with the framework's own evidence attached.
            # Does this child draw on the shared pot, or does it have its
            # own books?  A child that declared a budget is a delegation to
            # another department: its spend is accounted separately, it is
            # not clamped, and an exhausted pot does not refuse it.
            _own_budget = getattr(
                getattr(server, "_profile", None), "budget_control", None)
            server._draws_on_parent_budget = _own_budget is None

            _pool = server._cascade_budget_pool
            if _pool is not None:
                from shared.budget_control import CascadeExhaustedError
                try:
                    _pool.child_config(
                        getattr(server, "_profile", None)
                        and getattr(server._profile, "budget_control", None))
                except CascadeExhaustedError as exc:
                    _sess = self._sessions.get(session_id)
                    self._emit_cascade_refusal(
                        client_id, session_id, exc,
                        request_id=getattr(_sess, "create_request_id", None))
                    return False

            spawn_session_runner(
                server=server,
                session_id=session_id,
                workspace_path=workspace_path,
                profile_name=profile_name,
                daemon_loop=getattr(self, "_daemon_loop", None),
                disable_confine=(profile_name == ""),
                cgroup_attach=cgroup_attach,
                pool_manager=getattr(self, "_pool_manager_ref", None),
                cascade_driver_id=cascade_driver_id,
            )
            # Phase 3 post-Step-7 regression fix (Path B):
            # synchronously dispatch ``session.bootstrap`` so the
            # runner-side ``JaatoSession`` host is populated BEFORE
            # ``server.initialize()`` runs.  Pre-fix the IPC path
            # spawned the runner but never sent the bootstrap RPC,
            # leaving ``RunnerRPC._session_host = None`` for the
            # session lifetime.  Every daemon-side
            # ``self._runner_rpc.session_X_threadsafe()`` call
            # then raced against an unbootstrapped runner-side
            # session — handler correctly returned
            # ``stage="no_session"`` per the §3.3c surface
            # defensive contract, but the wrapper raised
            # ``RunnerCallError`` and crashed daemon-side init.
            #
            # The WS path has shipped this synchronous dispatch
            # since §7c step 2 (commit 6e31d375 era) at
            # ``websocket.py:690``.  This commit mirrors that
            # pattern for IPC sessions, closing the structural
            # asymmetry.
            #
            # Inline try/except so a bootstrap-dispatch hiccup
            # doesn't roll back the spawn-success return — spawn
            # already succeeded by this point.  Bootstrap-RPC
            # failures log WARNING via ``dispatch_bootstrap_envelope``
            # itself; the inline guard here additionally tolerates
            # ``server.runner_rpc`` being absent (test-stub
            # JaatoServer fakes that don't fully replicate the
            # post-spawn attribute set).
            try:
                dispatch_bootstrap_envelope(
                    server=server,
                    session_id=session_id,
                    workspace_path=workspace_path,
                    profile_name=profile_name,
                )
                # Phase 3 cascade-sharing: if this session inherited a
                # pool slot that previously served session
                # ``slot.last_session_id``, the runner just transitioned
                # away from that prior session's profile via
                # aa_change_profile (in the bootstrap step 1c re-entry
                # path).  The prior profile is now unreferenced — unload
                # it via apparmor_parser --remove so the kernel doesn't
                # accumulate stale per-session profiles across the
                # cascade's lifetime.  Best-effort: failure (EBUSY,
                # apparmor unavailable, etc.) logs a warning and
                # continues; the cascade-idle teardown sweep will reap
                # any leftovers.  Skipped for unconfined sessions
                # (profile_name empty) and non-pool sessions (no slot
                # reference to read last_session_id from).
                self._teardown_prior_apparmor_profile_after_transition(
                    server=server, current_session_id=session_id,
                    current_profile_name=profile_name,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "IPC session.bootstrap dispatch failed for %s "
                    "(%s: %s) — session will start with an "
                    "unbootstrapped runner-side host; downstream "
                    "session.* RPCs may race-fail until the "
                    "runner-side bootstrap completes some other way",
                    session_id, type(exc).__name__, exc,
                )
            # Phase 4 §4.3.3: wire the spawn_isolated_runner handler
            # with this SessionManager so the §4.3.7 opt-in branch
            # can spawn additional runners for isolated subagents.
            # The handler itself was registered earlier inside
            # ``JaatoServer.set_runner_rpc()`` (§4.3.2); here we
            # bridge it to our own ``_spawn_isolated_runner`` helper
            # by calling ``set_spawn_dependencies(self)``.  The
            # SessionManager reference isn't available at the
            # ``set_runner_rpc`` seam (would require widening
            # ``spawn_session_runner``'s signature) so the wire
            # lands here instead, where SessionManager is ``self``.
            # Best-effort: failure to wire the bridge logs WARNING
            # but doesn't roll back the spawn-success return — the
            # parent session is up and the runner is healthy; the
            # only impact is that ``agent_params.isolated=true``
            # subagents (§4.3.7) would get the "handler not yet
            # wired" stub envelope instead of the routed helper.
            handler = getattr(
                server, "_spawn_isolated_runner_handler", None,
            )
            if handler is not None:
                try:
                    handler.set_spawn_dependencies(session_manager=self)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    logger.warning(
                        "set_spawn_dependencies failed for session "
                        "%s (%s: %s) — agent_params.isolated=true "
                        "subagents will receive the handler-not-"
                        "wired stub until next session restart",
                        session_id, type(exc).__name__, exc,
                    )
            return True
        except Exception as exc:  # noqa: BLE001 — boundary
            self._notify_apparmor(
                client_id, session_id,
                f"runner spawn failed ({type(exc).__name__}: {exc}) "
                "— falling back to in-process tool execution; "
                "session is NOT kernel-confined",
                style="warning",
            )
            logger.exception(
                "runner spawn failed for session %s", session_id,
            )
            return False

    def _spawn_isolated_runner(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        profile_payload: Dict[str, Any],
        task: str,
        workspace_path: str,
        agent_params: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
        sub_profile_tightenings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Daemon-side helper for the isolated-subagent opt-in
        (Phase 4 §4.3.3).

        Invoked by ``SpawnIsolatedRunnerHandler.handle()`` (once
        ``set_spawn_dependencies(self)`` has been called) when a
        runner-side subagent plugin sends a
        ``subagent.spawn_isolated_runner`` RPC.  Per parent design
        §4.3, the supervisor's ``agent_params.isolated=true`` opt-in
        is meant to spawn the subagent in its own runner subprocess
        with a fresh AppArmor sub-profile
        (``jaato-ws-{parent}//{subagent}``) and its own sub-cgroup.

        §4.3.3 scope (THIS commit):

        1. Reconstruct ``SubagentProfile`` from the wire-shape
           ``profile_payload`` dict via ``build_inline_profile`` —
           same path SDK inline-spec callers use, so the JSON shape
           is identical to ``session.new`` inline specs.
        2. Generate the isolated session id
           (``{parent_session_id}__sub_{subagent_id}``).
        3. Return a ``stage="sub_profile"`` envelope with would-be
           values for diagnostic visibility — the next stage
           (sub-AppArmor profile generation + provisioning) lands
           in §4.3.4.

        Deferred to subsequent sub-commits:

        - §4.3.4: sub-AppArmor profile name generation + load.
        - §4.3.5: sub-cgroup creation + attach.
        - §4.3.6: cross-runner forwarding (prompt-forward INTO new
          runner; output-forward BACK to parent).
        - The actual ``_spawn_session_runner_unconditional(...)``
          call lands once both §4.3.4 and §4.3.5 are wired — passing
          ``profile_name=""`` today would spawn an unconfined
          sub-runner, which is a worse security posture than the
          §4.3 default-share path the §4.3.1 stub points callers
          to.  Refusing to spawn until the sub-profile is ready
          keeps the security gradient monotonic.

        Args:
            parent_session_id: Parent session's id.  Already echo-
                checked daemon-side by the handler; we re-use here
                only for the isolated session-id template.
            subagent_id: Pre-generated subagent id from the runner-
                side subagent plugin.
            profile_payload: Serialized ``SubagentProfile`` as a
                dict.  Field set mirrors ``build_inline_profile``'s
                contract (model, provider, plugins, plugin_configs,
                system_instructions, gc, env, runtime_limits,
                completion_payload_schema, spawn_payload_schema,
                completion_processors, model_tiers,
                suppress_base_instructions, max_turns).
            task: First-turn prompt for the isolated runner.  §4.3.3
                does NOT consume this; §4.3.6's forwarding will.
            workspace_path: Inherited from parent (§4.3 invariant).
                Forwarded for diagnostics only in §4.3.3.
            agent_params: Forwarded ``case_data``.  ``isolated`` key
                already stripped daemon-side by the handler.
            display_name: Custom display name; defaults to
                ``profile_payload.name``.
            parent_agent_id: For multi-hop subagent trees.

        Returns:
            Domain-failure envelope shape (parallels
            ``SpawnIsolatedRunnerHandler.handle``'s return):

                {"ok": False,
                 "error": "<reason>",
                 "stage": "validation" | "sub_profile" | "spawn" |
                          "sub_cgroup" | "forwarding",
                 # Diagnostic fields (would-be values for the next
                 # stage to debug against — only present when the
                 # stage advanced past the corresponding failure):
                 "isolated_session_id": "...",
                 "profile_name": "..."}

            On full success (post-§4.3.6 readiness):

                {"ok": True,
                 "session_id": "...",
                 "subagent_id": "...",
                 "runner_pid": <int>,
                 "apparmor_profile": "...",
                 "cgroup_path": "..."}
        """
        # ── Stage: validation — reconstruct SubagentProfile ────
        # ``build_inline_profile`` is the canonical "dict →
        # SubagentProfile" path used by SDK inline-spec session
        # creation; reuse here so the wire shape stays consistent
        # with what session.new accepts.  This also gives the
        # caller a precise validation-failure message if the
        # payload is malformed (e.g., bad gc / runtime_limits dict).
        try:
            from shared.plugins.subagent.config import build_inline_profile
            profile = build_inline_profile(
                profile_payload,
                name=profile_payload.get("name") or "<isolated>",
                description=profile_payload.get("description") or (
                    "Isolated subagent profile (Phase 4 §4.3 opt-in)"
                ),
            )
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "ok": False,
                "error": (
                    f"profile_payload reconstruction failed "
                    f"({type(exc).__name__}: {exc}).  Expected the "
                    f"same dict shape as session.new inline specs — "
                    f"see shared/plugins/subagent/config.py:"
                    f"build_inline_profile."
                ),
                "stage": "validation",
            }

        # ── Phase 5 §5.1: apply isolated-subagent runtime defaults ────
        # ``agent_params.isolated=true`` establishes the "isolation
        # implies bounds" invariant.  Profiles that omit
        # ``runtime_limits`` (or supply only a subset of fields) would
        # otherwise skip cgroup provision entirely and inherit the
        # daemon's default cgroup — the §4.3.9 item 1 hardening gap.
        # Merge supplied limits with the conservative default
        # (per-field; supplied wins on non-None) so the effective
        # value below carries the safety floor regardless of profile
        # content.  See
        # docs/design/phase5_5_1_isolated_default_runtime_limits_audit.md.
        effective_runtime_limits = apply_isolated_defaults(
            profile.runtime_limits,
        )
        if profile.runtime_limits != effective_runtime_limits:
            logger.info(
                "_spawn_isolated_runner: applied isolated-subagent "
                "runtime-limits defaults for parent=%s subagent=%s "
                "(supplied=%s, effective=%s)",
                parent_session_id, subagent_id,
                profile.runtime_limits, effective_runtime_limits,
            )

        # ── Stage: id generation ───────────────────────────────
        # Template ``{parent}__sub_{subagent}`` keeps the isolated
        # session id parseable + correlatable back to its parent
        # for log / trace inspection.  Same template used in
        # parent design §4.3's sub-profile name
        # (``jaato-ws-{session_id}//{subagent_id}``) so the two
        # stay in sync — when §4.3.4 generates the sub-profile
        # name, it can derive it from this session id.
        isolated_session_id = f"{parent_session_id}__sub_{subagent_id}"

        # ── Stage: sub_profile (next stage — §4.3.4) ───────────
        # Stop here.  Refusing to spawn until §4.3.4 provisions a
        # sub-AppArmor profile is intentional: the alternative
        # would be ``profile_name=""`` (unconfined sub-runner),
        # which weakens security relative to the §4.3 default-
        # share path callers can use today.  Monotonic security
        # gradient through the sub-track.
        # ── Stage: sub_profile — provision sub-AppArmor profile ──
        # Phase 4 §4.3.4: ask the daemon's AppArmorManager to write
        # + load a sub-profile named ``jaato-ws-{parent}//{subagent}``.
        # Standalone-with-prefix-name (not a true hat) per Audit 6.
        # When AppArmor isn't available (host doesn't support it, or
        # AppArmorManager isn't wired into this SessionManager
        # instance), we treat the sub-profile as absent and return
        # ``stage=sub_profile`` — the isolated-runner spawn requires
        # kernel confinement.
        apparmor_manager = self._resolve_apparmor_manager()
        if apparmor_manager is None or not apparmor_manager.is_available():
            return {
                "ok": False,
                "error": (
                    f"sub-AppArmor profile cannot be provisioned: "
                    f"AppArmorManager unavailable on this host.  "
                    f"Isolated-runner spawn requires kernel-level "
                    f"confinement.  Profile reconstruction succeeded "
                    f"(name={profile.name!r}, model={profile.model!r}).  "
                    f"Would-be isolated session: {isolated_session_id!r}.  "
                    f"Workaround: set agent_params.isolated=false (or "
                    f"omit) to use the default-share path (subagent "
                    f"runs in the parent's runner) — works end-to-end "
                    f"today.  See docs/design/phase4_implementation_audits.md."
                ),
                "stage": "sub_profile",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
            }

        ok, sub_profile_or_err = apparmor_manager.provision_sub_profile(
            parent_session_id=parent_session_id,
            subagent_id=subagent_id,
            workspace_path=workspace_path,
            tightenings=sub_profile_tightenings,
        )
        if not ok:
            logger.warning(
                "_spawn_isolated_runner: sub-profile provision failed "
                "for parent=%s subagent=%s: %s",
                parent_session_id, subagent_id, sub_profile_or_err,
            )
            return {
                "ok": False,
                "error": (
                    f"sub-AppArmor profile provision failed: "
                    f"{sub_profile_or_err}.  Profile reconstruction "
                    f"succeeded (name={profile.name!r}).  Would-be "
                    f"isolated session: {isolated_session_id!r}.  "
                    f"Workaround: omit agent_params.isolated."
                ),
                "stage": "sub_profile",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
            }

        sub_profile_name = sub_profile_or_err
        logger.info(
            "_spawn_isolated_runner: sub-profile provisioned for "
            "parent=%s subagent=%s (sub_profile=%s)",
            parent_session_id, subagent_id, sub_profile_name,
        )

        # ── Stage: sub_cgroup — provision sub-cgroup ───────────
        # Phase 4 §4.3.5: when cgroups available + profile declares
        # kernel-enforceable runtime_limits, create the sub-cgroup
        # via the existing CgroupsManager.provision_cgroup API
        # (passing isolated_session_id as session_id — the cgroup
        # path naturally lands at jaato-ws-{parent}__sub_{subagent}/
        # via get_cgroup_name's template).  Sibling-not-nested
        # structure per Audit 7 (cgroup v2 "no internal processes"
        # rule + design intent).
        #
        # Graceful degradation: cgroups unavailable on this host →
        # skip cgroup creation, sub-runner inherits daemon's default
        # cgroup.  AppArmor isolation still applies.
        #
        # Phase 5 §5.1: ``effective_runtime_limits`` now always carries
        # kernel-enforceable fields (memory_max_mb / pids_max /
        # cpu_weight) via :func:`apply_isolated_defaults`, so the
        # ``has_kernel_limits()`` predicate is True for the default.
        # The cgroup-skip branch can only fire when ``CgroupsManager``
        # is unavailable (kernel without cgroup v2, or daemon wired
        # without one), which is a host-capability gap rather than a
        # profile-content gap.
        cgroup_path = ""  # Empty = no sub-cgroup provisioned.
        cgroups_manager = self._resolve_cgroups_manager()
        runtime_limits = effective_runtime_limits
        if (cgroups_manager is not None
                and cgroups_manager.is_available()
                and runtime_limits.has_kernel_limits()):
            cgroup_ok = cgroups_manager.provision_cgroup(
                isolated_session_id, runtime_limits,
            )
            if not cgroup_ok:
                # Roll back the §4.3.4 sub-AppArmor profile before
                # returning — otherwise it remains kernel-loaded
                # but unused, and the next provision attempt with
                # the same subagent_id sees the loaded profile.
                # Best-effort: teardown failure logs but doesn't
                # change the §4.3.5 return shape.  Idempotent
                # re-load via provision_sub_profile would succeed
                # anyway, so a stuck-loaded profile is not blocking.
                try:
                    apparmor_manager.teardown_sub_profile(
                        parent_session_id=parent_session_id,
                        subagent_id=subagent_id,
                    )
                except Exception:  # noqa: BLE001 — best-effort
                    logger.exception(
                        "_spawn_isolated_runner: sub-AppArmor "
                        "rollback failed after sub-cgroup provision "
                        "failure for parent=%s subagent=%s",
                        parent_session_id, subagent_id,
                    )
                logger.warning(
                    "_spawn_isolated_runner: sub-cgroup provision "
                    "failed for parent=%s subagent=%s "
                    "(isolated_session_id=%s)",
                    parent_session_id, subagent_id, isolated_session_id,
                )
                return {
                    "ok": False,
                    "error": (
                        f"sub-cgroup provision failed for isolated "
                        f"session {isolated_session_id!r}.  Sub-AppArmor "
                        f"profile {sub_profile_name!r} rolled back.  "
                        f"Workaround: omit agent_params.isolated to "
                        f"use default-share path."
                    ),
                    "stage": "sub_cgroup",
                    "isolated_session_id": isolated_session_id,
                    "profile_name": profile.name,
                    "apparmor_profile": "",  # Rolled back.
                }
            cgroup_path = str(
                cgroups_manager.get_cgroup_path(isolated_session_id),
            )
            logger.info(
                "_spawn_isolated_runner: sub-cgroup provisioned for "
                "parent=%s subagent=%s (cgroup_path=%s)",
                parent_session_id, subagent_id, cgroup_path,
            )
            # Phase 5 §5.2: nesting-visibility instrumentation.
            # Sibling cgroup structure (today, per §4.3.5) means
            # sub bounds don't compose under the parent's.  When
            # sub's kernel-enforced cap exceeds parent's, true
            # nesting WOULD have capped at parent's bound — log
            # so operators get visibility into where a Phase 6
            # nested-layout migration would change behavior.
            # Observability-only; no behavior change.  See
            # docs/design/phase5_5_2_nested_cgroup_deferral_audit.md.
            self._log_nesting_visibility(
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                sub_limits=effective_runtime_limits,
            )
        else:
            logger.info(
                "_spawn_isolated_runner: sub-cgroup skipped for "
                "parent=%s subagent=%s "
                "(cgroups_available=%s, has_kernel_limits=%s) — "
                "sub-runner will inherit default cgroup",
                parent_session_id, subagent_id,
                cgroups_manager is not None
                and cgroups_manager.is_available(),
                runtime_limits.has_kernel_limits(),
            )

        # ── Stage: forwarding — spawn sub-runner subprocess ────
        # Phase 4 §4.3.6a: actually spawn the sub-runner with the
        # provisioned sub-AppArmor profile + (optional) sub-cgroup
        # attach.  Sub-runner self-confines via ``change_profile``
        # on spawn; pre-exec preexec_fn migrates into the sub-cgroup.
        #
        # Stays at ``stage=forwarding`` because cross-runner event
        # forwarding (sub→parent_runner) is §4.3.6b's job and the
        # first-turn prompt dispatch is §4.3.6c's job.  §4.3.6a
        # only proves the spawn happens + handle is bookkept.
        #
        # On spawn failure: roll back sub-cgroup + sub-AppArmor in
        # order to keep kernel state consistent (no orphaned
        # profile/cgroup if the subprocess never starts).
        try:
            sub_handle = self._do_spawn_isolated_runner(
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                isolated_session_id=isolated_session_id,
                workspace_path=workspace_path,
                sub_apparmor_profile=sub_profile_name,
                cgroup_path=cgroup_path,
                profile=profile,
                effective_runtime_limits=effective_runtime_limits,
                agent_params=agent_params,
            )
        except Exception as spawn_exc:  # noqa: BLE001 — boundary
            logger.warning(
                "_spawn_isolated_runner: subprocess spawn failed "
                "for parent=%s subagent=%s: %s",
                parent_session_id, subagent_id, spawn_exc,
                exc_info=True,
            )
            # Rollback chain — cgroup then AppArmor.  Best-effort;
            # rollback failures log but don't change return shape.
            self._rollback_isolated_resources(
                parent_session_id=parent_session_id,
                subagent_id=subagent_id,
                isolated_session_id=isolated_session_id,
                cgroup_path=cgroup_path,
            )
            return {
                "ok": False,
                "error": (
                    f"sub-runner subprocess spawn failed: "
                    f"{type(spawn_exc).__name__}: {spawn_exc}.  "
                    f"Sub-cgroup + sub-AppArmor profile rolled back.  "
                    f"Workaround: omit agent_params.isolated to use "
                    f"default-share path."
                ),
                "stage": "forwarding",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
                "apparmor_profile": "",  # Rolled back.
                "cgroup_path": "",       # Rolled back.
            }

        # Register the handle so §4.3.6b/c/d can find it by id.
        with self._lock:
            self._isolated_sub_runners[isolated_session_id] = sub_handle
        logger.info(
            "_spawn_isolated_runner: sub-runner spawned for "
            "parent=%s subagent=%s (isolated_session_id=%s "
            "pid=%d sub_profile=%s cgroup=%s)",
            parent_session_id, subagent_id, isolated_session_id,
            sub_handle.spawned.pid, sub_profile_name,
            cgroup_path or "(none)",
        )

        # ── Phase 4 §4.3.6c — session.bootstrap + first-turn dispatch ──
        # The sub-runner subprocess is up but its runner-side
        # JaatoSession isn't yet — dispatch session.bootstrap so
        # the sub-runner constructs its session host before we
        # send the first-turn prompt.  Bootstrap failure is
        # surfaced as a stage=forwarding error envelope; the
        # sub-runner is left registered (caller can attempt
        # explicit teardown).
        envelope = self._build_isolated_envelope(
            profile=profile,
            isolated_session_id=isolated_session_id,
            workspace_path=workspace_path,
            sub_apparmor_profile=sub_profile_name,
            agent_params=agent_params,
        )
        bootstrap_ok = self._dispatch_isolated_session_bootstrap(
            sub_handle, envelope,
        )
        if not bootstrap_ok:
            return {
                "ok": False,
                "error": (
                    f"sub-runner subprocess spawned but "
                    f"session.bootstrap dispatch failed.  Sub-runner "
                    f"will not run the task.  Sub-resources remain "
                    f"provisioned until parent teardown (§4.3.6d "
                    f"will add cascade)."
                ),
                "stage": "forwarding",
                "isolated_session_id": isolated_session_id,
                "profile_name": profile.name,
                "apparmor_profile": sub_profile_name,
                "cgroup_path": cgroup_path,
                "sub_runner_pid": sub_handle.spawned.pid,
                "sub_session_id": isolated_session_id,
            }

        # Schedule the first-turn prompt as a background task — the
        # sub-runner runs the task; streaming output forwards back
        # via the §4.3.6b chain.  Returns immediately so the caller
        # (subagent plugin) can continue.
        self._schedule_isolated_first_turn(sub_handle, task)

        # §4.3.6c milestone — isolated subagent is running end-to-end.
        # ok=True signals the supervisor that the spawn succeeded;
        # streaming output will arrive via inject_prompt over the
        # cross-runner chain.
        return {
            "ok": True,
            "session_id": isolated_session_id,
            "subagent_id": subagent_id,
            "runner_pid": sub_handle.spawned.pid,
            "apparmor_profile": sub_profile_name,
            "cgroup_path": cgroup_path,
            # Diagnostic fields kept for backward-compat with §4.3.6a/b
            # test assertions + audit-trail logging.
            "isolated_session_id": isolated_session_id,
            "profile_name": profile.name,
            "sub_runner_pid": sub_handle.spawned.pid,
            "sub_session_id": isolated_session_id,
        }

    def _do_spawn_isolated_runner(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        isolated_session_id: str,
        workspace_path: str,
        sub_apparmor_profile: str,
        cgroup_path: str,
        profile: Any,  # SubagentProfile
        effective_runtime_limits: RuntimeLimits,
        agent_params: Optional[Dict[str, Any]],
    ) -> SubRunnerHandle:
        """Spawn the sub-runner subprocess + initialize its RPC
        channel (Phase 4 §4.3.6a).

        Mirrors ``runner_spawn.spawn_session_runner`` but for the
        isolated-subagent path: the daemon doesn't have a full
        JaatoServer for the sub-session (and doesn't need one — the
        runner-side JaatoSession holds all the state).  We hold the
        spawn handles on a :class:`SubRunnerHandle` instead.

        Raises any exception from the spawn / RPC start path; caller
        catches + rolls back kernel resources.
        """
        import asyncio
        from server.runner_rpc_client import RunnerRPCClient
        from server.runner_spawner import RunnerSpawner

        daemon_loop = getattr(self, "_daemon_loop", None)
        if daemon_loop is None:
            raise RuntimeError(
                "_do_spawn_isolated_runner: daemon loop unavailable; "
                "cannot start RunnerRPCClient"
            )

        spawner = RunnerSpawner()
        log_path: Optional[str] = None
        if workspace_path:
            log_dir = os.path.join(workspace_path, ".jaato", "logs")
            log_path = os.path.join(
                log_dir, f"runner-{isolated_session_id}.log",
            )

        # Cgroup attach: when §4.3.5 provisioned a sub-cgroup, build
        # the preexec_fn that migrates the forked child in.  When no
        # cgroup, pass ``None`` — sub-runner inherits daemon's
        # default cgroup (the documented Phase 5+ hardening gap).
        cgroup_attach = None
        if cgroup_path:
            cgroups_manager = self._resolve_cgroups_manager()
            if cgroups_manager is not None:
                cgroup_attach = cgroups_manager.make_attach_callback(
                    isolated_session_id,
                )

        # Phase 5 §5.1: forward the app-layer caps via the
        # ``JAATO_RUNNER_MAX_OUTPUT_CHARS`` / ``JAATO_RUNNER_TOOL_TIMEOUT_SECONDS``
        # env-passthrough already wired in :class:`RunnerSpawner`.  The
        # runner-side cli plugin reads those env vars at startup.  Without
        # this hookup the kernel-layer defaults would apply (via cgroup)
        # but the application-layer defaults — tool wall-clock + output
        # truncation — would silently no-op on the sub-runner subprocess.
        spawned = spawner.spawn(
            profile_name=sub_apparmor_profile,
            session_id=isolated_session_id,
            workspace_path=workspace_path,
            log_path=log_path,
            max_output_chars=effective_runtime_limits.max_output_bytes,
            tool_timeout_seconds=effective_runtime_limits.tool_timeout_seconds,
            disable_confine=False,  # Always confined for §4.3.6.
            cgroup_attach=cgroup_attach,
        )

        rpc = RunnerRPCClient(
            spawned.parent_socket,
            runner_pid=spawned.pid,
            loop=daemon_loop,
        )
        fut = asyncio.run_coroutine_threadsafe(rpc.start(), daemon_loop)
        fut.result(timeout=10.0)

        return SubRunnerHandle(
            parent_session_id=parent_session_id,
            subagent_id=subagent_id,
            isolated_session_id=isolated_session_id,
            rpc=rpc,
            spawned=spawned,
            sub_apparmor_profile=sub_apparmor_profile,
            cgroup_path=cgroup_path,
        )

    def _build_isolated_envelope(
        self,
        *,
        profile: Any,  # SubagentProfile
        isolated_session_id: str,
        workspace_path: str,
        sub_apparmor_profile: str,
        agent_params: Optional[Dict[str, Any]],
    ) -> Any:
        """Build a :class:`SessionInitEnvelope` for an isolated
        subagent's runner-side bootstrap (Phase 4 §4.3.6c).

        Mirrors ``server.runner_spawn.build_session_envelope`` but
        sources fields from the reconstructed SubagentProfile
        directly (no daemon-side JaatoServer with ``_profile``
        attribute — isolated subagents don't get one).
        """
        from shared.session_envelope import SessionInitEnvelope

        provider_name = getattr(profile, "provider", None) or ""
        # Second envelope builder (isolated subagents have no daemon-side
        # JaatoServer).  Same binder as runner_spawn's, so a tiers-only
        # profile does not produce "envelope.model_name is empty" here either.
        from shared.model_tiers import bound_model_for_profile
        model_name = bound_model_for_profile(profile) or ""
        plugins_list = list(getattr(profile, "plugins", []) or [])
        preloaded = set(
            getattr(profile, "preloaded_plugins", set()) or set(),
        )
        # Server 0.6.123+: values flow through ``expand_plugin_configs``
        # so ``${VAR}`` references AND secret URIs (``pass://`` /
        # ``vault://``) resolve daemon-side before the isolated
        # subagent's runner sees them.  Same wire-gap class as PR #139
        # (completion_validators) and the v118 zhipuai diagnosis.
        # AppArmor-confined runners can't exec ``pass``, so resolution
        # must happen here.
        from shared.plugins.subagent.config import expand_plugin_configs
        raw_plugin_configs = dict(
            getattr(profile, "plugin_configs", {}) or {},
        )
        plugin_configs = expand_plugin_configs(
            raw_plugin_configs,
            workspace_root_override=workspace_path,
        )
        profile_tool_scopes = getattr(profile, "tool_scopes", {}) or {}
        plugin_specs = []
        for name in plugins_list:
            entry = {"name": name, "preload": name in preloaded}
            cfg = plugin_configs.get(name)
            if cfg:
                entry["config"] = dict(cfg)
            # Per-plugin tool allow-list (profile ``tools:[...]`` modifier)
            # — carried on the envelope entry so the isolated subagent's
            # runner-side bootstrap scopes its wire surface.
            scope = profile_tool_scopes.get(name)
            if scope:
                entry["tools"] = list(scope)
            plugin_specs.append(entry)

        system_instructions = getattr(profile, "system_instructions", None)
        gc_dict = None
        gc_obj = getattr(profile, "gc", None)
        if gc_obj is not None:
            gc_type = getattr(gc_obj, "type", None)
            gc_config = getattr(gc_obj, "config", None) or {}
            if gc_type:
                gc_dict = {"type": gc_type, **dict(gc_config)}
        env_overrides = dict(getattr(profile, "env", {}) or {})

        if not provider_name:
            provider_name = "anthropic"

        try:
            project_val = os.environ.get("PROJECT_ID", "") or ""  # env: GCP project ID for Google GenAI / Vertex AI
            location_val = os.environ.get("LOCATION", "") or ""  # env: Vertex AI region for Google GenAI (e.g. us-central1 or global)
        except Exception:  # noqa: BLE001
            project_val = ""
            location_val = ""

        # Envelope v5, mirroring build_session_envelope.  NOTE: this
        # envelope does not carry ``model_tiers`` (pre-existing gap), so an
        # isolated subagent gets action rungs (abort) but NOT tier-overlay
        # rungs — an overlay needs a tier table to patch.
        _iso_budget = getattr(profile, "budget_control", None)
        return SessionInitEnvelope(
            session_id=isolated_session_id,
            budget_control=_iso_budget.to_dict() if _iso_budget else None,
            workspace_path=workspace_path,
            profile_name=sub_apparmor_profile,
            provider_name=provider_name,
            model_name=model_name,
            plugins=plugin_specs,
            # Phase 4 §C (envelope schema v2): carry the full
            # profile.plugin_configs map at the top level so
            # auto-loaded plugins (permission, gc_*, etc.) that
            # aren't named in ``plugins`` still receive their
            # profile overrides on the runner side.
            plugin_configs=plugin_configs,
            system_instructions=system_instructions,
            agent_id="main",
            gc=gc_dict,
            agent_params=dict(agent_params or {}),
            config_root=None,  # Isolated subagent doesn't inherit.
            env_overrides=env_overrides,
            project=project_val,
            location=location_val,
            completion_payload_schema=getattr(
                profile, "completion_payload_schema", None,
            ),
            completion_processors=[
                {
                    "script": getattr(p, "script", None),
                    "output": getattr(p, "output", None),
                    "on_error": getattr(p, "on_error", "fail_completion"),
                    "description": getattr(p, "description", None),
                    "phase": getattr(p, "phase", "finalization"),
                }
                for p in (getattr(profile, "completion_processors", []) or [])
                if hasattr(p, "script")
            ],
        )

    def _dispatch_isolated_session_bootstrap(
        self,
        sub_handle: SubRunnerHandle,
        envelope: Any,  # SessionInitEnvelope
        *,
        timeout: float = 30.0,
    ) -> bool:
        """Send the ``session.bootstrap`` RPC to the sub-runner so
        its runner-side JaatoSession host is populated before the
        first-turn prompt arrives (Phase 4 §4.3.6c).

        Synchronous: blocks until the sub-runner acknowledges.
        Returns True on success, False on failure (logged).
        """
        try:
            result = sub_handle.rpc.bootstrap_session_threadsafe(
                envelope, timeout=timeout,
            )
            logger.info(
                "isolated session.bootstrap acknowledged for %s: %s",
                sub_handle.isolated_session_id, result,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "isolated session.bootstrap failed for %s: %s",
                sub_handle.isolated_session_id, exc,
                exc_info=True,
            )
            return False

    def _schedule_isolated_first_turn(
        self,
        sub_handle: SubRunnerHandle,
        task: str,
    ) -> None:
        """Schedule the first-turn prompt dispatch as an async task
        on the daemon loop (Phase 4 §4.3.6c).

        Returns immediately — the actual ``session.send_message``
        runs in the background.  Output streams back to the parent
        runner via the §4.3.6b ``subagent.forward_event`` RPC chain
        triggered by the ``on_output`` / ``on_notification``
        callbacks registered here.

        On completion: forwards final response (as event_kind=output)
        + status=done.
        On exception: forwards status=error with the exception message.
        """
        import asyncio

        daemon_loop = getattr(self, "_daemon_loop", None)
        if daemon_loop is None:
            logger.warning(
                "_schedule_isolated_first_turn: no daemon loop; "
                "sub-runner %s will not receive task",
                sub_handle.isolated_session_id,
            )
            self._forward_subagent_event_to_parent(
                sub_handle, "error",
                {"message": "no daemon loop to dispatch first-turn prompt"},
            )
            return

        # on_output: forward streaming text chunks to parent.
        def on_output(source: str, text: str, mode: str) -> None:
            try:
                self._forward_subagent_event_to_parent(
                    sub_handle, "output",
                    {"text": text, "source": source},
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "on_output forwarding failed for %s",
                    sub_handle.isolated_session_id,
                )

        # on_notification: lifecycle notifications (instruction
        # budget, retry, etc.).  Phase 4 §4.3.6c forwards them
        # as status events; finer-grained kinds are Phase 5+.
        def on_notification(event_type: str, payload: Any) -> None:
            try:
                self._forward_subagent_event_to_parent(
                    sub_handle, "status",
                    {
                        "status": str(event_type),
                        "payload": payload if isinstance(payload, dict) else {},
                    },
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "on_notification forwarding failed for %s",
                    sub_handle.isolated_session_id,
                )

        async def _run_first_turn():
            try:
                response = await sub_handle.rpc.session_send_message(
                    prompt=task,
                    on_output=on_output,
                    on_notification=on_notification,
                )
                # Final response — forward as terminal output.
                self._forward_subagent_event_to_parent(
                    sub_handle, "output",
                    {"text": response, "source": "final"},
                )
                self._forward_subagent_event_to_parent(
                    sub_handle, "status",
                    {"status": "done"},
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "_run_first_turn: session.send_message failed for %s",
                    sub_handle.isolated_session_id,
                )
                self._forward_subagent_event_to_parent(
                    sub_handle, "error",
                    {"message": f"{type(exc).__name__}: {exc}"},
                )

        asyncio.run_coroutine_threadsafe(_run_first_turn(), daemon_loop)

    def _forward_subagent_event_to_parent(
        self,
        sub_handle: SubRunnerHandle,
        event_kind: str,
        event_payload: Dict[str, Any],
        *,
        timeout: float = 5.0,
    ) -> bool:
        """Forward a sub-runner event to the parent runner via the
        ``subagent.forward_event`` RPC (Phase 4 §4.3.6b).

        Looks up the parent session's runner-RPC handle (from the
        parent's JaatoServer in ``self._sessions``) and dispatches
        the event.  Runner-side handler routes to the SubagentPlugin's
        ``receive_forwarded_event``, which calls ``inject_prompt``
        on the parent session.

        Args:
            sub_handle: The sub-runner's :class:`SubRunnerHandle`.
                Provides ``parent_session_id`` for parent lookup.
            event_kind: ``"output"`` | ``"status"`` | ``"error"``.
            event_payload: Event-kind-specific payload dict.
            timeout: Wall-clock cap for the RPC.  Default 5s — the
                runner-side handler is fast (lookup + inject_prompt
                + return), no plugin discovery / provider connect.

        Returns:
            True when the forward succeeded; False when the parent
            runner is unavailable / the RPC failed / the plugin
            rejected the event.  Caller logs but doesn't retry —
            cross-runner forwarding is best-effort (CLAUDE.md §4.4).

        Phase 4 §4.3.6b: this is the daemon-side helper.  §4.3.6c
        wires the subscription that triggers it via the first-turn
        ``session.send_message`` call's ``on_notification`` callback.
        """
        with self._lock:
            parent_session = self._sessions.get(sub_handle.parent_session_id)
        if parent_session is None:
            logger.warning(
                "_forward_subagent_event_to_parent: parent session %s "
                "not found; dropping event_kind=%s for subagent_id=%s",
                sub_handle.parent_session_id, event_kind,
                sub_handle.subagent_id,
            )
            return False

        parent_runner_rpc = getattr(
            parent_session.server, "runner_rpc", None,
        )
        if parent_runner_rpc is None:
            logger.warning(
                "_forward_subagent_event_to_parent: parent session %s "
                "has no runner_rpc; dropping event_kind=%s for "
                "subagent_id=%s",
                sub_handle.parent_session_id, event_kind,
                sub_handle.subagent_id,
            )
            return False

        try:
            env = parent_runner_rpc.call_threadsafe(
                "subagent.forward_event",
                {
                    "subagent_id": sub_handle.subagent_id,
                    "event_kind": event_kind,
                    "event_payload": event_payload,
                },
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_forward_subagent_event_to_parent: RPC failed for "
                "subagent_id=%s event_kind=%s: %s",
                sub_handle.subagent_id, event_kind, exc,
            )
            return False

        if not env.ok:
            err_msg = (
                env.error.message if env.error else "no error message"
            )
            logger.warning(
                "_forward_subagent_event_to_parent: handler rejected "
                "subagent_id=%s event_kind=%s: %s",
                sub_handle.subagent_id, event_kind, err_msg,
            )
            return False
        return True

    def _cascade_teardown_isolated_subagents(
        self,
        parent_session_id: str,
    ) -> int:
        """Tear down every isolated sub-runner owned by a parent
        session (Phase 4 §4.3.6d).

        Called when a parent session is unloaded or shut down.
        Iterates ``_isolated_sub_runners`` filtering by
        ``parent_session_id``; for each handle:
          1. Close the sub-runner's RPC client (sends EOF, waits
             for the sub-runner subprocess to exit).
          2. Tear down the sub-cgroup (``cgroup.kill`` atomic
             termination, then rmdir).
          3. Tear down the sub-AppArmor profile (``apparmor_parser -R``).
          4. Remove the handle from ``_isolated_sub_runners``.

        Best-effort throughout — each step wrapped in try/except so
        a single teardown failure doesn't strand the others.  Returns
        the count of handles torn down for logging / metrics.

        Args:
            parent_session_id: Parent session whose isolated
                sub-runners should be torn down.

        Returns:
            Number of handles processed (whether or not each
            individual teardown step succeeded).
        """
        import asyncio

        with self._lock:
            owned_handles = [
                handle for handle in self._isolated_sub_runners.values()
                if handle.parent_session_id == parent_session_id
            ]

        if owned_handles:
            logger.info(
                "_cascade_teardown_isolated_subagents: tearing down "
                "%d sub-runner(s) for parent_session=%s",
                len(owned_handles), parent_session_id,
            )
        # Phase 5 §5.3: don't short-circuit on empty owned_handles —
        # the orphan-scan tail still runs to catch kernel state left
        # behind by rollback failures or crashes that never registered
        # a handle.  Per the ledger, the leak audit runs at every
        # parent teardown, not only when known handles existed.

        for handle in owned_handles:
            # 1. Close RPC client (signals EOF; sub-runner exits).
            try:
                daemon_loop = getattr(self, "_daemon_loop", None)
                if daemon_loop is not None and hasattr(handle.rpc, "close"):
                    fut = asyncio.run_coroutine_threadsafe(
                        handle.rpc.close(), daemon_loop,
                    )
                    try:
                        fut.result(timeout=5.0)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "cascade teardown: RPC close timed out "
                            "for %s — continuing",
                            handle.isolated_session_id,
                        )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "cascade teardown: RPC close failed for %s",
                    handle.isolated_session_id,
                )

            # 2. Tear down sub-cgroup (if one was provisioned).
            if handle.cgroup_path:
                try:
                    cgroups_manager = self._resolve_cgroups_manager()
                    if cgroups_manager is not None:
                        cgroups_manager.teardown_cgroup(
                            handle.isolated_session_id,
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "cascade teardown: sub-cgroup teardown "
                        "failed for %s",
                        handle.isolated_session_id,
                    )

            # 3. Tear down sub-AppArmor profile (if one was loaded).
            if handle.sub_apparmor_profile:
                try:
                    apparmor_manager = self._resolve_apparmor_manager()
                    if apparmor_manager is not None:
                        apparmor_manager.teardown_sub_profile(
                            parent_session_id=handle.parent_session_id,
                            subagent_id=handle.subagent_id,
                        )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "cascade teardown: sub-AppArmor teardown "
                        "failed for %s",
                        handle.isolated_session_id,
                    )

            # 4. Remove from registry.
            with self._lock:
                self._isolated_sub_runners.pop(
                    handle.isolated_session_id, None,
                )
            logger.info(
                "cascade teardown: completed for %s",
                handle.isolated_session_id,
            )

        # Phase 5 §5.3 — orphan sub-cgroup reaper.  After the known-
        # handles loop completes, scan for sub-cgroups under this
        # parent that exist on disk but aren't in the just-torn-down
        # set.  Sources of orphans: rollback failure mid-teardown
        # (e.g., transient EBUSY before _rollback_isolated_resources'
        # teardown_cgroup call) and mid-spawn crashes that left
        # kernel state without a corresponding handle.  Reap via the
        # existing teardown_cgroup path so behaviour matches known
        # handles exactly.  Best-effort: failures log WARNING but
        # don't change the cascade return value (which is the count
        # of HANDLES torn down, not orphans reaped).  See
        # docs/design/phase5_5_3_cgroup_leak_audit_audit.md.
        torn_down_ids = {h.isolated_session_id for h in owned_handles}
        try:
            cgroups_manager = self._resolve_cgroups_manager()
            if cgroups_manager is not None:
                orphans = cgroups_manager.list_orphan_sub_cgroups(
                    parent_session_id, torn_down_ids,
                )
                for orphan_id in orphans:
                    logger.warning(
                        "cascade teardown: reaped orphaned sub-cgroup "
                        "%s — likely cause: rollback or mid-spawn crash",
                        orphan_id,
                    )
                    try:
                        cgroups_manager.teardown_cgroup(orphan_id)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "cascade teardown: orphan reap failed "
                            "for %s — manual cleanup may be required",
                            orphan_id,
                        )
        except Exception:  # noqa: BLE001
            logger.exception(
                "cascade teardown: orphan scan failed for "
                "parent_session=%s — no orphans reaped",
                parent_session_id,
            )

        return len(owned_handles)

    def _rollback_isolated_resources(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        isolated_session_id: str,
        cgroup_path: str,
    ) -> None:
        """Tear down sub-cgroup + sub-AppArmor on §4.3.6a spawn
        failure (Phase 4 §4.3.6a).

        Best-effort: each teardown wrapped in try/except.  Rollback
        failures log but don't propagate — the helper's return
        shape is what matters to callers.
        """
        # Cgroup first (innermost resource).
        if cgroup_path:
            try:
                cgroups_manager = self._resolve_cgroups_manager()
                if cgroups_manager is not None:
                    cgroups_manager.teardown_cgroup(isolated_session_id)
            except Exception:  # noqa: BLE001 — best-effort
                logger.exception(
                    "_rollback_isolated_resources: sub-cgroup "
                    "teardown failed for parent=%s subagent=%s",
                    parent_session_id, subagent_id,
                )

        # AppArmor next.
        try:
            apparmor_manager = self._resolve_apparmor_manager()
            if apparmor_manager is not None:
                apparmor_manager.teardown_sub_profile(
                    parent_session_id=parent_session_id,
                    subagent_id=subagent_id,
                )
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "_rollback_isolated_resources: sub-AppArmor "
                "teardown failed for parent=%s subagent=%s",
                parent_session_id, subagent_id,
            )

    def _log_nesting_visibility(
        self,
        *,
        parent_session_id: str,
        subagent_id: str,
        sub_limits: RuntimeLimits,
    ) -> None:
        """Phase 5 §5.2 — log when sub-cgroup bounds would have been
        capped by parent under a nested-layout migration.

        Today's sub-cgroup is sibling to the parent's (per §4.3.5
        + cgroup v2 "no internal processes" rule); their kernel
        limits don't compose.  When the supervisor declares an
        isolated sub with `memory_max_mb` or `pids_max` greater
        than the parent's, true nesting WOULD have capped the sub
        at parent's value.  This helper surfaces those cases at
        `INFO` so operators can see the evidence base for a Phase 6
        nested-layout migration.

        Pure observability — no behavior change.  Sub-cgroup limits
        are already applied as declared by the time this helper
        runs.

        `cpu_weight` is intentionally omitted: it's a share within
        a cgroup hierarchy, not an absolute cap, so a parent-vs-sub
        comparison doesn't represent composition semantics.  When
        Phase 6 ships the restructure, cpu_weight composition
        becomes meaningful; until then, omitting it from the log
        avoids misleading operators.

        See `docs/design/phase5_5_2_nested_cgroup_deferral_audit.md`.
        """
        parent_session = self._sessions.get(parent_session_id)
        parent_server = getattr(parent_session, "server", None) if parent_session else None
        parent_profile = getattr(parent_server, "_profile", None) if parent_server else None
        parent_limits = getattr(parent_profile, "runtime_limits", None) if parent_profile else None

        if parent_limits is None:
            logger.info(
                "_spawn_isolated_runner: nesting-visibility — "
                "parent=%s has no kernel runtime_limits; "
                "sub %s bounds are independent "
                "(sub.memory_max_mb=%s, sub.pids_max=%s).  "
                "Phase 6 nested layout would have nothing to cap.",
                parent_session_id, subagent_id,
                sub_limits.memory_max_mb, sub_limits.pids_max,
            )
            return

        fields_exceeded: List[str] = []
        if (parent_limits.memory_max_mb is not None
                and sub_limits.memory_max_mb is not None
                and sub_limits.memory_max_mb > parent_limits.memory_max_mb):
            fields_exceeded.append(
                f"memory_max_mb (sub={sub_limits.memory_max_mb}, "
                f"parent={parent_limits.memory_max_mb})"
            )
        if (parent_limits.pids_max is not None
                and sub_limits.pids_max is not None
                and sub_limits.pids_max > parent_limits.pids_max):
            fields_exceeded.append(
                f"pids_max (sub={sub_limits.pids_max}, "
                f"parent={parent_limits.pids_max})"
            )

        if fields_exceeded:
            logger.info(
                "_spawn_isolated_runner: nesting-visibility — "
                "parent=%s subagent=%s sub-cgroup bounds EXCEED "
                "parent's on: %s.  Today (sibling layout per §4.3.5): "
                "sub gets its declared bounds independently.  Phase 6 "
                "nested layout would cap at parent's value.  See "
                "Phase 5 §5.2.",
                parent_session_id, subagent_id, "; ".join(fields_exceeded),
            )

    def _resolve_cgroups_manager(self) -> Optional[Any]:
        """Return the daemon's :class:`CgroupsManager` instance, or
        ``None`` if not wired.

        Phase 4 §4.3.5: mirrors :meth:`_resolve_apparmor_manager`.
        ``_cgroups_manager`` lives on session managers wired via the
        IPC apparmor opt-in pre-init hook.  Test fakes that bypass
        the wire-up won't have the attribute; ``getattr`` returns
        ``None`` in that case so the helper falls back to the
        skip-cgroup-creation branch.
        """
        return getattr(self, "_cgroups_manager", None)

    def _resolve_apparmor_manager(self) -> Optional[Any]:
        """Return the daemon's :class:`AppArmorManager` instance, or
        ``None`` if not wired.

        Phase 4 §4.3.4: ``_spawn_isolated_runner`` needs the
        AppArmorManager to provision sub-profiles.  The manager
        instance lives at ``self._apparmor_manager`` on session
        managers wired via the IPC AppArmor opt-in path (set in the
        pre-initialize hook).  Test fakes that bypass the wire-up
        (``SessionManager.__new__`` without the hook) won't have the
        attribute; ``getattr`` returns ``None`` in that case so the
        helper falls back to the unavailable-message branch.

        Returns:
            The wired :class:`AppArmorManager`, or ``None`` if
            unavailable (test fake, host without AppArmor, etc.).
        """
        return getattr(self, "_apparmor_manager", None)

    def add_pre_initialize_hook(self, hook: Callable) -> None:
        """Register a callback invoked BEFORE ``server.initialize()`` runs
        (server 0.6.49+).

        The hook is called with four arguments:

        1. ``server`` — the just-constructed ``JaatoServer`` instance.
           Plugin registry NOT YET populated; provider NOT YET connected.
           Hook can stash the reference but must not call methods that
           depend on init state.
        2. ``session_id`` — the unique session identifier string.
        3. ``workspace_path`` — the session's workspace dir (or ``None``).
        4. ``client_id`` — the requesting client's id, or ``None`` for
           non-client-driven session creation paths (currently
           ``_load_session_impl``).  Phase 2 task 2.3 added this so the
           IPC AppArmor pre-init hook can look up the creator's
           ``ClientConfigRequest.apparmor`` opt-in (the lookup via
           ``_client_to_session`` doesn't work pre-init because that
           mapping isn't populated yet).

        Pre-initialize hooks exist so transports (notably the WS server's
        AppArmor wiring) can provision per-session kernel resources
        BEFORE ``server.initialize()`` runs the agent's
        ``configure()`` — including dynamic-instructions expansion and
        any prefetch scripts.  Without this hook, the AppArmor profile
        would not exist yet at prefetch time, leaving prefetch scripts
        unconfined and able to write to ``.jaato`` (or anywhere else
        the unconfined daemon can reach) before the deny rules apply.

        Distinct from ``add_session_hook`` which fires AFTER ``initialize()``
        completes (used for set_apparmor_confinement, cgroup attach
        callback, sandbox wiring — anything that depends on the
        executor existing).

        Hooks are called in registration order.  Exceptions are logged
        and subsequent hooks still run.

        Args:
            hook: A callable with signature
                ``(server: JaatoServer, session_id: str,
                workspace_path: Optional[str],
                client_id: Optional[str]) -> None``.

        Backwards-compat: hooks accepting only the first three args
        (the pre-Phase-2 signature) are still called with positional
        args 1-3; the ``client_id`` is dropped via ``inspect.signature``
        introspection inside :meth:`_run_pre_initialize_hooks`.
        """
        self._pre_initialize_hooks.append(hook)

    def _run_session_hooks(self, server: JaatoServer, session_id: str) -> None:
        """Invoke all registered session hooks for a newly set-up session.

        Args:
            server: The JaatoServer instance to pass to each hook.
            session_id: The session identifier.
        """
        for hook in self._session_hooks:
            try:
                hook(server, session_id)
            except Exception as exc:
                logger.warning("Session hook failed: %s", exc, exc_info=True)

    def _bootstrap_session(
        self,
        envelope: 'BootstrapEnvelope',
    ) -> Tuple[Optional[JaatoServer], Optional['Session']]:
        """Construct + initialize a session from a BootstrapEnvelope.

        Phase 3 §3.12.0.  This is the single helper every
        session-creation path funnels through, replacing the
        ad-hoc kwarg-bag previously inlined in each call site
        (``_create_session_impl``, ``_load_session_impl``,
        ``run_ephemeral_session``, ``JaatoWSServer`` standalone).

        Body matches the §3.12.0 spec:

        1. Construct ``JaatoServer`` from the envelope's
           construction fields.
        2. Push ``config_root`` onto the server BEFORE
           ``_run_pre_initialize_hooks`` so plugins discovered
           during ``initialize`` see the override.
        3. Run pre-initialize hooks (legacy 3-arg + 4-arg both
           still supported via :meth:`_run_pre_initialize_hooks`'s
           ``inspect.signature`` introspection).
        4. ``server.initialize()`` — return ``(None, None)`` on
           failure (the underlying error already emitted to the
           client).
        5. Build the :class:`Session` record with ``sandbox_mode``
           resolved per the priority chain:
           a. ``envelope.sandbox_mode`` — disk-restore's pre-known
              value (the saved Session record's mode).
           b. Return value of
              :meth:`_provision_ipc_apparmor_and_spawn_runner`
              (§3.13's inline call) — apparmor opt-in result for
              the IPC creation path.
           c. ``None`` — no opt-in / non-confined session.

           Phase 3 §3.13 removed the legacy
           ``server._planned_sandbox_mode`` stash slot; the
           apparmor opt-in is now read directly inside the
           inline provisioning call (whose return value flows
           into priority chain step (b)) rather than via a
           transient attribute on JaatoServer.

        The helper does NOT do:

        - Spawn-payload validation — that's profile-resolution
          time, before the envelope is built.
        - Session storage / event-callback rewiring / TODO
          configuration / client-config application — those are
          post-bootstrap concerns the caller handles after this
          returns.
        - Auth-complete callback registration — caller-specific.

        Args:
            envelope: The bootstrap payload aggregating every
                input the JaatoServer construction +
                Session-record path needs.

        Returns:
            ``(JaatoServer, Session)`` on success;
            ``(None, None)`` if ``server.initialize()`` failed
            (error already reported via the in-init event sink).

        Phase 3 §3.12.0 migrated the IPC path.  Phase 3 §3.12
        disk-restore migration routes ``_load_session_impl`` through
        the inner :meth:`_construct_and_initialize_server` helper —
        the helper splits the JaatoServer-construction-and-init from
        the Session-record assembly so the disk-restore path can
        share the construction logic while building its own
        record (with ``last_activity`` / ``user_inputs`` / etc.
        from the saved state).  The ephemeral path follows in a
        future commit.

        An AST partition test (``test_bootstrap_partition.py``)
        tracks remaining direct ``JaatoServer(...)`` construction
        sites with an explicit allow-list so a contributor adding
        a NEW path is forced to either funnel through here (or the
        sub-helper) or extend the allow-list.
        """
        server, planned_sandbox = self._construct_and_initialize_server(envelope)
        if server is None:
            return None, None

        session = Session(
            session_id=envelope.session_id,
            name=envelope.name,
            server=server,
            created_at=(
                envelope.timestamp.isoformat()
                if envelope.timestamp is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            description=envelope.description,
            is_dirty=True,  # New session needs saving
            workspace_path=envelope.workspace_path,
            config_root=envelope.config_root,
            provisioned=envelope.provisioned,
            created_by=envelope.created_by,
            sandbox_mode=planned_sandbox,
            inline_profile_spec=envelope.inline_profile_spec,
            sibling_name=getattr(envelope, "sibling_name", None),
        )

        return server, session

    def _wire_session_manager_into_plugins(self, server: Any) -> None:
        """Give every plugin that asks for it a handle on this manager.

        Duck-typed on the method, like the rest of the plugin lifecycle:
        wiring by NAME (``get_plugin("session_ops")``) meant a second plugin
        needing daemon-side session state had to edit this file, and a plugin
        that GREW the hook without editing it would silently never receive
        the manager.

        Called from :meth:`_construct_and_initialize_server` -- the single
        sanctioned construction funnel (see
        ``server/tests/test_bootstrap_partition.py``) -- so create and
        disk-restore are wired by the same line.  The standalone WS-server
        path constructs its own ``JaatoServer`` and is deliberately NOT
        wired: that mode has no ``SessionManager`` and no cascade, so the
        plugins' "no session manager is attached" answer is accurate there
        rather than a defect.

        A failure to wire ONE plugin must not abort session construction, so
        each is attempted independently and logged at WARNING -- loud enough
        to find, not fatal.
        """
        registry = getattr(server, "registry", None)
        if registry is None:
            return
        for name in registry.list_exposed():
            plugin = registry.get_plugin(name)
            if plugin is not None and hasattr(plugin, "set_session_manager"):
                try:
                    plugin.set_session_manager(self)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "set_session_manager failed for plugin %s: %s",
                        name, exc,
                    )

    def _construct_and_initialize_server(
        self,
        envelope: 'BootstrapEnvelope',
    ) -> Tuple[Optional[JaatoServer], Optional[str]]:
        """JaatoServer construction + pre-init + initialize, shared
        across IPC and disk-restore bootstrap paths (Phase 3 §3.12).

        Splits out from :meth:`_bootstrap_session` so the
        disk-restore path (which assembles a different Session
        record from the saved state — ``last_activity``,
        ``user_inputs``, restored ``is_dirty``) can share the
        construction logic without inheriting the create-session
        Session-record shape.

        Body matches the §3.12.0 spec:

        1. Construct ``JaatoServer`` from envelope construction
           fields.
        2. Push ``config_root`` onto the server BEFORE pre-init
           hooks so plugins see the override.
        3. Call ``_provision_ipc_apparmor_and_spawn_runner`` (§3.13
           inline relocation) — clean no-op when ``client_id`` is
           ``None`` or no apparmor opt-in.
        4. Run remaining pre-init hooks (WS + third-party).
        5. ``server.initialize()`` — return ``(None, None)`` on
           failure.
        6. Resolve sandbox_mode: ``envelope.sandbox_mode`` wins
           (disk-restore's pre-known value); else IPC method
           result; else None.

        Returns:
            ``(JaatoServer, sandbox_mode)`` on success;
            ``(None, None)`` on init failure.
        """
        server = JaatoServer(
            env_file=envelope.env_file,
            provider=None,  # Let env_file determine provider
            on_event=envelope.on_event_during_init,
            workspace_path=envelope.workspace_path,
            session_id=envelope.session_id,
            env_overrides=envelope.env_overrides,
            instruction_token_cache=envelope.instruction_token_cache,
            profile=envelope.profile,
            system_instruction_override=envelope.system_instruction_override,
            suppress_base_instructions=envelope.suppress_base_instructions,
            agent_name=envelope.agent_name,
        )

        # Push config_root BEFORE initialize so plugins discovered
        # during init see the override on their first
        # set_config_root notification.
        if envelope.config_root:
            server.config_root = envelope.config_root

        # Phase 4 §D: stash agent_params transiently on the per-session
        # JaatoServer so build_session_envelope can pick them up and
        # forward them on the SessionInitEnvelope.  The JaatoServer is
        # per-session (one instance per session, GC'd at session end),
        # so this is not centralized daemon-state — it's just the
        # per-session daemon-side handle holding session data during
        # the envelope-build window.
        server._agent_params = dict(envelope.agent_params or {})

        # Phase 2 cascade-sharing (server 0.6.144+): stash the cascade
        # tenant ID on the per-session server.  Picked up by
        # ``build_session_envelope`` so it lands on the wire envelope,
        # and by WS pre-init hook so it threads to spawn_session_runner.
        # Same per-session-handle pattern as ``_agent_params`` above.
        server._cascade_driver_id = envelope.cascade_driver_id

        # Phase 4 §B: resolve workspace .env + profile.env + overrides
        # BEFORE the runner-spawn fork so secret URIs (pass://,
        # vault://, awssm://, sops://, keyring://) reach the runner
        # subprocess via inherited os.environ.  Pre-fix the resolution
        # ran inside server.initialize() step 1 which fires AFTER the
        # spawn — the fork inherited unresolved literal URIs, and the
        # runner-side resolver wasn't always able to re-resolve (entry-
        # point registration timing + GPG-agent priming concerns).
        # Resolving daemon-side is reliable: the daemon process has
        # premium's secret resolvers entry-point-discovered at startup
        # and the GPG-agent socket has been primed by the operator.
        server._resolve_session_env()

        # Phase 3 §3.13: IPC apparmor provisioning + runner spawn
        # used to live in a pre-initialize hook registered from
        # ``server/__main__.py``.  The hook indirection was a
        # transitional step from Phase 2 §2.3; Phase 3 inlines the
        # logic here so the call site is co-located with the rest
        # of the bootstrap helper.  The method is a clean no-op
        # when ``client_id`` is None or the client did not opt in
        # to apparmor; non-IPC paths continue unaffected.
        #
        # Phase 4 §B: wrapped in _with_session_env so the spawn's
        # fork() inherits the resolved env via os.environ.  See
        # docs/design/phase4_env_propagation_audit.md for the bug
        # this closes (workspace .env pass:// URIs not reaching the
        # runner post-§7c seat-flip).
        # Server 0.6.131+ (PR-148): create the plugin registry + run
        # discover BEFORE the apparmor profile is composed.  The
        # composer at ``resolve_plugin_apparmor_rules`` walks
        # ``profile.plugins`` and queries each via
        # ``registry.get_plugin`` — without a registry, the loop is
        # silently skipped and ZERO plugin-contributed apparmor
        # rules land in the rendered profile.  Pre-PR-148, the
        # registry was created inside ``server.initialize()`` which
        # runs AFTER this provisioning step (~line 2571 below) —
        # too late.  Discovered v126/v128: file_edit's backup-path
        # rules never made it into the profile despite PR-145's
        # ``get_apparmor_rules`` export.
        #
        # ``create_registry_and_discover`` is idempotent — the
        # ``server.initialize()`` call below skips its own
        # registry-setup step when the registry already exists.
        with server._with_session_env(), server._in_workspace():
            server.create_registry_and_discover()

        with server._with_session_env():
            ipc_sandbox_mode = self._provision_ipc_apparmor_and_spawn_runner(
                server,
                envelope.session_id,
                envelope.workspace_path,
                envelope.client_id,
                apparmor_override=envelope.apparmor,
                # 2026-05-14 unification: thread the envelope's
                # ``config_root`` + ``env_file`` so the AppArmor policy
                # generator receives them across all entry points (IPC,
                # disk-restore, reactor-spawned headless, ephemeral
                # subagent, WS standalone) — not just the IPC path
                # that historically populated ``client_config[client_id]``.
                config_root_override=envelope.config_root,
                env_file_override=envelope.env_file,
                cascade_driver_id=envelope.cascade_driver_id,
            )

        # Pre-initialize hooks fire BEFORE initialize() — gives
        # transports a window to provision kernel resources
        # (AppArmor profile, cgroup) so prefetch can run inside
        # the session's confinement instead of unconfined.  After
        # §3.13 the IPC apparmor hook is no longer registered
        # here (relocated to the inline call above); the WS hook
        # and any third-party hooks continue to use this surface.
        self._run_pre_initialize_hooks(
            server,
            envelope.session_id,
            envelope.workspace_path,
            envelope.client_id,
        )

        # Initialize.  On failure, core.py already emits a
        # ConfigurationError event via the in-init sink — no need
        # for a redundant SessionError here.
        if not server.initialize():
            return None, None

        # Hand the manager to every plugin that asks for it.  MUST live here,
        # in the ONE construction funnel, not at a caller: this helper is
        # shared by session CREATE and by disk-RESTORE (which is what an
        # ``attach`` to an unloaded session runs).  It used to sit in
        # ``_create_session_impl``, so a session reached by attach came back
        # with its tools present and their daemon wiring absent -- and the
        # tools that need it answered "no session manager is attached", the
        # same words a real misconfiguration produces.
        self._wire_session_manager_into_plugins(server)

        # Reactor-bus sink: forward every event on this session's per-session
        # EventBus into the daemon-wide reactor bus, so a reactor that
        # subscribes ONCE to the daemon-wide bus receives events from ALL
        # sessions.  Per-session subscribers stay isolated on the per-session
        # bus; only this forward crosses into the daemon-wide one.  (Events for
        # an already-unloaded session arrive via daemon-level sources publishing
        # straight to the reactor bus — the per-session bus is gone by then.)
        # See docs/design/reactor-bus-session-scope.md.
        _runtime = getattr(server, "_runtime", None)
        _session_bus = getattr(_runtime, "event_bus", None) if _runtime else None
        if _session_bus is not None:
            from jaato_sdk.event_bus import EventFilter
            _session_bus.subscribe(
                subscriber_name="reactor_bus_sink",
                filter=EventFilter(),
                callback=self.reactor_event_bus.publish,
                replay_history=False,
            )

        # Resolve sandbox_mode.  Priority:
        # 1. ``envelope.sandbox_mode`` — authoritative pre-resolved
        #    value (disk-restore's saved mode).
        # 2. Result of inline IPC apparmor provisioning.
        # 3. None — no opt-in / non-confined.
        planned_sandbox = envelope.sandbox_mode
        if planned_sandbox is None:
            planned_sandbox = ipc_sandbox_mode
        return server, planned_sandbox

    def _run_pre_initialize_hooks(
        self,
        server: JaatoServer,
        session_id: str,
        workspace_path: Optional[str],
        client_id: Optional[str] = None,
    ) -> None:
        """Invoke pre-initialize hooks (server 0.6.49+).

        Called before ``server.initialize()`` so that transport-level
        kernel-resource provisioning (AppArmor profile, cgroup) can
        happen before the agent's configure() runs prefetch scripts.

        Args:
            server: The JaatoServer instance (constructed but not
                initialized).
            session_id: The session identifier.
            workspace_path: Session's workspace dir, or None.
            client_id: The requesting client id (or ``None`` for
                non-client-driven paths like ``_load_session_impl``).
                Phase 2 task 2.3+ extends the hook signature with this
                parameter so the IPC AppArmor pre-init hook can look up
                the creator's apparmor opt-in.

        Backwards-compat: hooks declared with the legacy 3-arg
        signature (server, session_id, workspace_path) keep working —
        we introspect the callable's parameter count and drop
        ``client_id`` for old-style hooks.
        """
        import inspect
        for hook in self._pre_initialize_hooks:
            try:
                # Count positional params on the hook (ignoring *args/**kwargs).
                # 4 = new-style; 3 = legacy.
                try:
                    sig = inspect.signature(hook)
                    n_positional = sum(
                        1 for p in sig.parameters.values()
                        if p.kind in (
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        )
                    )
                except (TypeError, ValueError):
                    n_positional = 4  # assume new-style on introspection failure
                if n_positional >= 4:
                    hook(server, session_id, workspace_path, client_id)
                else:
                    hook(server, session_id, workspace_path)
            except Exception as exc:
                logger.warning(
                    "Pre-initialize hook failed: %s", exc, exc_info=True,
                )

    def set_event_callback(
        self,
        callback: Callable[[str, Event], None],
    ) -> None:
        """Set callback for routing events to clients.

        Args:
            callback: Called with (client_id, event) for each event.
        """
        self._event_callback = callback

    def set_broadcast_callback(
        self,
        callback: Callable[[Event], None],
    ) -> None:
        """Set callback for broadcasting events to **all** connected clients.

        Used for daemon-wide events that don't belong to a specific
        session — currently the HandoffGate event family
        (``gate.announced`` / ``gate.released`` / ``gates.snapshot``)
        emitted by the jaato-premium reactor framework.  Wired in
        ``__main__.py`` to ``CompositeEventSink.broadcast_event``,
        which fans out across the IPC and WS transports.

        Args:
            callback: Called with (event,) for each broadcast.
        """
        self._broadcast_callback = callback

    def broadcast_event(self, event: Event) -> None:
        """Deliver an event to every connected client across all transports.

        Used by daemon extensions (notably the jaato-premium reactor
        framework's HandoffGate registry) to publish events that aren't
        tied to a specific session.  Per-session events should still go
        through the regular ``_emit_to_client`` / ``_emit_to_session``
        paths.

        No-op if no broadcast callback is wired (e.g. early daemon
        startup before transports are up).  Thread-safe: the underlying
        ``CompositeEventSink.broadcast_event`` snapshots its client
        registries before iterating.

        Args:
            event: The Event to broadcast.
        """
        callback = self._broadcast_callback
        if callback is None:
            logger.debug("broadcast_event: no broadcast callback wired; dropping %s", type(event).__name__)
            return
        callback(event)

    def _emit_to_client(self, client_id: str, event: Event) -> None:
        """Emit an event to a specific client, attributed to its session.

        WHY THE STAMP IS HERE.  ``_emit_to_session`` stamps, and it has 10
        call sites; this has 64.  So the MAJORITY path was unattributed,
        and ``session_id`` arrived empty on most of what a consumer sees --
        including every ``PermissionRequestedEvent``, and
        ``InjectPromptResultEvent`` from #619, which shipped through here
        the same day the unstamped path was identified.

        The earlier audit concluded this method "has nothing to stamp
        WITH" because it takes a client_id rather than a session_id.  That
        was wrong: ``_client_to_session`` is right here, and is already
        read that way elsewhere in this class.  A structural audit of
        which emitters stamp answered "does this one call the stamper",
        not "could it".

        WHAT STILL ARRIVES UNSTAMPED, and why that is correct: the map is
        not populated pre-init, so events emitted while a session is being
        created have no session to name yet.  Those are genuinely
        unattributable rather than missed -- which is the residue worth
        measuring before deciding whether the base field should become
        ``Optional[str] = None`` to say so out loud.

        The stamper NEVER overwrites, so an event that names its own
        subject -- ``SlotSettledEvent`` means "the session that just
        ended", ``GateReleasedEvent`` means "the originating session" --
        keeps it.  Relabelling those with the recipient's session would
        replace a true fact with a plausible one.
        """
        _stamp_session_id(event, self._client_to_session.get(client_id))
        logger.debug(f"_emit_to_client: {client_id} <- {type(event).__name__}")
        if self._event_callback:
            logger.debug(f"  calling event_callback")
            self._event_callback(client_id, event)
        else:
            logger.warning(f"  NO event_callback set!")

    # ------------------------------------------------------------------
    # Cascade-as-client (Phase 1, server 0.6.154+)
    # ------------------------------------------------------------------
    # See docs/design/cascade-as-client.md.  Decisions locked
    # 2026-05-21; Phase 1 adds the daemon-side registry + dispatch +
    # default lifecycle policy + GC backstop.  Phase 2 adds the
    # IPC-RPC variant for SDK clients.

    def register_in_process_client(
        self,
        client_id: str,
        callback: Callable[[Any], None],
        cascade_driver_id: str,
        role: str = "observer",
        event_types: Optional[Set[str]] = None,
        delivery_target_id: Optional[str] = None,
    ) -> None:
        """Register an in-process cascade-client (Phase 1).

        Subscribes ``callback`` to events fanned out by
        :meth:`_emit_to_session` for any session stamped with
        ``cascade_driver_id``.  Filtered by ``event_types`` (type-name
        match; ``None`` = subscribe to all).

        Args:
            client_id: Unique identifier for this registration.
                Convention: ``f"_cascade:{cascade_driver_id}"`` for the
                cascade owner; observers may use any string (e.g., an
                extension-supplied label).  Multiple entries with the
                same client_id are rejected to avoid silent shadowing.
            callback: Invoked with one positional argument (the
                event) when an event matching the filter arrives.
                Runs synchronously inside ``_emit_to_session`` —
                callback MUST NOT re-enter ``SessionManager`` methods
                that take ``_lock`` (deadlock).  Quick handlers only;
                offload heavy work to a thread.
            cascade_driver_id: The cid this registration observes.
                Sessions stamped with this cid will route their
                events to ``callback``.
            role: ``"owner"`` (single per cid; lifecycle authority) or
                ``"observer"`` (multiple per cid; read-only).
                Default ``"observer"`` for the common observe-only
                case.
            event_types: Set of event type-names to subscribe to
                (e.g., ``{"SessionTerminatedEvent", "AgentCompletedEvent"}``).
                ``None`` (default) subscribes to all event types.

        Idempotency (PR #182, 2026-05-21):
            Re-registration with the SAME ``client_id`` is idempotent —
            mirrors ``GateRegistry.get_or_create`` semantics.  First
            registration wins; subsequent calls with the same client_id:
              - Silent no-op if config (role + callback + event_types)
                matches exactly.
              - Warn-but-keep-first if config diverges (caller should
                ``unregister_cascade_client`` first if they want to
                re-register with different config).

            This contract lets callers (reactor extension, IPC handlers)
            re-call ``register_in_process_client`` during retry /
            reconnect / cascade-restart cycles without wrapping every
            call in try/except.

        Raises:
            ValueError: when ``role == "owner"`` AND a DIFFERENT
                client_id already holds owner for this cid (single-
                owner rule per Decision 5).  Same-client_id re-register
                with role="owner" is idempotent, NOT a raise.
            ValueError: when ``role`` is neither "owner" nor "observer".
        """
        if role not in ("owner", "observer"):
            raise ValueError(
                f"role must be 'owner' or 'observer'; got {role!r}"
            )
        entry = CascadeClientEntry(
            client_id=client_id,
            role=role,
            callback=callback,
            event_types=event_types,
            delivery_target_id=delivery_target_id,
        )
        with self._cascade_clients_lock:
            entries = self._cascade_clients.setdefault(cascade_driver_id, [])
            for existing in entries:
                if existing.client_id == client_id:
                    # PR #182 (Phase 1.1, 2026-05-21): re-registration
                    # with the same client_id is IDEMPOTENT.  Matches
                    # ``GateRegistry.get_or_create`` semantics
                    # (jaato_premium/reactors/gates/registry.py:97) —
                    # first registration wins; subsequent calls with
                    # the same client_id are silent no-ops if config
                    # matches, or warn-but-keep-first if config
                    # diverges.
                    #
                    # Why this contract: cascade-client registration
                    # is a "get_or_create" primitive.  Callers
                    # (premium reactor extension, IPC handlers) may
                    # re-call with the same client_id during retry /
                    # reconnect / cascade-restart cycles.  Strict-mode
                    # raise (Phase 1 original behavior) forced callers
                    # to wrap every call in try/except — duplicating
                    # what the registry should own.  See PR #182.
                    config_matches = (
                        existing.role == role
                        and existing.callback is callback
                        and existing.event_types == event_types
                    )
                    if not config_matches:
                        logger.warning(
                            "register_in_process_client: client_id=%r "
                            "already registered for cid=%r with different "
                            "config; keeping the original.  "
                            "(existing role=%s callback=%r event_types=%s; "
                            "new role=%s callback=%r event_types=%s).  "
                            "Callers should re-register with matching "
                            "config or call unregister first.",
                            client_id, cascade_driver_id,
                            existing.role,
                            getattr(existing.callback, "__qualname__", existing.callback),
                            sorted(existing.event_types) if existing.event_types else "ALL",
                            role,
                            getattr(callback, "__qualname__", callback),
                            sorted(event_types) if event_types else "ALL",
                        )
                    return  # idempotent — no-op (silent on match, warn on mismatch)
                if role == "owner" and existing.role == "owner":
                    raise ValueError(
                        f"cascade-client owner already registered for "
                        f"cid={cascade_driver_id!r} "
                        f"(existing client_id={existing.client_id!r}); "
                        f"only ONE owner permitted per cid (Decision 5).  "
                        f"This is a DIFFERENT-client-id conflict — two "
                        f"separate callers both trying to own the cid.  "
                        f"Idempotent re-registration with the SAME "
                        f"client_id no-ops; this raise covers genuine "
                        f"ownership-conflict."
                    )
            entries.append(entry)
        # Lazy-start the GC sweep thread on first registration.
        self._ensure_cascade_client_sweep_running()
        logger.info(
            "register_in_process_client: registered %s (role=%s) "
            "for cid=%s event_types=%s",
            client_id, role, cascade_driver_id,
            sorted(event_types) if event_types else "ALL",
        )
        # Wake re-nudge (Option 2): if an observer (re)registers for a cid that
        # has a wake pending on one of its sessions, re-emit SessionWokenEvent so
        # a reconnecting bot is nudged to re-attach even if it missed the first.
        if role == "observer":
            self._reemit_pending_wakes_for_cid(cascade_driver_id)

    def _reemit_pending_wakes_for_cid(self, cascade_driver_id: str) -> None:
        """Re-emit ``SessionWokenEvent`` for any not-yet-driven wake whose
        session's cid matches — so an observer that (re)connects after the first
        emit still learns it must re-attach."""
        now = time.time()
        with self._lock:
            pending = [
                (sid, p) for sid, p in self._pending_wakes.items()
                if p.cascade_driver_id == cascade_driver_id and p.expires_at > now
            ]
        for sid, p in pending:
            self._emit_session_woken(sid, p.wake_ref, p.source)

    def unregister_cascade_client(
        self,
        cascade_driver_id: str,
        client_id: str,
    ) -> bool:
        """Remove a cascade-client registration (Phase 1, explicit
        unregister per Decision 6).

        Args:
            cascade_driver_id: The cid the entry was registered under.
            client_id: The client_id used at register time.

        Returns:
            True if an entry was removed; False if no match (already
            unregistered, GC'd, or never registered).  Idempotent.
        """
        with self._cascade_clients_lock:
            entries = self._cascade_clients.get(cascade_driver_id, [])
            for i, entry in enumerate(entries):
                if entry.client_id == client_id:
                    entries.pop(i)
                    if not entries:
                        # Last entry for this cid — drop the dict key
                        # so the registry doesn't grow unbounded with
                        # empty cid entries.
                        self._cascade_clients.pop(cascade_driver_id, None)
                    logger.info(
                        "unregister_cascade_client: removed %s "
                        "from cid=%s", client_id, cascade_driver_id,
                    )
                    return True
        return False

    # ----------------------------- cascade.cancel ------------------------

    def is_cid_cancelled(self, cascade_driver_id: str) -> bool:
        """Whether *cascade_driver_id* has been cancelled by an operator.

        Reactor extensions consult this predicate before firing on
        ``AgentCompletedEvent`` so a cancelled cascade stops spawning
        new sessions.  Predicate-only API (Shape A from the
        brainstorm): the framework owns the cancelled-set; the reactor
        polls.  Race-free because the cancel handler marks the cid
        BEFORE iterating sessions to stop — by the time the reactor
        sees AgentCompletedEvent from a cancelled session, the cid is
        already marked.

        Args:
            cascade_driver_id: The cid to check.  Empty / None returns
                False (no cid → cannot be cancelled).

        Returns:
            True if the cid was previously passed to
            :meth:`cancel_cascade`; False otherwise.
        """
        if not cascade_driver_id:
            return False
        with self._cancelled_cids_lock:
            return cascade_driver_id in self._cancelled_cids

    def cancel_cascade(self, cascade_driver_id: str) -> Dict[str, Any]:
        """Cancel every loaded session belonging to *cascade_driver_id*.

        Implements the ``cascade.cancel cid`` IPC verb.  Three steps,
        ordered for race-safety with the reactor:

        1. **Mark the cid cancelled** — flips
           :meth:`is_cid_cancelled` to True before any session is
           stopped.  The reactor's consult-point sees the marker
           before any AgentCompletedEvent from the stopping sessions
           arrives, so suppression engages even for sessions that
           complete naturally between this call and the stop().
        2. **Iterate matching sessions + stop** — find every entry in
           ``_sessions`` whose ``cascade_driver_id == cid`` and call
           ``server.stop()`` (the canonical mid-turn cancel path,
           same as session.end's stop).  Idle sessions get a no-op
           stop; in-flight sessions get cancelled via the cancel
           token.
        3. **Emit SessionTerminatedEvent per cancelled session** with
           ``reason="cascade_cancelled"`` so clients and observers
           can distinguish operator-driven cascade cancel from
           natural completion / individual session.end.

        Idempotent for already-cancelled cids: marker stays set;
        re-iterating finds zero matching sessions (the prior call
        stopped them all); returns ``{"stopped_count": 0, ...}``.

        Args:
            cascade_driver_id: The cid to cancel.  Empty / None is a
                no-op returning zero counts.

        Returns:
            ``{"cid": cid, "cancelled_session_ids": [...],
            "stopped_count": int}`` — the list of session_ids
            cancelled this call.  Caller emits a SystemMessageEvent
            from this dict so the operator sees what got reaped.
        """
        if not cascade_driver_id:
            return {
                "cid": cascade_driver_id,
                "cancelled_session_ids": [],
                "stopped_count": 0,
            }

        # Step 1: mark cancelled BEFORE iterating sessions.  Race
        # window between this and step 2 is safe because the reactor
        # checks the marker, not the session list.
        with self._cancelled_cids_lock:
            self._cancelled_cids.add(cascade_driver_id)

        # Step 2: collect matching sessions under _lock so we don't
        # race with concurrent session creation / deletion.  Pop the
        # list out of the lock — server.stop() can be slow + may
        # re-enter SessionManager (via emit), so we release the lock
        # before calling it.
        with self._lock:
            matching: List[Tuple[str, Session]] = [
                (sid, sess) for sid, sess in self._sessions.items()
                if getattr(sess, "cascade_driver_id", None) == cascade_driver_id
            ]

        # Step 3: stop + emit per matching session.  Same shape as
        # session.end's stop + emit (command_router.py:_handle_session_end).
        from jaato_sdk.events import SessionTerminatedEvent
        cancelled_ids: List[str] = []
        for sid, sess in matching:
            if sess.server is None:
                continue
            agent_id = getattr(sess.server, "_main_agent_id", None) or "main"
            try:
                sess.server.stop()  # idempotent — returns False if idle
            except Exception:  # noqa: BLE001 — best-effort cancel
                logger.exception(
                    "cancel_cascade: server.stop() raised for session=%s "
                    "cid=%s — continuing with remaining sessions",
                    sid, cascade_driver_id,
                )
            self._emit_to_session(
                sid,
                SessionTerminatedEvent(
                    session_id=sid,
                    agent_id=agent_id,
                    reason="cascade_cancelled",
                ),
            )
            cancelled_ids.append(sid)

        logger.info(
            "cancel_cascade: cid=%s cancelled %d session(s): %s",
            cascade_driver_id, len(cancelled_ids), cancelled_ids,
        )
        return {
            "cid": cascade_driver_id,
            "cancelled_session_ids": cancelled_ids,
            "stopped_count": len(cancelled_ids),
        }

    def unregister_all_cascade_clients_for_connection(
        self, connection_client_id: str,
    ) -> int:
        """Phase 2 cascade-as-client: remove every cascade-client
        entry registered by the given IPC/WS connection.

        Called from the transport-level disconnect handler
        (command_router.handle_client_disconnect) to clean up
        registrations on connection loss.  Matches entries by the
        namespaced suffix convention: cascade-client client_ids are
        formatted as ``f"_cascade:{cid}:{connection_client_id}"``
        when registered via IPC; this method strips entries whose
        client_id ends with ``f":{connection_client_id}"``.

        Idempotent — returns 0 if no entries match.

        Args:
            connection_client_id: The IPC/WS client_id that
                disconnected.

        Returns:
            Number of entries removed.
        """
        suffix = f":{connection_client_id}"
        removed = 0
        with self._cascade_clients_lock:
            for cid in list(self._cascade_clients.keys()):
                survivors = [
                    e for e in self._cascade_clients[cid]
                    if not e.client_id.endswith(suffix)
                ]
                removed += len(self._cascade_clients[cid]) - len(survivors)
                if survivors:
                    self._cascade_clients[cid] = survivors
                else:
                    self._cascade_clients.pop(cid, None)
        if removed > 0:
            logger.info(
                "unregister_all_cascade_clients_for_connection: "
                "removed %d entries for client %s",
                removed, connection_client_id,
            )
        return removed

    def _dispatch_to_cascade_clients(
        self, session: 'Session', event: Event,
    ) -> None:
        """Phase 1: fan out an event to cascade-clients matching
        ``session.cascade_driver_id``.

        Post-bootstrap entry point — extracts the cid from the Session
        object then delegates to :meth:`_dispatch_to_cascade_clients_by_cid`.
        Bootstrap-time emit paths (where the Session object isn't yet in
        ``self._sessions`` because bootstrap is still in flight) call the
        by-cid helper directly via :meth:`_route_bootstrap_event` — see
        that method for the rationale and the duplicate-delivery caveat
        for direct-attached cascade observers.
        """
        cid = getattr(session, "cascade_driver_id", None)
        # Dedup owner==observer: every attached client already received this
        # event via the direct ``_emit_to_client`` fan-out at the caller
        # (session_manager.py ~3750), so skip any cascade entry delivering to
        # that same raw connection — otherwise post-bootstrap turn events
        # (ToolCallStart/End, AgentOutput, AgentCompleted) double on the wire
        # for a client that is BOTH attached AND a cascade observer.  Parallels
        # the bootstrap-path skip in :meth:`_route_bootstrap_event`.
        self._dispatch_to_cascade_clients_by_cid(
            cid, event, skip_client_ids=set(session.attached_clients),
        )

    def _dispatch_to_cascade_clients_by_cid(
        self, cid: Optional[str], event: Event,
        skip_client_id: Optional[str] = None,
        skip_client_ids: Optional[Set[str]] = None,
    ) -> None:
        """Phase 1 dispatch core — fan out an event to cascade-clients
        registered for ``cid``.

        Extracted from :meth:`_dispatch_to_cascade_clients` (server
        0.6.166+) so bootstrap-time emit paths can reach cascade
        observers without requiring a Session object — the Session
        isn't in ``self._sessions`` during bootstrap, so the previous
        ``getattr(session, "cascade_driver_id", None)`` lookup would
        succeed (PR-192 stamps the dataclass field) but the
        :meth:`_emit_to_session` event-routing chain that hosts the
        post-bootstrap call site doesn't fire during init.

        Dispatch order: owners first, then observers (so an owner
        can call ``session_manager.delete_session(...)`` and preempt
        observer notifications for a now-deleted session — the
        observer loop's ``_sessions.get(sid)`` would return None
        next time around).

        Callback exceptions are logged but never propagate: one
        misbehaving observer must not break the dispatch chain.
        ``last_event_ts`` is updated regardless so GC doesn't reap
        an actively-firing entry just because its callback raises.

        ``cid is None`` is a no-op: standalone (non-cascade) sessions
        emit through here at post-bootstrap dispatch + standalone
        bootstrap, and neither needs cascade fan-out.

        ``skip_client_id`` (server 0.6.177+, semantics tightened in
        0.6.178+): when non-None, cascade entries whose
        ``delivery_target_id`` matches are skipped.  Used by
        :meth:`_route_bootstrap_event` to dedup bootstrap-time
        delivery when the same IPC client is BOTH the direct-attach
        client AND a cascade observer for the same cid
        (cascade_develop.py's canonical client-of-API pattern).

        Pre-0.6.178 this compared against ``entry.client_id`` which
        is the NAMESPACED registration id
        (``_cascade:{cid}:{conn}``), not the raw connection id —
        so the empirical dedup never fired for IPC-bound cascade
        registrations.  Kb-side report 2026-06-03 from
        cascade_develop.py walker against 0.6.177: discovery still
        printed two ``↳ session <id>`` lines per AgentCreatedEvent
        because ``_cascade:eb3...:ipc_1 == ipc_1`` is False.  Fix:
        ``CascadeClientEntry`` now carries a separate
        ``delivery_target_id`` field (set by command_router's
        cascade.register handler to the raw connection client_id);
        this skip-check compares against THAT.

        In-process callers that don't set ``delivery_target_id``
        (extensions wiring callbacks not tied to IPC) get None →
        skip never matches → existing behavior preserved.

        ``skip_client_ids`` (server 0.6.196+): the POST-bootstrap
        analogue of the single ``skip_client_id``.  The post-bootstrap
        caller (:meth:`_dispatch_to_cascade_clients`) direct-emits the
        event to EVERY ``session.attached_clients`` before this
        cascade fan-out, so a cascade entry whose
        ``delivery_target_id`` is any of those attached clients would
        double-deliver on the same raw connection (owner==observer:
        the client that fired the session AND registered as a cascade
        observer on the same IPC connection).  Passing the
        attached-clients set here dedups all such overlaps; a cascade
        observer on a SEPARATE connection (not attached) is not in the
        set and still receives the event.

        ``last_event_ts`` is NOT updated for skipped entries because
        they didn't actually fire; their next real delivery resets
        the timer.
        """
        if cid is None:
            return
        # Combined skip set: bootstrap passes a single ``skip_client_id``
        # (the direct-attach client); the post-bootstrap path passes
        # ``skip_client_ids`` = the session's ``attached_clients``.
        skip: Set[str] = set()
        if skip_client_id is not None:
            skip.add(skip_client_id)
        if skip_client_ids:
            skip.update(skip_client_ids)
        # Snapshot under lock to avoid mutation-during-iteration if
        # a callback unregisters concurrently.
        with self._cascade_clients_lock:
            entries = list(self._cascade_clients.get(cid, []))
        now = time.monotonic()
        # Owners first, observers second — see docstring.
        for entry in sorted(entries, key=lambda e: 0 if e.role == "owner" else 1):
            if (
                entry.delivery_target_id is not None
                and entry.delivery_target_id in skip
            ):
                # Dedup branch (server 0.6.177+, fixed comparand
                # 0.6.178+): this entry's callback delivers to the
                # same raw connection that the direct-IPC path at
                # the caller's site is also delivering to.  Skip
                # to avoid double-delivery on the SDK queue.
                continue
            if not entry.event_type_match(event):
                continue
            entry.last_event_ts = now
            try:
                entry.callback(event)
            except Exception as exc:  # noqa: BLE001 — callback boundary
                logger.warning(
                    "_dispatch_to_cascade_clients_by_cid: callback for %s "
                    "(role=%s) raised %s — continuing dispatch chain",
                    entry.client_id, entry.role,
                    type(exc).__name__, exc_info=True,
                )

    def _route_bootstrap_event(
        self,
        direct_client_id: Optional[str],
        cascade_driver_id: Optional[str],
        event: Event,
        session_id: Optional[str] = None,
    ) -> None:
        """Centralized bootstrap-time event router (server 0.6.166+).

        Stamps ``event.session_id`` (protocol 1.2+) for the same reason
        :meth:`_emit_to_session` does — bootstrap-time events bypass that
        chokepoint entirely (the Session isn't in ``self._sessions``
        yet), so without stamping here the FIRST events of a session's
        life — precisely the ones an observer uses to notice it exists —
        would arrive unattributed.

        Replaces the previous ``on_event_during_init`` lambda pattern
        which routed ONLY to the requesting client via
        ``_emit_to_client``, silently bypassing the cascade-client
        dispatch chain for the window between session-creation request
        and ``set_event_callback`` wiring (when the regular
        :meth:`_emit_to_session` path takes over).

        Empirical motivation (peer 7:1, retry-47, 2026-05-28): cascade
        observers subscribed to ``AgentCreatedEvent`` for downstream
        reactor-spawned sessions (context / host_validator / codegen)
        never received the event because it fires DURING bootstrap
        with ``client_id = _HEADLESS_CLIENT_ID`` — the synthetic
        client whose transport drops the event.  ``SessionTerminatedEvent``
        fires AFTER bootstrap so the post-bootstrap path delivered it
        correctly; that asymmetry made the bug seem random.

        The audit identified TWO call sites with the same pattern
        (``create_session`` main path at line ~4199; ``_load_session``
        disk-restore path at line ~4857).  This helper centralizes the
        routing so neither site can silently drop cascade observer
        delivery.

        Args:
            direct_client_id: requesting client of the bootstrap.
                ``None`` for some restore paths.  When set, the event
                is emitted to this client via
                :meth:`_emit_to_client` (existing behavior).
            cascade_driver_id: if set, the event is also dispatched to
                cascade-clients registered for this cid via
                :meth:`_dispatch_to_cascade_clients_by_cid` — reaches
                cascade observers regardless of whether the Session is
                in ``self._sessions`` yet.
            event: the event being emitted.

        **Bootstrap-time dedup (server 0.6.177+)**: for the case
        where ``direct_client_id`` is a real IPC client AND that
        same client has a cascade observer subscription via
        ``IPCClient.cascade_events()`` (the canonical
        cascade_develop.py / cascade.py pattern — client creates
        the session AND subscribes to its own cid as observer), the
        cascade-fan-out call below passes ``skip_client_id`` so the
        direct-IPC delivery isn't duplicated through the cascade
        route-back.  Pre-0.6.177 this manifested as
        ``AgentCreatedEvent`` arriving twice on the SDK queue and
        was empirically surfaced by cascade_develop.py's walker
        printing two ``↳ session <id>`` lines per main-agent
        bootstrap once PR-205 populated session_id.  The
        ``_HEADLESS_CLIENT_ID`` case (reactor-spawned downstream
        sessions where ``direct_client_id`` is the synthetic
        headless id) is unaffected: the headless id will never
        match a real cascade-observer's client_id, so the
        skip-branch is a no-op for that load-bearing path.
        """
        _stamp_session_id(event, session_id)
        if direct_client_id is not None:
            self._emit_to_client(direct_client_id, event)
        if cascade_driver_id is not None:
            self._dispatch_to_cascade_clients_by_cid(
                cascade_driver_id,
                event,
                skip_client_id=direct_client_id,
            )

    def _apply_default_cascade_policy(
        self, session: 'Session', event: Event,
    ) -> None:
        """Default lifecycle policy: on ``SessionTerminatedEvent`` for
        a session that is headless OR has a registered cascade-owner OR
        is cascade-stamped (``cascade_driver_id`` set), force an unload.

        The cascade-stamped disjunct (server 0.6.166+) covers the
        driver-attached DISCOVERY session, whose IPC client registers
        only as an *observer*: without it that session was neither
        headless nor owner-registered, hit the early-return, and its
        slot stayed pinned for minutes until the driver detached on its
        own — see the inline comment on the gate below.

        ``SessionTerminatedEvent`` is by definition a terminal-state
        signal (see :class:`jaato_sdk.events.SessionTerminatedEvent`
        — "Session has fully wound down — safe to disconnect").  All
        four current reasons (``natural``, ``error``, ``stopped``,
        ``client_request``) are terminal; any of them on a headless /
        cascade-owned / cascade-stamped session means it's safe to
        unload now, which
        triggers ``JaatoServer.shutdown()`` → ``session_end`` RPC →
        ``pool_manager.return_slot_after_session(...)``.  Without this
        unload, the runner subprocess stays alive and the pool slot
        never returns to ``_idle_slots`` — next same-cascade acquire
        MISSES instead of HITs.

        Originally shipped (Phase 1, PR #178) handling only
        ``reason="error"`` to close Finding B (cascade-stall on terminal
        error).  Server 0.6.158 extends to all four reasons after
        retry-17 evidence showed natural-completion cascades
        accumulated 9 concurrent runners + 1h37min runner-exit lag
        because the prior ``reason != "error"`` guard let
        natural-completion sessions stay loaded indefinitely.

        Real-client sessions (interactive UI / TUI) are NOT unloaded
        — the client may reconnect to view history before an explicit
        ``session.delete`` command.

        Runs AFTER ``_dispatch_to_cascade_clients`` so a cascade-owner
        handler can preempt: if the owner deletes the session, the
        default's ``_maybe_unload_session`` call becomes a no-op
        (session already gone from ``self._sessions``).
        """
        # Local import to avoid widening the module-level import
        # graph; SessionTerminatedEvent is a small SDK type.
        from jaato_sdk.events import SessionTerminatedEvent
        if not isinstance(event, SessionTerminatedEvent):
            return

        is_headless = (
            session.attached_clients == {self._HEADLESS_CLIENT_ID}
        )
        cid = getattr(session, "cascade_driver_id", None)
        has_cascade_owner = False
        if cid is not None:
            with self._cascade_clients_lock:
                entries = self._cascade_clients.get(cid, [])
                has_cascade_owner = any(e.role == "owner" for e in entries)

        # Server 0.6.166+ (γ'-guard fix): a cascade-stamped session
        # (``cid is not None``) ALWAYS passes this gate, even when it is
        # neither headless nor owner-registered.  Without this third
        # disjunct the driver-attached DISCOVERY session — created via
        # ``client.create_session("discovery", cascade_driver_id=...)``
        # over an IPC client that registers only as an *observer* (no
        # ``owner`` entry) — was ``is_headless=False`` AND
        # ``has_cascade_owner=False``, hit this early-return, and so the
        # γ' driver-detach block below NEVER ran.  Its slot stayed pinned
        # until the driver's IPC client detached on its own (measured
        # 2m50s–6m25s later, 2026-06-11), stalling the cascade's first
        # handoff while every headless handoff returned its slot in
        # ~250ms.  ``cid is not None`` makes the γ' detach reachable for
        # discovery so its slot returns at SessionTerminated like every
        # other stage.  TUI interactive sessions never set
        # ``cascade_driver_id`` so they are unaffected.
        if not (is_headless or has_cascade_owner or cid is not None):
            return

        # Pop the synthetic _HEADLESS_CLIENT_ID so the existing
        # _maybe_unload_session gate (`if session.attached_clients`)
        # passes through.
        session.attached_clients.discard(self._HEADLESS_CLIENT_ID)
        self._client_to_session.pop(self._HEADLESS_CLIENT_ID, None)
        # Server 0.6.165+ (γ'): when the session is cascade-stamped
        # (cascade_driver_id != None), ALSO pop real IPC clients
        # attached via ``client.create_session(cascade_driver_id=...)``.
        # Rationale: presence of cascade_driver_id IS the semantic
        # signal the client is participating in a cascade (observer
        # role by design); the IPC attachment is incidental to the
        # cascade-kickoff API, not a request to keep the session
        # loaded.  Without this, the cascade-driver's
        # ``client.create_session("discovery", cascade_driver_id=...)``
        # pinned the discovery slot for the entire IPC connection
        # lifetime — peer 7:1's retry-46 held a discovery slot 6m43s
        # past SessionTerminated until the driver was killed.  TUI
        # interactive sessions don't pass cascade_driver_id so they
        # keep the current "stay-loaded-for-history-inspection"
        # behavior; only cascade sessions auto-detach here.
        if cid is not None:
            for client_id in list(session.attached_clients):
                session.attached_clients.discard(client_id)
                # Only pop client_to_session if it still points at
                # this session — the client may have attached
                # elsewhere since.
                if self._client_to_session.get(client_id) == session.session_id:
                    self._client_to_session.pop(client_id, None)
        reason = getattr(event, "reason", "unknown")
        try:
            self._maybe_unload_session(session.session_id)
            logger.info(
                "_apply_default_cascade_policy: triggered unload for "
                "headless/cascade session %s after "
                "SessionTerminatedEvent(reason=%s)",
                session.session_id, reason,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning(
                "_apply_default_cascade_policy: _maybe_unload_session "
                "raised for %s: %s — cascade may stall; investigate",
                session.session_id, exc, exc_info=True,
            )

    def _record_cid_session_activity(self, cid: Optional[str]) -> None:
        """Stamp the most-recent-session-creation timestamp for a
        cascade_driver_id.  No-op when ``cid`` is None (standalone
        session).

        The cascade-client GC sweep consults this dict to keep an
        observer registration alive across the brief window between
        session N unloading (default policy) and session N+1 spawning
        — without this, the sweep would reap the observer mid-cascade.

        Called from every site that adds a session with a cid to
        ``self._sessions``.  Cleanup happens in the sweep when the
        last registration for the cid is reaped.

        Server 0.6.161+ (Bug B fix).
        """
        if cid is None:
            return
        self._cid_last_session_ts[cid] = time.monotonic()

    def _ensure_cascade_client_sweep_running(self) -> None:
        """Lazy-start the cascade-client GC backstop sweep thread on
        first registration.  Idempotent."""
        if (
            self._cascade_client_sweep_thread is not None
            and self._cascade_client_sweep_thread.is_alive()
        ):
            return
        self._cascade_client_sweep_stop = threading.Event()
        self._cascade_client_sweep_thread = threading.Thread(
            target=self._cascade_client_sweep_loop,
            name="cascade-client-gc-sweep",
            daemon=True,
        )
        self._cascade_client_sweep_thread.start()
        logger.info(
            "Cascade-client GC sweep thread started "
            "(idle_timeout=%.0fs)",
            self._cascade_client_idle_timeout,
        )

    def _cascade_client_sweep_loop(self) -> None:
        """Phase 1 GC backstop (Decision 6): periodically reap
        cascade-client entries with no recent events AND no active
        sessions in their cid.

        Runs every ``cascade_client_idle_timeout / 10`` seconds
        (default 30s for the 300s timeout) so each entry is checked
        ~10 times across its idle window — fast enough to free
        resources without burning CPU.

        Stops cleanly on ``_cascade_client_sweep_stop.set()``
        (daemon shutdown path).
        """
        check_interval = max(1.0, self._cascade_client_idle_timeout / 10.0)
        while not self._cascade_client_sweep_stop.is_set():
            try:
                self._cascade_client_sweep_once()
            except Exception:  # noqa: BLE001 — sweep boundary
                logger.exception(
                    "cascade-client GC sweep: unhandled error; continuing",
                )
            self._cascade_client_sweep_stop.wait(check_interval)

    def _cascade_client_sweep_once(self) -> None:
        """Single sweep pass: reap stale cascade-client entries.

        Stale = (now - last_event_ts) > timeout AND no active
        sessions in the cid.  An entry that has never fired
        (``last_event_ts is None``) uses ``registered_at`` as the
        reference instead — covers registrations that outlive their
        sessions without any event activity.
        """
        now = time.monotonic()
        timeout = self._cascade_client_idle_timeout
        cids_to_reap: List[Tuple[str, str]] = []

        # Snapshot active-cid set under _lock to avoid racing
        # session-record creation.
        with self._lock:
            active_cids = {
                getattr(s, "cascade_driver_id", None)
                for s in self._sessions.values()
            } - {None}

        with self._cascade_clients_lock:
            for cid, entries in list(self._cascade_clients.items()):
                if cid in active_cids:
                    continue  # cascade still active; skip
                # Wake durability (Option 2): while a LIVE wake binding carries
                # this cid, its observer must survive the session going cold — a
                # wake may still arrive.  Tie the observer's lifetime to the
                # binding, NOT to session activity (else a wake-bound but idle
                # bot loses its session.woken subscription after the idle timeout
                # and silently misses the wake).
                if self._wake_binding_registry.has_live_binding_for_cid(cid):
                    continue
                # Server 0.6.161+ (Bug B): also treat the cascade as
                # alive if a session with this cid was created within
                # ``timeout`` seconds, even if no session is currently
                # loaded (PR-183 default policy unloads each stage
                # before the next spawns — brief gaps are normal).
                last_session_ts = self._cid_last_session_ts.get(cid)
                if (
                    last_session_ts is not None
                    and (now - last_session_ts) <= timeout
                ):
                    continue  # cascade alive recently; skip
                survivors: List[CascadeClientEntry] = []
                for entry in entries:
                    ref_ts = (
                        entry.last_event_ts
                        if entry.last_event_ts is not None
                        else entry.registered_at
                    )
                    if (now - ref_ts) > timeout:
                        cids_to_reap.append((cid, entry.client_id))
                    else:
                        survivors.append(entry)
                if survivors:
                    self._cascade_clients[cid] = survivors
                else:
                    self._cascade_clients.pop(cid, None)
                    # Server 0.6.161+ (Bug B): clean up the
                    # cid-activity dict when its last registration
                    # is reaped — bounds growth.
                    self._cid_last_session_ts.pop(cid, None)

        for cid, client_id in cids_to_reap:
            logger.info(
                "cascade-client GC sweep: reaped %s (cid=%s) — "
                "idle > %.0fs + no active sessions",
                client_id, cid, timeout,
            )

    # ------------------------------------------------------------------

    def _emit_to_session(self, session_id: str, event: Event) -> None:
        """Emit an event to all clients attached to a session.

        Also STAMPS the event's ``session_id`` (protocol 1.2+).  This is
        the one fan-out chokepoint that knows the session and feeds both
        delivery paths — the direct-attach clients below and the
        cascade-observer dispatch — so stamping here attributes every
        routed event without any emit site having to remember.

        Why it is done here and not at the emit sites: attribution used
        to exist on 12 of 112 event types, and the 100 without it were
        exactly the ACTIVITY events (turn, tool, agent output) an
        observer needs in order to tell two siblings of one cascade
        apart.  ``agent_id`` cannot do it — it is ``"main"`` for every
        top-level session.
        """
        _stamp_session_id(event, session_id)
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                # Handle session description updates - update in-memory Session
                if isinstance(event, SessionDescriptionUpdatedEvent):
                    if event.session_id == session_id:
                        session.description = event.description
                        session.is_dirty = True
                        logger.debug(f"Updated session {session_id} description: {event.description}")

                # Handle turn tracking for interrupted tool recovery
                self._handle_turn_tracking_event(session, event)

                # Cascade budget: deplete the cid pool from this turn's spend.
                self._accumulate_cascade_budget(session, event)

                for client_id in session.attached_clients:
                    self._emit_to_client(client_id, event)

                # Phase 1 cascade-as-client: dispatch to cascade-clients
                # registered for this session's cascade_driver_id.
                # Owners fire first (lifecycle authority); observers
                # follow.  See docs/design/cascade-as-client.md §4.3.
                self._dispatch_to_cascade_clients(session, event)

                # Phase 1 default lifecycle policy: on terminal-error
                # for headless / cascade-owned sessions, trigger
                # unload (closes Finding B).  Runs AFTER cascade-
                # client dispatch so owner handlers can preempt.
                self._apply_default_cascade_policy(session, event)

    # ------------------------------------------------------------------
    # Workspace file monitoring
    # ------------------------------------------------------------------

    def _start_workspace_monitor(
        self,
        session_id: str,
        workspace_path: str,
        server: Optional[JaatoServer] = None,
    ) -> None:
        """Start (or restart) a workspace file monitor for a session.

        If a monitor already exists for this session it is stopped first.

        After starting the monitor, wires up the sandbox manager plugin
        so that readwrite sandbox paths are automatically watched.

        Args:
            session_id: The session to monitor.
            workspace_path: Absolute path to the workspace directory.
            server: The session's JaatoServer.  If not provided, looked up
                from ``self._sessions``.  Pass explicitly when the session
                is not yet registered (e.g., during ``_load_session``).
        """
        self._stop_workspace_monitor(session_id)

        def on_changed(changes: List[Dict[str, str]]) -> None:
            """Callback invoked by the debouncer on the timer thread."""
            # Mark session dirty so the tracked dict gets persisted.
            with self._lock:
                session = self._sessions.get(session_id)
                if session:
                    session.is_dirty = True

            self._emit_to_session(session_id, WorkspaceFilesChangedEvent(
                changes=changes,
            ))

        monitor = WorkspaceMonitor(workspace_path, on_changed=on_changed)
        monitor.start()
        self._workspace_monitors[session_id] = monitor
        logger.debug("Workspace monitor started for session %s", session_id)

        # Wire sandbox path monitoring: when the user runs
        # "sandbox add readwrite /some/path", the workspace monitor should
        # also watch that path for file changes.
        self._wire_sandbox_to_monitor(session_id, monitor, server=server)

    def _stop_workspace_monitor(self, session_id: str) -> None:
        """Stop the workspace monitor for a session if one exists.

        Also clears the sandbox manager callback so it doesn't reference
        a stale monitor.

        Args:
            session_id: The session whose monitor to stop.
        """
        monitor = self._workspace_monitors.pop(session_id, None)
        if monitor:
            monitor.stop()
            logger.debug("Workspace monitor stopped for session %s", session_id)

        # Clear the sandbox manager callback to avoid stale references
        with self._lock:
            session = self._sessions.get(session_id)
        if session and session.server:
            sandbox_plugin = session.server._find_plugin_for_command("sandbox")
            if sandbox_plugin and hasattr(sandbox_plugin, 'set_on_readwrite_paths_changed'):
                sandbox_plugin.set_on_readwrite_paths_changed(None)

    def _wire_sandbox_to_monitor(
        self,
        session_id: str,
        monitor: WorkspaceMonitor,
        server: Optional[JaatoServer] = None,
    ) -> None:
        """Connect the sandbox manager plugin to the workspace monitor.

        When the user adds or removes readwrite sandbox paths, the monitor
        should start or stop watching those directories.  This method:

        1. Finds the sandbox manager plugin in the session's registry.
        2. Seeds the monitor with current readwrite paths.
        3. Registers a callback so future changes are propagated.

        Args:
            session_id: The session whose sandbox plugin to wire.
            monitor: The workspace monitor to update.
            server: The session's JaatoServer.  If not provided, looked up
                from ``self._sessions``.
        """
        if server is None:
            with self._lock:
                session = self._sessions.get(session_id)
            if not session or not session.server:
                return
            server = session.server

        sandbox_plugin = server._find_plugin_for_command("sandbox")
        if not sandbox_plugin or not hasattr(sandbox_plugin, 'set_on_readwrite_paths_changed'):
            return

        # Seed the monitor with current readwrite paths
        if hasattr(sandbox_plugin, 'get_readwrite_paths'):
            current_paths = sandbox_plugin.get_readwrite_paths()
            if current_paths:
                monitor.update_sandbox_paths(current_paths)
                logger.debug(
                    "Seeded workspace monitor with %d sandbox readwrite paths for session %s",
                    len(current_paths), session_id,
                )

        # Register callback for future changes
        def on_readwrite_changed(paths: List[str]) -> None:
            """Called by sandbox manager when readwrite paths change."""
            monitor.update_sandbox_paths(paths)
            logger.debug(
                "Updated workspace monitor sandbox paths: %d paths for session %s",
                len(paths), session_id,
            )

        sandbox_plugin.set_on_readwrite_paths_changed(on_readwrite_changed)

    def _send_workspace_snapshot(self, session_id: str, client_id: str) -> None:
        """Send the full workspace file snapshot to a specific client.

        Used on reconnect / attach so the client can rebuild its mirror.

        Args:
            session_id: Session whose monitor state to send.
            client_id: Target client.
        """
        monitor = self._workspace_monitors.get(session_id)
        if not monitor:
            return

        snapshot = monitor.get_snapshot()
        if snapshot:
            self._emit_to_client(client_id, WorkspaceFilesSnapshotEvent(
                files=snapshot,
                total=monitor.active_file_count,
            ))

    #: Path-bearing ``ClientConfigRequest`` fields, in the order they are
    #: reported.  Every one of them is interpreted by the DAEMON's
    #: filesystem, so a relative value silently means a different directory
    #: on each side of the socket (issue #742).
    _CLIENT_CONFIG_PATH_FIELDS = (
        "working_dir",
        "config_root",
        "env_file",
        "trace_log_path",
        "provider_trace_log",
    )

    def _reject_relative_client_paths(
        self, client_id: str, event: 'ClientConfigRequest',
    ) -> bool:
        """Refuse a client handshake carrying a relative path.

        A relative path is not a portable value across a process boundary:
        its meaning depends on the receiver's cwd, which the sender does
        not share and cannot see.  Absolutising it here — the daemon's
        historical behaviour — supplies the sender's missing half from the
        WRONG process, splitting a session's workspace across two
        directories with no error on either side (issue #742).

        Every path-bearing field is checked, not just ``working_dir``:
        ``config_root``, ``env_file`` and the two trace-log paths cross the
        same boundary by the same mechanism.  All violations are reported
        in one error so a client fixing its handshake sees the whole list
        rather than one field per round trip.

        Nothing is applied when this returns True — a half-applied
        handshake (say, a good ``working_dir`` beside a dropped
        ``config_root``) is its own silent-wrong-directory bug.

        Args:
            client_id: The requesting client, which receives the error.
            event: The client config event to validate.

        Returns:
            True when the config was REJECTED (caller must not apply it),
            False when every path is absolute or absent.
        """
        from shared.path_utils import describe_relative_path

        violations = [
            message
            for message in (
                describe_relative_path(
                    field, getattr(event, field, None) or "",
                    origin="the daemon boundary",
                )
                for field in self._CLIENT_CONFIG_PATH_FIELDS
            )
            if message
        ]
        if not violations:
            return False

        error = (
            "client config rejected — a relative path cannot cross the "
            "daemon boundary:\n" + "\n".join(f"  - {m}" for m in violations)
        )
        logger.error("Client %s: %s", client_id, error)
        self._emit_to_client(client_id, ErrorEvent(
            error=error,
            error_type="RelativePathAcrossBoundary",
            recoverable=True,
        ))
        return True

    def _apply_client_config(self, client_id: str, event: 'ClientConfigRequest') -> None:
        """Apply client configuration settings.

        Updates environment and plugin settings based on client's config.
        This allows clients to use their own .env settings (like JAATO_TRACE_LOG)
        even when connecting to a shared server.

        A handshake carrying a RELATIVE path is refused outright and
        nothing is applied — see :meth:`_reject_relative_client_paths`.

        Args:
            client_id: The requesting client.
            event: The client config event with settings.
        """
        import os

        if self._reject_relative_client_paths(client_id, event):
            return

        # Apply trace log paths if provided
        if event.trace_log_path:
            os.environ['JAATO_TRACE_LOG'] = event.trace_log_path
            logger.info(f"Client {client_id} set JAATO_TRACE_LOG={event.trace_log_path}")

        if event.provider_trace_log:
            os.environ['JAATO_PROVIDER_TRACE'] = event.provider_trace_log
            logger.info(f"Client {client_id} set JAATO_PROVIDER_TRACE={event.provider_trace_log}")

        # Initialize client config dict if needed
        if client_id not in self._client_config:
            self._client_config[client_id] = {}

        # Store presentation context (display capabilities)
        if event.presentation:
            self._client_config[client_id]['presentation'] = event.presentation
            logger.info(f"Client {client_id} set presentation context (client_type={event.presentation.get('client_type', 'unknown')})")

        # Store and apply working directory
        if event.working_dir:
            self._client_config[client_id]['working_dir'] = event.working_dir
            logger.info(f"Client {client_id} set working_dir={event.working_dir}")

        # Store config_root override (read-only framework config root).
        # When unset, the daemon falls back to ``working_dir/.jaato/`` per
        # ``shared.config_resolver.resolve_config_search_path``.
        if event.config_root:
            self._client_config[client_id]['config_root'] = event.config_root
            logger.info(f"Client {client_id} set config_root={event.config_root}")

        # Store client's env_file path for session creation
        if event.env_file:
            self._client_config[client_id]['env_file'] = event.env_file
            logger.info(f"Client {client_id} set env_file={event.env_file}")

        # Store permission timeout override
        if event.permission_timeout is not None:
            self._client_config[client_id]['permission_timeout'] = event.permission_timeout
            logger.info(f"Client {client_id} set permission_timeout={event.permission_timeout}")

        # Store opt-in AppArmor confinement flag.  Default is False —
        # IPC sessions historically run unconfined because the local
        # user already has full filesystem access.  Setting True asks
        # the daemon to provision a per-session AppArmor profile when
        # the client's next session is created (see session-creation
        # hook registered in ``__main__.py``).  When AppArmor is
        # unavailable the session falls back to running unconfined,
        # but the hook emits a ``SystemMessageEvent`` describing the
        # outcome — never a silent fallback.
        if getattr(event, 'apparmor', False):
            self._client_config[client_id]['apparmor'] = True
            logger.info(f"Client {client_id} set apparmor=True")

        # Apply to current session if client is attached to one
        self._apply_client_config_to_live_session(client_id, event)

    def _apply_client_config_to_live_session(
        self, client_id: str, event: 'ClientConfigRequest',
    ) -> None:
        """Push a re-sent client config onto the client's ATTACHED session.

        A client may send ``ClientConfigRequest`` again after it already
        holds a session (a resized terminal, a re-handshake after
        reconnect).  The stored config alone would then only reach the
        NEXT session, so the live one is updated in place here.  A client
        with no attached session — the usual connect-time case — is a
        no-op.

        Only the fields whose effect is per-session are pushed:
        presentation context, workspace / config_root (mirrored onto both
        the ``Session`` record and its ``JaatoServer``, which read them
        from different places), and the permission timeout.

        Callers must validate paths first — see
        :meth:`_reject_relative_client_paths`; this method assumes the
        event has already been accepted.

        Args:
            client_id: The client whose attached session to update.
            event: The already-validated client config event.
        """
        session_id = self._client_to_session.get(client_id)
        if not session_id:
            return
        session = self._sessions.get(session_id)
        if not (session and session.server):
            return
        self._apply_presentation_to_server(event, session.server)
        if event.working_dir:
            session.server.workspace_path = event.working_dir
            session.workspace_path = event.working_dir
        if event.config_root:
            # Mirrors workspace_path propagation: stash on the
            # JaatoServer so JaatoRuntime / JaatoSession can read
            # it when threading through discovery sites.
            session.server.config_root = event.config_root
            session.config_root = event.config_root
        if event.permission_timeout is not None:
            self._apply_permission_timeout(session.server, event.permission_timeout)

    def _apply_client_config_to_server(self, client_id: str, server: 'JaatoServer') -> None:
        """Apply stored client configuration to a server.

        Called when a client creates or attaches to a session.

        Args:
            client_id: The client whose config to apply.
            server: The server to configure.
        """
        config = self._client_config.get(client_id, {})
        if 'presentation' in config:
            from jaato_sdk.plugins.model_provider.types import PresentationContext
            ctx = PresentationContext.from_dict(config['presentation'])
            server.set_presentation_context(ctx)
            logger.debug(f"Applied presentation context to server for client {client_id}")
        elif client_id == self._HEADLESS_CLIENT_ID:
            # Reactor / extension-spawned headless sessions never send a
            # ClientConfigRequest, so without this branch their server's
            # presentation_context stays None.  Pre-server-0.6.67 that
            # was effectively API-equivalent (the lifecycle filter
            # exposes signal_completion when pctx is None), but default
            # to ClientType.API explicitly so the contract is declarative
            # and any future client_type-dependent code (telemetry,
            # rendering hints, plugin-level routing) can rely on a
            # known value rather than treating None as "unknown".
            from jaato_sdk.events import ClientType
            from jaato_sdk.plugins.model_provider.types import PresentationContext
            server.set_presentation_context(PresentationContext(
                client_type=ClientType.API,
            ))
            logger.debug(
                "Applied default API presentation context to headless server"
            )
        if 'working_dir' in config:
            server.workspace_path = config['working_dir']
            logger.debug(f"Applied working_dir={config['working_dir']} to server for client {client_id}")
        if 'config_root' in config:
            server.config_root = config['config_root']
            logger.debug(f"Applied config_root={config['config_root']} to server for client {client_id}")
        if 'permission_timeout' in config:
            self._apply_permission_timeout(server, config['permission_timeout'])

    @staticmethod
    def _apply_permission_timeout(server: 'JaatoServer', timeout: int) -> None:
        """Apply a permission timeout override to the session's permission plugin.

        Args:
            server: The server whose permission config to update.
            timeout: Timeout in seconds. 0 means wait forever.
        """
        if server.permission_plugin and hasattr(server.permission_plugin, '_config'):
            config = server.permission_plugin._config
            if config:
                config.channel_timeout = timeout
                logger.debug(f"Applied permission_timeout={timeout} to session {server.session_id}")

    @staticmethod
    def _apply_presentation_to_server(event: 'ClientConfigRequest', server: 'JaatoServer') -> None:
        """Construct and apply PresentationContext from a ClientConfigRequest.

        Args:
            event: The client config event.
            server: The server to configure.
        """
        if event.presentation:
            from jaato_sdk.plugins.model_provider.types import PresentationContext
            ctx = PresentationContext.from_dict(event.presentation)
            server.set_presentation_context(ctx)

    # ---------------- cascade budgets (design note §8/b) ----------------

    def set_cascade_budget(self, cascade_driver_id: str, budget: Any) -> None:
        """Declare the AGGREGATE ceiling for a cascade.  Owner-only.

        ``budget`` is a :class:`~shared.budget_control.BudgetControlConfig`.
        Declaring it here rather than on a profile is deliberate (§3.1): a
        profile is a reusable template, but a cascade cap is a runtime
        aggregate over one live cid — putting it on a leaf profile makes
        "which profile owns the number" unanswerable the moment two cascades
        spawn the same profile.

        Idempotent per cid: re-declaring replaces the pool, which resets
        accumulated spend.  Callers should declare once, at cascade launch.
        """
        from shared.budget_control import CascadeBudgetPool
        with self._cascade_budgets_lock:
            self._cascade_budgets[cascade_driver_id] = CascadeBudgetPool(
                cascade_driver_id, budget)
        logger.info(
            "cascade budget declared for %s: limits=%s",
            cascade_driver_id, dict(budget.limits),
        )

    def get_cascade_budget(self, cascade_driver_id: Optional[str]) -> Optional[Any]:
        """The :class:`CascadeBudgetPool` for a cid, or ``None`` if uncapped."""
        if not cascade_driver_id:
            return None
        with self._cascade_budgets_lock:
            return self._cascade_budgets.get(cascade_driver_id)

    def clear_cascade_budget(self, cascade_driver_id: str) -> None:
        """Drop a cascade's pool (cascade finished / owner unregistered)."""
        with self._cascade_budgets_lock:
            self._cascade_budgets.pop(cascade_driver_id, None)

    def _emit_cascade_refusal(
        self, client_id: Optional[str], session_id: str, exc: Any,
        request_id: Optional[str] = None,
    ) -> None:
        """Tell the requesting client its child was refused, and why.

        A refused spawn must be as observable as a session-level budget
        refusal.  Without this the driver sees a session id, then nothing,
        then a turn timeout — it cannot tell "the cascade is out of budget"
        (finish gracefully) from "the daemon hung" (escalate), and those
        want opposite responses.

        Carries ``error_type=CascadeExhaustedError`` and the full
        ``as_payload()`` — exhausted dimensions, both min() inputs, and the
        rendered detail — so the refusal is evidence the framework handed
        over rather than something the driver inferred from a timeout.

        DELIVERED TWICE, to two different audiences:

        1. The requesting client, which is waiting on this specific spawn.
        2. Every cascade OBSERVER of the cid — the design's own observation
           surface, and the audience a *cascade-budget* refusal is most
           obviously for.  Emitting only to the requester meant a driver
           watching the cascade stream saw a session that never appeared and
           no reason, which is the timeout-versus-refusal ambiguity this
           method exists to remove, surviving on the one surface built to
           watch a cascade.

        It also matters for a spawn with no real requester: a reactor- or
        cascade-driven child carries the synthetic ``_HEADLESS_CLIENT_ID``,
        so step 1 reaches nobody and the refusal existed only as a log line.

        The cid comes off the EXCEPTION rather than a parameter — the
        refusal already carries it, and a second source could disagree with
        the first.  ``skip_client_id`` dedups the requester when it is also
        an observer on the same connection.

        The event is stamped with the refused ``session_id`` (protocol 1.2+),
        because on a shared cascade stream "a spawn was refused" is not
        actionable without knowing WHICH one.
        """
        from jaato_sdk.events import ErrorEvent
        try:
            payload = exc.as_payload() if hasattr(exc, "as_payload") else {}
            logger.warning(
                "cascade refused spawn of %s: %s", session_id, payload,
            )
            # THE CORRELATION IS WHAT MAKES THIS REACH THE CALLER.
            #
            # Without ``request_id`` the SDK's create-wait files this as an
            # incidental event and keeps waiting -- so the caller sees a 30s
            # runner-not-ready timeout while the real answer sits in the log.
            # The SDK's own ``SessionRefused`` contract names
            # ``CascadeExhaustedError`` as its example and could not fire on
            # the one path that produces it, because the event it was waiting
            # for was unaddressed.
            #
            # ``None`` stays ``None``: a spawn with no originating request
            # (reactor- or cascade-driven, carrying the synthetic headless
            # client id) has nothing to correlate, and inventing an id would
            # let this event satisfy some other caller's wait.
            #
            # PASSED IN, never looked up.  The first version read
            # ``self._sessions`` here -- and this method's ``except
            # Exception`` (which exists so a failing sink cannot break a
            # spawn) swallowed the resulting AttributeError and emitted
            # NOTHING.  A defensive catch around a new dependency turns a
            # crash into a silent no-emit, which is the failure this whole
            # method exists to prevent.
            event = ErrorEvent(
                error=str(exc),
                error_type="CascadeExhaustedError",
                recoverable=False,
                details=payload,
                session_id=session_id,
                request_id=request_id,
            )
            if client_id:
                self._emit_to_client(client_id, event)
            cid = getattr(exc, "cascade_driver_id", None) or payload.get(
                "cascade_driver_id")
            if cid:
                self._dispatch_to_cascade_clients_by_cid(
                    cid, event, skip_client_id=client_id,
                )
        except Exception as emit_exc:  # noqa: BLE001 — defensive
            logger.warning(
                "cascade refusal emit failed for %s (root cause %s): %s",
                session_id, exc, emit_exc,
            )

    def _push_cascade_degrade(self, cascade_driver_id: str, fired) -> None:
        """Push a crossed cascade rung to every LIVE child of the cascade.

        The mid-flight half of the aggregate ceiling.  Spawn-time clamping
        alone constrains only children not yet started; a pool that let
        already-running siblings keep the ceiling they were handed would not
        be a shared budget — the point of it being aggregate is that one
        child burning the envelope affects everyone still running.

        Applies to ALL live children rather than recomputing each one's own
        ceiling: "the cascade is running low, everyone downshift" is what a
        pool means, and a child that has barely spent still degrades because
        its siblings did.

        Runner-side the rungs go through the SAME ``_apply_budget_rungs``
        path a session's own ladder uses, tagged ``origin="cascade"`` — so
        each pushed child emits the ordinary per-session evidence (tier
        rebind, active-tier re-connect, client notice), which is what makes
        "the push landed on THIS child" observable rather than merely "the
        pool crossed".

        DISPATCHED OFF THE EMIT PATH, one thread per child.  This is called
        from ``_accumulate_cascade_budget`` inside ``_emit_to_session``,
        which holds ``self._lock`` for its whole body — so doing the RPC
        inline blocked the ENTIRE SessionManager (every session's event
        delivery, not just this cascade's) for the duration.  Measured: three
        children x a 10s RPC timeout = 30s of frozen event delivery, during
        which the pushes themselves also timed out because the work they
        needed could not proceed.  One cause, two symptoms — the push failing
        AND clients receiving almost nothing.

        Per-child threads rather than a serial loop: a best-effort fan-out
        must not be serialised behind a per-child timeout, or N children cost
        N x timeout before any of them degrade.
        """
        payload = []
        for rung in fired:
            try:
                payload.append(rung.to_dict())
            except Exception:  # noqa: BLE001
                continue
        if not payload:
            return
        pool = self.get_cascade_budget(cascade_driver_id)
        pressure = pool.describe_pressure() if pool is not None else None
        with self._lock:
            targets = [
                (sid, sess) for sid, sess in self._sessions.items()
                if getattr(sess, "cascade_driver_id", None) == cascade_driver_id
            ]
        for sid, sess in targets:
            if getattr(sess, "draws_on_parent_budget", True) is False:
                # Its own budget governs it; the parent's pot running low is
                # not that child's problem and not the parent's call.
                continue
            rpc = getattr(getattr(sess, "server", None), "runner_rpc", None)
            if rpc is None:
                continue
            threading.Thread(
                target=self._push_cascade_degrade_one,
                args=(cascade_driver_id, sid, rpc, payload, pressure),
                name=f"cascade-degrade-{sid}",
                daemon=True,
            ).start()

    def _push_cascade_degrade_one(
        self, cascade_driver_id: str, session_id: str, rpc: Any,
        payload: list, pool_pressure: Optional[str] = None,
    ) -> None:
        """Deliver a cascade degrade to ONE child.  Best-effort, off-thread.

        Never raises: one unreachable runner must not stop its siblings
        degrading.  The failure log names the exception TYPE because a bare
        ``TimeoutError`` stringifies to the empty string — a previous version
        logged ``failed ()`` and told an operator nothing about why.

        A TIMEOUT is treated as distinct from a failure, and deliberately
        logged at INFO.  It means the daemon stopped waiting, not that the
        rung was rejected: a child inside a model call does not service the
        RPC until its turn ends, and the rung then applies at that boundary.
        An earlier version told the operator such a child "keeps its
        spawn-time ceiling", which is false — it is degraded, just late.
        """
        from jaato_sdk.events import AgentOutputEvent
        try:
            result = rpc.session_apply_budget_degrade_threadsafe(
                payload, pool_pressure, timeout=10.0)
            logger.info(
                "cascade %s pushed degrade to %s: %s",
                cascade_driver_id, session_id, result,
            )
            # Emit the child's notices from HERE.  The runner collected them
            # rather than writing them out because every client-facing
            # channel a session has is turn-scoped, and a pushed rung can
            # land between turns — which is why server-side degrades were
            # invisible to the driver.  The daemon is not turn-scoped.
            for notice in (result or {}).get("notices") or []:
                self._emit_to_session(session_id, AgentOutputEvent(
                    agent_id="main", source="system",
                    text=str(notice), mode="write",
                ))
        except TimeoutError as exc:
            # A TIMEOUT is the DAEMON giving up waiting, not the runner
            # refusing.  A child busy inside a model call does not service
            # the RPC until its turn ends, so the rung lands at that
            # boundary — measured across three runs: pushes that "failed"
            # at the 50% crossing were applied together with the 75% rung
            # at the next boundary.  Nothing is lost, it is DELAYED by up
            # to one turn, which is exactly the overshoot the
            # cap + N x one-turn bound accounts for.  Logged at INFO
            # because it is expected behaviour, not a fault.
            logger.info(
                "cascade %s: degrade push to %s did not ack within the "
                "timeout (%s) — child is mid-turn; the rung is latched and "
                "applies at its next turn boundary",
                cascade_driver_id, session_id,
                exc_message(exc),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort boundary
            logger.warning(
                "cascade %s: degrade push to %s failed: %s: %s — unlike a "
                "timeout this child may never receive the rung",
                cascade_driver_id, session_id,
                type(exc).__name__, exc_message(exc),
            )

    def _reconcile_cascade_pool(self, cascade_driver_id: Optional[str]) -> None:
        """Refresh a cascade's pool from its live sessions' own trackers.

        Called at the spawn boundary — off the hot path, and precisely when
        the pool's value has to be correct, because the next child's ceiling
        is ``min(profile, cascade_remaining)``.

        Reconciliation is delta-based per session
        (:meth:`CascadeBudgetPool.reconcile_session`), so running it
        alongside the incremental event-driven accumulation double-counts
        nothing: whichever saw a token first, the absolute reading settles
        the total.  A session whose runner is gone or unresponsive is
        skipped — its last incremental contribution stands.
        """
        pool = self.get_cascade_budget(cascade_driver_id)
        if pool is None:
            return
        with self._lock:
            sessions = [
                (sid, sess) for sid, sess in self._sessions.items()
                if getattr(sess, "cascade_driver_id", None) == cascade_driver_id
            ]
        for sid, sess in sessions:
            try:
                if getattr(sess, "draws_on_parent_budget", True) is False:
                    continue        # own books — see _accumulate_cascade_budget
                rpc = getattr(getattr(sess, "server", None), "runner_rpc", None)
                if rpc is None:
                    continue
                usage = rpc.session_get_budget_usage_threadsafe(timeout=5.0)
                result = pool.reconcile_session(sid, usage or {})
                deltas = result.deltas
                if result.fired:
                    # A refresh can itself cross a rung — the incremental
                    # view lagged.  Push before the new child is clamped.
                    self._push_cascade_degrade(cascade_driver_id, result.fired)
                if deltas:
                    # Log the session's TOTAL contribution too.  The delta
                    # alone is misleading: the incremental event path has
                    # usually already applied most dimensions, so a reconcile
                    # that moves only `usd` reads as though tokens were
                    # dropped — when tokens were simply already counted.
                    logger.info(
                        "cascade %s reconciled %s: delta=%s total=%s "
                        "(remaining %s)",
                        cascade_driver_id, sid, deltas,
                        pool.session_contribution(sid), pool.remaining(),
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.debug(
                    "cascade %s: could not reconcile %s (%s) — last "
                    "incremental contribution stands", cascade_driver_id,
                    sid, exc,
                )

    def _accumulate_cascade_budget(self, session: Session, event: Event) -> None:
        """Deplete a cascade's pool from one completed turn.

        Reads ``usage.spend_total_tokens`` — the SUM over the turn's
        responses — never ``total_tokens``, which is the end-of-turn context
        size and undercounts spend by ~41% on tool-calling turns.  The pool
        and the per-session tracker must count the same thing, or
        ``min(profile, cascade_remaining)`` composes two numbers measured on
        different scales.

        Accumulates from the event stream at the TURN boundary, which is
        exactly-once per turn (a refused turn emits no TurnCompletedEvent).
        Never let a budget failure break event delivery — this runs on the
        emit path.
        """
        if not isinstance(event, TurnCompletedEvent):
            return
        pool = self.get_cascade_budget(getattr(session, "cascade_driver_id", None))
        if pool is None:
            return
        # A child with its own declared budget spends on its own books, so
        # it must not deplete the shared pot — otherwise the pot would be
        # charged twice for the same tokens and would starve the children
        # that genuinely draw on it.
        if getattr(session, "draws_on_parent_budget", True) is False:
            return
        try:
            usage = getattr(event, "usage", None)
            spend = getattr(usage, "spend_total_tokens", None) if usage else None
            # Delta-based against this session's running absolute, so the
            # spawn-time reconciliation (which reads the same tracker) can
            # run alongside without double-counting either source.
            sid = getattr(session, "session_id", None) or id(session)
            prior = pool.session_contribution(sid)
            absolute = {
                "tokens": prior.get("tokens", 0.0) + float(spend or 0),
                "seconds": (prior.get("seconds", 0.0)
                            + float(getattr(event, "duration_seconds", 0) or 0)),
                "tool_calls": (prior.get("tool_calls", 0.0)
                               + len(getattr(event, "function_calls", None) or ())),
                "turns": prior.get("turns", 0.0) + 1.0,
            }
            cost = getattr(usage, "cost_usd", None) if usage else None
            if cost is not None:
                absolute["usd"] = prior.get("usd", 0.0) + float(cost)
            fired = pool.reconcile_session(sid, absolute).fired
            if fired:
                logger.info(
                    "cascade %s pool crossed %s (%s) — pushing to live children",
                    pool.cascade_driver_id,
                    ", ".join(f"{r.at_percent:.0f}%" for r in fired),
                    pool.describe_pressure(),
                )
                self._push_cascade_degrade(pool.cascade_driver_id, fired)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cascade budget accumulation failed: %s", exc)

    def _handle_turn_tracking_event(self, session: Session, event: Event) -> None:
        """Handle events for turn tracking (interrupted tool recovery).

        Tracks tool execution state so that if the server crashes during tool
        execution, we can recover by injecting synthetic error results.

        Args:
            session: The session being tracked.
            event: The event to process.
        """
        # Track when agent becomes active (turn starts).
        # Compare against the session's main_agent_id (typically "main",
        # but may be the ``--agent <name>`` value when one was supplied
        # at session creation).
        main_agent_id = (
            session.server.main_agent_id if session.server else "main"
        )
        if isinstance(event, AgentStatusChangedEvent):
            if event.status == "active" and event.agent_id == main_agent_id:
                # Main agent starting a turn - initialize tracking
                # Note: We don't have user_prompt here, but we can still track tool calls
                if not session.interrupted_turn:
                    session.interrupted_turn = {
                        "agent_id": event.agent_id,
                        "pending_tool_calls": [],
                        "user_prompt": "",  # Not available at this point
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                    session.is_dirty = True
                    logger.debug(f"Started turn tracking for session {session.session_id}")
            elif event.status == "done":
                # Agent finished - clear tracking
                if session.interrupted_turn:
                    session.interrupted_turn = None
                    session.is_dirty = True
                    logger.debug(f"Cleared turn tracking for session {session.session_id} (agent done)")

                # Re-check deferred unload: if all clients disconnected
                # while the model was running, the session was kept alive.
                # Now that the model is done, unload if still orphaned.
                if not session.attached_clients:
                    self._maybe_unload_session(session.session_id)

        # Track tool calls as they start
        elif isinstance(event, ToolCallStartEvent):
            if session.interrupted_turn and event.agent_id == session.interrupted_turn.get("agent_id"):
                # Add this tool call to pending list
                pending = session.interrupted_turn.get("pending_tool_calls", [])
                pending.append({
                    "id": event.call_id or "",
                    "name": event.tool_name,
                    "args": event.tool_args,
                })
                session.interrupted_turn["pending_tool_calls"] = pending
                session.is_dirty = True
                # Path H (cycle 10): incremental save deferred off
                # the synchronous _emit_to_session path.  Pre-Path-H
                # this called _save_session synchronously, which made
                # 2 blocking runner-RPCs that raced against the
                # runner's active send_message — 35s timeout starved
                # the permission-response window.  Async deferral
                # keeps the recovery contract (pending_tool_calls
                # still persisted) without blocking the emit path.
                self._save_session_async(session)
                logger.debug(
                    f"Updated pending tool calls for session {session.session_id}: "
                    f"{len(pending)} call(s), saving incrementally (async)"
                )

        # Remove completed tool calls from pending list
        elif isinstance(event, ToolCallEndEvent):
            if session.interrupted_turn and event.agent_id == session.interrupted_turn.get("agent_id"):
                pending = session.interrupted_turn.get("pending_tool_calls", [])
                # Remove the completed tool call by matching call_id
                original_count = len(pending)
                pending = [p for p in pending if p.get("id") != event.call_id]
                if len(pending) < original_count:
                    session.interrupted_turn["pending_tool_calls"] = pending
                    session.is_dirty = True
                    logger.debug(
                        f"Tool {event.tool_name} completed, {len(pending)} pending call(s) remain "
                        f"for session {session.session_id}"
                    )

        # Clear tracking when turn completes
        elif isinstance(event, TurnCompletedEvent):
            if session.interrupted_turn:
                session.interrupted_turn = None
                session.is_dirty = True
                logger.debug(f"Cleared turn tracking for session {session.session_id} (turn completed)")

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    def create_session(self, *args: Any, **kwargs: Any) -> str:
        """Create a new session and attach the client (server 0.6.71+ entry).

        Runs :meth:`_create_session_impl` via
        :func:`shared.session_context.run_in_fresh_session_context` so
        the bootstrap is ISOLATED from any ContextVar values inherited
        from the caller's task.  See the helper's module docstring for
        the full rationale (long-lived-daemon ContextVar leak class
        documented in
        ``project_backlog_workspace_root_contextvar_leak_long_lived_daemon.md``).

        Args:
            (forwarded verbatim to :meth:`_create_session_impl`)

        Returns:
            The new session ID, or empty string on failure.
        """
        from shared.session_context import run_in_fresh_session_context
        return run_in_fresh_session_context(
            self._create_session_impl, *args, **kwargs,
        )

    def _allocate_session_id(self, workspace_path: Optional[str]) -> str:
        """Atomically CLAIM a unique session id.

        Session ids are second-resolution timestamps
        (``%Y%m%d_%H%M%S``) with ``_N`` appended on collision.  The
        collision check used to be plain check-then-act: the candidate was
        tested against ``_sessions`` here, but only inserted into
        ``_sessions`` after the runner had spawned — measured at ~7.3s
        later.  Every concurrent ``create_session`` inside that window saw
        the same id free and took it, so three simultaneous creates were
        issued ONE id between them; two of the three sessions then never
        ran.  Second-resolution ids alone would collide for simultaneous
        spawns, but the wide window means they collide even when spawns are
        seconds apart.

        The claim is therefore made under ``self._lock`` against three
        sources at once — persisted ids, live ``_sessions``, and other
        in-flight claims — which makes a duplicate IMPOSSIBLE rather than
        unlikely.  Sub-second entropy would only have narrowed the window.

        The claim is released by :meth:`_release_session_id` once the
        session is registered (after which ``_sessions`` is authoritative)
        or the creation fails.  A leaked claim is benign: it only prevents
        that one timestamp string being reused.
        """
        existing_ids = {
            sess.session_id
            for sess in self._get_persisted_sessions(workspace_path=workspace_path)
        }
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        with self._lock:
            candidate = base
            counter = 0
            while (
                candidate in existing_ids
                or candidate in self._sessions
                or candidate in self._reserved_session_ids
            ):
                counter += 1
                candidate = f"{base}_{counter}"
            self._reserved_session_ids.add(candidate)
        return candidate

    def _release_session_id(self, session_id: str) -> None:
        """Drop an in-flight id claim (registered, or creation failed)."""
        with self._lock:
            self._reserved_session_ids.discard(session_id)

    def _cascade_storage_workspace(
        self, viewer_session_id: Optional[str], cid: Optional[str] = None,
    ) -> Optional[str]:
        """The workspace whose on-disk sessions hold this cascade's cold members.

        ``_get_persisted_sessions(workspace_path=None)`` does NOT mean "every
        workspace" -- it falls through to the session plugin's DEFAULT storage
        path (``target_dir = storage_dir or self._storage_path``).  For a
        workspace-scoped daemon that is a different directory, so the listing
        comes back empty and a resting sibling reads as one that never
        existed.  Absent and empty again, one layer down.

        Derived here rather than asked of callers: the two sibling paths both
        forgot to pass it, and a parameter that is silently wrong when omitted
        is a worse contract than no parameter.

        Prefers the viewer's own workspace; falls back to any loaded session
        in the same cascade, since a cascade shares a workspace.
        """
        with self._lock:
            viewer = self._sessions.get(viewer_session_id) if viewer_session_id else None
            if viewer is not None and getattr(viewer, "workspace_path", None):
                return viewer.workspace_path
            if cid:
                for s in self._sessions.values():
                    if s.cascade_driver_id == cid and getattr(s, "workspace_path", None):
                        return s.workspace_path
        return None

    def build_sibling_roster(
        self, viewer_session_id: str, workspace_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """The sessions sharing a viewer's cascade, as ``list_siblings`` sees them.

        ``{"you": <own address or None>, "siblings": [<row>, ...]}``.

        NO SELF ROW.  An agent has no reason to address itself, and a self row
        is an invitation to ``send_to_sibling(my_own_name)`` -- a loop generator
        in a feature whose §8 is entirely about bounding loops.  The ``you``
        scalar carries the agent's own address (assigned by whoever called
        ``session.new``, so it cannot otherwise know it) without putting it
        somewhere it can be passed as a target.

        NO ``role`` AND NO ``owner``.  Every cid-bearing session is top-level --
        subagents are runtime-level and carry no cid -- so the set is flat and
        every row would read "sibling".  A field whose value cannot vary is not
        information, and shipping ``parent``/``child`` values nothing can
        produce is the same defect as a guard that cannot fire.  They return
        when there is a topology to describe.

        LIVE UNION COLD.  Sessions unload on ORPHAN constantly, so a roster
        built from the in-memory table alone would make idle stages blink out
        and back -- and ``no_such_sibling`` would become a race rather than a
        fact.

        ``description`` is the peer's OWN ``session_describe`` output and is
        therefore UNTRUSTED CONTENT: the tool carries
        ``TRAIT_UNTRUSTED_CONTENT`` so the result routes through the boundary
        that marks and escapes it.  ``profile_name`` is author-written and
        trusted.

        Args:
            viewer_session_id: The asking session.
            workspace_path: Workspace whose persisted sessions to include.

        Returns:
            The roster.  ``siblings`` is empty when the viewer is in no
            cascade -- which is correct: it has none.
        """
        viewer = self._sessions.get(viewer_session_id)
        cid = getattr(viewer, "cascade_driver_id", None) if viewer else None
        you = getattr(viewer, "sibling_name", None) if viewer else None
        if cid is None:
            return {"you": you, "siblings": []}

        # The plugin calls this with the session id alone, so without this
        # the cold half listed the DEFAULT storage dir and came back empty --
        # a resting sibling vanished from the roster it owns an address in.
        if workspace_path is None:
            workspace_path = self._cascade_storage_workspace(viewer_session_id, cid)

        rows: List[Dict[str, Any]] = []
        live_ids = set()
        for sid, s in self._sessions.items():
            if s.cascade_driver_id != cid or sid == viewer_session_id:
                continue
            live_ids.add(sid)
            running = bool(
                s.server is not None and getattr(s.server, "_model_running", False))
            rows.append({
                "sibling_name": s.sibling_name,
                "status": "active" if (running or s.attached_clients) else "idle",
                "profile_name": self._roster_profile_name(s),
                "description": s.description,
            })

        try:
            for info in self._get_persisted_sessions(workspace_path=workspace_path):
                if info.session_id in live_ids or info.session_id == viewer_session_id:
                    continue
                if getattr(info, "cascade_driver_id", None) != cid:
                    continue
                rows.append({
                    "sibling_name": getattr(info, "sibling_name", None),
                    "status": "cold",
                    "profile_name": getattr(info, "profile_name", None),
                    "description": info.description,
                })
        except Exception as exc:  # noqa: BLE001
            # WARNING, not debug: a roster silently missing its cold members
            # reads as "those siblings do not exist".
            logger.warning(
                "sibling roster could not read persisted sessions (%s) -- "
                "cold siblings are missing from this listing", exc,
            )

        rows = [r for r in rows if r["sibling_name"]]
        rows.sort(key=lambda r: r["sibling_name"])
        return {"you": you, "siblings": rows}

    # ``permission_response`` / ``clarification_response`` are PARENT
    # authority.  ``send_to_subagent``'s own instructions document sending
    # them through that channel -- a parent answering its child's request.
    # A sibling edge that reused the channel naively would let any peer
    # grant permissions to any other peer, which defeats the permission
    # system outright (design §7).
    #
    # Matched on the OPENING TAG ONLY, and case-insensitively: a sender
    # that can get the daemon to accept `<Permission_Response ...>` has
    # already won, and a closing-tag check would miss a self-closing form.
    _SIBLING_FORBIDDEN_TAGS = ("permission_response", "clarification_response")

    def _sibling_grammar_violation(self, text: str) -> Optional[str]:
        """Return the forbidden tag *text* attempts to use, or ``None``."""
        lowered = text.lower()
        for tag in self._SIBLING_FORBIDDEN_TAGS:
            if f"<{tag}" in lowered:
                return tag
        return None

    def deliver_sibling_message(
        self,
        sender_session_id: str,
        sibling_name: str,
        text: str,
    ) -> Dict[str, Any]:
        """Deliver *text* from one cascade member to another.  DAEMON-SIDE.

        FIRE AND FORGET WITH A RECEIPT.  There is no reply channel and no
        blocking form, so two siblings awaiting each other is not
        expressible (design §8).  The receipt says what happened to the
        MESSAGE, never what the peer decided:

        ``accepted``   the peer was idle, so a turn was DRIVEN on its own
                       session.  Its cost lands on the peer and depletes the
                       shared cid pool via ``_accumulate_cascade_budget`` --
                       a sibling cannot spend a budget nobody can see.
        ``queued``     the peer was mid-turn.  SIBLING is an idle-only tier,
                       so the message waits for the current turn to end
                       rather than interrupting it, and is drained at that
                       boundary by ``JaatoSession._drain_child_messages``.

        QUEUED-AND-UNDRAINED IS NOT A STATE THIS CAN PRODUCE, and it used to
        be the common one -- see ``shared.message_delivery``.

        ``queued`` is decided from the peer's state AT THE MOMENT OF THE
        CHECK.  If the peer goes idle between that check and the delivery,
        ``inject_prompt_to_session`` drives it instead -- a strictly better
        outcome that the receipt does not distinguish.  The receipt's promise
        holds either way: the message is delivered and will be acted on.  It
        is not a claim about which mechanism carried it.
        ``no_such_sibling`` / ``sibling_cold`` / ``refused``

        ``queued`` and ``accepted`` are both about DELIVERY.  Neither claims
        the peer read, understood, or acted on anything -- a receipt that
        implied processing would be a blocking call wearing a non-blocking
        name.

        COLD PEERS ARE NOT WOKEN.  Reaching a resting session is a bigger,
        more surprising act than reaching a running one, and it belongs
        behind an explicit request rather than as a side effect of ordinary
        coordination (design §11 Q2).  ``sibling_cold`` says the address is
        real and the peer is resting, which is a different fact from
        ``no_such_sibling`` and needs a different response.

        Args:
            sender_session_id: The asking session.  The daemon reads the
                sender's identity from ITS OWN table -- the sender never
                supplies it, so a peer cannot claim to be someone else
                (design §7).
            sibling_name: Cascade-scoped address of the target.
            text: The message body.

        Returns:
            A receipt dict.  ``status`` is one of the five above; failures
            carry ``error`` (the key ``tool_result_is_error`` reads).
        """
        with self._lock:
            sender = self._sessions.get(sender_session_id)
            if sender is None:
                return {"status": "refused",
                        "error": "send_to_sibling: the calling session is not loaded."}
            cid = getattr(sender, "cascade_driver_id", None)
            sender_name = getattr(sender, "sibling_name", None)

        if not cid:
            # Not a failure of addressing -- a statement about scope.  The
            # cid IS the blast radius (design §2/§10); a session outside a
            # cascade has no siblings to reach, and saying "no such sibling"
            # would misdescribe that as a lookup miss.
            return {"status": "refused",
                    "error": ("send_to_sibling: this session is not part of a "
                              "cascade, so it has no siblings. The cascade "
                              "(cascade_driver_id) is the addressing boundary.")}

        violation = self._sibling_grammar_violation(text)
        if violation:
            return {"status": "refused",
                    "error": (f"send_to_sibling: <{violation}> is parent authority "
                              f"and cannot travel sideways. Siblings coordinate; "
                              f"they do not approve, grant or cancel for one "
                              f"another.")}

        size = len(text.encode("utf-8"))
        if size > SIBLING_MESSAGE_MAX_BYTES:
            return {"status": "refused",
                    "error": (f"send_to_sibling: message is {size} bytes, over the "
                              f"{SIBLING_MESSAGE_MAX_BYTES}-byte cap. Send a "
                              f"pointer to the work, not the work.")}

        with self._lock:
            exchanges = self._sibling_exchanges.get(cid, 0)
        if exchanges >= SIBLING_CID_EXCHANGE_CAP:
            return {"status": "refused",
                    "error": (f"send_to_sibling: this cascade has used its "
                              f"{SIBLING_CID_EXCHANGE_CAP} sibling messages. "
                              f"Coordinate through the driver.")}

        target_id, target_status = self._resolve_sibling(sender_session_id, cid,
                                                         sibling_name)
        if target_status == "absent":
            return {"status": "no_such_sibling",
                    "error": (f"send_to_sibling: no sibling named {sibling_name!r} "
                              f"in this cascade. Use list_siblings for the roster.")}
        if target_status == "cold":
            return {"status": "sibling_cold",
                    "error": (f"send_to_sibling: {sibling_name!r} is resting "
                              f"(unloaded). Cold siblings are not woken by a "
                              f"sibling message.")}

        with self._lock:
            target = self._sessions.get(target_id)
            # The REPLICA of the peer's turn state, kept ONLY as a witness in
            # the diagnostic below -- it is no longer consulted for any
            # decision.  It clears later than the peer's own flag, which is
            # what made "queued" mean "stranded" for ~30s after every turn.
            replica_busy = bool(
                target is not None and target.server is not None
                and getattr(target.server, "_model_running", False))
            pending = self._sibling_pending.get(target_id, 0)
        # Backpressure ASKS rather than assumes.  ``pending`` counts
        # consecutive queued sends, so reaching the cap means "N sends with no
        # turn of its own" -- but whether the peer is STILL working is the
        # peer's fact, not the daemon's.  At the cap the delivery below is made
        # with ``require_idle``, so it lands only if a turn would start, and
        # the peer itself tells us when it did not.  The old form -- refusing
        # here on the daemon's ``_model_running`` -- could refuse a peer that
        # had drained its backlog half a minute earlier.
        at_cap = pending >= SIBLING_PENDING_CAP

        # The DAEMON stamps the sender, never the sender itself (design §7),
        # and the body is wrapped as untrusted content so the receiving model
        # treats it as a claim to weigh rather than an instruction to follow.
        # Both imported at call time, matching this module's existing
        # convention (see the SourceType import in the external-event
        # handler) -- session_manager stays importable from contexts that
        # don't load the shared tier.
        from jaato_sdk.plugins.model_provider.types import wrap_untrusted_content
        from shared.message_queue import SourceType
        wrapped = wrap_untrusted_content(
            text, source=f"sibling:{sender_name or sender_session_id}")

        # ONE primitive, and it asks the peer rather than a replica of it.
        #
        # This used to inject in BOTH branches and report ``accepted``, then
        # (from #612) branch on the daemon's ``_model_running``.  Both were
        # decisions taken away from the state they were about: the daemon's
        # flag clears only after ``session.send_message`` returns, so a peer
        # that had finished its turn ~30s earlier still read as busy and the
        # message was queued behind a drain that had already run.
        #
        # ``deliver_prompt_to_session`` now offers the message to the peer's
        # OWN session, which answers atomically against its live
        # ``_is_running``: queued (a drain WILL collect it) or a turn is
        # driven.  A driven turn runs on the peer's own session, keeping its
        # id, so the cost lands on the peer and depletes the shared cid pool.
        # The body is already wrapped as untrusted content with the sender
        # stamped by the daemon, so the receiving model still sees the
        # boundary on either branch.
        from shared.message_delivery import BUSY, DELIVERED, QUEUED
        status = self.deliver_prompt_to_session(
            target_id, wrapped,
            source_id=sender_name or sender_session_id,
            source_type=SourceType.SIBLING,
            require_idle=at_cap,
        )
        if status == BUSY:
            return {"status": "refused",
                    "error": (f"send_to_sibling: {sibling_name!r} has "
                              f"{pending} messages waiting and has not been "
                              f"idle since. Let it work.")}
        reached: Dict[str, bool] = {"ok": status in DELIVERED}
        outcome = status

        # DIAGNOSTIC — the busy decision, at the moment it was made.
        #
        # A ``queued`` receipt and a stranded message look identical from
        # every witness a consumer has: the receipt says the peer was busy,
        # and whether a drain ever ran is only visible in the provider trace
        # (which is off unless JAATO_PROVIDER_TRACE is set).  So "queued and
        # drained" and "queued and stranded" are the same observation.
        #
        # ``thread_alive`` is the discriminator.  ``_model_running`` is set
        # at the top of ``model_thread`` and cleared in its finally, so a
        # True flag with NO live thread means the flag is stale -- a
        # different bug from a peer that is genuinely mid-turn, and they
        # need opposite fixes.
        #
        # Daemon log, not the provider trace: a diagnostic nobody can read
        # without setting an env var is one that will be read after the run
        # it was needed for.  Greppable token: SIBLING_DELIVERY.
        _thread = getattr(getattr(target, "server", None), "_model_thread", None)
        logger.info(
            "SIBLING_DELIVERY: from=%s to=%s target_session=%s "
            "replica_busy=%s thread_alive=%s outcome=%s bytes=%d",
            sender_name or sender_session_id, sibling_name, target_id,
            replica_busy,
            (_thread is not None and getattr(_thread, "is_alive", lambda: None)()),
            outcome, size,
        )
        if not reached.get("ok"):
            return {"status": "refused",
                    "error": (f"send_to_sibling: {sibling_name!r} "
                              f"{_delivery_failure_reason(status)}.")}

        with self._lock:
            self._sibling_exchanges[cid] = exchanges + 1
            if outcome == QUEUED:
                self._sibling_pending[target_id] = pending + 1
            else:
                # Driven: a turn was STARTED on the peer, and its end-of-turn
                # drain collects everything that was waiting -- so the backlog
                # is gone.  Keyed off the peer's OWN answer now, not off a
                # replica that could say "busy" about a session idle for half
                # a minute.
                self._sibling_pending.pop(target_id, None)

        return {"status": outcome,
                "sibling_name": sibling_name,
                "bytes": size}

    def send_to_named_session(
        self, cascade_driver_id: str, sibling_name: str, text: str,
    ) -> Dict[str, Any]:
        """Deliver operator text to a cascade member BY NAME.  DAEMON-SIDE.

        The client-tier counterpart of ``send_to_sibling`` (design §9): a
        human or script nudges a named stage without the model relaying and
        without knowing an opaque session id.  Named addressing is the whole
        point -- an id you never saw cannot be typed by a human, put in a
        profile, or written into a persona (§4).

        THREE DELIBERATE DIFFERENCES FROM THE SIBLING PATH, each because the
        sender is an OPERATOR and not a peer:

        1. ``SourceType.USER``, not ``SIBLING``.  The tier is an AUTHORITY
           statement, and a human reaching a session really does hold user
           authority -- so this is processed mid-turn like any user message.
           Labelling it SIBLING to get idle-only behaviour would be a lie
           about who is speaking.
        2. NOT wrapped as untrusted content.  The transport is the
           authentication boundary (IPC socket mode / WS bearer token), and
           wrapping an authenticated operator's words would teach the model
           to discount a boundary that exists for attacker-authored text.
        3. NO §8 caps.  Those bound an agent-to-agent ping-pong; the caps are
           daemon-side precisely because no single agent can see the whole
           conversation.  An operator is not in that loop and rate-limiting a
           human at 200 messages per cascade would be theatre.

        The §7 grammar refusal DOES still apply.  A client with a coordination
        channel must not be able to forge a permission or clarification
        answer through it -- those have their own typed request
        (``PermissionResponseRequest``), which is where authority is checked.
        Accepting them here would create a second, unchecked door to the same
        decision.

        COLD SESSIONS ARE NOT REVIVED.  ``session.wake`` is the primitive for
        that, and it has the signature checks and event-id dedup this path
        does not.  A nudge that silently resurrected a resting stage would be
        a much bigger act than it looks.

        Returns:
            ``{"status": "accepted"|"queued"|...}`` -- the SAME vocabulary
            ``send_to_sibling`` uses, because it is the same act.  There is
            no separate ``delivered``: it used to mean both "a turn is
            running" and "it is queued", which is exactly the one-word-two-
            outcomes shape that made these receipts untrustworthy.  Neither
            status claims the target read or acted on anything.
        """
        if not cascade_driver_id or not sibling_name:
            return {"status": "refused",
                    "error": "session.send requires a cascade id and a sibling name."}
        if not text or not text.strip():
            return {"status": "refused", "error": "session.send: message is empty."}

        violation = self._sibling_grammar_violation(text)
        if violation:
            return {"status": "refused",
                    "error": (f"session.send: <{violation}> is not accepted here. "
                              f"Answer permission and clarification requests "
                              f"through their own request type, where authority "
                              f"is checked.")}

        target_id, status = self._resolve_sibling(
            None, cascade_driver_id, sibling_name)
        if status == "absent":
            return {"status": "no_such_sibling",
                    "error": (f"session.send: no session named {sibling_name!r} "
                              f"in cascade {cascade_driver_id!r}.")}
        if status == "cold":
            return {"status": "sibling_cold",
                    "error": (f"session.send: {sibling_name!r} is resting "
                              f"(unloaded). Use session.wake to revive it.")}

        # The SAME decision every other sender makes.  This path used to
        # inject unconditionally and report ``delivered`` either way -- so an
        # IDLE target was queued rather than driven, and ``delivered`` covered
        # both "it is being worked on" and "it is sitting in a queue nobody
        # will pop".  One word for two outcomes is the shape this whole class
        # of bug takes.
        from shared.message_delivery import DELIVERED
        from shared.message_queue import SourceType
        # USER is a HIGH-priority tier, so an operator's words are picked up
        # MID-TURN rather than waiting for the turn to end.  That is the
        # authority difference between an operator and a sibling, and it is
        # carried by the tier -- not by which branch delivery takes.
        #
        # The branch itself is no longer decided here: ``deliver_prompt_to_
        # session`` offers the message to the target's OWN session, which
        # answers against its live ``_is_running``.  The daemon-side flag this
        # used to read clears ~30s later, so an operator send could be queued
        # behind a drain that had already run.
        outcome = self.deliver_prompt_to_session(
            target_id, text,
            source_id="operator",
            source_type=SourceType.USER,
        )
        reached: Dict[str, bool] = {"ok": outcome in DELIVERED}
        if not reached.get("ok"):
            return {"status": "refused",
                    "error": (f"session.send: {sibling_name!r} "
                              f"{_delivery_failure_reason(outcome)}.")}
        return {"status": outcome,
                "sibling_name": sibling_name,
                "session_id": target_id}

    def _resolve_sibling(
        self, viewer_session_id: str, cid: str, sibling_name: str,
    ) -> "Tuple[Optional[str], str]":
        """Resolve *sibling_name* within *cid* to ``(session_id, status)``.

        ``status`` is ``"live"`` / ``"cold"`` / ``"absent"``.  Live is checked
        first and on-disk second, mirroring ``build_sibling_roster``'s LIVE
        UNION COLD: an address stays owned by a session that has unloaded, so
        resolving only against ``self._sessions`` would report a resting
        sibling as absent -- and ``no_such_sibling`` would become a race
        rather than a fact.
        """
        with self._lock:
            for sid, s in self._sessions.items():
                if sid == viewer_session_id:
                    continue
                if s.cascade_driver_id == cid and s.sibling_name == sibling_name:
                    return sid, "live"
        try:
            # Scoped to the cascade's workspace.  Omitting it read the
            # DEFAULT storage dir, so every cold sibling resolved as
            # "absent" -- and the sender was told the address does not
            # exist, which is a different fact and needs a different
            # response than "the peer is resting".
            for info in self._get_persisted_sessions(
                workspace_path=self._cascade_storage_workspace(
                    viewer_session_id, cid),
            ):
                if info.session_id == viewer_session_id:
                    continue
                if (getattr(info, "cascade_driver_id", None) == cid
                        and getattr(info, "sibling_name", None) == sibling_name):
                    return info.session_id, "cold"
        except Exception as exc:  # noqa: BLE001
            # WARNING, not debug: without the on-disk half a resting sibling
            # is indistinguishable from one that never existed, and the
            # caller would be told the address is free.
            logger.warning(
                "sibling resolution could not read persisted sessions (%s) -- "
                "a cold sibling may be reported as absent", exc,
            )
        return None, "absent"

    @staticmethod
    def _roster_profile_name(session: Any) -> Optional[str]:
        """The profile a LIVE session was built from, or None."""
        server = getattr(session, "server", None)
        profile = getattr(server, "_profile", None) if server else None
        return getattr(profile, "name", None)

    def _known_sibling_addresses(
        self, workspace_path: Optional[str] = None,
    ) -> "List[Tuple[Optional[str], Optional[str]]]":
        """Every ``(sibling_name, cascade_driver_id)`` currently claimed.

        IN-MEMORY UNION ON-DISK, and the union is the point.  Sessions unload
        on ORPHAN, so a sibling resting on disk still owns its address.
        Checking only ``self._sessions`` handed that address to a second
        claimant the moment the first went cold -- and when the cold one
        revived, one cascade held two sessions answering to one name, with a
        perfectly healthy delivery receipt on whichever the roster happened to
        return.  That is the exact failure ``validate_sibling_name``'s own
        docstring warns about.

        Reachable in practice because a cascade is not fixed at creation: a
        reactor rule matching an agent-caused event can read the cid off it and
        mint further cid-stamped sessions later, long after the original stages
        have cycled through ORPHAN.

        Extracted so it can be tested by CALLING it.

        Args:
            workspace_path: Workspace whose persisted sessions to include.

        Returns:
            Pairs from live sessions and from the persisted index.
        """
        claimed: "List[Tuple[Optional[str], Optional[str]]]" = [
            (s.sibling_name, s.cascade_driver_id)
            for s in self._sessions.values()
        ]
        live_ids = set(self._sessions)
        try:
            for info in self._get_persisted_sessions(workspace_path=workspace_path):
                if info.session_id in live_ids:
                    continue          # already counted, and fresher in memory
                name = getattr(info, "sibling_name", None)
                if name:
                    claimed.append((name, getattr(info, "cascade_driver_id", None)))
        except Exception as exc:  # noqa: BLE001
            # WARNING, not debug: falling back to the in-memory view silently
            # is how a duplicate address gets issued.
            logger.warning(
                "sibling-address uniqueness could not read persisted sessions "
                "(%s) -- a name held by a COLD sibling may be reissued", exc,
            )
        return claimed

    def _create_session_impl(
        self,
        client_id: str,
        session_name: Optional[str] = None,
        workspace_path: Optional[str] = None,
        env_overrides: Optional[Dict[str, str]] = None,
        profile_name: Optional[str] = None,
        provisioned: bool = False,
        created_by: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_params: Optional[Dict[str, str]] = None,
        system_instruction_override: Optional[str] = None,
        suppress_base_instructions: bool = False,
        initial_session_state: Optional[Dict[str, Any]] = None,
        inline_profile_data: Optional[Dict[str, Any]] = None,
        config_root: Optional[str] = None,
        apparmor: Optional[bool] = None,
        cascade_driver_id: Optional[str] = None,
        budget_control: Optional[Dict[str, Any]] = None,
        budget_usage: Optional[Dict[str, float]] = None,
        sibling_name: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Implementation of session creation, called via ``Context().run()``.

        See :meth:`create_session` for the isolation rationale and the
        public docstring.

        Args:
            client_id: The requesting client.
            session_name: Optional name (auto-generated if not provided).
            workspace_path: Client's working directory for file operations.
            env_overrides: Optional env vars that override the .env file
                          (e.g., JAATO_PROVIDER/MODEL_NAME from post-auth wizard).
            profile_name: Optional profile name for runtime config (model,
                provider, plugins, GC, env). Loaded from ``.jaato/profiles/``.
            provisioned: True if the workspace was auto-provisioned by the
                server (e.g., for WebSocket clients).  When True, the
                workspace_path is server-managed and should not be overridden
                by client config.
            created_by: Authenticated user who created the session.
            agent_name: Optional agent name. If provided, the agent's rendered
                markdown is ONE LAYER of the assembled system instructions -- not
                the whole of them; see IPCClient.create_session. Resolved
                from ``.jaato/agents/`` and ``.jaato/prompts/``.
            agent_params: Parameter values for the agent's ``{{param}}``
                placeholders.
            system_instruction_override: If provided, replaces the assembled
                system instruction passed to the model.  Use the empty string
                to send no system message at all.  Full replacement — the
                agent prompt and plugin instructions are also discarded.
            suppress_base_instructions: Partial suppression — drop only
                the BASE layer (``.jaato/instructions/*.md`` + any
                premium-provided baseline) while keeping the agent prompt,
                plugin instructions, and framework constants.  The usual
                choice for fitting a session into a small model's context
                window (the BASE layer is typically the single biggest
                token consumer).  Ignored when ``system_instruction_override``
                is also set.
            initial_session_state: Optional opaque dict seeded onto the
                new session's session-attached-state container BEFORE
                ``_run_session_hooks`` fires.  Consumer hooks read keys
                via ``session.get_session_state(...)`` to rebuild
                runtime structure (e.g. premium pseudonymization
                instantiates a ``PseudonymTable`` from the encrypted
                blob carried under ``"pseudonym_table"`` and registers
                a provider for it on the new session).  Forking the
                CURRENT state of an existing session is the caller's
                job: snapshot the source via
                ``source.get_all_session_state()`` before calling this
                method (the snapshot invokes registered providers so
                live values are captured, not stale set-state values).
                Values must be JSON-serialisable; encrypt before
                attach if confidentiality is needed (the framework
                treats values as opaque).
            inline_profile_data: Optional dict carrying the same shape
                as a profile JSON file on disk (model, provider,
                plugins, plugin_configs, system_instructions, gc, etc.).
                Lets SDK consumers create sessions with a custom
                runtime config without having to write a profile to
                ``.jaato/profiles/``.  Mutually exclusive with
                ``profile_name``.  The ``model`` key is required (no
                silent fallback) so caller intent is explicit.  The
                helper ``shared.plugins.subagent.config.build_inline_profile``
                does the parsing.

        Returns:
            The session ID (empty string on failure).
        """
        # Validate the sibling ADDRESS before anything is allocated: a rejected
        # name must not burn a session id, and the caller must learn about a
        # collision at session.new rather than discovering at send time that
        # its messages have been reaching somebody else.
        if sibling_name is not None:
            _bad = validate_sibling_name(
                sibling_name, cascade_driver_id,
                self._known_sibling_addresses(workspace_path),
            )
            if _bad:
                # TELL THE CLIENT.  A refusal a consumer cannot see is not a
                # refusal -- it is a hang that happens to be correct
                # server-side.  Returning "" alone left the caller blocking
                # until its own timeout, and the router's falsy branch emits
                # an AUTH-PROVIDER hint, so a naming violation surfaced as a
                # misleading suggestion about credentials.
                #
                # Worse for a COLLISION: the client's ``_await_session_info``
                # would pick up the SessionInfoEvent of the session created
                # moments earlier and return ITS id, so the caller believed it
                # had created a sibling it had not, holding an address clash it
                # could not see.  An explicit error ends the wait instead.
                #
                # ``recoverable=True`` and an ErrorEvent match every other
                # session.new failure, which the SDK documents as arriving
                # that way.
                logger.error("create_session refused: %s", _bad)
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"session.new: {_bad}",
                    error_type="InvalidSiblingName",
                    recoverable=True,
                    request_id=request_id,
                ))
                return ""

        # Claim the id ATOMICALLY — see _allocate_session_id for why a
        # plain check-then-act here handed the same id to concurrent creates.
        timestamp = datetime.now()
        session_id = self._allocate_session_id(workspace_path)
        name = session_name or f"Session {timestamp.strftime('%Y-%m-%d %H:%M')}"

        # Get env_file from client config or derive from workspace path
        # Sessions are workspace-bound: the workspace determines the .env file,
        # which in turn determines the provider.
        client_config = self._client_config.get(client_id, {})
        session_env_file = client_config.get('env_file')

        # Inherit config_root from the client's handshake (set via
        # ``ClientConfigRequest.config_root``) when the caller didn't
        # pass one explicitly.  An explicit per-session override (e.g.
        # from a future inline-spec field) takes precedence over the
        # client-level default.
        if config_root is None:
            config_root = client_config.get('config_root')
        import os
        if not session_env_file and workspace_path:
            # Default to workspace/.env
            workspace_env = os.path.join(workspace_path, '.env')
            if os.path.exists(workspace_env):
                session_env_file = workspace_env

        # Headless reactor-spawned sessions never get a client-config
        # entry (their ``client_id`` is the synthetic ``_headless`` id),
        # so the lookup above falls through.  When a ``config_root``
        # was supplied, the project's ``.env`` typically lives **next
        # to** the config_root dir (the orchestrator passes
        # ``<project>/.jaato`` as config_root, so ``<project>/.env``
        # is one level up).  Honoring that convention here means
        # reactor-spawned sessions inherit env-driven config — most
        # importantly the ``JAATO_REDACTION_ENABLED=off`` flag — so
        # PII pseudonymization stays disabled when the workspace owner
        # asked for it to be off.
        if not session_env_file and config_root:
            config_root_parent_env = os.path.join(
                os.path.dirname(os.path.abspath(config_root)), '.env',
            )
            if os.path.exists(config_root_parent_env):
                session_env_file = config_root_parent_env

        logger.info(f"Creating session for client {client_id}: env_file={session_env_file}")
        logger.info(f"  Client config: {client_config}")

        # Resolve agent profile if requested.  The two paths
        # (named-on-disk vs inline-spec from the SDK) are mutually
        # exclusive — a request that supplies both is rejected up
        # front rather than silently picking one.
        profile = None
        if profile_name and inline_profile_data:
            self._emit_to_client(client_id, ErrorEvent(
                error=(
                    "session.new: 'profile' name and inline 'spec' are "
                    "mutually exclusive — pass exactly one"
                ),
                error_type="InvalidSessionSpec",
                recoverable=True,
            ))
            self._release_session_id(session_id)
            return ""

        if profile_name:
            resolve_path = workspace_path or str(pathlib.Path.home())
            profile, error = self._resolve_profile(
                profile_name, resolve_path,
                config_root=config_root,
                env_file=session_env_file,
            )
            if profile is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error=error,
                    error_type="ProfileNotFoundError",
                    recoverable=True,
                ))
                return ""
            logger.info(f"  Using profile: {profile_name}")
        elif inline_profile_data is not None:
            from shared.plugins.subagent.config import build_inline_profile
            if not inline_profile_data.get("model"):
                self._emit_to_client(client_id, ErrorEvent(
                    error=(
                        "session.new: inline spec requires a 'model' field "
                        "— defaults are not silently applied"
                    ),
                    error_type="InvalidSessionSpec",
                    recoverable=True,
                ))
                return ""
            try:
                profile = build_inline_profile(inline_profile_data)
            except ValueError as exc:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"session.new: {exc}",
                    error_type="InvalidSessionSpec",
                    recoverable=True,
                ))
                return ""
            logger.info(
                f"  Using inline profile spec (model={profile.model}, "
                f"plugins={profile.plugins})"
            )

        # Resolve agent if requested — the agent's rendered markdown
        # is one LAYER of the assembled system instructions.
        agent_instructions = None
        if agent_name:
            agent_result = self._resolve_agent(
                agent_name, agent_params, workspace_path, config_root=config_root,
            )
            if agent_result is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"Agent '{agent_name}' not found in .jaato/agents/ or .jaato/prompts/",
                    error_type="AgentNotFoundError",
                    recoverable=True,
                ))
                return ""
            agent_instructions = agent_result["system_instructions"]
            # Use agent's default_profile if no explicit --profile was provided
            if not profile_name and agent_result.get("default_profile"):
                default_prof = agent_result["default_profile"]
                resolve_path = workspace_path or str(pathlib.Path.home())
                profile, error = self._resolve_profile(
                    default_prof, resolve_path,
                    config_root=config_root,
                    env_file=session_env_file,
                )
                if profile:
                    logger.info(f"  Using agent's default profile: {default_prof}")
                else:
                    logger.warning(f"  Agent's default_profile '{default_prof}' not found: {error}")
            if agent_result.get("missing_params"):
                logger.warning(f"  Agent has unresolved params: {agent_result['missing_params']}")
            logger.info(f"  Using agent: {agent_name}")

            # Set agent instructions on the profile (overrides deprecated
            # system_instructions if present).
            if profile:
                if profile.system_instructions:
                    logger.info("  Agent instructions override profile's system_instructions (deprecated)")
                profile.system_instructions = agent_instructions
            else:
                # No explicit --profile: synthesize a minimal profile to CARRY
                # the agent persona. runner_spawn reads system_instructions from
                # ``profile.system_instructions``; with ``profile=None`` the
                # resolved persona was extracted then DROPPED here, so an agent
                # specified without a profile went bare. Every other field takes
                # its default (plugins=[], model/provider=None, gc=None,
                # spawn_payload_schema=None, suppress_base_instructions=False) —
                # byte-identical to the ``profile=None`` path this replaces (same
                # plugins=[]/model=None downstream, same False/None reads in the
                # pre-spawn profile-gated branches). So it adds ONLY the persona,
                # with zero plugin / model / provider / gating change.
                from shared.plugins.subagent.config import SubagentProfile
                profile = SubagentProfile(
                    name=agent_name,
                    description=agent_result.get("description", ""),
                    system_instructions=agent_instructions,
                )

        # ── Spawn-payload schema validation ──────────────────────────
        # Symmetric to the subagent plugin's check at the function-call
        # boundary: when the resolved profile declares
        # ``spawn_payload_schema``, validate the caller-supplied
        # ``agent_params`` dict against it BEFORE creating the session.
        # Catches missing-required-field bugs from BOTH spawn paths
        # (model-driven spawn_subagent AND reactor-side
        # create_headless_session) at a single chokepoint.
        if (
            profile is not None
            and getattr(profile, 'spawn_payload_schema', None) is not None
        ):
            try:
                from shared.spawn_schema_loader import resolve_spawn_schema
                resolved_schema = resolve_spawn_schema(
                    profile.spawn_payload_schema,
                    workspace_path=workspace_path,
                    config_root=config_root,
                )
                if resolved_schema is not None:
                    import jsonschema
                    try:
                        jsonschema.validate(
                            instance=agent_params or {},
                            schema=resolved_schema,
                        )
                    except jsonschema.ValidationError as exc:
                        required = list(resolved_schema.get('required') or [])
                        missing = [
                            f for f in required
                            if not agent_params or f not in agent_params
                        ]
                        details = (
                            f"missing required fields: {missing}. "
                            if missing
                            else f"first failure: {exc.message}. "
                        )
                        err_msg = (
                            f"create_session(profile={profile_name!r}) failed "
                            f"agent_params validation: {details}"
                            f"The profile requires agent_params matching its "
                            f"spawn_payload_schema "
                            f"({profile.spawn_payload_schema!r})."
                        )
                        logger.error(err_msg)
                        self._emit_to_client(client_id, ErrorEvent(
                            error=err_msg,
                            error_type="SpawnPayloadValidationError",
                            recoverable=True,
                        ))
                        return ""
            except Exception as exc:
                # Schema-loader bug or jsonschema crash — log and skip
                # validation rather than blocking session creation.
                logger.warning(
                    "spawn_payload_schema validation skipped for profile "
                    "%s: %s", profile_name, exc,
                )

        # Create JaatoServer for this session
        # Provider is determined by env_file, with optional overrides.
        # ``agent_name`` propagates to the main agent's ``agent_id`` so
        # reactor rules and event consumers can match on the agent's
        # logical identity (e.g. ``"coordinator"``).  Without this, all
        # AgentCompletedEvents would carry ``agent_id="main"`` regardless
        # of which agent the session was launched with.
        # Resolve effective suppress_base_instructions: UNION of the explicit
        # kwarg (CLI --no-instructions / SDK bool-or-dict) and the profile's
        # field.  A piece is suppressed if EITHER source asks for it.  Both
        # normalize to the canonical frozenset (see instruction_suppression).
        # Caller-supplied CEILING, applied BEFORE the envelope is built.
        # ``profile.budget_control`` -> the envelope's wire field is the only
        # route a budget takes to the runner, so attaching it here means there
        # is no window in which the session exists unbudgeted.  Same ordering
        # the reload path uses (#583).  An authored profile budget wins.
        self._attach_budget_ceiling(budget_control, profile, session_id)

        effective_suppress_base = normalize_suppression(
            suppress_base_instructions
        ) | (
            getattr(profile, "suppress_base_instructions", frozenset())
            if profile else frozenset()
        )

        # Phase 3 §3.12.0: route the construction +
        # pre-init-hooks + initialize + Session-record assembly
        # through the unified _bootstrap_session helper.  All four
        # session-creation paths (IPC, disk-restore, ephemeral, WS)
        # will eventually funnel through here; §3.12.0 ships only
        # the IPC migration as the focal commit.
        envelope = BootstrapEnvelope(
            session_id=session_id,
            workspace_path=workspace_path,
            name=name,
            description=None,
            client_id=client_id,
            env_file=session_env_file,
            profile=profile,
            # Carry the UNRESOLVED inline spec so the created Session can
            # stash it for disk-restore (persisted as profile_spec).  Only
            # set for inline-spec sessions; None for named/no-profile.
            inline_profile_spec=inline_profile_data,
            sibling_name=sibling_name,
            agent_name=agent_name,
            system_instruction_override=system_instruction_override,
            suppress_base_instructions=effective_suppress_base,
            env_overrides=env_overrides,
            config_root=config_root,
            instruction_token_cache=self._instruction_token_cache,
            # Phase 4 §D: forward the originating create_session
            # ``agent_params`` so build_session_envelope can put them
            # on the wire envelope.  Without this, runner-side prefetch
            # scripts (e.g. tmux_pane in the documenter harness) see
            # empty agent_params and emit their "missing keys" error.
            agent_params=dict(agent_params or {}),
            provisioned=provisioned,
            created_by=created_by,
            timestamp=timestamp,
            # PR-A (2026-05-14): forward explicit AppArmor caller intent
            # to the bootstrap helper.  None = no override (consult
            # client_config then profile.apparmor); True/False short-
            # circuits that chain.
            apparmor=apparmor,
            # Phase 2 cascade-sharing (server 0.6.144+): forward the
            # opaque cascade tenant ID supplied by the IPC client.
            # ``None`` = standalone session.  See
            # docs/design/runner-cascade-sharing.md §4.1.
            cascade_driver_id=cascade_driver_id,
            # Server 0.6.166+: bootstrap-time emit now routes through
            # :meth:`_route_bootstrap_event` so cascade observers
            # receive events fired during init (AgentCreatedEvent,
            # initial AgentStatusChangedEvent, etc.) — pre-0.6.166
            # these went only to ``client_id`` via _emit_to_client,
            # which transport-dropped them for headless cascade
            # sessions (client_id == _HEADLESS_CLIENT_ID).  See the
            # _route_bootstrap_event docstring for the audit context.
            on_event_during_init=lambda e: self._route_bootstrap_event(
                client_id, cascade_driver_id, e, session_id,
            ),
        )
        server, session = self._bootstrap_session(envelope)
        if server is None or session is None:
            # server.initialize() failed; core.py already emitted a
            # detailed ConfigurationError to the in-init sink.
            self._release_session_id(session_id)
            return ""

        logger.info(f"Server initialized successfully for session {session_id}")

        # ---------------------------------------------------------------
        # NEVER HAND BACK A HANDLE TO A SESSION THAT CANNOT RUN.
        #
        # The cascade ceiling was checked only in the runner spawn, which
        # happens AFTER this method has emitted SessionInfoEvent and returned
        # the id.  So an exhausted pool produced an ordinary session id, and
        # the refusal -- correct, well-typed, correctly logged -- arrived too
        # late to be the answer: the caller's create-wait had already been
        # satisfied by the SessionInfoEvent.  Measured: pool of 1200 tokens,
        # one turn charging 1200, and the next create still returned a
        # handle while the daemon logged ``cascade_remaining=0.0``.
        #
        # Refusing HERE, before the handle exists, is what makes the SDK's
        # ``SessionRefused`` contract true on the one path that produces
        # ``CascadeExhaustedError``.  The spawn-side check stays: it is the
        # backstop for sessions that reach a ceiling by a route this one does
        # not see (a reload, a fork pre-charged with usage).
        #
        # A profile carrying its OWN ``budget_control`` is deliberately
        # exempt, matching the spawn-side rule: it keeps separate books, is
        # not clamped, and an exhausted pot does not refuse it.
        if cascade_driver_id:
            _pool = self.get_cascade_budget(cascade_driver_id)
            if _pool is not None and getattr(
                    server, "_draws_on_parent_budget", True):
                from shared.budget_control import CascadeExhaustedError
                try:
                    _pool.child_config(
                        getattr(server, "_profile", None)
                        and getattr(server._profile, "budget_control", None))
                except CascadeExhaustedError as exc:
                    self._emit_cascade_refusal(
                        client_id, session_id, exc, request_id=request_id)
                    self._release_session_id(session_id)
                    try:
                        server.shutdown()
                    except Exception:  # noqa: BLE001 — best-effort
                        logger.debug(
                            "server.shutdown after cascade refusal raised",
                            exc_info=True,
                        )
                    return ""

        # Caller-supplied USAGE, pre-charged onto the fresh session via the
        # same RPC a reload uses.  Safe in this window because the session has
        # not served a turn.  Without it a fork starts at ZERO against a full
        # ceiling, so N branches from an exhausted source each run the budget
        # again -- branching becomes a way out of the ceiling.
        if budget_usage:
            self._restore_budget_usage(
                server, budget_usage, None, session_id,
            )

        # Switch to session-based event emission now that init is complete
        server.set_event_callback(lambda e: self._emit_to_session(session_id, e))

        # Configure TODO plugin with session-scoped storage
        if workspace_path:
            session_dir = self._session_storage_dir(workspace_path) / session_id
            self._configure_todo_storage(server, session_dir)

        # Apply client-specific config (e.g., presentation context)
        self._apply_client_config_to_server(client_id, server)

        # Register callback for when auth completes (if it was pending)
        def on_auth_complete():
            self._emit_to_session(session_id, self._build_session_info_event(session))
            self._emit_to_session(session_id, SystemMessageEvent(
                message=f"Session created: {name} ({session_id})",
                style="info",
            ))
        server.set_auth_complete_callback(on_auth_complete)

        with self._lock:
            # Server 0.6.164+ (Bug B real root cause): stamp the
            # session-object's cid BEFORE adding to ``_sessions`` so
            # both ``_dispatch_to_cascade_clients`` (Phase 1) and
            # ``_record_cid_session_activity`` (PR-188) read the
            # correct value.  Pre-0.6.164 the dataclass had no
            # ``cascade_driver_id`` field; getattr returned None for
            # every cascade session, defeating dispatch + GC-skip.
            session.cascade_driver_id = cascade_driver_id
            # Stamped here for the same reason as the cid above: BEFORE the
            # session enters ``_sessions``, because the spawn that can refuse
            # it looks the session up by id and would otherwise find no
            # correlation to echo.
            session.create_request_id = request_id
            # Whether this child draws on the parent's shared pot or keeps
            # its own books (see _spawn_session_runner_unconditional).
            session.draws_on_parent_budget = getattr(
                server, "_draws_on_parent_budget", True)
            self._sessions[session_id] = session
            # ``_sessions`` is authoritative from here; drop the claim.
            self._reserved_session_ids.discard(session_id)
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id
            # Server 0.6.161+ (Bug B): record cid activity so the
            # cascade-client GC sweep treats this cascade as alive
            # across inter-stage gaps.  See
            # :meth:`_record_cid_session_activity`.
            self._record_cid_session_activity(
                session.cascade_driver_id,
            )

        # Seed session-attached state BEFORE hooks fire.  Consumer
        # hooks (e.g. premium pseudonymization) read these keys via
        # session.get_session_state(...) to rebuild runtime structure
        # and register providers for incrementally-mutated state.  Any
        # JSON-serialisability error surfaces here at the call site
        # (set_session_state validates the value at attach time).
        # Phase 3 §7c step 6.6.3.6: forward to runner-side via
        # the existing ``session.set_session_state`` RPC (§3.3c
        # precursor) instead of reaching into the daemon-side
        # session via ``server.get_session()``.
        if initial_session_state:
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is not None:
                forwarder = getattr(
                    rpc, "session_set_state_threadsafe", None,
                )
                if callable(forwarder):
                    for key, value in initial_session_state.items():
                        try:
                            forwarder(key, value, timeout=2.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "set_session_state forward failed for "
                                "key=%r: %s", key, exc,
                            )

        # Run session hooks after the Session is stored so hooks can
        # call get_session() to modify session attributes (e.g. sandbox_mode).
        self._run_session_hooks(server, session_id)

        # Start workspace file monitor
        if workspace_path:
            self._start_workspace_monitor(session_id, workspace_path)

        # Save initial state to disk
        self._save_session(session)

        logger.info(f"Session created: {session_id} ({name})")

        # Note: We don't call emit_current_state() here because the client
        # already received all events during initialize() via direct emission.

        # Send SessionInfoEvent to confirm session creation.  When auth is
        # pending the tool/model lists may be incomplete, but the client
        # needs the session_id immediately.  on_auth_complete() will send
        # an updated SessionInfoEvent once the provider is fully ready.
        try:
            _info = self._build_session_info_event(session)
            # Echo the correlation id so the caller can tell THIS answer
            # from a concurrent create's.  Without it the client matched on
            # shape and a stale buffered event satisfied the wrong wait.
            _info.request_id = request_id
            self._emit_to_client(client_id, _info)
        except Exception as exc:
            logger.error("Failed to build SessionInfoEvent: %s", exc, exc_info=True)
            # Send a minimal SessionInfoEvent so the client can still proceed
            self._emit_to_client(client_id, SessionInfoEvent(
                session_id=session.session_id,
                session_name=session.name,
                model_provider=session.server.model_provider if session.server else "",
                model_name=session.server.model_name if session.server else "",
            ))

        if not server.auth_pending:
            self._emit_to_client(client_id, SystemMessageEvent(
                message=f"Session created: {name} ({session_id})",
                style="info",
            ))

        return session_id

    # ------------------------------------------------------------------
    # Headless session creation (for daemon extensions / reactors)
    # ------------------------------------------------------------------

    _HEADLESS_CLIENT_ID = "_headless"

    # Cap on the wake dedup LRU (session.wake event_id de-duplication).
    _WAKE_DEDUP_CAP = 1024

    def create_headless_session(
        self,
        profile_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        workspace_path: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        initial_history: Optional[List[Any]] = None,
        initial_session_state: Optional[Dict[str, Any]] = None,
        session_name: Optional[str] = None,
        config_root: Optional[str] = None,
        agent_params: Optional[Dict[str, str]] = None,
        apparmor: Optional[bool] = None,
        cascade_driver_id: Optional[str] = None,
        inline_profile_data: Optional[Dict[str, Any]] = None,
        budget_control: Optional[Dict[str, Any]] = None,
        budget_usage: Optional[Dict[str, float]] = None,
        sibling_name: Optional[str] = None,
    ) -> str:
        """Create a top-level session not attached to any real client.

        Intended for daemon extensions (e.g., reactor rules) that need to
        spawn an independent session in response to an event.  The session
        is fully initialized and runs like any other, but its client-facing
        events are silently dropped — the transport layer ignores events
        addressed to the synthetic ``_headless`` client (IPC: unknown
        client → silent return; WS: same).

        The session **is** visible on the EventBus, so any reactor
        subscriptions (via the session hook) observe it normally.

        Args:
            profile_name: Optional profile to use (resolved from
                ``.jaato/profiles/``).  Controls model, plugins, GC.
            agent_name: Optional agent to use (resolved from
                ``.jaato/agents/``).  The agent's markdown becomes the
                session's system instructions.  This is typically the
                mandatory parameter — it defines *what* the session does.
            workspace_path: Workspace directory.  Defaults to the daemon's
                cwd if not provided.
            initial_prompt: If set, a ``SendMessageRequest`` is dispatched
                to the new session immediately after creation.
            initial_history: Optional list of ``Message`` objects to
                seed the new session's conversation history with.  Used
                by spawn-from-snapshot callers (premium handoff via
                ``fork_session_from_history``, waypoint fork-to-session)
                so the new agent picks up where another session left off.
                Loaded after ``server.initialize()`` succeeds and before
                any ``initial_prompt`` is dispatched.  Typed as ``Any``
                here to avoid a top-level SDK import; concrete type is
                ``List[jaato_sdk.plugins.model_provider.types.Message]``.
            initial_session_state: Optional opaque dict seeded onto the
                new session's session-attached-state container BEFORE
                its session-hook fires.  Threaded through to
                :meth:`create_session`; see that method's docstring for
                semantics and the fork-carry contract.  Forking the
                CURRENT state of an existing session is the caller's
                job — snapshot the source via
                ``source.get_all_session_state()`` first so the dict
                reflects live values (registered providers are
                invoked), not stale set-state.
            session_name: Optional human-readable name.
            cascade_driver_id: Phase 2 cascade-sharing tenant ID
                (server 0.6.144+).  When non-None, the spawned
                runner's pool slot is acquired with cascade-affinity
                routing — sessions sharing the same ID reuse the
                same slot (warm imports + warm plugin state + warm
                LSP server connections survive across cascade
                stages).  ``None`` (default) = standalone session,
                no slot reuse.  Forwarded verbatim to
                :meth:`_create_session_impl` → BootstrapEnvelope
                → spawn_session_runner → PoolManager.acquire_slot.
                Premium ``ActionContext.create_session`` passes
                this through from the originating reactor handler
                (`cascade_after_*.py`).  See
                ``docs/design/runner-cascade-sharing.md``.
            inline_profile_data: Optional dict carrying an inline
                ``SubagentProfile`` shape (model/provider/plugins/…), the
                canonical ``build_inline_profile`` input.  Forwarded to
                :meth:`create_session` so callers with a profile built
                on-the-fly (rather than name-resolved from
                ``.jaato/profiles/``) can spawn a headless session — used by
                the ephemeral remote-spawn path, which receives a serialized
                profile from the origin peer.  Mutually exclusive with
                ``profile_name`` in practice.

        Returns:
            The session ID (empty string on failure).
        """
        session_id = self.create_session(
            client_id=self._HEADLESS_CLIENT_ID,
            session_name=session_name,
            workspace_path=workspace_path,
            profile_name=profile_name,
            agent_name=agent_name,
            agent_params=agent_params,
            initial_session_state=initial_session_state,
            config_root=config_root,
            apparmor=apparmor,
            budget_control=budget_control,
            budget_usage=budget_usage,
            sibling_name=sibling_name,
            cascade_driver_id=cascade_driver_id,
            inline_profile_data=inline_profile_data,
        )
        if not session_id:
            # Server 0.6.50.1+: log at WARNING so reactor callers
            # (``ctx.create_session`` → ``create_headless_session``) get
            # a diagnostic trail when their headless spawn silently
            # fails.  ErrorEvents fired during create_session go to the
            # ``_HEADLESS_CLIENT_ID`` synthetic client which the
            # transport drops — without this log line, the reactor
            # caller sees only an empty session_id with no clue why
            # (cascade v8 finding from 7:3, 2026-05-06: a buggy
            # prefetch raised DynamicInstructionsError, the underlying
            # core.py:initialize correctly aborted, but the empty
            # session_id propagated up to the reactor handler without
            # any visible log entry).  The headless caller still has
            # to handle ``""`` as failure, but at least now there's a
            # log breadcrumb explaining WHY.
            logger.warning(
                "create_headless_session: session creation returned "
                "empty (profile=%s, agent=%s, workspace=%s).  Check "
                "earlier ERROR logs in this turn for the underlying "
                "cause (typically: prefetch failed, auth missing, "
                "provider connect failed).",
                profile_name, agent_name, workspace_path,
            )
            return ""

        if initial_history:
            # Seed history before any model turn fires.  Look up the
            # session record we just created and call the JaatoSession's
            # public initial-history primitive.  Any failure is
            # surfaced — partial state (session created, history not
            # loaded) would silently mislead the caller.
            with self._lock:
                session_record = self._sessions.get(session_id)
            if session_record is None:
                logger.error(
                    "create_headless_session: session %s vanished after "
                    "create; cannot seed initial_history",
                    session_id,
                )
                return ""
            # Phase 3 §7c step 6.6.3.6: forward to runner-side
            # via the new ``session.set_initial_history`` RPC
            # (§7c step 6.6.1.1, commit 3f859e3a) instead of
            # reaching into the daemon-side session.
            rpc = getattr(session_record.server, "_runner_rpc", None)
            if rpc is not None:
                forwarder = getattr(
                    rpc, "session_set_initial_history_threadsafe", None,
                )
                if callable(forwarder):
                    forwarder(initial_history, timeout=10.0)

        if initial_prompt:
            from jaato_sdk.events import SendMessageRequest
            self.handle_request(
                self._HEADLESS_CLIENT_ID,
                session_id,
                SendMessageRequest(text=initial_prompt),
            )

        return session_id

    def get_persisted_history(
        self,
        session_id: str,
        workspace_path: Optional[str] = None,
    ) -> Optional[List[Any]]:
        """Read a persisted session's conversation history from disk WITHOUT
        loading the session (no JaatoServer built, no runner spawned).

        The read-only counterpart to the history-restore half of
        :meth:`_load_session_impl`: it does the same ``self._session_plugin.load``
        record read and returns ``state.history`` — the list of ``Message``
        objects suitable for ``create_headless_session(initial_history=...)`` —
        but stops short of building a server or spawning a runner.

        This is the jaato-server piece of the §9 fork-from-PERSISTED resume: a
        reactor (e.g. reliability_revive) forking a continuation from an
        UNLOADED/ended session reads the persisted history here and feeds it to
        :meth:`create_headless_session` (the premium ActionContext wraps the two
        as ``ctx.fork_from_persisted_session``).  Contrast
        ``JaatoSession.get_history`` / ``fork_from_session``, which snapshot a
        LIVE session's in-memory history and so require it loaded.

        Args:
            session_id: The persisted session to read.
            workspace_path: Workspace whose ``.jaato/sessions/`` holds the
                record (same contract as :meth:`_load_session`).  ``None`` falls
                back to the session plugin's default storage location.

        Returns:
            The session's history (list of ``Message`` objects), or ``None`` if
            no record exists on disk for ``session_id``.
        """
        storage_dir = (
            self._session_storage_dir(workspace_path) if workspace_path else None
        )
        try:
            state = self._session_plugin.load(session_id, storage_dir=storage_dir)
        except FileNotFoundError:
            logger.debug(
                "get_persisted_history: session %s not found on disk", session_id)
            return None
        except Exception as exc:  # noqa: BLE001 — a missing/corrupt record must not crash the caller
            logger.error(
                "get_persisted_history: failed to read %s: %s", session_id, exc)
            return None
        return getattr(state, "history", None)

    def deliver_prompt_to_session(
        self,
        target_session_id: str,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[Any] = None,
        require_idle: bool = False,
    ) -> str:
        """Deliver a prompt to a loaded session and REPORT what happened.

        The status-returning form of :meth:`inject_prompt_to_session`, and
        the one every new caller should use.  It answers the question an
        injecting caller actually has -- **after this returns, will the
        target act on the message?** -- which a boolean cannot express,
        because "queued into a live turn" and "the target is dead" were
        both ``False``-or-``True`` depending on which failure you hit.

        Returns one of the ``shared.message_delivery`` constants:

        ``ACCEPTED``
            The target was idle, so a turn was STARTED on it.
        ``QUEUED``
            The target is mid-turn; its running turn will drain the message.
        ``TERMINATED``
            The target is loaded but terminal (``_terminal_reason`` stamped
            by the model thread on an error or an exhausted budget) and will
            run no further turns.  Read from the target's OWN stamp -- never
            inferred from silence, because a slow target and a dead one
            produce identical nothing and a caller that infers cannot be
            wrong and know it.
        ``NO_SESSION``
            No session with that id is loaded.  Deliberately distinct from
            ``TERMINATED``: "gone" and "dead but present" are different
            situations for a driver.
        ``BUSY``
            Only when *require_idle* is set: the target confirmed it is
            mid-turn, so nothing was enqueued.  Backpressure that asks the
            target rather than guessing from a replica.
        ``UNREACHABLE``
            Loaded and live, but NOTHING WAS PUT IN FLIGHT: no server
            attached, no runner channel, a runner too old to accept the offer
            verb, or a drive that failed after the target answered
            ``needs_turn``.  A transport fault, not a decision by the target,
            which is why it is not ``refused``.  RETRY IS SAFE -- nothing was
            enqueued, so it cannot duplicate.
        ``NOT_CONFIRMED``
            An offer WAS made and its answer was lost (the RPC raised or timed
            out).  The message may be in the target's queue right now, or may
            never have arrived; from here those are indistinguishable.  RETRY
            MAY DUPLICATE.

        Only ``ACCEPTED`` and ``QUEUED`` mean the message will be acted on
        (``message_delivery.DELIVERED``).

        UNREACHABLE AND NOT_CONFIRMED ARE THE SAME AXIS, SPLIT ONCE.

        They were one word, and the prose reason attached to it described
        only NOT_CONFIRMED's case.  For the four structural producers that
        sentence was FALSE, not vague -- it warned about a duplicate that
        could not exist, so a careful sender declined to re-send a message
        that had definitely never arrived.  Mechanism detail below that
        (which of the four) stays in the log: a sender cannot act on "no
        runner channel" differently from "no server attached", and a word it
        cannot act on is one more thing to get wrong.

        AN IDLE TARGET IS DRIVEN, NOT INJECTED.

        ``JaatoSession.inject_prompt`` starts a turn only while
        ``_on_continuation_needed`` is installed -- and that is for the
        DURATION of a ``session.send_message`` RPC, not whenever the session
        happens to be idle.  So injecting into an idle target that nobody is
        driving queues the message and NOTHING drains it: the call reported
        success and the message was discarded on unload.  That made the
        documented cascade-watchdog pattern a no-op, and worse, self-sealing
        -- the nudge sent to rescue a stalled session landed in the same dead
        queue as the message it was rescuing, so the recovery mechanism could
        not work by construction.

        ``send_to_sibling`` fixed its own copy of this in #612; this is the
        shared primitive, so the fix belongs here rather than at each caller
        -- see ``shared.message_delivery`` for why cloning the queue-or-drive
        decision is what produced the bug in the first place.

        Thread-safe.

        Args:
            target_session_id: The destination session.  Must be loaded in
                ``self._sessions`` (not just persisted on disk -- attach
                first if needed).
            text: The prompt to deliver.
            source_id: Identifier of the sender (e.g. ``"reactor"``,
                ``"webhook:github"``).  Defaults to ``"unknown"`` downstream.
            source_type: ``SourceType`` enum value controlling priority.  Any
                member of the enum; not re-listed here because a prose copy
                of it drifts (``sibling`` shipped while this said five).
                Whether a tier may interrupt a turn in progress is declared
                by ``HIGH_PRIORITY_SOURCES`` / ``IDLE_ONLY_SOURCES``.
                Defaults to USER downstream.  Typed as ``Any`` here to avoid
                a top-level import of the SDK enum.

        Returns:
            One of the status constants described above.
        """
        from shared.message_delivery import (
            ACCEPTED, BUSY, NO_SESSION, NOT_CONFIRMED, QUEUED, TERMINATED,
            UNREACHABLE,
        )

        with self._lock:
            session = self._sessions.get(target_session_id)
        if session is None:
            return NO_SESSION
        server = session.server
        if server is None:
            # Each structural site logs its OWN mechanism.  The caller gets
            # one word (all four are equally retry-safe and equally futile
            # until repaired); the operator gets which of the four, because
            # "no server attached" and "runner too old" want completely
            # different fixes and the status deliberately cannot say that.
            logger.warning(
                "DELIVERY_UNREACHABLE session=%s cause=no_server -- the "
                "session is loaded but has no JaatoServer attached, so there "
                "is no delivery path at all.  Nothing was enqueued.",
                target_session_id,
            )
            return UNREACHABLE

        # Terminal targets are REPORTED, not delivered to.  Without this the
        # only way a driver learned its target was dead was that nothing ever
        # happened -- which is exactly what a busy target looks like.
        if getattr(server, "_terminal_reason", None):
            return TERMINATED

        # ASK THE SESSION, DO NOT READ THE REPLICA.
        #
        # ``server._model_running`` is a daemon-side replica of the session's
        # ``_is_running`` and clears strictly LATER -- only once
        # ``session.send_message`` returns and the model thread unwinds,
        # which is after the session has finished its turn AND run its final
        # drain.  Deciding here would therefore decide on state that can
        # already be stale, and a message queued into a turn that has ended
        # is drained by nothing.  Measured live at ~30s of staleness.
        #
        # ``session.offer_message`` makes the check-and-enqueue atomic
        # against the turn's own ``_is_running`` flip, so ``queued`` is a
        # guarantee rather than a prediction.
        rpc = getattr(server, "_runner_rpc", None)
        if rpc is None:
            logger.warning(
                "DELIVERY_UNREACHABLE session=%s cause=no_runner_channel -- "
                "the session has a server but no runner RPC client, so the "
                "offer could not be made.  Nothing was enqueued.",
                target_session_id,
            )
            return UNREACHABLE
        offer = getattr(rpc, "session_offer_message_threadsafe", None)
        if not callable(offer):
            # A VERSION statement, not a fault: the runner predates the
            # atomic offer verb (#620).  Worth its own token because the fix
            # is "restart the runner on current code", which no other
            # unreachable cause shares.
            logger.warning(
                "DELIVERY_UNREACHABLE session=%s cause=offer_verb_absent -- "
                "the runner does not expose session_offer_message (it "
                "predates the atomic offer verb).  Nothing was enqueued.",
                target_session_id,
            )
            return UNREACHABLE
        try:
            outcome = offer(
                text,
                source_id=source_id,
                source_type=(
                    source_type.value if source_type is not None else None
                ),
                require_idle=require_idle,
                # An offer that TIMES OUT may still have been enqueued
                # runner-side.  That residual is real and unfixable from
                # here -- what changed is that it now has its OWN status
                # (NOT_CONFIRMED) instead of sharing one with four cases
                # that never sent anything.
                timeout=2.0,
            )
        except Exception as exc:  # noqa: BLE001
            # WARNING, not debug: the caller is being told the delivery
            # FAILED, so the reason has to be somewhere.  At debug it was
            # generated and discarded -- the same shape as a retry notice
            # nobody can read.
            #
            # ``exc_message`` because the most likely exception here is the
            # 2.0s timeout, and ``str(TimeoutError())`` is the EMPTY STRING:
            # the line rendered as "offer_message failed: " with nothing
            # after it, which is the absent-vs-empty trap this helper exists
            # to close.
            logger.warning(
                "DELIVERY_NOT_CONFIRMED session=%s cause=offer_failed "
                "(%s: %s) -- the offer WAS made and its answer was lost, so "
                "the message may be enqueued runner-side or may never have "
                "arrived.  Re-sending may deliver it twice.",
                target_session_id, type(exc).__name__, exc_message(exc),
            )
            return NOT_CONFIRMED

        if outcome == "queued":
            return QUEUED
        if outcome == "busy":
            # Only reachable with require_idle: the caller applies
            # backpressure and the TARGET confirmed it is still working.
            return BUSY

        # ``needs_turn``: the session has no turn running, so nothing would
        # ever drain this.  It was deliberately NOT enqueued -- drive instead.
        if self.send_message_to_session(target_session_id, text):
            return ACCEPTED
        # The target told us it was idle and the drive still did not start a
        # turn.  Nothing was enqueued on either path -- the offer declined to
        # queue (that is what ``needs_turn`` MEANS) and the drive failed --
        # so this is retry-safe, not not-confirmed.  ``send_message_to_session``
        # logs which of its own three ways it failed.
        logger.warning(
            "DELIVERY_UNREACHABLE session=%s cause=drive_failed -- the target "
            "answered needs_turn (idle) but dispatching a turn failed.  "
            "Nothing was enqueued on either path.",
            target_session_id,
        )
        return UNREACHABLE

    def inject_prompt_to_session(
        self,
        target_session_id: str,
        text: str,
        source_id: Optional[str] = None,
        source_type: Optional[Any] = None,
    ) -> bool:
        """Deliver a prompt to a loaded session by ID.

        Boolean adapter over :meth:`deliver_prompt_to_session`, kept for the
        daemon extensions (reactor rules, webhook handlers, peer-clustering
        message routers) that already call it.  ``True`` iff the status is
        one that means the message will be acted on
        (``message_delivery.DELIVERED`` -- ``accepted`` or ``queued``).

        Prefer :meth:`deliver_prompt_to_session` in new code: a boolean
        cannot distinguish a target that is dead from one that is merely
        busy, and a caller that cannot tell those apart has no way to
        recover from the first.

        Args:
            target_session_id: The destination session.
            text: The prompt to deliver.
            source_id: Identifier of the sender.
            source_type: ``SourceType`` enum value controlling priority.

        Returns:
            ``True`` if the prompt will be acted on, ``False`` otherwise.
        """
        from shared.message_delivery import DELIVERED
        return self.deliver_prompt_to_session(
            target_session_id, text,
            source_id=source_id, source_type=source_type,
        ) in DELIVERED

    def send_message_to_session(
        self,
        target_session_id: str,
        text: str,
    ) -> bool:
        """DRIVE a turn on an already-loaded session in place, keeping its id.

        The turn-DRIVING counterpart to :meth:`inject_prompt_to_session`:
        ``inject_prompt_to_session`` QUEUES a prompt into the runner's inject
        buffer (consumed only when some OTHER driver runs the next turn), so an
        idle headless session with no client / no poll never actually runs.
        This DISPATCHES a ``SendMessageRequest`` through the same daemon path a
        client send takes — it is the exact call
        :meth:`create_headless_session` uses for its ``initial_prompt`` (only
        that path forks a NEW session first; this targets the EXISTING id) — so
        the session RUNS a turn.

        Use for the T1 reactor-driven resume (idle-but-LOADED): keeps the SAME
        session id, NO fork-from-history.  For an unloaded/ended session (T2)
        reload it first (attach / ``_load_session``) or resume via
        ``create_headless_session(initial_history=..., initial_prompt=...)`` —
        a forked continuation with a new id.

        Thread-safe.

        Returns ``True`` if a turn was dispatched.  ``False`` has TWO causes,
        which the boolean cannot distinguish and the log therefore must: the
        target is not loaded in ``self._sessions``, or the dispatch raised.
        The docstring used to name only the first, so the second read as the
        first at every call site.

        Callers get a bool because that is all they can act on -- a drive
        either happened or did not.  ``deliver_prompt_to_session`` maps a
        ``False`` here onto ``UNREACHABLE`` (retry-safe: nothing was enqueued
        on either path).
        """
        with self._lock:
            session = self._sessions.get(target_session_id)
        if session is None:
            logger.warning(
                "DRIVE_FAILED session=%s cause=not_loaded -- no session with "
                "that id is in self._sessions; no turn was dispatched.",
                target_session_id,
            )
            return False
        from jaato_sdk.events import SendMessageRequest
        try:
            self.handle_request(
                self._HEADLESS_CLIENT_ID,
                target_session_id,
                SendMessageRequest(text=text),
            )
        except Exception as exc:  # noqa: BLE001 — a reactor resume must not crash the caller
            # WARNING, not debug, and for the reason #626 gave one layer up:
            # the caller is being told the drive FAILED, so the reason has to
            # be somewhere.  At debug it was generated and discarded.
            #
            # ``exc_message`` and the type name because ``str(TimeoutError())``
            # is the EMPTY STRING -- the old line could render as
            # "dispatch failed: " with nothing after it, which is precisely
            # the absent-vs-empty trap this codebase keeps re-finding.
            logger.warning(
                "DRIVE_FAILED session=%s cause=dispatch_raised (%s: %s) -- "
                "no turn was dispatched.",
                target_session_id, type(exc).__name__, exc_message(exc),
            )
            return False
        return True

    def set_session_state_for_session(
        self, target_session_id: str, key: str, value: Any,
    ) -> bool:
        """Write session-attached state to a loaded session by ID.

        The session-state sibling of :meth:`inject_prompt_to_session` — a
        routing primitive for daemon extensions (reactor rules, webhook
        handlers) that must write ``key → value`` into a session OTHER than the
        one whose event triggered them.  Generalises
        ``JaatoSession.set_session_state`` from "self-targeting" to
        "addressable by ID".  Used by the reliability T3 resume: the
        ``gate.released`` handler (a global bus event) writes the approved-tools
        set into the **parked** session, which is not the originating session.

        Thread-safe.  Returns ``True`` if delivered, ``False`` if the target
        isn't loaded or has no active runner.  ``value`` must be
        JSON-serialisable (validated runner-side).
        """
        with self._lock:
            session = self._sessions.get(target_session_id)
        if session is None:
            return False
        rpc = getattr(session.server, "_runner_rpc", None)
        if rpc is None:
            return False
        forwarder = getattr(rpc, "session_set_state_threadsafe", None)
        if not callable(forwarder):
            return False
        try:
            forwarder(key, value, timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("set_session_state forward failed: %s", exc)
            return False
        return True

    def get_session_workspace(self, session_id: str) -> Optional[str]:
        """Get the workspace path of a session.

        Args:
            session_id: The session ID.

        Returns:
            The session's workspace path, or None if session not found.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.workspace_path
        return None

    def check_workspace_mismatch(
        self,
        session_id: str,
        client_workspace: Optional[str],
    ) -> Optional[tuple]:
        """Check if there's a workspace mismatch between client and session.

        Args:
            session_id: The session to check.
            client_workspace: The client's workspace path.

        Returns:
            Tuple of (session_workspace, client_workspace) if there's a mismatch,
            None if no mismatch or session not found.
        """
        session_workspace: Optional[str] = None

        with self._lock:
            # First check in-memory sessions
            session = self._sessions.get(session_id)
            if session:
                session_workspace = session.workspace_path
            else:
                # Check persisted sessions on disk — try client's workspace first,
                # since that's the most likely location for the session file
                persisted = self._get_persisted_sessions(workspace_path=client_workspace)
                for s in persisted:
                    if s.session_id == session_id:
                        session_workspace = s.workspace_path
                        break

        if not session_workspace or not client_workspace:
            # No mismatch if either is not set
            return None

        # Use helper method to compare workspaces
        if not self._workspaces_match(session_workspace, client_workspace):
            return (session_workspace, client_workspace)

        return None

    def _count_pending_held_tool_calls(self, session: Session) -> int:
        """Phase 3 §3.12 + peer-review M5/N1: return the count of
        tool calls held by the runner-side permission plugin's
        defer-and-flush queue for *session*.

        The queue itself lives runner-side and is populated when a
        ``check_permission`` ASK lands while
        ``Session.restored_pending_attach`` is True (no client
        attached).  This commit lays the daemon-side foundation —
        the actual queue + drain logic ships in a §3.12 follow-on.
        For now the count is always 0; the helper exists so the
        ``SessionRestoredEvent`` field is wired end-to-end and the
        client side has a stable contract to integrate against.
        """
        # TODO §3.12 follow-on: query the runner-side permission
        # plugin via runner-RPC for the pending-call count.  Until
        # then, return 0.
        return 0

    def attach_session(
        self,
        client_id: str,
        session_id: str,
        workspace_path: Optional[str] = None,
    ) -> bool:
        """Attach a client to an existing session.

        If the session is not in memory, attempts to load from disk.

        Args:
            client_id: The requesting client.
            session_id: The session to attach to.
            workspace_path: Client's working directory for file operations.

        Returns:
            True if attached successfully.
        """
        # Track if session was already in memory (client missed init events)
        session_was_in_memory = False

        # Attach-vs-unload race guard (off-lock await): if an async unload has
        # committed for this session, its runner is mid-disposal and the session
        # is about to be evicted. Await the unload (it needs the lock to finish,
        # so we must wait OUTSIDE the lock), then fall through — the session
        # will be gone and the disk-restore path below loads fresh + re-spawns.
        with self._lock:
            pending_unload = self._unloading.get(session_id)
        if pending_unload is not None:
            logger.info(
                "attach_session: session %s is mid-unload — awaiting teardown "
                "before re-attach", session_id,
            )
            pending_unload.wait(timeout=30.0)

        with self._lock:
            # Atomic with the client-add below: if the unload marker is STILL
            # present (await timed out, or an unload committed in the gap since
            # the await), do NOT attach to a session whose runner is being
            # disposed — bail cleanly so the client retries (the retry takes the
            # now-clear disk-restore path).
            if session_id in self._unloading:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"Session {session_id} is being unloaded; please retry",
                    error_type="SessionError",
                ))
                return False
            # Check if session is in memory
            session = self._sessions.get(session_id)
            session_was_in_memory = session is not None

            if not session:
                # Try to load from disk (pass client_id for init progress events)
                logger.debug(f"attach_session: session {session_id} not in memory, loading from disk...")
                try:
                    session = self._load_persisted_with_index_fallback(
                        session_id, client_id, workspace_path)
                    logger.debug(f"attach_session: load returned {session is not None}")
                except Exception as e:
                    logger.error(f"attach_session: _load_session raised: {type(e).__name__}: {e}")
                    import traceback
                    logger.error(f"attach_session: traceback:\n{traceback.format_exc()}")
                    session = None
                if session:
                    self._sessions[session_id] = session

            if not session:
                self._emit_to_client(client_id, ErrorEvent(
                    error=f"Session not found: {session_id}",
                    error_type="SessionError",
                ))
                return False

            # Detach from current session if any
            current = self._client_to_session.get(client_id)
            if current and current in self._sessions:
                old_session = self._sessions[current]
                old_session.attached_clients.discard(client_id)
                # Consider unloading if no clients
                self._maybe_unload_session(current)

            # Attach to new session
            #
            # Phase 3 §3.12 + peer-review M5/N1: detect first-attach
            # to a disk-restored session BEFORE adding the client to
            # ``attached_clients`` so the "no client previously
            # attached" precondition is unambiguous.  The actual
            # SessionRestoredEvent emission happens after the lock
            # is released (event sinks should not run under the
            # SessionManager lock).
            was_first_attach_to_restored = (
                session.restored_pending_attach
                and not session.attached_clients
            )
            session.attached_clients.add(client_id)
            self._client_to_session[client_id] = session_id

            if was_first_attach_to_restored:
                # Clear the flag now under the lock so a parallel
                # ``check_permission`` ASK observes the new state
                # (defer-and-flush off, normal denial / prompt
                # behaviour resumes).  The pending-tool-call queue
                # drain happens in the runner-side permission plugin
                # in a §3.12 follow-on commit; this commit lays the
                # foundation by surfacing the count of held calls
                # via the SessionRestoredEvent emitted below.
                session.restored_pending_attach = False

            # Only set workspace if session doesn't have one yet.
            # If session already has a workspace, it keeps it - clients are warned
            # about workspace mismatches before attach via check_workspace_mismatch().
            if workspace_path and not session.workspace_path:
                session.workspace_path = workspace_path
                session.server.workspace_path = workspace_path

        logger.info(f"Client {client_id} attached to session {session_id}")

        # M5/N1: emit SessionRestoredEvent for the first attach to a
        # disk-restored session.  The pending-tool-call count is 0
        # for this commit (the queue infrastructure lives in the
        # runner-side permission plugin and lands in a §3.12 follow-
        # on); the event still fires so the client can distinguish a
        # fresh-attach from a restored-attach for telemetry / UX.
        if was_first_attach_to_restored:
            self._emit_to_client(client_id, SessionRestoredEvent(
                session_id=session_id,
                pending_tool_call_count=self._count_pending_held_tool_calls(
                    session,
                ),
            ))

        # Apply client-specific config (e.g., presentation context)
        self._apply_client_config_to_server(client_id, session.server)

        # Only emit current state if session was already in memory.
        # If we just loaded it from disk, the client received all events during init.
        if session_was_in_memory:
            session.server.emit_current_state(
                lambda e: self._emit_to_client(client_id, e),
                skip_session_info=True
            )
        else:
            # Session was loaded from disk - clear any stale pending requests
            # the client might have from before the session was saved/restored
            session.server.emit_current_state(
                lambda e: self._emit_to_client(client_id, e),
                skip_session_info=True,
                clear_stale_pending_requests=True
            )

        # Send complete SessionInfoEvent with state snapshot
        self._emit_to_client(client_id, self._build_session_info_event(session))

        # Send workspace files snapshot so client can rebuild its mirror
        self._send_workspace_snapshot(session_id, client_id)

        # Build attach message with description if available
        desc_part = f" - {session.description}" if session.description else ""
        self._emit_to_client(client_id, SystemMessageEvent(
            message=f"Attached to session: {session_id}{desc_part}",
            style="info",
        ))

        # DEFERRED-TURN drain (Option 2) is NOT done here: attach_session
        # completes BEFORE the re-attaching client's buffered host tools are
        # flushed to the runner (that flush happens transport-side, post-attach).
        # Driving here would build the turn's tool schema before the client's
        # host tools are wired, so the woken turn couldn't call them.  The
        # transport layer drives the pending wake AFTER wiring the client's
        # tools (or immediately when the client has none) — see
        # ipc.py / websocket.py client-tool flush.
        return True

    def resume_session(
        self,
        session_id: str,
        workspace_path: Optional[str] = None,
    ) -> Optional[str]:
        """Reload a persisted session into a LIVE, SAME-id session and restore
        its headless presentation, so a reactor can drive it in place via
        :meth:`send_message_to_session` — the PUBLIC same-id RESUME counterpart
        to the private :meth:`_load_session`.

        Use for the reliability T2 resume (parked session unloaded to free the
        runner, then revived on a late human approval).  Unlike fork-from-
        persisted (:meth:`get_persisted_history` + ``create_headless_session``,
        which mints a NEW id and a fresh, lossy headless reconstruction), this
        reloads the record UNDER THE SAME ID with full fidelity — history,
        session-attached state, profile, and the permission WHITELIST — then
        re-applies the headless/API presentation context.

        The presentation re-apply is the crux: the presentation is CLIENT
        config (set at connect via :meth:`_apply_client_config_to_server`'s
        ``_HEADLESS_CLIENT_ID`` branch); it is NOT persisted in the record and
        NOT restored by :meth:`_load_session`.  Without it the reloaded
        session's ``presentation_context`` is ``None`` → the permission base
        flow takes the INTERACTIVE path → ``channel.request_permission`` blocks
        waiting for a client response a reactor-driven session never gets.
        Re-applying the API presentation makes permission behave as the
        headless session it is, so the restored profile whitelist grants the
        retried call directly (no per-call prompt).

        Pairs with :meth:`send_message_to_session` for the unified resume→drive
        shape mirroring the validated T1 path::

            resume_session(sid, ws)            # reload + restore presentation
            send_message_to_session(sid, ...)  # drive the continuation turn

        Thread-safe (``_load_session`` runs in a fresh session context).

        Args:
            session_id: The persisted session to resume.
            workspace_path: Workspace whose ``.jaato/sessions/`` holds the
                record (same contract as :meth:`_load_session` /
                :meth:`get_persisted_history`).

        Returns:
            ``session_id`` on success (the session is now loaded + presentation-
            restored), or ``None`` if no record exists / the load failed.
        """
        session = self._load_session(
            session_id,
            client_id=self._HEADLESS_CLIENT_ID,
            workspace_path=workspace_path,
        )
        if session is None:
            logger.debug(
                "resume_session: %s not found on disk / load failed", session_id)
            return None
        # _load_session restores history / session-state / profile / whitelist
        # but NOT the presentation context (it's client-config, set at connect,
        # not persisted).  Re-apply the headless/API presentation so the
        # permission layer behaves headless — otherwise the interactive path
        # blocks on this no-client session.  See the docstring.
        self._apply_client_config_to_server(
            self._HEADLESS_CLIENT_ID, session.server)
        return session_id

    def wake_session(
        self,
        session_id: str,
        text: str,
        source: str = "user",
        event_id: Optional[str] = None,
        wake_ref: Optional[str] = None,
        cascade_driver_id: Optional[str] = None,
    ) -> Tuple["WakeOutcome", str]:
        """Start a USER turn on ``session_id``, reviving it if cold/unloaded.

        DEFERRED-TURN (Option 2): if the session is revived COLD with no attached
        client AND a ``cascade_driver_id`` is known (so an observer can be
        notified), the turn is NOT driven immediately — host tools (client-side)
        would have no client to dispatch to.  Instead a ``SessionWokenEvent`` is
        emitted to the cid's cascade observers and the wake is held pending until
        a client re-attaches (:meth:`attach_session` drains it).  Returns
        ``DEFERRED`` in that case.  Without a cid (direct/reactor wake) or with a
        client already attached, the turn drives immediately (``OK``).

        The client-agnostic wake primitive (``session.wake``): any authenticated
        caller — IPC, WS, an HTTP webhook shim, cron, a sibling — can drive a fresh
        turn on a session with NO client attached.  It composes the existing
        headless primitives, :meth:`resume_session` (cold-revive) then
        :meth:`send_message_to_session` (drive), and adds three wake-specific
        concerns:

        - **Workspace stays server-owned.**  A cold session's workspace is
          resolved from the daemon's :class:`SessionWorkspaceIndex`, NEVER from
          the caller, so an authenticated-but-untrusted caller cannot point
          revival at a weaker sandbox root.  The sandbox root itself always
          comes from the persisted record (``state.workspace_path`` in
          ``_load_session``).  A loaded session's workspace is already on its
          ``Session`` object, so the index is consulted only for cold sessions.
        - **Payload is untrusted.**  ``text`` can be attacker-influenced (e.g. a
          public PR-review comment), so it is wrapped via
          :func:`wrap_untrusted_content` — the model sees it as DATA to weigh,
          never as instructions.  The inject / USER-prompt path does NOT pass
          through the tool-result trait auto-wrap (#495 scopes that to
          web_fetch / web_search / MCP), so the wrap is applied explicitly here.
        - **Dedup.**  An ``event_id`` already actioned is dropped — external
          ingresses (GitHub, etc.) redeliver.

        Fire-and-forget: returns once the turn is DISPATCHED (mirroring the
        reactor resume→drive shape); the turn's output flows to whatever client
        later attaches (or persists to history).

        Returns ``(outcome, detail)`` where ``outcome`` is a :class:`WakeOutcome`
        (route on this, not the prose ``detail``) — a redelivered ``event_id``
        yields the benign ``DUPLICATE`` (idempotent no-op), NOT an error.
        """
        from shared.session_id import is_safe_session_id
        if not session_id or not is_safe_session_id(session_id):
            return (WakeOutcome.INVALID, "invalid or missing session_id")
        if not text:
            return (WakeOutcome.INVALID, "empty wake text")

        # Dedup CLAIM (up-front): claim the event_id so a concurrent duplicate
        # dedups immediately.  The claim is RELEASED on any failure below, so a
        # legitimate retry — e.g. an at-least-once sender redelivering after a
        # 5xx transient failure — can re-drive: dedup-on-SUCCESS,
        # retry-on-failure.  A redelivery of an already-SUCCEEDED wake stays
        # claimed → benign DUPLICATE.  (Marking before dispatch would wrongly
        # swallow the retry of a failed wake as a no-op.)
        claimed = False
        if event_id:
            with self._lock:
                if event_id in self._wake_seen_event_ids:
                    return (WakeOutcome.DUPLICATE,
                            f"event_id {event_id!r} already actioned "
                            f"(idempotent no-op)")
                self._wake_seen_event_ids[event_id] = None
                while len(self._wake_seen_event_ids) > self._WAKE_DEDUP_CAP:
                    self._wake_seen_event_ids.popitem(last=False)
                claimed = True

        def _release_claim() -> None:
            """Release the event_id claim so a retry of THIS (failed) wake is
            not deduped away.  No-op when there was no claim."""
            if claimed:
                with self._lock:
                    self._wake_seen_event_ids.pop(event_id, None)

        # Revive if cold.  Workspace is resolved server-side, never from the
        # caller (the security invariant above).
        with self._lock:
            loaded = self._sessions.get(session_id)
        if loaded is None:
            workspace = self._session_workspace_index.resolve(session_id)
            if workspace is None:
                _release_claim()
                return (WakeOutcome.UNRESOLVED,
                        f"cannot resolve workspace for cold session "
                        f"{session_id!r} (unknown or ambiguous in the "
                        f"session-workspace index)")
            if self.resume_session(session_id, workspace_path=workspace) is None:
                _release_claim()
                return (WakeOutcome.REVIVE_FAILED,
                        f"revive failed for session {session_id!r}")

        # DEFERRED-TURN gate: a cold-revived session with NO attached client and
        # a known cid → emit SessionWokenEvent + hold pending; do NOT drive into
        # the void (host tools have no client to dispatch to).  The event_id
        # claim is retained (a deferred wake is a success — a redelivery while
        # pending is a benign DUPLICATE, not a re-defer).
        with self._lock:
            session = self._sessions.get(session_id)
            has_client = bool(session and session.attached_clients)
        if cascade_driver_id and session is not None and not has_client:
            with self._lock:
                # Tag the revived session with its cid (under _lock — event
                # routing + the sweep read cascade_driver_id concurrently) so
                # _emit_to_session reaches the cid's observers and the
                # durability sweep sees it active.
                session.cascade_driver_id = cascade_driver_id
                self._pending_wakes[session_id] = _PendingWake(
                    text=text, source=source, wake_ref=wake_ref or "",
                    cascade_driver_id=cascade_driver_id,
                    expires_at=self._wake_pending_expiry(wake_ref))
            self._emit_session_woken(session_id, wake_ref or "", source)
            logger.info(
                "wake: session %s revived cold, no client — DEFERRED; "
                "SessionWokenEvent emitted to cid=%s observers, turn pends re-attach",
                session_id, cascade_driver_id)
            return (WakeOutcome.DEFERRED,
                    f"session {session_id!r} revived cold with no client; turn "
                    f"deferred until re-attach (SessionWokenEvent emitted)")

        # Warm (client attached) or no observer path: wrap + drive immediately.
        from jaato_sdk.plugins.model_provider.types import wrap_untrusted_content
        wrapped = wrap_untrusted_content(text, source=f"wake:{source}")
        if not self.send_message_to_session(session_id, wrapped):
            _release_claim()
            return (WakeOutcome.NOT_DRIVABLE,
                    f"session {session_id!r} not drivable after wake")
        return (WakeOutcome.OK, "woken")

    def _wake_pending_expiry(self, wake_ref: Optional[str]) -> float:
        """Expiry for a deferred wake — the wake binding's expiry if resolvable,
        else a bounded default so a permanently-detached bot's pending wake
        doesn't linger forever."""
        if wake_ref:
            binding = self._wake_binding_registry.resolve(wake_ref)
            if binding is not None:
                return binding.expires_at
        return time.time() + 24 * 3600.0

    def _emit_session_woken(
        self, session_id: str, wake_ref: str, source: str,
    ) -> None:
        """Emit ``SessionWokenEvent`` to the session's cascade observers (and any
        attached clients) via :meth:`_emit_to_session` — the same tier
        ``SessionTerminatedEvent`` uses.  A connected-but-detached observer
        learns it must re-attach to serve the deferred turn."""
        from jaato_sdk.events import SessionWokenEvent
        try:
            self._emit_to_session(session_id, SessionWokenEvent(
                session_id=session_id, wake_ref=wake_ref, source=source))
        except Exception:  # noqa: BLE001 — emission must not break the wake path
            logger.exception("failed to emit SessionWokenEvent for %s", session_id)

    def drive_pending_wake(self, session_id: str) -> bool:
        """Drive a wake that was DEFERRED for ``session_id``, if one is pending
        and not expired.  Called by :meth:`attach_session` after a client
        attaches (a client is now present to serve host tools).  Returns True if
        a pending wake was driven."""
        with self._lock:
            pending = self._pending_wakes.pop(session_id, None)
        if pending is None:
            return False
        if pending.expires_at <= time.time():
            logger.info("drive_pending_wake: dropping expired pending wake for %s",
                        session_id)
            return False
        from jaato_sdk.plugins.model_provider.types import wrap_untrusted_content
        wrapped = wrap_untrusted_content(pending.text, source=f"wake:{pending.source}")
        driven = self.send_message_to_session(session_id, wrapped)
        if driven:
            logger.info(
                "wake: drove DEFERRED turn for session %s on re-attach "
                "(wake_ref=%s) — host tools now available", session_id,
                pending.wake_ref)
        else:
            logger.warning("drive_pending_wake: %s not drivable on re-attach",
                           session_id)
        return driven

    def bind_wake(
        self,
        wake_ref: str,
        session_id: str,
        workspace_path: str,
        trust_keys: List[str],
        ttl_seconds: Optional[int] = None,
        cascade_driver_id: Optional[str] = None,
    ) -> "BindOutcome":
        """Owner-guarded bind of ``wake_ref`` → ``session_id`` with ``trust_keys``.

        The command handler passes the CALLER'S current session as
        ``session_id`` + its workspace (so a caller can only bind ITSELF —
        hijack-proof) + its ``cascade_driver_id`` (so a deferred wake can reach
        the session's cascade observers and the observer survives the session
        going cold — see :meth:`wake_session` / the sweep exemption).  Delegates
        to the :class:`WakeBindingRegistry`.  See ``wake_binding_registry.py``.
        """
        return self._wake_binding_registry.bind(
            wake_ref, session_id, workspace_path, trust_keys, ttl_seconds,
            cascade_driver_id=cascade_driver_id)

    def unbind_wake(self, wake_ref: str, session_id: str) -> "BindOutcome":
        """Owner-guarded removal of ``wake_ref`` (the caller's session)."""
        return self._wake_binding_registry.unbind(wake_ref, session_id)

    def set_wake_public_url(self, url: Optional[str]) -> None:
        """Wire the operator-declared public wake endpoint (from wake.json
        ``public_url``) so ``bind_wake`` can advertise it in its result.
        Whitespace is stripped so a blank/whitespace-only value reads as unset
        (``""``) rather than a marker that looks set but won't route."""
        self._wake_public_url = (url or "").strip() if isinstance(url, str) else ""

    @property
    def wake_public_url(self) -> str:
        """The operator-declared public wake endpoint, or ``""`` if unset."""
        return self._wake_public_url

    def _owner_session_record_exists(
        self, session_id: str, workspace_path: str,
    ) -> bool:
        """Whether a session record for ``session_id`` still exists — live in
        memory OR persisted on disk under ``workspace_path``.

        The wake-binding owner-guard's liveness oracle (see
        :class:`WakeBindingRegistry`).  A DELETED owner (record gone) frees its
        ``wake_ref`` for re-binding; a merely-UNLOADED (cold, revivable) owner
        keeps its record on disk and so keeps the ref protected — the
        distinction that preserves #520 cold-revive.  On any error determining
        the path it fails SAFE (owner assumed to exist → guard stays), so
        uncertainty never opens a hijack.
        """
        if session_id in self._sessions:
            return True
        try:
            storage_dir = self._session_storage_dir(workspace_path)
        except (ValueError, TypeError):
            return True
        return (storage_dir / f"{session_id}.json").exists()

    def resolve_wake_binding(self, wake_ref: str):
        """Resolve a live (non-expired) binding for the mode-B verify shim, or
        ``None``.  The shim then verifies the wake signature against
        ``binding.trust_keys`` and drives ``binding.session_id`` via
        :meth:`wake_session`."""
        return self._wake_binding_registry.resolve(wake_ref)

    def _load_session(
        self,
        session_id: str,
        client_id: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> Optional[Session]:
        """Load a session from disk (server 0.6.71+ entry).

        Wraps :meth:`_load_session_impl` in a fresh ContextVar context
        via :func:`shared.session_context.run_in_fresh_session_context`
        so the bootstrap is isolated from any ContextVar values
        inherited from the caller's task.  See the helper's docstring
        for the rationale.
        """
        from shared.session_context import run_in_fresh_session_context
        return run_in_fresh_session_context(
            self._load_session_impl, session_id, client_id, workspace_path,
        )

    def _load_persisted_with_index_fallback(
        self,
        session_id: str,
        client_id: Optional[str],
        workspace_path: Optional[str],
    ) -> Optional["Session"]:
        """Load a persisted session by id: client workspace first, then the
        server-side session-workspace index.

        The disk-restore path locates a record at ``<workspace>/.jaato/
        sessions/<id>.json``.  For a workspace-PINNED client (an IPC client
        whose ``working_dir`` *is* the session's workspace) the first
        attempt lands.  For a workspace-PINLESS client (a browser over WS,
        whose session lives in a server-provisioned ``ws_<hash>`` dir it
        cannot present) the first attempt misses, so we fall back to
        :class:`SessionWorkspaceIndex` — the authoritative
        ``session_id → workspace`` map, the SAME server-side resolution the
        wake path uses (``_wake_impl`` → ``resume_session``).  This is what
        lets ``jaato.session(mode="ws", recovery=True).attach_session(id)``
        cold-resume a persisted session with zero client-side workspace
        knowledge — IPC parity.

        The index ``resolve`` returns ``None`` for an unknown OR ambiguous
        id (a cross-workspace id collision), so the fallback never guesses;
        and it is skipped when the resolved workspace equals the one the
        client attempt already used (no redundant reload).

        Returns the loaded :class:`Session`, or ``None`` if no record is
        found by either route.
        """
        session = self._load_session(
            session_id, client_id=client_id, workspace_path=workspace_path)
        if session is not None:
            return session
        resolved_ws = self._session_workspace_index.resolve(session_id)
        if resolved_ws and resolved_ws != workspace_path:
            logger.info(
                "attach_session: %s not under client workspace; resolving via "
                "session-workspace index → %s", session_id, resolved_ws)
            session = self._load_session(
                session_id, client_id=client_id, workspace_path=resolved_ws)
        return session

    @staticmethod
    def _resolve_restore_config_root(
        saved_config_root: Optional[str],
        client_config_root: Optional[str],
        workspace_path: Optional[str],
    ) -> Optional[str]:
        """Resolve ``config_root`` for a disk-restore, with deterministic
        fallbacks so a pre-persistence session can't hang the re-spawned runner.

        Pre-``config_root``-persistence sessions deserialize with
        ``state.config_root=None`` (born under an earlier daemon).  Restoring
        with ``None`` spawns a runner that never calls ``set_config_root``
        (runner/session.py:353) → ``file_edit`` can't resolve its backup base
        dir, ``FilesystemQuery`` inits ``workspace=none``, and auth verification
        hangs forever → the user's message reaches no working runner (silent).

        Resolution order:
          1. the SAVED ``config_root`` (correct for post-persistence sessions),
          2. the ATTACHING client's ``config_root`` (the client sends it in
             ``ClientConfigRequest`` on every (re)attach — an authoritative,
             non-guessed value),
          3. the framework default ``<workspace_path>/.jaato`` (workspace_path
             is reliably persisted, so every session resolves to a working
             config_root and the runner never hangs).

        Returns ``None`` only in the degenerate case where ``workspace_path`` is
        also unset.
        """
        if saved_config_root:
            return saved_config_root
        if client_config_root:
            return client_config_root
        if workspace_path:
            return str(pathlib.Path(workspace_path) / ".jaato")
        return None

    def _attach_budget_ceiling(
        self, budget_control, profile, session_id,
    ) -> bool:
        """Attach a caller-supplied budget CEILING to a session's profile.

        Returns True when a ceiling was attached.  Two callers, one shape:

        RELOAD  ``_load_session_impl`` passes ``state.budget_control`` -- the
                effective ceiling persisted when the session unloaded.
        CREATE  ``_create_session_impl`` passes the ``budget_control`` kwarg --
                a caller (notably a FORK) declaring the ceiling this session
                runs under, when its profile cannot express it.

        Both must land BEFORE ``build_session_envelope`` reads the profile:
        that wire field is the only route a budget takes to the runner, so a
        ceiling applied after the spawn would leave a window in which the
        session exists unbudgeted.  Applying it here means there is no window
        at all rather than one that is merely hard to hit today.

        Extracted so it can be tested by CALLING it -- inline in
        ``_load_session_impl`` it was only reachable through several hundred
        lines of server construction, and a test that cannot call the code ends
        up asserting on its source text instead, which survives deleting the
        very line it means to protect.

        A budget reaches the runner ONLY as ``profile.budget_control``
        (runner_spawn: ``profile = server._profile`` -> the envelope's wire
        field -> ``configure(budget_control=...)`` -> the BudgetTracker).  So
        a budget declared OUTSIDE the profile has no vehicle across a reload:
        ``cascade_budget_set`` puts limits on the cascade pool, the pool is
        not re-established on restore, and the profile is deliberately
        budget-free when limits are a per-run operator choice rather than a
        property of the agent.  The revived session came back with NO tracker,
        so its cross-turn ceilings could not fire -- confirmed live
        2026-08-23, where a ``turns: 2`` ceiling let a goal run three turns
        and exit 0.

        A profile that declares its own budget WINS: that is authored policy,
        re-read from disk on every restore, and must not be shadowed by a
        stale snapshot.  The persisted config fills the gap only where there
        was no vehicle at all.
        """
        persisted = budget_control
        if not persisted or profile is None:
            return False
        if getattr(profile, "budget_control", None) is not None:
            return False  # authored policy wins over the snapshot
        from shared.budget_control import BudgetControlConfig
        try:
            profile.budget_control = BudgetControlConfig.from_dict(
                persisted)
        except (ValueError, TypeError) as exc:
            # Loud: a ceiling that fails to rebuild is a ceiling that silently
            # stops applying.
            logger.warning(
                "_load_session: persisted budget ceiling for session %s failed "
                "to rebuild (%s) -- this session will run UNBUDGETED and its "
                "cross-turn ceilings will not fire", session_id, exc)
            return False
        logger.info(
            "_load_session: re-attached persisted budget ceiling to session %s "
            "(limits=%s) -- the profile declares none, so without this the "
            "revived session runs unbudgeted",
            session_id, persisted.get("limits"))
        return True

    def _restore_budget_usage(
        self, server, usage, reason, session_id: str,
    ) -> bool:
        """Re-seed a reloaded session's budget usage.  Returns True if applied.

        Extracted so it can be tested by CALLING it.  The first version of
        this lived inline in ``_load_session_impl`` and its test grepped the
        method source for the RPC name -- which survived deleting the call
        line, because the ``getattr`` lookup above it still mentioned the
        name.  Checking that a symbol is mentioned is not checking it is used.

        Must run BEFORE the session takes a turn, so a ceiling crossed before
        the unload is still crossed after it.

        Returns True only when the RUNNER confirms it applied the snapshot.
        The RPC returning normally is NOT that confirmation -- a session that
        came back unbudgeted answers ``restored: False`` and the ceiling is
        gone.  Honouring that bool is the difference between an instrument
        and a decoration.
        """
        if not usage and not reason:
            return False
        # A snapshot on disk IS the evidence this session was budgeted.  So
        # the reloaded session MUST have come back with a budget to restore
        # into -- and the only way the budget reaches the runner is the
        # profile (runner_spawn.py: ``profile = server._profile`` ->
        # ``profile.budget_control`` -> the envelope's wire field).  A
        # profile that failed to rebuild, or rebuilt without its budget,
        # yields a session with NO BudgetTracker: every cross-turn ceiling
        # is gone, restore is a no-op, and nothing downstream can tell.
        #
        # Checked HERE, before the RPC, because the cause is visible here
        # and the symptom is not: the restore call returns cleanly either
        # way.  Verified live 2026-08-23 -- a suspend/resume cascade logged
        # "Budget control active" exactly ONCE across a run with a reload
        # (the initial configure), so the revived session ran unbudgeted and
        # a ``turns: 2`` ceiling never fired however many resumes it took.
        _profile = getattr(server, "_profile", None)
        if getattr(_profile, "budget_control", None) is None:
            logger.warning(
                "session %s has a persisted budget snapshot (%s) but its "
                "reloaded profile declares NO budget_control (%s) -- this "
                "session came back UNBUDGETED and its cross-turn ceilings "
                "will not fire.  The budget reaches the runner only via the "
                "profile, so this is a profile-restore failure, not a "
                "budget-restore failure.",
                session_id, usage,
                "no profile was rebuilt" if _profile is None
                else f"profile {getattr(_profile, 'name', '?')!r} rebuilt "
                     "without budget_control",
            )
        rpc = getattr(server, "_runner_rpc", None)
        restorer = getattr(
            rpc, "session_restore_budget_usage_threadsafe", None,
        ) if rpc is not None else None
        if not callable(restorer):
            return False
        try:
            applied = restorer(usage or {}, exhausted_reason=reason,
                               timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            # WARNING, not debug: a budget that silently fails to restore is
            # a ceiling that silently stops applying.
            logger.warning(
                "budget usage restore failed for session %s (%s) -- this "
                "session's cross-turn ceilings restart from zero",
                session_id, exc,
            )
            return False
        if not applied:
            # The RPC succeeded and the runner said it restored NOTHING.
            # That is the reloaded session reporting it has no BudgetTracker
            # (rpc.py ``_handle_session_restore_budget_usage`` -> restored
            # False), i.e. it came back UNBUDGETED -- every cross-turn
            # ceiling is now gone even though a snapshot existed to enforce
            # one.  Logging this as success is what made the whole
            # suspend/resume budget arc invisible: the observable said
            # "Restored" whether or not anything was.
            logger.warning(
                "budget usage restore did NOT apply for session %s "
                "(snapshot %s) -- the reloaded session reports no budget "
                "tracker, so its cross-turn ceilings restart from zero",
                session_id, usage,
            )
            return False
        logger.info(
            "Restored budget usage for session %s: %s%s", session_id, usage,
            f" (still exhausted: {reason})" if reason else "")
        return True

    def _load_session_impl(
        self,
        session_id: str,
        client_id: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> Optional[Session]:
        """Implementation of session loading, called via fresh-context wrap.

        See :meth:`_load_session` for the isolation rationale.

        Args:
            session_id: The session ID to load.
            client_id: Optional client ID to receive init progress events.
            workspace_path: Workspace directory for resolving the session
                storage path. Required so we know which workspace's
                ``.jaato/sessions/`` to look in.

        Returns:
            The loaded Session, or None if not found.
        """
        logger.debug(f"_load_session: attempting to load {session_id}")

        # Resolve storage directory from workspace
        storage_dir = self._session_storage_dir(workspace_path) if workspace_path else None

        try:
            state = self._session_plugin.load(session_id, storage_dir=storage_dir)
            logger.debug(f"_load_session: loaded state for {session_id}")
        except FileNotFoundError:
            logger.debug(f"_load_session: session {session_id} not found on disk")
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

        # Create JaatoServer and restore state
        logger.debug(f"_load_session: creating JaatoServer for {session_id}...")

        # Server 0.6.166+: centralized bootstrap-time routing via
        # :meth:`_route_bootstrap_event`.  Disk-restore doesn't
        # currently thread a cascade_driver_id (sessions being
        # restored from disk aren't part of an active cascade
        # observer subscription), so the cascade dispatch piece is
        # a no-op for this path — but routing through the same
        # helper means future cascade-restoration support drops in
        # without re-introducing the bypass.  When client_id is
        # provided, route to that client; otherwise fall through to
        # _emit_to_session (covers the no-client restore branch).
        if client_id:
            init_callback = lambda e: self._route_bootstrap_event(
                client_id, None, e, session_id,
            )
        else:
            init_callback = lambda e: self._emit_to_session(session_id, e)

        # Determine which env_file to use for this session:
        # 1. If client_id is provided, use client's env_file from their config
        # 2. If session has workspace_path, try workspace/.env
        # Sessions are workspace-bound: the workspace determines the .env file,
        # which in turn determines the provider.
        session_env_file = None
        if client_id:
            client_config = self._client_config.get(client_id, {})
            if client_config.get('env_file'):
                session_env_file = client_config['env_file']
                logger.debug(f"_load_session: using client's env_file: {session_env_file}")
        if not session_env_file and state.workspace_path:
            import os
            workspace_env = os.path.join(state.workspace_path, '.env')
            if os.path.exists(workspace_env):
                session_env_file = workspace_env
                logger.debug(f"_load_session: using workspace env_file: {session_env_file}")

        # Resolve the SubagentProfile from ``state.profile_name``
        # (persisted post-2.3) so disk-restore re-binds the full
        # recipe — model, provider, plugin_configs,
        # system_instructions, GC strategy.  Without this, the
        # restored server initializes from session env alone; in
        # multi-profile workspaces (where MODEL_NAME / JAATO_PROVIDER
        # live in the per-profile YAML, not workspace .env)
        # ``JaatoServer.initialize`` raises ConfigurationError on
        # missing MODEL_NAME (silently — the error event is emitted
        # to the target session's sink which has no attached client).
        # Pre-2.3 sessions deserialize with ``state.profile_name=None``;
        # they fall through to env-only resolution as before, which
        # only succeeds if the workspace .env carries MODEL_NAME +
        # JAATO_PROVIDER — same constraint as fresh-spawn-without-
        # profile.
        # Resolve config_root with disk-restore fallbacks (saved → attaching
        # client → <workspace>/.jaato default) so a pre-persistence session
        # saved with config_root=None can't hang the re-spawned runner at
        # file_edit/auth path resolution.  Used by BOTH the profile resolution
        # below and the BootstrapEnvelope.
        restore_config_root = self._resolve_restore_config_root(
            state.config_root,
            self._client_config.get(client_id, {}).get("config_root"),
            state.workspace_path,
        )

        # The RECIPE.  Issue #787: a revived session comes back with the
        # profile it ran under rather than with whatever the profile files
        # say today.  ``JAATO_REVIVE_PROFILE=disk`` opts back into
        # re-resolving; see ``server/revive_policy.py``.
        restored_profile = self._resolve_revive_profile(
            state,
            session_id=session_id,
            workspace_path=state.workspace_path or workspace_path or "",
            config_root=restore_config_root,
            env_file=session_env_file,
        )

        self._attach_budget_ceiling(
            getattr(state, "budget_control", None), restored_profile,
            session_id)

        # The PROMPT.  Issue #787: a revived session RESTORES the system
        # instruction it was rendered with rather than rebuilding it, so the
        # persona's ``{{!py:...}}`` prefetch scripts do not re-run.  When
        # this returns None the persona is rebuilt from disk exactly as
        # before (pre-2.8 records, and the ``JAATO_REVIVE_PERSONA=disk``
        # opt-in), and the helper has rebound
        # ``restored_profile.system_instructions`` for that path.
        restored_instruction_override = self._resolve_revive_persona(
            state,
            restored_profile,
            session_id=session_id,
            workspace_path=state.workspace_path or workspace_path or "",
            config_root=restore_config_root,
        )

        # Phase 3 §3.12 disk-restore migration: route the JaatoServer
        # construction + pre-init hooks + initialize through the
        # unified ``_construct_and_initialize_server`` sub-helper that
        # the IPC path also uses.  The disk-restore path supplies the
        # saved ``state.sandbox_mode`` via ``envelope.sandbox_mode``
        # so the Session record below reflects the pre-restart value
        # rather than re-running the apparmor opt-in lookup (which
        # is a client-driven concept that doesn't apply to disk
        # restore).
        envelope = BootstrapEnvelope(
            session_id=session_id,
            workspace_path=state.workspace_path,
            name=state.description or f"Session {session_id}",
            description=state.description,
            # Thread the ATTACHING client (was hardcoded None) so a restore-
            # AFTER-UNLOAD re-attach actually spawns the runner — else
            # _provision_ipc_apparmor_and_spawn_runner's step-1 (``client_id is
            # None``) skips the spawn, ``_runner_rpc`` stays None, and the first
            # message dies on the runner-readiness wait (the #370 re-attach
            # flaky-fail, root-caused via PROVISION_ENTER client_id=None).  None
            # on a clientless background restore preserves the old skip.
            client_id=client_id,
            sandbox_mode=getattr(state, "sandbox_mode", None),
            # Drive confinement from the SAVED sandbox_mode (precedence-1
            # apparmor_override in _provision) rather than re-running the
            # client-driven opt-in — preserves the "use saved sandbox_mode,
            # don't re-run the opt-in" intent now that a real client_id is
            # threaded above.  env_file stays a saved-driven override; config_root
            # is resolved saved→client→<workspace>/.jaato (restore_config_root
            # above) so a pre-persistence None can't hang the runner.
            apparmor=(getattr(state, "sandbox_mode", None) == "apparmor"),
            profile=restored_profile,
            # Re-apply the profile's ``suppress_base_instructions`` on restore.
            # Unlike plugins / plugin_configs / system_instructions / gc (which
            # flow through ``server._profile`` → ``build_session_envelope``),
            # this knob is read on the wire from ``server._suppress_base_
            # instructions`` (runner_spawn.py), set from the BootstrapEnvelope
            # field — which the restore envelope never populated, so a restored
            # session silently regained the ~3-5k framework base instructions
            # even when the profile suppressed them.  On tiny-context models
            # (Gemini Nano ~9k) that overflowed the window ("input too large").
            # The create path derives this from an explicit kwarg OR the
            # profile; on restore there is no client kwarg, so the reconstructed
            # profile is the sole source.  Fixes named AND inline profiles.
            # Reconstructed profile is already the canonical frozenset
            # (normalized in SubagentProfile.__post_init__).
            suppress_base_instructions=getattr(
                restored_profile, "suppress_base_instructions", frozenset()),
            config_root=restore_config_root,
            # Rebind the persona (--agent) on revive so persona-only guidance
            # (e.g. enter_tier on images) survives — else JaatoServer(agent_name
            # =None) drops it and multimodal revives confabulate.
            agent_name=getattr(state, "agent_name", None),
            # #787: the frozen prompt, when this revive is using it.  None
            # leaves the runner assembling normally (pre-2.8 records, or
            # ``JAATO_REVIVE_PERSONA=disk``).
            system_instruction_override=restored_instruction_override,
            # #787: the ORIGINAL agent_params.  Needed by the re-render
            # path (a prefetch reads ``context.agent_params``; handing it
            # an empty dict is what made such sessions unwakeable), and
            # carried on the default path too so a save→revive→save cycle
            # keeps them rather than dropping them on the first revive.
            agent_params=dict(getattr(state, "agent_params", None) or {}),
            restore_state={"loaded_state": state},
            env_file=session_env_file,
            instruction_token_cache=self._instruction_token_cache,
            on_event_during_init=init_callback,
        )
        try:
            server, _restore_sandbox = self._construct_and_initialize_server(envelope)
        except Exception as e:
            logger.error(
                f"_load_session: initialize() raised exception: "
                f"{type(e).__name__}: {e}"
            )
            import traceback
            logger.error(f"_load_session: traceback:\n{traceback.format_exc()}")
            return None
        if server is None:
            logger.error(f"Failed to initialize server for session {session_id}")
            return None
        logger.debug(f"_load_session: initialize() returned True")

        logger.debug(f"_load_session: server initialized for {session_id}")

        # Switch to session-based event emission now that init is complete
        server.set_event_callback(lambda e: self._emit_to_session(session_id, e))

        # Configure TODO plugin with session-scoped storage
        effective_workspace = workspace_path or state.workspace_path
        if effective_workspace:
            session_dir = self._session_storage_dir(effective_workspace) / session_id
        else:
            session_dir = pathlib.Path(self._session_config.storage_path) / session_id
        self._configure_todo_storage(server, session_dir)

        # Restore history to the runner-side session.
        # Phase 3 §7c step 6.6.4.5b: route through the
        # ``session.set_initial_history`` RPC (added §7c step 6.6.1.1)
        # instead of ``server._jaato.reset_session(state.history)``.
        # Semantically equivalent at this site: the runner-side session
        # was just bootstrapped with empty history, which matches
        # ``set_initial_history``'s "session must be idle and history
        # must be empty" precondition.
        if state.history and server._runner_rpc is not None:
            server._runner_rpc.session_set_initial_history_threadsafe(state.history)
            logger.debug(f"Restored {len(state.history)} messages for session {session_id}")

            # Resolve the session's main agent id once — it may be the
            # default ``"main"`` or the ``--agent <name>`` value used at
            # session creation.  Used as the ``_agents`` dict key and the
            # emitted ``agent_id`` so consumers see consistent identity.
            main_agent_id = server.main_agent_id

            # Also populate AgentState.history so emit_current_state() can
            # replay conversation content for reconnecting clients.
            # on_agent_history_updated is only called during send_message(),
            # so we must set it explicitly after disk load.
            if main_agent_id in server._agents:
                server._agents[main_agent_id].history = list(state.history)

            # Restore turn accounting (reset_session clears it, so we restore after).
            # Phase 3 §7c step 6.6.3.6: forward to runner-side
            # via the new ``session.restore_turn_accounting``
            # RPC (§7c step 6.6.1.2 at commit 82b8da29) instead
            # of reaching into the daemon-side session.
            if state.turn_accounting:
                rpc = getattr(server, "_runner_rpc", None)
                if rpc is not None:
                    forwarder = getattr(
                        rpc, "session_restore_turn_accounting_threadsafe", None,
                    )
                    if callable(forwarder):
                        try:
                            forwarder(state.turn_accounting, timeout=5.0)
                            logger.debug(f"Restored {len(state.turn_accounting)} turn accounting entries for session {session_id}")
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "restore_turn_accounting forward failed: %s",
                                exc,
                            )

                # Update server's agent state and emit context update.
                # Phase 3 §7c step 6.6.4.5b: route ``get_context_usage``
                # through the runner-RPC instead of the daemon-side
                # JaatoClient indirection.
                if main_agent_id in server._agents:
                    main_state = server._agents[main_agent_id]
                    main_state.turn_accounting = list(state.turn_accounting)
                    usage = server._runner_rpc.session_get_context_usage_threadsafe()
                    main_state.context_usage = {
                        'total_tokens': usage.get('total_tokens', 0),
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'output_tokens': usage.get('output_tokens', 0),
                        'percent_used': usage.get('percent_used', 0.0),
                    }
                    # Emit context update so clients show correct usage
                    server.emit(ContextUpdatedEvent(
                        agent_id=main_agent_id,
                        usage=server._build_usage(
                            prompt_tokens=usage.get('prompt_tokens', 0),
                            output_tokens=usage.get('output_tokens', 0),
                            total_tokens=usage.get('total_tokens', 0),
                        ),
                        context_limit=usage.get('context_limit', 0),
                        percent_used=usage.get('percent_used', 0.0),
                        tokens_remaining=usage.get('tokens_remaining', 0),
                        turns=usage.get('turns', 0),
                    ))
                    logger.debug(f"Emitted ContextUpdatedEvent: {usage.get('percent_used', 0.0):.1f}% used")

        # Restore conversation budget if present (other budget sources are
        # automatically populated during session recreation).
        # Phase 3 §7c step 6.6.1.0: use the public
        # JaatoSession.restore_conversation_budget() method instead
        # of reaching through ``session.instruction_budget`` into
        # the underlying InstructionBudget's
        # ``restore_conversation_from_snapshot``.  The public
        # surface is the prerequisite for the upcoming
        # ``session.restore_conversation_budget`` runner-RPC
        # handler (§7c step 6.6.1.3).  The method is no-op when
        # the session's instruction_budget is None, so we drop
        # the explicit guard.
        # Phase 3 §7c step 6.6.3.6: forward to runner-side via the
        # new ``session.restore_conversation_budget`` RPC (§7c step
        # 6.6.1.3 at commit b40d2439); use the existing
        # ``session.snapshot_instruction_budget`` (§7c step 6.1
        # (2/3) at commit 1043bfde) for the post-restore emit.
        self._restore_budget_usage(
            server, getattr(state, "budget_usage", None),
            getattr(state, "budget_exhausted_reason", None),
            session_id)

        if state.budget_state:
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is not None:
                restorer = getattr(
                    rpc, "session_restore_conversation_budget_threadsafe", None,
                )
                snapshotter = getattr(
                    rpc, "session_snapshot_instruction_budget_threadsafe", None,
                )
                if callable(restorer):
                    try:
                        restorer(state.budget_state, timeout=5.0)
                        logger.debug(
                            f"Restored conversation budget for session {session_id}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "restore_conversation_budget forward failed: %s",
                            exc,
                        )
                if callable(snapshotter):
                    try:
                        snapshot = snapshotter(timeout=5.0)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "snapshot_instruction_budget forward failed: %s",
                            exc,
                        )
                        snapshot = None
                    if snapshot is not None:
                        # Emit budget event so clients show correct budget.
                        # ``agent_id`` is a top-level key in the snapshot
                        # dict (per InstructionBudget.snapshot() schema).
                        server.emit(InstructionBudgetEvent(
                            agent_id=snapshot.get('agent_id', server.main_agent_id),
                            budget_snapshot=snapshot,
                        ))

        # Restore subagent state if present in metadata.
        # Phase 3 §7c step 6.6.4.5e: truthiness check pivoted from
        # ``server._jaato`` (deleted) to ``server._runtime`` (the
        # daemon-side canonical handle since 5d).
        if state.metadata.get('subagents') and server._runtime is not None:
            self._restore_subagent_states(
                session_id,
                state.metadata['subagents'],
                server,
                workspace_path=effective_workspace,
            )

        # Restore TODO plugin state (agent-plan mapping, blocked steps)
        self._load_todo_state(server, session_dir)

        # Generic plugin state restoration: iterate plugin_states saved by
        # the generic persistence loop and call restore_persistence_state()
        # on each plugin that implements it.
        if state.metadata.get('plugin_states') and server.registry:
            for plugin_name, plugin_state in state.metadata['plugin_states'].items():
                plugin = server.registry.get_plugin(plugin_name)
                if plugin and hasattr(plugin, 'restore_persistence_state'):
                    try:
                        plugin.restore_persistence_state(plugin_state)
                        logger.debug(f"Restored persistence state for plugin: {plugin_name}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to restore persistence state for plugin "
                            f"'{plugin_name}': {e}"
                        )

        # Check for and recover from interrupted turn
        recovered_count = 0
        if state.interrupted_turn:
            recovered_count = self._recover_interrupted_turn(
                session_id,
                state.interrupted_turn,
                server
            )
            if recovered_count > 0:
                logger.info(f"Recovered {recovered_count} interrupted tool calls for session {session_id}")

        session = Session(
            session_id=session_id,
            name=state.description or f"Session {session_id}",
            server=server,
            created_at=state.created_at.isoformat(),
            last_activity=state.updated_at.isoformat(),
            description=state.description,
            is_dirty=recovered_count > 0,  # Mark dirty if recovery happened
            workspace_path=state.workspace_path,
            user_inputs=state.user_inputs or [],  # Command history for prompt restoration
            provisioned=state.metadata.get('provisioned', False),
            sandbox_mode=getattr(state, "sandbox_mode", None),
            # Carry the inline spec forward so a re-save of the restored
            # session re-persists it (survives restore → save → restore).
            inline_profile_spec=getattr(state, "profile_spec", None),
            # #787: carry the frozen recipe + frozen prompt forward so a
            # re-save of the restored session re-persists the ORIGINALS.
            # Without this, the write-once capture in ``_save_session``
            # would see empty fields on the restored record and re-snapshot
            # from the live server -- which on a ``=disk`` revive means the
            # re-derived value replacing the artifact it was meant to be
            # compared against.
            profile_snapshot=getattr(state, "profile_snapshot", None),
            rendered_instructions=getattr(state, "rendered_instructions", None),
            agent_params=getattr(state, "agent_params", None),
            # A sibling ADDRESS that does not survive a reload is not an
            # address: sessions unload on ORPHAN, so a sibling that came back
            # nameless would be unreachable by every sibling still holding
            # its name.  Same shape as the budget ceiling that did not
            # survive an unload (#583).
            sibling_name=getattr(state, "sibling_name", None),
            # Restore cascade MEMBERSHIP, not just the address.  Without
            # this a stage that unloaded on ORPHAN came back with None and
            # silently left its cascade: its sibling_name addressed a
            # cascade it was no longer in, _emit_to_session stopped
            # reaching the cid's observers, and the durability sweep
            # stopped seeing it.
            cascade_driver_id=getattr(state, "cascade_driver_id", None),
            # Phase 3 §3.12 + peer-review M5/N1: mark this session as
            # awaiting first client-attach.  While set, the runner-
            # side permission plugin queues ASK prompts rather than
            # denying them (defer-and-flush posture).  Cleared in
            # ``attach_session`` after emitting SessionRestoredEvent.
            restored_pending_attach=True,
        )

        # Restore workspace file monitor with persisted tracked state
        if state.workspace_path:
            self._start_workspace_monitor(session_id, state.workspace_path, server=server)
            monitor = self._workspace_monitors.get(session_id)
            if monitor and state.workspace_files:
                monitor.restore(state.workspace_files)
                # Reconcile: detect changes that happened while server was down
                reconcile_changes = monitor.reconcile()
                if reconcile_changes:
                    session.is_dirty = True
                    logger.info(
                        "Workspace reconciliation found %d changes for session %s",
                        len(reconcile_changes),
                        session_id,
                    )

        # Store session before running hooks so hooks can call get_session().
        with self._lock:
            self._sessions[session_id] = session
        self._run_session_hooks(server, session_id)

        logger.info(f"Loaded session from disk: {session_id}")
        return session

    def _restore_subagent_states(
        self,
        session_id: str,
        subagent_registry: Dict[str, Any],
        server: JaatoServer,
        workspace_path: Optional[str] = None,
    ) -> int:
        """Restore subagent states from persisted data.

        Args:
            session_id: The parent session ID.
            subagent_registry: Registry dict from state.metadata["subagents"].
            server: The JaatoServer to restore subagents into.
            workspace_path: Workspace directory for resolving storage path.

        Returns:
            Number of subagents successfully restored.
        """
        if not server.registry:
            logger.warning("Cannot restore subagents: no registry available")
            return 0

        subagent_plugin = server.registry.get_plugin("subagent")
        if not subagent_plugin or not hasattr(subagent_plugin, 'restore_persistence_state'):
            logger.warning("Cannot restore subagents: subagent plugin not available")
            return 0

        # Load per-agent state files
        if workspace_path:
            subagents_dir = self._session_storage_dir(workspace_path) / session_id / "subagents"
        else:
            subagents_dir = pathlib.Path(
                self._session_config.storage_path
            ) / session_id / "subagents"

        agent_states: Dict[str, Dict[str, Any]] = {}
        if subagents_dir.exists():
            for agent_file in subagents_dir.glob("*.json"):
                agent_id = agent_file.stem
                try:
                    with open(agent_file, 'r', encoding='utf-8') as f:
                        agent_states[agent_id] = json.load(f)
                    logger.debug(f"Loaded subagent state file: {agent_file}")
                except Exception as e:
                    logger.error(f"Failed to load subagent state {agent_file}: {e}")

        # Phase 3 §7c step 6.6.4.5a: read ``server._runtime`` directly
        # instead of going through ``server._jaato.get_runtime()``.
        # ``self._runtime`` has been the daemon-side runtime field since
        # §7c step 4 first pass (commit 7c34f218); this site completes
        # the pattern.  Behavior-preserving: ``_runtime`` is non-None
        # iff ``_jaato`` was successfully connected.
        runtime = server._runtime
        if not runtime:
            logger.warning("Cannot restore subagents: no runtime available")
            return 0

        # Restore subagents
        restored = subagent_plugin.restore_persistence_state(
            subagent_registry,
            agent_states,
            runtime
        )

        # Emit AgentCreatedEvent for each restored subagent so clients see them
        for agent_id, info in subagent_plugin._active_sessions.items():
            profile = info.get('profile')
            created_at = info.get('created_at')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()

            server.emit(AgentCreatedEvent(
                agent_id=agent_id,
                agent_name=profile.name if profile else agent_id,
                agent_type="subagent",
                profile_name=profile.name if profile else "",
                parent_agent_id=server.main_agent_id,
                created_at=created_at,
                session_id=session_id,
            ))

            # Emit context update for restored subagent
            session = info.get('session')
            if session:
                usage = session.get_context_usage()
                context_limit = session.get_context_limit()
                server.emit(ContextUpdatedEvent(
                    agent_id=agent_id,
                    usage=server._build_usage(
                        prompt_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('output_tokens', 0),
                        total_tokens=usage.get('total_tokens', 0),
                    ),
                    context_limit=context_limit,
                    percent_used=usage.get('percent_used', 0.0),
                    tokens_remaining=max(0, context_limit - usage.get('total_tokens', 0)),
                    turns=usage.get('turns', 0),
                ))

        logger.info(f"Restored {restored} subagents for session {session_id}")
        return restored

    def _recover_interrupted_turn(
        self,
        session_id: str,
        interrupted_state: Dict[str, Any],
        server: JaatoServer
    ) -> int:
        """Recover from an interrupted turn by injecting synthetic tool results.

        When a session is loaded with pending tool calls (from an interrupted turn),
        this method injects synthetic error results for each pending call. This
        completes the function_call/response pairs so the model sees what happened
        and can decide whether to retry.

        Args:
            session_id: The session ID being recovered.
            interrupted_state: The interrupted_turn dict from SessionState containing:
                - agent_id: Which agent was executing
                - pending_tool_calls: List of {id, name, args}
                - user_prompt: Original user prompt
                - started_at: When the turn started
            server: The JaatoServer to inject results into.

        Returns:
            Number of pending tool calls recovered.
        """
        from jaato_sdk.plugins.model_provider.types import Part, Message, Role, ToolResult

        pending_calls = interrupted_state.get('pending_tool_calls', [])
        if not pending_calls:
            logger.debug(f"No pending tool calls to recover for session {session_id}")
            return 0

        agent_id = interrupted_state.get('agent_id', 'main')

        # Build synthetic tool results for each pending call
        synthetic_parts = []
        for call in pending_calls:
            call_id = call.get('id', '')
            tool_name = call.get('name', 'unknown')

            synthetic_result = ToolResult(
                call_id=call_id,
                name=tool_name,
                result={
                    "error": "tool_interrupted",
                    "reason": "server_restart",
                    "message": f"Tool '{tool_name}' was interrupted by server restart. "
                               "You may retry this operation if appropriate."
                },
                is_error=True
            )
            synthetic_parts.append(Part.from_function_response(synthetic_result))

        # Create a TOOL message with all synthetic results
        synthetic_message = Message(role=Role.TOOL, parts=synthetic_parts)

        # Inject into history based on which agent was executing.
        # Phase 3 §7c step 6.6.3.6: forward the synthetic message
        # via the new ``session.append_history_message`` RPC
        # (§7c step 6.6.3.1 at commit aa9059ec) instead of the
        # daemon-side get-modify-reset dance.  The runner-side
        # JaatoSession.append_history_message wraps the same
        # get-history + append + reset_session flow internally
        # (preserves the ``_turn_accounting`` clear semantic).
        if agent_id == 'main':
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is not None:
                forwarder = getattr(
                    rpc, "session_append_history_message_threadsafe", None,
                )
                if callable(forwarder):
                    try:
                        forwarder(synthetic_message, timeout=5.0)
                        logger.info(
                            f"Recovered {len(pending_calls)} interrupted tool call(s) "
                            f"for main agent in session {session_id}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "append_history_message forward failed: %s",
                            exc,
                        )
        else:
            # Subagent recovery - find the subagent session
            if server.registry:
                subagent_plugin = server.registry.get_plugin("subagent")
                if subagent_plugin and hasattr(subagent_plugin, '_active_sessions'):
                    session_info = subagent_plugin._active_sessions.get(agent_id)
                    if session_info:
                        subagent_session = session_info.get('session')
                        if subagent_session:
                            # Append the synthetic tool message to history using proper API
                            current_history = subagent_session.get_history()
                            current_history.append(synthetic_message)
                            subagent_session.reset_session(current_history)
                            logger.info(
                                f"Recovered {len(pending_calls)} interrupted tool call(s) "
                                f"for subagent {agent_id} in session {session_id}"
                            )

        # Emit recovery event so clients know what happened
        server.emit(InterruptedTurnRecoveredEvent(
            session_id=session_id,
            agent_id=agent_id,
            recovered_calls=len(pending_calls),
            action_taken="synthetic_error",
        ))

        # Also emit a system message for user visibility
        tool_names = [call.get('name', 'unknown') for call in pending_calls]
        server.emit(SystemMessageEvent(
            message=f"Recovered from interrupted turn: {len(pending_calls)} tool call(s) "
                    f"({', '.join(tool_names)}) were interrupted by server restart.",
            style="warning",
        ))

        # Signal that the interrupted turn is now complete (agent is done)
        # This tells the client to stop showing the "thinking" spinner
        server.emit(AgentStatusChangedEvent(
            agent_id=agent_id,
            status="done",
        ))

        return len(pending_calls)

    def _save_session_async(self, session: Session) -> None:
        """Defer ``_save_session(session)`` to a background daemon thread.

        Path H (cycle 10).  Used by the ToolCallStartEvent branch in
        ``_handle_turn_tracking_event`` (line 1611) to persist
        ``pending_tool_calls`` for crash recovery WITHOUT blocking
        the synchronous ``_emit_to_session`` path.

        Pre-Path-H this site called ``_save_session(session)``
        synchronously, which made 2 blocking runner-RPCs
        (``session_get_history_threadsafe`` +
        ``session_snapshot_conversation_budget_threadsafe``) that
        raced against the runner's still-active ``send_message``.
        After 35s timeout the save failed silently AND the 35s
        delay starved the model loop's permission-response window.
        Architecturally same shape as Path E's Layer 5 race.

        Trade-off: the daemon-crash recovery window for IN-PROGRESS
        tool calls is narrowed (was synchronous fsync; becomes best-
        effort async fsync).  Recovery for COMPLETED turns is
        unaffected — the natural-boundary save on the
        AgentStatusChangedEvent(status=done) path also routes through this
        helper.  (This paragraph used to say that path was still synchronous;
        it stopped being so and the sentence did not follow.)

        Concurrent invocations for the same session serialize via
        ``Session.save_lock``, held inside ``_save_session`` itself, so
        parallel ToolCallStartEvents produce consistent last-writer-wins
        ordering.  Previously that guard was a global lock taken HERE, which
        serialized this one call site while the other eight ran unguarded.

        Args:
            session: The session to save.
        """
        def _do_save() -> None:
            try:
                # No lock taken HERE any more.  ``_save_session`` now holds
                # ``session.save_lock`` itself, so wrapping it again would
                # self-deadlock on a non-reentrant Lock -- and the outer take
                # was the bug: it guarded this ONE call site while eight
                # others ran unguarded.
                self._save_session(session)
            except Exception as exc:  # noqa: BLE001 — best-effort
                logger.warning(
                    "async save for session %s failed: %s",
                    session.session_id, exc,
                )

        thread = threading.Thread(
            target=_do_save,
            name=f"async-save-{session.session_id}",
            daemon=True,
        )
        thread.start()

    def _save_session(self, session: Session) -> bool:
        """Save a session to disk.

        Args:
            session: The session to save.

        Returns:
            True if saved successfully.
        """
        # SERIALIZED PER SESSION.  Nine call sites reach this
        # function and, before this change, exactly ONE of them held a
        # lock -- the wrapper in ``_save_session_async``.  The other
        # eight ran unguarded, so two saves of one session could
        # interleave: both write ``<id>.json.tmp``, the first rename
        # wins, the second raises ENOENT on a file that no longer
        # exists.  Observed live, and each of those saves also issues
        # its own ``session_get_history`` RPC.
        #
        # The guard is HERE, not at the call sites, so a tenth caller
        # inherits it instead of having to know it exists.  That is the
        # defect the old placement had: adding a caller looked like
        # read-only work while silently opting out of the only guard.
        with session.save_lock:
            try:
                # Get history directly from JaatoClient to ensure we capture
                # in-progress turns (the agent state cache is only updated at turn end)
                # Phase 3 §7c step 6.6.4.5b: fetch history via the
                # ``session.get_history`` RPC instead of the daemon-side
                # JaatoClient indirection.  Captures in-progress turns
                # (the agent state cache only updates at turn end).
                history = []
                if session.server and session.server._runner_rpc is not None:
                    history = session.server._runner_rpc.session_get_history_threadsafe()
                turn_accounting = []

                if session.server:
                    main_id = session.server.main_agent_id
                    if main_id in session.server._agents:
                        turn_accounting = session.server._agents[main_id].turn_accounting

                # Resolve storage directory from workspace
                if session.workspace_path:
                    storage_dir = self._session_storage_dir(session.workspace_path)
                    # Keep the wake index current: this is the authoritative
                    # session_id → workspace mapping, used to revive a cold session
                    # by id alone (session.wake) without a caller-supplied path.
                    self._session_workspace_index.record(
                        session.session_id, session.workspace_path)
                else:
                    storage_dir = pathlib.Path(self._session_config.storage_path)

                # Get subagent state if subagent plugin is available
                subagent_metadata = {}
                if session.server and session.server.registry:
                    subagent_plugin = session.server.registry.get_plugin("subagent")
                    if subagent_plugin and hasattr(subagent_plugin, 'get_persistence_state'):
                        subagent_registry = subagent_plugin.get_persistence_state()
                        if subagent_registry.get('agents'):
                            subagent_metadata['subagents'] = subagent_registry

                            # Save per-agent state files
                            self._save_subagent_states(
                                session.session_id,
                                subagent_plugin,
                                subagent_registry.get('agents', []),
                                storage_dir=storage_dir,
                            )

                # Save TODO plugin state
                session_dir = storage_dir / session.session_id
                if session.server:
                    self._save_todo_state(session.server, session_dir)

                # Generic plugin state persistence: iterate all exposed plugins
                # and collect state from any that implement get_persistence_state().
                # Plugins with dedicated persistence (subagent, todo) are skipped
                # since they're handled above with their own file-based storage.
                plugin_states = {}
                _DEDICATED_PLUGINS = {'subagent', 'todo'}
                if session.server and session.server.registry:
                    for plugin_name in session.server.registry.list_exposed():
                        if plugin_name in _DEDICATED_PLUGINS:
                            continue
                        plugin = session.server.registry.get_plugin(plugin_name)
                        if plugin and hasattr(plugin, 'get_persistence_state'):
                            try:
                                pstate = plugin.get_persistence_state()
                                if pstate:
                                    plugin_states[plugin_name] = pstate
                            except Exception as e:
                                logger.warning(
                                    f"Failed to get persistence state for plugin "
                                    f"'{plugin_name}': {e}"
                                )
                if plugin_states:
                    subagent_metadata['plugin_states'] = plugin_states

                # Get conversation budget for persistence (other budget sources are
                # automatically recreated when the session is restored).
                # Phase 3 §7c step 6.6.3.6: forward to runner-side via
                # the new ``session.snapshot_conversation_budget`` RPC
                # (§7c step 6.6.3.2 at commit abd7ec08) instead of
                # reaching into the daemon-side session's
                # instruction_budget.
                budget_state = None
                if session.server is not None:
                    rpc = getattr(session.server, "_runner_rpc", None)
                    if rpc is not None:
                        snapshotter = getattr(
                            rpc,
                            "session_snapshot_conversation_budget_threadsafe",
                            None,
                        )
                        if callable(snapshotter):
                            try:
                                budget_state = snapshotter(timeout=5.0)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "snapshot_conversation_budget forward failed: %s",
                                    exc,
                                )

                # budget_control usage.  Separate from ``budget_state`` above
                # (that is the conversation budget).  BudgetTracker accumulates in
                # memory only, so without this an unloaded session came back with
                # a zeroed tracker and every cross-turn ceiling silently
                # restarted -- and sessions unload on ORPHAN, so a suspend/resume
                # driver is evicted on every wait.
                budget_usage = None
                budget_exhausted_reason = None
                # The CEILING this session ran under, recorded on the server when
                # its runner envelope was built (runner_spawn: server.
                # _effective_budget_control).  Read from the server rather than
                # the profile: a cascade-declared budget never touches the
                # profile, and the effective value is post-clamp.
                budget_control_cfg = getattr(
                    session.server, "_effective_budget_control", None,
                ) if session.server is not None else None
                if session.server is not None:
                    rpc = getattr(session.server, "_runner_rpc", None)
                    if rpc is not None:
                        usage_reader = getattr(
                            rpc, "session_get_budget_usage_threadsafe", None,
                        )
                        if callable(usage_reader):
                            try:
                                # tracker_only: persistence must never write the
                                # unbudgeted ``{"tokens": N}`` fallback over a
                                # real multi-dimension snapshot.  Doing so turns
                                # "the ceiling stopped applying" into "the
                                # ceiling can never be restored again" -- the
                                # only layer of this class that poisons the
                                # input to its own fix.
                                budget_usage = usage_reader(
                                    tracker_only=True, timeout=5.0) or None
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "budget usage snapshot failed: %s", exc)

                        # The ENFORCEMENT latch travels with the usage. Usage
                        # alone left a reloaded session at its ceiling with no
                        # memory of being stopped, so it served one more turn.
                        reason_reader = getattr(
                            rpc, "session_get_budget_exhausted_threadsafe", None,
                        )
                        if callable(reason_reader):
                            try:
                                budget_exhausted_reason = (
                                    reason_reader(timeout=5.0) or None)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "budget latch snapshot failed: %s", exc)

                # Get workspace file tracking state for persistence
                workspace_files = None
                monitor = self._workspace_monitors.get(session.session_id)
                if monitor:
                    workspace_files = monitor.get_tracked_dict() or None

                # Persist provisioned flag in metadata so restored sessions
                # know their workspace is server-managed.
                if session.provisioned:
                    subagent_metadata['provisioned'] = True

                # Create SessionState.  Post-2.3: persist ``profile_name``
                # (denormalised from the server's bound SubagentProfile) so
                # disk-restore can re-resolve the full provider recipe
                # (model + provider + plugin_configs + system_instructions +
                # GC) via the profile registry at load time.  The legacy
                # ``model`` field on SessionState was retired alongside
                # ``project`` / ``location`` — they were Google-GenAI-era
                # connection scaffolding.
                server_profile = getattr(session.server, "_profile", None) if session.server else None
                profile_name = getattr(server_profile, "name", None) if server_profile else None
                # Persona identity (``--agent``), so orphan-revive rebinds the same
                # persona (see SessionState.agent_name) — else a revived multimodal
                # session loses its enter_tier guidance and confabulates on images.
                agent_name = (
                    getattr(session.server, "_main_agent_display_name", None)
                    if session.server else None
                )
                # Freeze the recipe and the prompt this session ran under
                # (issue #787).  Both are captured ONCE -- see the
                # write-once note on the Session fields -- so a revive that
                # deliberately re-derives one of them cannot overwrite the
                # original artifact on its next save.
                self._capture_revive_snapshots(session, server_profile)
                state = SessionState(
                    session_id=session.session_id,
                    history=history,
                    created_at=datetime.fromisoformat(session.created_at),
                    updated_at=datetime.now(),
                    description=session.description or session.name,
                    turn_count=len(history) // 2,  # Approximate
                    turn_accounting=turn_accounting,
                    user_inputs=session.user_inputs,  # Command history for prompt restoration
                    profile_name=profile_name,
                    # Persist the UNRESOLVED inline spec (if any) so disk-restore
                    # reconstructs an inline profile's recipe by id alone — the
                    # named-profile ``profile_name`` ("<inline>") isn't
                    # re-resolvable.  None for named-profile sessions.
                    profile_spec=session.inline_profile_spec,
                    # 2.8+ (#787): the frozen recipe + the frozen prompt, so
                    # the revive restores rather than re-derives.  Both are
                    # None for sessions whose runner never reported one, and
                    # the loader then falls back to re-deriving -- the
                    # pre-2.8 behaviour.
                    profile_snapshot=session.profile_snapshot,
                    rendered_instructions=session.rendered_instructions,
                    # Persisted for the OPT-IN re-render path only (the
                    # default revive never re-runs a prefetch).  Never put a
                    # credential here: agent_params are substituted into the
                    # persona, so they already reach the model -- and the
                    # rendered persona above is on disk regardless.
                    agent_params=session.agent_params,
                    budget_control=budget_control_cfg,
                    sibling_name=session.sibling_name,
                    cascade_driver_id=session.cascade_driver_id,
                    workspace_path=session.workspace_path,
                    config_root=session.config_root,
                    # Persist confinement so orphan-revive / disk-restore re-applies
                    # the SAME AppArmor mode on runner re-spawn (else the revive read
                    # of state.sandbox_mode was always None → unconfined revive).
                    sandbox_mode=session.sandbox_mode,
                    agent_name=agent_name,
                    metadata=subagent_metadata,
                    budget_state=budget_state,
                    budget_usage=budget_usage,
                    budget_exhausted_reason=budget_exhausted_reason,
                    interrupted_turn=session.interrupted_turn,  # For recovery on restart
                    workspace_files=workspace_files,
                )

                self._session_plugin.save(state, storage_dir=storage_dir)
                session.is_dirty = False

                logger.debug(f"Saved session: {session.session_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to save session {session.session_id}: {e}")
                return False

    def _resolve_revive_profile(
        self,
        state: Any,
        *,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str],
        env_file: Optional[str],
    ) -> Optional[Any]:
        """Rebind the profile a revived session runs under.

        Three sources, tried in this order (issue #787):

        1. ``state.profile_snapshot`` -- the RESOLVED profile the session
           actually ran under, frozen at creation.  The default, and the
           one that makes the operator ruling on #787 true: *a revived
           session keeps what it was created under.*  Skipped when the
           operator asked for ``JAATO_REVIVE_PROFILE=disk``.
        2. ``state.profile_spec`` -- an INLINE session's own recipe.
           Authoritative and self-contained; an inline session was never a
           named profile, so it is never name-resolved (which could match
           an unrelated same-named profile on disk).
        3. ``state.profile_name`` -- re-resolved against the profile files
           AS THEY STAND NOW.  The pre-2.8 behaviour, and still the right
           one in two cases: a record with no snapshot (every session
           written before 2.8), and the deliberate
           ``JAATO_REVIVE_PROFILE=disk`` opt-in, which interrogation needs
           because a ``JAATO_PROFILE_SET`` switch is resolved inside
           ``discover_profiles`` and a frozen profile would make it inert.

        A snapshot that fails to rebuild falls through to (3) rather than
        failing the load: the worst case is the pre-#787 behaviour, and a
        session that cannot be woken at all is exactly the failure this
        change exists to remove.

        Args:
            state: The deserialized :class:`SessionState`.
            session_id: For log messages.
            workspace_path: Workspace to resolve a named profile against.
            config_root: Framework-config root override for discovery.
            env_file: Session ``.env``, overlaid before ``discover_profiles``
                so a workspace-declared ``JAATO_PROFILE_SET`` is visible.

        Returns:
            The rebound profile, or ``None`` when the session had none (or
            none of the three sources produced one -- ``initialize`` then
            falls back to env-only resolution, as it always has).
        """
        from server.revive_policy import DISK, profile_source

        snapshot = getattr(state, "profile_snapshot", None)
        if (
            snapshot
            and state.profile_spec is None
            and profile_source() != DISK
        ):
            from shared.plugins.subagent.config import profile_from_snapshot
            try:
                profile = profile_from_snapshot(snapshot)
                logger.info(
                    "_load_session: session %s restored from its persisted "
                    "profile snapshot (name=%s, model=%s, provider=%s); set "
                    "JAATO_REVIVE_PROFILE=disk to re-resolve from disk",
                    session_id, profile.name, profile.model, profile.provider,
                )
                return profile
            except (ValueError, TypeError) as exc:
                logger.error(
                    "_load_session: persisted profile snapshot for session "
                    "%s failed to rebuild (%s) -- falling back to resolving "
                    "%r from disk", session_id, exc, state.profile_name,
                )

        if state.profile_spec:
            # Uses the SAME build_inline_profile path create uses
            # (re-resolving any pass:// secrets daemon-side -- nothing
            # sensitive is on disk).  The spec is also carried onto the
            # restored Session so re-saves re-persist it.
            from shared.plugins.subagent.config import build_inline_profile
            try:
                profile = build_inline_profile(state.profile_spec)
                logger.info(
                    "_load_session: reconstructed inline profile for session "
                    "%s from persisted profile_spec (name=%s, model=%s, "
                    "provider=%s)", session_id, profile.name, profile.model,
                    profile.provider,
                )
                return profile
            except ValueError as exc:
                logger.error(
                    "_load_session: persisted inline profile_spec for session "
                    "%s failed to rebuild: %s", session_id, exc)
                return None

        if not state.profile_name:
            return None

        profile, profile_err = self._resolve_profile(
            state.profile_name,
            workspace_path=workspace_path,
            config_root=config_root,
            env_file=env_file,
        )
        if profile is None:
            logger.error(
                "_load_session: profile %r for session %s not "
                "resolvable (%s) -- initialize will likely fail "
                "(workspace=%s config_root=%s) -- verify the "
                "profile still exists at "
                "<config_root>/profiles/[<JAATO_PROFILE_SET>/]<name>",
                state.profile_name, session_id, profile_err,
                workspace_path, config_root,
            )
        return profile

    def _resolve_revive_persona(
        self,
        state: Any,
        restored_profile: Optional[Any],
        *,
        session_id: str,
        workspace_path: str,
        config_root: Optional[str],
    ) -> Optional[str]:
        """Decide what system instruction a revived session comes back with.

        Two outcomes (issue #787):

        * **Restore.**  The session persisted the prompt it was rendered
          with, and the operator has not asked for a re-render.  Returned
          as a ``system_instruction_override``, so the runner skips
          assembly entirely: no instruction layers re-read, no agent
          markdown re-resolved, and — the point — **no prefetch scripts
          re-run**.  Re-running them is what made a session with a
          mandatory ``{{!py:...}}`` prefetch unwakeable: ``agent_params``
          were not persisted, the script was handed an empty dict, raised,
          and aborted session-prep.  Re-running also repeated whatever side
          effects the script has (the reported case materialises a git
          worktree) and could hand the session a prompt its own history was
          not produced under.

        * **Re-render.**  No prompt was persisted (every record written
          before 2.8), or ``JAATO_REVIVE_PERSONA=disk`` asked for one.  The
          persona is re-resolved from disk and rebound onto
          ``restored_profile.system_instructions``, exactly as before —
          except that the ORIGINAL ``agent_params`` are now passed, so
          ``{{param}}`` placeholders substitute and a prefetch reading
          ``context.agent_params`` sees what it saw at creation.

        Note the snapshot deliberately restores the CONFIGURE-TIME render,
        not the live prompt: instructions a plugin injects when one of its
        tools first activates are re-produced by the revived session
        itself, so restoring them too would double them once per revive.

        Args:
            state: The deserialized :class:`SessionState`.
            restored_profile: The rebound profile, mutated in place on the
                re-render path.  ``None`` for profile-less sessions.
            session_id: For log messages.
            workspace_path: Workspace to resolve the agent markdown against.
            config_root: Framework-config root override for that lookup.

        Returns:
            The prompt to send as ``system_instruction_override``, or
            ``None`` to let the runner assemble normally.
        """
        from server.revive_policy import DISK, persona_source

        rendered = getattr(state, "rendered_instructions", None)
        if rendered and persona_source() != DISK:
            logger.info(
                "_load_session: session %s restored with its persisted "
                "system instruction (%d chars); prefetch scripts are NOT "
                "re-run.  Set JAATO_REVIVE_PERSONA=disk to re-render.",
                session_id, len(rendered),
            )
            return rendered

        if not (state.profile_name and restored_profile is not None
                and state.agent_name):
            return None

        # Rebind the persona from disk.  Restoring ``agent_name`` alone
        # gives the revived session the agent IDENTITY but not the persona
        # prose, and persona-only guidance (e.g. "call
        # ``enter_tier('vision')`` on user images") was silently dropped —
        # a revived multimodal session kept the tool and lost the
        # instruction to use it.
        agent_result = self._resolve_agent(
            state.agent_name,
            dict(getattr(state, "agent_params", None) or {}) or None,
            workspace_path,
            config_root=config_root,
        )
        if agent_result is None:
            logger.warning(
                "_load_session: agent %r for session %s not resolvable — "
                "persona (e.g. enter_tier guidance) missing on restore "
                "(config_root=%s)",
                state.agent_name, session_id, config_root,
            )
            return None
        restored_profile.system_instructions = agent_result["system_instructions"]
        return None

    def _capture_revive_snapshots(
        self, session: Session, server_profile: Optional[Any],
    ) -> None:
        """Fill in this session's frozen recipe + frozen prompt, once.

        Issue #787.  A revived session is supposed to come back as the
        session that was saved, and before this it came back as whatever
        the profile files and the persona's prefetch scripts produced at
        revive time.  These two snapshots are what "wake from persisted
        state" actually requires; :meth:`_load_session_impl` consumes
        them.

        WRITE-ONCE.  Each field is filled only when empty, so:

        * the value persisted is the one from the ORIGINAL run, even after
          many save cycles;
        * a revive that deliberately re-derives one of them
          (``JAATO_REVIVE_PROFILE=disk`` / ``JAATO_REVIVE_PERSONA=disk``)
          cannot overwrite the original on its next save -- testing an
          alternative must not destroy the record it is compared against;
        * an already-persisted session picks its snapshots up on its next
          save, so records written before 2.8 migrate forward by being
          used rather than by a migration step.

        Failures are logged and swallowed: a session that cannot be
        snapshotted must still SAVE.  The cost of a missing snapshot is a
        revive that re-derives, which is the pre-#787 behaviour -- the
        cost of a raised exception here would be a lost session.

        Args:
            session: The record being saved; mutated in place.
            server_profile: The resolved profile bound to the session's
                server, or ``None`` for profile-less sessions.
        """
        # --- the recipe ---------------------------------------------------
        # Inline sessions are already frozen: ``profile_spec`` persists the
        # spec itself because there is no name to re-resolve.  Snapshotting
        # them again would store the same recipe twice and give the loader
        # two sources to disagree about.
        if (
            session.profile_snapshot is None
            and session.inline_profile_spec is None
            and server_profile is not None
        ):
            try:
                from shared.plugins.subagent.config import profile_to_snapshot
                session.profile_snapshot = profile_to_snapshot(server_profile)
            except Exception as exc:  # noqa: BLE001 -- never block a save
                logger.warning(
                    "session %s: could not snapshot profile %r for revive "
                    "(%s: %s); a revive will re-resolve it from disk",
                    session.session_id,
                    getattr(server_profile, "name", None),
                    type(exc).__name__, exc,
                )

        # --- the prefetch inputs -------------------------------------------
        if session.agent_params is None and session.server is not None:
            params = getattr(session.server, "_agent_params", None)
            session.agent_params = dict(params) if params else None

        # --- the prompt ---------------------------------------------------
        if session.rendered_instructions is not None:
            return
        rpc = getattr(session.server, "_runner_rpc", None) if session.server else None
        reader = getattr(
            rpc, "session_get_rendered_system_instruction_threadsafe", None,
        ) if rpc is not None else None
        if not callable(reader):
            return
        try:
            rendered = reader(timeout=5.0)
        except Exception as exc:  # noqa: BLE001 -- never block a save
            logger.debug(
                "session %s: rendered-instruction snapshot failed: %s",
                session.session_id, exc,
            )
            return
        if rendered:
            session.rendered_instructions = rendered

    def _get_todo_plugin(self, server: JaatoServer) -> Optional[Any]:
        """Get the TODO plugin from a server's registry.

        Args:
            server: The JaatoServer instance.

        Returns:
            The TodoPlugin instance, or None if not available.
        """
        if not server or not server.registry:
            return None
        return server.registry.get_plugin("todo")

    def _configure_todo_storage(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Configure TODO plugin with session-scoped file storage.

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin:
            return

        # Resolve to absolute path to avoid issues with CWD changes
        # (e.g., when subagents call os.chdir() in background threads)
        plans_dir = (session_dir / "plans").resolve()
        todo_plugin.initialize({
            "storage_type": "file",
            "storage_path": str(plans_dir),
            "storage_use_directory": True,  # One file per plan
        })
        logger.debug(f"Configured TODO storage at: {plans_dir}")

    def _save_todo_state(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Save TODO plugin state (agent-plan mapping, blocked steps).

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin or not hasattr(todo_plugin, 'get_persistence_state'):
            return

        state = todo_plugin.get_persistence_state()
        if not state.get('agent_plan_ids'):
            # No plans to save
            return

        # Resolve to absolute path to avoid CWD issues
        state_path = (session_dir / "plans" / "_state.json").resolve()
        try:
            from shared.atomic_write import atomic_write_json
            atomic_write_json(state_path, state)
            logger.debug(f"Saved TODO state: {state_path}")
        except Exception as e:
            logger.error(f"Failed to save TODO state: {e}")

    def _load_todo_state(self, server: JaatoServer, session_dir: pathlib.Path) -> None:
        """Load TODO plugin state from disk.

        Args:
            server: The JaatoServer instance.
            session_dir: The session's storage directory.
        """
        # Resolve to absolute path to avoid CWD issues
        state_path = (session_dir / "plans" / "_state.json").resolve()
        if not state_path.exists():
            return

        todo_plugin = self._get_todo_plugin(server)
        if not todo_plugin or not hasattr(todo_plugin, 'restore_persistence_state'):
            return

        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            todo_plugin.restore_persistence_state(state)
            logger.debug(f"Loaded TODO state: {state_path}")
        except Exception as e:
            logger.error(f"Failed to load TODO state: {e}")

    def _save_subagent_states(
        self,
        session_id: str,
        subagent_plugin: Any,
        agents: List[Dict[str, Any]],
        storage_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """Save per-agent state files for subagents.

        Args:
            session_id: The parent session ID.
            subagent_plugin: The SubagentPlugin instance.
            agents: List of agent info dicts from the registry.
            storage_dir: Workspace-resolved session storage directory.
        """
        # Create subagents directory
        base = storage_dir or pathlib.Path(self._session_config.storage_path)
        subagents_dir = base / session_id / "subagents"
        subagents_dir.mkdir(parents=True, exist_ok=True)

        for agent_info in agents:
            agent_id = agent_info.get('agent_id')
            if not agent_id:
                continue

            # Get full state from plugin
            full_state = subagent_plugin.get_agent_full_state(agent_id)
            if not full_state:
                continue

            # Write to file (atomically — Phase 3 §3.14).  A SIGTERM
            # mid-write would otherwise leave a corrupt subagent state
            # file that the disk-restore path can't parse.
            agent_file = subagents_dir / f"{agent_id}.json"
            try:
                from shared.atomic_write import atomic_write_json
                atomic_write_json(agent_file, full_state)
                logger.debug(f"Saved subagent state: {agent_file}")
            except Exception as e:
                logger.error(f"Failed to save subagent {agent_id}: {e}")

    def _maybe_unload_session(self, session_id: str) -> None:
        """Unload a session from memory if no clients attached.

        Saves to disk first if dirty.  The heavy cleanup work (save,
        workspace monitor stop, subagent teardown, server.shutdown)
        is deferred to a background thread because callers may
        invoke this from the asyncio loop's IPC disconnect handler.
        The deferred-to-thread work performs ``run_coroutine_threadsafe``
        RPCs to the runner; if those ran on the same loop they
        deadlock (the scheduled coro can't progress while the
        calling task holds the loop).  Pre-2026-05-14 the chain ran
        inline and froze the loop for ~55 s (30 s save + 10 s
        session.shutdown + 10 s runner-rpc close + 5 s buffer)
        per disconnect; the per-session unload error message logged
        as ``Failed to save session ...:`` (empty ``str(exc)``) was
        the timeout's tell.  Render exceptions with
        :func:`shared.utils.errors.exc_message` so that tell can never be
        an empty slot again.

        Args:
            session_id: The session to potentially unload.
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        if session.attached_clients:
            return  # Still has clients

        # Don't unload while the model thread is still running — the
        # client may have disconnected (WS ping timeout, network blip)
        # but the model is mid-turn processing tools.  The session will
        # be unloaded when the model thread finishes and checks again.
        if session.server and session.server._model_running:
            logger.info(
                "Deferring unload of session %s — model thread still active",
                session_id,
            )
            return

        # Defer heavy cleanup to a background thread so the asyncio
        # loop (when this is called from an IPC disconnect handler)
        # doesn't deadlock on ``run_coroutine_threadsafe`` RPCs to
        # the runner.  Fire-and-forget — no caller awaits the unload
        # completion (verified against the 3 call sites).
        thread = threading.Thread(
            target=self._do_session_unload,
            args=(session_id,),
            name=f"unload-{session_id}",
            daemon=True,
        )
        thread.start()

    def _do_session_unload(self, session_id: str) -> None:
        """Background-thread body of session unload.

        Invoked by :meth:`_maybe_unload_session` after the pre-checks
        confirm no attached clients and no active model thread.  Does
        the actual save + plugin teardown + server.shutdown on a
        thread distinct from the daemon's asyncio loop so the
        ``run_coroutine_threadsafe`` RPCs to the runner can complete
        normally.

        Re-checks ``self._sessions`` under the lock — a new client
        might have attached between the pre-check and this thread's
        first action.  If so, abort the unload.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            # Re-check attached clients: a new client could have
            # attached between _maybe_unload_session's check and
            # this thread starting.
            if session.attached_clients:
                logger.info(
                    "Unload of session %s aborted — a client re-attached "
                    "while the unload thread was scheduling", session_id,
                )
                return
            if session.server and session.server._model_running:
                logger.info(
                    "Unload of session %s aborted — model thread became "
                    "active while the unload thread was scheduling",
                    session_id,
                )
                return
            # Commit point: no clients, no active model → we ARE unloading.
            # Publish the in-flight marker UNDER THE SAME LOCK as the re-check
            # above, so a concurrent attach_session either added its client
            # first (the re-check aborted us) or sees this marker and
            # awaits+reloads. The lock makes the two paths mutually exclusive.
            self._unloading[session_id] = threading.Event()

        # Save before unloading.  Now running on a real thread so
        # session_get_history_threadsafe + budget snapshot RPCs work
        # without deadlocking.
        if session.is_dirty:
            self._save_session(session)

        # Close session-specific log handlers
        handler = get_session_handler()
        if handler:
            handler.close_session(session_id)

        # Stop workspace monitor
        self._stop_workspace_monitor(session_id)

        # Phase 4 §4.3.6d: cascade-teardown any isolated subagents
        # owned by this parent.  Must run BEFORE the parent's
        # server.shutdown() so we have a chance to close sub-runner
        # RPCs cleanly; running after shutdown leaves orphaned
        # processes that the daemon loses track of.
        try:
            n_torn_down = self._cascade_teardown_isolated_subagents(
                parent_session_id=session_id,
            )
            if n_torn_down > 0:
                logger.info(
                    "Unload: cascade-teardown completed for %d isolated "
                    "subagent(s) of session %s",
                    n_torn_down, session_id,
                )
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "Unload: cascade-teardown raised for session %s — "
                "continuing with server.shutdown",
                session_id,
            )

        # Shutdown server and remove from memory.
        try:
            session.server.shutdown()
        except Exception:  # noqa: BLE001 — best-effort
            logger.exception(
                "Unload: server.shutdown raised for session %s — "
                "removing from sessions anyway", session_id,
            )
        with self._lock:
            self._sessions.pop(session_id, None)
            # Unload complete: signal any attach_session awaiting this teardown,
            # then drop the in-flight marker so a fresh attach takes the
            # disk-restore path (_load_session) and re-spawns the runner.
            done_evt = self._unloading.pop(session_id, None)
        if done_evt is not None:
            done_evt.set()
        logger.info(f"Unloaded session: {session_id}")

    # ------------------------------------------------------------------
    # Prompt skill expansion
    # ------------------------------------------------------------------

    _PROMPT_REF_PATTERN = re.compile(r'%([a-zA-Z][a-zA-Z0-9_-]*)')

    # ``%<name> --help`` / ``%<name> -h`` — the leading whitespace and the
    # flag are consumed together so the span can be excised cleanly from
    # the outgoing message without leaving a dangling ``--help`` token.
    _PROMPT_HELP_REF_PATTERN = re.compile(
        r'%([a-zA-Z][a-zA-Z0-9_-]*)[ \t]+(--help|-h)\b'
    )

    def _intercept_prompt_help_refs(
        self,
        text: str,
        server: 'JaatoServer',
        client_id: Optional[str],
    ) -> str:
        """Handle ``%<name> --help`` references before model dispatch.

        Scans *text* for ``%name --help`` (or ``-h``) tokens, resolves
        each via the prompt library plugin, emits the resulting help as
        a :class:`HelpTextEvent` (pager) directly to *client_id*, and
        returns *text* with those tokens excised.  This runs *before*
        :meth:`_expand_prompt_references` so the help content never
        reaches the model — the user types ``%foo --help`` and sees the
        help, without the model also being asked to respond.

        If the prompt library plugin is unavailable, or a referenced
        prompt doesn't exist, the reference is left untouched and
        flows through to the normal expansion path for graceful
        degradation.

        Args:
            text: Raw message text from the client.
            server: The session's :class:`JaatoServer` for plugin access.
            client_id: Target client for the emitted help events. When
                ``None`` (rare — only during offline replay), the
                matched spans are still stripped but no events fire.

        Returns:
            The message text with matched ``%<name> --help`` spans
            removed.  If the entire original message consisted only of
            such references, the return value is the surrounding
            whitespace — the caller should check ``.strip()`` and skip
            the model call.
        """
        if not text or '%' not in text:
            return text

        matches = list(self._PROMPT_HELP_REF_PATTERN.finditer(text))
        if not matches:
            return text

        # Phase 3 §7c step 6.6.4.5a: read ``server._runtime`` directly.
        prompt_plugin = None
        runtime = server._runtime
        if runtime and runtime.registry:
            prompt_plugin = runtime.registry.get_plugin("prompt_library")
        if prompt_plugin is None or not hasattr(
            prompt_plugin, '_execute_prompt_command'
        ):
            return text

        from jaato_sdk.plugins.base import HelpLines
        from jaato_sdk.events import HelpTextEvent, SystemMessageEvent

        result = text
        # Walk matches in reverse so earlier spans' positions remain
        # valid while we cut later spans out of ``result``.
        for match in reversed(matches):
            # ``%`` must stand alone — skip if preceded by an
            # alphanumeric (e.g. ``100%foo`` is not a prompt reference).
            if match.start() > 0 and text[match.start() - 1].isalnum():
                continue

            prompt_name = match.group(1)
            try:
                help_result = prompt_plugin._execute_prompt_command(
                    {'args': [prompt_name, '--help']}
                )
            except Exception:
                continue

            if isinstance(help_result, HelpLines):
                if client_id is not None:
                    self._emit_to_client(
                        client_id, HelpTextEvent(lines=help_result.lines)
                    )
            elif isinstance(help_result, str):
                # Fallback: plugin returned an error string (e.g.
                # "Prompt not found") — surface it as a system message
                # rather than pushing it to the model.
                if client_id is not None:
                    self._emit_to_client(
                        client_id, SystemMessageEvent(
                            message=help_result, style="error"
                            if help_result.startswith('Prompt not found')
                            else "info",
                        )
                    )
            else:
                # Unknown return type — leave reference in place and
                # let the normal expansion path handle it.
                continue

            result = result[:match.start()] + result[match.end():]

        return result

    def _expand_prompt_references(self, text: str, server: 'JaatoServer') -> str:
        """Expand ``%prompt-name`` references in a message.

        Scans *text* for ``%name`` tokens, resolves each via the prompt
        library plugin, strips the ``%`` prefix from the message body,
        and appends the prompt content in a ``--- Referenced Prompts ---``
        section (same format as the TUI client).

        This runs server-side so that **all** clients (TUI, WS, IPC)
        get automatic prompt skill injection regardless of client-side
        capabilities.

        If the prompt library plugin is not available or a prompt is not
        found, the ``%`` prefix is stripped but no content is appended
        (same graceful degradation as the TUI).

        Args:
            text: Raw message text from the client.
            server: The session's ``JaatoServer`` for plugin access.

        Returns:
            The message with prompt references expanded, or the
            original text if no references are found.
        """
        if not text or '%' not in text:
            return text

        matches = list(self._PROMPT_REF_PATTERN.finditer(text))
        if not matches:
            return text

        # Phase 3 §7c step 6.6.4.5a: read ``server._runtime`` directly.
        prompt_plugin = None
        runtime = server._runtime
        if runtime and runtime.registry:
            prompt_plugin = runtime.registry.get_plugin("prompt_library")

        # Process matches in reverse to preserve positions
        result = text
        expanded = []
        for match in reversed(matches):
            # Skip if preceded by alphanumeric (not a standalone reference)
            if match.start() > 0 and text[match.start() - 1].isalnum():
                continue

            prompt_name = match.group(1)

            # Capture args after the prompt name on the same line so the
            # expansion can substitute them.  Args run from the end of the
            # name match to the next newline (or end of text), supporting
            # both positional tokens and ``key=value`` named tokens.  The
            # original args are LEFT in the message body — they remain
            # visible to the user, while the expansion gets a fully
            # populated copy via _execute_prompt_command.
            args_start = match.end()
            newline_pos = text.find('\n', args_start)
            args_end = newline_pos if newline_pos != -1 else len(text)
            args_text = text[args_start:args_end].strip()
            # Tokenize via the prompt library's shared splitter so that
            # ``%name key="value with spaces"`` and ``%name "positional
            # value"`` work consistently across the text path and the
            # slash-command path.
            from shared.plugins.prompt_library.plugin import tokenize_prompt_args
            prompt_args = tokenize_prompt_args(args_text)

            # Strip the % prefix regardless of whether we find the prompt
            result = result[:match.start()] + prompt_name + result[match.end():]

            # Try to expand via prompt library
            if prompt_plugin and hasattr(prompt_plugin, '_execute_prompt_command'):
                try:
                    content = prompt_plugin._execute_prompt_command(
                        {'args': [prompt_name] + prompt_args}
                    )
                    # _execute_prompt_command usually returns strings,
                    # but ``help``-style calls can return HelpLines — the
                    # latter shouldn't be embedded in the outgoing message
                    # (both because it has no sensible f-string form and
                    # because it's meant for the user, not the model).
                    if isinstance(content, str) and content and not content.startswith('Prompt not found'):
                        expanded.append({'name': prompt_name, 'content': content})
                except Exception:
                    pass  # Graceful degradation

        if expanded:
            # Reverse to restore original order
            expanded.reverse()
            parts = [result, "\n\n--- Referenced Prompts ---\n"]
            for prompt in expanded:
                parts.append(f"\n[Prompt: {prompt['name']}]\n")
                parts.append(f"{prompt['content']}\n")
            return ''.join(parts)

        return result

    def detach_client(self, client_id: str) -> None:
        """Detach a client from its current session.

        Args:
            client_id: The client to detach.
        """
        with self._lock:
            session_id = self._client_to_session.pop(client_id, None)
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.attached_clients.discard(client_id)
                logger.info(f"Client {client_id} detached from session {session_id}")

                # Maybe unload if no more clients
                self._maybe_unload_session(session_id)

    def save_session(self, session_id: str) -> bool:
        """Explicitly save a session to disk.

        Args:
            session_id: The session to save.

        Returns:
            True if saved successfully.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
        # The save happens OUTSIDE ``self._lock``.  ``_save_session`` issues
        # ``session_get_history_threadsafe()``, which schedules a coroutine
        # onto the daemon loop and blocks this thread until it completes --
        # so holding the manager lock across it lets a worker wait for the
        # loop while the loop waits for the lock.  See the module note on
        # THE MANAGER LOCK IS A DICT GUARD.
        return self._save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from memory and disk.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted.
        """
        workspace_path = None
        with self._lock:
            # Remove from memory — capture workspace_path before popping
            session = self._sessions.pop(session_id, None)
            if session:
                workspace_path = session.workspace_path
                # Notify attached clients
                for client_id in session.attached_clients:
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=f"Session deleted: {session.name}",
                        style="warning",
                    ))
                    self._client_to_session.pop(client_id, None)

        # Shutdown the server OUTSIDE ``self._lock``.  ``JaatoServer.shutdown``
        # issues ``session_end_threadsafe`` / ``session_shutdown_threadsafe``
        # and closes the RPC via ``run_coroutine_threadsafe`` -- all of which
        # block this thread until the daemon loop runs them.  The session is
        # already popped above, so nothing else can reach it.
        if session is not None:
            session.server.shutdown()

        # Stop the session's egress proxy if one was started (Phase 5 §5.11).
        # Idempotent + guarded — a no-op for sessions without an allowlist.
        try:
            from server.egress_proxy import wireup as _egress_wireup
            _egress_wireup.egress_teardown(session_id)
        except Exception:  # pragma: no cover - defensive
            logger.warning("egress proxy teardown failed for %s", session_id,
                           exc_info=True)

        # Delete from disk
        storage_dir = self._session_storage_dir(workspace_path) if workspace_path else None
        deleted = self._session_plugin.delete(session_id, storage_dir=storage_dir)

        # Release this session's daemon-side declarations so they don't outlive
        # it.  A wake binding is the session's "wake me" invitation; a deleted
        # session is gone for good, so its bindings must go too — else the
        # owner-guard blocks a NEW session from re-binding that wake_ref until
        # TTL (days), and any wake that resolved would target a dead session.
        # (Distinct from UNLOAD, which keeps bindings so a cold session stays
        # wakeable — the #520 cold-revive durability.)
        released = self._wake_binding_registry.release_for_session(session_id)
        if released:
            logger.info("released %d wake binding(s) for deleted session %s",
                        released, session_id)
        # Drop the id→workspace index entry too (no TTL there — a stale entry
        # would otherwise persist forever and mis-resolve a later id collision).
        self._session_workspace_index.forget(session_id)

        logger.info(f"Session deleted: {session_id}")
        return deleted or session is not None

    def _normalize_workspace(self, path: Optional[str]) -> Optional[str]:
        """Normalize a workspace path for comparison.

        Args:
            path: The path to normalize.

        Returns:
            Normalized absolute path, or None if path is None.
        """
        if not path:
            return None
        import os
        return os.path.normpath(os.path.abspath(path))

    def _workspaces_match(
        self,
        path1: Optional[str],
        path2: Optional[str],
    ) -> bool:
        """Check if two workspace paths match.

        Args:
            path1: First path.
            path2: Second path.

        Returns:
            True if both paths are set and point to the same directory.
        """
        norm1 = self._normalize_workspace(path1)
        norm2 = self._normalize_workspace(path2)
        if not norm1 or not norm2:
            return False
        return norm1 == norm2

    def get_or_create_default(
        self,
        client_id: str,
        workspace_path: Optional[str] = None,
    ) -> str:
        """Get the default session for a workspace, or create a new one.

        Finds the most recently used session for the given workspace.
        Creates a new session if no matching session exists.

        Args:
            client_id: The requesting client.
            workspace_path: Client's working directory for file operations.

        Returns:
            The session ID.
        """
        logger.debug(f"get_or_create_default called for client {client_id}, workspace={workspace_path}")

        # Check in-memory sessions first - find one matching the workspace
        with self._lock:
            if self._sessions and workspace_path:
                # Find sessions matching this workspace
                matching_sessions = [
                    s for s in self._sessions.values()
                    if self._workspaces_match(s.workspace_path, workspace_path)
                ]
                if matching_sessions:
                    # Use the first matching session (they're all for the same workspace)
                    session = matching_sessions[0]
                    logger.debug(f"  found in-memory session for workspace: {session.session_id}")
                    session.attached_clients.add(client_id)
                    self._client_to_session[client_id] = session.session_id
                    # Emit current agent state to the newly attached client
                    session.server.emit_current_state(
                        lambda e: self._emit_to_client(client_id, e),
                        skip_session_info=True
                    )
                    # Send complete SessionInfoEvent with state snapshot
                    self._emit_to_client(client_id, self._build_session_info_event(session))
                    return session.session_id

        # Check persisted sessions (already sorted by updated_at descending)
        logger.debug(f"  checking persisted sessions...")
        persisted = self._get_persisted_sessions(workspace_path=workspace_path)
        logger.debug(f"  found {len(persisted)} persisted session(s)")

        if persisted and workspace_path:
            # Find sessions matching this workspace
            matching_persisted = [
                s for s in persisted
                if self._workspaces_match(s.workspace_path, workspace_path)
            ]
            logger.debug(f"  found {len(matching_persisted)} session(s) for workspace")

            if matching_persisted:
                # Use the most recent one for this workspace
                most_recent = matching_persisted[0]
                logger.debug(f"  attaching to workspace session: {most_recent.session_id}")
                if self.attach_session(client_id, most_recent.session_id, workspace_path):
                    return most_recent.session_id

        # No matching sessions exist - create a new one for this workspace
        logger.debug(f"  creating new session for workspace...")
        return self.create_session(client_id, workspace_path=workspace_path)

    # =========================================================================
    # Session Queries
    # =========================================================================

    def _get_persisted_sessions(
        self,
        workspace_path: Optional[str] = None,
    ) -> List[PluginSessionInfo]:
        """Get list of sessions from disk.

        Args:
            workspace_path: Workspace directory to list sessions for.
                When provided, lists sessions from that workspace's storage.
        """
        storage_dir = self._session_storage_dir(workspace_path) if workspace_path else None
        try:
            return self._session_plugin.list_sessions(storage_dir=storage_dir)
        except Exception as e:
            logger.error(f"Failed to list persisted sessions: {e}")
            return []

    def list_profiles(
        self,
        workspace_path: Optional[str] = None,
        config_root: Optional[str] = None,
    ) -> Tuple[List["ProfileSummary"], List["ProfileParseError"]]:
        """List available agent profiles.

        Discovers profiles from three sources in decreasing precedence:
        workspace ``.jaato/profiles/``, user ``~/.jaato/profiles/``,
        and premium entry-point profiles.

        Profiles that fail to parse are returned in the second element
        (parse errors) so the caller can surface them distinctly from
        usable profiles — picker UIs typically render the two lists
        differently (badges vs. the main grid).

        Sensitive material is filtered: env *values* are dropped (only
        variable names survive); ``system_instructions`` and
        ``inherits`` are not exposed (deprecated or already resolved).

        Args:
            workspace_path: Workspace directory to discover profiles from.
                May be ``None`` (user-level and premium profiles are still
                returned).

        Returns:
            Tuple ``(profiles, parse_errors)``.  ``profiles`` is the
            list of typed ``ProfileSummary`` records; ``parse_errors``
            is the list of files that failed discovery.
        """
        from dataclasses import asdict
        from shared.plugins.subagent.config import discover_profiles
        from jaato_sdk.events import ProfileSummary, ProfileParseError

        discovery = discover_profiles(
            ".jaato/profiles",
            base_path=workspace_path or ".",
            config_root=config_root,
        )
        summaries: List[ProfileSummary] = []
        for name, profile in discovery.profiles.items():
            summaries.append(ProfileSummary(
                name=name,
                description=profile.description,
                plugins=profile.plugins,
                preloaded_plugins=sorted(profile.preloaded_plugins),
                plugin_configs=profile.plugin_configs,
                model=profile.model,
                provider=profile.provider,
                max_turns=profile.max_turns,
                model_tiers=profile.model_tiers,
                budget_control=(
                    profile.budget_control.to_dict()
                    if profile.budget_control else None
                ),
                gc=asdict(profile.gc) if profile.gc else None,
                runtime_limits=(
                    asdict(profile.runtime_limits)
                    if profile.runtime_limits else None
                ),
                completion_payload_schema=profile.completion_payload_schema,
                env_var_names=sorted(profile.env.keys()),
            ))

        parse_errors: List[ProfileParseError] = [
            ProfileParseError(name=stem, error=error)
            for stem, error in discovery.errors.items()
        ]
        return summaries, parse_errors

    def list_sessions(self) -> List[RuntimeSessionInfo]:
        """List all sessions (in-memory and on-disk).

        Returns merged view with runtime status for loaded sessions.
        Collects persisted sessions from all known workspaces (in-memory
        sessions + client configs).
        """
        result: Dict[str, RuntimeSessionInfo] = {}

        # Collect all known workspace paths from in-memory sessions and client configs
        known_workspaces: Set[str] = set()
        with self._lock:
            for session in self._sessions.values():
                if session.workspace_path:
                    norm = self._normalize_workspace(session.workspace_path)
                    if norm:
                        known_workspaces.add(norm)
            for config in self._client_config.values():
                wp = config.get('working_dir')
                if wp:
                    norm = self._normalize_workspace(wp)
                    if norm:
                        known_workspaces.add(norm)
        # Union the session-workspace index's workspaces so cold, persisted
        # sessions in server-provisioned ``ws_<hash>`` dirs (which no
        # in-memory session or attached client references) still surface —
        # the discovery half of workspace-pinless (browser/WS) resume, mirror
        # of the attach-time index fallback above.
        for norm in (self._normalize_workspace(w)
                     for w in self._session_workspace_index.workspaces()):
            if norm:
                known_workspaces.add(norm)

        # Add persisted sessions from all known workspaces.
        # ``model_name`` left blank for persisted-only entries — the
        # post-2.3 SessionInfo carries ``profile_name`` instead of
        # ``model`` (profile is the post-multi-provider recipe
        # source).  Clients that want the model resolve the profile
        # registry on-demand rather than denormalising into the
        # listing index.
        for wp in known_workspaces:
            for info in self._get_persisted_sessions(workspace_path=wp):
                result[info.session_id] = RuntimeSessionInfo(
                    session_id=info.session_id,
                    name=info.description or info.session_id,
                    description=info.description,
                    created_at=info.created_at.isoformat(),
                    last_activity=info.updated_at.isoformat(),
                    model_provider="",
                    model_name="",
                    is_processing=False,
                    is_loaded=False,
                    client_count=0,
                    turn_count=info.turn_count,
                    workspace_path=info.workspace_path,
                )

        # Overlay in-memory sessions (have more current info)
        with self._lock:
            for session in self._sessions.values():
                result[session.session_id] = RuntimeSessionInfo(
                    session_id=session.session_id,
                    name=session.name,
                    description=session.description,
                    created_at=session.created_at,
                    last_activity=session.last_activity,
                    model_provider=session.server.model_provider,
                    model_name=session.server.model_name,
                    is_processing=session.server.is_processing,
                    is_loaded=True,
                    client_count=len(session.attached_clients),
                    turn_count=len(session.server.get_history()) // 2,
                    workspace_path=session.workspace_path,
                    created_by=session.created_by,
                )

        # Sort by last activity
        sessions = list(result.values())
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions

    def _build_session_info_event(self, session: "Session") -> SessionInfoEvent:
        """Build a complete SessionInfoEvent with state snapshot.

        Includes current session info plus:
        - sessions: All available sessions for completion/display
        - tools: All tools with enabled status
        - models: Available model names
        """
        # Get sessions list
        # Build sessions list. Enrich with sandbox_mode from Session objects
        # (sandbox_mode is set by the WS server during workspace provisioning).
        session_lookup = {s.session_id: s for s in self._sessions.values()}
        sessions_data = []
        for s in self.list_sessions():
            entry = {
                "id": s.session_id,
                "name": s.name or "",
                "description": s.description or "",
                "model_provider": s.model_provider or "",
                "model_name": s.model_name or "",
                "is_loaded": s.is_loaded,
                "client_count": s.client_count,
                "turn_count": s.turn_count,
                "workspace_path": s.workspace_path or "",
            }
            sess = session_lookup.get(s.session_id)
            if sess and sess.sandbox_mode:
                entry["sandbox_mode"] = sess.sandbox_mode
            sessions_data.append(entry)

        # Get tools list from the session's server
        tools_data = []
        if session.server:
            tools_data = session.server.get_tool_status()

        # Models list is lazy-loaded on demand to avoid API calls during init
        # Client fetches models when user requests completions
        models_data = []

        # Get memory metadata from the session's server for completion cache
        memories_data = []
        if session.server:
            mem_plugin = session.server._find_plugin_for_command("memory")
            if mem_plugin and hasattr(mem_plugin, 'get_memory_metadata'):
                memories_data = mem_plugin.get_memory_metadata()

        # Get sandbox paths from the session's server for @@ completion cache
        sandbox_paths_data = []
        if session.server:
            sandbox_paths_data = session.server._get_sandbox_paths()

        # Get service metadata from the session's server for completion cache
        services_data = []
        if session.server:
            svc_plugin = session.server._find_plugin_for_command("services")
            if svc_plugin and hasattr(svc_plugin, 'get_service_metadata'):
                services_data = svc_plugin.get_service_metadata()

        # Build tool ID mappings for client-side display resolution
        tool_id_mappings = {}
        if session.server:
            tool_id_mappings = session.server._build_tool_id_mappings()

        return SessionInfoEvent(
            session_id=session.session_id,
            session_name=session.name,
            model_provider=session.server.model_provider if session.server else "",
            model_name=session.server.model_name if session.server else "",
            profile_name=session.server.profile_name if session.server else None,
            sessions=sessions_data,
            tools=tools_data,
            models=models_data,
            user_inputs=session.user_inputs,  # Command history for prompt restoration
            memories=memories_data,
            sandbox_paths=sandbox_paths_data,
            services=services_data,
            tool_id_mappings=tool_id_mappings,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID (in-memory only)."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_client_session(self, client_id: str) -> Optional[Session]:
        """Get the session a client is attached to."""
        with self._lock:
            session_id = self._client_to_session.get(client_id)
            if session_id:
                return self._sessions.get(session_id)
        return None

    # =========================================================================
    # Workspace Snapshot
    # =========================================================================

    def snapshot_workspace(
        self,
        target_session_id: str,
        requester_workspace: str,
    ) -> Dict[str, Any]:
        """Copy a target session's workspace to the requester's replay area.

        For git-managed workspaces, uses ``git archive`` for committed
        content and manually copies untracked files.  For non-git
        workspaces, uses ``shutil.copytree``.

        The snapshot is created inside the requester's workspace at
        ``<requester_workspace>/.jaato/replay/<uuid>/`` so the
        requester's AppArmor profile can access it without any
        confinement changes.

        The target session is NOT paused during the copy.  The files
        the fine-tuner cares about (profiles, agent markdown, prompts,
        service configs) are written once at session creation and never
        modified during a turn, so a consistent snapshot does not
        require pausing.

        Args:
            target_session_id: The session whose workspace to snapshot.
            requester_workspace: The requesting session's workspace
                path (destination parent).

        Returns:
            Dict with ``snapshot_path``, ``source_session_id``, and
            ``source_commit`` (``None`` if not a git repo).

        Raises:
            ValueError: If the target session is not found, has no
                workspace, or if the workspace doesn't exist.
            OSError: If file operations fail.
        """
        import shutil
        import subprocess
        import uuid as _uuid

        session = self.get_session(target_session_id)
        if session is None:
            raise ValueError(f"Session '{target_session_id}' not found")
        workspace = session.workspace_path
        if not workspace:
            raise ValueError(
                f"Session '{target_session_id}' has no workspace"
            )
        if not os.path.isdir(workspace):
            raise ValueError(f"Workspace does not exist: {workspace}")

        dest_dir = os.path.join(
            requester_workspace, ".jaato", "replay", str(_uuid.uuid4()),
        )
        os.makedirs(dest_dir, exist_ok=True)

        source_commit: Optional[str] = None
        is_git = os.path.isdir(os.path.join(workspace, ".git"))

        try:
            if is_git:
                # Committed files via git archive
                archive = subprocess.run(
                    ["git", "-C", workspace, "archive", "HEAD"],
                    capture_output=True,
                    check=True,
                )
                subprocess.run(
                    ["tar", "-x", "-C", dest_dir],
                    input=archive.stdout,
                    check=True,
                )
                # Capture current commit
                rev = subprocess.run(
                    ["git", "-C", workspace, "rev-parse", "HEAD"],
                    capture_output=True, text=True,
                )
                source_commit = rev.stdout.strip() if rev.returncode == 0 else None

                # Copy untracked files
                status = subprocess.run(
                    ["git", "-C", workspace, "status", "--porcelain"],
                    capture_output=True, text=True,
                )
                if status.returncode == 0:
                    for line in status.stdout.splitlines():
                        if line.startswith("??"):
                            rel_path = line[3:].strip()
                            src = os.path.join(workspace, rel_path)
                            dst = os.path.join(dest_dir, rel_path)
                            if os.path.isdir(src):
                                shutil.copytree(
                                    src, dst,
                                    ignore=shutil.ignore_patterns(
                                        "__pycache__", "*.pyc",
                                    ),
                                )
                            elif os.path.isfile(src):
                                os.makedirs(os.path.dirname(dst), exist_ok=True)
                                shutil.copy2(src, dst)
            else:
                # Non-git: plain copytree
                shutil.copytree(
                    workspace, dest_dir,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        ".git", ".venv", "__pycache__", "node_modules",
                        "*.pyc",
                    ),
                )
        except Exception:
            # Clean up partial snapshot on failure
            if os.path.isdir(dest_dir):
                shutil.rmtree(dest_dir, ignore_errors=True)
            raise

        logger.info(
            "Snapshot workspace '%s' → '%s' (commit=%s)",
            workspace, dest_dir, source_commit,
        )
        return {
            "snapshot_path": dest_dir,
            "source_session_id": target_session_id,
            "source_commit": source_commit,
        }

    # =========================================================================
    # Turn Tracking for Recovery
    # =========================================================================

    def start_turn_tracking(
        self,
        session_id: str,
        user_prompt: str,
        agent_id: str = "main"
    ) -> None:
        """Mark a turn as in-progress for recovery purposes.

        Call this when a turn starts (user sends message) to enable recovery
        if the server crashes during tool execution.

        Args:
            session_id: The session ID.
            user_prompt: The user's original prompt.
            agent_id: Which agent is executing ("main" or subagent ID).
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.interrupted_turn = {
                    "agent_id": agent_id,
                    "pending_tool_calls": [],
                    "user_prompt": user_prompt,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
                session.is_dirty = True
                logger.debug(f"Started turn tracking for session {session_id}, agent {agent_id}")

    def update_pending_tool_calls(
        self,
        session_id: str,
        function_calls: List[Dict[str, Any]]
    ) -> None:
        """Update pending tool calls after model response.

        Call this after the model returns function calls, before tool execution.
        This triggers an incremental save so the pending calls are persisted.

        Args:
            session_id: The session ID.
            function_calls: List of {id, name, args} dicts from model response.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.interrupted_turn:
                session.interrupted_turn["pending_tool_calls"] = function_calls
                session.is_dirty = True
            else:
                session = None
        if session is not None:
            # Outside the lock: the save round-trips to the loop.  The
            # MUTATION above needs the lock; the save does not.
            self._save_session(session)
            logger.debug(
                    f"Updated pending tool calls for session {session_id}: "
                    f"{len(function_calls)} call(s)"
                )

    def clear_turn_tracking(self, session_id: str) -> None:
        """Clear turn tracking on successful completion.

        Call this when a turn completes successfully (no more function calls).

        Args:
            session_id: The session ID.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.interrupted_turn = None
                session.is_dirty = True
                logger.debug(f"Cleared turn tracking for session {session_id}")

    # =========================================================================
    # Request Routing
    # =========================================================================

    def handle_request(
        self,
        client_id: str,
        session_id: str,
        event: Event,
    ) -> None:
        """Route a request to the appropriate session.

        Args:
            client_id: The requesting client.
            session_id: The target session.
            event: The request event.
        """
        from jaato_sdk.events import ClientConfigRequest

        # Handle client config before session lookup (doesn't require session)
        if isinstance(event, ClientConfigRequest):
            self._apply_client_config(client_id, event)
            return

        session = self.get_session(session_id)
        if not session:
            self._emit_to_client(client_id, ErrorEvent(
                error=f"Session not found: {session_id}",
                error_type="SessionError",
            ))
            return

        # Update activity timestamp
        session.last_activity = datetime.now(timezone.utc).isoformat()
        session.is_dirty = True

        # Route to session's server
        server = session.server

        # Set logging context for session-specific log routing
        workspace_path = session.workspace_path
        session_env = server.get_all_session_env() if server else {}
        set_logging_context(
            session_id=session_id,
            client_id=client_id,
            workspace_path=workspace_path,
            session_env=session_env,
        )

        from jaato_sdk.events import (
            SendMessageRequest,
            PermissionResponseRequest,
            ClarificationResponseRequest,
            ClarificationBatchResponseEvent,
            ReferenceSelectionResponseRequest,
            StopRequest,
            CommandRequest,
            GetInstructionBudgetRequest,
            InstructionBudgetEvent,
            InjectPromptRequest,
            InjectPromptResultEvent,
            ReplayMessagesRequest,
            ReplayMessagesResultEvent,
            ResolveForkPointRequest,
            ResolveForkPointResultEvent,
            PermissionAddWhitelistRequest,
            PermissionAddBlacklistRequest,
            PermissionRemoveRequest,
            PermissionClearRequest,
            PermissionSetDefaultRequest,
            PermissionPolicySnapshotRequest,
            PermissionPolicySnapshotEvent,
        )

        if isinstance(event, SendMessageRequest):
            # Per-call parallel-tools override.  When the request
            # specifies parallel_tools (True/False), stash it on the
            # session so the next turn's tool-execution branch
            # consults it instead of the JAATO_PARALLEL_TOOLS env var.
            # The session clears the override after the turn.  None
            # leaves env-driven behaviour unchanged.
            if event.parallel_tools is not None:
                # Phase 3 §7c step 6.6.3.6: forward to runner-side
                # via the new ``session.set_parallel_tools_override``
                # RPC (§7c step 6.6.3.3 at commit b678ce2c) instead
                # of the private-attr write on the daemon-side
                # session.
                rpc = getattr(server, "_runner_rpc", None)
                if rpc is not None:
                    forwarder = getattr(
                        rpc,
                        "session_set_parallel_tools_override_threadsafe",
                        None,
                    )
                    if callable(forwarder):
                        try:
                            forwarder(event.parallel_tools, timeout=2.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "set_parallel_tools_override forward failed: %s",
                                exc,
                            )

            # Track user input for command history restoration
            if event.text and event.text.strip():
                session.user_inputs.append(event.text)
                session.is_dirty = True

            # ``%<name> --help`` is intercepted first: the help is
            # emitted straight to the client via a HelpTextEvent and the
            # reference is stripped so it never reaches the model.
            message_text = self._intercept_prompt_help_refs(
                event.text, server, client_id
            )

            # If the message was purely help requests, the user just
            # wanted documentation — don't dispatch anything to the
            # model.  The help was already delivered as a HelpTextEvent by
            # ``_intercept_prompt_help_refs``; persist the (already-appended)
            # user input and return early WITHOUT a model turn.
            if not (message_text and message_text.strip()):
                # Close the turn lifecycle even though no model turn ran.  A
                # client that waits for a per-message completion signal (WS /
                # chat renderers) would otherwise see zero further events and
                # trip its stall detector, killing the session — the observed
                # ``%name --help`` stall.  A synthetic TurnCompletedEvent
                # (finish_reason="stop") resets that timer and closes the turn;
                # it is targeted at the requesting client only (no model turn
                # exists to fan out to the whole session).
                if client_id is not None:
                    self._emit_to_client(client_id, TurnCompletedEvent())
                self._save_session(session)
                return

            # Server-side prompt skill expansion: expand %prompt-name
            # references so all clients (TUI, WS, IPC) get automatic
            # prompt content injection regardless of client capabilities.
            message_text = self._expand_prompt_references(
                message_text, server
            )

            # Capture context for thread (ContextVars don't propagate to threads)
            ctx_session_id = session_id
            ctx_client_id = client_id
            ctx_workspace = workspace_path
            ctx_session_env = session_env

            # Run in thread to not block
            def run_message():
                # Set logging context in thread
                set_logging_context(
                    session_id=ctx_session_id,
                    client_id=ctx_client_id,
                    workspace_path=ctx_workspace,
                    session_env=ctx_session_env,
                )
                # Ensure provider traces from the main agent go to the
                # base provider_trace.log (not a subagent-specific file).
                try:
                    from jaato_sdk.trace import set_trace_agent_context
                    set_trace_agent_context("main")
                except ImportError:
                    pass  # Older jaato_sdk without per-agent trace routing
                try:
                    server.send_message(
                        message_text,
                        event.attachments if event.attachments else None
                    )
                    # Auto-save after turn
                    self._save_session(session)
                finally:
                    clear_logging_context()

            threading.Thread(target=run_message, daemon=True).start()

        elif isinstance(event, PermissionResponseRequest):
            server.respond_to_permission(
                event.request_id, event.response,
                edited_arguments=event.edited_arguments,
            )

        elif isinstance(event, ClarificationResponseRequest):
            server.respond_to_clarification(event.request_id, event.response)

        elif isinstance(event, ClarificationBatchResponseEvent):
            server.respond_to_clarification_batch(
                event.request_id, event.answers, cancelled=event.cancelled,
            )

        elif isinstance(event, ReferenceSelectionResponseRequest):
            server.respond_to_reference_selection(event.request_id, event.response)

        elif isinstance(event, StopRequest):
            server.stop()

        elif isinstance(event, CommandRequest):
            # Intercept workspace command — it needs the monitor from
            # session manager, not from the plugin system.
            if event.command == "workspace":
                from .workspace_command import handle_workspace_command

                monitor = self._workspace_monitors.get(session_id)
                result = handle_workspace_command(monitor, event.args or [])
            else:
                result = server.execute_command(event.command, event.args)
            # Format result properly
            if isinstance(result, dict):
                if "_pager" in result:
                    # HelpLines result already emitted via HelpTextEvent, skip
                    pass
                elif "error" in result:
                    # Error result
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=result["error"],
                        style="error",
                    ))
                elif "result" in result:
                    # Simple result - show the text directly
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message=result["result"],
                        style="info",
                    ))
                else:
                    # Dict result with multiple keys - format each
                    lines = []
                    for key, value in result.items():
                        if not key.startswith('_'):
                            if isinstance(value, list):
                                # Format lists nicely
                                if value:
                                    lines.append(f"{key}:")
                                    for item in value:
                                        # Extract short name for model paths
                                        if isinstance(item, str) and '/' in item:
                                            item = item.split('/')[-1]
                                        lines.append(f"  • {item}")
                                else:
                                    lines.append(f"{key}: (none)")
                            else:
                                lines.append(f"{key}: {value}")
                    self._emit_to_client(client_id, SystemMessageEvent(
                        message="\n".join(lines) if lines else str(result),
                        style="info",
                    ))
            else:
                self._emit_to_client(client_id, SystemMessageEvent(
                    message=str(result),
                    style="info",
                ))

        elif isinstance(event, GetInstructionBudgetRequest):
            # Get instruction budget for the requested agent.  ``None`` (or
            # the legacy default ``"main"``) targets this server's main
            # agent, whose actual id may be a custom ``--agent <name>``.
            main_id = server.main_agent_id
            agent_id = event.agent_id or main_id

            if agent_id == main_id or agent_id == "main":
                # Main agent budget — Phase 3 §7c step 6.6.3.6:
                # forward to runner-side via the existing
                # ``session.snapshot_instruction_budget`` RPC (§7c
                # step 6.1 (2/3) at commit 1043bfde).
                snapshot = None
                rpc = getattr(server, "_runner_rpc", None)
                if rpc is not None:
                    snapshotter = getattr(
                        rpc,
                        "session_snapshot_instruction_budget_threadsafe",
                        None,
                    )
                    if callable(snapshotter):
                        try:
                            snapshot = snapshotter(timeout=5.0)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug(
                                "snapshot_instruction_budget forward failed: %s",
                                exc,
                            )
                if snapshot is not None:
                    self._emit_to_client(client_id, InstructionBudgetEvent(
                        agent_id=agent_id,
                        budget_snapshot=snapshot,
                    ))
                else:
                    self._emit_to_client(client_id, ErrorEvent(
                        error="No instruction budget available for main agent",
                        error_type="BudgetNotFound",
                    ))
            else:
                # Subagent budget from SubagentPlugin
                subagent_plugin = server.registry.get_plugin("subagent") if server.registry else None
                if subagent_plugin and hasattr(subagent_plugin, '_active_sessions'):
                    session_info = subagent_plugin._active_sessions.get(agent_id)
                    if session_info:
                        subagent_session = session_info.get('session')
                        if subagent_session and hasattr(subagent_session, 'instruction_budget') and subagent_session.instruction_budget:
                            self._emit_to_client(client_id, InstructionBudgetEvent(
                                agent_id=agent_id,
                                budget_snapshot=subagent_session.instruction_budget.snapshot(),
                            ))
                        else:
                            self._emit_to_client(client_id, ErrorEvent(
                                error=f"No instruction budget available for agent {agent_id}",
                                error_type="BudgetNotFound",
                            ))
                    else:
                        self._emit_to_client(client_id, ErrorEvent(
                            error=f"Agent not found: {agent_id}",
                            error_type="AgentNotFound",
                        ))
                else:
                    self._emit_to_client(client_id, ErrorEvent(
                        error=f"Subagent plugin not available",
                        error_type="PluginNotFound",
                    ))

        # ─── SDK feature parity — session-primitive verbs ───────────────
        # Typed WS verbs over JaatoSession's public primitives so SDK
        # consumers can drive inject_prompt / replay_messages /
        # resolve_fork_point without going through the model loop.
        # See ``project_backlog_sdk_feature_parity.md``.

        elif isinstance(event, InjectPromptRequest):
            # ROUTE THROUGH THE QUEUE-OR-DRIVE DECISION.
            #
            # This handler used to call ``session_inject_prompt_threadsafe``
            # directly -- bypassing ``inject_prompt_to_session``, and with it
            # ``shared.message_delivery.deliver`` -- so it made NO busy/idle
            # decision at all.  It could not drive a turn under any
            # circumstances, in any session state.  Since the runner-side
            # ``inject_prompt`` only starts a turn while a ``send_message``
            # RPC is in flight, an inject into an idle session queued into a
            # queue with no drainer, forever: the session became permanently
            # UNREACHABLE, and a watchdog's nudge-on-silence landed in the
            # same dead queue as the message it was sent to rescue.
            #
            # Trace evidence (perpetual-monologue cascade, runs 10 and 11,
            # session_20260825_232315): queue_size_after 1 -> 2 -> 3 across a
            # sibling message and two user nudges six minutes apart, never a
            # single pop.  This was the FOURTH copy of the queue-or-drive
            # decision and the only one that omitted it rather than getting
            # it wrong.
            from shared.message_delivery import DELIVERED
            from shared.message_queue import SourceType
            try:
                source_type = SourceType(event.source_type)
            except ValueError:
                self._emit_to_client(client_id, ErrorEvent(
                    error=(
                        f"Invalid source_type: {event.source_type!r}. "
                        f"Valid values: {[s.value for s in SourceType]}"
                    ),
                    error_type="ValidationError",
                    request_id=event.request_id,
                ))
                return

            status = self.deliver_prompt_to_session(
                session_id, event.text,
                source_id=event.source_id,
                source_type=source_type,
            )

            if event.request_id:
                # Protocol 1.3+ caller asked to be told.  A status is a
                # status -- failures ride the same event as successes, so a
                # caller waiting on one correlation id never waits on two
                # channels to learn one outcome.
                self._emit_to_client(client_id, InjectPromptResultEvent(
                    request_id=event.request_id,
                    status=status,
                ))
            elif status not in DELIVERED:
                # Pre-1.3 caller: no result channel, so a failure has to
                # surface as an error or it is silent.  Silence is the
                # expensive direction -- the caller assumes delivery and
                # stalls somewhere it cannot attribute.
                self._emit_to_client(client_id, ErrorEvent(
                    error=(
                        f"inject_prompt was not delivered (status={status}). "
                        f"Pass a request_id to receive an "
                        f"InjectPromptResultEvent instead."
                    ),
                    error_type="SessionError",
                ))

        elif isinstance(event, ReplayMessagesRequest):
            # Phase 3 §7c step 6.6.3.6: forward to runner-side via
            # the new ``session.replay_messages`` RPC (§7c step
            # 6.6.3.4 at commit 24ed6c0f) instead of reaching
            # into the daemon-side session.  When ``event.messages``
            # is None, the runner's handler falls back to the
            # session's history — but the wrapper requires a
            # messages list, so we read history first via the
            # existing ``session.get_history`` RPC (§3.3c precursor).
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is None:
                self._emit_to_client(client_id, ReplayMessagesResultEvent(
                    request_id=event.request_id,
                    error="No active JaatoSession",
                ))
            else:
                replayer = getattr(
                    rpc, "session_replay_messages_threadsafe", None,
                )
                if not callable(replayer):
                    self._emit_to_client(client_id, ReplayMessagesResultEvent(
                        request_id=event.request_id,
                        error="No active JaatoSession",
                    ))
                else:
                    # Resolve messages: use the request's if provided;
                    # else fall back to the runner-side session's
                    # current history via the existing get_history
                    # RPC.  Deserialize daemon-side first when the
                    # request supplied a list (the runner handler
                    # also does this; we deserialize early to surface
                    # malformed input as a typed daemon error).
                    if event.messages is not None:
                        from shared.plugins.session.serializer import deserialize_history
                        try:
                            messages = deserialize_history(event.messages)
                        except Exception as exc:  # noqa: BLE001
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error=f"deserialize failed: {exc}",
                            ))
                            return
                    else:
                        history_getter = getattr(
                            rpc, "session_get_history_threadsafe", None,
                        )
                        if not callable(history_getter):
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error="No active JaatoSession",
                            ))
                            return
                        try:
                            messages = history_getter(timeout=5.0)
                        except Exception as exc:  # noqa: BLE001
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error=f"get_history failed: {exc}",
                            ))
                            return

                    # Run in a worker thread — replay_messages
                    # blocks until the provider call completes.
                    def run_replay():
                        try:
                            response_text = replayer(
                                messages,
                                replay_timeout=event.timeout_seconds,
                                timeout=event.timeout_seconds + 60.0,
                            )
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                response_text=response_text,
                            ))
                        except Exception as exc:
                            self._emit_to_client(client_id, ReplayMessagesResultEvent(
                                request_id=event.request_id,
                                error=f"{type(exc).__name__}: {exc}",
                            ))
                    threading.Thread(target=run_replay, daemon=True).start()

        elif isinstance(event, ResolveForkPointRequest):
            # Phase 3 §7c step 6.6.3.6: forward to runner-side
            # via the new ``session.resolve_fork_point`` RPC (§7c
            # step 6.6.3.5 at commit e4eddc0e).  The runner-side
            # handler defaults ``history`` to its own
            # ``session.get_history()`` when omitted (matches the
            # pre-§7c daemon-side pattern at line 4573).
            rpc = getattr(server, "_runner_rpc", None)
            if rpc is None:
                self._emit_to_client(client_id, ResolveForkPointResultEvent(
                    request_id=event.request_id,
                    fork_index=-1,
                    error="No active JaatoSession",
                ))
            else:
                resolver = getattr(
                    rpc, "session_resolve_fork_point_threadsafe", None,
                )
                if not callable(resolver):
                    self._emit_to_client(client_id, ResolveForkPointResultEvent(
                        request_id=event.request_id,
                        fork_index=-1,
                        error="No active JaatoSession",
                    ))
                else:
                    try:
                        fork_index = resolver(
                            after_message=event.after_message,
                            after_tool_call=event.after_tool_call,
                            after_timestamp=event.after_timestamp,
                            timeout=5.0,
                        )
                        self._emit_to_client(client_id, ResolveForkPointResultEvent(
                            request_id=event.request_id,
                            fork_index=fork_index,
                        ))
                    except Exception as exc:
                        self._emit_to_client(client_id, ResolveForkPointResultEvent(
                            request_id=event.request_id,
                            fork_index=-1,
                            error=f"{type(exc).__name__}: {exc}",
                        ))

        # ─── SDK feature parity — permission policy verbs ───────────────
        # Typed verbs replacing stringly-typed CommandRequest("permissions",
        # [...]) for SDK consumers.  CLI command path stays for users.

        elif isinstance(event, PermissionAddWhitelistRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                if event.tools:
                    permission_plugin.add_whitelist_tools(event.tools)
                for pattern in event.patterns:
                    permission_plugin._policy.add_session_whitelist(pattern)

        elif isinstance(event, PermissionAddBlacklistRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                # PermissionPolicy tracks blacklist rules in a single
                # session_blacklist set — tools and patterns live in
                # the same set (the matcher checks exact match first,
                # then pattern match).  Both add through the same call.
                for tool in event.tools:
                    permission_plugin._policy.add_session_blacklist(tool)
                for pattern in event.patterns:
                    permission_plugin._policy.add_session_blacklist(pattern)

        elif isinstance(event, PermissionRemoveRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            elif event.target == "whitelist":
                # Direct set mutation — no remove method on the
                # policy, but the sets are public attributes.  discard
                # is a no-op for missing items so the call is idempotent.
                policy = permission_plugin._policy
                for tool in event.tools:
                    policy.whitelist_tools.discard(tool)
                    policy.session_whitelist.discard(tool)
                for pattern in event.patterns:
                    policy.session_whitelist.discard(pattern)
            elif event.target == "blacklist":
                policy = permission_plugin._policy
                for tool in event.tools:
                    policy.blacklist_tools.discard(tool)
                    policy.session_blacklist.discard(tool)
                for pattern in event.patterns:
                    policy.session_blacklist.discard(pattern)
            else:
                self._emit_to_client(client_id, ErrorEvent(
                    error=(
                        f"Invalid target: {event.target!r}. "
                        f"Valid values: 'whitelist', 'blacklist'"
                    ),
                    error_type="ValidationError",
                ))

        elif isinstance(event, PermissionClearRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                policy = permission_plugin._policy
                # Clear targets the SESSION-level overrides only
                # (matches the semantics of `permissions clear`).
                # Base policy from permissions.json is unaffected.
                if event.target in ("whitelist", "all"):
                    policy.session_whitelist.clear()
                if event.target in ("blacklist", "all"):
                    policy.session_blacklist.clear()
                if event.target == "all":
                    policy.session_default_policy = None

        elif isinstance(event, PermissionSetDefaultRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, ErrorEvent(
                    error="Permission plugin not available",
                    error_type="PluginNotFound",
                ))
            else:
                try:
                    permission_plugin._policy.set_session_default_policy(
                        event.policy
                    )
                except ValueError as exc:
                    self._emit_to_client(client_id, ErrorEvent(
                        error=str(exc),
                        error_type="ValidationError",
                    ))

        elif isinstance(event, PermissionPolicySnapshotRequest):
            permission_plugin = (
                server.registry.get_plugin("permission")
                if server.registry else None
            )
            if permission_plugin is None or permission_plugin._policy is None:
                self._emit_to_client(client_id, PermissionPolicySnapshotEvent(
                    request_id=event.request_id,
                    default_policy="ask",
                ))
            else:
                policy = permission_plugin._policy
                self._emit_to_client(client_id, PermissionPolicySnapshotEvent(
                    request_id=event.request_id,
                    default_policy=policy.default_policy,
                    session_default_policy=policy.session_default_policy,
                    whitelist_tools=sorted(policy.whitelist_tools),
                    whitelist_patterns=list(policy.whitelist_patterns),
                    blacklist_tools=sorted(policy.blacklist_tools),
                    blacklist_patterns=list(policy.blacklist_patterns),
                    session_whitelist=sorted(policy.session_whitelist),
                    session_blacklist=sorted(policy.session_blacklist),
                ))

        else:
            self._emit_to_client(client_id, ErrorEvent(
                error=f"Unknown request type: {type(event).__name__}",
                error_type="RequestError",
            ))

    # =========================================================================
    # Cleanup
    # =========================================================================

    def save_all(self) -> int:
        """Save all dirty sessions to disk.

        Returns:
            Number of sessions saved.
        """
        with self._lock:
            # Snapshot under the lock, save outside it.  Iterating the dict
            # is what the lock is for; the save round-trips to the loop and
            # must not be done while holding it.
            dirty = [s for s in self._sessions.values() if s.is_dirty]
        return sum(1 for session in dirty if self._save_session(session))

    # ------------------------------------------------------------------
    # Ephemeral sessions (Phase 3 — remote subagent delegation)
    # ------------------------------------------------------------------

    def run_ephemeral_session(self, *args: Any, **kwargs: Any) -> str:
        """Run an ephemeral session for a remote subagent (server 0.6.71+ entry).

        Wraps :meth:`_run_ephemeral_session_impl` in a fresh ContextVar
        context via :func:`shared.session_context.run_in_fresh_session_context`
        so the bootstrap is isolated from any ContextVar values
        inherited from the caller's task.  See the helper's docstring
        for the rationale.
        """
        from shared.session_context import run_in_fresh_session_context
        return run_in_fresh_session_context(
            self._run_ephemeral_session_impl, *args, **kwargs,
        )

    def _run_ephemeral_session_impl(
        self,
        profile_json: str,
        inline_config_json: str,
        prompt: str,
        agent_name: str,
        on_output: Any,
        workspace_path: Optional[str] = None,
        on_started: Any = None,
    ) -> str:
        """Implementation of ephemeral session run, called via fresh-context wrap.

        See :meth:`run_ephemeral_session` for the isolation rationale.

        This is a blocking call intended to be run from a background thread
        (via ``asyncio.to_thread``).  The session is not persisted to disk
        and is not visible in the session list.

        When ``workspace_path`` is provided (Phase 5 workspace replication),
        the ephemeral session runs with that directory as its working
        directory, and ``JAATO_WORKSPACE_ROOT`` is set accordingly.  The
        server's plugin registry also gets the workspace path so plugins
        can discover project files.

        Args:
            profile_json: JSON-serialized SubagentProfile from the origin.
            inline_config_json: JSON-serialized inline config (empty if profile-based).
            prompt: The full prompt to send.
            agent_name: Display name for the subagent.
            on_output: Callback ``(source: str, text: str, mode: str) -> None``
                invoked for each output chunk.
            workspace_path: Optional workspace directory for the session.
                When provided, CWD and ``JAATO_WORKSPACE_ROOT`` are set to
                this path for the duration of the session.

        Returns:
            Summary string from the model's final response.
        """
        import json as _json
        import uuid

        # Parse profile/config
        profile_data = _json.loads(profile_json) if profile_json else {}
        inline_data = _json.loads(inline_config_json) if inline_config_json else {}

        # Phase 3 §3.12 ephemeral migration: route through the unified
        # ``_construct_and_initialize_server`` sub-helper.  Compose the
        # ephemeral inputs (model/provider/plugins/system_instructions
        # /max_turns) into a single inline ``SubagentProfile`` so the
        # construction shape matches the IPC + disk-restore paths
        # (env_file-driven JaatoServer construction with a profile
        # override) rather than the pre-§3.12 broken
        # ``JaatoServer().initialize(model=, provider_name=, tools=)``
        # signature (the underlying ``JaatoServer.initialize`` takes
        # no kwargs — that call would have raised TypeError; the
        # ephemeral path was effectively unreachable in production).
        merged: Dict[str, Any] = {}
        for source in (profile_data, inline_data):
            for key, value in source.items():
                merged.setdefault(key, value)
        import queue
        import threading

        # Spike (§7c remote-spawn repair): route the ephemeral run through the
        # PROVEN create_headless_session path instead of constructing an
        # in-daemon JaatoServer.  This gives the delegated agent a real
        # confined runner + AppArmor (the old client_id=None path ran it
        # UNCONFINED in the daemon) and fixes the post-seat-flip TypeError
        # (JaatoServer.send_message no longer takes on_output).  Output is
        # sourced from the EventBus via an in-process cascade client keyed by a
        # per-spawn cascade_driver_id; a forwarder thread relays it to
        # ``on_output`` PER-EMIT so streaming granularity is preserved WITHOUT
        # calling on_output under SessionManager._lock (the event callback runs
        # under the lock and must stay trivial).  Workspace is per-session via
        # create_headless_session(workspace_path=...) — no global chdir /
        # JAATO_WORKSPACE_ROOT mutation (the old path raced sibling sessions by
        # mutating daemon-global cwd).
        from jaato_sdk.events import AgentOutputEvent, SessionTerminatedEvent, TurnCompletedEvent

        # Source taxonomy (locked with gossip): relay ONLY model output into
        # the forwarded stream.  Drop the prompt echo (source="user"),
        # tool/status/lifecycle, and thinking (CoT is an origin opt-in, never
        # forwarded by default across a federation boundary).  Fail-closed: any
        # unlisted source is dropped, never relayed.
        relay_sources = frozenset({"model"})

        cid = f"ephemeral-{uuid.uuid4().hex[:12]}"
        in_process_client_id = f"_ephemeral:{cid}"
        try:
            timeout_s = float(os.environ.get("JAATO_EPHEMERAL_TIMEOUT_S", "1800"))  # env: seconds an ephemeral relay session may run before the daemon gives up (default 1800)
        except ValueError:
            timeout_s = 1800.0

        out_q: "queue.Queue[Any]" = queue.Queue()
        collected: List[str] = []
        terminal: Dict[str, Any] = {}
        done = threading.Event()
        sentinel = object()

        def _on_event(event: Any) -> None:
            # Runs synchronously on the daemon thread under
            # SessionManager._lock — MUST stay trivial: enqueue / signal only;
            # never call on_output (network-bound) or re-enter SessionManager.
            if isinstance(event, AgentOutputEvent):
                if event.source in relay_sources:
                    out_q.put((event.source, event.text, event.mode))
            elif isinstance(event, SessionTerminatedEvent):
                terminal["reason"] = event.reason
                terminal["error_type"] = event.error_type
                terminal["error_summary"] = event.error_summary
                out_q.put(sentinel)
                done.set()
            elif isinstance(event, TurnCompletedEvent):
                # A single-shot ephemeral runs exactly one turn (one
                # prompt -> one turn, including all tool iterations).  A
                # NATURAL success emits turn.completed and then goes IDLE:
                # a headless session does NOT self-terminate after its
                # prompt, so NO SessionTerminatedEvent fires on the happy
                # path.  Without treating turn.completed as the terminal,
                # done.wait() blocks until the 1800s timeout and
                # execute_spawn never sends PeerAgentCompleted (origin gets
                # the full output stream but never the completion).
                # turn.completed fires AFTER all of the turn's
                # agent.output, so the sentinel lands behind every relayed
                # chunk.  Errors still arrive as SessionTerminatedEvent
                # (error) above; setdefault lets a real error reason win if
                # it somehow raced in first.
                terminal.setdefault("reason", "natural")
                out_q.put(sentinel)
                done.set()

        def _forward() -> None:
            # Drains off-lock and relays each chunk per-emit (live streaming).
            while True:
                item = out_q.get()
                if item is sentinel:
                    return
                src, text, mode = item
                collected.append(text)
                if on_output:
                    try:
                        on_output(src, text, mode)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "ephemeral %s: on_output relay raised", cid,
                        )

        forwarder = threading.Thread(
            target=_forward, name=f"ephemeral-fwd-{cid}", daemon=True,
        )

        session_id = ""
        try:
            self.register_in_process_client(
                client_id=in_process_client_id,
                callback=_on_event,
                cascade_driver_id=cid,
                role="owner",
            )
            forwarder.start()

            # An inline ephemeral spawn (e.g. gossip remote-spawn) carries no
            # workspace — there's no git/workspace replication for an
            # inline-config subagent.  But the runner needs a cwd to spawn:
            # without a workspace, ``_provision_ipc_apparmor_and_spawn_runner``
            # skips the runner spawn, ``server._runner_rpc`` stays None, and
            # the first model turn crashes with
            # ``NoneType.session_send_message_threadsafe`` (the §7c seat-flip
            # path forwards send_message to the runner).  Provision a scratch
            # workspace so a runner spawns; the ephemeral session is
            # short-lived and writes nothing meaningful there.
            if not workspace_path:
                import tempfile
                workspace_path = tempfile.mkdtemp(prefix="jaato-ephemeral-")
                logger.info(
                    "ephemeral %s: no workspace supplied (inline spawn); "
                    "provisioned scratch workspace %s so a runner can spawn",
                    cid, workspace_path,
                )

            # The runner's core plugins (canonical case: file_edit's
            # backup manager) require a config_root.  The auto
            # "<workspace>/.jaato" fallback was removed (PR-147), so it
            # must be set explicitly or those plugins fail to expose with
            # a loud RuntimeError traceback on every ephemeral spawn
            # (non-fatal, but it masquerades as a failure in logs).
            # Mirror normal-session semantics: derive config_root from
            # whatever workspace this ephemeral session runs in (scratch
            # or caller-supplied).  Deterministic, no hardcoded fallback.
            ephemeral_config_root = str(pathlib.Path(workspace_path) / ".jaato")

            # For an inline-config spawn the caller's ``agent_name`` is a
            # human-readable DISPLAY label (e.g. a gossip remote-spawn's
            # "remote-subagent") — NOT a ``.jaato/agents/<name>.md`` persona.
            # The inline profile IS the complete config, so there is no
            # persona to resolve.  Passing the label as ``agent_name`` makes
            # ``_create_session_impl`` run ``_resolve_agent(label)``, which
            # fails (no such file) and returns an empty session_id — the
            # remote-leg empty-init bug surfaced in gossip co-validation.
            # Route the label to ``session_name`` for identity and pass
            # ``agent_name=None`` when an inline profile is supplied so agent
            # resolution is skipped; real-agent spawns (no inline profile)
            # keep their ``agent_name`` resolution intact.
            _display_name = agent_name or "remote-subagent"
            session_id = self.create_headless_session(
                agent_name=None if merged else (agent_name or "main"),
                session_name=_display_name,
                workspace_path=workspace_path,
                initial_prompt=prompt,
                inline_profile_data=merged,
                config_root=ephemeral_config_root,
                cascade_driver_id=cid,
            )
            if not session_id:
                out_q.put(sentinel)
                forwarder.join(timeout=5)
                return "Ephemeral session initialization failed."

            # STOP (ii) hook: hand the real session_id to the caller BEFORE
            # blocking so gossip can map request_id -> session_id and route a
            # PeerStopRequest to the live session.  Optional — callers that
            # don't pass on_started keep the pure blocking contract.
            if on_started is not None:
                try:
                    on_started(session_id)
                except Exception:  # noqa: BLE001
                    logger.exception("ephemeral %s: on_started raised", cid)

            # Block on the terminal.  Two paths set ``done``: a NATURAL
            # single-shot success arrives as turn.completed (a headless
            # session goes idle, NOT terminated, after its one prompt),
            # while error/stop winds down via SessionTerminatedEvent.
            # ``terminal['reason']`` classifies it (default "natural").
            finished = done.wait(timeout=timeout_s)
            out_q.put(sentinel)  # drain + stop the forwarder even on timeout
            forwarder.join(timeout=5)

            if not finished:
                raise RuntimeError(
                    f"Ephemeral session {session_id} did not reach a terminal "
                    f"event within {timeout_s}s"
                )

            # Failure surfacing (locked contract with gossip): raise on any
            # non-clean terminal, return the collected string on natural
            # completion.  gossip leans on a raised exception from its
            # to_thread call to emit PeerAgentCompletedEvent(success=False).
            reason = terminal.get("reason")
            if reason and reason != "natural":
                detail = (
                    f" ({terminal.get('error_type')}: "
                    f"{terminal.get('error_summary')})"
                    if reason == "error" else ""
                )
                raise RuntimeError(
                    f"Ephemeral session {session_id} terminated "
                    f"reason={reason}{detail}"
                )
            return "".join(collected) or "Task completed."
        finally:
            try:
                self.unregister_cascade_client(cid, in_process_client_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "ephemeral %s: unregister_cascade_client raised", cid,
                )
            out_q.put(sentinel)  # belt-and-suspenders forwarder stop
            if session_id:
                try:
                    self.delete_session(session_id)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "ephemeral %s: delete_session(%s) raised",
                        cid, session_id,
                    )

    def shutdown(self) -> None:
        """Shutdown all sessions, saving to disk first."""
        logger.info("SessionManager shutting down...")

        # Stop all workspace monitors
        for sid in list(self._workspace_monitors):
            self._stop_workspace_monitor(sid)

        with self._lock:
            # Snapshot under the lock; BOTH the save and the server shutdown
            # below round-trip to the daemon loop, and holding the manager
            # lock across either is what lets a worker wait for the loop while
            # the loop waits for the lock.
            closing = list(self._sessions.values())

        for session in closing:
            self._save_session(session)
            session.server.shutdown()

        with self._lock:
            self._sessions.clear()
            self._client_to_session.clear()

        # Close all session log handlers
        handler = get_session_handler()
        if handler:
            handler.close()

        # Stop every per-session egress proxy (Phase 5 §5.11).
        try:
            from server.egress_proxy import wireup as _egress_wireup
            _egress_wireup.shutdown_all()
        except Exception:  # pragma: no cover - defensive
            logger.warning("egress proxy shutdown_all failed", exc_info=True)

        self._session_plugin.shutdown()
        logger.info("SessionManager shutdown complete")
