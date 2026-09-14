"""Per-session runtime resource caps — the value type plumbed through profiles.

Lives in ``shared/`` (rather than ``server/``) because :class:`SubagentProfile`
references it statically and ``shared/`` cannot import from ``server/``.
The cgroups runtime in :mod:`server.cgroups` imports this dataclass back
out to consume it; profile JSON loaders construct it via
:meth:`RuntimeLimits.from_dict`.

Why this is a separate module
-----------------------------

Two things need this type:

* ``SubagentProfile.runtime_limits`` (in
  ``shared/plugins/subagent/config.py``) — for static typing on the
  profile field, equivalent to how ``GCProfileConfig`` is co-located
  with the profile.
* :class:`server.cgroups.CgroupsManager.provision_cgroup` —
  the kernel-side consumer that writes the kernel-enforced subset of
  these limits to cgroup v2 controller files.

If this dataclass lived in ``server.cgroups``, ``shared`` would have to
import server, which is a layering violation; if it lived in
``subagent/config.py``, the cgroups runtime would have to import a
plugin module.  A free-standing module sidesteps both.

Design notes
------------

The dataclass is **frozen + validated in __post_init__** so an invalid
profile fails at *parse* time, not when a session tries to launch its
first subprocess.  The two-layer split (kernel-enforced vs
application-enforced) is documented on the class itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional


# cgroup v2 cpu.weight bounds (kernel: include/uapi/linux/cgroupv2.h).
# Kept here because validation runs at config-construction time, before
# any cgroup machinery is touched.
_CPU_WEIGHT_MIN = 1
_CPU_WEIGHT_MAX = 10_000

# Sanity ceilings for user-supplied values.  Not security boundaries —
# just guardrails to catch obvious typos in profile JSON ("1024" GB
# instead of MB, etc.) at load time rather than at session-start time.
_MEMORY_MAX_MB_LIMIT = 1024 * 1024  # 1 TiB
_PIDS_MAX_LIMIT = 1_000_000

# Concurrency width the framework uses when nothing declares one.  This
# is the value ``jaato_session`` capped its two thread pools at as a bare
# literal before #862 made it configurable, so an unconfigured tree keeps
# byte-identical behaviour.
DEFAULT_MAX_PARALLEL_TOOLS = 8

# Sanity ceiling for ``max_parallel_tools``.  Not a security boundary —
# the number of tool calls a model emits in one turn is already small;
# a profile asking for 4096 workers has a typo, not a workload.
_MAX_PARALLEL_TOOLS_LIMIT = 256

# How long a session may keep running with NO consumer at all -- no attached
# client, no headless marker, nothing -- before the daemon stops it (#812).
#
# This is the ONE field in this block that carries a framework default, and
# the reason is the incident it comes from: a session whose client died kept
# executing tools for seven minutes and spent $2.52, and the only thing that
# eventually stopped it was a ``budget_control`` ceiling its profile happened
# to declare.  A profile that declared none had nothing at all.  A bound that
# must be declared to exist would have left that session running just as long,
# so the default is what makes the answer to "what stops an unattended
# session?" independent of what its author remembered to write.
#
# 900s rather than something tight: the sweep cannot distinguish "the harness
# crashed" from "the harness is being restarted", and the cost of being wrong
# is destroyed work, while the cost of being slow is bounded spend.  An
# operator who wants it tighter declares ``max_orphan_seconds`` and gets it;
# one who genuinely runs unattended-forever sessions declares 0.
DEFAULT_MAX_ORPHAN_SECONDS = 900.0

# The value both wall-clock fields read as "explicitly unbounded".  Zero
# rather than a string because these are numeric deadlines, and 0-disables is
# already this tree's spelling for one (``JAATO_GC_MEDIA_BYTES``,
# ``gc.media_bytes_threshold``, the three OpenRouter timeouts).
UNBOUNDED_SECONDS = 0


def _positive_int(
    name: str,
    value: Any,
    *,
    ceiling: Optional[int] = None,
    ceiling_unit: str = "",
    ceiling_note: str = "",
) -> None:
    """Reject anything that is not a positive ``int`` below *ceiling*.

    ``bool`` is excluded explicitly: it is an ``int`` subclass, so
    ``max_parallel_tools: true`` would otherwise validate and silently
    mean "one worker" — which is not what an author writing a bool meant.

    Args:
        name: Field name, for the message.
        value: The declared value; ``None`` (unset) always passes.
        ceiling: Optional sanity ceiling — a guardrail against typos, not
            a security boundary.
        ceiling_unit: Unit to print after the ceiling (e.g. ``"MiB"``).
        ceiling_note: Why the ceiling exists, appended to the message.

    Raises:
        ValueError: With the field name and the offending value.
    """
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    if ceiling is not None and value > ceiling:
        unit = f" {ceiling_unit}" if ceiling_unit else ""
        note = f" — {ceiling_note}" if ceiling_note else ""
        raise ValueError(
            f"{name}={value} exceeds sanity ceiling {ceiling}{unit}{note}"
        )


def _positive_number(name: str, value: Any) -> None:
    """Reject anything that is not a positive ``int`` or ``float``.

    Args:
        name: Field name, for the message.
        value: The declared value; ``None`` (unset) always passes.

    Raises:
        ValueError: With the field name and the offending value.
    """
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"{name} must be a number, got {type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _non_negative_number(name: str, value: Any) -> None:
    """Reject anything that is not a non-negative ``int`` or ``float``.

    The sibling of :func:`_positive_number` for the two wall-clock bounds,
    which accept ``0`` as "explicitly unbounded" -- so 0 must validate here
    where it would be rejected there.  ``bool`` is excluded for the reason
    given on :func:`_positive_int`.

    Args:
        name: Field name, for the message.
        value: The declared value; ``None`` (unset) always passes.

    Raises:
        ValueError: With the field name and the offending value.
    """
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"{name} must be a number, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(
            f"{name} must be >= 0, got {value} (0 means explicitly unbounded)"
        )


def _in_range(name: str, value: Any, low: int, high: int) -> None:
    """Reject anything that is not an ``int`` within ``[low, high]``.

    Args:
        name: Field name, for the message.
        value: The declared value; ``None`` (unset) always passes.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Raises:
        ValueError: With the field name and the offending value.
    """
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be int, got {type(value).__name__}")
    if not low <= value <= high:
        raise ValueError(f"{name}={value} out of range [{low}, {high}]")


@dataclass(frozen=True)
class RuntimeLimits:
    """Per-session resource consumption caps.

    Top-level field on a session profile (``runtime_limits``,
    parallel to ``gc``).  Answers the question "how much can this
    session consume?" — orthogonal to *sandboxing* (AppArmor), which
    answers "what can it touch?".

    Any field left as ``None`` means "no limit / inherit host default".
    The fields split into two enforcement layers, but a profile author
    treats them as one knob set:

    * **Kernel-enforced** (cgroup v2) — written once into the cgroup
      controller files; the kernel enforces them for every process in
      the slice for the lifetime of the session:
      ``memory_max_mb`` → ``memory.max``,
      ``pids_max`` → ``pids.max``,
      ``cpu_weight`` → ``cpu.weight``.
    * **Application-enforced** — read by the CLI / interactive_shell
      plugins and applied per-tool-call at the Python layer because
      cgroup v2 has no kernel knob for them:
      ``tool_timeout_seconds`` (passed to ``subprocess.run(timeout=)``),
      ``max_output_bytes`` (truncates captured stdout/stderr),
      ``max_parallel_tools`` (width of the session's tool thread pool).

    Single config, two layers — a profile author writing JSON shouldn't
    need to know which limits happen in the kernel vs in Python.

    ``max_parallel_tools`` is the odd one out in *who* reads it: the
    other application-enforced caps are consumed by the subprocess
    plugins, this one by :class:`shared.jaato_session.JaatoSession`
    itself, which owns the thread pool.  It lives here anyway because it
    answers the same question the block exists for — how much may this
    session consume at once — and because the ceilings it has to respect
    are its neighbours: eight simultaneous ``cli`` subprocesses under a
    small ``pids_max`` hit the cgroup limit non-deterministically, and
    eight simultaneous calls into a rate-limited service are the wrong
    shape whatever the memory ceiling says.  Before #862 the only lever
    was ``JAATO_PARALLEL_TOOLS``, which turns parallelism off entirely.
    """

    memory_max_mb: Optional[int] = None
    pids_max: Optional[int] = None
    cpu_weight: Optional[int] = None
    tool_timeout_seconds: Optional[float] = None
    max_output_bytes: Optional[int] = None
    # ``None`` means "no profile said anything" and the session applies
    # :data:`DEFAULT_MAX_PARALLEL_TOOLS`.  1 is a legitimate value and is
    # NOT the same as ``JAATO_PARALLEL_TOOLS=false``: the pool still runs
    # (single-worker), so the parallel path's ordering, hooks and
    # cancellation semantics are unchanged.
    max_parallel_tools: Optional[int] = None
    # Wall-clock ceilings, enforced DAEMON-SIDE by the session-lifetime
    # watchdog (#812).  The odd ones out in WHERE they are enforced: every
    # other field here is applied inside the session (by the kernel, by a
    # subprocess plugin, or by ``JaatoSession``), and these two are applied
    # by the ``SessionManager`` that owns the session from the outside.
    #
    # That placement is the whole point.  The ceilings that already existed
    # were all held by something the session could outlive -- the eval
    # harness's ``--arm-timeout`` lived in the client process that died, and
    # the task pool's ``seconds`` is reconciled when a session ENDS, so a
    # session that never ends never consumes it.  A bound held by the daemon
    # is held by the one process that is still there.
    #
    # ``None`` = nothing declared (``max_orphan_seconds`` then takes
    # :data:`DEFAULT_MAX_ORPHAN_SECONDS`; ``max_session_seconds`` is
    # unbounded).  ``0`` = explicitly unbounded, see
    # :data:`UNBOUNDED_SECONDS`.
    #
    #: Total wall-clock a session may stay LOADED, attended or not.  Opt-in
    #: and unbounded by default: an interactive TUI session left open over a
    #: lunch break is not a defect, and a default here would kill it.
    max_session_seconds: Optional[float] = None
    #: Wall-clock a session may keep running with NO consumer -- no attached
    #: client and not even the synthetic headless marker a woken or
    #: cascade-driven session carries.  Defaulted, because the session this
    #: field exists for is precisely the one whose profile declared nothing.
    max_orphan_seconds: Optional[float] = None

    # Future-proof: forward-compat passthrough for fields the runtime
    # doesn't recognise yet.  Profile schema validation should reject
    # truly unknown keys; this is just a safe landing pad if a newer
    # profile is loaded by an older server.
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate every declared field, at PARSE time.

        Each field delegates to one of the three checkers below, so the
        method stays a table of what-is-checked rather than a wall of
        inline branches (and so a new field is one line, not five).
        """
        _positive_int("memory_max_mb", self.memory_max_mb,
                      ceiling=_MEMORY_MAX_MB_LIMIT, ceiling_unit="MiB",
                      ceiling_note="likely a unit typo")
        _positive_int("pids_max", self.pids_max, ceiling=_PIDS_MAX_LIMIT)
        _in_range("cpu_weight", self.cpu_weight,
                  _CPU_WEIGHT_MIN, _CPU_WEIGHT_MAX)
        _positive_number("tool_timeout_seconds", self.tool_timeout_seconds)
        _positive_int("max_output_bytes", self.max_output_bytes)
        _positive_int("max_parallel_tools", self.max_parallel_tools,
                      ceiling=_MAX_PARALLEL_TOOLS_LIMIT,
                      ceiling_note="a model emits a handful of calls per "
                                   "turn, not hundreds")
        _non_negative_number("max_session_seconds", self.max_session_seconds)
        _non_negative_number("max_orphan_seconds", self.max_orphan_seconds)

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "RuntimeLimits":
        """Build limits from a profile dict, parking unknown keys in ``extra``.

        Returns the default (no-limits) instance when ``data`` is
        ``None`` or empty, so callers don't need to special-case
        missing config.
        """
        if not data:
            return cls()
        known_fields = {"memory_max_mb", "pids_max", "cpu_weight",
                        "tool_timeout_seconds", "max_output_bytes",
                        "max_parallel_tools", "max_session_seconds",
                        "max_orphan_seconds"}
        kwargs: Dict[str, Any] = {k: data[k] for k in known_fields if k in data}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        return cls(extra=extra, **kwargs)

    def has_kernel_limits(self) -> bool:
        """True if any cgroup-enforced limit is set.

        When this returns ``False``, :meth:`CgroupsManager.provision_cgroup`
        short-circuits without creating a cgroup directory at all — the
        session runs with the host's default limits and no bookkeeping
        cost.
        """
        return any(v is not None for v in (
            self.memory_max_mb, self.pids_max, self.cpu_weight,
        ))


class ConfinementUnavailableError(RuntimeError):
    """A profile requiring kernel confinement was used to create an in-process
    session (shared runtime, no runner subprocess), which cannot apply it.

    Raised fail-closed instead of silently running unconfined: kernel limits
    (and AppArmor) are applied at a runner subprocess's ``fork()/exec()``.  An
    in-process session is objects in the parent process — there is no
    subprocess boundary to confine, so its profile's ``runtime_limits`` would
    otherwise be silently ignored.
    """


def profile_requires_kernel_confinement(profile: Any) -> bool:
    """True if ``profile`` declares cgroup-enforced ``runtime_limits``.

    This is the unambiguous, profile-level kernel-confinement signal.  (AppArmor
    is a client-level opt-in, orthogonal to the profile, so it is not consulted
    here.)
    """
    limits = getattr(profile, "runtime_limits", None)
    return bool(limits is not None and limits.has_kernel_limits())


def assert_inprocess_can_honor(profile: Any) -> None:
    """Fail closed if an IN-PROCESS session is being created from a profile that
    requires kernel confinement — see :class:`ConfinementUnavailableError`."""
    if profile_requires_kernel_confinement(profile):
        name = getattr(profile, "name", "?")
        raise ConfinementUnavailableError(
            f"profile {name!r} declares kernel runtime_limits (memory/pids/cpu) "
            f"but is being created IN-PROCESS (shared runtime, no runner "
            f"subprocess), which cannot enforce them. Spawn it as an isolated "
            f"runner, or remove runtime_limits for in-process use — do not run "
            f"it silently unconfined."
        )


# Phase 5 §5.1: default `RuntimeLimits` applied to subagents spawned with
# ``agent_params.isolated=true`` whenever the profile omits the
# corresponding field.  The opt-in establishes the "isolation implies
# bounds" invariant; without a default, a profile that forgot to declare
# ``runtime_limits`` would silently skip cgroup provision and inherit the
# daemon's default cgroup (no caps) — the documented Phase 4 §4.3.9 item 1
# hardening gap.
#
# Values chosen for typical LLM-driven subagent workloads:
# * 2 GiB memory — comfortable for tool-running subagents; OOM-kills runaways.
# * 128 pids — generous for shell/cli workloads; rejects fork-bomb classes.
# * cpu.weight=100 — cgroup v2 default fair-share weight.
# * 120s tool timeout — conservative wall-clock cap for individual subprocesses.
# * 1 MiB output cap — prevents chatty tools from saturating the wire.
# * max_parallel_tools deliberately UNSET — the framework default (8) is
#   already comfortable under pids_max=128, and pinning it here would
#   make an isolated subagent's concurrency independent of the pids
#   ceiling an operator tightens.  A profile that wants a narrower pool
#   declares one; the default then applies as it does everywhere else.
#
# See ``docs/design/phase5_5_1_isolated_default_runtime_limits_audit.md``
# for the per-field rationale and merge semantics.
ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS = RuntimeLimits(
    memory_max_mb=2048,
    pids_max=128,
    cpu_weight=100,
    tool_timeout_seconds=120.0,
    max_output_bytes=1_048_576,
)


def apply_isolated_defaults(
    supplied: Optional[RuntimeLimits],
) -> RuntimeLimits:
    """Merge *supplied* with :data:`ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS`.

    Per-field semantics: when *supplied* sets a field, that value wins;
    when *supplied* leaves a field as ``None`` (or is itself ``None``),
    the default fills in.  The ``extra`` forward-compat dict is taken
    from *supplied* verbatim — defaults don't contribute unknown keys.

    Used by :meth:`SessionManager._spawn_isolated_runner` to compute the
    effective `RuntimeLimits` for an isolated subagent before provisioning
    its sub-cgroup and forwarding app-layer caps to the runner subprocess.

    Returns a fresh :class:`RuntimeLimits` instance — never returns the
    module-level default object directly, so the caller can't mutate the
    shared default through field-by-field assignment.  (The dataclass is
    frozen anyway, but defensive copying preserves the invariant under
    future-frozen-removal scenarios.)

    The per-field walk is driven by ``dataclasses.fields`` rather than
    written out by hand: a hand-written constructor call silently DROPS
    any field added to :class:`RuntimeLimits` afterwards, so an isolated
    subagent would lose a cap its own profile declared and no test of the
    new field would notice (it was the isolated path that lost it, not
    the field).  ``max_parallel_tools`` (#862) was the first field added
    after this function existed.
    """
    if supplied is None:
        supplied = RuntimeLimits()
    default = ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
    merged: Dict[str, Any] = {}
    for f in fields(RuntimeLimits):
        if f.name == "extra":
            continue
        value = getattr(supplied, f.name)
        merged[f.name] = (
            value if value is not None else getattr(default, f.name)
        )
    return RuntimeLimits(extra=dict(supplied.extra), **merged)
