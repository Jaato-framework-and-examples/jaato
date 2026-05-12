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

from dataclasses import dataclass, field
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
      ``max_output_bytes`` (truncates captured stdout/stderr).

    Single config, two layers — a profile author writing JSON shouldn't
    need to know which limits happen in the kernel vs in Python.
    """

    memory_max_mb: Optional[int] = None
    pids_max: Optional[int] = None
    cpu_weight: Optional[int] = None
    tool_timeout_seconds: Optional[float] = None
    max_output_bytes: Optional[int] = None

    # Future-proof: forward-compat passthrough for fields the runtime
    # doesn't recognise yet.  Profile schema validation should reject
    # truly unknown keys; this is just a safe landing pad if a newer
    # profile is loaded by an older server.
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.memory_max_mb is not None:
            if not isinstance(self.memory_max_mb, int) or self.memory_max_mb <= 0:
                raise ValueError(
                    f"memory_max_mb must be a positive int, got {self.memory_max_mb!r}"
                )
            if self.memory_max_mb > _MEMORY_MAX_MB_LIMIT:
                raise ValueError(
                    f"memory_max_mb={self.memory_max_mb} exceeds sanity ceiling "
                    f"{_MEMORY_MAX_MB_LIMIT} MiB — likely a unit typo"
                )
        if self.pids_max is not None:
            if not isinstance(self.pids_max, int) or self.pids_max <= 0:
                raise ValueError(
                    f"pids_max must be a positive int, got {self.pids_max!r}"
                )
            if self.pids_max > _PIDS_MAX_LIMIT:
                raise ValueError(
                    f"pids_max={self.pids_max} exceeds sanity ceiling {_PIDS_MAX_LIMIT}"
                )
        if self.cpu_weight is not None:
            if not isinstance(self.cpu_weight, int):
                raise ValueError(
                    f"cpu_weight must be int, got {type(self.cpu_weight).__name__}"
                )
            if not _CPU_WEIGHT_MIN <= self.cpu_weight <= _CPU_WEIGHT_MAX:
                raise ValueError(
                    f"cpu_weight={self.cpu_weight} out of range "
                    f"[{_CPU_WEIGHT_MIN}, {_CPU_WEIGHT_MAX}]"
                )
        if self.tool_timeout_seconds is not None:
            if not isinstance(self.tool_timeout_seconds, (int, float)):
                raise ValueError(
                    f"tool_timeout_seconds must be a number, got "
                    f"{type(self.tool_timeout_seconds).__name__}"
                )
            if self.tool_timeout_seconds <= 0:
                raise ValueError(
                    f"tool_timeout_seconds must be > 0, got {self.tool_timeout_seconds}"
                )
        if self.max_output_bytes is not None:
            if not isinstance(self.max_output_bytes, int) or self.max_output_bytes <= 0:
                raise ValueError(
                    f"max_output_bytes must be a positive int, got {self.max_output_bytes!r}"
                )

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
                        "tool_timeout_seconds", "max_output_bytes"}
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
    """
    if supplied is None:
        supplied = RuntimeLimits()
    default = ISOLATED_SUBAGENT_DEFAULT_RUNTIME_LIMITS
    return RuntimeLimits(
        memory_max_mb=(
            supplied.memory_max_mb
            if supplied.memory_max_mb is not None
            else default.memory_max_mb
        ),
        pids_max=(
            supplied.pids_max
            if supplied.pids_max is not None
            else default.pids_max
        ),
        cpu_weight=(
            supplied.cpu_weight
            if supplied.cpu_weight is not None
            else default.cpu_weight
        ),
        tool_timeout_seconds=(
            supplied.tool_timeout_seconds
            if supplied.tool_timeout_seconds is not None
            else default.tool_timeout_seconds
        ),
        max_output_bytes=(
            supplied.max_output_bytes
            if supplied.max_output_bytes is not None
            else default.max_output_bytes
        ),
        extra=dict(supplied.extra),
    )
