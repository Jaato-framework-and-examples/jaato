"""Phase 5 §5.9 — typed validation of supervisor-declared
sub-profile tightening flags.

The §4.3.4 sub-profile body is daemon-trusted: the runner can
request sub-profile creation, but the daemon decides the rules.
§5.9 lets the supervisor declare **additive tightenings** (less
capability than the default sub-profile, never more) through
``agent_params``, with a daemon-side allow-list enforcing the
permitted-keys + value-shape contract.

Tightenings flow runner → daemon through
``subagent.spawn_isolated_runner``.  This module is the daemon-
side validator; it sits at the same trust boundary as
``profile_payload_schema.py`` (§5.8) and shares its
reject-unknown posture.

v1 keys (post-§5.9 ship):

| Key                              | Effect                                 |
|----------------------------------|----------------------------------------|
| ``isolated_workspace_subpath``   | Narrow workspace allow to ``{ws}/X/``  |
| ``isolated_read_only_workspace`` | Downgrade workspace ``rwkl`` → ``r``   |

Forward-compat contract: adding a new tightening requires
extending three places in lockstep — this module's allow-list +
per-key validator + the renderer in ``server/apparmor.py``.

Design doc:
``docs/design/phase5_5_9_sub_profile_tightening_flags_audit.md``.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional


# ──────────────────────────────────────────────────────────────────
# Allow-list of supervisor-declared sub-profile tightening keys.
# ──────────────────────────────────────────────────────────────────
#
# Keys are namespaced with the ``isolated_`` prefix so they share
# the existing ``agent_params.isolated`` control-flag convention.
# The handler strips them from ``agent_params`` before forwarding
# the rest as template data to the new sub-runner — tightenings
# are daemon-side control flags, NOT case_data for {{name}}
# substitution.
SUB_PROFILE_TIGHTENING_KEYS: FrozenSet[str] = frozenset({
    "isolated_workspace_subpath",
    "isolated_read_only_workspace",
})


# Per-key size caps.  Generous (~10× normal usage) — DoS prevention
# against the unconfined daemon, not policy enforcement.
_MAX_WORKSPACE_SUBPATH_LEN = 256

# Characters forbidden in workspace_subpath.  Two reasons:
#
# 1. AppArmor profile-file syntax safety: ``"``, ``\n``, NUL would
#    break the quoted path in the rendered profile body.
# 2. Glob injection: ``*``, ``?``, ``[``, ``{`` would let the
#    supervisor express patterns the renderer wasn't designed to
#    quote-escape — defense in depth.
_FORBIDDEN_SUBPATH_CHARS: FrozenSet[str] = frozenset(
    '"\n\x00*?[]{}|;&`$<>\\\r\t',
)


def extract_and_validate_tightenings(
    agent_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract + validate sub-profile tightening keys from
    ``agent_params``.

    Returns a NEW dict containing only the validated tightening
    keys.  Does NOT mutate the input — caller is responsible for
    stripping the tightening keys from ``agent_params`` before
    forwarding the rest as template data.

    Args:
        agent_params: The ``agent_params`` arg of a
            ``subagent.spawn_isolated_runner`` RPC.  May be
            ``None`` (no template data + no tightenings).

    Returns:
        Dict containing each validated tightening key present in
        the input.  Empty dict when no tightening keys are
        supplied.

    Raises:
        ValueError: On any shape violation — wrong type, value
            out of range, path traversal, glob injection, etc.
            The exception message names the offending key + the
            specific violation.  Callers wrap with the RPC-
            method-name prefix per the existing handler
            convention.
    """
    if agent_params is None:
        return {}
    if not isinstance(agent_params, dict):
        raise ValueError(
            f"agent_params must be a dict or None, got "
            f"{type(agent_params).__name__}"
        )

    out: Dict[str, Any] = {}

    if "isolated_workspace_subpath" in agent_params:
        out["isolated_workspace_subpath"] = (
            _validate_workspace_subpath(
                agent_params["isolated_workspace_subpath"],
            )
        )

    if "isolated_read_only_workspace" in agent_params:
        v = agent_params["isolated_read_only_workspace"]
        # Reject str/int — bool is a subclass of int in Python, so
        # ``isinstance(True, int)`` is True.  Explicit type-check
        # via ``type(v) is bool`` rules out the int-truthy case
        # without also rejecting the legit bool case.
        if type(v) is not bool:
            raise ValueError(
                f"isolated_read_only_workspace must be a bool, "
                f"got {type(v).__name__}"
            )
        out["isolated_read_only_workspace"] = v

    return out


def _validate_workspace_subpath(value: Any) -> str:
    """Validate the ``isolated_workspace_subpath`` value.

    Returns the value when valid (str passthrough); raises
    ``ValueError`` otherwise.

    Rules:
      - Must be a non-empty str ≤ 256 chars.
      - No path traversal (``..`` segments, absolute paths,
        Windows drive letters).
      - No leading/trailing slash.
      - No glob/wildcards.
      - No characters that would break AppArmor profile-file
        quoting.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"isolated_workspace_subpath must be a str, got "
            f"{type(value).__name__}"
        )
    if not value:
        raise ValueError(
            "isolated_workspace_subpath must be non-empty"
        )
    if len(value) > _MAX_WORKSPACE_SUBPATH_LEN:
        raise ValueError(
            f"isolated_workspace_subpath exceeds length cap "
            f"({len(value)} > {_MAX_WORKSPACE_SUBPATH_LEN})"
        )
    # Absolute paths (POSIX or Windows-drive).
    if value.startswith("/") or value.startswith("\\"):
        raise ValueError(
            f"isolated_workspace_subpath must be relative, got "
            f"absolute path {value!r}"
        )
    if len(value) >= 2 and value[1] == ":":
        # Windows drive letter form like ``C:``.
        raise ValueError(
            f"isolated_workspace_subpath must be relative, got "
            f"drive-letter path {value!r}"
        )
    # No trailing slash — the renderer appends ``/`` to form the
    # workspace allow.  Trailing slash would produce ``//``.
    if value.endswith("/") or value.endswith("\\"):
        raise ValueError(
            f"isolated_workspace_subpath must not have a "
            f"trailing slash, got {value!r}"
        )
    # Path traversal — reject any ``..`` segment.  Normalize
    # separators first so ``..\\`` on a hypothetical Windows-style
    # input is also caught.
    normalized = value.replace("\\", "/")
    segments = normalized.split("/")
    if ".." in segments:
        raise ValueError(
            f"isolated_workspace_subpath must not contain "
            f"``..`` segments, got {value!r}"
        )
    # AppArmor-quote-breaking + glob chars.
    bad_chars = sorted({c for c in value if c in _FORBIDDEN_SUBPATH_CHARS})
    if bad_chars:
        raise ValueError(
            f"isolated_workspace_subpath contains forbidden "
            f"characters {bad_chars!r}; rejected to prevent "
            f"AppArmor-quote escape and glob injection"
        )
    return value
