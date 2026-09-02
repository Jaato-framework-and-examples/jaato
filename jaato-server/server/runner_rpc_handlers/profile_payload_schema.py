"""Phase 5 §5.8 — typed validation of the ``profile_payload`` RPC arg.

The ``subagent.spawn_isolated_runner`` RPC carries a serialized
``SubagentProfile`` as a free-form ``Dict[str, Any]`` (Phase 4 §4.3.2
wire shape).  ``SpawnIsolatedRunnerHandler`` only checked the top-
level type (``isinstance(value, dict)``); per-key validation was
deferred to §5.8 (§4.3.9 item 10 in
``docs/design/phase4_implementation_audits.md``).

This module supplies the boundary validator.  The runner→daemon RPC
is the trust boundary in Phase 4/5's threat model: the runner runs
model-controlled subprocess code under AppArmor + cgroup
confinement; the daemon holds provider tokens, OAuth credentials,
and cgroup/AppArmor management capability outside that confinement.
Field-level validation at the handler boundary makes the wire shape
**explicit + auditable** — extra keys are rejected (no silent
acceptance), wrong types fail with precise diagnostics, and size
caps prevent DoS-via-malformed-payload pivots against the
unconfined daemon.

Caller contract: ``validate_profile_payload`` raises ``ValueError``
on any shape violation.  The RPC dispatcher translates that into a
typed error envelope on the wire (the existing §3.2 / §4.3.2
``stage=validation`` contract).

Design doc: ``docs/design/phase5_5_8_profile_payload_typed_model_audit.md``.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional

from shared.instruction_suppression import normalize_suppression


# ──────────────────────────────────────────────────────────────────
# Allow-list — the single source of truth for which keys the
# daemon accepts from a runner-supplied profile_payload.
# ──────────────────────────────────────────────────────────────────
#
# Mirrors the producer-side wire shape in
# ``shared/plugins/subagent/plugin.py:2283-2325``.  When a new
# field needs to ride this wire, THREE files change in lockstep:
#
#   1. Producer (``shared/plugins/subagent/plugin.py``): add the
#      key to the ``profile_payload`` dict.
#   2. Consumer (``shared/plugins/subagent/config.py:
#      build_inline_profile``): read the key during reconstruction.
#   3. Allow-list (this constant) + the matching per-key rule in
#      ``validate_profile_payload``: extend the schema.
#
# The third step is **mandatory** by design — a producer cannot
# quietly start riding new keys past an unupgraded daemon.  This is
# the §4.3.9 item 10 hardening posture: reject unknown fields.
PROFILE_PAYLOAD_ALLOWED_KEYS: FrozenSet[str] = frozenset({
    "name",
    "description",
    "model",
    "provider",
    "plugins",
    "plugin_configs",
    "system_instructions",
    "suppress_base_instructions",
    "max_turns",
    "env",
    "gc",
    "trace",
    "runtime_limits",
})


# Per-key size caps.  Generous (~10× normal usage) — the goal is
# DoS prevention against the unconfined daemon, not field-level
# policy enforcement.  Profiles legitimately needing more should
# be split or reviewed at a higher level.
_MAX_NAME_LEN = 256
_MAX_DESCRIPTION_LEN = 1024
_MAX_MODEL_LEN = 128
_MAX_PROVIDER_LEN = 64
_MAX_PLUGINS_COUNT = 64
_MAX_PLUGIN_NAME_LEN = 128
_MAX_PLUGIN_CONFIGS_COUNT = 64
_MAX_PLUGIN_CONFIG_KEY_LEN = 64
_MAX_SYSTEM_INSTRUCTIONS_LEN = 65536
_MIN_MAX_TURNS = 1
_MAX_MAX_TURNS = 1000
_MAX_ENV_COUNT = 128
_MAX_ENV_KEY_LEN = 256
_MAX_ENV_VALUE_LEN = 4096

# Allowed sub-keys inside the ``gc`` dict.  The producer side emits
# ``{type, config}`` only (see plugin.py:2298-2304).
_GC_ALLOWED_KEYS: FrozenSet[str] = frozenset({"type", "config"})


def validate_profile_payload(payload: Any) -> None:
    """Validate a runner-supplied ``profile_payload`` dict.

    Args:
        payload: The ``profile_payload`` arg of a
            ``subagent.spawn_isolated_runner`` RPC.  Caller has
            already verified ``isinstance(payload, dict)`` at the
            handler boundary; this function re-verifies defensively
            so it's safe to call standalone.

    Raises:
        ValueError: On any shape violation — unknown top-level key,
            missing required key, wrong type, value out of range,
            or nested shape problem.  The exception message names
            the offending key + the specific violation.  Callers
            wrap with the RPC-method-name prefix per the existing
            handler convention.

    Notes:
        - The validator is **structural**, not semantic.  It does
          not check whether ``model`` names a registered model, or
          whether ``provider`` names a registered provider — those
          belong further downstream where the relevant lookup
          tables exist.
        - ``runtime_limits`` gets a top-level type check only; full
          structural validation is delegated to
          ``RuntimeLimits.from_dict`` during profile reconstruction.
          Splitting it would duplicate logic that already has its
          own test coverage.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"profile_payload must be a dict, "
            f"got {type(payload).__name__}"
        )

    # ── Unknown top-level keys ─────────────────────────────────
    extra_keys = set(payload.keys()) - PROFILE_PAYLOAD_ALLOWED_KEYS
    if extra_keys:
        # Sort for deterministic error messages — important for
        # tests that match on the message text.
        raise ValueError(
            f"profile_payload has unknown key(s): "
            f"{sorted(extra_keys)!r}.  "
            f"Allowed: {sorted(PROFILE_PAYLOAD_ALLOWED_KEYS)!r}"
        )

    # ── Required: name ─────────────────────────────────────────
    _require_nonempty_str(payload, "name", _MAX_NAME_LEN)

    # ── Optional fields ────────────────────────────────────────
    _optional_str_or_none(payload, "description", _MAX_DESCRIPTION_LEN)
    _optional_str_or_none(payload, "model", _MAX_MODEL_LEN)
    _optional_str_or_none(payload, "provider", _MAX_PROVIDER_LEN)
    _optional_str_or_none(
        payload, "system_instructions", _MAX_SYSTEM_INSTRUCTIONS_LEN,
    )

    if "suppress_base_instructions" in payload:
        v = payload["suppress_base_instructions"]
        # bool | mapping over {disk,constants,security} | list of piece names
        # (wire form).  normalize_suppression fails loud on an unknown piece
        # or unsupported type — surface that as the payload validation error.
        try:
            normalize_suppression(v)
        except ValueError as exc:
            raise ValueError(
                f"profile_payload.suppress_base_instructions invalid: {exc}"
            ) from exc

    if "max_turns" in payload:
        v = payload["max_turns"]
        # Reject bool explicitly — bool is a subclass of int in
        # Python.  ``max_turns=True`` would otherwise type-check as
        # int(1), which is misleading.
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(
                f"profile_payload.max_turns must be an int, "
                f"got {type(v).__name__}"
            )
        if v < _MIN_MAX_TURNS or v > _MAX_MAX_TURNS:
            raise ValueError(
                f"profile_payload.max_turns out of range "
                f"[{_MIN_MAX_TURNS}, {_MAX_MAX_TURNS}], got {v}"
            )

    if "plugins" in payload:
        _check_plugins(payload["plugins"])

    if "plugin_configs" in payload:
        _check_plugin_configs(payload["plugin_configs"])

    if "env" in payload:
        _check_env(payload["env"])

    if "gc" in payload:
        _check_gc(payload["gc"])

    if "runtime_limits" in payload:
        v = payload["runtime_limits"]
        if not isinstance(v, dict):
            raise ValueError(
                f"profile_payload.runtime_limits must be a dict, "
                f"got {type(v).__name__}"
            )
        # Structural content delegated to RuntimeLimits.from_dict
        # during profile reconstruction — its existing test
        # coverage validates the field set + per-field types.

    _check_trace(payload.get("trace"))


# ──────────────────────────────────────────────────────────────────
# Per-key helpers
# ──────────────────────────────────────────────────────────────────


def _check_trace(value: Any) -> None:
    """Validate a ``trace`` block at the runner->daemon boundary.

    Unlike ``runtime_limits``, this one is validated HERE rather than
    only at reconstruction: the block's entire reason for existing is
    that a trace path arriving as a switch (``"1"``) is accepted by
    every string-typed surface it crosses and produces a file named
    ``1`` in a workspace (issue #775).  A boundary that re-checks a
    thing it already knows how to reject costs one call.

    ``TraceProfileConfig.from_dict`` is the single rule -- duplicating
    its vocabulary here would be a second place for it to drift.

    ``None`` means the key was absent, which is legal: taking the
    absent-check here rather than at the call site keeps
    ``validate_profile_payload`` at its complexity baseline.
    """
    from shared.plugins.subagent.config import TraceProfileConfig

    if value is None:          # key absent — the block is optional
        return
    if not isinstance(value, dict):
        raise ValueError(
            f"profile_payload.trace must be a dict, "
            f"got {type(value).__name__}"
        )
    try:
        TraceProfileConfig.from_dict(value)
    except ValueError as exc:
        raise ValueError(f"profile_payload.{exc}") from exc


def _require_nonempty_str(
    payload: Dict[str, Any], key: str, max_len: int,
) -> None:
    if key not in payload:
        raise ValueError(
            f"profile_payload.{key} is required but missing"
        )
    v = payload[key]
    if not isinstance(v, str):
        raise ValueError(
            f"profile_payload.{key} must be a str, "
            f"got {type(v).__name__}"
        )
    if not v:
        raise ValueError(
            f"profile_payload.{key} must be a non-empty str"
        )
    if len(v) > max_len:
        raise ValueError(
            f"profile_payload.{key} exceeds length cap "
            f"({len(v)} > {max_len})"
        )


def _optional_str_or_none(
    payload: Dict[str, Any], key: str, max_len: int,
) -> None:
    if key not in payload:
        return
    v = payload[key]
    if v is None:
        return
    if not isinstance(v, str):
        raise ValueError(
            f"profile_payload.{key} must be a str or None, "
            f"got {type(v).__name__}"
        )
    if len(v) > max_len:
        raise ValueError(
            f"profile_payload.{key} exceeds length cap "
            f"({len(v)} > {max_len})"
        )


def _check_plugins(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"profile_payload.plugins must be a list, "
            f"got {type(value).__name__}"
        )
    if len(value) > _MAX_PLUGINS_COUNT:
        raise ValueError(
            f"profile_payload.plugins exceeds count cap "
            f"({len(value)} > {_MAX_PLUGINS_COUNT})"
        )
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"profile_payload.plugins[{i}] must be a str, "
                f"got {type(item).__name__}"
            )
        if len(item) > _MAX_PLUGIN_NAME_LEN:
            raise ValueError(
                f"profile_payload.plugins[{i}] exceeds length "
                f"cap ({len(item)} > {_MAX_PLUGIN_NAME_LEN})"
            )


def _check_plugin_configs(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"profile_payload.plugin_configs must be a dict, "
            f"got {type(value).__name__}"
        )
    if len(value) > _MAX_PLUGIN_CONFIGS_COUNT:
        raise ValueError(
            f"profile_payload.plugin_configs exceeds count cap "
            f"({len(value)} > {_MAX_PLUGIN_CONFIGS_COUNT})"
        )
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(
                f"profile_payload.plugin_configs key must be a "
                f"str, got {type(k).__name__}"
            )
        if len(k) > _MAX_PLUGIN_CONFIG_KEY_LEN:
            raise ValueError(
                f"profile_payload.plugin_configs[{k!r}] key "
                f"exceeds length cap "
                f"({len(k)} > {_MAX_PLUGIN_CONFIG_KEY_LEN})"
            )
        if not isinstance(v, dict):
            raise ValueError(
                f"profile_payload.plugin_configs[{k!r}] must "
                f"be a dict, got {type(v).__name__}"
            )


def _check_env(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"profile_payload.env must be a dict, "
            f"got {type(value).__name__}"
        )
    if len(value) > _MAX_ENV_COUNT:
        raise ValueError(
            f"profile_payload.env exceeds count cap "
            f"({len(value)} > {_MAX_ENV_COUNT})"
        )
    for k, v in value.items():
        if not isinstance(k, str):
            raise ValueError(
                f"profile_payload.env key must be a str, "
                f"got {type(k).__name__}"
            )
        if len(k) > _MAX_ENV_KEY_LEN:
            raise ValueError(
                f"profile_payload.env[{k!r}] key exceeds length "
                f"cap ({len(k)} > {_MAX_ENV_KEY_LEN})"
            )
        if not isinstance(v, str):
            raise ValueError(
                f"profile_payload.env[{k!r}] value must be a str, "
                f"got {type(v).__name__}"
            )
        if len(v) > _MAX_ENV_VALUE_LEN:
            raise ValueError(
                f"profile_payload.env[{k!r}] value exceeds length "
                f"cap ({len(v)} > {_MAX_ENV_VALUE_LEN})"
            )


def _check_gc(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            f"profile_payload.gc must be a dict, "
            f"got {type(value).__name__}"
        )
    extra = set(value.keys()) - _GC_ALLOWED_KEYS
    if extra:
        raise ValueError(
            f"profile_payload.gc has unknown key(s): "
            f"{sorted(extra)!r}.  Allowed: "
            f"{sorted(_GC_ALLOWED_KEYS)!r}"
        )
    if "type" in value and not isinstance(value["type"], str):
        raise ValueError(
            f"profile_payload.gc.type must be a str, "
            f"got {type(value['type']).__name__}"
        )
    if "config" in value and not isinstance(value["config"], dict):
        raise ValueError(
            f"profile_payload.gc.config must be a dict, "
            f"got {type(value['config']).__name__}"
        )
