"""What a revive takes from a session record it cannot authenticate (#1529).

:mod:`server.record_seal` answers whether a record is the one the daemon
wrote.  This module answers the next question: when it is NOT (a record a
session edited, one written before seals existed, one written by a
runner-side ``save`` tool, one sealed under another daemon's key), which of
its fields may a revive still use?

The rule is: **a field that decides the boundary is re-derived from
something the daemon owns, or narrowed, or dropped -- never read from the
record.**  :func:`distrust_record` rewrites the deserialized
``SessionState`` in place to that fail-safe form, so the revive code that
follows (``SessionManager._load_session_impl``) needs no second branch for
it, and returns a :class:`DistrustOutcome` saying what it changed.

==========================  ===============================================
Record field                On an unverified record
==========================  ===============================================
``profile_snapshot``        dropped: the profile is re-resolved from disk by
                            ``profile_ref`` (#1588; a qualified ref binds
                            only that set's file), else ``profile_name``
                            (the ``JAATO_REVIVE_PROFILE=disk``
                            path), so plugins, plugin configs (the permission
                            policy among them), AppArmor fragments,
                            ``runtime_limits`` (seccomp, pids, memory, cpu,
                            lifetime bounds), ``scrub_secret_env`` and every
                            ``env`` secret URI come from the operator's files.
                            A name that no longer resolves REFUSES the revive
                            (``require_disk_profile``): falling back to an
                            env-only session would hand it the default plugin
                            set, which may be wider than its profile
``profile_spec``            an inline recipe has no on-disk counterpart to
                            re-derive from: the revive is REFUSED
``sandbox_mode``            kept as EVIDENCE only; the caller arms kernel
                            confinement regardless of what it says
``workspace_path``          the workspace the record was READ FROM, else the
                            daemon's session-workspace index
``config_root``             dropped: the attaching client's, else
                            ``<workspace>/.jaato``
``created_by``,             from the daemon's index (``membership``), else
``cascade_driver_id``,      ``None``: the record's claim would put the session
``sibling_name``            in another user's group (courier messaging)
``agent_params``            dropped: they are fed to the persona's prefetch
                            scripts on a re-render
``metadata.plugin_states``  ``permission`` is NARROWED (deny rules and a
                            ``deny`` default kept; ``always`` rules dropped);
                            every other plugin's state is dropped
``metadata.subagents``      dropped: no subagent is restored
``metadata.model_override`` only ``MODEL_NAME`` / ``JAATO_PROVIDER`` kept
==========================  ===============================================

Still taken from the record, and why:

* ``history``, ``turn_accounting``, ``budget_state``, ``interrupted_turn``,
  ``workspace_files``, ``user_inputs`` -- the conversation and its
  bookkeeping.  The agent already controls what its own context says.
* ``rendered_instructions`` -- the prompt.  Lower impact for the same
  reason, and re-rendering instead would re-run the persona's
  ``{{!py:...}}`` prefetch scripts on a revive (#787).
* ``budget_control`` (only when the disk profile declares none) and
  ``budget_usage`` -- there is no daemon-owned copy, and dropping them is no
  narrower than an edited value: an edited ceiling can only be raised to
  "none", which is what dropping it gives.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: The ``model_override_env`` keys a no-profile ``session.new --model``
#: writes (``session_manager._apply_model_override``).  Anything else in an
#: unverified record's override is dropped.
MODEL_OVERRIDE_KEYS = frozenset({"MODEL_NAME", "JAATO_PROVIDER"})

#: The membership facts re-derived from the daemon's index.
MEMBERSHIP_FIELDS = ("created_by", "cascade_driver_id", "sibling_name")


@dataclass
class DistrustOutcome:
    """What :func:`distrust_record` did to one unverified record.

    Attributes:
        refusal: Set when the record cannot be revived safely at all; the
            caller refuses the revive with this sentence.
        require_disk_profile: ``True`` when the record names a profile:
            the caller must refuse the revive if that name does not
            resolve from disk.
        changes: One short phrase per field changed, for the WARNING.  Names
            keys, never values (a value may be a secret URI).
    """

    refusal: Optional[str] = None
    require_disk_profile: bool = False
    changes: List[str] = field(default_factory=list)


def narrow_permission_state(state: Any) -> Optional[Dict[str, Any]]:
    """The NARROWING half of a ``PermissionPlugin.get_persistence_state()``.

    Keeps ``session_blacklist`` (a ``never`` can only take a tool away) and a
    ``session_default_policy`` of ``deny``; drops ``session_whitelist`` (an
    ``always``) and any other default.  ``None`` when nothing narrowing is
    left.  Malformed entries are skipped, as the plugin's own restore skips
    them.
    """
    if not isinstance(state, dict):
        return None
    deny = [p for p in state.get("session_blacklist") or []
            if isinstance(p, str) and p]
    out: Dict[str, Any] = {}
    if deny:
        out["session_blacklist"] = sorted(set(deny))
    if state.get("session_default_policy") == "deny":
        out["session_default_policy"] = "deny"
    return out or None


def _same_path(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return a == b
    return os.path.realpath(a) == os.path.realpath(b)


def _distrust_workspace(state: Any, loaded_from: Optional[str],
                        indexed: Optional[str], out: DistrustOutcome) -> None:
    trusted = loaded_from or indexed
    if not _same_path(getattr(state, "workspace_path", None), trusted):
        out.changes.append("workspace_path (taken from where the record was "
                           "read / the daemon's index)")
    state.workspace_path = trusted


def _distrust_profile(state: Any, out: DistrustOutcome) -> None:
    if getattr(state, "profile_spec", None):
        out.refusal = (
            "the record carries an inline profile, which has no on-disk "
            "counterpart to re-derive it from; start a new session")
    if getattr(state, "profile_snapshot", None) is not None:
        out.changes.append("profile_snapshot (re-resolved from disk)")
        state.profile_snapshot = None
    if getattr(state, "profile_name", None) or getattr(state, "profile_ref", None):
        out.require_disk_profile = True
    if getattr(state, "config_root", None) is not None:
        out.changes.append("config_root")
        state.config_root = None
    if getattr(state, "agent_params", None):
        out.changes.append("agent_params")
    state.agent_params = None


def _distrust_membership(state: Any, membership: Optional[Dict[str, Any]],
                         out: DistrustOutcome) -> None:
    row = membership or {}
    for name in MEMBERSHIP_FIELDS:
        value = row.get(name) or None
        if getattr(state, name, None) != value:
            out.changes.append(f"{name} (from the daemon's index)")
        setattr(state, name, value)


def _distrust_plugin_states(metadata: Dict[str, Any],
                            out: DistrustOutcome) -> None:
    states = metadata.pop("plugin_states", None)
    if not isinstance(states, dict) or not states:
        return
    narrowed = narrow_permission_state(states.get("permission"))
    kept = {"permission": narrowed} if narrowed else {}
    dropped = sorted(k for k in states if k not in kept)
    if "permission" in states and narrowed != states.get("permission"):
        out.changes.append("plugin_states.permission (narrowed to deny rules)")
    others = [k for k in dropped if k != "permission"]
    if others:
        out.changes.append("plugin_states." + ",".join(others) + " (dropped)")
    if kept:
        metadata["plugin_states"] = kept


def _distrust_metadata(state: Any, out: DistrustOutcome) -> None:
    metadata = dict(getattr(state, "metadata", None) or {})
    _distrust_plugin_states(metadata, out)
    if metadata.pop("subagents", None):
        out.changes.append("metadata.subagents (not restored)")
    override = metadata.pop("model_override_env", None)
    if isinstance(override, dict):
        kept = {k: v for k, v in override.items()
                if k in MODEL_OVERRIDE_KEYS and isinstance(v, str)}
        if kept != override:
            out.changes.append("metadata.model_override_env (filtered)")
        if kept:
            metadata["model_override_env"] = kept
    state.metadata = metadata


def distrust_record(
    state: Any,
    *,
    loaded_from: Optional[str],
    indexed_workspace: Optional[str],
    membership: Optional[Dict[str, Any]],
) -> DistrustOutcome:
    """Rewrite an UNVERIFIED ``SessionState`` in place to its fail-safe form.

    See the module docstring for the field table.  Never raises on a
    malformed field: a record that cannot be read safely is narrowed or
    refused, never a crash.

    Args:
        state: The deserialized record (``record_verified`` is not True).
        loaded_from: The workspace whose ``.jaato/sessions/`` the record
            was read from, or ``None`` (the plugin's default storage).
        indexed_workspace: The daemon's session-workspace index answer for
            this id, used when ``loaded_from`` is ``None``.
        membership: The index's membership row for this id, or ``None``.

    Returns:
        A :class:`DistrustOutcome`.
    """
    out = DistrustOutcome()
    _distrust_workspace(state, loaded_from, indexed_workspace, out)
    _distrust_profile(state, out)
    _distrust_membership(state, membership, out)
    _distrust_metadata(state, out)
    return out
