"""Slot-scoped plugin carry-over across cascade session boundaries.

A pre-warm pool slot serves several sessions of one cascade in turn: the
daemon calls ``session.end``, returns the slot to the pool, and the next
stage's ``session.bootstrap`` lands on the same process.  ``reset_for_next_
session()`` exists for exactly that boundary and several plugins answer it
with "keep everything — the next stage benefits".

That answer was aspirational.  Every ``session.bootstrap`` built a fresh
:class:`~shared.plugins.registry.PluginRegistry` and re-ran ``discover()``,
which calls ``module.create_plugin()``, so the preserved state belonged to
an object the next session discarded.  Nothing shut the outgoing registry
down either, so a plugin holding an OS process kept it: a 5-subphase
cascade ended with three live jdtls (~2.3 GB) parented to one slot, and
every stage paid the multi-minute cold start it was preserving state to
avoid (#890).

This module is the missing half.  At the session boundary the runner hands
its registry here; instances whose plugin declares
:data:`~jaato_sdk.plugins.base.TRAIT_SLOT_SCOPED` move into a process-level
store and everything else is shut down.  The next bootstrap adopts the
stored instances into its new registry BEFORE discovery runs, so discovery
skips those names and never builds a rival.

**The store is process-level because the slot is the process.**  One slot
serves one session at a time, so a module global has exactly the slot's
scope — there is no second session in this interpreter to confuse it with.

**Carry-over is conditional.**  A slot returns to the pool and can be
acquired by unrelated work, so an instance is reused only when the arriving
session matches the one that parked it on every axis that could make the
warm resource wrong: same cascade, same workspace, same config root, and
the same declared config for that plugin.  Anything else is a miss, and a
miss shuts the stored instance down rather than leaving it running — the
leak this module exists to close.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover — trivial import guard
    from jaato_sdk.plugins.base import TRAIT_SLOT_SCOPED
except ImportError:  # pragma: no cover — pre-trait SDK
    TRAIT_SLOT_SCOPED = "slot_scoped"


#: Config keys that identify the SESSION rather than the plugin's
#: configuration.  They differ on every session by construction, so
#: including them in the reuse fingerprint would make every comparison a
#: miss and the carry-over dead code.  ``workspace_path`` / ``config_root``
#: are excluded here because they are part of the carry KEY instead — a
#: stronger check than fingerprint equality, since they gate the whole set.
_IDENTITY_CONFIG_KEYS = frozenset({
    "session_id",
    "agent_name",
    "workspace_path",
    "config_root",
})


def _fingerprint(config: Optional[Dict[str, Any]]) -> str:
    """Stable string for a plugin's declared config, session identity aside.

    Compared between the session that parked an instance and the one asking
    to adopt it: a profile that changed ``plugin_configs.lsp.languageServers``
    between stages must get a fresh plugin, not a server started under the
    old declaration.

    Unserialisable values fall back to ``repr`` so a config carrying a live
    object degrades to "compares equal only to itself", which is the safe
    direction — a miss costs a cold start, a false hit reuses the wrong
    server.
    """
    if not config:
        return "{}"
    trimmed = {
        k: v for k, v in config.items()
        if k not in _IDENTITY_CONFIG_KEYS
    }
    try:
        return json.dumps(trimmed, sort_keys=True, default=repr)
    except Exception:  # noqa: BLE001 — fingerprint must never raise
        return repr(sorted(trimmed.items(), key=lambda kv: kv[0]))


@dataclass(frozen=True)
class SlotCarryKey:
    """Identity a carried plugin set belongs to.

    A slot is affine to a cascade, but affinity is not a guarantee — a slot
    returns to the pool and the pool may hand it to unrelated work.  The key
    is what makes reuse safe rather than merely likely: a warm language
    server rooted in one workspace must never be handed to a session running
    in another.

    Attributes:
        cascade_driver_id: The cascade whose stages share this slot.  ``None``
            for a standalone session, which never carries (there is no next
            stage to benefit, so parking state would only defer its teardown).
        workspace_path: Workspace root the carried resources are rooted in.
        config_root: Framework-config root the carried config was read under.
    """

    cascade_driver_id: Optional[str]
    workspace_path: Optional[str]
    config_root: Optional[str]

    @property
    def carryable(self) -> bool:
        """True when a session under this key may park plugins.

        Requires a cascade: without one there is no next session on this
        slot that the state is being kept for, and the contract
        (``reset_for_next_session``) is explicitly about "the next session
        of the same cascade".
        """
        return bool(self.cascade_driver_id)


@dataclass
class _Carried:
    """One parked plugin instance plus what it takes to re-register it."""

    plugin: Any
    origin: Any            # PluginOrigin from the parking registry
    enrichment_only: bool
    fingerprint: str


# The parked instances, and the identity they belong to.  Module-level
# because the slot IS the process; see the module docstring.
_carried: Dict[str, _Carried] = {}
_carry_key: Optional[SlotCarryKey] = None


def carry_key_for(envelope: Any) -> SlotCarryKey:
    """Build the carry key for a session envelope."""
    return SlotCarryKey(
        cascade_driver_id=getattr(envelope, "cascade_driver_id", None) or None,
        workspace_path=getattr(envelope, "workspace_path", None) or None,
        config_root=getattr(envelope, "config_root", None) or None,
    )


def is_slot_scoped(plugin: Any) -> bool:
    """True when ``plugin`` declares :data:`TRAIT_SLOT_SCOPED`.

    Read defensively: ``plugin_traits`` is an optional class attribute and a
    plugin from an older tree, or a test double, simply does not have one.
    """
    try:
        traits = getattr(plugin, "plugin_traits", frozenset()) or frozenset()
        return TRAIT_SLOT_SCOPED in traits
    except Exception:  # noqa: BLE001 — trait probe must never raise
        return False


def adopt_into(registry: Any, envelope: Any, plugin_configs: Dict[str, Any]) -> List[str]:
    """Adopt parked instances into a freshly-built registry.

    MUST be called before ``registry.discover()``: both discovery paths skip
    a name that is already registered, and that skip is what stops discovery
    building a second instance alongside the warm one.

    A parked plugin whose fingerprint no longer matches the arriving
    session's config, or whose key does not match at all, is shut down here
    rather than left running — the slot must not accumulate abandoned
    resources, which is the whole failure this module addresses.

    Args:
        registry: The new session's :class:`PluginRegistry`.
        envelope: The arriving :class:`SessionInitEnvelope`.
        plugin_configs: The effective per-plugin config map this bootstrap
            will pass to ``expose_all`` — the same dict the parking session
            was fingerprinted against.

    Returns:
        Names actually adopted, for logging.
    """
    global _carry_key

    if not _carried:
        return []

    key = carry_key_for(envelope)
    if _carry_key != key:
        logger.info(
            "slot carry: discarding %d parked plugin(s) — arriving session "
            "does not match the parked identity (parked=%r arriving=%r)",
            len(_carried), _carry_key, key,
        )
        release_all(reason="carry key mismatch")
        return []

    adopted: List[str] = []
    stale: List[str] = []
    for name, entry in list(_carried.items()):
        if entry.fingerprint != _fingerprint(plugin_configs.get(name)):
            stale.append(name)
            continue
        try:
            registry.adopt_plugin(
                name, entry.plugin, entry.origin,
                enrichment_only=entry.enrichment_only,
            )
            adopted.append(name)
        except Exception:  # noqa: BLE001 — per-plugin boundary
            logger.exception(
                "slot carry: adopting '%s' failed; falling back to a fresh "
                "instance", name,
            )
            stale.append(name)

    for name in stale:
        _shutdown_one(name, _carried.pop(name), reason="config changed")
    for name in adopted:
        _carried.pop(name, None)

    if not _carried:
        _carry_key = None

    if adopted:
        logger.info(
            "slot carry: adopted %s into session %s (warm — no re-initialize, "
            "no re-connect)",
            ", ".join(sorted(adopted)),
            getattr(envelope, "session_id", "?"),
        )
    return adopted


def park_from(
    registry: Any, envelope: Any, *, allow_carry: bool = True,
) -> Tuple[List[str], List[str]]:
    """Release an outgoing session's registry at the session boundary.

    Slot-scoped plugins are moved into the process-level store; every other
    initialized plugin is shut down.  Before #890 neither happened — the
    registry reference was dropped and the plugins were left to garbage
    collection, which does not terminate a subprocess a plugin started.

    Args:
        registry: The outgoing session's :class:`PluginRegistry`.
        envelope: The outgoing :class:`SessionInitEnvelope`, whose cascade /
            workspace identity the parked set is keyed on.
        allow_carry: False tears everything down instead of parking.  Passed
            by the caller when the slot is not going back to the pool (a
            failed reset, a cold shutdown), because parking state for a next
            session that will never arrive is just a slower leak.

    Returns:
        ``(parked_names, shutdown_errors)``.
    """
    global _carry_key

    # Anything still parked from an earlier boundary was not adopted, so
    # nothing is coming for it.
    release_all(reason="superseded by a newer session boundary")

    key = carry_key_for(envelope)
    may_carry = allow_carry and key.carryable

    parked: List[str] = []
    skip: Set[str] = set()
    if may_carry:
        for name in list(registry.list_available() or []):
            plugin = registry.get_plugin(name)
            if plugin is None or not is_slot_scoped(plugin):
                continue
            _carried[name] = _Carried(
                plugin=plugin,
                origin=_origin_of(registry, name),
                enrichment_only=name in getattr(registry, "_enrichment_only", set()),
                fingerprint=_fingerprint(_config_of(registry, name)),
            )
            skip.add(name)
            parked.append(name)
        _carry_key = key if parked else None

    # Duck-typed, like every other reach across this boundary: a registry
    # from an older tree (or a test double) may not have ``shutdown_all``.
    # Missing it means the pre-#890 behaviour — nothing is torn down — which
    # is a leak, not a crash, so say so once rather than failing the slot.
    shutdown_all = getattr(registry, "shutdown_all", None)
    if callable(shutdown_all):
        errors = shutdown_all(skip=skip)
    else:
        errors = []
        logger.warning(
            "slot carry: registry %s has no shutdown_all(); the outgoing "
            "session's plugins will not be torn down",
            type(registry).__name__,
        )

    if parked:
        logger.info(
            "slot carry: parked %s for the next session of cascade %s — "
            "warm resources kept; every other initialized plugin shut down",
            ", ".join(sorted(parked)), key.cascade_driver_id,
        )
    else:
        logger.debug(
            "slot carry: nothing parked (carryable=%s allow_carry=%s); "
            "the outgoing session's plugins were shut down",
            key.carryable, allow_carry,
        )
    return parked, errors


def release_all(reason: str = "slot teardown") -> List[str]:
    """Shut down every parked instance and empty the store.

    The final teardown a slot-scoped plugin's ``shutdown()`` docstring means
    by "slot-end".  Called when the slot is going away, and whenever the
    parked set is superseded or unclaimable — a parked plugin nobody will
    adopt is exactly the abandoned resource this module exists to prevent.

    Returns:
        Names whose ``shutdown()`` raised.  The store is emptied regardless.
    """
    global _carried, _carry_key
    errors: List[str] = []
    for name, entry in list(_carried.items()):
        if not _shutdown_one(name, entry, reason=reason):
            errors.append(name)
    _carried = {}
    _carry_key = None
    return errors


def carried_names() -> List[str]:
    """Names currently parked.  Introspection for tests and diagnostics."""
    return sorted(_carried)


def _shutdown_one(name: str, entry: _Carried, *, reason: str) -> bool:
    """Shut one parked instance down.  Returns False when it raised."""
    try:
        entry.plugin.shutdown()
        logger.info("slot carry: shut down parked plugin '%s' (%s)", name, reason)
        return True
    except Exception:  # noqa: BLE001 — per-plugin boundary
        logger.exception(
            "slot carry: shutdown of parked plugin '%s' raised (%s); "
            "dropping it anyway", name, reason,
        )
        return False


def _origin_of(registry: Any, name: str) -> Any:
    """The provenance the parking registry recorded for ``name``.

    Re-registering under the same origin keeps the adoption quiet: the
    registry's name-collision warning exempts two registrations of the same
    module, so an adopted built-in does not read as a shadowing attempt.
    """
    try:
        return registry.get_plugin_source(name)
    except Exception:  # noqa: BLE001
        return None


def _config_of(registry: Any, name: str) -> Optional[Dict[str, Any]]:
    """The effective config the parking registry initialized ``name`` with."""
    try:
        return dict(getattr(registry, "_configs", {}).get(name) or {})
    except Exception:  # noqa: BLE001
        return None
