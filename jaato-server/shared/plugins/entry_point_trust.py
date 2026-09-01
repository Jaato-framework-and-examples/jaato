"""Trust policy for ``jaato.*`` entry-point plugin discovery.

Background (issue #684)
-----------------------
:meth:`PluginRegistry.discover` runs entry-point discovery *before* the
directory scan, and the directory scan skips any name already
registered.  Entry-point loading had no name reservation and no
allowlist, so **any** distribution installed in the same virtualenv
could declare::

    [project.entry-points."jaato.plugins"]
    permission = "anything:create_plugin"

and take over the built-in ``permission`` plugin — the real one was
never imported, and nothing was logged.  ``ep.load()`` imports the
module at discovery time, so merely being installed was enough to run
code.

The policy in this module inverts that default:

1. **Built-in names are reserved.**  An entry point whose name matches a
   built-in plugin module is refused unless it is *provided by the
   built-in package itself* (jaato-server declares its own plugins
   through the same entry-point groups) or an operator names it in
   :data:`ENV_ALLOW_SHADOW`.
2. **A security-critical subset is never shadowable** — not even with
   the opt-in.  See :data:`NEVER_SHADOWABLE`.
3. **An optional distribution allowlist** (:data:`ENV_ENTRY_POINT_ALLOWLIST`)
   narrows the set of distributions that may contribute plugins at all.
4. Every refusal, and every shadow that *is* honoured, produces a
   ``WARNING`` naming the distribution involved.

All decisions are made from the entry point's *metadata*
(``ep.name`` / ``ep.value`` / ``ep.dist``) — that is, **before**
``ep.load()`` — so a refused entry point never gets its module
imported.
"""

from __future__ import annotations

import os
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Optional, Set

__all__ = [
    "ENV_ENTRY_POINT_ALLOWLIST",
    "ENV_ALLOW_SHADOW",
    "BUILTIN_PLUGIN_PACKAGE",
    "NEVER_SHADOWABLE",
    "PluginOrigin",
    "TrustDecision",
    "builtin_plugin_names",
    "entry_point_distribution",
    "entry_point_module",
    "evaluate_entry_point",
    "is_builtin_module",
    "normalize_distribution",
]


#: Operator knob: comma-separated distribution names that may contribute
#: plugins through the ``jaato.*`` entry-point groups.  When unset (or
#: empty) every installed distribution participates — today's behaviour.
#: When set, distributions outside the list are refused with a WARNING.
#: The built-in package is always honoured and never needs listing.
ENV_ENTRY_POINT_ALLOWLIST = "JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST"

#: Operator knob: comma-separated built-in plugin names that a
#: third-party distribution IS allowed to replace.  Names in
#: :data:`NEVER_SHADOWABLE` are refused even when listed here.
ENV_ALLOW_SHADOW = "JAATO_PLUGIN_ALLOW_SHADOW"

#: Import package that owns the built-in plugins.  An entry point whose
#: target module lives here is the framework's own declaration, not a
#: third-party claim.
BUILTIN_PLUGIN_PACKAGE = "shared.plugins"

#: Built-ins that may never be replaced by an out-of-tree entry point.
#: These mediate command execution, file writes, sandboxing, external
#: tool servers and the permission prompt itself — replacing any of them
#: silently is a full compromise of the agent's guard rails, so the
#: opt-in in :data:`ENV_ALLOW_SHADOW` deliberately does not reach them.
NEVER_SHADOWABLE: FrozenSet[str] = frozenset({
    "cli",
    "file_edit",
    "interactive_shell",
    "mcp",
    "permission",
    "sandbox_manager",
})


# --------------------------------------------------------------- names

_builtin_names_cache: Optional[FrozenSet[str]] = None


def builtin_plugin_names(refresh: bool = False) -> FrozenSet[str]:
    """Every top-level module name inside the built-in plugin package.

    Read with :func:`pkgutil.iter_modules` — a directory listing, **no
    imports** — so this is safe to call before any plugin has loaded and
    cheap enough to call per entry point (the result is cached).

    The set is deliberately wider than "plugins that exist": it also
    covers helper modules (``path_safety``, ``sandbox_utils``, …).
    Reserving those costs nothing and stops an out-of-tree package from
    claiming a name the framework may grow into.

    Consequence worth knowing before adding a built-in: because the set
    is derived rather than declared, **a jaato release that adds a
    plugin retroactively reserves that name**, and an out-of-tree
    plugin already using it stops loading on upgrade.  That is the
    intended trade — a name collision with a built-in has to resolve in
    the built-in's favour — and the refusal message names
    :data:`ENV_ALLOW_SHADOW`, so the operator gets a clear error naming
    the remedy rather than a plugin that quietly vanished.  Prefer a
    distinctive name for a new built-in over a generic one for the same
    reason.

    Args:
        refresh: Recompute instead of returning the cached set.  Only
            useful in tests that add or remove plugin directories.

    Returns:
        Frozen set of reserved built-in names.
    """
    global _builtin_names_cache
    if _builtin_names_cache is not None and not refresh:
        return _builtin_names_cache
    plugin_dir = Path(__file__).parent
    names = {
        name
        for _finder, name, _ispkg in pkgutil.iter_modules([str(plugin_dir)])
        if not name.startswith("_")
    }
    _builtin_names_cache = frozenset(names)
    return _builtin_names_cache


def normalize_distribution(name: Any) -> str:
    """Normalise a distribution name for comparison (PEP 503).

    ``Jaato_Server`` / ``jaato-server`` / ``jaato.server`` all compare
    equal.  Non-string input (a ``MagicMock`` in a test, a missing
    ``dist``) normalises to the empty string rather than raising.
    """
    if not isinstance(name, str):
        return ""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def entry_point_module(ep: Any) -> str:
    """The module an entry point targets, read from its ``value``.

    ``"shared.plugins.cli:create_plugin"`` → ``"shared.plugins.cli"``.
    Returns ``""`` when the value is missing or not a string — an
    unreadable target is treated as third-party, i.e. the conservative
    side of every decision below.
    """
    value = getattr(ep, "value", None)
    if not isinstance(value, str):
        return ""
    return value.split(":", 1)[0].strip()


def entry_point_distribution(ep: Any) -> Optional[str]:
    """The name of the distribution that declared *ep*, if discoverable.

    ``importlib.metadata`` attaches ``ep.dist`` for entry points obtained
    from :func:`importlib.metadata.entry_points` on Python 3.10+.  Older
    metadata objects — and the mocks used in tests — may not, so anything
    that is not a plain string yields ``None``.
    """
    dist = getattr(ep, "dist", None)
    name = getattr(dist, "name", None) if dist is not None else None
    return name if isinstance(name, str) else None


def is_builtin_module(module: str) -> bool:
    """Whether *module* lives inside the built-in plugin package."""
    return (
        module == BUILTIN_PLUGIN_PACKAGE
        or module.startswith(BUILTIN_PLUGIN_PACKAGE + ".")
    )


# ------------------------------------------------------------- policy

def _env_name_set(var: str) -> Set[str]:
    """Parse a comma-separated operator knob into a set of names."""
    raw = os.environ.get(var, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def allowed_distributions() -> Set[str]:
    """Normalised distribution allowlist from :data:`ENV_ENTRY_POINT_ALLOWLIST`.

    An empty set means "no allowlist configured" — every distribution
    participates, which is the pre-#684 behaviour and the default.

    **The opt-in default is deliberate, not an oversight.**  Callers
    guard on ``if allowlist and ...``, so an unset knob blocks nothing.
    Flipping this to default-deny would refuse every out-of-tree
    distribution the moment it was released — jaato-premium registers
    ``profile_tools`` / ``session_ops`` through ``jaato.plugins`` and
    ``auto_steering`` through ``jaato.enrichment_plugins`` — and would
    break those installs on upgrade rather than at a moment anyone
    chose.  Narrowing the participating set is an operator decision
    with an operator's knowledge of what is installed; the reservation
    above is what protects the default configuration.
    """
    return {normalize_distribution(n) for n in _env_name_set(ENV_ENTRY_POINT_ALLOWLIST)}


def shadow_opt_ins() -> Set[str]:
    """Built-in names an operator has explicitly opened up for shadowing."""
    return _env_name_set(ENV_ALLOW_SHADOW)


@dataclass(frozen=True)
class TrustDecision:
    """Outcome of applying the entry-point trust policy to one claim.

    Attributes:
        allowed: Whether the entry point may be loaded / registered.
            A refused claim is never ``ep.load()``-ed, so its module is
            not imported.
        reason: Stable machine-readable tag — one of ``"builtin"``,
            ``"not_allowlisted"``, ``"reserved"``, ``"never_shadowable"``,
            ``"shadow_opt_in"`` or ``"external"``.  Tests and callers
            branch on this rather than on the prose.
        message: Operator-facing WARNING text, or ``""`` when the
            decision is unremarkable and should stay quiet.
    """

    allowed: bool
    reason: str
    message: str = ""

    @property
    def warn(self) -> bool:
        """Whether :attr:`message` should be logged at WARNING."""
        return bool(self.message)


def _describe(claim: str, module: str, distribution: Optional[str]) -> str:
    """Render the claimant for a warning line."""
    dist = distribution or "<unknown distribution>"
    return f"'{claim}' from {dist} ({module or '<unknown module>'})"


def evaluate_entry_point(
    claim: str,
    module: str,
    distribution: Optional[str],
) -> TrustDecision:
    """Decide whether an entry point may claim the plugin name *claim*.

    Called twice per entry point: once on ``ep.name`` **before**
    ``ep.load()`` (so a refusal prevents the import), and once on the
    instantiated plugin's own ``.name`` — a plugin whose ``name``
    property returns something other than its entry-point name would
    otherwise slip a reserved name past the pre-load check.

    Args:
        claim: The plugin name being claimed.
        module: Import path of the entry point's target module (see
            :func:`entry_point_module`).
        distribution: Declaring distribution name, if known.

    Returns:
        A :class:`TrustDecision`.  ``allowed=False`` means "skip this
        entry point"; ``message`` is non-empty whenever the operator
        should hear about it.
    """
    if is_builtin_module(module):
        # LOAD-BEARING, and not obviously so from a source checkout: on
        # an INSTALLED tree the built-ins are themselves entry points
        # (``cli = shared.plugins.cli`` and 16 more in jaato-server's
        # pyproject).  Drop this bypass and the framework refuses its
        # own declarations — discovery does not merely lose the
        # reservation, it collapses.  Verified by sabotage: making this
        # branch unreachable fails 12 of the guards in
        # ``test_entry_point_trust.py``, not one.
        return TrustDecision(True, "builtin")

    who = _describe(claim, module, distribution)
    allowlist = allowed_distributions()
    if allowlist and normalize_distribution(distribution) not in allowlist:
        return TrustDecision(
            False,
            "not_allowlisted",
            f"Refusing plugin entry point {who}: its distribution is not "
            f"in {ENV_ENTRY_POINT_ALLOWLIST}="
            f"{','.join(sorted(allowlist))}. Add it there to load this "
            f"plugin.",
        )

    if claim not in builtin_plugin_names():
        return TrustDecision(True, "external")

    if claim in NEVER_SHADOWABLE:
        return TrustDecision(
            False,
            "never_shadowable",
            f"Refusing plugin entry point {who}: '{claim}' is a "
            f"security-critical built-in and can never be replaced by an "
            f"out-of-tree plugin. Rename the entry point.",
        )

    if claim in shadow_opt_ins():
        return TrustDecision(
            True,
            "shadow_opt_in",
            f"Plugin entry point {who} SHADOWS the built-in '{claim}' "
            f"— honoured because '{claim}' is listed in "
            f"{ENV_ALLOW_SHADOW}. The built-in will not be loaded.",
        )

    return TrustDecision(
        False,
        "reserved",
        f"Refusing plugin entry point {who}: '{claim}' is a reserved "
        f"built-in plugin name. Rename the entry point, or set "
        f"{ENV_ALLOW_SHADOW}={claim} to let this distribution replace "
        f"the built-in.",
    )


# ------------------------------------------------------------ origins

@dataclass(frozen=True)
class PluginOrigin:
    """Where a registered plugin came from.

    Recorded by :class:`~shared.plugins.registry.PluginRegistry` for every
    plugin it registers, so a shadow is visible from ``jaato-scaffold
    plugins`` without reading logs (issue #684 item 4), and so the
    registry can name the incumbent when a later claim on the same name
    is skipped.

    Attributes:
        name: The plugin name it registered under.
        via: How it was discovered — ``"entry_point"``, ``"directory"``
            or ``"registered"`` (handed to ``register_plugin()`` directly,
            e.g. the session plugin and premium's in-process plugins).
        module: Import path of the module that provided it.
        distribution: Declaring distribution, when discovery knew it
            (entry points only).
        entry_point: The entry-point name, when it differs from
            :attr:`name` — a plugin whose ``.name`` property disagrees
            with the name it was declared under.
    """

    name: str
    via: str
    module: str = ""
    distribution: Optional[str] = None
    entry_point: Optional[str] = None

    @property
    def builtin(self) -> bool:
        """Whether this plugin is the framework's own built-in."""
        return is_builtin_module(self.module)

    def describe(self) -> str:
        """One-line provenance, e.g. ``jaato-server (shared.plugins.cli)``.

        Used both in shadow WARNINGs and in the ``jaato-scaffold
        plugins`` table, so the same wording identifies a plugin
        everywhere it is reported.
        """
        if self.builtin:
            label = "built-in"
        else:
            label = self.distribution or "unknown distribution"
        module = self.module or "<unknown module>"
        suffix = ""
        if self.entry_point and self.entry_point != self.name:
            suffix = f" [entry point '{self.entry_point}']"
        return f"{label} ({module}){suffix}"
