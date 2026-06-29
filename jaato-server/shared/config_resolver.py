"""Resolve ordered search paths for read-only framework config.

The framework reads many kinds of configuration off disk: agent ``.md``
files, profiles, prompts, references, completion schemas, system
instructions, scripts, service definitions, and ad-hoc JSON knobs
(``gc.json``, ``thinking.json``, ``formatters.json``, …).  Historically
every loader hard-coded the same three-step lookup —

  1. ``<workspace_path>/.jaato/<domain>/``
  2. ``~/.jaato/<domain>/``
  3. premium entry-point fallback (only some loaders)

— with the ``.jaato/`` segment baked in.  That coupling makes the
agent's filesystem (the workspace its tools see) and the framework's
read source (where profiles, agents, etc. live) inseparable: to expose
config to the daemon you have to expose it to the agent's ``ls`` too.
The orchestrator's sandbox use case wants exactly the opposite — let
agents see only the sandbox dir, while the daemon reads config from
the project's real ``.jaato/`` outside the sandbox.

This module introduces ``config_root`` as an optional override for the
primary (workspace) tier of the lookup.  When unset, behavior is
unchanged.  When set, the workspace tier is replaced with whatever
path the caller provides.  The user tier (``~/.jaato/``) is always
appended as a secondary tier so per-user prompts/profiles/skills keep
working regardless.

Two helpers:

* :func:`resolve_config_search_path` — bare roots, useful when the
  caller wants to join a non-trivial subpath (or no subpath at all).
* :func:`resolve_domain_search_path` — joins the supplied ``domain``
  subdirectory and tags each root with its tier name, matching the
  shape :func:`bundle_common.bundle.resolve_bundle_roots` produces
  today.  The thin shape lets ``resolve_bundle_roots`` and other
  domain-scoped loaders migrate without changing their downstream
  callers.

Runtime/state writes (sessions, logs, backups, waypoints, vision,
replay workspaces, artifact-tracker) deliberately do **not** go
through this module — see :func:`workspace_state_path` for that path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union


# Tier name constants — kept lowercase to match the existing
# ``BUNDLE_TIER_WORKSPACE`` / ``BUNDLE_TIER_USER`` strings already in
# ``shared.plugins.bundle_common.bundle``.  The constants are duplicated
# here only so this module has no upward dependency on ``plugins``.
TIER_WORKSPACE: str = "workspace"
TIER_USER: str = "user"

# Conventional name of the per-workspace config directory.  Single
# source of truth so callers don't sprinkle ``".jaato"`` literals
# alongside this resolver.
JAATO_DIRNAME: str = ".jaato"


def resolve_config_search_path(
    workspace_path: Optional[str],
    config_root: Optional[str] = None,
    *,
    user_home: Optional[Path] = None,
) -> List[Path]:
    """Ordered list of root directories to scan for read-only config.

    Each loader is expected to join its own domain subdir on top —
    e.g. the profiles loader uses ``[r / "profiles" for r in roots]``.
    The returned paths may not exist on disk; callers should handle a
    missing root the same way they handle a missing user-tier dir
    (treat as "no entries in this tier", not as an error).

    Resolution chain:

    1. **Primary tier** —
       * if ``config_root`` is set: ``Path(config_root).resolve()``
       * else if ``workspace_path`` is set: ``<workspace_path>/.jaato``
       * else: omitted (only the user tier remains)
    2. **Secondary tier (always appended)** — ``user_home / ".jaato"``,
       defaulting to ``Path.home() / ".jaato"``.

    Args:
        workspace_path: Workspace root, when known.  Used for the
            primary tier only when ``config_root`` is ``None``.
        config_root: Explicit override for the primary tier.  When
            given, it replaces the workspace-anchored ``.jaato``
            subdir — the path is used **as-is**, *without* a further
            ``.jaato`` join (so callers pass e.g.
            ``/home/user/project/.jaato``, not ``/home/user/project``).
        user_home: Override for ``Path.home()``.  Test seam — production
            code passes ``None`` to use the real home directory.

    Returns:
        Ordered list of absolute root paths.  Length 1 when only the
        user tier is available (no workspace and no explicit config_root),
        length 2 in the normal case.

    Examples:
        Default (today's behavior, no config_root)::

            resolve_config_search_path("/proj")
            # → [Path("/proj/.jaato"), Path("/home/me/.jaato")]

        With explicit config_root::

            resolve_config_search_path("/proj/sandbox", config_root="/proj/.jaato")
            # → [Path("/proj/.jaato"), Path("/home/me/.jaato")]

        No workspace, no override (rare — daemon-internal lookups)::

            resolve_config_search_path(None)
            # → [Path("/home/me/.jaato")]
    """
    roots: List[Path] = []
    if config_root:
        roots.append(Path(config_root).expanduser().resolve())
    elif workspace_path:
        roots.append(Path(workspace_path).expanduser().resolve() / JAATO_DIRNAME)
    home = user_home if user_home is not None else Path.home()
    roots.append(home / JAATO_DIRNAME)
    return roots


def resolve_domain_search_path(
    workspace_path: Optional[str],
    config_root: Optional[str] = None,
    *,
    domain: Union[str, Path],
    user_home: Optional[Path] = None,
) -> List[Tuple[Path, str]]:
    """Domain-scoped search path with tier names.

    Joins ``domain`` to every root from :func:`resolve_config_search_path`
    and tags each result with its tier (``"workspace"`` for the primary,
    ``"user"`` for the secondary).  The shape matches what the existing
    ``bundle_common.bundle.resolve_bundle_roots`` returns, so loaders
    using that primitive can migrate by swapping the call.

    Args:
        workspace_path: Workspace root.  See
            :func:`resolve_config_search_path`.
        config_root: Explicit override for the primary tier.  See
            :func:`resolve_config_search_path`.
        domain: Domain subdir name — e.g. ``"profiles"``,
            ``"references"``, ``"agents"``.  May also be a multi-segment
            path (e.g. ``Path("services/_discovered")``); pass without
            a leading ``.jaato/`` (this helper adds it implicitly via
            the workspace tier and treats ``config_root`` as the
            already-anchored equivalent).
        user_home: Test seam.  See :func:`resolve_config_search_path`.

    Returns:
        Ordered list of ``(absolute_path, tier_name)`` pairs.
    """
    base_roots = resolve_config_search_path(
        workspace_path, config_root, user_home=user_home
    )
    if not base_roots:
        return []
    # First root is the primary tier ("workspace"); any subsequent
    # roots are the user tier.  Today there's exactly one of each, but
    # the list-shape keeps room for future tiers (premium overlay,
    # site-wide /etc/jaato, etc.) without changing the helper API.
    out: List[Tuple[Path, str]] = []
    for index, root in enumerate(base_roots):
        tier = TIER_WORKSPACE if index == 0 and (config_root or workspace_path) else TIER_USER
        out.append((root / domain, tier))
    return out


def workspace_state_path(
    workspace_path: str,
    subpath: Union[str, Path],
) -> Path:
    """Resolve a runtime/state write path under ``<workspace>/.jaato/``.

    **Never** honors ``config_root``.  Runtime artifacts (sessions,
    logs, backups, vision, replay, waypoints, artifact-tracker) are
    workspace-scoped writes; redirecting them to a shared
    ``config_root`` would either mix read-only/read-write semantics or
    collide across sessions that share the same config_root.

    Use :func:`resolve_config_search_path` for *reads* of framework
    configuration the user authors; use this helper for *writes* of
    state the daemon owns.

    Args:
        workspace_path: Workspace root.  Required — runtime writes
            must be anchored.
        subpath: Path under ``<workspace>/.jaato/`` (e.g.
            ``"sessions/abc123.json"`` or ``Path("logs") / sid``).

    Returns:
        Absolute path under the workspace's ``.jaato/`` directory.
        The path is **not** created on disk; callers are responsible
        for ``mkdir(parents=True, exist_ok=True)`` as appropriate.
    """
    return Path(workspace_path).expanduser().resolve() / JAATO_DIRNAME / subpath


def resolve_secret_uri(value: str) -> str:
    """Resolve a ``scheme://path[#key]`` secret URI to its secret value.

    Public entry point for secret-URI resolution. Returns *value* unchanged
    when it is not a resolvable secret URI — a plain string, a still-present
    ``${VAR}`` reference, or a standard network scheme (``https://``,
    ``wss://``, ...). A secret scheme with no registered resolver raises
    ``SecretResolutionError`` (importable from the same place this delegates
    to). Resolvers are discovered from the ``jaato.premium`` →
    ``secret_resolvers`` entry point and cached on first call.

    **In-process / embedding hosts must call this themselves.** When the
    daemon runs a session it resolves credential config values (e.g.
    ``plugin_configs[provider].api_key = pass://...``) upstream during
    profile/spec resolution. An embedded :class:`~jaato.JaatoClient` has no
    such upstream step, so it must resolve credential URIs with this function
    BEFORE building a provider — otherwise the raw URI reaches the provider
    credential boundary unresolved and fails loud.

    This is the public name for the resolver that previously had to be reached
    via the private ``shared.plugins.subagent.config._resolve_secret_uri``.
    """
    from shared.plugins.subagent.config import _resolve_secret_uri
    return _resolve_secret_uri(value)
