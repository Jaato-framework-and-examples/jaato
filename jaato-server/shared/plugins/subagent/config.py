"""Configuration models for subagent plugin."""

import importlib.metadata
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol, Tuple, Union
from typing import runtime_checkable

from shared.runtime_limits import RuntimeLimits
from shared.budget_control import BudgetControlConfig, merge_limits
from shared.instruction_suppression import normalize_suppression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Secret resolver protocol and registry
#
# Allows premium (or third-party) packages to register secret backends
# (HashiCorp Vault, AWS Secrets Manager, SOPS, OS keyring, etc.) via the
# ``jaato.premium`` → ``secret_resolvers`` entry point.
#
# Config values like ``vault://secret/myapp#db_password`` are resolved
# transparently during variable expansion.
# ---------------------------------------------------------------------------

# Regex for ``scheme://path`` or ``scheme://path#key``.
_SECRET_URI_RE = re.compile(
    r'^(?P<scheme>[a-z][a-z0-9_+-]*)://'  # scheme (lowercase, RFC-ish)
    r'(?P<path>[^#]+)'                     # path (everything up to optional #)
    r'(?:#(?P<key>.+))?$'                  # optional #key fragment
)


# Standard network protocol schemes that are NEVER secret-resolver
# indirections — they're literal URLs awaiting plain HTTP/WS/FTP
# resolution.  Pre-server-0.6.57 ``_resolve_secret_uri`` matched these
# against the URI regex, fired a "no resolver registered" warning, and
# returned the literal URI unchanged.  Cosmetically noisy AND, when the
# URI contained an unresolved ``${VAR}`` substitution (e.g. handoff_test's
# ``http://127.0.0.1:${ANTIFRAUDE_PORT}``), prevented the env-file env-var
# expansion from running at the right point in the chain because the
# secret-URI machinery short-circuited to "literal URL".  Network-scheme
# values now bypass secret-URI resolution entirely; standard env-var
# expansion downstream handles ``${VAR}``.
_NETWORK_SCHEMES = frozenset({"http", "https", "ws", "wss", "ftp", "ftps"})


@runtime_checkable
class SecretResolver(Protocol):
    """Protocol for secret backend resolvers.

    Each resolver handles one or more URI schemes (e.g. ``vault``, ``awssm``,
    ``sops``, ``keyring``).  The framework discovers resolvers via the
    ``jaato.premium`` → ``secret_resolvers`` entry point and dispatches
    ``scheme://path#key`` references to the matching resolver.

    Implementations live in the premium package (or any third-party package
    that registers the entry point).  The core framework only defines this
    protocol and the dispatch logic.
    """

    @property
    def schemes(self) -> FrozenSet[str]:
        """URI schemes this resolver handles (e.g. ``frozenset({"vault"})``).

        Must be lowercase.  A resolver may handle multiple schemes — for
        example a "cloud" resolver might handle both ``awssm`` and ``gcpsm``.
        """
        ...

    def resolve(self, scheme: str, path: str, key: Optional[str] = None) -> str:
        """Resolve a secret reference to its plaintext value.

        Args:
            scheme: The URI scheme (e.g. ``"vault"``).
            path: The path portion of the URI (e.g. ``"secret/myapp"``).
            key: Optional key/field within the secret (from the ``#fragment``).

        Returns:
            The resolved secret value as a string.

        Raises:
            SecretResolutionError: If the secret cannot be resolved (not found,
                auth failure, backend unreachable, etc.).
        """
        ...


class SecretResolutionError(Exception):
    """Raised when a secret URI cannot be resolved.

    Attributes:
        uri: The original ``scheme://path#key`` string that failed.
        reason: Human-readable explanation of the failure.
    """

    def __init__(self, uri: str, reason: str) -> None:
        self.uri = uri
        self.reason = reason
        super().__init__(f"Failed to resolve secret '{uri}': {reason}")


# ---------------------------------------------------------------------------
# Resolver registry — populated lazily from entry points.
# ---------------------------------------------------------------------------

#: Discovered resolvers, or ``None`` until discovery has COMPLETED.
#:
#: ``None`` means "not discovered yet"; an empty dict means "discovered, and
#: there are none".  Those are different answers and the fast path below
#: distinguishes them with ``is not None`` -- which only works if the dict is
#: published when it is FINISHED.
_resolvers: Optional[Dict[str, 'SecretResolver']] = None

#: Serialises discovery.  Two threads may both find ``_resolvers is None``;
#: only one should pay for the entry-point scan, and neither may observe a
#: half-built registry.
_resolvers_lock = threading.Lock()


def _discover_secret_resolvers() -> Dict[str, 'SecretResolver']:
    """Discover secret resolvers from ``jaato.premium`` entry points.

    Looks for the ``secret_resolvers`` entry point which must return
    an iterable of :class:`SecretResolver` instances.

    Results are cached for the process lifetime, and the cache is
    **published only once it is complete** -- a concurrent caller either
    waits for discovery or sees the finished registry, never a partial one.
    Discovery runs at most once; losers of the race pay only the lock.

    Returns:
        Dict mapping URI scheme → resolver instance.  Empty means
        "discovered, and there are none" -- never "not discovered yet".
    """
    global _resolvers
    if _resolvers is not None:
        return _resolvers

    with _resolvers_lock:
        # Re-check: another thread may have completed discovery while this
        # one waited.
        if _resolvers is not None:
            return _resolvers
        discovered = _discover_secret_resolvers_uncached()
        # PUBLISHED ONLY WHEN COMPLETE.  This used to assign ``_resolvers =
        # {}`` and then fill it, so a second thread arriving during the
        # (slow) entry-point scan and premium import saw ``is not None``,
        # took the fast path, and got an EMPTY registry -- reporting
        # "(available: none)" and passing a literal ``pass://`` URI through
        # to a provider as its api_key.
        #
        # Observed on a cold daemon's first two CONCURRENT sessions, 2 for 2;
        # never on a warm one, because once populated the registry is never
        # empty again.  The condition is first-use concurrency, not elapsed
        # time.
        _resolvers = discovered
        return _resolvers


def _discover_secret_resolvers_uncached() -> Dict[str, 'SecretResolver']:
    """Do the discovery, into a LOCAL dict nothing else can observe.

    Split out so the caller can publish the result atomically.  Everything
    here is slow enough to matter: ``entry_points()`` scans installed
    distributions and ``ep.load()`` imports jaato-premium.
    """
    resolvers: Dict[str, 'SecretResolver'] = {}

    eps = importlib.metadata.entry_points()
    if sys.version_info >= (3, 12):
        matches = eps.select(group="jaato.premium", name="secret_resolvers")
    elif sys.version_info >= (3, 10):
        matches = [ep for ep in eps.select(group="jaato.premium")
                   if ep.name == "secret_resolvers"]
    else:
        matches = [ep for ep in eps.get("jaato.premium", [])
                   if ep.name == "secret_resolvers"]

    for ep in matches:
        try:
            provider_fn = ep.load()
            provider_fn_result = provider_fn()
            for resolver in provider_fn_result:
                for scheme in resolver.schemes:
                    if scheme in resolvers:
                        logger.warning(
                            "Duplicate secret resolver for scheme '%s' — "
                            "keeping first registered",
                            scheme,
                        )
                        continue
                    resolvers[scheme] = resolver
                    logger.debug("Registered secret resolver: %s://", scheme)
        except Exception:
            logger.warning(
                "Failed to load secret_resolvers entry point",
                exc_info=True,
            )

    if resolvers:
        logger.info(
            "Secret resolvers available for schemes: %s",
            ", ".join(sorted(resolvers.keys())),
        )

    return resolvers


def _resolve_secret_uri(value: str) -> str:
    """If *value* is a ``scheme://path[#key]`` URI with a registered resolver, resolve it.

    Returns the original string unchanged if:
    - It doesn't match the URI pattern.
    - The scheme is a standard network protocol (http/https/ws/wss/ftp/ftps) —
      these are literal URLs, not secret-resolver indirections (server
      0.6.57+).
    - The value contains ``${VAR}`` substitution markers — they're
      pending env-var expansion, not secret URIs (server 0.6.57+).
    - No resolver is registered for the scheme.

    Raises:
        SecretResolutionError: Propagated from the resolver on failure.
    """
    # Server 0.6.57+: skip values with unresolved ``${VAR}`` substitutions.
    # The env-file expansion runs downstream (``expand_variables`` in
    # http_client + general os.environ), so a literal-with-pending-var
    # passes through here unchanged and gets resolved at the right
    # point.  Pre-0.6.57 the secret-URI machinery returned the literal
    # ``http://127.0.0.1:${ANTIFRAUDE_PORT}`` as-is, blocking downstream
    # expansion and breaking handoff_test cascade post daemon-restart.
    if "${" in value:
        return value

    m = _SECRET_URI_RE.match(value)
    if not m:
        return value

    scheme = m.group('scheme')

    # Server 0.6.57+: standard network schemes are literal URLs.
    # ``https://search.maven.org`` matched the URI regex pre-0.6.57,
    # fired a "no resolver registered" warning that was just noise,
    # and the literal URL was returned unchanged anyway.  Skip the
    # whole code path now — these schemes never have a resolver
    # because they aren't secret-resolver indirections.
    if scheme in _NETWORK_SCHEMES:
        return value

    resolvers = _discover_secret_resolvers()
    resolver = resolvers.get(scheme)
    if resolver is None:
        logger.warning(
            "Secret URI '%s' uses scheme '%s' but no resolver is registered "
            "for it (available: %s). The literal URI will be used as the value, "
            "which is almost certainly wrong.",
            value, scheme, list(resolvers.keys()) or "none",
        )
        return value

    path = m.group('path')
    key = m.group('key')  # May be None

    try:
        return resolver.resolve(scheme, path, key)
    except SecretResolutionError:
        raise
    except Exception as exc:
        raise SecretResolutionError(value, str(exc)) from exc


def looks_like_unresolved_secret_uri(value: Any) -> bool:
    """Return True if *value* is a non-network ``scheme://`` secret-URI that
    was NOT resolved (it passed through literally because no resolver is
    registered for its scheme).

    Used to FAIL LOUD at the provider credential boundary: a *resolved* secret
    is a plain string, so a credential field still shaped like ``pass://...`` /
    ``vault://...`` means the providing resolver plugin (e.g. jaato-premium's
    ``secret_resolvers`` entry point) isn't installed.  ``_resolve_secret_uri``
    intentionally passes such values through (so non-provider consumers like
    ``service_connector`` can report "credential missing" with provenance), but
    a provider must NOT send a literal secret URI as an API key — that produces
    a confusing upstream 401.  Network schemes (http/ws/...), ``${VAR}``
    placeholders, and non-URI strings return False.
    """
    if not isinstance(value, str) or "${" in value:
        return False
    m = _SECRET_URI_RE.match(value)
    if not m:
        return False
    return m.group("scheme") not in _NETWORK_SCHEMES


# Regex for a MALFORMED single-colon secret reference: ``scheme:path`` with
# the ``//`` dropped (``pass:x`` instead of ``pass://x`` — the #1 secret-URI
# typo).  The ``(?!//)`` lookahead excludes the well-formed ``scheme://`` form
# (that's :data:`_SECRET_URI_RE`), so this matches ONLY the ``//``-less shape.
_MALFORMED_SECRET_URI_RE = re.compile(
    r'^(?P<scheme>[a-z][a-z0-9_+-]*):(?!//)(?P<rest>\S+)$'
)


def looks_like_malformed_secret_uri(value: Any) -> Optional[str]:
    """Return the scheme name if *value* is a MALFORMED single-colon secret
    reference for a REGISTERED resolver — e.g. ``pass:jaato/x`` when the user
    meant ``pass://jaato/x`` — else ``None``.

    The well-formed ``scheme://`` form is handled by
    :func:`looks_like_unresolved_secret_uri`.  This catches the common
    ``//``-dropped typo, which is invisible to the resolver machinery: a single
    colon fails :data:`_SECRET_URI_RE`, so the value is passed through
    literally and reaches the provider as a bearer token — producing exactly
    the confusing upstream 401 the ``//`` guard exists to prevent.

    Only a scheme that is an ACTIVELY REGISTERED resolver
    (:func:`_discover_secret_resolvers`) is flagged, so on a host without that
    resolver a literal ``word:word`` value is left untouched and there is no
    hardcoded scheme list.  Network schemes (http/ws/...) and ``${VAR}``
    placeholders return ``None``.
    """
    if not isinstance(value, str) or "${" in value:
        return None
    m = _MALFORMED_SECRET_URI_RE.match(value)
    if not m:
        return None
    scheme = m.group("scheme")
    if scheme in _NETWORK_SCHEMES:
        return None
    if scheme in _discover_secret_resolvers():
        return scheme
    return None


def reset_secret_resolvers() -> None:
    """Reset the cached secret resolvers (for testing).

    Takes the discovery lock so that *every* write to ``_resolvers`` happens
    under it -- a reader can then rely on seeing either ``None`` or a
    finished registry, with no third state.
    """
    global _resolvers
    with _resolvers_lock:
        _resolvers = None


# Valid values for the ``mode`` modifier knob.  ``discover`` (the
# default) leaves the plugin's discoverable tools deferred — the model
# finds them via ``list_tools`` / ``get_tool_schemas`` introspection.
# ``preload`` forces all of the plugin's tools (including discoverable
# ones) into the initial wire context.  The vocabulary mirrors
# ``ToolSchema.discoverability`` (``core`` | ``discoverable``): a
# deferred tool is precisely one the model *discovers*.
_PLUGIN_MODES = ("preload", "discover")


def _split_top_level_commas(s: str) -> List[str]:
    """Split ``s`` on commas that are NOT inside ``[...]`` brackets.

    Needed so ``"mode:preload, tools:[readFile,writeFile]"`` splits into
    two tokens (``mode:preload`` and ``tools:[readFile,writeFile]``)
    rather than four — the commas inside the bracketed list belong to
    the list, not the modifier separator.
    """
    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in s:
        if ch == '[':
            depth += 1
            cur.append(ch)
        elif ch == ']':
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur))
    return parts


def _parse_tool_allowlist(val: str) -> List[str]:
    """Parse a ``tools`` value into a list of tool names.

    Accepts the bracketed form ``[readFile, writeFile]`` and the bare
    single-value form ``readFile``.  Whitespace around names is
    stripped; empty entries are dropped.
    """
    val = val.strip()
    if val.startswith('[') and val.endswith(']'):
        val = val[1:-1]
    return [t.strip() for t in val.split(',') if t.strip()]


def parse_plugin_entry(entry: str) -> Tuple[str, bool, Optional[List[str]]]:
    """Parse a plugin entry that may carry a ``(...)`` modifier.

    Plugin names in profile ``plugins`` lists can carry an optional
    parenthesised modifier expressing two orthogonal knobs:

    - **mode** (``preload`` | ``discover``, default ``discover``) —
      whether to eagerly load all of the plugin's tools into the
      initial wire context (``preload``) or leave discoverable tools
      deferred (``discover``).
    - **tools** (an allow-list) — restrict the plugin to exactly the
      named tools; every other tool the plugin ships is dropped from
      this session's wire body AND its xgrammar grammar surface.  When
      absent, all of the plugin's tools are exposed (current default).

    Both knobs accept an **implicit** (positional, by token shape) form
    and an **explicit** (tagged ``key:value``) form, freely mixed:

    - a bare word (``preload`` / ``discover``) → the **mode**
    - a ``[...]`` token → the **tools** allow-list
    - a ``key:value`` token → an explicit tag (``mode:`` / ``tools:``)

    Token order is irrelevant.  An optional space before the
    parenthesis is accepted (``"file_edit (preload)"``).

    The bare legacy flag ``(preload)`` still parses — it is just the
    implicit-mode form.

    Args:
        entry: Plugin entry string, e.g. ``"cli"``, ``"file_edit(preload)"``,
            ``"file_edit([readFile])"``,
            ``"file_edit(mode:preload, tools:[readFile,writeFile])"``.

    Returns:
        Tuple of ``(plugin_name, is_preloaded, tool_allowlist)`` where
        ``tool_allowlist`` is ``None`` (all tools) or a list of allowed
        tool names.

    Raises:
        ValueError: when a modifier token is unrecognised (not a valid
            mode, not a ``[...]`` list, not a known ``key:value`` tag).

    Examples:
        >>> parse_plugin_entry("cli")
        ('cli', False, None)
        >>> parse_plugin_entry("file_edit(preload)")
        ('file_edit', True, None)
        >>> parse_plugin_entry("file_edit([readFile])")
        ('file_edit', False, ['readFile'])
        >>> parse_plugin_entry("file_edit(mode:preload, tools:[readFile,writeFile])")
        ('file_edit', True, ['readFile', 'writeFile'])
        >>> parse_plugin_entry("file_edit([readFile], preload)")
        ('file_edit', True, ['readFile'])
    """
    match = re.match(r'^(\w+)\s*(?:\((.*)\))?$', entry.strip())
    if not match:
        # Not a recognisable ``name`` or ``name(...)`` shape — treat the
        # whole string as a bare plugin name (lenient; downstream
        # expose_tool will surface an unknown-plugin error if invalid).
        return entry.strip(), False, None

    name = match.group(1)
    inner = match.group(2)
    if inner is None:
        return name, False, None

    preload = False
    tools: Optional[List[str]] = None
    for raw_token in _split_top_level_commas(inner):
        token = raw_token.strip()
        if not token:
            continue
        if ':' in token:
            key, _, val = token.partition(':')
            key = key.strip()
            val = val.strip()
            if key == 'mode':
                if val not in _PLUGIN_MODES:
                    raise ValueError(
                        f"invalid mode {val!r} in plugin entry {entry!r}; "
                        f"expected one of {_PLUGIN_MODES}"
                    )
                preload = (val == 'preload')
            elif key == 'tools':
                tools = _parse_tool_allowlist(val)
            else:
                raise ValueError(
                    f"unknown modifier key {key!r} in plugin entry "
                    f"{entry!r}; expected 'mode' or 'tools'"
                )
        elif token.startswith('['):
            tools = _parse_tool_allowlist(token)
        elif token in _PLUGIN_MODES:
            preload = (token == 'preload')
        else:
            raise ValueError(
                f"unrecognised modifier token {token!r} in plugin entry "
                f"{entry!r}; expected a mode ({_PLUGIN_MODES}), a "
                f"'[tool,...]' allow-list, or a 'mode:'/'tools:' tag"
            )
    return name, preload, tools


def parse_plugin_list(
    entries: List[str],
) -> Tuple[List[str], set, Dict[str, List[str]]]:
    """Parse a list of plugin entries into names, preload set, and scopes.

    Args:
        entries: List of plugin entry strings, possibly carrying ``(...)``
            modifiers (see :func:`parse_plugin_entry`).

    Returns:
        Tuple of ``(clean_plugin_names, preloaded_plugin_names_set,
        tool_scopes)`` where ``tool_scopes`` maps a plugin name to its
        allow-list of tool names.  Plugins without a ``tools`` modifier
        do not appear in ``tool_scopes`` (meaning: all tools exposed).
    """
    clean_names: List[str] = []
    preloaded: set = set()
    tool_scopes: Dict[str, List[str]] = {}
    for entry in entries:
        name, is_preloaded, tools = parse_plugin_entry(entry)
        clean_names.append(name)
        if is_preloaded:
            preloaded.add(name)
        if tools is not None:
            tool_scopes[name] = tools
    return clean_names, preloaded, tool_scopes


def expand_variables(
    value: Any,
    context: Optional[Dict[str, str]] = None,
    workspace_root_override: Optional[str] = None
) -> Any:
    """Expand ${variable} references in a value.

    Supports:
    - Environment variables: ${HOME}, ${USER}, ${PATH}
    - Context variables: ${projectPath}, ${workspaceRoot}, ${cwd}
    - Nested expansion in dicts and lists

    Args:
        value: Value to expand (string, dict, list, or other)
        context: Optional dict of context variables to expand
        workspace_root_override: Explicit workspace root to use instead of auto-detection.
            This is useful when the calling code knows the correct workspace root
            (e.g., from parent agent's config or environment).

    Returns:
        Value with variables expanded

    Examples:
        >>> expand_variables("${HOME}/projects", {})
        '/home/user/projects'

        >>> expand_variables({"path": "${projectPath}/.lsp.json"}, {"projectPath": "/app"})
        {'path': '/app/.lsp.json'}
    """
    if context is None:
        context = {}

    # Add default context variables
    # Use workspace_root_override if provided, otherwise auto-detect
    # via the per-task ContextVar (race-free) → os.environ fallback.
    from shared.session_context import get_workspace_root
    effective_cwd = workspace_root_override or get_workspace_root() or os.getcwd()
    workspace_root = _find_workspace_root(workspace_root_override)
    default_context = {
        'cwd': effective_cwd,
        'workspaceRoot': workspace_root,
        # Family IV (PR-217): sibling-of-workspace path that holds
        # framework-managed per-session state which cannot live inside
        # the project boundary (e.g. Eclipse / jdtls workspace metadata
        # — Eclipse Platform Core forbids workspace_location ⊆
        # project_location).  Naming convention:
        # ``<workspace.parent>/.<workspace.basename>-jdtls-state``
        # (dotfile-prefix lowest-collision pattern, suffix scopes the
        # data class so future framework-managed siblings can coexist).
        # Sibling stays under the same per-session AppArmor profile
        # → no data-leak (indexed source copies stay inside tenant
        # confinement boundary, just not inside project-root subtree).
        # Operator-facing template var symmetric with workspaceRoot.
        'jdtlsStateRoot': _compute_jdtls_state_root(workspace_root),
        'HOME': os.environ.get('HOME', ''),  # env: ambient — expanded as a template variable in profile/persona files
        'USER': os.environ.get('USER', ''),  # env: ambient — expanded as a template variable in profile/persona files
    }
    # Merge with provided context (provided takes precedence)
    effective_context = {**default_context, **context}

    if isinstance(value, str):
        return _expand_string(value, effective_context)
    elif isinstance(value, dict):
        return {k: expand_variables(v, context, workspace_root_override) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_variables(item, context, workspace_root_override) for item in value]
    else:
        return value


def _expand_string(s: str, context: Dict[str, str]) -> str:
    """Expand ``${variable}`` references and secret URIs in a string.

    Two-phase expansion:

    1. **Variable substitution** — ``${VAR}`` patterns are replaced in
       this order: *context* (caller-provided), then the session-scoped
       env (``get_session_env``, populated from the session's
       ``env_file``), then the daemon's ``os.environ``.  Undefined
       variables are kept as-is (``${UNKNOWN}`` stays literal).

       The session-env tier exists so that plugins reading expanded
       config values (e.g. service_connector resolving
       ``base_url: http://127.0.0.1:${SERVICE_PORT}``) see the variables
       defined in the session's ``.env`` file even when the daemon was
       started without those vars in its process env.

    2. **Secret URI resolution** — if the *fully expanded* string matches
       ``scheme://path[#key]`` and a :class:`SecretResolver` is registered
       for that scheme, the value is resolved to its plaintext secret.
       This phase is a no-op when no premium resolvers are installed.

    Secret resolution only applies when the **entire** string is a URI
    (e.g. a config value ``vault://secret/myapp#db_password``).  URIs
    embedded in a larger string are not resolved — use ``${VAR}``
    indirection for those cases.

    Args:
        s: String containing ``${variable}`` references or a secret URI.
        context: Dict of variable names to values.

    Returns:
        String with variables expanded and secrets resolved.

    Raises:
        SecretResolutionError: If a secret URI is recognised but the
            resolver fails (auth error, not found, backend unreachable).
    """
    # Imported lazily to avoid a circular import — session_context lives
    # at the package root and importing it at module load time would
    # pull in the JaatoSession TYPE_CHECKING ladder.
    from shared.session_context import get_session_env

    # Phase 1: ${VAR} expansion
    if '${' in s:
        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            # First check context, then session env, then process env.
            if var_name in context:
                return context[var_name]
            session_val = get_session_env(var_name)
            if session_val is not None:
                return session_val
            return os.environ.get(var_name, match.group(0))  # Keep original if not found

        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        s = re.sub(pattern, replace_var, s)

    # Phase 2: secret URI resolution (entire-string match only)
    return _resolve_secret_uri(s)


def _resolve_workspace_path(path: str) -> str:
    """Resolve a workspace path, handling relative paths.

    Relative paths (like ".") are resolved relative to the current
    working directory.

    Args:
        path: The workspace path (absolute or relative like ".").

    Returns:
        Absolute path to the workspace.
    """
    p = Path(path)
    if not p.is_absolute():
        from shared.session_context import get_workspace_root
        workspace = get_workspace_root() or os.getcwd()
        p = Path(workspace) / p
    return str(p.resolve())


def _compute_jdtls_state_root(workspace_root: str) -> str:
    """Compute the sibling jdtls state directory for *workspace_root*.

    Family IV (PR-217) naming convention.  Given a workspace at
    ``/foo/bar/cascade_smoke``, returns
    ``/foo/bar/.cascade_smoke-jdtls-state``.

    Why a sibling and not an in-workspace location: Eclipse Platform
    Core forbids the workspace metadata directory (jdtls's ``-data``
    arg) from being nested inside any imported project's root.  jdtls
    imports the workspace as a Maven project (pom.xml at root), so
    placing ``-data`` at ``<workspaceRoot>/.jaato/jdtls-data`` fires
    ``Invalid project description ... overlaps the workspace location``
    at ``ProjectsManager.importProjects``; no diagnostics flow.

    Why a sibling and not ``~/.cache`` / ``/var/lib/jaato/jdtls/...``:
    jdtls metadata is a transformed copy of source code (indexed
    classpath data, syntax trees, etc.).  Keeping it inside the
    tenant confinement boundary (per-session AppArmor profile) avoids
    a data-leak class where the indexed copy survives in shared
    system territory.  Sibling is under the same parent dir as the
    workspace; the per-session AppArmor profile grants r/w to both.

    Why dotfile-prefix + suffix: ``.<basename>-jdtls-state`` is:
    - hidden in ``ls`` by default (operators don't see framework state)
    - lowest-collision with operator-created directories (dotfile +
      framework-specific suffix is unlikely to be picked accidentally)
    - explicit about ownership (``-jdtls-state`` reads as
      framework-managed, distinct from any operator-named
      ``-jdtls`` / ``-state`` etc.)

    Args:
        workspace_root: Absolute workspace path.  Empty string returns
            empty string (callers that compute on a missing workspace
            get a no-op; the apparmor composer + LSP plugin both
            tolerate the empty case).

    Returns:
        Absolute path to the sibling jdtls state directory, or empty
        string when ``workspace_root`` is falsy.
    """
    if not workspace_root:
        return ""
    p = Path(workspace_root).resolve()
    return str(p.parent / f".{p.name}-jdtls-state")


def _find_workspace_root(override: Optional[str] = None) -> str:
    """Find the workspace root by looking for .git directory.

    Priority order:
    1. Explicit override parameter
    2. JAATO_WORKSPACE_ROOT environment variable
    3. workspaceRoot environment variable (from .env file)
    4. Search for .git or .jaato directory from cwd

    Note: Relative paths (like ".") are resolved relative to cwd.

    Args:
        override: Explicit workspace root path to use (takes precedence).

    Returns:
        Path to workspace root, or cwd if not found
    """
    # Priority 1: Explicit override
    if override:
        return _resolve_workspace_path(override)

    # Priority 2: per-task workspace root (ContextVar, race-free) →
    # JAATO_WORKSPACE_ROOT env var fallback for daemon-startup callers.
    from shared.session_context import get_workspace_root
    env_root = get_workspace_root()
    if env_root:
        return _resolve_workspace_path(env_root)

    # Priority 3: workspaceRoot environment variable (common in .env files)
    env_workspace_root = os.environ.get('workspaceRoot')
    if env_workspace_root:
        return _resolve_workspace_path(env_workspace_root)

    # Priority 4: Search for .git or .jaato directory
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / '.git').exists():
            return str(parent)
        if (parent / '.jaato').exists():
            return str(parent)
    return str(current)


def detect_workspace_tech_stack(workspace_path: str) -> str:
    """Detect the primary technology stack of a workspace by scanning for marker files.

    Checks for common build/config files at the workspace root to determine
    the project's language and build system.

    Args:
        workspace_path: Absolute path to the workspace root.

    Returns:
        Concise summary string (e.g., "Java project (Maven - pom.xml detected)")
        or empty string if nothing detected.
    """
    root = Path(workspace_path)
    detections = []

    # Check marker files in priority order
    markers = [
        ("pom.xml", "Java project (Maven - pom.xml detected)"),
        ("build.gradle", "Java/Kotlin project (Gradle - build.gradle detected)"),
        ("build.gradle.kts", "Kotlin project (Gradle KTS - build.gradle.kts detected)"),
        ("Cargo.toml", "Rust project (Cargo - Cargo.toml detected)"),
        ("go.mod", "Go project (go.mod detected)"),
        ("package.json", "JavaScript/TypeScript project (Node.js - package.json detected)"),
        ("pyproject.toml", "Python project (pyproject.toml detected)"),
        ("setup.py", "Python project (setup.py detected)"),
        ("requirements.txt", "Python project (requirements.txt detected)"),
        ("Gemfile", "Ruby project (Gemfile detected)"),
        ("composer.json", "PHP project (Composer - composer.json detected)"),
    ]

    for filename, description in markers:
        if (root / filename).exists():
            detections.append(description)

    # Check glob patterns for .NET
    if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
        detections.append("C#/.NET project (.sln/.csproj detected)")

    if not detections:
        return ""

    return "; ".join(detections)


def expand_plugin_configs(
    plugin_configs: Dict[str, Dict[str, Any]],
    context: Optional[Dict[str, str]] = None,
    workspace_root_override: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Expand variables in all plugin configurations.

    Args:
        plugin_configs: Dict of plugin name -> config dict
        context: Optional context variables (e.g., projectPath)
        workspace_root_override: Explicit workspace root to use instead of auto-detection.
            If provided, ${workspaceRoot} will expand to this value.

    Returns:
        Plugin configs with all variables expanded

    Example:
        >>> configs = {
        ...     "lsp": {"config_path": "${projectPath}/.lsp.json"},
        ...     "mcp": {"config_path": "${projectPath}/.mcp.json"}
        ... }
        >>> expand_plugin_configs(configs, {"projectPath": "/app"})
        {'lsp': {'config_path': '/app/.lsp.json'}, 'mcp': {'config_path': '/app/.mcp.json'}}
    """
    return expand_variables(plugin_configs, context, workspace_root_override)


@dataclass
class CacheProfileConfig:
    """Prompt-cache configuration for a profile — the common `cache:` field.

    The cross-provider default. Caching is delivered three different ways
    (Anthropic breakpoints, Google ``CachedContent``, OpenRouter's gateway
    annotation) with three different knob spellings, layers and defaults,
    so before this field a profile author had to know which mechanism
    their provider used in order to turn caching on at all. See
    ``docs/design/model-tier-prompt-cache.md`` §7.

    This sets the default; ``plugin_configs.<provider>`` overrides it for
    mechanism-specific tuning. More specific wins, which is the same
    child-wins rule ``resolve_provider_extra`` already applies to
    provider extras — the common field is a layer BENEATH an existing
    one, not a new precedence concept.

    Attributes:
        enabled: ``"auto"`` (default), ``True`` or ``False``. ``auto``
            means "leave the provider's own default alone", and on a
            provider that cannot cache at all it is a no-op rather than
            an error — ``ProviderCapabilities.prompt_caching`` is what
            makes that well-defined.
        ttl: Cache lifetime in the ``5m`` / ``1h`` vocabulary. Translated
            per provider (Google wants a duration in seconds).
        history: Cache the conversation prefix, not only system+tools.
            Honoured by providers whose mechanism can place a history
            breakpoint; ignored by the others.
    """
    enabled: Any = "auto"
    ttl: str = "5m"
    history: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheProfileConfig':
        """Build from a profile's ``cache:`` block.

        Raises:
            ValueError: on an unusable value, rather than silently
                falling back to a default. A cache knob that is quietly
                ignored is the exact failure §4 documents.
        """
        enabled = data.get('enabled', "auto")
        if isinstance(enabled, str):
            lowered = enabled.strip().lower()
            if lowered == "auto":
                enabled = "auto"
            elif lowered in ("true", "yes", "on"):
                enabled = True
            elif lowered in ("false", "no", "off"):
                enabled = False
            else:
                raise ValueError(
                    f"cache.enabled must be auto/true/false, got {enabled!r}")
        elif not isinstance(enabled, bool):
            raise ValueError(
                f"cache.enabled must be auto/true/false, got {enabled!r}")

        ttl = str(data.get('ttl', "5m")).strip().lower()
        if ttl not in VALID_CACHE_TTLS:
            raise ValueError(
                f"cache.ttl must be one of {sorted(VALID_CACHE_TTLS)}, "
                f"got {ttl!r}")

        history = data.get('history', True)
        if not isinstance(history, bool):
            raise ValueError(
                f"cache.history must be a boolean, got {history!r}")

        return cls(enabled=enabled, ttl=ttl, history=history)


#: Profile ``trace:`` key -> the env var it seeds.  ONE mapping, read by
#: :meth:`TraceProfileConfig.from_dict` (key validation), by
#: :meth:`TraceProfileConfig.as_env` (the export) and by the env-scope
#: catalog's ``typed_key`` entries, so a rename cannot desynchronise them.
TRACE_ENV_VARS: Dict[str, str] = {
    "session_log": "JAATO_TRACE_LOG",
    "provider_log": "JAATO_PROVIDER_TRACE",
}

#: Values that mean "on" to a human and nothing at all to a path reader.
#: An author who writes one of these into ``trace:`` is reaching for a
#: switch; there is no file they could plausibly have meant, so the value
#: is refused rather than turned into a file with that name.
_TRACE_BOOLEAN_TOKENS = frozenset({
    "0", "1", "true", "false", "yes", "no", "on", "off", "y", "n",
    "enable", "enabled", "disable", "disabled", "none", "null",
})


@dataclass
class TraceProfileConfig:
    """Diagnostic trace-log paths for a profile -- the ``trace:`` block.

    THE KNOB THAT MADE THE CASE FOR TYPED KEYS (issue #775).  Both values
    are *paths*, and before this block the only per-session route to them
    was the profile's ``env:`` map::

        env:
          JAATO_PROVIDER_TRACE: "1"      # accepted, and catastrophic

    ``env`` is ``Dict[str, str]`` and ``"1"`` is a valid string, so
    nothing rejected it -- and nothing was in a position to.  Every
    session then wrote its provider trace to a file literally named ``1``,
    including eval-arm workspaces, contaminating the very trees a
    comparative judge was diffing.  The failure is silent on both sides:
    the trace is written, the run completes, and the contamination is
    visible only by listing the arm directories afterwards.

    WHAT THIS BLOCK REJECTS, AND WHAT IT DELIBERATELY DOES NOT.  The
    defect in ``"1"`` is not that it is relative -- a *relative* trace
    path is the supported per-session idiom, resolved against
    ``JAATO_WORKSPACE_ROOT`` by ``jaato_sdk.trace._resolve_trace_file`` so
    each session gets its own file in its own workspace.  Rejecting
    relative paths here would break that.  The defect is that ``"1"`` is a
    *boolean written into a path field*: the author reached for a switch,
    and a string-typed map had no way to say so.  So this block refuses
    the boolean vocabulary (:data:`_TRACE_BOOLEAN_TOKENS`), a value that
    names a directory rather than a file, and anything that is not a
    non-empty string -- and passes every real path through untouched.

    The env vars remain the lower-precedence default -- a workspace
    ``.env`` or an ``env:`` entry still works -- and this block simply
    outranks them (see ``JaatoServer._resolve_session_env``).  Nothing
    downstream reads the block: it is a validated *producer* of the two
    env vars the framework already reads, which is why promoting a knob
    costs no reader changes.

    Attributes:
        session_log: Path for the framework's own session event trace
            (``JAATO_TRACE_LOG``).  Absolute, or relative to the session
            workspace.
        provider_log: Path for the provider request/response trace
            (``JAATO_PROVIDER_TRACE``).  Same resolution.
    """

    session_log: Optional[str] = None
    provider_log: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceProfileConfig':
        """Build from a profile's ``trace:`` block.

        Raises:
            ValueError: on a non-mapping block, an unknown key, a
                non-string or empty value, a boolean-shaped token, or a
                value that names a directory.  Failing loud is the whole
                point -- a trace path that is quietly wrong produces a
                file nobody looks for and a diagnosis nobody can make.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"trace: must be a mapping, got {type(data).__name__}")

        unknown = set(data) - set(TRACE_ENV_VARS)
        if unknown:
            raise ValueError(
                f"trace: unknown key(s) {sorted(unknown)}. "
                f"Allowed: {sorted(TRACE_ENV_VARS)}")

        values: Dict[str, Optional[str]] = {}
        for key in TRACE_ENV_VARS:
            value = data.get(key)
            values[key] = (None if value is None
                           else _validate_trace_path(key, value))
        return cls(**values)

    def as_env(self) -> Dict[str, str]:
        """The env vars this block seeds, omitting the keys left unset."""
        return {
            TRACE_ENV_VARS[key]: value
            for key, value in (("session_log", self.session_log),
                               ("provider_log", self.provider_log))
            if value
        }


def _validate_trace_path(key: str, value: Any) -> str:
    """Return *value* as a usable trace path, or raise ``ValueError``.

    Split out of :meth:`TraceProfileConfig.from_dict` so the rule has one
    home and one set of tests; the two keys are validated identically.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"trace.{key} must be a non-empty string path, got {value!r}")
    text = value.strip()

    if text.lower() in _TRACE_BOOLEAN_TOKENS:
        raise ValueError(
            f"trace.{key}={value!r} is a switch, not a path. This knob is "
            f"the FILE the trace is written to -- setting it to {value!r} "
            f"through the untyped `env:` map is what produced a file "
            f"literally named {value!r} in every session's workspace "
            f"(issue #775). Give a path: an absolute one is shared by "
            f"every session using this profile, a relative one resolves "
            f"against each session's own workspace "
            f"(e.g. .jaato/logs/{key}.jsonl).")

    if text.endswith(("/", "\\")) or os.path.isdir(text):
        raise ValueError(
            f"trace.{key}={value!r} names a directory; this knob is the "
            f"trace FILE. Append a filename "
            f"(e.g. {text.rstrip('/')}/{key}.jsonl).")

    return text


@dataclass
class GCProfileConfig:
    """Garbage collection configuration for a profile.

    Defines the GC strategy and its configuration for a subagent or main agent.

    Attributes:
        type: GC strategy type ('truncate', 'summarize', 'hybrid', 'budget').
        threshold_percent: Trigger GC when context usage exceeds this percentage.
        target_percent: Target usage after GC (default: 60.0).
        pressure_percent: When PRESERVABLE can be touched (0 = continuous mode).
        preserve_recent_turns: Number of recent turns to always preserve.
        notify_on_gc: Whether to inject a notification into history after GC.
        summarize_middle_turns: For hybrid strategy, number of middle turns to summarize.
        max_turns: Trigger GC when turn count exceeds this limit.
        plugin_config: Additional plugin-specific configuration.
    """
    type: str = "truncate"
    threshold_percent: float = 80.0
    target_percent: float = 60.0
    pressure_percent: Optional[float] = 90.0  # 0 or None = continuous mode
    preserve_recent_turns: int = 5
    notify_on_gc: bool = True
    summarize_middle_turns: Optional[int] = None  # For hybrid strategy
    max_turns: Optional[int] = None
    plugin_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def continuous_mode(self) -> bool:
        """True if continuous GC is enabled (pressure_percent is 0 or None)."""
        return not self.pressure_percent

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GCProfileConfig':
        """Create GCProfileConfig from a dictionary."""
        return cls(
            type=data.get('type', 'truncate'),
            threshold_percent=data.get('threshold_percent', 80.0),
            target_percent=data.get('target_percent', 60.0),
            pressure_percent=data.get('pressure_percent', 90.0),
            preserve_recent_turns=data.get('preserve_recent_turns', 5),
            notify_on_gc=data.get('notify_on_gc', True),
            summarize_middle_turns=data.get('summarize_middle_turns'),
            max_turns=data.get('max_turns'),
            plugin_config=data.get('plugin_config', {}),
        )


@dataclass
class CompletionProcessor:
    """A profile-declared completion processor.

    Replaces the prior split between ``completion_artifacts``
    (renderers that produce files) and ``completion_validators`` (kb
    Python that returns error lists) — both surfaces collapsed into
    one ``completion_processors`` config field as of server 0.6.125+.
    The two had the same plumbing (kb Python under
    ``.jaato/scripts/``, loaded via ``script_loader``, run after
    ``jsonschema.validate`` passes, block completion on failure); the
    split was an artifact of incremental shipping (PR-138 added
    validators while artifacts already existed and supported a
    "validator-as-renderer" mode).

    Each kb processor module exposes one or both of these top-level
    callables — the framework probes for which symbols are present
    and dispatches accordingly:

    - ``render(payload: dict, context: RenderContext) -> str | bytes``
      Produces output content.  When the processor entry declares an
      ``output:`` path template, the returned bytes are written to
      disk (atomic ``.tmp`` + ``rename``).  When ``output:`` is
      omitted, the return is logged for audit but not persisted —
      "validator-as-renderer" use case.

    - ``validate(payload: dict, context: RenderContext) -> list[str]``
      Returns a list of error strings.  Empty list → pass.  Non-empty
      → completion blocked per the entry's ``on_error`` policy.  Has
      access to ``context.tool_calls`` — the pre-computed ledger of
      every function_call + function_response in the session, paired
      by call_id.  Use to cross-check payload claims against actual
      tool outcomes (e.g. agent claimed file X rendered, but the
      corresponding ``renderTemplateToFile`` call returned an error).

    Both functions can be present in one module — useful when a
    single processor both writes an audit record AND checks payload
    consistency.

    Attributes:
        script: kb Python file path resolved through the standard
            ``script_loader`` tier (absolute → ``<config_root>/<path>``
            → ``~/.jaato/<path>``).
        name: Optional stable identifier for this processor, used by an
            inheriting profile's ``suppress_inherited_processors`` to
            decline it by name.  When ``None`` the ``script`` path is the
            identity (a suppression entry matches either).  Declare one on
            a base profile's processors when you want the identity to
            survive moving the script file.  Ignored at runtime.
        output: Optional output file path with simple ``{field}``
            templating.  When set, the ``render`` symbol's return is
            written to this path.  Substitutes from the payload first,
            then ``agent_params``, then session-derived values
            (``case_id``, ``agent_id``, ``workspace_path``).  Relative
            paths resolve under ``workspace_path``.  ``None`` means the
            processor runs for side-effect / validation only — useful
            for ``validate``-only processors or ``render`` calls that
            consult state without producing files.
        on_error: How a failure (script raised, file write failed,
            ``validate`` returned non-empty list) is surfaced.
            ``"fail_completion"`` returns a validation_failed shape to
            the model so it retries; ``"warn"`` logs and lets the
            completion proceed (the operator can clean up after).
            Default ``"fail_completion"``.
        description: Optional human-readable note on what this
            processor does and why it's wired in.  Ignored at runtime;
            consumed by docs / introspection tooling.
        phase: WHEN this processor runs (server 0.6.199+).
            ``"finalization"`` (default) — runs at ``signal_completion``
            only, exactly as every processor did before this field
            existed: ``render`` writes output, ``validate`` blocks
            completion on errors.  ``"completeness"`` — runs DURING
            ``prepare_completion`` (gated: only once the schema-required
            floor is met, so it fires ~once near the end, not per
            field) and its ``validate`` return participates in the
            COMPOSITE ``is_complete`` verdict via the ``incomplete[]``
            channel of :class:`jaato_sdk.cascade_authoring.ProcessorResult`.
            A completeness processor's job is SEMANTIC done-ness:
            "does the accumulated payload have every field the
            downstream cascade stages actually consume for THIS run?"
            — distinct from the schema's STRUCTURAL floor (required[]).
            It must be cheap (pure payload/context inspection, no
            subprocess / Maven / LSP) because it runs mid-accumulation.
            Its ``incomplete[]`` entries gate ``is_complete`` to False
            and surface to the model as neutral "still needed" guidance
            (no retry penalty); its ``errors[]`` still reject as usual.
    """
    script: str
    output: Optional[str] = None
    on_error: str = "fail_completion"
    description: Optional[str] = None
    phase: str = "finalization"
    name: Optional[str] = None

    @property
    def identity(self) -> str:
        """What ``suppress_inherited_processors`` matches this entry by.

        The declared ``name`` when there is one, else the ``script``
        path.  Used for the "no inherited processor matched" diagnostic;
        the match itself accepts EITHER form so a child can suppress by
        script path even when the parent named the processor.
        """
        return self.name or self.script


@dataclass
class SubagentProfile:
    """Configuration profile for a subagent.

    Defines what tools and capabilities a subagent has access to,
    allowing the parent model to delegate specialized tasks.

    Attributes:
        name: Unique identifier for this subagent profile.
        description: Human-readable description of what this subagent does.
        plugins: List of plugin names to enable for this subagent (clean names,
            ``(preload)`` suffixes stripped during parsing).
        preloaded_plugins: Set of plugin names that should bypass deferred tool
            loading — all their tools (including discoverable) are loaded into
            the initial context. Derived from ``(preload)`` annotations in the
            raw ``plugins`` list during profile parsing.
        plugin_configs: Per-plugin configuration overrides.
        system_instructions: **Deprecated.** Use agents (``.jaato/agents/``) instead.
            When an agent is specified via ``--agent``, its rendered markdown
            replaces this field.  Profiles should contain runtime config only.
        model: Optional model override (uses parent's model if not specified).
        provider: Optional provider override (e.g., 'anthropic', 'google_genai').
                  Allows subagents to use a different provider than the parent.
        max_turns: Maximum conversation turns before returning (default: 10).
        gc: Optional garbage collection configuration for this subagent.
        trace: Optional diagnostic trace-log paths (``session_log`` /
            ``provider_log``).  The typed, validated sibling of
            ``JAATO_TRACE_LOG`` / ``JAATO_PROVIDER_TRACE``, which remain
            the lower-precedence default.  Seeded into the session env by
            ``JaatoServer._resolve_session_env`` -- see
            :class:`TraceProfileConfig` for the failure that motivated it
            (issue #775).
        env: Session-scoped environment variables for this profile.
            Values support ``${VAR}`` expansion and secret URI resolution
            (e.g. ``vault://secret/myapp#db_password``).  For main sessions
            these are merged into ``JaatoServer._session_env``; for subagents
            they are applied to ``os.environ`` for the duration of the
            subagent thread and restored on exit.  Never leaks to other
            sessions or agents.
        inherits: Optional list of parent profile names. When set, this
            profile inherits fields from its parents. Resolved during
            ``discover_profiles()`` — after resolution, ``inherits`` is
            cleared and the profile is fully flattened.
        completion_payload_schema: Optional JSON Schema constraining the
            ``payload`` argument of ``signal_completion``. Either an inline
            dict or a string path resolved via the
            ``.jaato/completion_schemas/`` tier (absolute → workspace →
            home). When set, ``signal_completion``'s parameters carry the
            schema so providers enforce it at sampling time and
            ``LifecycleTools`` validates the payload server-side before
            emitting ``AgentCompletedEvent``. When ``None``, the legacy
            ``summary: str`` parameter is used. Inheritance follows the
            scalar-override rule (parents must agree or child overrides).
        runtime_limits: Optional per-session resource consumption caps —
            memory, PIDs, CPU weight, tool wall-clock timeout, and stdout
            cap.  Orthogonal to sandboxing (AppArmor): answers "how much
            can this session consume?" rather than "what can it touch?".
            The kernel-enforceable subset (memory, PIDs, CPU weight) is
            applied via cgroup v2 by ``server.cgroups.CgroupsManager``;
            the rest is read by the CLI/interactive_shell plugins at
            tool-call time.  ``None`` means "no limits" (host defaults).
        budget_control: Optional multi-dimensional budget ceilings
            (``limits``: usd / tokens / seconds / tool_calls / turns) plus
            a ``degrade`` ladder applied as those ceilings are
            approached.  Where ``runtime_limits`` caps HOST resources,
            this caps agent ECONOMICS.  A degrade rung rebinds
            ``model_tiers`` bindings (a brownout) rather than moving the
            agent between tiers, so the model's cognitive role is
            untouched.  Inheritance: ``limits`` min-wins (a child may only
            tighten), ``degrade`` is scalar-override.  ``None`` means
            "unbudgeted".
    """
    name: str = field(metadata={
        "description": "Unique profile identifier (the <agent>.yaml stem)."})
    description: str = field(metadata={
        "description": "Human-readable summary of what this (sub)agent does."})
    plugins: List[str] = field(default_factory=list, metadata={
        "description": "Plugin names to enable. `name(preload)` bypasses "
        "deferred tool-loading (all its tools, incl. discoverable, enter the "
        "initial context); `name(tools:[a,b])` scopes which tools are exposed "
        "(see tool_scopes)."})
    preloaded_plugins: set = field(default_factory=set, metadata={
        "description": "DERIVED from `(preload)` annotations in `plugins` during "
        "parsing — not set directly."})
    # Per-plugin tool allow-lists derived from ``tools:[...]`` modifiers
    # in the raw ``plugins`` list (see :func:`parse_plugin_entry`).  Maps
    # plugin name → list of tool names to expose; every other tool the
    # plugin ships is dropped from this session's wire body AND its
    # xgrammar grammar surface.  A plugin absent from this dict exposes
    # all its tools (the default).  The filter is applied **per session**
    # in ``JaatoSession`` (mirroring the ``_tool_plugins`` plugin-level
    # filter) — it never mutates the shared registry, so sibling
    # subagents on the same runtime are unaffected.
    #
    # CAVEAT for profile authors: a tool dropped here is invisible to the
    # model — its schema never reaches the wire.  If an agent persona (or
    # cross-persona instruction) names a tool that the allow-list omits,
    # the model will be told to use a tool it cannot see.  Keep the
    # allow-list and the persona's referenced tools in sync.
    tool_scopes: Dict[str, List[str]] = field(default_factory=dict, metadata={
        "description": "Per-plugin tool allow-lists (plugin -> [tool names to "
        "expose]); every other tool that plugin ships is dropped from this "
        "session's wire + grammar. Absent plugin = all its tools. Keep in sync "
        "with the persona's referenced tools."})
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict, metadata={
        "description": "Per-plugin config overrides (plugin name -> config "
        "dict), e.g. plugin_configs.<provider>.api_params / .extra_body, "
        "plugin_configs.permission.policy."})
    system_instructions: Optional[str] = field(default=None, metadata={
        "description": "DEPRECATED — use agents (.jaato/agents/<name>.md) "
        "instead; an `--agent`'s rendered markdown replaces this. Profiles "
        "should carry runtime config only."})
    # When True, drop the framework's BASE instructions layer (the
    # "Principle 1: Transparency Mandate" and other always-on framework
    # instructions) from this profile's system prompt.  Plugin-contributed
    # instructions and the agent's own ``system_instructions`` are still
    # included.  Useful for simple goal-focused agents (body-wired echo
    # specialists, narrow-scope narrators) that don't benefit from the
    # framework's general-purpose guidance and would rather have those
    # tokens back for their actual work — typical savings 3-5k tokens
    # per turn, which can be the difference between fitting in a small
    # model's context window and triggering aggressive GC.  Defaults to
    # ``False`` (full framework instructions).
    suppress_base_instructions: Any = field(default=False, metadata={
        "description": "Drop framework-injected instruction layers from the system "
        "prompt (plugin + agent instructions always kept).  `true` drops the disk "
        "BASE layer (.jaato/instructions/*.md) AND the framework constants "
        "(task-completion, parallel guidance, turn-summary); the security "
        "untrusted-content boundary is kept.  A dict gives granular control, e.g. "
        "`{disk: true, constants: true, security: false}` (absent key = keep).  "
        "Saves ~3-5k tokens/turn for narrow goal-focused agents.  Default False.  "
        "Normalized to a canonical frozenset of piece names in __post_init__."})
    model: Optional[str] = field(default=None, metadata={
        "description": "Model override (uses the parent's model if unset). "
        "Silently ignored when model_tiers is non-empty."})
    provider: Optional[str] = field(default=None, metadata={
        "description": "Provider override (e.g. anthropic, nebius, vllm). A "
        "profile binds exactly one provider + model."})
    max_turns: int = field(default=10, metadata={
        "description": "Max conversation turns before the (sub)agent returns."})
    cache: Optional['CacheProfileConfig'] = field(default=None, metadata={
        "description": "Prompt-cache defaults, cross-provider. "
        "{enabled: auto|true|false, ttl: 5m|1h, history: bool}. "
        "'auto' leaves the provider's own default alone and is a no-op on "
        "a provider that cannot cache. plugin_configs.<provider> overrides "
        "this for mechanism-specific tuning (more specific wins)."})
    trace: Optional['TraceProfileConfig'] = field(default=None, metadata={
        "description": "Diagnostic trace-log paths: {session_log, provider_log}. "
        "Typed sibling of the JAATO_TRACE_LOG / JAATO_PROVIDER_TRACE env vars, "
        "which stay the lower-precedence default (this block outranks them). "
        "Absolute = one shared file; relative = one file per session, resolved "
        "against the workspace. Refuses a switch written into a path field -- "
        "`env: {JAATO_PROVIDER_TRACE: \'1\'}` is a valid str and wrote every "
        "session\'s trace to a file named `1` (#775)."})
    gc: Optional[GCProfileConfig] = field(default=None, metadata={
        "description": "Garbage-collection strategy + thresholds for this "
        "session (type + threshold_percent / target / preserve_recent_turns). "
        "None = framework default."})
    env: Dict[str, str] = field(default_factory=dict, metadata={
        "description": "Session-scoped env vars (support ${VAR} expansion + "
        "secret URIs, e.g. vault://secret/app#key). Applied for the session's "
        "duration, never leak to other sessions."})
    inherits: Optional[List[str]] = field(default=None, metadata={
        "description": "Parent profile names to inherit from (tier-1 _base_* / "
        "profile-set composition). Resolved + flattened at discover_profiles(); "
        "cleared after."})
    completion_payload_schema: Optional[Union[str, Dict[str, Any]]] = field(
        default=None, metadata={
        "description": "JSON Schema constraining signal_completion's `payload` "
        "(inline dict or a path under .jaato/completion_schemas/). Carried on "
        "the tool so providers enforce it at sampling time + LifecycleTools "
        "validates server-side. None = legacy `summary: str`."})
    spawn_payload_schema: Optional[Union[str, Dict[str, Any]]] = field(
        default=None, metadata={
        "description": "JSON Schema constraining the spawn-time payload "
        "(input boundary), mirror of completion_payload_schema. Inline dict or "
        "a .jaato/completion_schemas/ path."})
    # Unified completion-processor surface (server 0.6.125+).  Replaces
    # the prior split between ``completion_artifacts`` (renderers that
    # produce files) and ``completion_validators`` (kb Python that
    # returns error lists) — same plumbing under the hood (kb Python
    # under ``.jaato/scripts/processors/``, loaded via
    # ``script_loader``, run after ``jsonschema.validate`` passes,
    # block completion on failure).  See :class:`CompletionProcessor`
    # docstring for the full kb author contract (probe-by-symbol:
    # ``render`` and/or ``validate``).  Inheritance concatenates
    # parent + child — each processor is independent and all fire.
    completion_processors: List[CompletionProcessor] = field(
        default_factory=list, metadata={
        "description": "kb Python hooks (.jaato/scripts/processors/, probed for "
        "`render` and/or `validate`) run after jsonschema.validate passes; a "
        "validator's error list blocks completion, a renderer produces files. "
        "Inheritance concatenates parent + child; all fire. To drop ONE "
        "inherited processor, name it in `suppress_inherited_processors` — "
        "an empty list here adds nothing, it does not clear the parents'."})
    # The one opt-out of ``completion_processors``' concatenation
    # (#791).  Every other inherited key can be scoped down by the
    # child somehow — a scalar by replacing it, a dict per key — but
    # concatenation only ever grows, so a child whose stage genuinely
    # completes differently had no move except to stop inheriting, and
    # silently lose ``budget_control`` / ``max_turns`` /
    # ``runtime_limits`` / ``env`` / ``plugin_configs`` with it.
    #
    # Scoped deliberately narrow:
    #   - BY NAME, never "drop them all", so a base that later adds a
    #     second processor does not silently re-enable the one this
    #     child declined;
    #   - applies only to what the PARENTS contributed — the child's own
    #     ``completion_processors`` are its to edit directly;
    #   - an entry matching no inherited processor is a profile-load
    #     ERROR, not a silent no-op, because a stale suppression means
    #     the base moved or renamed the processor and this child is
    #     once again running one it declared it did not want;
    #   - NOT itself inherited: it is consumed at the merge that
    #     resolves this profile, so a grandchild never re-applies (and
    #     never trips over) an ancestor's suppression.
    suppress_inherited_processors: List[str] = field(
        default_factory=list, metadata={
        "description": "Names (or script paths) of INHERITED "
        "completion_processors this profile declines — the only way to scope "
        "down a key that otherwise only concatenates. Matches a parent "
        "entry's `name` or its `script`. An entry matching nothing is a load "
        "error. Not inherited further; only meaningful alongside `inherits`."})
    runtime_limits: Optional[RuntimeLimits] = field(default=None, metadata={
        "description": "Per-session resource caps (memory, PIDs, CPU weight, "
        "tool wall-clock timeout, stdout). 'How much can it consume' — "
        "orthogonal to AppArmor's 'what can it touch'. None = host defaults."})
    # Per-turn model-tier config.  Empty dict means "single-model
    # mode" — the framework falls back to env vars (JAATO_TIER_*) at
    # session-init time, and from there to single-model behavior using
    # ``model``.  When non-empty, ``model`` is silently ignored (with a
    # warning at load time) because the active model is selected per
    # turn from ``model_tiers[<active_tier>]``.
    #
    # Single-level dict mixing tier→model entries (keys in
    # ``VALID_TIER_NAMES``) with reserved control keys (``initial`` /
    # ``fallback``).  Each tier entry is either a model-name string or a
    # dict with ``model`` (required) plus optional ``provider`` (tiers may
    # name different ones) and ``description`` (prose the MODEL reads as
    # that tier's bullet in the ``enter_tier`` tool).  See
    # ``shared/model_tiers.py`` for the resolver and validation, and
    # ``project_backlog_per_turn_model`` for the full design.
    model_tiers: Dict[str, Any] = field(default_factory=dict, metadata={
        "description": "Per-turn model-tier selection. Single-level dict "
        "mapping a tier key to a model (a model-name string, or "
        "{model (required), provider (optional; tiers may span providers), "
        "description (optional; prose the model reads as that tier's bullet "
        "in the enter_tier tool — default is the framework's own wording for "
        "the tier name)}), plus the reserved control keys initial / fallback. "
        "The enter_tier tool advertises ONLY the declared tiers. "
        "Non-empty silently ignores `model` (warns at load) — the active model "
        "is picked per turn from model_tiers[<active_tier>]. Empty = "
        "single-model mode (falls back to the JAATO_TIER_* env vars, then "
        "`model`)."})
    # Budget ceilings + the degradation ladder applied as they're
    # approached.  Distinct from ``runtime_limits`` (host resources:
    # memory / pids / cpu) — this caps agent ECONOMICS.  A ``degrade``
    # rung REBINDS ``model_tiers`` entries rather than moving the agent
    # between tiers, because tier labels are a cognitive/role axis with
    # no inherent cost ordering (see shared/model_tiers.py).  Parsed +
    # validated by ``shared/budget_control.py``; full design in
    # ``docs/design/budget-control-degradation.md``.
    budget_control: Optional[BudgetControlConfig] = field(
        default=None, metadata={
        "description": "Multi-dimensional budget ceilings + graceful degradation. "
        "`limits`: usd / tokens / seconds / tool_calls / turns (omit a dimension to "
        "leave it unbounded). `degrade`: an ordered ladder of rungs; a rung fires "
        "when ANY dimension crosses its `at` percentage and overlays new model_tiers "
        "bindings (a brownout — the tier vocabulary and the model's role are "
        "untouched, only the model each tier points at changes) and/or takes a "
        "terminal action (finalize / abort / escalate). Inheritance: limits "
        "MIN-WINS (a child may only tighten), degrade is scalar-override. "
        "None = unbudgeted."})
    # AppArmor confinement intent for the session (PR-A, 2026-05-14).
    #
    # ``False`` (default, back-compat) — the session bootstraps
    # unconfined regardless of host AppArmor capability.  Same posture
    # as every pre-PR-A caller of ``SessionManager.create_headless_session``.
    #
    # ``True`` — the session opts into per-session AppArmor confinement
    # (same mechanism as the IPC client's ``apparmor=True`` toggle):
    # the daemon provisions a per-session profile, the runner self-
    # confines to it in ``bootstrap_session`` step 1c, and any //child
    # subprocesses transition to ``//child``.  On hosts without
    # AppArmor (macOS, BSD, containerised Linux without policy load
    # support) the ``_maybe_self_confine`` helper no-ops just like
    # ``IPCClient(apparmor=True)`` already does today — the field is a
    # statement of intent; the actual enforcement is best-effort.
    #
    # **Default-flip planned (PR-B):**  Once cascade workloads
    # (kb-enablement-2.0) have validated the field on non-trivial
    # graphs, PR-B will flip the default to ``True`` so the security
    # gradient closes.  Legacy callers that haven't opted out by then
    # will need to set ``apparmor: false`` explicitly.  See
    # ``project_backlog_apparmor_kwarg_for_headless_sessions`` for the
    # full migration plan.
    #
    # Resolution precedence at session-creation time:
    #   1. Explicit ``apparmor=`` kwarg on
    #      ``SessionManager.create_headless_session`` (kwarg wins).
    #   2. This profile field.
    #   3. Legacy unconfined default (``False`` until PR-B).
    apparmor: bool = field(default=False, metadata={
        "description": "Opt into per-session kernel-enforced AppArmor "
        "confinement (best-effort; no-ops on hosts without AppArmor). "
        "False = unconfined. Resolution: create_headless_session kwarg > this "
        "field > legacy unconfined default."})

    # Provider/model quirks declarations (server 0.6.194+).
    #
    # Top-level profile field that opts the active provider into known
    # wire-format / model-behavior workarounds.  A profile binds to
    # exactly one ``provider`` + ``model``; the provider plugin reads
    # the quirks dict at session init and acts on the keys it knows.
    # Unknown keys log a warning and are ignored.
    #
    # Currently shipped quirks (see provider plugin docstrings for the
    # canonical list):
    #
    # - ``coerce_typed_tool_args`` (vllm, others may follow): when the
    #   model emits a JSON string where the tool schema expects an
    #   array / object / integer / number / boolean, attempt
    #   ``ast.literal_eval`` (handles Python repr with single quotes
    #   that ``json.loads`` cannot parse) then ``json.loads`` as a
    #   fallback, then re-validate.  Workaround for Llama 3.1 on vLLM
    #   0.22.1 with the ``llama3_json`` parser, which passes the
    #   model's stringified args through verbatim because vLLM has not
    #   registered a structural-tag enforcement for that parser.
    #   See ``feedback_llama31_vllm_auto_mode_stringifies_args`` for
    #   the full diagnosis.
    #
    # Defaults to an empty dict (no quirks active).  Inheritance follows
    # the collection-union rule: child + parent keys are merged; on
    # key collision the child wins.
    quirks: Dict[str, Any] = field(default_factory=dict, metadata={
        "description": "Opt the active provider into known wire-format / "
        "model-behavior workarounds (the provider reads the keys it knows; "
        "unknown keys warn). e.g. coerce_typed_tool_args (vllm)."})

    # Per-profile AppArmor fragment scoping (Piece 1, 2026-05-14).
    #
    # When set, the per-session AppArmor policy composes only the
    # fragments whose basename (without ``.rules``) matches an entry
    # in this list, looked up in the fragment search path (user-tier
    # ``~/.jaato/apparmor-fragments/``, workspace-tier
    # ``<workspace>/.jaato/apparmor-fragments/``, and the
    # walker-generated cache layer
    # ``<workspace>/.jaato/.cache/apparmor-fragments/`` with cache
    # taking precedence on basename collision).
    #
    # ``None`` (default, absent) — back-compat: the profile composes
    # ALL fragments from the search path, just like pre-Piece-1
    # behaviour.  This is the right default for workspaces with no
    # cascade intent.
    #
    # ``[]`` (explicit empty list) — distinct from ``None``: the
    # profile composes NO fragments.  Maximally locked-down stage in
    # a cascade.
    #
    # Non-empty list — compose ONLY the listed fragments.  Unknown
    # fragment names log WARNING but don't abort (operator may have
    # removed the fragment after authoring the profile).
    #
    # **Inheritance semantics — child REPLACES parent.**  When the
    # field is declared (not None) on the child, the child's value
    # wins.  When the child doesn't declare it, the resolved value
    # comes from the nearest parent in the ``inherits:`` chain.
    # Replace (not union) lets cascade authors SCOPE DOWN from a
    # parent's broader set — necessary for least-privilege, since
    # union would only let children ADD permissions, never remove.
    # Matches the ``model`` field's child-wins rationale (vs
    # ``plugins`` which is union).
    #
    # See ``project_backlog_per_profile_apparmor_fragments`` for the
    # design ask + the cascade footgun this closes (workspace-tier
    # fragments bleeding binary-exec across all cascade stages when
    # only one stage should have it).
    apparmor_fragments: Optional[List[str]] = field(default=None, metadata={
        "description": "Scope WHICH AppArmor .rules fragments compose this "
        "session's policy, by basename, from the fragment search path "
        "(~/.jaato/apparmor-fragments/, <workspace>/.jaato/apparmor-fragments/, "
        "+ the .cache/ layer). None = compose ALL fragments; [] = none "
        "(maximally locked-down)."})

    def __post_init__(self) -> None:
        """Normalize ``suppress_base_instructions`` to its canonical form.

        Accepts the authored bool / dict / list (or an already-normalized
        frozenset, idempotently) and stores a ``frozenset`` of piece names.
        Centralizing here means every construction path — ``from_dict``,
        ``build_inline_profile``, inheritance merge, direct kwargs — ends up
        with one representation, and an unknown piece name fails loud at
        profile-load time rather than silently keeping a layer.
        """
        self.suppress_base_instructions = normalize_suppression(
            self.suppress_base_instructions
        )


def _normalize_inherits(value: Any) -> Optional[List[str]]:
    """Normalize the ``inherits`` field to a list of strings or None.

    Accepts a single string (``"readonly"``), a list of strings
    (``["readonly", "web_capable"]``), or None/absent.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return None


def _normalize_apparmor_fragments(value: Any) -> Optional[List[str]]:
    """Coerce ``apparmor_fragments`` field value into the canonical shape.

    Distinguishes three input states (Piece 1, 2026-05-14):

    - **Absent** (``value is None`` — key not in the dict) → returns
      ``None``.  The profile inherits the workspace-default
      "compose all fragments" behaviour.  Back-compat for every
      pre-Piece-1 profile.
    - **Explicit empty list** (``value == []``) → returns ``[]``.
      The profile composes NO fragments — maximally locked-down
      stage in a cascade.  Distinct from absent (the field IS
      declared, just empty).
    - **Non-empty list of strings** → returns the same list with
      entries coerced to ``str``.  These fragment basenames will
      be looked up at policy-render time.

    Anything else (non-list value, list of non-strings) is rejected
    by raising :class:`ValueError`.  Quiet coercion would mask
    operator typos in the YAML/JSON.

    Args:
        value: Raw value pulled from ``data.get('apparmor_fragments')``.

    Returns:
        ``None``, ``[]``, or ``List[str]`` per the rules above.

    Raises:
        ValueError: If ``value`` is non-None but isn't a list of
            strings.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            f"apparmor_fragments must be a list of strings, "
            f"got {type(value).__name__}: {value!r}"
        )
    normalised: List[str] = []
    for i, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"apparmor_fragments[{i}] must be a non-empty string, "
                f"got {type(entry).__name__}: {entry!r}"
            )
        normalised.append(entry.strip())
    return normalised


def _normalize_suppress_inherited_processors(value: Any) -> List[str]:
    """Normalize ``suppress_inherited_processors`` to a list of strings.

    Accepts a single string (``"acceptance"``), a list, or None/absent.
    Entries are coerced with ``str()`` rather than dropped: a non-string
    entry becomes a string that matches no inherited processor, and
    :func:`_merge_completion_processors` then fails the profile load with
    the same "matched nothing" diagnostic a typo gets.  Dropping it
    quietly would leave the author with a processor they declared they
    did not want, which is precisely the silent loss #791 is about.

    A value that is neither a string nor a list (a dict, an int) is not
    coercible to a *set* of identities at all, so it warns and yields
    ``[]``.

    Args:
        value: Raw value pulled from ``data.get('suppress_inherited_processors')``.

    Returns:
        A list of identity strings; empty when nothing was declared.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and v != ""]
    logger.warning(
        "suppress_inherited_processors must be a string or a list of "
        "strings, got %s: %r; ignoring",
        type(value).__name__, value,
    )
    return []


def _processor_opt_str(
    entry: Dict[str, Any], key: str, script: str,
) -> Optional[str]:
    """Read an optional free-text key off one ``completion_processors`` entry.

    Shared by ``description`` and ``name``: both are optional strings that
    are stripped, both collapse to ``None`` when blank, and both warn
    (naming the offending ``script``) rather than raise when the author
    wrote a non-string — a typo'd annotation must not take the whole
    profile down with it.

    Args:
        entry: One raw ``completion_processors`` list item.
        key: The key to read (``"description"`` / ``"name"``).
        script: The entry's script path, for the warning message.

    Returns:
        The stripped string, or ``None`` when absent, blank, or invalid.
    """
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        logger.warning(
            "completion_processors: %r must be a string for script=%r "
            "(got %s); ignoring",
            key, script, type(value).__name__,
        )
        return None
    return value.strip() or None


def _processor_enum(
    entry: Dict[str, Any],
    key: str,
    allowed: Tuple[str, ...],
    default: str,
    script: str,
) -> str:
    """Read a closed-vocabulary key off one ``completion_processors`` entry.

    Shared by ``on_error`` and ``phase``.  An unrecognised value warns and
    falls back to *default* rather than rejecting the entry: the safe
    default (``fail_completion`` / ``finalization``) is the conservative
    one in both cases, so a typo degrades to strictness, never past it.

    Args:
        entry: One raw ``completion_processors`` list item.
        key: The key to read.
        allowed: The permitted values.
        default: Value used when the key is absent or unrecognised.
        script: The entry's script path, for the warning message.

    Returns:
        One of *allowed*.
    """
    value = entry.get(key, default)
    if value not in allowed:
        logger.warning(
            "completion_processors: invalid %s=%r for script=%r "
            "(expected one of %r); defaulting to %r",
            key, value, script, list(allowed), default,
        )
        return default
    return value


def _parse_completion_processors(value: Any) -> List[CompletionProcessor]:
    """Parse a profile's ``completion_processors`` list from raw JSON/YAML.

    Replaces the prior ``_parse_completion_artifacts`` +
    ``_parse_completion_validators`` (server 0.6.125+).  Each entry is
    a dict shaped like::

        {"script": "scripts/processors/foo.py",
         "output": "out/{case_id}/foo",      # optional
         "on_error": "fail_completion",      # default
         "phase": "finalization",            # default
         "name": "acceptance",               # optional
         "description": "..."}               # optional

    ``output`` is optional — when omitted, the processor runs for
    side-effect (validator-only) and ``render``'s return is logged
    but not written.  ``description`` travels with the wiring for
    documentation; ignored at runtime.  ``name`` is the stable
    identity an inheriting profile's ``suppress_inherited_processors``
    can decline the processor by (#791); when absent the ``script``
    path is that identity.

    Skips malformed entries with a warning rather than raising —
    partial profiles still load and the missing/typo'd processor
    surfaces at completion time as a load error the agent sees.

    Returns an empty list when ``value`` is ``None``, missing, or
    not a list.
    """
    if not isinstance(value, list):
        return []
    out: List[CompletionProcessor] = []
    for entry in value:
        if not isinstance(entry, dict):
            logger.warning(
                "completion_processors: skipping non-dict entry: %r", entry,
            )
            continue
        script = entry.get("script")
        output = entry.get("output")
        if not isinstance(script, str) or not script.strip():
            logger.warning(
                "completion_processors: skipping entry without 'script': %r",
                entry,
            )
            continue
        normalized_output: Optional[str]
        if output is None or output == "":
            normalized_output = None
        elif isinstance(output, str) and output.strip():
            normalized_output = output.strip()
        else:
            logger.warning(
                "completion_processors: invalid 'output' value (must be a "
                "non-empty string or omitted) for script=%r: %r",
                script, output,
            )
            continue
        out.append(CompletionProcessor(
            script=script.strip(),
            output=normalized_output,
            on_error=_processor_enum(
                entry, "on_error", ("fail_completion", "warn"),
                "fail_completion", script,
            ),
            description=_processor_opt_str(entry, "description", script),
            phase=_processor_enum(
                entry, "phase", ("finalization", "completeness"),
                "finalization", script,
            ),
            name=_processor_opt_str(entry, "name", script),
        ))
    return out


#: The ``cache.ttl`` vocabulary.  Deliberately the Anthropic/OpenRouter
#: spelling rather than a duration string: those are the two mechanisms
#: that expose a TTL choice at all, and Google's seconds format is
#: derived from these rather than the other way round.
VALID_CACHE_TTLS = frozenset({"5m", "1h"})


def parse_cache_block(data: Dict[str, Any]) -> Optional['CacheProfileConfig']:
    """Parse a profile dict's optional ``cache:`` block.

    Sibling of :func:`parse_gc_block`, and the reason that one exists:
    both are called from all four profile ingresses, and a block field
    wired into three of them is silently inert in the fourth.
    """
    block = data.get('cache')
    if not block:
        return None
    return CacheProfileConfig.from_dict(block)


def parse_gc_block(data: Dict[str, Any]) -> Optional['GCProfileConfig']:
    """Parse a profile dict's optional ``gc:`` block.

    ONE definition, because there are FOUR ingresses that build a
    ``SubagentProfile`` from a dict — ``build_inline_profile``,
    ``_scan_profiles_dir``, ``_discover_premium_profiles`` and
    ``SubagentConfig.from_dict`` — and each carried its own copy of the
    same three lines, in two spellings of the identical guard
    (``data.get('gc')`` and ``'gc' in data and data['gc']``).

    Four copies is four places to forget when a sibling block field is
    added, and a field wired into three ingresses and missed in the
    fourth is silently inert in exactly one code path — which is the
    failure this branch opened by fixing (§4: a cache knob that reached
    no ingress at all). Collapsing them now means the next block field
    is added once.

    Returns ``None`` when the block is absent or empty; an empty ``gc:``
    is deliberately not a default-constructed config, matching what all
    four sites already did.
    """
    block = data.get('gc')
    if not block:
        return None
    return GCProfileConfig.from_dict(block)


def parse_trace_block(data: Dict[str, Any]) -> Optional['TraceProfileConfig']:
    """Parse a profile dict's optional ``trace:`` block.

    Third sibling of :func:`parse_cache_block` / :func:`parse_gc_block`,
    for the reason both of those exist: FOUR ingresses build a
    ``SubagentProfile`` from a dict, and a block field wired into three
    of them is silently inert in the fourth.

    Returns ``None`` when the block is absent or empty; raises
    ``ValueError`` (from ``TraceProfileConfig.from_dict``) on an
    unusable one, which is the whole reason the block exists.
    """
    block = data.get('trace')
    if not block:
        return None
    return TraceProfileConfig.from_dict(block)


def build_inline_profile(
    data: Dict[str, Any],
    name: str = "<inline>",
    description: str = "Inline session spec",
) -> 'SubagentProfile':
    """Construct a ``SubagentProfile`` from a dict supplied by an SDK client.

    Mirrors the field set understood by ``_load_profiles_from_directory``
    so an inline spec on ``session.new`` accepts the same JSON shape as a
    profile file on disk. ``inherits`` is intentionally ignored — inline
    specs are atomic, not chained. ``name`` is taken from ``data['name']``
    when the client supplied one (like disk profiles), else the ``name``
    param (default ``<inline>``); ``description`` defaults to a placeholder.

    Args:
        data: The dict carried in ``CommandRequest.payload['spec']``.
            Recognized keys: ``model``, ``provider``, ``plugins``,
            ``plugin_configs``, ``system_instructions``, ``max_turns``,
            ``gc``, ``env``, ``completion_payload_schema``,
            ``completion_processors``,
            ``runtime_limits``, ``budget_control``, ``model_tiers``.
        name: Display name for logs and traces. Default ``<inline>``.
        description: Human-readable description for the profile.

    Returns:
        A fully-formed ``SubagentProfile`` ready to hand to
        ``JaatoServer(profile=...)``.

    Raises:
        ValueError: If a structured sub-field (``gc``, ``runtime_limits``)
            fails to parse. Surfaced so the caller can emit a clear
            ``ErrorEvent`` rather than swallowing the failure.
    """
    cache_config = parse_cache_block(data)
    gc_config = parse_gc_block(data)
    trace_config = parse_trace_block(data)

    runtime_limits = None
    if data.get('runtime_limits'):
        if not isinstance(data['runtime_limits'], dict):
            raise ValueError(
                "Invalid runtime_limits in inline spec: expected dict, "
                f"got {type(data['runtime_limits']).__name__}"
            )
        try:
            runtime_limits = RuntimeLimits.from_dict(data['runtime_limits'])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid runtime_limits in inline spec: {exc}")

    budget_control = None
    if data.get('budget_control'):
        try:
            budget_control = BudgetControlConfig.from_dict(data['budget_control'])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid budget_control in inline spec: {exc}")

    # ``plugins`` is REQUIRED on every profile / inline spec — absent
    # vs. explicitly-empty have meaningfully different downstream
    # semantics (see the workspace-tier scanner for the full
    # rationale).  Inline specs raise immediately because the caller
    # has the surface area to fix it.
    if 'plugins' not in data:
        raise ValueError(
            "Inline session spec is missing the required 'plugins' "
            "key.  Use 'plugins': [] for the minimal framework set "
            "(permission, reliability, lifecycle only), or list "
            "plugin names to expose."
        )
    raw_plugins = data['plugins']
    clean_plugins, preloaded, tool_scopes = parse_plugin_list(raw_plugins)

    raw_env = data.get('env', {})
    env = (
        {str(k): str(v) for k, v in raw_env.items()}
        if isinstance(raw_env, dict) else {}
    )

    raw_model_tiers = data.get('model_tiers') or {}
    model_tiers = (
        {str(k): v for k, v in raw_model_tiers.items()}
        if isinstance(raw_model_tiers, dict) else {}
    )

    raw_quirks = data.get('quirks') or {}
    quirks = (
        {str(k): v for k, v in raw_quirks.items()}
        if isinstance(raw_quirks, dict) else {}
    )

    return SubagentProfile(
        # Honor the spec's own ``name`` (e.g. "nano-chat") when the SDK
        # client supplied one; fall back to the ``name`` param (default
        # "<inline>") otherwise.  Mirrors disk profiles, which take
        # ``data.get('name')`` — previously an inline spec's name was
        # silently dropped, so ``profile_name`` / the agent display always
        # read "<inline>".  Inline restore no longer depends on the name
        # being an unresolvable sentinel (it reconstructs from
        # ``profile_spec`` directly — see SessionManager._load_session_impl).
        name=data.get('name') or name,
        description=description,
        plugins=clean_plugins,
        preloaded_plugins=preloaded,
        tool_scopes=tool_scopes,
        plugin_configs=data.get('plugin_configs', {}),
        system_instructions=data.get('system_instructions'),
        suppress_base_instructions=data.get('suppress_base_instructions', False),
        model=data.get('model'),
        provider=data.get('provider'),
        max_turns=data.get('max_turns', 10),
        gc=gc_config,
            cache=cache_config,
            trace=trace_config,
        env=env,
        inherits=None,
        completion_payload_schema=data.get('completion_payload_schema'),
        spawn_payload_schema=data.get('spawn_payload_schema'),
        completion_processors=_parse_completion_processors(data.get('completion_processors')),
        runtime_limits=runtime_limits,
        budget_control=budget_control,
        model_tiers=model_tiers,
        apparmor=bool(data.get('apparmor', False)),
        # Use ``data.get('apparmor_fragments')`` (returns None when
        # absent) rather than ``data.get(..., default)`` — None
        # signals "absent / inherit workspace default", which is
        # semantically distinct from explicit ``[]`` (compose no
        # fragments).  See :func:`_normalize_apparmor_fragments`.
        apparmor_fragments=_normalize_apparmor_fragments(data.get('apparmor_fragments')),
        quirks=quirks,
    )


# ---------------------------------------------------------------------------
# Resolved-profile snapshots (issue #787)
#
# ``build_inline_profile`` reconstructs a profile from an AUTHORED spec —
# the JSON/YAML shape a human writes.  A snapshot is the other direction:
# the RESOLVED ``SubagentProfile`` a session actually ran under, frozen so
# a revive can rebuild it without re-reading (and re-resolving, and
# re-merging) the profile files on disk.
#
# Why not reuse ``build_inline_profile``: it accepts the authored key set
# only.  A resolved profile carries fields inheritance produced
# (``suppress_inherited_processors``, merged ``completion_processors``) and
# fields derived at parse time (``preloaded_plugins``, ``tool_scopes``,
# normalized ``suppress_base_instructions``).  Round-tripping through the
# authored shape would silently drop them, which is the opposite of what a
# snapshot is for.
#
# Secrets: the snapshot stores what the resolved profile holds, and a
# resolved profile holds ``pass://`` / ``vault://`` URIs UNRESOLVED —
# expansion happens later, at ``expand_plugin_configs`` / ``env`` overlay
# time on the daemon (see ``runner_spawn.build_session_envelope``).  So a
# snapshot lands the same unresolved URIs on disk that ``profile_spec``
# already does, and no plaintext credential is introduced by persisting it.
# ---------------------------------------------------------------------------

#: Snapshot format version.  Bumped when the field set changes in a way a
#: reader must notice; readers tolerate an unknown-but-newer minor by
#: ignoring keys they do not recognise (``SubagentProfile`` construction is
#: keyword-explicit, so an unknown key is simply never read).
PROFILE_SNAPSHOT_VERSION = 1


def _emit_plugin_entry(
    name: str,
    preloaded: bool,
    tools: Optional[List[str]],
) -> str:
    """Render one ``plugins:`` entry back to its authored string form.

    Inverse of :func:`parse_plugin_entry`.  Emits the explicit tagged form
    (``name(mode:preload, tools:[a,b])``) rather than the positional one so
    the result is unambiguous regardless of which knobs are present.

    Args:
        name: Plugin name.
        preloaded: Whether the plugin is in the profile's preload set.
        tools: Per-plugin tool allow-list, or ``None`` for "all tools".

    Returns:
        A string :func:`parse_plugin_entry` parses back to the same triple.
    """
    parts: List[str] = []
    if preloaded:
        parts.append("mode:preload")
    if tools is not None:
        parts.append("tools:[" + ",".join(str(t) for t in tools) + "]")
    if not parts:
        return name
    return f"{name}({', '.join(parts)})"


def _runtime_limits_to_dict(
    limits: Optional['RuntimeLimits'],
) -> Optional[Dict[str, Any]]:
    """Render :class:`RuntimeLimits` back to its authored mapping.

    Inverse of :meth:`RuntimeLimits.from_dict`, which splits a profile's
    ``runtime_limits:`` block into known fields plus an ``extra`` bag for
    keys it does not recognise.  ``dataclasses.asdict`` would emit that bag
    as a nested ``extra`` key, which ``from_dict`` then parks in ANOTHER
    ``extra`` -- one level deeper per save/restore cycle.  Flattening here
    keeps the round trip a fixed point.

    Args:
        limits: The resolved limits, or ``None``.

    Returns:
        A mapping ``RuntimeLimits.from_dict`` reconstructs identically, or
        ``None`` when there were no limits.
    """
    if limits is None:
        return None
    import dataclasses as _dc
    data = _dc.asdict(limits)
    extra = data.pop("extra", None) or {}
    data.update(extra)
    return data


def profile_to_snapshot(profile: 'SubagentProfile') -> Dict[str, Any]:
    """Freeze a RESOLVED profile into a JSON-serializable snapshot.

    The inverse of :func:`profile_from_snapshot`.  Every field of
    :class:`SubagentProfile` is carried, with the derived plugin knobs
    (``preloaded_plugins`` / ``tool_scopes``) folded back into the
    ``plugins`` entry strings so there is one source of truth for them on
    the way back in.

    Args:
        profile: The resolved profile a session was created with.

    Returns:
        A JSON-serializable dict.  ``pass://`` / ``vault://`` URIs inside
        ``env`` / ``plugin_configs`` are carried UNRESOLVED (see the module
        note above) — the snapshot never lands a plaintext credential that
        the profile file did not already contain.
    """
    import dataclasses as _dc

    preloaded = set(getattr(profile, "preloaded_plugins", None) or ())
    scopes = dict(getattr(profile, "tool_scopes", None) or {})
    plugins = [
        _emit_plugin_entry(name, name in preloaded, scopes.get(name))
        for name in (profile.plugins or [])
    ]

    def _block(value: Any) -> Optional[Dict[str, Any]]:
        return _dc.asdict(value) if value is not None else None

    budget = getattr(profile, "budget_control", None)

    return {
        "snapshot_version": PROFILE_SNAPSHOT_VERSION,
        "name": profile.name,
        "description": profile.description,
        "plugins": plugins,
        "plugin_configs": dict(profile.plugin_configs or {}),
        "system_instructions": profile.system_instructions,
        "suppress_base_instructions": sorted(
            profile.suppress_base_instructions or ()
        ),
        "model": profile.model,
        "provider": profile.provider,
        "max_turns": profile.max_turns,
        "cache": _block(getattr(profile, "cache", None)),
        "trace": _block(getattr(profile, "trace", None)),
        "gc": _block(getattr(profile, "gc", None)),
        "env": dict(profile.env or {}),
        # ``inherits`` is deliberately dropped: a snapshot is POST-merge, so
        # re-declaring the parents would re-apply them on top of a profile
        # that already carries their fields.
        "completion_payload_schema": profile.completion_payload_schema,
        "spawn_payload_schema": profile.spawn_payload_schema,
        "completion_processors": [
            _dc.asdict(p) for p in (profile.completion_processors or [])
        ],
        "suppress_inherited_processors": list(
            getattr(profile, "suppress_inherited_processors", None) or []
        ),
        # ``RuntimeLimits`` parks unknown keys in an ``extra`` dict and
        # ``from_dict`` re-parks anything it does not recognise -- so a
        # plain ``asdict`` would nest ``extra`` one level deeper on every
        # round trip.  Flatten it back into the mapping ``from_dict``
        # expects, which is the shape the profile file had.
        "runtime_limits": _runtime_limits_to_dict(
            getattr(profile, "runtime_limits", None)
        ),
        "model_tiers": dict(getattr(profile, "model_tiers", None) or {}),
        "budget_control": budget.to_dict() if budget is not None else None,
        "apparmor": bool(getattr(profile, "apparmor", False)),
        "apparmor_fragments": (
            list(profile.apparmor_fragments)
            if getattr(profile, "apparmor_fragments", None) is not None
            else None
        ),
        "quirks": dict(getattr(profile, "quirks", None) or {}),
    }


def _snapshot_blocks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Re-parse a snapshot's five structured sub-blocks.

    Split out of :func:`profile_from_snapshot` so that function stays under
    the cyclomatic-complexity ceiling: each block is an
    absent-or-parse pair, and five of them in one body is most of its
    branching.

    Args:
        data: The snapshot dict.

    Returns:
        ``{"gc", "cache", "trace", "runtime_limits", "budget_control"}``,
        each either the parsed config or ``None``.

    Raises:
        ValueError: When a block is present but unparseable.  Loud, because
            a session silently revived with (say) no GC strategy is worse
            than one that refuses to revive: the first is discovered when
            the context window overflows.
    """
    limits_raw = data.get("runtime_limits")
    budget_raw = data.get("budget_control")
    try:
        return {
            # The SHARED block parsers, deliberately -- a snapshot is one
            # more profile ingress, and the whole point of
            # ``parse_gc_block`` / ``parse_cache_block`` /
            # ``parse_trace_block`` is that a new block field is wired in
            # once rather than once per ingress.  They read their own key
            # out of the dict, so the whole snapshot is what they take.
            "gc": parse_gc_block(data),
            "cache": parse_cache_block(data),
            "trace": parse_trace_block(data),
            "runtime_limits": (
                RuntimeLimits.from_dict(limits_raw) if limits_raw else None
            ),
            "budget_control": (
                BudgetControlConfig.from_dict(budget_raw)
                if budget_raw else None
            ),
        }
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid block in profile snapshot: {exc}") from exc


def profile_from_snapshot(data: Dict[str, Any]) -> 'SubagentProfile':
    """Rebuild a resolved profile from a :func:`profile_to_snapshot` dict.

    Args:
        data: A snapshot dict.

    Returns:
        The reconstructed :class:`SubagentProfile`.

    Raises:
        ValueError: When ``data`` is not a dict, or a structured sub-block
            fails to parse.  Callers surface this rather than silently
            reviving a session under a half-built recipe.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"profile snapshot must be a dict, got {type(data).__name__}"
        )

    clean_plugins, preloaded, tool_scopes = parse_plugin_list(
        list(data.get("plugins") or [])
    )
    blocks = _snapshot_blocks(data)

    return SubagentProfile(
        name=data.get("name") or "<snapshot>",
        description=data.get("description") or "",
        plugins=clean_plugins,
        preloaded_plugins=preloaded,
        tool_scopes=tool_scopes,
        plugin_configs=data.get("plugin_configs") or {},
        system_instructions=data.get("system_instructions"),
        suppress_base_instructions=data.get(
            "suppress_base_instructions", False
        ),
        model=data.get("model"),
        provider=data.get("provider"),
        max_turns=data.get("max_turns", 10),
        cache=blocks["cache"],
        trace=blocks["trace"],
        gc=blocks["gc"],
        env=dict(data.get("env") or {}),
        inherits=None,
        completion_payload_schema=data.get("completion_payload_schema"),
        spawn_payload_schema=data.get("spawn_payload_schema"),
        completion_processors=_parse_completion_processors(
            data.get("completion_processors")
        ),
        suppress_inherited_processors=list(
            data.get("suppress_inherited_processors") or []
        ),
        runtime_limits=blocks["runtime_limits"],
        model_tiers=dict(data.get("model_tiers") or {}),
        budget_control=blocks["budget_control"],
        apparmor=bool(data.get("apparmor", False)),
        apparmor_fragments=_normalize_apparmor_fragments(
            data.get("apparmor_fragments")
        ),
        quirks=dict(data.get("quirks") or {}),
    )


def resolve_profiles(
    profiles: Dict[str, 'SubagentProfile'],
) -> Tuple[Dict[str, 'SubagentProfile'], Dict[str, str]]:
    """Resolve profile inheritance by merging parent fields into children.

    Performs a topological traversal of the inheritance graph. Each profile
    with an ``inherits`` field has its parents' fields merged in, following
    the merge semantics from ``docs/design/profile-inheritance.md``:

    - **Collection fields** (union): ``plugins``, ``preloaded_plugins``,
      ``env``, ``plugin_configs``
    - **Scalar fields** (agreement-or-override): ``model``, ``provider``,
      ``max_turns``, ``gc``
    - **Concatenation**: ``system_instructions`` (grandparent → parent → child)
    - **Never inherited**: ``name``, ``description``

    Conflicts between parents on scalar fields are hard errors unless the
    child explicitly overrides.  Cycles are detected and reported.

    After resolution, ``inherits`` is cleared on each profile (fully
    flattened).

    Args:
        profiles: Dict of profile name → SubagentProfile (unresolved).

    Returns:
        Tuple of (resolved_profiles, errors). ``errors`` maps profile
        names to error messages.  Profiles with errors are excluded from
        the resolved dict.
    """
    resolved: Dict[str, SubagentProfile] = {}
    errors: Dict[str, str] = {}
    resolving: set = set()  # cycle detection

    def _resolve(name: str) -> Optional[SubagentProfile]:
        """Recursively resolve a single profile."""
        if name in resolved:
            return resolved[name]
        if name in errors:
            return None
        if name in resolving:
            errors[name] = f"Inheritance cycle detected involving '{name}'"
            return None
        if name not in profiles:
            errors[name] = f"Profile '{name}' not found"
            return None

        profile = profiles[name]

        # No inheritance — resolve immediately.  Nothing to suppress
        # here, so say so rather than letting the key look effective:
        # ``suppress_inherited_processors`` only ever removes what a
        # PARENT contributed (#791).
        if not profile.inherits:
            if profile.suppress_inherited_processors:
                logger.warning(
                    "Profile '%s' declares suppress_inherited_processors %r "
                    "but does not inherit from anything; it has no effect. "
                    "Remove the entries, or the processors themselves from "
                    "completion_processors.",
                    name, list(profile.suppress_inherited_processors),
                )
            resolved[name] = profile
            return profile

        # Resolve parents first
        resolving.add(name)
        parent_profiles: List[SubagentProfile] = []
        for parent_name in profile.inherits:
            parent = _resolve(parent_name)
            if parent is None:
                errors[name] = (
                    f"Profile '{name}' inherits from '{parent_name}' "
                    f"which failed to resolve"
                )
                resolving.discard(name)
                return None
            parent_profiles.append(parent)
        resolving.discard(name)

        # Merge parents then apply child overrides
        merged = _merge_profiles(name, parent_profiles, profile, errors)
        if merged is None:
            return None

        resolved[name] = merged
        return merged

    # Resolve all profiles
    for name in list(profiles.keys()):
        _resolve(name)

    return resolved, errors


def _merge_budget_control(
    parents: List['SubagentProfile'],
    child: 'SubagentProfile',
) -> Optional[BudgetControlConfig]:
    """Merge ``budget_control`` across parents + child.

    Two different rules, one per half of the block:

    * ``limits`` — **min-wins** across every layer that declares a
      dimension (:func:`shared.budget_control.merge_limits`).  A child
      may only ever TIGHTEN a ceiling; it must never grant itself a
      bigger budget than the profile that spawned it.  This is the same
      safety direction ``max_turns`` already takes (most restrictive
      value across parents), and it deliberately differs from the
      child-replaces-parent rule used by most scalar fields — for a
      resource ceiling, "child wins" would be an escape hatch.
      Divergent parent values do NOT conflict: the minimum is
      well-defined and is the safe resolution.
    * ``degrade`` — **scalar-override**: the child's whole ladder wins
      when it declares one, else the first parent that declares one is
      inherited whole.  Matches ``model_tiers`` (a ladder is a coherent
      unit; interleaving rungs from two layers would produce a ladder
      neither author wrote).

    Returns ``None`` when no layer declares anything, so an unbudgeted
    profile stays unbudgeted.
    """
    parent_configs = [
        p.budget_control for p in parents if p.budget_control is not None
    ]
    child_config = child.budget_control
    if not parent_configs and child_config is None:
        return None

    limits: Dict[str, float] = {}
    for cfg in parent_configs:
        limits = merge_limits(limits, cfg.limits)
    if child_config is not None:
        limits = merge_limits(limits, child_config.limits)

    degrade: Tuple[Any, ...] = ()
    if child_config is not None and child_config.degrade:
        degrade = child_config.degrade
    else:
        for cfg in parent_configs:
            if cfg.degrade:
                degrade = cfg.degrade
                break

    if not limits and not degrade:
        return None
    return BudgetControlConfig(limits=limits, degrade=degrade)


def _processor_identities(
    processors: List[CompletionProcessor],
) -> FrozenSet[str]:
    """Every string a ``suppress_inherited_processors`` entry may match.

    A processor answers to its declared ``name`` AND to its ``script``
    path, so a child can decline one without knowing which of the two the
    base chose to write.  See :attr:`CompletionProcessor.identity` for the
    single canonical form used in diagnostics.

    Args:
        processors: The inherited processors to index.

    Returns:
        The union of every entry's script path and (when set) name.
    """
    ids: set = set()
    for proc in processors:
        ids.add(proc.script)
        if proc.name:
            ids.add(proc.name)
    return frozenset(ids)


def _unmatched_suppression_error(
    child_name: str,
    unmatched: List[str],
    parents: List['SubagentProfile'],
    inherited: List[CompletionProcessor],
) -> str:
    """Explain a ``suppress_inherited_processors`` entry that matched nothing.

    Names what was asked for AND what was actually on offer, because the
    two ways to get here — a typo, or a base that renamed/moved the
    processor — are told apart by reading the available list.

    Args:
        child_name: Profile that declared the suppression.
        unmatched: The entries that matched no inherited processor.
        parents: The resolved parents, named in the message.
        inherited: Everything the parents contributed.

    Returns:
        The error message recorded against *child_name*.
    """
    listing = sorted(proc.identity for proc in inherited) or ["(none)"]
    return (
        f"Profile '{child_name}' declares suppress_inherited_processors "
        f"{unmatched!r}, which match no inherited completion_processor.  "
        f"Inherited from {[p.name for p in parents]!r}: {listing!r}.  A "
        f"suppression that matches nothing means the processor was renamed "
        f"or moved and this profile is running it again — fix the entry or "
        f"drop it."
    )


def _merge_completion_processors(
    parents: List['SubagentProfile'],
    child: 'SubagentProfile',
) -> Tuple[List[CompletionProcessor], Optional[str]]:
    """Merge ``completion_processors`` across parents + child (#791).

    The default is **concatenation**, parent → child: each processor is
    independent (writes a different artefact or checks a different
    invariant) and all of them fire.  A child's ``completion_processors``
    only ever ADDS — an empty list there clears nothing.

    ``suppress_inherited_processors`` is the single, deliberately narrow
    opt-out.  It names inherited processors to drop, matching either a
    parent entry's ``name`` or its ``script`` path.  Three properties are
    load-bearing:

    * **By name, never wholesale.**  A base that later adds a second
      processor must not silently re-enable the one this child declined,
      and "drop everything the parent completes with" is not a thing a
      cascade stage should be able to say in passing.
    * **Parents only.**  The child's own processors are its to edit
      directly, so suppression never has to disambiguate between "the one
      I inherited" and "the one I declared".
    * **A stale entry is an error.**  If nothing matches, the base moved
      or renamed the processor and this child is running one it declared
      it did not want.  Failing the profile load says so; a silent no-op
      is how #791's "an interrogation ran with no cost ceiling" happens.

    One entry drops EVERY inherited processor it matches — with multiple
    parents contributing the same script, declining it declines all of
    them, which is what "I do not complete that way" means.

    Args:
        parents: Resolved parent profiles, in declaration order.
        child: The child profile with its own overrides.

    Returns:
        ``(processors, error)``.  ``error`` is ``None`` on success, else a
        message naming the unmatched entries and what WAS available; the
        caller records it and drops the profile.
    """
    inherited: List[CompletionProcessor] = []
    for parent in parents:
        inherited.extend(parent.completion_processors)

    suppress = list(child.suppress_inherited_processors or [])
    if not suppress:
        return inherited + list(child.completion_processors), None

    available = _processor_identities(inherited)
    unmatched = [entry for entry in suppress if entry not in available]
    if unmatched:
        return [], _unmatched_suppression_error(
            child.name, unmatched, parents, inherited,
        )

    # Keep a processor when NEITHER of its identities is suppressed.
    suppress_set = set(suppress)
    kept = [
        proc for proc in inherited
        if suppress_set.isdisjoint((proc.script, proc.identity))
    ]
    return kept + list(child.completion_processors), None


def _merge_profiles(
    child_name: str,
    parents: List['SubagentProfile'],
    child: 'SubagentProfile',
    errors: Dict[str, str],
) -> Optional['SubagentProfile']:
    """Merge parent profiles with a child profile.

    Args:
        child_name: Name of the child profile (for error messages).
        parents: Resolved parent profiles in declaration order.
        child: The child profile with its own overrides.
        errors: Error dict to populate on conflict.

    Returns:
        Merged SubagentProfile, or None if conflicts detected.
    """
    # --- Collection fields: union ---

    # plugins: parents first (in order), then child, deduplicated
    seen_plugins: set = set()
    merged_plugins: List[str] = []
    for parent in parents:
        for p in parent.plugins:
            if p not in seen_plugins:
                seen_plugins.add(p)
                merged_plugins.append(p)
    for p in child.plugins:
        if p not in seen_plugins:
            seen_plugins.add(p)
            merged_plugins.append(p)

    # preloaded_plugins: union
    merged_preloaded: set = set()
    for parent in parents:
        merged_preloaded |= parent.preloaded_plugins
    merged_preloaded |= child.preloaded_plugins

    # tool_scopes: per-plugin override (dict update — child wins on a
    # plugin key it specifies; a parent's scope for a plugin the child
    # doesn't re-scope survives).  Note this means a child that re-lists
    # a plugin WITHOUT a ``tools:`` modifier inherits the parent's
    # allow-list — consistent with preloaded_plugins' additive
    # semantics.  To widen back to all tools, a child must list the
    # plugin with an explicit ``tools:[...]`` enumerating the wider set.
    merged_tool_scopes: Dict[str, List[str]] = {}
    for parent in parents:
        merged_tool_scopes.update(getattr(parent, 'tool_scopes', {}) or {})
    merged_tool_scopes.update(child.tool_scopes or {})

    # env: merge with conflict detection
    merged_env: Dict[str, str] = {}
    env_sources: Dict[str, str] = {}  # key → source profile name
    conflict_details: List[str] = []

    for parent in parents:
        for key, val in parent.env.items():
            if key in merged_env and merged_env[key] != val:
                # Conflict between parents
                if key not in child.env:
                    conflict_details.append(
                        f"  env['{key}']: '{env_sources[key]}' sets "
                        f"'{merged_env[key]}', '{parent.name}' sets '{val}'"
                    )
            else:
                merged_env[key] = val
                env_sources[key] = parent.name
    # Child overrides last
    merged_env.update(child.env)

    # plugin_configs: deep merge by plugin name
    merged_configs: Dict[str, Dict[str, Any]] = {}
    config_sources: Dict[str, Dict[str, str]] = {}  # plugin → key → source

    for parent in parents:
        for plugin_name, config in parent.plugin_configs.items():
            if plugin_name not in merged_configs:
                merged_configs[plugin_name] = {}
                config_sources[plugin_name] = {}
            for key, val in config.items():
                existing = merged_configs[plugin_name].get(key)
                if existing is not None and existing != val:
                    src = config_sources[plugin_name].get(key, "?")
                    # Only conflict if child doesn't override this specific key
                    child_plugin_config = child.plugin_configs.get(plugin_name, {})
                    if key not in child_plugin_config:
                        conflict_details.append(
                            f"  plugin_configs['{plugin_name}']['{key}']: "
                            f"'{src}' sets {existing!r}, '{parent.name}' sets {val!r}"
                        )
                else:
                    merged_configs[plugin_name][key] = val
                    config_sources[plugin_name][key] = parent.name
    # Child overrides last (per-plugin, per-key)
    for plugin_name, config in child.plugin_configs.items():
        if plugin_name not in merged_configs:
            merged_configs[plugin_name] = {}
        merged_configs[plugin_name].update(config)

    # --- Scalar fields: agreement-or-override ---

    # Collect parent values for scalar fields
    scalar_conflicts: List[str] = []

    def _resolve_scalar(field_name: str, child_val, default=None):
        """Resolve a scalar field across parents + child."""
        parent_vals = {}
        for parent in parents:
            val = getattr(parent, field_name)
            if val is not None and val != default:
                parent_vals[parent.name] = val

        # Child explicitly overrides
        if child_val is not None and child_val != default:
            return child_val

        # No parents set it
        if not parent_vals:
            return child_val  # keep child default

        # All parents agree
        unique_vals = set(str(v) for v in parent_vals.values())
        if len(unique_vals) == 1:
            return next(iter(parent_vals.values()))

        # Conflict
        details = ", ".join(
            f"'{name}': {val!r}" for name, val in parent_vals.items()
        )
        scalar_conflicts.append(
            f"  {field_name}: {details}"
        )
        return child_val

    merged_model = _resolve_scalar('model', child.model)
    merged_provider = _resolve_scalar('provider', child.provider)

    # max_turns: most restrictive (minimum) across parents, child can override
    parent_max_turns = [p.max_turns for p in parents if p.max_turns != 10]
    if child.max_turns != 10:
        merged_max_turns = child.max_turns
    elif parent_max_turns:
        merged_max_turns = min(parent_max_turns)
    else:
        merged_max_turns = 10

    # gc: agreement-or-override (compare as dicts for equality)
    merged_cache = _resolve_scalar('cache', child.cache)
    merged_gc = _resolve_scalar('gc', child.gc)
    # trace: scalar-override.  A stage that redirects its trace redirects
    # the whole of it -- merging session_log from one layer with
    # provider_log from another produces a split diagnosis nobody asked
    # for, which is the class of failure the block exists to prevent.
    merged_trace = _resolve_scalar('trace', child.trace)

    # runtime_limits: scalar-override (parents must agree or child
    # overrides).  Compared via str() inside _resolve_scalar — frozen
    # dataclasses with the same field values produce identical reprs,
    # so two parents declaring the same limits don't conflict.
    merged_runtime_limits = _resolve_scalar('runtime_limits', child.runtime_limits)

    # model_tiers: scalar-override (the child's whole tier-config wins; inherit
    # the parent's when the child declares none).  default={} so an unset child
    # ({}) is treated as "not set" and inherits.  WAS DROPPED entirely pre-fix:
    # the construction below omitted model_tiers, so inherits/set-based tiered
    # profiles silently lost their tiers and fell back to single-model.
    merged_model_tiers = _resolve_scalar(
        'model_tiers', child.model_tiers, default={})

    # budget_control: NOT a plain scalar-override.  ``limits`` merge
    # MIN-WINS across parents + child (a child may only TIGHTEN a
    # ceiling — it must never grant itself a bigger budget than the
    # profile that spawned it, the same safety direction as max_turns
    # above), while ``degrade`` is scalar-override (the child's whole
    # ladder wins, matching model_tiers).
    merged_budget_control = _merge_budget_control(parents, child)

    # completion_payload_schema: scalar-override (parents must agree or
    # child overrides). Inline dicts and string paths both compared as-is
    # via str() in _resolve_scalar.
    merged_completion_schema = _resolve_scalar(
        'completion_payload_schema', child.completion_payload_schema
    )

    # spawn_payload_schema: same scalar-override semantics — symmetric
    # to completion_payload_schema but constrains the agent_params dict
    # passed to spawn_subagent at the spawn boundary.
    merged_spawn_schema = _resolve_scalar(
        'spawn_payload_schema', child.spawn_payload_schema
    )

    # completion_processors: concatenation across parent → child, minus
    # whatever the child declines by name in
    # ``suppress_inherited_processors``.  Each processor is independent
    # (writes a different artefact or checks a different invariant);
    # concatenating fires all of them.  Child entries appear last; the
    # framework invokes them sequentially and aggregates ALL errors so
    # the agent sees the full set on the retry prompt rather than
    # playing whack-a-mole turn by turn.  See
    # :func:`_merge_completion_processors` for the opt-out's rules.
    merged_completion_processors, processor_error = (
        _merge_completion_processors(parents, child)
    )

    # --- Concatenation: system_instructions ---
    instruction_parts = []
    for parent in parents:
        if parent.system_instructions:
            instruction_parts.append(parent.system_instructions)
    if child.system_instructions:
        instruction_parts.append(child.system_instructions)
    merged_instructions = "\n\n".join(instruction_parts) if instruction_parts else None

    # --- Check for conflicts ---
    if processor_error:
        errors[child_name] = processor_error
        return None

    all_conflicts = conflict_details + scalar_conflicts
    if all_conflicts:
        conflict_msg = (
            f"Profile '{child_name}' inherits from "
            f"{[p.name for p in parents]}.\n"
            f"Conflicts (override in '{child_name}' to resolve):\n"
            + "\n".join(all_conflicts)
        )
        errors[child_name] = conflict_msg
        return None

    # ``suppress_base_instructions`` follows UNION semantics: a piece is
    # suppressed if ANY layer in the chain (parents or child) suppresses
    # it.  Rationale: a base saying "drop the framework constants, I'm
    # minimal" shouldn't be silently overridable by an inheritor; an
    # inheritor that genuinely wants a layer back should not inherit from
    # a parent that drops it.  Each profile's field is already the
    # canonical frozenset (normalized in __post_init__), so the merge is
    # a plain set union.
    merged_suppress_base = frozenset().union(
        *(getattr(p, 'suppress_base_instructions', frozenset()) for p in parents),
        getattr(child, 'suppress_base_instructions', frozenset()),
    )

    # ``apparmor`` follows OR semantics: True if any layer in the chain
    # (parents or child) sets it True.  Same rationale as
    # ``suppress_base_instructions`` — a security primitive shouldn't be
    # silently downgradeable by an inheritor.  An inheritor that
    # genuinely wants unconfined operation should not inherit from a
    # confined parent.
    merged_apparmor = any(
        getattr(p, 'apparmor', False) for p in parents
    ) or getattr(child, 'apparmor', False)

    # ``apparmor_fragments`` follows CHILD-REPLACES-PARENT semantics
    # (Piece 1, 2026-05-14).  Distinct from ``apparmor`` above:
    #
    # - ``apparmor`` is a bool, OR-merged so children can't silently
    #   downgrade a confined parent.
    # - ``apparmor_fragments`` is a list expressing WHICH fragments
    #   the profile wants.  Replace lets cascade authors SCOPE DOWN
    #   from a broader parent set — necessary for least-privilege,
    #   since union would only let children ADD permissions, never
    #   remove.  Matches ``model`` field's child-wins rationale.
    #
    # Resolution: walk the inheritance chain in order child → parents.
    # The first profile that declares the field (i.e. has a non-None
    # value) wins.  ``None`` everywhere → resolved value is ``None``
    # (workspace-default "compose all fragments" applies at render
    # time).  See ``project_backlog_per_profile_apparmor_fragments``
    # for the cascade footgun this design closes.
    merged_apparmor_fragments: Optional[List[str]] = None
    child_fragments = getattr(child, 'apparmor_fragments', None)
    if child_fragments is not None:
        merged_apparmor_fragments = list(child_fragments)
    else:
        for p in parents:
            parent_fragments = getattr(p, 'apparmor_fragments', None)
            if parent_fragments is not None:
                merged_apparmor_fragments = list(parent_fragments)
                break

    # quirks: dict-union with child-wins-on-key-collision.  Same shape
    # as plugin_configs / env merging — parent keys flow through, child
    # keys override.  Rationale: quirks are additive declarations of
    # which provider workarounds the profile opts into; a child that
    # disables a parent's quirk does so explicitly with ``key: false``,
    # not by omitting the key.
    merged_quirks: Dict[str, Any] = {}
    for parent in parents:
        merged_quirks.update(getattr(parent, 'quirks', {}) or {})
    merged_quirks.update(child.quirks or {})

    return SubagentProfile(
        name=child.name,
        description=child.description,
        plugins=merged_plugins,
        preloaded_plugins=merged_preloaded,
        tool_scopes=merged_tool_scopes,
        plugin_configs=merged_configs,
        system_instructions=merged_instructions,
        suppress_base_instructions=merged_suppress_base,
        model=merged_model,
        provider=merged_provider,
        max_turns=merged_max_turns,
        gc=merged_gc,
        cache=merged_cache,
        trace=merged_trace,
        env=merged_env,
        inherits=None,  # Fully resolved
        completion_payload_schema=merged_completion_schema,
        spawn_payload_schema=merged_spawn_schema,
        completion_processors=merged_completion_processors,
        # Consumed by the merge above — the resolved profile carries no
        # residual suppression, so a grandchild inheriting THIS profile
        # neither re-applies it nor trips the "matched nothing" error on
        # a processor that is already gone.
        suppress_inherited_processors=[],
        runtime_limits=merged_runtime_limits,
        budget_control=merged_budget_control,
        model_tiers=merged_model_tiers,
        apparmor=merged_apparmor,
        apparmor_fragments=merged_apparmor_fragments,
        quirks=merged_quirks,
    )


def _parse_profile_file(
    file_path: Path,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Parse a single profile file (JSON or YAML).

    Args:
        file_path: Path to the profile file.

    Returns:
        Tuple of ``(profile_name, profile_data, None)`` on success, or
        ``(None, None, error_message)`` on parse/read failure.
    """
    try:
        content = file_path.read_text(encoding='utf-8')

        if file_path.suffix in ('.yaml', '.yml'):
            try:
                import yaml
                data = yaml.safe_load(content)
            except ImportError:
                msg = f"PyYAML not installed, cannot parse YAML profile: {file_path.name}"
                logger.warning(msg)
                return None, None, msg
        elif file_path.suffix == '.json':
            data = json.loads(content)
        else:
            logger.debug("Skipping non-profile file: %s", file_path)
            return None, None, None

        if not isinstance(data, dict):
            msg = f"Profile file must contain a JSON object: {file_path.name}"
            logger.warning(msg)
            return None, None, msg

        # Profile name is either explicit 'name' field or derived from filename
        name = data.get('name') or file_path.stem

        return name, data, None

    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {file_path.name}: {e}"
        logger.warning(msg)
        return None, None, msg
    except Exception as e:
        msg = f"Error reading {file_path.name}: {e}"
        logger.warning(msg)
        return None, None, msg


@dataclass
class ProfileDiscoveryResult:
    """Result of profile discovery, carrying both valid profiles and parse errors.

    Callers that only need the profiles dict can use ``result.profiles``.
    Callers that want to surface actionable diagnostics (e.g. when a
    requested profile is missing) should also inspect ``result.errors``
    — a mapping from the profile file stem (e.g. ``"github-resolver"``)
    to the human-readable error message.
    """

    profiles: Dict[str, 'SubagentProfile'] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)


def _scan_profiles_dir(
    directory: Path,
    profiles: Dict[str, 'SubagentProfile'],
    errors: Dict[str, str],
) -> None:
    """Scan a directory for profile files and populate profiles/errors dicts.

    Existing entries in ``profiles`` are never overwritten, so earlier
    directories (higher precedence) win over later ones.

    Args:
        directory: Directory to scan for .json/.yaml/.yml profile files.
        profiles: Accumulator dict — discovered profiles are added here.
        errors: Accumulator dict — parse errors are added here.
    """
    try:
        if not directory.is_dir():
            return
        entries = list(directory.iterdir())
    except OSError as exc:
        # The directory is inaccessible: missing, or a confined session
        # correctly denied this tier (e.g. ~/.jaato/profiles under AppArmor —
        # is_dir()/iterdir() raise PermissionError, not return False).  This is
        # an OPTIONAL tier: skip it so the other tiers (workspace, premium)
        # still discover.  A denied tier must NEVER abort the whole discovery.
        logger.debug("Profiles tier %s not scannable (%s); skipping", directory, exc)
        return

    # Track names actually registered IN THIS PASS so the summary
    # log line below reports the correct provenance.  Pre-fix the
    # log printed ``profiles.items()`` which is the cumulative dict
    # across all preceding passes — names from earlier directories
    # appeared in the line claiming they came from ``directory``,
    # which is misleading to operators ("why does my home dir have
    # 'codegen' in it?  it doesn't.").  See 2026-05-15 finding.
    found_names: List[str] = []
    found = 0
    for file_path in entries:
        if not file_path.is_file():
            continue
        if file_path.suffix not in ('.json', '.yaml', '.yml'):
            continue

        name, data, error = _parse_profile_file(file_path)
        if error:
            if file_path.stem not in errors:
                errors[file_path.stem] = error
            continue
        if name is None or data is None:
            continue
        if name in profiles:
            continue  # higher-precedence source already registered this name

        cache_config = parse_cache_block(data)

        gc_config = parse_gc_block(data)
        trace_config = parse_trace_block(data)

        runtime_limits = None
        if 'runtime_limits' in data and data['runtime_limits']:
            try:
                runtime_limits = RuntimeLimits.from_dict(data['runtime_limits'])
            except (ValueError, TypeError) as exc:
                err = f"Invalid runtime_limits in profile '{name}': {exc}"
                logger.warning(err)
                if name not in errors:
                    errors[name] = err
                continue

        budget_control = None
        if 'budget_control' in data and data['budget_control']:
            try:
                budget_control = BudgetControlConfig.from_dict(
                    data['budget_control'])
            except (ValueError, TypeError) as exc:
                err = f"Invalid budget_control in profile '{name}': {exc}"
                logger.warning(err)
                if name not in errors:
                    errors[name] = err
                continue

        # ``plugins:`` is a REQUIRED profile key as of server 0.6.x
        # (this PR).  Absent vs. explicitly-empty have meaningfully
        # different semantics:
        #   - Absent  → was conflated with "load all exposed plugins"
        #               by server/core.py:_apply_profile_overrides via
        #               a falsy check (``if self._profile.plugins``)
        #               that swallowed both ``None`` and ``[]``.  The
        #               2026-06-07 vLLM smoke investigation surfaced
        #               this: ``plugins: []`` in the profile YAML
        #               produced ~30 tools on the wire instead of 0.
        #   - Empty   → explicit "no non-framework plugins" — the
        #               framework still wires permission, reliability,
        #               and lifecycle (signal_completion) regardless.
        # Requiring the key forces profile authors to pick one
        # intentionally and eliminates the ambiguous middle case.
        if 'plugins' not in data:
            err = (
                f"Profile '{name}' is missing the required 'plugins' "
                f"key.  Use 'plugins: []' for the minimal framework "
                f"set (permission, reliability, lifecycle only), or "
                f"list plugin names to expose (e.g. 'plugins: "
                f"[cli, todo]')."
            )
            logger.warning(err)
            if name not in errors:
                errors[name] = err
            continue
        raw_plugins = data['plugins']
        clean_plugins, preloaded, tool_scopes = parse_plugin_list(raw_plugins)

        # Parse env: must be a flat dict of string→string
        raw_env = data.get('env', {})
        env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}

        raw_model_tiers = data.get('model_tiers') or {}
        model_tiers = (
            {str(k): v for k, v in raw_model_tiers.items()}
            if isinstance(raw_model_tiers, dict) else {}
        )
        if model_tiers and data.get('model'):
            logger.warning(
                "Profile '%s' declares both 'model' and 'model_tiers'; "
                "'model' will be ignored — the active model is selected "
                "per turn from 'model_tiers[<active_tier>]', starting at "
                "the initial tier. Keeping 'model' is harmless; removing it "
                "is also fine (the session bootstraps from the initial "
                "tier).", name,
            )

        raw_quirks = data.get('quirks') or {}
        quirks = (
            {str(k): v for k, v in raw_quirks.items()}
            if isinstance(raw_quirks, dict) else {}
        )

        profiles[name] = SubagentProfile(
            name=name,
            description=data.get('description', ''),
            plugins=clean_plugins,
            preloaded_plugins=preloaded,
            tool_scopes=tool_scopes,
            plugin_configs=data.get('plugin_configs', {}),
            system_instructions=data.get('system_instructions'),
            suppress_base_instructions=data.get('suppress_base_instructions', False),
            model=data.get('model'),
            provider=data.get('provider'),
            max_turns=data.get('max_turns', 10),
            gc=gc_config,
            cache=cache_config,
            trace=trace_config,
            env=env,
            inherits=_normalize_inherits(data.get('inherits')),
            completion_payload_schema=data.get('completion_payload_schema'),
            spawn_payload_schema=data.get('spawn_payload_schema'),
            completion_processors=_parse_completion_processors(data.get('completion_processors')),
            suppress_inherited_processors=_normalize_suppress_inherited_processors(
                data.get('suppress_inherited_processors')),
            runtime_limits=runtime_limits,
            budget_control=budget_control,
            model_tiers=model_tiers,
            apparmor=bool(data.get('apparmor', False)),
            apparmor_fragments=_normalize_apparmor_fragments(data.get('apparmor_fragments')),
            quirks=quirks,
        )
        if data.get('system_instructions'):
            import warnings
            warnings.warn(
                f"Profile '{name}' has 'system_instructions' which is deprecated. "
                f"Move the prompt to .jaato/agents/{name}.md and remove "
                f"system_instructions from the profile.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(
                "Profile '%s' has deprecated 'system_instructions'. "
                "Move to .jaato/agents/%s.md instead.",
                name, name,
            )
        found += 1
        found_names.append(name)
        logger.debug("Discovered profile '%s' from %s", name, file_path)

    if found:
        logger.info(
            "Discovered %d profile(s) from %s: %s",
            found, directory,
            ", ".join(found_names),
        )


def resolve_agent(
    agent_name: str,
    params: Optional[Dict[str, str]],
    workspace_path: Optional[str],
    config_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve an agent by name from .jaato/agents/ and .jaato/prompts/.

    Scans agent directories (workspace then user-level), reads the markdown
    file, parses frontmatter, substitutes params, and returns the rendered
    system instructions.

    The single source of truth for agent-persona resolution: the daemon's
    ``SessionManager._resolve_agent`` delegates here, and the embedded
    in-process client (``jaato_embedded.client``) imports it directly — so a
    daemon-free embedded session resolves ``agent=<name>`` the same way the
    daemon does, without depending on ``server`` (mirrors how
    ``shared.config_resolver.resolve_secret_uri`` was lifted out of the daemon).

    Args:
        agent_name: Agent name (filename stem).
        params: Parameter values for ``{{param}}`` placeholders.
        workspace_path: Workspace directory for agent resolution.
        config_root: Optional override for the workspace tier.  When set, scans
            ``<config_root>/agents/`` and ``<config_root>/prompts/`` instead of
            the workspace-anchored paths.  See
            :func:`shared.config_resolver.resolve_config_search_path`.

    Returns:
        Dict with ``system_instructions``, ``description``, ``default_profile``,
        ``missing_params``, ``source_path``, or ``None`` if not found.
    """
    search_dirs = []
    if config_root:
        cr = Path(config_root).expanduser().resolve()
        search_dirs.append(cr / "agents")
        search_dirs.append(cr / "prompts")
    elif workspace_path:
        search_dirs.append(Path(workspace_path) / ".jaato" / "agents")
        search_dirs.append(Path(workspace_path) / ".jaato" / "prompts")
    search_dirs.append(Path.home() / ".jaato" / "agents")
    search_dirs.append(Path.home() / ".jaato" / "prompts")

    # Find the agent file
    agent_path = None
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        # Single file: agents/gen-references.md
        candidate = search_dir / f"{agent_name}.md"
        if candidate.is_file():
            agent_path = candidate
            break
        # Directory: agents/gen-references/PROMPT.md
        candidate_dir = search_dir / agent_name
        if candidate_dir.is_dir():
            for entry_name in ("PROMPT.md", "SKILL.md"):
                entry = candidate_dir / entry_name
                if entry.is_file():
                    agent_path = entry
                    break
            if agent_path:
                break

    if not agent_path:
        return None

    raw = agent_path.read_text(encoding="utf-8")

    # Parse YAML frontmatter
    frontmatter: Dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
        if match:
            try:
                import yaml
                frontmatter = yaml.safe_load(match.group(1)) or {}
            except Exception:
                pass
            body = raw[match.end():]

    # Substitute params
    effective_params = dict(params or {})
    param_defs = frontmatter.get("params", {})

    # Apply frontmatter defaults for params not provided
    if isinstance(param_defs, dict):
        for pname, pdef in param_defs.items():
            if pname not in effective_params:
                if isinstance(pdef, dict) and "default" in pdef:
                    default = pdef["default"]
                    if default is not None:
                        effective_params[pname] = str(default)

    # Pre-scan: collect inline ``{{name:default}}`` defaults declared anywhere
    # in the body so a later bare ``{{name}}`` can fall back to the same
    # default.  Without this, an agent that uses a parameter both with and
    # without an inline default would mark it missing on the bare occurrences
    # and leave literal ``{{name}}`` placeholders in the rendered system
    # instructions — which then bloat every turn's prompt.
    inline_defaults: Dict[str, str] = {}
    inline_pattern = re.compile(r"\{\{(\w+)(?::([^}]*))?\}\}")
    for m in inline_pattern.finditer(body):
        name = m.group(1)
        default = m.group(2)
        if default is not None and name not in inline_defaults:
            inline_defaults[name] = default

    # Use a set for O(1) dedup; the public missing list is built once at the
    # end so the same name never appears twice.
    missing_set: set = set()

    def replace_param(m: "re.Match") -> str:
        name = m.group(1)
        inline_default = m.group(2)

        if name in effective_params:
            return effective_params[name]
        if inline_default is not None:
            return inline_default
        if name in inline_defaults:
            return inline_defaults[name]
        missing_set.add(name)
        return m.group(0)  # Keep unresolved (debugging signal)

    rendered = inline_pattern.sub(replace_param, body)

    return {
        "system_instructions": rendered,
        "description": frontmatter.get("description", ""),
        "default_profile": frontmatter.get("default_profile"),
        "missing_params": sorted(missing_set),
        "source_path": str(agent_path),
    }


def discover_profiles(
    profiles_dir: str,
    base_path: Optional[str] = None,
    config_root: Optional[str] = None,
    force_profile_set: Optional[str] = None,
) -> ProfileDiscoveryResult:
    """Discover subagent profiles from multiple sources.

    Scans up to three locations for ``.json`` / ``.yaml`` / ``.yml``
    profile files, in decreasing order of precedence:

    1. **Workspace** —
       * if ``config_root`` is set: ``<config_root>/profiles/``;
       * else: ``{base_path}/{profiles_dir}`` (today's behavior).
    2. **User**      — ``~/.jaato/profiles/``
    3. **Premium**   — profiles registered via ``jaato.premium`` entry points

    When the same profile name appears in multiple sources, the
    higher-precedence source wins.

    Args:
        profiles_dir: Directory path to scan (relative or absolute).
            Ignored when ``config_root`` is set.
        base_path: Base path for resolving relative profiles_dir.
                   Defaults to current working directory.
        config_root: Optional override for the workspace tier.  When set,
            scans ``<config_root>/profiles/`` instead of
            ``{base_path}/{profiles_dir}``.  See
            :func:`shared.config_resolver.resolve_config_search_path`.
        force_profile_set: Optional override for the profile-set name.
            When set, takes precedence over the ``JAATO_PROFILE_SET``
            env-var lookup at step 1.a — the named set is scanned even
            if the env var is empty or names a different set.  Used by
            callers that resolve a qualified ``set/name`` profile path
            and need the matching set's subdirectory to be scanned
            regardless of the per-session env state.  When ``None`` /
            empty, falls back to the env-var read (pre-existing
            behavior).

    Returns:
        ProfileDiscoveryResult with discovered profiles and any parse errors.
    """
    # Server 0.6.68+: read workspace_root / config_root via the per-task
    # ``ContextVar`` first (race-free across concurrent sessions); fall
    # back to ``os.environ`` for daemon-startup callers.  Pre-0.6.68
    # this read directly from ``os.environ``, which clobbered across
    # concurrent overlapping sessions and made the daemon's profile
    # discovery for client A read client B's workspace.
    from shared.session_context import get_config_root, get_workspace_root
    if base_path is None:
        base_path = get_workspace_root() or os.getcwd()

    # When no explicit ``config_root`` is provided, fall back to the
    # session-scoped value set by ``JaatoServer._in_workspace`` —
    # plugins whose ``initialize()`` runs inside that context (including
    # the subagent plugin's first call here) pick up the per-session
    # override even though the registry's ``set_config_root`` broadcast
    # hasn't fired yet (broadcasts run AFTER plugin init).
    effective_config_root = config_root or get_config_root()

    profiles: Dict[str, SubagentProfile] = {}
    errors: Dict[str, str] = {}

    # 1.a Workspace profile-set overlay (optional).
    #
    # When ``JAATO_PROFILE_SET`` is set and ``<config_root>/profiles/<set>/``
    # exists, scan it FIRST so its entries land in the profiles dict before
    # the regular profiles/ scan — ``_scan_profiles_dir`` skips already-present
    # names, so first-scanned wins.  Used by the model-set switcher (e.g.
    # ``--model-set dumb``) to override per-agent ``model`` / ``provider`` /
    # ``plugin_configs`` while inheriting everything else from the regular
    # ``profiles/`` tier (typically via ``inherits: [_base_<agent>]``).
    #
    # The set lives as a subdirectory under ``profiles/`` (e.g.
    # ``profiles/dumb/``, ``profiles/tailored/``) — the regular scan is
    # non-recursive so subdirectories aren't accidentally pulled into the
    # default set.
    #
    # When the env var isn't set, this is a no-op and behaviour matches
    # the pre-existing single-dir scan.
    #
    # Resolution rule (per the workspace-tied env-var contract): the
    # value is read via ``get_session_env``, which checks the per-
    # session contextvar (populated from the session's ``env_file``)
    # before falling back to the daemon's ``os.environ``.  This keeps
    # profile-set selection workspace-scoped — different sessions on
    # the same daemon can run different sets concurrently, and switching
    # sets does NOT require restarting the daemon.
    from shared.session_context import get_session_env
    # ``force_profile_set`` (explicit kwarg) wins over the env-var read
    # so callers resolving a qualified ``set/name`` path can pin the
    # set without mutating the per-session env contextvar.
    profile_set = force_profile_set or get_session_env('JAATO_PROFILE_SET')
    if profile_set and effective_config_root:
        set_path = (
            Path(effective_config_root).expanduser().resolve()
            / "profiles" / profile_set
        )
        _scan_profiles_dir(set_path, profiles, errors)
    elif profile_set and not effective_config_root:
        # No ``config_root`` override — fall back to scanning
        # ``<base_path>/<profiles_dir>/<set>/`` so qualified resolution
        # still works in test harnesses and ad-hoc layouts that don't
        # set a config_root.
        fallback_set_path = Path(profiles_dir)
        if not fallback_set_path.is_absolute():
            fallback_set_path = Path(base_path) / fallback_set_path
        _scan_profiles_dir(fallback_set_path / profile_set, profiles, errors)

    # 1.b Workspace tier — config_root override takes precedence; fall
    #    back to <base_path>/<profiles_dir> when no override is in effect.
    if effective_config_root:
        profiles_path = Path(effective_config_root).expanduser().resolve() / "profiles"
    else:
        profiles_path = Path(profiles_dir)
        if not profiles_path.is_absolute():
            profiles_path = Path(base_path) / profiles_path
    _scan_profiles_dir(profiles_path, profiles, errors)

    # 2. User-level profiles from ~/.jaato/profiles/
    #    Workspace profiles take precedence.
    user_profiles_path = Path.home() / ".jaato" / "profiles"
    _scan_profiles_dir(user_profiles_path, profiles, errors)

    # 3. Premium entry-point profiles (if installed).
    #    Workspace and user profiles take precedence over premium ones.
    premium_profiles = _discover_premium_profiles()
    for name, profile in premium_profiles.items():
        if name not in profiles:
            profiles[name] = profile

    # Resolve inheritance after all sources are scanned
    resolved, inheritance_errors = resolve_profiles(profiles)
    errors.update(inheritance_errors)

    return ProfileDiscoveryResult(profiles=resolved, errors=errors)


def _discover_premium_profiles() -> Dict[str, 'SubagentProfile']:
    """Discover profiles provided by the ``jaato.premium`` → ``profiles`` entry point.

    Returns an empty dict if no premium package is installed.
    """
    try:
        from shared.jaato_runtime import _get_premium_content_path
    except ImportError:
        return {}

    premium_dir = _get_premium_content_path("profiles")
    if not premium_dir or not Path(premium_dir).is_dir():
        return {}

    # Re-use discover_profiles logic on the premium path, but pass
    # an absolute path to avoid workspace-relative resolution.
    profiles: Dict[str, SubagentProfile] = {}
    for file_path in Path(premium_dir).iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix not in ('.json', '.yaml', '.yml'):
            continue

        name, data, _error = _parse_profile_file(file_path)
        if name is None or data is None:
            continue

        cache_config = parse_cache_block(data)

        gc_config = parse_gc_block(data)
        trace_config = parse_trace_block(data)

        runtime_limits = None
        if 'runtime_limits' in data and data['runtime_limits']:
            try:
                runtime_limits = RuntimeLimits.from_dict(data['runtime_limits'])
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping premium profile '%s': invalid runtime_limits: %s",
                    name, exc,
                )
                continue

        budget_control = None
        if 'budget_control' in data and data['budget_control']:
            try:
                budget_control = BudgetControlConfig.from_dict(
                    data['budget_control'])
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping premium profile '%s': invalid budget_control: %s",
                    name, exc,
                )
                continue

        # ``plugins:`` is REQUIRED on premium profiles too — see the
        # workspace scanner for the full rationale.
        if 'plugins' not in data:
            logger.warning(
                "Skipping premium profile '%s': missing required "
                "'plugins' key.  Use 'plugins: []' for the minimal "
                "framework set, or list plugin names to expose.",
                name,
            )
            continue
        raw_plugins = data['plugins']
        clean_plugins, preloaded, tool_scopes = parse_plugin_list(raw_plugins)

        raw_env = data.get('env', {})
        env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}

        raw_model_tiers = data.get('model_tiers') or {}
        model_tiers = (
            {str(k): v for k, v in raw_model_tiers.items()}
            if isinstance(raw_model_tiers, dict) else {}
        )
        if model_tiers and data.get('model'):
            logger.warning(
                "Premium profile '%s' declares both 'model' and 'model_tiers'; "
                "'model' will be ignored.", name,
            )

        raw_quirks = data.get('quirks') or {}
        quirks = (
            {str(k): v for k, v in raw_quirks.items()}
            if isinstance(raw_quirks, dict) else {}
        )

        profile = SubagentProfile(
            name=name,
            description=data.get('description', ''),
            plugins=clean_plugins,
            preloaded_plugins=preloaded,
            tool_scopes=tool_scopes,
            plugin_configs=data.get('plugin_configs', {}),
            system_instructions=data.get('system_instructions'),
            suppress_base_instructions=data.get('suppress_base_instructions', False),
            model=data.get('model'),
            provider=data.get('provider'),
            max_turns=data.get('max_turns', 10),
            gc=gc_config,
            cache=cache_config,
            trace=trace_config,
            env=env,
            inherits=_normalize_inherits(data.get('inherits')),
            completion_payload_schema=data.get('completion_payload_schema'),
            spawn_payload_schema=data.get('spawn_payload_schema'),
            completion_processors=_parse_completion_processors(data.get('completion_processors')),
            suppress_inherited_processors=_normalize_suppress_inherited_processors(
                data.get('suppress_inherited_processors')),
            runtime_limits=runtime_limits,
            budget_control=budget_control,
            model_tiers=model_tiers,
            apparmor=bool(data.get('apparmor', False)),
            apparmor_fragments=_normalize_apparmor_fragments(data.get('apparmor_fragments')),
            quirks=quirks,
        )
        profiles[name] = profile
        logger.debug("Discovered premium profile '%s' from %s", name, file_path)

    if profiles:
        logger.info(
            "Discovered %d premium profile(s): %s",
            len(profiles), ", ".join(profiles.keys())
        )
    return profiles


def validate_profile(data: Any) -> Tuple[bool, List[str], List[str]]:
    """Validate a subagent profile JSON structure.

    Checks required fields, type constraints, and GC sub-configuration
    for a single profile definition (the format stored in .jaato/profiles/*.json).

    Args:
        data: Parsed JSON data from a profile file.

    Returns:
        Tuple of (is_valid, errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(data, dict):
        return False, ["File must contain a JSON object"], []

    # Required fields
    if not data.get("name"):
        errors.append("'name' is required")
    if not data.get("description"):
        errors.append("'description' is required")

    # plugins: list of strings
    plugins = data.get("plugins")
    if plugins is not None:
        if not isinstance(plugins, list):
            errors.append("'plugins' must be an array")
        elif not all(isinstance(p, str) for p in plugins):
            errors.append("'plugins' must contain only strings")

    # plugin_configs: dict of dicts
    plugin_configs = data.get("plugin_configs")
    if plugin_configs is not None:
        if not isinstance(plugin_configs, dict):
            errors.append("'plugin_configs' must be an object")
        else:
            for key, val in plugin_configs.items():
                if not isinstance(val, dict):
                    errors.append(f"plugin_configs['{key}'] must be an object")

    # max_turns: positive int
    max_turns = data.get("max_turns")
    if max_turns is not None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            errors.append("'max_turns' must be an integer")
        elif max_turns <= 0:
            errors.append("'max_turns' must be a positive integer")

    # model: string or null
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        errors.append("'model' must be a string or null")

    # provider: string or null
    provider = data.get("provider")
    if provider is not None and not isinstance(provider, str):
        errors.append("'provider' must be a string or null")

    # env: dict of string keys to string values, or null
    env_data = data.get("env")
    if env_data is not None:
        if not isinstance(env_data, dict):
            errors.append("'env' must be an object or null")
        else:
            for key, val in env_data.items():
                if not isinstance(key, str):
                    errors.append(f"env key {key!r} must be a string")
                if not isinstance(val, str):
                    errors.append(f"env['{key}'] must be a string")

    # inherits: string, list of strings, or null
    inherits = data.get("inherits")
    if inherits is not None:
        if isinstance(inherits, str):
            pass  # Single string is valid (normalized to list during parsing)
        elif isinstance(inherits, list):
            for item in inherits:
                if not isinstance(item, str):
                    errors.append("'inherits' must be a string or list of strings")
                    break
        else:
            errors.append("'inherits' must be a string or list of strings")

    # GC sub-validation
    gc_data = data.get("gc")
    if gc_data is not None:
        if not isinstance(gc_data, dict):
            errors.append("'gc' must be an object or null")
        else:
            valid_gc_types = ("truncate", "summarize", "hybrid", "budget")
            gc_type = gc_data.get("type", "truncate")
            if gc_type not in valid_gc_types:
                errors.append(
                    f"gc.type '{gc_type}' is invalid. "
                    f"Must be one of: {', '.join(valid_gc_types)}"
                )

            # Numeric range checks
            for field_name in ("threshold_percent", "target_percent", "pressure_percent"):
                val = gc_data.get(field_name)
                if val is not None:
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        errors.append(f"gc.{field_name} must be a number")
                    elif val < 0 or val > 100:
                        errors.append(f"gc.{field_name} must be between 0 and 100")

            gc_preserve = gc_data.get("preserve_recent_turns")
            if gc_preserve is not None:
                if not isinstance(gc_preserve, int) or isinstance(gc_preserve, bool):
                    errors.append("gc.preserve_recent_turns must be an integer")
                elif gc_preserve < 0:
                    errors.append("gc.preserve_recent_turns must be non-negative")

            gc_max_turns = gc_data.get("max_turns")
            if gc_max_turns is not None:
                if not isinstance(gc_max_turns, int) or isinstance(gc_max_turns, bool):
                    errors.append("gc.max_turns must be an integer")
                elif gc_max_turns <= 0:
                    errors.append("gc.max_turns must be a positive integer")

    # runtime_limits sub-validation: delegate to RuntimeLimits.from_dict
    # which raises ValueError for any out-of-range value (kept in one
    # place rather than duplicating the rules here).
    runtime_data = data.get("runtime_limits")
    if runtime_data is not None:
        if not isinstance(runtime_data, dict):
            errors.append("'runtime_limits' must be an object or null")
        else:
            try:
                RuntimeLimits.from_dict(runtime_data)
            except (ValueError, TypeError) as exc:
                errors.append(f"runtime_limits: {exc}")

    # budget_control sub-validation: delegate to BudgetControlConfig
    # .from_dict, which raises for unknown dimensions / bad thresholds /
    # bad overlay tier names — kept in one place rather than duplicated.
    budget_data = data.get("budget_control")
    if budget_data is not None:
        if not isinstance(budget_data, dict):
            errors.append("'budget_control' must be an object or null")
        else:
            try:
                BudgetControlConfig.from_dict(budget_data)
            except (ValueError, TypeError) as exc:
                errors.append(f"budget_control: {exc}")

    return len(errors) == 0, errors, warnings


@dataclass
class SubagentConfig:
    """Top-level configuration for the subagent plugin.

    Attributes:
        project: GCP project ID for Vertex AI.
        location: Vertex AI region (e.g., 'us-central1').
        default_model: Default model for subagents. None = inherit from parent.
        default_provider: Default provider for subagents. None = inherit from parent.
                         If set, MUST match default_model's provider.
        profiles: Dict of named subagent profiles.
        allow_inline: Whether to allow inline subagent creation.
        inline_allowed_plugins: Plugins allowed for inline subagent creation.
        auto_discover_profiles: Whether to auto-discover profiles from profiles_dir.
        profiles_dir: Directory to scan for profile files (default: .jaato/profiles).
    """
    project: str
    location: str
    default_model: Optional[str] = None  # None = inherit from parent
    default_provider: Optional[str] = None  # None = inherit from parent
    profiles: Dict[str, SubagentProfile] = field(default_factory=dict)
    allow_inline: bool = True
    inline_allowed_plugins: List[str] = field(default_factory=list)
    auto_discover_profiles: bool = True
    profiles_dir: str = ".jaato/profiles"

    def add_profile(self, profile: SubagentProfile) -> None:
        """Add a subagent profile."""
        self.profiles[profile.name] = profile

    def get_profile(self, name: str) -> Optional[SubagentProfile]:
        """Get a subagent profile by name."""
        return self.profiles.get(name)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubagentConfig':
        """Create SubagentConfig from a dictionary.

        Args:
            data: Configuration dictionary with structure:
                {
                    "project": "...",
                    "location": "...",
                    "default_model": "...",
                    "profiles": {
                        "profile_name": {
                            "description": "...",
                            "plugins": [...],
                            ...
                        }
                    },
                    "allow_inline": true,
                    "inline_allowed_plugins": [...],
                    "auto_discover_profiles": true,
                    "profiles_dir": ".jaato/profiles"
                }

        Returns:
            SubagentConfig instance.
        """
        profiles = {}
        for name, profile_data in data.get('profiles', {}).items():
            # Parse GC configuration if present
            cache_config = parse_cache_block(profile_data)
            gc_config = parse_gc_block(profile_data)
            trace_config = parse_trace_block(profile_data)

            # Parse runtime_limits (cgroup-enforced + app-enforced caps).
            # Validation runs in __post_init__ — bad values raise here so
            # the inline config rejects them at load time, same as gc.
            runtime_limits = None
            if 'runtime_limits' in profile_data and profile_data['runtime_limits']:
                runtime_limits = RuntimeLimits.from_dict(profile_data['runtime_limits'])

            # Same fail-at-load-time contract as runtime_limits above.
            budget_control = None
            if 'budget_control' in profile_data and profile_data['budget_control']:
                budget_control = BudgetControlConfig.from_dict(
                    profile_data['budget_control'])

            # Parse plugin entries, separating (preload) annotations
            # and per-plugin tool allow-lists.
            raw_plugins = profile_data.get('plugins', [])
            clean_plugins, preloaded, tool_scopes = parse_plugin_list(raw_plugins)

            raw_env = profile_data.get('env', {})
            env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}

            raw_model_tiers = profile_data.get('model_tiers') or {}
            model_tiers = (
                {str(k): v for k, v in raw_model_tiers.items()}
                if isinstance(raw_model_tiers, dict) else {}
            )
            if model_tiers and profile_data.get('model'):
                logger.warning(
                    "Inline profile '%s' declares both 'model' and "
                    "'model_tiers'; 'model' will be ignored.", name,
                )

            profiles[name] = SubagentProfile(
                name=name,
                description=profile_data.get('description', ''),
                plugins=clean_plugins,
                preloaded_plugins=preloaded,
                tool_scopes=tool_scopes,
                plugin_configs=profile_data.get('plugin_configs', {}),
                system_instructions=profile_data.get('system_instructions'),
                suppress_base_instructions=profile_data.get('suppress_base_instructions', False),
                model=profile_data.get('model'),
                provider=profile_data.get('provider'),
                max_turns=profile_data.get('max_turns', 10),
                gc=gc_config,
            cache=cache_config,
            trace=trace_config,
                env=env,
                inherits=_normalize_inherits(profile_data.get('inherits')),
                completion_payload_schema=profile_data.get('completion_payload_schema'),
                completion_processors=_parse_completion_processors(profile_data.get('completion_processors')),
                suppress_inherited_processors=_normalize_suppress_inherited_processors(
                    profile_data.get('suppress_inherited_processors')),
                runtime_limits=runtime_limits,
                budget_control=budget_control,
                model_tiers=model_tiers,
                apparmor=bool(profile_data.get('apparmor', False)),
                apparmor_fragments=_normalize_apparmor_fragments(profile_data.get('apparmor_fragments')),
            )

        return cls(
            project=data.get('project', ''),
            location=data.get('location', ''),
            default_model=data.get('default_model'),  # None = inherit from parent
            default_provider=data.get('default_provider'),  # None = inherit from parent
            profiles=profiles,
            allow_inline=data.get('allow_inline', True),
            inline_allowed_plugins=data.get('inline_allowed_plugins', []),
            auto_discover_profiles=data.get('auto_discover_profiles', True),
            profiles_dir=data.get('profiles_dir', '.jaato/profiles'),
        )


def gc_profile_to_plugin_config(
    gc_profile: GCProfileConfig,
    agent_name: Optional[str] = None
) -> tuple:
    """Convert a GCProfileConfig to a (GCPlugin, GCConfig) tuple.

    This helper function takes a profile's GC configuration and creates
    the actual GC plugin and config objects needed by JaatoSession.

    Args:
        gc_profile: GCProfileConfig from a subagent profile.
        agent_name: Optional agent name for trace logging identification.

    Returns:
        Tuple of (GCPlugin, GCConfig) ready to pass to session.set_gc_plugin().

    Raises:
        ValueError: If the GC plugin type is not found.

    Example:
        if profile.gc:
            gc_plugin, gc_config = gc_profile_to_plugin_config(profile.gc, agent_id)
            session.set_gc_plugin(gc_plugin, gc_config)
    """
    from ..gc import load_gc_plugin, GCConfig

    gc_type = gc_profile.type
    gc_plugin_name = gc_type if gc_type.startswith('gc_') else f'gc_{gc_type}'

    # Build plugin init config
    gc_init_config = {
        'preserve_recent_turns': gc_profile.preserve_recent_turns,
        'notify_on_gc': gc_profile.notify_on_gc,
    }
    if agent_name:
        gc_init_config['agent_name'] = agent_name
    if gc_profile.summarize_middle_turns is not None:
        gc_init_config['summarize_middle_turns'] = gc_profile.summarize_middle_turns
    # Merge plugin-specific config
    gc_init_config.update(gc_profile.plugin_config)

    gc_plugin = load_gc_plugin(gc_plugin_name, gc_init_config)

    # Create GCConfig for the session
    gc_config = GCConfig(
        threshold_percent=gc_profile.threshold_percent,
        target_percent=gc_profile.target_percent,
        pressure_percent=gc_profile.pressure_percent,
        max_turns=gc_profile.max_turns,
        preserve_recent_turns=gc_profile.preserve_recent_turns,
        plugin_config=gc_profile.plugin_config,
    )

    return gc_plugin, gc_config


@dataclass
class SubagentResult:
    """Result from a subagent execution.

    Attributes:
        success: Whether the subagent completed successfully.
        response: The subagent's final response text.
        turns_used: Number of conversation turns used.
        error: Error message if success is False.
        token_usage: Token usage statistics if available.
        agent_id: ID of the subagent session (for multi-turn conversations).
        output_streamed: Whether output was streamed via UI hooks (prevents double-display).
    """
    success: bool
    response: str
    turns_used: int = 0
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    agent_id: Optional[str] = None
    output_streamed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool response.

        When output_streamed is True, the response text is omitted from the
        result since it was already displayed to the user via UI hooks.
        This prevents the model from echoing the response in its output.
        """
        result: Dict[str, Any] = {
            'success': self.success,
            'turns_used': self.turns_used,
        }
        # Only include response text if it wasn't already streamed to UI
        if self.output_streamed:
            result['response_note'] = 'Response was streamed to the user interface. Do not repeat it.'
        else:
            result['response'] = self.response
        if self.error:
            result['error'] = self.error
        if self.token_usage:
            result['token_usage'] = self.token_usage
        if self.agent_id:
            result['agent_id'] = self.agent_id
        return result
