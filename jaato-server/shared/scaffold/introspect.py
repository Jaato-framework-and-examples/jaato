"""The introspection core — the single source of framework metadata.

This module reads, **offline** (no daemon, no network), what a profile author
needs to know to write a correct profile: which plugins exist (and their
tools / core-vs-discoverable / config knobs), which providers exist (and their
capabilities / config knobs / quirks / modalities), and which GC strategies
exist.  Both verbs of the scaffold tool consume it:

- ``explain`` renders this data by scope.
- ``validate`` checks a hand-authored asset against it.

Sharing this one reader is the point — there is no second code path that
re-derives "what is a valid plugin / knob / quirk".

**Provider metadata is read by AST-exec, not import.**  A provider's
``__init__.py`` does ``from .provider import …`` which would pull in the vendor
SDK; an author scaffolding for provider X should still be able to *explain*
provider Y whose SDK isn't installed.  So we parse each ``__init__.py``, pull
out only the ``PROVIDER_CAPABILITIES`` / ``PROVIDER_KNOBS`` / ``PROVIDER_QUIRKS``
assignments, and ``eval`` them in a controlled namespace holding just the
metadata classes (which live in ``base.py`` — pure, no vendor SDK).  Same
declared source of truth as the CI guard, no import side-effects.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import io
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from shared.plugins.model_provider import base as _pbase
from jaato_sdk.plugins.model_provider.types import DISCOVERABILITY_DEFERRED

# Same exclusions as the contract guards — test stubs / non-providers.
_EXCLUDE = {"tests", "__pycache__", "bundle_common", "echo"}

_PROVIDER_DIR = Path(_pbase.__file__).resolve().parent

# Controlled namespace for eval-ing the declared provider constants.  Only the
# metadata classes + frozenset — no __builtins__, so a malformed declaration
# can't do anything but build these objects.
_PROVIDER_NS = {
    "ProviderCapabilities": _pbase.ProviderCapabilities,
    "ProviderKnobs": _pbase.ProviderKnobs,
    "KnobLayer": _pbase.KnobLayer,
    "KnobSpec": _pbase.KnobSpec,
    "AuthSource": _pbase.AuthSource,
    "frozenset": frozenset,
    "__builtins__": {},
}

_WANTED_CONSTS = ("PROVIDER_CAPABILITIES", "PROVIDER_KNOBS", "PROVIDER_QUIRKS",
                  "PROVIDER_AUTH_RESOLUTION")


# -------------------------------------------------------------------- models

@dataclass
class ProviderInfo:
    """Introspected metadata for one model provider."""

    dir_name: str                       # filesystem dir (PROVIDER_KNOBS lives here)
    capabilities: Optional[_pbase.ProviderCapabilities] = None
    knobs: Optional[_pbase.ProviderKnobs] = None
    quirks: frozenset = field(default_factory=frozenset)
    auth: tuple = ()                    # ordered AuthSource credential chain

    def normalized_names(self) -> set:
        """Names a profile's ``provider:`` field might use for this provider.

        Registry names sometimes hyphenate where the dir underscores
        (``zhipuai-openai`` vs ``zhipuai_openai``); accept both spellings.
        """
        return {self.dir_name, self.dir_name.replace("_", "-")}


@dataclass
class ToolInfo:
    name: str
    discoverability: str = DISCOVERABILITY_DEFERRED
    description: str = ""
    #: The tool's JSON-Schema ``parameters`` block, verbatim.
    #:
    #: Carried so a consumer can machine-check a tool SIGNATURE without a
    #: live session.  Its absence was a real gap: the cascade-coordination
    #: example wanted to validate its published spec against the framework
    #: and found ``explain plugin --json`` surfaced name, discoverability and
    #: description only -- so the signature its document specified could not
    #: be compared to the one that shipped, and four drifts (a parameter that
    #: was never implemented, a renamed one, and two stale return shapes)
    #: went unnoticed in a public repo.
    #:
    #: ``None`` when the plugin's schema omitted it, which is distinct from
    #: ``{}`` -- a tool that genuinely takes no arguments.
    parameters: "Optional[Dict[str, Any]]" = None


@dataclass
class CommandInfo:
    """One user-facing command a plugin exposes (from ``get_user_commands``).

    User commands are the operator's runtime control surface — invoked directly
    in the TUI (``permissions allow *``, ``memory …``, auth switches), NOT via
    the model's function calling.  ``subcommands`` are the first-argument
    completions (``CommandCompletion`` entries from ``get_command_completions``)
    — where the real surface lives: ``permissions`` alone tells an operator
    nothing actionable.
    """
    name: str
    description: str = ""
    share_with_model: bool = False
    subcommands: List[str] = field(default_factory=list)


@dataclass
class ConfigSetting:
    """One configurable plugin setting (from ``get_config_schema``)."""
    name: str
    type: str = ""
    default: Any = None
    description: str = ""


@dataclass
class PluginInfo:
    """Introspected metadata for one plugin (best-effort, offline)."""

    name: str
    kind: str = "tool"
    tier: Optional[str] = None
    description: str = ""               # plugin class docstring, first line
    tools: List[ToolInfo] = field(default_factory=list)
    commands: List[CommandInfo] = field(default_factory=list)
    config_keys: List[str] = field(default_factory=list)
    config_settings: List["ConfigSetting"] = field(default_factory=list)
    dynamic: bool = False               # tool list needs a live session (mcp, …)
    # Provenance (issue #684): who supplied this plugin.  ``source`` is
    # the rendered one-liner (``built-in (shared.plugins.cli)``), and
    # ``builtin`` says whether it came from the framework's own package —
    # so an out-of-tree distribution that claimed a built-in name is
    # visible in ``jaato-scaffold plugins`` without reading daemon logs.
    source: str = ""
    builtin: bool = True


@dataclass
class ProfileField:
    """One ``SubagentProfile`` schema field (name/type/default/description)."""
    name: str
    type: str = ""
    default: Any = None
    description: str = ""
    allowed: str = ""               # resolved value-constraint (actual values)


@dataclass
class EventField:
    """One field on an event class (name + rendered type annotation)."""
    name: str
    type: str = ""


@dataclass
class EventInfo:
    """One event in the client/server protocol (from the SDK's ``EventType``).

    Sourced by AST-scanning ``jaato_sdk/events.py`` — the ``EventType`` enum is
    the authoritative catalog, and its section headers + trailing comments carry
    the DIRECTION (``Server -> Client`` etc.) that runtime reflection can't see.
    ``event_class``/``doc``/``fields`` are filled from the matching
    ``class …Event(Event)`` whose ``type:`` field defaults to this member.
    """

    name: str                           # enum member (e.g. AGENT_OUTPUT)
    wire: str = ""                       # wire value (e.g. "agent.output")
    direction: str = ""                 # "Server → Client" | "Client → Server" | "Server ↔ Client"
    domain: str = ""                    # section label (e.g. "Agent lifecycle")
    note: str = ""                      # trailing-comment remainder after the direction
    event_class: Optional[str] = None   # matching Event subclass name, if any
    doc: str = ""                       # event class docstring, first line
    fields: List["EventField"] = field(default_factory=list)


@dataclass
class EnvVar:
    """A process-level env var the daemon/plugins actually read (from code).

    Distinct from a provider's ``plugin_configs`` knob: these are read off
    ``os.environ`` at process scope, not from a profile.  Discovered by
    scanning ``os.environ.get`` / ``os.getenv`` / ``os.environ[...]`` sites,
    so the list reflects the INSTALLED code, never prose.

    ``scope`` / ``typed_key`` / ``scope_note`` are the exception: they are
    DECLARED in ``shared/env_scope.py`` rather than derived, because "is a
    per-session value meaningful here?" is not a property the source can be
    asked (issue #775).  They are stamped on by :func:`_apply_env_scope`
    after the scan, and default to ``"unclassified"`` when the catalog does
    not know the var.
    """

    name: str
    default: Optional[str] = None       # literal default at the read site, if any
    category: str = "framework"         # provider:<x> | plugin:<x> | daemon | framework | rate_limit | telemetry | proxy
    tier: str = "daemon"                # daemon | runner | daemon_callable | unknown (PLUGIN_TIER of the reader)
    sources: List[str] = field(default_factory=list)  # relative file paths
    description: Optional[str] = None   # one-line goal, from an `# env: ...` comment on the read line
    # WHAT the var is, from the declared catalog in ``shared/env_scope.py``
    # (issue #775).  The scan can only report that a var is READ; whether a
    # per-session value is meaningful, and whether a typed profile key
    # already covers it, are declarations -- and unclassified is a real
    # answer, meaning the guard has not run since the var appeared.
    scope: str = "unclassified"         # session | host | ambient | internal | unclassified
    typed_key: Optional[str] = None     # dotted path of the typed equivalent, if any
    scope_note: Optional[str] = None    # one line on why that scope


# ---------------------------------------------------------------- providers

def _provider_dirs() -> List[str]:
    out = []
    for entry in sorted(p.name for p in _PROVIDER_DIR.iterdir()):
        d = _PROVIDER_DIR / entry
        if not d.is_dir() or entry in _EXCLUDE or entry.startswith("_"):
            continue
        if (d / "__init__.py").exists():
            out.append(entry)
    return out


def _read_provider_consts(dir_name: str) -> Dict[str, Any]:
    """AST-exec the declared metadata constants from a provider __init__.py."""
    src = (_PROVIDER_DIR / dir_name / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    out: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        for const in _WANTED_CONSTS:
            if const in targets:
                seg = ast.get_source_segment(src, node.value)
                out[const] = eval(seg, dict(_PROVIDER_NS))  # noqa: S307 controlled ns
    return out


def providers() -> Dict[str, ProviderInfo]:
    """All model providers, keyed by directory name (offline, SDK-independent)."""
    out: Dict[str, ProviderInfo] = {}
    for name in _provider_dirs():
        consts = _read_provider_consts(name)
        out[name] = ProviderInfo(
            dir_name=name,
            capabilities=consts.get("PROVIDER_CAPABILITIES"),
            knobs=consts.get("PROVIDER_KNOBS"),
            quirks=consts.get("PROVIDER_QUIRKS") or frozenset(),
            auth=consts.get("PROVIDER_AUTH_RESOLUTION") or (),
        )
    return out


def resolve_provider(name: str) -> Optional[ProviderInfo]:
    """Find a provider by any accepted spelling of ``name`` (hyphen/underscore)."""
    allp = providers()
    if name in allp:
        return allp[name]
    norm = name.replace("-", "_")
    if norm in allp:
        return allp[norm]
    for info in allp.values():
        if name in info.normalized_names():
            return info
    return None


# ---------------------------------------------------------------------- gc

def gc_strategies() -> Dict[str, List[str]]:
    """GC strategy names → the GCConfig field names that tune them."""
    from shared.plugins.gc import discover_gc_plugins
    from shared.plugins.gc.base import GCConfig

    fields = [f.name for f in dataclasses.fields(GCConfig)]
    return {name: fields for name in sorted(discover_gc_plugins().keys())}


# ------------------------------------------------------------------ profile

def _type_name(t) -> str:
    """Readable display name for a dataclass field annotation (type or string).

    Plain types render as their name (``str``, not ``<class 'str'>``); generics
    keep their shape but drop ``typing.`` and module qualifiers
    (``shared...GCProfileConfig`` -> ``GCProfileConfig``); ``NoneType`` -> ``None``.
    """
    import re
    if isinstance(t, type):
        return t.__name__
    s = (t if isinstance(t, str) else str(t)).replace("typing.", "").replace(
        "NoneType", "None")
    return re.sub(r"\b\w+(?:\.\w+)+\.(\w+)", r"\1", s)  # a.b.C -> C


def _profile_field_constraints() -> Dict[str, str]:
    """Resolve value-constraints bounded by a framework constant — so an author
    sees the ACTUAL allowed values, not a source symbol to chase.

    e.g. ``model_tiers`` keys are constrained by
    ``shared.model_tiers.VALID_TIER_NAMES``; surfaced here as the real tier
    names (the same "introspect the installed code" principle as the rest of
    ``explain``).  Soft — a field with no resolvable constraint is simply absent.
    """
    out: Dict[str, str] = {}
    try:
        from shared import model_tiers as mt
        tiers = ", ".join(sorted(mt.VALID_TIER_NAMES))
        reserved = ", ".join(sorted(mt.RESERVED_KEYS))
        out["model_tiers"] = f"tier keys: {tiers}  |  reserved control keys: {reserved}"
    except Exception:
        pass
    try:
        from shared import budget_control as bc
        dims = ", ".join(sorted(bc.VALID_DIMENSIONS))
        actions = ", ".join(sorted(bc.VALID_ACTIONS))
        out["budget_control"] = (
            f"limits dimensions: {dims}  |  degrade[].at: percentage in (0, 100] "
            f"(70 or '70%'; strictly increasing across rungs)  |  "
            f"degrade[].model_tiers: overlay keyed by tier name  |  "
            f"degrade[].action: {actions}")
    except Exception:
        pass
    return out


def profile_schema() -> List[ProfileField]:
    """The ``SubagentProfile`` schema — the knobs a profile author can set.

    Names / types / defaults come from ``dataclasses.fields``; descriptions come
    from each field's ``metadata['description']`` (the structured source on the
    dataclass); and value-constraints that reference a framework constant (e.g.
    ``model_tiers``' valid tier keys) are RESOLVED to their actual values via
    :func:`_profile_field_constraints` — so an author discovers every knob,
    incl. the AppArmor knobs and the real tier names, without reading any
    source.  Fields with no metadata show name / type / default only.
    """
    from shared.plugins.subagent.config import SubagentProfile

    constraints = _profile_field_constraints()
    out: List[ProfileField] = []
    for f in dataclasses.fields(SubagentProfile):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except Exception:
                default = None
        else:
            default = "<required>"
        out.append(ProfileField(
            name=f.name,
            type=_type_name(f.type),
            default=default,
            description=f.metadata.get("description", ""),
            allowed=constraints.get(f.name, ""),
        ))
    return out


# ----------------------------------------------------------------- plugins

def _stamp_origin(info: "PluginInfo", origin: Any) -> None:
    """Copy a registry ``PluginOrigin`` onto *info*, if one was recorded.

    Provenance is optional: a registry that predates #684 tracking, or a
    plugin registered by some path that never stamped one, yields
    ``None`` and leaves the defaults ("built-in", unnamed source) in
    place rather than misreporting an unknown origin as foreign.
    """
    if origin is None:
        return
    info.source = origin.describe()
    info.builtin = origin.builtin


def _collect_user_commands(plugin: Any) -> List["CommandInfo"]:
    """The plugin's user-facing commands, or ``[]`` — never raising.

    Split out of :func:`plugins` rather than inlined: that function is over
    the complexity ceiling and frozen in the audit baseline, so new logic
    belongs in a helper (see ``test_cyclomatic_complexity_audit``).

    Best-effort on BOTH levels, and deliberately so.  ``get_user_commands``
    is optional protocol — a plugin that lacks it or raises must not break
    the whole registry walk — and the per-command completion lookup is
    independently guarded, so one plugin with a broken completer costs its
    own subcommands rather than every command after it.

    Subcommands come from the first-argument completions, which is where the
    actionable surface lives: ``permissions`` alone tells an operator
    nothing, ``permissions allow|deny|suspend|…`` does.
    """
    out: List[CommandInfo] = []
    try:
        commands = plugin.get_user_commands() or []
    except Exception:
        return out
    for cmd in commands:
        out.append(CommandInfo(
            name=getattr(cmd, "name", "?"),
            description=getattr(cmd, "description", "") or "",
            share_with_model=bool(getattr(cmd, "share_with_model", False)),
            subcommands=_command_subcommands(plugin, getattr(cmd, "name", "")),
        ))
    return out


def _command_subcommands(plugin: Any, command: str) -> List[str]:
    """First-argument completions for *command*, de-duplicated, or ``[]``."""
    subs: List[str] = []
    try:
        completions = plugin.get_command_completions(command, []) or []
    except Exception:
        return subs
    for comp in completions:
        value = getattr(comp, "value", None)
        if value and value not in subs:
            subs.append(value)
    return subs


def plugins() -> Dict[str, PluginInfo]:
    """All tool/enrichment plugins, best-effort offline.

    Tool-schema extraction is wrapped per-plugin: a plugin whose schema needs
    a live session (mcp, cross-tier forwarders) is reported with
    ``dynamic=True`` rather than crashing the walk.
    """
    from shared.plugins.registry import PluginRegistry

    reg = PluginRegistry()
    try:
        reg.discover()  # tool (+ auto enrichment)
    except Exception:
        return {}

    out: Dict[str, PluginInfo] = {}
    for name in sorted(reg.list_available()):
        info = PluginInfo(name=name)
        plugin = reg.get_plugin(name)
        _stamp_origin(info, reg.get_plugin_source(name))
        # kind / tier from the plugin's module (module-level constants)
        mod = type(plugin).__module__
        try:
            import importlib
            pkg = importlib.import_module(mod.rsplit(".", 1)[0])
            info.kind = getattr(pkg, "PLUGIN_KIND", "tool")
            info.tier = getattr(pkg, "PLUGIN_TIER", None)
        except Exception:
            pass
        # tools (best-effort)
        try:
            for schema in plugin.get_tool_schemas() or []:
                _params = getattr(schema, "parameters", None)
                info.tools.append(ToolInfo(
                    name=getattr(schema, "name", "?"),
                    discoverability=getattr(schema, "discoverability", DISCOVERABILITY_DEFERRED),
                    description=(getattr(schema, "description", "") or "").split("\n")[0],
                    # Copied, not referenced: a consumer mutating what it was
                    # handed must not reshape the live plugin's schema.
                    parameters=(dict(_params) if isinstance(_params, dict)
                                else None),
                ))
        except Exception:
            info.dynamic = True
        info.commands.extend(_collect_user_commands(plugin))
        # plugin-level description (class docstring, first line)
        doc = (type(plugin).__doc__ or "").strip()
        info.description = doc.split("\n", 1)[0].strip() if doc else ""
        # config schema (best-effort) — names + descriptions / types / defaults.
        # Two shapes exist in the wild; normalize BOTH into ConfigSetting so
        # explain/validate see the knobs either way:
        #   - a list of ``PluginSetting`` objects (``.name`` / ``.type`` / …),
        #   - a raw JSON-schema dict ``{"type":"object","properties":{k:{…}}}``
        #     (permission, cli, interactive_shell, notebook).  Previously the
        #     dict form yielded NO knobs (iterating a dict gives key strings,
        #     which have no ``.name``), so those plugins' knobs were invisible.
        try:
            schema = reg.get_plugin_config_schema(name) or []
            settings: List[ConfigSetting] = []
            if isinstance(schema, dict):
                props = schema.get("properties")
                if isinstance(props, dict):
                    for knob, spec in props.items():
                        spec = spec if isinstance(spec, dict) else {}
                        settings.append(ConfigSetting(
                            name=str(knob),
                            type=str(spec.get("type", "") or ""),
                            default=spec.get("default", None),
                            description=str(spec.get("description", "") or ""),
                        ))
            else:
                for s in schema:
                    if hasattr(s, "name"):
                        settings.append(ConfigSetting(
                            name=s.name,
                            type=str(getattr(s, "type", "") or ""),
                            default=getattr(s, "default", None),
                            description=getattr(s, "description", "") or "",
                        ))
            info.config_keys = [s.name for s in settings]
            info.config_settings = settings
        except Exception:
            pass
        out[name] = info
    return out


# ------------------------------------------------------------------- env vars

# Scan roots: the daemon (server/) and the shared core + plugins.
_SERVER_ROOT = Path(_pbase.__file__).resolve().parents[3]  # …/jaato-server
_SCAN_DIRS = ("server", "shared")


def _literal(node: ast.expr) -> Optional[str]:
    """Render a constant AST node as a string, else None (dynamic default)."""
    if isinstance(node, ast.Constant):
        return None if node.value is None else str(node.value)
    return None


def _const_str_map(tree: ast.AST) -> Dict[str, str]:
    """Map module/function string-constant names → value (for key indirection).

    Providers commonly do ``ENV_X = "JAATO_X"`` then ``os.getenv(ENV_X)`` — so
    resolving same-file string constants is what surfaces the provider API-key
    vars a literal-only scan would miss.
    """
    m: Dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) \
                and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    m[t.id] = n.value.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Constant) \
                and isinstance(n.value.value, str) and isinstance(n.target, ast.Name):
            m[n.target.id] = n.value.value
    return m


def _key_of(node: ast.expr, const_map: Dict[str, str]) -> Optional[str]:
    """Resolve an env-read key node to a string (literal or same-file const)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in const_map:
        return const_map[node.id]
    return None


# An ``# env: <one-line goal>`` comment on (or just above) an env-read line
# documents that var.  Deliberately code-co-located so the description can't
# drift from the reader; undocumented vars simply have ``description=None``.
_ENV_DOC_RE = re.compile(r"#\s*env:\s*(.+?)\s*$")


def _env_doc_comments(source: str) -> Dict[int, str]:
    """Map line-number -> description for every ``# env: ...`` comment."""
    out: Dict[int, str] = {}
    try:
        toks = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                m = _ENV_DOC_RE.match(tok.string.strip())
                if m:
                    out[tok.start[0]] = m.group(1).strip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def _env_reads(node: ast.AST, const_map: Dict[str, str]):
    """Yield (name, default_node_or_None, lineno) for each os.environ read.

    Matches ``os.getenv``/``os.environ.get``/``environ.get`` (call) and
    ``os.environ[...]``/``environ[...]`` (subscript).  Keys may be string
    literals OR same-file string constants (resolved via ``const_map``).
    ``lineno`` is the read site's line, for `# env:` doc-comment lookup.
    """
    for n in ast.walk(node):
        # Call forms: os.getenv / os.environ.get / environ.get
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            attr = n.func.attr
            v = n.func.value
            is_getenv = attr == "getenv" and isinstance(v, ast.Name) and v.id == "os"
            is_environ_get = (
                attr == "get" and (
                    (isinstance(v, ast.Attribute) and v.attr == "environ")
                    or (isinstance(v, ast.Name) and v.id == "environ")
                )
            )
            if (is_getenv or is_environ_get) and n.args:
                key = _key_of(n.args[0], const_map)
                if key is not None:
                    default = n.args[1] if len(n.args) >= 2 else None
                    yield key, default, getattr(n, "lineno", 0)
        # Subscript form: os.environ["X"] / environ["X"]
        elif isinstance(n, ast.Subscript):
            v = n.value
            if (isinstance(v, ast.Attribute) and v.attr == "environ") or \
               (isinstance(v, ast.Name) and v.id == "environ"):
                key = _key_of(n.slice, const_map)
                if key is not None:
                    yield key, None, getattr(n, "lineno", 0)


def _categorize(name: str, rel_path: str) -> str:
    """Classify an env var by cross-cutting prefix, else by reading path."""
    if name.startswith(("AI_REQUEST", "AI_RETRY")) or name.startswith("AI_"):
        return "rate_limit"
    if name.startswith("OTEL_") or "TELEMETRY" in name:
        return "telemetry"
    if "PROXY" in name or name in ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"):
        return "proxy"
    parts = rel_path.replace("\\", "/").split("/")
    if "model_provider" in parts:
        i = parts.index("model_provider")
        if i + 1 < len(parts):
            return f"provider:{parts[i + 1]}"
    if "plugins" in parts:
        i = parts.index("plugins")
        if i + 1 < len(parts) and parts[i + 1] != "model_provider":
            return f"plugin:{parts[i + 1]}"
    if parts and parts[0] == "server":
        return "daemon"
    return "framework"


_PLUGIN_DIR = _SERVER_ROOT / "shared" / "plugins"
_TIER_CACHE: Dict[str, str] = {}


def _plugin_tier(plugin: str) -> str:
    """AST-read a plugin's ``PLUGIN_TIER`` from its ``__init__.py`` (cached)."""
    if plugin in _TIER_CACHE:
        return _TIER_CACHE[plugin]
    tier = "unknown"
    try:
        tree = ast.parse((_PLUGIN_DIR / plugin / "__init__.py")
                         .read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if any(isinstance(t, ast.Name) and t.id == "PLUGIN_TIER"
                       for t in node.targets):
                    tier = str(node.value.value)
    except (OSError, SyntaxError, UnicodeDecodeError):
        pass
    _TIER_CACHE[plugin] = tier
    return tier


def _tier_for(category: str) -> str:
    """The PLUGIN_TIER the reader of a category's env var runs in.

    Plugin vars take the plugin's declared ``PLUGIN_TIER``; model providers are
    daemon-tier; server / framework-core / cross-cutting all run daemon-side.
    """
    if category.startswith("plugin:"):
        return _plugin_tier(category.split(":", 1)[1])
    return "daemon"


# Memo for :func:`_scan_env_vars`.  ``None`` until the first scan; afterwards
# the authoritative result, handed out only as deep copies by ``env_vars()``.
_ENV_VARS_CACHE: Optional[Dict[str, EnvVar]] = None


def _scan_env_vars() -> Dict[str, EnvVar]:
    """AST-scan the installed tree for env reads.  The expensive half of
    :func:`env_vars` — parses every non-test ``.py`` under ``_SCAN_DIRS``
    (500+ files, ~3s).  Call ``env_vars()`` instead; it memoizes this.
    """
    out: Dict[str, EnvVar] = {}
    for d in _SCAN_DIRS:
        root = _SERVER_ROOT / d
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            if "__pycache__" in py.parts or "/tests/" in str(py):
                continue
            try:
                source = py.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            rel = str(py.relative_to(_SERVER_ROOT))
            const_map = _const_str_map(tree)
            doc_comments = _env_doc_comments(source)
            for name, default_node, lineno in _env_reads(tree, const_map):
                ev = out.get(name)
                if ev is None:
                    cat = _categorize(name, rel)
                    ev = EnvVar(name=name, category=cat, tier=_tier_for(cat))
                    out[name] = ev
                if rel not in ev.sources:
                    ev.sources.append(rel)
                if ev.default is None and default_node is not None:
                    ev.default = _literal(default_node)
                # First `# env:` comment found for this var wins (like default).
                if ev.description is None:
                    desc = doc_comments.get(lineno)
                    if desc:
                        ev.description = desc
    _apply_env_scope(out)
    return out


def _apply_env_scope(found: Dict[str, EnvVar]) -> None:
    """Stamp each scanned var with its declared scope + typed equivalent.

    The scan and the catalog answer different questions -- "is this read?"
    versus "what is it?" -- and only the first can be derived from source.
    A var the catalog does not know keeps ``scope="unclassified"`` rather
    than being given a plausible default, so ``explain env`` shows the gap
    and the guard fails on it.
    """
    from shared.env_scope import CATALOG

    for name, ev in found.items():
        entry = CATALOG.get(name)
        if entry is None:
            continue
        ev.scope = entry.scope
        ev.typed_key = entry.typed_key
        ev.scope_note = entry.note or None


#: Memo for :func:`plugin_config_keys` — plugin name → its config-key set.
_PLUGIN_CONFIG_KEYS_CACHE: Dict[str, FrozenSet[str]] = {}


def _config_base(node: ast.AST) -> str:
    """The receiver name of a ``<x>.get(...)`` / ``<x>[...]`` expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def plugin_config_keys(plugin: str) -> FrozenSet[str]:
    """Every ``plugin_configs.<plugin>`` key the plugin actually consumes.

    AST-scans the plugin package for the two ways a config value is read --
    ``<something>config.get("key")`` and ``<something>config["key"]`` -- and
    unions them with the ``properties`` of any ``get_config_schema`` the
    plugin declares.  No imports: same offline discipline as
    :func:`env_vars`, which matters because importing every plugin makes an
    unrelated dependency skew look like a catalog error.

    WHY THE READ SITES AND NOT THE SCHEMA.  ``get_config_schema`` is the
    obvious source and is not sufficient on its own: four of the plugins
    this is used for declare none at all, and ``todo`` declares one that
    omits ``reporter_config`` -- a key it genuinely reads.  Verifying
    against the schema alone would fail correct entries, which is worse
    than not checking.  The union is what the plugin will actually honour.

    Returns an empty set when the plugin reads no config at all; callers
    treat that as "cannot verify" rather than as a pass.
    """
    if plugin in _PLUGIN_CONFIG_KEYS_CACHE:
        return _PLUGIN_CONFIG_KEYS_CACHE[plugin]

    keys: set = set()
    root = _PLUGIN_DIR / plugin
    if root.is_dir():
        for py in root.rglob("*.py"):
            if "__pycache__" in py.parts or "/tests/" in str(py):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            keys |= _config_keys_in(tree)
    result = frozenset(keys)
    _PLUGIN_CONFIG_KEYS_CACHE[plugin] = result
    return result


def _key_from_config_get(node: ast.AST) -> set:
    """``<something>config.get("key")`` → ``{"key"}``, else empty."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args):
        return set()
    base = _config_base(node.func.value).lower()
    if not (base.endswith("config") or base in ("cfg", "opts")):
        return set()
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return {arg.value}
    return set()


def _key_from_config_subscript(node: ast.AST) -> set:
    """``<something>config["key"]`` → ``{"key"}``, else empty."""
    if not isinstance(node, ast.Subscript):
        return set()
    if not _config_base(node.value).lower().endswith("config"):
        return set()
    sl = node.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return {sl.value}
    return set()


def _keys_from_schema_properties(node: ast.AST) -> set:
    """A JSON-schema ``"properties": {...}`` block → its declared keys."""
    if not isinstance(node, ast.Dict):
        return set()
    out: set = set()
    for key, val in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == "properties" \
                and isinstance(val, ast.Dict):
            out |= {k.value for k in val.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return out


def _config_keys_in(tree: ast.AST) -> set:
    """Config keys read (or schema-declared) in one parsed module.

    Three independent node shapes, one helper each — the shapes share
    nothing but the walk, and inlining them put this function over the
    complexity ceiling.
    """
    out: set = set()
    for n in ast.walk(tree):
        out |= _key_from_config_get(n)
        out |= _key_from_config_subscript(n)
        out |= _keys_from_schema_properties(n)
    return out


def env_vars() -> Dict[str, EnvVar]:
    """All env vars the installed daemon + plugins read (offline source scan).

    Keyed by var name; merges read sites (union of sources; first literal
    default wins).  Each var carries the ``tier`` (daemon/runner/...) of the
    code that reads it.  Reflects the INSTALLED code — no prose.

    The one declared part is ``scope`` / ``typed_key`` / ``scope_note``,
    stamped on from ``shared/env_scope.py`` (see :class:`EnvVar`): the scan
    can say a var is read, not what it is for.

    **Memoized for the life of the process.**  The underlying scan
    (:func:`_scan_env_vars`) AST-parses the whole installed tree and costs
    ~3s; the tree cannot change under a running process, so repeat calls —
    ``build``/``explain``/``validate`` each make one, and the scaffold test
    suite made ~37 — are served from the memo.

    Callers get a **deep copy**, so the mutable ``EnvVar`` fields (``sources``,
    ``description``, ``default``) stay caller-owned and no caller can poison
    the shared memo.  The copy is ~1000x cheaper than the scan.
    """
    global _ENV_VARS_CACHE
    if _ENV_VARS_CACHE is None:
        _ENV_VARS_CACHE = _scan_env_vars()
    return copy.deepcopy(_ENV_VARS_CACHE)


# ----------------------------------------------------------------- events

import importlib.util as _ilu  # noqa: E402  (co-located with its sole use)

# Direction arrow in a section header ``(Server -> Client)`` or a trailing
# member comment ``# Client -> Server``.  ``<->`` (or ``<=>``) is bidirectional.
_EVENT_DIR_RE = re.compile(
    r"(Server|Client)\s*(<->|<=>|->|→|↔)\s*(Client|Server)")


def _events_file() -> Optional[Path]:
    """Locate the SDK's ``events.py`` WITHOUT importing/executing it.

    ``find_spec`` resolves the module's origin (importing only the parent
    ``jaato_sdk`` package, never ``events`` itself), so the catalog stays a
    pure source-of-truth read — mirrors the module docstring's no-import rule.
    """
    try:
        spec = _ilu.find_spec("jaato_sdk.events")
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None or not spec.origin:
        return None
    p = Path(spec.origin)
    return p if p.is_file() else None


def _parse_direction(text: str) -> str:
    """Normalize a ``Server -> Client`` style token to a display string.

    Returns ``""`` when no direction token is present.
    """
    m = _EVENT_DIR_RE.search(text)
    if not m:
        return ""
    left, arrow, right = m.group(1), m.group(2), m.group(3)
    if arrow in ("<->", "<=>", "↔"):
        # Order-independent for bidirectional; present Server-first for stability.
        return "Server ↔ Client"
    return f"{left} → {right}"


def _event_class_map(tree: ast.AST) -> Dict[str, tuple]:
    """Map ``EventType`` member name → (class_name, doc, [EventField]).

    Reads every ``class …(Event)`` whose ``type: EventType`` field defaults to
    ``Field(default=EventType.<MEMBER>)``; the field's member is the join key.
    """
    out: Dict[str, tuple] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(isinstance(b, ast.Name) and b.id == "Event" for b in node.bases):
            continue
        member: Optional[str] = None
        fields: List[EventField] = []
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            fname = stmt.target.id
            if fname == "type":
                # type: EventType = Field(default=EventType.XXX)
                member = _member_of_type_default(stmt.value)
                continue
            try:
                ftype = ast.unparse(stmt.annotation)
            except Exception:  # noqa: BLE001
                ftype = ""
            fields.append(EventField(name=fname, type=ftype))
        if member is not None:
            doc = (ast.get_docstring(node) or "").strip().splitlines()
            out[member] = (node.name, doc[0] if doc else "", fields)
    return out


def _member_of_type_default(value: Optional[ast.expr]) -> Optional[str]:
    """Extract ``XXX`` from ``Field(default=EventType.XXX)`` (or a bare
    ``EventType.XXX`` default)."""
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
            and value.value.id == "EventType":
        return value.attr
    if isinstance(value, ast.Call):
        for kw in value.keywords:
            if kw.arg == "default":
                return _member_of_type_default(kw.value)
    return None


def events() -> Dict[str, EventInfo]:
    """The client/server event protocol, keyed by ``EventType`` member name.

    AST-scans ``jaato_sdk/events.py``: enumerates the ``EventType`` members
    (wire value from the assignment, DOMAIN from the nearest preceding section
    comment, DIRECTION from that section header or a trailing per-member comment
    which wins), then joins each to its ``…Event`` class (docstring + fields).
    Reflects the INSTALLED SDK source — no import side-effects, no prose.
    """
    out: Dict[str, EventInfo] = {}
    path = _events_file()
    if path is None:
        return out
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return out

    # Trailing comment per line + standalone (section-header) comment lines.
    trailing: Dict[int, str] = {}       # lineno -> comment text (comment shares a code line)
    standalone: Dict[int, str] = {}     # lineno -> comment text (comment is alone on its line)
    try:
        prev_code_row = -1
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
        code_rows = {t.start[0] for t in toks
                     if t.type not in (tokenize.COMMENT, tokenize.NL,
                                       tokenize.NEWLINE, tokenize.INDENT,
                                       tokenize.DEDENT, tokenize.ENCODING)}
        for tok in toks:
            if tok.type != tokenize.COMMENT:
                continue
            row = tok.start[0]
            text = tok.string.lstrip("#").strip()
            if row in code_rows:
                trailing[row] = text
            else:
                standalone[row] = text
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    # Locate the EventType enum class and walk its members in source order.
    enum_node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef) and n.name == "EventType"), None)
    if enum_node is None:
        return out

    # A SECTION HEADER is a standalone comment whose preceding physical line is
    # BLANK (the file's convention: blank line → one-line topical header → the
    # members).  A comment block that directly follows a MEMBER (no blank line)
    # is that member's own doc-comment, NOT a section header — so it must not
    # reset the domain (e.g. AGENT_ERROR's multi-line doc keeps it under "Agent
    # lifecycle").  The header is the FIRST line of each blank-delimited block;
    # continuation lines (prev line is itself a comment) are not headers.
    comment_rows = set(standalone) | set(trailing)
    section: Dict[int, str] = {}
    for ln in sorted(standalone):
        prev = ln - 1
        if prev in code_rows or prev in comment_rows:
            continue                    # mid-block, or a member's own doc-comment
        section[ln] = standalone[ln]    # blank (or class top) above → section header

    class_map = _event_class_map(tree)
    section_lines = sorted(section)

    for stmt in enum_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not (isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            continue
        targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        if not targets:
            continue
        member = targets[0]
        lineno = stmt.lineno

        # DOMAIN + section direction: nearest standalone comment above.
        sec_line = None
        for sl in section_lines:
            if sl < lineno:
                sec_line = sl
            else:
                break
        domain, sec_dir = "", ""
        if sec_line is not None:
            raw = section[sec_line]
            sec_dir = _parse_direction(raw)
            # Drop any ``(...)`` group carrying the direction, wherever it sits
            # (``Agent lifecycle (Server -> Client)``, ``... (Client <-> Server,
            # WS only)``) — a non-direction parenthetical
            # (``(server-to-server gossip)``) is part of the label and kept.
            cleaned = re.sub(
                r"\s*\(([^()]*)\)",
                lambda m: "" if _EVENT_DIR_RE.search(m.group(1)) else m.group(0),
                raw)
            # Headers authored as ``Short label: long explanation`` or ``Short
            # label — explanation`` collapse to the short label (also groups
            # sibling headers, e.g. both wake blocks → "Wake primitive").
            domain = re.split(r"\s*[:—]\s*|\s+-\s+", cleaned, maxsplit=1)[0].strip(" .")

        # DIRECTION: a trailing per-member comment overrides the section.
        note = ""
        direction = sec_dir
        tc = trailing.get(lineno)
        if tc:
            td = _parse_direction(tc)
            if td:
                direction = td
            note = _EVENT_DIR_RE.sub("", tc).strip(" ()-")

        cls = class_map.get(member)
        out[member] = EventInfo(
            name=member,
            wire=stmt.value.value,
            direction=direction,
            domain=domain,
            note=note,
            event_class=cls[0] if cls else None,
            doc=cls[1] if cls else "",
            fields=list(cls[2]) if cls else [],
        )
    return out
