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
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.plugins.model_provider import base as _pbase

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
    discoverability: str = "discoverable"
    description: str = ""


@dataclass
class PluginInfo:
    """Introspected metadata for one plugin (best-effort, offline)."""

    name: str
    kind: str = "tool"
    tier: Optional[str] = None
    tools: List[ToolInfo] = field(default_factory=list)
    config_keys: List[str] = field(default_factory=list)
    dynamic: bool = False               # tool list needs a live session (mcp, …)


@dataclass
class EnvVar:
    """A process-level env var the daemon/plugins actually read (from code).

    Distinct from a provider's ``plugin_configs`` knob: these are read off
    ``os.environ`` at process scope, not from a profile.  Discovered by
    scanning ``os.environ.get`` / ``os.getenv`` / ``os.environ[...]`` sites,
    so the list reflects the INSTALLED code, never prose.
    """

    name: str
    default: Optional[str] = None       # literal default at the read site, if any
    category: str = "framework"         # provider:<x> | plugin:<x> | daemon | framework | rate_limit | telemetry | proxy
    tier: str = "daemon"                # daemon | runner | daemon_callable | unknown (PLUGIN_TIER of the reader)
    sources: List[str] = field(default_factory=list)  # relative file paths


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


# ----------------------------------------------------------------- plugins

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
                info.tools.append(ToolInfo(
                    name=getattr(schema, "name", "?"),
                    discoverability=getattr(schema, "discoverability", "discoverable"),
                    description=(getattr(schema, "description", "") or "").split("\n")[0],
                ))
        except Exception:
            info.dynamic = True
        # config schema keys (best-effort)
        try:
            schema = reg.get_plugin_config_schema(name) or []
            info.config_keys = [getattr(s, "name", str(s)) for s in schema]
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


def _env_reads(node: ast.AST, const_map: Dict[str, str]):
    """Yield (name, default_node_or_None) for each os.environ read in ``node``.

    Matches ``os.getenv``/``os.environ.get``/``environ.get`` (call) and
    ``os.environ[...]``/``environ[...]`` (subscript).  Keys may be string
    literals OR same-file string constants (resolved via ``const_map``).
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
                    yield key, default
        # Subscript form: os.environ["X"] / environ["X"]
        elif isinstance(n, ast.Subscript):
            v = n.value
            if (isinstance(v, ast.Attribute) and v.attr == "environ") or \
               (isinstance(v, ast.Name) and v.id == "environ"):
                key = _key_of(n.slice, const_map)
                if key is not None:
                    yield key, None


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


def env_vars() -> Dict[str, EnvVar]:
    """All env vars the installed daemon + plugins read (offline source scan).

    Keyed by var name; merges read sites (union of sources; first literal
    default wins).  Each var carries the ``tier`` (daemon/runner/...) of the
    code that reads it.  Reflects the INSTALLED code — no prose, no declaration.
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
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, OSError, UnicodeDecodeError):
                continue
            rel = str(py.relative_to(_SERVER_ROOT))
            const_map = _const_str_map(tree)
            for name, default_node in _env_reads(tree, const_map):
                ev = out.get(name)
                if ev is None:
                    cat = _categorize(name, rel)
                    ev = EnvVar(name=name, category=cat, tier=_tier_for(cat))
                    out[name] = ev
                if rel not in ev.sources:
                    ev.sources.append(rel)
                if ev.default is None and default_node is not None:
                    ev.default = _literal(default_node)
    return out

