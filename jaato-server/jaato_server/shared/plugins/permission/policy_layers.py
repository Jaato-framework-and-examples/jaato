"""Where a session's permission policy comes from, layer by layer (#1474).

Before #1474 a ``permissions.json`` was read and then thrown away on every
daemon session: :meth:`PermissionPlugin.initialize` let an inline ``policy``
key replace the file, and both enforcer builders (the runner's
``build_session_permission_plugin`` and the daemon-local
``JaatoServer.initialize``) always passed one -- a hard-coded
``defaultPolicy: ask`` dict, written out twice.  The file never decided
anything, although ``docs/jaato_permission_system.md`` called it the STATIC
layer.

This module is the one place the effective policy is assembled.  Four
layers, lowest to highest (:data:`POLICY_LAYERS`):

1. the framework default -- ``defaultPolicy: ask``, empty lists
   (:data:`FRAMEWORK_DEFAULT_POLICY`, the only copy of that dict);
2. ``~/.jaato/permissions.json`` -- the daemon user's (on a runner, the
   snapshot the daemon shipped on the envelope, #1465);
3. ``<config_root>/permissions.json``, else
   ``<workspace>/.jaato/permissions.json`` -- the project file.  An
   explicit ``config_path`` (or ``PERMISSION_CONFIG_PATH``) names this
   file instead of discovering it;
4. the profile's ``plugin_configs.permission.policy``.  For a #957
   subagent with a block of its own, that block is layer 4 for the
   subagent, still on top of the files.

How they combine (:data:`MERGE_RULES`):

- ``defaultPolicy`` is a scalar: the highest layer that SETS it wins.  A
  file that omits the key does not set it (before #1474 such a file meant
  ``deny`` on the paths that read it at all; that reading is gone with the
  path that applied it).
- ``whitelist`` and ``blacklist`` (``tools``, ``patterns``, ``arguments``)
  are a UNION across every layer.  A profile adds to a file; it never
  replaces it.
- the blacklist beats the whitelist, as :class:`PermissionPolicy` already
  decides -- so a deny written in ANY layer survives a higher layer's allow
  of the same tool.
- any other policy key (``sanitization``, ``cwd`` ...) is replaced WHOLE by
  the highest layer that sets it; nested keys are not merged.

The consequence worth knowing on upgrade: a host whose ``permissions.json``
says ``defaultPolicy: allow`` starts auto-approving the tools no list names.
:func:`announce_effective_policy` logs that at WARNING, naming the file, and
``jaato-scaffold validate`` reports it as ``permission_file_allow`` before
any session runs.

Stdlib only, beyond the permission package's own loader: the scaffold reads
:data:`POLICY_LAYERS` / :data:`MERGE_RULES` to render ``explain plugin
permission``, so the page and the code cannot describe two precedences.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from jaato_server.shared.user_tier import path as _user_tier_path

from .config_loader import ConfigValidationError, validate_config

logger = logging.getLogger(__name__)

#: The file name every file layer reads.
PERMISSION_FILE_NAME = "permissions.json"

#: Env var naming the project-layer file explicitly (pre-#1474 spelling,
#: kept so a deployment that used it still points at the same file).
PERMISSION_CONFIG_PATH_ENV = "PERMISSION_CONFIG_PATH"

#: Layer 1.  The ONLY copy of the framework's default policy: both enforcer
#: builders used to carry their own, and a third one in ``initialize`` would
#: have been free to disagree with them.
FRAMEWORK_DEFAULT_POLICY: Dict[str, Any] = {
    "defaultPolicy": "ask",
    "whitelist": {"tools": [], "patterns": []},
    "blacklist": {"tools": [], "patterns": []},
}

#: Layer ids, lowest precedence first, with what each one is.  Rendered by
#: ``jaato-scaffold explain plugin permission``.
POLICY_LAYERS: Tuple[Tuple[str, str], ...] = (
    ("framework", "the framework default: defaultPolicy ask, empty lists"),
    ("user", "~/.jaato/permissions.json (the daemon user's; on a confined "
             "runner the snapshot the daemon ships, #1465)"),
    ("workspace", "<config_root>/permissions.json, else "
                  "<workspace>/.jaato/permissions.json (or the file "
                  "config_path / PERMISSION_CONFIG_PATH names)"),
    ("profile", "the profile's plugin_configs.permission.policy (a #957 "
                "subagent's own block, for that subagent)"),
)

#: The keys whose rule lists are UNIONED across layers.
LIST_KEYS: Tuple[str, ...] = ("whitelist", "blacklist")

#: The sub-keys of a list key: two lists and one nested map.
_LIST_SUBKEYS: Tuple[str, ...] = ("tools", "patterns")
_ARGUMENTS_SUBKEY = "arguments"

#: File-only keys that are not policy (the file format carries them beside
#: the policy; :func:`resolve_effective_policy` never hands them on).
FILE_META_KEYS: Tuple[str, ...] = ("version", "channel")

#: How the layers combine, one sentence each.  Rendered beside
#: :data:`POLICY_LAYERS`.
MERGE_RULES: Tuple[str, ...] = (
    "defaultPolicy: the highest layer that sets it wins; a file that omits "
    "the key does not set it",
    "whitelist / blacklist (tools, patterns, arguments): UNION across all "
    "layers -- a profile adds to the files, never replaces them",
    "blacklist beats whitelist, so a deny written in ANY layer survives a "
    "higher layer's allow of the same tool",
    "any other key (sanitization, cwd, ...): the highest layer that sets it "
    "replaces it whole; nested keys are not merged",
    "a file layer whose defaultPolicy is allow is announced at WARNING per "
    "session, and `validate` reports it as permission_file_allow",
)


@dataclass(frozen=True)
class PolicySource:
    """One layer that contributed to an effective policy.

    Attributes:
        layer: One of the ids in :data:`POLICY_LAYERS`.
        path: The file read, for a file layer; ``None`` otherwise.
    """

    layer: str
    path: Optional[str] = None

    @property
    def is_file(self) -> bool:
        return self.layer in ("user", "workspace")

    @property
    def label(self) -> str:
        """How a log line or a page names this layer."""
        if self.layer == "framework":
            return "the framework default"
        if self.layer == "profile":
            return "the profile's plugin_configs.permission.policy"
        return f"{self.layer} file {self.path}"


@dataclass
class EffectivePolicy:
    """The merged policy and where each part of it came from.

    Attributes:
        policy: The dict :meth:`PermissionPolicy.from_config` reads.
        default_policy_source: The layer whose ``defaultPolicy`` won.
        list_sources: Every layer that contributed at least one whitelist
            or blacklist entry, lowest first.
        files: Every ``permissions.json`` that was read, lowest first.
    """

    policy: Dict[str, Any]
    default_policy_source: PolicySource
    list_sources: List[PolicySource] = field(default_factory=list)
    files: List[str] = field(default_factory=list)

    @property
    def default_policy(self) -> str:
        return str(self.policy.get("defaultPolicy"))

    @property
    def file_allow(self) -> Optional[str]:
        """The file that made ``defaultPolicy`` ``allow``, or ``None``."""
        src = self.default_policy_source
        if self.default_policy == "allow" and src.is_file:
            return src.path
        return None

    def describe(self) -> str:
        """One line: the effective default and every contributing layer."""
        lists = ", ".join(s.label for s in self.list_sources) or "none"
        return (f"defaultPolicy={self.default_policy} "
                f"(decided by {self.default_policy_source.label}); "
                f"list entries from: {lists}")


# ---------------------------------------------------------------- the files

def user_permission_file() -> Path:
    """Layer 2's file: the daemon user's, or the runner's shipped copy."""
    return _user_tier_path(PERMISSION_FILE_NAME)


def project_permission_file(
    workspace_path: Optional[str],
    config_root: Optional[str],
    config_path: Optional[str] = None,
) -> Optional[Path]:
    """Layer 3's file, or ``None`` when no project file exists.

    An explicit ``config_path`` (then ``PERMISSION_CONFIG_PATH``) names it
    outright and is returned whether or not it exists, so the caller can say
    a named file is missing rather than silently falling back.  Otherwise the
    first existing of ``<config_root>/permissions.json`` and
    ``<workspace>/.jaato/permissions.json``.
    """
    explicit = config_path or os.environ.get(PERMISSION_CONFIG_PATH_ENV)
    if explicit:
        return Path(explicit)
    candidates: List[Path] = []
    if config_root:
        candidates.append(Path(config_root) / PERMISSION_FILE_NAME)
    if workspace_path:
        candidates.append(Path(workspace_path) / ".jaato" / PERMISSION_FILE_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_policy_file(path: Path) -> Dict[str, Any]:
    """The policy half of one ``permissions.json``.

    Raises:
        ConfigValidationError: the file is not a valid permissions file.
            Deliberately not swallowed: a policy file that cannot be read
            must not quietly leave a session on the framework default.
        json.JSONDecodeError / OSError: as :func:`json.load` / ``open``.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ConfigValidationError([f"{path}: top level must be an object"])
    _, errors = validate_config(raw)
    actual = [e for e in errors if not e.startswith("Warning:")]
    if actual:
        raise ConfigValidationError([f"{path}: {e}" for e in actual])
    return {k: v for k, v in raw.items() if k not in FILE_META_KEYS}


# ---------------------------------------------------------------- merging

def _union(into: List[Any], values: Iterable[Any]) -> bool:
    """Append each value not already present; True if any was added."""
    added = False
    for value in values:
        if value not in into:
            into.append(value)
            added = True
    return added


def _merge_arguments(into: Dict[str, Dict[str, List[str]]],
                     incoming: Any) -> bool:
    """Union a ``{tool: {arg: [values]}}`` map into ``into``."""
    if not isinstance(incoming, dict):
        return False
    added = False
    for tool, arg_rules in incoming.items():
        if not isinstance(arg_rules, dict):
            continue
        slot = into.setdefault(tool, {})
        for arg, values in arg_rules.items():
            if isinstance(values, list):
                added |= _union(slot.setdefault(arg, []), values)
    return added


def _merge_list_key(into: Dict[str, Any], incoming: Any) -> bool:
    """Union one ``whitelist`` / ``blacklist`` block into ``into``."""
    if not isinstance(incoming, dict):
        return False
    added = False
    for sub in _LIST_SUBKEYS:
        values = incoming.get(sub)
        if isinstance(values, list):
            added |= _union(into.setdefault(sub, []), values)
    if _ARGUMENTS_SUBKEY in incoming:
        added |= _merge_arguments(into.setdefault(_ARGUMENTS_SUBKEY, {}),
                                  incoming[_ARGUMENTS_SUBKEY])
    return added


def merge_policy_layers(
    layers: Sequence[Tuple[PolicySource, Optional[Dict[str, Any]]]],
) -> EffectivePolicy:
    """Combine policy dicts, lowest precedence first, by :data:`MERGE_RULES`.

    A ``None`` layer contributes nothing.  The first layer should be the
    framework default; :func:`resolve_effective_policy` always supplies it.
    """
    merged: Dict[str, Any] = {key: {} for key in LIST_KEYS}
    default_source = PolicySource("framework")
    list_sources: List[PolicySource] = []
    for source, layer in layers:
        if not isinstance(layer, dict):
            continue
        contributed = False
        for key, value in layer.items():
            if key in LIST_KEYS:
                contributed |= _merge_list_key(merged[key], value)
            elif key == "defaultPolicy":
                merged[key] = value
                default_source = source
            else:
                merged[key] = copy.deepcopy(value)
        if contributed:
            list_sources.append(source)
    return EffectivePolicy(policy=merged,
                           default_policy_source=default_source,
                           list_sources=list_sources)


def resolve_effective_policy(
    profile_policy: Optional[Dict[str, Any]] = None,
    *,
    workspace_path: Optional[str] = None,
    config_root: Optional[str] = None,
    config_path: Optional[str] = None,
) -> EffectivePolicy:
    """The policy a session is judged by: all four layers, merged.

    Called by :meth:`PermissionPlugin.initialize` (the runtime policy, so by
    both enforcer builders) and :meth:`PermissionPlugin.set_scoped_policy`
    (a #957 subagent's own block).  Nothing else assembles a policy.

    Args:
        profile_policy: Layer 4 -- ``plugin_configs.permission.policy``, or
            ``None`` when the profile declares none.
        workspace_path: The session's workspace, for the project file.
        config_root: The session's config root; its ``permissions.json``
            outranks ``<workspace>/.jaato/permissions.json``.
        config_path: An explicit project file (the block's ``config_path``).

    Raises:
        ConfigValidationError, json.JSONDecodeError, OSError: a file layer
            exists and cannot be read as a permissions file.
    """
    layers: List[Tuple[PolicySource, Optional[Dict[str, Any]]]] = [
        (PolicySource("framework"), FRAMEWORK_DEFAULT_POLICY),
    ]
    files: List[str] = []
    user_file = user_permission_file()
    project_file = project_permission_file(workspace_path, config_root,
                                           config_path)
    for layer, path in (("user", user_file), ("workspace", project_file)):
        if path is None:
            continue
        if not path.is_file():
            if layer == "workspace" and (config_path or os.environ.get(
                    PERMISSION_CONFIG_PATH_ENV)):
                logger.warning("permission: the policy file %s named by "
                               "config_path / %s does not exist; ignored",
                               path, PERMISSION_CONFIG_PATH_ENV)
            continue
        if layer == "workspace" and _same_file(path, user_file):
            continue
        layers.append((PolicySource(layer, str(path)), read_policy_file(path)))
        files.append(str(path))
    if isinstance(profile_policy, dict):
        layers.append((PolicySource("profile"), profile_policy))
    effective = merge_policy_layers(layers)
    effective.files = files
    return effective


def _same_file(a: Path, b: Path) -> bool:
    """Whether two paths name one file (a workspace whose root is ``~``)."""
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


# ---------------------------------------------------------------- announcing

#: ``(session, scope, line)`` already logged.  A session's policy is built
#: by more than one plugin instance (the registry's copy and the enforcer),
#: so "once per session" needs a memory.  Bounded: cleared when it grows.
_ANNOUNCED: Set[Tuple[str, str, str]] = set()
_ANNOUNCED_MAX = 4096


def _first_time(session_id: Optional[str], scope: str, line: str) -> bool:
    if not session_id:
        return True
    key = (session_id, scope, line)
    if key in _ANNOUNCED:
        return False
    if len(_ANNOUNCED) >= _ANNOUNCED_MAX:
        _ANNOUNCED.clear()
    _ANNOUNCED.add(key)
    return True


def announce_effective_policy(
    effective: EffectivePolicy,
    *,
    session_id: Optional[str] = None,
    scope: str = "runtime",
) -> None:
    """Log where a session's policy came from -- once per session and scope.

    INFO always: the effective ``defaultPolicy`` and the layer that decided
    it.  WARNING when a FILE is what made it ``allow``: on upgrade past #1474
    such a host starts auto-approving, and that must be visible.
    """
    line = effective.describe()
    if _first_time(session_id, scope, line):
        logger.info("permission[%s] session=%s: %s", scope,
                    session_id or "-", line)
    allow_file = effective.file_allow
    if allow_file and _first_time(session_id, scope, "allow:" + allow_file):
        logger.warning(
            "permission[%s] session=%s: defaultPolicy=allow comes from %s -- "
            "every tool no list names is auto-approved without a prompt.  "
            "Since #1474 permissions.json is applied; remove the key or set "
            "it to ask/deny to restore prompting.", scope,
            session_id or "-", allow_file)


def file_layers_setting_allow(
    workspace_path: Optional[str],
    config_root: Optional[str] = None,
) -> List[str]:
    """The ``permissions.json`` files a session would read that say
    ``defaultPolicy: allow`` -- for ``validate``'s ``permission_file_allow``.

    Unreadable or invalid files are skipped: ``validate`` reports the
    finding it is asked about and must not raise on the way.
    """
    out: List[str] = []
    paths = [user_permission_file(),
             project_permission_file(workspace_path, config_root)]
    for path in paths:
        if path is None or not path.is_file():
            continue
        try:
            data = read_policy_file(path)
        except (ConfigValidationError, ValueError, OSError):
            continue
        if data.get("defaultPolicy") == "allow" and str(path) not in out:
            out.append(str(path))
    return out


# ---------------------------------------------------------------- builders

def enforcer_init_config(
    profile_block: Optional[Dict[str, Any]],
    *,
    workspace_path: Optional[str],
    config_root: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``initialize`` config of a daemon session's enforcer.

    Called by both enforcer builders -- the runner's
    ``build_session_permission_plugin`` and the daemon-local
    ``JaatoServer.initialize`` -- so they cannot disagree.  It carries NO
    default policy: before #1474 each builder wrote one out, and its mere
    presence made ``initialize`` discard ``permissions.json``.  The default
    is :data:`FRAMEWORK_DEFAULT_POLICY`, layer 1 of
    :func:`resolve_effective_policy`, and the profile's ``policy`` (when it
    declares one) is layer 4.

    The profile's ``plugin_configs.permission`` block replaces top-level
    keys (Phase 4 §C), as before.
    """
    config: Dict[str, Any] = {
        "channel_type": "queue",
        "channel_config": {"use_colors": False},
        "workspace_path": workspace_path,
    }
    if config_root:
        config["config_root"] = config_root
    if session_id:
        config["session_id"] = session_id
    if profile_block:
        config.update(profile_block)
    return config
