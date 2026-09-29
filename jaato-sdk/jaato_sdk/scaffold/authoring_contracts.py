"""The facts ``jaato-scaffold new`` needs, live or from a snapshot.

``new`` asks two questions about the framework:

- **which providers exist**, and for one of them, which env var holds its key
  (the ``AuthSource`` chain) and which config keys it accepts (the
  ``PROVIDER_KNOBS`` layers).  This validates ``--provider``, names the key
  var in a generated profile and ``.env``, and decides which worked examples
  a profile carries.
- **which env vars the daemon reads**, with their categories and defaults.
  This fills the commented knob catalogue in every generated ``.env``.
- **what a profile file may say, and where it is found** (#1267, tier 2):
  the keys the loader reads, the keys it derives or withdrew, the field
  types, the closed vocabularies, and the discovery layout (tiers, set
  directories, extensions).  ``new profile-set`` checks what it emitted
  against these when the framework validator is not installed.  Live, they
  come from ``jaato_server.shared.plugins.subagent.config`` (and the three
  modules owning a vocabulary); the snapshot carries the same projection.

jaato-server's ``introspect`` answers both by reading the INSTALLED
jaato-server source with ``ast``: the 27 provider ``__init__.py`` files and
~630 modules under ``jaato_server/{server,shared}``.  Nothing is imported, so
the dependency does not show up in ``sys.modules``, but it is real.  This
module ships in jaato-sdk (#1267), where those files are not on disk unless
jaato-server is installed beside it.

This module is the one door ``build`` goes through:

- **live** when jaato-server's ``introspect`` imports and the provider tree it
  reads is present.  The objects returned are introspect's own, so output is
  unchanged.
- **snapshot** otherwise: :data:`SNAPSHOT_FILE`, a projection of the live
  answers to exactly the fields ``build`` reads.  The snapshot objects
  (:class:`ProviderView`, :class:`EnvView`) are duck-compatible with
  ``introspect.ProviderInfo`` / ``introspect.EnvVar`` for those fields and no
  others.

**Provenance.**  The snapshot records the jaato-server version it was
projected from (``jaato_server_version``).  In this repository a test diffs it
against the live tree, but an SDK wheel carries it alone: nothing ties it to
whichever jaato-server a user has.  So when the snapshot answers and a
jaato-server IS installed at another version (its tree unusable to
``introspect``, or forced off), authoring warns once, naming both versions.
The version is part of the projection, so bumping jaato-server's version makes
the checked-in snapshot stale and the guard below fails until it is
regenerated: the release bump regenerates it, or does not pass CI.

The snapshot is checked in and regenerated with::

    python -m jaato_server.shared.scaffold.authoring_contracts --write

(the same entry point as ``python -m jaato_sdk.scaffold.authoring_contracts``;
either needs jaato-server importable, because a snapshot is only ever
projected from the live tree).  The repo's ``.githooks/pre-commit`` runs it
automatically, and stages the result, whenever a commit touches non-test
jaato-server source or ``jaato-server/pyproject.toml`` (activate the hooks once
per clone with ``git config core.hooksPath .githooks``).
``test_authoring_does_not_load_introspection_1267.py`` fails when it differs
from the live projection, naming that command.  A snapshot nobody checks is a
second source of truth that drifts silently; the guard is what keeps this one
from being that.

Imports only the stdlib at module level.  jaato-server is imported inside
:func:`_live`, :func:`_live_profile_config` and :func:`project_profiles`, and
only there.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

#: The checked-in projection read when the live tree is unavailable.
SNAPSHOT_FILE = Path(__file__).with_name("authoring_snapshot.json")

#: Bumped when the snapshot's SHAPE changes, so a reader never half-parses a
#: file written for another shape.  2: the ``profiles`` section (#1267, tier 2).
SNAPSHOT_VERSION = 2

#: Tests set this to exercise the snapshot branch on a tree where the live
#: one is available.  Deliberately not an env var: an env read would be one
#: more knob for the env-scope catalog, and nothing but a test needs it.
_FORCE_SNAPSHOT = False

_SNAPSHOT_CACHE: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------ snapshot views


@dataclass(frozen=True)
class AuthStep:
    """One credential source, as ``introspect`` reports it (``AuthSource``)."""

    kind: str
    name: str = ""
    note: str = ""


@dataclass(frozen=True)
class KnobsView:
    """The layer/key membership of a provider's ``PROVIDER_KNOBS``.

    Carries only what :meth:`accepts` needs.  Its answer must match
    ``ProviderKnobs.accepts``: an undeclared layer is rejected, an opaque
    (pass-through) layer accepts any key, a closed layer only its own keys.
    """

    layers: Tuple[Tuple[str, bool, FrozenSet[str]], ...] = ()

    def accepts(self, layer: str, key: str) -> bool:
        """Whether ``key`` is valid in ``layer``; see ``ProviderKnobs.accepts``."""
        for name, opaque, keys in self.layers:
            if name == layer:
                return True if opaque else key in keys
        return False


@dataclass(frozen=True)
class ProviderView:
    """A provider as the snapshot records it: the fields ``build`` reads."""

    dir_name: str
    auth: Tuple[AuthStep, ...] = ()
    knobs: Optional[KnobsView] = None

    def normalized_names(self) -> set:
        """Same spellings as ``ProviderInfo.normalized_names``."""
        return {self.dir_name, self.dir_name.replace("_", "-")}


@dataclass(frozen=True)
class EnvView:
    """An env var as the snapshot records it: the fields ``build`` reads."""

    name: str
    category: str = "framework"
    default: Optional[str] = None


# -------------------------------------------------------------- projections


def project_providers(infos: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON form of the provider facts ``build`` reads.

    Accepts ``introspect.ProviderInfo`` objects or :class:`ProviderView` ones,
    so the guard can compare the live tree and the snapshot through the same
    function.
    """
    out: Dict[str, Any] = {}
    for name in sorted(infos):
        info = infos[name]
        knobs = getattr(info, "knobs", None)
        out[name] = {
            "auth": [[s.kind, s.name, s.note] for s in (info.auth or ())],
            "knobs": None if knobs is None else _project_knobs(knobs),
        }
    return out


def _project_knobs(knobs: Any) -> Dict[str, Any]:
    """``{layer: {opaque, keys}}`` from a ``ProviderKnobs`` or a :class:`KnobsView`."""
    if isinstance(knobs, KnobsView):
        return {name: {"opaque": opaque, "keys": sorted(keys)}
                for name, opaque, keys in knobs.layers}
    return {lyr.layer: {"opaque": lyr.opaque, "keys": sorted(lyr.keys)}
            for lyr in knobs.layers}


def _env_var_is_read_by_build(category: str, default: Optional[str]) -> bool:
    """Whether ``build._compose_env`` can render this var.

    It renders every ``provider:<x>`` var for the bound provider, and every
    other var that has a non-empty default.  The snapshot keeps exactly that
    set.  A var outside it changes no generated file, so recording it would
    only make the snapshot churn on edits that cannot matter to ``new``.
    """
    return category.startswith("provider:") or default not in (None, "")


def project_env_vars(evs: Dict[str, Any]) -> Dict[str, Any]:
    """The JSON form of the env-var facts ``build`` reads."""
    return {
        name: {"category": ev.category, "default": ev.default}
        for name, ev in sorted(evs.items())
        if _env_var_is_read_by_build(ev.category, ev.default)
    }


@dataclass(frozen=True)
class ProfileFacts:
    """What a profile FILE may say, and where discovery looks for one.

    Built from :func:`project_profiles`'s JSON form on both paths, so the
    live answer and the snapshot answer are the same object for the same
    tree.  Facts only: whether a given profile is VALID is the framework
    validator's question (``inherits`` merging, block construction, plugin
    schemas), and that stays with jaato-server.

    Attributes:
        file_keys: Every top-level key the loader reads
            (``PROFILE_FILE_KEYS``).
        derived_keys: Dataclass fields a file may not set, with what supplies
            them (``PROFILE_DERIVED_FIELDS``).
        removed_keys: Keys the loader withdrew, with the successor
            (``PROFILE_REMOVED_FIELDS``).
        reserved_names: Profile names a file may not claim.
        field_types: The declared type of each file key that is a
            ``SubagentProfile`` field, as ``explain profile`` renders it.
        enums: Closed VALUE vocabularies by dotted path
            (``completion_processors[].phase``).
        key_vocabularies: Closed KEY sets of nested blocks by path
            (``regulatory``, ``budget_control.limits``).
        layout: The discovery layout: ``extensions``, ``yaml_extensions``,
            ``profiles_subdir``, ``profile_set_env_var`` and ``tiers``
            (``[{tier, location, rule}]``, highest precedence first).
    """

    file_keys: FrozenSet[str]
    derived_keys: Dict[str, str]
    removed_keys: Dict[str, str]
    reserved_names: FrozenSet[str]
    field_types: Dict[str, str]
    enums: Dict[str, Tuple[str, ...]]
    key_vocabularies: Dict[str, FrozenSet[str]]
    layout: Dict[str, Any]

    @classmethod
    def from_json(cls, rec: Dict[str, Any]) -> "ProfileFacts":
        """The view of one ``profiles`` section (live projection or snapshot)."""
        return cls(
            file_keys=frozenset(rec["file_keys"]),
            derived_keys=dict(rec["derived_keys"]),
            removed_keys=dict(rec["removed_keys"]),
            reserved_names=frozenset(rec["reserved_names"]),
            field_types=dict(rec["field_types"]),
            enums={k: tuple(v) for k, v in rec["enums"].items()},
            key_vocabularies={k: frozenset(v)
                              for k, v in rec["key_vocabularies"].items()},
            layout=dict(rec["layout"]),
        )


def _type_name(t: Any) -> str:
    """A field annotation as ``explain profile`` shows it (``Optional[GCProfileConfig]``).

    The rendering of ``introspect._type_name`` (plain types by name,
    ``typing.`` and module qualifiers dropped, ``NoneType`` as ``None``),
    plus ``ForwardRef('X')`` shown as ``X``.
    """
    import re

    if isinstance(t, type):
        return t.__name__
    text = (t if isinstance(t, str) else str(t)).replace(
        "typing.", "").replace("NoneType", "None")
    # A quoted annotation resolves to ``ForwardRef('X')``; show the ``X``.
    text = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", text)
    return re.sub(r"\b\w+(?:\.\w+)+\.(\w+)", r"\1", text)


def project_profiles(cfg: Any) -> Dict[str, Any]:
    """The JSON form of the profile facts, read from the live modules.

    *cfg* is ``jaato_server.shared.plugins.subagent.config``.  The
    vocabularies owned elsewhere (``budget_control``,
    ``instruction_suppression``) are imported beside it; all three are
    stdlib + jaato_sdk at module level.
    Every list is sorted, so a regeneration that changes nothing writes the
    same bytes.
    """
    import dataclasses

    from jaato_server.shared import budget_control as bc
    from jaato_server.shared import instruction_suppression as sup

    fields = {f.name: f for f in dataclasses.fields(cfg.SubagentProfile)}
    enums = {
        "budget_control.degrade[].action": bc.VALID_ACTIONS,
        "budget_control.on_unmetered": bc.UNMETERED_POLICIES,
        "cache.ttl": cfg.VALID_CACHE_TTLS,
        "completion_processors[].on_error": cfg.PROCESSOR_ON_ERROR,
        "completion_processors[].on_exhausted": cfg.PROCESSOR_ON_EXHAUSTED,
        "completion_processors[].phase": cfg.PROCESSOR_PHASES,
        "record_keeping.integrity": cfg.INTEGRITY_MODES,
        "regulatory.risk_class": cfg.RISK_CLASSES,
    }
    vocabularies = {
        "budget_control.limits": bc.VALID_DIMENSIONS,
        "record_keeping": cfg.RECORD_KEEPING_KEYS,
        "regulatory": cfg.REGULATORY_KEYS,
        "suppress_base_instructions": sup.SUPPRESSION_PIECES,
    }
    return {
        "file_keys": sorted(cfg.PROFILE_FILE_KEYS),
        "derived_keys": dict(sorted(cfg.PROFILE_DERIVED_FIELDS.items())),
        "removed_keys": dict(sorted(cfg.PROFILE_REMOVED_FIELDS.items())),
        "reserved_names": sorted(cfg.RESERVED_PROFILE_NAMES),
        "field_types": {k: _type_name(fields[k].type)
                        for k in sorted(cfg.PROFILE_FILE_KEYS) if k in fields},
        "enums": {k: sorted(v) for k, v in sorted(enums.items())},
        "key_vocabularies": {k: sorted(v)
                             for k, v in sorted(vocabularies.items())},
        "layout": {
            "extensions": list(cfg.PROFILE_FILE_EXTENSIONS),
            "yaml_extensions": list(cfg.PROFILE_YAML_EXTENSIONS),
            "profiles_subdir": cfg.PROFILES_SUBDIR,
            "profile_set_env_var": cfg.PROFILE_SET_ENV_VAR,
            "tiers": [{"tier": t, "location": loc, "rule": rule}
                      for t, loc, rule in cfg.PROFILE_DISCOVERY_TIERS],
        },
    }


# -------------------------------------------------------------- live / snapshot


def _live():
    """:mod:`introspect`, when it imports and the provider tree it reads exists.

    ``None`` means the snapshot answers.  Checking the directory, not just the
    import, covers an install that ships the scaffold package without the
    provider sources.
    """
    if _FORCE_SNAPSHOT:
        return None
    try:
        from jaato_server.shared.scaffold import introspect
    except ImportError:
        return None
    if not introspect._PROVIDER_DIR.is_dir():
        return None
    return introspect


def _live_profile_config():
    """``subagent.config`` when jaato-server imports, else ``None``.

    Separate from :func:`_live` because it reads a different part of the
    tree: the profile schema is an import, not a source scan, so it is live
    whenever jaato-server is importable, whatever the provider directory.
    """
    if _FORCE_SNAPSHOT:
        return None
    try:
        from jaato_server.shared.plugins.subagent import config
    except ImportError:
        return None
    return config


def source() -> str:
    """``"live"`` or ``"snapshot"``: which one the accessors below answer from."""
    return "live" if _live() is not None else "snapshot"


def _snapshot() -> Dict[str, Any]:
    """The parsed snapshot, validated for version, cached for the process."""
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is None:
        data = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
        version = data.get("snapshot_version")
        if version != SNAPSHOT_VERSION:
            raise RuntimeError(
                f"{SNAPSHOT_FILE.name} is snapshot_version {version!r}; this "
                f"reader understands {SNAPSHOT_VERSION}.  Regenerate it with "
                "`python -m jaato_server.shared.scaffold.authoring_contracts "
                "--write`.")
        _SNAPSHOT_CACHE = data
        _warn_on_version_skew(data)
    return _SNAPSHOT_CACHE


#: Set once the skew warning has been printed, so it is printed once per
#: process however many accessors read the snapshot.
_SKEW_WARNED = False


def installed_server_version() -> Optional[str]:
    """The installed jaato-server's version from its metadata, or ``None``.

    Metadata only: this is asked on the snapshot path, where importing
    jaato-server is exactly what did not work.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:          # pragma: no cover - stdlib since 3.8
        return None
    try:
        return version("jaato-server")
    except PackageNotFoundError:
        return None
    except Exception:            # noqa: BLE001 - a broken dist-info says nothing
        return None


def _warn_on_version_skew(data: Dict[str, Any]) -> None:
    """Warn once when an installed jaato-server differs from the snapshot's.

    Silent when no jaato-server is installed (the SDK-only case the snapshot
    exists for), and when the snapshot records no version (it cannot say).
    A warning, never a refusal: the provider list and env vars rarely change
    in a way that breaks a generated file, and refusing would take ``new``
    away from exactly the install that needs it.
    """
    global _SKEW_WARNED
    if _SKEW_WARNED:
        return
    recorded = data.get("jaato_server_version")
    installed = installed_server_version()
    if not recorded or not installed or recorded == installed:
        return
    _SKEW_WARNED = True
    print(f"jaato-scaffold: warning: provider, env-var and profile facts "
          f"come from the snapshot of jaato-server {recorded} shipped with "
          f"jaato-sdk, but "
          f"jaato-server {installed} is installed here and its source tree "
          f"could not be read; generated files may name providers or knobs "
          f"that differ from the installed server's.  Upgrading jaato-sdk to "
          f"the release cut with jaato-server {installed} brings them back in "
          f"line.", file=sys.stderr)


def providers() -> Dict[str, Any]:
    """Every provider, keyed by directory name."""
    live = _live()
    if live is not None:
        return live.providers()
    out: Dict[str, Any] = {}
    for name, rec in _snapshot()["providers"].items():
        knobs = rec["knobs"]
        out[name] = ProviderView(
            dir_name=name,
            auth=tuple(AuthStep(*step) for step in rec["auth"]),
            knobs=None if knobs is None else KnobsView(tuple(
                (layer, spec["opaque"], frozenset(spec["keys"]))
                for layer, spec in knobs.items())),
        )
    return out


def resolve_provider(name: str) -> Optional[Any]:
    """Find a provider by any accepted spelling, as ``introspect.resolve_provider`` does."""
    live = _live()
    if live is not None:
        return live.resolve_provider(name)
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


def env_vars() -> Dict[str, Any]:
    """The env vars ``build`` can render, keyed by name.

    Live, this is the whole of ``introspect.env_vars()``; the snapshot holds
    the subset :func:`_env_var_is_read_by_build` keeps.  ``build`` renders the
    same text from either.
    """
    live = _live()
    if live is not None:
        return live.env_vars()
    return {name: EnvView(name=name, **rec)
            for name, rec in _snapshot()["env_vars"].items()}


def profile_facts() -> ProfileFacts:
    """The profile facts, live when jaato-server imports, else the snapshot's.

    Resolution order, as for the provider facts: the installed jaato-server,
    then the bundled snapshot.  A reachable daemon is not asked (#1267 lists
    it second, optionally): no daemon verb serves these facts yet.
    """
    cfg = _live_profile_config()
    if cfg is not None:
        return ProfileFacts.from_json(project_profiles(cfg))
    return ProfileFacts.from_json(_snapshot()["profiles"])


# ------------------------------------------------------------- regeneration


def build_snapshot() -> Dict[str, Any]:
    """The snapshot as the live tree would write it now.

    Raises:
        RuntimeError: the live tree is unavailable, so there is nothing to
            project.  A snapshot is never regenerated from itself.
    """
    live = _live()
    cfg = _live_profile_config()
    if live is None or cfg is None:
        raise RuntimeError("the jaato-server source tree is not available; "
                           "the snapshot can only be generated from it")
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "jaato_server_version": _live_server_version(live),
        "providers": project_providers(live.providers()),
        "env_vars": project_env_vars(live.env_vars()),
        "profiles": project_profiles(cfg),
    }


def _live_server_version(live) -> str:
    """The version of the jaato-server tree the snapshot is projected from.

    The ``pyproject.toml`` beside the ``jaato_server`` package when there is
    one (a checkout: what the tree declares, even when an editable install's
    metadata was written before the last bump), else installed metadata.
    """
    import tomllib

    pkg = Path(live.__file__).resolve().parents[2]      # .../jaato_server
    pyproject = pkg.parent / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        if data.get("project", {}).get("name") == "jaato-server":
            return str(data["project"]["version"])
    except (OSError, ValueError, KeyError):
        pass
    return installed_server_version() or "unknown"


def render_snapshot(data: Dict[str, Any]) -> str:
    """The on-disk text of a snapshot: sorted keys, one key per line."""
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    """``--write`` regenerates :data:`SNAPSHOT_FILE`; ``--check`` compares it."""
    args = sys.argv[1:] if argv is None else argv
    text = render_snapshot(build_snapshot())
    if args == ["--write"]:
        SNAPSHOT_FILE.write_text(text, encoding="utf-8")
        print(f"wrote {SNAPSHOT_FILE}")
        return 0
    if args == ["--check"]:
        current = (SNAPSHOT_FILE.read_text(encoding="utf-8")
                   if SNAPSHOT_FILE.is_file() else "")
        if current == text:
            print(f"{SNAPSHOT_FILE.name} is current")
            return 0
        print(f"{SNAPSHOT_FILE.name} is stale; run with --write")
        return 1
    print("usage: python -m jaato_server.shared.scaffold.authoring_contracts "
          "--write | --check  (needs jaato-server importable)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
