"""The provider and env-var facts ``jaato-scaffold new`` needs, live or from a snapshot.

``new`` asks two questions about the framework:

- **which providers exist**, and for one of them, which env var holds its key
  (the ``AuthSource`` chain) and which config keys it accepts (the
  ``PROVIDER_KNOBS`` layers).  This validates ``--provider``, names the key
  var in a generated profile and ``.env``, and decides which worked examples
  a profile carries.
- **which env vars the daemon reads**, with their categories and defaults.
  This fills the commented knob catalogue in every generated ``.env``.

:mod:`introspect` answers both by reading the INSTALLED jaato-server source
with ``ast``: the 27 provider ``__init__.py`` files and ~630 modules under
``jaato_server/{server,shared}``.  Nothing is imported, so the dependency
does not show up in ``sys.modules``, but it is real.  Once the authoring
commands ship without jaato-server (#1267), those files are not on disk.

This module is the one door ``build`` goes through:

- **live** when :mod:`introspect` imports and the provider tree it reads is
  present.  The objects returned are introspect's own, so output is
  unchanged.
- **snapshot** otherwise: :data:`SNAPSHOT_FILE`, a projection of the live
  answers to exactly the fields ``build`` reads.  The snapshot objects
  (:class:`ProviderView`, :class:`EnvView`) are duck-compatible with
  ``introspect.ProviderInfo`` / ``introspect.EnvVar`` for those fields and no
  others.

The snapshot is checked in and regenerated with::

    python -m jaato_server.shared.scaffold.authoring_contracts --write

``test_authoring_snapshot_matches_the_tree.py`` fails when it differs from
the live projection, naming that command.  A snapshot nobody checks is a
second source of truth that drifts silently; the guard is what keeps this one
from being that.

Imports only the stdlib at module level.  :mod:`introspect` is imported
inside :func:`_live`, and only there.
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
#: file written for another shape.
SNAPSHOT_VERSION = 1

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
        from . import introspect
    except ImportError:
        return None
    if not introspect._PROVIDER_DIR.is_dir():
        return None
    return introspect


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
    return _SNAPSHOT_CACHE


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


# ------------------------------------------------------------- regeneration


def build_snapshot() -> Dict[str, Any]:
    """The snapshot as the live tree would write it now.

    Raises:
        RuntimeError: the live tree is unavailable, so there is nothing to
            project.  A snapshot is never regenerated from itself.
    """
    live = _live()
    if live is None:
        raise RuntimeError("the jaato-server source tree is not available; "
                           "the snapshot can only be generated from it")
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "providers": project_providers(live.providers()),
        "env_vars": project_env_vars(live.env_vars()),
    }


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
          "--write | --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
