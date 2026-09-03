"""Guards for the two halves of issue #721.

A test asserted ``creds is None`` after isolating only the *project*
tier of a two-tier credential lookup.  On a machine where the developer
had actually run ``openrouter-auth key``, the home tier answered, the
assertion failed, and pytest rendered the credential object into the
failure message — putting a live ``sk-or-v1-…`` into terminal
scrollback and into any CI log that captures pytest output.

Two independent things had to be true for that to happen, so there are
two guards here, and neither subsumes the other:

``TestNoTestReadsTheRealHome``
    the *isolation* half.  Fixing it makes the assertion true again —
    which removes the symptom that exposed the key without removing the
    ability to expose one.  Scoped to what the framework resolves:
    every provider's credential path, under both read and write.

``TestCredentialsCannotBePrinted``
    the *disclosure* half.  Every dataclass in the tree that carries a
    secret must render a redacted ``repr``, so the next mis-isolated
    assertion — or log line, or traceback — prints ``<redacted>``.
    Discovery is by AST scan rather than a hand-written list, so a
    provider added next year is covered without editing this file.

The isolation itself lives in ``jaato-server/conftest.py``; the
redaction helper in :mod:`shared.secret_repr`.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from shared.secret_repr import REDACTED, SECRET_FIELD_NAMES

#: Root of the installed server tree (``jaato-server/``), two levels up
#: from ``shared/tests/``.
SERVER_ROOT = Path(__file__).resolve().parents[2]

#: A value no real credential could be, distinctive enough that a
#: partial leak (a prefix, a suffix, a fingerprint) is still caught.
SENTINEL = "sk-SENTINEL-b6f0d1e2-DO-NOT-PRINT"


# ── the isolation half ──────────────────────────────────────────────

def _auth_modules() -> List[str]:
    """Return the dotted names of every provider credential module."""
    provider_root = SERVER_ROOT / "shared" / "plugins" / "model_provider"
    names = []
    for path in sorted(provider_root.glob("*/auth.py")) + sorted(
        provider_root.glob("*/oauth.py")
    ):
        names.append(
            "shared.plugins.model_provider."
            f"{path.parent.name}.{path.stem}"
        )
    return names


class TestNoTestReadsTheRealHome:
    """The developer's ``~/.jaato`` must be unreachable from a test."""

    def test_home_points_at_the_isolated_tree(self, isolated_home, real_home):
        """``Path.home()`` is the empty session directory, not the user's.

        The one assertion the whole file rests on: if this fails,
        every "no credential is configured" test below it is reading
        the machine instead of the case it names.
        """
        assert Path.home() == isolated_home
        assert Path(os.environ["HOME"]) == isolated_home, (
            "``Path.home`` is pinned but ``HOME`` is not — code reading "
            "the env var directly, or a subprocess, would still find the "
            "real home."
        )
        assert Path.home() != real_home, (
            "HOME still resolves to the real home directory — the "
            "isolation in jaato-server/conftest.py is not in effect."
        )

    def test_credential_env_is_cleared(self, credential_env_vars):
        """No provider credential arrives from the ambient environment.

        The env-tier twin of the ``HOME`` redirect: a developer with
        ``JAATO_OVHCLOUD_API_KEY`` exported fails the "no key
        configured" assertions for exactly the same reason.
        """
        assert credential_env_vars, (
            "the credential env-var set came back empty — the derivation "
            "from shared.env_scope.CATALOG in conftest.py has broken, and "
            "the ambient environment is no longer being isolated."
        )
        leaked = [name for name in credential_env_vars if name in os.environ]
        assert not leaked, f"credential env vars visible to tests: {leaked}"

    @pytest.mark.parametrize("module_name", _auth_modules())
    def test_credential_paths_stay_out_of_the_real_home(
        self, module_name, real_home, tmp_path,
    ):
        """No provider resolves a credential path into the real home.

        The workspace argument pins the *project* tier at an empty
        directory — the same isolation every caller of these loaders is
        expected to do — so what is left under test is the tier that
        falls back to ``~/.jaato``.  That has to be checked for reads
        *and* writes: a resolver that honours ``HOME`` for reads but
        hardcodes something else for writes would still overwrite the
        developer's own credential file.

        A provider that stops honouring ``HOME`` — hardcoding a path,
        or reading the ``pwd`` database directly — fails here, rather
        than by silently loading a real key inside some other test.
        """
        module = importlib.import_module(module_name)
        resolver = getattr(module, "_get_token_storage_path", None)
        if resolver is None:
            pytest.skip(f"{module_name} stores no credential file")

        kwargs = {}
        if "workspace_path" in inspect.signature(resolver).parameters:
            kwargs["workspace_path"] = str(tmp_path)

        for for_write in (False, True):
            resolved = Path(resolver(for_write=for_write, **kwargs)).resolve()
            assert not resolved.is_relative_to(real_home), (
                f"{module_name} resolves the "
                f"{'write' if for_write else 'read'} credential path to "
                f"{resolved}, inside the real home {real_home} — a test "
                "run would read (or overwrite) the developer's own "
                "credentials."
            )


# ── the disclosure half ─────────────────────────────────────────────

def _secret_bearing_dataclasses() -> List[Tuple[str, str]]:
    """Find every ``@dataclass`` in the tree with a secret-named field.

    An AST scan, so nothing is imported to answer the question and a
    module with a heavy or optional dependency cannot make the guard
    quietly skip itself.  Test modules are excluded — a fixture holding
    a fake key is not a disclosure risk.

    Returns:
        ``(dotted module name, class name)`` pairs, sorted.
    """
    found: List[Tuple[str, str]] = []
    for path in sorted(SERVER_ROOT.rglob("*.py")):
        rel = path.relative_to(SERVER_ROOT)
        parts = rel.parts
        if "tests" in parts or parts[0] not in ("shared", "server"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            decorators = {
                ast.unparse(d).split("(")[0] for d in node.decorator_list
            }
            if not decorators & {"dataclass", "dataclasses.dataclass"}:
                continue
            fields = {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            }
            if fields & SECRET_FIELD_NAMES:
                dotted = ".".join(rel.with_suffix("").parts)
                found.append((dotted, node.name))
    return sorted(found)


SECRET_DATACLASSES = _secret_bearing_dataclasses()


def _placeholder(annotation: Any) -> Any:
    """Return a harmless value of roughly the annotated type.

    Only used to fill the fields that have no default, so the dataclass
    can be constructed at all.  Precision does not matter — the value
    is never used for anything but being rendered next to the sentinel.
    ``None`` is the fallback, which every dataclass here tolerates.
    """
    text = getattr(annotation, "__name__", None) or str(annotation)
    for needle, value in (
        ("bool", False),
        ("int", 0),
        ("float", 0.0),
        ("str", "placeholder"),
        ("Dict", {}),
        ("dict", {}),
        ("List", []),
        ("list", []),
    ):
        if needle in text:
            return value
    return None


def _build_with_sentinels(cls: type) -> Tuple[Any, List[str]]:
    """Instantiate ``cls`` with :data:`SENTINEL` in every secret field.

    Returns the instance and the names of the fields the sentinel went
    into, so a failure can say which field leaked.
    """
    kwargs: Dict[str, Any] = {}
    planted: List[str] = []
    for field in dataclasses.fields(cls):
        is_secret = field.name in SECRET_FIELD_NAMES
        has_default = (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        )
        if is_secret:
            kwargs[field.name] = SENTINEL
            planted.append(field.name)
        elif not has_default:
            kwargs[field.name] = _placeholder(field.type)
    return cls(**kwargs), planted


class TestCredentialsCannotBePrinted:
    """A secret-bearing object must not render its secret."""

    def test_the_scan_found_the_known_credential_types(self):
        """The AST scan is wired up and actually finds things.

        A discovery bug would make every parametrised test below
        vacuous — zero cases, green suite, no coverage.  Pinning a
        couple of names that must always be in the set is what stops
        that from passing silently.
        """
        names = {cls for _module, cls in SECRET_DATACLASSES}
        assert "OpenRouterCredentials" in names, (
            "the dataclass scan no longer finds the type from #721; "
            f"found: {sorted(names)}"
        )
        assert "ProviderConfig" in names

    @pytest.mark.parametrize(
        "module_name,class_name",
        SECRET_DATACLASSES,
        ids=[f"{m.rsplit('.', 1)[-1]}.{c}" for m, c in SECRET_DATACLASSES],
    )
    def test_secret_never_appears_in_repr(self, module_name, class_name):
        """Rendering the object never renders the secret.

        ``repr`` is what pytest prints for a failed assertion, what
        ``logging`` prints for ``%s``, and what a traceback prints for
        a local — so covering it covers all three.  ``str`` and
        ``format`` are checked too, in case a class overrides one and
        not the other.
        """
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance, planted = _build_with_sentinels(cls)
        assert planted, f"{class_name} has no secret field to plant"

        renderings = {
            "repr()": repr(instance),
            "str()": str(instance),
            "format()": f"{instance}",
            "nested in a container": repr([instance]),
        }
        for how, text in renderings.items():
            assert SENTINEL not in text, (
                f"{module_name}.{class_name} discloses "
                f"{planted} through {how}.  Give the dataclass a "
                f'redacting repr: `__repr__ = secret_safe_repr("'
                f'{planted[0]}")` from shared.secret_repr.'
            )
            assert REDACTED in text, (
                f"{module_name}.{class_name} rendered neither the secret "
                f"nor a redaction marker through {how} — check the repr "
                "still names the secret fields."
            )

    def test_serialisation_still_carries_the_real_secret(self):
        """Redaction guards display, never storage.

        The credential file is written from ``to_dict()``; a redaction
        that reached it would write ``<redacted>`` into
        ``~/.jaato/openrouter_auth.json`` and lock the user out with a
        credential that looks present and fails to authenticate.
        """
        from shared.plugins.model_provider.openrouter.auth import (
            OpenRouterCredentials,
        )

        creds = OpenRouterCredentials(api_key=SENTINEL, created_at=0.0)
        assert creds.to_dict()["api_key"] == SENTINEL
        assert creds.api_key == SENTINEL
        assert SENTINEL not in repr(creds)

    def test_empty_secret_is_not_hidden(self):
        """An absent credential renders as absent, not as ``<redacted>``.

        "The key is missing" is not a secret, and hiding the difference
        between a missing credential and a present one is what sends
        someone hunting through the wrong half of the stack.
        """
        from shared.plugins.model_provider.openrouter.auth import (
            OpenRouterCredentials,
        )

        assert "api_key=''" in repr(
            OpenRouterCredentials(api_key="", created_at=0.0)
        )
