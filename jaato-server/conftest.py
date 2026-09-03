"""Test-wide isolation from the machine the tests are running on.

WHAT THIS FIXES.  Credential resolution is multi-tier: a project
``.jaato/<provider>_auth.json``, then ``~/.jaato/<provider>_auth.json``,
with ambient env vars consulted alongside.  A test that isolates only
the project tier — ``try_load_credentials_with_reason(workspace_path=
str(tmp_path))`` — has isolated one tier of three, and the other two
still resolve against whatever the developer running the suite has
installed.

That is not a cosmetic gap.  Three concrete consequences, all observed:

- **It disclosed a live key.**  ``assert creds is None`` fails on an
  authenticated machine, and pytest renders the left-hand operand into
  the failure message: a real ``sk-or-v1-…`` went into terminal
  scrollback, and would go into any CI log capturing pytest output
  (issue #721).  The credential dataclasses now carry a redacting
  ``__repr__`` (:mod:`shared.secret_repr`) so the disclosure cannot
  recur, but the assertion was still reading the wrong thing.
- **It inverted the coverage.**  ``test_file_missing_returns_none``
  exists to cover the no-credential path.  On an authenticated machine
  the file is not missing, so the case the test names is the one case
  it stops exercising.
- **It fails only for authenticated developers**, i.e. the people most
  likely to run the auth suites, and it fails on their branch — which
  cost a false regression attribution once already (#734).

WHAT IT DOES.  Three tiers are neutralised for every test under
``jaato-server/``:

1. ``HOME`` (and the Windows / XDG equivalents) points at an empty
   per-session directory, so ``Path.home() / ".jaato"`` resolves
   somewhere with nothing in it.
2. The ambient config roots (``JAATO_CONFIG_ROOT``,
   ``JAATO_WORKSPACE_ROOT``) are cleared, because an exported
   ``JAATO_CONFIG_ROOT`` re-points the *project* tier and so survives
   an explicit ``workspace_path=`` argument.
3. Every credential env var the framework reads is cleared — derived
   from :data:`shared.env_scope.CATALOG`, which a guard keeps
   exhaustive, so a provider added later is covered without editing
   this file.  This is the env-tier twin of (1): a developer with
   ``JAATO_OVHCLOUD_API_KEY`` exported fails the "no key configured"
   assertions for the same reason.

WHY THE ENV IS SET AT IMPORT TIME, NOT ONLY IN A FIXTURE.  Several
modules capture a home-derived path at import (``DEFAULT_REGISTRY_PATH
= Path.home() / ".jaato" / "workspaces.json"`` and friends).  A fixture
runs too late to affect those — the module is imported during
collection.  conftest import happens first, so the redirect is in place
before any test module is imported.  The autouse fixture then re-asserts
it per test, so a test that legitimately repoints ``HOME`` (there are a
few) cannot leak that into the next one.

WHAT IT DELIBERATELY DOES NOT DO.  It does not clear ``PATH``, proxy
settings or the rest of the ambient environment: tests that shell out
need a working machine.  It is scoped to the tiers credential and
config resolution actually consults.

Tests that want the opposite — to prove the home tier IS read — set
``HOME`` themselves with ``monkeypatch``, which wins over this fixture
because function-scoped monkeypatching happens after it.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest

#: The real home directory of whoever is running the suite, captured
#: BEFORE the redirect below.  The guard in
#: ``shared/tests/test_credential_hygiene.py`` uses it to assert that
#: no auth module resolves a credential path back into it.
REAL_HOME = Path.home()

#: Config roots that re-point the *project* credential tier.  An
#: exported ``JAATO_CONFIG_ROOT`` outranks an explicit ``workspace_path``
#: argument in every provider's ``_get_token_storage_path``, so passing
#: a ``tmp_path`` workspace is not on its own enough to isolate a test.
AMBIENT_CONFIG_VARS = ("JAATO_CONFIG_ROOT", "JAATO_WORKSPACE_ROOT")

#: Names in the env catalog that hold a secret.  Matched against
#: :data:`shared.env_scope.CATALOG` rather than against ``os.environ``,
#: so only vars the framework actually reads are touched — an unrelated
#: ``COMPANY_API_KEY`` in the developer's shell is left alone.
_CREDENTIAL_NAME_RE = re.compile(
    r"(API_KEY|_KEY|_TOKEN|_TOKEN_FILE|_SECRET|_PASSWORD|CREDENTIALS)$"
)


def _credential_env_vars() -> tuple:
    """Return the credential env vars the framework reads.

    Derived from the scope catalog so a provider added later is covered
    without touching this file.  Falls back to an empty tuple if the
    catalog cannot be imported — an import failure here must not take
    the whole suite down, and the ``HOME`` isolation above is the part
    that carries the disclosure fix.
    """
    try:
        from shared.env_scope import CATALOG
    except Exception:  # pragma: no cover - defensive
        return ()
    return tuple(sorted(n for n in CATALOG if _CREDENTIAL_NAME_RE.search(n)))


CREDENTIAL_ENV_VARS = _credential_env_vars()

_ISOLATED_HOME = Path(tempfile.mkdtemp(prefix="jaato-test-home-"))
atexit.register(shutil.rmtree, _ISOLATED_HOME, ignore_errors=True)

#: Env vars that steer ``Path.home()`` / ``expanduser`` and the XDG
#: lookups, mapped to their isolated values.  ``HOME`` is *set*, never
#: deleted: ``os.path.expanduser`` falls back to the ``pwd`` database
#: when it is absent, which would resolve to the real home again.
ISOLATED_HOME_ENV = {
    "HOME": str(_ISOLATED_HOME),
    "USERPROFILE": str(_ISOLATED_HOME),          # Windows
    "HOMEDRIVE": "",                             # Windows: HOMEDRIVE+HOMEPATH
    "HOMEPATH": str(_ISOLATED_HOME),             # is expanduser's fallback pair
    "XDG_CONFIG_HOME": str(_ISOLATED_HOME / ".config"),
    "XDG_DATA_HOME": str(_ISOLATED_HOME / ".local" / "share"),
    "XDG_STATE_HOME": str(_ISOLATED_HOME / ".local" / "state"),
    "XDG_CACHE_HOME": str(_ISOLATED_HOME / ".cache"),
}


def _apply_isolation() -> None:
    """Point the home / config / credential env at the isolated tree."""
    os.environ.update(ISOLATED_HOME_ENV)
    for name in AMBIENT_CONFIG_VARS + CREDENTIAL_ENV_VARS:
        os.environ.pop(name, None)


# Applied at import so module-level constants computed from ``Path.home()``
# during collection see the isolated tree, not the developer's.
_apply_isolation()


@pytest.fixture(autouse=True)
def isolated_machine_state(monkeypatch):
    """Re-assert home / config / credential isolation for each test.

    The import-time redirect above covers collection; this covers the
    run, and undoes any repointing a previous test did.  Yields the
    isolated home so a test can assert against it, or plant a file in
    it to exercise the user tier on purpose.
    """
    for name, value in ISOLATED_HOME_ENV.items():
        monkeypatch.setenv(name, value)
    for name in AMBIENT_CONFIG_VARS + CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # ``Path.home()`` reads ``HOME`` — until a test clears the whole
    # environment (``patch.dict("os.environ", {}, clear=True)`` is a
    # common way to assert "no credentials are configured").  With
    # ``HOME`` gone, ``expanduser`` falls back to the ``pwd`` database
    # and resolves the real home again, so the tests most determined to
    # start from nothing were the ones that reached furthest into the
    # developer's machine.  Pinning the attribute closes that.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: _ISOLATED_HOME))
    return _ISOLATED_HOME


@pytest.fixture(scope="session")
def real_home() -> Path:
    """The home directory of whoever is running the suite.

    Captured before the redirect, and exposed so the hygiene guard in
    ``shared/tests/test_credential_hygiene.py`` can assert that no
    credential path resolves back into it.  Nothing else should want
    this: a test that reads the real home is the bug this file exists
    to prevent.
    """
    return REAL_HOME


@pytest.fixture(scope="session")
def isolated_home() -> Path:
    """The empty directory ``HOME`` points at for the whole session."""
    return _ISOLATED_HOME


@pytest.fixture(scope="session")
def credential_env_vars() -> tuple:
    """Credential env vars cleared for every test (see module docstring)."""
    return CREDENTIAL_ENV_VARS
