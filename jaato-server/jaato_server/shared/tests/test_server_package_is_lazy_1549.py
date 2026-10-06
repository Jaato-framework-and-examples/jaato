"""Reading a constant from ``jaato_server.server.*`` does not load the daemon (#1549).

``jaato_server/server/__init__.py`` imported ``JaatoServer`` (``.core``) and
``SessionManager`` (``.session_manager``) at module level.  Every import of a
submodule runs the package ``__init__`` first, so ``jaato-scaffold explain
runner-user`` and ``explain pool``, which read only constant tables, loaded
~116 ``jaato_server`` modules: core, the session manager, permission, memory,
todo, telemetry.  The ``__init__`` is now lazy (a module ``__getattr__``, the
shape #1267 gave the subagent plugin package), and the re-exported names still
resolve on first access.

Footprints are measured in a fresh interpreter, because a module this process
already imported says nothing about what a cold import loads.  The child
inherits ``PYTHONPATH``, so under the reversion meta-guard it imports the
sandbox's copy.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from jaato_server.shared.tests.reversion import Reversion

_INIT = "jaato-server/jaato_server/server/__init__.py"

REVERSIONS = [
    Reversion(
        target=_INIT,
        find="_LAZY_IMPORTS = {\n",
        replace=("from .core import JaatoServer\n"
                 "from .session_manager import SessionManager\n"
                 "_LAZY_IMPORTS = {\n"),
        because="an eager package import makes every server.* submodule "
                "load the daemon core and the session manager",
        test="test_a_constant_module_does_not_load_the_daemon",
    ),
]

_DAEMON = ("jaato_server.server.core", "jaato_server.server.session_manager")


def _cold_modules(code: str) -> set:
    """``jaato_server.*`` modules loaded by *code* in a fresh interpreter."""
    probe = (
        "import sys, json\n"
        f"{code}\n"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('jaato_server'))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe],
                          capture_output=True, text=True, check=True)
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("module", [
    "jaato_server.server.runner_user",
    "jaato_server.server.pool_admin",
])
def test_a_constant_module_does_not_load_the_daemon(module):
    loaded = _cold_modules(f"import {module}")
    assert module in loaded
    assert not loaded.intersection(_DAEMON), sorted(loaded)


def test_the_re_exported_names_still_resolve():
    """``from jaato_server.server import X`` keeps working for every name."""
    code = (
        "import jaato_server.server as s\n"
        "missing = [n for n in s.__all__ if getattr(s, n, None) is None]\n"
        "assert not missing, missing\n"
        "from jaato_server.server import JaatoServer, SessionManager\n"
        "assert JaatoServer.__module__ == 'jaato_server.server.core'\n"
        "assert SessionManager.__module__ == "
        "'jaato_server.server.session_manager'\n"
        "assert set(s.__all__) <= set(dir(s))\n"
    )
    loaded = _cold_modules(code)
    assert loaded.issuperset(_DAEMON)


def test_an_unknown_name_is_an_attribute_error():
    """A submodule is still importable by name, and a typo still fails."""
    import jaato_server.server as s

    with pytest.raises(AttributeError):
        s.NoSuchThing  # noqa: B018
    from jaato_server.server import pool_admin
    assert pool_admin.__name__ == "jaato_server.server.pool_admin"
