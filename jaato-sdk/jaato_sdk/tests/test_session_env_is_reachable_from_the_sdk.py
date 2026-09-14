"""A plugin can read a credential without importing jaato-server (#918).

The plugin contract is otherwise cleanly SDK-shaped: an out-of-tree
distribution declaring ``[project.entry-points."jaato.plugins"]`` gets
``ToolPlugin`` / ``UserCommand`` / ``TRAIT_*`` from
``jaato_sdk.plugins.base`` and ``ToolSchema`` from
``jaato_sdk.plugins.model_provider.types``, and the SDK never imports
``shared``.

The one hole was the thing a connector plugin exists to do.
``get_session_env`` lived only in ``shared.session_context``, so a
third-party plugin either depended on jaato-server or wrote
``os.environ.get(...)`` -- and that is not merely a missed abstraction.
``JaatoServer._with_session_env()`` overlays each session's ``env:`` map
onto the daemon's process environment for the duration of a turn, so on
a daemon serving two tenants a plain read can return the *other*
session's token, non-deterministically and silently.

Two properties make the fix real, and both are pinned here:

1. **One ContextVar object.**  A second var declared in the SDK would
   read empty, fall through to ``os.environ``, and reproduce the bug in
   a form that looks fixed.  So the server imports the SDK's, and the
   identity is asserted rather than assumed.
2. **No server import.**  If reaching the function still requires
   ``shared``, nothing was gained.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

_HAS_SERVER = importlib.util.find_spec("shared") is not None


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=60,
    )


# ---------------------------------------------------------------- reachability


def test_reaching_it_does_not_import_the_server():
    """Measured in a fresh interpreter, because another test in this
    session may already have imported ``shared`` for its own reasons —
    which would make the import-purity check pass vacuously."""
    proc = _run(
        """
        import sys
        from jaato_sdk.session_env import get_session_env
        from jaato_sdk.plugins.base import get_session_env as via_base
        assert via_base is get_session_env
        assert "shared" not in sys.modules, sorted(
            m for m in sys.modules if m.startswith("shared")
        )
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_the_three_documented_import_routes_agree():
    """``jaato_sdk`` / ``jaato_sdk.session_env`` / ``jaato_sdk.plugins.base``
    must all hand back the same function — an author who finds one route
    must not get a different implementation from another."""
    import jaato_sdk
    from jaato_sdk.plugins.base import get_session_env as via_base
    from jaato_sdk.session_env import get_session_env as via_module

    assert jaato_sdk.get_session_env is via_module
    assert via_base is via_module
    assert "get_session_env" in jaato_sdk.__all__


# ---------------------------------------------------------------- the semantics


def test_session_value_wins_over_the_process_environment(monkeypatch):
    """The whole point: the overlay is what a plugin reads, not the
    global dict a concurrent session may have just rewritten."""
    from jaato_sdk.session_env import (
        clear_session_env, get_session_env, set_session_env,
    )

    monkeypatch.setenv("GRAPH_CLIENT_SECRET", "tenant-a-leaked-into-os-environ")
    set_session_env({"GRAPH_CLIENT_SECRET": "tenant-b"})
    try:
        assert get_session_env("GRAPH_CLIENT_SECRET") == "tenant-b"
    finally:
        clear_session_env()


def test_it_falls_back_to_the_process_environment(monkeypatch):
    """Outside a session context — a test, a CLI, daemon startup — the
    call must still work, which is what makes it safe to write
    unconditionally instead of behind a soft import."""
    from jaato_sdk.session_env import clear_session_env, get_session_env

    clear_session_env()
    monkeypatch.setenv("GRAPH_CLIENT_SECRET", "from-the-process")
    assert get_session_env("GRAPH_CLIENT_SECRET") == "from-the-process"


def test_absent_everywhere_returns_the_default(monkeypatch):
    from jaato_sdk.session_env import clear_session_env, get_session_env

    clear_session_env()
    monkeypatch.delenv("NO_SUCH_JAATO_VAR", raising=False)
    assert get_session_env("NO_SUCH_JAATO_VAR") is None
    assert get_session_env("NO_SUCH_JAATO_VAR", "fallback") == "fallback"


def test_a_key_absent_from_the_overlay_still_falls_through(monkeypatch):
    """The overlay is not a whitelist.  A session env carrying one key
    must not hide every other variable the process holds — a connector
    reading ``HTTPS_PROXY`` alongside its token depends on this."""
    from jaato_sdk.session_env import (
        clear_session_env, get_session_env, set_session_env,
    )

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    set_session_env({"GRAPH_CLIENT_SECRET": "tenant-b"})
    try:
        assert get_session_env("HTTPS_PROXY") == "http://proxy:8080"
    finally:
        clear_session_env()


# -------------------------------------------------- one var, not two copies


@pytest.mark.skipif(not _HAS_SERVER, reason="jaato-server not installed")
def test_the_server_and_the_sdk_share_one_contextvar():
    """The failure this guards against is the quiet one: two vars, a
    plugin reading the SDK's, the daemon writing the server's, and the
    read silently falling through to ``os.environ`` — i.e. the original
    cross-session leak, wearing the fix as a disguise.

    Asserting object identity is what makes that undetectable failure
    detectable, so it is asserted on the objects rather than inferred
    from behaviour.
    """
    from shared import session_context
    from jaato_sdk import session_env

    assert session_context.get_session_env is session_env.get_session_env
    assert session_context.set_session_env is session_env.set_session_env
    assert session_context.clear_session_env is session_env.clear_session_env
    assert session_context._session_env is session_env._session_env


@pytest.mark.skipif(not _HAS_SERVER, reason="jaato-server not installed")
def test_a_daemon_side_write_is_visible_to_an_sdk_side_read(monkeypatch):
    """The behavioural half of the identity check, in the direction that
    actually occurs: ``_with_session_env`` writes through the server
    module, an out-of-tree plugin reads through the SDK."""
    from shared.session_context import set_session_env as server_set
    from shared.session_context import clear_session_env as server_clear
    from jaato_sdk.session_env import get_session_env as plugin_read

    monkeypatch.setenv("GRAPH_CLIENT_SECRET", "wrong-tenant")
    server_set({"GRAPH_CLIENT_SECRET": "right-tenant"})
    try:
        assert plugin_read("GRAPH_CLIENT_SECRET") == "right-tenant"
    finally:
        server_clear()
    assert plugin_read("GRAPH_CLIENT_SECRET") == "wrong-tenant"


@pytest.mark.skipif(not _HAS_SERVER, reason="jaato-server not installed")
def test_in_tree_callers_keep_their_import_path():
    """Fourteen plugin packages do ``from shared.session_context import
    get_session_env``.  The move to the SDK is a re-export, not a
    relocation — an in-tree import that breaks would be a migration
    nobody asked for."""
    from shared.session_context import get_session_env  # noqa: F401
    from shared.session_context import get_current_session  # noqa: F401


def test_get_current_session_is_deliberately_not_exported():
    """It hands back a ``JaatoSession`` — server-side by nature — and a
    plugin reaching into ``session._runtime`` is not something to make
    easier from out of tree.  Its absence is a decision, so it is
    pinned like one."""
    import jaato_sdk
    from jaato_sdk import session_env

    assert not hasattr(session_env, "get_current_session")
    assert "get_current_session" not in getattr(jaato_sdk, "__all__", ())
