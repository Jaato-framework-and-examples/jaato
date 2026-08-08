"""Regression guards for the sdk-only `jaato` facade (server may be remote).

The whole point of shipping the top-level `jaato` package in jaato-sdk is that a
thin client — `import jaato` + `jaato.session(mode="ipc"/"ws")` — needs only the
SDK, because the daemon it talks to may be on another host. These tests lock that
in:

1. **Import purity** — `import jaato` must NOT drag in the server runtime
   (`shared`) or the embedded backend (`jaato_embedded`), even when they happen to
   be installed. Run in a subprocess so a clean `sys.modules` measures the eager
   import, not what other tests already loaded.
2. **in-process fail-loud** — with the embedded backend unavailable,
   `session(mode="in_process")` raises a clear `ImportError` with an install hint,
   while `mode="ipc"/"ws"` still construct (they never touch `jaato_embedded`).
3. **lazy server names** — when jaato-server IS installed, `from jaato import
   JaatoClient/PluginRegistry/InProcessClient` resolve via PEP 562, and those lazy
   names stay OUT of `__all__` (so `from jaato import *` is sdk-only-safe).
"""

import importlib.util
import subprocess
import sys
import textwrap

import pytest

_HAS_EMBEDDED = importlib.util.find_spec("jaato_embedded") is not None


def _run(code: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter; return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=60,
    )


def test_import_jaato_does_not_pull_server_runtime():
    """`import jaato` must not import `shared` or `jaato_embedded` eagerly."""
    proc = _run(
        """
        import sys
        import jaato
        assert "shared" not in sys.modules, "import jaato pulled `shared`"
        assert "jaato_embedded" not in sys.modules, "import jaato pulled `jaato_embedded`"
        # Eager sdk-only surface + the transport dispatcher are available.
        from jaato import Message, ToolSchema, session, PresentationContext
        assert callable(session)
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_in_process_mode_fails_loud_without_embedded_backend():
    """With `jaato_embedded` unavailable, `session(mode="in_process")` raises an
    ImportError carrying the install hint; ipc/ws still construct (they never
    import the embedded backend)."""
    proc = _run(
        """
        import sys, importlib.abc

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "jaato_embedded" or name.startswith("jaato_embedded."):
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        sys.modules.pop("jaato_embedded", None)
        sys.meta_path.insert(0, _Block())

        import jaato

        # in_process must fail loud with the hint.
        try:
            jaato.session(mode="in_process", model="m", provider="p")
            raise SystemExit("expected ImportError for in_process without backend")
        except ImportError as e:
            assert "jaato-server" in str(e), f"missing install hint: {e}"

        # ipc/ws construct fine with the backend blocked (sdk-only client path).
        jaato.session(mode="ipc", socket_path="/tmp/nonexistent.sock")
        jaato.session(mode="ws", url="ws://localhost:1/none")
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_all_excludes_lazy_server_names():
    """The lazy server-only names must not be in __all__ (else `from jaato import
    *` would trigger __getattr__ and break an sdk-only install)."""
    import jaato

    assert "session" in jaato.__all__
    for name in ("JaatoClient", "PluginRegistry", "InProcessClient"):
        assert name not in jaato.__all__, f"{name} must not be in jaato.__all__"


@pytest.mark.skipif(
    not _HAS_EMBEDDED,
    reason="jaato-server (jaato_embedded) not installed — lazy-resolve guard is "
           "server-side only",
)
def test_lazy_server_names_resolve_when_server_installed():
    """When jaato-server IS installed, the lazy names resolve via PEP 562."""
    from jaato import JaatoClient, PluginRegistry, InProcessClient

    assert JaatoClient.__name__ == "JaatoClient"
    assert PluginRegistry.__name__ == "PluginRegistry"
    assert InProcessClient.__name__ == "InProcessClient"
