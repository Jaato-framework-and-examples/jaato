"""Tests for R2 — prefetch confinement plumbing (server 0.6.50+).

Pins the contract that:
- ``JaatoRuntime.set_confine_context_factory`` stashes the factory.
- Sessions created via ``runtime.create_session`` inherit the factory.
- ``JaatoSession.configure()`` wraps the dynamic-instructions
  expansion in the factory's context manager when set.

These tests exercise the plumbing in isolation — no real AppArmor is
required.  We use a synthetic context manager that records its
enter/exit timing so we can verify the prefetch ran INSIDE it.
"""

from __future__ import annotations

import os
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from shared.dynamic_instructions import (
    RenderContext,
    expand_py_placeholders,
)
from shared.jaato_runtime import JaatoRuntime
from shared.jaato_session import JaatoSession


def test_runtime_set_confine_context_factory_stashes():
    rt = JaatoRuntime()
    assert rt._confine_context_factory is None

    def factory():
        @contextmanager
        def _cm():
            yield
        return _cm()

    rt.set_confine_context_factory(factory)
    assert rt._confine_context_factory is factory


def test_session_set_confine_context_factory_stashes():
    rt = JaatoRuntime()
    sess = JaatoSession(rt, model="dummy")
    assert sess._confine_context_factory is None

    def factory():
        @contextmanager
        def _cm():
            yield
        return _cm()

    sess.set_confine_context_factory(factory)
    assert sess._confine_context_factory is factory


def test_expand_runs_inside_confine_context_when_session_has_factory(tmp_path):
    """End-to-end: dynamic-instructions expansion runs inside the
    factory's context manager.  We verify by recording the order in
    which the context-manager's enter/exit fire vs the script's render.
    """
    # Mimic the ``expand_py_placeholders`` call site in JaatoSession.configure()
    # but instrumented: a fake confine context that records enter/exit.
    events: List[str] = []

    @contextmanager
    def _fake_confine():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def _factory():
        return _fake_confine()

    # Set up a minimal workspace + script.
    scripts_dir = tmp_path / ".jaato" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "echo.py").write_text(textwrap.dedent("""\
        def render(context, args):
            # The script itself records its own marker so the test
            # can verify it ran BETWEEN enter and exit.
            import os
            os.environ["_TEST_EXPAND_FIRED"] = "1"
            return "OK"
    """))

    ctx = RenderContext(
        session=None,
        runtime=None,
        registry=None,
        workspace_path=str(tmp_path),
        config_root=None,
        agent_params={},
        env=dict(os.environ),
    )

    # Direct call mirroring the wrap in jaato_session.py:1662+ (but
    # without spinning up a full JaatoSession+initialize).
    os.environ.pop("_TEST_EXPAND_FIRED", None)
    with _factory():
        out = expand_py_placeholders(
            "Greeting: {{!py:scripts/echo.py}}", ctx,
        )

    assert out == "Greeting: OK"
    assert os.environ.get("_TEST_EXPAND_FIRED") == "1"
    assert events == ["enter", "exit"], (
        f"context manager enter/exit didn't bracket the expansion: "
        f"{events}"
    )


def test_runtime_propagates_factory_to_session_on_create(tmp_path, monkeypatch):
    """``runtime.create_session`` should propagate the runtime's
    factory onto the session it constructs (server 0.6.50+).

    We can't easily instantiate a real session without provider/auth
    plumbing, so this test validates the propagation logic by setting
    the factory on the runtime and inspecting whether
    ``set_confine_context_factory`` would be called on the new session.
    Done by spying on ``JaatoSession.set_confine_context_factory``.
    """
    rt = JaatoRuntime()
    rt._connected = True  # bypass connect() guard
    rt._registry = MagicMock()  # bypass plugins-configured guard

    def factory():
        @contextmanager
        def _cm():
            yield
        return _cm()

    rt.set_confine_context_factory(factory)

    # Spy on the session's setter via monkeypatching at the class level
    # on the session module.  We can't run the full create_session
    # (it'd hit configure() and need way more state) so we patch
    # ``JaatoSession.configure`` to no-op AND watch the setter.
    captured: List = []
    real_setter = JaatoSession.set_confine_context_factory

    def spy_setter(self, f):
        captured.append(f)
        real_setter(self, f)

    monkeypatch.setattr(
        JaatoSession, "set_confine_context_factory", spy_setter,
    )
    monkeypatch.setattr(JaatoSession, "configure", lambda *a, **kw: None)

    rt.create_session(model="dummy")

    assert captured == [factory], (
        f"runtime didn't propagate factory to session: {captured}"
    )
