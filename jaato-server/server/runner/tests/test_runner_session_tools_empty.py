"""Runner-side ``_build_session`` tools-list passthrough (the live
production path for profile.plugins resolution).

Pins the fix shipped 2026-06-07 for the vLLM smoke wire-bug
investigation:

- Pre-fix: ``tools=tool_names or None`` swallowed the explicit-empty
  list.  When ``profile.plugins: []`` was declared, ``tool_names``
  ended up ``[]``; ``[] or None`` evaluates to ``None``; runtime
  interpreted ``None`` as "load ALL exposed plugins" per its
  docstring.  Empirical proof: 22 tools on the vLLM wire instead of
  the expected ~8 (introspection + lifecycle + framework infra).
- Post-fix: ``tools=tool_names`` passes the empty list verbatim.
  Runtime receives ``tools=[]`` and produces the minimal set.

The earlier ``server/core.py:_build_profile_session_kwargs`` fix
(PR #240) targeted the SAME falsy bug but in a function that turned
out to be dead code (only test callers).  This is the live path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from server.runner.session import _build_session
from shared.session_envelope import SessionInitEnvelope


class _CapturingRuntime:
    """Records the args ``_build_session`` passes to
    :meth:`JaatoRuntime.create_session`."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create_session(self, **kwargs: Any) -> object:
        self.calls.append(dict(kwargs))
        return object()


def _envelope_with_plugins(
    plugins: Optional[List[Dict[str, Any]]],
) -> SessionInitEnvelope:
    """Build a minimal envelope with the given plugins list.

    ``envelope.plugins`` entries are dicts ``{"name": str, "preload":
    bool, "config": dict?}``; the helper extracts the name list as
    ``tool_names`` and the preload set.
    """
    return SessionInitEnvelope(
        session_id="sess-tools-test",
        workspace_path="/tmp/ws",
        profile_name="test-profile",
        provider_name="lmstudio",
        model_name="test-model",
        plugins=plugins,
    )


def test_empty_plugins_passes_empty_list_not_none() -> None:
    """The bug: ``profile.plugins: []`` → ``tool_names = []`` →
    pre-fix ``[] or None`` → runtime got ``tools=None`` → loaded
    ALL exposed plugins (~22 tools on the vLLM smoke wire).

    Post-fix: ``tools=[]`` flows through, runtime returns just the
    minimal framework set."""
    runtime = _CapturingRuntime()
    env = _envelope_with_plugins([])

    _build_session(runtime, env)  # type: ignore[arg-type]

    assert len(runtime.calls) == 1
    kwargs = runtime.calls[0]
    assert kwargs["plugins"] == [], (
        f"Empty plugins list must produce plugins=[] (NOT None).  "
        f"Got plugins={kwargs['plugins']!r}.  When None reaches "
        f"runtime.create_session it falls back to 'load all exposed "
        f"plugins' — the exact bug the vLLM smoke surfaced "
        f"2026-06-07 (22 tools instead of the expected ~8)."
    )


def test_non_empty_plugins_passes_through_verbatim() -> None:
    """Sanity: non-empty plugin lists still reach the runtime."""
    runtime = _CapturingRuntime()
    env = _envelope_with_plugins([
        {"name": "cli", "preload": False},
        {"name": "todo", "preload": True},
    ])

    _build_session(runtime, env)  # type: ignore[arg-type]

    kwargs = runtime.calls[0]
    assert kwargs["plugins"] == ["cli", "todo"]
    assert kwargs["preloaded_plugins"] == {"todo"}


def test_single_plugin_passes_through() -> None:
    runtime = _CapturingRuntime()
    env = _envelope_with_plugins([{"name": "cli", "preload": False}])

    _build_session(runtime, env)  # type: ignore[arg-type]

    assert runtime.calls[0]["plugins"] == ["cli"]


def test_absent_profile_passes_none_not_empty_list() -> None:
    """The third case: NO profile at all.

    ``plugins: [x, y]`` means those; ``plugins: []`` means the minimal
    set; **no profile** means "nobody asked for a subset" and must reach
    ``runtime.create_session`` as ``None``, which expands to every
    exposed plugin — so the eager wire keeps the core tools (introspection
    among them) and the model can discover the rest.

    Regression: ``SessionInitEnvelope.plugins`` used to be a
    ``default_factory=list`` field, so a profile-less session arrived
    here as ``[]`` — indistinguishable from an explicit ``plugins: []``.
    It got the minimal set, and ``_should_drop_introspection`` then
    removed list_tools/get_tool_schemas as well (nothing left to
    discover), leaving a session with NO tools while the system prompt
    still instructed the model to discover them.  The model narrated
    tool calls in prose and fabricated their results.
    """
    runtime = _CapturingRuntime()
    env = _envelope_with_plugins(None)

    _build_session(runtime, env)  # type: ignore[arg-type]

    kwargs = runtime.calls[0]
    assert kwargs["plugins"] is None, (
        f"A profile-less session must reach the runtime as plugins=None "
        f"(\"all exposed plugins\").  Got plugins={kwargs['plugins']!r}.  "
        f"An empty list here means 'a profile asked for the minimal set' "
        f"— a different answer, and the one that empties the tool wire."
    )
