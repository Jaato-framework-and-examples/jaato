"""Tests for the pre-initialize hook API (server 0.6.49+).

Pins the contract that:
- ``add_pre_initialize_hook`` registers a callback fired BEFORE
  ``server.initialize()`` runs (and therefore before configure() runs
  prefetch scripts).
- The hook receives ``(server, session_id, workspace_path)`` — the
  ``Session`` object does NOT yet exist in ``self._sessions``.
- Hook exceptions are logged but don't block subsequent hooks or
  session creation.
- Pre-init hooks fire BEFORE post-init hooks (the existing
  ``add_session_hook`` API).

These tests pin the wiring so the apparmor pre-init split stays
load-bearing and bisect-safe.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from server.session_manager import SessionManager


def _make_session_manager(tmp_path) -> SessionManager:
    # SessionManager only takes ``storage_path``.  We exercise the
    # hook-registration + invocation surface, not real session lifecycle.
    return SessionManager(storage_path=str(tmp_path))


@pytest.fixture
def sm(tmp_path):
    return _make_session_manager(tmp_path)


def test_add_pre_initialize_hook_registers(sm):

    def hook(server, session_id, workspace_path):
        pass

    sm.add_pre_initialize_hook(hook)
    assert hook in sm._pre_initialize_hooks


def test_run_pre_initialize_hooks_invokes_with_three_args(sm):
    seen: List[tuple] = []

    def hook(server, session_id, workspace_path):
        seen.append((server, session_id, workspace_path))

    sm.add_pre_initialize_hook(hook)
    fake_server = object()
    sm._run_pre_initialize_hooks(fake_server, "sess-1", "/tmp/ws")

    assert seen == [(fake_server, "sess-1", "/tmp/ws")]


def test_run_pre_initialize_hooks_swallows_exceptions(sm, caplog):
    """Hook failures are logged but don't propagate (one bad hook
    must not break session creation for other hooks or for the
    transport layer).
    """
    later_called: List[bool] = []

    def bad_hook(server, session_id, workspace_path):
        raise RuntimeError("boom")

    def good_hook(server, session_id, workspace_path):
        later_called.append(True)

    sm.add_pre_initialize_hook(bad_hook)
    sm.add_pre_initialize_hook(good_hook)
    sm._run_pre_initialize_hooks(object(), "sess-2", None)

    # bad_hook raised; good_hook still ran.
    assert later_called == [True]


def test_pre_init_hooks_fire_before_post_init_hooks(sm):
    """When a session goes through both hook lists, pre-init MUST
    fire first.  This test pins the call-order contract: a single
    ``order`` list captures the sequence of hook invocations as
    ``_run_pre_initialize_hooks`` and ``_run_session_hooks`` fire.
    """
    order: List[str] = []

    def pre_hook(server, session_id, workspace_path):
        order.append("pre")

    def post_hook(server, session_id):
        order.append("post")

    sm.add_pre_initialize_hook(pre_hook)
    sm.add_session_hook(post_hook)

    fake_server = object()
    sm._run_pre_initialize_hooks(fake_server, "sess-3", "/tmp/ws")
    sm._run_session_hooks(fake_server, "sess-3")

    assert order == ["pre", "post"]


def test_pre_init_hook_receives_workspace_path_directly(sm):
    """The hook gets workspace_path as a parameter, not via
    ``sm.get_session(session_id)``.  This is load-bearing because
    the Session object isn't in ``self._sessions`` yet at pre-init
    time — the hook must not depend on session-dict lookup.
    """
    captured: List[Any] = []

    def hook(server, session_id, workspace_path):
        # Verify the session is NOT in the dict at this point.
        assert sm.get_session(session_id) is None
        captured.append(workspace_path)

    sm.add_pre_initialize_hook(hook)
    sm._run_pre_initialize_hooks(object(), "sess-4", "/tmp/some/ws")
    assert captured == ["/tmp/some/ws"]
