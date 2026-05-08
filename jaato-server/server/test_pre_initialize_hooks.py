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


# ----------------------------------------------------------------------
# Phase 2 task 2.3 (post-rebase): client_id parameter
# ----------------------------------------------------------------------


def test_pre_init_hook_receives_client_id(sm):
    """4-arg hooks see the requesting client_id.

    Phase 2 task 2.3 needs this so the IPC AppArmor pre-init hook can
    look up ``ClientConfigRequest.apparmor`` from
    ``_client_config[client_id]``.  ``_client_to_session`` isn't
    populated yet at pre-init, so the hook MUST receive client_id as
    a parameter.
    """
    captured: List[Any] = []

    def hook(server, session_id, workspace_path, client_id):
        captured.append(client_id)

    sm.add_pre_initialize_hook(hook)
    sm._run_pre_initialize_hooks(
        object(), "sess-5", "/tmp/ws", client_id="client-A",
    )
    assert captured == ["client-A"]


def test_pre_init_hook_legacy_3arg_still_works(sm):
    """Hooks with the legacy 3-arg signature keep working — the
    SessionManager introspects the callable's parameter count and
    drops client_id when calling old-style hooks.

    Backwards-compat is required because the WS server's apparmor
    pre-init hook (``server/websocket.py:_apparmor_pre_init_hook``)
    still uses the 3-arg signature; converting it to 4-arg is
    Phase 3 work alongside the WS-side runner spawn.
    """
    captured: List[tuple] = []

    def legacy_hook(server, session_id, workspace_path):
        captured.append((server, session_id, workspace_path))

    sm.add_pre_initialize_hook(legacy_hook)
    sm._run_pre_initialize_hooks(
        "S", "sess-6", "/tmp/ws", client_id="ignored-by-legacy",
    )
    # Legacy hook called with 3 args; client_id silently dropped.
    assert captured == [("S", "sess-6", "/tmp/ws")]


def test_pre_init_hooks_legacy_and_new_compose(sm):
    """Mix: legacy 3-arg + new 4-arg hooks both run cleanly."""
    seen_legacy: List[Any] = []
    seen_new: List[Any] = []

    def legacy(server, session_id, workspace_path):
        seen_legacy.append(workspace_path)

    def new(server, session_id, workspace_path, client_id):
        seen_new.append((workspace_path, client_id))

    sm.add_pre_initialize_hook(legacy)
    sm.add_pre_initialize_hook(new)
    sm._run_pre_initialize_hooks(
        object(), "sess-7", "/tmp/ws", client_id="C",
    )
    assert seen_legacy == ["/tmp/ws"]
    assert seen_new == [("/tmp/ws", "C")]


def test_pre_init_hook_client_id_default_none(sm):
    """Calling ``_run_pre_initialize_hooks`` without ``client_id``
    (e.g. ``_load_session_impl`` doesn't have a client) passes
    ``None`` to 4-arg hooks — they must handle the non-IPC case.
    """
    captured: List[Any] = []

    def hook(server, session_id, workspace_path, client_id):
        captured.append(client_id)

    sm.add_pre_initialize_hook(hook)
    # No client_id keyword — the disk-restore path doesn't pass one.
    sm._run_pre_initialize_hooks(object(), "sess-8", "/tmp/ws")
    assert captured == [None]
