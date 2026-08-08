"""Tests for ``JaatoServer.create_registry_and_discover`` (PR-148).

Structural fix for the apparmor-composer-runs-before-registry-exists
bug discovered v126/v128 (kb-enablement-2.0 transform-step
PermissionError on ``<config_root>/sessions/...`` backups).

The per-session AppArmor profile is composed in
``SessionManager._provision_ipc_apparmor_and_spawn_runner`` BEFORE
``server.initialize()`` runs.  Pre-PR-148, the registry was created
inside ``initialize()`` — so when the composer queried
``registry.get_plugin(name)`` for each plugin's ``get_apparmor_rules``
contribution, ``server.registry`` was ``None`` and the entire
contribution loop was silently skipped.  ZERO plugin rules landed in
the rendered profile for ANY plugin.

PR-148 extracts the registry-creation + discover steps into a
separate ``create_registry_and_discover`` method callable BEFORE the
apparmor provisioning step.  ``initialize()`` is idempotent: when
the registry already exists, the registry-setup step is a no-op.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from server.core import JaatoServer


@pytest.fixture
def server(tmp_path):
    """Minimal JaatoServer for unit tests of registry lifecycle."""
    return JaatoServer(
        workspace_path=str(tmp_path),
        session_id="test-session",
    )


class TestCreateRegistryAndDiscover:

    def test_creates_registry_on_first_call(self, server):
        """Pin: ``server.registry`` starts as None, becomes a
        PluginRegistry instance after the call."""
        assert server.registry is None
        server.create_registry_and_discover()
        assert server.registry is not None
        # Plugins should have been discovered (non-empty _plugins)
        assert len(server.registry._plugins) > 0

    def test_idempotent_when_registry_already_populated(self, server):
        """Pin: calling twice doesn't re-create the registry or
        re-run discover.  Required because ``initialize()`` ALSO
        calls the equivalent setup steps; with PR-148 the second
        call must skip them."""
        server.create_registry_and_discover()
        first_registry = server.registry
        first_plugin_count = len(server.registry._plugins)

        # Second call — registry should be the same instance.
        server.create_registry_and_discover()
        assert server.registry is first_registry
        assert len(server.registry._plugins) == first_plugin_count

    def test_pre_populates_workspace_path(self, server, tmp_path):
        """Pin: registry's ``_workspace_path`` is set so the apparmor
        composer + plugin.initialize call sites both see the value."""
        server.create_registry_and_discover()
        assert server.registry._workspace_path == str(tmp_path)

    def test_pre_populates_session_id(self, server):
        """Pin: registry's ``_session_id`` is set (per PR-146 framework-
        key injection contract)."""
        server.create_registry_and_discover()
        assert server.registry._session_id == "test-session"

    def test_pre_populates_config_root_when_set(self, tmp_path):
        """Pin: when server has ``_config_root`` set, registry inherits it."""
        config_root = tmp_path / "kb" / ".jaato"
        config_root.mkdir(parents=True)
        server = JaatoServer(
            workspace_path=str(tmp_path),
            session_id="s",
        )
        server.config_root = str(config_root)  # use the public setter
        # The public setter (line 547) calls registry.set_config_root
        # if registry exists — but here it doesn't yet.  Just set the
        # attribute directly to test the create_registry path.
        server._config_root = str(config_root)

        server.create_registry_and_discover()
        assert server.registry._config_root == str(config_root)

    def test_can_call_before_or_after_workspace_path_set(self, tmp_path):
        """Pin: callable from either path order:
        (a) workspace set on server.__init__ → create_registry_and_discover
        (b) server created bare → workspace set later → create_registry_and_discover
        Both should result in registry knowing workspace_path."""
        # Path (a) — workspace at __init__
        s_a = JaatoServer(workspace_path=str(tmp_path), session_id="a")
        s_a.create_registry_and_discover()
        assert s_a.registry._workspace_path == str(tmp_path)

        # Path (b) — workspace set later
        s_b = JaatoServer(workspace_path=None, session_id="b")
        s_b._workspace_path = str(tmp_path)
        s_b.create_registry_and_discover()
        assert s_b.registry._workspace_path == str(tmp_path)
