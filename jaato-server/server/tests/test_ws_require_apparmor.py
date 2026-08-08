"""Tests for the require-AppArmor opt-in gate on the WS server.

Covers the constructor-level wiring that feeds the fail-closed gate in
``JaatoWSServer.start()``:

- ``JAATO_REQUIRE_APPARMOR=1`` promotes auto (``None``) mode to required
  (``True``) so startup fails closed instead of degrading silently.
- An explicit ``apparmor=False`` is *not* silently overridden by the env
  var; the contradiction is preserved for ``start()`` to reject.
- The ``_env_flag`` truthy spelling matches the framework convention.

The ``start()`` raise paths themselves are documented inline in
websocket.py; here we lock the wiring that determines required-ness.
"""

from __future__ import annotations

import pytest

from server.websocket import JaatoWSServer, _env_flag


class TestEnvFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes"])
    def test_truthy_values(self, value, monkeypatch):
        monkeypatch.setenv("JAATO_REQUIRE_APPARMOR", value)
        assert _env_flag("JAATO_REQUIRE_APPARMOR") is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
    def test_falsey_values(self, value, monkeypatch):
        monkeypatch.setenv("JAATO_REQUIRE_APPARMOR", value)
        assert _env_flag("JAATO_REQUIRE_APPARMOR") is False

    def test_absent_is_false(self, monkeypatch):
        monkeypatch.delenv("JAATO_REQUIRE_APPARMOR", raising=False)
        assert _env_flag("JAATO_REQUIRE_APPARMOR") is False


class TestRequireApparmorWiring:
    def test_env_promotes_auto_to_required(self, monkeypatch):
        monkeypatch.setenv("JAATO_REQUIRE_APPARMOR", "1")
        srv = JaatoWSServer(workspace_root="/tmp/ws", apparmor=None)
        assert srv._apparmor_required_env is True
        assert srv._apparmor_mode is True

    def test_env_does_not_override_explicit_disable(self, monkeypatch):
        # Contradiction is preserved (not silently flipped) so start()
        # can surface it as a config error.
        monkeypatch.setenv("JAATO_REQUIRE_APPARMOR", "1")
        srv = JaatoWSServer(workspace_root="/tmp/ws", apparmor=False)
        assert srv._apparmor_required_env is True
        assert srv._apparmor_mode is False

    def test_explicit_required_without_env(self, monkeypatch):
        monkeypatch.delenv("JAATO_REQUIRE_APPARMOR", raising=False)
        srv = JaatoWSServer(workspace_root="/tmp/ws", apparmor=True)
        assert srv._apparmor_required_env is False
        assert srv._apparmor_mode is True

    def test_auto_mode_unchanged_without_env(self, monkeypatch):
        monkeypatch.delenv("JAATO_REQUIRE_APPARMOR", raising=False)
        srv = JaatoWSServer(workspace_root="/tmp/ws", apparmor=None)
        assert srv._apparmor_required_env is False
        assert srv._apparmor_mode is None
