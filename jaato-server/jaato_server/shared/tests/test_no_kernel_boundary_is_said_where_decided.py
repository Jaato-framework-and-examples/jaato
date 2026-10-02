""""No kernel boundary" is said by the code that decides it, not by a probe.

``AppArmorManager.is_available()`` used to log "workspace isolation falls
back to directory sandboxing only" at WARNING whenever AppArmor was absent.
Under ``JAATO_CONFINEMENT=auto`` the daemon asks AppArmor before SELinux, so
on every SELinux host that line was printed and then contradicted: the
phase 2b kernel run showed it above ``jaato-doctor``'s ``selinux — mode
enforcing`` PASS.  The probe now records its reason; the WARNING comes from
the daemon's backend selection and from the WS server's startup, each only
when it has actually settled on no kernel boundary.
"""

import logging
from unittest.mock import patch

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.server.websocket import JaatoWSServer
from jaato_server.shared.tests.reversion import Reversion

_AA = "jaato-server/jaato_server/server/apparmor.py"
_WS = "jaato-server/jaato_server/server/websocket.py"
_MAIN = "jaato-server/jaato_server/server/__main__.py"
_WS_GATE = '        elif getattr(self, "_selinux_backend", None) is None:\n'

REVERSIONS = [
    Reversion(
        target=_AA,
        find='            logger.info(\n                "AppArmor not available: %s",\n',
        replace='            logger.warning(\n                "AppArmor not available: %s",\n',
        test="test_the_probe_does_not_warn",
        because="every SELinux host would log a false 'falls back to "
                "directory sandboxing' line at startup and in the doctor",
    ),
    Reversion(
        target=_MAIN,
        find="        (logger.info if choice.backend is not None else logger.warning)(\n",
        replace="        (logger.info)(\n",
        test="test_the_daemon_warns_when_no_backend_is_available",
        because="a daemon with no kernel boundary would say so only at INFO",
    ),
    Reversion(
        target=_WS,
        find=_WS_GATE,
        replace="        elif False:\n",
        test="test_the_ws_server_warns_without_apparmor_or_selinux",
        because="a standalone WS server with no AppArmor would start "
                "unconfined without a WARNING",
    ),
    Reversion(
        target=_WS,
        find=_WS_GATE,
        replace="        else:\n",
        test="test_the_ws_server_is_quiet_when_selinux_confines",
        because="the daemon's WS server would contradict the SELinux "
                "backend it was handed",
    ),
]

_SANDBOX = "falls back to directory sandboxing"


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.fixture
def no_apparmor():
    with patch("jaato_server.server.apparmor.platform.system", return_value="Darwin"):
        yield


def test_the_probe_does_not_warn(no_apparmor, caplog, tmp_path):
    manager = AppArmorManager(workspace_root=str(tmp_path))
    with caplog.at_level(logging.DEBUG):
        assert manager.is_available() is False
    assert "Linux" in (manager.unavailable_reason or "")
    assert not _warnings(caplog)


def test_the_daemon_warns_when_no_backend_is_available(monkeypatch, caplog):
    from jaato_server.server.__main__ import JaatoDaemon

    monkeypatch.setenv("JAATO_CONFINEMENT", "none")
    monkeypatch.delenv("JAATO_REQUIRE_CONFINEMENT", raising=False)
    daemon = JaatoDaemon.__new__(JaatoDaemon)
    with caplog.at_level(logging.INFO):
        daemon._select_confinement_backend()
    assert any("kernel confinement: none" in r.getMessage()
               for r in _warnings(caplog))


def _ws(monkeypatch, tmp_path, selinux_backend):
    monkeypatch.delenv("JAATO_REQUIRE_APPARMOR", raising=False)
    srv = JaatoWSServer(workspace_root=str(tmp_path), apparmor=None)
    srv._selinux_backend = selinux_backend
    return srv


def test_the_ws_server_warns_without_apparmor_or_selinux(
        no_apparmor, monkeypatch, caplog, tmp_path):
    srv = _ws(monkeypatch, tmp_path, None)
    with caplog.at_level(logging.INFO):
        srv._init_apparmor(None)
    assert any(_SANDBOX in r.getMessage() for r in _warnings(caplog))


def test_the_ws_server_is_quiet_when_selinux_confines(
        no_apparmor, monkeypatch, caplog, tmp_path):
    srv = _ws(monkeypatch, tmp_path, object())
    with caplog.at_level(logging.INFO):
        srv._init_apparmor(None)
    assert not _warnings(caplog)


def test_required_apparmor_still_refuses_to_start(no_apparmor, monkeypatch, tmp_path):
    monkeypatch.delenv("JAATO_REQUIRE_APPARMOR", raising=False)
    srv = JaatoWSServer(workspace_root=str(tmp_path), apparmor=True)
    with pytest.raises(RuntimeError, match="Refusing to start unconfined"):
        srv._init_apparmor(None)
