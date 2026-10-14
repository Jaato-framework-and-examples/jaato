"""``tool_hat`` and ``//child`` carry ``attach_disconnected`` (#1662, v48).

Under a private ``/tmp`` (#1381) the runner unshares a mount namespace
before it confines.  A file reached through a mount of the old namespace
then has no name from the new root: the ``/dev/null`` the runner's stdio
was opened on, inherited by a ``cli`` command, and the bound path of a unix
socket whose listener is outside the namespace.  The kernel refuses such an
access with "Failed name lookup - disconnected path" (name ``dev/null``,
without its slash) unless the profile asks for ``attach_disconnected``,
whatever the rules grant.  Measured on an enforcing kernel: a ``//child``
python denied ``getattr`` on ``/dev/null``, and ``connect()`` to a
fragment-granted ``<ws>/relay.sock`` answered ``EACCES`` under
``private_tmp: true`` and worked under ``false``.

The base profile has had the flag since v19.  Flags are per profile, so
the sub-profiles never got it.

These tests read rendered profile text; CI has no AppArmor kernel.
"""

from __future__ import annotations

import re

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.shared.confinement_grants import profile_body_rules
from jaato_server.shared.tests.reversion import Reversion

_AA = "jaato-server/jaato_server/server/apparmor.py"
_GRANTS = "jaato-server/jaato_server/shared/confinement_grants.py"

REVERSIONS = [
    Reversion(
        target=_AA,
        find='            " flags=(attach_disconnected, complain)" if complain\n'
             '            else " flags=(attach_disconnected)"\n',
        replace='            " flags=(complain)" if complain\n            else ""\n',
        test="test_every_sub_profile_attaches_disconnected_paths",
        because="//child cannot name its own inherited /dev/null",
    ),
    Reversion(
        target=_GRANTS,
        find='(?: flags=\\([^)]*\\))? \\{{',
        replace=' \\{{',
        test="test_the_grant_record_reads_a_flagged_child_body",
        because="the #1348 grant record would read no //child rules at all",
    ),
]


def _render(tmp_path, monkeypatch, *, complain: bool, private_tmp: bool) -> str:
    if complain:
        monkeypatch.setenv("JAATO_APPARMOR_COMPLAIN", "1")
    else:
        monkeypatch.delenv("JAATO_APPARMOR_COMPLAIN", raising=False)
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    manager = AppArmorManager(workspace_root=str(tmp_path))
    return manager._render_profile(
        "sid", str(ws),
        private_tmp_dir=str(ws / ".tmp") if private_tmp else None,
    )


def _headers(text: str) -> dict:
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*profile (\S+)(?: flags=\(([^)]*)\))? \{$", line)
        if m:
            found[m.group(1)] = {f.strip() for f in (m.group(2) or "").split(",") if f.strip()}
    return found


def test_the_template_version_moved():
    assert AppArmorManager._TEMPLATE_VERSION >= 48


@pytest.mark.parametrize("complain", [False, True], ids=["enforce", "complain"])
@pytest.mark.parametrize("private_tmp", [False, True], ids=["host_tmp", "private_tmp"])
def test_every_sub_profile_attaches_disconnected_paths(
        tmp_path, monkeypatch, complain, private_tmp):
    headers = _headers(_render(tmp_path, monkeypatch,
                               complain=complain, private_tmp=private_tmp))
    assert {"tool_hat", "child"} <= set(headers), headers
    for name, flags in headers.items():
        assert "attach_disconnected" in flags, f"{name}: {flags}"
        assert ("complain" in flags) is complain, f"{name}: {flags}"


@pytest.mark.parametrize("complain", [False, True], ids=["enforce", "complain"])
def test_the_grant_record_reads_a_flagged_child_body(tmp_path, monkeypatch, complain):
    text = _render(tmp_path, monkeypatch, complain=complain, private_tmp=True)
    rules = profile_body_rules(text, "child")
    assert rules, "no //child rules read from the rendered profile"
    assert any(r.startswith("/tmp/**") for r in rules), rules
