"""#1686: every profile jaato renders pins an AppArmor ABI.

On kernel 7.0 with ``apparmor_parser`` 5.0.2, a profile with no ``abi``
line had its ``//child``'s ``deny network inet,`` enforced on the BASE
profile too: the runner was refused ``inet dgram``, lost DNS and could
not reach its model.  The same profiles with ``abi <abi/4.0>,`` or
``abi <abi/5.0>,`` kept the deny in the child.  Measured on that host
with throwaway profiles holding nothing jaato-specific.

These tests read the rendered TEXT (and compile it where
``apparmor_parser`` is installed).  No enforcing kernel runs here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from jaato_server.shared.tests.reversion import Reversion

if "jaato_server.server" not in sys.modules:
    _stub = types.ModuleType("jaato_server.server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["jaato_server.server"] = _stub

import jaato_server.server.apparmor as _apparmor_mod

AppArmorManager = _apparmor_mod.AppArmorManager

_APPARMOR = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_APPARMOR,
        find="            abi_rule=self._abi_rule,\n",
        replace="            abi_rule=\"\",\n",
        test="test_the_session_profile_pins_the_abi",
        because="the session profile compiles with no ABI, and a child-only "
                "deny network reaches the runner on kernel 7.0",
    ),
    Reversion(
        target=_APPARMOR,
        find="{self._abi_rule}#include <tunables/global>\n",
        replace="#include <tunables/global>\n",
        test="test_the_isolated_sub_profile_pins_the_abi",
        because="the isolated sub-runner's profile compiles with no ABI",
    ),
    Reversion(
        target=_APPARMOR,
        find='PINNED_ABIS = ("5.0", "4.0")\n',
        replace='PINNED_ABIS = ("4.0", "5.0")\n',
        test="test_the_newest_available_abi_wins",
        because="a host shipping abi/5.0 is pinned to the older 4.0",
    ),
]


def _abi_dir(tmp_path: Path, *versions: str) -> Path:
    d = tmp_path / "abi"
    d.mkdir(parents=True, exist_ok=True)
    for v in versions:
        (d / v).write_text("")
    return d


def _manager(tmp_path: Path, abi_dir: Path) -> AppArmorManager:
    root = tmp_path / "workspaces"
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    pdir = tmp_path / "profiles"
    pdir.mkdir(exist_ok=True)
    return AppArmorManager(
        workspace_root=str(root), venv_path=sys.prefix,
        profile_dir=str(pdir), abi_dir=str(abi_dir))


def _first_rule(text: str) -> str:
    """The first line that is neither blank nor a comment.

    ``#include`` is a rule, not a comment, so it counts.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") and not line.startswith("#include"):
            continue
        return line
    return ""


def test_the_newest_available_abi_wins(tmp_path):
    assert _apparmor_mod.pinned_abi_rule(str(_abi_dir(tmp_path, "3.0", "4.0", "5.0"))) \
        == "abi <abi/5.0>,\n"
    assert _apparmor_mod.pinned_abi_rule(str(_abi_dir(tmp_path / "x", "3.0", "4.0"))) \
        == "abi <abi/4.0>,\n"


def test_a_host_with_no_pinnable_abi_renders_as_before(tmp_path, caplog):
    abi = _abi_dir(tmp_path, "3.0")
    assert _apparmor_mod.pinned_abi_rule(str(abi)) == ""
    with caplog.at_level("WARNING"):
        m = _manager(tmp_path, abi)
    assert "#1686" in caplog.text
    text = m._render_profile("s1", str(tmp_path / "ws"), requested_fragments=[])
    assert "abi <" not in text


def test_the_session_profile_pins_the_abi(tmp_path):
    m = _manager(tmp_path, _abi_dir(tmp_path, "4.0", "5.0"))
    text = m._render_profile("s1", str(tmp_path / "ws"), requested_fragments=[])
    # An abi rule must come before every profile and include.
    assert _first_rule(text) == "abi <abi/5.0>,"


def test_the_isolated_sub_profile_pins_the_abi(tmp_path):
    m = _manager(tmp_path, _abi_dir(tmp_path, "4.0"))
    body = m._render_sub_profile("parent", "sub1", str(tmp_path / "ws"))
    assert _first_rule(body) == "abi <abi/4.0>,"


def test_the_abi_is_part_of_the_boundary(tmp_path):
    ws = str(tmp_path / "ws")
    four = _manager(tmp_path / "a", _abi_dir(tmp_path / "a", "4.0"))
    five = _manager(tmp_path / "b", _abi_dir(tmp_path / "b", "5.0"))
    assert four.confinement_id_for_boundary(ws, requested_fragments=[]) != \
        five.confinement_id_for_boundary(ws, requested_fragments=[])


@pytest.mark.skipif(shutil.which("apparmor_parser") is None,
                    reason="apparmor_parser not installed")
def test_the_pinned_render_compiles(tmp_path):
    real = Path(_apparmor_mod.DEFAULT_ABI_DIR)
    if not _apparmor_mod.pinned_abi_rule(str(real)):
        pytest.skip("this host's parser ships no pinnable ABI")
    ws = tmp_path / "ws"
    frags = ws / ".jaato" / "apparmor-fragments"
    frags.mkdir(parents=True)
    (frags / "no_network.rules").write_text("deny network inet,\ndeny network inet6,\n")
    m = _manager(tmp_path, real)
    bodies = {
        "base": m._render_profile("s1", str(ws), requested_fragments=[],
                                  child_fragments=["no_network"]),
        "isolated": m._render_sub_profile("parent", "sub1", str(ws)),
    }
    for key, body in bodies.items():
        profile = tmp_path / f"profile-{key}"
        profile.write_text(body)
        result = subprocess.run(
            ["apparmor_parser", "-Q", "-K", str(profile)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (key, result.stderr)
