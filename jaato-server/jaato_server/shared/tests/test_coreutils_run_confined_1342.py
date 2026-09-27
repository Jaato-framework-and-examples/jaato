"""Coreutils and script commands run in a confined session (#1342).

Ubuntu 25.10+ ships Rust coreutils: ``/usr/bin/ls`` is a symlink into
``/usr/lib/cargo/bin/coreutils/``, and AppArmor checks the RESOLVED path.
``/usr/bin/** ix`` therefore does not cover it, and ``/usr/lib/** rm``
cannot exec, so on those hosts no coreutils command ran in any confined
session.  Kernel log from the reporting host:

    apparmor="DENIED" operation="exec" profile="…//child"
        name="/usr/lib/cargo/bin/coreutils/ls" requested_mask="x"

Script commands fail for a second reason: ``/usr/bin/which`` resolves to
``which.debianutils``, a shell script its interpreter must OPEN, and no
body granted ``r`` on ``/usr/bin/**``:

    apparmor="DENIED" operation="open" name="/usr/bin/which.debianutils"
        requested_mask="r"

Template v38 adds ``/usr/lib/cargo/bin/** ix`` and makes the in-PATH
directories ``r`` beside ``ix`` in base, ``tool_hat`` and a non-scoping ``//child``.
A scoped ``//child`` keeps fragment-only exec (v18).

These tests read rendered profile text; CI has no AppArmor kernel.  Where
``apparmor_parser`` is installed they also compile the render, which
catches a conflicting exec mode the text checks cannot see.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.shared.tests.reversion import Reversion

_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_AA,
        find='                "    /usr/lib/cargo/bin/**    ix,"',
        replace='                "    # (no cargo exec)"',
        test="test_every_broad_body_can_exec_rust_coreutils",
        because="a non-scoping //child cannot exec /usr/lib/cargo/bin/coreutils/ls",
    ),
    Reversion(
        target=_AA,
        find="  # Script commands (v38, #1342): the interpreter must OPEN the script.\n  /usr/bin/**          r,\n",
        replace="  # Script commands (v38, #1342): the interpreter must OPEN the script.\n",
        test="test_every_broad_body_can_read_script_commands",
        because="the base profile cannot open a script command like which.debianutils",
    ),
]

def _bodies(root: Path, requested_fragments: Optional[List[str]]) -> Dict[str, str]:
    """Render a profile for a workspace under *root* and split its bodies.

    The workspace name keeps a space, the #1305 shape.  It lives under a
    test directory, never ``/root``: CI does not run as root, and the
    renderer stats the workspace's fragment directory.
    """
    workspace = root / "Test env"
    workspace.mkdir(parents=True, exist_ok=True)
    manager = AppArmorManager(workspace_root=str(root))
    text = manager._render_profile(
        "sid", str(workspace), requested_fragments=requested_fragments,
        plugin_rules=['"/usr/bin/python3.14" ix,'],
    )
    hat = text.index("profile tool_hat {")
    child = text.index("profile child {")
    return {
        "base": text[:hat],
        "tool_hat": text[hat:child],
        "child": text[child:],
        "_all": text,
    }


def _grants(body: str) -> Dict[str, bool]:
    lines = {" ".join(line.split()) for line in body.splitlines()}
    return {
        "cargo": "/usr/lib/cargo/bin/** ix," in lines,
        "usr_bin_r": "/usr/bin/** r," in lines,
        "usr_local_bin_r": "/usr/local/bin/** r," in lines,
        "bin_r": "/bin/** r," in lines,
    }


def test_the_template_version_moved():
    assert AppArmorManager._TEMPLATE_VERSION >= 38


def test_every_broad_body_can_exec_rust_coreutils(tmp_path):
    bodies = _bodies(tmp_path, None)
    for name in ("base", "tool_hat", "child"):
        assert _grants(bodies[name])["cargo"], f"{name} cannot exec rust coreutils"


def test_every_broad_body_can_read_script_commands(tmp_path):
    bodies = _bodies(tmp_path, None)
    for name in ("base", "tool_hat", "child"):
        g = _grants(bodies[name])
        assert g["usr_bin_r"] and g["usr_local_bin_r"] and g["bin_r"], (
            f"{name} cannot read a script command: {g}"
        )


def test_a_scoped_child_keeps_fragment_only_exec(tmp_path):
    bodies = _bodies(tmp_path, [])
    assert not any(_grants(bodies["child"]).values()), (
        "a scoped //child must get no broad exec or read (v18)"
    )
    # The parent bodies are not scoped and keep the grants.
    assert _grants(bodies["base"])["cargo"]
    assert _grants(bodies["tool_hat"])["cargo"]


@pytest.mark.skipif(shutil.which("apparmor_parser") is None,
                    reason="apparmor_parser not installed")
@pytest.mark.parametrize("fragments", [None, []], ids=["unscoped", "scoped"])
def test_the_render_compiles(tmp_path: Path, fragments):
    profile = tmp_path / "profile"
    profile.write_text(_bodies(tmp_path / "ws", fragments)["_all"])
    result = subprocess.run(
        ["apparmor_parser", "-Q", "-K", str(profile)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
