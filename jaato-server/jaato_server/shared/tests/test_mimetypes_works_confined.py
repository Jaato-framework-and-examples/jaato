"""``mimetypes`` works in a confined process (template v42).

``mimetypes.init()`` stats ``/etc/mime.types`` (allowed) and then opens it.
No body granted the open, so every ``mimetypes.guess_type()`` in a confined
process raised ``PermissionError: '/etc/mime.types'``: the model's own code,
and jaato's runner-side attachment handling, which guesses a file's mime
from its name.  Found by an environment assessment run in a web-coder
workspace, where the SDK's own clarification-attachment test failed on it.

Every body a confined process can run under gets ``/etc/mime.types r``:
base, ``tool_hat``, ``//child`` (scoped or not) and the isolated sub-runner.

These tests read rendered profile text; CI has no AppArmor kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from jaato_server.server.apparmor import AppArmorManager
from jaato_server.shared.tests.reversion import Reversion

_AA = "jaato-server/jaato_server/server/apparmor.py"

REVERSIONS = [
    Reversion(
        target=_AA,
        find='  # v42: mimetypes.init() opens it (see the base body).\n  /etc/mime.types r,\n',
        replace='',
        test="test_every_body_can_read_the_mime_table",
        because="the isolated sub-runner cannot open /etc/mime.types",
    ),
    Reversion(
        target=_AA,
        find="  /etc/nsswitch.conf   r,\n  # v42: the stdlib's mimetypes.init() opens it; without it every\n"
             "  # mimetypes.guess_type() in a confined process raises EACCES.\n  /etc/mime.types      r,\n",
        replace="  /etc/nsswitch.conf   r,\n",
        test="test_every_body_can_read_the_mime_table",
        because="the base profile cannot open /etc/mime.types",
    ),
]


def _bodies(root: Path, requested_fragments: Optional[List[str]]) -> Dict[str, str]:
    workspace = root / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    manager = AppArmorManager(workspace_root=str(root))
    text = manager._render_profile(
        "sid", str(workspace), requested_fragments=requested_fragments,
    )
    hat = text.index("profile tool_hat {")
    child = text.index("profile child {")
    return {
        "base": text[:hat],
        "tool_hat": text[hat:child],
        "child": text[child:],
        "isolated_sub_runner": manager._render_sub_profile(
            parent_session_id="parent-A",
            subagent_id="agent-B",
            workspace_path=str(workspace),
        ),
    }


def _grants_mime_table(body: str) -> bool:
    return "/etc/mime.types r," in {" ".join(l.split()) for l in body.splitlines()}


def test_the_template_version_moved():
    assert AppArmorManager._TEMPLATE_VERSION >= 42


@pytest.mark.parametrize("fragments", [None, []], ids=["unscoped", "scoped"])
def test_every_body_can_read_the_mime_table(tmp_path, fragments):
    for name, body in _bodies(tmp_path, fragments).items():
        assert _grants_mime_table(body), f"{name} cannot open /etc/mime.types"
