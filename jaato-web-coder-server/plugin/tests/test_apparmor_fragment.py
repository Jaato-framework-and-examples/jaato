"""The shipped AppArmor fragment reaches every body a session's tools run in.

``apparmor/jaato-toolchain-offer.rules`` is installed into the daemon
account's user tier (``~/.jaato/apparmor-fragments/``).  Rendered through the
framework's own ``AppArmorManager._render_profile``, it must appear in the
base body, ``tool_hat`` and ``//child`` of an unscoped session, so no tool
the model drives can rewrite the offer.  No kernel here: this checks what
the kernel would be handed.
"""

import re
import shutil
from pathlib import Path

import pytest

FRAGMENT = Path(__file__).resolve().parents[1] / "apparmor" / "jaato-toolchain-offer.rules"
RULE = "audit deny /**/.jaato/toolchain-offer.json wlk,"


def _render(monkeypatch, tmp_path, requested=None):
    from jaato_server.server.apparmor import AppArmorManager

    home = tmp_path / "home"
    (home / ".jaato" / "apparmor-fragments").mkdir(parents=True, exist_ok=True)
    shutil.copy(FRAGMENT, home / ".jaato" / "apparmor-fragments" / FRAGMENT.name)
    monkeypatch.setenv("HOME", str(home))
    ws = tmp_path / "workspaces" / "alice-1"
    ws.mkdir(parents=True, exist_ok=True)
    mgr = AppArmorManager(workspace_root=str(tmp_path / "workspaces"))
    return mgr._render_profile("jaato-ws-test", str(ws), requested_fragments=requested)


def test_the_fragment_carries_the_deny():
    lines = [ln.strip() for ln in FRAGMENT.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert lines == [RULE]


def _bodies(profile):
    """The base body (outside the hats) and each ``profile <name> {`` sub-body, as text."""
    bodies, stack, base = {}, [], []
    for line in profile.splitlines():
        m = re.match(r"^\s+profile\s+(\w+)\s*\{", line)
        if m:
            stack.append(m.group(1)); bodies[m.group(1)] = []
            continue
        if stack and line.strip() == "}" and line.startswith("  ") and not line.startswith("    "):
            stack.pop()
            continue
        (bodies[stack[-1]] if stack else base).append(line)
    return {"base": "\n".join(base), **{k: "\n".join(v) for k, v in bodies.items()}}


def test_an_unscoped_session_denies_the_offer_in_every_body(monkeypatch, tmp_path):
    bodies = _bodies(_render(monkeypatch, tmp_path))
    assert set(bodies) >= {"base", "tool_hat", "child"}
    for name in ("base", "tool_hat", "child"):
        assert RULE in bodies[name], name


def test_a_scoped_session_gets_it_only_when_it_names_it(monkeypatch, tmp_path):
    assert RULE not in _render(monkeypatch, tmp_path, requested=["something-else"])
    assert RULE in _render(monkeypatch, tmp_path, requested=["jaato-toolchain-offer"])
