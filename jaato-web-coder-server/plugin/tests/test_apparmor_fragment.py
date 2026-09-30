"""The shipped AppArmor fragment reaches every body a session's tools run in.

``apparmor/jaato-web-coder-toolchains.rules`` is installed into the daemon
account's user tier (``~/.jaato/apparmor-fragments/``).  Rendered through the
framework's own ``AppArmorManager._render_profile``, it must appear in the
base body, ``tool_hat`` and ``//child`` of an unscoped session, so no tool
the model drives can rewrite the offer, and a bound JDK can load its own
libraries.  No kernel here: this checks what
the kernel would be handed.
"""

import re
import shutil
from pathlib import Path

import pytest

FRAGMENT = Path(__file__).resolve().parents[1] / "apparmor" / "jaato-web-coder-toolchains.rules"
RULE = "audit deny /**/.jaato/toolchain-offer.json wlk,"
RULES = [
    RULE,
    "/**/.home/.local/share/mise/installs/**/bin/* ix,",
    "/**/.home/.local/share/mise/installs/**/*.so m,",
    "/**/.home/.local/share/mise/installs/**/*.so.* m,",
    "/**/.home/.local/share/mise/installs/java/**/lib/jspawnhelper ix,",
    "/**/.home/.local/share/mise/installs/go/*/pkg/tool/*/* ix,",
]


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
    assert lines == RULES


def _bodies(profile):
    """The base body (outside the hats) and each ``profile <name> {`` or ``hat <name> {`` sub-body, as text."""
    bodies, stack, base = {}, [], []
    for line in profile.splitlines():
        m = re.match(r"^\s+(?:profile|hat)\s+(\w+)\b[^{]*\{", line)
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
        for rule in RULES:
            assert rule in bodies[name], (name, rule)


def test_a_scoped_session_gets_it_only_when_it_names_it(monkeypatch, tmp_path):
    assert RULE not in _render(monkeypatch, tmp_path, requested=["something-else"])
    assert RULE in _render(monkeypatch, tmp_path, requested=["jaato-web-coder-toolchains"])
