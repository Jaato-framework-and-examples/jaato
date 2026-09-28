"""Template v40 (#1385): the cache fragment tier is write-denied.

``_render_profile`` composes ``*.rules`` from three tiers: user
(``~/.jaato/apparmor-fragments/``), workspace
(``<ws>/.jaato/apparmor-fragments/``) and cache
(``<ws>/.jaato/.cache/apparmor-fragments/``).  The cache tier wins a
basename collision, and an unscoped session composes every fragment it
finds.  Every body write-denied the workspace tier and none named the
cache tier, while the workspace-wide ``rwkl`` grant covered
``.jaato/.cache/``.  So a confined session could write a fragment the next
provisioning on that workspace composes: allow rules for itself, or a file
named like a user-tier fragment that shadows it and drops its denies.

These tests assert the rendered profile TEXT, in each of the four bodies a
confined process can run under.  AppArmor sub-profiles do not inherit the
base's rules, so a deny in the base alone would leave ``tool_hat`` and
``//child`` open.  Nothing here was run against an enforcing kernel.
"""

import os
import re
import sys
import types

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
        find='  # walker that fills it runs outside the confined runner; the\n'
             '  # workspace-wide rwkl grant would otherwise cover it.\n'
             '  audit deny "{workspace_path}/.jaato/.cache/apparmor-fragments/**" wlk,\n',
        replace='  # walker that fills it runs outside the confined runner; the\n'
                '  # workspace-wide rwkl grant would otherwise cover it.\n',
        test="test_every_body_denies_writes_to_the_cache_tier",
        because="the base profile body stops denying writes to the cache "
                "fragment tier, so a confined runner can author its next "
                "session's profile",
    ),
    Reversion(
        target=_APPARMOR,
        find='    # rules for the next session\'s profile).\n'
             '    audit deny "{workspace_path}/.jaato/apparmor-fragments/**" wlk,\n'
             '    # Cache-tier fragments (#1385) — mirrors base: the tier that\n'
             '    # wins a basename collision, so a planted file could also\n'
             '    # shadow a user-tier fragment and drop its denies.\n'
             '    audit deny "{workspace_path}/.jaato/.cache/apparmor-fragments/**" wlk,\n',
        replace='    # rules for the next session\'s profile).\n'
                '    audit deny "{workspace_path}/.jaato/apparmor-fragments/**" wlk,\n',
        test="test_every_body_denies_writes_to_the_cache_tier",
        because="the //child body (every model-driven subprocess) loses the "
                "cache-tier deny while base keeps it; sub-profiles do not "
                "inherit base rules",
    ),
]

_CACHE_DENY = re.compile(
    r'audit deny "/workspace/\.jaato/\.cache/apparmor-fragments/\*\*" wlk,')


@pytest.fixture
def manager(tmp_path):
    workspace_root = tmp_path / "workspaces"
    (workspace_root / "sessions").mkdir(parents=True)
    profile_dir = tmp_path / "apparmor_profiles"
    profile_dir.mkdir()
    return AppArmorManager(
        workspace_root=str(workspace_root),
        venv_path="/usr/local/venv",
        profile_dir=str(profile_dir),
    )


def _brace_body(text: str, anchor: str) -> str:
    """The text inside the ``{...}`` that follows ``anchor``."""
    start = text.find(anchor)
    assert start != -1, f"anchor {anchor!r} not in profile"
    open_at = text.find("{", start)
    depth = 0
    for i, ch in enumerate(text[open_at:], open_at):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:i]
    raise AssertionError(f"unmatched brace after {anchor!r}")


def _bodies(manager) -> dict:
    """The four bodies, each WITHOUT the sub-profiles nested in it, so a
    rule present only in ``tool_hat`` cannot satisfy the base check."""
    profile = manager._render_profile("s1", "/workspace")
    tool_hat = _brace_body(profile, "profile tool_hat")
    child = _brace_body(profile, "profile child")
    base = profile.replace(tool_hat, "").replace(child, "")
    sub = manager._render_sub_profile(
        parent_session_id="parent-A",
        subagent_id="agent-B",
        workspace_path="/workspace",
    )
    return {"base": base, "tool_hat": tool_hat, "child": child,
            "isolated_sub_runner": sub}


def test_every_body_denies_writes_to_the_cache_tier(manager):
    missing = [name for name, body in _bodies(manager).items()
               if not _CACHE_DENY.search(body)]
    assert not missing, (
        f"no write-deny on <ws>/.jaato/.cache/apparmor-fragments/ in: "
        f"{missing}.  That tier is composed into the next profile and wins "
        f"a basename collision (#1385).")


def test_the_bodies_are_really_separated(manager):
    """The check above is only per-body if the base text no longer holds
    the sub-profiles' rules."""
    bodies = _bodies(manager)
    assert "profile tool_hat" in bodies["base"]
    assert len(_CACHE_DENY.findall(bodies["base"])) == 1


def test_the_search_tiers_still_include_the_cache_tier():
    """The deny is only meaningful while discovery reads that directory;
    if the tier moves, the deny must move with it."""
    src = open(_apparmor_mod.__file__, encoding="utf-8").read()
    assert '("cache", (Path(workspace_path) / ".jaato" / ".cache" / ' \
           '"apparmor-fragments")' in src
