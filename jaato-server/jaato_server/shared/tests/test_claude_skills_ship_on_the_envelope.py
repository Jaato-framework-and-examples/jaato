"""Claude Code's ~/.claude/skills reaches a confined runner on the envelope.

AppArmor grants the prompt library a read of ``~/.claude/skills``; the
SELinux module does not (the directory keeps the home's type), and the
phase 2b kernel run logged 17 enforced ``search`` denials for it, with the
skills silently missing.  The daemon now ships the directory in the #1465
user-tier snapshot under ``@home/``, and the plugin reads it through
``user_tier.home_path``.
"""

import os
from pathlib import Path

import pytest

from jaato_server.shared import user_tier
from jaato_server.shared.tests.reversion import Reversion

_UT = "jaato-server/jaato_server/shared/user_tier.py"
_PL = "jaato-server/jaato_server/shared/plugins/prompt_library/plugin.py"

REVERSIONS = [
    Reversion(
        target=_UT,
        find='HOME_SHIPPED_DIRS: Tuple[str, ...] = (\n    ".claude/skills",\n',
        replace="HOME_SHIPPED_DIRS: Tuple[str, ...] = (\n",
        test="test_the_snapshot_carries_claude_skills",
        because="a confined runner's prompt library would silently miss the "
                "user's Claude Code skills under SELinux",
    ),
    Reversion(
        target=_UT,
        find="            if not stat.S_ISDIR(os.lstat(current).st_mode):\n                return False\n",
        replace="            if not os.path.isdir(current):\n                return False\n",
        test="test_a_symlinked_claude_dir_is_not_followed",
        because="a root daemon reading a dropped user's home would follow "
                "~/.claude -> anywhere and ship that",
    ),
    Reversion(
        target=_PL,
        find='                path=user_tier.home_path(".claude/skills"),\n',
        replace='                path=home / ".claude" / "skills",\n',
        test="test_the_prompt_library_reads_the_shipped_copy",
        because="the plugin would read the real ~/.claude/skills, which a "
                "confined SELinux runner cannot search",
    ),
]


def _skill(home: Path, name: str = "greet") -> Path:
    d = home / ".claude" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: greet\n---\nSay hello.\n")
    return d


@pytest.fixture(autouse=True)
def _no_installed_snapshot():
    yield
    user_tier.install(None, "/nonexistent", "x")


def test_the_snapshot_carries_claude_skills(tmp_path):
    home = tmp_path / "home"
    (home / ".jaato").mkdir(parents=True)
    _skill(home)
    snap = user_tier.collect(str(home / ".jaato"))
    assert snap["@home/.claude/skills/greet/SKILL.md"].endswith("Say hello.\n")


def test_claude_skills_ship_even_without_a_jaato_dir(tmp_path):
    home = tmp_path / "home"
    _skill(home)
    snap = user_tier.collect(str(home / ".jaato"))
    assert "@home/.claude/skills/greet/SKILL.md" in snap


def test_a_symlinked_claude_dir_is_not_followed(tmp_path):
    home = tmp_path / "home"
    (home / ".jaato").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    _skill(elsewhere)
    (home / ".claude").symlink_to(elsewhere / ".claude")
    snap = user_tier.collect(str(home / ".jaato"))
    assert not any(k.startswith("@home/") for k in snap)


def test_home_path_points_at_the_installed_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert user_tier.home_path(".claude/skills") == tmp_path / "home" / ".claude" / "skills"
    root = user_tier.install({"@home/.claude/skills/greet/SKILL.md": "x"}, str(tmp_path), "s1")
    assert user_tier.home_path(".claude/skills") == root / "@home" / ".claude" / "skills"
    assert (user_tier.home_path(".claude/skills") / "greet" / "SKILL.md").read_text() == "x"


def test_the_prompt_library_reads_the_shipped_copy(tmp_path, monkeypatch):
    from jaato_server.shared.plugins.prompt_library.plugin import PromptLibraryPlugin

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = user_tier.install({"@home/.claude/skills/greet/SKILL.md": "x"}, str(tmp_path), "s1")
    plugin = PromptLibraryPlugin()
    sources = {s.source_name: s.path for s in plugin._get_prompt_sources()}
    assert sources["claude-global"] == root / "@home" / ".claude" / "skills"
