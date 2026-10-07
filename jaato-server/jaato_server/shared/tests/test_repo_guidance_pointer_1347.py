"""A repository's own guidance is pointed at, never copied (#1347).

When the workspace ROOT holds ``AGENTS.md``, ``CLAUDE.md``,
``CONTRIBUTING.md``, ``.github/copilot-instructions.md`` or ``.cursor/rules``,
the system instruction gains one line naming them.  The properties pinned
here, each attached to a way it could silently stop holding:

A. the line names what is present, in a fixed order, and nothing when
   nothing is present -- root only, the tree is never walked;
B. the line reaches the prompt the session actually renders
   (``JaatoRuntime.get_system_instructions``, the call ``configure()``
   makes), and the files' CONTENTS never do;
C. it belongs to the ``disk`` piece: ``suppress_base_instructions: true``
   or ``{disk: true}`` drops it;
D. a name the web coder's managed ``30-repo-guidance.md`` already lists is
   not pointed at twice.

The rendered prompt is persisted for revive (#787), so a revived session
keeps the line it was rendered with; nothing here needs revive handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jaato_server.shared import jaato_runtime as _runtime_module
from jaato_server.shared.instruction_suppression import PIECE_DISK, normalize_suppression
from jaato_server.shared.repo_guidance import repo_guidance_pointer
from jaato_server.shared.tests.reversion import Reversion

_GUIDANCE = "jaato-server/jaato_server/shared/repo_guidance.py"
_RUNTIME = "jaato-server/jaato_server/shared/jaato_runtime.py"

SENTINEL = "SENTINEL-AGENTS-CONTENT-7f3a"

REVERSIONS = [
    Reversion(
        target=_GUIDANCE,
        find="            elif path.is_file():\n                found.append(name)\n",
        replace=(
            "            elif path.is_file():\n"
            "                found.append(name + ': ' + path.read_text())\n"
        ),
        because="the repository's text must never enter the trusted system prompt",
        test="test_the_pointer_reaches_the_rendered_prompt_and_the_contents_do_not",
    ),
    Reversion(
        target=_RUNTIME,
        find="            self._append_repo_guidance_pointer()\n",
        replace="",
        because="a pointer computed and never appended points at nothing",
        test="test_the_pointer_reaches_the_rendered_prompt_and_the_contents_do_not",
    ),
    Reversion(
        target=_RUNTIME,
        find="        if include_base:\n            base = self.get_base_system_instructions()\n",
        replace="        if True:\n            base = self.get_base_system_instructions()\n",
        because="the pointer is part of the disk piece; suppressing disk must drop it",
        test="test_disk_suppression_drops_the_pointer",
    ),
    Reversion(
        target=_GUIDANCE,
        find='    names = [n for n in present if n.rstrip("/") not in skipped and n not in skipped]\n',
        replace="    names = list(present)\n",
        because="a root file the managed file already lists would be pointed at twice",
        test="test_a_name_the_managed_file_lists_is_skipped",
    ),
]


def _write(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _managed(ws: Path, *paths: str) -> None:
    body = "\n".join(f"- `{p}`" for p in paths)
    _write(
        ws / ".jaato" / "instructions" / "30-repo-guidance.md",
        "<!-- jaato-managed: repo-guidance v1 — delete this line to keep your own edits -->\n"
        "This workspace contains repositories with their own agent guidance.\n"
        f"{body}\n",
    )


# ------------------------------------------------------------ A. the function

def test_nothing_present_yields_nothing(tmp_path):
    assert repo_guidance_pointer(tmp_path) is None


def test_a_missing_root_yields_nothing(tmp_path):
    assert repo_guidance_pointer(tmp_path / "absent") is None
    assert repo_guidance_pointer(None) is None


def test_names_are_in_the_fixed_order(tmp_path):
    _write(tmp_path / "CONTRIBUTING.md")
    _write(tmp_path / ".cursor" / "rules" / "a.mdc")
    _write(tmp_path / "AGENTS.md")
    _write(tmp_path / ".github" / "copilot-instructions.md")
    _write(tmp_path / "CLAUDE.md")
    line = repo_guidance_pointer(tmp_path)
    assert line is not None
    order = ["`AGENTS.md`", "`CLAUDE.md`", "`CONTRIBUTING.md`",
             "`.github/copilot-instructions.md`", "`.cursor/rules/`"]
    positions = [line.index(n) for n in order]
    assert positions == sorted(positions)
    assert "readFile" in line and "the relevant one" in line


def test_a_single_name_reads_naturally(tmp_path):
    _write(tmp_path / "AGENTS.md")
    line = repo_guidance_pointer(tmp_path)
    assert line == (
        "This repository's own guidance is in `AGENTS.md` (at the workspace "
        "root); read it with readFile before working."
    )


def test_cursor_rules_as_a_file(tmp_path):
    _write(tmp_path / ".cursor" / "rules")
    assert "`.cursor/rules`" in repo_guidance_pointer(tmp_path)


def test_only_the_root_is_looked_at(tmp_path):
    _write(tmp_path / "sub" / "AGENTS.md")
    _write(tmp_path / "sub" / "deeper" / "CLAUDE.md")
    assert repo_guidance_pointer(tmp_path) is None


def test_a_directory_named_like_a_file_is_not_a_file(tmp_path):
    (tmp_path / "AGENTS.md").mkdir()
    assert repo_guidance_pointer(tmp_path) is None


# ---------------------------------------------------- D. no double pointer

def test_a_name_the_managed_file_lists_is_skipped(tmp_path):
    _write(tmp_path / "AGENTS.md")
    _write(tmp_path / "CONTRIBUTING.md")
    _managed(tmp_path, "AGENTS.md", "api-server/AGENTS.md")
    line = repo_guidance_pointer(tmp_path)
    assert line is not None
    assert "`AGENTS.md`" not in line
    assert "`CONTRIBUTING.md`" in line


def test_all_names_skipped_emits_nothing(tmp_path):
    _write(tmp_path / "AGENTS.md")
    _managed(tmp_path, "AGENTS.md")
    assert repo_guidance_pointer(tmp_path) is None


def test_a_subdirectory_path_does_not_skip_the_root_name(tmp_path):
    _write(tmp_path / "AGENTS.md")
    _managed(tmp_path, "api-server/AGENTS.md")
    assert "`AGENTS.md`" in repo_guidance_pointer(tmp_path)


def test_an_unmarked_instruction_file_does_not_skip(tmp_path):
    _write(tmp_path / "AGENTS.md")
    _write(tmp_path / ".jaato" / "instructions" / "10-mine.md", "Read `AGENTS.md`.\n")
    assert "`AGENTS.md`" in repo_guidance_pointer(tmp_path)


def test_a_marker_below_the_first_line_does_not_count(tmp_path):
    _write(tmp_path / "AGENTS.md")
    _write(
        tmp_path / ".jaato" / "instructions" / "30-repo-guidance.md",
        "my own edits\n<!-- jaato-managed: repo-guidance v1 -->\n- `AGENTS.md`\n",
    )
    assert "`AGENTS.md`" in repo_guidance_pointer(tmp_path)


def test_a_managed_file_in_an_explicit_dir_is_honoured(tmp_path):
    _write(tmp_path / "ws" / "AGENTS.md")
    _write(
        tmp_path / "cfg" / "instructions" / "30-repo-guidance.md",
        "<!-- jaato-managed: repo-guidance v1 -->\n- `AGENTS.md`\n",
    )
    assert repo_guidance_pointer(
        tmp_path / "ws", instructions_dirs=[tmp_path / "cfg" / "instructions"]
    ) is None


# ------------------------------------------------- B/C. the rendered prompt

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A workspace, with the user and premium instruction tiers emptied."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(_runtime_module, "_get_premium_content_path", lambda _k: None)
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _render(ws: Path, **kw) -> str:
    runtime = _runtime_module.JaatoRuntime(provider_name="anthropic", workspace_path=ws)
    return runtime.get_system_instructions(plugin_names=[], additional="PERSONA", **kw)


def test_the_pointer_reaches_the_rendered_prompt_and_the_contents_do_not(isolated):
    _write(isolated / "AGENTS.md", f"# Agents\n{SENTINEL}\n")
    _write(isolated / ".jaato" / "instructions" / "00-base.md", "BASE-LAYER\n")
    prompt = _render(isolated)
    assert "`AGENTS.md`" in prompt
    assert SENTINEL not in prompt
    assert prompt.index("BASE-LAYER") < prompt.index("`AGENTS.md`") < prompt.index("PERSONA")


def test_the_pointer_arrives_with_no_instruction_files(isolated):
    _write(isolated / "CLAUDE.md", SENTINEL)
    prompt = _render(isolated)
    assert "`CLAUDE.md`" in prompt
    assert SENTINEL not in prompt


def test_no_guidance_adds_nothing(isolated):
    assert "own guidance" not in _render(isolated)


def test_disk_suppression_drops_the_pointer(isolated):
    _write(isolated / "AGENTS.md")
    assert PIECE_DISK in normalize_suppression(True)
    assert PIECE_DISK in normalize_suppression({"disk": True})
    prompt = _render(isolated, include_base=False)
    assert "`AGENTS.md`" not in prompt
    assert "PERSONA" in prompt


def test_a_read_error_never_fails_the_assembly(isolated, monkeypatch):
    _write(isolated / "AGENTS.md")

    def boom(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr(
        "jaato_server.shared.repo_guidance.repo_guidance_pointer", boom
    )
    assert "PERSONA" in _render(isolated)
