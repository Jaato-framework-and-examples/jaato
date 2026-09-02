"""``configure()`` freezes the prompt it rendered, prefetch output included.

Issue #787.  A revived session restores this snapshot instead of rebuilding
its prompt, and rebuilding is what re-ran the persona's ``{{!py:...}}``
prefetch against ``agent_params`` that were never persisted — so a session
with a mandatory prefetch could be created and run but never woken.

Two properties make the snapshot usable as that restore artifact, and both
are easy to break by moving one line:

* it is taken AFTER placeholder expansion, so it carries what the script
  produced rather than the directive that produced it.  A snapshot taken
  before expansion would restore a prompt containing ``{{!py:...}}``, which
  the revived session would then expand — re-running the script, i.e. the
  bug, with an extra step;
* it is taken at configure time and never updated, so the runtime additions
  that follow (a plugin's deferred instructions when one of its tools first
  activates, a pinned reference's content) are NOT in it.  Those are
  re-produced by the revived session itself, so a snapshot that included
  them would double them once per revive.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from shared.jaato_session import JaatoSession


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace whose prefetch script REQUIRES an agent_param.

    Shaped after the script in the issue: it does not merely read the
    params, it refuses to render without them — which is what turned a
    missing dict into an aborted session-prep.
    """
    scripts = tmp_path / ".jaato" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "checkout_worktree.py").write_text(textwrap.dedent("""\
        def render(context, args):
            params = context.agent_params or {}
            missing = [k for k in ("repo", "issue_id") if k not in params]
            if missing:
                raise RuntimeError(
                    "input.agent_params must carry both 'repo' and "
                    "'issue_id' - the task.yaml for this arm is missing "
                    "one of them"
                )
            return "worktree at /w/%s-%s" % (params["repo"], params["issue_id"])
    """))
    return tmp_path


def _session(workspace: Path, assembled: str) -> JaatoSession:
    runtime = MagicMock()
    runtime.registry = None
    runtime.reliability_plugin = None
    runtime._config_root = None
    runtime.get_system_instructions.return_value = assembled
    session = JaatoSession(runtime, model="dummy")
    session._workspace_path = str(workspace)
    session._config_root = None
    return session


def test_the_snapshot_carries_the_prefetch_output_not_the_directive(workspace):
    session = _session(workspace, "PERSONA\n{{!py:scripts/checkout_worktree.py}}")
    session._agent_params = {"repo": "jaato", "issue_id": "787"}

    session.configure(skip_provider=True)

    rendered = session.get_rendered_system_instruction()
    assert rendered == "PERSONA\nworktree at /w/jaato-787"
    assert "{{!py:" not in rendered, (
        "the snapshot was taken before expansion; restoring it would make "
        "the revived session expand the directive again — the very re-run "
        "this change removes"
    )


def test_the_snapshot_does_not_follow_later_prompt_growth(workspace):
    """The live prompt keeps growing; the snapshot must not.

    ``_system_instruction`` is appended to at runtime — deferred plugin
    instructions when a tool first activates, pinned reference content.
    The revived session re-produces those for itself, so a snapshot that
    tracked them would add them a second time on every revive.
    """
    session = _session(workspace, "PERSONA")
    session.configure(skip_provider=True)
    frozen = session.get_rendered_system_instruction()

    session._system_instruction += "\n\n<deferred plugin instructions>"

    assert session.get_rendered_system_instruction() == frozen
    assert session.get_system_instruction() != frozen, (
        "the live accessor stopped tracking the live value — the two "
        "accessors have to differ or the snapshot proves nothing"
    )


def test_a_restored_override_becomes_the_snapshot(workspace):
    """The revive path is a fixed point: restore in, same value out.

    A revived session is configured with the persisted prompt as
    ``system_instruction_override``.  Its own snapshot must then be that
    same prompt, so the next save re-persists the original rather than
    something derived from it.
    """
    restored = "PERSONA\nworktree at /w/jaato-787"
    session = _session(workspace, "SOMETHING ELSE ENTIRELY")

    session.configure(skip_provider=True, system_instruction_override=restored)

    assert session.get_rendered_system_instruction() == restored
    assert session._system_instruction == restored


def test_nothing_is_snapshotted_before_configure_runs(workspace):
    session = _session(workspace, "PERSONA")
    assert session.get_rendered_system_instruction() is None, (
        "an unconfigured session has rendered nothing; reporting a value "
        "would make the daemon persist a prompt that was never used"
    )


def test_the_original_failure_still_fails_without_params(workspace):
    """Guard the guard: the prefetch really is mandatory.

    If this stopped raising, every test above would pass for the wrong
    reason — there would be no re-run worth avoiding.
    """
    from shared.dynamic_instructions import DynamicInstructionsError

    session = _session(workspace, "PERSONA\n{{!py:scripts/checkout_worktree.py}}")
    session._agent_params = {}

    with pytest.raises(DynamicInstructionsError) as exc:
        session.configure(skip_provider=True)
    assert "agent_params" in str(exc.value)
