"""``default_agent`` bound a subagent and not a top-level session (#1159).

A profile's ``default_agent:`` is resolved by ``spawn_subagent``
(``shared/plugins/subagent/plugin.py``) and was resolved **nowhere** on
the path a client takes.  ``grep -rn 'default_agent' jaato-server/server/``
returned only ``default_agent_id``, an unrelated tool-output-dispatch
identifier, so a session created as ``create_session(profile="organize")``
came up with no persona layer at all.

The sharp form is that BOTH bindings meet in one function and it resolved
one of them.  Measured against ``main`` @ ``54d4abef``::

    agent -> profile   default_profile   session_manager.py:8010   HONOURED
    profile -> agent   default_agent     --                        IGNORED

Twelve lines apart, in ``_create_session_impl``, for the same session.

It is silent at every layer, which is why it cost several full cascade
runs before anyone looked at a session record:

    jaato-scaffold validate   PASS  -- _check_default_agent_exists fires
                                       only when the persona is MISSING;
                                       it checks that the file resolves,
                                       never that anything reads it
    session log               --    -- never carries persona text, even
                                       when a persona IS loaded
    model output              plausible -- the trigger prompt and the
                                       completion schema alone shape a
                                       result

Only the persisted session record (which carries the rendered persona,
per #787) shows it, by containing none of the persona's text.

**The AST guard below is the load-bearing test.**  ``_agent_for_session``
is a pure function, so a test that imports it and calls it passes just as
happily on the broken tree — the defect was never that the resolution was
wrong, it was that nothing performed it.  That is #1133's lesson in this
file: *a test that imports ``_install_gc`` and calls it passes on the
broken tree, since the broken tree's defect was precisely that the
resolution existed and nothing invoked it.*  So the call site is asserted
structurally, and its ORDER is asserted too: resolving after the
``if agent_name:`` block would be a no-op.
"""

from __future__ import annotations

import ast
import pathlib

from server.session_manager import (
    _agent_for_session,
    _agent_not_found_error,
)
from shared.tests.reversion import Reversion

_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "session_manager.py"


class _Profile:
    """Minimal stand-in — the helpers read two attributes by ``getattr``."""

    def __init__(self, name="organize", default_agent=None):
        self.name = name
        self.default_agent = default_agent


def _impl_node() -> ast.FunctionDef:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_create_session_impl"):
            return node
    raise AssertionError("_create_session_impl not found in session_manager.py")


def _calls_named(node: ast.AST, name: str) -> list:
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == name]


class TestTheResolver:
    """What persona a session gets, and where the name came from."""

    def test_the_profile_supplies_its_persona_when_no_agent_is_named(self):
        name, from_profile = _agent_for_session(
            None, _Profile(default_agent="organize"))
        assert name == "organize"
        assert from_profile is True

    def test_an_explicit_agent_still_wins(self):
        # The profile's binding is a default, not a ceiling.
        name, from_profile = _agent_for_session(
            "specialist", _Profile(default_agent="organize"))
        assert name == "specialist"
        assert from_profile is False

    def test_a_profile_without_the_key_changes_nothing(self):
        assert _agent_for_session(None, _Profile()) == (None, False)

    def test_no_profile_at_all_is_not_an_error(self):
        # create_session is reachable with profile=None (a bare session
        # over the workspace .env); the resolver must not raise there.
        assert _agent_for_session(None, None) == (None, False)


class TestTheMessage:
    """A failure blames whoever actually got it wrong."""

    def test_a_missing_default_agent_blames_the_profile(self):
        msg = _agent_not_found_error(
            "organize", _Profile(name="organize"), "organize", True)
        assert "Profile 'organize'" in msg
        assert "default_agent" in msg
        # The caller passed no agent= at all, so the message must not
        # send them looking for a bug in their own call.
        assert not msg.startswith("Agent ")

    def test_an_explicit_missing_agent_still_blames_the_agent(self):
        msg = _agent_not_found_error("ghost", _Profile(), "organize", False)
        assert msg.startswith("Agent 'ghost' not found")


class TestTheCallSite:
    """The half a unit test on a pure helper cannot see."""

    def test_the_impl_actually_resolves_the_profiles_persona(self):
        assert _calls_named(_impl_node(), "_agent_for_session"), (
            "_create_session_impl does not call _agent_for_session, so a "
            "profile's default_agent binds nothing on the client path — "
            "which is #1159 exactly"
        )

    def test_the_resolution_happens_before_the_agent_is_looked_up(self):
        impl = _impl_node()
        resolve = _calls_named(impl, "_agent_for_session")[0]

        lookup = [
            n for n in ast.walk(impl)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name)
            and n.test.id == "agent_name"
            and any(
                isinstance(c.func, ast.Attribute)
                and c.func.attr == "_resolve_agent"
                for c in ast.walk(n) if isinstance(c, ast.Call)
            )
        ]
        assert lookup, "the `if agent_name:` persona-resolution block moved"
        assert resolve.lineno < lookup[0].lineno, (
            "default_agent is resolved AFTER the block that reads "
            "agent_name, so it is a no-op"
        )

    def test_the_not_found_branch_can_blame_the_profile(self):
        assert _calls_named(_impl_node(), "_agent_not_found_error"), (
            "the AgentNotFoundError branch no longer distinguishes a "
            "persona the caller named from one the profile named"
        )


REVERSIONS = [
    Reversion(
        target="jaato-server/server/session_manager.py",
        find=(
            "        agent_name, agent_from_profile = _agent_for_session("
            "agent_name, profile)\n"
        ),
        replace="        agent_from_profile = False\n",
        test=(
            "TestTheCallSite::"
            "test_the_impl_actually_resolves_the_profiles_persona"
        ),
        because="#1159 itself — the resolver exists and the client path "
                "never calls it, so a top-level session created by profile "
                "name comes up with no persona layer at all, silently",
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find=(
            "    default_agent = getattr(profile, \"default_agent\", None) "
            "if profile else None\n"
        ),
        replace="    default_agent = None\n",
        test=(
            "TestTheResolver::"
            "test_the_profile_supplies_its_persona_when_no_agent_is_named"
        ),
        because="the resolver being called and answering None anyway, "
                "which is the same silent no-persona session with a "
                "call site that looks correct",
    ),
    Reversion(
        target="jaato-server/server/session_manager.py",
        find="    if from_profile:\n        pname = getattr(profile, \"name\", None)",
        replace="    if False:\n        pname = getattr(profile, \"name\", None)",
        test="TestTheMessage::test_a_missing_default_agent_blames_the_profile",
        because="a workspace configuration error reported as though the "
                "caller passed a bad agent= argument they never passed",
    ),
]
