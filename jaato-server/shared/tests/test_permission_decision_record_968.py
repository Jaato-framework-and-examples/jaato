"""Structural guards for the permission decision record (issue #968).

The behaviour lives in
``shared/plugins/permission/tests/test_decision_record_968.py``.  What is
here is the two properties that behaviour cannot protect, because both
fail *silently* rather than wrongly:

- **``reason=`` is written last.**  Every other field of a DECISION line
  is whitespace-free, so :func:`~shared.plugins.permission.plugin.parse_decision_trace`
  splits on ``" reason="`` and reads the remainder as a repr.  A field
  appended after it is swallowed into the reason string — the record
  still parses, still looks right, and has quietly lost a field.
- **The resolved hook has exactly one call site.**  ``_publish_decision``
  suppresses a duplicate event by asking whether a branch already fired
  the hook, and it can only know about a branch that went through
  ``_emit_resolved``.  A branch calling ``self._on_permission_resolved``
  directly would fire, not set the flag, and the opt-in would then emit a
  SECOND event for the same decision — which an auditor counting
  approvals reads as two approvals.

They live in ``shared/tests`` rather than beside the plugin because
``test_every_guard_detects_its_own_reversion`` discovers ``REVERSIONS``
only under ``jaato-server/shared/tests`` and ``jaato-server/server/tests``,
and a guard the meta-suite cannot find is a guard nobody proves works.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from shared.plugins.permission import plugin as permission_plugin_module
from shared.plugins.permission.plugin import PermissionPlugin
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


def _decision_trace_fstring() -> ast.JoinedStr:
    """The ``self._trace(...)`` argument that writes the DECISION line.

    Read structurally rather than by substring, so the guard answers
    "which field is last" rather than "is this token present anywhere".
    """
    source = textwrap.dedent(
        inspect.getsource(PermissionPlugin.check_permission))
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        arg = node.args[0]
        if not isinstance(arg, ast.JoinedStr):
            continue
        if any(isinstance(v, ast.Name) and v.id == "DECISION_TRACE_PREFIX"
               for v in ast.walk(arg)):
            return arg
    raise AssertionError(
        "check_permission no longer writes a DECISION_TRACE_PREFIX line — "
        "the record #968 ships has gone")


def test_reason_is_written_last() -> None:
    """``reason=`` must be the final field of the DECISION line.

    ``parse_decision_trace`` takes everything after ``" reason="`` as the
    reason, so a field written after it is not merely unread — its text
    is appended to the value of another field, and the record still
    parses and still looks right.

    Asserted on the f-string's own structure: the last literal segment
    must end with ``reason=``, and exactly one interpolation may follow
    it.  A substring check cannot express this — a field appended as a
    bare ``{helper(info)}`` contributes no literal to search for, which
    is precisely the reversion this guard is paired with.
    """
    values = _decision_trace_fstring().values
    assert len(values) >= 2, "the DECISION line lost its interpolations"
    literal, interpolation = values[-2], values[-1]
    assert isinstance(literal, ast.Constant), (
        "the DECISION line ends with two interpolations in a row; the last "
        "field must be `reason={...!r}` with nothing after it")
    assert isinstance(interpolation, ast.FormattedValue)
    assert literal.value.endswith("reason="), (
        "the DECISION line ends with %r, not with `reason=` + its value — "
        "a field written after reason= is folded into the reason string "
        "by parse_decision_trace" % (literal.value,))


def test_every_resolved_emission_goes_through_the_one_door() -> None:
    """The resolved hook is invoked from ``_emit_resolved`` and nowhere else.

    That method is what records "this decision already reached the event
    bus", which is the only thing standing between the ``#968`` opt-in
    and a doubled event.
    """
    source = inspect.getsource(permission_plugin_module.PermissionPlugin)
    direct = source.count("self._on_permission_resolved(")
    assert direct == 1, (
        "the resolved hook must be invoked only from _emit_resolved (which "
        "sets the already-announced flag _publish_decision reads); found "
        f"{direct} direct call site(s)")


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/permission/plugin.py",
        find='f"{_describe_permission_decider(info)}"\n'
             '            f" reason={info.get(\'reason\', \'\')!r}"',
        replace='f" reason={info.get(\'reason\', \'\')!r}"\n'
                '            f"{_describe_permission_decider(info)}"',
        test="test_reason_is_written_last",
        because="a scalar field written after the free-text reason, which "
                "parse_decision_trace silently folds into the reason value",
    ),
    Reversion(
        target="jaato-server/shared/plugins/permission/plugin.py",
        find="            if not is_subagent_mode:\n"
             '                self._emit_resolved(tool_name, "", False, method)',
        replace="            if not is_subagent_mode and self._on_permission_resolved:\n"
                '                self._on_permission_resolved(tool_name, "", False, method)',
        test="test_every_resolved_emission_goes_through_the_one_door",
        because="a branch firing the resolved hook without recording that it "
                "did, which makes the #968 opt-in emit a second event for "
                "the same decision",
    ),
]


def test_every_reversion_here_names_a_test_in_this_module() -> None:
    """A reversion naming a renamed test is BLOCKED in the meta-guard, which
    it reports; this makes the rename fail here first, where it is edited."""
    names = set(globals())
    for reversion in REVERSIONS:
        assert reversion.test in names, reversion.test
