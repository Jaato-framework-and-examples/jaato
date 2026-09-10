"""A `completion_payload_schema` that does not resolve must say so.

`_should_hide_signal_completion` reads `self._payload_schema is None`, which
is true in two very different situations:

  * the profile declared no schema — the documented way to opt OUT of
    `signal_completion`, and correct;
  * the profile DID declare one and the path did not resolve — an author
    mistake whose entire visible consequence was that the tool disappeared.

The second one cost a whole run: the model hunts for `signal_completion`
through `list_tools`, never finds it, ends its turn, the framework spends the
completion-nudge budget re-prompting, and the driver gets `None` from a
session that looked like it ran.  The only trace was a resolver WARNING about
a path — nothing connecting that path to the tool that vanished because of it.

`jaato-scaffold validate` now reports the same two mistakes BEFORE a run
(`redundant_config_root_prefix`, `completion_asset_missing`); this is the
runtime backstop for a session that did not go through it.
"""

import logging
import types

import pytest

from shared.lifecycle_tools import LifecycleTools


def _session(schema_ref, workspace=None):
    """A stand-in carrying only what LifecycleTools.__init__ reads."""
    return types.SimpleNamespace(
        _completion_payload_schema=schema_ref,
        workspace_path=str(workspace) if workspace else None,
        runtime=types.SimpleNamespace(_config_root=None),
        _parent_session=None,
        _presentation_context=None,
    )


def test_no_schema_declared_is_silent(caplog):
    """Opting out is the documented path — it must not look like an error."""
    with caplog.at_level(logging.WARNING):
        lt = LifecycleTools(_session(None))
    assert lt._schema_declared_but_unresolved is False
    assert lt._should_hide_signal_completion() is True      # still hidden
    assert "completion_payload_schema" not in caplog.text


def test_declared_but_unresolved_is_announced(caplog, tmp_path):
    with caplog.at_level(logging.WARNING):
        lt = LifecycleTools(_session("completion_schemas/nope.json", tmp_path))
    assert lt._schema_declared_but_unresolved is True
    assert lt._should_hide_signal_completion() is True
    assert "signal_completion is HIDDEN" in caplog.text
    assert "completion_schemas/nope.json" in caplog.text


def test_the_warning_names_the_jaato_prefix_trap(caplog, tmp_path):
    """The commonest way to get here: writing the prefix the resolver adds."""
    with caplog.at_level(logging.WARNING):
        LifecycleTools(_session(".jaato/completion_schemas/step.json", tmp_path))
    assert "'.jaato/'" in caplog.text
    assert "jaato-scaffold validate" in caplog.text


def test_a_resolvable_schema_is_silent_and_exposes_the_tool(caplog, tmp_path):
    d = tmp_path / ".jaato" / "completion_schemas"
    d.mkdir(parents=True)
    (d / "step.json").write_text('{"type": "object"}')
    with caplog.at_level(logging.WARNING):
        lt = LifecycleTools(_session("completion_schemas/step.json", tmp_path))
    assert lt._schema_declared_but_unresolved is False
    assert lt._should_hide_signal_completion() is False
    assert "HIDDEN" not in caplog.text


def test_an_inline_schema_never_reports_unresolved(caplog):
    with caplog.at_level(logging.WARNING):
        lt = LifecycleTools(_session({"type": "object"}))
    assert lt._schema_declared_but_unresolved is False
    assert lt._should_hide_signal_completion() is False
