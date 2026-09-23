"""`explain prefetch` documents every RenderContext attribute, derived (#911).

The topic hand-listed nine of ``RenderContext``'s ten attributes, and the
missing one -- ``session``, the handle the dataclass documents as reaching
``session.workspace_path`` / ``session.history`` -- was NOT doc-lag behind a
new field.  ``git log -S`` puts ``session`` in the first generation of the
dataclass (2026-05-01), ``explain prefetch`` at 2026-06-24 and its last edit at
2026-09-08: the two genuinely later fields (``tool_calls``, ``session_id``) are
both listed, and the oldest one never was.  A hand-copied list, incomplete on
the day it was written.

So the SET comes from ``dataclasses.fields(RenderContext)`` and cannot be
incomplete; ``_RENDER_CONTEXT_NOTES`` carries only the per-attribute prose no
derivation could know, and an attribute it does not describe is rendered by
name alone rather than dropped.

This guard asserts the derivation in both destinations -- the ``--json`` data
and the human text -- because they are written separately and either one can be
re-hardcoded on its own.
"""

import dataclasses

from jaato_server.shared.dynamic_instructions import RenderContext
from jaato_server.shared.scaffold import explain
from jaato_server.shared.tests.reversion import Reversion

_EXPLAIN = "jaato-server/jaato_server/shared/scaffold/explain.py"

#: The list as it stood when the issue was filed: nine of ten, `session`
#: missing.
_HAND_LISTED = (
    '        "context_attrs": ["agent_params", "registry", "runtime", '
    '"workspace_path",\n'
    '                          "config_root", "env", "session_id", "logger", '
    '"tool_calls"],'
)

_HAND_LISTED_TEXT = '''        "      context = RenderContext: agent_params (the agent's params dict),",
        "        registry (registry.get_plugin('<name>') to reach a plugin),",
        "        runtime, workspace_path, config_root, env (os.environ snapshot),",
        "        session_id, logger, tool_calls (completion-time only; [] for",
        "        input-side prefetch).",'''

REVERSIONS = [
    Reversion(
        target=_EXPLAIN,
        find='        "context_attrs": [name for name, _ in attrs],',
        replace=_HAND_LISTED,
        test="test_json_lists_every_render_context_attribute",
        because="the --json attribute list is hand-copied again, so `session` "
                "is missing from it",
    ),
    Reversion(
        target=_EXPLAIN,
        find="        *_context_attr_lines(attrs),",
        replace=_HAND_LISTED_TEXT,
        test="test_the_text_names_every_render_context_attribute",
        because="the human page is hand-copied again, so a reader is told "
                "about nine of the ten handles",
    ),
]


def _fields():
    return [f.name for f in dataclasses.fields(RenderContext)]


def test_json_lists_every_render_context_attribute():
    """``data["context_attrs"]`` IS the dataclass's field list, in order."""
    data, _ = explain.prefetch()
    assert data["context_attrs"] == _fields(), (
        "explain prefetch's context_attrs disagrees with RenderContext. It "
        "must be derived from dataclasses.fields, not written out."
    )


def test_the_text_names_every_render_context_attribute():
    """The human page names each attribute on its own line."""
    _, text = explain.prefetch()
    for name in _fields():
        assert f"\n        {name}" in text, (
            f"`explain prefetch` never names the RenderContext attribute "
            f"{name!r} -- the exact shape of #911."
        )


def test_session_is_documented():
    """The attribute the hand-copied list missed, named.

    Kept separate from the derived checks: it is the reported symptom, and a
    reader of this file should see it asserted rather than inferred.
    """
    data, text = explain.prefetch()
    assert "session" in data["context_attrs"]
    assert "\n        session — " in text


def test_notes_describe_only_real_attributes():
    """A note for an attribute that no longer exists is documentation of a lie.

    The set is derived, so a renamed field silently drops its prose; this is
    what says so.
    """
    unknown = sorted(set(explain._RENDER_CONTEXT_NOTES) - set(_fields()))
    assert not unknown, (
        f"_RENDER_CONTEXT_NOTES describes attributes RenderContext does not "
        f"have: {unknown}"
    )


def test_an_undescribed_attribute_is_still_rendered():
    """A field with no note appears by NAME, never silently.

    This is the property that makes the derivation safe to leave alone: adding
    a field to the dataclass documents it immediately, at worst namelessly
    annotated.
    """
    undescribed = [n for n in _fields()
                   if n not in explain._RENDER_CONTEXT_NOTES]
    # workspace_path / config_root / logger carry no note today; if that ever
    # stops being true the assertion below is vacuous, so say so.
    assert undescribed, (
        "every RenderContext attribute now carries a note, so this test no "
        "longer exercises the undescribed path -- give it a synthetic pair "
        "instead of deleting it."
    )
    lines = explain._context_attr_lines([(n, "") for n in undescribed])
    _, text = explain.prefetch()
    for line in lines:
        assert line in text


def test_the_json_notes_are_a_subset_of_the_attributes():
    """``context_attr_notes`` annotates the same set it is rendered beside."""
    data, _ = explain.prefetch()
    assert set(data["context_attr_notes"]) <= set(data["context_attrs"])
