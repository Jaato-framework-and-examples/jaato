"""``jaato-scaffold validate`` catches the two traps that hang a harness.

Both were measured on 2026-09-08 while bringing up an SDK-driven cascade, and
both share a shape: the run does its work, the daemon is content, and the
DRIVER waits forever with nothing logged on its side.  A validator finding is
the only cheap place to meet them, because at runtime they present as a
timeout rather than as an error.

1. ``echo`` with no ``usage`` — a turn is recorded only when the provider
   reported tokens, and the post-turn hook gated on that record is the single
   emitter of both ``TurnCompletedEvent`` and ``SessionTerminatedEvent``.  No
   usage, no terminal event, and every ``Session.complete``/``.ask`` waits out
   its timeout.

2. A ``spawn_payload_schema`` property typed as anything but a string —
   ``agent_params`` cross the IPC wire as ``key=value`` argv tokens, so the
   daemon validates strings and the spawn is refused on every attempt.
"""
import json
from pathlib import Path
from types import SimpleNamespace

from shared.scaffold import introspect
from shared.scaffold.validate import (
    _check_spawn_schema_wire_types,
    validate_profile,
)


def _validate(provider, plugin_configs=None):
    prof = SimpleNamespace(provider=provider, model="m", plugins=[],
                           plugin_configs=plugin_configs or {})
    return validate_profile(
        prof, providers=introspect.providers(), plugins=introspect.plugins(),
        gc_names=list(introspect.gc_strategies().keys()))


def _codes(diags, code):
    return [d for d in diags if d.code == code]


# ------------------------------------------------------------------- echo

def test_echo_is_not_reported_as_an_unknown_provider():
    """It is excluded from the catalogue but installed, and the live
    conformance suite runs on it — calling it unknown told every harness
    author their working profile was invalid."""
    diags = _validate("echo")
    assert _codes(diags, "unknown_provider") == []
    assert _codes(diags, "echo_is_a_test_double")


def test_echo_without_usage_warns_that_no_terminal_event_will_fire():
    diags = _validate("echo", {"echo": {"response": "hi"}})
    found = _codes(diags, "echo_reports_no_usage")
    assert found, "an echo profile with no usage must be flagged"
    assert found[0].severity == "warn"
    assert found[0].where == "plugin_configs.echo.usage"
    # The message has to name the consequence, not just the missing key:
    # the author's symptom is a hang, and nothing else will tell them why.
    assert "TurnCompletedEvent" in found[0].message


def test_echo_with_usage_is_quiet():
    diags = _validate("echo", {"echo": {"usage": {"prompt_tokens": 1000,
                                                  "output_tokens": 200}}})
    assert _codes(diags, "echo_reports_no_usage") == []


def test_a_real_provider_is_untouched_by_the_echo_branch():
    diags = _validate("anthropic")
    assert _codes(diags, "echo_is_a_test_double") == []
    assert _codes(diags, "echo_reports_no_usage") == []


# ----------------------------------------------------- spawn payload schema

def _spawn(tmp_path: Path, schema: dict, *, inline: bool = False):
    if inline:
        prof = SimpleNamespace(spawn_payload_schema=schema)
    else:
        d = tmp_path / "spawn_schemas"
        d.mkdir(parents=True, exist_ok=True)
        (d / "s.json").write_text(json.dumps(schema), encoding="utf-8")
        prof = SimpleNamespace(spawn_payload_schema="spawn_schemas/s.json")
    out: list = []
    _check_spawn_schema_wire_types({"p": prof}, str(tmp_path), out)
    return out


def test_integer_property_is_unreachable_over_the_wire(tmp_path):
    out = _spawn(tmp_path, {"type": "object",
                            "properties": {"iteration": {"type": "integer"}}})
    assert [d.code for d in out] == ["spawn_schema_type_unreachable"]
    assert out[0].severity == "error"
    assert out[0].where == "spawn_payload_schema.properties.iteration"


def test_string_properties_are_accepted(tmp_path):
    out = _spawn(tmp_path, {"type": "object",
                            "properties": {"loop": {"type": "string"},
                                           "subphase": {"type": "string",
                                                        "pattern": "^[0-9.]+$"}}})
    assert out == []


def test_inline_schema_is_checked_too(tmp_path):
    out = _spawn(tmp_path, {"type": "object",
                            "properties": {"flag": {"type": "boolean"}}},
                 inline=True)
    assert [d.code for d in out] == ["spawn_schema_type_unreachable"]


def test_union_type_is_flagged_on_its_non_string_member(tmp_path):
    out = _spawn(tmp_path, {"type": "object",
                            "properties": {"n": {"type": ["string", "integer"]}}})
    assert [d.code for d in out] == ["spawn_schema_type_unreachable"]


def test_a_profile_without_a_spawn_schema_is_quiet(tmp_path):
    out: list = []
    _check_spawn_schema_wire_types({"p": SimpleNamespace()}, str(tmp_path), out)
    assert out == []


# ------------------------------------------------------- default_agent (#944)
#
# ``default_agent`` binds a profile's persona to the profile, so a spawn
# naming only the profile still gets instructions.  A name that resolves to
# no file fails at the spawn — the one moment the caller can do nothing
# about it, having passed no agent at all.

def _default_agent(tmp_path: Path, agent_name, *, on_disk=None, layout="file"):
    from shared.scaffold.validate import _check_default_agent_exists

    cr = tmp_path / ".jaato"
    (cr / "agents").mkdir(parents=True, exist_ok=True)
    if on_disk:
        if layout == "file":
            (cr / "agents" / f"{on_disk}.md").write_text("hi", encoding="utf-8")
        else:
            d = cr / "agents" / on_disk
            d.mkdir(parents=True, exist_ok=True)
            (d / "PROMPT.md").write_text("hi", encoding="utf-8")
    prof = SimpleNamespace(default_agent=agent_name)
    out: list = []
    _check_default_agent_exists({"documentalista": prof}, tmp_path, str(cr), out)
    return out


def test_default_agent_naming_no_file_is_an_error(tmp_path):
    out = _default_agent(tmp_path, "gone")
    assert [d.code for d in out] == ["default_agent_missing"]
    assert out[0].severity == "error"
    assert out[0].profile == "documentalista"
    # The message must name the consequence, not just the missing file.
    assert "spawn_subagent(profile='documentalista')" in out[0].message


def test_default_agent_present_on_disk_is_quiet(tmp_path):
    assert _default_agent(tmp_path, "writer", on_disk="writer") == []


def test_default_agent_as_a_directory_persona_is_quiet(tmp_path):
    assert _default_agent(tmp_path, "writer", on_disk="writer",
                          layout="dir") == []


def test_a_profile_without_a_default_agent_is_quiet(tmp_path):
    assert _default_agent(tmp_path, None) == []


def test_the_check_does_not_render_the_persona(tmp_path):
    """Rendering runs the persona's ``{{!py:...}}`` prefetch scripts, and
    validate is side-effect free — so a persona whose prefetch would blow
    up must still validate clean."""
    cr = tmp_path / ".jaato"
    (cr / "agents").mkdir(parents=True)
    (cr / "agents" / "writer.md").write_text(
        "{{!py:scripts/does_not_exist.py}}", encoding="utf-8")
    from shared.scaffold.validate import _check_default_agent_exists
    out: list = []
    _check_default_agent_exists(
        {"p": SimpleNamespace(default_agent="writer")}, tmp_path, str(cr), out)
    assert out == []
