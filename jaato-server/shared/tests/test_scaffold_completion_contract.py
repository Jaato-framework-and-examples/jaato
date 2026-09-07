"""Guard: scaffold's completion-processor docs track the framework, not prose.

`jaato-scaffold explain` had a `prefetch` scope for the INPUT-side script
hook and nothing at all for the output-side one — ``grep -rn
completion_processor jaato-server/shared/scaffold/`` returned zero files
(jaato #769).  So an author wanting a self-correcting agent rediscovered
the shape by burning an eval arm, and the failure modes are not the ones
you would guess.

Adding an `explain completion` scope closes the gap once.  Keeping it
closed is a different problem, and the issue names the evidence: scaffold
templates rot fast, and an archetype doc in this very package asserted
framework behaviour in prose ("waits on the FIRST of {TURN_COMPLETED,
SESSION_TERMINATED}") that the framework had since changed, on a branch
that touched zero scaffold files.  Prose cannot notice.

So every claim this suite checks is COMPUTED from the framework:

* the wiring fields are ``dataclasses.fields(CompletionProcessor)`` — add
  one without documenting it and this goes red;
* the closed vocabularies are the ``PROCESSOR_*`` constants the parser
  itself reads;
* the ``validate`` return channels are ``ProcessorResult.__annotations__``;
* and the generated processor is not merely inspected but LOADED through
  ``load_processors`` and driven through ``invoke_processors``, so "it
  bounds its refusals" is a behaviour this suite observes rather than a
  sentence it greps for.

The last of those carries the dependency #769 flagged: now that
``max_refusals`` exists, the generator must emit the DECLARATIVE form.  A
generator still shipping a hand-rolled module-level counter would codify
exactly the folklore #768 set out to retire, and would do it in the one
place authors are told to copy from.
"""

from __future__ import annotations

import ast
import dataclasses

import pytest

from jaato_sdk.cascade_authoring import ProcessorResult
from shared.completion_processors import invoke_processors, load_processors
from shared.plugins.subagent import config as _cfg
from shared.scaffold import archetypes as A
from shared.scaffold import _processor_template as _tpl
from shared.scaffold import explain, introspect
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Put the defect back: spell the vocabularies instead of reading them.
#: A hand-kept copy of a framework vocabulary is the exact shape of rot
#: this scope exists to be immune to — it renders plausibly and drifts
#: silently, which is how `archetypes.py` came to document a `complete()`
#: rule the framework no longer had.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find='    vocab = S["vocabularies"]',
        replace=('    vocab = {"on_error": ["fail_completion", "warn"],\n'
                 '             "phase": ["finalization", "completeness"]}'),
        test="test_the_vocabularies_are_the_frameworks_own",
        because="the rendered vocabularies being READ from the framework's "
                "PROCESSOR_* constants rather than copied into the doc, "
                "where they rot without anything noticing",
    ),
]


@pytest.fixture(scope="module")
def rendered():
    data, text = explain.completion()
    return data, text


# --------------------------------------------- 1. the doc reads the framework

def test_every_wiring_field_is_documented(rendered):
    """Add a field to ``CompletionProcessor``, document it or fail here.

    The gap #769 records is not that a doc was wrong, it is that nothing
    connected the doc to the thing it described.  Iterating the dataclass
    is that connection.
    """
    data, text = rendered
    documented = {e["name"] for e in data["entries"]}
    actual = {f.name for f in dataclasses.fields(_cfg.CompletionProcessor)}
    assert documented == actual, (
        f"`explain completion` documents {sorted(documented)} but "
        f"CompletionProcessor has {sorted(actual)}; the undocumented ones "
        f"are invisible to anyone learning this hook from scaffold")
    for name in actual:
        assert name in text, f"{name} is missing from the human rendering"


def test_the_vocabularies_are_the_frameworks_own(rendered):
    """The closed vocabularies must be the constants the PARSER reads.

    Not a copy of them: a copy renders just as plausibly on the day the
    framework adds a value and stops being true.
    """
    data, text = rendered
    assert data["vocabularies"] == {
        "on_error": list(_cfg.PROCESSOR_ON_ERROR),
        "phase": list(_cfg.PROCESSOR_PHASES),
        "on_exhausted": list(_cfg.PROCESSOR_ON_EXHAUSTED),
    }
    for values in (_cfg.PROCESSOR_ON_ERROR, _cfg.PROCESSOR_PHASES,
                   _cfg.PROCESSOR_ON_EXHAUSTED):
        for value in values:
            assert value in text, (
                f"{value!r} is accepted by the profile parser but does not "
                f"appear in `explain completion`")


def test_every_validate_channel_is_documented(rendered):
    """All four channels, and each one's answer to the two questions that
    matter: does it block, and does it spend a refusal."""
    data, text = rendered
    assert set(data["validate_channels"]) == set(ProcessorResult.__annotations__)
    for channel in ProcessorResult.__annotations__:
        assert channel in text, (
            f"the {channel!r} channel exists in ProcessorResult but "
            f"`explain completion` never mentions it")


def test_the_documented_defaults_match_the_dataclass(rendered):
    """A default is the claim most likely to be quietly wrong."""
    data, _text = rendered
    documented = {e["name"]: e["default"] for e in data["entries"]}
    for f in dataclasses.fields(_cfg.CompletionProcessor):
        expected = ("<required>" if f.default is dataclasses.MISSING
                    else f.default)
        assert documented[f.name] == expected


def test_the_scope_is_reachable_from_the_overview():
    """The discovery path. A scope nothing points at is not discoverable,
    which was the state of this hook entirely."""
    _data, text = explain.overview()
    assert "explain completion" in text
    from shared.scaffold.__main__ import _SIMPLE_SCOPES, _SCOPES_HELP
    assert _SIMPLE_SCOPES["completion"] is explain.completion
    assert "completion" in _SCOPES_HELP


def test_it_points_at_its_input_side_sibling(rendered):
    """`prefetch` is the closest analogue and the issue says to read it
    first; the symmetry is the fastest way to understand this hook."""
    _data, text = rendered
    assert "prefetch" in text
    assert "max_turns" in text, (
        "the retry budget IS max_turns and there is no second knob — an "
        "author who does not learn that here builds the second one")
    assert any(f.name == "max_turns"
               for f in dataclasses.fields(_cfg.SubagentProfile)), (
        "the doc names max_turns as the retry budget; it must be a real "
        "profile key")


# ----------------------------- 2. the generator emits the declarative form

def test_the_generator_does_not_hand_roll_a_refusal_counter():
    """#769's stated dependency, and the reason it is stated.

    ``max_refusals`` now exists, so a generated processor that kept its own
    module-level counter would teach the pattern the framework just
    replaced — in the one file authors are told to copy from.
    """
    module = _tpl.render("gate", "test")
    wiring = _tpl.wiring_for("gate")
    assert "max_refusals" in wiring and "on_exhausted" in wiring, (
        "the emitted wiring must declare the ceiling; that is the whole "
        "point of it being a framework key")

    # Read the module rather than grep it: `max_refusals` legitimately
    # appears in its prose, and a substring match cannot tell the key being
    # documented from a counter being kept.
    tree = ast.parse(module)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)], (
        "the generated processor declares a `global` — the only reason a "
        "processor ever needed one was to carry a refusal counter across "
        "calls, which is now the framework's job")
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign)
                   else [])
        for t in targets:
            name = getattr(t, "id", "")
            assert "refusal" not in name.lower(), (
                f"the generated processor keeps a module-level {name!r} — a "
                f"hand-rolled refusal counter, which is the folklore #768 "
                f"retired and which survives only on an undocumented "
                f"caching guarantee")


def test_the_generated_processor_documents_its_own_wiring():
    """The module and the wiring cannot drift: one renders the other."""
    module = _tpl.render("gate", "test")
    for line in _tpl.wiring_for("gate").splitlines():
        assert line.strip() in module


# ------------------- 3. the generated processor BEHAVES, observed not asserted

class _Ctx:
    config_root = None
    agent_params: dict = {}
    tool_calls = [{"name": "cli", "success": False,
                   "result": {"error": "boom"}, "turn_index": 0}]

    def __init__(self, workspace):
        self.workspace_path = str(workspace)


@pytest.fixture
def emitted(tmp_path):
    """The generated processor, loaded exactly as the daemon loads it."""
    path = tmp_path / "gate.py"
    path.write_text(_tpl.render("gate", "test"), encoding="utf-8")
    return path


def _loaded(path, **entry):
    return load_processors(
        [_cfg.CompletionProcessor(script=str(path), **entry)],
        workspace_path=str(path.parent), config_root=None,
    )


def test_the_generated_processor_loads_and_gates(emitted):
    """It refuses a payload claiming a clean run over a failed tool call,
    and accepts one that reports it — the check it ships with, run."""
    loaded = _loaded(emitted)
    assert loaded[0].load_error is None
    assert loaded[0].validate_fn is not None

    ctx = _Ctx(emitted.parent)
    dishonest = invoke_processors(loaded, payload={}, context=ctx)
    assert dishonest.has_fatal
    honest = invoke_processors(
        loaded, payload={"errors": ["the cli call failed"]}, context=ctx)
    assert not honest.has_fatal


def test_the_generated_processor_terminates_under_its_emitted_ceiling(emitted):
    """The refusal loop ends — observed by driving it, not by reading it.

    #768's incident was seven refusals in 156 seconds on an unchanging
    payload.  Here the same unchanging payload is re-submitted and must
    stop being refused.
    """
    ceiling = 3
    loaded = _loaded(emitted, max_refusals=ceiling, on_exhausted="allow")
    ctx = _Ctx(emitted.parent)
    blocked = 0
    for _ in range(10):
        if invoke_processors(loaded, payload={}, context=ctx).has_fatal:
            blocked += 1
    assert blocked == ceiling, (
        f"the same unchanging payload was refused {blocked} times under "
        f"max_refusals={ceiling}; that loop is what BLOCKED the arm")


def test_the_generated_processor_calls_a_missing_checker_a_fault(emitted, tmp_path):
    """An environment fault must not spend the agent's retries (#768 rule 6).

    Pointed at a command that does not exist, the emitted gate must block
    (the checks did not run) while leaving the refusal budget untouched.
    """
    src = emitted.read_text(encoding="utf-8").replace(
        "CHECKS_COMMAND: str | None = None",
        'CHECKS_COMMAND: str | None = "definitely-not-a-real-command --all"')
    faulty = tmp_path / "faulty.py"
    faulty.write_text(src, encoding="utf-8")

    loaded = _loaded(faulty, max_refusals=2)
    ctx = _Ctx(tmp_path)
    first = invoke_processors(loaded, payload={"errors": ["x"]}, context=ctx)
    assert first.has_fatal, "a gate that could not run must not read as a pass"
    assert loaded[0].refusals == 0, (
        "an unfixable environment fault consumed a refusal; a retryable "
        "message about one burns the whole budget without ever producing a "
        "verdict")


def test_the_generated_processor_never_reads_a_broken_checker_as_a_pass(
        emitted, tmp_path):
    """#768 rule 5, run rather than asserted.

    With one-line-per-failure semantics stdout IS the error list, so a
    non-zero exit with EMPTY stdout means the checker never got as far as
    checking.  Returning no errors there waves the completion through on a
    gate that is not running — the day's most repeated defect class.
    """
    src = emitted.read_text(encoding="utf-8").replace(
        "CHECKS_COMMAND: str | None = None",
        'CHECKS_COMMAND: str | None = "sh -c \'exit 7\'"')
    broken = tmp_path / "broken.py"
    broken.write_text(src, encoding="utf-8")

    loaded = _loaded(broken)
    out = invoke_processors(
        loaded, payload={"errors": ["x"]}, context=_Ctx(tmp_path))
    assert out.has_fatal, (
        "the checker exited non-zero and printed nothing, and the generated "
        "processor accepted the completion anyway")


# ------------------------------------------------- 4. the two halves agree

def test_explain_and_the_archetype_point_at_each_other():
    """A reader arriving at either must find the other.

    `explain completion` teaches the contract; `new processor` emits it.
    Documentation that does not mention the generator leaves the reader
    hand-writing the thing that got every rule wrong on its first pass.
    """
    _data, text = explain.completion()
    assert "new processor" in text
    doc = A.ARCHETYPES[A.PROCESSOR]
    assert any("explain completion" in s for s in doc.next_steps)
