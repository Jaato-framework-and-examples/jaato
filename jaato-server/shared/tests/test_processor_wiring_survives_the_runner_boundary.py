"""Guard: a completion_processors entry crosses into the runner INTACT.

A profile is parsed daemon-side; the session that reads it runs in the runner
subprocess. Everything between is a serialise/deserialise, and this one was
written out by hand — three separate places listing the fields they knew
about:

* ``runner_spawn.build_session_envelope`` (the spawn path),
* ``SessionManager``'s isolated-subagent envelope,
* ``runner/session._processors_from_envelope`` (the far side).

All three named ``script`` / ``output`` / ``on_error`` / ``description`` /
``phase`` and stopped. ``CompletionProcessor`` has eight fields. So ``name``,
``max_refusals`` and ``on_exhausted`` were dropped in transit — parsed from
the profile, persisted into the snapshot, and gone by the time a session used
them.

WHAT THAT COST, measured rather than reasoned about: a live daemon running a
profile that declared ``max_refusals: 2`` refused completion **494 times**
without ever exhausting, because the entry that reached ``invoke_processors``
had ``max_refusals=None``. That is jaato #768's non-terminating loop,
reintroduced by a serialiser, with the profile still saying the ceiling was
there and every unit test still green — the ceiling was enforced correctly by
code that was never given a ceiling. ``suppress_inherited_processors`` (#791)
was lost the same way, since it matches on ``name``.

WHY THE EXISTING GUARDS COULD NOT SEE IT. Both other guards on this feature
construct their own objects: the invocation-level one builds a
``LoadedProcessor`` by hand, and the loop-level one builds a
``LifecycleTools`` over a session double. Neither crosses a process boundary,
so neither can notice a boundary that drops fields. The defect was only
visible from a real session — which is what jaato #770 meant by "watch the
session loop instead", and is why that instruction produced a fix rather than
a test.

So this module asserts the boundary itself, and asserts it from
``dataclasses.fields`` rather than from a list of field names: a list is what
failed, and a second copy of it would fail the same way. Adding a field to
``CompletionProcessor`` without teaching the wire about it fails here.
"""

from __future__ import annotations

import dataclasses

import pytest

from shared.plugins.subagent.config import (
    CompletionProcessor,
    PROCESSOR_ON_ERROR,
    PROCESSOR_ON_EXHAUSTED,
    PROCESSOR_PHASES,
    completion_processors_from_wire,
    completion_processors_to_wire,
)
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Put the defect back: serialise the fields somebody thought of.
#:
#: The exact five the three call sites listed. It is a plausible-looking
#: serialiser — every field it names it handles correctly — and it silently
#: discards the refusal ceiling, which is the only reason the loop that #768
#: exists to bound terminates at all.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/plugins/subagent/config.py",
        find="        elif _dc.is_dataclass(entry) and not isinstance(entry, type):\n"
             "            out.append(_dc.asdict(entry))",
        replace="        elif _dc.is_dataclass(entry) and not isinstance(entry, type):\n"
                "            out.append({k: getattr(entry, k) for k in (\n"
                "                'script', 'output', 'on_error', 'description',\n"
                "                'phase')})",
        test="test_every_field_of_the_dataclass_survives_the_wire",
        because="a hand-written field list in the daemon->runner serialiser, "
                "which dropped max_refusals / on_exhausted / name and left a "
                "declared refusal ceiling with no effect on the running "
                "session",
    ),
]


def _non_default(field: dataclasses.Field):
    """A value for *field* that is NOT its default.

    The point of the round-trip test is that a field ARRIVES, and a field
    left at its default arrives whether or not the wire carries it — the
    reconstruction would supply the same value by defaulting. So every field
    has to be set to something the far side could only know by being told.

    Closed vocabularies pick their non-default member rather than a made-up
    string, because the parser validates them and would fall back to the
    default, which is exactly the silent pass this avoids.
    """
    name = field.name
    if name == "on_error":
        return next(v for v in PROCESSOR_ON_ERROR if v != field.default)
    if name == "phase":
        return next(v for v in PROCESSOR_PHASES if v != field.default)
    if name == "on_exhausted":
        return next(v for v in PROCESSOR_ON_EXHAUSTED if v != field.default)
    if name == "max_refusals":
        return 7
    # Everything else on this dataclass is a string or an optional string.
    return f"wire-{name}"


def _fully_populated() -> CompletionProcessor:
    """A ``CompletionProcessor`` with every field set away from its default."""
    kwargs = {f.name: _non_default(f)
              for f in dataclasses.fields(CompletionProcessor)}
    return CompletionProcessor(**kwargs)


def test_every_field_of_the_dataclass_survives_the_wire():
    """Serialise, deserialise, compare the WHOLE dataclass.

    Driven from ``dataclasses.fields`` so this cannot rot into the thing it
    guards against: a field added to ``CompletionProcessor`` is automatically
    part of the assertion, and if the wire does not carry it the equality
    fails naming it.
    """
    original = _fully_populated()
    restored = completion_processors_from_wire(
        completion_processors_to_wire([original])
    )

    assert len(restored) == 1, (
        f"one processor in, {len(restored)} out — the wire dropped the entry "
        f"entirely"
    )
    assert restored[0] == original, (
        "a completion_processors entry lost fields crossing the daemon -> "
        "runner boundary. Differences: "
        + ", ".join(
            f"{f.name}: sent {getattr(original, f.name)!r}, "
            f"got {getattr(restored[0], f.name)!r}"
            for f in dataclasses.fields(CompletionProcessor)
            if getattr(original, f.name) != getattr(restored[0], f.name)
        )
    )


def test_the_wire_dict_names_every_field():
    """The serialised form carries every field, by name.

    Separate from the round-trip because the two fail differently and a
    reader should be able to tell which half broke: this one says the
    SERIALISER forgot a field, the round-trip says either half did.
    """
    wire = completion_processors_to_wire([_fully_populated()])[0]
    expected = {f.name for f in dataclasses.fields(CompletionProcessor)}
    missing = expected - set(wire)

    assert not missing, (
        f"the daemon -> runner wire dict omits {sorted(missing)}; the runner "
        f"will default them, so a profile that declared them runs without "
        f"them and says nothing"
    )


def test_the_ceiling_specifically_reaches_the_far_side():
    """Named on its own, because this is the field whose loss does not show.

    An entry that arrives without ``output`` writes no file and somebody
    notices. An entry that arrives without ``max_refusals`` behaves exactly
    as it did before the ceiling existed — it just never stops refusing — and
    the profile still says the ceiling is there. That is the shape of the
    defect this module was written for, so it gets an assertion that names it
    rather than being one line of a dataclass comparison.
    """
    entry = CompletionProcessor(script="scripts/processors/gate.py",
                                name="gate", max_refusals=3,
                                on_exhausted="fail")
    restored = completion_processors_from_wire(
        completion_processors_to_wire([entry])
    )[0]

    assert restored.max_refusals == 3, (
        f"max_refusals arrived as {restored.max_refusals!r}; a None ceiling "
        f"is the pre-#768 unbounded loop, and the profile still claims a "
        f"bound"
    )
    assert restored.on_exhausted == "fail", (
        f"on_exhausted arrived as {restored.on_exhausted!r} — the exhaustion "
        f"policy is a declared choice and defaulting it silently reverses "
        f"what the author asked for"
    )
    assert restored.name == "gate", (
        "name arrived empty; suppress_inherited_processors matches on it, so "
        "a child profile's declining of an inherited processor would stop "
        "matching (jaato #791)"
    )


def test_a_raw_wire_dict_passes_through_unchanged():
    """Subagent spawn specs may carry dicts that never became dataclasses.

    The serialiser has to tolerate them — the path it replaced did — and must
    not quietly normalise them, because a spec is the caller's and the
    framework has no basis for editing it.
    """
    raw = {"script": "scripts/processors/x.py", "max_refusals": 5,
           "on_exhausted": "fail", "name": "x"}
    assert completion_processors_to_wire([raw]) == [raw]


@pytest.mark.parametrize("junk", [None, 42, "not-a-list", [None], [42],
                                  [{"no_script": True}], [{"script": "  "}]])
def test_malformed_entries_are_skipped_not_raised(junk):
    """A bad entry loses that processor, never the session.

    Same policy as a profile file declaring one, because it is the same
    parser: the gate surfaces as a load error the agent sees at completion
    time, which is louder than a bootstrap that failed with a traceback the
    agent never gets to read.
    """
    assert completion_processors_from_wire(junk) == []


def test_a_duck_typed_entry_keeps_every_field_too():
    """A stand-in exposing ``script`` is serialised through the dataclass.

    The path this replaced duck-typed on ``hasattr(entry, "script")``, and
    callers (subagent spawn specs, and the envelope suite's own doubles) rely
    on that tolerance. Keeping it is not the interesting part; keeping it
    WITHOUT a second hand-written field list is, so this asserts the ceiling
    survives that route as well as the dataclass one.
    """
    from types import SimpleNamespace

    entry = SimpleNamespace(script="scripts/processors/gate.py", output=None,
                            on_error="warn", description=None,
                            phase="finalization", name="gate",
                            max_refusals=2, on_exhausted="fail")
    restored = completion_processors_from_wire(
        completion_processors_to_wire([entry])
    )[0]

    assert (restored.max_refusals, restored.on_exhausted, restored.name) == (
        2, "fail", "gate")


def test_an_entry_that_cannot_be_serialised_is_not_dropped_in_silence(caplog):
    """A vanished processor is a gate that stopped gating.

    Whatever the reason an entry cannot be carried, the one outcome that must
    not happen is the session running as though no gate had been declared.
    The profile still says it is there, so the loss has to be findable.
    """
    with caplog.at_level("WARNING"):
        assert completion_processors_to_wire([object()]) == []
    assert any("dropping unusable entry" in r.getMessage()
               for r in caplog.records), (
        f"an unusable completion_processors entry was dropped without a "
        f"warning: {[r.getMessage() for r in caplog.records]}")


def test_the_runner_side_extraction_uses_the_shared_parser():
    """The far side is wired to the same parser, not a second reconstruction.

    Asserted through ``_processors_from_envelope`` — the function the runner
    actually calls — rather than by reading the source, so moving it back to
    a hand-rolled rebuild fails here.
    """
    from server.runner.session import _processors_from_envelope

    class _Envelope:
        completion_processors = [{
            "script": "scripts/processors/gate.py",
            "name": "gate", "max_refusals": 4, "on_exhausted": "fail",
        }]

    got = _processors_from_envelope(_Envelope())

    assert len(got) == 1
    assert got[0].max_refusals == 4 and got[0].on_exhausted == "fail", (
        f"the runner rebuilt the entry without its ceiling ({got[0]!r}); the "
        f"daemon can send max_refusals and it still will not take effect"
    )
    assert got[0].name == "gate"
