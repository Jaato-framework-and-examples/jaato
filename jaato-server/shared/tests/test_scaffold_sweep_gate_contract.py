"""Guard: the sweep archetype's emitted completion gate is a WORKING set.

``new sweep`` emits four files that only mean anything together — the checks
(``acceptance.sh``), the processor that runs them, the schema without which
there is nothing to gate, and the profile carrying the two keys that connect
them (jaato #772).  This module holds them to that.

WHY IT READS THE FRAMEWORK INSTEAD OF THE TEMPLATES.  #772 asks for "a test
that fails when the framework moves", and cites a live example of the
alternative: ``archetypes.py`` carried a confident description of ``complete()``
waiting on the first of ``{TURN_COMPLETED, SESSION_TERMINATED}`` while #767 was
in flight changing exactly that, and #767's branch touched zero scaffold files.
Prose asserting a contract cannot notice the contract moving.  So nothing here
greps a template for a reassuring string:

* the schema gate is checked by driving :class:`LifecycleTools` and looking at
  whether ``signal_completion`` is in the tool surface — if that gate is ever
  relaxed, the emitted profile's claim that the schema is load-bearing becomes
  false and ``test_the_schema_is_what_makes_signal_completion_exist`` says so;
* the profile keys are checked by parsing the emitted YAML with the framework's
  own parser and reading the ``CompletionProcessor`` dataclass, so renaming
  ``max_refusals`` breaks this rather than silently emitting a dead key;
* the paths are checked through the real ``script_loader`` /
  ``completion_schema_loader`` resolvers, so moving the ``.jaato/`` tier breaks
  this rather than emitting files the daemon cannot find;
* the gate's behaviour is checked by RUNNING it — the emitted ``acceptance.sh``
  is really executed, and the emitted processor is really driven through
  ``invoke_processors``, in each of the three states that matter.

THE STATE THAT MATTERS MOST is the unconfigured one.  As emitted, ``run_checks``
is empty, and the obvious behaviour for a script with nothing to check is to
exit 0.  That is #768 rule 5 — an error path returning the same value as
success — and here it would mean every arm of a graded sweep waved through by a
gate that never ran.  ``test_an_unconfigured_gate_refuses_rather_than_passes``
is the assertion this whole module exists for, and the reversion below puts
exactly that defect back.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from shared.completion_processors import invoke_processors, load_processors
from shared.completion_schema_loader import _resolve_schema_path
from shared.lifecycle_tools import LifecycleTools
from shared.plugins.subagent.config import build_inline_profile
from shared.scaffold import archetypes as A
from shared.scaffold import build
from shared.script_loader import resolve_script_path
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion
from shared.tests.test_lifecycle_tools import StubSession


#: Put the defect back: let an unconfigured ``acceptance.sh`` exit 0.
#:
#: This is the whole reason the gate is generated rather than described.  A
#: checks script with nothing configured that reports success is
#: indistinguishable, to the gate above it, from one that ran every check and
#: found no failures — so every arm of a graded sweep signals completion having
#: been checked against nothing, and the sweep reports a clean board.  The
#: emitted script exits 78 with an empty stdout instead, which the processor
#: reads as "the checker did not run".
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/_gate_templates.py",
        find="    exit __EX_CONFIG__\nfi",
        replace="    exit 0\nfi",
        test="test_an_unconfigured_gate_refuses_rather_than_passes",
        because="an unconfigured checks script reporting success, which lets "
                "the gate wave through every arm of a graded sweep on checks "
                "that never ran (#768 rule 5)",
    ),
    #: Put the OTHER defect back: emit the gate unwired.
    #:
    #: This is #772 itself.  `new processor` leaves CHECKS_COMMAND as None and
    #: prints the wiring for the author to paste — which is what left three
    #: files and the relationship between them as an exercise.  With the
    #: checks command unset the emitted processor skips the subprocess gate
    #: entirely and grades on the ledger check alone, so a set that LOOKS
    #: complete accepts an arm nobody checked.
    Reversion(
        target="jaato-server/shared/scaffold/build.py",
        find="checks_command=_gate.CHECKS_COMMAND,\n"
             "                                 wiring=wiring),",
        replace="wiring=wiring),",
        test="test_an_unconfigured_gate_refuses_rather_than_passes",
        because="the gate being emitted merely wireable rather than wired — "
                "an unset CHECKS_COMMAND means acceptance.sh is never run, so "
                "the set accepts an arm that met no criterion (jaato #772)",
    ),
]


GATE = "acceptance"


def _args(ws: Path, **kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    defaults = dict(archetype="sweep", workspace=str(ws), provider=None,
                    model=None, set=None, agents=None, force=False,
                    json=False, recoverable=False, dry_run=False,
                    secrets=None, secret_path=None, transport="ipc",
                    url=None, token=None, ca=None, name=None,
                    no_gate=False, gate_name=None)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def _run_new(args) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return build.run(args)


@pytest.fixture(scope="module")
def swept(tmp_path_factory) -> Path:
    """One real ``new sweep`` run, shared by every assertion below.

    Module-scoped: the run resolves providers and drives the emitted gate
    through the framework, which is seconds.  Nothing here mutates the
    workspace — the tests that need a CONFIGURED acceptance.sh copy it into
    their own tmp dir (:func:`_configured`) rather than editing this one.
    """
    ws = tmp_path_factory.mktemp("sweep-gate")
    assert _run_new(_args(ws)) == 0, "new sweep failed on a clean workspace"
    return ws


class _Ctx:
    """The RenderContext surface a completion processor actually reads."""

    def __init__(self, ws: Path, tool_calls=None):
        self.tool_calls = tool_calls or []
        self.agent_params: dict = {}
        self.workspace_path = str(ws)
        self.config_root = None
        self.env: dict = {}
        self.session_id = "gate-contract"
        self.logger = logging.getLogger(__name__)


HONEST = {"summary": "done", "errors": [], "warnings": []}


def _entry(ws: Path):
    """The ``CompletionProcessor`` the emitted profile declares.

    Parsed with the framework's own parser rather than read as YAML keys, so
    a field the framework stops understanding stops arriving here.
    """
    raw = yaml.safe_load(
        (ws / ".jaato" / "profiles" / f"{GATE}.yaml").read_text("utf-8"))
    profile = build_inline_profile(raw, name=GATE)
    assert profile.completion_processors, (
        "the emitted profile declares no completion_processors — the gate is "
        "not wired into anything")
    return profile, profile.completion_processors[0]


def _drive(ws: Path, loaded, payload=None):
    """One ``signal_completion``-shaped invocation of the loaded gate."""
    return invoke_processors(loaded, payload=payload or HONEST,
                             context=_Ctx(ws), phase_filter="finalization")


def _configured(ws: Path, tmp_path: Path, *checks: str) -> Path:
    """A copy of the emitted set whose ``acceptance.sh`` has real checks.

    The generator deliberately emits an EMPTY ``run_checks`` — it cannot know
    a sweep's acceptance criteria — so the states where the gate grades rather
    than faults only exist once an author has filled it in.  This does what
    that author does, textually, to the emitted script.

    Args:
        ws: The scaffolded workspace to copy from.
        tmp_path: Where the copy goes.
        checks: ``check`` invocations to place in ``run_checks``.

    Returns:
        The new workspace root.
    """
    import shutil

    dst = tmp_path / "configured"
    shutil.copytree(ws, dst)
    script = dst / "acceptance.sh"
    body = "\n".join(f"    {c}" for c in checks)
    text = script.read_text("utf-8")
    marker = "run_checks() {\n    :\n}"
    assert marker in text, (
        "the emitted acceptance.sh no longer has the empty run_checks this "
        "helper edits — update the helper, and check the emitted script still "
        "ships with no checks configured")
    script.write_text(text.replace(marker, "run_checks() {\n%s\n}" % body))
    script.chmod(0o755)
    return dst


# --------------------------------------------------------------------------
# 1. The framework facts the emitted set is built on.  These fail when the
#    framework moves, which is what #772 asks for.
# --------------------------------------------------------------------------

def test_the_schema_is_what_makes_signal_completion_exist(swept):
    """The emitted profile's two keys are one unit, per the REAL gate.

    ``LifecycleTools._should_hide_signal_completion`` hides the tool outright
    when no ``completion_payload_schema`` is declared.  That is why the gate
    set emits the schema alongside the processor: without it a profile with
    ``completion_processors:`` has no lenient gate, it has NO gate — the agent
    cannot signal, so the processor never runs, and a driver calling
    ``complete()`` waits for a payload that cannot arrive.

    Driven rather than asserted: if that gate is ever relaxed, this test goes
    red and the emitted profile's comment saying so is the thing to fix.
    """
    schema = json.loads(
        (swept / ".jaato" / "completion_schemas" / f"{GATE}.json")
        .read_text("utf-8"))

    without = LifecycleTools(StubSession(schema=None))
    assert all(s.name != "signal_completion" for s in without.get_tool_schemas()), (
        "the framework no longer hides signal_completion when no schema is "
        "declared — the emitted profile documents that it does")

    with_schema = LifecycleTools(StubSession(schema=schema))
    assert any(s.name == "signal_completion"
               for s in with_schema.get_tool_schemas()), (
        "the EMITTED schema does not make signal_completion appear, so the "
        "generated gate has nothing to gate")


def test_the_emitted_ceiling_is_a_field_the_framework_reads(swept):
    """``max_refusals`` reaches the framework as a parsed field, not a comment.

    #772's hard precondition: the gate must arrive with the DECLARATIVE
    ceiling rather than the module-level counter authors used to hand-roll.
    Renaming or dropping the field in ``CompletionProcessor`` breaks this
    instead of leaving scaffold emitting a key nothing reads.
    """
    _, entry = _entry(swept)
    assert entry.max_refusals == 3, (
        f"the emitted entry's max_refusals did not survive parsing "
        f"(got {entry.max_refusals!r}) — an unbounded gate does not "
        f"terminate on its own (#768 rule 2)")
    assert entry.on_exhausted == "allow", (
        f"the emitted entry's on_exhausted did not survive parsing "
        f"(got {entry.on_exhausted!r})")


def test_the_generated_processor_holds_no_counter_of_its_own(swept):
    """The module must not carry a second budget beside the framework's.

    By AST, not by grep: ``max_refusals`` legitimately appears in the module's
    prose, and a guard that matched the word would pass on a module that
    talked about the ceiling while keeping its own.  What is banned is
    ``global`` and a module-level integer that a ``validate`` mutates.
    """
    import ast

    src = (swept / ".jaato" / "scripts" / "processors" / f"{GATE}.py"
           ).read_text("utf-8")
    tree = ast.parse(src)
    globals_used = [n for n in ast.walk(tree) if isinstance(n, ast.Global)]
    assert not globals_used, (
        "the emitted processor declares `global` — the refusal ceiling is the "
        "framework's, and a module-level counter is the folklore #768 retired "
        "(it survives only on an undocumented per-session caching guarantee)")


def test_the_emitted_paths_resolve_through_the_frameworks_own_loaders(swept):
    """``script:`` and ``completion_payload_schema:`` land where the daemon looks.

    Both are relative references resolved against ``<workspace>/.jaato/``.  A
    file written anywhere else is a profile that parses and a gate that never
    loads — so this resolves them with the real resolvers rather than
    asserting the paths look right.
    """
    _, entry = _entry(swept)
    profile, _ = _entry(swept)

    script = resolve_script_path(entry.script, workspace_path=str(swept))
    assert script is not None and script.is_file(), (
        f"the emitted script: {entry.script!r} does not resolve under "
        f"{swept}/.jaato/")

    schema = _resolve_schema_path(profile.completion_payload_schema,
                                  workspace_path=str(swept))
    assert schema is not None and schema.is_file(), (
        f"the emitted completion_payload_schema "
        f"{profile.completion_payload_schema!r} does not resolve under "
        f"{swept}/.jaato/")


# --------------------------------------------------------------------------
# 2. The emitted acceptance.sh honours the contract the processor depends on.
#    Run, not read.
# --------------------------------------------------------------------------

def _sh(ws: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(ws / "acceptance.sh"), *args], cwd=ws,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the emitted checks script is POSIX sh")
def test_the_checks_script_reports_failures_one_per_line_on_stdout(swept, tmp_path):
    """stdout IS the failure list — that is what makes the gate's
    broken-gate discrimination possible, so it has to be true."""
    ws = _configured(swept, tmp_path,
                     'check "first thing is not done" false',
                     'check "second thing is fine" true',
                     'check "third thing is not done" false')
    proc = _sh(ws, "--all")
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}"
    assert lines == ["first thing is not done", "third thing is not done"], (
        f"stdout must carry exactly one line per FAILING check; got {lines!r}")


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the emitted checks script is POSIX sh")
def test_the_checks_script_is_silent_when_everything_passes(swept, tmp_path):
    """Nothing on stdout on success: a success line would be read as a failure."""
    ws = _configured(swept, tmp_path, 'check "fine" true')
    proc = _sh(ws, "--all")
    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}"
    assert proc.stdout.strip() == "", (
        f"a passing run must print nothing on stdout; got {proc.stdout!r}")


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the emitted checks script is POSIX sh")
def test_an_unconfigured_checks_script_is_not_silent_success(swept):
    """Unconfigured: non-zero AND an empty stdout.

    Both halves matter.  Non-zero is what stops it reading as a pass; the
    EMPTY stdout is what tells the processor the checks did not run, as
    opposed to ran and failed.  A diagnostic printed on stdout here would be
    handed to the agent as a failing check it cannot fix.
    """
    proc = _sh(swept, "--all")
    assert proc.returncode != 0, (
        "an acceptance.sh with no checks configured exited 0 — a gate that is "
        "not running must never read as a gate that passed")
    assert proc.stdout.strip() == "", (
        f"the unconfigured script wrote to stdout ({proc.stdout!r}); stdout is "
        f"the failure list, so the gate would read this as a failing check")
    assert proc.stderr.strip(), (
        "the unconfigured script said nothing on stderr — the author has to "
        "learn that the gate is unconfigured from somewhere")


# --------------------------------------------------------------------------
# 3. The set, driven end to end through the framework.
# --------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32",
                    reason="the emitted checks script is POSIX sh")
def test_an_unconfigured_gate_refuses_rather_than_passes(swept):
    """THE assertion of this module.  See the module docstring.

    Fresh from the generator the gate blocks, and blocks as an environment
    FAULT: it costs the agent no refusal (nothing it does will configure your
    acceptance criteria) and it blocks for exactly the one round-trip it needs
    to record the fault.  Both halves are checked behaviourally rather than by
    reading a message, because a fault and an error land in the same bucket
    for the round-trip they block — invoking twice is what separates them.
    """
    _, entry = _entry(swept)
    loaded = load_processors([entry], workspace_path=str(swept),
                             config_root=None)
    assert not loaded[0].load_error, loaded[0].load_error

    first = _drive(swept, loaded)
    assert first.has_fatal, (
        "the emitted gate ACCEPTED a completion while acceptance.sh has no "
        "checks configured — every arm of a graded sweep would signal "
        "completion having been checked against nothing")

    second = _drive(swept, loaded)
    assert loaded[0].refusals == 0, (
        f"an unconfigured checks script spent {loaded[0].refusals} of the "
        f"agent's refusals; an environment fault it cannot clear must not "
        f"consume the retry budget (#768 rule 6)")
    assert not second.has_fatal, (
        "the emitted gate blocked twice on a condition no retry can clear — "
        "a fault blocks for one round-trip; blocking forever is the "
        "non-terminating loop the budget exists to prevent")


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the emitted checks script is POSIX sh")
def test_a_failing_check_blocks_and_spends_exactly_one_refusal(swept, tmp_path):
    """A real failing check is a wrong ANSWER: it blocks and costs a refusal,
    and the failure line reaches the agent so it knows what to fix."""
    ws = _configured(swept, tmp_path,
                     'check "the widget is not built — run make, then commit" false')
    _, entry = _entry(ws)
    loaded = load_processors([entry], workspace_path=str(ws), config_root=None)

    outcome = _drive(ws, loaded)
    assert outcome.has_fatal, "a failing acceptance check did not block"
    assert loaded[0].refusals == 1, (
        f"expected exactly one refusal per invocation, got "
        f"{loaded[0].refusals}")
    blob = " ".join(msg for _, msg in outcome.failed)
    assert "the widget is not built" in blob, (
        f"the check's own message did not reach the agent; got {blob!r}")


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the emitted checks script is POSIX sh")
def test_a_passing_check_accepts_the_completion(swept, tmp_path):
    """The gate has to be passable, or it is not a gate."""
    ws = _configured(swept, tmp_path, 'check "all good" true')
    _, entry = _entry(ws)
    loaded = load_processors([entry], workspace_path=str(ws), config_root=None)

    outcome = _drive(ws, loaded)
    assert not outcome.has_fatal, (
        f"the gate refused a completion although every check passed: "
        f"{[m for _, m in outcome.failed]}")
    assert loaded[0].refusals == 0


@pytest.mark.skipif(sys.platform == "win32",
                    reason="the emitted checks script is POSIX sh")
def test_the_emitted_ceiling_actually_terminates_the_loop(swept, tmp_path):
    """A gate that always refuses must STOP refusing at the declared ceiling.

    Bounded by iteration count rather than by wall clock, deliberately: the
    failure being guarded against is a loop that does not stop on its own, and
    a test that waited for it to stop would hang rather than fail.  The
    observed incident behind #768 was seven refusals in 156 seconds on the
    same two errors, ending with the arm's whole budget spent.
    """
    ws = _configured(swept, tmp_path, 'check "never satisfied" false')
    _, entry = _entry(ws)
    assert entry.max_refusals == 3
    loaded = load_processors([entry], workspace_path=str(ws), config_root=None)

    blocked = []
    for _ in range(entry.max_refusals + 3):
        blocked.append(_drive(ws, loaded).has_fatal)

    assert blocked[:entry.max_refusals] == [True] * entry.max_refusals, (
        f"the gate stopped blocking before its ceiling: {blocked}")
    assert not any(blocked[entry.max_refusals:]), (
        f"the gate was still refusing after max_refusals={entry.max_refusals} "
        f"({blocked}) — on_exhausted: allow lets the unfinished completion "
        f"stand, because a FAIL verdict carries information and a BLOCKED arm "
        f"carries none")


# --------------------------------------------------------------------------
# 4. The set is coherent with the client, the docs and the generator.
# --------------------------------------------------------------------------

def test_the_client_names_the_profile_that_was_written_beside_it(swept):
    """The JOBS matrix must point at the emitted gate profile.

    The whole complaint in #772 is that the pieces arrive unrelated.  A client
    still naming ``"your-profile"`` while a gate profile sits next to it is
    that complaint intact.
    """
    client = (swept / "run_sweep.py").read_text("utf-8")
    assert f'"{GATE}"' in client, (
        f"the emitted run_sweep.py does not name the {GATE!r} profile")
    assert '"your-profile"' not in client, (
        "the emitted run_sweep.py still carries the placeholder profile "
        "although a real gate profile was written beside it")


def test_no_gate_emits_none_of_the_set(tmp_path):
    """``--no-gate`` is all-or-nothing: a partial set is worse than none.

    Half a gate is the failure mode this archetype exists to remove — a
    processor with no checks script, or a profile whose schema is missing,
    are each inert in a way that reports as something else.
    """
    ws = tmp_path / "ungated"
    assert _run_new(_args(ws, no_gate=True)) == 0
    for path in ("acceptance.sh",
                 f".jaato/scripts/processors/{GATE}.py",
                 f".jaato/completion_schemas/{GATE}.json",
                 f".jaato/profiles/{GATE}.yaml"):
        assert not (ws / path).exists(), f"--no-gate still emitted {path}"
    assert '"your-profile"' in (ws / "run_sweep.py").read_text("utf-8"), (
        "without a gate profile the JOBS matrix must keep its placeholder")


def test_the_gate_name_flag_renames_the_whole_set(tmp_path):
    """One name across the four files, so they read as one thing."""
    ws = tmp_path / "renamed"
    assert _run_new(_args(ws, gate_name="grade")) == 0
    for path in (".jaato/scripts/processors/grade.py",
                 ".jaato/completion_schemas/grade.json",
                 ".jaato/profiles/grade.yaml"):
        assert (ws / path).is_file(), f"--gate-name did not produce {path}"
    assert '"grade"' in (ws / "run_sweep.py").read_text("utf-8")


def test_every_gated_archetype_documents_its_gate_files():
    """``build.GATED_ARCHETYPES`` and the archetype docs cannot disagree.

    The registry decides what is emitted and the docs decide what ``explain``
    and ``--dry-run`` promise.  Two lists naming the same thing drift; this is
    the comparison that stops them, and it is the shape of drift that produced
    #716 (an archetype count SPELLED rather than counted).
    """
    gate_globs = {ef.glob() for ef in A._GATE_FILES}
    for name in build.GATED_ARCHETYPES:
        doc = A.resolve(name)
        assert doc is not None, f"{name} is gated but has no archetype doc"
        declared = {ef.glob() for ef in doc.writes}
        missing = gate_globs - declared
        assert not missing, (
            f"{name} emits a completion gate but its doc does not declare "
            f"{sorted(missing)} — `explain archetype {name}` and `--dry-run` "
            f"would both under-report what lands in the workspace")

    # The reverse direction is about the SET, not about individual paths:
    # `new processor` legitimately writes .jaato/scripts/processors/<n>.py,
    # which is one of the gate's four.  What no ungated archetype may claim is
    # the whole set — that would be a doc promising a gate its generator never
    # writes.
    for name, doc in A.ARCHETYPES.items():
        if name in build.GATED_ARCHETYPES:
            continue
        declared = {ef.glob() for ef in doc.writes}
        assert not gate_globs.issubset(declared), (
            f"{name} documents the whole completion-gate set but "
            f"build.GATED_ARCHETYPES does not list it, so `new {name}` never "
            f"writes it")


def test_the_generator_refuses_to_vouch_for_a_gate_it_did_not_write(swept, tmp_path):
    """Re-running ``new`` keeps an edited acceptance.sh rather than reverting it.

    These are files an author is expected to EDIT — the checks above all — so
    a re-run that silently restored the template would destroy the work the
    archetype exists to prompt.  The files are kept and the skip is reported.
    """
    import shutil

    ws = tmp_path / "rerun"
    shutil.copytree(swept, ws)
    edited = "# my own checks\n"
    (ws / "acceptance.sh").write_text(edited)
    # Re-running to regenerate the CLIENT, which is the case that reaches the
    # gate's skip path: with run_sweep.py still present `new` refuses before
    # writing anything, so nothing would be exercised.
    (ws / "run_sweep.py").unlink()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = build.run(_args(ws))
    assert rc == 0, buf.getvalue()
    assert (ws / "acceptance.sh").read_text("utf-8") == edited, (
        "re-running `new sweep` clobbered an edited acceptance.sh")
    assert "kept" in buf.getvalue(), (
        f"the kept file was not reported; a silent skip reads as a rewrite:\n"
        f"{buf.getvalue()}")
