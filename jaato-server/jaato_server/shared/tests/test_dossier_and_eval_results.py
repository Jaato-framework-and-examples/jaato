"""Guard: the Annex IV dossier and the accuracy section it is filled from.

jaato #1121 (the dossier + the Article 25(4) component pack) and #1124 (the
``jaato-eval`` adapter that fills Annex IV §4).

Both deliverables are DOCUMENTS, which is what makes them easy to get wrong in
a way nothing notices.  A generated legal document fails silently in three
directions, and each has its own assertion here:

1. **By omission.**  A section the framework cannot fill is dropped, and an
   absent section in a legal document reads as *nothing to declare*.  Every
   Annex IV heading must be present, and ``new dossier`` reads its own output
   back to prove it.
2. **By assertion.**  A fact the framework does not know is printed anyway --
   ``risk_class: minimal`` for a profile that declared none is a risk
   determination made on the provider's behalf, and Article 6(4) makes that
   determination theirs.
3. **By quotation.**  A number is rendered without the limits of the
   instrument that produced it.  ``jaato-eval``'s judge grader is one model
   scoring another; the caveat travels in the results file and is rendered
   verbatim, because a consumer that wrote its own copy would go on quoting it
   after the limit was fixed, or miss one added later.

The adapter reads a DECLARED FORMAT rather than importing the engine
(``jaato_eval`` imports ``jaato_sdk`` and nothing else from this tree), so the
version field is the whole of what stops it rendering a file it cannot read --
and an absent version is an unknown version, not a version 1 record.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path

import pytest

from jaato_server.shared.scaffold import archetypes as A
from jaato_server.shared.scaffold import build, dossier
from jaato_server.shared.scaffold import eval_results as ER
from jaato_server.shared.tests.reversion import Reversion


ROOT = Path(__file__).resolve().parents[4]


REVERSIONS = [
    # -------------------------------------------------- #1121, the dossier
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/dossier.py",
        find="""    return tuple(f"{number}. {title}"
                 for number, title in ANNEX_IV_SECTIONS
                 if section_heading(number, title) not in text)""",
        replace="""    return ()""",
        test="test_the_read_back_notices_a_dropped_heading",
        because="the emit-then-check reporting every document as complete -- "
                "a read-back that cannot fail is the silent omission it was "
                "written to catch, wearing the fix as a disguise",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/build.py",
        find="    missing = _dossier.missing_sections(text) if not component else ()",
        replace="    missing = ()",
        test="test_new_dossier_refuses_its_own_broken_render",
        because="`new dossier` trusting its own renderer instead of reading "
                "the document back",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/dossier.py",
        find="""        return _todo(
            "This profile declares no `regulatory:` block, so its intended "
            "purpose and risk class are UNDECLARED — not `minimal`. """,
        replace="""        return _todo(
            "This profile declares no `regulatory:` block, so its risk "
            "class is `minimal`. """,
        test="test_an_undeclared_risk_class_is_never_printed_as_minimal",
        because="the framework making the Article 6(4) risk determination on "
                "the provider's behalf -- absent is UNDECLARED, and a "
                "document that says `minimal` is one somebody will rely on",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/dossier.py",
        find="""    for title, detail in NON_GUARANTEES:
        parts += [f"**{title}.** {detail}", ""]""",
        replace="""    for title, detail in ():
        parts += [f"**{title}.** {detail}", ""]""",
        test="test_the_component_pack_says_what_it_does_not_guarantee",
        because="a 25(4) pack listing only guarantees -- the limits are the "
                "half a provider relies on when deciding what they still owe",
    ),

    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/dossier.py",
        find="                cwd=str(checkout),\n",
        replace="",
        because=(
            "the framework's commit read from the process CWD -- an "
            "operator generating a dossier from their own project gets "
            "THEIR commit stamped as the framework's, in the one field "
            "that exists to make the document auditable later"
        ),
        test="test_the_commit_stamp_describes_the_FRAMEWORK",
    ),

    # ---------------------------------------------- #1124, the eval adapter
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/eval_results.py",
        find="""    version = seen[0]
    if version not in SUPPORTED_VERSIONS:""",
        replace="""    version = seen[0]
    if False:""",
        test="test_an_unknown_results_version_is_refused_by_name",
        because="rendering a results file whose format this reader does not "
                "know -- a half-understood accuracy table looks exactly like "
                "a complete one",
    ),
    Reversion(
        target="jaato-server/jaato_server/shared/scaffold/eval_results.py",
        find="""    for caveat in _caveats(records):""",
        replace="""    for caveat in ():""",
        test="test_the_harness_caveat_is_rendered_verbatim",
        because="an accuracy number rendered without the limits of the "
                "instrument that produced it -- the number is the part a "
                "dossier reader quotes",
    ),
]


# ----------------------------------------------------------------- fixtures

def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    defaults = dict(archetype=A.DOSSIER, workspace=None, provider="nebius",
                    model="m", set=None, agents=None, force=False, json=False,
                    recoverable=False, dry_run=False, secrets=None,
                    secret_path=None, transport="ipc", url=None, token=None,
                    ca=None, name=None, profile=None, component=False,
                    eval_results=None)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def _run(args):
    """Run ``new`` with its chatter captured."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = build.run(args)
    return rc, buf.getvalue()


@pytest.fixture
def workspace(tmp_path) -> Path:
    """A workspace declaring one profile, written by hand.

    Hand-written rather than scaffolded so the test states exactly what the
    dossier is reading; the scaffolded shape is covered by
    ``test_scaffold_archetype_docs``.
    """
    ws = tmp_path / "ws"
    (ws / ".jaato" / "profiles").mkdir(parents=True)
    (ws / ".jaato" / "profiles" / "worker.yaml").write_text(
        "name: worker\n"
        "description: a worker\n"
        "provider: nebius\n"
        "model: some-model\n"
        "plugins: [cli]\n",
        encoding="utf-8")
    return ws


def _results(tmp_path, records) -> Path:
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8")
    return path


JUDGE_CAVEAT = ("The `judge` grader is one language model scoring another's "
                "output, and it is not a calibrated instrument.")


def _arm(state="PASS", task="t", pset="s", graders=("script:x",),
         caveats=(), version="1", repeat=0):
    return {
        "results_version": version,
        "caveats": list(caveats),
        "arm_id": f"{task}@{pset}#{repeat}", "task_id": task,
        "profile_set": pset, "repeat": repeat, "state": state,
        "verdicts": [{"grader_id": g, "claim": "c", "state": state}
                     for g in graders],
        "provenance": {"jaato_sdk_version": "9.9.9",
                       "jaato_sdk_path": "/x/jaato_sdk/__init__.py"},
    }


# ================================================== #1121 -- the dossier

def test_every_annex_iv_heading_is_present(workspace):
    text = dossier.render_dossier("worker", str(workspace))
    for number, title in dossier.ANNEX_IV_SECTIONS:
        assert dossier.section_heading(number, title) in text, (
            f"Annex IV point {number} is missing. An absent section in a "
            f"legal document reads as *nothing to declare*, which is the one "
            f"thing this generator must never say by accident")


def test_a_section_the_framework_cannot_fill_says_so(workspace):
    """Not silence, and not an empty heading: a named TODO."""
    text = dossier.render_dossier("worker", str(workspace))
    # §8 (the declaration of conformity) is a document the provider signs;
    # nothing in the tree generates it, so it is pure TODO.
    body = text.split(dossier.section_heading("8", dossier.ANNEX_IV_SECTIONS[7][1]))[1]
    body = body.split("## 9.")[0]
    assert "TODO — the framework cannot supply this" in body
    assert "Article 47" in body or "declaration of conformity" in body.lower()


def test_the_read_back_notices_a_dropped_heading(workspace):
    """``missing_sections`` is a real check, not a decoration."""
    text = dossier.render_dossier("worker", str(workspace))
    assert dossier.missing_sections(text) == ()

    number, title = dossier.ANNEX_IV_SECTIONS[4]
    mutilated = text.replace(dossier.section_heading(number, title), "## gone")
    assert dossier.missing_sections(mutilated) == (f"{number}. {title}",), (
        "a heading removed from the rendering must be NAMED -- a read-back "
        "that answers 'fine' to a broken document is worse than none")


def test_new_dossier_refuses_its_own_broken_render(workspace, monkeypatch):
    """The generator does not trust itself; it reads the document back."""
    good = dossier.render_dossier("worker", str(workspace))
    number, title = dossier.ANNEX_IV_SECTIONS[2]
    broken = good.replace(dossier.section_heading(number, title), "## dropped")

    monkeypatch.setattr(dossier, "render_dossier",
                        lambda *a, **k: broken)
    rc, out = _run(_args(workspace=str(workspace), profile="worker"))
    assert rc == 1, "a document missing a heading must not be reported as written"
    assert f"{number}. {title}" in out


def test_an_undeclared_risk_class_is_never_printed_as_minimal(workspace):
    """Article 6(4) makes the determination the provider's, not ours."""
    text = dossier.render_dossier("worker", str(workspace))
    assert "UNDECLARED" in text
    assert "risk class is `minimal`" not in text
    assert "Article 6(4)" in text


def test_the_computed_sections_are_stamped(workspace):
    """A fact about a tree is a fact about a COMMIT, and about a date."""
    text = dossier.render_dossier("worker", str(workspace))
    assert "Computed from the installed framework at commit" in text
    assert "Regenerate" in text


def test_the_commit_stamp_describes_the_FRAMEWORK(tmp_path, monkeypatch):
    """Not whatever repository the operator was standing in.

    ``git rev-parse`` inherits the process CWD, so an operator generating
    a dossier from their own project -- with jaato installed from a wheel
    -- had THEIR commit stamped as the framework's, in the one field that
    exists to make the document auditable six months later.  A wrong fact
    presented as provenance is worse than an absent one.
    """
    from jaato_server.shared.scaffold import dossier as D

    foreign = tmp_path / "someone-elses-repo"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    D._commit.cache_clear()
    try:
        stamp = D._commit()
    finally:
        D._commit.cache_clear()

    checkout = D._framework_checkout()
    assert checkout is not None, "this tree IS a checkout; the probe is broken"
    assert checkout.resolve() == ROOT.resolve()
    assert stamp != "unknown"
    assert str(foreign) not in stamp


def test_the_commit_is_resolved_once_per_run():
    """One subprocess, not one per section.

    ``_stamp()`` is called per Annex IV section, so an unmemoised probe
    spawned nine `git` processes -- each with its own timeout -- to
    answer a question that cannot change during a run.
    """
    from jaato_server.shared.scaffold import dossier as D

    assert hasattr(D._commit, "cache_clear"), (
        "_commit must be memoised for the run")


def test_a_wheel_install_states_versions_rather_than_borrowing_a_commit(
        monkeypatch):
    """No checkout means no commit, and no commit means say so.

    Falling through to a bare `git rev-parse` is exactly how somebody
    else's commit ends up in the document; the honest answer a wheel
    install can give is which distributions are installed.
    """
    from jaato_server.shared.scaffold import dossier as D

    monkeypatch.setattr(D, "_framework_checkout", lambda: None)
    D._commit.cache_clear()
    try:
        stamp = D._commit()
    finally:
        D._commit.cache_clear()
    assert stamp.startswith("installed ") or stamp == "unknown"
    assert "jaato" in stamp or stamp == "unknown"


def test_the_dossier_never_renders_a_persona():
    """Side-effect free, like ``validate``.

    Rendering a persona runs its ``{{!py:...}}`` prefetch scripts, which is
    arbitrary code with side effects; a document generator must locate files,
    never execute them.  Asserted on the source because the failure is a call
    that is not there, and a behavioural test can only prove that one
    particular profile did not reach it.
    """
    src = (ROOT / "jaato-server" / "jaato_server" / "shared" / "scaffold" / "dossier.py").read_text()
    tree = ast.parse(src)
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    for forbidden in ("resolve_agent", "render_agent", "render_persona"):
        assert forbidden not in called, (
            f"dossier.py calls {forbidden}() -- generating a document must "
            f"not execute a persona's prefetch scripts")


# ------------------------------------------------ the component pack

def test_the_component_pack_denies_the_foss_carve_out():
    text = dossier.render_component_pack()
    assert "BUSL-1.1" in text
    assert "25(4)" in text
    assert "written agreement" in text


def test_every_guarantee_names_its_enforcer():
    """A guarantee with no enforcer is a claim, and a claim in a 25(4) pack
    is what a provider relies on."""
    assert dossier.GUARANTEES
    for title, detail in dossier.GUARANTEES:
        assert title.strip()
        assert "`" in detail, (
            f"guarantee {title!r} names no enforcing mechanism -- every entry "
            f"must point at the thing in the tree that holds it")


def test_the_component_pack_says_what_it_does_not_guarantee():
    text = dossier.render_component_pack()
    assert dossier.NON_GUARANTEES
    for title, _detail in dossier.NON_GUARANTEES:
        assert title in text, (
            f"the pack omits the non-guarantee {title!r}; a supplier's "
            f"document that lists only its guarantees invites reliance on "
            f"the parts it does not hold")


# ------------------------------------------------------- the CLI surface

def test_neither_flag_is_refused_and_writes_nothing(workspace):
    rc, out = _run(_args(workspace=str(workspace)))
    assert rc == 2
    assert "--profile" in out and "--component" in out
    assert not (workspace / "docs").exists()


def test_both_flags_together_are_refused(workspace):
    rc, out = _run(_args(workspace=str(workspace), profile="worker",
                         component=True))
    assert rc == 2
    assert "two documents for two readers" in out


def test_eval_results_on_the_component_pack_is_refused(workspace, tmp_path):
    rc, out = _run(_args(workspace=str(workspace), component=True,
                         eval_results=str(tmp_path / "r.jsonl")))
    assert rc == 2
    assert "accuracy section" in out


def test_a_profile_that_does_not_resolve_writes_nothing(workspace):
    rc, out = _run(_args(workspace=str(workspace), profile="ghost"))
    assert rc == 2
    assert "ghost" in out
    assert not (workspace / "docs").exists(), (
        "a dossier generated for the wrong system is worse than none, so a "
        "refusal must leave nothing behind")


def test_the_conditional_files_appear_under_their_condition(workspace):
    """Both of this archetype's declared files are conditional, so the
    archetype-docs guard skips them.  Assert each condition produces its
    file, or 'conditional' becomes a place to hide undocumented output."""
    assert _run(_args(workspace=str(workspace), profile="worker"))[0] == 0
    assert (workspace / "docs" / "annex-iv-worker.md").is_file()

    assert _run(_args(workspace=str(workspace), component=True))[0] == 0
    assert (workspace / "docs" / "jaato-component-pack.md").is_file()


def test_regenerating_an_existing_document_needs_force(workspace):
    assert _run(_args(workspace=str(workspace), profile="worker"))[0] == 0
    rc, out = _run(_args(workspace=str(workspace), profile="worker"))
    assert rc == 1
    assert "--force" in out
    assert _run(_args(workspace=str(workspace), profile="worker",
                      force=True))[0] == 0


# ============================================ #1124 -- the accuracy section

def test_the_harness_caveat_is_rendered_verbatim(tmp_path):
    path = _results(tmp_path, [
        _arm(graders=("judge:rubric",), caveats=(JUDGE_CAVEAT,))])
    text = "\n".join(ER.render_section(str(path)))
    assert JUDGE_CAVEAT in text, (
        "the harness's caveat must appear WORD FOR WORD: a summary of it is "
        "a second copy of a fact, and it is the copy that gets quoted after "
        "the limit it describes has moved")


def test_the_caveat_is_not_written_here():
    """This module must carry no caveat text of its own.

    A hand-written warning about LLM judges in the consumer outlives the
    limit it describes.  The producer measured it; the producer states it.
    """
    src = (ROOT / "jaato-server" / "jaato_server" / "shared" / "scaffold"
           / "eval_results.py").read_text()
    body = src.split('"""', 2)[-1]  # past the module docstring
    for phrase in ("language model scoring", "not a calibrated instrument",
                   "one run in four"):
        assert phrase not in body, (
            f"eval_results.py spells out {phrase!r} -- the caveat belongs to "
            f"the harness that measured it, and a copy here would be quoted "
            f"after the harness's own had changed")


def test_the_adapter_does_not_import_the_engine():
    """``jaato_eval`` imports ``jaato_sdk`` and nothing else from this tree;
    a consumer inside ``shared`` importing it would run that rule backwards,
    and would make a dossier ungeneratable wherever the harness is absent."""
    src = (ROOT / "jaato-server" / "jaato_server" / "shared" / "scaffold"
           / "eval_results.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("jaato_eval"), (
                "the adapter reads a declared format; it must not import the "
                "engine that writes it")


def test_an_unknown_results_version_is_refused_by_name(tmp_path):
    path = _results(tmp_path, [_arm(version="42")])
    with pytest.raises(ER.EvalResultsError) as exc:
        ER.render_section(str(path))
    message = str(exc.value)
    assert "'42'" in message, "the refusal must name the version it found"
    assert "'1'" in message, "and the versions this reader knows"
    assert "docs/eval-results.md" in message


def test_a_record_with_no_version_is_an_unknown_version(tmp_path):
    """Absent is not version 1.

    The field has been written since the format was declared, so its absence
    says the file predates the contract -- reading it as today's format is
    exactly the guess this refusal exists to prevent.
    """
    path = _results(tmp_path, [{"task_id": "t", "state": "PASS"}])
    with pytest.raises(ER.EvalResultsError) as exc:
        ER.render_section(str(path))
    assert "no results_version" in str(exc.value)


def test_records_disagreeing_about_their_version_are_refused(tmp_path):
    path = _results(tmp_path, [_arm(), _arm(version="2", repeat=1)])
    with pytest.raises(ER.EvalResultsError) as exc:
        ER.render_section(str(path))
    assert "mixes results_version" in str(exc.value)


def test_a_file_with_no_records_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ER.EvalResultsError) as exc:
        ER.render_section(str(path))
    assert "no readable results record" in str(exc.value)


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(ER.EvalResultsError):
        ER.render_section(str(tmp_path / "absent.jsonl"))


def test_blocked_arms_leave_the_denominator(tmp_path):
    path = _results(tmp_path, [
        _arm(state="PASS"), _arm(state="FAIL", repeat=1),
        _arm(state="BLOCKED", repeat=2)])
    text = "\n".join(ER.render_section(str(path)))
    assert "| 3 | 2 | 50% |" in text, (
        "three arms, two exercised, one pass -- 50%, not 33%: a blocked arm "
        "was never exercised, so it is neither a pass nor a failure of the "
        "system under test")


def test_a_cell_where_everything_blocked_has_no_pass_rate(tmp_path):
    path = _results(tmp_path, [_arm(state="BLOCKED")])
    text = "\n".join(ER.render_section(str(path)))
    assert "| — |" in text
    assert "| 0% |" not in text, (
        "0% says 'it always failed'; the truth is 'we never found out', and "
        "the two must not print the same")


def test_every_row_leaves_the_threshold_to_the_provider(tmp_path):
    path = _results(tmp_path, [_arm(), _arm(task="u", repeat=1)])
    lines = ER.render_section(str(path))
    rows = [ln for ln in lines if ln.startswith("| ") and "---" not in ln][1:]
    assert len(rows) == 2
    for row in rows:
        assert row.rstrip().endswith("| **TODO** |")
    assert any("Article 15(3)" in ln for ln in lines), (
        "the TODO must name the Article that makes the threshold the "
        "provider's call")


def test_a_refused_file_empties_the_section_rather_than_faking_it(workspace,
                                                                  tmp_path):
    """The dossier renders the refusal; it does not quietly drop §4."""
    bad = _results(tmp_path, [_arm(version="42")])
    text = dossier.render_dossier("worker", str(workspace),
                                  eval_results=str(bad))
    assert "Refused:" in text
    assert "'42'" in text
    assert dossier.missing_sections(text) == ()


# --------------------------------------------- the producer's half

def test_the_engine_stamps_the_version_and_the_caveats():
    """``ArmResult.to_dict`` writes both contract fields.

    Read from the source rather than by importing ``jaato_eval``: the harness
    is a separate distribution and need not be installed for this guard to
    run -- and a guard that silently skips where the package is absent is a
    guard that does not run in the environment that matters.
    """
    src = (ROOT / "jaato-eval" / "jaato_eval" / "arm.py").read_text()
    assert '"results_version": RESULTS_FORMAT_VERSION' in src
    assert '"caveats": caveats_for(' in src


def test_the_producer_and_this_reader_agree_on_today_s_version():
    src = (ROOT / "jaato-eval" / "jaato_eval" / "results_format.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "RESULTS_FORMAT_VERSION"):
            declared = node.value.value
            break
    else:  # pragma: no cover - the constant is the contract
        pytest.fail("jaato_eval declares no RESULTS_FORMAT_VERSION")
    assert declared in ER.SUPPORTED_VERSIONS, (
        f"the harness writes {declared!r} and this reader accepts "
        f"{ER.SUPPORTED_VERSIONS} -- every file the harness produces would "
        f"be refused, which is the check succeeding at being useless")


def test_the_format_is_documented_as_a_contract():
    doc = (ROOT / "docs" / "eval-results.md").read_text()
    for token in ("results_version", "caveats", "verbatim", "BLOCKED"):
        assert token in doc
