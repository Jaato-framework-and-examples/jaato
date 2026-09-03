"""Guard: every archetype ``new`` accepts is documented, and the docs are true.

The gap this closes (jaato #716): ``explain`` documented the framework's
INPUTS and nothing documented what ``new`` PRODUCES, so the only way to learn
what an archetype writes was to run it against a throwaway directory and diff,
or to read ``build.py`` / ``_client_templates.py``.  A generator whose output
is undocumented cannot be trusted sight-unseen — an agent briefed to prefer
``jaato-scaffold new profile-set`` over hand-writing spent ~20 minutes and 250+
tool calls reverse-engineering it and never ran it.

The gap AROSE by drift, and the banner is where it shows: ``5aa82e1`` (#624)
shipped FIVE client templates under a literal reading "4 client archetypes"
(``host-tools`` was uncounted from the first commit), and ``ad016d8`` (#649)
added ``sweep`` without touching that literal.  The counter was never
incremented — it was simply never right, and with no drill-down behind it
nothing made that visible.  So the guard is not "does a doc scope exist" but
"does the doc still match the generator":

1. every archetype ``new`` dispatches on has an entry in ``ARCHETYPES``;
2. a real ``new`` run writes nothing the entry does not declare, and writes
   every file the entry declares unconditionally;
3. ``--dry-run`` rehearses EXACTLY what the real run writes (same paths, same
   create/update actions) and touches nothing;
4. the banner counter is counted, not spelled.

Adding a template to ``TEMPLATES`` without documenting it fails (1); changing
what an archetype emits without updating its doc fails (2).
"""

from __future__ import annotations

import argparse
import io
import contextlib

import pytest

from shared.scaffold import archetypes as A
from shared.scaffold import build, explain
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Put the defect back: spell the archetype count instead of counting it.
#: That single literal is the whole of how #716 shipped — the banner said
#: "4 client archetypes" while `new` accepted six client templates plus
#: profile-set — so it is the reversion this guard must notice.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find='f"{len(GC)} gc strategies   {n_arch} archetypes\\n\\n"',
        replace='f"{len(GC)} gc strategies   4 client archetypes\\n\\n"',
        test="test_the_banner_counter_is_counted_not_spelled",
        because="the overview banner advertising a hardcoded archetype count "
                "— wrong from the commit that wrote it (host-tools was never "
                "in the four), and never revisited when sweep landed",
    ),
]


# The provider used for every generation here.  Any installed provider works —
# nothing in this module asserts on provider-specific output.
PROVIDER = "nebius"


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    defaults = dict(archetype=None, workspace=None, provider=PROVIDER,
                    model="m", set=None, agents=None, force=False, json=False,
                    recoverable=False, dry_run=False, secrets=None,
                    secret_path=None, transport="ipc", url=None, token=None,
                    ca=None, name=None)
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def _invocation(name: str, ws) -> argparse.Namespace:
    """A minimal VALID ``new`` invocation for archetype *name*."""
    if name == A.PROFILE_SET:
        return _args(archetype=name, workspace=str(ws), set="s1",
                     agents="alpha,beta")
    if name == A.PROCESSOR:
        return _args(archetype=name, workspace=str(ws), name="gate")
    return _args(archetype=name, workspace=str(ws))


def _run(args) -> int:
    """Run ``new`` with its chatter swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return build.run(args)


def _written(ws) -> set:
    """Every file *scaffolded* under *ws*, workspace-relative, POSIX-separated.

    ``__pycache__`` is excluded: the client archetypes' emit-then-check runs
    ``py_compile`` on the script they just wrote, and CPython drops the .pyc
    beside it.  That is a byproduct of checking the output, not part of it, and
    documenting it would tell the reader nothing they need.
    """
    return {str(p.relative_to(ws)).replace("\\", "/")
            for p in ws.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> dict:
    """Run every archetype ONCE, real and rehearsed, and share the results.

    Module-scoped because each ``new`` call resolves providers and (for a
    profile-set) re-runs the validator — seconds apiece.  The per-archetype
    assertions below all read from this one pass.

    Returns:
        ``{archetype: {"ws", "files", "real_plan", "dry_plan", "dry_ws"}}`` —
        the real run's workspace and the files in it, both runs' plan entries,
        and the (never-created) workspace the rehearsal was pointed at.
    """
    root = tmp_path_factory.mktemp("archetypes")
    out = {}
    for name in sorted(A.ARCHETYPES):
        ws = root / name
        dry_ws = root / f"{name}_dry"
        dry_args = _invocation(name, dry_ws)
        dry_args.dry_run = True
        out[name] = {
            "ws": ws,
            "dry_ws": dry_ws,
            "dry_plan": _capture_plan(dry_args),
            "real_plan": _capture_plan(_invocation(name, ws)),
        }
        out[name]["files"] = _written(ws)
    return out


# ------------------------------------------------------- 1. coverage

def test_every_accepted_archetype_is_documented():
    """`new` dispatches on the registry, so an undocumented one cannot exist.

    Stated as an assertion anyway: the dispatch could be widened back to a
    literal list, which is how six archetypes came to be accepted while
    ``explain`` advertised four.
    """
    for name in A.CLIENT_ARCHETYPES:
        assert name in A.ARCHETYPES, (
            f"archetype '{name}' is in TEMPLATES (so `new {name}` runs) but has "
            f"no entry in scaffold/archetypes.py — add one; that omission is "
            f"the whole of jaato #716")
    assert A.PROFILE_SET in A.ARCHETYPES
    for alias in A.PROFILE_SET_ALIASES:
        assert A.resolve(alias) is A.ARCHETYPES[A.PROFILE_SET]
    assert A.resolve(None) is A.ARCHETYPES[A.PROFILE_SET], (
        "`new` with no archetype builds a profile-set; the docs must resolve "
        "the same way")
    assert A.resolve("no-such-archetype") is None


@pytest.mark.parametrize("name", sorted(A.ARCHETYPES))
def test_each_doc_is_complete(name):
    """No entry may be a stub — a stub reads as documentation and answers
    nothing, which is the state this guard exists to prevent."""
    doc = A.ARCHETYPES[name]
    assert doc.summary and doc.summary.strip()
    assert doc.requires, f"{name}: no required flags declared"
    assert doc.writes, f"{name}: declares no output — `new` writes something"
    assert doc.check, f"{name}: no self-check declared"
    assert doc.next_steps, f"{name}: no next step declared"
    for ef in doc.writes:
        assert ef.what.strip(), f"{name}: {ef.path} has no purpose line"
        assert ef.status in ("generated", "fill-in", "edit", "merged"), (
            f"{name}: {ef.path} has unknown status {ef.status!r}")


@pytest.mark.parametrize("name", sorted(A.ARCHETYPES))
def test_explain_renders_every_archetype(name):
    """`explain archetype <name>` must render for every accepted name, so a
    reader following the drill-down line never hits a dead end."""
    data, text = explain.archetype(name)
    assert "error" not in data
    assert data["name"] == name
    doc = A.ARCHETYPES[name]
    rendered = {w["path"]: w for w in data["writes"]}
    assert len(rendered) == len(doc.writes), f"{name}: rendered no file list"
    for ef in doc.writes:
        path = ef.render_path(archetype=name, set="<set>",
                              agent="<agent>", name="<name>")
        assert path in rendered, f"{name}: {path} missing from the rendering"
        assert rendered[path]["what"] == ef.what
        assert rendered[path]["status"] == ef.status
        # the human rendering wraps prose, so assert on the path (never wrapped)
        assert path in text
    _, listing = explain.archetypes()
    assert name in listing


def test_the_banner_counter_is_counted_not_spelled():
    """The literal '4 client archetypes' outlived two new archetypes.

    Asserting the rendered number tracks the registry is what makes the next
    addition self-documenting instead of silently wrong.
    """
    _, text = explain.overview()
    assert f"{len(A.ARCHETYPES)} archetypes" in text
    assert "4 client archetypes" not in text
    assert "explain archetypes" in text, (
        "the overview must offer the drill-down; a counter with nothing behind "
        "it is exactly how #716 arose")


# --------------------------------------- 2. the docs match what `new` writes

@pytest.mark.parametrize("name", sorted(A.ARCHETYPES))
def test_a_real_run_writes_only_documented_files(generated, name):
    """Every file that lands is declared, and every unconditional declaration
    lands.  This is the assertion that rots if an archetype's output changes
    and its doc does not."""
    doc = A.ARCHETYPES[name]
    actual = generated[name]["files"]
    assert actual, "the run wrote nothing"

    for rel in sorted(actual):
        assert A.documents(doc, rel) is not None, (
            f"`new {name}` wrote {rel} but scaffold/archetypes.py does not "
            f"declare it — a reader of `explain archetype {name}` would be "
            f"surprised by that file")

    for ef in doc.writes:
        if ef.when:
            continue  # conditional — not required to appear
        assert any(A.documents(doc, rel) is ef for rel in actual), (
            f"scaffold/archetypes.py declares {ef.path} unconditionally for "
            f"'{name}', but `new {name}` did not write it")


def test_the_conditional_files_appear_under_their_condition(tmp_path):
    """profile-set's two conditional files — .gitignore (env/none secrets) and
    .jaato/scaffold.json (an explicit --secrets) — are declared conditional, so
    the check above skips them.  Assert the condition actually produces them,
    or 'conditional' becomes a place to hide undocumented output."""
    ws = tmp_path / "cond"
    assert _run(_args(archetype=A.PROFILE_SET, workspace=str(ws), set="s1",
                      agents="alpha", secrets="env")) == 0
    actual = _written(ws)
    assert ".gitignore" in actual
    assert ".jaato/scaffold.json" in actual


# ------------------------------------------ 3. --dry-run tells the truth

@pytest.mark.parametrize("name", sorted(A.ARCHETYPES))
def test_dry_run_writes_nothing(generated, name):
    """Not even the workspace directory: a rehearsal is run against a real
    workspace, so creating one is a side effect the reader did not ask for."""
    dry_ws = generated[name]["dry_ws"]
    assert not dry_ws.exists(), (
        f"`new {name} --dry-run` created {dry_ws} — a rehearsal must not touch "
        f"the filesystem")


@pytest.mark.parametrize("name", sorted(A.ARCHETYPES))
def test_dry_run_predicts_the_real_run_exactly(generated, name):
    """The rehearsal and the real run must agree on paths AND on create-vs-
    update — 'will this clobber my .env?' is the question --dry-run is asked.

    Compared as PLANS rather than as printed text: the plan is the single
    object both paths build, so a divergence here means the two paths really
    would produce different trees, not that a caption changed.
    """
    assert generated[name]["dry_plan"] == generated[name]["real_plan"]


def _capture_plan(args) -> list:
    """Run ``new`` and return its plan entries ``[(relpath, action), ...]``."""
    seen = []
    real_init = build._Plan.__init__

    def spy(self, ws, doc=None, *, dry_run=False):
        real_init(self, ws, doc, dry_run=dry_run)
        seen.append(self)

    build._Plan.__init__ = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = build.run(args)
    finally:
        build._Plan.__init__ = real_init
    assert rc == 0, f"new returned {rc}"
    assert len(seen) == 1, "expected exactly one plan per invocation"
    return list(seen[0].entries)


def test_dry_run_over_an_existing_workspace_reports_updates(tmp_path):
    """A second `new` into a populated workspace appends rather than clobbers;
    the rehearsal must say so, marking those entries 'update'."""
    ws = tmp_path / "existing"
    ws.mkdir()
    (ws / ".env").write_text("FOO=1\n", encoding="utf-8")
    args = _args(archetype=A.PROFILE_SET, workspace=str(ws), set="s1",
                 agents="alpha")
    args.dry_run = True
    entries = dict(_capture_plan(args))
    assert entries[".env"] == "update", (
        "an existing .env is appended to, and the rehearsal must not imply it "
        "is created fresh")
    assert (ws / ".env").read_text(encoding="utf-8") == "FOO=1\n"


# ------------------------------------------------ 4. the CLI surfaces it all

def test_new_help_epilog_names_the_output_of_every_archetype():
    """`new --help` listed every flag and zero lines about the output.  A
    reader who never leaves --help must still learn what lands."""
    from shared.scaffold.__main__ import _new_epilog
    epilog = _new_epilog()
    for name, doc in A.ARCHETYPES.items():
        assert name in epilog
        for ef in doc.writes:
            leaf = ef.render_path(archetype=name, set="<set>",
                                  agent="<agent>", name="<name>")
            assert leaf in epilog, f"{name}: {leaf} missing from `new --help`"
    assert "--dry-run" in epilog
    assert "explain archetype" in epilog


def test_unknown_archetype_names_the_accepted_ones(capsys):
    rc = build.run(_args(archetype="nope", workspace="/tmp/unused"))
    assert rc == 2
    out = capsys.readouterr().out
    for name in A.accepted():
        assert name in out, (
            "the error must list every accepted archetype — that list is the "
            "reader's only clue, and a stale literal is what shipped before")
