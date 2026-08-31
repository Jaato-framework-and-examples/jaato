"""Guard: the profile ``env:`` block is documented, correctly, in BOTH places.

The gap this closes (jaato #752).  A session's env vars have two routes and
the scaffold named only one of them::

    env vars read by the installed daemon + plugins (186 total, 171 documented)
      (set these in the workspace .env; ...)

The route it omitted -- the profile ``env:`` block -- OUTRANKS the one it
named (``server/core.py:_resolve_session_env`` applies it over the workspace
``.env``), and for a whole class of callers it is the only route there is:
``jaato-eval`` writes each arm's ``.env`` itself, so a task author cannot put
anything in it.  The author who needed it found it by reading ``core.py``,
which is precisely the failure ``explain`` exists to prevent.  The generated
tier-2 profile said nothing either, though it already carried a commented
``model_tiers`` stanza in exactly the right shape.

WHY THE TWO HALVES ARE CHECKED TOGETHER.  #716's
``test_a_real_run_writes_only_documented_files`` asserts WHICH FILES ``new``
writes, never their content.  So a commented block in the generated profile and
the ``explain env`` text describing it could drift apart -- reworded in one,
deleted from the other -- with nothing failing: "documentation about a
generator rots", reintroduced one level down.  The mechanism against that is
:data:`explain.PROFILE_ENV_FACTS` / :data:`explain.ENV_EXAMPLE_VAR`, ONE
definition rendered by both halves, and the binding assertion is
``test_both_halves_render_the_same_facts``.

WHAT IS ASSERTED, AND WHY EACH.

1. ``explain env`` names the profile route, its precedence, and the path rule.
2. ``new`` emits a commented ``env:`` block with a worked example + a pointer.
3. Both halves render the SAME fact strings (the anti-drift binding).
4. The worked example names an env var the framework ACTUALLY reads -- so
   renaming or deleting ``JAATO_PROVIDER_TRACE`` fails here instead of leaving
   a generated profile advertising a variable nothing consumes.
5. Uncommenting the block yields a profile the validator ACCEPTS.  An example
   that is merely present is not an example that is true; this is the only
   assertion that runs it.
6. The worked example RUNS, and the file lands where the note says.
7. All three shared facts are what ``JaatoServer._resolve_session_env``
   actually does, and the runner's cwd is what the note says it is.

WHY 6 AND 7 EXIST, WHICH IS THE MORE IMPORTANT LESSON.  The first version of
this module had 1-5 and shipped a resolution claim that was simply false: that
a relative value lands in the daemon's cwd, and that an author wanting a
per-session file should therefore pass an ABSOLUTE path.  Twelve tests passed,
because every one of them asserted the TEXT.  Nothing set a relative env var
and looked at where the file went, so nothing could notice that the advice
inverted the safe pattern -- an absolute path is fixed at the PROFILE, so every
session sharing that profile appends to one interleaved file, which is the
contamination the relative form avoids.  Caught in review by someone who ran
it (jaato #752 review, sweep run 12).  A guard that pins prose cannot notice
the prose is wrong; 6 and 7 run the thing.

THE MECHANISM, since three plausible answers are all wrong and each was
believed by someone here.  The runner does NOT chdir into the workspace (7
measures this -- its cwd is the daemon's; ``core.py:336`` claims otherwise and
is stale, ``core.py:1029`` is right).  ``${workspaceRoot}`` does NOT help: it
expands daemon-side in ``_resolve_session_env``, before the session's workspace
is in scope.  And the value is NOT rewritten on the way in (7 measures this
too).  What makes a relative trace path land per-session is the READER:
``jaato_sdk.trace._resolve_trace_file`` joins a relative trace path onto
``JAATO_WORKSPACE_ROOT``, which the runner seeds per session.  That is why the
note names the reader instead of stating a rule about paths, and why 6 asserts
against the trace writer rather than against a string.
"""

from __future__ import annotations

import argparse
import contextlib
import io

import pytest
import yaml

from shared.scaffold import build, explain, introspect
from shared.scaffold import validate as V
from shared.scaffold.__main__ import _FILTER_SCOPES
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Put each half of #752 back.  The defect was an ABSENCE in two places, so
#: there are two reversions: one deletes the route + facts from ``explain
#: env`` (the state the issue was filed against), the other deletes the
#: commented block from what ``new`` writes.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find='''        "  TWO ROUTES SET THESE — the profile block wins:",
        "    <workspace>/.env                     VAR=value"
        "           per-WORKSPACE  (lower)",
        "    .jaato/profiles/<set>/<agent>.yaml   env: {VAR: value}"
        "   per-SESSION    (higher)",
        "  The profile `env:` block:",
    ]
    lines += [f"    - {fact}" for fact in PROFILE_ENV_FACTS]''',
        replace='''        "  (set these in the workspace .env)",
    ]''',
        test="test_explain_env_names_the_profile_env_block_as_a_route",
        because="`explain env` documenting only the workspace .env — the "
                "lower-precedence route, and the one a jaato-eval task author "
                "cannot write to at all",
    ),
    Reversion(
        target="jaato-sdk/jaato_sdk/trace.py",
        find="""    workspace = os.environ.get("JAATO_WORKSPACE_ROOT")
    if workspace:
        return os.path.join(workspace, file_path)
    return os.path.abspath(file_path)""",
        replace="""    return os.path.abspath(file_path)""",
        test="test_the_worked_example_lands_where_the_note_says_it_lands",
        because="the reader that makes the worked example per-session — "
                "without it a relative trace path resolves against the "
                "process cwd, and the generated profile would be "
                "recommending a value that pools every session's trace",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/build.py",
        find='''        "# Optional per-SESSION env vars.  This block:",
    ]
    lines += [f"#   - {fact}" for fact in _explain.PROFILE_ENV_FACTS]
    lines += [
        "#     — the trace vars resolve theirs against the session workspace,",
        "#     so the RELATIVE form below writes one file per session, in its",
        "#     own workspace, where an absolute path would be fixed at this",
        "#     profile and shared by every session using it.",
        "#     See `jaato-scaffold explain env`.",
        "# env:",
        f"#   {_explain.ENV_EXAMPLE_VAR}: {_explain.ENV_EXAMPLE_VALUE}",
    ]''',
        replace='''    ]''',
        test="test_new_emits_a_commented_env_block_with_a_pointer",
        because="the generated tier-2 profile teaching model_tiers and staying "
                "silent about the env: block, which is just as undiscoverable",
    ),
]


# Any installed provider works — nothing here asserts on provider-specific
# output.  Matched to the sibling scaffold guards so one uninstalled provider
# does not take out two suites for different reasons.
PROVIDER = "nebius"
MODEL = "deepseek-ai/DeepSeek-R1"
AGENT = "planner"
SET = "s1"


@pytest.fixture(scope="module")
def explain_env_text() -> str:
    """The full, unfiltered ``jaato-scaffold explain env`` rendering."""
    _data, text = explain.env()
    return text


@pytest.fixture
def generated_profile(tmp_path) -> str:
    """Run ``new profile-set`` for real; return the tier-2 profile's text.

    Generated rather than read off a fixture on purpose: a fixture is a copy
    of the generator's output that stops tracking it the moment it is written,
    which is the drift this module exists to catch.
    """
    ns = argparse.Namespace(
        archetype=build._archetypes.PROFILE_SET, workspace=str(tmp_path),
        provider=PROVIDER, model=MODEL, set=SET, agents=AGENT, force=False,
        json=False, recoverable=False, dry_run=False, secrets=None,
        secret_path=None, transport="ipc", url=None, token=None, ca=None,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        build.run(ns)
    prof = tmp_path / ".jaato" / "profiles" / SET / f"{AGENT}.yaml"
    assert prof.is_file(), f"new wrote no tier-2 profile:\n{buf.getvalue()}"
    return prof.read_text()


def _uncomment_env_block(text: str) -> str:
    """Return *text* with its commented ``env:`` block made live.

    Strips the ``# `` prefix from the ``# env:`` line and the indented
    ``#   VAR: value`` lines under it, leaving every other comment alone —
    which is exactly the edit a profile author makes when they take the
    generated hint up.
    """
    out, in_block = [], False
    for line in text.splitlines():
        if line.startswith("# env:"):
            in_block, line = True, line[2:]
        elif in_block and line.startswith("#   "):
            line = line[2:]
        elif in_block:
            in_block = False
        out.append(line)
    assert in_block or any(ln == "env:" for ln in out), \
        "no commented `env:` block to uncomment"
    return "\n".join(out) + "\n"


# ------------------------------------------------------- half 1: explain env

def test_explain_env_names_the_profile_env_block_as_a_route(explain_env_text):
    """`explain env` must name the higher-precedence route, not just the .env.

    Naming it is the whole of #752: the author had both a route and a
    precedence to learn and the tool volunteered neither.
    """
    text = explain_env_text
    assert ".jaato/profiles/<set>/<agent>.yaml" in text, \
        "explain env does not say WHERE the profile env: block lives"
    assert "env: {VAR: value}" in text, \
        "explain env does not show the profile block's shape"
    assert "<workspace>/.env" in text, \
        "explain env dropped the workspace .env route"
    assert explain.PROFILE_ENV_FACTS[0] in text, \
        "explain env does not state the precedence between the two routes"


def test_explain_env_names_the_reader_that_resolves_a_relative_path(
        explain_env_text):
    """Stating "verbatim" without naming the reader leaves the trap in place.

    "The value is applied verbatim" is true and, on its own, useless: it tells
    an author nothing about where their file will appear.  Both plausible
    completions of that sentence are wrong (see the module docstring), so the
    note has to name the two readers that differ — the trace helpers, which
    join a relative path onto ``JAATO_WORKSPACE_ROOT``, and everything else,
    which gets the runner's cwd.  An author who reads only the first half is
    exactly as stuck as before #752.
    """
    text = explain_env_text
    assert explain.PROFILE_ENV_FACTS[2] in text
    assert "JAATO_WORKSPACE_ROOT" in text, \
        "explain env says the reader resolves it but never names what it uses"
    assert "_resolve_trace_file" in text, \
        "explain env does not cite the reader an author would have to read"
    assert "RELATIVE value gives each session its own file" in text, \
        "explain env states the mechanism but never the recommendation"


def test_explain_env_no_longer_points_only_at_the_workspace_dotenv(
        explain_env_text):
    """The original misleading sentence must be gone, not merely appended to.

    ``explain env``'s header said "set these in the workspace .env" full stop.
    Leaving that in place and adding the profile route below it still tells a
    skimming reader the wrong thing first.
    """
    assert "set these in the workspace .env" not in explain_env_text


def test_the_explain_pointer_names_a_real_scope(generated_profile):
    """`jaato-scaffold explain env` must be a scope the CLI actually accepts.

    A pointer to a scope that does not exist is worse than no pointer: it
    costs the reader a command and teaches them the tool is stale.
    """
    assert "`jaato-scaffold explain env`" in generated_profile
    assert "env" in _FILTER_SCOPES


# --------------------------------------------------- half 2: what `new` emits

def test_new_emits_a_commented_env_block_with_a_pointer(generated_profile):
    """The generated tier-2 profile teaches the block where it is edited.

    Same shape as the ``model_tiers`` stanza it sits beside — commented, one
    worked example, one ``explain`` pointer — because that shape is what
    earned its place there.
    """
    text = generated_profile
    assert "# env:" in text, "no commented env: block in the generated profile"
    assert f"#   {explain.ENV_EXAMPLE_VAR}:" in text, "no worked example"
    assert "`jaato-scaffold explain env`" in text, "no explain pointer"


def test_the_generated_block_stays_commented_out(generated_profile):
    """It is a HINT, not a setting: an active env: block would take effect.

    A generated profile that silently redirected the provider trace log would
    be a surprise the author never asked for.
    """
    doc = yaml.safe_load(generated_profile)
    assert "env" not in doc, \
        "the env: example is live in the generated profile, not commented out"


def test_the_worked_example_lands_where_the_note_says_it_lands(
        tmp_path, monkeypatch):
    """RUN the worked example and look at where the file goes.

    This is the assertion the first version of this module lacked, and its
    absence is how a false resolution claim shipped with twelve green tests
    (see the module docstring).  It takes the example's own value, puts it in
    the environment the way the runner and the profile block would, drives the
    real trace writer, and checks the file.

    The claim under test is the useful half of the note: the RELATIVE form
    gives each session its own file in its own workspace.  If
    ``_resolve_trace_file`` ever stopped joining onto ``JAATO_WORKSPACE_ROOT``,
    the generated profile would be recommending a value that silently pools
    every session's trace in one place, and this fails instead of shipping.
    """
    import os

    from jaato_sdk.trace import provider_trace

    workspace = tmp_path / "arm-a"
    workspace.mkdir()
    # A PRISTINE cwd, not the repo's.  The negative half of this test ("it did
    # not land in the process's cwd") is only meaningful against a directory
    # nothing else writes to — asserted against the repo root it fails on any
    # stray provider_trace.log, which a sabotage run of this very test left
    # behind once.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    saved = {k: os.environ.get(k)
             for k in ("JAATO_WORKSPACE_ROOT", "JAATO_PROVIDER_TRACE")}
    try:
        # Exactly what the runner and the profile block put in place.
        os.environ["JAATO_WORKSPACE_ROOT"] = str(workspace)
        os.environ["JAATO_PROVIDER_TRACE"] = explain.ENV_EXAMPLE_VALUE
        provider_trace("guard", "the worked example, actually written")
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    landed = workspace / explain.ENV_EXAMPLE_VALUE
    assert landed.is_file(), (
        f"the worked example did not write into the session workspace; "
        f"{explain.ENV_EXAMPLE_VALUE!r} is documented as landing there")
    assert landed.read_text().strip(), "the trace file is empty"
    assert not (elsewhere / explain.ENV_EXAMPLE_VALUE).exists(), (
        "the worked example wrote into the process cwd — the note says a "
        "relative trace path follows the session, not the process")


def test_the_worked_example_is_relative(generated_profile):
    """An absolute example would be fixed at the profile, not per session.

    Stated separately from the behavioural test because it is the property
    the GENERATED text must carry: ``_resolve_trace_file`` only follows the
    session for a relative value, so an absolute one silently turns a
    per-session example into one shared file.
    """
    line = next(ln for ln in generated_profile.splitlines()
                if ln.startswith(f"#   {explain.ENV_EXAMPLE_VAR}:"))
    value = line.split(":", 1)[1].strip()
    assert value == explain.ENV_EXAMPLE_VALUE
    assert not value.startswith("/"), (
        f"worked example is absolute: {value!r} — that is one file for every "
        "session using this profile, not one per session")


def test_uncommenting_the_block_yields_a_profile_validate_accepts(
        tmp_path, generated_profile):
    """The example must be TRUE, not merely present.

    Everything above checks that text is there.  This one takes the hint up —
    uncomments the block exactly as an author would — and puts the result
    through the same validator ``new`` runs, so a stale key name or a wrong
    value shape fails here rather than at session bootstrap.
    """
    pdir = tmp_path / ".jaato" / "profiles"
    (pdir / f"_base_{AGENT}.yaml").write_text(
        f"name: _base_{AGENT}\ndescription: b\nplugins: []\n")
    (pdir / SET).mkdir(parents=True, exist_ok=True)
    live = _uncomment_env_block(generated_profile)
    assert "\nenv:" in live, "uncommenting produced no live env: block"
    (pdir / SET / f"{AGENT}.yaml").write_text(live)

    diags = V.validate_workspace(str(tmp_path), profile_set=SET, only=AGENT)
    ours = [d for d in diags
            if getattr(d, "tier", None) != "user" and d.severity == "error"]
    assert not ours, f"the worked example does not validate: {ours}"

    doc = yaml.safe_load(live)
    assert doc["env"] == {explain.ENV_EXAMPLE_VAR: explain.ENV_EXAMPLE_VALUE}


# ------------------------------------------------- the two halves, in step

def test_both_halves_render_the_same_facts(explain_env_text,
                                           generated_profile):
    """The anti-drift binding: one definition, rendered twice.

    #716's guard checks which FILES ``new`` writes and never their content,
    so nothing else would notice a fact reworded in one half and not the
    other.  Both halves render :data:`explain.PROFILE_ENV_FACTS` verbatim, so
    the only way to change the wording is to change it once, for both.
    """
    assert explain.PROFILE_ENV_FACTS, "the shared fact list is empty"
    for fact in explain.PROFILE_ENV_FACTS:
        assert fact in explain_env_text, f"explain env dropped: {fact!r}"
        assert fact in generated_profile, \
            f"the generated profile dropped: {fact!r}"


def test_the_worked_example_var_is_one_the_framework_reads():
    """The example must name a variable that still exists.

    ``JAATO_PROVIDER_TRACE`` is introspected from its ``# env:`` comment at
    the read site.  Renaming or deleting the variable must fail here, not
    leave two documents advertising a knob nothing consumes.
    """
    assert explain.ENV_EXAMPLE_VAR in introspect.env_vars(), (
        f"{explain.ENV_EXAMPLE_VAR} is no longer an env var the framework "
        "reads — pick a live one for the worked example in both halves")


def test_the_worked_example_var_appears_in_the_catalogue(explain_env_text):
    """A reader who follows the pointer must find the example listed.

    The routes note and the catalogue below it are the same screen; an
    example naming a variable absent from that catalogue reads as a typo.
    """
    assert explain_env_text.count(explain.ENV_EXAMPLE_VAR) >= 2


# ------------------------------------------- the facts ARE the behaviour

def test_the_documented_facts_are_what_the_resolver_actually_does(tmp_path):
    """The two halves agree with each other; this pins them to the code.

    Everything above keeps the two DOCUMENTS in step.  Nothing kept either in
    step with ``JaatoServer._resolve_session_env``, which is what all three
    shared facts are claims ABOUT: that the profile block wins on a key the
    workspace ``.env`` also sets (and only on that key), that values go
    through ``${VAR}`` expansion, and that they are otherwise applied
    verbatim.  Documentation that states behaviour and is not tied to it is
    the same rot one layer further down.

    The resolver is RUN rather than read, because the third fact is an
    ABSENCE — no path rewriting happens — and an absence cannot be grepped
    for.  The absence is what makes the reader responsible, which is the whole
    of why the note is worded the way it is.
    """
    import os

    from server.core import JaatoServer
    from shared.plugins.subagent.config import SubagentProfile

    ws = tmp_path / "ws"
    (ws / ".jaato").mkdir(parents=True)
    env_file = ws / ".env"
    env_file.write_text("SHARED=from_dotenv\nDOTENV_ONLY=1\n")

    profile = SubagentProfile(
        name="p", description="d", plugins=[],
        env={"SHARED": "from_profile",
             "EXPANDED": "${HOME}/trace.log",
             "RELATIVE": "provider_trace.log"})
    server = JaatoServer(env_file=str(env_file), workspace_path=str(ws),
                         profile=profile, session_id="s")
    server._resolve_session_env()
    resolved = server._session_env

    # fact 0 — "outranks the workspace .env, PER KEY"
    assert resolved["SHARED"] == "from_profile", explain.PROFILE_ENV_FACTS[0]
    assert resolved["DOTENV_ONLY"] == "1", \
        "the profile block replaced the .env wholesale instead of per key"

    # fact 1 — "takes ${VAR} expansion + secret URIs"
    assert resolved["EXPANDED"] == os.environ["HOME"] + "/trace.log", \
        explain.PROFILE_ENV_FACTS[1]

    # fact 2 — "is applied verbatim — a relative path is resolved by its
    # READER".  The verbatim half: nothing rewrites the value on the way in,
    # so the reader is genuinely the one that decides.
    assert resolved["RELATIVE"] == "provider_trace.log", (
        "the resolver now rewrites relative values — `explain env` and the "
        "generated profile both tell authors it does not, and both then "
        "explain what the READER does with them")
    assert str(ws) not in resolved["RELATIVE"]


def test_a_spawned_runner_starts_in_the_daemons_cwd(tmp_path):
    """The note's caveat, measured: the spawn does not enter the workspace.

    ``explain env`` warns that a reader which simply ``open()``s a relative
    value gets the runner process's cwd, and that this is not the session's
    workspace.  That is the non-obvious half — ``core.py`` carries BOTH
    claims, the stale one at :336 ("the server will chdir to this path") and
    the true one at :1029 ("Does NOT call ``os.chdir()``") — and a reader who
    believes the stale one concludes that ANY relative value is per-session.
    Only readers that consult ``JAATO_WORKSPACE_ROOT`` are.

    Forks a real runner and reads ``/proc/<pid>/cwd``, because that is the
    only way to tell the two docstrings apart.

    SCOPED TO THE SPAWN, and named for it.  It measures a runner at rest, so
    it would not see a chdir performed later in the process's life — and there
    IS one: ``subagent/plugin.py`` chdirs the runner into the workspace when
    it spawns a subagent.  That is not a hole to plug here but the reason the
    note tells authors to depend on the READER rather than on the process cwd:
    a cwd that moves mid-session is not something to write a path against.
    """
    import os
    import time

    from server.runner_spawner import RunnerSpawner

    if not os.path.isdir("/proc/self"):
        pytest.skip("needs /proc to read a process's cwd")

    workspace = tmp_path / "arm-a"
    (workspace / ".jaato").mkdir(parents=True)
    daemon_cwd = os.path.realpath(os.getcwd())

    runner = RunnerSpawner().spawn(
        profile_name="", session_id="guard-cwd-probe",
        workspace_path=str(workspace), disable_confine=True)
    try:
        # The runner is up once it can be read; give the bootstrap a moment
        # so a late chdir (there is none today) would still be observed.
        deadline = time.monotonic() + 20.0
        cwd = None
        while time.monotonic() < deadline:
            try:
                cwd = os.readlink(f"/proc/{runner.pid}/cwd")
                break
            except OSError:
                time.sleep(0.2)
        assert cwd is not None, "runner never became readable via /proc"
        time.sleep(1.0)
        cwd = os.readlink(f"/proc/{runner.pid}/cwd")
    finally:
        try:
            os.kill(runner.pid, 9)
        except OSError:
            pass

    assert cwd != os.path.realpath(workspace), (
        "the runner now chdirs into the session workspace — `explain env` "
        "tells authors it does not, and that a relative value read by a "
        "plain open() therefore lands in the daemon's cwd")
    assert cwd == daemon_cwd, f"runner cwd is neither workspace nor daemon: {cwd}"
