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
6. The example obeys its own advice: it is an ABSOLUTE path.  The block warns
   that a relative value lands in the daemon's cwd, so an example that shipped
   a relative one would teach the bug it documents.
7. All three shared facts are what ``JaatoServer._resolve_session_env``
   actually does.  1-6 keep the two documents in step with EACH OTHER; this
   keeps both in step with the code they describe.

A NOTE ON THE SECOND FACT, because #752 states it the other way round.  The
issue reports that a relative value resolves against the session's working
directory, and that this is what gives each session its own file.  It does
not: the resolver applies the value verbatim (asserted below), the runner
subprocess never ``chdir``s to the workspace, and ``${workspaceRoot}`` /
``${cwd}`` expand daemon-side -- so a relative value lands in the DAEMON's
cwd, which every session on that daemon shares.  That is the same single
interleaved file the issue was chasing, reached by the other route, so the
note documents the behaviour rather than the expectation and says to pass an
absolute path.
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
        target="jaato-server/shared/scaffold/build.py",
        find='''        "# Optional per-SESSION env vars.  This block:",
    ]
    lines += [f"#   - {fact}" for fact in _explain.PROFILE_ENV_FACTS]
    lines += [
        "#     (so pass an ABSOLUTE path when each session must write its own",
        "#     file).  See `jaato-scaffold explain env`.",
        "# env:",
        f"#   {_explain.ENV_EXAMPLE_VAR}: /abs/path/{agent}-provider_trace.log",
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


def test_explain_env_states_the_verbatim_path_rule(explain_env_text):
    """The relative-path trap must be stated, not left to be rediscovered.

    A relative value is applied to the session process's environment as-is;
    the runner never chdirs to the workspace, so it resolves against the
    DAEMON's cwd — shared by every session on that daemon.  That is the half
    that produces a single interleaved file when the author expected one per
    session, and it is not guessable from the profile schema.
    """
    assert explain.PROFILE_ENV_FACTS[2] in explain_env_text
    assert "ABSOLUTE path" in explain_env_text, \
        "explain env states the trap but not the way out of it"


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


def test_the_worked_example_is_an_absolute_path(generated_profile):
    """The example must obey the rule the comment above it states.

    The block warns that a relative value lands in the daemon's cwd.  An
    example that shipped a relative path would demonstrate the bug.
    """
    line = next(ln for ln in generated_profile.splitlines()
                if ln.startswith(f"#   {explain.ENV_EXAMPLE_VAR}:"))
    value = line.split(":", 1)[1].strip()
    assert value.startswith("/"), f"worked example is not absolute: {value!r}"


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
    assert doc["env"] == {
        explain.ENV_EXAMPLE_VAR: f"/abs/path/{AGENT}-provider_trace.log"}


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
    for.  It is also the fact worth the most: #752 was filed believing the
    opposite (that a relative value resolves against the session workspace),
    and the runner never chdirs there, so a relative value lands in the
    daemon's cwd, shared by every session on it.
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

    # fact 2 — "is applied verbatim: a RELATIVE path lands in the DAEMON's cwd"
    assert resolved["RELATIVE"] == "provider_trace.log", (
        "the resolver now rewrites relative values — `explain env` and the "
        "generated profile both tell authors it does not")
    assert str(ws) not in resolved["RELATIVE"]
