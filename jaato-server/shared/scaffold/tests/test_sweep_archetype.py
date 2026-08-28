"""The `sweep` archetype — N INDEPENDENT arms, none feeding another.

`new cascade` is a linear CHAIN: stage 1 then stage 2, output feeding
forward, one session at a time.  A sweep is a different topology and a common
one — an eval matrix, a batch job, any fan-out over a work-list, any A/B
across profile sets.  Two consumers wrote it by hand for the same reason.

EVERY ASSERTION HERE CORRESPONDS TO SOMETHING SOMEONE GOT WRONG BY HAND.
That is the whole basis for encoding it: a scaffold is the one form of
documentation that gets executed, and these are the traps it exists to
carry.  The requirements come from the consumer who hit each of them.
"""

from __future__ import annotations

import ast

import pytest

from shared.scaffold._client_templates import TEMPLATES


@pytest.fixture(scope="module")
def rendered() -> str:
    """The template with the builder's placeholders filled.

    Rendered, not raw: a template that only compiles WITH placeholders left
    in is a template that never compiles for a reader.
    """
    _, tmpl, _ = TEMPLATES["sweep"]
    return (tmpl
            .replace("__TITLE__", "Sweep driver")
            .replace("__PROVENANCE__", "jaato-scaffold new sweep")
            .replace("__WORKSPACE__", "/ws")
            .replace("__ENV_FILE__", ".env")
            .replace("__MODEL__", "echo")
            .replace("__PROVIDER__", "echo")
            .replace("__CLIENT_IMPORT__",
                     "from jaato_sdk import IPCClient, ClientType, EventType")
            .replace("__CONN_CONSTANTS__", "SOCKET = '/tmp/j.sock'")
            .replace("__ON_STATUS_DEF__", "")
            .replace("__NEW_CLIENT_CALL__", "IPCClient(socket_path=SOCKET)"))


def test_the_archetype_is_registered():
    assert "sweep" in TEMPLATES
    _, _, desc = TEMPLATES["sweep"]
    assert "INDEPENDENT" in desc, (
        "the description must distinguish it from `cascade`, or a reader "
        "picks the chain when they wanted the fan-out"
    )


def test_the_generated_script_compiles(rendered):
    """A scaffold that emits invalid Python is worse than none."""
    ast.parse(rendered)


def test_it_subscribes_before_creating_the_session(rendered):
    """Requirement 2, and the one with the least forgiving failure mode.

    A refusal is announced WHILE the create is in flight, so a handler
    installed afterwards never sees it.  Current daemons raise, but a
    generated driver ships to whatever daemon the reader has — and without
    this the failure is a 30s timeout naming nothing.
    """
    tree = ast.parse(rendered)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_arm")

    def _first_line(pred):
        return min((n.lineno for n in ast.walk(fn)
                    if isinstance(n, ast.Call) and pred(n)), default=None)

    subscribe = _first_line(
        lambda n: isinstance(n.func, ast.Attribute) and n.func.attr == "subscribe")
    create = _first_line(
        lambda n: isinstance(n.func, ast.Attribute)
        and n.func.attr == "create_session")

    assert subscribe is not None, "the arm never subscribes to errors"
    assert create is not None, "the arm never creates a session"
    assert subscribe < create, (
        "create_session is called before the error subscription; a refusal "
        "announced during the create would reach no handler"
    )


def test_a_refusal_becomes_a_typed_outcome(rendered):
    """Requirement 5.  An exhausted pool means the ceiling worked and nothing
    ran — a different call to action from "the daemon is broken"."""
    assert "except SessionCreateFailed" in rendered
    assert "may_exist" in rendered, (
        "the refusal outcome drops may_exist, so a caller cannot tell a "
        "safe retry from one that may create a second session"
    )


def test_completeness_uses_the_sdk_rule_not_finish_reason(rendered):
    """Requirement 4.  ``finish_reason != "stop"`` blocks every schema-driven
    arm as truncated; a consumer shipped exactly that into two graders."""
    assert "truncation_reason(" in rendered

    # Checked against CODE, with comments stripped.  The template contains a
    # comment WARNING against the wrong branch, and that comment is desirable
    # -- a naive substring check fails on the guard's own explanation, which
    # is a trap this repo has now sprung four times.
    code = "\n".join(
        line.split("#", 1)[0] for line in rendered.splitlines()
    )
    assert 'finish_reason != "stop"' not in code
    assert "finish_reason != 'stop'" not in code


def test_the_owner_client_outlives_the_arms(rendered):
    """Requirement 1.  A pool belongs to the connection that declared it — an
    owner opened and closed around a single arm takes the pool with it."""
    tree = ast.parse(rendered)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.AsyncFunctionDef) and n.name == "main")

    budget = [n.lineno for n in ast.walk(main)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "cascade_budget_set"]
    gather = [n.lineno for n in ast.walk(main)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "gather"]

    assert budget and gather, "main no longer declares a pool or fans out"
    assert budget[0] < gather[0], (
        "the pool is declared after the arms run; the arms would draw on "
        "nothing"
    )


def test_the_arms_run_concurrently_and_independently(rendered):
    """The topology itself.  A sweep whose arms are sequential is a chain
    with extra steps, and one whose failure propagates is not a sweep."""
    assert "asyncio.gather(" in rendered
    assert "Never raises" in rendered, (
        "_run_arm's contract is not stated; an arm that raises takes its "
        "siblings down through gather()"
    )


def test_both_budget_gates_are_named(rendered):
    """Requirement 8.  A generated harness is where a consumer first meets
    them, and they do not compose the way they look: a session carrying its
    own ceiling does not draw on the pool, so declaring both leaves the pool
    inert."""
    assert "budget_control" in rendered
    assert "own-books" in rendered.lower()
    assert "inert" in rendered


def test_the_pool_forecloses_ledger_grading_and_says_so(rendered):
    """The constraint a consumer hit only after building on the assumption.

    A pooled arm is detached when it terminates — that is what returns the
    warm slot — so ``request_history`` afterwards answers with an error, not
    a ledger.
    """
    assert "FORECLOSES" in rendered or "forecloses" in rendered
    assert "request_history" in rendered


def test_it_stamps_what_actually_ran(rendered):
    """Requirement 7.  The branch you have checked out does not determine
    which SDK ran — an editable install resolves elsewhere."""
    assert "jaato_sdk.__file__" in rendered


def test_it_does_not_emit_a_cascade_register_call(rendered):
    """Retired deliberately, not forgotten.

    Observer registration WAS required: a cid'd session withheld
    TURN_COMPLETED unless the creator registered.  Fixing the terminal-event
    ordering removed the need — a pooled arm reports turns=1 without it —
    and the consumer verified that by A/B before deleting theirs.  Emitting
    it now would ship a redundant RPC plus a comment explaining behaviour
    that no longer exists, and a wrong explanation in generated code costs
    more than a redundant call.
    """
    assert "cascade_register" not in rendered


def test_the_sweep_varies_the_profile_not_the_model(rendered):
    """``model`` and ``provider`` are PROFILE properties.

    The first version of this template gave every arm an identical inline
    ``{"model": MODEL, "provider": PROVIDER}`` — so the "matrix" varied
    nothing but the prompt, while the dict implied model/provider were the
    sweep dimension.  ``create_session(profile=...)`` takes a profile name,
    and a profile is what carries plugins, GC strategy, instructions and a
    completion schema; an inline spec can only vary the thinnest part of what
    an agent is.
    """
    tree = ast.parse(rendered)
    arms = next(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "ARMS" for t in n.targets)
    )

    profiles = [row.elts[1] for row in arms.elts]        # (name, profile, agent, prompt)
    assert all(isinstance(p, ast.Constant) and isinstance(p.value, str)
               for p in profiles), (
        "an arm's profile is not a name; the default sweep should vary "
        "PROFILES, since model/provider are profile-expressible"
    )
    # NOT asserting the profiles differ: the shipped example varies the
    # PERSONA with capabilities held fixed, which is the cleaner
    # demonstration of orthogonality.  What must hold is that SOMETHING
    # varies -- see test_the_example_actually_sweeps_something.
    assert all(isinstance(p.value, str) for p in profiles)


def test_no_unsubstituted_placeholder_survives(rendered):
    """A ``__PLACEHOLDER__`` the builder does not fill reaches the reader.

    Nearly shipped: this template referenced ``__PROFILE__``, which is not
    one of the builder's placeholders, so it would have appeared verbatim in
    generated code.  Checked against the builder's real list rather than a
    copy of it.
    """
    import re

    import pathlib

    build_src = pathlib.Path(
        "jaato-server/shared/scaffold/build.py").read_text(encoding="utf-8")
    known = set(re.findall(r"__[A-Z_]+__", build_src))

    left = set(re.findall(r"__[A-Z_]+__", rendered))
    assert not left, (
        f"unsubstituted placeholder(s) {sorted(left)} in the generated "
        f"script; the builder only fills {sorted(known)}"
    )


def test_the_tuple_carries_the_persona(rendered):
    """``agent`` is ORTHOGONAL to ``profile`` and is its own sweep axis.

    profile = CAPABILITIES (model, provider, plugins, GC, ceilings).
    agent   = PERSONA (the markdown that becomes system instructions).

    The SDK says they "compose freely", and the sibling ``cascade``
    archetype already carries an agent slot — a sweep without one cannot ask
    the most common eval question, "does this persona work better?", and is
    inconsistent with its own family.
    """
    tree = ast.parse(rendered)
    arms = next(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "ARMS" for t in n.targets)
    )

    widths = {len(row.elts) for row in arms.elts}
    assert widths == {4}, (
        f"ARMS rows are {widths}-wide; expected 4 — "
        "(name, profile, agent, prompt)"
    )
    assert "agent=agent" in rendered, (
        "the agent column is collected and never passed to create_session, "
        "so the persona axis is decorative"
    )


def test_the_example_actually_sweeps_something(rendered):
    """A sweep whose rows are identical teaches the wrong thing.

    The first version varied nothing but the prompt while implying
    model/provider were the dimension.  Whatever the shipped example varies,
    it must vary SOMETHING other than the label.
    """
    tree = ast.parse(rendered)
    arms = next(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "ARMS" for t in n.targets)
    )

    rows = [
        tuple(getattr(e, "value", ast.dump(e)) for e in row.elts[1:])
        for row in arms.elts
    ]
    assert len(set(rows)) == len(rows), (
        f"every arm is identical apart from its name ({rows[0]}); the "
        "example demonstrates no axis at all"
    )


def test_the_sdk_does_not_call_the_agent_its_system_instructions():
    """An agent is not "the system instructions", and the SDK said it was.

    The instructions are an ASSEMBLY — the agent, plus the
    ``.jaato/instructions/`` base layer, plugin instructions, framework
    constants and the untrusted-content boundary.  ``suppress_base_
    instructions`` can drop every one of those EXCEPT the agent and its
    plugins, which is the framework itself saying the agent is not the
    stack.

    Naming an identity after its transport is how it comes to look
    swappable with a prompt string.  Pinned here rather than left to
    prose, because a wrong sentence in a docstring readers DO consult
    already cost this project two bad graders once today.
    """
    import pathlib

    # ALL of them, not just the SDK pair: a guard that checks two files
    # while the same sentence survives in a third is a guard aimed at the
    # wrong place, and this repo keeps writing those.
    for path in ("jaato-sdk/jaato_sdk/client/ipc.py",
                 "jaato-sdk/jaato_sdk/client/recovery.py",
                 "jaato-server/server/session_manager.py"):
        src = pathlib.Path(path).read_text(encoding="utf-8")
        assert "becomes the session's system instructions" not in src, (
            f"{path} describes the agent as BECOMING the system "
            "instructions; it is one layer of them"
        )
