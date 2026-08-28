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
