"""A CANCELLED turn must still report its spend; a REFUSED one must not.

Both end a send_message without a normal return, but they are different
events: a cancelled turn RAN and spent tokens, a refused turn never started.
The cancelled path used to `return` early, skipping the post-turn
forwarding entirely — so that turn's tokens reached no TurnCompletedEvent
and were invisible to anything accumulating from it, the cascade pool most
consequentially (a child that exhausts its own budget ends exactly this
way, so the leak hit precisely the children whose spend mattered).
"""

from types import SimpleNamespace

from server.runner.rpc import RunnerRPC


class _Hooks:
    def __init__(self):
        self.turns = []

    def on_agent_turn_completed(self, **kw):
        self.turns.append(kw)

    def on_agent_context_updated(self, **kw):
        pass

    def on_agent_history_updated(self, **kw):
        pass


def _session(turn_accounting, hooks):
    return SimpleNamespace(
        _ui_hooks=hooks, _agent_id="main",
        get_turn_accounting=lambda: turn_accounting,
        get_context_usage=lambda: {},
        get_history=lambda: [],
    )


def _fwd(session, turns_before):
    RunnerRPC._forward_post_turn_hooks(
        SimpleNamespace(), session, turns_before)


def test_a_turn_that_ran_is_forwarded_with_its_spend():
    hooks = _Hooks()
    _fwd(_session([{"total": 1600, "spend_total": 1600}], hooks), turns_before=0)
    assert len(hooks.turns) == 1
    assert hooks.turns[0]["spend_total_tokens"] == 1600


def test_no_new_turn_is_not_forwarded():
    """The refused-turn case, now enforced mechanically: firing here would
    re-emit turn_accounting[-1], i.e. the PREVIOUS turn's numbers."""
    hooks = _Hooks()
    existing = [{"total": 2504, "spend_total": 4654}]
    _fwd(_session(existing, hooks), turns_before=1)   # count unchanged
    assert hooks.turns == []


def test_cancelled_turn_spend_is_not_lost():
    """The leak: 3 responses landed, the turn was cancelled mid-flight, and
    its spend must still reach a TurnCompletedEvent."""
    hooks = _Hooks()
    accounting = [
        {"total": 2503, "spend_total": 4646},
        {"total": 1578, "spend_total": 3105},
        {"total": 1600, "spend_total": 1600},   # the cancelled turn
    ]
    _fwd(_session(accounting, hooks), turns_before=2)
    assert len(hooks.turns) == 1
    assert hooks.turns[0]["spend_total_tokens"] == 1600
    # and the pool summing these now sees every token the tracker saw
    assert sum(t["spend_total"] for t in accounting) == 9351


def test_forwarding_never_raises_on_a_broken_session():
    hooks = _Hooks()
    bad = SimpleNamespace(
        _ui_hooks=hooks, _agent_id="main",
        get_turn_accounting=lambda: [{"total": 1}],
        get_context_usage=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        get_history=lambda: [],
    )
    _fwd(bad, turns_before=0)          # must not raise


def test_no_hooks_is_a_noop():
    _fwd(SimpleNamespace(_ui_hooks=None), turns_before=0)


# ------------- session.apply_budget_degrade wire contract (slice 3) -------

def test_apply_budget_degrade_dispatch_passes_the_args_through(monkeypatch):
    """Pin the DISPATCH, not the handler.

    The dispatch line read ``env.params``, which does not exist on
    RequestEnvelope (the field is ``args``, and that is what _call_named
    populates). Every cascade degrade push therefore died with
    'AttributeError: RequestEnvelope object has no attribute params'.

    Handler-level tests all passed throughout — they call the handler with a
    dict and never cross the boundary that was broken. So this drives
    _dispatch_method with a real envelope: it is the only shape that would
    have caught it.
    """
    from server.runner.rpc import RunnerRPC
    from server.runner.envelope import RequestEnvelope

    seen = {}

    class _Sess:
        def apply_cascade_degrade(self, rungs, pool_pressure=None):
            seen["rungs"] = rungs
            return {"applied": len(rungs)}

    rpc = RunnerRPC.__new__(RunnerRPC)
    monkeypatch.setattr(
        rpc, "_require_ready_session", lambda: (True, None, _Sess()),
        raising=False)

    rung = {"at": 50.0, "action": "abort"}
    env = RequestEnvelope(
        id="r1", method="session.apply_budget_degrade",
        args={"rungs": [rung]},
    )
    ok, result = rpc._dispatch_method(env)
    assert ok is True, f"dispatch failed: {result}"
    assert result == {"applied": 1}
    assert seen["rungs"] == [rung], "args never reached the handler"


def test_apply_budget_degrade_tolerates_an_empty_envelope():
    from server.runner.rpc import RunnerRPC

    class _Sess:
        def apply_cascade_degrade(self, rungs, pool_pressure=None):
            return {"applied": 0}

    rpc = SimpleNamespace(_require_ready_session=lambda: (True, None, _Sess()))
    ok, result = RunnerRPC._handle_session_apply_budget_degrade(rpc, {})
    assert ok is True and result == {"applied": 0}


def test_apply_budget_degrade_on_a_session_without_the_method():
    from server.runner.rpc import RunnerRPC

    rpc = SimpleNamespace(
        _require_ready_session=lambda: (True, None, SimpleNamespace()))
    ok, result = RunnerRPC._handle_session_apply_budget_degrade(rpc, {"rungs": []})
    assert ok is False and "apply_cascade_degrade" in result["error"]
