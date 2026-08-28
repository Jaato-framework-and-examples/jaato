"""A provider-reported cost must reach ``TurnCompletedEvent``.

It did not, for any provider that reports one.  A consumer measured, for the
SAME turn:

    UsageBreakdown : {"spend_total_tokens": 26469, "cost_usd": null, ...}
    budget tracker : {'usd': 0.0027821, 'tokens': 26469.0, ...}

The token counts agree TO THE UNIT, so these are not two measurements that
disagree about cost -- they are ONE measurement whose cost survives on the
tracker/telemetry path (``_resolve_span_cost`` reads ``usage.cost_usd``) and
dies on the event path.

THERE WAS NO SINGLE DROP POINT, which is why it could not be isolated from
outside.  Five links, every one missing:

    1. the session never accumulated ``usage.cost_usd`` into turn accounting
    2. ``on_agent_turn_completed`` had no cost parameter to carry it
    3. the runner's RPC payload did not include it
    4. the daemon-side unpacker did not read it
    5. ``_build_usage`` -- which HAS a ``cost_usd_override`` -- was never
       given one

Any one of those alone would have been a bug with an obvious fix.  All five
together look like "the framework does not do cost", which is what it looked
like.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


def test_the_builder_accepts_an_override_and_uses_it():
    """``_build_usage`` was always ready; nobody fed it."""
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._pricing_loaded = True
    srv._pricing = None
    srv._workspace_path = None
    # __init__ always sets this; a __new__-built double that omits it is a
    # shape production never has, and the no-override branch reads it.
    srv._model_name = None

    usage = srv._build_usage(
        prompt_tokens=1000, output_tokens=200, total_tokens=1200,
        cost_usd_override=0.0042,
    )

    assert usage.cost_usd == 0.0042


def test_no_reported_cost_stays_None_not_zero():
    """``None`` and ``0.0`` are different answers.

    ``None`` = the provider reported no cost.  ``0.0`` = it reported free.
    A default of ``0.0`` anywhere in the chain would claim the second on
    behalf of a provider that said the first -- and a free-tier turn and an
    unpriced one want different handling from anything totalling spend.
    """
    from server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._pricing_loaded = True
    srv._pricing = None
    srv._workspace_path = None
    # __init__ always sets this; a __new__-built double that omits it is a
    # shape production never has, and the no-override branch reads it.
    srv._model_name = None

    usage = srv._build_usage(prompt_tokens=10, output_tokens=1, total_tokens=11)

    assert usage.cost_usd is None


def test_cost_accumulates_across_a_turns_responses():
    """A turn with a tool call has >= 2 billed responses.

    Replacing rather than accumulating would report only the last one -- the
    same error the surrounding code documents for ``spend_total``, which is
    why the cost accumulates at exactly that site.
    """
    src = pathlib.Path(
        "jaato-server/shared/jaato_session.py").read_text(encoding="utf-8")

    assert "turn_tokens['cost_usd'] = (" in src, (
        "cost is no longer accumulated onto the turn"
    )
    assert "(turn_tokens.get('cost_usd') or 0.0)" in src, (
        "cost is being REPLACED rather than accumulated; a turn with a tool "
        "call would report only its final response's cost"
    )


@pytest.mark.parametrize("path,what", [
    ("jaato-server/shared/plugins/subagent/ui_hooks.py", "the hook signature"),
    ("jaato-server/server/runner/rpc.py", "the RPC payload"),
    ("jaato-server/server/core.py", "the daemon-side unpack + build"),
    ("jaato-server/shared/jaato_client.py", "the facade driver"),
])
def test_every_link_in_the_chain_carries_cost(path, what):
    """All five links, pinned per file.

    Four of them silently forwarding and one dropping produces exactly what
    was reported: a cost that exists everywhere except where a consumer
    reads it.  Checked per link so a break names WHICH link.
    """
    src = pathlib.Path(path).read_text(encoding="utf-8")

    assert "cost_usd" in src, f"{what} no longer carries cost_usd"


def test_the_event_builder_is_actually_given_the_override():
    """The last link, checked as an AST rather than by grepping.

    ``cost_usd_override`` appearing anywhere in core.py is not enough -- it
    is a parameter of ``_build_usage`` itself, so the string is present even
    when nobody passes it.  That is precisely the state this bug was in.
    """
    src = pathlib.Path("jaato-server/server/core.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    # THE TURN CALL SPECIFICALLY, not "some call somewhere".
    #
    # core.py has several ``_build_usage`` call sites and ANOTHER of them
    # already passed ``cost_usd_override``.  So the first version of this
    # assertion -- "at least one call passes it" -- was true before the fix
    # and stayed true when the fix was removed: it PASSED under sabotage.
    # Third vacuous AST guard of the month, and the only reason it was
    # caught is that the sabotage was actually run.
    #
    # The turn call is identified by a keyword only it carries.
    turn_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_build_usage"
        and any(kw.arg == "spend_total_tokens" for kw in n.keywords)
    ]
    assert turn_calls, (
        "no _build_usage call takes spend_total_tokens; this test can no "
        "longer identify the turn-event call and must be re-aimed"
    )
    missing = [
        n.lineno for n in turn_calls
        if not any(kw.arg == "cost_usd_override" for kw in n.keywords)
    ]
    assert not missing, (
        f"_build_usage call(s) at line(s) {missing} build a turn's usage "
        "without passing cost_usd_override — the parameter exists and is not "
        "supplied, which is exactly the shape of the original bug"
    )
