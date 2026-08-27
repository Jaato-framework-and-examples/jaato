"""Echo can REPORT a spend, so the budget subsystem is assertable for free.

Every assertion about budgets and cost -- does a ceiling refuse, does a
reported ``cost_usd`` survive to ``UsageBreakdown``, does
``limits={"tokens": 0}`` mean zero headroom or unbounded -- currently needs a
provider that ACTUALLY SPENDS, against a real endpoint, nondeterministically.

That is why those tests do not exist, and it is why three defects in that
subsystem reached a consumer before anyone here saw them: a cascade that
hands back a session id for a budget it cannot honour, a ``cost_usd`` of
``None`` beside a tracker holding the real figure for the same turn, and a
zero limit that refuses nothing.  All three were found by someone building
against a live daemon on their own tokens.

Making spend a CONFIGURED VALUE turns each of those into an assertion that
costs nothing and never varies.  The numbers are arbitrary by construction --
whatever the test declares -- and deliberately NOT derived from
``count_tokens``, because a simulated spend that tracked the prompt would
drift with the fixture's wording, which is the opposite of what a ceiling
test needs.
"""

from __future__ import annotations

import pytest

from shared.plugins.model_provider.base import ProviderConfig
from shared.plugins.model_provider.echo.provider import EchoProvider
from jaato_sdk.plugins.model_provider.types import (
    FinishReason, Message, Part, Role,
)


def _provider(**extra):
    p = EchoProvider()
    p.initialize(ProviderConfig(extra=extra))
    return p


def _one_turn(provider, text="hello"):
    return provider.complete(
        messages=[Message(role=Role.USER, parts=[Part.from_text(text)])],
        tools=None,
    )


# ----------------------------------------------------------- the default

def test_without_the_knob_echo_still_costs_nothing():
    """The existing contract, unchanged.

    Every profile using echo today omits ``usage``; none of them may start
    reporting a spend, or an unrelated budget test would begin charging.
    """
    p = _provider()
    _one_turn(p)

    usage = p.get_token_usage()
    assert usage.total_tokens == 0
    assert usage.cost_usd is None


# --------------------------------------------------------- the reported spend

def test_a_configured_cost_is_reported():
    p = _provider(usage={"prompt_tokens": 1000, "output_tokens": 200,
                         "cost_usd": 0.0042})
    _one_turn(p)

    usage = p.get_token_usage()
    assert usage.cost_usd == 0.0042
    assert usage.prompt_tokens == 1000
    assert usage.output_tokens == 200


def test_total_tokens_defaults_to_prompt_plus_output():
    """A silent zero total would make a ceiling test pass by never charging.

    ``{"prompt_tokens": 10, "output_tokens": 5}`` means a 15-token turn.  If
    the total defaulted to 0, a budget of 1 token would never be reached and
    the test would go green having proved nothing -- passing for the wrong
    reason, which is the failure this knob exists to prevent.
    """
    p = _provider(usage={"prompt_tokens": 10, "output_tokens": 5})
    _one_turn(p)

    assert p.get_token_usage().total_tokens == 15


def test_an_explicit_total_is_not_overridden():
    """Providers report totals that are not the sum (cache reads, reasoning),
    so a caller stating one means it."""
    p = _provider(usage={"prompt_tokens": 10, "output_tokens": 5,
                         "total_tokens": 99})
    _one_turn(p)

    assert p.get_token_usage().total_tokens == 99


def test_the_spend_is_identical_on_every_turn():
    """Per-turn constancy is what makes 'how many turns to the ceiling'
    arithmetic rather than observation."""
    p = _provider(usage={"prompt_tokens": 100, "output_tokens": 10,
                         "cost_usd": 0.001})

    seen = []
    for i in range(3):
        _one_turn(p, f"turn {i}")
        u = p.get_token_usage()
        seen.append((u.total_tokens, u.cost_usd))

    assert seen == [(110, 0.001)] * 3


def test_the_spend_is_reported_on_the_tool_call_turn_too():
    """A tool-calling turn costs the same as a prose one.

    The tool-call branch builds its own ProviderResponse; if it were left
    reporting zero, a budget test driving a tool loop would charge nothing on
    exactly the turns a real agent spends most.
    """
    p = _provider(
        tool_call={"name": "signal_completion", "args": {}},
        usage={"prompt_tokens": 500, "output_tokens": 50, "cost_usd": 0.002},
    )
    result = _one_turn(p)

    assert result.finish_reason == FinishReason.TOOL_USE
    assert p.get_token_usage().cost_usd == 0.002
    assert p.get_token_usage().total_tokens == 550


# ------------------------------------------------------------- typos raise

def test_an_unknown_key_raises_rather_than_being_ignored():
    """A typo'd ``cost`` for ``cost_usd`` would leave the cost at ``None``.

    The test asserting that cost would then read the FRAMEWORK's silence as
    its own bug -- a fixture defect wearing a product defect's clothes, which
    is the exact confusion this whole area keeps producing.
    """
    with pytest.raises(ValueError) as excinfo:
        _provider(usage={"cost": 0.5})

    assert "cost" in str(excinfo.value)
    assert "cost_usd" in str(excinfo.value), (
        "the error must name the field the caller meant, not just reject theirs"
    )


def test_every_TokenUsage_field_is_accepted():
    """The knob must not drift from the dataclass it fills.

    Checked against the FIELDS rather than a hardcoded list: a hardcoded copy
    is a second statement of the contract, and this codebase has been bitten
    three times this week by exactly that.
    """
    import dataclasses
    from jaato_sdk.plugins.model_provider.types import TokenUsage

    spec = {f.name: 1 for f in dataclasses.fields(TokenUsage)}
    p = _provider(usage=spec)          # must not raise
    _one_turn(p)

    usage = p.get_token_usage()
    for name in spec:
        assert getattr(usage, name) == 1, f"{name} did not survive the knob"
