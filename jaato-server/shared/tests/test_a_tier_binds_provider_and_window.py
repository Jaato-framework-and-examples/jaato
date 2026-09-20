"""A tier binds (provider, model) — and two halves of that did not take.

WHAT WAS REPORTED: "two tiers of the same profile do not share history;
each tier keeps its own."  The premise is false and worth stating first,
because it is what locates the real defects: a session has ONE
``_history`` (``jaato_session.py``), ``switch_tier`` never touches it, and
every request is built from ``_history_for_provider()`` whose only
per-tier filter withholds BINARY content.  Driven end to end, the tier
entered second is handed the tier entered first's turns verbatim —
``test_history_is_shared_across_a_tier_switch`` is that measurement, and
it is here so the claim stops being folklore.

What a tier DOES bind is a pair, ``(provider, model)``, and the session
applied one half of it in two places.

DEFECT 1 -- THE PROVIDER HALF OF THE INITIAL TIER WAS DROPPED.
``configure()`` overrode ``self._model_name`` from the initial tier and
left ``self._provider_name_override`` at the profile's top-level
``provider:`` (``None`` when the profile declares none, because every
tier declares its own).  Two consequences, and the second is the one the
report is about:

  * turn 0 ran the initial tier's MODEL on somebody else's PROVIDER --
    the runtime default, or a top-level value that disagrees;
  * ``_active_provider_name`` -- which ``_connect_tier_entry`` compares
    ``entry.provider`` against to decide whether to SWAP -- was that same
    wrong value.  Entering a tier naming the provider the session was
    already running compared unequal and built a SECOND instance of it.

For a stateless provider a duplicate instance is a wasted handshake.
``claude_cli`` is not stateless: it sends ``messages[-1]`` and nothing
else, delegating the transcript to the CLI's own ``--resume`` session
(``_cli_session_id``).  A second instance is a second CLI session, so
that tier starts empty and accumulates only its own turns -- literally
"each tier keeps its own history", from the one provider in the tree
that can produce it.

DEFECT 2 -- THE WINDOW DID NOT FOLLOW THE MODEL.
``InstructionBudget.context_limit`` is the denominator for the after-turn
GC threshold, the pre-send refusal guard, and ``get_context_usage`` (so
``aspect="context"`` and every client's readout).  It was stamped once,
at first provider materialisation, and never again.  Measured: a session
that booted on a 200k tier and entered an 8k one reported
``get_context_limit() == 8000`` while the budget still said ``200000``.

The direction that hurts is entering a SMALLER window: GC cannot fire
before the request overflows, the guard lets it through, the upstream
rejects it, and ``_try_gc_for_context_recovery`` then trims history in
the STORE -- destructive, and shared, so the tier that had the room loses
the conversation too.  Which is how a stale denominator ends up looking
like the isolation that was reported.
"""

from unittest.mock import MagicMock

import pytest

from jaato_sdk.plugins.model_provider.types import (
    FinishReason, Message, Part, ProviderResponse, Role, TokenUsage, TurnResult,
)

from shared.jaato_session import JaatoSession
from shared.model_tiers import (
    ModelTierConfig, RESERVED_FALLBACK_KEY, RESERVED_INITIAL_KEY,
)
from shared.tests.reversion import Reversion


REVERSIONS = [
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        if not entry.provider:
            return
        if (self._provider_name_override
                and self._provider_name_override != entry.provider):
            logger.info(
                "Tier mode active: overriding session provider %s with "
                "initial tier %s's provider %s",
                self._provider_name_override, tier_config.initial_tier,
                entry.provider,
            )
        self._provider_name_override = entry.provider
""",
        replace="",
        test="test_the_initial_tiers_provider_is_the_sessions_provider",
        because="turn 0 running the initial tier's model on whichever "
                "provider the profile's top level (or the runtime default) "
                "names, and _active_provider_name naming it too",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""            resolved_provider_name = (
                cfg['provider_name']
                or getattr(self._runtime, 'provider_name', None)
            )
            self._active_provider_name = resolved_provider_name
            if resolved_provider_name is not None:
                self._provider_cache[resolved_provider_name] = self._provider""",
        replace="""            self._active_provider_name = cfg['provider_name']
            if cfg['provider_name'] is not None:
                self._provider_cache[cfg['provider_name']] = self._provider""",
        test="test_entering_the_provider_already_in_use_does_not_fork_it",
        because="a session left unnamed by an absent override, so a tier "
                "naming the provider it is already on builds a second "
                "instance of it -- a second CLI session under claude_cli",
    ),
    Reversion(
        target="jaato-server/shared/jaato_session.py",
        find="""        try:
            self._refresh_context_limit_from_provider()
        except Exception as exc:  # noqa: BLE001
            self._tier_context_limit_refresh_failures = getattr(
                self, '_tier_context_limit_refresh_failures', 0) + 1
            logger.warning(
                "tier context-window refresh for %s failed; GC and the "
                "context readout keep measuring against the previous "
                "model's window: %s", entry.model, exc,
            )
""",
        replace="",
        test="test_the_context_window_follows_the_tier",
        because="GC, the refusal guard and every context readout measuring "
                "against the window of the model the session booted on",
    ),
]


# ============================================================ fixtures

WINDOWS = {"big-model": 200_000, "small-model": 8_000}


def _provider(registered_as):
    """A stateless provider that records the message list of every request."""
    p = MagicMock()
    p.name = registered_as
    p.model_name = None
    p.supports_streaming.return_value = True
    p.get_retry_after.return_value = None
    p.get_context_limit.side_effect = lambda: WINDOWS.get(p.model_name, 0)
    p.requests = []

    def connect(model, skip_model_test=True):
        p.model_name = model
    p.connect.side_effect = connect

    def complete(messages, **kwargs):
        p.requests.append([
            Message(role=m.role, parts=list(m.parts or [])) for m in messages
        ])
        on_chunk = kwargs.get("on_chunk")
        if on_chunk is not None:
            on_chunk("ok")
        return TurnResult.from_provider_response(ProviderResponse(
            parts=[Part(text="ok")],
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=1, output_tokens=1, total_tokens=2),
        ))
    p.complete.side_effect = complete
    return p


def _tiers(initial_provider=None, second_provider=None):
    return ModelTierConfig.from_unified_dict({
        "executor": {"model": "big-model",
                     **({"provider": initial_provider} if initial_provider else {})},
        "planner": {"model": "small-model",
                    **({"provider": second_provider} if second_provider else {})},
        RESERVED_INITIAL_KEY: "executor",
        RESERVED_FALLBACK_KEY: "executor",
    })


def _session(tier_config, *, runtime_provider_name=None,
             profile_provider=None):
    """A live session whose runtime records every provider it is asked for.

    ``runtime_provider_name`` stands in for the daemon's configured default
    (what ``create_provider(provider_name=None)`` resolves to);
    ``profile_provider`` for a top-level ``provider:`` in the profile.
    """
    built = []
    provider = _provider(runtime_provider_name or "fake")

    def create_provider(model=None, provider_name=None, **_kw):
        built.append((provider_name, model))
        provider.connect(model)          # create_provider connects, so do we
        return provider

    runtime = MagicMock()
    runtime.provider_name = runtime_provider_name
    runtime.create_provider.side_effect = create_provider
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    runtime.registry.enrich_prompt.side_effect = (
        lambda prompt, **_k: MagicMock(prompt=prompt, metadata={})
    )

    session = JaatoSession(runtime, "profile-model",
                           provider_name=profile_provider)
    session.configure(tier_config=tier_config)
    return session, provider, built


# ============================================================ the premise

def test_history_is_shared_across_a_tier_switch():
    """The reported symptom's premise, measured rather than assumed.

    Asserted on the WIRE, not on ``get_history()``: what the second tier
    is HANDED is the question, and a session could keep a shared store
    while sending each tier a slice of it.
    """
    session, provider, _ = _session(_tiers("fake", "fake"),
                                    runtime_provider_name="fake")

    session.send_message("turn one, in executor", lambda *a, **k: None)
    session.switch_tier("planner")
    session.send_message("turn two, in planner", lambda *a, **k: None)

    second = provider.requests[1]
    texts = ["".join(p.text or "" for p in (m.parts or [])) for m in second]
    assert "turn one, in executor" in texts, (
        "the tier entered second was not handed the turn the first tier "
        f"took; it received {texts!r}.  One session has one history and "
        "switch_tier does not touch it -- if this fails, something now "
        "partitions it per tier."
    )
    assert any(m.role == Role.MODEL for m in second), (
        "the first tier's ANSWER is missing from the second tier's request"
    )


# ============================================================ defect 1

def test_the_initial_tiers_provider_is_the_sessions_provider():
    """Turn 0 runs on the provider the initial tier declared.

    The profile's top-level ``provider:`` disagrees here on purpose: a
    tier binds a pair, and the tier wins for the same reason its model
    already did.
    """
    session, _, built = _session(
        _tiers("anthropic", "openrouter"),
        runtime_provider_name="google_genai",
        profile_provider="google_genai",
    )
    session.send_message("first turn", lambda *a, **k: None)

    assert built[0][0] == "anthropic", (
        f"the session's first provider was built as {built[0][0]!r}.  The "
        "initial tier declares anthropic, and configure() overrode the "
        "model half of that binding while leaving the provider half at the "
        "profile's top-level value."
    )
    assert session._active_provider_name == "anthropic", (
        "_active_provider_name is what _connect_tier_entry compares "
        "entry.provider against; wrong here, every later swap decision is "
        "taken against a provider the session is not running."
    )


def test_a_tier_declaring_no_provider_leaves_the_choice_alone():
    """The override is the tier's to make only when the tier makes it.

    A profile whose tiers name no provider is the single-provider shape
    that predates cross-provider tiers, and it must reach create_provider
    exactly as it did before.
    """
    session, _, built = _session(
        _tiers(None, None),
        runtime_provider_name="anthropic",
        profile_provider="anthropic",
    )
    session.send_message("first turn", lambda *a, **k: None)

    assert built == [("anthropic", "big-model")], (
        f"a profile whose initial tier declares no provider built {built!r}"
    )


# ============================================================ defect 2

def test_entering_the_provider_already_in_use_does_not_fork_it():
    """Two tiers on one provider are served by ONE provider instance.

    The reversion of this one is the mechanism behind the report: under
    ``claude_cli`` a second instance is a second ``--resume`` CLI session,
    and since that provider sends only ``messages[-1]``, the new session
    starts with nothing and accumulates only the turns taken while its
    tier holds the wheel.
    """
    # The shape that isolates THIS half: nobody names the provider up front
    # -- no top-level ``provider:``, and the INITIAL tier declares none
    # either, so the session runs whatever the runtime resolved -- and the
    # tier being entered names that same provider.  (When the initial tier
    # does name one, the override above already makes the comparison true;
    # this is the case it cannot reach.)
    session, _, built = _session(
        _tiers(None, "claude_cli"),
        runtime_provider_name="claude_cli",
        profile_provider=None,
    )
    session.send_message("first turn", lambda *a, **k: None)
    session.switch_tier("planner")

    assert len(built) == 1, (
        f"entering a tier on the provider the session was already running "
        f"built a second instance of it: {built!r}"
    )
    assert session._provider_cache.get("claude_cli") is session._provider, (
        "the live provider is not in the per-provider cache under the name "
        "a tier would ask for, so the next switch back builds another one"
    )


def test_an_unnamed_provider_stays_unnamed():
    """Neither side naming a provider leaves the field ``None``, as before.

    The fallback asks the RUNTIME what it resolved; when the runtime names
    nothing there is nothing to key a cache on, and inventing a name (from
    ``provider.name``, say) would key zhipuai's instance under its parent
    anthropic's name.
    """
    session, _, _ = _session(_tiers(None, None), runtime_provider_name=None)
    session.send_message("first turn", lambda *a, **k: None)

    assert session._active_provider_name is None
    assert session._provider_cache == {}


# ============================================================ defect 3

def test_the_context_window_follows_the_tier():
    """The budget denominator is re-read on every tier connect."""
    session, _, _ = _session(_tiers("fake", "fake"),
                             runtime_provider_name="fake")
    session.send_message("first turn", lambda *a, **k: None)
    assert session._instruction_budget.context_limit == 200_000

    session.switch_tier("planner")          # small-model, an 8k window

    assert session.get_context_limit() == 8_000, "fixture sanity"
    assert session._instruction_budget.context_limit == 8_000, (
        "the GC / refusal-guard / context-readout denominator still names "
        "the window of the model the session BOOTED on.  Entering a "
        "smaller-window tier then overflows without GC firing, and the "
        "recovery path trims the SHARED store."
    )
    assert session.get_context_usage()["context_limit"] == 8_000, (
        "get_context_usage reads the budget, so aspect='context' and every "
        "client readout inherit the stale window too"
    )


def test_a_failed_refresh_is_counted_not_raised():
    """A tier switch already connected must not fail on bookkeeping.

    The block it joins is explicitly non-fatal, and its comment says the
    cost of that is a regression visible only in a log nobody reads -- so
    the failure is counted onto the LLM span beside its two siblings.
    """
    session, provider, _ = _session(_tiers("fake", "fake"),
                                    runtime_provider_name="fake")
    session.send_message("first turn", lambda *a, **k: None)

    provider.get_context_limit.side_effect = RuntimeError("provider is sulking")
    session.switch_tier("planner")           # must not raise

    assert session._tier_context_limit_refresh_failures == 1
    attrs = session._build_llm_span_attributes()
    assert attrs["jaato.tier.context_limit_refresh_failures"] == 1, (
        "a silently degraded denominator is exactly what made this "
        "unobservable in the first place"
    )


def test_a_session_with_no_tiers_is_untouched():
    """Single-model mode keeps the one stamp it always had."""
    built = []
    provider = _provider("fake")

    def create_provider(model=None, provider_name=None, **_kw):
        built.append((provider_name, model))
        provider.connect(model)
        return provider

    runtime = MagicMock()
    runtime.provider_name = "fake"
    runtime.create_provider.side_effect = create_provider
    runtime.get_tool_schemas.return_value = []
    runtime.get_executors.return_value = {}
    runtime.get_system_instructions.return_value = None
    runtime.permission_plugin = None
    runtime.ledger = None
    runtime.reliability_plugin = None
    runtime.registry = MagicMock()
    runtime.registry.get_exposed_tool_schemas.return_value = []
    runtime.registry.enrich_prompt.side_effect = (
        lambda prompt, **_k: MagicMock(prompt=prompt, metadata={})
    )

    session = JaatoSession(runtime, "big-model")
    session.configure()
    session.send_message("only turn", lambda *a, **k: None)

    assert session._tier_config is None
    # No profile provider and no tier to supply one: create_provider is still
    # asked for None and the runtime still resolves it, exactly as before.
    assert built == [(None, "big-model")]
    assert session._instruction_budget.context_limit == 200_000
    assert "jaato.tier" not in session._build_llm_span_attributes()
    # What DID change for this session: the field now names the provider the
    # runtime resolved instead of echoing the absent override.  Nothing
    # outside tier mode reads _provider_cache, so this is honesty, not
    # behaviour -- and it is the half that stops a tier forking the instance.
    assert session._active_provider_name == "fake"


def test_the_provider_cache_tracks_a_model_select():
    """``model select`` replaces the instance the cache points at.

    Reachable before this change for any profile with a top-level
    ``provider:``, and the fix above makes the cache populated in the
    no-override case too -- so leaving it stale would have widened an
    existing hole rather than closed one.  ``_provider_for_tier`` hands a
    tier whatever is cached under its provider name, so a stale entry
    sends a later ``enter_tier`` back to the instance the operator just
    replaced.
    """
    session, first, _ = _session(_tiers("fake", "fake"),
                                 runtime_provider_name="fake")
    session.send_message("first turn", lambda *a, **k: None)
    assert session._provider_cache["fake"] is first

    replacement = _provider("fake")
    session._runtime.create_provider.side_effect = (
        lambda *a, **k: replacement)
    session._runtime.list_available_models.return_value = ["small-model"]
    session._execute_model_command(
        {"subcommand": "select", "model_name": "small-model"})

    assert session._provider is replacement, "fixture sanity"
    assert session._provider_cache["fake"] is replacement, (
        "the cache still names the instance model select replaced; the "
        "next enter_tier would switch back to it"
    )
