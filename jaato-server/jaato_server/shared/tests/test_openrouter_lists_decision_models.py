"""OpenRouter's catalog, read whole, and a decision model with no window.

Two findings of the 2026-10-17 live run (docs/design/decision-models.md,
step 10):

* ``GET /api/v1/models`` lists only models whose output includes text, so
  every decision model was missing from it and cost a per-model
  ``/endpoints`` request.  The provider now asks for
  ``?output_modalities=all``; the ``/endpoints`` document remains the
  fallback for a model the full listing omits.
* Some catalog entries report a context length of 0
  (``respan/span-01-lite``), and ``connect()`` refused such a model.  A
  decisions-only model needs no window: it is used only by ``decide()``'s
  pre-send size check, which is skipped when the window is unknown (as on
  the ``typesafe`` provider).  A chat model with no window is still
  refused.

Driven against the local decision stand-in, which mirrors the catalog
filter; nothing touches the network.
"""

from __future__ import annotations

import pytest

from jaato_sdk.plugins.model_provider.decisions import DecisionQuestion
from jaato_server.shared.tests.decision_standin import (
    JEV,
    NO_WINDOW,
    DecisionStandIn,
    openrouter_against,
)
from jaato_server.shared.tests.reversion import Reversion

_OR = "jaato-server/jaato_server/shared/plugins/model_provider/openrouter/provider.py"

REVERSIONS = [
    Reversion(
        target=_OR,
        find="response = httpx.get(url, params=CATALOG_PARAMS, timeout=15)",
        replace="response = httpx.get(url, timeout=15)",
        test="test_a_decision_model_is_found_in_the_listing",
        because="every decision model would cost a per-model /endpoints "
                "request, because the default listing omits it",
    ),
    Reversion(
        target=_OR,
        find="        if not self._context_length and not self._decisions_only:\n",
        replace="        if not self._context_length:\n",
        test="test_a_decision_model_with_no_window_connects_and_decides",
        because="a decision model whose catalog reports no window could "
                "not be connected without a manual context_length",
    ),
]

Q = {"urgent": DecisionQuestion("noul", "Is this urgent?")}


@pytest.fixture
def standin():
    with DecisionStandIn() as server:
        yield server


def _gets(server: DecisionStandIn):
    return [path for method, path, _h, _b in server.seen if method == "GET"]


def test_a_decision_model_is_found_in_the_listing(standin):
    provider = openrouter_against(standin, JEV)
    assert provider.get_context_limit() == 32000
    assert provider.is_decisions_only()
    gets = _gets(standin)
    assert "/api/v1/models?output_modalities=all" in gets
    assert not [g for g in gets if g.endswith("/endpoints")]


def test_a_decision_model_with_no_window_connects_and_decides(standin):
    provider = openrouter_against(standin, NO_WINDOW)
    assert provider.get_context_limit() == 0
    result = provider.decide("Payouts failing for 3 days.", Q, model=NO_WINDOW)
    assert "urgent" in result.answers
    (path, _headers, body), = standin.posts()
    assert path == "/api/alpha/decisions" and body["model"] == NO_WINDOW


def test_a_chat_model_with_no_window_is_still_refused(standin):
    with pytest.raises(ValueError, match="context_length could not be resolved"):
        openrouter_against(standin, "acme/unlisted-chat")
