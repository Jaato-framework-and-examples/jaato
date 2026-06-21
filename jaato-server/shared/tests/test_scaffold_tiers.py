"""Scaffold `explain tiers` coverage — renders the model_tiers / V2 cross-provider
topic and introspects the REAL tier names from the installed framework.
"""

from shared.scaffold import explain


def test_explain_tiers_introspects_tier_names():
    data, text = explain.tiers()
    assert "vision" in data["tier_names"]          # from VALID_TIER_NAMES
    assert "cross_provider" in data                # V2 documented
    assert "enter_tier" in text
