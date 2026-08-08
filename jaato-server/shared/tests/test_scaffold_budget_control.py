"""Scaffold tooling coverage for budget_control — the discoverability
contract from ``docs/design/budget-control-degradation.md`` §6.3.

A new profile block is invisible to ``explain`` and silently accepted by
``validate`` unless wired at the same points ``model_tiers`` is.  These
tests are the regression guard against sliding back to the silent-ignore
path (a block that LOOKS configured while doing nothing).

Mirrors ``test_scaffold_tiers.py``.
"""

from shared.scaffold import explain
from shared.scaffold import validate as V


def _write_set(tmp_path, set_name, profile_name, body):
    """Emit a minimal workspace with one inheriting profile."""
    pdir = tmp_path / ".jaato" / "profiles" / set_name
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_b.yaml").write_text(
        "name: _base_b\ndescription: b\nplugins: []\n")
    (pdir / f"{profile_name}.yaml").write_text(
        f"name: {profile_name}\ninherits: [_base_b]\nplugins: []\n{body}")
    return V.validate_workspace(
        str(tmp_path), profile_set=set_name, only=profile_name)


# ------------------------------------------------------------- explain

def test_explain_profile_surfaces_budget_control():
    """`explain profile` must list the field — else an author can't discover
    it without reading source."""
    data, text = explain.profile()
    row = next((d for d in data if d["name"] == "budget_control"), None)
    assert row is not None, "budget_control missing from profile_schema()"
    assert row["description"], "field must carry a metadata description"
    assert "budget_control" in text


def test_explain_profile_resolves_real_constraints():
    """Allowed-values come from the installed constants, not a hardcoded
    string — same principle as model_tiers' tier names."""
    from shared.budget_control import VALID_ACTIONS, VALID_DIMENSIONS
    data, _ = explain.profile()
    allowed = next(d for d in data if d["name"] == "budget_control")["allowed"]
    for dim in VALID_DIMENSIONS:
        assert dim in allowed
    for action in VALID_ACTIONS:
        assert action in allowed


# ------------------------------------------------------------ validate

def test_validate_accepts_well_formed_budget_control(tmp_path):
    diags = _write_set(tmp_path, "setOK", "ok",
        "model_tiers:\n"
        "  executor: {model: m, provider: openrouter}\n"
        "  planner: {model: p, provider: openrouter}\n"
        "  initial: executor\n  fallback: executor\n"
        "budget_control:\n"
        "  limits: {usd: 3.0, tokens: 300000}\n"
        "  degrade:\n"
        "    - at: 70%\n"
        "      model_tiers:\n"
        "        planner: {model: google/gemini-flash, provider: openrouter}\n"
        "    - at: 100\n"
        "      action: finalize\n")
    budget_codes = {d.code for d in diags if "budget" in d.code
                    or d.code in ("parse_error", "unknown_provider")}
    assert budget_codes == set(), f"unexpected findings: {[d.message for d in diags]}"


def test_validate_flags_overlay_provider_not_installed(tmp_path):
    """The check load-time CANNOT do: it doesn't know what's installed."""
    diags = _write_set(tmp_path, "setP", "p",
        "model_tiers:\n"
        "  planner: {model: p, provider: openrouter}\n"
        "  initial: planner\n  fallback: planner\n"
        "budget_control:\n"
        "  limits: {usd: 1.0}\n"
        "  degrade:\n"
        "    - at: 70\n"
        "      model_tiers:\n"
        "        planner: {model: m, provider: notaprovider}\n")
    assert "unknown_provider" in {d.code for d in diags}


def test_validate_flags_overlay_without_model_tiers(tmp_path):
    """A cross-field defect: an overlay with no tier table to patch would
    silently do nothing at runtime."""
    diags = _write_set(tmp_path, "setN", "n",
        "model: some-model\nprovider: openrouter\n"
        "budget_control:\n"
        "  limits: {usd: 1.0}\n"
        "  degrade:\n"
        "    - at: 70\n"
        "      model_tiers:\n"
        "        planner: {model: m, provider: openrouter}\n")
    assert "budget_overlay_without_tiers" in {d.code for d in diags}


def test_validate_warns_on_overlay_of_undeclared_tier(tmp_path):
    diags = _write_set(tmp_path, "setU", "u",
        "model_tiers:\n"
        "  executor: {model: m, provider: openrouter}\n"
        "  initial: executor\n  fallback: executor\n"
        "budget_control:\n"
        "  limits: {usd: 1.0}\n"
        "  degrade:\n"
        "    - at: 70\n"
        "      model_tiers:\n"
        "        vision: {model: v, provider: openrouter}\n")
    assert "budget_overlay_undeclared_tier" in {d.code for d in diags}


def test_action_only_ladder_needs_no_tiers(tmp_path):
    """A single-model profile may still budget — only OVERLAY rungs require
    model_tiers."""
    diags = _write_set(tmp_path, "setA", "a",
        "model: some-model\nprovider: openrouter\n"
        "budget_control:\n"
        "  limits: {usd: 1.0, seconds: 60}\n"
        "  degrade:\n"
        "    - at: 100\n      action: finalize\n")
    assert not [d for d in diags if "budget" in d.code or d.code == "parse_error"]


def test_malformed_budget_control_surfaces_as_parse_error(tmp_path):
    """The silent-ignore regression guard: a bad block must be REPORTED, never
    dropped."""
    diags = _write_set(tmp_path, "setBad", "bad",
        "budget_control:\n  limits: {bogus: 1}\n")
    parse = [d for d in diags if d.code == "parse_error"]
    assert parse, f"malformed budget_control was silently ignored: {diags}"
    assert "budget_control" in parse[0].message
