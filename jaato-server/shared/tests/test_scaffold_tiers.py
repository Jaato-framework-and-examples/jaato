"""Scaffold tooling coverage for model_tiers / V2 cross-provider tiers:
explain renders + introspects the real tier names; validate flags tier typos +
cross-provider tiers naming an uninstalled provider (now that model_tiers
survives the inherits/set merge — see config._merge_profiles).
"""

from shared.scaffold import explain
from shared.scaffold import validate as V


def test_explain_tiers_introspects_tier_names():
    data, text = explain.tiers()
    assert "vision" in data["tier_names"]          # from VALID_TIER_NAMES
    assert "cross_provider" in data                # V2 documented
    assert "enter_tier" in text


def test_validate_flags_bad_model_tiers(tmp_path):
    pdir = tmp_path / ".jaato" / "profiles" / "setT"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_t.yaml").write_text(
        "name: _base_t\ndescription: b\nplugins: []\n")
    (pdir / "t.yaml").write_text(
        "name: t\ninherits: [_base_t]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: m, provider: nebius}\n"
        "  visionn: {model: v, provider: openrouter}\n"     # tier-name typo
        "  planner: {model: p, provider: notaprovider}\n"   # unknown provider
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setT", only="t")
    codes = {d.code for d in diags}
    assert "unknown_tier" in codes        # visionn
    assert "unknown_provider" in codes    # notaprovider


def test_validate_accepts_cross_provider_tiers(tmp_path):
    # V2: a vision tier on a DIFFERENT (real) provider is valid.
    pdir = tmp_path / ".jaato" / "profiles" / "setOK"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_ok.yaml").write_text(
        "name: _base_ok\ndescription: b\nplugins: []\n")
    (pdir / "ok.yaml").write_text(
        "name: ok\ninherits: [_base_ok]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: glm-4.6, provider: zhipuai}\n"
        "  vision: {model: google/gemini-2.5-flash-lite, provider: openrouter}\n"
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setOK", only="ok")
    tier_codes = {d.code for d in diags
                  if d.code in ("unknown_tier", "unknown_provider")}
    assert tier_codes == set()   # cross-provider accepted, no tier findings


def test_explain_tiers_documents_the_description_key():
    data, text = explain.tiers()
    assert "description" in data
    assert "description" in data["shape"]
    assert "DESCRIPTION" in text


def test_validate_flags_a_malformed_tier_description(tmp_path):
    # A description reaches the MODEL (it becomes the tier's bullet in the
    # enter_tier schema), so a malformed one is worth catching statically.
    pdir = tmp_path / ".jaato" / "profiles" / "setD"
    pdir.mkdir(parents=True)
    (tmp_path / ".jaato" / "profiles" / "_base_d.yaml").write_text(
        "name: _base_d\ndescription: b\nplugins: []\n")
    (pdir / "d.yaml").write_text(
        "name: d\ninherits: [_base_d]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: m, description: 'grind through edits'}\n"  # ok
        "  planner:  {model: p, description: 42}\n"                    # bad
        "  initial: executor\n  fallback: executor\n")
    diags = V.validate_workspace(str(tmp_path), profile_set="setD", only="d")
    bad = [d for d in diags if d.code == "invalid_tier_description"]
    assert len(bad) == 1
    assert "planner" in bad[0].where
