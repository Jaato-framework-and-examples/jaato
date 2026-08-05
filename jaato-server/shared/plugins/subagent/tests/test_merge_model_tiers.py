"""Regression: _merge_profiles must PRESERVE model_tiers.

Pre-fix the inherits/set merge constructed the merged SubagentProfile without
model_tiers, so any profile using ``inherits:`` (which is how scaffolded
set-profiles are structured) silently lost its tiers and fell back to
single-model.  Now the child's tier-config wins (inherit the parent's when the
child declares none).
"""

from shared.plugins.subagent.config import discover_profiles


def _discover(tmp_path, child_yaml, base_yaml="name: _base_m\ndescription: b\nplugins: []\n"):
    pdir = tmp_path / ".jaato" / "profiles"
    (pdir / "setM").mkdir(parents=True)
    (pdir / "_base_m.yaml").write_text(base_yaml)
    (pdir / "setM" / "m.yaml").write_text(child_yaml)
    r = discover_profiles(
        profiles_dir=".jaato/profiles", base_path=str(tmp_path),
        config_root=str(tmp_path / ".jaato"), force_profile_set="setM")
    return r.profiles["m"]


def test_child_model_tiers_survive_inherits(tmp_path):
    prof = _discover(tmp_path,
        "name: m\ninherits: [_base_m]\nplugins: []\n"
        "model_tiers:\n"
        "  executor: {model: x, provider: zhipuai}\n"
        "  vision: {model: g, provider: openrouter}\n"
        "  initial: executor\n  fallback: executor\n")
    assert prof.model_tiers.get("executor") == {"model": "x", "provider": "zhipuai"}
    assert prof.model_tiers.get("vision") == {"model": "g", "provider": "openrouter"}


def test_parent_model_tiers_inherited_when_child_has_none(tmp_path):
    prof = _discover(tmp_path,
        "name: m\ninherits: [_base_m]\nplugins: []\n",
        base_yaml="name: _base_m\ndescription: b\nplugins: []\n"
                  "model_tiers:\n  executor: {model: x, provider: zhipuai}\n"
                  "  initial: executor\n  fallback: executor\n")
    assert prof.model_tiers.get("executor") == {"model": "x", "provider": "zhipuai"}
