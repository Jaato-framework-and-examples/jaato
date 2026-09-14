"""The profile-level ``scrub_secret_env`` key (#863).

Pins:

- the field parses from a profile file as written (str or list) and is
  ``None`` when absent — absence is NOT "off": the plugins apply the
  framework set on their own, so ``None`` must reach them untouched;
- inheritance is scalar-override (child replaces; silent child inherits;
  disagreeing parents conflict);
- ``inject_scrub_secret_env`` folds the profile value BENEATH an explicit
  per-surface knob, only into the surfaces the profile enables;
- ``validate_profile`` (the data-level validator) rejects a malformed value
  at either position.
"""

from __future__ import annotations

import pytest
import yaml

from shared.plugins.subagent.config import (
    SubagentProfile,
    build_inline_profile,
    discover_profiles,
    inject_scrub_secret_env,
    resolve_profiles,
    validate_profile,
)


# ---- parsing --------------------------------------------------------------

def _discover(tmp_path, **profiles):
    pdir = tmp_path / ".jaato" / "profiles"
    pdir.mkdir(parents=True)
    for name, body in profiles.items():
        (pdir / f"{name}.yaml").write_text(yaml.safe_dump(
            {"name": name, "description": name, "plugins": ["cli"], **body}))
    result = discover_profiles(
        profiles_dir=".jaato/profiles", base_path=str(tmp_path),
        config_root=str(tmp_path / ".jaato"))
    assert not result.errors, result.errors
    return result.profiles


def test_absent_is_none_not_off(tmp_path):
    assert _discover(tmp_path, a={})["a"].scrub_secret_env is None


def test_parses_shorthand_and_list_as_written(tmp_path):
    profiles = _discover(
        tmp_path,
        off={"scrub_secret_env": "none"},
        dflt={"scrub_secret_env": "default"},
        lst={"scrub_secret_env": ["default", "!GH_TOKEN"]},
    )
    assert profiles["off"].scrub_secret_env == "none"
    assert profiles["dflt"].scrub_secret_env == "default"
    assert profiles["lst"].scrub_secret_env == ["default", "!GH_TOKEN"]


def test_inline_profile_carries_the_key():
    p = build_inline_profile({"plugins": ["cli"], "scrub_secret_env": "none"})
    assert p.scrub_secret_env == "none"


# ---- inheritance -----------------------------------------------------------

def _resolve(**profiles):
    resolved, errors = resolve_profiles(profiles)
    return resolved, errors


def test_child_inherits_parent_value():
    resolved, errors = _resolve(
        base=SubagentProfile(name="base", description="b", scrub_secret_env="none"),
        child=SubagentProfile(name="child", description="c", inherits=["base"]),
    )
    assert not errors
    assert resolved["child"].scrub_secret_env == "none"


def test_child_replaces_parent_value_outright():
    resolved, errors = _resolve(
        base=SubagentProfile(name="base", description="b",
                             scrub_secret_env=["default", "!GH_TOKEN"]),
        child=SubagentProfile(name="child", description="c", inherits=["base"],
                              scrub_secret_env="none"),
    )
    assert not errors
    assert resolved["child"].scrub_secret_env == "none"


def test_disagreeing_parents_conflict_unless_child_overrides():
    resolved, errors = _resolve(
        p1=SubagentProfile(name="p1", description="1", scrub_secret_env="none"),
        p2=SubagentProfile(name="p2", description="2", scrub_secret_env="default"),
        child=SubagentProfile(name="child", description="c", inherits=["p1", "p2"]),
    )
    assert "child" in errors and "scrub_secret_env" in errors["child"]
    resolved, errors = _resolve(
        p1=SubagentProfile(name="p1", description="1", scrub_secret_env="none"),
        p2=SubagentProfile(name="p2", description="2", scrub_secret_env="default"),
        child=SubagentProfile(name="child", description="c", inherits=["p1", "p2"],
                              scrub_secret_env=["*_TOKEN"]),
    )
    assert not errors and resolved["child"].scrub_secret_env == ["*_TOKEN"]


# ---- the fold into plugin_configs -----------------------------------------

def test_inject_reaches_every_enabled_surface_beneath_explicit_knobs():
    p = SubagentProfile(
        name="p", description="p",
        plugins=["cli", "mcp", "interactive_shell", "todo"],
        plugin_configs={"mcp": {"scrub_secret_env": ["*_TOKEN"]},
                        "cli": {"extra_paths": ["/x"]}},
        scrub_secret_env="none",
    )
    cfg = dict(p.plugin_configs)
    out = inject_scrub_secret_env(p, cfg)
    assert out is cfg
    assert cfg["cli"] == {"extra_paths": ["/x"], "scrub_secret_env": "none"}
    assert cfg["interactive_shell"] == {"scrub_secret_env": "none"}
    # explicit per-surface knob wins
    assert cfg["mcp"] == {"scrub_secret_env": ["*_TOKEN"]}
    # a non-surface plugin is untouched
    assert "todo" not in cfg
    # the profile's own dict is not mutated
    assert "scrub_secret_env" not in p.plugin_configs["cli"]


def test_inject_skips_surfaces_the_profile_does_not_enable():
    p = SubagentProfile(name="p", description="p", plugins=["cli"],
                        scrub_secret_env="none")
    cfg = {}
    inject_scrub_secret_env(p, cfg)
    assert cfg == {"cli": {"scrub_secret_env": "none"}}


def test_inject_with_no_profile_value_changes_nothing():
    # The plugins then apply the framework default on their own — the
    # flipped default does NOT depend on this fold.
    p = SubagentProfile(name="p", description="p", plugins=["cli", "mcp"])
    cfg = {"cli": {"extra_paths": []}}
    inject_scrub_secret_env(p, cfg)
    assert cfg == {"cli": {"extra_paths": []}}


# ---- data-level validation -------------------------------------------------

def _errors(data):
    ok, errors, _ = validate_profile({"name": "x", "description": "x",
                                      "plugins": [], **data})
    return errors


def test_validate_accepts_the_grammar():
    assert _errors({"scrub_secret_env": "none"}) == []
    assert _errors({"scrub_secret_env": ["default", "!GH_TOKEN"]}) == []
    assert _errors({"plugin_configs": {"cli": {"scrub_secret_env": "default"}}}) == []


def test_validate_rejects_malformed_at_either_position():
    errs = _errors({"scrub_secret_env": 7})
    assert errs and "'scrub_secret_env'" in errs[0]
    errs = _errors({"plugin_configs": {"mcp": {"scrub_secret_env": [{}]}}})
    assert errs and "plugin_configs['mcp'].scrub_secret_env" in errs[0]
