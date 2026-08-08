"""scaffold `explain` surfaces descriptions (gap 1) + the profile schema (gap 2).

Gap 1: plugin-level description (class docstring) + config-setting descriptions
(``PluginSetting.description``, previously dropped — only names were kept).
Gap 2: a new ``explain profile`` scope surfacing the ``SubagentProfile`` knobs
with descriptions, incl. the AppArmor knobs (``apparmor`` /
``apparmor_fragments``) an author would otherwise have to find in config.py.
"""

from shared.scaffold import explain, introspect


# ----------------------------------------------------------- gap 2: profile

def test_profile_schema_surfaces_apparmor_knobs_with_descriptions():
    fields = {f.name: f for f in introspect.profile_schema()}
    for knob in ("apparmor", "apparmor_fragments", "quirks", "model_tiers", "gc"):
        assert knob in fields, f"profile schema missing {knob}"
    # descriptions come from the dataclass field metadata
    assert "confin" in fields["apparmor"].description.lower()
    assert "fragment" in fields["apparmor_fragments"].description.lower()
    assert fields["quirks"].description and fields["gc"].description


def test_explain_profile_renders_apparmor_fragment_search_path():
    data, text = explain.profile()
    names = {f["name"] for f in data}
    assert {"apparmor", "apparmor_fragments"} <= names
    # the client-side extra-rules mechanism (search path) is spelled out
    assert "apparmor_fragments" in text
    assert ".jaato/apparmor-fragments" in text


def test_explain_profile_resolves_constants_not_source_symbols():
    """A builder must see ACTUAL values, never a constant to chase in source.

    model_tiers' valid keys are resolved from shared.model_tiers at runtime.
    """
    data, text = explain.profile()
    mt = next(f for f in data if f["name"] == "model_tiers")
    assert "dispatcher" in mt["allowed"] and "executor" in mt["allowed"]
    assert "initial" in mt["allowed"] and "fallback" in mt["allowed"]
    # no source symbol / file pointer leaks anywhere in the rendering
    assert "VALID_TIER_NAMES" not in text
    assert "model_tiers.py" not in text


# ------------------------------------------------------------ gap 1: plugins

def test_plugininfo_carries_description_and_config_settings():
    PL = introspect.plugins()
    assert PL, "no plugins discovered"
    # plugin-level description (class docstring first line) for at least one
    assert any(pi.description for pi in PL.values())
    # config_settings is a structured list with a description field
    for pi in PL.values():
        for s in pi.config_settings:
            assert hasattr(s, "name") and hasattr(s, "description")


def test_explain_plugin_threads_config_setting_descriptions():
    """A plugin that declares real PluginSettings surfaces their descriptions.

    Asserted conditionally on such a plugin being installed (lsp is the
    reference), so the test is robust to plugin-set variation.
    """
    PL = introspect.plugins()
    described = [
        s for pi in PL.values() for s in pi.config_settings if s.description
    ]
    if described:  # at least one PluginSetting-based plugin installed
        data, text = explain.plugin(
            next(pi.name for pi in PL.values()
                 if any(s.description for s in pi.config_settings)))
        assert any(c["description"] for c in data["config"])
