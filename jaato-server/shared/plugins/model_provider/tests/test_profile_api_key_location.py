"""Tests for the profile api_key checked-location line.

An ``APIKeyNotFoundError`` / ``TokenNotFoundError`` reports a "Checked
locations" list so users see what the provider looked at before giving up.
The runtime promotes ``plugin_configs.<provider>.api_key`` to
``config.api_key`` (``jaato_runtime.py``) — the HIGHEST-precedence key
source, consulted before env vars and stored credentials — but the
per-provider ``get_checked_credential_locations()`` helpers were env-only
and could not see it, so the error omitted the very first source it
checked.

:func:`shared.plugins.model_provider.base.profile_api_key_location` builds
that line; each provider's ``get_checked_credential_locations(config=...)``
prepends it.  These tests pin: (1) the helper's set/not-set + masking
behaviour, and (2) that every affected provider surfaces the profile knob
as the FIRST checked location, named for that provider.
"""

from __future__ import annotations

import pytest

from ..base import ProviderConfig, profile_api_key_location


# ---- helper unit behaviour --------------------------------------------------

def test_none_config_reports_not_set():
    assert profile_api_key_location(None, "openrouter") == \
        "plugin_configs.openrouter.api_key (profile): not set"


def test_promoted_api_key_field_reported_masked():
    # initialize-time config: runtime promoted the profile knob to config.api_key
    line = profile_api_key_location(
        ProviderConfig(api_key="sk-or-abcdefgh1234"), "openrouter")
    assert line.startswith("plugin_configs.openrouter.api_key (profile): set (")
    assert "sk-or-ab...1234" in line
    assert "abcdefgh1234" not in line  # unmasked middle must not leak


def test_extra_api_key_reported():
    # verify-time config: key still under config.extra['api_key'] (not promoted)
    line = profile_api_key_location(
        ProviderConfig(extra={"api_key": "sk-nim-wxyz98765432"}), "nim")
    assert line == "plugin_configs.nim.api_key (profile): set (sk-nim-w...5432)"


def test_short_key_fully_masked():
    line = profile_api_key_location(ProviderConfig(api_key="short"), "nebius")
    assert line == "plugin_configs.nebius.api_key (profile): set (***)"


def test_empty_config_reports_not_set():
    assert profile_api_key_location(ProviderConfig(extra={}), "github_models") == \
        "plugin_configs.github_models.api_key (profile): not set"


# ---- each provider prepends the profile line --------------------------------

@pytest.mark.parametrize("provider_name, import_path, extra_kwargs", [
    ("openrouter", "shared.plugins.model_provider.openrouter.env", {}),
    ("nebius", "shared.plugins.model_provider.nebius.env", {}),
    ("ovhcloud", "shared.plugins.model_provider.ovhcloud.env", {}),
    ("doubleword", "shared.plugins.model_provider.doubleword.env", {}),
    ("helmcode", "shared.plugins.model_provider.helmcode.env", {}),
    ("nim", "shared.plugins.model_provider.nim.env", {}),
    # github_models' helper also takes an auth_method positional arg.
    ("github_models", "shared.plugins.model_provider.github_models.env",
     {"auth_method": "auto"}),
])
def test_provider_checked_locations_prepend_profile_line(
    provider_name, import_path, extra_kwargs, monkeypatch,
):
    import importlib
    env = importlib.import_module(import_path)

    # github_models' helper is positional-first for auth_method.
    if "auth_method" in extra_kwargs:
        locs = env.get_checked_credential_locations(
            extra_kwargs["auth_method"], config=ProviderConfig(extra={}))
        locs_set = env.get_checked_credential_locations(
            extra_kwargs["auth_method"],
            config=ProviderConfig(api_key="key-abcdefgh1234"))
    else:
        locs = env.get_checked_credential_locations(config=ProviderConfig(extra={}))
        locs_set = env.get_checked_credential_locations(
            config=ProviderConfig(api_key="key-abcdefgh1234"))

    # FIRST checked location is the profile knob, named for this provider.
    assert locs[0] == \
        f"plugin_configs.{provider_name}.api_key (profile): not set"
    assert locs_set[0].startswith(
        f"plugin_configs.{provider_name}.api_key (profile): set (")
