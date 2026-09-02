"""Application identity: who the framework says is making the requests.

The bug these guard: every product built on the SDK reported to OpenRouter as
``jaato``, because the framework's name was hardcoded as the app's name.  The
tests below pin the three properties that fix has to keep true at once —

1. an application that names itself is the one attributed,
2. jaato keeps its credit through the ``(powered by ...)`` suffix, and
3. an *unconfigured* deployment still sends exactly what it sent before,

— plus the header-safety invariant, since these strings are written verbatim
into HTTP headers.
"""

import pytest

from shared.app_identity import (
    FRAMEWORK_NAME,
    FRAMEWORK_URL,
    MAX_NAME_LENGTH,
    AppIdentity,
    framework_version,
    resolve_app_identity,
)

_APP_ENV_VARS = (
    "JAATO_APP_NAME",
    "JAATO_APP_URL",
    "JAATO_APP_VERSION",
    "JAATO_APP_POWERED_BY",
)


@pytest.fixture(autouse=True)
def clean_app_env(monkeypatch):
    """Resolve against a known-empty environment.

    Autouse because a developer with ``JAATO_APP_NAME`` exported would
    otherwise see every "unconfigured deployment" assertion fail for reasons
    that have nothing to do with the code.
    """
    for var in _APP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# ==================== Defaults ====================


class TestFrameworkDefault:
    """An unconfigured deployment is indistinguishable from before."""

    def test_bare_identity_is_the_framework(self):
        identity = resolve_app_identity()
        assert identity.name == FRAMEWORK_NAME
        assert identity.is_framework

    def test_framework_gets_no_powered_by_suffix(self):
        # "jaato (powered by jaato)" helps nobody.
        assert resolve_app_identity().attribution_title() == FRAMEWORK_NAME

    def test_framework_url_is_the_attribution_url(self):
        assert resolve_app_identity().attribution_url() == FRAMEWORK_URL

    def test_naming_the_app_jaato_is_treated_as_the_framework(self, monkeypatch):
        monkeypatch.setenv("JAATO_APP_NAME", "JAATO")
        assert resolve_app_identity().attribution_title() == "JAATO"


# ==================== Naming an application ====================


class TestApplicationNaming:

    def test_env_name_becomes_the_title(self, monkeypatch):
        monkeypatch.setenv("JAATO_APP_NAME", "Acme Copilot")
        assert (
            resolve_app_identity().attribution_title()
            == "Acme Copilot (powered by jaato)"
        )

    def test_env_url_becomes_the_referer(self, monkeypatch):
        monkeypatch.setenv("JAATO_APP_NAME", "Acme Copilot")
        monkeypatch.setenv("JAATO_APP_URL", "https://acme.example")
        assert resolve_app_identity().attribution_url() == "https://acme.example"

    def test_app_without_url_falls_back_to_the_framework_url(self, monkeypatch):
        # OpenRouter keys rankings on the referer, so an app with no URL is
        # attributed to jaato rather than to nothing.
        monkeypatch.setenv("JAATO_APP_NAME", "Acme Copilot")
        identity = resolve_app_identity()
        assert identity.url is None
        assert identity.attribution_url() == FRAMEWORK_URL

    def test_powered_by_can_be_switched_off(self, monkeypatch):
        monkeypatch.setenv("JAATO_APP_NAME", "Acme Copilot")
        monkeypatch.setenv("JAATO_APP_POWERED_BY", "false")
        assert resolve_app_identity().attribution_title() == "Acme Copilot"

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", ""])
    def test_falsey_words_all_disable_the_suffix(self, monkeypatch, raw):
        monkeypatch.setenv("JAATO_APP_NAME", "Acme")
        monkeypatch.setenv("JAATO_APP_POWERED_BY", raw)
        assert resolve_app_identity().powered_by is False

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on"])
    def test_truthy_words_keep_the_suffix(self, monkeypatch, raw):
        monkeypatch.setenv("JAATO_APP_NAME", "Acme")
        monkeypatch.setenv("JAATO_APP_POWERED_BY", raw)
        assert resolve_app_identity().powered_by is True


# ==================== Precedence ====================


class TestPrecedence:
    """Explicit overrides beat the environment; the environment beats the
    framework default."""

    def test_identity_instance_is_returned_unchanged(self, monkeypatch):
        monkeypatch.setenv("JAATO_APP_NAME", "FromEnv")
        explicit = AppIdentity(name="Explicit")
        assert resolve_app_identity(explicit) is explicit

    def test_mapping_overrides_win_per_field(self, monkeypatch):
        monkeypatch.setenv("JAATO_APP_NAME", "FromEnv")
        monkeypatch.setenv("JAATO_APP_VERSION", "9.9.9")
        identity = resolve_app_identity({"name": "FromCode"})
        assert identity.name == "FromCode"
        # Untouched fields still come from the environment.
        assert identity.version == "9.9.9"

    def test_none_valued_override_keys_do_not_erase_the_environment(
        self, monkeypatch,
    ):
        monkeypatch.setenv("JAATO_APP_NAME", "FromEnv")
        assert resolve_app_identity({"name": None}).name == "FromEnv"

    def test_env_is_read_per_call_not_cached(self, monkeypatch):
        # The daemon overlays a session's env for the duration of a turn; an
        # identity frozen at import would attribute every session to whoever
        # started the process.
        assert resolve_app_identity().is_framework
        monkeypatch.setenv("JAATO_APP_NAME", "Later")
        assert resolve_app_identity().name == "Later"


# ==================== Header safety ====================


class TestSanitisation:
    """These values land verbatim in HTTP headers."""

    def test_crlf_is_stripped_from_the_name(self):
        identity = AppIdentity(name="Acme\r\nX-Injected: 1")
        assert "\r" not in identity.name and "\n" not in identity.name

    def test_crlf_is_stripped_from_the_url(self):
        identity = AppIdentity(name="Acme", url="https://a.example\r\nEvil: 1")
        assert "\r" not in identity.url and "\n" not in identity.url

    def test_name_is_length_capped(self):
        identity = AppIdentity(name="A" * (MAX_NAME_LENGTH + 50))
        assert len(identity.name) == MAX_NAME_LENGTH

    def test_whitespace_only_name_falls_back_to_the_framework(self):
        assert AppIdentity(name="   ").name == FRAMEWORK_NAME

    def test_whitespace_only_env_values_read_as_unset(self, monkeypatch):
        monkeypatch.setenv("JAATO_APP_NAME", "  ")
        assert resolve_app_identity().is_framework

    def test_surrounding_whitespace_is_trimmed(self):
        assert AppIdentity(name="  Acme  ").name == "Acme"


# ==================== User agent ====================


class TestUserAgent:

    def test_framework_user_agent_names_only_the_framework(self):
        ua = AppIdentity().user_agent()
        version = framework_version()
        assert ua == (f"jaato/{version}" if version else "jaato")

    def test_application_user_agent_carries_both(self):
        ua = AppIdentity(name="Acme Copilot", version="1.4.0").user_agent()
        assert ua.startswith("Acme-Copilot/1.4.0 (powered by jaato")

    def test_spaces_in_the_name_do_not_split_the_product_token(self):
        assert AppIdentity(name="Acme Copilot").user_agent().startswith(
            "Acme-Copilot ("
        )

    def test_versionless_app_omits_the_slash(self):
        assert AppIdentity(name="Acme").user_agent().startswith("Acme (")

    def test_opted_out_app_drops_the_framework_half(self):
        ua = AppIdentity(name="Acme", version="2.0", powered_by=False).user_agent()
        assert ua == "Acme/2.0"


# ==================== Serialisation ====================


class TestSerialisation:
    """``to_dict`` is how the identity crosses onto ``ProviderConfig.extra``."""

    def test_round_trip_preserves_every_field(self):
        identity = AppIdentity(
            name="Acme", url="https://acme.example", version="1.0",
            powered_by=False,
        )
        assert AppIdentity.from_dict(identity.to_dict()) == identity

    def test_unset_fields_are_omitted(self):
        assert AppIdentity(name="Acme").to_dict() == {
            "name": "Acme", "powered_by": True,
        }

    def test_from_dict_of_none_is_the_framework(self):
        assert AppIdentity.from_dict(None).is_framework

    def test_unknown_keys_are_ignored(self):
        # A newer producer talking to an older consumer degrades rather
        # than raising.
        identity = AppIdentity.from_dict({"name": "Acme", "future_key": 1})
        assert identity.name == "Acme"


# ==================== Runtime stamping ====================


class TestRuntimeStamping:
    """``JaatoRuntime`` stamps the identity onto every provider config it
    builds, which is how a provider sees it without reading env itself."""

    def _stamped(self, runtime):
        from shared.plugins.model_provider.base import ProviderConfig
        return runtime._inject_session_extras(ProviderConfig(), "sess-1").extra

    def test_explicit_identity_reaches_provider_config(self):
        from shared.jaato_runtime import JaatoRuntime
        runtime = JaatoRuntime(
            app_identity=AppIdentity(name="Acme", url="https://acme.example"),
        )
        assert self._stamped(runtime)["app_identity"] == {
            "name": "Acme", "powered_by": True, "url": "https://acme.example",
        }

    def test_env_identity_reaches_provider_config(self, monkeypatch):
        from shared.jaato_runtime import JaatoRuntime
        monkeypatch.setenv("JAATO_APP_NAME", "FromEnv")
        assert self._stamped(JaatoRuntime())["app_identity"]["name"] == "FromEnv"

    def test_unconfigured_runtime_stamps_nothing(self):
        # Nobody named an application, so the config stays exactly what it
        # was before this mechanism existed and the provider falls back to
        # its own framework defaults.
        from shared.jaato_runtime import JaatoRuntime
        assert "app_identity" not in self._stamped(JaatoRuntime())

    def test_identity_is_resolved_per_call_not_at_construction(
        self, monkeypatch,
    ):
        # A runtime outlives the env overlay of any one session.
        from shared.jaato_runtime import JaatoRuntime
        runtime = JaatoRuntime()
        assert "app_identity" not in self._stamped(runtime)
        monkeypatch.setenv("JAATO_APP_NAME", "Later")
        assert self._stamped(runtime)["app_identity"]["name"] == "Later"


def test_framework_version_matches_installed_metadata():
    """The version reported is the package's, not a literal that can drift."""
    from importlib.metadata import version as pkg_version
    try:
        expected = pkg_version("jaato-server")
    except Exception:  # pragma: no cover — source checkout, never installed
        pytest.skip("jaato-server is not installed in this environment")
    assert framework_version() == expected


def test_module_reads_only_the_catalogued_env_vars():
    """No stowaway env var: the four names are the whole surface.

    ``shared/env_scope.py`` classifies exactly these, and its guard test
    re-derives the list from source — this asserts the same thing from the
    module's side so the intent is visible where the code is.
    """
    import re
    from pathlib import Path
    import shared.app_identity as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    found = set(re.findall(r'os\.environ\.get\(\s*(ENV_\w+)', source))
    assert found == {
        "ENV_APP_NAME", "ENV_APP_URL", "ENV_APP_VERSION", "ENV_APP_POWERED_BY",
    }
    assert {getattr(module, name) for name in found} == set(_APP_ENV_VARS)
