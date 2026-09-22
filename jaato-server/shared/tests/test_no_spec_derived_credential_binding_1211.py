"""An untrusted spec must not bind a daemon credential to its host (#1211).

A model steered by text in its own context (a fetched page, an issue body,
an API response) could set up a credential-exfiltration chain entirely
through AUTO-APPROVED verbs: ``discover_service`` registered a spec whose
``securitySchemes`` KEY the spec author controls, and the OpenAPI parser
derived a credential env-var name from that key
(``value_env=f"{name.upper()}_API_KEY"``).  A scheme keyed ``ANTHROPIC``
named ``ANTHROPIC_API_KEY``, which ``call_service`` then resolved from the
daemon's own environment and sent to the spec's ``servers:`` host.  The one
gate (``call_service``) asks the operator to approve a domain, and the
credential is redacted in the preview, so which secret leaves is invisible
even at the gate that fires.

The agreed fix has two halves, both guarded here:

  A. **Both setup verbs require permission.** ``discover_service`` (registers
     a service + destination) and ``configure_service_auth`` (binds a
     credential) are off the auto-approved list.

  B. **No credential REFERENCE is derived from parsed/imported content.** The
     parser keeps the non-secret STRUCTURE a spec legitimately describes
     (scheme type, header/query location, header/param name) and leaves every
     ``*_env`` unbound.  A human binds the reference afterwards, via the
     now-gated ``configure_service_auth`` or by editing
     ``<service>/_service.yaml`` — so ``from_dict`` still honours an
     operator-authored ``value_env``, which this guard also asserts.

  C. **The resolution invariant.** With a reference stripped, an
     ``AuthConfig`` may carry a non-NONE ``type`` and no reference.
     ``_resolve_credential`` must resolve an empty/None name to "nothing",
     reading NO env var — never ``get_session_env(None)`` and never a
     spec-derived name — and ``get_auth_headers`` must raise a clear "not
     configured" error rather than silently sending an unauthenticated
     request or a daemon credential.
"""

import pytest

from shared.plugins.service_connector import auth as auth_mod
from shared.plugins.service_connector.auth import (
    METHOD_API_KEY,
    AuthError,
    AuthManager,
    _resolve_credential,
)
from shared.plugins.service_connector.openapi_parser import parse_openapi_spec
from shared.plugins.service_connector.plugin import create_plugin
from shared.plugins.service_connector.types import AuthConfig, AuthType

from shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/shared/plugins/service_connector/plugin.py"
_PARSER = "jaato-server/shared/plugins/service_connector/openapi_parser.py"
_AUTH = "jaato-server/shared/plugins/service_connector/auth.py"


REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="""        return [
            # Read-only discovery tools
            "list_endpoints",""",
        replace="""        return [
            # Read-only discovery tools
            "discover_service",
            "list_endpoints",""",
        test="test_setup_verbs_are_not_auto_approved",
        because="discover_service is auto-approved again, so a service can be "
                "registered with no permission prompt",
    ),
    Reversion(
        target=_PARSER,
        find="""                    key_name=scheme.get("name"),
                )
            elif scheme_type == "http":""",
        replace="""                    key_name=scheme.get("name"),
                    value_env=f"{name.upper()}_API_KEY",
                )
            elif scheme_type == "http":""",
        test="test_parser_binds_no_credential_reference_from_the_scheme_key",
        because="the parser again derives an env-var name from the "
                "attacker-controlled securitySchemes key",
    ),
    Reversion(
        target=_AUTH,
        find="""    source = _make_source(name_or_uri, method)

    if not name_or_uri:
        return None, source""",
        replace="""    source = _make_source(name_or_uri, method)

    if not name_or_uri and False:
        return None, source""",
        test="test_a_stripped_config_reads_no_env_var",
        because="an unset credential reference falls through to reading an "
                "env var (get_session_env(None)) instead of resolving to "
                "nothing",
    ),
]


# --- Part A: neither setup verb is auto-approved -------------------------

def test_setup_verbs_are_not_auto_approved():
    approved = create_plugin().get_auto_approved_tools()
    assert "discover_service" not in approved
    assert "configure_service_auth" not in approved
    # call_service was never auto-approved and stays gated.
    assert "call_service" not in approved
    # The read-only / local-file verbs are still auto-approved.
    for read_only in (
        "list_endpoints",
        "get_endpoint_schema",
        "list_schemas",
        "preview_request",
        "save_schema",
        "import_bruno_collection",
        "services",
    ):
        assert read_only in approved


# --- Part B: the parser derives no credential reference ------------------

def _spec_keyed(scheme_name: str) -> dict:
    """An OpenAPI 3 spec whose single apiKey scheme is keyed on a real
    provider name (the exact shape the issue describes)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Evil", "version": "1.0"},
        "servers": [{"url": "https://reports-api.example"}],
        "components": {
            "securitySchemes": {
                scheme_name: {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                },
            },
        },
        "paths": {"/x": {"get": {"responses": {"200": {"description": "OK"}}}}},
    }


@pytest.mark.parametrize("scheme_name", ["ANTHROPIC", "OPENAI"])
def test_parser_binds_no_credential_reference_from_the_scheme_key(scheme_name):
    result = parse_openapi_spec(_spec_keyed(scheme_name), "svc")
    auth = result.config.auth
    # The non-secret STRUCTURE the spec describes is kept...
    assert auth.type == AuthType.API_KEY
    assert auth.key_name == "X-API-Key"
    # ...but NO credential reference is derived from the scheme key.
    assert auth.value_env is None
    assert auth.username_env is None
    assert auth.password_env is None
    assert auth.client_id_env is None
    assert auth.client_secret_env is None


def test_parser_derives_no_bearer_or_basic_reference():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Evil", "version": "1.0"},
        "components": {
            "securitySchemes": {
                "GITHUB": {"type": "http", "scheme": "bearer"},
            },
        },
        "paths": {"/x": {"get": {"responses": {"200": {"description": "OK"}}}}},
    }
    auth = parse_openapi_spec(spec, "svc").config.auth
    assert auth.type == AuthType.BEARER
    assert auth.value_env is None


def test_operator_authored_reference_still_binds():
    """The fix changes only what the PARSER auto-populates.  A hand-authored
    _service.yaml that already sets value_env is an operator binding and must
    keep working unchanged."""
    auth = AuthConfig.from_dict(
        {"type": "apiKey", "in": "header", "name": "PRIVATE-TOKEN",
         "value_env": "GITLAB_TOKEN"}
    )
    assert auth.type == AuthType.API_KEY
    assert auth.value_env == "GITLAB_TOKEN"


# --- Part C: the resolution invariant ------------------------------------

def test_a_stripped_config_reads_no_env_var(monkeypatch):
    """An AuthConfig with a type but no operator-supplied reference must
    resolve to "no credential configured" without reading ANY env var — never
    get_session_env(None), never a spec-derived name."""
    reads = []

    def spy(name):
        reads.append(name)
        return "SHOULD-NEVER-BE-SENT"

    monkeypatch.setattr(auth_mod, "get_session_env", spy)

    # The primitive: an unset reference resolves to None, reading nothing.
    value, _source = _resolve_credential(None, METHOD_API_KEY)
    assert value is None
    assert reads == []

    value, _source = _resolve_credential("", METHOD_API_KEY)
    assert value is None
    assert reads == []

    # End to end: a stripped API_KEY config (type set, no reference) raises a
    # clear "not configured" error and injects no header, reading nothing.
    cfg = AuthConfig(type=AuthType.API_KEY, key_name="X-API-Key")
    with pytest.raises(AuthError):
        AuthManager().get_auth_headers(cfg, service_name="svc")
    assert reads == []
