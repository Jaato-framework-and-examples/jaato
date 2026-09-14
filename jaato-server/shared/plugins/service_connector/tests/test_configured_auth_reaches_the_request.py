"""`configure_service_auth` must actually change what the next request sends.

Reported from a live cascade: an `apiKey`/header scheme was configured, the
call returned `env_vars_present: [GITLAB_TOKEN]`, and `preview_request`
showed the request going out with no `PRIVATE-TOKEN` header at all.  Every
reasonable auth spelling was tried; the workaround was passing the header by
hand on every `call_service`.  Three defects in one chain:

1. **`preview_request` had the config precedence backwards.**  `call_service`
   reads the stored `<service>/_service.yaml` first and falls back to the
   in-memory discovered cache; `preview_request` did the opposite.  Since
   `configure_service_auth` writes to disk, the preview kept showing the auth
   the OpenAPI spec was PARSED with — an invented `<scheme>_API_KEY` env var,
   because a spec declares the header name and never the credential.  The two
   verbs disagreed about the request, and the one that lied was the one an
   agent uses to check its work.

2. **The in-memory entry was never refreshed**, so the stale config outlived
   the call that was supposed to replace it, for the life of the session.

3. **`build_request` swallowed the resulting `AuthError`** ("for preview, we
   can skip auth errors") and returned a request with no auth header — which
   reads as "this endpoint needs none": the one answer a caller acts on and
   the one that is wrong.

And one that only became reachable once the first three were fixed: the
credential then arrives in a preview that is returned to the MODEL, and
`redact_headers` matched a hardcoded four-name list that no operator-chosen
header (`PRIVATE-TOKEN`, `X-Acme-Key`) is in.  Redaction is now by
provenance — the header names the auth manager actually injected into.
"""

import json
import os

import pytest

from shared.plugins.service_connector.plugin import ServiceConnectorPlugin


SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "gl", "version": "1"},
    "servers": [{"url": "https://gitlab.example/api/v4"}],
    "paths": {"/projects": {"get": {"responses": {}}}},
    "components": {"securitySchemes": {
        "PRIVATE-TOKEN": {"type": "apiKey", "in": "header",
                          "name": "PRIVATE-TOKEN"}}},
    "security": [{"PRIVATE-TOKEN": []}],
}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-0123456789abcdef")
    spec_path = tmp_path / "openapi.json"
    spec_path.write_text(json.dumps(SPEC))

    p = ServiceConnectorPlugin()
    p.set_workspace_path(str(tmp_path))
    p.initialize({})
    ex = p.get_executors()
    assert "error" not in ex["discover_service"]({"source": str(spec_path),
                                                  "alias": "gl"})
    return p, ex


def _configure(ex):
    return ex["configure_service_auth"]({
        "service": "gl",
        "auth": {"type": "apiKey", "in": "header", "name": "PRIVATE-TOKEN",
                 "value_env": "GITLAB_TOKEN"},
    })


def test_configure_reports_the_credential_it_found(svc):
    _, ex = svc
    result = _configure(ex)
    assert result["auth_type"] == "apiKey"
    assert result["env_vars_present"] == ["GITLAB_TOKEN"]
    assert result["env_vars_missing"] == []


def test_the_configured_header_reaches_the_preview(svc):
    """The reported symptom, directly."""
    _, ex = svc
    _configure(ex)
    preview = ex["preview_request"]({"service": "gl", "method": "GET",
                                     "path": "/projects"})
    assert "PRIVATE-TOKEN" in preview["headers"]
    assert "auth_unresolved" not in preview


def test_the_credential_is_redacted_in_what_goes_back_to_the_model(svc):
    """`preview_request`'s output is a tool result — it must not carry a token.

    `PRIVATE-TOKEN` is in no static list of sensitive header names, and never
    could be: an apiKey scheme's header name belongs to the operator.
    """
    _, ex = svc
    _configure(ex)
    preview = ex["preview_request"]({"service": "gl", "method": "GET",
                                     "path": "/projects"})
    rendered = json.dumps(preview)
    assert "glpat-0123456789abcdef" not in rendered
    assert preview["headers"]["PRIVATE-TOKEN"] != "glpat-0123456789abcdef"


def test_the_in_memory_cache_is_refreshed_not_left_stale(svc):
    """Otherwise every later reader serves the spec-parsed auth all session."""
    p, ex = svc
    spec_auth = p._discovered_services["gl"].config.auth
    assert spec_auth.value_env != "GITLAB_TOKEN"     # what discovery invented
    _configure(ex)
    assert p._discovered_services["gl"].config.auth.value_env == "GITLAB_TOKEN"


def test_preview_and_call_agree_on_which_config_is_authoritative(svc):
    """They read the same stored config, so a dry run describes the real call."""
    p, ex = svc
    _configure(ex)
    stored = p._schema_store.load_service_config("gl")
    assert stored.auth.value_env == "GITLAB_TOKEN"
    assert stored.auth.key_name == "PRIVATE-TOKEN"
    preview = ex["preview_request"]({"service": "gl", "method": "GET",
                                     "path": "/projects"})
    assert stored.auth.key_name in preview["headers"]


def test_an_unresolvable_credential_is_reported_not_silently_dropped(tmp_path,
                                                                     monkeypatch):
    monkeypatch.delenv("NO_SUCH_TOKEN", raising=False)
    spec_path = tmp_path / "openapi.json"
    spec_path.write_text(json.dumps(SPEC))
    p = ServiceConnectorPlugin()
    p.set_workspace_path(str(tmp_path))
    p.initialize({})
    ex = p.get_executors()
    ex["discover_service"]({"source": str(spec_path), "alias": "gl"})
    ex["configure_service_auth"]({
        "service": "gl",
        "auth": {"type": "bearer", "token_env": "NO_SUCH_TOKEN"}})

    preview = ex["preview_request"]({"service": "gl", "method": "GET",
                                     "path": "/projects"})
    assert "Authorization" not in preview["headers"]      # honestly absent
    assert "NO_SUCH_TOKEN" in preview["auth_unresolved"]
    assert "UNAUTHENTICATED" in preview["warning"]


def test_an_explicit_auth_override_still_wins_over_the_stored_config(svc,
                                                                    monkeypatch):
    monkeypatch.setenv("OTHER_TOKEN", "bearer-value-here")
    _, ex = svc
    _configure(ex)
    preview = ex["preview_request"]({
        "service": "gl", "method": "GET", "path": "/projects",
        "auth": {"type": "bearer", "token_env": "OTHER_TOKEN"}})
    assert "Authorization" in preview["headers"]
    assert "PRIVATE-TOKEN" not in preview["headers"]
    assert "bearer-value-here" not in json.dumps(preview)   # redacted too
