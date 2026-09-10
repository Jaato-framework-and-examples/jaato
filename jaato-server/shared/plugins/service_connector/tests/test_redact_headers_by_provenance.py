"""Credential redaction cannot work from a fixed list of header names.

`redact_headers` matched four names — `authorization`, `x-api-key`,
`api-key`, `apikey` — which is fine for the schemes that own their header and
useless for the one that does not: an `apiKey` scheme carries whatever header
the API chose (`PRIVATE-TOKEN`, `X-Gitlab-Token`, `X-Acme-Key`), declared in
the OpenAPI spec and unknowable to any list written in advance.

That matters because `preview_request` returns these headers to the MODEL.
The gap was latent only because the auth never reached the preview at all (see
`test_configured_auth_reaches_the_request`); fixing that is what would have
put a live token in the transcript.

So redaction is by PROVENANCE: the caller passes the header names the auth
manager actually resolved a credential into on this request.  The name list
stays for headers a caller supplied by hand, which have no provenance to read.
"""

from shared.plugins.service_connector.auth import AuthManager


LONG = "glpat-0123456789abcdef"


def _mgr():
    return AuthManager()


def test_a_header_the_list_cannot_know_is_redacted_by_provenance():
    out = _mgr().redact_headers({"PRIVATE-TOKEN": LONG},
                                injected=["PRIVATE-TOKEN"])
    assert out["PRIVATE-TOKEN"] != LONG
    assert LONG not in out["PRIVATE-TOKEN"]


def test_provenance_matching_is_case_insensitive():
    out = _mgr().redact_headers({"Private-Token": LONG},
                                injected=["PRIVATE-TOKEN"])
    assert out["Private-Token"] != LONG


def test_the_conventional_names_still_redact_without_provenance():
    """A caller-supplied Authorization header has no provenance to read."""
    out = _mgr().redact_headers({"Authorization": f"Bearer {LONG}"})
    assert LONG not in out["Authorization"]


def test_a_short_secret_is_elided_entirely_not_previewed():
    out = _mgr().redact_headers({"X-Key": "abc"}, injected=["X-Key"])
    assert out["X-Key"] == "***"


def test_ordinary_headers_are_left_readable():
    """Redacting everything would make the preview useless."""
    out = _mgr().redact_headers(
        {"Accept": "application/json", "User-Agent": "jaato/1.0",
         "PRIVATE-TOKEN": LONG},
        injected=["PRIVATE-TOKEN"])
    assert out["Accept"] == "application/json"
    assert out["User-Agent"] == "jaato/1.0"


def test_the_input_dict_is_not_mutated():
    """The REAL request headers are built from the same dict."""
    headers = {"PRIVATE-TOKEN": LONG}
    _mgr().redact_headers(headers, injected=["PRIVATE-TOKEN"])
    assert headers["PRIVATE-TOKEN"] == LONG


def test_no_provenance_argument_keeps_the_old_behaviour():
    """Existing callers pass one argument; they must not start leaking."""
    out = _mgr().redact_headers({"Authorization": LONG, "Accept": "x"})
    assert LONG not in out["Authorization"]
    assert out["Accept"] == "x"


def test_cookie_and_proxy_authorization_are_covered():
    """Both carry credentials and neither was in the original four."""
    out = _mgr().redact_headers({"Cookie": f"session={LONG}",
                                 "Proxy-Authorization": f"Basic {LONG}"})
    assert LONG not in out["Cookie"]
    assert LONG not in out["Proxy-Authorization"]
