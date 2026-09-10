"""`explain plugin <name>` must describe a parameter a signature cannot.

``_signature`` renders ``configure_service_auth(service, auth)`` — accurate,
and unusable, because ``auth`` is an object whose five accepted shapes ARE the
tool.  The schema declared them all along and ``--json`` carried it, but the
human page showed neither, so the only route to a correct call was reading the
plugin source — for a page whose entire purpose is to remove that step.

The expansion is deliberately shallow (one level) and conditional: a tool
whose arguments are plain scalars gains nothing, and a schema dump is what
``--json`` is for.
"""

from shared.scaffold.explain import _nested_params, _spec_summary, plugin


def test_scalar_only_parameters_add_nothing():
    """No noise on the common case."""
    schema = {"type": "object",
              "properties": {"path": {"type": "string"},
                             "limit": {"type": "integer"}},
              "required": ["path"]}
    assert _nested_params(schema) == []


def test_absent_schema_is_not_an_error():
    assert _nested_params(None) == []
    assert _nested_params({}) == []
    assert _nested_params({"properties": "not-a-dict"}) == []


def test_a_top_level_enum_is_surfaced():
    """A closed set is the whole contract of a parameter; a bare type is not."""
    out = "\n".join(_nested_params({
        "type": "object",
        "properties": {"method": {"type": "string",
                                  "enum": ["GET", "POST"]}}}))
    assert "method" in out and "one of: GET, POST" in out


def test_object_parameter_expands_with_required_markers():
    out = "\n".join(_nested_params({
        "type": "object",
        "properties": {"auth": {
            "type": "object",
            "properties": {"type": {"type": "string", "enum": ["bearer"]},
                           "token_env": {"type": "string",
                                         "description": "env var holding it"}},
            "required": ["type"]}},
        "required": ["auth"]}))
    assert "auth (object, required):" in out
    assert "type*" in out                    # required within the object
    assert "token_env" in out and "*" not in out.split("token_env")[1][:2]
    assert "one of: bearer" in out
    assert "env var holding it" in out       # sub-field descriptions render
    assert "(* = required within this object)" in out


def test_optional_object_says_so():
    out = "\n".join(_nested_params({
        "type": "object",
        "properties": {"opts": {"type": "object",
                                "properties": {"x": {"type": "string"}}}}}))
    assert "opts (object, optional):" in out
    assert "(* = required" not in out        # nothing required — no legend


def test_union_type_renders_joined_not_as_a_repr():
    assert _spec_summary({"type": ["string", "null"]}) == "string|null"


def test_service_connector_page_shows_the_auth_shape():
    """The reported gap, end to end: the page a session actually reads."""
    _, text = plugin("service_connector")
    assert "auth (object, required):" in text
    for field in ("value_env", "token_env", "username_env", "password_env",
                  "token_url", "client_id_env", "client_secret_env"):
        assert field in text
    assert "oauth2_client" in text


def test_configure_service_auth_description_names_the_per_type_fields():
    """The MODEL reads the description, and it used to say almost nothing.

    "Credentials are read from environment variables" is true and does not
    tell a caller that `bearer` wants `token_env` while `apiKey` wants
    `in` + `name` + `value_env`.
    """
    data, _ = plugin("service_connector")
    tool = next(t for t in data["tools"] if t["name"] == "configure_service_auth")
    desc = tool["description"]
    for token in ("apiKey", "bearer", "basic", "oauth2_client",
                  "value_env", "token_env", "username_env", "client_secret_env"):
        assert token in desc, f"{token!r} missing from the tool description"
    # And it must not invite a literal secret — every field names an env var.
    assert "NEVER" in desc and "ENVIRONMENT VARIABLE" in desc
