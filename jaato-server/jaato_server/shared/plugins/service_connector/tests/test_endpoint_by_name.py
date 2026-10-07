"""Tests for the ``endpoint`` argument in ``call_service`` (server 0.6.57+).

The framework now accepts an ``endpoint`` parameter that names the YAML
basename of an endpoint under ``<config_root>/services/<service>/``.
When supplied, ``_execute_call_service`` looks up the endpoint via
``schema_store.list_endpoint_schemas(service)`` and copies the schema's
``method`` and ``path`` into the request — the agent doesn't need to
type the URL path or even the HTTP method.

Empirical motivation: 7:3's build_descriptor agent kept passing the
endpoint *filename* (``"search_full_class"``) where the framework
expected the URL path (``"/solrsearch/select"``).  Maven Central
returned 403 because the filename isn't a real URL path.  Splitting
the two — one arg for the lookup key, the other for the URL — closes
the variance class without forcing the agent to know URL structure.
"""

from unittest.mock import MagicMock

import pytest

from ..plugin import ServiceConnectorPlugin
from ..types import AuthConfig, AuthType, EndpointSchema, ServiceConfig


@pytest.fixture
def plugin():
    """Plugin with mocked schema store + http client.

    The real ``_execute_call_service`` calls ``self._http_client.execute(...)``
    via ``_retry_http_call``, then ``response.to_dict()``.  Mock both
    layers so the test can introspect the kwargs the executor passed
    to ``execute()`` — that's where the resolved ``method`` and ``path``
    end up after endpoint-by-name lookup.
    """
    p = ServiceConnectorPlugin()
    p._schema_store = MagicMock()
    p._http_client = MagicMock()
    p._validator = None
    p._initialized = True

    response = MagicMock()
    response.status = 200
    response.auth_attempts = []
    response.to_dict.return_value = {"status": 200, "body": {}}
    p._http_client.execute.return_value = response

    return p


def _service(name: str = "maven_central") -> ServiceConfig:
    return ServiceConfig(
        name=name,
        base_url="https://search.maven.org",
        title="Maven Central",
        version="1.0",
        auth=AuthConfig(type=AuthType.NONE),
    )


def _schema(method: str = "GET", path: str = "/solrsearch/select") -> EndpointSchema:
    return EndpointSchema(method=method, path=path, summary="Search artifacts")


class TestEndpointByName:
    """Endpoint-by-name resolution via the ``endpoint`` parameter."""

    def test_endpoint_arg_resolves_path_and_method(self, plugin):
        """``endpoint=`` looks up YAML by basename → schema's path + method are used."""
        plugin._schema_store.load_service_config.return_value = _service()
        plugin._schema_store.list_endpoint_schemas.return_value = [
            ("search_full_class", _schema("GET", "/solrsearch/select")),
            ("search_artifact", _schema("GET", "/solrsearch/select")),
        ]

        result = plugin._execute_call_service({
            "service": "maven_central",
            "endpoint": "search_full_class",
            "query": {"q": 'fc:"java.lang.String"', "rows": 1, "wt": "json"},
        })

        assert result.get("error") is None, result
        kwargs = plugin._http_client.execute.call_args.kwargs
        # Method + path resolved from the schema, not the agent's args.
        assert kwargs["method"] == "GET"
        assert kwargs["path"] == "/solrsearch/select"

    def test_endpoint_arg_method_explicit_overrides(self, plugin):
        """When agent passes both ``endpoint`` and ``method``, agent's method wins.

        The schema is authoritative for path (the lookup key); method
        defaults from the schema only when the agent didn't pass one.
        Lets a single endpoint YAML serve method-overlapping verbs
        without ambiguity.
        """
        plugin._schema_store.load_service_config.return_value = _service()
        plugin._schema_store.list_endpoint_schemas.return_value = [
            ("upsert", _schema("PUT", "/items/{id}")),
        ]

        result = plugin._execute_call_service({
            "service": "maven_central",
            "endpoint": "upsert",
            "method": "PATCH",
        })

        assert result.get("error") is None, result
        kwargs = plugin._http_client.execute.call_args.kwargs
        assert kwargs["method"] == "PATCH"
        assert kwargs["path"] == "/items/{id}"

    def test_endpoint_arg_unknown_returns_error(self, plugin):
        """Unknown ``endpoint`` basename surfaces a clear error.

        No silent fallback to the agent's path arg — the contract is
        "endpoint identifies a curated YAML; missing YAML is the
        agent's bug, not a network call we should attempt."
        """
        plugin._schema_store.load_service_config.return_value = _service()
        plugin._schema_store.list_endpoint_schemas.return_value = [
            ("search_artifact", _schema()),
        ]

        result = plugin._execute_call_service({
            "service": "maven_central",
            "endpoint": "does_not_exist",
        })

        assert "error" in result
        assert "does_not_exist" in result["error"]
        assert plugin._http_client.execute.call_count == 0

    def test_endpoint_and_path_are_mutually_exclusive(self, plugin):
        """Passing both ``endpoint`` and ``path`` is rejected.

        The two args drive URL resolution from different sources;
        accepting both creates ambiguity over which one wins.
        """
        result = plugin._execute_call_service({
            "service": "maven_central",
            "endpoint": "search_full_class",
            "path": "/solrsearch/select",
            "method": "GET",
        })

        assert "error" in result
        assert "mutually exclusive" in result["error"]
        assert plugin._http_client.execute.call_count == 0

    def test_path_only_still_works(self, plugin):
        """The legacy path-based lookup remains untouched.

        Agents that already type the URL path verbatim (handoff_test's
        ``policy_admin`` agent, callers without a curated services tree)
        keep working without modification.
        """
        plugin._schema_store.load_service_config.return_value = _service()
        plugin._schema_store.find_endpoint.return_value = _schema("GET", "/solrsearch/select")

        result = plugin._execute_call_service({
            "service": "maven_central",
            "method": "GET",
            "path": "/solrsearch/select",
        })

        assert result.get("error") is None, result
        # endpoint-by-name code path is NOT taken when ``endpoint`` is empty
        # (proves the new path doesn't shadow the old one).
        plugin._schema_store.list_endpoint_schemas.assert_not_called()

    def test_endpoint_without_method_no_schema_match(self, plugin):
        """Endpoint missing AND method missing → clear error, no request fired.

        Tightens the existing ``method is required`` check so the new
        ``endpoint`` parameter doesn't accidentally let methodless calls
        through when the schema lookup fails.
        """
        plugin._schema_store.load_service_config.return_value = _service()
        plugin._schema_store.list_endpoint_schemas.return_value = []

        result = plugin._execute_call_service({
            "service": "maven_central",
            "endpoint": "anything",
        })

        # Either "endpoint not found" OR "method is required" — both are
        # acceptable failures (the lookup-fail check is hit first today).
        assert "error" in result
        assert plugin._http_client.execute.call_count == 0
