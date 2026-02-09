"""Tests for OpenAPI parser."""

import pytest

from ..openapi_parser import OpenAPIParseError, parse_openapi_spec
from ..types import AuthType, ParameterLocation


class TestOpenAPIParser:
    """Tests for OpenAPI spec parsing."""

    def test_parse_openapi_3_minimal(self):
        """Test parsing minimal OpenAPI 3 spec."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Test API",
                "version": "1.0.0",
            },
            "servers": [{"url": "https://api.test.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "summary": "List users",
                        "responses": {
                            "200": {"description": "Success"},
                        },
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "test-api")

        assert result.name == "test-api"
        assert result.base_url == "https://api.test.com"
        assert result.config.title == "Test API"
        assert result.config.version == "1.0.0"
        assert len(result.endpoints) == 1
        assert result.endpoints[0].method == "GET"
        assert result.endpoints[0].path == "/users"

    def test_parse_swagger_2_minimal(self):
        """Test parsing minimal Swagger 2 spec."""
        spec = {
            "swagger": "2.0",
            "info": {
                "title": "Legacy API",
                "version": "2.0.0",
            },
            "host": "api.legacy.com",
            "basePath": "/v2",
            "schemes": ["https"],
            "paths": {
                "/items": {
                    "post": {
                        "summary": "Create item",
                        "responses": {
                            "201": {"description": "Created"},
                        },
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "legacy-api")

        assert result.name == "legacy-api"
        assert result.base_url == "https://api.legacy.com/v2"
        assert result.config.title == "Legacy API"
        assert len(result.endpoints) == 1
        assert result.endpoints[0].method == "POST"

    def test_parse_with_parameters(self):
        """Test parsing endpoints with parameters."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/users/{id}": {
                    "get": {
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "include",
                                "in": "query",
                                "schema": {"type": "string"},
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "param-api")
        endpoint = result.endpoints[0]

        assert len(endpoint.parameters) == 2

        path_param = next(p for p in endpoint.parameters if p.name == "id")
        assert path_param.location == ParameterLocation.PATH
        assert path_param.required is True

        query_param = next(p for p in endpoint.parameters if p.name == "include")
        assert query_param.location == ParameterLocation.QUERY

    def test_parse_with_request_body(self):
        """Test parsing endpoint with request body."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/users": {
                    "post": {
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                        "responses": {"201": {"description": "Created"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "body-api")
        endpoint = result.endpoints[0]

        assert endpoint.request_body is not None
        assert endpoint.request_body.required is True
        assert endpoint.request_body.content_type == "application/json"
        assert "properties" in endpoint.request_body.schema

    def test_parse_swagger_2_body_parameter(self):
        """Test parsing Swagger 2 body parameter."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/users": {
                    "post": {
                        "parameters": [
                            {
                                "name": "body",
                                "in": "body",
                                "required": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "email": {"type": "string"},
                                    },
                                },
                            },
                        ],
                        "responses": {"201": {"description": "Created"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "swagger-body")
        endpoint = result.endpoints[0]

        assert endpoint.request_body is not None
        assert endpoint.request_body.required is True

    def test_parse_with_responses(self):
        """Test parsing endpoint with response schemas."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/users": {
                    "get": {
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {"type": "object"},
                                        },
                                    },
                                },
                            },
                            "404": {"description": "Not found"},
                        },
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "response-api")
        endpoint = result.endpoints[0]

        assert 200 in endpoint.responses
        assert 404 in endpoint.responses
        assert endpoint.responses[200].schema["type"] == "array"

    def test_parse_with_security_schemes(self):
        """Test parsing security schemes."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "components": {
                "securitySchemes": {
                    "api_key": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                    },
                    "bearer": {
                        "type": "http",
                        "scheme": "bearer",
                    },
                },
            },
            "paths": {
                "/secure": {
                    "get": {
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "secure-api")

        assert len(result.auth_schemes) == 2
        assert "api_key" in result.auth_schemes
        assert "bearer" in result.auth_schemes
        # First scheme becomes default auth config
        assert result.config.auth.type == AuthType.API_KEY

    def test_parse_with_server_variables(self):
        """Test parsing server URL with variables."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "servers": [
                {
                    "url": "https://{environment}.api.com/{version}",
                    "variables": {
                        "environment": {"default": "prod"},
                        "version": {"default": "v1"},
                    },
                },
            ],
            "paths": {},
        }

        result = parse_openapi_spec(spec, "var-api")

        assert result.base_url == "https://prod.api.com/v1"

    def test_parse_with_tags(self):
        """Test parsing endpoint tags."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/users": {
                    "get": {
                        "tags": ["users", "public"],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "tag-api")
        endpoint = result.endpoints[0]

        assert endpoint.tags == ["users", "public"]

    def test_parse_skips_deprecated(self):
        """Test that deprecated endpoints are skipped."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/old": {
                    "get": {
                        "deprecated": True,
                        "responses": {"200": {"description": "OK"}},
                    },
                },
                "/new": {
                    "get": {
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "deprecated-api")

        assert len(result.endpoints) == 1
        assert result.endpoints[0].path == "/new"

    def test_parse_invalid_version(self):
        """Test error on unsupported version."""
        spec = {
            "info": {"title": "Test", "version": "1.0"},
            "paths": {},
        }

        with pytest.raises(OpenAPIParseError) as exc_info:
            parse_openapi_spec(spec, "invalid")

        assert "Unsupported" in str(exc_info.value)

    def test_parse_multiple_methods(self):
        """Test parsing multiple methods on same path."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/items": {
                    "get": {"responses": {"200": {"description": "List"}}},
                    "post": {"responses": {"201": {"description": "Create"}}},
                    "delete": {"responses": {"204": {"description": "Delete all"}}},
                },
            },
        }

        result = parse_openapi_spec(spec, "multi-method")

        assert len(result.endpoints) == 3
        methods = {e.method for e in result.endpoints}
        assert methods == {"GET", "POST", "DELETE"}
