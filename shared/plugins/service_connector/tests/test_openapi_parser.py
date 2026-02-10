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

    def test_parse_swagger_2_formdata_urlencoded(self):
        """Test that formData params become a url-encoded RequestBody."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test", "version": "1.0"},
            "host": "api.example.com",
            "paths": {
                "/upload": {
                    "post": {
                        "parameters": [
                            {
                                "name": "name",
                                "in": "formData",
                                "type": "string",
                                "required": True,
                                "description": "Item name",
                            },
                            {
                                "name": "status",
                                "in": "formData",
                                "type": "string",
                                "enum": ["available", "sold"],
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "form-api")
        endpoint = result.endpoints[0]

        assert endpoint.request_body is not None
        assert endpoint.request_body.content_type == "application/x-www-form-urlencoded"
        assert endpoint.request_body.required is True
        schema = endpoint.request_body.schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "status" in schema["properties"]
        assert schema["properties"]["name"]["description"] == "Item name"
        assert schema["properties"]["status"]["enum"] == ["available", "sold"]
        assert schema["required"] == ["name"]
        # formData params should NOT appear as regular parameters
        assert len(endpoint.parameters) == 0

    def test_parse_swagger_2_formdata_multipart_for_file(self):
        """Test that formData with file type uses multipart/form-data."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test", "version": "1.0"},
            "host": "api.example.com",
            "paths": {
                "/upload": {
                    "post": {
                        "parameters": [
                            {
                                "name": "file",
                                "in": "formData",
                                "type": "file",
                                "required": True,
                            },
                            {
                                "name": "description",
                                "in": "formData",
                                "type": "string",
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "file-api")
        endpoint = result.endpoints[0]

        assert endpoint.request_body is not None
        assert endpoint.request_body.content_type == "multipart/form-data"

    def test_parse_swagger_2_body_wins_over_formdata(self):
        """Test that body param takes precedence over formData (spec violation)."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test", "version": "1.0"},
            "host": "api.example.com",
            "paths": {
                "/mixed": {
                    "post": {
                        "parameters": [
                            {
                                "name": "body",
                                "in": "body",
                                "required": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {"id": {"type": "integer"}},
                                },
                            },
                            {
                                "name": "extra",
                                "in": "formData",
                                "type": "string",
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "mixed-api")
        endpoint = result.endpoints[0]

        # body wins — request_body should be JSON from the body param
        assert endpoint.request_body is not None
        assert endpoint.request_body.content_type == "application/json"

    def test_unknown_parameter_location_produces_warning(self):
        """Test that unknown parameter location is skipped with a warning."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test", "version": "1.0"},
            "host": "api.example.com",
            "paths": {
                "/items": {
                    "get": {
                        "parameters": [
                            {
                                "name": "token",
                                "in": "cookie",
                                "type": "string",
                            },
                            {
                                "name": "limit",
                                "in": "query",
                                "type": "integer",
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "bad-param")

        # The endpoint is still parsed, just without the bad parameter
        assert len(result.endpoints) == 1
        assert len(result.endpoints[0].parameters) == 1
        assert result.endpoints[0].parameters[0].name == "limit"
        # Warning recorded
        assert len(result.warnings) == 1
        assert "token" in result.warnings[0]
        assert "cookie" in result.warnings[0]

    def test_unknown_parameter_location_v3_produces_warning(self):
        """Test that unknown parameter location in v3 is skipped with a warning."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0"},
            "paths": {
                "/items": {
                    "get": {
                        "parameters": [
                            {
                                "name": "token",
                                "in": "banana",
                                "schema": {"type": "string"},
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "bad-param-v3")

        # Endpoint parsed, but the bad parameter was skipped
        assert len(result.endpoints) == 1
        assert len(result.endpoints[0].parameters) == 0
        assert len(result.warnings) == 1
        assert "token" in result.warnings[0]
        assert "banana" in result.warnings[0]

    def test_unresolvable_ref_skipped_with_warning(self):
        """Test that unresolvable $ref in parameter is skipped gracefully."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "z/OS API", "version": "1.0"},
            "host": "mainframe.example.com",
            "basePath": "/api",
            "paths": {
                "/volumes": {
                    "get": {
                        "summary": "List volumes",
                        "parameters": [
                            {"$ref": "#/definitions/z/OS volume object"},
                            {
                                "name": "limit",
                                "in": "query",
                                "type": "integer",
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "zos-api")

        # Service info is still returned
        assert result.name == "zos-api"
        assert result.base_url == "https://mainframe.example.com/api"
        assert len(result.endpoints) == 1
        # The resolvable parameter is kept
        assert len(result.endpoints[0].parameters) == 1
        assert result.endpoints[0].parameters[0].name == "limit"
        # Warning about the unresolvable ref
        assert any("$ref" in w for w in result.warnings)

    def test_unresolvable_ref_in_response_skipped(self):
        """Test that unresolvable $ref in response schema produces a warning."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Test", "version": "1.0"},
            "host": "api.example.com",
            "paths": {
                "/items": {
                    "get": {
                        "summary": "List items",
                        "responses": {
                            "200": {
                                "description": "OK",
                                "schema": {
                                    "$ref": "#/definitions/NonExistent",
                                },
                            },
                        },
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "ref-resp-api")

        assert len(result.endpoints) == 1
        assert any("NonExistent" in w for w in result.warnings)

    def test_mixed_good_and_bad_endpoints(self):
        """Test that good endpoints survive when others fail."""
        spec = {
            "swagger": "2.0",
            "info": {"title": "Mixed API", "version": "1.0"},
            "host": "api.example.com",
            "paths": {
                "/healthy": {
                    "get": {
                        "summary": "Health check",
                        "responses": {"200": {"description": "OK"}},
                    },
                },
                "/broken": {
                    "$ref": "#/definitions/does/not/exist",
                },
                "/also-healthy": {
                    "post": {
                        "summary": "Create thing",
                        "responses": {"201": {"description": "Created"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "mixed-api")

        assert len(result.endpoints) == 2
        methods = {e.method for e in result.endpoints}
        assert methods == {"GET", "POST"}
        assert any("broken" in w or "does/not/exist" in w for w in result.warnings)

    def test_no_warnings_on_clean_spec(self):
        """Test that a valid spec produces no warnings."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Clean API", "version": "1.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {
                        "parameters": [
                            {
                                "name": "page",
                                "in": "query",
                                "schema": {"type": "integer"},
                            },
                        ],
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        }

        result = parse_openapi_spec(spec, "clean-api")

        assert len(result.endpoints) == 1
        assert result.warnings == []
