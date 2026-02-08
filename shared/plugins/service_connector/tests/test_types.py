"""Tests for service connector types."""

import pytest

from ..types import (
    AuthConfig,
    AuthType,
    EndpointSchema,
    Parameter,
    ParameterLocation,
    RequestBody,
    ResponseSpec,
    ServiceConfig,
    ValidationError,
    ValidationResult,
)


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_none_auth(self):
        """Test none auth type."""
        auth = AuthConfig()
        assert auth.type == AuthType.NONE

        data = auth.to_dict()
        assert data == {"type": "none"}

    def test_api_key_auth(self):
        """Test API key auth type."""
        auth = AuthConfig(
            type=AuthType.API_KEY,
            key_location=ParameterLocation.HEADER,
            key_name="X-API-Key",
            value_env="MY_API_KEY",
        )

        data = auth.to_dict()
        assert data["type"] == "apiKey"
        assert data["in"] == "header"
        assert data["name"] == "X-API-Key"
        assert data["value_env"] == "MY_API_KEY"

        # Round-trip
        restored = AuthConfig.from_dict(data)
        assert restored.type == AuthType.API_KEY
        assert restored.key_location == ParameterLocation.HEADER
        assert restored.key_name == "X-API-Key"
        assert restored.value_env == "MY_API_KEY"

    def test_bearer_auth(self):
        """Test bearer auth type."""
        auth = AuthConfig(
            type=AuthType.BEARER,
            value_env="MY_TOKEN",
        )

        data = auth.to_dict()
        assert data["type"] == "bearer"
        assert data["token_env"] == "MY_TOKEN"

    def test_basic_auth(self):
        """Test basic auth type."""
        auth = AuthConfig(
            type=AuthType.BASIC,
            username_env="SVC_USER",
            password_env="SVC_PASS",
        )

        data = auth.to_dict()
        assert data["type"] == "basic"
        assert data["username_env"] == "SVC_USER"
        assert data["password_env"] == "SVC_PASS"

    def test_oauth2_auth(self):
        """Test OAuth2 client credentials auth type."""
        auth = AuthConfig(
            type=AuthType.OAUTH2_CLIENT,
            token_url="https://auth.example.com/token",
            client_id_env="CLIENT_ID",
            client_secret_env="CLIENT_SECRET",
            scope="read write",
        )

        data = auth.to_dict()
        assert data["type"] == "oauth2_client"
        assert data["token_url"] == "https://auth.example.com/token"
        assert data["scope"] == "read write"


class TestParameter:
    """Tests for Parameter."""

    def test_query_parameter(self):
        """Test query parameter."""
        param = Parameter(
            name="limit",
            location=ParameterLocation.QUERY,
            param_type="integer",
            default=20,
            description="Max results",
        )

        data = param.to_dict()
        assert data["name"] == "limit"
        assert data["in"] == "query"
        assert data["type"] == "integer"
        assert data["default"] == 20

    def test_path_parameter(self):
        """Test path parameter."""
        param = Parameter(
            name="user_id",
            location=ParameterLocation.PATH,
            param_type="string",
            required=True,
        )

        data = param.to_dict()
        assert data["name"] == "user_id"
        assert data["in"] == "path"
        assert data["required"] is True

    def test_enum_parameter(self):
        """Test parameter with enum."""
        param = Parameter(
            name="status",
            location=ParameterLocation.QUERY,
            param_type="string",
            enum=["active", "inactive", "pending"],
        )

        data = param.to_dict()
        assert data["enum"] == ["active", "inactive", "pending"]


class TestEndpointSchema:
    """Tests for EndpointSchema."""

    def test_simple_endpoint(self):
        """Test simple GET endpoint."""
        endpoint = EndpointSchema(
            method="GET",
            path="/users",
            summary="List users",
        )

        data = endpoint.to_dict()
        assert data["method"] == "GET"
        assert data["path"] == "/users"
        assert data["summary"] == "List users"

    def test_endpoint_with_parameters(self):
        """Test endpoint with parameters."""
        endpoint = EndpointSchema(
            method="GET",
            path="/users/{id}",
            summary="Get user by ID",
            parameters=[
                Parameter(
                    name="id",
                    location=ParameterLocation.PATH,
                    param_type="string",
                    required=True,
                ),
            ],
        )

        data = endpoint.to_dict()
        assert len(data["parameters"]) == 1
        assert data["parameters"][0]["name"] == "id"

    def test_endpoint_with_request_body(self):
        """Test endpoint with request body."""
        endpoint = EndpointSchema(
            method="POST",
            path="/users",
            summary="Create user",
            request_body=RequestBody(
                content_type="application/json",
                required=True,
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                },
            ),
        )

        data = endpoint.to_dict()
        assert data["request_body"]["required"] is True
        assert "properties" in data["request_body"]["schema"]

    def test_endpoint_with_responses(self):
        """Test endpoint with responses."""
        endpoint = EndpointSchema(
            method="GET",
            path="/users",
            responses={
                200: ResponseSpec(
                    status_code=200,
                    description="Success",
                    schema={"type": "array"},
                ),
                404: ResponseSpec(
                    status_code=404,
                    description="Not found",
                ),
            },
        )

        data = endpoint.to_dict()
        assert "200" in data["responses"]
        assert "404" in data["responses"]
        assert data["responses"]["200"]["description"] == "Success"

    def test_round_trip(self):
        """Test from_dict and to_dict round-trip."""
        original = EndpointSchema(
            method="POST",
            path="/orders",
            summary="Create order",
            tags=["orders"],
            parameters=[
                Parameter(
                    name="dry_run",
                    location=ParameterLocation.QUERY,
                    param_type="boolean",
                ),
            ],
            request_body=RequestBody(
                content_type="application/json",
                schema={"type": "object"},
            ),
            responses={
                201: ResponseSpec(status_code=201, description="Created"),
            },
        )

        data = original.to_dict()
        restored = EndpointSchema.from_dict(data)

        assert restored.method == original.method
        assert restored.path == original.path
        assert restored.summary == original.summary
        assert restored.tags == original.tags
        assert len(restored.parameters) == len(original.parameters)
        assert restored.request_body is not None
        assert len(restored.responses) == len(original.responses)


class TestServiceConfig:
    """Tests for ServiceConfig."""

    def test_simple_config(self):
        """Test simple service config."""
        config = ServiceConfig(
            name="test-api",
            base_url="https://api.test.com",
            title="Test API",
        )

        data = config.to_dict()
        assert data["name"] == "test-api"
        assert data["base_url"] == "https://api.test.com"
        assert data["title"] == "Test API"

    def test_config_with_auth(self):
        """Test config with authentication."""
        config = ServiceConfig(
            name="secure-api",
            base_url="https://api.secure.com",
            auth=AuthConfig(
                type=AuthType.BEARER,
                value_env="SECURE_TOKEN",
            ),
        )

        data = config.to_dict()
        assert "auth" in data
        assert data["auth"]["type"] == "bearer"

    def test_config_with_headers(self):
        """Test config with default headers."""
        config = ServiceConfig(
            name="custom-api",
            base_url="https://api.custom.com",
            default_headers={
                "X-Client-ID": "jaato",
            },
        )

        data = config.to_dict()
        assert data["default_headers"]["X-Client-ID"] == "jaato"


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_valid_result(self):
        """Test valid result."""
        result = ValidationResult(valid=True)

        data = result.to_dict()
        assert data["valid"] is True
        assert "errors" not in data

    def test_invalid_result(self):
        """Test invalid result with errors."""
        result = ValidationResult(
            valid=False,
            errors=[
                ValidationError(field="email", error="required field missing"),
                ValidationError(field="age", error="expected integer, got string"),
            ],
        )

        data = result.to_dict()
        assert data["valid"] is False
        assert len(data["errors"]) == 2
