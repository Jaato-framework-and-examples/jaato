"""Tests for service connector validation."""

import pytest

from ..types import (
    EndpointSchema,
    Parameter,
    ParameterLocation,
    RequestBody,
    ResponseSpec,
)
from ..validation import SchemaValidator


@pytest.fixture
def validator():
    """Create a SchemaValidator instance."""
    return SchemaValidator()


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_validate_simple_object(self, validator):
        """Test validating a simple object against schema."""
        endpoint = EndpointSchema(
            method="POST",
            path="/users",
            request_body=RequestBody(
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],
                }
            ),
        )

        # Valid data
        result = validator.validate_request_body(
            {"name": "Alice", "age": 30},
            endpoint
        )
        assert result.valid is True
        assert len(result.errors) == 0

        # Missing required field
        result = validator.validate_request_body(
            {"age": 30},
            endpoint
        )
        assert result.valid is False
        assert any("required" in e.error for e in result.errors)

    def test_validate_type_mismatch(self, validator):
        """Test type validation."""
        endpoint = EndpointSchema(
            method="POST",
            path="/data",
            request_body=RequestBody(
                schema={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "enabled": {"type": "boolean"},
                    },
                }
            ),
        )

        # Wrong type
        result = validator.validate_request_body(
            {"count": "not-a-number", "enabled": True},
            endpoint
        )
        assert result.valid is False
        assert any("count" in e.field for e in result.errors)

    def test_validate_string_constraints(self, validator):
        """Test string constraint validation."""
        endpoint = EndpointSchema(
            method="POST",
            path="/users",
            request_body=RequestBody(
                schema={
                    "type": "object",
                    "properties": {
                        "username": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 20,
                        },
                        "email": {
                            "type": "string",
                            "format": "email",
                        },
                    },
                }
            ),
        )

        # Too short
        result = validator.validate_request_body(
            {"username": "ab", "email": "test@example.com"},
            endpoint
        )
        assert result.valid is False
        assert any("minLength" in e.error for e in result.errors)

        # Invalid email
        result = validator.validate_request_body(
            {"username": "alice", "email": "not-an-email"},
            endpoint
        )
        assert result.valid is False
        assert any("email" in e.error for e in result.errors)

    def test_validate_enum(self, validator):
        """Test enum validation."""
        endpoint = EndpointSchema(
            method="POST",
            path="/orders",
            request_body=RequestBody(
                schema={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["pending", "active", "completed"],
                        },
                    },
                }
            ),
        )

        # Valid enum value
        result = validator.validate_request_body(
            {"status": "active"},
            endpoint
        )
        assert result.valid is True

        # Invalid enum value
        result = validator.validate_request_body(
            {"status": "invalid"},
            endpoint
        )
        assert result.valid is False
        assert any("must be one of" in e.error for e in result.errors)

    def test_validate_array(self, validator):
        """Test array validation."""
        endpoint = EndpointSchema(
            method="POST",
            path="/batch",
            request_body=RequestBody(
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                        },
                        "required": ["id"],
                    },
                }
            ),
        )

        # Valid array
        result = validator.validate_request_body(
            [{"id": 1}, {"id": 2}],
            endpoint
        )
        assert result.valid is True

        # Invalid item
        result = validator.validate_request_body(
            [{"id": 1}, {"name": "missing-id"}],
            endpoint
        )
        assert result.valid is False

    def test_validate_number_constraints(self, validator):
        """Test number constraint validation."""
        endpoint = EndpointSchema(
            method="POST",
            path="/config",
            request_body=RequestBody(
                schema={
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                }
            ),
        )

        # Valid value
        result = validator.validate_request_body({"value": 50}, endpoint)
        assert result.valid is True

        # Below minimum
        result = validator.validate_request_body({"value": -1}, endpoint)
        assert result.valid is False

        # Above maximum
        result = validator.validate_request_body({"value": 101}, endpoint)
        assert result.valid is False

    def test_validate_response(self, validator):
        """Test response validation."""
        endpoint = EndpointSchema(
            method="GET",
            path="/users",
            responses={
                200: ResponseSpec(
                    status_code=200,
                    schema={
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                            },
                        },
                    },
                ),
            },
        )

        # Valid response
        result = validator.validate_response_body(
            [{"id": 1, "name": "Alice"}],
            200,
            endpoint
        )
        assert result.valid is True

        # Type mismatch in response
        result = validator.validate_response_body(
            [{"id": "not-int", "name": "Bob"}],
            200,
            endpoint
        )
        assert result.valid is False

    def test_validate_parameters(self, validator):
        """Test parameter validation."""
        endpoint = EndpointSchema(
            method="GET",
            path="/users",
            parameters=[
                Parameter(
                    name="limit",
                    location=ParameterLocation.QUERY,
                    param_type="integer",
                    required=True,
                ),
                Parameter(
                    name="status",
                    location=ParameterLocation.QUERY,
                    param_type="string",
                    enum=["active", "inactive"],
                ),
            ],
        )

        # Valid params
        result = validator.validate_parameters(
            {"limit": 10, "status": "active"},
            endpoint
        )
        assert result.valid is True

        # Missing required
        result = validator.validate_parameters(
            {"status": "active"},
            endpoint
        )
        assert result.valid is False

        # Invalid enum
        result = validator.validate_parameters(
            {"limit": 10, "status": "invalid"},
            endpoint
        )
        assert result.valid is False

    def test_no_schema(self, validator):
        """Test validation when no schema is defined."""
        endpoint = EndpointSchema(method="GET", path="/health")

        # No request body schema - always valid
        result = validator.validate_request_body({"any": "data"}, endpoint)
        assert result.valid is True

        # No response schema - always valid
        result = validator.validate_response_body({"any": "data"}, 200, endpoint)
        assert result.valid is True
