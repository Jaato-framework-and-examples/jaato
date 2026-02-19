"""Request and response validation for service connector.

Validates data against JSON schemas extracted from OpenAPI specs
or manually defined endpoint schemas.
"""

from typing import Any, Dict, List, Optional

from .types import (
    EndpointSchema,
    ValidationError,
    ValidationResult,
)


def _validate_type(value: Any, expected_type: str, path: str) -> List[ValidationError]:
    """Validate value against expected JSON schema type.

    Args:
        value: Value to validate.
        expected_type: Expected JSON schema type.
        path: Path to the value for error reporting.

    Returns:
        List of validation errors (empty if valid).
    """
    errors = []

    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    if expected_type not in type_map:
        return errors  # Unknown type, skip validation

    expected_python_type = type_map[expected_type]

    # Handle null
    if value is None:
        if expected_type != "null":
            errors.append(ValidationError(
                field=path,
                error=f"expected {expected_type}, got null"
            ))
        return errors

    # Check type
    if not isinstance(value, expected_python_type):
        actual_type = type(value).__name__
        errors.append(ValidationError(
            field=path,
            error=f"expected {expected_type}, got {actual_type}"
        ))

    return errors


def _validate_schema(
    value: Any,
    schema: Dict[str, Any],
    path: str = ""
) -> List[ValidationError]:
    """Recursively validate value against JSON schema.

    Args:
        value: Value to validate.
        schema: JSON schema.
        path: Current path for error reporting.

    Returns:
        List of validation errors.
    """
    errors: List[ValidationError] = []

    if not schema:
        return errors

    # Handle type validation
    schema_type = schema.get("type")
    if schema_type:
        errors.extend(_validate_type(value, schema_type, path))
        if errors:
            return errors  # Type mismatch, skip further validation

    # Handle enum
    if "enum" in schema:
        if value not in schema["enum"]:
            errors.append(ValidationError(
                field=path,
                error=f"value must be one of: {schema['enum']}"
            ))

    # Handle string constraints
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(ValidationError(
                field=path,
                error=f"string length must be >= {schema['minLength']}"
            ))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(ValidationError(
                field=path,
                error=f"string length must be <= {schema['maxLength']}"
            ))
        if "pattern" in schema:
            import re
            if not re.match(schema["pattern"], value):
                errors.append(ValidationError(
                    field=path,
                    error=f"string must match pattern: {schema['pattern']}"
                ))
        if "format" in schema:
            # Basic format validation
            fmt = schema["format"]
            if fmt == "email" and "@" not in value:
                errors.append(ValidationError(
                    field=path,
                    error="invalid email format"
                ))
            elif fmt == "uri" and not value.startswith(("http://", "https://")):
                errors.append(ValidationError(
                    field=path,
                    error="invalid URI format"
                ))

    # Handle number constraints
    if isinstance(value, (int, float)):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(ValidationError(
                field=path,
                error=f"value must be >= {schema['minimum']}"
            ))
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(ValidationError(
                field=path,
                error=f"value must be <= {schema['maximum']}"
            ))

    # Handle object validation
    if isinstance(value, dict) and schema_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required fields
        for req_field in required:
            if req_field not in value:
                errors.append(ValidationError(
                    field=f"{path}.{req_field}" if path else req_field,
                    error="required field missing"
                ))

        # Validate each property
        for prop_name, prop_value in value.items():
            if prop_name in properties:
                prop_path = f"{path}.{prop_name}" if path else prop_name
                errors.extend(_validate_schema(
                    prop_value,
                    properties[prop_name],
                    prop_path
                ))

    # Handle array validation
    if isinstance(value, list) and schema_type == "array":
        items_schema = schema.get("items", {})
        for i, item in enumerate(value):
            item_path = f"{path}[{i}]"
            errors.extend(_validate_schema(item, items_schema, item_path))

    # Handle allOf
    if "allOf" in schema:
        for sub_schema in schema["allOf"]:
            errors.extend(_validate_schema(value, sub_schema, path))

    # Handle oneOf (must match exactly one)
    if "oneOf" in schema:
        matches = 0
        for sub_schema in schema["oneOf"]:
            sub_errors = _validate_schema(value, sub_schema, path)
            if not sub_errors:
                matches += 1
        if matches != 1:
            errors.append(ValidationError(
                field=path,
                error=f"must match exactly one of {len(schema['oneOf'])} schemas"
            ))

    # Handle anyOf (must match at least one)
    if "anyOf" in schema:
        any_match = False
        for sub_schema in schema["anyOf"]:
            sub_errors = _validate_schema(value, sub_schema, path)
            if not sub_errors:
                any_match = True
                break
        if not any_match:
            errors.append(ValidationError(
                field=path,
                error=f"must match at least one of {len(schema['anyOf'])} schemas"
            ))

    return errors


class SchemaValidator:
    """Validates requests and responses against endpoint schemas.

    Provides methods for validating:
    - Request bodies against request schema
    - Response bodies against response schema
    - Parameters against parameter definitions
    """

    def validate_request_body(
        self,
        body: Any,
        endpoint_schema: EndpointSchema
    ) -> ValidationResult:
        """Validate a request body against endpoint schema.

        Args:
            body: Request body to validate.
            endpoint_schema: Endpoint schema with request_body definition.

        Returns:
            ValidationResult with valid flag and any errors.
        """
        if not endpoint_schema.request_body:
            # No schema defined, always valid
            return ValidationResult(valid=True)

        schema = endpoint_schema.request_body.schema
        if not schema:
            return ValidationResult(valid=True)

        errors = _validate_schema(body, schema)
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )

    def validate_response_body(
        self,
        body: Any,
        status_code: int,
        endpoint_schema: EndpointSchema
    ) -> ValidationResult:
        """Validate a response body against endpoint schema.

        Args:
            body: Response body to validate.
            status_code: HTTP status code of the response.
            endpoint_schema: Endpoint schema with response definitions.

        Returns:
            ValidationResult with valid flag and any warnings.
        """
        if not endpoint_schema.responses:
            return ValidationResult(valid=True)

        # Find matching response schema
        response_spec = endpoint_schema.responses.get(status_code)

        # Fall back to default (status 0) if defined
        if not response_spec and 0 in endpoint_schema.responses:
            response_spec = endpoint_schema.responses[0]

        if not response_spec or not response_spec.schema:
            return ValidationResult(valid=True)

        errors = _validate_schema(body, response_spec.schema)
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )

    def validate_parameters(
        self,
        params: Dict[str, Any],
        endpoint_schema: EndpointSchema
    ) -> ValidationResult:
        """Validate parameters against endpoint schema.

        Args:
            params: Dict of parameter values.
            endpoint_schema: Endpoint schema with parameter definitions.

        Returns:
            ValidationResult with valid flag and any errors.
        """
        errors: List[ValidationError] = []

        if not endpoint_schema.parameters:
            return ValidationResult(valid=True)

        # Check required parameters
        for param in endpoint_schema.parameters:
            if param.required and param.name not in params:
                errors.append(ValidationError(
                    field=param.name,
                    error="required parameter missing"
                ))
                continue

            if param.name in params:
                value = params[param.name]

                # Type check
                type_errors = _validate_type(value, param.param_type, param.name)
                errors.extend(type_errors)

                # Enum check
                if param.enum and value not in param.enum:
                    errors.append(ValidationError(
                        field=param.name,
                        error=f"value must be one of: {param.enum}"
                    ))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )

    def create_response_validator(
        self,
        endpoint_schema: EndpointSchema
    ):
        """Create a response validator function for use with HTTP client.

        Args:
            endpoint_schema: Endpoint schema for validation.

        Returns:
            Callable that takes (body, status_code) and returns ValidationResult.
        """
        def validator(body: Any, status_code: int) -> ValidationResult:
            return self.validate_response_body(body, status_code, endpoint_schema)

        return validator
