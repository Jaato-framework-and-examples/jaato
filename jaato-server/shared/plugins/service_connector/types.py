"""Data types for the service connector plugin.

Defines the core data structures for API services, endpoints, schemas,
and authentication configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AuthType(str, Enum):
    """Supported authentication types."""
    NONE = "none"
    API_KEY = "apiKey"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2_CLIENT = "oauth2_client"


class ParameterLocation(str, Enum):
    """Where a parameter is located in the request."""
    PATH = "path"
    QUERY = "query"
    HEADER = "header"


@dataclass
class AuthConfig:
    """Authentication configuration for a service.

    All sensitive values are read from environment variables,
    never stored directly in files.

    Attributes:
        type: Authentication type.
        key_location: For apiKey - where to send it (header/query).
        key_name: For apiKey - the header or query param name.
        value_env: Env var containing the API key or bearer token.
        username_env: For basic auth - env var with username.
        password_env: For basic auth - env var with password.
        token_url: For oauth2_client - token endpoint URL.
        client_id_env: For oauth2_client - env var with client ID.
        client_secret_env: For oauth2_client - env var with client secret.
        scope: For oauth2_client - requested scope.
    """
    type: AuthType = AuthType.NONE

    # apiKey specific
    key_location: Optional[ParameterLocation] = None
    key_name: Optional[str] = None
    value_env: Optional[str] = None

    # basic auth specific
    username_env: Optional[str] = None
    password_env: Optional[str] = None

    # oauth2_client specific
    token_url: Optional[str] = None
    client_id_env: Optional[str] = None
    client_secret_env: Optional[str] = None
    scope: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthConfig":
        """Create AuthConfig from a dictionary."""
        auth_type = AuthType(data.get("type", "none"))
        return cls(
            type=auth_type,
            key_location=ParameterLocation(data["in"]) if data.get("in") else None,
            key_name=data.get("name"),
            value_env=data.get("value_env") or data.get("token_env"),
            username_env=data.get("username_env"),
            password_env=data.get("password_env"),
            token_url=data.get("token_url"),
            client_id_env=data.get("client_id_env"),
            client_secret_env=data.get("client_secret_env"),
            scope=data.get("scope"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {"type": self.type.value}

        if self.type == AuthType.API_KEY:
            if self.key_location:
                result["in"] = self.key_location.value
            if self.key_name:
                result["name"] = self.key_name
            if self.value_env:
                result["value_env"] = self.value_env

        elif self.type == AuthType.BEARER:
            if self.value_env:
                result["token_env"] = self.value_env

        elif self.type == AuthType.BASIC:
            if self.username_env:
                result["username_env"] = self.username_env
            if self.password_env:
                result["password_env"] = self.password_env

        elif self.type == AuthType.OAUTH2_CLIENT:
            if self.token_url:
                result["token_url"] = self.token_url
            if self.client_id_env:
                result["client_id_env"] = self.client_id_env
            if self.client_secret_env:
                result["client_secret_env"] = self.client_secret_env
            if self.scope:
                result["scope"] = self.scope

        return result


@dataclass
class Parameter:
    """A request parameter (path, query, or header).

    Attributes:
        name: Parameter name.
        location: Where the parameter goes (path, query, header).
        param_type: JSON schema type (string, integer, etc.).
        required: Whether the parameter is required.
        default: Default value if not provided.
        description: Human-readable description.
        enum: List of allowed values (if constrained).
    """
    name: str
    location: ParameterLocation
    param_type: str = "string"
    required: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None
    enum: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Parameter":
        """Create Parameter from a dictionary."""
        return cls(
            name=data["name"],
            location=ParameterLocation(data.get("in", "query")),
            param_type=data.get("type", "string"),
            required=data.get("required", False),
            default=data.get("default"),
            description=data.get("description"),
            enum=data.get("enum"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "in": self.location.value,
            "type": self.param_type,
        }
        if self.required:
            result["required"] = True
        if self.default is not None:
            result["default"] = self.default
        if self.description:
            result["description"] = self.description
        if self.enum:
            result["enum"] = self.enum
        return result


@dataclass
class RequestBody:
    """Request body specification.

    Attributes:
        content_type: MIME type (e.g., application/json).
        required: Whether a body is required.
        schema: JSON schema for the body.
    """
    content_type: str = "application/json"
    required: bool = False
    schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestBody":
        """Create RequestBody from a dictionary."""
        return cls(
            content_type=data.get("content_type", "application/json"),
            required=data.get("required", False),
            schema=data.get("schema", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {"content_type": self.content_type}
        if self.required:
            result["required"] = True
        if self.schema:
            result["schema"] = self.schema
        return result


@dataclass
class ResponseSpec:
    """Response specification for a status code.

    Attributes:
        status_code: HTTP status code.
        description: Human-readable description.
        schema: JSON schema for the response body.
    """
    status_code: int
    description: Optional[str] = None
    schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, status_code: int, data: Dict[str, Any]) -> "ResponseSpec":
        """Create ResponseSpec from a dictionary."""
        return cls(
            status_code=status_code,
            description=data.get("description"),
            schema=data.get("schema", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {}
        if self.description:
            result["description"] = self.description
        if self.schema:
            result["schema"] = self.schema
        return result


@dataclass
class EndpointSchema:
    """Complete schema for an API endpoint.

    Attributes:
        method: HTTP method (GET, POST, etc.).
        path: URL path (may contain {placeholders}).
        summary: Brief description.
        description: Detailed description.
        parameters: List of parameters.
        request_body: Request body spec (if applicable).
        responses: Dict of status code -> ResponseSpec.
        tags: OpenAPI tags for grouping.
        base_url: Override base URL for this endpoint.
    """
    method: str
    path: str
    summary: Optional[str] = None
    description: Optional[str] = None
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[RequestBody] = None
    responses: Dict[int, ResponseSpec] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    base_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EndpointSchema":
        """Create EndpointSchema from a dictionary."""
        parameters = [
            Parameter.from_dict(p) for p in data.get("parameters", [])
        ]

        request_body = None
        if data.get("request_body"):
            request_body = RequestBody.from_dict(data["request_body"])

        responses = {}
        for code_str, resp_data in data.get("responses", {}).items():
            code = int(code_str)
            responses[code] = ResponseSpec.from_dict(code, resp_data)

        return cls(
            method=data["method"].upper(),
            path=data["path"],
            summary=data.get("summary"),
            description=data.get("description"),
            parameters=parameters,
            request_body=request_body,
            responses=responses,
            tags=data.get("tags", []),
            base_url=data.get("base_url"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "method": self.method,
            "path": self.path,
        }

        if self.summary:
            result["summary"] = self.summary
        if self.description:
            result["description"] = self.description
        if self.parameters:
            result["parameters"] = [p.to_dict() for p in self.parameters]
        if self.request_body:
            result["request_body"] = self.request_body.to_dict()
        if self.responses:
            result["responses"] = {
                str(code): spec.to_dict()
                for code, spec in self.responses.items()
            }
        if self.tags:
            result["tags"] = self.tags
        if self.base_url:
            result["base_url"] = self.base_url

        return result


@dataclass
class ServiceConfig:
    """Configuration for a discovered or defined service.

    Attributes:
        name: Service name/alias.
        base_url: Base URL for API requests.
        title: Human-readable title.
        version: API version.
        description: Service description.
        auth: Authentication configuration.
        default_headers: Headers to include in every request.
        timeout: Default timeout in milliseconds.
        ssl_trusted: Whether the user has explicitly trusted this service's
            SSL certificate despite verification failures (e.g., weak key,
            self-signed). When True, SSL verification is skipped for all
            requests to this service. Defaults to False (verify SSL).
        proxy_bypass: Whether the user has explicitly opted to bypass the
            configured proxy for this service (e.g., because the service is
            on a local network or the proxy blocks it). When True, requests
            to this service connect directly. Defaults to False (use proxy).
    """
    name: str
    base_url: str
    title: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    auth: AuthConfig = field(default_factory=AuthConfig)
    default_headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30000
    ssl_trusted: bool = False
    proxy_bypass: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceConfig":
        """Create ServiceConfig from a dictionary."""
        auth = AuthConfig()
        if data.get("auth"):
            auth = AuthConfig.from_dict(data["auth"])

        return cls(
            name=data["name"],
            base_url=data["base_url"],
            title=data.get("title"),
            version=data.get("version"),
            description=data.get("description"),
            auth=auth,
            default_headers=data.get("default_headers", {}),
            timeout=data.get("timeout", 30000),
            ssl_trusted=data.get("ssl_trusted", False),
            proxy_bypass=data.get("proxy_bypass", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: Dict[str, Any] = {
            "name": self.name,
            "base_url": self.base_url,
        }

        if self.title:
            result["title"] = self.title
        if self.version:
            result["version"] = self.version
        if self.description:
            result["description"] = self.description
        if self.auth.type != AuthType.NONE:
            result["auth"] = self.auth.to_dict()
        if self.default_headers:
            result["default_headers"] = self.default_headers
        if self.timeout != 30000:
            result["timeout"] = self.timeout
        if self.ssl_trusted:
            result["ssl_trusted"] = True
        if self.proxy_bypass:
            result["proxy_bypass"] = True

        return result


@dataclass
class DiscoveredService:
    """A service discovered from an OpenAPI/Swagger spec.

    Attributes:
        config: Service configuration.
        endpoints: List of endpoint schemas.
        auth_schemes: List of supported auth scheme names.
        source: Where the spec was loaded from (URL or file path).
        warnings: Non-fatal issues encountered during parsing (e.g.,
            unresolvable ``$ref``, unsupported parameter locations).
            The service is still usable; these describe endpoints or
            details that were skipped.
    """
    config: ServiceConfig
    endpoints: List[EndpointSchema] = field(default_factory=list)
    auth_schemes: List[str] = field(default_factory=list)
    source: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Service alias/name."""
        return self.config.name

    @property
    def base_url(self) -> str:
        """Service base URL."""
        return self.config.base_url

    @property
    def endpoint_count(self) -> int:
        """Number of endpoints."""
        return len(self.endpoints)


@dataclass
class ValidationError:
    """A schema validation error.

    Attributes:
        field: Path to the invalid field (e.g., "data.user.email").
        error: Description of the validation failure.
    """
    field: str
    error: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"field": self.field, "error": self.error}


@dataclass
class ValidationResult:
    """Result of schema validation.

    Attributes:
        valid: Whether the data is valid.
        errors: List of validation errors (if any).
    """
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {"valid": self.valid}
        if self.errors:
            result["errors" if not self.valid else "warnings"] = [
                e.to_dict() for e in self.errors
            ]
        return result


@dataclass
class HttpResponse:
    """HTTP response from a service call.

    Attributes:
        status: HTTP status code.
        headers: Response headers.
        body: Response body (parsed JSON or string).
        elapsed_ms: Request duration in milliseconds.
        truncated: Whether body was truncated.
        full_length: Original body length before truncation.
        request_validation: Validation result for the request (if schema exists).
        response_validation: Validation result for the response (if schema exists).
    """
    status: int
    headers: Dict[str, str]
    body: Any
    elapsed_ms: int
    truncated: bool = False
    full_length: Optional[int] = None
    request_validation: Optional[ValidationResult] = None
    response_validation: Optional[ValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool response."""
        result: Dict[str, Any] = {
            "status": self.status,
            "headers": self.headers,
            "body": self.body,
            "elapsed_ms": self.elapsed_ms,
        }

        if self.truncated:
            result["truncated"] = True
            if self.full_length:
                result["full_length"] = self.full_length

        if self.request_validation:
            result["request_validation"] = self.request_validation.to_dict()
        if self.response_validation:
            result["response_validation"] = self.response_validation.to_dict()

        return result


@dataclass
class PreviewedRequest:
    """A previewed HTTP request (dry-run result).

    Attributes:
        method: HTTP method.
        url: Fully resolved URL with query string.
        headers: All headers that would be sent.
        body: Serialized request body.
        curl: Equivalent curl command.
    """
    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[str] = None
    curl: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool response."""
        result: Dict[str, Any] = {
            "method": self.method,
            "url": self.url,
            "headers": self.headers,
        }
        if self.body is not None:
            result["body"] = self.body
        if self.curl:
            result["curl"] = self.curl
        return result
