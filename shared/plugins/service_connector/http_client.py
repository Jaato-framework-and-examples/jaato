"""HTTP client for service connector.

Handles HTTP request execution with authentication, validation,
and response processing.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode, urljoin, urlparse

from .auth import AuthError, AuthManager
from .types import (
    AuthConfig,
    AuthType,
    EndpointSchema,
    HttpResponse,
    ParameterLocation,
    PreviewedRequest,
    ServiceConfig,
    ValidationResult,
)


DEFAULT_TIMEOUT = 30000  # 30 seconds in milliseconds
DEFAULT_TRUNCATE_AT = 10000  # 10k characters
DEFAULT_USER_AGENT = "jaato-service-connector/1.0"


class HttpClientError(Exception):
    """HTTP client error."""
    pass


def _substitute_path_params(path: str, params: Dict[str, Any]) -> str:
    """Substitute path parameters into URL path.

    Args:
        path: URL path with {placeholders}.
        params: Dict of parameter values.

    Returns:
        Path with placeholders substituted.
    """
    result = path
    for key, value in params.items():
        placeholder = f"{{{key}}}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    return result


def _build_curl_command(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[str] = None,
    redact_auth: bool = True
) -> str:
    """Build equivalent curl command.

    Args:
        method: HTTP method.
        url: Full URL.
        headers: Request headers.
        body: Request body.
        redact_auth: Whether to redact auth headers.

    Returns:
        Curl command string.
    """
    parts = ["curl", "-X", method]

    for key, value in headers.items():
        if redact_auth and key.lower() in {"authorization", "x-api-key"}:
            value = '"$' + key.upper().replace("-", "_") + '"'
        else:
            value = f'"{value}"'
        parts.append(f'-H "{key}: {value}"')

    if body:
        # Escape for shell
        escaped_body = body.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'-d "{escaped_body}"')

    parts.append(f'"{url}"')

    return " \\\n  ".join(parts)


class ServiceHttpClient:
    """HTTP client for making service requests.

    Handles:
    - URL building from service config and endpoint schema
    - Path parameter substitution
    - Query parameter encoding
    - Request body serialization
    - Response truncation
    - Authentication header injection

    Attributes:
        auth_manager: AuthManager instance for handling authentication.
        default_timeout: Default request timeout in milliseconds.
        default_truncate_at: Default response truncation limit.
    """

    def __init__(
        self,
        auth_manager: Optional[AuthManager] = None,
        default_timeout: int = DEFAULT_TIMEOUT,
        default_truncate_at: int = DEFAULT_TRUNCATE_AT
    ):
        """Initialize the HTTP client.

        Args:
            auth_manager: AuthManager instance. Created if not provided.
            default_timeout: Default timeout in milliseconds.
            default_truncate_at: Default truncation limit in characters.
        """
        self._auth_manager = auth_manager or AuthManager()
        self._default_timeout = default_timeout
        self._default_truncate_at = default_truncate_at

    def build_request(
        self,
        method: str,
        url: Optional[str] = None,
        service_config: Optional[ServiceConfig] = None,
        endpoint_schema: Optional[EndpointSchema] = None,
        path: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[Dict[str, Any], str]] = None,
        auth_override: Optional[Dict[str, Any]] = None,
    ) -> PreviewedRequest:
        """Build a request without executing it.

        Args:
            method: HTTP method.
            url: Full URL (if not using service_config).
            service_config: Service configuration.
            endpoint_schema: Endpoint schema for path/parameters.
            path: URL path (if not using endpoint_schema).
            query: Query parameters.
            headers: Additional headers.
            body: Request body.
            auth_override: Override auth configuration.

        Returns:
            PreviewedRequest with all details.

        Raises:
            HttpClientError: If URL cannot be determined.
        """
        # Determine base URL and path
        if url:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            request_path = parsed.path
            # Include existing query params
            if parsed.query:
                existing_query = dict(
                    pair.split("=", 1) for pair in parsed.query.split("&") if "=" in pair
                )
                query = {**existing_query, **(query or {})}
        elif service_config:
            base_url = service_config.base_url.rstrip("/")
            if endpoint_schema:
                request_path = endpoint_schema.path
            elif path:
                request_path = path
            else:
                raise HttpClientError("Either endpoint_schema or path is required")
        else:
            raise HttpClientError("Either url or service_config is required")

        # Extract path parameters from query and substitute
        path_params = {}
        remaining_query = {}

        if query:
            # Find {placeholders} in path
            placeholders = set(re.findall(r"\{(\w+)\}", request_path))
            for key, value in query.items():
                if key in placeholders:
                    path_params[key] = value
                else:
                    remaining_query[key] = value

        request_path = _substitute_path_params(request_path, path_params)

        # Build full URL
        full_url = urljoin(base_url + "/", request_path.lstrip("/"))

        # Add query parameters
        if remaining_query:
            query_string = urlencode(remaining_query)
            full_url = f"{full_url}?{query_string}"

        # Build headers
        request_headers: Dict[str, str] = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }

        # Add service default headers
        if service_config and service_config.default_headers:
            request_headers.update(service_config.default_headers)

        # Add custom headers
        if headers:
            request_headers.update(headers)

        # Add authentication
        auth_config = None
        if auth_override:
            auth_config = AuthConfig.from_dict(auth_override)
        elif service_config:
            auth_config = service_config.auth

        if auth_config and auth_config.type != AuthType.NONE:
            try:
                auth_headers, auth_query = self._auth_manager.get_auth_headers(
                    auth_config,
                    service_config.name if service_config else None
                )
                request_headers.update(auth_headers)
                if auth_query:
                    # Add auth query params to URL
                    separator = "&" if "?" in full_url else "?"
                    full_url = f"{full_url}{separator}{urlencode(auth_query)}"
            except AuthError:
                # For preview, we can skip auth errors
                pass

        # Serialize body
        body_str = None
        if body is not None:
            if isinstance(body, dict):
                body_str = json.dumps(body)
                request_headers.setdefault("Content-Type", "application/json")
            else:
                body_str = str(body)

        # Build curl command
        curl = _build_curl_command(
            method.upper(),
            full_url,
            self._auth_manager.redact_headers(request_headers),
            body_str,
            redact_auth=True
        )

        return PreviewedRequest(
            method=method.upper(),
            url=full_url,
            headers=self._auth_manager.redact_headers(request_headers),
            body=body_str,
            curl=curl,
        )

    def execute(
        self,
        method: str,
        url: Optional[str] = None,
        service_config: Optional[ServiceConfig] = None,
        endpoint_schema: Optional[EndpointSchema] = None,
        path: Optional[str] = None,
        query: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[Dict[str, Any], str]] = None,
        auth_override: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        truncate_at: Optional[int] = None,
        request_validation: Optional[ValidationResult] = None,
        response_validator: Optional[Any] = None,  # Callable for validation
        verify_ssl: bool = True,
    ) -> HttpResponse:
        """Execute an HTTP request.

        Args:
            method: HTTP method.
            url: Full URL (if not using service_config).
            service_config: Service configuration.
            endpoint_schema: Endpoint schema for path/parameters.
            path: URL path (if not using endpoint_schema).
            query: Query parameters.
            headers: Additional headers.
            body: Request body.
            auth_override: Override auth configuration.
            timeout: Request timeout in milliseconds.
            truncate_at: Response body truncation limit.
            request_validation: Pre-computed request validation result.
            response_validator: Callable to validate response body.
            verify_ssl: Whether to verify SSL certificates. Defaults to True.
                Set to False only for explicitly trusted services with
                certificate issues (e.g., weak key, self-signed).

        Returns:
            HttpResponse with status, headers, body, etc.

        Raises:
            HttpClientError: If request fails.
        """
        # Use defaults
        timeout_ms = timeout or (
            service_config.timeout if service_config else self._default_timeout
        )
        truncate_limit = truncate_at or self._default_truncate_at

        # Build the request
        preview = self.build_request(
            method=method,
            url=url,
            service_config=service_config,
            endpoint_schema=endpoint_schema,
            path=path,
            query=query,
            headers=headers,
            body=body,
            auth_override=auth_override,
        )

        # Re-build headers without redaction for actual request
        request_headers: Dict[str, str] = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }

        if service_config and service_config.default_headers:
            request_headers.update(service_config.default_headers)

        if headers:
            request_headers.update(headers)

        # Add authentication
        auth_config = None
        if auth_override:
            auth_config = AuthConfig.from_dict(auth_override)
        elif service_config:
            auth_config = service_config.auth

        full_url = preview.url

        if auth_config and auth_config.type != AuthType.NONE:
            auth_headers, auth_query = self._auth_manager.get_auth_headers(
                auth_config,
                service_config.name if service_config else None
            )
            request_headers.update(auth_headers)
            if auth_query and "?" not in full_url:
                full_url = f"{full_url}?{urlencode(auth_query)}"
            elif auth_query:
                full_url = f"{full_url}&{urlencode(auth_query)}"

        # Add content type for body
        body_str = preview.body
        if body_str:
            request_headers.setdefault("Content-Type", "application/json")

        # Execute request
        timeout_sec = timeout_ms / 1000.0
        start_time = time.time()

        try:
            import httpx
        except ImportError:
            try:
                import requests
                from shared.http import get_requests_kwargs

                proxy_kwargs = get_requests_kwargs(full_url)

                response = requests.request(
                    method=preview.method,
                    url=full_url,
                    headers=request_headers,
                    data=body_str,
                    timeout=timeout_sec,
                    verify=verify_ssl,
                    **proxy_kwargs
                )
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Parse response
                response_headers = dict(response.headers)
                status = response.status_code

                # Try to parse as JSON
                try:
                    response_body = response.json()
                except (json.JSONDecodeError, ValueError):
                    response_body = response.text

            except ImportError:
                raise HttpClientError(
                    "httpx or requests is required. Install with: pip install httpx"
                )
            except requests.exceptions.Timeout:
                raise HttpClientError(f"Request timed out after {timeout_ms}ms")
            except requests.exceptions.RequestException as e:
                raise HttpClientError(f"Request failed: {e}")
        else:
            from shared.http import get_httpx_kwargs

            proxy_kwargs = get_httpx_kwargs(full_url)

            try:
                with httpx.Client(
                    timeout=timeout_sec, verify=verify_ssl, **proxy_kwargs
                ) as client:
                    response = client.request(
                        method=preview.method,
                        url=full_url,
                        headers=request_headers,
                        content=body_str,
                    )
                    elapsed_ms = int((time.time() - start_time) * 1000)

                    # Parse response
                    response_headers = dict(response.headers)
                    status = response.status_code

                    # Try to parse as JSON
                    try:
                        response_body = response.json()
                    except (json.JSONDecodeError, ValueError):
                        response_body = response.text

            except httpx.TimeoutException:
                raise HttpClientError(f"Request timed out after {timeout_ms}ms")
            except httpx.HTTPError as e:
                raise HttpClientError(f"Request failed: {e}")

        # Truncate response body if needed
        truncated = False
        full_length = None

        if isinstance(response_body, str):
            if len(response_body) > truncate_limit:
                full_length = len(response_body)
                response_body = response_body[:truncate_limit]
                truncated = True
        elif isinstance(response_body, dict):
            # Serialize to check length
            serialized = json.dumps(response_body)
            if len(serialized) > truncate_limit:
                full_length = len(serialized)
                # Truncate the serialized version
                response_body = serialized[:truncate_limit] + "... (truncated)"
                truncated = True

        # Validate response if validator provided
        response_validation = None
        if response_validator and isinstance(response_body, (dict, list)):
            response_validation = response_validator(response_body, status)

        return HttpResponse(
            status=status,
            headers=response_headers,
            body=response_body,
            elapsed_ms=elapsed_ms,
            truncated=truncated,
            full_length=full_length,
            request_validation=request_validation,
            response_validation=response_validation,
        )

    def get_auth_manager(self) -> AuthManager:
        """Get the auth manager instance."""
        return self._auth_manager
