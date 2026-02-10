"""OpenAPI/Swagger specification parser.

Parses OpenAPI 3.x and Swagger 2.x specifications to extract
service configuration and endpoint schemas.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from .types import (
    AuthConfig,
    AuthType,
    DiscoveredService,
    EndpointSchema,
    Parameter,
    ParameterLocation,
    RequestBody,
    ResponseSpec,
    ServiceConfig,
)


class OpenAPIParseError(Exception):
    """Error parsing OpenAPI specification."""
    pass


def _resolve_ref(spec: Dict[str, Any], ref: str) -> Dict[str, Any]:
    """Resolve a JSON reference in the spec.

    Args:
        spec: The full OpenAPI spec.
        ref: Reference string (e.g., "#/components/schemas/Pet").

    Returns:
        The resolved object.

    Raises:
        OpenAPIParseError: If reference cannot be resolved.
    """
    if not ref.startswith("#/"):
        raise OpenAPIParseError(f"External references not supported: {ref}")

    parts = ref[2:].split("/")
    current = spec

    for part in parts:
        # URL decode part (e.g., %2F -> /)
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise OpenAPIParseError(f"Cannot resolve reference: {ref}")

    return current


def _resolve_all_refs(
    spec: Dict[str, Any],
    obj: Any,
    warnings: Optional[List[str]] = None,
    _seen: Optional[set] = None,
) -> Any:
    """Recursively resolve all $ref in an object.

    Handles circular references by tracking which ``$ref`` paths have
    already been visited in the current resolution chain.  When a cycle
    is detected, the circular ``$ref`` is replaced with an empty dict
    and a warning is recorded (if *warnings* is provided) or an error
    is raised.

    When *warnings* is provided, unresolvable references are replaced
    with an empty dict and a warning message is appended instead of
    raising ``OpenAPIParseError``.  This allows the parser to return
    partial results for specs that contain broken references.

    Args:
        spec: The full OpenAPI spec for reference resolution.
        obj: Object to process.
        warnings: If provided, collects warning strings for
            unresolvable or circular references instead of raising.
        _seen: Internal set tracking ``$ref`` strings already being
            resolved in the current chain (for cycle detection).
            Callers should not pass this parameter.

    Returns:
        Object with all references resolved (or placeholders for
        unresolvable/circular ones when *warnings* is not None).

    Raises:
        OpenAPIParseError: If a reference cannot be resolved and
            *warnings* is None (strict mode).
    """
    if _seen is None:
        _seen = set()

    if isinstance(obj, dict):
        if "$ref" in obj:
            ref = obj["$ref"]

            # Cycle detection
            if ref in _seen:
                if warnings is not None:
                    warnings.append(
                        f"Circular $ref skipped: {ref}"
                    )
                return {}

            try:
                resolved = _resolve_ref(spec, ref)
            except OpenAPIParseError:
                if warnings is not None:
                    warnings.append(
                        f"Unresolvable $ref: {ref}"
                    )
                    return {}
                raise
            # Recursively resolve refs in the resolved object,
            # adding this ref to the seen set for cycle detection.
            return _resolve_all_refs(
                spec, resolved, warnings, _seen | {ref},
            )
        return {
            k: _resolve_all_refs(spec, v, warnings, _seen)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_resolve_all_refs(spec, item, warnings, _seen) for item in obj]
    return obj


def _extract_json_schema(schema_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a simplified JSON schema from OpenAPI schema.

    Args:
        schema_obj: OpenAPI schema object.

    Returns:
        Simplified JSON schema suitable for validation.
    """
    if not schema_obj:
        return {}

    result: Dict[str, Any] = {}

    if "type" in schema_obj:
        result["type"] = schema_obj["type"]

    if "properties" in schema_obj:
        result["properties"] = {
            k: _extract_json_schema(v)
            for k, v in schema_obj["properties"].items()
        }

    if "required" in schema_obj:
        result["required"] = schema_obj["required"]

    if "items" in schema_obj:
        result["items"] = _extract_json_schema(schema_obj["items"])

    if "enum" in schema_obj:
        result["enum"] = schema_obj["enum"]

    if "format" in schema_obj:
        result["format"] = schema_obj["format"]

    if "description" in schema_obj:
        result["description"] = schema_obj["description"]

    if "default" in schema_obj:
        result["default"] = schema_obj["default"]

    if "minimum" in schema_obj:
        result["minimum"] = schema_obj["minimum"]

    if "maximum" in schema_obj:
        result["maximum"] = schema_obj["maximum"]

    if "minLength" in schema_obj:
        result["minLength"] = schema_obj["minLength"]

    if "maxLength" in schema_obj:
        result["maxLength"] = schema_obj["maxLength"]

    if "pattern" in schema_obj:
        result["pattern"] = schema_obj["pattern"]

    # Handle allOf, oneOf, anyOf
    for key in ("allOf", "oneOf", "anyOf"):
        if key in schema_obj:
            result[key] = [_extract_json_schema(s) for s in schema_obj[key]]

    return result


def _parse_parameter_v3(param: Dict[str, Any]) -> Parameter:
    """Parse an OpenAPI 3.x parameter.

    Args:
        param: Parameter object from spec.

    Returns:
        Parsed Parameter.

    Raises:
        OpenAPIParseError: If parameter location is not supported.
    """
    schema = param.get("schema", {})
    location = param.get("in", "query")
    try:
        parsed_location = ParameterLocation(location)
    except ValueError:
        raise OpenAPIParseError(
            f"Parameter '{param.get('name', '?')}' has unsupported location "
            f"'{location}'. Supported: path, query, header."
        )

    return Parameter(
        name=param["name"],
        location=parsed_location,
        param_type=schema.get("type", "string"),
        required=param.get("required", False),
        default=schema.get("default"),
        description=param.get("description"),
        enum=schema.get("enum"),
    )


def _parse_parameter_v2(param: Dict[str, Any]) -> Parameter:
    """Parse a Swagger 2.x parameter.

    Args:
        param: Parameter object from spec.

    Returns:
        Parsed Parameter.

    Raises:
        OpenAPIParseError: If parameter location is not supported.
    """
    location = param.get("in", "query")
    try:
        parsed_location = ParameterLocation(location)
    except ValueError:
        raise OpenAPIParseError(
            f"Parameter '{param.get('name', '?')}' has unsupported location "
            f"'{location}'. Supported: path, query, header."
        )

    return Parameter(
        name=param["name"],
        location=parsed_location,
        param_type=param.get("type", "string"),
        required=param.get("required", False),
        default=param.get("default"),
        description=param.get("description"),
        enum=param.get("enum"),
    )


def _parse_request_body_v3(
    request_body: Dict[str, Any]
) -> Optional[RequestBody]:
    """Parse OpenAPI 3.x request body.

    Args:
        request_body: RequestBody object from spec.

    Returns:
        Parsed RequestBody or None.
    """
    if not request_body:
        return None

    content = request_body.get("content", {})

    # Prefer JSON, fall back to other content types
    content_type = "application/json"
    schema = {}

    if "application/json" in content:
        schema = _extract_json_schema(content["application/json"].get("schema", {}))
    elif content:
        # Take first available content type
        content_type = next(iter(content.keys()))
        schema = _extract_json_schema(content[content_type].get("schema", {}))

    return RequestBody(
        content_type=content_type,
        required=request_body.get("required", False),
        schema=schema,
    )


def _parse_responses(responses: Dict[str, Any]) -> Dict[int, ResponseSpec]:
    """Parse response definitions.

    Args:
        responses: Responses object from spec.

    Returns:
        Dict mapping status codes to ResponseSpec.
    """
    result = {}

    for code_str, response in responses.items():
        # Handle "default" response
        if code_str == "default":
            code = 0
        else:
            try:
                code = int(code_str)
            except ValueError:
                continue

        schema = {}
        # OpenAPI 3.x
        if "content" in response:
            content = response["content"]
            if "application/json" in content:
                schema = _extract_json_schema(
                    content["application/json"].get("schema", {})
                )
        # Swagger 2.x
        elif "schema" in response:
            schema = _extract_json_schema(response["schema"])

        result[code] = ResponseSpec(
            status_code=code,
            description=response.get("description"),
            schema=schema,
        )

    return result


def _parse_security_schemes_v3(
    components: Dict[str, Any]
) -> Tuple[AuthConfig, List[str]]:
    """Parse OpenAPI 3.x security schemes.

    Args:
        components: Components object from spec.

    Returns:
        Tuple of (default AuthConfig, list of scheme names).
    """
    schemes = components.get("securitySchemes", {})
    auth_config = AuthConfig()
    scheme_names = []

    for name, scheme in schemes.items():
        scheme_type = scheme.get("type", "")
        scheme_names.append(name)

        # Use the first scheme as default auth config template
        if auth_config.type == AuthType.NONE:
            if scheme_type == "apiKey":
                auth_config = AuthConfig(
                    type=AuthType.API_KEY,
                    key_location=ParameterLocation(scheme.get("in", "header")),
                    key_name=scheme.get("name"),
                    value_env=f"{name.upper()}_API_KEY",
                )
            elif scheme_type == "http":
                if scheme.get("scheme") == "bearer":
                    auth_config = AuthConfig(
                        type=AuthType.BEARER,
                        value_env=f"{name.upper()}_TOKEN",
                    )
                elif scheme.get("scheme") == "basic":
                    auth_config = AuthConfig(
                        type=AuthType.BASIC,
                        username_env=f"{name.upper()}_USERNAME",
                        password_env=f"{name.upper()}_PASSWORD",
                    )
            elif scheme_type == "oauth2":
                flows = scheme.get("flows", {})
                # Prefer client credentials flow
                if "clientCredentials" in flows:
                    flow = flows["clientCredentials"]
                    auth_config = AuthConfig(
                        type=AuthType.OAUTH2_CLIENT,
                        token_url=flow.get("tokenUrl"),
                        client_id_env=f"{name.upper()}_CLIENT_ID",
                        client_secret_env=f"{name.upper()}_CLIENT_SECRET",
                        scope=" ".join(flow.get("scopes", {}).keys()),
                    )

    return auth_config, scheme_names


def _parse_security_schemes_v2(
    security_defs: Dict[str, Any]
) -> Tuple[AuthConfig, List[str]]:
    """Parse Swagger 2.x security definitions.

    Args:
        security_defs: SecurityDefinitions object from spec.

    Returns:
        Tuple of (default AuthConfig, list of scheme names).
    """
    auth_config = AuthConfig()
    scheme_names = []

    for name, scheme in security_defs.items():
        scheme_type = scheme.get("type", "")
        scheme_names.append(name)

        if auth_config.type == AuthType.NONE:
            if scheme_type == "apiKey":
                auth_config = AuthConfig(
                    type=AuthType.API_KEY,
                    key_location=ParameterLocation(scheme.get("in", "header")),
                    key_name=scheme.get("name"),
                    value_env=f"{name.upper()}_API_KEY",
                )
            elif scheme_type == "basic":
                auth_config = AuthConfig(
                    type=AuthType.BASIC,
                    username_env=f"{name.upper()}_USERNAME",
                    password_env=f"{name.upper()}_PASSWORD",
                )
            elif scheme_type == "oauth2":
                if scheme.get("flow") == "application":
                    auth_config = AuthConfig(
                        type=AuthType.OAUTH2_CLIENT,
                        token_url=scheme.get("tokenUrl"),
                        client_id_env=f"{name.upper()}_CLIENT_ID",
                        client_secret_env=f"{name.upper()}_CLIENT_SECRET",
                        scope=" ".join(scheme.get("scopes", {}).keys()),
                    )

    return auth_config, scheme_names


def _get_base_url_v3(spec: Dict[str, Any]) -> str:
    """Extract base URL from OpenAPI 3.x spec.

    Args:
        spec: OpenAPI spec.

    Returns:
        Base URL string.
    """
    servers = spec.get("servers", [])
    if servers:
        url = servers[0].get("url", "")
        # Handle server variables
        variables = servers[0].get("variables", {})
        for var_name, var_config in variables.items():
            default = var_config.get("default", "")
            url = url.replace(f"{{{var_name}}}", default)
        return url
    return ""


def _get_base_url_v2(spec: Dict[str, Any]) -> str:
    """Extract base URL from Swagger 2.x spec.

    Args:
        spec: Swagger spec.

    Returns:
        Base URL string.
    """
    schemes = spec.get("schemes", ["https"])
    scheme = schemes[0] if schemes else "https"
    host = spec.get("host", "")
    base_path = spec.get("basePath", "")

    if host:
        return f"{scheme}://{host}{base_path}"
    return ""


def _build_form_data_body(form_data_params: List[Dict[str, Any]]) -> RequestBody:
    """Build a RequestBody from Swagger 2.x formData parameters.

    Args:
        form_data_params: List of parameter objects with "in": "formData".

    Returns:
        RequestBody with schema built from the individual parameters.
    """
    has_file = any(p.get("type") == "file" for p in form_data_params)
    content_type = (
        "multipart/form-data" if has_file
        else "application/x-www-form-urlencoded"
    )

    properties = {}
    required_fields = []
    for p in form_data_params:
        prop: Dict[str, Any] = {}
        if "type" in p and p["type"] != "file":
            prop["type"] = p["type"]
        if "description" in p:
            prop["description"] = p["description"]
        if "enum" in p:
            prop["enum"] = p["enum"]
        if "default" in p:
            prop["default"] = p["default"]
        properties[p["name"]] = prop
        if p.get("required", False):
            required_fields.append(p["name"])

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required_fields:
        schema["required"] = required_fields

    return RequestBody(
        content_type=content_type,
        required=bool(required_fields),
        schema=schema,
    )


def parse_openapi_spec(
    spec: Dict[str, Any],
    alias: str,
    source: Optional[str] = None
) -> DiscoveredService:
    """Parse an OpenAPI/Swagger specification.

    Supports both OpenAPI 3.x and Swagger 2.x formats.  The parser is
    *tolerant*: when individual endpoints or parameters contain errors
    (e.g. unresolvable ``$ref``, unsupported parameter locations) they
    are skipped and a warning is recorded.  The caller still receives
    all successfully parsed endpoints and can inspect the ``warnings``
    list on the returned ``DiscoveredService``.

    Args:
        spec: Parsed OpenAPI/Swagger specification dict.
        alias: Alias name for the service.
        source: Source URL or file path.

    Returns:
        DiscoveredService with parsed configuration and endpoints.
        The ``warnings`` field contains human-readable descriptions of
        any non-fatal issues encountered during parsing.

    Raises:
        OpenAPIParseError: If the specification is fundamentally
            invalid (e.g. unsupported version).  Errors scoped to a
            single endpoint or parameter are captured as warnings
            instead.
    """
    warnings: List[str] = []

    # Detect version
    openapi_version = spec.get("openapi", "")
    swagger_version = spec.get("swagger", "")

    is_v3 = openapi_version.startswith("3.")
    is_v2 = swagger_version.startswith("2.")

    if not is_v3 and not is_v2:
        raise OpenAPIParseError(
            f"Unsupported OpenAPI/Swagger version. "
            f"Found openapi={openapi_version}, swagger={swagger_version}"
        )

    # Extract info
    info = spec.get("info", {})
    title = info.get("title", alias)
    version = info.get("version", "")
    description = info.get("description", "")

    # Extract base URL
    if is_v3:
        base_url = _get_base_url_v3(spec)
    else:
        base_url = _get_base_url_v2(spec)

    # Extract security schemes
    if is_v3:
        components = spec.get("components", {})
        auth_config, auth_schemes = _parse_security_schemes_v3(components)
    else:
        security_defs = spec.get("securityDefinitions", {})
        auth_config, auth_schemes = _parse_security_schemes_v2(security_defs)

    # Build service config
    service_config = ServiceConfig(
        name=alias,
        base_url=base_url,
        title=title,
        version=version,
        description=description,
        auth=auth_config,
    )

    # Parse endpoints
    endpoints = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        # Resolve any $ref at path level
        if "$ref" in path_item:
            try:
                path_item = _resolve_ref(spec, path_item["$ref"])
            except OpenAPIParseError as exc:
                warnings.append(
                    f"Skipped path {path}: {exc}"
                )
                continue

        # Get parameters defined at path level
        path_parameters = path_item.get("parameters", [])

        for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
            if method not in path_item:
                continue

            operation = path_item[method]

            # Skip deprecated operations
            if operation.get("deprecated", False):
                continue

            op_label = f"{method.upper()} {path}"

            try:
                endpoint = _parse_single_endpoint(
                    spec, path, method, operation, path_parameters,
                    is_v3, warnings,
                )
            except (OpenAPIParseError, KeyError, TypeError, ValueError) as exc:
                warnings.append(
                    f"Skipped endpoint {op_label}: {exc}"
                )
                continue

            endpoints.append(endpoint)

    return DiscoveredService(
        config=service_config,
        endpoints=endpoints,
        auth_schemes=auth_schemes,
        source=source,
        warnings=warnings,
    )


def _parse_single_endpoint(
    spec: Dict[str, Any],
    path: str,
    method: str,
    operation: Dict[str, Any],
    path_parameters: List[Dict[str, Any]],
    is_v3: bool,
    warnings: List[str],
) -> EndpointSchema:
    """Parse a single endpoint operation from the spec.

    Parameters with unsupported locations or other non-fatal issues are
    skipped and recorded in *warnings*.  Reference resolution errors
    within parameters, request bodies, and responses are also handled
    gracefully.

    Args:
        spec: The full OpenAPI/Swagger spec.
        path: The URL path for this endpoint.
        method: HTTP method (lowercase).
        operation: The operation dict from the spec.
        path_parameters: Parameters defined at the path level.
        is_v3: True if OpenAPI 3.x, False if Swagger 2.x.
        warnings: Mutable list to append warning messages to.

    Returns:
        Parsed EndpointSchema.

    Raises:
        OpenAPIParseError: If the endpoint cannot be parsed at all
            (propagated to caller for skipping).
    """
    op_label = f"{method.upper()} {path}"

    # Combine path-level and operation-level parameters
    op_parameters = operation.get("parameters", [])
    all_parameters = path_parameters + op_parameters

    # Resolve all references (tolerant mode)
    all_parameters = _resolve_all_refs(spec, all_parameters, warnings)

    # Parse parameters
    parameters = []
    request_body = None
    form_data_params = []

    for param in all_parameters:
        # Skip empty dicts produced by unresolvable $refs
        if not param or not param.get("name"):
            continue

        param_in = param.get("in")
        # In Swagger 2.x, body parameter becomes request body
        if param_in == "body":
            schema = _extract_json_schema(param.get("schema", {}))
            request_body = RequestBody(
                content_type="application/json",
                required=param.get("required", False),
                schema=schema,
            )
        elif param_in == "formData":
            form_data_params.append(param)
        else:
            try:
                if is_v3:
                    parameters.append(_parse_parameter_v3(param))
                else:
                    parameters.append(_parse_parameter_v2(param))
            except (OpenAPIParseError, KeyError, ValueError) as exc:
                warnings.append(
                    f"Skipped parameter '{param.get('name', '?')}' "
                    f"on {op_label}: {exc}"
                )

    # Convert formData params to RequestBody (body param takes precedence)
    if form_data_params and request_body is None:
        request_body = _build_form_data_body(form_data_params)

    # OpenAPI 3.x request body
    if is_v3 and "requestBody" in operation:
        rb = _resolve_all_refs(spec, operation["requestBody"], warnings)
        request_body = _parse_request_body_v3(rb)

    # Parse responses
    responses_raw = _resolve_all_refs(
        spec, operation.get("responses", {}), warnings,
    )
    responses = _parse_responses(responses_raw)

    return EndpointSchema(
        method=method.upper(),
        path=path,
        summary=operation.get("summary"),
        description=operation.get("description"),
        parameters=parameters,
        request_body=request_body,
        responses=responses,
        tags=operation.get("tags", []),
    )


def load_spec_from_file(file_path: str) -> Dict[str, Any]:
    """Load an OpenAPI spec from a file.

    Supports JSON and YAML formats.

    Args:
        file_path: Path to the spec file.

    Returns:
        Parsed specification dict.

    Raises:
        OpenAPIParseError: If file cannot be loaded or parsed.
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise OpenAPIParseError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8")

    # Try JSON first
    if path.suffix.lower() == ".json":
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise OpenAPIParseError(f"Invalid JSON: {e}")

    # Try YAML
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        # Fall back to JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise OpenAPIParseError(
                "File appears to be YAML but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            )
    except Exception as e:
        raise OpenAPIParseError(f"Failed to parse spec: {e}")


async def fetch_spec_from_url(
    url: str, timeout: int = 30, verify_ssl: bool = True,
    use_proxy: bool = True,
) -> Dict[str, Any]:
    """Fetch an OpenAPI spec from a URL.

    Args:
        url: URL to fetch the spec from.
        timeout: Request timeout in seconds.
        verify_ssl: Whether to verify SSL certificates. Defaults to True.
            Set to False only for explicitly trusted services with
            certificate issues (e.g., weak key, self-signed).
        use_proxy: Whether to use the configured proxy. Defaults to True.
            Set to False for services that should connect directly.

    Returns:
        Parsed specification dict.

    Raises:
        OpenAPIParseError: If spec cannot be fetched or parsed.
    """
    try:
        import httpx
    except ImportError:
        raise OpenAPIParseError(
            "httpx is required for fetching specs from URLs. "
            "Install with: pip install httpx"
        )

    try:
        client_kwargs: Dict[str, Any] = {"timeout": timeout, "verify": verify_ssl}
        if not use_proxy:
            client_kwargs["proxy"] = None
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(url)
            response.raise_for_status()
            content = response.text
    except httpx.HTTPError as e:
        raise OpenAPIParseError(f"Failed to fetch spec from {url}: {e}")

    # Try JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try YAML
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        raise OpenAPIParseError(
            "Response appears to be YAML but PyYAML is not installed. "
            "Install with: pip install pyyaml"
        )
    except Exception as e:
        raise OpenAPIParseError(f"Failed to parse spec: {e}")


def fetch_spec_from_url_sync(
    url: str, timeout: int = 30, verify_ssl: bool = True,
    use_proxy: bool = True,
) -> Dict[str, Any]:
    """Fetch an OpenAPI spec from a URL (synchronous version).

    Args:
        url: URL to fetch the spec from.
        timeout: Request timeout in seconds.
        verify_ssl: Whether to verify SSL certificates. Defaults to True.
            Set to False only for explicitly trusted services with
            certificate issues (e.g., weak key, self-signed).
        use_proxy: Whether to use the configured proxy. Defaults to True.
            Set to False for services that should connect directly.

    Returns:
        Parsed specification dict.

    Raises:
        OpenAPIParseError: If spec cannot be fetched or parsed.
    """
    try:
        import httpx
    except ImportError:
        try:
            import requests
            from shared.http import get_requests_kwargs

            proxy_kwargs = get_requests_kwargs(url) if use_proxy else {"proxies": {}}
            response = requests.get(
                url, timeout=timeout, verify=verify_ssl, **proxy_kwargs
            )
            response.raise_for_status()
            content = response.text
        except ImportError:
            raise OpenAPIParseError(
                "httpx or requests is required for fetching specs. "
                "Install with: pip install httpx"
            )
        except Exception as e:
            raise OpenAPIParseError(f"Failed to fetch spec from {url}: {e}")
    else:
        from shared.http import get_httpx_kwargs

        proxy_kwargs = get_httpx_kwargs(url) if use_proxy else {"proxy": None}

        try:
            with httpx.Client(
                timeout=timeout, verify=verify_ssl, **proxy_kwargs
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                content = response.text
        except httpx.HTTPError as e:
            raise OpenAPIParseError(f"Failed to fetch spec from {url}: {e}")

    # Try JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try YAML
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        raise OpenAPIParseError(
            "Response appears to be YAML but PyYAML is not installed. "
            "Install with: pip install pyyaml"
        )
    except Exception as e:
        raise OpenAPIParseError(f"Failed to parse spec: {e}")
