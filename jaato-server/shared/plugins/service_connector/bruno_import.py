"""Bruno collection importer for service connector.

Parses Bruno .bru files and converts them to service connector schemas.
Bruno is an open-source API client that stores collections as files.

Bruno .bru file format:
    meta {
      name: Request Name
      type: http
      seq: 1
    }

    get {
      url: {{baseUrl}}/path
      body: none
      auth: bearer
    }

    headers {
      Content-Type: application/json
    }

    query {
      param1: value1
    }

    auth:bearer {
      token: {{accessToken}}
    }

    body:json {
      {
        "key": "value"
      }
    }
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    AuthConfig,
    AuthType,
    EndpointSchema,
    Parameter,
    ParameterLocation,
    RequestBody,
    ServiceConfig,
)


class BrunoParseError(Exception):
    """Error parsing Bruno file."""
    pass


def _parse_bru_block(content: str) -> Dict[str, Dict[str, Any]]:
    """Parse Bruno file content into blocks.

    Args:
        content: Raw .bru file content.

    Returns:
        Dict mapping block names to their contents.
    """
    blocks: Dict[str, Dict[str, Any]] = {}

    # Match block patterns: name { ... } or name:subtype { ... }
    # Use non-greedy matching and handle nested braces
    block_pattern = re.compile(
        r'(\w+(?::\w+)?)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        re.MULTILINE | re.DOTALL
    )

    for match in block_pattern.finditer(content):
        block_name = match.group(1)
        block_content = match.group(2).strip()

        if ':' in block_name:
            # Handle typed blocks like body:json, auth:bearer
            main_name, sub_type = block_name.split(':', 1)
            if main_name not in blocks:
                blocks[main_name] = {}
            blocks[main_name]['_type'] = sub_type
            blocks[main_name]['_content'] = block_content
        else:
            blocks[block_name] = _parse_block_content(block_content)

    return blocks


def _parse_block_content(content: str) -> Dict[str, Any]:
    """Parse key-value pairs from block content.

    Args:
        content: Block content string.

    Returns:
        Dict of parsed key-value pairs.
    """
    result: Dict[str, Any] = {}

    for line in content.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue

        # Handle key: value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Remove trailing comma if present
            if value.endswith(','):
                value = value[:-1].strip()

            # Try to parse as JSON value
            try:
                parsed_value = json.loads(value)
                result[key] = parsed_value
            except json.JSONDecodeError:
                # Keep as string
                result[key] = value

    return result


def _resolve_variables(value: str, env_vars: Dict[str, str]) -> str:
    """Resolve {{variable}} placeholders.

    Args:
        value: String with potential {{var}} placeholders.
        env_vars: Dict of variable names to values.

    Returns:
        String with variables resolved, or placeholder for env lookup.
    """
    def replace_var(match):
        var_name = match.group(1)
        if var_name in env_vars:
            return env_vars[var_name]
        # Return as env var reference
        return f"${{{var_name.upper()}}}"

    return re.sub(r'\{\{(\w+)\}\}', replace_var, value)


def _extract_base_url(url: str) -> Tuple[str, str]:
    """Extract base URL and path from full URL.

    Args:
        url: Full URL (may contain {{variables}}).

    Returns:
        Tuple of (base_url, path).
    """
    # Handle variable base URLs
    if url.startswith('{{'):
        # Find end of variable
        var_end = url.find('}}')
        if var_end != -1:
            path_start = var_end + 2
            base_url = url[:path_start]
            path = url[path_start:] if path_start < len(url) else '/'
            return base_url, path

    # Parse regular URL
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path or '/'
        return base_url, path

    # Assume it's just a path
    return '', url


def parse_bru_file(file_path: str) -> Tuple[Optional[EndpointSchema], Dict[str, Any]]:
    """Parse a Bruno .bru file.

    Args:
        file_path: Path to the .bru file.

    Returns:
        Tuple of (EndpointSchema or None, metadata dict).

    Raises:
        BrunoParseError: If file cannot be parsed.
    """
    path = Path(file_path)
    if not path.exists():
        raise BrunoParseError(f"File not found: {file_path}")

    content = path.read_text(encoding='utf-8')
    blocks = _parse_bru_block(content)

    # Extract metadata
    meta = blocks.get('meta', {})
    name = meta.get('name', path.stem)

    # Find HTTP method block
    http_methods = ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']
    method = None
    method_block = {}

    for m in http_methods:
        if m in blocks:
            method = m.upper()
            method_block = blocks[m]
            break

    if not method:
        # Not an HTTP request file
        return None, {'name': name, 'type': meta.get('type', 'unknown')}

    # Extract URL and parse
    url = method_block.get('url', '')
    base_url, path_str = _extract_base_url(url)

    # Parse parameters
    parameters = []

    # Query parameters
    query_block = blocks.get('query', {})
    for param_name, param_value in query_block.items():
        if not param_name.startswith('_'):
            parameters.append(Parameter(
                name=param_name,
                location=ParameterLocation.QUERY,
                param_type='string',
                default=str(param_value) if param_value else None,
            ))

    # Path parameters (extract from URL)
    path_params = re.findall(r':(\w+)', path_str)
    for param_name in path_params:
        parameters.append(Parameter(
            name=param_name,
            location=ParameterLocation.PATH,
            param_type='string',
            required=True,
        ))
        # Convert :param to {param} format
        path_str = path_str.replace(f':{param_name}', f'{{{param_name}}}')

    # Header parameters
    headers_block = blocks.get('headers', {})

    # Parse request body
    request_body = None
    body_block = blocks.get('body', {})

    if body_block:
        body_type = body_block.get('_type', 'json')
        body_content = body_block.get('_content', '')

        if body_type == 'json' and body_content:
            try:
                # Try to parse as JSON to extract schema
                parsed_body = json.loads(body_content)
                schema = _infer_json_schema(parsed_body)
                request_body = RequestBody(
                    content_type='application/json',
                    schema=schema,
                )
            except json.JSONDecodeError:
                request_body = RequestBody(
                    content_type='application/json',
                )
        elif body_type in ('text', 'xml', 'graphql'):
            content_types = {
                'text': 'text/plain',
                'xml': 'application/xml',
                'graphql': 'application/json',
            }
            request_body = RequestBody(
                content_type=content_types.get(body_type, 'text/plain'),
            )
        elif body_type == 'formUrlEncoded':
            request_body = RequestBody(
                content_type='application/x-www-form-urlencoded',
            )
        elif body_type == 'multipartForm':
            request_body = RequestBody(
                content_type='multipart/form-data',
            )

    # Create endpoint schema
    endpoint = EndpointSchema(
        method=method,
        path=path_str,
        summary=name,
        parameters=parameters,
        request_body=request_body,
        base_url=base_url if base_url else None,
    )

    # Collect metadata
    metadata = {
        'name': name,
        'sequence': meta.get('seq', 0),
        'headers': headers_block,
        'auth_type': method_block.get('auth', 'none'),
    }

    # Parse auth block if present
    auth_block = blocks.get('auth', {})
    if auth_block:
        metadata['auth'] = auth_block

    return endpoint, metadata


def _infer_json_schema(value: Any, max_depth: int = 5) -> Dict[str, Any]:
    """Infer JSON schema from a sample value.

    Args:
        value: Sample value to infer schema from.
        max_depth: Maximum recursion depth.

    Returns:
        Inferred JSON schema.
    """
    if max_depth <= 0:
        return {}

    if value is None:
        return {'type': 'null'}
    elif isinstance(value, bool):
        return {'type': 'boolean'}
    elif isinstance(value, int):
        return {'type': 'integer'}
    elif isinstance(value, float):
        return {'type': 'number'}
    elif isinstance(value, str):
        return {'type': 'string'}
    elif isinstance(value, list):
        if value:
            items_schema = _infer_json_schema(value[0], max_depth - 1)
            return {'type': 'array', 'items': items_schema}
        return {'type': 'array'}
    elif isinstance(value, dict):
        properties = {}
        for k, v in value.items():
            properties[k] = _infer_json_schema(v, max_depth - 1)
        return {
            'type': 'object',
            'properties': properties,
        }

    return {}


def parse_bruno_collection(
    collection_path: str,
    service_name: str,
    base_url_override: Optional[str] = None
) -> Tuple[ServiceConfig, List[EndpointSchema], List[str]]:
    """Parse an entire Bruno collection directory.

    Args:
        collection_path: Path to Bruno collection directory.
        service_name: Name for the service.
        base_url_override: Override base URL from environment files.

    Returns:
        Tuple of (ServiceConfig, list of EndpointSchemas, list of warnings).

    Raises:
        BrunoParseError: If collection cannot be parsed.
    """
    collection_dir = Path(collection_path)
    if not collection_dir.exists():
        raise BrunoParseError(f"Collection directory not found: {collection_path}")

    if not collection_dir.is_dir():
        raise BrunoParseError(f"Path is not a directory: {collection_path}")

    warnings: List[str] = []
    endpoints: List[EndpointSchema] = []
    collected_base_urls: List[str] = []
    auth_config = AuthConfig()

    # Load environment variables from Bruno's environments folder
    env_vars: Dict[str, str] = {}
    env_dir = collection_dir / 'environments'
    if env_dir.exists():
        for env_file in env_dir.glob('*.bru'):
            try:
                content = env_file.read_text(encoding='utf-8')
                blocks = _parse_bru_block(content)
                vars_block = blocks.get('vars', {})
                env_vars.update(vars_block)
            except Exception as e:
                warnings.append(f"Failed to parse environment {env_file.name}: {e}")

    # Parse collection config (bruno.json)
    bruno_json = collection_dir / 'bruno.json'
    collection_title = service_name
    if bruno_json.exists():
        try:
            config_data = json.loads(bruno_json.read_text(encoding='utf-8'))
            collection_title = config_data.get('name', service_name)
        except Exception as e:
            warnings.append(f"Failed to parse bruno.json: {e}")

    # Recursively find all .bru files
    bru_files = list(collection_dir.rglob('*.bru'))

    # Skip environment files
    bru_files = [f for f in bru_files if 'environments' not in f.parts]

    for bru_file in sorted(bru_files):
        try:
            endpoint, metadata = parse_bru_file(str(bru_file))
            if endpoint:
                endpoints.append(endpoint)

                # Collect base URLs
                if endpoint.base_url:
                    resolved_url = _resolve_variables(endpoint.base_url, env_vars)
                    if resolved_url and not resolved_url.startswith('$'):
                        collected_base_urls.append(resolved_url)

                # Check for auth configuration
                auth_type = metadata.get('auth_type', 'none')
                if auth_type != 'none' and auth_config.type == AuthType.NONE:
                    if auth_type == 'bearer':
                        auth_config = AuthConfig(
                            type=AuthType.BEARER,
                            value_env=f"{service_name.upper()}_TOKEN",
                        )
                    elif auth_type == 'basic':
                        auth_config = AuthConfig(
                            type=AuthType.BASIC,
                            username_env=f"{service_name.upper()}_USERNAME",
                            password_env=f"{service_name.upper()}_PASSWORD",
                        )
                    elif auth_type == 'apikey':
                        auth_config = AuthConfig(
                            type=AuthType.API_KEY,
                            key_location=ParameterLocation.HEADER,
                            key_name='X-API-Key',
                            value_env=f"{service_name.upper()}_API_KEY",
                        )

        except BrunoParseError as e:
            warnings.append(f"Failed to parse {bru_file.name}: {e}")
        except Exception as e:
            warnings.append(f"Unexpected error parsing {bru_file.name}: {e}")

    # Determine base URL
    base_url = base_url_override or ''
    if not base_url and collected_base_urls:
        # Use the most common base URL
        from collections import Counter
        base_url = Counter(collected_base_urls).most_common(1)[0][0]

    # Resolve base URL variables
    if base_url.startswith('{{'):
        var_name = re.match(r'\{\{(\w+)\}\}', base_url)
        if var_name:
            var_key = var_name.group(1)
            if var_key in env_vars:
                base_url = env_vars[var_key]
            else:
                # Use env var reference
                base_url = f"${{{var_key.upper()}}}"
                warnings.append(
                    f"Base URL uses variable {var_key}, "
                    f"set {var_key.upper()} environment variable"
                )

    # Create service config
    service_config = ServiceConfig(
        name=service_name,
        base_url=base_url,
        title=collection_title,
        auth=auth_config,
    )

    return service_config, endpoints, warnings
