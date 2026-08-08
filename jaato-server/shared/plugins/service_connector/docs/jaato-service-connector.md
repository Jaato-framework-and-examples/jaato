# Jaato Service Connector — Complete Reference

> Scope: The `service_connector` plugin that enables jaato agents to discover, configure, authenticate, and consume external web services via OpenAPI specs, Bruno collections, or manually defined schemas.

## Table of Contents

1. [What Is the Service Connector?](#1-what-is-the-service-connector)
2. [Architecture Overview](#2-architecture-overview)
3. [Tool Reference](#3-tool-reference)
4. [Data Types](#4-data-types)
5. [Schema Storage (Filesystem)](#5-schema-storage-filesystem)
6. [Authentication](#6-authentication)
7. [HTTP Client](#7-http-client)
8. [OpenAPI Parser](#8-openapi-parser)
9. [Bruno Import](#9-bruno-import)
10. [Validation](#10-validation)
11. [Plugin Lifecycle & Wiring](#11-plugin-lifecycle--wiring)
12. [Configuration Reference](#12-configuration-reference)
13. [Source Code Map](#13-source-code-map)

---

## 1. What Is the Service Connector?

The service connector is a jaato **tool plugin** (`PLUGIN_KIND = "tool"`) that gives agents the ability to interact with external HTTP APIs. It provides:

- **Service discovery** — load OpenAPI/Swagger specifications from URLs or files
- **Endpoint browsing** — list and filter available endpoints by method, path, or tag
- **Schema inspection** — get full request/response schemas for any endpoint
- **HTTP execution** — make authenticated requests with automatic validation
- **Request preview** — dry-run mode showing exactly what would be sent (including curl command)
- **Schema persistence** — store service configs and endpoint schemas as YAML files
- **Bruno import** — import API collections from the Bruno API client format
- **Auth configuration** — set up API key, bearer, basic, or OAuth2 client credentials authentication

The plugin is session-aware: credentials are read from session-scoped environment variables (via `get_session_env()`), not global `os.environ`, so concurrent sessions don't clobber each other's secrets.

---

## 2. Architecture Overview

```
Agent (tool call)
  │
  ▼
ServiceConnectorPlugin
  ├── SchemaStore (filesystem: .jaato/services/)
  ├── AuthManager (credentials from session env)
  ├── ServiceHttpClient (httpx/requests execution)
  ├── SchemaValidator (JSON schema validation)
  ├── OpenAPI parser (spec loading, $ref resolution)
  └── Bruno importer (.bru file parsing)
```

**Data flow for a typical API call:**

1. Agent calls `discover_service(source="https://api.example.com/openapi.json", alias="example")`
2. OpenAPI parser fetches and parses the spec → `DiscoveredService` with endpoints
3. SchemaStore persists to `.jaato/services/_discovered/example.yaml`
4. Agent calls `call_service(service="example", method="GET", path="/users")`
5. Plugin resolves: service config from store → endpoint schema → auth headers from env → URL building → HTTP execution → response validation → truncated result

---

## 3. Tool Reference

### `discover_service`

Load and parse an OpenAPI/Swagger specification.

```json
{
  "source": "https://api.github.com/openapi.json",
  "alias": "github",
  "insecure": false,
  "no_proxy": false
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source` | string | Yes | URL or file path to the OpenAPI/Swagger spec |
| `alias` | string | Yes | Short name to reference this service |
| `insecure` | boolean | No | Skip SSL verification; marks service as trusted for future requests |
| `no_proxy` | boolean | No | Bypass configured HTTP proxy; marks service for future direct connections |

Supports both OpenAPI 3.x and Swagger 2.x. The parser resolves `$ref` references (with cycle detection) and extracts endpoints, auth schemes, and base URLs. When `insecure` or `no_proxy` is set, the resulting `ServiceConfig` persists `ssl_trusted` or `proxy_bypass` flags.

### `list_endpoints`

List available endpoints from a discovered service or schema directory.

```json
{
  "service": "github",
  "filter_method": "GET",
  "filter_path": "/repos/*",
  "filter_tag": "repositories"
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `service` | string | Yes | Service alias or schema directory name |
| `filter_method` | string | No | Filter by HTTP method |
| `filter_path` | string | No | Filter by path pattern (glob-style) |
| `filter_tag` | string | No | Filter by OpenAPI tag |

### `get_endpoint_schema`

Get the full request/response schema for a specific endpoint.

```json
{
  "service": "github",
  "method": "GET",
  "path": "/repos/{owner}/{repo}"
}
```

Returns parameters, request body schema, and response schemas with types and constraints.

### `call_service`

Execute an HTTP request. Can use a discovered service (with auth and base URL) or a raw URL.

```json
{
  "service": "github",
  "method": "GET",
  "path": "/repos/{owner}/{repo}",
  "query": {"owner": "octocat", "repo": "hello-world"},
  "headers": {"Accept": "application/vnd.github.v3+json"},
  "timeout": 10000,
  "truncate_at": 5000,
  "save_to": "response.json",
  "insecure": false,
  "no_proxy": false
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `service` | string | No | Service alias (uses stored base_url and auth) |
| `url` | string | No | Full URL (alternative to service + path) |
| `method` | string | Yes | HTTP method |
| `path` | string | No | Endpoint path (when using service) |
| `query` | object | No | Query parameters |
| `headers` | object | No | Additional headers |
| `body` | object/string | No | Request body |
| `auth` | object | No | Override auth for this request |
| `timeout` | integer | No | Timeout in milliseconds (default: 30000) |
| `truncate_at` | integer | No | Truncate response at N characters (default: 10000) |
| `save_to` | string | No | Save response body to file |
| `insecure` | boolean | No | Skip SSL verification for this request |
| `no_proxy` | boolean | No | Bypass proxy for this request |

Request body is validated against schema if available. Response is validated and truncated. The response includes `request_validation` and `response_validation` results when schemas exist.

### `preview_request`

Dry-run showing exactly what would be sent, including curl command.

```json
{
  "service": "github",
  "method": "POST",
  "path": "/repos/{owner}/{repo}/issues",
  "body": {"title": "Bug", "body": "Description"}
}
```

Returns method, full URL, headers, body, and equivalent curl command. Auth headers are redacted in the curl output.

### `save_schema`

Persist a request/response schema to the filesystem.

```json
{
  "service": "github",
  "name": "create_issue",
  "method": "POST",
  "path": "/repos/{owner}/{repo}/issues"
}
```

### `list_schemas`

List all stored schemas across all services. Returns service name, endpoint name, method, path, and summary.

### `import_bruno_collection`

Import endpoints from a Bruno API collection directory.

```json
{
  "path": "/path/to/bruno-collection",
  "service_name": "myapi",
  "base_url": "https://api.example.com"
}
```

### `configure_service_auth`

Set up authentication for a service.

```json
{
  "service": "github",
  "auth": {
    "type": "bearer",
    "token_env": "GITHUB_TOKEN"
  }
}
```

---

## 4. Data Types

### `AuthType` (enum)

| Value | Description |
|-------|-------------|
| `none` | No authentication |
| `apiKey` | API key in header or query parameter |
| `bearer` | Bearer token in Authorization header |
| `basic` | HTTP Basic authentication |
| `oauth2_client` | OAuth2 client credentials flow |

### `AuthConfig`

```python
@dataclass
class AuthConfig:
    type: AuthType = AuthType.NONE
    key_location: Optional[ParameterLocation] = None  # header or query
    key_name: Optional[str] = None                     # e.g., "X-API-Key"
    value_env: Optional[str] = None                    # env var for API key/token
    username_env: Optional[str] = None                 # env var for basic auth
    password_env: Optional[str] = None                 # env var for basic auth
    token_url: Optional[str] = None                    # OAuth2 token endpoint
    client_id_env: Optional[str] = None                # env var for OAuth2 client ID
    client_secret_env: Optional[str] = None            # env var for OAuth2 client secret
    scope: Optional[str] = None                        # OAuth2 scope
```

All sensitive values reference environment variables — secrets are never stored directly.

### `ServiceConfig`

```python
@dataclass
class ServiceConfig:
    name: str                              # Service alias
    base_url: str                          # Base URL for API requests
    title: Optional[str]                   # Human-readable title
    version: Optional[str]                 # API version
    description: Optional[str]             # Service description
    auth: AuthConfig                       # Authentication configuration
    default_headers: Dict[str, str]        # Headers included in every request
    timeout: int = 30000                   # Default timeout (ms)
    ssl_trusted: bool = False             # Skip SSL verification
    proxy_bypass: bool = False            # Bypass configured proxy
```

### `EndpointSchema`

```python
@dataclass
class EndpointSchema:
    method: str                            # HTTP method (GET, POST, etc.)
    path: str                              # URL path with {placeholders}
    summary: Optional[str]
    description: Optional[str]
    parameters: List[Parameter]            # Path, query, header params
    request_body: Optional[RequestBody]
    responses: Dict[int, ResponseSpec]     # Status code → response spec
    tags: List[str]
    base_url: Optional[str]               # Per-endpoint base URL override
```

### `Parameter`

```python
@dataclass
class Parameter:
    name: str
    location: ParameterLocation            # path, query, header
    param_type: str = "string"
    required: bool = False
    default: Optional[Any]
    description: Optional[str]
    enum: Optional[List[str]]
```

### `RequestBody`

```python
@dataclass
class RequestBody:
    content_type: str = "application/json"
    required: bool = False
    schema: Dict[str, Any]
```

### `ResponseSpec`

```python
@dataclass
class ResponseSpec:
    status_code: int
    description: Optional[str]
    schema: Dict[str, Any]
```

### `HttpResponse`

```python
@dataclass
class HttpResponse:
    status: int
    headers: Dict[str, str]
    body: Any                              # Parsed JSON or string
    elapsed_ms: int
    truncated: bool = False
    full_length: Optional[int]
    request_validation: Optional[ValidationResult]
    response_validation: Optional[ValidationResult]
```

### `PreviewedRequest`

```python
@dataclass
class PreviewedRequest:
    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[str]
    curl: Optional[str]
```

### `ValidationResult`

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError]
```

### `ValidationError`

```python
@dataclass
class ValidationError:
    field: str        # e.g., "data.user.email"
    error: str        # e.g., "expected string, got integer"
```

### `DiscoveredService`

```python
@dataclass
class DiscoveredService:
    config: ServiceConfig
    endpoints: List[EndpointSchema]
    auth_schemes: List[str]
    source: Optional[str]                 # URL or file path where spec was loaded
    warnings: List[str]                   # Non-fatal parsing issues
```

---

## 5. Schema Storage (Filesystem)

Services and schemas are stored as YAML files under the workspace:

```
.jaato/services/
├── _discovered/                    # Auto-cached from OpenAPI specs
│   └── github.yaml                 # Single file: config + all endpoints
├── myapi/                          # Manually defined service
│   ├── _service.yaml               # Service configuration
│   ├── create_user.yaml            # Endpoint schema
│   └── list_users.yaml             # Endpoint schema
```

### `SchemaStore` — Key Operations

| Method | Description |
|--------|-------------|
| `save_service_config(config)` | Save `_service.yaml` for a manual service |
| `load_service_config(name)` | Load config (checks manual dir, then `_discovered/`) |
| `list_services()` | List all service names |
| `delete_service(name)` | Remove service and all its schemas |
| `save_endpoint_schema(service, name, schema)` | Save endpoint YAML |
| `load_endpoint_schema(service, name)` | Load endpoint YAML |
| `list_endpoint_schemas(service)` | List all endpoints for a service |
| `find_endpoint(service, method, path)` | Find by method+path across discovered and manual |
| `save_discovered_service(name, config, endpoints, raw_spec, source)` | Cache full OpenAPI parse result |
| `load_discovered_service(name)` | Load cached service (config + endpoints) |

**Persistence format** — discovered services are a single YAML file containing `config`, `endpoints`, `source`, and optionally `raw_spec`. Manual services use separate YAML files per endpoint alongside `_service.yaml`.

The store requires PyYAML (`pip install pyyaml`). Workspace path is set via `set_workspace_path()` — the store resolves relative to `{workspace}/.jaato/services/`.

---

## 6. Authentication

The `AuthManager` handles credential resolution and authentication header injection.

### Credential Resolution

Credentials are read from **session-scoped environment variables** via `get_session_env()`, not `os.environ`. This prevents concurrent sessions from overwriting each other's secrets. The function checks a `ContextVar` first, falling back to `os.environ`.

### Supported Auth Types

| Type | Mechanism | Required Config |
|------|-----------|----------------|
| `none` | No auth | — |
| `apiKey` | Key in header or query param | `key_location`, `key_name`, `value_env` |
| `bearer` | `Authorization: Bearer <token>` | `value_env` (env var containing token) |
| `basic` | `Authorization: Basic <base64>` | `username_env`, `password_env` |
| `oauth2_client` | Client credentials flow → `Authorization: <type> <token>` | `token_url`, `client_id_env`, `client_secret_env`, `scope` |

### OAuth2 Client Credentials Flow

1. Fetch token from `token_url` using `client_id` and `client_secret`
2. Cache token in memory with expiry timestamp (60s buffer before expiry)
3. Tokens are cached per service name for reuse
4. Token URL supports `${VAR}` expansion at request time

### Header Redaction

Sensitive headers (`authorization`, `x-api-key`, `api-key`) are redacted in logging and curl preview output. Values are truncated to first 4 and last 4 characters.

### `configure_service_auth` Tool

Sets auth config for a service and persists it to `_service.yaml`. The auth config references env vars by name — actual secrets are never written to files.

---

## 7. HTTP Client

`ServiceHttpClient` handles request building, execution, and response processing.

### Request Building

1. **URL resolution**: If `service_config` is provided, base URL is prepended to path. If `endpoint_schema` exists, its path is used with parameter substitution.
2. **Path parameters**: `{placeholder}` values in the path are substituted from the `query` dict.
3. **Query parameters**: Encoded and appended to URL.
4. **Headers**: Merged from service defaults, auth headers, and caller-provided headers.
5. **Body**: JSON-serialized if dict, passed as-is if string.

### Execution

Uses `httpx` (preferred) or `requests` (fallback). Respects proxy configuration via `get_httpx_kwargs()` / `get_requests_kwargs()` from `shared.http`. When `ssl_trusted` is set on the service config, SSL verification is skipped. When `proxy_bypass` is set, proxy kwargs are omitted.

### Response Processing

- JSON responses are parsed automatically; non-JSON returned as string
- Responses exceeding `truncate_at` are truncated with `full_length` preserved
- User-Agent defaults to `jaato-service-connector/1.0`

### Default Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_TIMEOUT` | 30000 (30s) | Default request timeout |
| `DEFAULT_TRUNCATE_AT` | 10000 (10KB) | Default response truncation |
| `DEFAULT_USER_AGENT` | `jaato-service-connector/1.0` | HTTP User-Agent header |

---

## 8. OpenAPI Parser

Parses OpenAPI 3.x and Swagger 2.x specifications.

### Spec Loading

- **From URL**: `fetch_spec_from_url_sync()` — synchronous HTTP fetch using httpx/requests
- **From file**: `load_spec_from_file()` — reads local JSON/YAML file
- **Parsing**: `parse_openapi_spec()` — extracts `DiscoveredService` from raw spec dict

### Reference Resolution (`$ref`)

The parser resolves all `$ref` references in the spec:

- Only internal references (`#/...`) are supported — external references are rejected
- Circular references are detected via a `_seen` set; circular refs are replaced with empty dicts
- When `warnings` list is provided (lenient mode), unresolvable refs produce warnings instead of errors, allowing partial results

### What Gets Extracted

- `ServiceConfig`: name, base URL, title, version, description, auth schemes
- `EndpointSchema` for each operation: method, path, parameters, request body, responses, tags
- `auth_schemes`: list of supported security scheme names from `components/securitySchemes`

### Error Handling

`OpenAPIParseError` is raised for:
- External references (not `#/...`)
- Unresolvable references (in strict mode, without warnings list)
- Missing required fields in the spec

---

## 9. Bruno Import

Imports API collections from [Bruno](https://www.usebruno.com/), an open-source API client that stores collections as `.bru` files.

### Bruno File Format

```bash
meta {
  name: Request Name
  type: http
}

get {
  url: {{baseUrl}}/path
}

headers {
  Content-Type: application/json
}

auth:bearer {
  token: {{accessToken}}
}

body:json {
  {"key": "value"}
}
```

### Import Process

1. Scan directory for `.bru` files
2. Parse each file's sections: `meta`, method blocks (`get`, `post`, etc.), `headers`, `query`, `auth:*`, `body:*`
3. Convert to `EndpointSchema` objects
4. Create a `ServiceConfig` from the collection name and base URL
5. Store as a regular service in `SchemaStore`

### Supported Bruno Features

- HTTP methods: GET, POST, PUT, DELETE, PATCH
- URL with `{{variable}}` placeholders
- Headers, query params
- Auth types: bearer, basic, apikey, none
- Body formats: json, form, text, graphql

### Error Handling

`BrunoParseError` is raised for malformed `.bru` files.

---

## 10. Validation

`SchemaValidator` validates request/response data against JSON schemas.

### Validation Scope

- **Type checking**: string, integer, number, boolean, array, object, null
- **Enum constraints**: value must be in allowed list
- **String constraints**: minLength, maxLength, pattern (regex)
- **Number constraints**: minimum, maximum, exclusiveMinimum, exclusiveMaximum
- **Array constraints**: minItems, maxItems
- **Object constraints**: required fields, additionalProperties
- **Nested schemas**: recursive validation of `properties` and `items`

### Validation Flow in `call_service`

1. If endpoint schema exists and has a request body schema → validate request body
2. Execute HTTP request
3. If endpoint schema exists and has a response schema for the status code → validate response
4. Both validation results included in `HttpResponse`

---

## 11. Plugin Lifecycle & Wiring

### Initialization

```python
plugin = ServiceConnectorPlugin()
plugin.initialize(config={"workspace_path": "/path/to/workspace"})
```

Creates `AuthManager`, `ServiceHttpClient`, `SchemaValidator`, and `SchemaStore`. The workspace path determines where `.jaato/services/` is rooted.

### Session Persistence

The plugin supports session persistence:

- `get_persistence_state()` → returns `{"discovered_services": ["github", "stripe"], "version": 1}`
- `restore_persistence_state(state)` → pre-warms in-memory cache from `SchemaStore`

This means discovered services survive session restart without re-fetching specs.

### Workspace Changes

`set_workspace_path(path)` updates the `SchemaStore` root. Called by plugin wiring when the workspace changes.

### Shutdown

`shutdown()` clears the in-memory service cache and marks plugin as uninitialized.

### Plugin Registration

The plugin is registered as a tool plugin:

```python
# In __init__.py
PLUGIN_KIND = "tool"

# Factory function
def create_plugin(config=None):
    plugin = ServiceConnectorPlugin()
    plugin.initialize(config=config)
    return plugin
```

---

## 12. Configuration Reference

### Workspace Configuration

No separate config file is required. Configuration is driven by:

1. **Environment variables** — referenced by `AuthConfig` (e.g., `GITHUB_TOKEN`, `STRIPE_API_KEY`)
2. **YAML schema files** — stored in `.jaato/services/`
3. **Proxy settings** — read from `shared.http` module (respects `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`)
4. **Plugin init config** — `{"workspace_path": "..."}` passed during initialization

### Service YAML Schema

```yaml
# .jaato/services/myapi/_service.yaml
name: myapi
base_url: https://api.example.com/v1
title: My API
version: "2.0"
description: Example API service
auth:
  type: bearer
  token_env: MYAPI_TOKEN
default_headers:
  Accept: application/json
timeout: 15000
ssl_trusted: false
proxy_bypass: false
```

### Discovered Service YAML Schema

```yaml
# .jaato/services/_discovered/github.yaml
config:
  name: github
  base_url: https://api.github.com
  title: GitHub API
  version: "3.0"
  auth:
    type: bearer
    token_env: GITHUB_TOKEN
source: https://api.github.com/openapi.json
endpoints:
  - method: GET
    path: /repos/{owner}/{repo}
    summary: Get a repository
    parameters:
      - name: owner
        in: path
        type: string
        required: true
      - name: repo
        in: path
        type: string
        required: true
    responses:
      200:
        description: OK
        schema: ...
```

### Endpoint YAML Schema

```yaml
# .jaato/services/myapi/list_users.yaml
method: GET
path: /users
summary: List all users
parameters:
  - name: limit
    in: query
    type: integer
    required: false
    default: 20
responses:
  200:
    description: List of users
    schema:
      type: array
      items:
        type: object
```

---

## 13. Source Code Map

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 18 | Plugin registration, exports `ServiceConnectorPlugin` and `create_plugin` |
| `types.py` | 539 | All data classes: `AuthConfig`, `ServiceConfig`, `EndpointSchema`, `Parameter`, `RequestBody`, `ResponseSpec`, `DiscoveredService`, `HttpResponse`, `PreviewedRequest`, `ValidationResult`, `ValidationError` |
| `plugin.py` | 2017 | Main plugin class with all tool implementations and schema definitions |
| `auth.py` | 368 | `AuthManager`: credential resolution, header building, OAuth2 token lifecycle |
| `http_client.py` | 516 | `ServiceHttpClient`: request building, HTTP execution, response processing, curl generation |
| `schema_store.py` | 443 | `SchemaStore`: filesystem-based YAML persistence for services and endpoints |
| `openapi_parser.py` | 927 | OpenAPI 3.x / Swagger 2.x spec parser with `$ref` resolution and cycle detection |
| `validation.py` | 344 | `SchemaValidator`: JSON schema validation for requests and responses |
| `bruno_import.py` | 487 | Bruno `.bru` file parser and collection importer |
| `tests/test_user_commands.py` | 19K | Tests for user-facing tool commands |
| `tests/test_persistence.py` | 5K | Tests for session persistence |
| `tests/test_types.py` | 10K | Tests for data type serialization |
| `tests/test_validation.py` | 10K | Tests for schema validation |
| `tests/test_openapi_parser.py` | 27K | Tests for OpenAPI spec parsing |

**Total**: ~7,175 lines across 10 source files (excluding tests).
