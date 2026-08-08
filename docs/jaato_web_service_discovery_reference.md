# Web Service Pre-Definition and Discovery

How to define, discover, and consume external web services in jaato using the
service connector plugin — with schemas, authentication, validation, and the
filesystem-based schema store.

## Overview

Jaato's **service connector plugin** lets agents interact with external REST
APIs through a structured workflow:

1. **Pre-define** services manually via YAML files in `.jaato/services/`
2. **Discover** services automatically from OpenAPI/Swagger specs
3. **Import** API collections from Bruno
4. **Call** services with automatic validation, auth, and retries

All service configurations and endpoint schemas are stored as YAML on the
filesystem — no database, no runtime-only state.  This makes services
portable, versionable, and available across sessions.

## The Three Ways to Register a Service

### 1. Manual Pre-Definition (Recommended for Mock Services)

Create a directory under `.jaato/services/<service>/` with a
`_service.yaml` config and one YAML file per endpoint:

```
.jaato/services/
└── sinco/
    ├── _service.yaml          # Service configuration
    ├── query-sinco.yaml       # GET /v1/siniestros/{dni}
    └── register-claim.yaml    # POST /v1/siniestros
```

**`_service.yaml`** — service-level configuration:

```yaml
name: sinco
base_url: http://localhost:8080
title: SINCO / SIHSA — Insurance Claims Database
version: "2.0"
description: Spain's central insurance claims database (TIREA/UNESPA)
auth:
  type: bearer
  token_env: SINCO_API_TOKEN
timeout: 5000
default_headers:
  Accept: application/json
  X-Client: jaato-e2e-test
```

**`query-sinco.yaml`** — endpoint schema:

```yaml
method: GET
path: /v1/siniestros/{dni}
summary: Query claims history by DNI
description: Returns 5-year claims history with bonus-malus coefficient.
parameters:
  - name: dni
    in: path
    type: string
    required: true
    description: Policyholder DNI
  - name: matricula
    in: query
    type: string
    required: false
    description: Vehicle registration number
responses:
  200:
    description: Claims history
    schema:
      type: object
      properties:
        query_id:
          type: string
        dni:
          type: string
        ant_siniestros_5y:
          type: integer
        ant_bonus_malus:
          type: integer
        siniestros:
          type: array
          items:
            type: object
            properties:
              id:
                type: string
              fecha:
                type: string
              importe:
                type: number
              responsable:
                type: boolean
```

### 2. OpenAPI/Swagger Discovery

Point `discover_service` at an OpenAPI spec URL or local file:

```
discover_service(
  source="https://api.example.com/openapi.json",
  alias="example_api"
)
```

This parses the spec, extracts all endpoints, auth schemes, and schemas,
and caches the result in `.jaato/services/_discovered/<alias>.yaml`.

The service is then available via `call_service(service="example_api", ...)`.

### 3. Bruno Collection Import

Import endpoints from a Bruno API collection:

```
import_bruno_collection(
  path="./bruno-collections/sinco/",
  service_name="sinco",
  base_url="http://localhost:8080"
)
```

## Schema Store: Tiered Filesystem Lookup

Schemas are stored in YAML under `.jaato/services/` with two tiers:

| Tier | Path | Read/Write |
|------|------|-------------|
| **Workspace** | `<workspace>/.jaato/services/` | Read + Write |
| **User** | `~/.jaato/services/` | Read only |

Lookup precedence: **workspace first, then user home**.  A workspace-tier
service with the same name shadows a user-tier one.

Within each tier, two sub-directories coexist:

```
.jaato/services/
├── _discovered/              # Auto-cached from OpenAPI specs
│   ├── github.yaml
│   └── stripe.yaml
├── sinco/                    # Manually defined
│   ├── _service.yaml
│   └── query-sinco.yaml
└── dgt/
    ├── _service.yaml
    └── verify-vehicle.yaml
```

**Writes always go to the workspace tier.**  The user tier is populated
out of band (e.g., manually copying a discovered service for cross-project
reuse).

## Service Configuration Reference

The `_service.yaml` file supports these fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `string` | Yes | — | Service alias used in `call_service(service=...)` |
| `base_url` | `string` | Yes | — | Base URL for all endpoints |
| `title` | `string` | No | — | Human-readable title |
| `version` | `string` | No | — | API version |
| `description` | `string` | No | — | Service description |
| `auth` | `object` | No | (none) | Authentication configuration |
| `default_headers` | `object` | No | `{}` | Headers included in every request |
| `timeout` | `integer` | No | `30000` | Default timeout in milliseconds |
| `ssl_trusted` | `boolean` | No | `false` | Skip SSL verification |
| `proxy_bypass` | `boolean` | No | `false` | Bypass HTTP proxy |

## Authentication

### Supported Auth Types

| Type | Config Fields | When to Use |
|------|--------------|-------------|
| `none` | *(empty)* | Public APIs |
| `bearer` | `token_env` | OAuth2 bearer tokens |
| `apiKey` | `in`, `name`, `value_env` | API keys in header or query |
| `basic` | `username_env`, `password_env` | HTTP Basic Auth |
| `oauth2_client` | `token_url`, `client_id_env`, `client_id_env`, `scope` | OAuth2 client credentials |

All sensitive values reference **environment variables** — never stored
directly.  Use `${VAR}` expansion in YAML values.

```yaml
# Bearer token
auth:
  type: bearer
  token_env: SINCO_API_TOKEN

# API key in header
auth:
  type: apiKey
  in: header
  name: X-API-Key
  value_env: MY_API_KEY

# Basic auth
auth:
  type: basic
  username_env: SERVICE_USER
  password_env: SERVICE_PASS
```

Auth can also be configured at runtime via `configure_service_auth`:

```
configure_service_auth(
  service="sinco",
  auth={"type": "bearer", "token_env": "SINCO_API_TOKEN"}
)
```

## Endpoint Schema Reference

Each endpoint YAML file defines:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | `string` | Yes | HTTP method (GET, POST, etc.) |
| `path` | `string` | Yes | URL path with `{placeholders}` |
| `summary` | `string` | No | Brief description |
| `description` | `string` | No | Detailed description |
| `parameters` | `array` | No | Request parameters |
| `request_body` | `object` | No | Request body spec |
| `responses` | `object` | No | Response specs by status code |
| `tags` | `array` | No | OpenAPI tags for grouping |
| `base_url` | `string` | No | Override base URL for this endpoint |

### Parameter Location Values

| Value | Origin | Notes |
|-------|--------|-------|
| `path` | OAS 2.0 + 3.0 | URL path segment |
| `query` | OAS 2.0 + 3.0 | Query string |
| `header` | OAS 2.0 + 3.0 | HTTP header |
| `body` | OAS 2.0 only | Request body (flat) |
| `formData` | OAS 2.0 only | Form-encoded body |
| `cookie` | OAS 3.0 only | Cookie header |

## Using Services from Profiles

Subagent profiles can declare the `service_connector` plugin to give agents
access to predefined services:

```json
{
  "name": "pricing",
  "plugins": ["service_connector"],
  "provider": "zhipuai",
  "model": "glm-5-turbo"
}
```

The agent can then:

1. List endpoints: `list_endpoints(service="sinco")`
2. Get a schema: `get_endpoint_schema(service="sinco", method="GET", path="/v1/siniestros/{dni}")`
3. Call the service: `call_service(service="sinco", method="GET", path="/v1/siniestros/12345678Z")`

### Request Validation

When a schema exists for the endpoint being called, the request body is
validated against it before sending.  On validation failure, the error is
returned without making the HTTP call.

### Response Validation

When a response schema exists, the response body is validated after
receiving.  Warnings are returned (the response is still returned).

## Using Services from the Python SDK

The `tools` module in notebooks exposes service connector tools:

```python
from tools import call_service, list_endpoints, get_endpoint_schema

# List endpoints
endpoints = list_endpoints(service="sinco")

# Get full schema
schema = get_endpoint_schema(service="sinco", method="GET", path="/v1/siniestros/{dni}")

# Call the service
result = call_service(
    service="sinco",
    method="GET",
    path="/v1/siniestros/12345678Z",
    query={"matricula": "2691-BJX"}
)
```

## Best Practices for Mock Services

### For E2E Tests

Define mock services as YAML files in `.jaato/services/` alongside your test
fixtures.  This gives you:

- **Type safety**: Request/response schemas validated at call time
- **Discoverability**: Agents can `list_endpoints` to see what's available
- **Portability**: Services move with the workspace

### Base URL Override for Mocks

Point `base_url` at a local mock server or use `save_to` for file-based
mocks.  Each endpoint's `base_url` can override the service-level URL,
enabling mixed real/mock configurations:

```yaml
# _service.yaml
name: sinco
base_url: http://localhost:8080  # production

# register-claim.yaml — override to point at mock
method: POST
path: /v1/siniestros
base_url: http://localhost:9999  # mock server
```

### `${VAR}` Expansion

All string values in YAML support `${VAR}` expansion and secret URI
resolution:

```yaml
auth:
  type: bearer
  token_env: SINCO_API_TOKEN   # reads os.environ["SINCO_API_TOKEN"]
```

## Tools Reference

| Tool | Description |
|------|-------------|
| `discover_service` | Load OpenAPI/Swagger spec from URL or file |
| `list_endpoints` | Browse available endpoints for a service |
| `get_endpoint_schema` | Get full request/response schema |
| `call_service` | Execute HTTP request with validation |
| `preview_request` | Dry-run without executing |
| `save_schema` | Save an endpoint schema for reuse |
| `list_schemas` | List all stored schemas |
| `import_bruno_collection` | Import from Bruno API collection |
| `configure_service_auth` | Set up authentication |

## Mock Servers

For e2e tests and development, jaato services can be backed by **local mock
HTTP servers** instead of real external APIs.  This section documents the
recommended pattern using Python's built-in `http.server` (stdlib, zero
dependencies).

### Architecture

```
.jaato/services/sinco/_service.yaml   <- schema definition (service_connector reads this)
         |
call_service(service="sinco", ...)    <- validated by schema, dispatched to base_url
         |
http://localhost:8081                  <- MockRESTServer (threaded, returns fixtures)
```

The service connector validates requests/responses against the YAML schemas
in `.jaato/services/`.  The mock server just needs to return valid JSON on
the right routes -- the schema store handles the contract.

### MockRESTServer

A reusable `http.server.HTTPServer` subclass that serves deterministic JSON
responses from a route table.  No external dependencies.

```python
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class MockRESTHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        handler = self.server.routes.get((method, path))
        if handler is None:
            self._respond(404, {"error": "not_found", "path": path})
            return

        try:
            status, body = handler(method, path, query, self._read_body())
        except Exception as e:
            self._respond(500, {"error": str(e)})
            return

        self._respond(status, body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return None

    def _respond(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class MockRESTServer:
    """Threaded mock HTTP server with a declarative route table."""

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self._server = HTTPServer((host, port), MockRESTHandler)
        self._server.routes = {}
        self._thread = None

    @property
    def port(self):
        return self._server.server_address[1]

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def route(self, method, path):
        """Decorator to register a handler for (method, path)."""
        def decorator(fn):
            self._server.routes[(method, path.rstrip("/"))] = fn
            return fn
        return decorator

    def start(self):
        """Start the server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def stop(self):
        self._server.shutdown()
```

### Registering Mock Routes

Define handlers that match the endpoint paths from the service schemas:

```python
server = MockRESTServer(port=8081)

@server.route("GET", "/v1/siniestros/{dni}")
def query_sinco(method, path, query, body):
    dni = path.split("/")[-1]
    return 200, {
        "query_id": f"SINCO-Q-{dni[:3]}",
        "dni": dni,
        "matricula": query.get("matricula", [""])[0],
        "ant_siniestros_5y": 1,
        "ant_bonus_malus": 5,
        "siniestros": [
            {
                "id": "SIN-2024-001",
                "fecha": "2024-03-15",
                "importe": 3200.0,
                "responsable": True,
            }
        ],
    }

server.start()
# server.base_url -> "http://127.0.0.1:8081"
```

### Wiring Schemas to Mock Servers

The `_service.yaml` `base_url` points at the mock server.  The schema
store and `call_service` do not care whether the backend is real or mock --
they validate against the same YAML schemas either way.

```yaml
# .jaato/services/sinco/_service.yaml
name: sinco
base_url: http://127.0.0.1:8081   # mock server
```

### Multi-Service Mock Setup

Run multiple mock servers on different ports, one per external service:

```python
sinco_server = MockRESTServer(port=8081)
dgt_server = MockRESTServer(port=8082)
kyc_server = MockRESTServer(port=8083)
aml_server = MockRESTServer(port=8084)

for s in [sinco_server, dgt_server, kyc_server, aml_server]:
    s.start()
```

```yaml
# .jaato/services/sinco/_service.yaml
base_url: http://127.0.0.1:8081

# .jaato/services/dgt/_service.yaml
base_url: http://127.0.0.1:8082

# .jaato/services/kyc/_service.yaml
base_url: http://127.0.0.1:8083

# .jaato/services/aml/_service.yaml
base_url: http://127.0.0.1:8084
```

### Port 0 for Dynamic Allocation

Use `port=0` to let the OS assign a free port.  Read it from
`server.port` and inject into environment variables:

```python
server = MockRESTServer(port=0)
server.start()

import os
os.environ["SINCO_BASE_URL"] = server.base_url
```

Then in `_service.yaml`:

```yaml
base_url: ${SINCO_BASE_URL}
```

### Fixture Integration

Combine mock servers with pytest fixtures for lifecycle management:

```python
import pytest

@pytest.fixture(scope="session")
def mock_servers():
    servers = {}
    for name, port in {"sinco": 8081, "dgt": 8082, "kyc": 8083, "aml": 8084}.items():
        s = MockRESTServer(port=port)
        s.start()
        servers[name] = s
    yield servers
    for s in servers.values():
        s.stop()
```


## Mock Servers

For e2e tests and development, jaato services can be backed by **local mock
HTTP servers** instead of real external APIs.  This section documents the
recommended pattern using Python's built-in `http.server` (stdlib, zero
dependencies).

### Architecture

```
.jaato/services/sinco/_service.yaml   ← schema definition (service_connector reads this)
         ↓
call_service(service="sinco", ...)    ← validated by schema, dispatched to base_url
         ↓
http://localhost:8081                  ← MockRESTServer (threaded, returns fixtures)
```

The service connector validates requests/responses against the YAML schemas
in `.jaato/services/`.  The mock server just needs to return valid JSON on
the right routes — the schema store handles the contract.

### MockRESTServer

A reusable `http.server.HTTPServer` subclass that serves deterministic JSON
responses from a route table.  No external dependencies.

```python
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class MockRESTHandler(BaseHTTPRequestHandler):
    """Routes requests to registered handlers.  Returns 404 for unknown routes."""

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        handler = self.server.routes.get((method, path))
        if handler is None:
            self._respond(404, {"error": "not_found", "path": path})
            return

        try:
            status, body = handler(method, path, query, self._read_body())
        except Exception as e:
            self._respond(500, {"error": str(e)})
            return

        self._respond(status, body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return None

    def _respond(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # suppress default stderr logging


class MockRESTServer:
    """Threaded mock HTTP server with a declarative route table."""

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self._server = HTTPServer((host, port), MockRESTHandler)
        self._server.routes = {}
        self._thread = None

    @property
    def port(self):
        return self._server.server_address[1]

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def route(self, method, path):
        def decorator(fn):
            self._server.routes[(method, path.rstrip("/"))] = fn
            return fn
        return decorator

    def start(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.shutdown()
```

### Registering Mock Routes

Define handlers that match the endpoint paths from the service schemas:

```python
server = MockRESTServer(port=8081)

@server.route("GET", "/v1/siniestros/{dni}")
def query_sinco(method, path, query, body):
    dni = path.split("/")[-1]
    return 200, {
        "query_id": f"SINCO-Q-{dni[:3]}",
        "dni": dni,
        "matricula": query.get("matricula", [""])[0],
        "ant_siniestros_5y": 1,
        "ant_bonus_malus": 5,
        "siniestros": [
            {
                "id": "SIN-2024-001",
                "fecha": "2024-03-15",
                "importe": 3200.0,
                "responsable": True,
            }
        ],
    }

server.start()
# server.base_url -> "http://127.0.0.1:8081"
```

### Wiring Schemas to Mock Servers

The `_service.yaml` `base_url` points at the mock server.  The schema
store and `call_service` don't care whether the backend is real or mock
— they validate against the same YAML schemas either way.

```yaml
# .jaato/services/sinco/_service.yaml
name: sinco
base_url: http://127.0.0.1:8081   # mock server
```

For tests that need deterministic responses per test case, register different
route handlers or swap the `base_url` to different mock server ports.

### Multi-Service Mock Setup

Run multiple mock servers on different ports, one per external service:

```python
sinco_server = MockRESTServer(port=8081)
dgt_server = MockRESTServer(port=8082)
kyc_server = MockRESTServer(port=8083)
aml_server = MockRESTServer(port=8084)

# ... register routes for each ...

for s in [sinco_server, dgt_server, kyc_server, aml_server]:
    s.start()
```

```yaml
# .jaato/services/sinco/_service.yaml
base_url: http://127.0.0.1:8081

# .jaato/services/dgt/_service.yaml
base_url: http://127.0.0.1:8082

# .jaato/services/kyc/_service.yaml
base_url: http://127.0.0.1:8083

# .jaato/services/aml/_service.yaml
base_url: http://127.0.0.1:8084
```

### Port 0 for Dynamic Allocation

Use `port=0` to let the OS assign a free port.  Read it from
`server.port` and inject into the schema store or environment:

```python
server = MockRESTServer(port=0)
server.start()

import os
os.environ["SINCO_BASE_URL"] = server.base_url
```

Then in `_service.yaml`:

```yaml
base_url: ${SINCO_BASE_URL}
```

### Fixture Integration

Combine mock servers with pytest fixtures for lifecycle management:

```python
import pytest

@pytest.fixture(scope="session")
def mock_servers():
    servers = {}
    services = {
        "sinco": 8081,
        "dgt": 8082,
        "kyc": 8083,
        "aml": 8084,
    }
    for name, port in services.items():
        s = MockRESTServer(port=port)
        s.start()
        servers[name] = s
    yield servers
    for s in servers.values():
        s.stop()
```
