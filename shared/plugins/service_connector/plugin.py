"""Service connector plugin for webservice discovery and consumption.

Provides tools for:
- Discovering APIs via OpenAPI/Swagger specifications
- Managing request/response schemas stored on the filesystem
- Making HTTP requests with validation
- Importing API collections from Bruno
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..base import CommandCompletion, CommandParameter, HelpLines, UserCommand
from ..model_provider.types import EditableContent, ToolSchema

from .auth import AuthError, AuthManager
from .bruno_import import BrunoParseError, parse_bruno_collection
from .http_client import HttpClientError, ServiceHttpClient
from .openapi_parser import (
    OpenAPIParseError,
    fetch_spec_from_url_sync,
    load_spec_from_file,
    parse_openapi_spec,
)
from .schema_store import SchemaStore
from .types import (
    AuthConfig,
    AuthType,
    DiscoveredService,
    EndpointSchema,
    ServiceConfig,
)
from .validation import SchemaValidator


class ServiceConnectorPlugin:
    """Plugin for discovering and consuming web services.

    Provides tools for:
    - discover_service: Load OpenAPI/Swagger specs
    - list_endpoints: Browse available endpoints
    - get_endpoint_schema: Get request/response schemas
    - call_service: Execute HTTP requests
    - preview_request: Dry-run showing what would be sent
    - save_schema: Persist schemas to filesystem
    - list_schemas: List stored schemas
    - import_bruno_collection: Import from Bruno
    - configure_service_auth: Set up authentication
    """

    def __init__(self):
        """Initialize the service connector plugin."""
        self._schema_store: Optional[SchemaStore] = None
        self._http_client: Optional[ServiceHttpClient] = None
        self._auth_manager: Optional[AuthManager] = None
        self._validator: Optional[SchemaValidator] = None

        # In-memory cache of discovered services
        self._discovered_services: Dict[str, DiscoveredService] = {}

        self._initialized = False
        self._workspace_path: Optional[str] = None

    @property
    def name(self) -> str:
        return "service_connector"

    def _trace(self, msg: str) -> None:
        """Write trace message to log file for debugging."""
        trace_path = os.environ.get(
            'JAATO_TRACE_LOG',
            os.path.join(tempfile.gettempdir(), "rich_client_trace.log")
        )
        if trace_path:
            try:
                with open(trace_path, "a") as f:
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"[{ts}] [SERVICE_CONNECTOR] {msg}\n")
                    f.flush()
            except (IOError, OSError):
                pass

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the service connector plugin.

        Args:
            config: Optional configuration dict with:
                - workspace_path: Base directory for schema storage
        """
        config = config or {}

        workspace = config.get("workspace_path")
        if workspace:
            self._workspace_path = workspace

        self._auth_manager = AuthManager()
        self._http_client = ServiceHttpClient(auth_manager=self._auth_manager)
        self._validator = SchemaValidator()
        self._schema_store = SchemaStore(workspace_path=self._workspace_path)

        self._initialized = True
        self._trace(f"initialize: workspace={self._workspace_path}")

    def shutdown(self) -> None:
        """Shutdown the service connector plugin."""
        self._trace("shutdown")
        self._discovered_services.clear()
        self._initialized = False

    def set_workspace_path(self, path: str) -> None:
        """Set the workspace path for schema storage.

        Called by plugin wiring when workspace is set.

        Args:
            path: Workspace directory path.
        """
        self._workspace_path = path
        if self._schema_store:
            self._schema_store.set_workspace_path(path)
        self._trace(f"set_workspace_path: {path}")

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Return tool schemas for service connector tools."""
        return [
            ToolSchema(
                name="discover_service",
                description=(
                    "Load and parse an OpenAPI/Swagger specification from a URL or file. "
                    "This discovers available endpoints, authentication requirements, and "
                    "request/response schemas. Use the returned alias to reference this "
                    "service in subsequent calls."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "URL or file path to the OpenAPI/Swagger spec"
                        },
                        "alias": {
                            "type": "string",
                            "description": "Short name to reference this service (e.g., 'github', 'stripe')"
                        }
                    },
                    "required": ["source", "alias"]
                },
                category="web",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="list_endpoints",
                description=(
                    "List available endpoints from a discovered service or schema directory. "
                    "Returns method, path, summary, and tags for each endpoint."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Service alias (from discover_service) or schema directory name"
                        },
                        "filter_method": {
                            "type": "string",
                            "description": "Filter by HTTP method (GET, POST, etc.)"
                        },
                        "filter_path": {
                            "type": "string",
                            "description": "Filter by path pattern (glob-style, e.g., '/users/*')"
                        },
                        "filter_tag": {
                            "type": "string",
                            "description": "Filter by OpenAPI tag"
                        }
                    },
                    "required": ["service"]
                },
                category="web",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="get_endpoint_schema",
                description=(
                    "Get the full request/response schema for a specific endpoint. "
                    "Returns parameters, request body schema, and response schemas."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Service alias or schema directory"
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP method (GET, POST, etc.)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Endpoint path (e.g., '/users/{id}')"
                        }
                    },
                    "required": ["service", "method", "path"]
                },
                category="web",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="call_service",
                description=(
                    "Execute an HTTP request. Can use a discovered service (with auth and "
                    "base URL) or a raw URL. Request body is validated against schema if "
                    "available. Response is validated and truncated if too large."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Service alias (uses stored base_url and auth)"
                        },
                        "url": {
                            "type": "string",
                            "description": "Full URL (alternative to service + path)"
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)"
                        },
                        "path": {
                            "type": "string",
                            "description": "Endpoint path (when using service)"
                        },
                        "query": {
                            "type": "object",
                            "description": "Query parameters"
                        },
                        "headers": {
                            "type": "object",
                            "description": "Additional headers"
                        },
                        "body": {
                            "type": ["object", "string"],
                            "description": "Request body (object or string)"
                        },
                        "auth": {
                            "type": "object",
                            "description": "Override auth config (type, token_env, etc.)"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Request timeout in milliseconds (default: 30000)"
                        },
                        "truncate_at": {
                            "type": "integer",
                            "description": "Response truncation limit in chars (default: 10000)"
                        }
                    },
                    "required": ["method"]
                },
                category="web",
                discoverability="discoverable",
                editable=EditableContent(
                    parameters=["method", "path", "url", "query", "headers", "body"],
                    format="json",
                    template="# Edit the request below. Save and exit to continue.\n",
                ),
            ),
            ToolSchema(
                name="preview_request",
                description=(
                    "Dry-run showing exactly what would be sent without executing. "
                    "Returns the full URL, headers, body, and equivalent curl command."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Service alias"
                        },
                        "url": {
                            "type": "string",
                            "description": "Full URL (alternative to service + path)"
                        },
                        "method": {
                            "type": "string",
                            "description": "HTTP method"
                        },
                        "path": {
                            "type": "string",
                            "description": "Endpoint path (when using service)"
                        },
                        "query": {
                            "type": "object",
                            "description": "Query parameters"
                        },
                        "headers": {
                            "type": "object",
                            "description": "Additional headers"
                        },
                        "body": {
                            "type": ["object", "string"],
                            "description": "Request body"
                        },
                        "auth": {
                            "type": "object",
                            "description": "Override auth config"
                        }
                    },
                    "required": ["method"]
                },
                category="web",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="save_schema",
                description=(
                    "Save an endpoint schema to the filesystem for reuse. "
                    "Useful for APIs without OpenAPI specs."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Service directory name"
                        },
                        "name": {
                            "type": "string",
                            "description": "Schema name (e.g., 'get-users')"
                        },
                        "schema": {
                            "type": "object",
                            "description": "Endpoint schema with method, path, parameters, etc.",
                            "properties": {
                                "method": {"type": "string"},
                                "path": {"type": "string"},
                                "base_url": {"type": "string"},
                                "summary": {"type": "string"},
                                "parameters": {"type": "array"},
                                "request_body": {"type": "object"},
                                "responses": {"type": "object"}
                            },
                            "required": ["method", "path"]
                        }
                    },
                    "required": ["service", "name", "schema"]
                },
                category="web",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="list_schemas",
                description=(
                    "List all stored schemas across services, or filter by service name."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Filter by service name (optional)"
                        }
                    },
                    "required": []
                },
                category="web",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="import_bruno_collection",
                description=(
                    "Import endpoints from a Bruno API collection directory. "
                    "Bruno is an open-source API client that stores collections as files."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to Bruno collection directory"
                        },
                        "service_name": {
                            "type": "string",
                            "description": "Name for the imported service"
                        },
                        "base_url": {
                            "type": "string",
                            "description": "Override base URL (optional)"
                        }
                    },
                    "required": ["path", "service_name"]
                },
                category="web",
                discoverability="discoverable",
            ),
            ToolSchema(
                name="configure_service_auth",
                description=(
                    "Configure authentication for a service. "
                    "Credentials are read from environment variables."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "Service name"
                        },
                        "auth": {
                            "type": "object",
                            "description": "Auth configuration",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["none", "apiKey", "bearer", "basic", "oauth2_client"]
                                },
                                "in": {"type": "string", "enum": ["header", "query"]},
                                "name": {"type": "string"},
                                "value_env": {"type": "string"},
                                "token_env": {"type": "string"},
                                "username_env": {"type": "string"},
                                "password_env": {"type": "string"},
                                "token_url": {"type": "string"},
                                "client_id_env": {"type": "string"},
                                "client_secret_env": {"type": "string"},
                                "scope": {"type": "string"}
                            },
                            "required": ["type"]
                        }
                    },
                    "required": ["service", "auth"]
                },
                category="web",
                discoverability="discoverable",
            ),
        ]

    def get_executors(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """Return the executor mapping."""
        return {
            "discover_service": self._execute_discover_service,
            "list_endpoints": self._execute_list_endpoints,
            "get_endpoint_schema": self._execute_get_endpoint_schema,
            "call_service": self._execute_call_service,
            "preview_request": self._execute_preview_request,
            "save_schema": self._execute_save_schema,
            "list_schemas": self._execute_list_schemas,
            "import_bruno_collection": self._execute_import_bruno_collection,
            "configure_service_auth": self._execute_configure_service_auth,
            # User command
            "services": lambda args: self.execute_user_command("services", args),
        }

    def get_system_instructions(self) -> Optional[str]:
        """Return system instructions for the service connector tools."""
        return """You have access to web service discovery and consumption tools.

**Workflow for discovered APIs (OpenAPI/Swagger):**
1. `discover_service(source="https://api.example.com/openapi.json", alias="example")`
2. `list_endpoints(service="example")` - browse available endpoints
3. `get_endpoint_schema(service="example", method="GET", path="/users")` - get details
4. `call_service(service="example", method="GET", path="/users")` - make request

**Workflow for raw URLs:**
1. `call_service(url="https://api.example.com/data", method="GET")`
2. Or with auth: `call_service(url="...", method="GET", auth={"type": "bearer", "token_env": "MY_TOKEN"})`

**Workflow for Bruno collections:**
1. `import_bruno_collection(path="./api-collection", service_name="myapi")`
2. Then use `list_endpoints`, `call_service` as above

**Authentication:**
- Credentials are NEVER stored in files - always read from environment variables
- Use `configure_service_auth` to set up auth for a service
- Check `env_vars_missing` in the response to know which vars to set

**Request validation:**
- Request bodies are validated against schema before sending
- Response bodies are validated after receiving (warnings only)
- Use `preview_request` to see exactly what would be sent

**Tips:**
- Use `preview_request` to debug auth issues (shows curl command)
- Responses over 10k chars are truncated; use `truncate_at` to adjust
- Path parameters can be in query: `call_service(..., query={"id": "123"})` for path `/{id}`"""

    def get_auto_approved_tools(self) -> List[str]:
        """Return tools that don't require permission.

        Note: call_service requires permission as it makes HTTP requests.
        The permission plugin handles domain-based approval.
        """
        return [
            # Read-only discovery tools
            "discover_service",
            "list_endpoints",
            "get_endpoint_schema",
            "list_schemas",
            # Dry-run preview
            "preview_request",
            # Schema management (local files only)
            "save_schema",
            "import_bruno_collection",
            "configure_service_auth",
            # User command (invoked by user directly)
            "services",
        ]

    def get_user_commands(self) -> List[UserCommand]:
        """Return user commands for service administration."""
        return [
            UserCommand(
                name='services',
                description='Manage discovered web services (list, show, endpoints, auth, remove)',
                share_with_model=False,
                parameters=[
                    CommandParameter(
                        name='subcommand',
                        description='Subcommand: list, show, endpoints, auth, remove, help',
                        required=True,
                    ),
                    CommandParameter(
                        name='rest',
                        description='Arguments for the subcommand',
                        required=False,
                        capture_rest=True,
                    ),
                ],
            ),
        ]

    def get_command_completions(
        self, command: str, args: List[str]
    ) -> List[CommandCompletion]:
        """Return completion options for services command arguments."""
        if command != 'services':
            return []

        subcommands = [
            CommandCompletion('list', 'List all known services'),
            CommandCompletion('show', 'Show details for a service'),
            CommandCompletion('endpoints', 'List endpoints for a service'),
            CommandCompletion('auth', 'Show auth status for a service'),
            CommandCompletion('remove', 'Remove a service'),
            CommandCompletion('help', 'Show help'),
        ]

        if not args:
            return subcommands

        if len(args) == 1:
            partial = args[0].lower()
            return [c for c in subcommands if c.value.startswith(partial)]

        subcommand = args[0].lower()

        # Level 2: service names for subcommands that take a service argument
        if len(args) == 2 and subcommand in ('show', 'endpoints', 'auth', 'remove'):
            partial = args[1].lower()
            return self._get_service_completions(partial)

        # Level 3: HTTP method filter for 'endpoints <service> <method>'
        if len(args) == 3 and subcommand == 'endpoints':
            partial = args[2].upper()
            service_name = args[1]
            methods = self._get_service_methods(service_name)
            return [m for m in methods if m.value.startswith(partial)]

        return []

    def _get_service_completions(self, partial: str) -> List[CommandCompletion]:
        """Get service name completions from memory and filesystem."""
        names = self._get_all_service_names()
        return [
            CommandCompletion(name, 'Service')
            for name in names
            if name.lower().startswith(partial)
        ]

    def _get_methods_for_service(self, service_name: str) -> List[str]:
        """Get sorted list of HTTP methods used in a service's endpoints."""
        methods: set = set()
        # _get_service checks memory first, then loads from disk
        discovered = self._get_service(service_name)
        if discovered:
            for ep in discovered.endpoints:
                methods.add(ep.method)
        # Also check per-endpoint schema files
        if self._schema_store:
            for _ep_name, schema in self._schema_store.list_endpoint_schemas(service_name):
                methods.add(schema.method)
        return sorted(methods)

    def _get_service_methods(self, service_name: str) -> List[CommandCompletion]:
        """Get HTTP method completions for a service's endpoints."""
        methods = self._get_methods_for_service(service_name)
        if methods:
            return [CommandCompletion(m, f'{m} requests') for m in methods]
        # Fallback to common methods if service has no endpoints yet
        return [
            CommandCompletion('GET', 'GET requests'),
            CommandCompletion('POST', 'POST requests'),
            CommandCompletion('PUT', 'PUT requests'),
            CommandCompletion('DELETE', 'DELETE requests'),
            CommandCompletion('PATCH', 'PATCH requests'),
        ]

    def _get_all_service_names(self) -> List[str]:
        """Get all known service names from memory cache and filesystem."""
        names = set(self._discovered_services.keys())
        if self._schema_store:
            names.update(self._schema_store.list_services())
        return sorted(names)

    def get_service_metadata(self) -> List[Dict[str, Any]]:
        """Return lightweight service metadata for completion caches.

        Returns:
            List of dicts with name and methods for each service.
        """
        return [
            {"name": name, "methods": self._get_methods_for_service(name)}
            for name in self._get_all_service_names()
        ]

    def execute_user_command(self, command: str, args: Dict[str, Any]) -> Any:
        """Execute a user command."""
        if command != 'services':
            return f"Unknown command: {command}"

        subcommand = args.get('subcommand', '').lower()
        rest = args.get('rest', '').strip()

        if subcommand == 'list' or subcommand == '':
            return self._cmd_list()
        elif subcommand == 'show':
            if not rest:
                return "Usage: services show <service>"
            return self._cmd_show(rest)
        elif subcommand == 'endpoints':
            parts = rest.split(None, 1)
            if not parts:
                return "Usage: services endpoints <service> [method]"
            service_name = parts[0]
            method_filter = parts[1].upper() if len(parts) > 1 else None
            return self._cmd_endpoints(service_name, method_filter)
        elif subcommand == 'auth':
            if not rest:
                return "Usage: services auth <service>"
            return self._cmd_auth(rest)
        elif subcommand == 'remove':
            if not rest:
                return "Usage: services remove <service>"
            return self._cmd_remove(rest)
        elif subcommand == 'help':
            return self._cmd_help()
        else:
            return (
                f"Unknown subcommand: {subcommand}\n"
                "Use 'services help' for available commands."
            )

    # === User Command Handlers ===

    def _cmd_list(self) -> Any:
        """List all known services. Returns HelpLines for pager display."""
        names = self._get_all_service_names()
        if not names:
            return HelpLines(lines=[
                ("No services discovered.", "dim"),
                ("", ""),
                ("Use the discover_service tool or 'services help' for guidance.", ""),
            ])

        lines: List[tuple] = [("Known Services", "bold"), ("", "")]
        for name in names:
            discovered = self._get_service(name)
            if discovered:
                auth_type = discovered.config.auth.type.value
                ep_count = discovered.endpoint_count
                base_url = discovered.config.base_url
                auth_label = f"auth={auth_type}" if auth_type != "none" else "no auth"
                lines.append((f"  {name}", "bold"))
                lines.append(
                    (f"    {base_url}  |  {ep_count} endpoints  |  {auth_label}", "dim")
                )
            else:
                lines.append((f"  {name}", "bold"))
                lines.append(("    (config only)", "dim"))

        lines.append(("", ""))
        lines.append((f"{len(names)} service(s) total", "bold"))
        return HelpLines(lines=lines)

    def _cmd_show(self, service_name: str) -> Any:
        """Show details of a specific service. Returns HelpLines for pager display."""
        discovered = self._get_service(service_name)

        # Fall back to manual config
        config = None
        if discovered:
            config = discovered.config
        elif self._schema_store:
            config = self._schema_store.load_service_config(service_name)

        if not config:
            return f"Service not found: {service_name}"

        lines: List[tuple] = [(f"Service: {config.name}", "bold"), ("", "")]

        if config.title:
            lines.append((f"  Title:       {config.title}", ""))
        if config.version:
            lines.append((f"  Version:     {config.version}", ""))
        lines.append((f"  Base URL:    {config.base_url or '(not set)'}", ""))

        if config.description:
            desc = config.description
            if len(desc) > 120:
                desc = desc[:117] + "..."
            lines.append((f"  Description: {desc}", "dim"))

        # Auth summary
        auth = config.auth
        lines.append((f"  Auth type:   {auth.type.value}", ""))

        # Endpoint count
        if discovered:
            lines.append((f"  Endpoints:   {discovered.endpoint_count}", ""))

        # Source
        if discovered and discovered.source:
            lines.append((f"  Source:      {discovered.source}", "dim"))

        # Schema store path
        if self._schema_store:
            source_on_disk = self._schema_store.get_discovered_source(service_name)
            if source_on_disk and (not discovered or source_on_disk != discovered.source):
                lines.append((f"  Disk source: {source_on_disk}", "dim"))

        return HelpLines(lines=lines)

    def _cmd_endpoints(
        self, service_name: str, method_filter: Optional[str] = None
    ) -> Any:
        """List endpoints for a service. Returns HelpLines for pager display."""
        discovered = self._get_service(service_name)

        endpoints: List[Dict[str, Any]] = []

        if discovered:
            for ep in discovered.endpoints:
                if method_filter and ep.method != method_filter:
                    continue
                endpoints.append({
                    "method": ep.method,
                    "path": ep.path,
                    "summary": ep.summary or "",
                })
        elif self._schema_store:
            for _name, schema in self._schema_store.list_endpoint_schemas(service_name):
                if method_filter and schema.method != method_filter:
                    continue
                endpoints.append({
                    "method": schema.method,
                    "path": schema.path,
                    "summary": schema.summary or "",
                })

        if not endpoints and not discovered:
            return f"Service not found: {service_name}"

        if not endpoints:
            filter_note = f" matching {method_filter}" if method_filter else ""
            return f"No endpoints{filter_note} for service: {service_name}"

        filter_note = f" (filtered: {method_filter})" if method_filter else ""
        lines: List[tuple] = [
            (f"Endpoints for {service_name}{filter_note}", "bold"),
            ("", ""),
        ]

        for ep in endpoints:
            method_str = f"  {ep['method']:7s} {ep['path']}"
            if ep['summary']:
                lines.append((method_str, ""))
                lines.append((f"           {ep['summary']}", "dim"))
            else:
                lines.append((method_str, ""))

        lines.append(("", ""))
        lines.append((f"{len(endpoints)} endpoint(s)", "bold"))
        return HelpLines(lines=lines)

    def _cmd_auth(self, service_name: str) -> Any:
        """Show auth configuration and credential status. Returns HelpLines for pager display."""
        discovered = self._get_service(service_name)

        config = None
        if discovered:
            config = discovered.config
        elif self._schema_store:
            config = self._schema_store.load_service_config(service_name)

        if not config:
            return f"Service not found: {service_name}"

        auth = config.auth
        lines: List[tuple] = [
            (f"Auth for {service_name}", "bold"),
            ("", ""),
            (f"  Type: {auth.type.value}", ""),
        ]

        if auth.type == AuthType.NONE:
            lines.append(("  No authentication configured.", "dim"))
            return HelpLines(lines=lines)

        if auth.type == AuthType.API_KEY:
            loc = auth.key_location.value if auth.key_location else "header"
            lines.append((f"  Location: {loc}", ""))
            lines.append((f"  Key name: {auth.key_name or '(default)'}", ""))

        if auth.type == AuthType.OAUTH2_CLIENT and auth.token_url:
            lines.append((f"  Token URL: {auth.token_url}", ""))
            if auth.scope:
                lines.append((f"  Scope: {auth.scope}", ""))

        # Credential check
        if self._auth_manager:
            cred = self._auth_manager.check_credentials(auth)
            required = cred.get("env_vars_required", [])
            present = set(cred.get("env_vars_present", []))
            missing = set(cred.get("env_vars_missing", []))

            if required:
                lines.append(("", ""))
                lines.append(("  Environment variables:", "bold"))
                for var in required:
                    if var in present:
                        lines.append((f"    {var} = (set)", ""))
                    else:
                        lines.append((f"    {var} = (MISSING)", "bold"))

                if missing:
                    lines.append(("", ""))
                    lines.append((
                        f"  Warning: {len(missing)} required variable(s) not set.",
                        "bold",
                    ))

        return HelpLines(lines=lines)

    def _cmd_remove(self, service_name: str) -> str:
        """Remove a service from memory and filesystem."""
        removed_memory = service_name in self._discovered_services
        if removed_memory:
            del self._discovered_services[service_name]

        removed_disk = False
        if self._schema_store:
            removed_disk = self._schema_store.delete_service(service_name)

        if removed_memory or removed_disk:
            parts = []
            if removed_memory:
                parts.append("memory")
            if removed_disk:
                parts.append("disk")
            return f"Removed service '{service_name}' from {' and '.join(parts)}."

        return f"Service not found: {service_name}"

    def _cmd_help(self) -> HelpLines:
        """Return formatted help for the services command."""
        return HelpLines(lines=[
            ("Services Command", "bold"),
            ("", ""),
            ("Manage discovered web services. View service details, check auth", ""),
            ("status, browse endpoints, and clean up services.", ""),
            ("", ""),
            ("USAGE", "bold"),
            ("    services [subcommand] [args]", ""),
            ("", ""),
            ("SUBCOMMANDS", "bold"),
            ("    list                          List all known services (default)", "dim"),
            ("    show <service>                Show details for a service", "dim"),
            ("    endpoints <service> [method]  List endpoints, optionally by HTTP method", "dim"),
            ("    auth <service>                Show auth type and env var status", "dim"),
            ("    remove <service>              Remove a service from memory and disk", "dim"),
            ("    help                          Show this help", "dim"),
            ("", ""),
            ("EXAMPLES", "bold"),
            ("    services                          List all services", "dim"),
            ("    services show github               Show github service details", "dim"),
            ("    services endpoints petstore GET    List GET endpoints for petstore", "dim"),
            ("    services auth stripe               Check auth config and env vars", "dim"),
            ("    services remove old-api            Remove a service", "dim"),
            ("", ""),
            ("NOTES", "bold"),
            ("    Services are discovered by the model via the discover_service tool", ""),
            ("    or imported from Bruno collections. This command lets you inspect", ""),
            ("    and manage what the model currently knows about.", ""),
            ("", ""),
            ("    Auth credentials are never stored in files - only environment", ""),
            ("    variable names are stored. Use 'services auth <name>' to check", ""),
            ("    which variables are set.", ""),
        ])

    # === Tool Executors ===

    def _execute_discover_service(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute discover_service tool."""
        source = args.get("source", "").strip()
        alias = args.get("alias", "").strip()

        self._trace(f"discover_service: source={source}, alias={alias}")

        if not source:
            return {"error": "source is required"}
        if not alias:
            return {"error": "alias is required"}

        try:
            # Load spec
            if source.startswith(("http://", "https://")):
                spec = fetch_spec_from_url_sync(source)
            else:
                spec = load_spec_from_file(source)

            # Parse spec
            discovered = parse_openapi_spec(spec, alias, source)

            # Cache in memory
            self._discovered_services[alias] = discovered

            # Save to filesystem for persistence
            if self._schema_store:
                self._schema_store.save_discovered_service(
                    service_name=alias,
                    config=discovered.config,
                    endpoints=discovered.endpoints,
                    source=source,
                )

            return {
                "alias": alias,
                "base_url": discovered.base_url,
                "title": discovered.config.title,
                "version": discovered.config.version,
                "endpoint_count": discovered.endpoint_count,
                "auth_schemes": discovered.auth_schemes,
            }

        except OpenAPIParseError as e:
            return {"error": f"Failed to parse spec: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}

    def _get_service(self, service_name: str) -> Optional[DiscoveredService]:
        """Get a discovered service by name."""
        # Check memory cache first
        if service_name in self._discovered_services:
            return self._discovered_services[service_name]

        # Try to load from storage
        if self._schema_store:
            result = self._schema_store.load_discovered_service(service_name)
            if result:
                config, endpoints = result
                source = self._schema_store.get_discovered_source(service_name)
                discovered = DiscoveredService(
                    config=config,
                    endpoints=endpoints,
                    source=source,
                )
                self._discovered_services[service_name] = discovered
                return discovered

        return None

    def _execute_list_endpoints(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_endpoints tool."""
        service_name = args.get("service", "").strip()
        filter_method = args.get("filter_method", "").upper()
        filter_path = args.get("filter_path", "")
        filter_tag = args.get("filter_tag", "")

        self._trace(f"list_endpoints: service={service_name}")

        if not service_name:
            return {"error": "service is required"}

        # Get discovered service
        discovered = self._get_service(service_name)
        if not discovered:
            # Check for manual schemas
            if self._schema_store:
                schemas = self._schema_store.list_endpoint_schemas(service_name)
                if schemas:
                    endpoints = []
                    for name, schema in schemas:
                        if filter_method and schema.method != filter_method:
                            continue
                        if filter_path and not self._match_path(schema.path, filter_path):
                            continue
                        if filter_tag and filter_tag not in schema.tags:
                            continue
                        endpoints.append({
                            "method": schema.method,
                            "path": schema.path,
                            "summary": schema.summary,
                            "tags": schema.tags,
                        })
                    return {"endpoints": endpoints}

            return {"error": f"Service not found: {service_name}"}

        # Filter endpoints
        endpoints = []
        for endpoint in discovered.endpoints:
            if filter_method and endpoint.method != filter_method:
                continue
            if filter_path and not self._match_path(endpoint.path, filter_path):
                continue
            if filter_tag and filter_tag not in endpoint.tags:
                continue

            endpoints.append({
                "method": endpoint.method,
                "path": endpoint.path,
                "summary": endpoint.summary,
                "tags": endpoint.tags,
            })

        return {"endpoints": endpoints}

    def _match_path(self, path: str, pattern: str) -> bool:
        """Match path against glob-style pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

    def _execute_get_endpoint_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_endpoint_schema tool."""
        service_name = args.get("service", "").strip()
        method = args.get("method", "").upper()
        path = args.get("path", "").strip()

        self._trace(f"get_endpoint_schema: service={service_name}, {method} {path}")

        if not service_name:
            return {"error": "service is required"}
        if not method:
            return {"error": "method is required"}
        if not path:
            return {"error": "path is required"}

        # Find endpoint
        endpoint = None

        discovered = self._get_service(service_name)
        if discovered:
            for ep in discovered.endpoints:
                if ep.method == method and ep.path == path:
                    endpoint = ep
                    break

        if not endpoint and self._schema_store:
            endpoint = self._schema_store.find_endpoint(service_name, method, path)

        if not endpoint:
            return {"error": f"Endpoint not found: {method} {path}"}

        return endpoint.to_dict()

    def _execute_call_service(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute call_service tool."""
        service_name = args.get("service", "").strip()
        url = args.get("url", "").strip()
        method = args.get("method", "").upper()
        path = args.get("path", "").strip()
        query = args.get("query")
        headers = args.get("headers")
        body = args.get("body")
        auth_override = args.get("auth")
        timeout = args.get("timeout")
        truncate_at = args.get("truncate_at")

        self._trace(f"call_service: service={service_name}, url={url}, {method} {path}")

        if not method:
            return {"error": "method is required"}
        if not url and not service_name:
            return {"error": "Either url or service is required"}

        # Get service config and endpoint schema
        service_config = None
        endpoint_schema = None

        if service_name:
            discovered = self._get_service(service_name)
            if discovered:
                service_config = discovered.config
                # Find endpoint schema for validation
                if path:
                    for ep in discovered.endpoints:
                        if ep.method == method and ep.path == path:
                            endpoint_schema = ep
                            break

            # Also check manual config
            if not service_config and self._schema_store:
                service_config = self._schema_store.load_service_config(service_name)
                if path:
                    endpoint_schema = self._schema_store.find_endpoint(
                        service_name, method, path
                    )

            if not service_config:
                return {"error": f"Service not found: {service_name}"}

        # Validate request body
        request_validation = None
        if endpoint_schema and body and self._validator:
            request_validation = self._validator.validate_request_body(
                body, endpoint_schema
            )

        # Create response validator
        response_validator = None
        if endpoint_schema and self._validator:
            response_validator = self._validator.create_response_validator(
                endpoint_schema
            )

        # Execute request
        try:
            response = self._http_client.execute(
                method=method,
                url=url or None,
                service_config=service_config,
                endpoint_schema=endpoint_schema,
                path=path or None,
                query=query,
                headers=headers,
                body=body,
                auth_override=auth_override,
                timeout=timeout,
                truncate_at=truncate_at,
                request_validation=request_validation,
                response_validator=response_validator,
            )
            return response.to_dict()

        except AuthError as e:
            return {"error": f"Authentication error: {e}"}
        except HttpClientError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Request failed: {e}"}

    def _execute_preview_request(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute preview_request tool."""
        service_name = args.get("service", "").strip()
        url = args.get("url", "").strip()
        method = args.get("method", "").upper()
        path = args.get("path", "").strip()
        query = args.get("query")
        headers = args.get("headers")
        body = args.get("body")
        auth_override = args.get("auth")

        self._trace(f"preview_request: service={service_name}, {method}")

        if not method:
            return {"error": "method is required"}
        if not url and not service_name:
            return {"error": "Either url or service is required"}

        # Get service config
        service_config = None
        endpoint_schema = None

        if service_name:
            discovered = self._get_service(service_name)
            if discovered:
                service_config = discovered.config
                if path:
                    for ep in discovered.endpoints:
                        if ep.method == method and ep.path == path:
                            endpoint_schema = ep
                            break

            if not service_config and self._schema_store:
                service_config = self._schema_store.load_service_config(service_name)

            if not service_config:
                return {"error": f"Service not found: {service_name}"}

        try:
            preview = self._http_client.build_request(
                method=method,
                url=url or None,
                service_config=service_config,
                endpoint_schema=endpoint_schema,
                path=path or None,
                query=query,
                headers=headers,
                body=body,
                auth_override=auth_override,
            )
            return preview.to_dict()

        except HttpClientError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to build request: {e}"}

    def _execute_save_schema(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute save_schema tool."""
        service_name = args.get("service", "").strip()
        schema_name = args.get("name", "").strip()
        schema_data = args.get("schema", {})

        self._trace(f"save_schema: service={service_name}, name={schema_name}")

        if not service_name:
            return {"error": "service is required"}
        if not schema_name:
            return {"error": "name is required"}
        if not schema_data:
            return {"error": "schema is required"}

        if "method" not in schema_data or "path" not in schema_data:
            return {"error": "schema must have method and path"}

        try:
            endpoint = EndpointSchema.from_dict(schema_data)
            path = self._schema_store.save_endpoint_schema(
                service_name, schema_name, endpoint
            )
            return {"path": str(path)}
        except Exception as e:
            return {"error": f"Failed to save schema: {e}"}

    def _execute_list_schemas(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute list_schemas tool."""
        service_filter = args.get("service", "").strip()

        self._trace(f"list_schemas: service={service_filter}")

        if not self._schema_store:
            return {"schemas": []}

        all_schemas = self._schema_store.list_all_schemas()

        if service_filter:
            all_schemas = [s for s in all_schemas if s["service"] == service_filter]

        return {"schemas": all_schemas}

    def _execute_import_bruno_collection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute import_bruno_collection tool."""
        collection_path = args.get("path", "").strip()
        service_name = args.get("service_name", "").strip()
        base_url = args.get("base_url", "").strip()

        self._trace(f"import_bruno_collection: path={collection_path}, service={service_name}")

        if not collection_path:
            return {"error": "path is required"}
        if not service_name:
            return {"error": "service_name is required"}

        try:
            config, endpoints, warnings = parse_bruno_collection(
                collection_path,
                service_name,
                base_url or None,
            )

            # Save to storage
            if self._schema_store:
                self._schema_store.save_discovered_service(
                    service_name=service_name,
                    config=config,
                    endpoints=endpoints,
                    source=collection_path,
                )

            # Cache in memory
            discovered = DiscoveredService(
                config=config,
                endpoints=endpoints,
                source=collection_path,
            )
            self._discovered_services[service_name] = discovered

            return {
                "service": service_name,
                "imported": len(endpoints),
                "endpoints": [
                    {"method": e.method, "path": e.path, "summary": e.summary}
                    for e in endpoints
                ],
                "warnings": warnings,
            }

        except BrunoParseError as e:
            return {"error": f"Failed to parse Bruno collection: {e}"}
        except Exception as e:
            return {"error": f"Import failed: {e}"}

    def _execute_configure_service_auth(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute configure_service_auth tool."""
        service_name = args.get("service", "").strip()
        auth_data = args.get("auth", {})

        self._trace(f"configure_service_auth: service={service_name}")

        if not service_name:
            return {"error": "service is required"}
        if not auth_data:
            return {"error": "auth is required"}

        try:
            auth_config = AuthConfig.from_dict(auth_data)

            # Load or create service config
            service_config = None
            if self._schema_store:
                service_config = self._schema_store.load_service_config(service_name)

            if not service_config:
                # Create minimal config
                service_config = ServiceConfig(
                    name=service_name,
                    base_url="",  # Will need to be set separately
                    auth=auth_config,
                )
            else:
                service_config.auth = auth_config

            # Save
            if self._schema_store:
                self._schema_store.save_service_config(service_config)

            # Check credentials
            cred_check = self._auth_manager.check_credentials(auth_config)

            return {
                "service": service_name,
                "auth_type": auth_config.type.value,
                **cred_check,
            }

        except Exception as e:
            return {"error": f"Failed to configure auth: {e}"}


def create_plugin() -> ServiceConnectorPlugin:
    """Factory function to create the service connector plugin instance."""
    return ServiceConnectorPlugin()
