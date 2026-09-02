"""Filesystem-based schema storage for service connector.

Manages the storage and retrieval of service configurations and endpoint
schemas from the filesystem. Uses YAML format for human readability.

Tiered lookup
-------------

Reads traverse two tiers in precedence order — the same pattern used by
agents, profiles, prompts, skills, and themes elsewhere in jaato:

1. **Workspace tier** — ``<workspace>/.jaato/services/``.  Per-project
   services; the only writable tier.
2. **User tier** — ``~/.jaato/services/``.  Shared across all workspaces
   for the user.

The first tier containing a given service wins (workspace shadows home).
Writes always target the workspace tier — the user-tier is populated out
of band (e.g. by the user manually copying a discovered service they
want available everywhere).

Directory structure (per tier):
    .jaato/services/
    ├── _discovered/              # Auto-cached OpenAPI specs
    │   └── {service}.yaml
    ├── {service}/                # Manually defined services
    │   ├── _service.yaml         # Service configuration
    │   └── {endpoint}.yaml       # Endpoint schemas
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .types import (
    AuthConfig,
    EndpointSchema,
    ServiceConfig,
)


# Default storage directory relative to workspace OR user home
DEFAULT_SERVICES_DIR = ".jaato/services"
DISCOVERED_DIR = "_discovered"
SERVICE_CONFIG_FILE = "_service.yaml"


def _default_home_base_path() -> Optional[Path]:
    """Return the user-tier services path, or None when HOME is unset.

    ``~/.jaato/services/``.  The directory is not created on demand —
    only read from.  When it doesn't exist the tiered lookup simply
    skips it, so missing user-tier content is free of warnings.
    """
    try:
        return Path.home() / DEFAULT_SERVICES_DIR
    except (RuntimeError, OSError):
        # Path.home() can raise on exotic configurations with no HOME.
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file, returning empty dict if not found."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for schema storage. Install with: pip install pyyaml")

    if not path.exists():
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        return data if data else {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Save data to YAML file, creating directories as needed."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for schema storage. Install with: pip install pyyaml")

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


class SchemaStore:
    """Manages service and endpoint schemas on the filesystem.

    Attributes:
        base_path: Root directory for schema storage (.jaato/services).
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        home_base_path: Optional[Path] = None,
        config_root: Optional[str] = None,
    ):
        """Initialize the schema store.

        Args:
            workspace_path: Base directory for the workspace. If None,
                workspace-relative paths will not resolve until
                set_workspace_path() is called.
            home_base_path: Override for the user-tier base path.  When
                ``None`` (the default), uses ``~/.jaato/services/``.
                Passing an explicit path is mainly for tests that need
                to isolate the user tier; production callers rely on
                the default.
            config_root: Optional override for the workspace tier.  When
                set, the workspace base becomes ``<config_root>/services``
                instead of ``<workspace>/.jaato/services``.  See
                ``shared/config_resolver.py`` for the resolver chain.
        """
        self._workspace: Optional[Path] = Path(workspace_path) if workspace_path else None
        self._config_root: Optional[Path] = (
            Path(config_root).expanduser().resolve() if config_root else None
        )
        if self._config_root is not None:
            self._base_path: Optional[Path] = self._config_root / "services"
        elif self._workspace is not None:
            self._base_path = self._workspace / DEFAULT_SERVICES_DIR
        else:
            self._base_path = None
        # User-tier base path.  Unlike the workspace tier, this is
        # independent of the workspace setting and populated eagerly at
        # construction time (it comes from HOME, not from any caller-
        # supplied state).  Read-only from this class's perspective;
        # writes always go to self._base_path.
        self._home_base_path: Optional[Path] = (
            home_base_path if home_base_path is not None
            else _default_home_base_path()
        )

    @property
    def base_path(self) -> Optional[Path]:
        """Get the workspace base path for service storage, or None if no workspace set."""
        return self._base_path

    @property
    def home_base_path(self) -> Optional[Path]:
        """Get the user-tier base path (``~/.jaato/services/``), or None."""
        return self._home_base_path

    def set_workspace_path(self, path: str) -> None:
        """Update the workspace path.

        Called by plugin wiring when workspace is set.  Does not touch
        the user-tier base path, which stays pinned to ``~/.jaato/services/``
        regardless of workspace changes.  When a ``config_root`` is
        already in effect (set via :meth:`set_config_root`), the
        workspace tier stays anchored to it and this call only updates
        the underlying workspace reference for any future config_root
        clearance.

        Args:
            path: New workspace path.
        """
        self._workspace = Path(path)
        if self._config_root is None:
            self._base_path = self._workspace / DEFAULT_SERVICES_DIR

    def set_config_root(self, path: Optional[str]) -> None:
        """Adopt the registry-broadcast ``config_root`` override.

        When ``path`` is non-None, the workspace-tier base becomes
        ``<path>/services`` instead of ``<workspace>/.jaato/services``.
        When ``path`` is ``None``, falls back to the workspace tier
        (today's default behavior).
        """
        self._config_root = (
            Path(path).expanduser().resolve() if path else None
        )
        if self._config_root is not None:
            self._base_path = self._config_root / "services"
        elif self._workspace is not None:
            self._base_path = self._workspace / DEFAULT_SERVICES_DIR
        else:
            self._base_path = None

    # ------------------------------------------------------------------
    # Tier iteration
    # ------------------------------------------------------------------

    def _read_base_paths(self) -> Iterator[Path]:
        """Yield base paths to READ from, in precedence order.

        Workspace first (when configured), then user-tier (when it
        exists on disk).  Caller is responsible for stopping at the
        first hit when semantics demand "first tier wins" — this
        helper just iterates.

        Non-existent tiers are skipped — ``~/.jaato/services/`` is
        optional and most users won't have it.
        """
        if self._base_path is not None and self._base_path.exists():
            yield self._base_path
        if self._home_base_path is not None and self._home_base_path.exists():
            yield self._home_base_path

    def _find_service_base(self, service_name: str) -> Optional[Path]:
        """Return the first tier base containing ``service_name``.

        Checks for either a manual service (``<base>/<service>/_service.yaml``)
        or a discovered service (``<base>/_discovered/<service>.yaml``).
        Workspace wins on conflict.
        """
        for base in self._read_base_paths():
            if (base / service_name / SERVICE_CONFIG_FILE).exists():
                return base
            if (base / DISCOVERED_DIR / f"{service_name}.yaml").exists():
                return base
        return None

    # ------------------------------------------------------------------
    # Write-tier path helpers (workspace only)
    # ------------------------------------------------------------------

    def _get_service_dir(self, service_name: str) -> Path:
        """Get the WRITE directory for a service (always workspace-tier)."""
        return self._base_path / service_name

    def _get_discovered_dir(self) -> Path:
        """Get the WRITE directory for discovered services (always workspace-tier)."""
        return self._base_path / DISCOVERED_DIR

    # ------------------------------------------------------------------
    # Read-tier path helpers (tiered)
    # ------------------------------------------------------------------

    def _get_service_dir_for_read(self, service_name: str) -> Optional[Path]:
        """Get the READ directory for a service, respecting tier precedence.

        Returns ``None`` when the service doesn't exist in any tier.
        Callers that write should use :meth:`_get_service_dir` instead.
        """
        base = self._find_service_base(service_name)
        return base / service_name if base else None

    def _get_discovered_path_for_read(self, service_name: str) -> Optional[Path]:
        """Get the READ path for a discovered service, respecting tier precedence."""
        for base in self._read_base_paths():
            path = base / DISCOVERED_DIR / f"{service_name}.yaml"
            if path.exists():
                return path
        return None

    # === Service Operations ===

    def save_service_config(self, config: ServiceConfig) -> Path:
        """Save a service configuration.

        Args:
            config: Service configuration to save.

        Returns:
            Path to the saved file.
        """
        service_dir = self._get_service_dir(config.name)
        config_path = service_dir / SERVICE_CONFIG_FILE

        _save_yaml(config_path, config.to_dict())
        return config_path

    def load_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Load a service configuration.

        Tiered: checks workspace first, then user home.  First tier
        wins — a workspace-tier service with the same name as a
        user-tier one shadows the latter.

        Args:
            service_name: Name of the service.

        Returns:
            ServiceConfig if found, None otherwise.
        """
        for base in self._read_base_paths():
            # Check regular service at this tier
            config_path = base / service_name / SERVICE_CONFIG_FILE
            if config_path.exists():
                data = _load_yaml(config_path)
                if data:
                    data["name"] = service_name  # Ensure name is set
                    return ServiceConfig.from_dict(data)

            # Check discovered service at this tier
            discovered_path = base / DISCOVERED_DIR / f"{service_name}.yaml"
            if discovered_path.exists():
                data = _load_yaml(discovered_path)
                if data and data.get("config"):
                    config_data = data["config"]
                    config_data["name"] = service_name
                    return ServiceConfig.from_dict(config_data)

        return None

    def list_services(self) -> List[str]:
        """List all available service names across tiers.

        Returns:
            List of service names (both manual and discovered) from all
            readable tiers, deduplicated.  Workspace names shadow user-
            tier names with the same string — but since this returns
            names only (not configs), the deduplication is transparent.
        """
        services = set()

        for base in self._read_base_paths():
            for item in base.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    if item.name == DISCOVERED_DIR:
                        # List discovered services from YAML files
                        for yaml_file in item.glob("*.yaml"):
                            services.add(yaml_file.stem)
                    else:
                        # Regular service directory
                        if (item / SERVICE_CONFIG_FILE).exists():
                            services.add(item.name)

        return sorted(services)

    def delete_service(self, service_name: str) -> bool:
        """Delete a service and all its schemas.

        Args:
            service_name: Name of the service to delete.

        Returns:
            True if deleted, False if not found.
        """
        import shutil

        # Check regular service
        service_dir = self._get_service_dir(service_name)
        if service_dir.exists():
            shutil.rmtree(service_dir)
            return True

        # Check discovered service
        discovered_path = self._get_discovered_dir() / f"{service_name}.yaml"
        if discovered_path.exists():
            discovered_path.unlink()
            return True

        return False

    # === Endpoint Schema Operations ===

    def save_endpoint_schema(
        self,
        service_name: str,
        endpoint_name: str,
        schema: EndpointSchema
    ) -> Path:
        """Save an endpoint schema.

        Args:
            service_name: Service name/directory.
            endpoint_name: Name for the endpoint file (without extension).
            schema: Endpoint schema to save.

        Returns:
            Path to the saved file.
        """
        service_dir = self._get_service_dir(service_name)
        schema_path = service_dir / f"{endpoint_name}.yaml"

        _save_yaml(schema_path, schema.to_dict())
        return schema_path

    def load_endpoint_schema(
        self,
        service_name: str,
        endpoint_name: str
    ) -> Optional[EndpointSchema]:
        """Load an endpoint schema (tiered: workspace first, then user home).

        Args:
            service_name: Service name/directory.
            endpoint_name: Endpoint file name (without extension).

        Returns:
            EndpointSchema if found, None otherwise.
        """
        service_dir = self._get_service_dir_for_read(service_name)
        if service_dir is None:
            return None
        schema_path = service_dir / f"{endpoint_name}.yaml"
        if not schema_path.exists():
            return None

        data = _load_yaml(schema_path)
        if not data or "method" not in data or "path" not in data:
            return None

        return EndpointSchema.from_dict(data)

    def list_endpoint_schemas(
        self,
        service_name: str
    ) -> List[Tuple[str, EndpointSchema]]:
        """List all endpoint schemas for a service (tiered).

        Returns endpoints from whichever tier the service lives in —
        workspace takes precedence; if the service is workspace-
        defined, user-tier endpoints of the same service name are NOT
        merged in.  This matches the "first tier wins" semantics of
        ``load_service_config``.

        Args:
            service_name: Service name/directory.

        Returns:
            List of (endpoint_name, schema) tuples.
        """
        service_dir = self._get_service_dir_for_read(service_name)
        if service_dir is None or not service_dir.exists():
            return []

        endpoints = []
        for yaml_file in service_dir.glob("*.yaml"):
            # Skip service config
            if yaml_file.name == SERVICE_CONFIG_FILE:
                continue

            data = _load_yaml(yaml_file)
            if data and "method" in data and "path" in data:
                schema = EndpointSchema.from_dict(data)
                endpoints.append((yaml_file.stem, schema))

        return endpoints

    def delete_endpoint_schema(self, service_name: str, endpoint_name: str) -> bool:
        """Delete an endpoint schema.

        Args:
            service_name: Service name/directory.
            endpoint_name: Endpoint file name (without extension).

        Returns:
            True if deleted, False if not found.
        """
        schema_path = self._get_service_dir(service_name) / f"{endpoint_name}.yaml"
        if schema_path.exists():
            schema_path.unlink()
            return True
        return False

    # === Discovered Service Operations ===

    def save_discovered_service(
        self,
        service_name: str,
        config: ServiceConfig,
        endpoints: List[EndpointSchema],
        raw_spec: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> Path:
        """Save a discovered service (from OpenAPI spec).

        Args:
            service_name: Alias for the service.
            config: Service configuration.
            endpoints: List of endpoint schemas.
            raw_spec: Original OpenAPI spec (for caching).
            source: URL or path where spec was loaded from.

        Returns:
            Path to the saved file.
        """
        discovered_dir = self._get_discovered_dir()
        service_path = discovered_dir / f"{service_name}.yaml"

        data = {
            "config": config.to_dict(),
            "endpoints": [e.to_dict() for e in endpoints],
        }

        if source:
            data["source"] = source

        # Optionally store raw spec for reference
        if raw_spec:
            data["raw_spec"] = raw_spec

        _save_yaml(service_path, data)
        return service_path

    def load_discovered_service(
        self,
        service_name: str
    ) -> Optional[Tuple[ServiceConfig, List[EndpointSchema]]]:
        """Load a discovered service (tiered: workspace first, then user home).

        Args:
            service_name: Service alias.

        Returns:
            Tuple of (config, endpoints) if found, None otherwise.
        """
        service_path = self._get_discovered_path_for_read(service_name)
        if service_path is None:
            return None

        data = _load_yaml(service_path)
        if not data or "config" not in data:
            return None

        config_data = data["config"]
        config_data["name"] = service_name
        config = ServiceConfig.from_dict(config_data)

        endpoints = []
        for e_data in data.get("endpoints", []):
            endpoints.append(EndpointSchema.from_dict(e_data))

        return config, endpoints

    def get_discovered_source(self, service_name: str) -> Optional[str]:
        """Get the original source URL/path for a discovered service (tiered).

        Args:
            service_name: Service alias.

        Returns:
            Source URL/path if available, None otherwise.
        """
        service_path = self._get_discovered_path_for_read(service_name)
        if service_path is None:
            return None

        data = _load_yaml(service_path)
        return data.get("source")

    # === Query Operations ===

    def find_endpoint(
        self,
        service_name: str,
        method: str,
        path: str
    ) -> Optional[EndpointSchema]:
        """Find an endpoint schema by method and path.

        Args:
            service_name: Service name.
            method: HTTP method (GET, POST, etc.).
            path: URL path.

        Returns:
            EndpointSchema if found, None otherwise.
        """
        method = method.upper()

        # Check discovered service first
        discovered = self.load_discovered_service(service_name)
        if discovered:
            _, endpoints = discovered
            for endpoint in endpoints:
                if endpoint.method == method and endpoint.path == path:
                    return endpoint

        # Check manual endpoint schemas
        for name, schema in self.list_endpoint_schemas(service_name):
            if schema.method == method and schema.path == path:
                return schema

        return None

    def list_all_schemas(self) -> List[Dict[str, Any]]:
        """List all schemas across all services.

        Returns:
            List of dicts with service, name, method, path info.
        """
        all_schemas = []

        for service_name in self.list_services():
            # Manual schemas
            for endpoint_name, schema in self.list_endpoint_schemas(service_name):
                all_schemas.append({
                    "service": service_name,
                    "name": endpoint_name,
                    "method": schema.method,
                    "path": schema.path,
                    "summary": schema.summary,
                })

            # Discovered service endpoints
            discovered = self.load_discovered_service(service_name)
            if discovered:
                _, endpoints = discovered
                for i, endpoint in enumerate(endpoints):
                    all_schemas.append({
                        "service": service_name,
                        "name": f"endpoint_{i}",
                        "method": endpoint.method,
                        "path": endpoint.path,
                        "summary": endpoint.summary,
                        "discovered": True,
                    })

        return all_schemas
