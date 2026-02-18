"""Filesystem-based schema storage for service connector.

Manages the storage and retrieval of service configurations and endpoint
schemas from the filesystem. Uses YAML format for human readability.

Directory structure:
    .jaato/services/
    ├── _discovered/              # Auto-cached OpenAPI specs
    │   └── {service}.yaml
    ├── {service}/                # Manually defined services
    │   ├── _service.yaml         # Service configuration
    │   └── {endpoint}.yaml       # Endpoint schemas
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    AuthConfig,
    EndpointSchema,
    ServiceConfig,
)


# Default storage directory relative to workspace
DEFAULT_SERVICES_DIR = ".jaato/services"
DISCOVERED_DIR = "_discovered"
SERVICE_CONFIG_FILE = "_service.yaml"


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

    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize the schema store.

        Args:
            workspace_path: Base directory for the workspace. If None,
                workspace-relative paths will not resolve until
                set_workspace_path() is called.
        """
        self._workspace: Optional[Path] = Path(workspace_path) if workspace_path else None
        self._base_path: Optional[Path] = self._workspace / DEFAULT_SERVICES_DIR if self._workspace else None

    @property
    def base_path(self) -> Optional[Path]:
        """Get the base path for service storage, or None if no workspace set."""
        return self._base_path

    def set_workspace_path(self, path: str) -> None:
        """Update the workspace path.

        Called by plugin wiring when workspace is set.

        Args:
            path: New workspace path.
        """
        self._workspace = Path(path)
        self._base_path = self._workspace / DEFAULT_SERVICES_DIR

    def _get_service_dir(self, service_name: str) -> Path:
        """Get the directory for a service."""
        return self._base_path / service_name

    def _get_discovered_dir(self) -> Path:
        """Get the directory for discovered (cached) services."""
        return self._base_path / DISCOVERED_DIR

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

        Args:
            service_name: Name of the service.

        Returns:
            ServiceConfig if found, None otherwise.
        """
        # Check regular services first
        config_path = self._get_service_dir(service_name) / SERVICE_CONFIG_FILE
        if config_path.exists():
            data = _load_yaml(config_path)
            if data:
                data["name"] = service_name  # Ensure name is set
                return ServiceConfig.from_dict(data)

        # Check discovered services
        discovered_path = self._get_discovered_dir() / f"{service_name}.yaml"
        if discovered_path.exists():
            data = _load_yaml(discovered_path)
            if data and data.get("config"):
                config_data = data["config"]
                config_data["name"] = service_name
                return ServiceConfig.from_dict(config_data)

        return None

    def list_services(self) -> List[str]:
        """List all available service names.

        Returns:
            List of service names (both manual and discovered).
        """
        services = []

        if not self._base_path.exists():
            return services

        for item in self._base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if item.name == DISCOVERED_DIR:
                    # List discovered services from YAML files
                    for yaml_file in item.glob("*.yaml"):
                        services.append(yaml_file.stem)
                else:
                    # Regular service directory
                    if (item / SERVICE_CONFIG_FILE).exists():
                        services.append(item.name)

        return sorted(set(services))

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
        """Load an endpoint schema.

        Args:
            service_name: Service name/directory.
            endpoint_name: Endpoint file name (without extension).

        Returns:
            EndpointSchema if found, None otherwise.
        """
        schema_path = self._get_service_dir(service_name) / f"{endpoint_name}.yaml"
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
        """List all endpoint schemas for a service.

        Args:
            service_name: Service name/directory.

        Returns:
            List of (endpoint_name, schema) tuples.
        """
        service_dir = self._get_service_dir(service_name)
        if not service_dir.exists():
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
        """Load a discovered service.

        Args:
            service_name: Service alias.

        Returns:
            Tuple of (config, endpoints) if found, None otherwise.
        """
        service_path = self._get_discovered_dir() / f"{service_name}.yaml"
        if not service_path.exists():
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
        """Get the original source URL/path for a discovered service.

        Args:
            service_name: Service alias.

        Returns:
            Source URL/path if available, None otherwise.
        """
        service_path = self._get_discovered_dir() / f"{service_name}.yaml"
        if not service_path.exists():
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
