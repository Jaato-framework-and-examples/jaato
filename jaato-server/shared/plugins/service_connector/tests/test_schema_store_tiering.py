"""Tests for SchemaStore's tiered lookup (workspace + user home).

Covers the regression where ``list_schemas`` was only finding services
in ``<workspace>/.jaato/services/`` and silently ignored user-tier
services in ``~/.jaato/services/`` — even though the rest of jaato
(agents, profiles, prompts, skills, themes) uses tiered lookup.
"""

from pathlib import Path

import pytest

from ..schema_store import SchemaStore


SERVICE_CONFIG_YAML = """\
base_url: https://{host}
auth:
  type: none
title: test
version: "1.0"
"""


ENDPOINT_YAML = """\
method: GET
path: /ping
summary: Ping endpoint
parameters: []
"""


DISCOVERED_YAML = """\
config:
  base_url: https://{host}
  auth:
    type: none
  title: discovered
  version: "1.0"
endpoints:
  - method: GET
    path: /discovered-endpoint
    summary: discovered
source: https://example.com/openapi.json
"""


def _make_manual_service(base: Path, name: str, host: str) -> None:
    """Create a manual service with one endpoint under ``base/<name>/``."""
    service_dir = base / name
    service_dir.mkdir(parents=True, exist_ok=True)
    (service_dir / "_service.yaml").write_text(
        SERVICE_CONFIG_YAML.format(host=host)
    )
    (service_dir / "ping.yaml").write_text(ENDPOINT_YAML)


def _make_discovered_service(base: Path, name: str, host: str) -> None:
    """Create a discovered service under ``base/_discovered/<name>.yaml``."""
    discovered_dir = base / "_discovered"
    discovered_dir.mkdir(parents=True, exist_ok=True)
    (discovered_dir / f"{name}.yaml").write_text(
        DISCOVERED_YAML.format(host=host)
    )


@pytest.fixture
def tmp_workspace_home(tmp_path):
    """Create workspace and home base directories for testing.

    Returns:
        Tuple of (workspace_base_path, home_base_path) — both pointing
        at ``.jaato/services/`` inside their respective roots.
    """
    ws_base = tmp_path / "workspace" / ".jaato" / "services"
    home_base = tmp_path / "home" / ".jaato" / "services"
    return ws_base, home_base


@pytest.fixture
def store_factory(tmp_workspace_home):
    """Return a factory that builds a SchemaStore rooted at the test tmpdir.

    Example::

        store = store_factory()  # workspace + home tier both pointing at tmp
    """
    ws_base, home_base = tmp_workspace_home

    def _make() -> SchemaStore:
        workspace_root = ws_base.parent.parent  # strip .jaato/services
        return SchemaStore(
            workspace_path=str(workspace_root),
            home_base_path=home_base,
        )

    return _make


class TestListServicesTiered:

    def test_empty_both_tiers(self, store_factory):
        store = store_factory()
        assert store.list_services() == []

    def test_workspace_only(self, tmp_workspace_home, store_factory):
        ws_base, _ = tmp_workspace_home
        _make_manual_service(ws_base, "svc-a", "a.example")
        store = store_factory()
        assert store.list_services() == ["svc-a"]

    def test_home_only_visible(self, tmp_workspace_home, store_factory):
        """Regression: user-tier services must be listed even when the
        workspace tier is empty.  This is the core bug the user
        reported: they had ``~/.jaato/services/github`` and
        ``~/.jaato/services/phoenix`` but ``list_schemas`` returned
        ``[]`` because workspace was empty."""
        _, home_base = tmp_workspace_home
        _make_manual_service(home_base, "github", "api.github.com")
        _make_discovered_service(home_base, "phoenix", "phoenix.local")
        store = store_factory()
        assert store.list_services() == ["github", "phoenix"]

    def test_merges_both_tiers_deduplicated(self, tmp_workspace_home, store_factory):
        ws_base, home_base = tmp_workspace_home
        _make_manual_service(ws_base, "workspace-svc", "ws.example")
        _make_manual_service(home_base, "home-svc", "home.example")
        _make_manual_service(ws_base, "overlap", "ws-overlap.example")
        _make_manual_service(home_base, "overlap", "home-overlap.example")
        store = store_factory()
        # Deduplicated; sorted
        assert store.list_services() == ["home-svc", "overlap", "workspace-svc"]


class TestLoadServiceConfigTiered:

    def test_workspace_wins_on_overlap(self, tmp_workspace_home, store_factory):
        ws_base, home_base = tmp_workspace_home
        _make_manual_service(ws_base, "x", "workspace.example")
        _make_manual_service(home_base, "x", "home.example")
        store = store_factory()
        cfg = store.load_service_config("x")
        assert cfg is not None
        assert cfg.base_url == "https://workspace.example"

    def test_home_tier_visible_when_workspace_empty(
        self, tmp_workspace_home, store_factory
    ):
        _, home_base = tmp_workspace_home
        _make_manual_service(home_base, "home-only", "home.example")
        store = store_factory()
        cfg = store.load_service_config("home-only")
        assert cfg is not None
        assert cfg.base_url == "https://home.example"

    def test_missing_returns_none(self, store_factory):
        assert store_factory().load_service_config("nope") is None

    def test_discovered_home_tier_visible(self, tmp_workspace_home, store_factory):
        _, home_base = tmp_workspace_home
        _make_discovered_service(home_base, "discovered-home", "d.example")
        store = store_factory()
        cfg = store.load_service_config("discovered-home")
        assert cfg is not None


class TestListAllSchemasTiered:

    def test_includes_home_tier_services(self, tmp_workspace_home, store_factory):
        """The original bug: ``list_all_schemas`` returned ``[]`` when
        the workspace tier was empty even though the home tier had
        services."""
        _, home_base = tmp_workspace_home
        _make_manual_service(home_base, "github", "api.github.com")
        _make_discovered_service(home_base, "phoenix", "phoenix.local")
        store = store_factory()
        all_schemas = store.list_all_schemas()
        service_names = {s["service"] for s in all_schemas}
        assert service_names == {"github", "phoenix"}

    def test_marks_discovered_entries(self, tmp_workspace_home, store_factory):
        _, home_base = tmp_workspace_home
        _make_discovered_service(home_base, "phoenix", "phoenix.local")
        store = store_factory()
        results = store.list_all_schemas()
        discovered = [s for s in results if s.get("discovered")]
        assert len(discovered) == 1
        assert discovered[0]["service"] == "phoenix"


class TestWritesAreWorkspaceOnly:
    """The user tier is read-only from the workspace's perspective.

    A ``save_*`` call should always target the workspace tier, even if
    a same-named service already exists in the user tier.  This matches
    the convention elsewhere in jaato: the workspace overrides, it
    does not mutate user-level state.
    """

    def test_save_service_config_goes_to_workspace(
        self, tmp_workspace_home, store_factory
    ):
        ws_base, home_base = tmp_workspace_home
        # Pre-existing home-tier service
        _make_manual_service(home_base, "shared-name", "old-home.example")

        store = store_factory()
        from ..types import ServiceConfig, AuthConfig, AuthType

        new_config = ServiceConfig(
            name="shared-name",
            base_url="https://new-workspace.example",
            auth=AuthConfig(type=AuthType.NONE),
        )
        saved_path = store.save_service_config(new_config)

        # Saved under workspace base, NOT home base
        assert ws_base in saved_path.parents
        assert home_base not in saved_path.parents

        # Home tier file still has the old value
        home_yaml = (home_base / "shared-name" / "_service.yaml").read_text()
        assert "old-home.example" in home_yaml

        # Subsequent load returns the workspace-tier copy (tier precedence)
        loaded = store.load_service_config("shared-name")
        assert loaded.base_url == "https://new-workspace.example"


class TestNoHomeTier:
    """When ``home_base_path=None`` (the construction path a plugin
    would take when ``Path.home()`` raises), only workspace tier is
    consulted — matches pre-tiering behavior exactly."""

    def test_list_services_only_workspace(self, tmp_path):
        workspace = tmp_path / "workspace"
        ws_base = workspace / ".jaato" / "services"
        _make_manual_service(ws_base, "svc", "svc.example")

        store = SchemaStore(
            workspace_path=str(workspace),
            home_base_path=None,  # explicit: no home tier
        )
        # Manually clear the auto-loaded home path too
        store._home_base_path = None
        assert store.list_services() == ["svc"]
