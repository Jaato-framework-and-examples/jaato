"""Tests for WorkspaceProvisioner."""

import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Avoid importing server.__init__ which pulls heavy deps (google, etc.)
# by pre-populating sys.modules with a stub package and importing directly.
import types
if "server" not in sys.modules:
    _stub = types.ModuleType("server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["server"] = _stub

import server.workspace_provisioner as _mod
WorkspaceProvisioner = _mod.WorkspaceProvisioner
ProvisionedWorkspace = _mod.ProvisionedWorkspace


@pytest.fixture
def workspace_root(tmp_path):
    """Create a workspace root with template."""
    root = tmp_path / "workspaces"
    root.mkdir()
    # Create default template
    template_dir = root / "templates" / "default"
    template_dir.mkdir(parents=True)
    (template_dir / ".env").write_text("JAATO_PROVIDER=anthropic\nMODEL_NAME=claude-sonnet-4-20250514\n")
    jaato_dir = template_dir / ".jaato"
    jaato_dir.mkdir()
    (jaato_dir / "config.json").write_text('{"key": "value"}')
    return root


@pytest.fixture
def provisioner(workspace_root):
    return WorkspaceProvisioner(str(workspace_root))


class TestProvision:
    def test_creates_workspace_directory(self, provisioner, workspace_root):
        ws = provisioner.provision("session_001")
        assert Path(ws.path).is_dir()
        assert ws.session_id == "session_001"

    def test_copies_template(self, provisioner, workspace_root):
        ws = provisioner.provision("session_002")
        env_file = Path(ws.path) / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "JAATO_PROVIDER=anthropic" in content

    def test_creates_jaato_subdir(self, provisioner):
        ws = provisioner.provision("session_003")
        assert (Path(ws.path) / ".jaato").is_dir()

    def test_template_config_copied(self, provisioner):
        ws = provisioner.provision("session_004")
        config = Path(ws.path) / ".jaato" / "config.json"
        assert config.exists()
        assert json.loads(config.read_text()) == {"key": "value"}

    def test_duplicate_session_raises(self, provisioner):
        provisioner.provision("session_005")
        with pytest.raises(ValueError, match="already provisioned"):
            provisioner.provision("session_005")

    def test_records_template_name(self, provisioner):
        ws = provisioner.provision("session_006")
        assert ws.template == "default"

    def test_no_template_dir(self, tmp_path):
        """Provision works even without a template."""
        root = tmp_path / "empty_root"
        root.mkdir()
        p = WorkspaceProvisioner(str(root))
        ws = p.provision("session_007")
        assert Path(ws.path).is_dir()
        assert ws.template is None

    def test_records_client_id(self, provisioner):
        ws = provisioner.provision("session_008", client_id="client_1")
        assert ws.client_id == "client_1"

    def test_custom_template(self, workspace_root):
        # Create a custom template
        custom = workspace_root / "templates" / "research"
        custom.mkdir(parents=True)
        (custom / ".env").write_text("JAATO_PROVIDER=google\n")

        p = WorkspaceProvisioner(str(workspace_root))
        ws = p.provision("session_009", template="research")
        assert ws.template == "research"
        content = (Path(ws.path) / ".env").read_text()
        assert "google" in content


class TestTeardown:
    def test_removes_directory(self, provisioner):
        ws = provisioner.provision("session_td1")
        assert Path(ws.path).exists()
        provisioner.teardown("session_td1")
        assert not Path(ws.path).exists()

    def test_removes_from_tracking(self, provisioner):
        provisioner.provision("session_td2")
        provisioner.teardown("session_td2")
        assert provisioner.get_workspace("session_td2") is None

    def test_teardown_unknown_session_is_noop(self, provisioner):
        provisioner.teardown("nonexistent")  # should not raise


class TestLookup:
    def test_get_workspace(self, provisioner):
        provisioner.provision("session_lu1")
        ws = provisioner.get_workspace("session_lu1")
        assert ws is not None
        assert ws.session_id == "session_lu1"

    def test_get_unknown_returns_none(self, provisioner):
        assert provisioner.get_workspace("unknown") is None

    def test_list_workspaces(self, provisioner):
        provisioner.provision("s1")
        provisioner.provision("s2")
        workspaces = provisioner.list_workspaces()
        ids = {ws.session_id for ws in workspaces}
        assert ids == {"s1", "s2"}


class TestManifest:
    def test_persists_across_instances(self, workspace_root):
        p1 = WorkspaceProvisioner(str(workspace_root))
        p1.provision("persist_1")
        p1.provision("persist_2")

        # Create new instance — should load manifest
        p2 = WorkspaceProvisioner(str(workspace_root))
        assert p2.get_workspace("persist_1") is not None
        assert p2.get_workspace("persist_2") is not None

    def test_stale_manifest_entries_pruned(self, workspace_root):
        p1 = WorkspaceProvisioner(str(workspace_root))
        ws = p1.provision("prune_1")
        # Manually remove the directory (simulating external cleanup)
        import shutil
        shutil.rmtree(ws.path)
        # New instance should not load the stale entry
        p2 = WorkspaceProvisioner(str(workspace_root))
        assert p2.get_workspace("prune_1") is None


class TestReapExpired:
    def test_reaps_old_workspaces(self, provisioner):
        ws = provisioner.provision("old_session")
        # Manually backdate the last_activity
        ws.last_activity = "2020-01-01T00:00:00+00:00"
        reaped = provisioner.reap_expired(max_age_seconds=60)
        assert "old_session" in reaped
        assert not Path(ws.path).exists()

    def test_keeps_recent_workspaces(self, provisioner):
        provisioner.provision("recent_session")
        reaped = provisioner.reap_expired(max_age_seconds=86400)
        assert "recent_session" not in reaped


class TestUpdateActivity:
    def test_bumps_timestamp(self, provisioner):
        ws = provisioner.provision("active_1")
        old_ts = ws.last_activity
        time.sleep(0.01)
        provisioner.update_activity("active_1")
        ws2 = provisioner.get_workspace("active_1")
        assert ws2.last_activity > old_ts


class TestListTemplates:
    def test_lists_available_templates(self, provisioner, workspace_root):
        templates = provisioner.list_templates()
        assert "default" in templates

    def test_empty_when_no_templates(self, tmp_path):
        root = tmp_path / "empty"
        root.mkdir()
        p = WorkspaceProvisioner(str(root))
        assert p.list_templates() == []
