"""The workspace picker offers every provider, named as the runtime names it.

``workspace_manager`` used to carry a six-entry table of providers and
their env vars, written when those were the providers.  It fell behind
in both directions: the picker offered six of the twenty-odd providers in
the tree, and two of the six under names the runtime does not know
(``google`` / ``github`` for the ``google_genai`` / ``github_models``
modules), so a ``.env`` the form wrote could fail at ``load_provider``.

The list is now derived from each provider's own ``PROVIDER_AUTH_RESOLUTION``
contract -- the same declaration ``jaato-scaffold explain provider``
renders -- so these tests pin the DERIVATION, not a list: a provider added
to the tree appears here with no edit, and the names are the directory
names the runtime and every profile use.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jaato_server.server import workspace_manager as wm
from jaato_server.server.workspace_manager import WorkspaceManager


def _provider_dirs() -> set:
    root = Path(wm.__file__).resolve().parents[1] / "shared" / "plugins" / "model_provider"
    return {
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "__init__.py").exists()
        and not p.name.startswith("_") and p.name not in {"tests", "echo", "bundle_common"}
    }


@pytest.fixture
def manager(tmp_path: Path) -> WorkspaceManager:
    root = tmp_path / "ws"
    (root / "proj").mkdir(parents=True)
    (root / "proj" / ".env").write_text("")
    m = WorkspaceManager(str(root), registry_path=tmp_path / "registry.json")
    m.discover_workspaces()
    return m


def test_every_provider_in_the_tree_is_offered_under_its_runtime_name() -> None:
    offered = set(wm.available_providers())
    assert offered == _provider_dirs()
    # The two names the old table got wrong are gone, and the modules are in.
    assert "google" not in offered and "github" not in offered
    assert {"google_genai", "github_models", "openrouter", "openai", "bedrock"} <= offered
    # The test double never reaches a picker.
    assert "echo" not in offered


def test_the_list_is_sorted_and_stable() -> None:
    names = wm.available_providers()
    assert names == sorted(names)
    assert wm.available_providers() == names


def test_credential_var_is_the_providers_own_first_env_step() -> None:
    # Read straight off the contracts the CLI renders, so a provider that
    # renames its variable renames it here too.
    assert wm.credential_env_var("openrouter") == "JAATO_OPENROUTER_API_KEY"
    assert wm.credential_env_var("github_models") == "GITHUB_TOKEN"
    # anthropic lists the OAuth token vars first; the api key is a later step
    assert wm.credential_env_var("anthropic") in wm.provider_catalog()["anthropic"]
    # OAuth / CLI / local providers take no key from the environment.
    for p in ("antigravity", "claude_cli", "ollama"):
        assert wm.credential_env_var(p) is None, p


def test_config_status_reports_the_full_catalog(manager: WorkspaceManager) -> None:
    status = manager.get_config_status("proj")
    assert status["available_providers"] == wm.available_providers()
    assert status["configured"] is False


def test_detection_reads_every_providers_credential_var(manager: WorkspaceManager) -> None:
    assert manager._detect_provider({"JAATO_OPENROUTER_API_KEY": "sk-or-x"}) == "openrouter"
    assert manager._detect_provider({"GITHUB_TOKEN": "ghp_x"}) == "github_models"
    assert manager._detect_provider({"ANTHROPIC_API_KEY": "sk-ant-x"}) == "anthropic"
    # Vertex AI configures through PROJECT_ID and authenticates through ADC:
    # no env credential step, so a detection hint carries it.
    assert manager._detect_provider({"PROJECT_ID": "my-gcp-project"}) == "google_genai"
    # An explicit JAATO_PROVIDER wins over any credential present.
    assert manager._detect_provider({"JAATO_PROVIDER": "kimi", "GITHUB_TOKEN": "x"}) == "kimi"
    assert manager._detect_provider({"UNRELATED": "1"}) is None


def test_update_config_writes_the_key_where_the_provider_reads_it(manager: WorkspaceManager, tmp_path: Path) -> None:
    manager.select_workspace("proj")
    manager.update_config("openrouter", model="openai/gpt-5", api_key="sk-or-abc")
    env = (tmp_path / "ws" / "proj" / ".env").read_text()
    assert "JAATO_PROVIDER=openrouter" in env
    assert "MODEL_NAME=openai/gpt-5" in env
    assert "JAATO_OPENROUTER_API_KEY=sk-or-abc" in env
    status = manager.get_config_status("proj")
    assert (status["configured"], status["provider"], status["model"]) == (True, "openrouter", "openai/gpt-5")


def test_update_config_refuses_a_provider_the_runtime_does_not_know(manager: WorkspaceManager) -> None:
    manager.select_workspace("proj")
    # The old table's own spelling is exactly the kind of name to refuse.
    with pytest.raises(ValueError, match="Unknown provider 'google'"):
        manager.update_config("google", model="gemini-2.5-flash")


def test_update_config_refuses_a_key_for_a_provider_with_no_env_credential(manager: WorkspaceManager) -> None:
    manager.select_workspace("proj")
    with pytest.raises(ValueError, match="takes no API key from the environment"):
        manager.update_config("antigravity", model="antigravity-gemini-3-pro", api_key="whatever")
    # Without a key the selection itself is fine: the auth command signs in.
    manager.update_config("antigravity", model="antigravity-gemini-3-pro")
    assert manager.get_config_status("proj")["provider"] == "antigravity"


def test_no_provider_sdk_is_imported_to_build_the_catalog() -> None:
    # The catalog is read from source (AST), so a daemon whose venv lacks a
    # provider's SDK still lists that provider -- and nothing here paid for
    # importing twenty SDKs.  ``openai`` is the heaviest common one.  The
    # bare ``google`` namespace is NOT the test: ``google.protobuf`` and
    # ``google.api_core`` arrive through unrelated dependencies (grpc,
    # OpenTelemetry), so the check names the SDK packages themselves.
    import subprocess, sys
    code = (
        "import sys; from server import workspace_manager as wm; wm.available_providers(); "
        "sdk = {'openai','anthropic','boto3','botocore','google.genai','google.generativeai','vertexai'}; "
        "print(sorted(m for m in sys.modules if m in sdk or any(m.startswith(s + '.') for s in sdk)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(Path(wm.__file__).resolve().parents[1]), check=True)
    assert out.stdout.strip() == "[]", out.stdout
