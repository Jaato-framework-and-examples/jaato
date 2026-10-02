"""A driver can create the reference bundle it promotes into (#1478).

A cascade driver promotes each run's claims into a sub-bundle named after the
run.  The bundle had to exist first, and the only way to make one was the
references plugin's ``references bundle create`` -- a SESSION command that
also refused outright when no embedding provider was active.  So a driver,
which holds no session, got ``unknown_bundle`` on every promotion (200 of them
on one run), and a workspace without embeddings could not have a sub-bundle
at all.

Two fixes, guarded here:

* ``ReferenceBundleCreateRequest`` (protocol 1.36) creates an UNINDEXED
  workspace sub-bundle, daemon-side and without a session, under the owner
  rule promotion uses.  Driven end to end over a real ``JaatoIPCServer`` and
  ``IPCClient`` with no session and no embedding provider anywhere.
* ``references bundle create`` with no provider creates an unindexed bundle
  instead of refusing; ``references bundle index`` adds the index later.

Every wait is short, so a regression fails in seconds.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import pytest

from jaato_server.shared.tests.reversion import Reversion

_CURATION = "jaato-server/jaato_server/server/reference_curation.py"
_EVENTS = "jaato-sdk/jaato_sdk/events.py"
_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"

#: Seconds a correlated call may wait; a working path answers in milliseconds.
_WAIT = 3.0


class _NoSessionManager:
    """No sessions; the workspace owner is whatever the test sets."""

    owner: Optional[str] = None

    def get_client_session(self, client_id: str) -> Any:
        return None

    def get_session(self, session_id: str) -> Any:
        return None

    def creator_in_workspace(self, *args: Any, **kwargs: Any) -> Optional[str]:
        return None

    def _workspace_owner_of(self, workspace: str) -> Optional[str]:
        return self.owner

    def handle_request(self, *args: Any, **kwargs: Any) -> None:
        """``ClientConfigRequest`` lands here; nothing to apply."""

    def detach_client(self, client_id: str) -> None:
        pass

    def unregister_all_cascade_clients_for_connection(self, *a: Any, **k: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *a, **k: None


def _drive(body: Callable[[Any, Path], Awaitable[None]], *,
           owner: Optional[str] = None) -> None:
    """Run ``body(client, workspace)`` against a live, session-less IPC server."""
    from jaato_sdk.client.ipc import IPCClient
    from jaato_sdk.events import ClientType

    from jaato_server.server.command_router import CommandRouter
    from jaato_server.server.ipc import JaatoIPCServer

    tmp = tempfile.mkdtemp(prefix="jaato1478-", dir="/tmp")
    workspace = Path(tmp) / "ws"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "deploy.md").write_text("# Deploy\n")
    (workspace / ".jaato").mkdir(parents=True)
    sock = os.path.join(tmp, "d.sock")
    manager = _NoSessionManager()
    manager.owner = owner

    async def main() -> None:
        server = JaatoIPCServer(socket_path=sock)
        router = CommandRouter(session_manager=manager,  # type: ignore[arg-type]
                               event_sink=server, daemon_plugins={})
        server._on_session_request = router.handle_request
        task = asyncio.create_task(server.start())
        try:
            deadline = time.monotonic() + 5
            while not os.path.exists(sock):
                assert time.monotonic() < deadline, "IPC server did not bind"
                await asyncio.sleep(0.02)
            client = IPCClient(sock, auto_start=False, client_type=ClientType.API,
                               workspace_path=str(workspace))
            assert await client.connect(timeout=5)
            deadline = time.monotonic() + 5
            while not any(server.get_client_workspace(cid) for cid in list(server._clients)):
                assert time.monotonic() < deadline, "workspace not declared"
                await asyncio.sleep(0.02)
            try:
                await body(client, workspace)
            finally:
                await client.disconnect()
        finally:
            await server.stop()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    try:
        asyncio.run(main())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_claim(workspace: Path) -> str:
    from jaato_server.shared.plugins.references.claims import (
        build_proposed_reference,
        new_claim,
        write_claim,
    )
    entry, errors = build_proposed_reference(
        {"id": "runbook", "name": "Runbook", "description": "how to deploy",
         "path": "docs/deploy.md"},
        workspace=str(workspace), catalog_ids=[])
    assert entry is not None, errors
    claim = new_claim(entry, None)
    write_claim(str(workspace), claim)
    return claim["claim_id"]


def test_a_session_less_driver_creates_an_unindexed_bundle_and_promotes_into_it() -> None:
    """(a) No session, no embedding provider: create, then promote, both succeed."""

    async def body(client: Any, workspace: Path) -> None:
        created = await client.create_reference_bundle("run-41861364", timeout=_WAIT)
        assert created.ok is True, created
        assert created.indexed is False
        assert {"name": "run-41861364", "indexed": False} in created.bundles
        bundle_dir = workspace / ".jaato" / "references" / "run-41861364"
        assert (bundle_dir / "bundle.json").is_file()
        assert not (bundle_dir / "embedding_config.json").exists()

        claim_id = _write_claim(workspace)
        promoted = await client.promote_reference_claim(
            claim_id, bundle="run-41861364", timeout=_WAIT)
        assert promoted.ok is True, promoted
        assert promoted.reconcile == "none"
        assert (bundle_dir / "runbook.json").is_file()

    _drive(body)


def test_an_existing_bundle_is_a_collision_and_untouched() -> None:
    """(b) A second create of the same name answers ``collision``."""

    async def body(client: Any, workspace: Path) -> None:
        first = await client.create_reference_bundle("kb", timeout=_WAIT)
        assert first.ok is True, first
        manifest = workspace / ".jaato" / "references" / "kb" / "bundle.json"
        before = manifest.read_bytes()
        again = await client.create_reference_bundle("kb", timeout=_WAIT)
        assert again.ok is False
        assert again.category == "collision", again
        assert manifest.read_bytes() == before
        assert [b["name"] for b in again.bundles] == ["kb"]

    _drive(body)


def test_a_non_owner_is_refused() -> None:
    """(c) On an owned workspace, a connection that is not its owner is refused."""

    async def body(client: Any, workspace: Path) -> None:
        refused = await client.create_reference_bundle("kb", timeout=_WAIT)
        assert refused.ok is False
        assert refused.category == "not_owner", refused
        assert not (workspace / ".jaato" / "references" / "kb").exists()

    _drive(body, owner="app:alice")


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", "/abs", "root", ""])
def test_a_name_that_is_not_one_flat_component_is_refused(name: str, tmp_path: Path) -> None:
    """(d) Traversal, nesting, absolute and root names are ``invalid_request``."""
    from jaato_server.server.reference_curation import create_bundle

    ws = tmp_path / "ws"
    (ws / ".jaato").mkdir(parents=True)
    outcome = create_bundle(str(ws), name, owner=None, user_id=None)
    assert outcome.ok is False
    assert outcome.category == "invalid_request", outcome
    assert not (tmp_path / "escape").exists()
    assert not (ws / ".jaato" / "references").exists()


def test_the_session_command_creates_an_unindexed_bundle_without_a_provider(
        tmp_path: Path, monkeypatch) -> None:
    """(e) ``references bundle create`` with no provider no longer refuses."""
    from jaato_server.shared.plugins.references.plugin import ReferencesPlugin

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    (ws / ".jaato" / "references").mkdir(parents=True)
    plugin = ReferencesPlugin()
    plugin.set_workspace_path(str(ws))
    plugin.initialize({"lookup_strategy": "tags_only"})
    plugin._embedding_provider = None

    result = plugin._execute_bundle_cmd({"subcommand": "create", "target": "kb"})
    assert result.get("status") == "ok", result
    assert result["indexed"] is False
    bundle_dir = ws / ".jaato" / "references" / "kb"
    assert json.loads((bundle_dir / "bundle.json").read_text())["name"] == "kb"
    assert not (bundle_dir / "embedding_config.json").exists()
    assert any(b.name == "kb" and not b.has_index for b in plugin._bundles)


def test_bundle_index_without_a_provider_is_refused_and_changes_nothing(
        tmp_path: Path, monkeypatch) -> None:
    """Indexing is a separate step, and needs a provider."""
    from jaato_server.shared.plugins.references.plugin import ReferencesPlugin

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    ws = tmp_path / "ws"
    (ws / ".jaato" / "references").mkdir(parents=True)
    plugin = ReferencesPlugin()
    plugin.set_workspace_path(str(ws))
    plugin.initialize({"lookup_strategy": "tags_only"})
    plugin._embedding_provider = None
    assert plugin._execute_bundle_cmd({"subcommand": "create", "target": "kb"})["status"] == "ok"

    result = plugin._execute_bundle_cmd({"subcommand": "index", "target": "kb"})
    assert "embedding provider" in result.get("error", ""), result
    assert not (ws / ".jaato" / "references" / "kb" / "embedding_config.json").exists()


REVERSIONS = [
    Reversion(
        target=_EVENTS,
        find="    ReferenceLinksUpdateRequest,\n    ReferenceBundleCreateRequest,\n)\n",
        replace="    ReferenceLinksUpdateRequest,\n)\n",
        test="test_a_session_less_driver_creates_an_unindexed_bundle_and_promotes_into_it",
        because=("the create request is dropped at the IPC session gate, so a "
                 "session-less driver cannot create its bundle"),
    ),
    Reversion(
        target=_CURATION,
        find="    if target and os.path.lexists(target):\n",
        replace="    if False:\n",
        test="test_an_existing_bundle_is_a_collision_and_untouched",
        because="an existing bundle's manifest would be overwritten",
    ),
    Reversion(
        target=_CURATION,
        find="    allowed = may_curate(owner, user_id)\n",
        replace="    allowed = True\n",
        test="test_a_non_owner_is_refused",
        because="anyone could create bundles in a workspace another user owns",
    ),
    Reversion(
        target=_CURATION,
        find="    if not valid_id(name) or name in _RESERVED_BUNDLE_NAMES:\n",
        replace="    if not name:\n",
        test="test_a_name_that_is_not_one_flat_component_is_refused",
        because="a name with '..' or '/' would write a bundle outside one flat component",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if self._embedding_provider is None:\n"
             "            return None, None\n",
        replace="        if self._embedding_provider is None:\n"
                "            return None, \"bundle create requires an embedding provider\"\n",
        test="test_the_session_command_creates_an_unindexed_bundle_without_a_provider",
        because="a session with no embedding provider could not create a bundle at all",
    ),
]
