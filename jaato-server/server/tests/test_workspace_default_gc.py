"""A workspace created by a client gets a GC strategy, so its sessions have one.

``create_workspace`` made ``.jaato/`` and touched an empty ``.env``, and
nothing else.  A session driven against such a workspace is a bare
``session.new`` over the ``JAATO_PROVIDER`` / ``MODEL_NAME`` pair in that
``.env``, so it has no profile and therefore no ``gc:`` block; with no
``gc.json`` beside it either, ``JaatoServer.initialize``'s

    if not gc_result:
        gc_result = load_gc_from_file(workspace_root=...)

leaves ``gc_result`` at ``None`` and the session runs with **no context
garbage collection at all** -- the history grows until the pre-send guard
refuses it or the upstream does.

The vehicle is ``gc.json`` and not ``.env``, which is the thing worth
pinning: there is no ``JAATO_GC_TYPE``.  The four ``JAATO_GC_*`` variables
are read by ``GCConfig``'s field defaults, and that object is constructed
only once a strategy has been selected -- so a threshold written into
``.env`` with nothing selecting a strategy configures nothing, silently.

These tests assert the fix is real end to end: the file lands where
``load_gc_from_file`` looks, and that function -- the real one, not a
stand-in -- returns a plugin for it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from server.workspace_manager import WorkspaceManager
from shared.plugins.gc import load_gc_from_file


@pytest.fixture
def manager(tmp_path: Path) -> WorkspaceManager:
    root = tmp_path / "ws"
    root.mkdir()
    return WorkspaceManager(str(root), registry_path=tmp_path / "registry.json")


def test_a_new_workspace_carries_a_gc_config(manager: WorkspaceManager) -> None:
    info = manager.create_workspace("proj")
    gc_json = Path(info.path) / ".jaato" / "gc.json"
    assert gc_json.is_file(), "a workspace with no gc.json has no GC at all"
    assert json.loads(gc_json.read_text())["type"] == "budget"


def test_load_gc_from_file_finds_it(manager: WorkspaceManager) -> None:
    """The path the daemon actually searches, exercised by the real loader.

    ``JaatoServer.initialize`` calls ``load_gc_from_file(workspace_root=...)``
    and installs nothing when it answers ``None``.  Asserting only that a
    file exists would pass just as well for a file written one directory
    over, which is the mistake worth guarding against.
    """
    info = manager.create_workspace("proj")
    result = load_gc_from_file(workspace_root=info.path)
    assert result is not None
    plugin, config = result
    assert plugin.name == "gc_budget"


def test_the_generated_file_re_spells_no_framework_default(
    manager: WorkspaceManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only ``type`` is written, so the env knobs still decide the rest.

    ``shared.plugins.gc._media_settings`` states the rule: omission has to
    mean "the dataclass decides", not "the default I happened to type".  A
    generated file that spelled out today's ``threshold_percent: 80.0``
    would outrank ``JAATO_GC_THRESHOLD`` for every workspace created before
    that number next moves -- so the check is that the env var still wins,
    which is a fact about behaviour rather than about the file's key set.
    """
    info = manager.create_workspace("proj")
    assert json.loads((Path(info.path) / ".jaato" / "gc.json").read_text()) == {
        "type": "budget"
    }

    monkeypatch.setenv("JAATO_GC_THRESHOLD", "55.0")
    result = load_gc_from_file(workspace_root=info.path)
    assert result is not None
    _, config = result
    assert config.threshold_percent == 55.0, (
        "the generated gc.json overrode an env knob it never mentioned"
    )


def test_an_unwritable_config_dir_does_not_fail_the_creation(
    manager: WorkspaceManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort: no gc.json is the state every workspace was already in.

    Losing the default is a workspace whose sessions have no GC -- exactly
    today's behaviour -- while raising here would lose the workspace.
    """
    def _boom(*_a, **_k):
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", _boom)
    info = manager.create_workspace("proj")
    assert Path(info.path).is_dir()
    assert not (Path(info.path) / ".jaato" / "gc.json").exists()


def test_an_existing_workspace_is_not_migrated(
    manager: WorkspaceManager, tmp_path: Path,
) -> None:
    """Discovery must not write into a directory it did not create.

    The file is a starting point its owner is meant to edit, and a
    workspace that predates this change may have been deliberately left
    without one.
    """
    existing = tmp_path / "ws" / "old"
    (existing / ".jaato").mkdir(parents=True)
    (existing / ".env").write_text("JAATO_PROVIDER=anthropic\n")
    manager.discover_workspaces()
    assert not (existing / ".jaato" / "gc.json").exists()
