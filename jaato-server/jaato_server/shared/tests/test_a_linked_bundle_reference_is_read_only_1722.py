"""A reference read through a linked sub-bundle is read-only, and both refusals say so (#1722).

A workspace may link a sub-bundle into its catalog
(``.jaato/references/<bundle>`` -> a folder elsewhere with ``bundle.json``).
The loader reads it; the files a curator of this workspace may write never
include it, because promotion does not write through a link.  Before, a
plain ``proposeReference(id)`` of such an id was refused toward
``revises``, and ``revises`` was refused as "in no workspace catalog
file": the model alternated between the two until it gave up.

Properties, each a way it could go wrong:

1. **The plain proposal does not send the model to ``revises``**: it says
   the reference is read-only here and carries no ``revises`` hint.
2. **The revision gives the same reason**, naming the linked bundle.
3. **A reference in a real sub-bundle is unchanged**: still refused toward
   ``revises``, and still revisable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jaato_server.shared.plugins.references.plugin import create_plugin
from jaato_server.shared.session_context import isolated_current_session
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_CLAIMS = "jaato-server/jaato_server/shared/plugins/references/claims.py"

REVERSIONS = [
    Reversion(
        target=_CLAIMS,
        find="        read_only = read_only_reason(workspace, ref_id)\n"
             "        return None, [read_only or already_in_catalog(ref_id)]\n",
        replace="        return None, [already_in_catalog(ref_id)]\n",
        because="a plain proposal of a linked reference would send the model to revises again",
        test="TestALinkedBundle::test_a_plain_proposal_says_read_only",
    ),
    Reversion(
        target=_PLUGIN,
        find="            error = read_only_reason(workspace, entry[\"id\"]) or error\n",
        replace="",
        because="the revision refusal would not say why, nor name the link",
        test="TestALinkedBundle::test_a_revision_gives_the_same_reason",
    ),
    Reversion(
        target=_PLUGIN,
        find="            if workspace and read_only_reason(workspace, ref_id):\n",
        replace="            if False:\n",
        because="the machine-readable hint would still name the refused route",
        test="TestALinkedBundle::test_a_plain_proposal_says_read_only",
    ),
]


def _bundle(folder: Path, ref_id: str) -> None:
    folder.mkdir(parents=True)
    (folder / "bundle.json").write_text(json.dumps({"name": folder.name}), encoding="utf-8")
    (folder / f"{ref_id}.json").write_text(json.dumps({
        "id": ref_id, "name": "Ficha", "description": "old", "type": "inline",
        "mode": "selectable", "tags": ["ficha"], "content": "# old"}), encoding="utf-8")


def _session(ws: Path):
    plugin = create_plugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only"})
    plugin.set_workspace_path(str(ws))
    return plugin


def _propose(plugin, **args: Any):
    with isolated_current_session():
        return plugin.get_executors()["proposeReference"](args)


def _linked_workspace(tmp_path: Path) -> Path:
    _bundle(tmp_path / "elsewhere" / "u-1", "ficha")
    ws = tmp_path / "ws"
    refs = ws / ".jaato" / "references"
    refs.mkdir(parents=True)
    os.symlink(tmp_path / "elsewhere" / "u-1", refs / "u-1")
    return ws


class TestALinkedBundle:
    def test_the_loader_reads_it(self, tmp_path):
        plugin = _session(_linked_workspace(tmp_path))
        assert "ficha" in {s.id for s in plugin._sources}

    def test_a_plain_proposal_says_read_only(self, tmp_path):
        plugin = _session(_linked_workspace(tmp_path))
        ok, payload = _propose(plugin, id="ficha", name="Ficha", description="new",
                               tags=["ficha"], content="# new")
        assert ok is False
        assert "read-only in this workspace" in payload["error"]
        assert "revises='ficha'" not in payload["error"]
        assert "revises" not in payload
        assert payload.get("read_only") is True

    def test_a_revision_gives_the_same_reason(self, tmp_path):
        plugin = _session(_linked_workspace(tmp_path))
        ok, payload = _propose(plugin, revises="ficha", name="Ficha", description="new",
                               tags=["ficha"], content="# new")
        assert ok is False
        assert "read-only in this workspace" in payload["error"]
        assert "'u-1'" in payload["error"]


class TestARealSubBundle:
    def test_is_still_refused_toward_revises_and_revisable(self, tmp_path):
        ws = tmp_path / "ws"
        _bundle(ws / ".jaato" / "references" / "u-1", "ficha")
        plugin = _session(ws)
        ok, payload = _propose(plugin, id="ficha", name="Ficha", description="new",
                               tags=["ficha"], content="# new")
        assert ok is False and payload["revises"] == "ficha"
        result = _propose(plugin, revises="ficha", name="Ficha", description="new",
                          tags=["ficha"], content="# new")
        assert not (isinstance(result, tuple) and result[0] is False), result
        assert result["revises"] == "ficha"
