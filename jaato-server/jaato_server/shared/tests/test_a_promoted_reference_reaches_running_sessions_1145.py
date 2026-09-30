"""A reference promoted after a session started reaches that session (#1145).

A session loaded its references catalog once, at ``initialize()``.  Every
later write (a promotion, a link edit, a bundle reconcile, a ``git pull``)
was invisible to it until ``references reload`` was typed or a new session
started, and after #1420 a promoted page went further: the running session
listed it as ``proposed`` until promotion deleted the claim, then had it
nowhere.

The fix, pinned here:

1. Every read path the model reaches refreshes first: one ``stat`` per
   catalog directory and config file, a reload only when one moved
   (``references/catalog_watch.py``).  A promotion, a link edit and a new
   sub-bundle reference each move a directory's mtime, because every
   framework writer replaces files by rename.
2. The reload is the one ``references reload`` takes, which now includes
   the bundle half; it used to reload only the workspace root, so a reload
   dropped every sub-bundle reference.
3. A selection whose reference left the catalog is dropped, said once on
   the next references result, and said at the turn boundary.
4. A write landing during or just after a load is not lost
   (``catalog_watch.settle``).
5. A session configured with inline ``sources`` is not refreshed from
   disk, which would replace them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from jaato_server.server.reference_catalog import update_links
from jaato_server.server.reference_curation import curate_claim
from jaato_server.shared.plugins.bundle_common.bundle import write_bundle_manifest
from jaato_server.shared.plugins.references import catalog_watch
from jaato_server.shared.plugins.references.plugin import create_plugin
from jaato_server.shared.session_context import isolated_current_session
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_WATCH = "jaato-server/jaato_server/shared/plugins/references/catalog_watch.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="            self._refresh_catalog_if_changed()\n            result = execute(args)\n",
        replace="            result = execute(args)\n",
        because=(
            "listReferences and selectReferences would read the catalog the "
            "session loaded at start, so a page another session promoted is "
            "in no running session's catalog"
        ),
        test="TestAPromotionReachesARunningSession::test_the_promoted_page_is_listed",
    ),
    Reversion(
        target=_PLUGIN,
        find="        self._reload_catalog(workspace)\n        self._discover_and_load_bundles()\n",
        replace="        self._reload_catalog(workspace)\n",
        because=(
            "a reload would read only the workspace root, dropping every "
            "sub-bundle and user-tier reference from a session that had them"
        ),
        test="TestTheReloadIsWhole::test_a_sub_bundle_reference_arrives_and_stays",
    ),
    Reversion(
        target=_PLUGIN,
        find="        if dropped:\n            # The turn boundary",
        replace="        if False:\n            # The turn boundary",
        because=(
            "a selection removed from the catalog would be dropped silently "
            "at the turn boundary, and the model would go on citing a "
            "reference whose files it can no longer read"
        ),
        test="TestARemovedSelection::test_the_turn_says_so",
    ),
    Reversion(
        target=_WATCH,
        find="        stored[path] = UNSETTLED if (moved or recent) else stamp\n",
        replace="        stored[path] = stamp\n",
        because=(
            "a write landing in the same clock tick as the load leaves the "
            "directory's mtime unchanged, so the next check sees nothing "
            "and the reference is missed until something else changes"
        ),
        test="TestAWriteRacingTheLoad::test_a_recent_mtime_is_not_trusted",
    ),
    Reversion(
        target=_PLUGIN,
        find='config.get("refresh_catalog") is not False and "sources" not in config',
        replace='config.get("refresh_catalog") is not False',
        because=(
            "a session configured with inline sources would have them "
            "replaced by the disk catalog at the first changed file"
        ),
        test="TestInlineSourcesAreKept::test_a_disk_change_does_not_replace_them",
    ),
]


@pytest.fixture(autouse=True)
def _private_home(tmp_path, monkeypatch):
    """Keep the user tier (``~/.jaato/references``) out of every case."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("REFERENCES_CONFIG_PATH", raising=False)


def _refs(ws: Path) -> Path:
    return ws / ".jaato" / "references"


def _write_ref(directory: Path, ref_id: str, **extra: Any) -> None:
    data = {"id": ref_id, "name": ref_id, "description": f"about {ref_id}",
            "type": "inline", "mode": "selectable", "tags": ["ops"],
            "content": f"# {ref_id}", **extra}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{ref_id}.json").write_text(json.dumps(data), encoding="utf-8")


def _age(*paths: Path) -> None:
    """Move mtimes well out of the unsettled window, as an hour-old tree."""
    past = 1_000_000_000
    for p in paths:
        os.utime(p, (past, past))


def _session(ws: Path, **config: Any):
    plugin = create_plugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only", **config})
    plugin.set_workspace_path(str(ws))
    return plugin


def _tools(plugin) -> Dict[str, Any]:
    return plugin.get_executors()


def _listed(plugin) -> Dict[str, Any]:
    return _tools(plugin)["listReferences"]({})


def _ids(result: Dict[str, Any]):
    return {s["id"] for s in result["sources"]}


def _promote(ws: Path, ref_id: str) -> None:
    """Propose ``ref_id`` from another session and promote it as the daemon does."""
    other = _session(ws)
    with isolated_current_session():
        claim = other._execute_propose({"id": ref_id, "name": ref_id,
                                        "description": "promoted", "content": f"# {ref_id}"})
    outcome = curate_claim(str(ws), "promote", claim["claim_id"], owner=None, user_id=None,
                           creator_in_workspace=lambda _s, _w: None)
    assert outcome.ok, outcome.error


class TestAPromotionReachesARunningSession:
    def test_the_promoted_page_is_listed(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(_refs(ws), "base")
        running = _session(ws)
        assert _ids(_listed(running)) == {"base"}

        _promote(ws, "runbook")

        result = _listed(running)
        assert _ids(result) == {"base", "runbook"}
        assert result["catalog_changed"]["added"] == ["runbook"]
        # Said once.
        assert "catalog_changed" not in _listed(running)

    def test_a_link_edit_reaches_it_too(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(_refs(ws), "base")
        _write_ref(_refs(ws), "other")
        running = _session(ws)
        _listed(running)

        outcome = update_links(str(ws), "base", [{"to": "other", "rel": "depends-on"}],
                               owner=None, user_id=None)
        assert outcome.ok, outcome.error

        base = next(s for s in _listed(running)["sources"] if s["id"] == "base")
        assert base["links"] == [{"to": "other", "rel": "depends-on"}]

    def test_nothing_changed_reloads_nothing(self, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        _write_ref(_refs(ws), "base")
        _age(_refs(ws), _refs(ws).parent, ws)
        running = _session(ws)
        calls = []
        real = running._reload_from_disk
        monkeypatch.setattr(running, "_reload_from_disk",
                            lambda: calls.append(1) or real())
        for _ in range(3):
            _listed(running)
        assert calls == []

    def test_refresh_off_keeps_the_snapshot(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(_refs(ws), "base")
        running = _session(ws, refresh_catalog=False)
        _promote(ws, "runbook")
        assert _ids(_listed(running)) == {"base"}


class TestTheReloadIsWhole:
    def test_a_sub_bundle_reference_arrives_and_stays(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(_refs(ws), "base")
        running = _session(ws)
        _listed(running)

        team = _refs(ws) / "team"
        team.mkdir(parents=True)
        write_bundle_manifest(team, name="team")
        _write_ref(team, "team-guide")

        assert "team-guide" in _ids(_listed(running))
        # And a typed reload keeps it (it used to read the root alone).
        running._cmd_references_reload()
        assert "team-guide" in {s.id for s in running._sources}


class TestARemovedSelection:
    def _selected(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(_refs(ws), "base")
        _write_ref(_refs(ws), "gone")
        running = _session(ws)
        _tools(running)["selectReferences"]({"ids": ["gone"]})
        assert "gone" in running._selected_source_ids
        (_refs(ws) / "gone.json").unlink()
        return running

    def test_the_next_result_says_so(self, tmp_path):
        running = self._selected(tmp_path)
        result = _listed(running)
        assert result["catalog_changed"]["dropped_selected"] == ["gone"]
        assert "gone" not in running._selected_source_ids

    def test_the_turn_says_so(self, tmp_path):
        running = self._selected(tmp_path)
        enriched = running.enrich_prompt("what next?")
        assert "gone" in enriched.prompt
        assert "no longer authorized" in enriched.prompt


class TestAWriteRacingTheLoad:
    def test_a_recent_mtime_is_not_trusted(self, tmp_path):
        d = tmp_path / "d"
        d.mkdir()
        stamp = catalog_watch.stat_path(str(d))
        stored = catalog_watch.settle({}, {str(d): stamp}, now_ns=stamp[0] + 1_000)
        # Nothing on disk moved, and the next check still reloads: a write
        # in the same tick would not have moved the stamp either.
        assert catalog_watch.has_changed(stored)

    def test_a_write_during_the_load_is_not_trusted(self, tmp_path):
        d = tmp_path / "d"
        d.mkdir()
        stamp = catalog_watch.stat_path(str(d))
        later = stamp[0] + 10 ** 12
        stored = catalog_watch.settle({str(d): (0, 0, 0)}, {str(d): stamp}, now_ns=later)
        assert catalog_watch.has_changed(stored)

    def test_an_old_unchanged_stamp_is_trusted(self, tmp_path):
        d = tmp_path / "d"
        d.mkdir()
        _age(d)
        stamp = catalog_watch.stat_path(str(d))
        stored = catalog_watch.settle({str(d): stamp}, {str(d): stamp})
        assert not catalog_watch.has_changed(stored)


class TestInlineSourcesAreKept:
    def test_a_disk_change_does_not_replace_them(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(_refs(ws), "base")
        inline = {"id": "inline-only", "name": "inline-only", "description": "d",
                  "type": "inline", "mode": "selectable", "content": "x"}
        running = _session(ws, sources=[inline])
        assert _ids(_listed(running)) == {"inline-only"}
        _write_ref(_refs(ws), "new-on-disk")
        assert _ids(_listed(running)) == {"inline-only"}
