"""An agent proposes a new version of a catalog reference, and running sessions see it (#1437).

``proposeReference`` with ``revises: <id>`` records a REVISION claim: the
new entry plus the sha256 of the catalog file it was written against, the
digest promotion checks.  The catalog itself is still never written by the
runner.

Properties, each a way it could go wrong:

1. **The claim carries ``revises`` and the digest** of the file's bytes as
   they were when the agent wrote it.
2. **Only a reference in the agent's catalog can be revised**, and the id
   cannot change.
3. **A plain proposal of a catalog id is refused toward ``revises``** --
   naming it first, inviting no other id (a writer told "a different id"
   invented ``-r2`` ids in the kbwiki case), with ``revises`` in the payload.
4. **``listReferences`` shows a proposed revision as one**, with ``stale``.
5. **A running session sees an in-place edit.**  A reference file
   rewritten without a rename (a hand edit, a writer that does not use
   ``os.replace``) moves no directory mtime, so the #1145 check also stamps
   each reference file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from jaato_server.server.reference_curation import curate_claim
from jaato_server.shared.plugins.references.plugin import create_plugin
from jaato_server.shared.session_context import isolated_current_session
from jaato_server.shared.tests.reversion import Reversion

_PLUGIN = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_CLAIMS = "jaato-server/jaato_server/shared/plugins/references/claims.py"

REVERSIONS = [
    Reversion(
        target=_PLUGIN,
        find="            claim[REVISES_KEY] = revision_record(target)\n",
        replace="            pass\n",
        because="a revision would be written with nothing to check staleness against",
        test="TestTheClaim::test_it_carries_revises_and_the_files_digest",
    ),
    Reversion(
        target=_CLAIMS,
        find="    if ref_id not in set(catalog_ids):\n"
             "        return None, [f\"'{ref_id}' is not in the catalog; propose it as a new reference.\"]\n",
        replace="",
        because="an agent could 'revise' an id that is not in its catalog",
        test="TestTheClaim::test_an_id_not_in_the_catalog_is_refused",
    ),
    Reversion(
        target=_PLUGIN,
        find="            refusal[REVISES_KEY] = ref_id\n",
        replace="            pass\n",
        because="a driver could not tell a collision from any other refusal without parsing prose",
        test="TestTheClaim::test_a_new_id_that_collides_names_the_revision_route",
    ),
    Reversion(
        target=_PLUGIN,
        find="            entry[\"stale\"] = stale\n",
        replace="",
        because="an agent would not know its proposed revision can no longer be promoted",
        test="TestTheListing::test_a_revision_is_listed_as_one",
    ),
    Reversion(
        target=_PLUGIN,
        find="        paths.update(catalog_watch.json_files(\n"
             "            [p for p in paths if os.path.isdir(p)]))\n",
        replace="",
        because="an in-place edit of a reference would never reach a running session",
        test="TestARunningSession::test_an_in_place_edit_is_seen",
    ),
]


def _refs(ws: Path) -> Path:
    return ws / ".jaato" / "references"


def _write_ref(ws: Path, ref_id: str = "runbook", **extra: Any) -> Path:
    data = {"id": ref_id, "name": "Runbook", "description": "old steps",
            "type": "inline", "mode": "selectable", "tags": ["ops"],
            "content": "# old", **extra}
    _refs(ws).mkdir(parents=True, exist_ok=True)
    path = _refs(ws) / f"{ref_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _age(*paths: Path) -> None:
    past = 1_000_000_000
    for p in paths:
        os.utime(p, (past, past))


def _session(ws: Path):
    plugin = create_plugin()
    plugin.initialize({"workspace_path": str(ws), "lookup_strategy": "tags_only"})
    plugin.set_workspace_path(str(ws))
    return plugin


def _propose(plugin, **args: Any):
    with isolated_current_session():
        return plugin.get_executors()["proposeReference"](args)


def _listed(plugin) -> Dict[str, Any]:
    return plugin.get_executors()["listReferences"]({})


class TestTheClaim:
    def test_it_carries_revises_and_the_files_digest(self, tmp_path):
        ws = tmp_path / "ws"
        path = _write_ref(ws)
        result = _propose(_session(ws), revises="runbook", name="Runbook",
                          description="current steps", content="# new")
        assert result["success"] and result["revises"] == "runbook"
        claim = json.loads((ws / result["claim_file"]).read_text())
        assert claim["revises"] == {"id": "runbook", "file": ".jaato/references/runbook.json",
                                    "digest": hashlib.sha256(path.read_bytes()).hexdigest()}
        assert claim["reference"]["id"] == "runbook"
        assert claim["origin"]["kind"] == "agent"

    def test_the_catalog_is_not_written(self, tmp_path):
        ws = tmp_path / "ws"
        path = _write_ref(ws)
        before = path.read_bytes()
        _propose(_session(ws), revises="runbook", name="Runbook", content="# new")
        assert path.read_bytes() == before

    def test_an_id_not_in_the_catalog_is_refused(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(ws)
        ok, payload = _propose(_session(ws), revises="nope", name="x", content="y")
        assert ok is False and "not in the catalog" in payload["error"]

    def test_the_id_cannot_change(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(ws)
        ok, payload = _propose(_session(ws), revises="runbook", id="runbook-2",
                               name="x", content="y")
        assert ok is False and "supersedes" in payload["error"]

    def test_the_inline_cap_still_applies(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(ws)
        ok, payload = _propose(_session(ws), revises="runbook", name="x",
                               content="y" * (33 * 1024))
        assert ok is False and "exceeds" in payload["error"]

    def test_a_new_id_that_collides_names_the_revision_route(self, tmp_path):
        """The kbwiki case: told "propose a different id", a writer invented
        ``-r2`` ids and spent its budget.  The refusal names ``revises``
        first, invites no other id, and carries it for a driver to re-call."""
        ws = tmp_path / "ws"
        _write_ref(ws)
        ok, payload = _propose(_session(ws), id="runbook", name="x", content="y")
        assert ok is False
        assert payload["error"].startswith("'runbook' is already in the catalog. To change it")
        assert "revises='runbook'" in payload["error"]
        assert "propose a different id" not in payload["error"]
        assert payload["revises"] == "runbook"

    def test_the_suggested_call_works(self, tmp_path):
        ws = tmp_path / "ws"
        _write_ref(ws)
        plugin = _session(ws)
        _ok, refusal = _propose(plugin, id="runbook", name="Runbook", content="# fixed")
        result = _propose(plugin, revises=refusal["revises"], name="Runbook",
                          content="# fixed")
        assert result["success"] and result["revises"] == "runbook"


class TestTheListing:
    def test_a_revision_is_listed_as_one(self, tmp_path):
        ws = tmp_path / "ws"
        path = _write_ref(ws)
        plugin = _session(ws)
        _propose(plugin, revises="runbook", name="Runbook", content="# new")
        row = _listed(plugin)["proposed"][0]
        assert (row["id"], row["revises"], row["stale"]) == ("runbook", "runbook", False)
        path.write_text(path.read_text().replace("old", "edited"))
        row = _listed(plugin)["proposed"][0]
        assert row["stale"] is True


class TestARunningSession:
    def test_an_in_place_edit_is_seen(self, tmp_path):
        ws = tmp_path / "ws"
        path = _write_ref(ws)
        _age(path, _refs(ws), _refs(ws).parent, ws)
        running = _session(ws)
        assert _listed(running)["sources"][0]["description"] == "old steps"
        dir_stamp = os.stat(_refs(ws)).st_mtime_ns
        with open(path, "r+", encoding="utf-8") as fh:
            data = json.load(fh)
            data["description"] = "steps edited in place"
            fh.seek(0)
            fh.write(json.dumps(data))
            fh.truncate()
        assert os.stat(_refs(ws)).st_mtime_ns == dir_stamp, "an in-place edit moves no dir mtime"
        assert _listed(running)["sources"][0]["description"] == "steps edited in place"

    def test_a_promoted_revision_is_seen(self, tmp_path):
        ws = tmp_path / "ws"
        path = _write_ref(ws)
        _age(path, _refs(ws), _refs(ws).parent, ws)
        running = _session(ws)
        assert _listed(running)["sources"][0]["description"] == "old steps"
        result = _propose(_session(ws), revises="runbook", name="Runbook",
                          description="revised steps", content="# new")
        outcome = curate_claim(str(ws), "promote", result["claim_id"], owner=None,
                               user_id=None, creator_in_workspace=lambda _s, _w: None)
        assert outcome.ok, outcome.error
        assert _listed(running)["sources"][0]["description"] == "revised steps"
