"""A reference claim says who approved it at the prompt, and only when somebody did.

Step 2 of the wikiLLM write path (``docs/design/wikillm-brainstorm.md`` §6):
a ``proposeReference`` call a PERSON approved is worth more than one a policy
let through, and the claim is written during the call, so the approval has
to be visible to the tool body.  ``ToolExecutor`` now binds the call's
permission verdict for the duration of the body
(:mod:`jaato_server.shared.call_witness`), and the claim's origin carries it
as ``witnessed_by``.

Driven end to end through a real ``ToolExecutor``, a real
``PermissionPlugin`` whose channel answers the prompt, and the real
references executor.  Properties, each a way the stamp could lie:

1. **Asked-and-approved is a witness; a policy approval is not.**  The
   plugin's ``asked`` flag (#968) decides, never ``method`` -- ``allow_all``
   is produced both by a person answering ``a`` and by a pre-approval.
2. **Who answered is the channel's attribution (#859)**, and nothing is
   invented when it named nobody.
3. **No witness leaks** into a later, or an enclosing, call.
4. **``witness_proposals`` is what makes it reachable**: ``proposeReference``
   is auto-approved (whitelisted, asked nobody) unless the knob is set.
5. **Promotion carries it** to the catalog as recorded.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest

from jaato_server.shared.ai_tool_runner import ToolExecutor
from jaato_server.shared.call_witness import (
    WITNESS_VIA_PERMISSION_PROMPT,
    bound_call_witness,
    current_call_witness,
    witness_from_verdict,
)
from jaato_server.shared.plugins.permission.channels import (
    ChannelDecision,
    ChannelResponse,
)
from jaato_server.shared.plugins.permission.plugin import PermissionPlugin
from jaato_server.shared.plugins.references.claims import claims_dir
from jaato_server.shared.plugins.references.models import ReferenceOrigin
from jaato_server.shared.plugins.references.plugin import ReferencesPlugin
from jaato_server.shared.session_context import (
    isolated_current_session,
    set_current_session,
)
from jaato_server.shared.tests.reversion import Reversion

if "jaato_server.server" not in sys.modules:
    _stub = types.ModuleType("jaato_server.server")
    _stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "server")]
    sys.modules["jaato_server.server"] = _stub

from jaato_server.server.reference_curation import curate_claim  # noqa: E402

_WITNESS = "jaato-server/jaato_server/shared/call_witness.py"
_RUNNER = "jaato-server/jaato_server/shared/ai_tool_runner.py"
_CLAIMS = "jaato-server/jaato_server/shared/plugins/references/claims.py"
_REFS = "jaato-server/jaato_server/shared/plugins/references/plugin.py"
_PERM = "jaato-server/jaato_server/shared/plugins/permission/plugin.py"
_CURATION = "jaato-server/jaato_server/server/reference_curation.py"

REVERSIONS = [
    Reversion(
        target=_WITNESS,
        find='    if perm_info.get("asked") is not True:\n        return None\n',
        replace="",
        because=(
            "every allowed call becomes a witness, so a claim the whitelist "
            "let through reads as one a person approved"
        ),
        test="TestAWitnessIsAPersonsApproval::test_a_policy_approval_is_no_witness",
    ),
    Reversion(
        target=_RUNNER,
        find="                with bound_call_witness(gate.witness):\n",
        replace="                with bound_call_witness(None):\n",
        because=(
            "the verdict never reaches the tool body, so a claim a person "
            "approved at the prompt records nobody"
        ),
        test="TestAWitnessIsAPersonsApproval::test_an_answered_prompt_is_recorded",
    ),
    Reversion(
        target=_CLAIMS,
        find="        witnessed_by=current_call_witness(),\n",
        replace="",
        because="the claim never asks for the witness the executor bound",
        test="TestAWitnessIsAPersonsApproval::test_an_answered_prompt_is_recorded",
    ),
    Reversion(
        target=_PERM,
        find='        info = {**info, "asked": asked}\n',
        replace="        info = dict(info)\n",
        because=(
            "the gate's caller cannot tell a person's approval from a "
            "policy's, so no call is ever witnessed"
        ),
        test="TestAWitnessIsAPersonsApproval::test_an_answered_prompt_is_recorded",
    ),
    Reversion(
        target=_RUNNER,
        find="                with bound_call_witness(gate.witness):\n",
        replace="                with (bound_call_witness(gate.witness) if gate.witness "
                "else contextlib.nullcontext()):\n",
        because=(
            "a call nobody approved inherits the witness of whatever "
            "enclosing call bound one"
        ),
        test="TestNoWitnessLeaks::test_an_enclosing_witness_does_not_reach_a_policy_call",
    ),
    Reversion(
        target=_REFS,
        find='        if self._witness_proposals:\n            tools.remove("proposeReference")\n',
        replace="",
        because=(
            "proposeReference stays auto-approved, so no deployment can "
            "have a person approve (and be recorded approving) a proposal"
        ),
        test="TestTheKnobMakesItReachable::test_witness_proposals_puts_it_to_the_prompt",
    ),
    Reversion(
        target=_CURATION,
        find="        witnessed_by=recorded.witnessed_by if recorded else None,\n",
        replace="",
        because="promotion drops who approved the proposal",
        test="TestPromotionCarriesIt::test_the_catalog_entry_keeps_the_witness",
    ),
]


class _Session:
    """What the stamp and the permission gate read off a session."""

    _client_user_id = "acme:alice"

    def get_session_state(self, _key: str) -> None:
        return None

    def _model_provenance(self) -> Dict[str, Any]:
        return {"kind": "ai", "provider": "p", "model": "m",
                "session_id": "S1", "agent_id": "writer"}


def _references(ws: Path, **config: Any) -> ReferencesPlugin:
    plugin = ReferencesPlugin()
    plugin._workspace_path = str(ws)
    plugin._sources = []
    plugin._witness_proposals = config.get("witness_proposals") is True
    return plugin


def _asking_permission(response: ChannelResponse) -> PermissionPlugin:
    """``defaultPolicy: ask``, and a channel that answers with *response*."""
    plugin = PermissionPlugin()
    plugin.initialize({"policy": {"defaultPolicy": "ask"}})
    channel = Mock()
    channel.name = "webhook"
    channel.request_permission.return_value = response
    plugin._channel = channel
    return plugin


def _executor(refs: ReferencesPlugin, perm: PermissionPlugin) -> ToolExecutor:
    executor = ToolExecutor(auto_background_enabled=False)
    executor.register("proposeReference", refs.get_executors()["proposeReference"])
    perm.add_whitelist_tools(refs.get_auto_approved_tools())
    executor.set_permission_plugin(perm, {"session_id": "S1"})
    return executor


def _propose(executor: ToolExecutor, ref_id: str = "runbook") -> Any:
    with isolated_current_session():
        set_current_session(_Session())
        return executor.execute("proposeReference",
                                {"id": ref_id, "name": "Runbook", "content": "steps"})


def _origin(ws: Path, ref_id: str = "runbook") -> Dict[str, Any]:
    for path in claims_dir(str(ws)).glob("*.json"):
        claim = json.loads(path.read_text(encoding="utf-8"))
        if claim["reference"]["id"] == ref_id:
            return claim["origin"]
    raise AssertionError(f"no claim for {ref_id}")


def _approve(user: Optional[str] = "acme:bob", approver: Optional[str] = None) -> ChannelResponse:
    return ChannelResponse(request_id="r", decision=ChannelDecision.ALLOW,
                           reason="ok", user_id=user, approver=approver)


class TestAWitnessIsAPersonsApproval:
    def test_an_answered_prompt_is_recorded(self, tmp_path: Path) -> None:
        refs = _references(tmp_path, witness_proposals=True)
        ok, result = _propose(_executor(refs, _asking_permission(_approve())))
        assert ok is True and result["witnessed"] is True
        witness = _origin(tmp_path)["witnessed_by"]
        assert witness["via"] == WITNESS_VIA_PERMISSION_PROMPT
        assert witness["user"] == "acme:bob"
        assert "proposed by" in ReferenceOrigin.from_dict(_origin(tmp_path)).describe()
        assert "approved at the prompt by acme:bob" in (
            ReferenceOrigin.from_dict(_origin(tmp_path)).describe())

    def test_a_policy_approval_is_no_witness(self, tmp_path: Path) -> None:
        # The default: proposeReference is auto-approved (whitelisted), so
        # the call is allowed without anybody being asked.
        refs = _references(tmp_path)
        perm = _asking_permission(_approve())
        ok, result = _propose(_executor(refs, perm))
        assert ok is True and result["witnessed"] is False
        assert "witnessed_by" not in _origin(tmp_path)
        perm._channel.request_permission.assert_not_called()

    def test_an_unattributed_answer_is_a_witness_naming_nobody(self, tmp_path: Path) -> None:
        refs = _references(tmp_path, witness_proposals=True)
        _propose(_executor(refs, _asking_permission(_approve(user=None))))
        witness = _origin(tmp_path)["witnessed_by"]
        assert "user" not in witness and "approver" not in witness

    def test_a_refused_prompt_writes_no_claim(self, tmp_path: Path) -> None:
        refs = _references(tmp_path, witness_proposals=True)
        deny = ChannelResponse(request_id="r", decision=ChannelDecision.DENY, reason="no")
        ok, _ = _propose(_executor(refs, _asking_permission(deny)))
        assert ok is False
        assert not claims_dir(str(tmp_path)).exists()

    def test_a_verdict_without_asked_is_no_witness(self) -> None:
        # An out-of-tree policy engine that does not report ``asked``.
        assert witness_from_verdict(True, {"method": "user_approved"}) is None
        assert witness_from_verdict(False, {"asked": True}) is None


class TestNoWitnessLeaks:
    def test_a_later_call_on_the_same_executor_is_unwitnessed(self, tmp_path: Path) -> None:
        refs = _references(tmp_path, witness_proposals=True)
        perm = _asking_permission(_approve())
        executor = _executor(refs, perm)
        _propose(executor, "first")
        perm.add_whitelist_tools(["proposeReference"])  # now a policy decision
        _propose(executor, "second")
        assert "witnessed_by" in _origin(tmp_path, "first")
        assert "witnessed_by" not in _origin(tmp_path, "second")
        assert current_call_witness() is None

    def test_an_enclosing_witness_does_not_reach_a_policy_call(self, tmp_path: Path) -> None:
        refs = _references(tmp_path)
        executor = _executor(refs, _asking_permission(_approve()))
        with bound_call_witness({"via": WITNESS_VIA_PERMISSION_PROMPT, "method": "x"}):
            _propose(executor)
        assert "witnessed_by" not in _origin(tmp_path)


class TestTheKnobMakesItReachable:
    def test_proposals_are_auto_approved_by_default(self, tmp_path: Path) -> None:
        assert "proposeReference" in _references(tmp_path).get_auto_approved_tools()

    def test_witness_proposals_puts_it_to_the_prompt(self, tmp_path: Path) -> None:
        refs = _references(tmp_path, witness_proposals=True)
        assert "proposeReference" not in refs.get_auto_approved_tools()
        assert "listReferences" in refs.get_auto_approved_tools()

    def test_the_knob_is_read_from_the_config(self, tmp_path: Path) -> None:
        plugin = ReferencesPlugin()
        plugin.initialize({"witness_proposals": True, "workspace_path": str(tmp_path)})
        try:
            assert "proposeReference" not in plugin.get_auto_approved_tools()
        finally:
            plugin.shutdown()


class TestPromotionCarriesIt:
    def test_the_catalog_entry_keeps_the_witness(self, tmp_path: Path) -> None:
        refs = _references(tmp_path, witness_proposals=True)
        ok, result = _propose(_executor(refs, _asking_permission(_approve())))
        assert ok is True
        outcome = curate_claim(str(tmp_path), "promote", result["claim_id"],
                               owner=None, user_id="acme:carol",
                               creator_in_workspace=lambda _s, _w: None)
        assert outcome.ok, outcome.error
        entry = json.loads((tmp_path / outcome.reference_file).read_text(encoding="utf-8"))
        assert entry["origin"]["witnessed_by"]["user"] == "acme:bob"
        assert entry["origin"]["curated_by"]["user"] == "acme:carol"
