"""``jaato-scaffold explain oversight`` -- the Article 14 measures, computed.

The framework's stop button exists twice -- ``jaato-server --stop`` for the
deployment, ``session.stop <id>`` for one session -- and no surface a
deployer reads named either as the human-oversight measure Regulation (EU)
2024/1689 Art. 14(4)(e) asks for.  The framework's rule for facts about
itself is that they are COMPUTED (``explain`` exists so a documented order
cannot disagree with the loaded one), so this topic reads every value off
the module that enforces it.  ``docs/design/eu-ai-act.md`` §4.5.

Pinned here: the page is dispatched and advertised like every topic; each
figure on it is the framework's own constant rather than a copy; and the
named form reports what a RESOLVED profile has armed -- including a
ladder that declares limits and cannot stop, which is the state #947
found in production.
"""

from __future__ import annotations

from pathlib import Path

from shared.scaffold import explain
from shared.scaffold.__main__ import _SCOPES, scope_catalog
from shared.tests.reversion import Reversion

_EXPLAIN = "jaato-server/shared/scaffold/explain.py"

REVERSIONS = [
    Reversion(
        target=_EXPLAIN,
        find='        "stops": bool(getattr(budget, "has_abort_rung", False)),',
        replace='        "stops": bool(rungs),',
        because=(
            "a ladder with rungs and no abort does NOT stop the run (#947); "
            "reporting it as stopping certifies a ceiling nobody enforces"
        ),
        test="test_a_ladder_without_abort_is_reported_as_not_stopping",
    ),
]


def test_the_topic_is_dispatched_and_reads_the_workspace():
    spec = _SCOPES["oversight"]
    assert spec.kind == "optional_named"
    assert spec.render is explain.oversight
    assert spec.render_named is explain.oversight_profile
    row = next(t for t in scope_catalog() if t["scope"] == "oversight")
    assert row["reads_workspace"]


def test_the_two_stop_verbs_carry_the_frameworks_own_constants():
    from jaato_sdk.client.ipc import DEFAULT_PID_FILE, DEFAULT_SOCKET_PATH, IPCClient

    data, text = explain.oversight()
    verbs = [s["verb"] for s in data["stop"]]
    assert verbs == ["jaato-server --stop", "session.stop <id>"]
    assert DEFAULT_PID_FILE in data["stop"][0]["invocation"]
    assert DEFAULT_SOCKET_PATH in data["stop"][0]["invocation"]
    assert IPCClient.MIN_SESSION_STOP_PROTOCOL in data["stop"][1]["where"]
    assert "14(4)(e)" in text


def test_the_constraints_are_read_from_their_enforcers():
    from shared import budget_control as bc
    from shared import runtime_limits as rl

    data, _ = explain.oversight()
    by_mech = {c["mechanism"]: c["detail"] for c in data["built_in_constraints"]}
    ladder = by_mech["budget_control degrade ladder"]
    for dim in bc.DIMENSIONS:
        assert dim in ladder
    assert f"{rl.DEFAULT_MAX_ORPHAN_SECONDS:g}s" in by_mech["session-lifetime watchdog (runtime_limits)"]


def test_the_policy_vocabulary_is_the_permission_plugins_own():
    """Read off the plugin's declared schema, never a copy: the enum here is
    the one ``validate`` enforces."""
    from shared.scaffold import introspect

    data, _ = explain.oversight()
    vocab = data["decision_gate"]["policy_vocabulary"]
    info = introspect.plugins().get("permission")
    if info is None:  # a build without the plugin renders nothing, not a guess
        assert vocab == {}
        return
    policy = next(s for s in info.config_settings if s.name == "policy")
    default = next(c for c in policy.children if c.name == "defaultPolicy")
    assert vocab["policy.defaultPolicy"] == list(default.enum)


def _workspace(tmp_path: Path, child_budget: str) -> Path:
    pdir = tmp_path / ".jaato" / "profiles"
    pdir.mkdir(parents=True)
    (pdir / "_base.yaml").write_text(
        "name: _base\ndescription: base\nplugins: []\n"
        "plugin_configs:\n  permission:\n    policy:\n      defaultPolicy: deny\n"
        "      whitelist: {tools: [readFile]}\n", encoding="utf-8")
    (pdir / "worker.yaml").write_text(
        "name: worker\ndescription: worker\ninherits: [_base]\nplugins: [cli]\n"
        + child_budget, encoding="utf-8")
    return tmp_path


def test_the_named_form_reports_what_a_resolved_profile_armed(tmp_path):
    ws = _workspace(tmp_path, (
        "budget_control:\n  limits: {usd: 5}\n"
        "  degrade: [{at: 100, action: abort}]\n"))
    data, text = explain.oversight_profile("worker", str(ws))
    assert data["found"]
    # The permission policy came from the BASE -- resolved, not raw.
    assert data["permission_policy"]["defaultPolicy"] == "deny"
    assert data["permission_policy"]["whitelist_tools"] == 1
    assert data["budget_control"]["stops"] is True
    assert data["irreversible_surfaces"] == ["cli"]
    assert "STOPS the run" in text
    assert "defaultPolicy=deny" in text


def test_a_ladder_without_abort_is_reported_as_not_stopping(tmp_path):
    ws = _workspace(tmp_path, (
        "budget_control:\n  limits: {usd: 5}\n"
        "  degrade: [{at: 100, action: finalize}]\n"))
    data, text = explain.oversight_profile("worker", str(ws))
    assert data["budget_control"]["rungs"] == [{"at": 100.0, "action": "finalize"}]
    assert data["budget_control"]["stops"] is False
    assert "does NOT stop the run" in text


def test_an_unknown_profile_is_an_error_not_a_page(tmp_path):
    data, _ = explain.oversight_profile("nobody", str(_workspace(tmp_path, "")))
    assert "error" in data
