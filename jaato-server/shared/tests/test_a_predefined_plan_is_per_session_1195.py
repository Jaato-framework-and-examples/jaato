"""A predefined plan belongs to the session that declared it (#1195).

``plugin_configs.todo.initial_plan_name: <id>`` loads
``<config_root>/plans/<id>.yaml`` as a session's active plan before turn 1,
puts a fixed hint in that session's system prompt, and takes ``createPlan``
off that session's tool surface.

The todo plugin INSTANCE is not the session's: one ``PluginRegistry`` is
shared by a parent and its in-process subagents, and the instance is
carried across cascade stages on a pool slot (#890).  So every effect of the
knob must bind to ONE session — gating the instance would strip
``createPlan`` from every session sharing it, the #944 lesson.  The
mechanism is the one #957 built for permission policies: the session mints
a key (``JaatoSession.plugin_scope``), the plugin files per-session state
under it, and the gate itself is the session's own ``_tool_scopes`` filter.

These tests drive real ``JaatoSession`` objects on a real runtime and a real
``PluginRegistry`` holding one real ``TodoPlugin`` — two sessions on one
registry is the configuration the defect lives in, so it is the
configuration the tests build.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.jaato_runtime import JaatoRuntime
from shared.jaato_session import JaatoSession
from shared.plugins.registry import PluginRegistry
from shared.plugins.todo.initial_plan import (
    PRELOADED_PLAN_HINT, InitialPlanError, load_initial_plan,
)
from shared.plugins.todo.plugin import TodoPlugin
from shared.session_context import isolated_current_session, set_current_session
from shared.tests.reversion import Reversion

REPO = Path(__file__).resolve().parents[3]
README = REPO / "jaato-server" / "shared" / "plugins" / "todo" / "README.md"

_TODO = "jaato-server/shared/plugins/todo/plugin.py"
_PLAN = "jaato-server/shared/plugins/todo/initial_plan.py"
_SESSION = "jaato-server/shared/jaato_session.py"

REVERSIONS = [
    Reversion(
        target=_TODO,
        find=("        if tool_name not in PLAN_REQUIRED_TOOLS:\n"
              "            return True\n"),
        replace=("        if tool_name == GATED_TOOL and self._preloaded_scopes:\n"
                 "            return False\n"
                 "        if tool_name not in PLAN_REQUIRED_TOOLS:\n"
                 "            return True\n"),
        test="test_a_session_without_the_knob_still_sees_create_plan",
        because="the createPlan gate made instance-wide: any preload on the "
                "shared plugin hides createPlan from every session on it",
    ),
    Reversion(
        target=_TODO,
        find=("        if self._current_scope() in self._preloaded_scopes:\n"
              "            text +="),
        replace=("        if self._preloaded_scopes:\n"
                 "            text +="),
        test="test_the_hint_is_only_in_the_declaring_sessions_prompt",
        because="the hint contributed per instance: a sibling that declared "
                "nothing is told it was invoked with a predefined plan",
    ),
    Reversion(
        target=_TODO,
        find="        plan = load_initial_plan(plan_name, config_root, workspace_path)\n",
        replace=("        self._preloaded_scopes.add(scope)\n"
                 "        plan = load_initial_plan(plan_name, config_root, workspace_path)\n"),
        test="test_no_hint_when_the_plan_failed_to_load",
        because="the session recorded as preloaded before the plan loaded — "
                "a hint pointing at a plan that does not exist",
    ),
    Reversion(
        target=_SESSION,
        find="        self._apply_initial_plan(plugins)\n",
        replace=("        try:\n"
                 "            self._apply_initial_plan(plugins)\n"
                 "        except Exception:\n"
                 "            pass\n"),
        test="test_a_missing_plan_refuses_the_session",
        because="a missing plan silently degrades to a session with no plan",
    ),
    Reversion(
        target=_PLAN,
        find='    doc["plan_id"] = str(uuid.uuid4())\n',
        replace='    doc["plan_id"] = path.stem\n',
        test="test_progress_never_lands_on_the_authored_file",
        because="the stored copy shares the authored file's identity, so a "
                "file-backed storage writes progress over the authored plan",
    ),
    Reversion(
        target=_TODO,
        find=("        if scope is None or scope not in self._scoped_plan_ids:\n"
              "            return self._current_plan_ids.get(self._get_agent_name())\n"),
        replace=("        carried = self._current_plan_ids.get(self._get_agent_name())\n"
                 "        if carried or scope is None or scope not in self._scoped_plan_ids:\n"
                 "            return carried\n"),
        test="test_the_preload_overrides_a_plan_carried_from_an_earlier_stage",
        because="the per-agent map outranks the session's preload, so the "
                "previous stage's plan wins",
    ),
    Reversion(
        target=_TODO,
        find="            if getattr(session, 'agent_type', 'main') != 'subagent':\n",
        replace="            if True:\n",
        test="test_a_subagents_preload_never_replaces_its_parents_plan",
        because="a subagent (configured as agent 'main') binds its plan into "
                "the per-agent map and replaces its parent's",
    ),
    Reversion(
        target=_PLAN,
        find='            or name in (".", "..") or Path(name).is_absolute()\n',
        replace='            or Path(name).is_absolute()\n',
        test="test_a_name_that_is_not_an_id_is_refused",
        because="'..' accepted as a plan id resolves outside plans/",
    ),
    Reversion(
        target=_PLAN,
        find='    if (name != name.strip() or "/" in name or "\\\\" in name\n',
        replace='    if (name != name.strip() or "\\\\" in name\n',
        test="test_a_name_that_is_not_an_id_is_refused",
        because="a path accepted as a plan id",
    ),
    Reversion(
        target=_PLAN,
        find='        data = yaml.safe_load(path.read_text(encoding="utf-8"))\n',
        replace='        data = yaml.unsafe_load(path.read_text(encoding="utf-8"))\n',
        test="test_the_plan_is_read_with_safe_load",
        because="a hand-authored file parsed with a loader that constructs "
                "arbitrary Python objects",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/validate.py",
        find="    _check_initial_plans(result.profiles, ws, config_root, out)\n",
        replace="",
        test="test_validate_reports_a_missing_and_a_malformed_plan",
        because="validate never asks whether a declared plan loads",
    ),
]


PLAN_YAML = (
    "title: Onboard a new service\n"
    "steps:\n"
    "  - description: Read the README\n"
    "  - description: Run the tests\n"
)


# ------------------------------------------------------------------ harness


@pytest.fixture(autouse=True)
def _isolated():
    """``configure()`` and tool dispatch publish the current session to a
    process-global ContextVar; restore it so no test leaks one (#974)."""
    from shared.plugins.todo import plugin as todo_module
    with isolated_current_session():
        yield
    todo_module._thread_local.session = None


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    plans = tmp_path / ".jaato" / "plans"
    plans.mkdir(parents=True)
    (plans / "onboard.yaml").write_text(PLAN_YAML, encoding="utf-8")
    return tmp_path


def _runtime(ws: Path, todo_config=None):
    """One runtime, one registry, ONE todo instance every session shares."""
    rt = JaatoRuntime(provider_name="echo", workspace_path=ws)
    reg = PluginRegistry()
    reg.set_workspace_path(str(ws))
    todo = TodoPlugin()
    reg.register_plugin(todo, expose=True,
                        config=todo_config or {"reporter_type": "memory"})
    rt.configure_plugins(reg)
    return rt, todo


def _session(rt, plan=None, *, agent_id="main", agent_type=None) -> JaatoSession:
    session = JaatoSession(rt, "test-model", agent_id=agent_id)
    configs = {"todo": {"initial_plan_name": plan}} if plan is not None else None
    session.configure(skip_provider=True, plugins=["todo"],
                      plugin_configs=configs)
    if agent_type:
        session.set_agent_context(agent_type=agent_type, agent_name=agent_id)
    session._provider = SimpleNamespace(uses_external_tools=lambda: True)
    return session


def _wire(session: JaatoSession):
    """The tool names this session's next provider call would carry."""
    return {t.name for t in session._get_tools_for_provider()}


def _call(session: JaatoSession, todo: TodoPlugin, tool: str, args=None):
    """Run one todo tool as ``_execute_single_tool`` would for *session*."""
    set_current_session(session)
    todo.set_session(session)
    return todo.get_executors()[tool](dict(args or {}))


# ------------------------------------------------------------ the gate


def test_create_plan_is_hidden_in_the_session_that_declared_a_plan(ws):
    rt, _ = _runtime(ws)
    a = _session(rt, "onboard")

    assert "createPlan" not in {t.name for t in a._tools}
    wire = _wire(a)
    assert "createPlan" not in wire
    # ... and the plan is readable on turn 1: the plan-required tools are on.
    assert {"getPlanStatus", "setStepStatus", "addStep"} <= wire


def test_a_session_without_the_knob_still_sees_create_plan(ws):
    """The #944 lesson: the gate is this session's, never the instance's."""
    rt, _ = _runtime(ws)
    before = _session(rt)           # configured before the preload
    _session(rt, "onboard")
    after = _session(rt)            # and after it

    for sibling in (before, after):
        assert "createPlan" in {t.name for t in sibling._tools}
        assert "createPlan" in _wire(sibling)
    # The plugin's schema itself is untouched.
    assert "createPlan" in {s.name for s in rt.registry.get_plugin(
        "todo").get_tool_schemas()}


def test_create_plan_executor_refuses_only_in_the_declaring_session(ws):
    rt, todo = _runtime(ws)
    a = _session(rt, "onboard", agent_id="alpha")
    b = _session(rt, agent_id="beta")

    refused = _call(a, todo, "createPlan", {"title": "t", "steps": ["x"]})
    assert "predefined plan" in refused.get("error", "")

    made = _call(b, todo, "createPlan", {"title": "t", "steps": ["x"]})
    assert "error" not in made and made["title"] == "t"


# ------------------------------------------------------------- the hint


def test_the_hint_is_only_in_the_declaring_sessions_prompt(ws):
    rt, _ = _runtime(ws)
    a = _session(rt, "onboard")
    b = _session(rt)

    assert PRELOADED_PLAN_HINT in (a._system_instruction or "")
    assert PRELOADED_PLAN_HINT not in (b._system_instruction or "")


def test_no_hint_when_the_plan_failed_to_load(ws):
    rt, todo = _runtime(ws)
    session = _session(rt)
    set_current_session(session)

    with pytest.raises(InitialPlanError):
        todo.preload_plan(session.plugin_scope, "absent",
                          config_root=None, workspace_path=str(ws))

    assert not todo.has_preloaded_plan(session.plugin_scope)
    assert PRELOADED_PLAN_HINT not in (todo.get_system_instructions() or "")


# ----------------------------------------------------- failing loud


def test_a_missing_plan_refuses_the_session(ws):
    rt, _ = _runtime(ws)
    with pytest.raises(InitialPlanError) as exc:
        _session(rt, "absent")
    assert exc.value.code == "initial_plan_missing"
    assert "absent.yaml" in str(exc.value)


def test_a_malformed_plan_refuses_the_session(ws):
    (ws / ".jaato" / "plans" / "broken.yaml").write_text(
        "title: no steps here\n", encoding="utf-8")
    rt, _ = _runtime(ws)
    with pytest.raises(InitialPlanError) as exc:
        _session(rt, "broken")
    assert exc.value.code == "initial_plan_invalid"


def test_a_name_that_is_not_an_id_is_refused(ws):
    rt, _ = _runtime(ws)
    for bad in ("../x", "/abs", "a/b", "..", "", "onboard.yaml"):
        with pytest.raises(InitialPlanError) as exc:
            _session(rt, bad)
        assert exc.value.code == "initial_plan_name_invalid", bad


def test_the_plan_is_read_with_safe_load(ws):
    """A tag that only a Python-object loader understands is refused, not
    constructed: the file is hand-authored data."""
    (ws / ".jaato" / "plans" / "tagged.yaml").write_text(
        "title: !!python/name:os.sep\n"
        "steps:\n  - description: x\n", encoding="utf-8")
    with pytest.raises(InitialPlanError):
        load_initial_plan("tagged", None, str(ws))


# ---------------------------------------------- the authored file is inert


def test_the_authored_file_is_unchanged_after_set_step_status(ws):
    authored = ws / ".jaato" / "plans" / "onboard.yaml"
    before = authored.read_bytes()
    rt, todo = _runtime(ws)
    a = _session(rt, "onboard")

    status = _call(a, todo, "getPlanStatus")
    first = status["steps"][0]["step_id"]
    done = _call(a, todo, "setStepStatus",
                 {"step_id": first, "status": "completed"})

    assert "error" not in done, done
    assert authored.read_bytes() == before
    again = _call(a, todo, "getPlanStatus")
    assert again["steps"][0]["status"] == "completed"


def test_progress_never_lands_on_the_authored_file(ws):
    """Even a file storage rooted at the plans directory writes a COPY: the
    stored plan's id is minted per load, never the file's name."""
    plans = ws / ".jaato" / "plans"
    authored = plans / "onboard.yaml"
    before = authored.read_bytes()
    rt, todo = _runtime(ws, {"reporter_type": "memory", "storage_type": "file",
                             "storage_path": str(plans),
                             "storage_use_directory": True})
    a = _session(rt, "onboard")

    first = _call(a, todo, "getPlanStatus")["steps"][0]["step_id"]
    _call(a, todo, "setStepStatus", {"step_id": first, "status": "completed"})

    assert authored.read_bytes() == before


# ------------------------------------------- per agent, per stage, per child


def test_the_preload_overrides_a_plan_carried_from_an_earlier_stage(ws):
    """Stage 1 of agent ``discovery`` leaves a plan in the per-agent map
    (#890 carries the instance across stages).  Stage 2 of the same agent
    declares a predefined plan: that is the plan it gets."""
    rt, todo = _runtime(ws)
    stage1 = _session(rt, agent_id="discovery")
    _call(stage1, todo, "createPlan", {"title": "stage one", "steps": ["a"]})
    stage1.close_session()

    stage2 = _session(rt, "onboard", agent_id="discovery")
    assert _call(stage2, todo, "getPlanStatus")["title"] == \
        "Onboard a new service"
    # A root session binds its plan for the NEXT stage's handoff.
    stage3 = _session(rt, agent_id="discovery")
    assert _call(stage3, todo, "getPlanStatus")["title"] == \
        "Onboard a new service"


def test_a_subagents_preload_never_replaces_its_parents_plan(ws):
    """An in-process subagent is configured while its agent_id is still the
    default ``main`` — its parent's key.  Its plan stays its own."""
    rt, todo = _runtime(ws)
    parent = _session(rt, agent_id="main")
    _call(parent, todo, "createPlan", {"title": "parent's plan", "steps": ["a"]})

    child = _session(rt, "onboard", agent_id="main", agent_type="subagent")
    assert _call(child, todo, "getPlanStatus")["title"] == \
        "Onboard a new service"
    assert _call(parent, todo, "getPlanStatus")["title"] == "parent's plan"


# ------------------------------------------------ the documented schema


def test_the_readme_example_loads(tmp_path):
    """The schema example in the todo README is loaded as written, so the
    documentation cannot drift from what the loader accepts."""
    text = README.read_text(encoding="utf-8")
    m = re.search(r"<!-- initial-plan-example -->\s*```yaml\n(.*?)```",
                  text, re.S)
    assert m, "the README lost its marked initial-plan example"
    plans = tmp_path / ".jaato" / "plans"
    plans.mkdir(parents=True)
    (plans / "example.yaml").write_text(m.group(1), encoding="utf-8")

    plan = load_initial_plan("example", None, str(tmp_path))

    assert plan.title == "Onboard a new service"
    assert [s.sequence for s in plan.steps] == [1, 2, 3]
    assert plan.steps[1].step_id == "tests"
    assert plan.steps[2].validation_required is True
    assert plan.started is True
    assert plan.context["owner"] == "platform-team"
    assert plan.context["initial_plan_name"] == "example"


# ---------------------------------------------------------- validate


def test_validate_reports_a_missing_and_a_malformed_plan(ws):
    from shared.scaffold.validate import validate_workspace

    (ws / ".jaato" / "plans" / "broken.yaml").write_text(
        "- not a mapping\n", encoding="utf-8")
    profiles = ws / ".jaato" / "profiles"
    profiles.mkdir(parents=True)
    for name, plan in (("ok", "onboard"), ("gone", "absent"),
                       ("bad", "broken"), ("path", "../x")):
        (profiles / f"{name}.yaml").write_text(
            f"name: {name}\ndescription: d\nplugins: [todo]\n"
            f"plugin_configs:\n  todo:\n    initial_plan_name: '{plan}'\n",
            encoding="utf-8")

    found = {(d.profile, d.code) for d in validate_workspace(str(ws))
             if d.code.startswith("initial_plan")}

    assert found == {("gone", "initial_plan_missing"),
                     ("bad", "initial_plan_invalid"),
                     ("path", "initial_plan_name_invalid")}


def test_no_default_storage_path_lands_in_the_authored_plans_dir(monkeypatch):
    """The plugin's own writes must never target ``.jaato/plans/``, which a
    confined runner may not write and which holds authored input.  The one
    default path (hybrid storage) is a cwd-relative YAML file."""
    from pathlib import PurePosixPath
    from shared.plugins.todo import storage as S

    monkeypatch.delenv("TODO_STORAGE_PATH", raising=False)
    default = PurePosixPath(S.DEFAULT_STORAGE_PATH)
    assert default.suffix == ".yaml"
    assert ".jaato" not in default.parts and "plans" not in default.parts
