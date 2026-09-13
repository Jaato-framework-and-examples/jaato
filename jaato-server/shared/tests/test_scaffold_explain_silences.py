"""Guard: the four places ``explain`` held the answer and would not say it.

Each of jaato #904, #905, #908 and #909 is the same shape — the framework
knows a fact, and the topic a reader consults does not state it, so the
source is the only recourse.  Needing to read framework source IS the
defect signal, and four sessions across three repos hit it.

WHY THESE ASSERTIONS ARE NOT PROSE GREPS.  A doc test that greps for a word
appearing elsewhere in the same output passes for the wrong reason, so every
claim here is anchored to something COMPUTED:

* the timeout defaults are read from the live ``jaato_sdk`` signatures in
  the test, then looked for in the rendered text — hard-code one in
  ``explain`` and it goes red the day the SDK moves;
* the turn-method rule is asserted to be the SAME OBJECT
  (``archetypes.TURN_METHOD_RULE``) that ``explain archetype <client>``
  renders, plus a source scan that there is exactly one definition of it —
  #909 asked for the rule at the front door, and a second wording that can
  drift from the first would be a worse answer than the silence;
* the inheritance claims are MEASURED through ``discover_profiles`` in the
  same test that asserts the documentation states them, so the two cannot
  disagree.  #908 is a semantic trap rather than a gap — ``plugins: []`` in
  a child means "keep the parent's" — and a doc fix that got the direction
  wrong would be worse than none;
* the lifecycle tool names are PROBED from ``LifecycleTools`` rather than
  listed, so a tool added there must appear in the docs.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from shared.plugins.subagent.config import discover_profiles
from shared.scaffold import archetypes as A
from shared.scaffold import explain, introspect
from shared.tests.test_every_guard_detects_its_own_reversion import Reversion


#: Put each defect back.  ``find`` is the fixed text, ``replace`` what the
#: tree said before the fix — a number written down instead of read, a
#: clause deleted, a topic that denies the thing exists.
REVERSIONS = [
    Reversion(
        target="jaato-server/shared/scaffold/introspect.py",
        find='            default=_num(_sig_default(IPCClient.create_session, "timeout")),',
        replace="            default=30.0,",
        test="test_every_documented_timeout_default_is_the_live_one",
        because="the session.new budget being READ from the live signature "
                "rather than written down, where it drifts silently and "
                "sends an author to ipc.py to find the real number",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find="    turn = _turn_method_block()",
        replace="    turn = []",
        test="test_the_turn_method_rule_is_rendered_at_the_front_door",
        because="the ask/complete decision reaching `explain clients`, "
                "instead of only `explain archetype observer` where a "
                "reader who is not scaffolding that archetype never meets it",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find='        "                                 parents\'. There is NO spelling that clears them.\\n"',
        replace='        "\\n"',
        test="test_the_docs_state_that_an_empty_child_list_keeps_the_parents",
        because="`plugins: []` in a child reading as 'no tools' when it "
                "means 'keep the parent's' — the security-shaped misreading "
                "that hands a stage the parent's whole tool surface",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find='        "                                 NESTED dict is NOT merged recursively:\\n"',
        replace='        "\\n"',
        test="test_the_docs_state_that_a_nested_config_value_is_replaced",
        because="a nested plugin_configs dict being REPLACED rather than "
                "merged — the reading that silently drops a hoisted "
                "temperature: 0.0 from the stages that override api_params",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find="        if name == LIFECYCLE_TOPIC:\n            return lifecycle()",
        replace="        pass",
        test="test_explain_plugin_lifecycle_resolves",
        because="`explain plugin lifecycle` answering `unknown plugin` and "
                "pointing at a list that by construction cannot contain it, "
                "leaving signal_completion with no discoverable owner",
    ),
    Reversion(
        target="jaato-server/shared/scaffold/explain.py",
        find="            + _session_tools_note())",
        replace="            )",
        test="test_explain_plugins_names_the_session_tools_it_cannot_list",
        because="the plugin list omitting the session-level tools in "
                "silence, which is what made `explain plugins | grep "
                "lifecycle` return 0 and send two repos to the source",
    ),
]


_SCAFFOLD = Path(explain.__file__).resolve().parent


@pytest.fixture(scope="module")
def clients_text() -> str:
    return explain.clients()[1]


@pytest.fixture(scope="module")
def profile_text() -> str:
    return explain.profile()[1]


# ------------------------------------------------------------------ #909
#
# The rule exists and lives one topic away.  Relocating it is only correct
# if there is still exactly ONE of it.

def test_the_turn_method_rule_is_rendered_at_the_front_door(clients_text):
    """``explain clients`` carries the decision, not only the three names."""
    # The block is wrapped for width, so compare on collapsed whitespace —
    # this still pins the WORDS, which is the point.
    flat = " ".join(clients_text.split())
    assert " ".join(A.TURN_METHOD_RULE.split()) in flat, (
        "explain clients does not render archetypes.TURN_METHOD_RULE"
    )


def test_every_turn_method_row_is_the_frameworks_own(clients_text):
    """The table is built from ``TURN_METHODS``, so each row must show up."""
    for m in A.TURN_METHODS:
        assert m.name in clients_text
        assert m.settles_on in clients_text, (
            f"{m.name} is listed without what settles it"
        )


def test_the_turn_method_rule_has_exactly_one_definition():
    """Relocated, never forked.

    #909 asked for the rule where the question arrives.  Answering it by
    typing a second copy into ``explain`` would reproduce the failure one
    layer up: two wordings, free to drift, with nothing to notice.  So the
    distinctive opening may appear exactly once in the package source — in
    the constant both surfaces render.
    """
    hits = [
        p for p in _SCAFFOLD.rglob("*.py")
        if "WHICH turn method" in p.read_text(encoding="utf-8")
    ]
    assert [p.name for p in hits] == ["archetypes.py"], (
        f"the rule is spelled in {[p.name for p in hits]}; it must have one "
        f"definition (archetypes.TURN_METHOD_RULE) that every surface renders"
    )
    body = (_SCAFFOLD / "archetypes.py").read_text(encoding="utf-8")
    assert body.count("WHICH turn method") == 1


# ------------------------------------------------------------------ #904.1
#
# Timeouts.  Every number is read from the SDK here, in the test, so a
# written-down copy in explain cannot pass.

def _live_defaults() -> dict:
    from jaato_sdk.client.convenience import Session, open_session
    from jaato_sdk.client.ipc import IPCClient

    def d(fn, param):
        p = inspect.signature(fn).parameters.get(param)
        return None if p is None or p.default is inspect.Parameter.empty \
            else p.default

    return {
        "facade_connect": d(open_session, "connect_timeout"),
        "bare_connect": d(IPCClient.connect, "timeout"),
        "autostart": d(IPCClient.__init__, "autostart_timeout"),
        "create": d(IPCClient.create_session, "timeout"),
        "turn": d(Session.ask, "timeout"),
    }


def test_every_documented_timeout_default_is_the_live_one(clients_text):
    """The rendered defaults ARE the installed SDK's, not a transcription."""
    live = _live_defaults()
    reported = {t.where: t.default for t in introspect.client_timeouts()}
    assert reported["IPCClient.create_session(timeout=)"] == live["create"]
    assert reported["IPCClient.connect(timeout=)"] == live["bare_connect"]
    assert reported["IPCClient(autostart_timeout=)"] == live["autostart"]
    assert (reported["jaato.session(...) / IPCClient.session(...)"]
            == live["facade_connect"])
    # …and each one reaches the page a reader actually opens.
    for key in ("create", "bare_connect", "facade_connect"):
        assert f"{live[key]:g}s" in clients_text, (
            f"the {key} default ({live[key]}) is not shown in explain clients"
        )


def test_the_session_new_budget_says_where_it_is_settable():
    """Whether the FACADE can set it is asked of the signature.

    #904's sting was not the 60s itself but that a facade user cannot
    change it (jaato #899).  Asserting the answer against the live
    signature means the day the parameter is added this flips on its own
    instead of documenting a limitation that no longer exists.
    """
    from jaato_sdk.client.convenience import open_session

    forwards = "create_timeout" in inspect.signature(open_session).parameters
    row = next(t for t in introspect.client_timeouts()
               if t.where == "IPCClient.create_session(timeout=)")
    assert row.settable_via == ("both" if forwards else "bare client only")


def test_the_timeout_that_may_leave_a_session_running_says_so(clients_text):
    """A `session.new` timeout is the one failure with a side effect."""
    assert "SessionNotConfirmed" in clients_text
    assert "MAY EXIST" in clients_text


# ------------------------------------------------------- #908 and #904.2
#
# MEASURED first, documented second — in the same test, so a doc that
# states the rule backwards cannot pass.

def _resolve(tmp_path: Path, files: dict) -> dict:
    d = tmp_path / ".jaato" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (d / f"{name}.yaml").write_text(body, encoding="utf-8")
    res = discover_profiles(str(d))
    assert not res.errors, res.errors
    return res.profiles


_PARENT = """
name: _base_x
description: base
plugins: [memory, todo]
plugin_configs:
  openrouter:
    context_length: 32768
    api_params:
      temperature: 0.0
"""


def test_an_empty_child_plugin_list_keeps_the_parents(tmp_path):
    """The behaviour #908 went to the source for, pinned."""
    profs = _resolve(tmp_path, {
        "_base_x": _PARENT,
        "kid": ("name: kid\ndescription: k\ninherits: [_base_x]\n"
                "plugins: []\n"),
        "kid2": ("name: kid2\ndescription: k\ninherits: [_base_x]\n"
                 "plugins: [cli]\n"),
    })
    # [] adds nothing and clears nothing.
    assert profs["kid"].plugins == ["memory", "todo"]
    # A listed plugin is UNIONed on top; the parent's are still there.
    assert profs["kid2"].plugins == ["memory", "todo", "cli"]


def _plugins_inheritance_row(profile_text: str) -> str:
    """Just the ``plugins, preloaded_plugins`` row of the inheritance block.

    Scoping the assertion to this slice is the difference between a real
    check and a decorative one.  ``completion_processors`` — the very next
    row — has carried the sentence "it does NOT clear the parents'" all
    along, so a bare ``in profile_text`` would find THAT one and pass with
    the ``plugins`` clause deleted: a grep matching a word that appears
    elsewhere in the same output, passing for the wrong reason.
    """
    start = profile_text.index("    plugins, preloaded_plugins")
    end = profile_text.index("    completion_processors", start)
    return profile_text[start:end]


def test_the_docs_state_that_an_empty_child_list_keeps_the_parents(
        profile_text):
    """And the topic says so, in the row for the key it is about.

    ``completion_processors`` already had this sentence; ``plugins`` had
    only the union rule, and a neighbouring block then stated the
    standalone case unconditionally.  Read while holding a child profile —
    which is when the question is asked — that block answered "none".
    """
    row = _plugins_inheritance_row(profile_text)
    assert "does NOT clear the" in row, (
        "the plugins row does not carry the empty-list sentence "
        "(completion_processors' copy does not count)"
    )
    assert "NO spelling that clears them" in row
    # The standalone block must be scoped to the case it is true for.
    assert "WITH NO PARENT" in profile_text


def test_the_only_ways_to_narrow_are_named(profile_text, tmp_path):
    """There is no clearing spelling, so the alternatives must be named.

    Measured: re-listing a plugin with ``tools:[…]`` narrows THAT plugin's
    surface while the plugin itself stays inherited — which is why the
    answer is tool_scopes, not the plugins list.
    """
    profs = _resolve(tmp_path, {
        "_base_x": _PARENT,
        "kid": ("name: kid\ndescription: k\ninherits: [_base_x]\n"
                'plugins: ["memory(tools:[retrieve_memories])"]\n'),
    })
    assert profs["kid"].plugins == ["memory", "todo"]
    assert profs["kid"].tool_scopes == {"memory": ["retrieve_memories"]}
    assert "tool_scopes" in profile_text


def test_a_nested_plugin_config_value_is_replaced_not_merged(tmp_path):
    """#904.2, measured: the merge stops one level below the plugin name."""
    profs = _resolve(tmp_path, {
        "_base_x": _PARENT,
        "kid": ("name: kid\ndescription: k\ninherits: [_base_x]\n"
                "plugins: []\nplugin_configs:\n  openrouter:\n"
                "    api_params:\n      enable_thinking: true\n"),
    })
    cfg = profs["kid"].plugin_configs["openrouter"]
    # The sibling KEY survives …
    assert cfg["context_length"] == 32768
    # … and the nested dict at api_params does NOT: temperature is gone.
    assert cfg["api_params"] == {"enable_thinking": True}
    assert "temperature" not in cfg["api_params"]


def test_the_docs_state_that_a_nested_config_value_is_replaced(profile_text):
    """"the parent's other keys survive" invites the losing design."""
    assert "NESTED dict is NOT merged recursively" in profile_text
    assert "plugin_configs" in profile_text


# ------------------------------------------------------------------ #905
#
# The owner of signal_completion.  Probed, never listed.

def test_session_tools_are_probed_from_the_live_lifecycle_tools():
    """``introspect.session_tools`` exercises the gates rather than restating.

    If it listed names, a tool added to ``lifecycle_tools.py`` would be
    documented nowhere and nothing would notice.
    """
    import types

    from shared.lifecycle_tools import LifecycleTools

    stub = types.SimpleNamespace(
        _completion_payload_schema={"type": "object", "properties": {}})
    direct = {s.name for s in LifecycleTools(stub).get_tool_schemas()}
    probed = {s.name for s in introspect.session_tools()}
    assert direct, "the probe stub no longer yields any lifecycle tool"
    assert direct <= probed, f"missing from explain: {direct - probed}"


def test_explain_plugin_lifecycle_resolves():
    """The topic the profile loader's own error sends a reader to."""
    data, text = explain.plugin("lifecycle")
    assert "unknown plugin" not in text
    assert data.get("selectable") is False
    for s in introspect.session_tools():
        assert s.name in text, f"{s.name} is not named by explain plugin lifecycle"
    # and it must say WHY it is not in the registry list
    assert "NOT a registry plugin" in text


def test_an_unknown_plugin_still_reports_unknown():
    """The lifecycle branch is a named exception, not a blanket catch."""
    _, text = explain.plugin("definitely_not_a_plugin")
    assert "unknown plugin" in text


def test_explain_plugins_names_the_session_tools_it_cannot_list():
    """The list ends by saying what is NOT in it.

    ``plugins()`` walks ``PluginRegistry``; lifecycle is wired onto the
    session, so this surface could only ever omit it.  Omission in silence
    is what produced ``explain plugins | grep -ci lifecycle`` → 0.
    """
    text = explain.plugins()[1]
    names = {s.name for s in introspect.session_tools()}
    assert names, "nothing probed — the rest of this assertion is vacuous"
    for n in names:
        assert n in text, f"{n} is absent from explain plugins"
    assert "lifecycle" in text


def test_explain_completion_names_the_owner_and_both_finish_shapes():
    """#905's comment: a watcher must tell the protocol from a loop."""
    text = explain.completion()[1]
    assert "lifecycle_tools.py" in text, "the gate never names its owner"
    for tool in ("prepare_completion", "query_completion"):
        assert tool in text, f"{tool} is undocumented in explain completion"
    # the key that turns gating on at all
    assert "completion_payload_schema" in text


def test_the_schema_key_gates_the_whole_lifecycle_surface():
    """Measured: no ``completion_payload_schema`` → no signal_completion.

    This is the fact that makes "which turn method" answerable, so it is
    pinned rather than trusted: ``complete()`` returns None on a profile
    with no schema because there is no payload, and there is no payload
    because the tool was never on the wire.
    """
    import types

    from shared.lifecycle_tools import LifecycleTools

    bare = types.SimpleNamespace()
    assert LifecycleTools(bare).get_tool_schemas() == []
    gated = types.SimpleNamespace(
        _completion_payload_schema={"type": "object", "properties": {}})
    assert [s.name for s in LifecycleTools(gated).get_tool_schemas()][0] \
        == "signal_completion"


def test_the_gate_is_stated_in_the_lifecycle_topic():
    """A reader must not have to infer the opt-in from an empty tool list."""
    text = explain.plugin("lifecycle")[1]
    assert re.search(r"NO completion_payload_schema\s+→\s+NO signal_completion",
                     text), "the opt-in gate is not stated"
