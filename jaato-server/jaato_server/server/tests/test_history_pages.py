"""Paged, rendered history replay (protocol 1.28).

Covers the three properties the feature exists for:

* pages are cut from the END, walk older by cursor, and never split a
  renderable unit (fence, table, quoted notebook markup, one message's tool
  calls);
* model text goes through the formatter the live stream uses -- both on the
  paged path and on the full ``emit_current_state`` replay;
* a cursor the history no longer contains is reported ``stale``, never
  silently resolved to another page.
"""

from types import SimpleNamespace

import pytest

from jaato_server.server import history_pages as hp
from jaato_server.server.history_pages import (
    build_units, paginate, resolve_cursor, segment_text,
)
from jaato_sdk.plugins.model_provider.types import (
    FunctionCall, Message, Part, Role, ToolResult,
)


def _user(text):
    return Message(role=Role.USER, parts=[Part(text=text)])


def _model(*parts):
    return Message(role=Role.MODEL, parts=list(parts))


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------

FENCED = (
    "Intro paragraph.\n"
    "\n"
    "```python\n"
    "def f():\n"
    "\n"
    "    return 1\n"
    "```\n"
    "\n"
    "| a | b |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    "\n"
    "Outro.\n"
)


def test_segments_round_trip_and_keep_fences_and_tables_whole():
    segs = segment_text(FENCED)
    assert "".join(segs) == FENCED
    assert len(segs) == 4
    fence = [s for s in segs if s.startswith("```python")]
    assert len(fence) == 1 and "\n\n    return 1\n```" in fence[0]
    table = [s for s in segs if s.startswith("| a |")]
    assert len(table) == 1 and "| 1 | 2 |" in table[0]


def test_blank_line_inside_quoted_notebook_markup_does_not_split():
    text = ('<notebook-cell type="output" exec="1">\nline one\n\n'
            'line two\n</notebook-cell>\n\nafter\n')
    segs = segment_text(text)
    assert len(segs) == 2
    assert segs[0].rstrip().endswith("</notebook-cell>")


def test_unterminated_fence_keeps_the_rest_in_one_segment():
    text = "a\n\n```\ncode\n\nmore code\n"
    assert segment_text(text) == ["a\n\n", "```\ncode\n\nmore code\n"]


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def _history():
    return [
        _user("first question"),
        _model(Part(thought="let me think"), Part(text=FENCED)),
        _user("run it"),
        _model(Part(text="Running."),
               Part(function_call=FunctionCall(id="c1", name="cli_based_tool",
                                               args={"command": "ls"})),
               Part(function_call=FunctionCall(id="c2", name="readFile",
                                               args={"path": "x"}))),
        Message(role=Role.TOOL, parts=[
            Part(function_response=ToolResult(call_id="c1", name="cli_based_tool",
                                              result="ok")),
            Part(function_response=ToolResult(call_id="c2", name="readFile",
                                              result="nope", is_error=True)),
        ]),
        _model(Part(text="Done.")),
    ]


def test_units_kinds_turns_groups_and_tool_outcomes():
    units = build_units(_history())
    kinds = [u.kind for u in units]
    assert kinds == ["user", "thinking", "model", "model", "model", "model",
                     "user", "model", "tools", "model"]
    assert [u.turn for u in units] == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    fenced_groups = {u.group for u in units[2:6]}
    assert len(fenced_groups) == 1  # one text part, four segments
    tools = units[8].tools
    assert [t["success"] for t in tools] == [True, False]
    assert tools[0]["tool_args"] == {"command": "ls"}


def test_formatter_is_applied_to_model_text_only_and_a_raising_one_keeps_text():
    seen = []

    def fmt(seg):
        seen.append(seg)
        if "Outro" in seg:
            raise RuntimeError("boom")
        return f"<fmt>{seg}</fmt>"

    units = build_units(_history(), fmt)
    assert all(s for s in seen)
    by_kind = {}
    for u in units:
        by_kind.setdefault(u.kind, u)
    assert by_kind["user"].text == "first question"
    assert by_kind["thinking"].text == "let me think"
    outro = [u for u in units if "Outro" in u.raw][0]
    assert outro.text == outro.raw  # formatter raised: text kept raw
    running = [u for u in units if u.raw == "Running."][0]
    assert running.text == "<fmt>Running.</fmt>"


def test_digest_is_over_raw_content_not_formatted_text():
    a = build_units(_history(), lambda s: s.upper())
    b = build_units(_history(), None)
    assert [u.digest for u in a] == [u.digest for u in b]


def test_hidden_content_is_stripped_and_an_empty_user_turn_dropped():
    units = build_units([_user("<hidden>internal</hidden>"), _user("x <hidden>y</hidden>")])
    assert [u.text for u in units] == ["x "]


# ---------------------------------------------------------------------------
# paging
# ---------------------------------------------------------------------------

def _line_units(n, lines=10):
    return [hp.ReplayUnit(kind="model", raw=f"u{i}", text=f"u{i}", lines=lines)
            for i in range(n)]


def test_latest_page_is_the_end_and_cursors_walk_older_to_the_start():
    units = _line_units(25)
    seen = []
    before = ""
    for _ in range(10):
        page = paginate(units, before=before, max_lines=40)
        seen = [i for i, _ in page.units] + seen
        if not page.has_more:
            break
        before = page.before
    assert seen == list(range(25))
    first = paginate(units, max_lines=40)
    assert [i for i, _ in first.units] == [21, 22, 23, 24]
    assert first.total == 25


def test_a_unit_taller_than_the_budget_is_a_page_by_itself():
    units = _line_units(3, lines=10)
    units[1].lines = 500
    p1 = paginate(units, max_lines=40)
    assert [i for i, _ in p1.units] == [2]
    p2 = paginate(units, before=p1.before, max_lines=40)
    assert [i for i, _ in p2.units] == [1]
    p3 = paginate(units, before=p2.before, max_lines=40)
    assert [i for i, _ in p3.units] == [0] and not p3.has_more


def test_budget_is_clamped():
    assert hp.clamp_page_lines(0) == hp.DEFAULT_PAGE_LINES
    assert hp.clamp_page_lines("junk") == hp.DEFAULT_PAGE_LINES
    assert hp.clamp_page_lines(10 ** 9) == hp.MAX_PAGE_LINES


def test_cursor_survives_gc_dropping_the_front_of_the_history():
    units = _line_units(10)
    page = paginate(units, max_lines=30)
    shifted = units[3:]  # GC dropped three units from the front
    idx = resolve_cursor(shifted, page.before)
    assert shifted[idx].raw == units[int(page.before.split(":")[0])].raw


def test_a_cursor_the_history_no_longer_contains_is_stale():
    units = _line_units(10)
    page = paginate(units, max_lines=30)
    rewritten = _line_units(2)
    for u in rewritten:
        u.raw = "other " + u.raw
    stale = paginate(rewritten, before=page.before)
    assert stale.stale and stale.units == []
    assert paginate(units, before="garbage").stale


def test_to_dict_shapes():
    units = build_units(_history())
    d_tools = units[8].to_dict(8)
    assert "text" not in d_tools and len(d_tools["tools"]) == 2
    d_text = units[0].to_dict(0)
    assert d_text["id"] == units[0].cursor(0) and d_text["text"] == "first question"


# ---------------------------------------------------------------------------
# JaatoServer integration
# ---------------------------------------------------------------------------

def _server(history, presentation=None):
    from jaato_server.server.core import JaatoServer

    srv = JaatoServer.__new__(JaatoServer)
    srv._agents = {"main": SimpleNamespace(agent_id="main")}
    srv._presentation_context = presentation
    srv._formatter_pipeline = None
    srv.get_history = lambda agent_id: history
    srv._format_replay_text = lambda text: f"[F]{text}"
    return srv


def test_history_page_event_carries_formatted_units_and_request_id():
    srv = _server(_history())
    ev = srv.history_page("main", max_lines=5, request_id="r1")
    assert ev.request_id == "r1" and ev.ok and ev.has_more
    assert ev.units[-1]["text"] == "[F]Done."
    older = srv.history_page("main", before=ev.before, max_lines=5)
    assert older.units[-1]["id"] != ev.units[0]["id"]


def test_history_page_answers_not_ok_when_the_history_cannot_be_read():
    srv = _server([])

    def boom(agent_id):
        raise RuntimeError("runner gone")
    srv.get_history = boom
    ev = srv.history_page("main", request_id="r2")
    assert not ev.ok and "runner gone" in ev.error and ev.request_id == "r2"


def test_full_replay_goes_through_the_formatter_and_joins_a_part():
    from jaato_sdk.events import AgentOutputEvent, ToolCallEndEvent
    events = []
    _server(_history())._emit_conversation_replay(events.append)
    model = [e for e in events if isinstance(e, AgentOutputEvent) and e.source == "model"]
    assert all(e.text.startswith("[F]") for e in model)
    # the fenced reply is ONE part in four segments: write, then appends
    assert [e.mode for e in model[:4]] == ["write", "append", "append", "append"]
    ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
    assert [e.success for e in ends] == [True, False]


@pytest.mark.parametrize("pres,expected", [
    (None, "full"),
    ({"client_type": "chat"}, "none"),
    ({"client_type": "chat", "history_replay": "paged"}, "paged"),
    ({"client_type": "terminal", "history_replay": "none"}, "none"),
    ({"client_type": "web", "history_replay": "bogus"}, "full"),
])
def test_history_replay_mode(pres, expected):
    from jaato_sdk.events import PresentationContext
    ctx = PresentationContext.from_dict(pres) if pres is not None else None
    assert _server([], ctx).history_replay_mode() == expected


def test_paged_attach_sends_one_page_instead_of_the_full_replay():
    from jaato_sdk.events import HistoryPageEvent, PresentationContext
    ctx = PresentationContext.from_dict(
        {"client_type": "chat", "history_replay": "paged", "history_page_lines": 3})
    events = []
    _server(_history(), ctx)._emit_history_on_attach(events.append)
    assert len(events) == 1 and isinstance(events[0], HistoryPageEvent)
    assert events[0].request_id == "" and events[0].has_more


def test_router_answers_a_page_request_without_a_session():
    from jaato_sdk.events import HistoryPageEvent, HistoryPageRequest
    from jaato_server.server.command_router import CommandRouter

    sent = []
    router = CommandRouter.__new__(CommandRouter)
    router._session_manager = SimpleNamespace(get_client_session=lambda cid: None)
    router._event_sink = SimpleNamespace(send_event=lambda cid, ev: sent.append(ev))
    router._handle_history_page_request("c1", HistoryPageRequest(request_id="q"))
    assert isinstance(sent[0], HistoryPageEvent)
    assert not sent[0].ok and sent[0].request_id == "q"


def test_the_real_default_pipeline_renders_fences_and_tables_per_unit():
    """Against the shipped formatters, not a stub: the thing a replay lost."""
    from jaato_server.shared.plugins.formatter_pipeline import create_registry
    registry = create_registry()
    registry.discover()
    registry.use_defaults()
    pipeline = registry.create_pipeline(80)

    def fmt(seg):
        try:
            return "".join(pipeline.process_chunk(seg)) + "".join(pipeline.flush())
        finally:
            pipeline.reset()

    units = build_units([_model(Part(text=FENCED))], fmt)
    rendered = [u.text for u in units]
    assert any(t.startswith("<j-code") and "</j-code>" in t for t in rendered)
    assert any(t.startswith("<j-table>") and "</j-table>" in t for t in rendered)
    assert not any("```" in t for t in rendered)


def test_standalone_ws_mode_answers_a_page_request():
    """WS clients (the web coder) reach the verb in BOTH WS modes: daemon mode
    goes through the CommandRouter (covered above); standalone has its own
    per-type dispatch, which must know the verb too."""
    import asyncio
    from jaato_sdk.events import HistoryPageRequest
    from jaato_server.server.websocket import JaatoWSServer

    sent = []
    ws = JaatoWSServer.__new__(JaatoWSServer)
    ws._jaato_server = _server(_history())

    async def fake_send(cid, ev):
        sent.append(ev)
    ws._send_to_client = fake_send
    asyncio.run(ws._handle_message_standalone(
        "c1", HistoryPageRequest(request_id="w", max_lines=5)))
    assert sent and sent[0].request_id == "w" and sent[0].units
