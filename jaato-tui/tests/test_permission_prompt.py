"""A ``PermissionRequestedEvent`` renders its content in the TUI.

The defect these cover: on a runner-served session (the default) the
permission ASK arrives as a ``PermissionRequestedEvent`` carrying the
prompt lines and warnings, followed by the input-mode event, and no
``AgentOutputEvent`` is emitted.  The TUI rendered only the latter shape,
so the person saw "Permission required" and the options bar with no diff
and no warning between them.
"""

from jaato_sdk.events import PermissionRequestedEvent

import permission_prompt as pp


class FakeBuffer:
    def __init__(self):
        self.appended = []

    def append(self, source, text, mode):
        self.appended.append((source, text, mode))


class FakeRegistry:
    def __init__(self, buffers=None, selected=None):
        self.buffers = buffers or {}
        self.selected = selected

    def get_buffer(self, agent_id):
        return self.buffers.get(agent_id)

    def get_selected_buffer(self):
        return self.selected


def test_content_is_the_block_the_daemon_local_hook_emits():
    text = pp.permission_prompt_content(
        ["Update file: src/app.py", "-print('hello')", "+print('hi')"],
        warnings="The path is outside the sandbox allowlist.",
        warning_level="warning",
    )
    assert text == (
        '<security-warning level="warning">\n'
        "The path is outside the sandbox allowlist.\n"
        "</security-warning>\n"
        "\n"
        "Update file: src/app.py\n-print('hello')\n+print('hi')"
    )


def test_warning_level_defaults_to_warning_and_no_warning_means_no_block():
    assert pp.security_warning_block("careful", None).startswith('<security-warning level="warning">')
    assert pp.security_warning_block(None, "error") == ""
    assert pp.permission_prompt_content(["Tool: x"]) == "Tool: x"


def test_requested_event_is_appended_under_the_permission_source():
    """Same source the daemon-local ``AgentOutputEvent`` used, so
    ``set_tool_awaiting_approval`` attaches it to the tool exactly as
    before."""
    main = FakeBuffer()
    reg = FakeRegistry(buffers={"main": main})
    ev = PermissionRequestedEvent(agent_id="main", request_id="r1", tool_name="updateFile",
                                  prompt_lines=["Update file: a.py", "+x"], format_hint="diff")
    assert pp.render_permission_requested(ev, reg) is True
    assert main.appended == [("permission", "Update file: a.py\n+x", "write")]


def test_relayed_event_with_no_agent_id_lands_on_the_selected_buffer():
    selected = FakeBuffer()
    reg = FakeRegistry(selected=selected)
    ev = PermissionRequestedEvent(agent_id="", request_id="r1", tool_name="run", prompt_lines=["Tool: run"])
    assert pp.render_permission_requested(ev, reg) is True
    assert selected.appended[0][0] == "permission"


def test_an_event_with_nothing_to_show_appends_nothing():
    """The bare header the tool tree already renders beats an empty block."""
    main = FakeBuffer()
    ev = PermissionRequestedEvent(agent_id="main", request_id="r1", tool_name="run")
    assert pp.render_permission_requested(ev, FakeRegistry(buffers={"main": main})) is False
    assert main.appended == []


class FakeDisplay:
    def __init__(self):
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1


def test_loop_entry_point_ignores_other_events_and_refreshes_on_content():
    """Called for EVERY event the loop receives, so it must be inert for
    the rest and refresh only when it appended something."""
    from jaato_sdk.events import PermissionInputModeEvent
    main = FakeBuffer()
    reg = FakeRegistry(buffers={"main": main})
    display = FakeDisplay()
    assert pp.maybe_render_permission_requested(PermissionInputModeEvent(agent_id="main", request_id="r1"), reg, display) is False
    assert pp.maybe_render_permission_requested("not an event", reg, display) is False
    assert main.appended == [] and display.refreshed == 0
    ev = PermissionRequestedEvent(agent_id="main", request_id="r1", tool_name="run", prompt_lines=["Tool: run"])
    assert pp.maybe_render_permission_requested(ev, reg, display) is True
    assert display.refreshed == 1
