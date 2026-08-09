"""askPermission grants ONE call, not the whole tool for the session.

Regression cover for a privilege escalation: ``_execute_ask_permission`` used
to call ``add_session_whitelist(tool_name)`` on ANY approval, including one
that came from a per-COMMAND whitelist pattern.  A policy of
``defaultPolicy=deny`` + ``whitelist.patterns=['git *']`` therefore denied
``rm -rf /tmp`` on a fresh plugin but ALLOWED it once ``git status`` had been
approved -- the pattern grant was escalated to a whole-tool session grant and
every later command bypassed the pattern.

The line existed to stop the user being prompted twice for one decision (once
for the model's askPermission pre-check, once for the real execution).  That
is now a single-use grant keyed to the exact call.
"""
from shared.plugins.permission.channels import ChannelDecision, ChannelResponse
from shared.plugins.permission.plugin import create_plugin

_PATTERN_POLICY = {
    "policy": {"defaultPolicy": "deny", "whitelist": {"patterns": ["git *"]}},
}


def _ask(plugin, command):
    return plugin.get_executors()["askPermission"]({
        "tool_name": "cli_based_tool",
        "intent": "exercise the permission path",
        "arguments": {"command": command},
    })


def test_pattern_approval_does_not_whitelist_the_whole_tool():
    """THE escalation: one `git *` match must not unlock every command."""
    plugin = create_plugin()
    plugin.initialize(_PATTERN_POLICY)

    assert _ask(plugin, "git status")["allowed"] is True

    denied = _ask(plugin, "rm -rf /tmp")
    assert denied["allowed"] is False
    assert denied["method"] == "default"


def test_denial_is_not_order_dependent():
    """Same verdict whether or not an allowed command ran first."""
    fresh = create_plugin()
    fresh.initialize(_PATTERN_POLICY)
    first = _ask(fresh, "rm -rf /tmp")

    after = create_plugin()
    after.initialize(_PATTERN_POLICY)
    _ask(after, "git status")
    second = _ask(after, "rm -rf /tmp")

    assert first["allowed"] == second["allowed"] is False


def test_pattern_still_allows_matching_commands():
    """The fix must not break the whitelist it is protecting."""
    plugin = create_plugin()
    plugin.initialize(_PATTERN_POLICY)
    assert _ask(plugin, "git log --oneline")["allowed"] is True


def test_explicit_session_grant_still_works():
    """`permissions allow <tool>` is the SUPPORTED way to grant a session."""
    plugin = create_plugin()
    plugin.initialize(_PATTERN_POLICY)
    plugin._permissions_allow("cli_based_tool")
    allowed, _ = plugin.check_permission("cli_based_tool", {"command": "rm -rf /tmp"})
    assert allowed is True


class _CountingChannel:
    """Channel that approves ALLOW_ONCE and counts how often it was asked."""

    name = "counting"

    def __init__(self):
        self.prompts = []

    def initialize(self, config):
        pass

    def shutdown(self):
        pass

    def request_permission(self, request):
        self.prompts.append(request.tool_name)
        return ChannelResponse(
            request_id=request.request_id,
            decision=ChannelDecision.ALLOW_ONCE,
            reason="User chose: yes",
        )


def _asking_plugin():
    plugin = create_plugin()
    plugin.initialize({"policy": {"defaultPolicy": "ask"}})
    channel = _CountingChannel()
    plugin._channel = channel
    return plugin, channel


def test_interactive_approval_does_not_prompt_twice():
    """The behaviour the removed line existed for, preserved."""
    plugin, channel = _asking_plugin()
    args = {"command": "ls"}

    assert _ask_tool(plugin, args)["method"] == "user_approved"
    assert len(channel.prompts) == 1

    allowed, meta = plugin.check_permission("cli", args)
    assert allowed is True
    assert meta["method"] == "ask_permission_once"
    assert len(channel.prompts) == 1, "real execution must not re-prompt"


def test_grant_is_single_use():
    """A second execution of the same call asks again."""
    plugin, channel = _asking_plugin()
    args = {"command": "ls"}
    _ask_tool(plugin, args)
    plugin.check_permission("cli", args)          # consumes the grant

    plugin.check_permission("cli", args)
    assert len(channel.prompts) == 2


def test_grant_does_not_cover_different_arguments():
    """Pre-approving `ls` must not pre-approve `rm -rf /`."""
    plugin, channel = _asking_plugin()
    _ask_tool(plugin, {"command": "ls"})

    plugin.check_permission("cli", {"command": "rm -rf /"})
    assert len(channel.prompts) == 2, "different args must be asked separately"


def _ask_tool(plugin, args):
    return plugin.get_executors()["askPermission"]({
        "tool_name": "cli", "intent": "list files", "arguments": args,
    })
