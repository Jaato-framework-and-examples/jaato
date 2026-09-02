"""Answering a permission prompt: one answer, and options the user can read.

Three behaviours, each with a failure that was observed live rather than
imagined:

* **One answer per prompt.**  Selection is reachable from Enter and from
  space, and the client stays in permission mode until the server replies.
  Two space keypresses 59ms apart sent two responses for the same request
  id; the server rejects the second with ``StateError: Unknown permission
  request``, and if focus moved in between it answers something the user
  never chose.

* **The options hint always renders.**  It used to sit inside
  ``if tool.permission_content:``, and that field is populated only by an
  ``AgentOutputEvent(source="permission")`` the server does not reliably
  emit — so the prompt showed a bare "Permission required" with no
  indication of what to press.

* **No completion float.**  The options were also offered as completions in
  a cursor-anchored ``Float``, which reserves no layout space and painted
  over the permission payload the user was being asked to approve.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pt_display import PTDisplay  # noqa: E402


def _options(shape="dict"):
    """The response options in one of the two representations in use."""
    raw = [("y", "yes"), ("n", "no"), ("a", "always")]
    if shape == "dict":
        return [{"key": s, "label": f} for s, f in raw]

    class Opt:
        def __init__(self, short, full):
            self.short, self.full = short, full
    return [Opt(s, f) for s, f in raw]


def _display(focus=0, shape="dict"):
    """A PTDisplay carrying only what the selection path touches."""
    d = object.__new__(PTDisplay)
    d._permission_response_options = _options(shape)
    d._permission_focus_index = focus
    d._comment_mode_active = False
    d._submitted = []
    d._input_callback = d._submitted.append
    return d


class TestOneAnswerPerPrompt:
    """A resolved prompt must not accept a second answer."""

    def test_first_selection_submits(self):
        d = _display()
        d._select_focused_permission_option()
        assert d._submitted == ["y"]

    def test_second_selection_is_inert(self):
        """The double-space case, measured at 59ms between keypresses."""
        d = _display()
        d._select_focused_permission_option()
        d._select_focused_permission_option()
        assert d._submitted == ["y"], "a resolved prompt accepted a second answer"

    def test_a_moved_focus_cannot_answer_after_the_fact(self):
        """The dangerous variant: the second submit chose a different option.

        One observed run sent ``y`` then ``a`` — whitelisting the tool for
        the whole session without the user ever selecting that.
        """
        d = _display()
        d._select_focused_permission_option()
        d._permission_focus_index = 2          # focus moves to "always"
        d._select_focused_permission_option()
        assert d._submitted == ["y"]

    def test_options_are_cleared_so_every_path_disarms(self):
        """Enter, space and tab all gate on _permission_response_options."""
        d = _display()
        d._select_focused_permission_option()
        assert d._permission_response_options is None

    def test_a_new_prompt_rearms_selection(self):
        d = _display()
        d._select_focused_permission_option()
        d._permission_response_options = _options()   # next request arrives
        d._permission_focus_index = 1
        d._select_focused_permission_option()
        assert d._submitted == ["y", "n"]

    @pytest.mark.parametrize("shape", ["dict", "obj"])
    def test_both_option_shapes_submit_the_short_form(self, shape):
        d = _display(focus=1, shape=shape)
        d._select_focused_permission_option()
        assert d._submitted == ["n"]

    def test_comment_mode_still_requires_typed_text(self):
        d = _display()
        d._comment_mode_active = True
        d._select_focused_permission_option()
        assert d._submitted == []
        assert d._permission_response_options is not None


class TestOptionsHintRendering:
    """The hint line, and why it carries no ANSI."""

    def _buffer(self, options, focus=0, style=None):
        from output_buffer import OutputBuffer
        b = object.__new__(OutputBuffer)
        b._permission_response_options = options
        b._permission_focus_index = focus
        b._style = style or (lambda name, fallback: fallback)
        return b

    def _opts(self):
        return [{"key": "y", "label": "yes"}, {"key": "n", "label": "no"}]

    def test_no_escape_characters_are_emitted(self):
        """Raw ANSI made Rich count escape bytes as visible columns, pad on
        the inflated width, and leave the row ~69 columns short — putting the
        panel's right border in the middle of the line."""
        from rich.text import Text
        out = Text()
        self._buffer(self._opts())._append_focused_options(out)
        assert "\x1b" not in out.plain

    def test_width_equals_the_visible_text(self):
        """cell_len is what the panel pads against."""
        from rich.text import Text
        out = Text()
        self._buffer(self._opts())._append_focused_options(out)
        assert out.cell_len == len(out.plain)

    def test_every_option_label_is_present(self):
        from rich.text import Text
        out = Text()
        self._buffer(self._opts())._append_focused_options(out)
        assert "[yes]" in out.plain and "[no]" in out.plain

    def test_the_navigation_hint_is_shown(self):
        """Selection is keyboard-only, so the keys must be on screen."""
        from rich.text import Text
        out = Text()
        self._buffer(self._opts())._append_focused_options(out)
        assert "cycle" in out.plain and "select" in out.plain

    def test_focus_is_styled_through_the_theme(self):
        from rich.text import Text
        out = Text()
        b = self._buffer(self._opts(), focus=1, style=lambda name, fb: name)
        b._append_focused_options(out)
        styles = [str(sp.style) for sp in out.spans]
        assert "permission_bar_focused" in styles
        assert "permission_bar_option" in styles


class TestHintRendersWithoutContent:
    """The hint must not depend on permission_content.

    It used to be nested inside ``if tool.permission_content:``.  That field
    is filled only by an ``AgentOutputEvent(source="permission")``, and in
    every session measured the server emitted none — so the branch never ran
    and the prompt showed a bare "Permission required" with no options and no
    way to know what to press.
    """

    class _Tool:
        def __init__(self, content=None):
            self.permission_content = content

    def _buffer(self):
        from output_buffer import OutputBuffer
        b = object.__new__(OutputBuffer)
        b._permission_response_options = [
            {"key": "y", "label": "yes", "description": "Allow this tool execution"},
            {"key": "n", "label": "no", "description": "Deny this tool execution"},
        ]
        b._permission_focus_index = 0
        b._style = lambda name, fallback: fallback
        return b

    def test_options_render_with_no_content_at_all(self):
        from rich.text import Text
        out = Text()
        self._buffer()._render_permission_prompt(out, self._Tool(None), is_last=True)
        assert "[yes]" in out.plain and "[no]" in out.plain

    def test_description_renders_with_no_content_at_all(self):
        from rich.text import Text
        out = Text()
        self._buffer()._render_permission_prompt(out, self._Tool(None), is_last=True)
        assert "Allow this tool execution" in out.plain

    def test_header_is_still_shown(self):
        from rich.text import Text
        out = Text()
        self._buffer()._render_permission_prompt(out, self._Tool(None), is_last=True)
        assert "Permission required" in out.plain

    def test_empty_string_content_is_treated_as_absent(self):
        from rich.text import Text
        out = Text()
        self._buffer()._render_permission_prompt(out, self._Tool(""), is_last=True)
        assert "[yes]" in out.plain


class TestFocusedOptionDescription:
    """The gloss that replaced the completion menu's description column."""

    def _buffer(self, options, focus=0):
        from output_buffer import OutputBuffer
        b = object.__new__(OutputBuffer)
        b._permission_response_options = options
        b._permission_focus_index = focus
        return b

    def _described(self):
        return [
            {"key": "y", "label": "yes", "description": "Allow this tool execution"},
            {"key": "i", "label": "idle", "description": "Allow until session goes idle"},
        ]

    def test_describes_the_focused_option(self):
        b = self._buffer(self._described(), focus=1)
        assert b._render_focused_option_description() == "Allow until session goes idle"

    def test_follows_the_focus_index(self):
        b = self._buffer(self._described(), focus=0)
        assert b._render_focused_option_description() == "Allow this tool execution"

    def test_object_shaped_options_are_described_too(self):
        class Opt:
            def __init__(self):
                self.short, self.full = "n", "no"
                self.description = "Deny this tool execution"
        assert self._buffer([Opt()])._render_focused_option_description() == \
            "Deny this tool execution"

    def test_no_options_means_no_line(self):
        assert self._buffer(None)._render_focused_option_description() == ""

    def test_missing_description_means_no_line(self):
        b = self._buffer([{"key": "y", "label": "yes"}])
        assert b._render_focused_option_description() == ""

    def test_out_of_range_focus_is_not_an_index_error(self):
        """A focus index can outlive a shorter option list across prompts."""
        b = self._buffer(self._described(), focus=7)
        assert b._render_focused_option_description() == ""


class TestNoCompletionFloat:
    """Permission mode must offer no completions at all."""

    def _completer(self):
        from file_completer import CombinedCompleter
        return CombinedCompleter(commands=[("help", "Show help")])

    def test_permission_mode_yields_no_completions(self):
        from prompt_toolkit.document import Document
        c = self._completer()
        c.set_permission_mode(True, options=_options("obj"))
        assert list(c.get_completions(Document("y"), None)) == []

    def test_empty_input_in_permission_mode_yields_nothing_either(self):
        from prompt_toolkit.document import Document
        c = self._completer()
        c.set_permission_mode(True, options=_options("obj"))
        assert list(c.get_completions(Document(""), None)) == []

    def test_leaving_permission_mode_restores_normal_completion(self):
        """The suppression is scoped to permission mode, not global."""
        from prompt_toolkit.document import Document
        c = self._completer()
        c.set_permission_mode(True, options=_options("obj"))
        c.set_permission_mode(False)
        completions = list(c.get_completions(Document("hel"), None))
        assert any("help" in x.text for x in completions)


class TestFeedbackOptionsUnlockTyping:
    """Both feedback options must unlock typing, and submit their own prefix.

    Two decisions carry free text: ``c``/``comment`` (deny with feedback,
    ``ChannelDecision.COMMENT``) and ``yc``/``allow-comment`` (allow with
    feedback, ``ChannelDecision.ALLOW_COMMENT``).  Recognising only the
    first left ``allow-comment`` focusable but untypeable, so the feedback
    it exists to collect could not be entered.

    The prefix matters as much as the unlock: the server parses ``yc:`` and
    ``c:`` as different decisions, so submitting a hardcoded ``c:`` for an
    allow-comment selection DENIED the tool the user had chosen to allow.
    """

    class _Obj:
        def __init__(self, short, full, decision=None):
            self.short, self.full = short, full
            if decision is not None:
                self.decision = decision

    @pytest.mark.parametrize("option", [
        {"key": "c", "label": "comment", "action": "comment"},
        {"key": "yc", "label": "allow-comment", "action": "allow_comment"},
        {"key": "c", "label": "comment"},              # no action field
        {"key": "yc", "label": "allow-comment"},       # no action field
    ], ids=["comment-action", "allow-comment-action",
            "comment-key-only", "allow-comment-key-only"])
    def test_dict_feedback_options_are_recognised(self, option):
        assert PTDisplay._is_comment_option(option) is True

    @pytest.mark.parametrize("short,decision", [
        ("c", "comment"), ("yc", "allow_comment"), ("c", None), ("yc", None),
    ])
    def test_object_feedback_options_are_recognised(self, short, decision):
        opt = self._Obj(short, "x", decision)
        assert PTDisplay._is_comment_option(opt) is True

    @pytest.mark.parametrize("option", [
        {"key": "y", "label": "yes", "action": "allow"},
        {"key": "n", "label": "no", "action": "deny"},
        {"key": "a", "label": "always", "action": "allow_session"},
    ])
    def test_plain_options_are_not_feedback_options(self, option):
        assert PTDisplay._is_comment_option(option) is False

    def test_prefix_follows_the_focused_option(self):
        """yc must submit "yc:", not "c:" — they are opposite decisions."""
        d = object.__new__(PTDisplay)
        d._permission_response_options = [
            {"key": "c", "label": "comment"},
            {"key": "yc", "label": "allow-comment"},
        ]
        d._permission_focus_index = 1
        assert d._focused_option_short() == "yc"
        d._permission_focus_index = 0
        assert d._focused_option_short() == "c"

    def test_prefix_is_empty_when_there_is_nothing_focused(self):
        """Empty must mean submit nothing, never fall back to a guess."""
        d = object.__new__(PTDisplay)
        d._permission_response_options = None
        d._permission_focus_index = 0
        assert d._focused_option_short() == ""

    def test_prefix_is_empty_on_an_out_of_range_focus(self):
        d = object.__new__(PTDisplay)
        d._permission_response_options = [{"key": "c", "label": "comment"}]
        d._permission_focus_index = 5
        assert d._focused_option_short() == ""
