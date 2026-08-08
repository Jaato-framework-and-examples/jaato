"""Integration test for the rewind-with-hint flow.

Exercises the SessionHistory rewrite helper end-to-end with realistic
message shapes.  The full JaatoSession wiring is covered by the
detector's unit tests plus the existing session test suite — this
module focuses on the two behaviors most likely to regress:

1. ``rewrite_last_dropping_tool_use`` preserves text parts and drops
   function_call parts from the most recent assistant message,
   maintaining message_id continuity so GC and ledger see a coherent
   turn.
2. The hint-text builder (``JaatoSession._build_rewind_hint_text``)
   produces a human-readable user-role message that references the
   specific tool, the detection reason, and the preserved narration.
"""

from jaato_sdk.plugins.model_provider.types import (
    FunctionCall,
    Message,
    Part,
    Role,
)

from shared.session_history import SessionHistory


class TestRewriteLastDroppingToolUse:
    """Covers the history-surgery side of the rewind flow."""

    def _msg(self, parts, role=Role.MODEL, message_id="m1"):
        return Message(
            role=role, parts=parts, message_id=message_id,
            model="test-model", provider="test-provider",
        )

    def test_drops_function_call_keeps_text(self):
        history = SessionHistory()
        narration = Part(text="I am about to write a large file.")
        bad_call = Part(function_call=FunctionCall(
            id="c1", name="writeNewFile", args={},
        ))
        history.append(self._msg([narration, bad_call]))

        preserved = history.rewrite_last_dropping_tool_use()
        assert preserved == "I am about to write a large file."

        last = history.last
        assert last is not None
        assert len(last.parts) == 1
        assert last.parts[0].text == "I am about to write a large file."
        assert not any(p.function_call for p in last.parts)

    def test_preserves_message_id_and_provenance(self):
        """GC/ledger consumers key off message_id, model, provider —
        the rewrite must not regenerate those fields."""
        history = SessionHistory()
        history.append(self._msg(
            parts=[
                Part(text="narration"),
                Part(function_call=FunctionCall(id="c1", name="x", args={})),
            ],
            message_id="stable-id-42",
        ))

        history.rewrite_last_dropping_tool_use()

        last = history.last
        assert last.message_id == "stable-id-42"
        assert last.model == "test-model"
        assert last.provider == "test-provider"

    def test_multiple_text_parts_concatenated(self):
        history = SessionHistory()
        history.append(self._msg([
            Part(text="First thought. "),
            Part(function_call=FunctionCall(id="c1", name="x", args={})),
            Part(text="Second thought."),
        ]))

        preserved = history.rewrite_last_dropping_tool_use()
        # ``Message.text`` concatenates all text parts; preserved returns
        # that joined string.
        assert preserved == "First thought. Second thought."

    def test_skips_when_no_narration(self):
        """Function-call-only assistant turn can't anchor a hint —
        signal the caller to fall through to normal error handling."""
        history = SessionHistory()
        history.append(self._msg([
            Part(function_call=FunctionCall(id="c1", name="x", args={})),
        ]))

        assert history.rewrite_last_dropping_tool_use() is None
        # History left untouched
        assert len(history) == 1
        assert any(p.function_call for p in history.last.parts)

    def test_skips_when_last_message_not_model(self):
        """Defensive: if history ends in a user or tool message,
        something upstream is out of order — don't mutate."""
        history = SessionHistory()
        history.append(Message.from_text(Role.USER, "hello"))

        assert history.rewrite_last_dropping_tool_use() is None
        assert history.last.role == Role.USER

    def test_skips_on_empty_history(self):
        history = SessionHistory()
        assert history.rewrite_last_dropping_tool_use() is None

    def test_dirty_flag_set_on_rewrite(self):
        history = SessionHistory()
        history.append(self._msg([
            Part(text="narration"),
            Part(function_call=FunctionCall(id="c1", name="x", args={})),
        ]))
        # Reset dirty to confirm the rewrite itself sets it
        history._dirty = False

        history.rewrite_last_dropping_tool_use()
        assert history.dirty is True


class TestHintTextBuilder:
    """Verifies the shape of the synthetic user-role hint.

    Uses a minimal JaatoSession stand-in (the builder doesn't need
    runtime/registry/etc.) — we just want to confirm the output text
    references the tool name, reason, and narration.
    """

    def _build_hint(self, tool: str, reason: str, narration: str) -> str:
        # Import inside to avoid pulling the full session module at
        # collection time.
        from shared.jaato_session import JaatoSession

        # Bypass __init__ via object.__new__ — the builder doesn't touch
        # any instance state.  Keeps the test free of session fixtures.
        session = object.__new__(JaatoSession)
        return session._build_rewind_hint_text(tool, reason, narration)

    def test_hint_names_the_tool(self):
        hint = self._build_hint(
            "writeNewFile", "max_tokens_empty_args",
            "I'll write a large doc now."
        )
        assert "writeNewFile" in hint

    def test_hint_quotes_preserved_narration(self):
        hint = self._build_hint(
            "writeNewFile", "max_tokens_empty_args",
            "I'll write a large doc now."
        )
        assert "I'll write a large doc now." in hint

    def test_hint_empty_args_explanation(self):
        hint = self._build_hint(
            "writeNewFile", "max_tokens_empty_args", "narration"
        )
        assert "empty arguments" in hint.lower()

    def test_hint_missing_required_explanation(self):
        hint = self._build_hint(
            "writeNewFile", "max_tokens_missing_required", "narration"
        )
        assert "missing" in hint.lower()

    def test_hint_unknown_reason_falls_back(self):
        """Builder is conservative — unknown reason strings still
        produce a valid message; they just use a generic explanation."""
        hint = self._build_hint("writeNewFile", "not_a_known_reason", "n")
        assert "writeNewFile" in hint
        assert "truncated" in hint.lower()

    def test_hint_truncates_long_narration(self):
        """Hint stays compact — the model has its own narration in the
        preceding turn anyway; we just need an anchor."""
        long_narration = "x" * 500
        hint = self._build_hint(
            "writeNewFile", "max_tokens_empty_args", long_narration
        )
        # Truncated to 200 chars + ellipsis
        assert "xxx…" in hint or "x…" in hint
        # And shouldn't contain the full 500-char blob
        assert long_narration not in hint

    def test_hint_suggests_chunking_recipe(self):
        hint = self._build_hint(
            "writeNewFile", "max_tokens_empty_args", "n"
        )
        # Some recipe-like phrasing should appear
        assert "skeleton" in hint.lower() or "sections" in hint.lower()
