"""Tests for FileReferenceProcessor.expand_references — specifically
the rule that ``@`` is stripped only when the reference resolves to an
existing file or directory.

Regression: prior to 2026-04-25 the processor stripped ``@`` from every
matched reference regardless of whether the file existed.  That made
typing or pasting any non-file ``@`` impossible — ``@username``,
``@deprecated``, email addresses (the partial match after the first
``@``) all silently lost their ``@`` on submit.  This file locks in
the corrected behaviour.
"""

import os
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location

_TUI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TUI_DIR not in sys.path:
    sys.path.insert(0, _TUI_DIR)

_spec = spec_from_file_location(
    "file_completer", os.path.join(_TUI_DIR, "file_completer.py")
)
_fc = module_from_spec(_spec)
_spec.loader.exec_module(_fc)

FileReferenceProcessor = _fc.FileReferenceProcessor


class TestExpandReferencesPreservesAt:
    """The user-facing fix: typing ``@<not-a-file>`` must round-trip."""

    def _processor(self, base_path):
        return FileReferenceProcessor(base_path=base_path, include_contents=False)

    def test_at_handle_preserved_when_no_file_match(self, tmp_path):
        """``@username`` where ``username`` isn't a file → text unchanged."""
        proc = self._processor(str(tmp_path))
        out = proc.expand_references("hi @username how are you")
        assert out == "hi @username how are you"

    def test_at_decorator_style_preserved(self, tmp_path):
        """Common Python idiom: ``@deprecated`` should survive even when
        no file by that name exists in the workspace."""
        proc = self._processor(str(tmp_path))
        out = proc.expand_references("note that this method is @deprecated now")
        assert "@deprecated" in out

    def test_email_address_preserved(self, tmp_path):
        """Pasting ``user@example.com`` — the regex matches ``@example.com``
        but ``example.com`` resolves to nothing under tmp_path, so the
        whole email string must come through unchanged.  Without the
        fix the input would arrive at the model as ``userexample.com``."""
        proc = self._processor(str(tmp_path))
        out = proc.expand_references("ping daniel@example.com please")
        assert out == "ping daniel@example.com please"

    def test_existing_file_strips_at_and_inlines(self, tmp_path):
        """Sanity: the happy path is unchanged — when the file exists
        the ``@`` is stripped from the prompt and the contents land in
        the Referenced Files section."""
        f = tmp_path / "hello.py"
        f.write_text("print('hi')\n")
        proc = FileReferenceProcessor(base_path=str(tmp_path), include_contents=True)
        out = proc.expand_references("review @hello.py")

        # @ stripped from inline mention.
        assert "@hello.py" not in out
        assert "review hello.py" in out
        # Contents section appended.
        assert "--- Referenced Files ---" in out
        assert "print('hi')" in out

    def test_mixed_existing_and_non_existing(self, tmp_path):
        """When some refs resolve and others don't: existing ones get
        their @ stripped + content inlined; non-existing ones keep
        their @ as the user wrote it."""
        f = tmp_path / "real.py"
        f.write_text("ok\n")
        proc = FileReferenceProcessor(base_path=str(tmp_path), include_contents=True)
        out = proc.expand_references("compare @real.py with @ghost.py")

        # Existing reference: @ stripped.
        assert "@real.py" not in out
        # Non-existing reference: @ preserved.
        assert "@ghost.py" in out
        # Contents section only mentions the real one.
        assert "--- Referenced Files ---" in out
        assert "[real.py]" in out
        assert "[ghost.py]" not in out

    def test_text_without_at_returns_unchanged(self, tmp_path):
        """No @ in input → no work to do, identity output."""
        proc = self._processor(str(tmp_path))
        text = "just regular conversation, no markup at all"
        assert proc.expand_references(text) == text

    def test_at_at_start_of_message_preserved(self, tmp_path):
        """Leading ``@`` is the most common file-reference position;
        confirm non-existent leading ``@`` still preserves the literal."""
        proc = self._processor(str(tmp_path))
        out = proc.expand_references("@nobody come help me")
        assert out == "@nobody come help me"
