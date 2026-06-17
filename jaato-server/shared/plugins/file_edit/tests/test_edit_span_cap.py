"""Tests for the per-profile ``max_edit_span_chars`` targeted-edit span cap.

Covers the three surfaces of the knob:
- config parsing/validation (``_parse_edit_span_cap``)
- runtime enforcement at the ``updateFile`` executor + permission-preview sites
- the dynamic tool-description advertisement built in ``get_tool_schemas``

The cap exists to stop weak models (e.g. GPT-5-mini) from issuing
``updateFile`` targeted edits where ``old`` ≈ the whole file and ``new`` is the
whole rewrite — which can never anchor-match → 0 mutations.  It applies
symmetrically to ``old`` and ``new`` but never to ``new_content``
(full-replacement mode), so legitimate whole-file rewrites remain legal.
"""

import pytest

from ..plugin import FileEditPlugin


def _make_plugin(tmp_path, cap=None, allow_full_replace=None):
    """Initialize a plugin with an optional span cap / full-replace gate."""
    plugin = FileEditPlugin()
    config = {"backup_dir": str(tmp_path / "backups")}
    if cap is not None:
        config["max_edit_span_chars"] = cap
    if allow_full_replace is not None:
        config["allow_full_replace"] = allow_full_replace
    plugin.initialize(config)
    return plugin


class TestParseEditSpanCap:
    """Validation of the raw ``max_edit_span_chars`` config value."""

    def test_none_is_unlimited(self):
        assert FileEditPlugin._parse_edit_span_cap(None) is None

    def test_positive_int_passes_through(self):
        assert FileEditPlugin._parse_edit_span_cap(800) == 800

    def test_numeric_string_accepted(self):
        # Profile loaders may hand the value through as a string.
        assert FileEditPlugin._parse_edit_span_cap("1200") == 1200

    def test_zero_rejected_as_unlimited(self):
        assert FileEditPlugin._parse_edit_span_cap(0) is None

    def test_negative_rejected_as_unlimited(self):
        assert FileEditPlugin._parse_edit_span_cap(-5) is None

    def test_bool_rejected_as_unlimited(self):
        # bool is an int subclass; True/False as a span cap is always a mistake.
        assert FileEditPlugin._parse_edit_span_cap(True) is None
        assert FileEditPlugin._parse_edit_span_cap(False) is None

    def test_non_numeric_rejected_as_unlimited(self):
        assert FileEditPlugin._parse_edit_span_cap("abc") is None
        assert FileEditPlugin._parse_edit_span_cap([800]) is None

    def test_initialize_caches_validated_cap(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=500)
        assert plugin._max_edit_span_chars == 500

    def test_initialize_without_cap_is_unlimited(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        assert plugin._max_edit_span_chars is None


class TestCheckEditSpan:
    """Direct unit tests of the symmetric enforcement helper."""

    def test_unlimited_is_noop(self, tmp_path):
        plugin = _make_plugin(tmp_path)  # no cap
        assert plugin._check_edit_span("x" * 10_000, "y" * 10_000) is None

    def test_under_cap_passes(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=100)
        assert plugin._check_edit_span("short old", "short new") is None

    def test_old_over_cap_rejected(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=50)
        err = plugin._check_edit_span("o" * 60, "new")
        assert err is not None
        assert "'old' is 60 chars" in err
        # #3 anchor guidance: minimal old/new pinned by prologue/epilogue, NOT a
        # widened anchor or a "single distinctive line".
        assert "prologue" in err and "epilogue" in err
        assert "minimal" in err
        assert "single distinctive line" not in err
        # Default profile keeps full-replacement, so the remedy is mentioned and
        # is only reachable by OMITting 'old' (new_content ignored when old set).
        assert "new_content" in err
        assert "OMIT 'old'" in err

    def test_new_over_cap_rejected(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=50)
        err = plugin._check_edit_span("old", "n" * 60)
        assert err is not None
        assert "'new' is 60 chars" in err

    def test_both_over_cap_named(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=50)
        err = plugin._check_edit_span("o" * 60, "n" * 70)
        assert "'old' is 60 chars" in err
        assert "'new' is 70 chars" in err

    def test_at_cap_boundary_passes(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=10)
        # exactly cap chars on each side → allowed (cap is inclusive)
        assert plugin._check_edit_span("a" * 10, "b" * 10) is None

    def test_none_args_treated_as_empty(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=10)
        assert plugin._check_edit_span(None, None) is None


class TestUpdateFileExecutionWithCap:
    """End-to-end enforcement through ``_execute_update_file``."""

    def test_targeted_edit_over_cap_rejected_and_file_unchanged(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=20)
        f = tmp_path / "f.txt"
        original = "alpha\nbeta\ngamma\n" * 5
        f.write_text(original)

        result = plugin._execute_update_file({
            "path": str(f),
            "old": original,             # whole-file old → over cap
            "new": original.upper(),     # whole-file new → over cap
        })

        assert "error" in result
        assert "each be ≤ 20 characters" in result["error"]
        # Critically: nothing was written.
        assert f.read_text() == original

    def test_targeted_edit_under_cap_succeeds(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=100)
        f = tmp_path / "f.txt"
        f.write_text("value = 1\nother = 2\n")

        result = plugin._execute_update_file({
            "path": str(f),
            "old": "value = 1",
            "new": "value = 42",
        })

        assert "error" not in result
        assert result["success"] is True
        assert "value = 42" in f.read_text()

    def test_full_replacement_over_cap_is_allowed(self, tmp_path):
        """``new_content`` is the legitimate whole-file path — never capped."""
        plugin = _make_plugin(tmp_path, cap=20)
        f = tmp_path / "f.txt"
        f.write_text("old content")

        big = "x" * 500  # far over the cap, but full-replacement mode
        result = plugin._execute_update_file({
            "path": str(f),
            "new_content": big,
        })

        assert "error" not in result
        assert result["success"] is True
        assert f.read_text() == big

    def test_no_cap_allows_large_targeted_edit(self, tmp_path):
        plugin = _make_plugin(tmp_path)  # unlimited
        f = tmp_path / "f.txt"
        original = "line\n" * 200
        f.write_text(original)

        result = plugin._execute_update_file({
            "path": str(f),
            "old": original,
            "new": original + "tail\n",
        })

        assert "error" not in result
        assert result["success"] is True


class TestPermissionPreviewWithCap:
    """The preview path surfaces the cap as a ``pre_validation_error``."""

    def test_preview_reports_span_pre_validation_error(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=20)
        f = tmp_path / "f.txt"
        f.write_text("abcdefghij" * 10)

        info = plugin._format_update_file({
            "path": str(f),
            "old": "abcdefghij" * 10,   # 100 chars > cap
            "new": "z" * 80,
        })

        assert info is not None
        assert info.pre_validation_error is not None
        assert "each be ≤ 20 characters" in info.pre_validation_error


class TestDynamicSchema:
    """The tool description advertises the cap only when it is set."""

    def _update_schema(self, plugin):
        schemas = plugin.get_tool_schemas()
        return [s for s in schemas if s.name == "updateFile"][0]

    def test_description_unchanged_without_cap(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        schema = self._update_schema(plugin)
        assert "must each be" not in schema.description
        props = schema.parameters["properties"]
        assert "Must be ≤" not in props["old"]["description"]
        assert "Must be ≤" not in props["new"]["description"]

    def test_description_advertises_cap_when_set(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=800)
        schema = self._update_schema(plugin)
        assert "≤ 800 characters" in schema.description
        assert "new_content" in schema.description
        props = schema.parameters["properties"]
        assert "≤ 800 characters" in props["old"]["description"]
        assert "≤ 800 characters" in props["new"]["description"]
        # new_content (full replacement) carries no span limit.
        assert "Must be ≤" not in props["new_content"]["description"]


class TestParseAllowFullReplace:
    """Validation of the ``allow_full_replace`` config value."""

    def test_absent_defaults_true(self):
        assert FileEditPlugin._parse_allow_full_replace(None) is True

    def test_bool_passthrough(self):
        assert FileEditPlugin._parse_allow_full_replace(True) is True
        assert FileEditPlugin._parse_allow_full_replace(False) is False

    def test_string_spellings(self):
        for t in ("false", "False", "0", "no", "off"):
            assert FileEditPlugin._parse_allow_full_replace(t) is False
        for t in ("true", "True", "1", "yes", "on"):
            assert FileEditPlugin._parse_allow_full_replace(t) is True

    def test_garbage_defaults_true(self):
        # A typo must not silently strip a tool mode.
        assert FileEditPlugin._parse_allow_full_replace("maybe") is True
        assert FileEditPlugin._parse_allow_full_replace(42) is True

    def test_initialize_caches_gate(self, tmp_path):
        assert _make_plugin(tmp_path, allow_full_replace=False)._allow_full_replace is False
        assert _make_plugin(tmp_path)._allow_full_replace is True


class TestFullReplaceGate:
    """allow_full_replace=False removes the whole-file path entirely."""

    def test_schema_drops_new_content_when_gated(self, tmp_path):
        plugin = _make_plugin(tmp_path, allow_full_replace=False)
        schema = [s for s in plugin.get_tool_schemas() if s.name == "updateFile"][0]
        # The arg is gone — the model literally cannot pass new_content.
        assert "new_content" not in schema.parameters["properties"]
        # The mode is not advertised as usable, but IS flagged as disabled.
        assert "provide 'new_content'" not in schema.description
        assert "disabled for this profile" in schema.description

    def test_schema_keeps_new_content_by_default(self, tmp_path):
        plugin = _make_plugin(tmp_path)
        schema = [s for s in plugin.get_tool_schemas() if s.name == "updateFile"][0]
        assert "new_content" in schema.parameters["properties"]

    def test_runtime_rejects_full_replace_when_gated(self, tmp_path):
        plugin = _make_plugin(tmp_path, allow_full_replace=False)
        f = tmp_path / "f.txt"
        f.write_text("original")
        result = plugin._execute_update_file({
            "path": str(f),
            "new_content": "wholesale rewrite",
        })
        assert "error" in result
        assert "disabled for this profile" in result["error"]
        assert f.read_text() == "original"  # nothing written

    def test_targeted_edit_still_works_when_gated(self, tmp_path):
        plugin = _make_plugin(tmp_path, allow_full_replace=False)
        f = tmp_path / "f.txt"
        f.write_text("value = 1\n")
        result = plugin._execute_update_file({
            "path": str(f),
            "old": "value = 1",
            "new": "value = 2",
        })
        assert "error" not in result
        assert f.read_text() == "value = 2\n"

    def test_preview_reports_gate_pre_validation_error(self, tmp_path):
        plugin = _make_plugin(tmp_path, allow_full_replace=False)
        f = tmp_path / "f.txt"
        f.write_text("x")
        info = plugin._format_update_file({
            "path": str(f),
            "new_content": "y" * 50,
        })
        assert info is not None
        assert info.pre_validation_error is not None
        assert "disabled for this profile" in info.pre_validation_error


class TestConditionalCapWording:
    """The cap-rejection error mentions new_content ONLY when it is enabled."""

    def test_gated_cap_error_omits_new_content_remedy(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=50, allow_full_replace=False)
        err = plugin._check_edit_span("o" * 60, "n" * 60)
        assert err is not None
        # Anchor guidance always present...
        assert "prologue" in err and "epilogue" in err
        # ...but the new_content remedy (the clobber path) is suppressed.
        assert "new_content" not in err

    def test_allowed_cap_error_includes_new_content_remedy(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=50)  # default allow_full_replace=True
        err = plugin._check_edit_span("o" * 60, "n" * 60)
        assert "new_content" in err
        assert "OMIT 'old'" in err

    def test_gated_schema_hint_omits_new_content(self, tmp_path):
        plugin = _make_plugin(tmp_path, cap=800, allow_full_replace=False)
        schema = [s for s in plugin.get_tool_schemas() if s.name == "updateFile"][0]
        assert "≤ 800 characters" in schema.description
        # The cap-hint's whole-file remedy ("To replace the entire file …") is
        # suppressed; the only new_content mention left is the "disabled" notice.
        assert "To replace the entire file" not in schema.description
        assert "provide 'new_content'" not in schema.description
