"""Tests for shared/completion_schema_loader.py."""

import json
from unittest.mock import patch

from shared.completion_schema_loader import (
    COMPLETION_SCHEMAS_SUBDIR,
    resolve_completion_schema,
    _resolve_schema_path,
)


SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


class TestResolveCompletionSchema:

    def test_none_returns_none(self):
        assert resolve_completion_schema(None) is None

    def test_inline_dict_returned_as_is(self):
        assert resolve_completion_schema(SAMPLE_SCHEMA) == SAMPLE_SCHEMA

    def test_invalid_type_returns_none(self):
        assert resolve_completion_schema(123) is None  # type: ignore[arg-type]
        assert resolve_completion_schema(["not", "a", "dict"]) is None  # type: ignore[arg-type]

    def test_path_resolved_from_workspace(self, tmp_path):
        ws = tmp_path / "workspace"
        schemas_dir = ws / ".jaato" / COMPLETION_SCHEMAS_SUBDIR
        schemas_dir.mkdir(parents=True)
        schema_file = schemas_dir / "triage.json"
        schema_file.write_text(json.dumps(SAMPLE_SCHEMA))

        result = resolve_completion_schema(
            "triage.json", workspace_path=str(ws)
        )
        assert result == SAMPLE_SCHEMA

    def test_absolute_path_resolved(self, tmp_path):
        schema_file = tmp_path / "abs.json"
        schema_file.write_text(json.dumps(SAMPLE_SCHEMA))
        result = resolve_completion_schema(str(schema_file))
        assert result == SAMPLE_SCHEMA

    def test_path_not_found_returns_none(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        result = resolve_completion_schema(
            "missing.json", workspace_path=str(ws)
        )
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        ws = tmp_path / "workspace"
        schemas_dir = ws / ".jaato" / COMPLETION_SCHEMAS_SUBDIR
        schemas_dir.mkdir(parents=True)
        bad = schemas_dir / "bad.json"
        bad.write_text("{ not valid json")
        result = resolve_completion_schema(
            "bad.json", workspace_path=str(ws)
        )
        assert result is None

    def test_non_object_json_returns_none(self, tmp_path):
        ws = tmp_path / "workspace"
        schemas_dir = ws / ".jaato" / COMPLETION_SCHEMAS_SUBDIR
        schemas_dir.mkdir(parents=True)
        arr = schemas_dir / "array.json"
        arr.write_text(json.dumps([1, 2, 3]))
        result = resolve_completion_schema(
            "array.json", workspace_path=str(ws)
        )
        assert result is None


class TestResolveSchemaPath:

    def test_workspace_tier_wins_over_home(self, tmp_path):
        fake_home = tmp_path / "home"
        home_dir = fake_home / ".jaato" / COMPLETION_SCHEMAS_SUBDIR
        home_dir.mkdir(parents=True)
        home_schema = home_dir / "shared.json"
        home_schema.write_text(json.dumps({"v": "home"}))

        ws = tmp_path / "workspace"
        ws_dir = ws / ".jaato" / COMPLETION_SCHEMAS_SUBDIR
        ws_dir.mkdir(parents=True)
        ws_schema = ws_dir / "shared.json"
        ws_schema.write_text(json.dumps({"v": "workspace"}))

        with patch(
            "shared.completion_schema_loader.Path.home",
            return_value=fake_home,
        ):
            assert (
                _resolve_schema_path("shared.json", str(ws)) == ws_schema
            )

    def test_home_fallback(self, tmp_path):
        fake_home = tmp_path / "home"
        home_dir = fake_home / ".jaato" / COMPLETION_SCHEMAS_SUBDIR
        home_dir.mkdir(parents=True)
        schema = home_dir / "global.json"
        schema.write_text(json.dumps(SAMPLE_SCHEMA))

        with patch(
            "shared.completion_schema_loader.Path.home",
            return_value=fake_home,
        ):
            assert _resolve_schema_path("global.json", None) == schema

    def test_no_match_returns_none(self, tmp_path):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        with patch(
            "shared.completion_schema_loader.Path.home",
            return_value=fake_home,
        ):
            assert _resolve_schema_path("nope.json", None) is None
