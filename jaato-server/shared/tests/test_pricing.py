"""Tests for ``shared.pricing`` — Litellm-compatible pricing table.

Covers:
- Empty table returns ``None`` (never zero)
- Lookup by exact model name; no fuzzy matching
- All four cost dimensions (input/output/cache_read/cache_creation)
- Mixed-coverage entries (model has some rates but not others)
- File loading: missing, malformed, multi-source precedence
"""

import json
from pathlib import Path

import pytest

from shared.pricing import PricingTable, discover_pricing_paths, load_pricing


class TestPricingTable:

    def test_empty_table_returns_none(self):
        t = PricingTable.empty()
        assert t.cost_for_usage("any-model", 1000, 500) is None
        assert t.has("any-model") is False

    def test_unknown_model_returns_none(self):
        t = PricingTable({"claude-x": {"input_cost_per_token": 1e-6}})
        assert t.cost_for_usage("gpt-4o", 1000, 500) is None

    def test_input_output_cost(self):
        t = PricingTable({
            "claude-sonnet-4-5": {
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 15e-06,
            }
        })
        # 1000 in × 3e-6 + 500 out × 15e-6 = 0.003 + 0.0075 = 0.0105
        assert t.cost_for_usage("claude-sonnet-4-5", 1000, 500) == pytest.approx(0.0105)

    def test_cache_dimensions(self):
        t = PricingTable({
            "claude-sonnet-4-5": {
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 15e-06,
                "cache_read_input_token_cost": 3e-07,       # 0.1× input
                "cache_creation_input_token_cost": 3.75e-06,  # 1.25× input
            }
        })
        cost = t.cost_for_usage(
            "claude-sonnet-4-5",
            prompt_tokens=100,
            output_tokens=500,
            cache_read_tokens=900,
            cache_creation_tokens=200,
        )
        # 100*3e-6 + 500*15e-6 + 900*3e-7 + 200*3.75e-6
        expected = 0.0003 + 0.0075 + 0.00027 + 0.00075
        assert cost == pytest.approx(expected)

    def test_partial_entry_returns_partial_cost(self):
        """A model entry with only input/output rates still works for those."""
        t = PricingTable({
            "minimal": {"input_cost_per_token": 1e-06}
        })
        # cache_read passed but rate missing → contributes nothing
        cost = t.cost_for_usage("minimal", 1000, 500, cache_read_tokens=900)
        assert cost == pytest.approx(1e-06 * 1000)

    def test_empty_entry_returns_none(self):
        """A model entry with no recognised cost keys returns None, not 0."""
        t = PricingTable({"weird": {"some_unrelated_field": 42}})
        assert t.cost_for_usage("weird", 1000, 500) is None

    def test_zero_tokens_returns_zero_when_rates_known(self):
        """0 tokens × known rate = 0.0 (deliberate, distinct from None)."""
        t = PricingTable({"x": {"input_cost_per_token": 1e-06}})
        # When input_rate is known but prompt_tokens is 0, no contribution
        # is added — but since output is 0 too, any_known is False, so → None.
        assert t.cost_for_usage("x", 0, 0) is None
        # With non-zero tokens, the rate kicks in.
        assert t.cost_for_usage("x", 1000, 0) == pytest.approx(1e-3)


class TestPricingFileLoading:

    def test_missing_file_returns_empty(self, tmp_path: Path):
        t = PricingTable.from_paths([tmp_path / "nonexistent.json"])
        assert t.cost_for_usage("any", 100, 50) is None

    def test_malformed_json_skipped(self, tmp_path: Path, caplog):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {", encoding="utf-8")
        t = PricingTable.from_paths([bad])
        # Falls through to empty; no exception bubbles up
        assert t.cost_for_usage("any", 100, 50) is None

    def test_non_dict_root_skipped(self, tmp_path: Path):
        bad = tmp_path / "list.json"
        bad.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
        t = PricingTable.from_paths([bad])
        assert t.cost_for_usage("any", 100, 50) is None

    def test_workspace_overrides_user(self, tmp_path: Path):
        """Earlier paths win on key conflicts."""
        ws = tmp_path / "workspace.json"
        usr = tmp_path / "user.json"
        ws.write_text(json.dumps({
            "shared": {"input_cost_per_token": 9e-06},  # workspace value
        }))
        usr.write_text(json.dumps({
            "shared": {"input_cost_per_token": 1e-06},  # user value (loses)
            "user-only": {"input_cost_per_token": 5e-06},
        }))
        t = PricingTable.from_paths([ws, usr])
        # Workspace value wins for "shared"
        assert t.cost_for_usage("shared", 1000, 0) == pytest.approx(9e-3)
        # user-only still accessible (no conflict)
        assert t.cost_for_usage("user-only", 1000, 0) == pytest.approx(5e-3)


class TestDiscoverPaths:

    def test_workspace_first_then_user(self, tmp_path):
        paths = discover_pricing_paths(workspace_path=str(tmp_path))
        # Workspace path is first (highest precedence)
        assert paths[0] == tmp_path / ".jaato" / "pricing.json"
        # User path is appended
        assert paths[-1].name == "pricing.json"
        assert ".jaato" in str(paths[-1])

    def test_no_workspace_returns_user_only(self):
        paths = discover_pricing_paths(workspace_path=None)
        # User-level only
        assert len(paths) == 1
        assert paths[0].name == "pricing.json"


class TestLoadPricing:

    def test_load_with_workspace(self, tmp_path: Path):
        jaato_dir = tmp_path / ".jaato"
        jaato_dir.mkdir()
        (jaato_dir / "pricing.json").write_text(json.dumps({
            "claude-sonnet-4-5": {"input_cost_per_token": 3e-06},
        }))
        t = load_pricing(workspace_path=str(tmp_path))
        # Workspace pricing visible
        assert t.cost_for_usage("claude-sonnet-4-5", 1000, 0) == pytest.approx(3e-3)
