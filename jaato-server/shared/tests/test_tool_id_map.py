"""Tests for shared/tool_id_map.py — deterministic tool/category name-to-ID mapping."""

from shared.tool_id_map import id_to_name, name_to_id


class TestNameToId:

    def test_returns_prefixed_hash(self):
        result = name_to_id("readFile")
        assert result.startswith("t_")
        assert len(result) == 10  # "t_" + 8 hex chars

    def test_deterministic(self):
        assert name_to_id("readFile") == name_to_id("readFile")

    def test_different_names_produce_different_ids(self):
        assert name_to_id("readFile") != name_to_id("writeFile")

    def test_custom_prefix(self):
        result = name_to_id("filesystem", prefix="c")
        assert result.startswith("c_")

    def test_special_characters_in_name(self):
        result = name_to_id("mcp.server.tool_name")
        assert result.startswith("t_")
        assert len(result) == 10

    def test_empty_name(self):
        result = name_to_id("")
        assert result.startswith("t_")


class TestStreamingVariant:
    """Server 0.6.65+: streaming variants use ``-stream`` suffix.

    Each variant gets its own independent hash (no shared base prefix),
    so the model-facing ID is uniformly ``t_<hash>`` and conforms to
    every strict upstream tool-name regex (Anthropic / Bedrock / OpenAI
    all enforce ``^[a-zA-Z0-9_-]{1,128}$``).
    """

    def test_stream_variant_gets_own_id(self):
        base_id = name_to_id("grep_content")
        stream_id = name_to_id("grep_content-stream")
        # Different hashes — they're different inputs.
        assert base_id != stream_id

    def test_stream_variant_id_is_regex_safe(self):
        stream_id = name_to_id("grep_content-stream")
        # Anthropic / Bedrock pattern: ^[a-zA-Z0-9_-]{1,128}$
        import re
        assert re.match(r"^[a-zA-Z0-9_-]{1,128}$", stream_id)

    def test_stream_variant_roundtrip(self):
        stream_id = name_to_id("grep_content-stream")
        assert id_to_name(stream_id) == "grep_content-stream"

    def test_id_length_is_uniform(self):
        # Both base and -stream variant IDs are ``t_`` + 8 hex chars = 10.
        assert len(name_to_id("readFile")) == 10
        assert len(name_to_id("readFile-stream")) == 10


class TestIdToName:

    def test_roundtrip(self):
        tool_id = name_to_id("cli_based_tool")
        assert id_to_name(tool_id) == "cli_based_tool"

    def test_unknown_id_returns_input(self):
        assert id_to_name("t_nonexistent") == "t_nonexistent"

    def test_roundtrip_with_prefix(self):
        cat_id = name_to_id("filesystem", prefix="c")
        assert id_to_name(cat_id) == "filesystem"

    def test_multiple_tools_roundtrip(self):
        tools = ["readFile", "writeFile", "cli_based_tool", "mcp.server.search"]
        for name in tools:
            tool_id = name_to_id(name)
            assert id_to_name(tool_id) == name

    def test_unregistered_id_returns_input(self):
        # The model occasionally hallucinates IDs; these must round-trip
        # to themselves so downstream code can detect "unknown tool".
        assert id_to_name("t_deadbeef") == "t_deadbeef"


class TestStability:

    def test_ids_are_stable_across_calls(self):
        id1 = name_to_id("readFile")
        name_to_id("writeFile")
        name_to_id("cli_based_tool")
        id2 = name_to_id("readFile")
        assert id1 == id2

    def test_ids_independent_of_registration_order(self):
        id_a = name_to_id("alpha_tool")
        name_to_id("beta_tool")
        name_to_id("gamma_tool")
        # Start fresh conceptually — but since it's a pure hash,
        # the result must be the same regardless of what else was registered
        id_a2 = name_to_id("alpha_tool")
        assert id_a == id_a2
