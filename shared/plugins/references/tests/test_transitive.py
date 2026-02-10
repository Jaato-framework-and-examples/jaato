"""Tests for transitive reference detection and tag-based enrichment.

Tests the ability to automatically discover and inject references that are
mentioned within pre-selected reference content, and the tag-based content
enrichment hints that surface relevant unselected references.
"""

import os
import tempfile
import pytest

from ..plugin import ReferencesPlugin, create_plugin, MAX_TRANSITIVE_DEPTH
from ..models import ReferenceSource, SourceType, InjectionMode
from ...base import PromptEnrichmentResult


class TestFindReferencedIds:
    """Tests for _find_referenced_ids method."""

    def test_finds_exact_id_match(self):
        """Test finding exact reference ID in content."""
        plugin = create_plugin()

        content = "See skill-001-circuit-breaker for implementation details."
        catalog_ids = {"skill-001-circuit-breaker", "other-ref"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        assert "skill-001-circuit-breaker" in found
        assert "other-ref" not in found

    def test_finds_multiple_ids(self):
        """Test finding multiple reference IDs in content."""
        plugin = create_plugin()

        content = """
        This document references:
        - skill-001-circuit-breaker for resilience
        - api-guidelines for standards
        """
        catalog_ids = {"skill-001-circuit-breaker", "api-guidelines", "unused-ref"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        assert "skill-001-circuit-breaker" in found
        assert "api-guidelines" in found
        assert "unused-ref" not in found

    def test_finds_id_with_backticks(self):
        """Test finding reference ID wrapped in backticks."""
        plugin = create_plugin()

        content = "Use `my-reference-id` for this feature."
        catalog_ids = {"my-reference-id"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        assert "my-reference-id" in found

    def test_finds_id_with_brackets(self):
        """Test finding reference ID in wiki-style brackets."""
        plugin = create_plugin()

        content = "See [[my-reference-id]] for more info."
        catalog_ids = {"my-reference-id"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        assert "my-reference-id" in found

    def test_finds_id_with_at_syntax(self):
        """Test finding reference ID with @ref: syntax."""
        plugin = create_plugin()

        content = "Related: @ref:my-reference-id for details."
        catalog_ids = {"my-reference-id"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        assert "my-reference-id" in found

    def test_does_not_match_partial_id(self):
        """Test that partial ID matches are not returned."""
        plugin = create_plugin()

        content = "The skill-001-circuit-breaker-extended reference."
        catalog_ids = {"skill-001-circuit-breaker"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        # Should not match because the content has an extended version
        assert "skill-001-circuit-breaker" not in found

    def test_finds_id_at_line_start(self):
        """Test finding reference ID at the start of a line."""
        plugin = create_plugin()

        content = """First line.
my-reference-id is mentioned here."""
        catalog_ids = {"my-reference-id"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        assert "my-reference-id" in found

    def test_finds_id_at_line_end(self):
        """Test finding reference ID at the end of a line."""
        plugin = create_plugin()

        content = "See my-reference-id"
        catalog_ids = {"my-reference-id"}

        found = plugin._find_referenced_ids(content, catalog_ids)

        assert "my-reference-id" in found

    def test_empty_content_returns_empty_set(self):
        """Test that empty content returns no matches."""
        plugin = create_plugin()

        found = plugin._find_referenced_ids("", {"ref-1", "ref-2"})

        assert len(found) == 0

    def test_empty_catalog_returns_empty_set(self):
        """Test that empty catalog returns no matches."""
        plugin = create_plugin()

        found = plugin._find_referenced_ids("Some content with ref-1", set())

        assert len(found) == 0


class TestGetReferenceContent:
    """Tests for _get_reference_content method."""

    def test_returns_inline_content(self):
        """Test that INLINE source content is returned directly."""
        plugin = create_plugin()

        source = ReferenceSource(
            id="inline-ref",
            name="Inline Reference",
            description="Test inline content",
            type=SourceType.INLINE,
            mode=InjectionMode.SELECTABLE,
            content="This is the inline content."
        )

        content = plugin._get_reference_content(source)

        assert content == "This is the inline content."

    def test_returns_local_file_content(self):
        """Test that LOCAL source content is read from file."""
        plugin = create_plugin()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Content\n\nThis is test file content.")
            temp_path = f.name

        try:
            source = ReferenceSource(
                id="local-ref",
                name="Local Reference",
                description="Test local file",
                type=SourceType.LOCAL,
                mode=InjectionMode.SELECTABLE,
                path=temp_path,
                resolved_path=temp_path
            )

            content = plugin._get_reference_content(source)

            assert content is not None
            assert "# Test Content" in content
            assert "This is test file content." in content
        finally:
            os.unlink(temp_path)

    def test_returns_directory_content(self):
        """Test that LOCAL directory source concatenates file contents."""
        plugin = create_plugin()

        # Create a temporary directory with files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some files
            file1 = os.path.join(temp_dir, "file1.md")
            file2 = os.path.join(temp_dir, "file2.txt")

            with open(file1, 'w') as f:
                f.write("Content from file 1")
            with open(file2, 'w') as f:
                f.write("Content from file 2")

            source = ReferenceSource(
                id="dir-ref",
                name="Directory Reference",
                description="Test directory",
                type=SourceType.LOCAL,
                mode=InjectionMode.SELECTABLE,
                path=temp_dir,
                resolved_path=temp_dir
            )

            content = plugin._get_reference_content(source)

            assert content is not None
            assert "Content from file 1" in content
            assert "Content from file 2" in content

    def test_returns_none_for_url_source(self):
        """Test that URL source returns None (requires external fetch)."""
        plugin = create_plugin()

        source = ReferenceSource(
            id="url-ref",
            name="URL Reference",
            description="Test URL",
            type=SourceType.URL,
            mode=InjectionMode.SELECTABLE,
            url="https://example.com/doc"
        )

        content = plugin._get_reference_content(source)

        assert content is None

    def test_returns_none_for_mcp_source(self):
        """Test that MCP source returns None (requires external fetch)."""
        plugin = create_plugin()

        source = ReferenceSource(
            id="mcp-ref",
            name="MCP Reference",
            description="Test MCP",
            type=SourceType.MCP,
            mode=InjectionMode.SELECTABLE,
            server="some-server",
            tool="some-tool"
        )

        content = plugin._get_reference_content(source)

        assert content is None

    def test_returns_none_for_nonexistent_file(self):
        """Test that nonexistent file returns None."""
        plugin = create_plugin()

        source = ReferenceSource(
            id="missing-ref",
            name="Missing Reference",
            description="Test missing file",
            type=SourceType.LOCAL,
            mode=InjectionMode.SELECTABLE,
            path="/nonexistent/path/file.md",
            resolved_path="/nonexistent/path/file.md"
        )

        content = plugin._get_reference_content(source)

        assert content is None


class TestResolveTransitiveReferences:
    """Tests for _resolve_transitive_references method."""

    def test_returns_initial_ids_when_no_content(self):
        """Test that initial IDs are returned when no content to parse."""
        plugin = create_plugin()

        catalog = {
            "ref-1": ReferenceSource(
                id="ref-1",
                name="Reference 1",
                description="Test",
                type=SourceType.URL,  # URL has no content
                mode=InjectionMode.SELECTABLE,
                url="https://example.com"
            )
        }

        result = plugin._resolve_transitive_references(["ref-1"], catalog)

        assert result == ["ref-1"]

    def test_discovers_single_transitive_reference(self):
        """Test discovering a single transitive reference."""
        plugin = create_plugin()

        catalog = {
            "ref-1": ReferenceSource(
                id="ref-1",
                name="Reference 1",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="This document mentions ref-2 for more details."
            ),
            "ref-2": ReferenceSource(
                id="ref-2",
                name="Reference 2",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="Additional content here."
            )
        }

        result = plugin._resolve_transitive_references(["ref-1"], catalog)

        assert "ref-1" in result
        assert "ref-2" in result
        assert len(result) == 2

    def test_discovers_chain_of_references(self):
        """Test discovering a chain of transitive references."""
        plugin = create_plugin()

        catalog = {
            "ref-1": ReferenceSource(
                id="ref-1",
                name="Reference 1",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="See ref-2 for next step."
            ),
            "ref-2": ReferenceSource(
                id="ref-2",
                name="Reference 2",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="See ref-3 for final details."
            ),
            "ref-3": ReferenceSource(
                id="ref-3",
                name="Reference 3",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="Final content here."
            )
        }

        result = plugin._resolve_transitive_references(["ref-1"], catalog)

        assert "ref-1" in result
        assert "ref-2" in result
        assert "ref-3" in result
        assert len(result) == 3

    def test_handles_circular_references(self):
        """Test that circular references don't cause infinite loops."""
        plugin = create_plugin()

        catalog = {
            "ref-1": ReferenceSource(
                id="ref-1",
                name="Reference 1",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="See ref-2 for more."
            ),
            "ref-2": ReferenceSource(
                id="ref-2",
                name="Reference 2",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="See ref-1 for context."  # Circular reference
            )
        }

        result = plugin._resolve_transitive_references(["ref-1"], catalog)

        # Should not hang, and should include both refs exactly once
        assert "ref-1" in result
        assert "ref-2" in result
        assert len(result) == 2

    def test_handles_self_reference(self):
        """Test that self-references are ignored."""
        plugin = create_plugin()

        catalog = {
            "ref-1": ReferenceSource(
                id="ref-1",
                name="Reference 1",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="This is ref-1 and it mentions ref-1 again."
            )
        }

        result = plugin._resolve_transitive_references(["ref-1"], catalog)

        assert result == ["ref-1"]

    def test_respects_max_depth(self):
        """Test that max depth limit is respected."""
        plugin = create_plugin()

        # Create a chain longer than MAX_TRANSITIVE_DEPTH
        catalog = {}
        for i in range(MAX_TRANSITIVE_DEPTH + 5):
            catalog[f"ref-{i}"] = ReferenceSource(
                id=f"ref-{i}",
                name=f"Reference {i}",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content=f"See ref-{i+1} for next."
            )

        result = plugin._resolve_transitive_references(["ref-0"], catalog)

        # Should stop at max depth, not all refs should be included
        assert len(result) <= MAX_TRANSITIVE_DEPTH + 1

    def test_empty_initial_ids_returns_empty(self):
        """Test that empty initial IDs returns empty list."""
        plugin = create_plugin()

        result = plugin._resolve_transitive_references([], {})

        assert result == []

    def test_preserves_order_of_discovery(self):
        """Test that initial IDs come first, then discovered ones."""
        plugin = create_plugin()

        catalog = {
            "initial": ReferenceSource(
                id="initial",
                name="Initial",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="Mentions discovered-ref here."
            ),
            "discovered-ref": ReferenceSource(
                id="discovered-ref",
                name="Discovered",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="No more refs."
            )
        }

        result = plugin._resolve_transitive_references(["initial"], catalog)

        assert result[0] == "initial"
        assert result[1] == "discovered-ref"


class TestInitializeWithTransitive:
    """Tests for transitive resolution during plugin initialization."""

    def test_transitive_enabled_by_default(self):
        """Test that transitive resolution is enabled by default."""
        plugin = create_plugin()

        # Create temp files for sources
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f1:
            f1.write("This mentions ref-2 for details.")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f2:
            f2.write("Additional content.")
            path2 = f2.name

        try:
            plugin.initialize({
                "sources": [
                    {
                        "id": "ref-1",
                        "name": "Reference 1",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": path1
                    },
                    {
                        "id": "ref-2",
                        "name": "Reference 2",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": path2
                    }
                ],
                "preselected": ["ref-1"]
            })

            # ref-2 should be transitively discovered and selected
            selected = plugin.get_selected_ids()
            assert "ref-1" in selected
            assert "ref-2" in selected
        finally:
            plugin.shutdown()
            os.unlink(path1)
            os.unlink(path2)

    def test_transitive_can_be_disabled(self):
        """Test that transitive resolution can be disabled via config."""
        plugin = create_plugin()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f1:
            f1.write("This mentions ref-2 for details.")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f2:
            f2.write("Additional content.")
            path2 = f2.name

        try:
            plugin.initialize({
                "sources": [
                    {
                        "id": "ref-1",
                        "name": "Reference 1",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": path1
                    },
                    {
                        "id": "ref-2",
                        "name": "Reference 2",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": path2
                    }
                ],
                "preselected": ["ref-1"],
                "transitive_injection": False  # Disable transitive
            })

            # Only ref-1 should be selected, not ref-2
            selected = plugin.get_selected_ids()
            assert "ref-1" in selected
            assert "ref-2" not in selected
        finally:
            plugin.shutdown()
            os.unlink(path1)
            os.unlink(path2)

    def test_transitive_with_inline_sources(self):
        """Test transitive resolution with inline sources."""
        plugin = create_plugin()

        plugin.initialize({
            "sources": [
                {
                    "id": "main-doc",
                    "name": "Main Document",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Main content. See appendix-a for more."
                },
                {
                    "id": "appendix-a",
                    "name": "Appendix A",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Appendix content."
                }
            ],
            "preselected": ["main-doc"]
        })

        try:
            selected = plugin.get_selected_ids()
            assert "main-doc" in selected
            assert "appendix-a" in selected
        finally:
            plugin.shutdown()

    def test_transitive_adds_sources_to_available(self):
        """Test that transitively discovered sources are added to available sources."""
        plugin = create_plugin()

        plugin.initialize({
            "sources": [
                {
                    "id": "doc-1",
                    "name": "Document 1",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Mentions doc-2 here."
                },
                {
                    "id": "doc-2",
                    "name": "Document 2",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Second doc."
                }
            ],
            "preselected": ["doc-1"]
        })

        try:
            sources = plugin.get_sources()
            source_ids = [s.id for s in sources]
            assert "doc-1" in source_ids
            assert "doc-2" in source_ids
        finally:
            plugin.shutdown()

    def test_no_transitive_when_no_preselected(self):
        """Test that transitive resolution doesn't run without preselected."""
        plugin = create_plugin()

        plugin.initialize({
            "sources": [
                {
                    "id": "doc-1",
                    "name": "Document 1",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Mentions doc-2 here."
                },
                {
                    "id": "doc-2",
                    "name": "Document 2",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Second doc."
                }
            ]
            # No preselected
        })

        try:
            selected = plugin.get_selected_ids()
            assert len(selected) == 0
        finally:
            plugin.shutdown()


def _make_plugin_with_selectable_tags(sources_config):
    """Helper to create a plugin with selectable sources and tags for enrichment tests.

    Sets up a plugin with the given source configs, none preselected, so that
    _enrich_content() Pass 2 (tag matching) is exercised.

    Args:
        sources_config: List of dicts with id, name, tags, and optional description.

    Returns:
        Initialized ReferencesPlugin instance. Caller must call shutdown().
    """
    plugin = create_plugin()
    sources = []
    for cfg in sources_config:
        sources.append({
            "id": cfg["id"],
            "name": cfg.get("name", cfg["id"]),
            "description": cfg.get("description", "Test"),
            "type": "inline",
            "mode": "selectable",
            "content": cfg.get("content", "placeholder"),
            "tags": cfg["tags"],
        })
    plugin.initialize({"sources": sources})
    return plugin


class TestEnrichContentTagMatching:
    """Tests for the tag-based hint matching in _enrich_content() Pass 2.

    These tests verify that tags match as whole words with proper boundary
    detection, and that the hint content includes the triggering tags.
    """

    def test_tag_matches_standalone_word(self):
        """Tag matches when it appears as a standalone word."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            result = plugin._enrich_content("We need to use java for this project", "prompt")
            assert "tag_matched_references" in result.metadata
            assert "ref-1" in result.metadata["tag_matched_references"]
        finally:
            plugin.shutdown()

    def test_tag_does_not_match_in_dotted_name(self):
        """Tag should NOT match inside a dotted package/class name."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            result = plugin._enrich_content(
                "Import java.util.concurrent.TimeoutException from the SDK",
                "prompt"
            )
            # "java" appears only as part of "java.util..." — should not match
            assert result.metadata is None or "tag_matched_references" not in result.metadata
        finally:
            plugin.shutdown()

    def test_tag_does_not_match_in_file_extension(self):
        """Tag should NOT match inside a file extension like '.java'."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            result = plugin._enrich_content(
                "Edit the file CircuitBreaker.java to add the retry logic",
                "prompt"
            )
            # "java" appears only as ".java" extension — should not match
            assert result.metadata is None or "tag_matched_references" not in result.metadata
        finally:
            plugin.shutdown()

    def test_tag_does_not_match_in_path(self):
        """Tag should NOT match inside a file path segment."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            result = plugin._enrich_content(
                "The binary is at /usr/lib/java/bin/javac",
                "prompt"
            )
            # "java" appears only inside path segments — should not match
            assert result.metadata is None or "tag_matched_references" not in result.metadata
        finally:
            plugin.shutdown()

    def test_tag_matches_case_insensitive(self):
        """Tag matching is case-insensitive."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            result = plugin._enrich_content("JAVA is a programming language", "prompt")
            assert "tag_matched_references" in result.metadata
            assert "ref-1" in result.metadata["tag_matched_references"]
        finally:
            plugin.shutdown()

    def test_tag_matches_with_punctuation_boundary(self):
        """Tag matches when bounded by punctuation (not dots/slashes)."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            # Parentheses and commas are valid boundaries
            result = plugin._enrich_content("languages (java, python) are supported", "prompt")
            assert "tag_matched_references" in result.metadata
            assert "ref-1" in result.metadata["tag_matched_references"]
        finally:
            plugin.shutdown()

    def test_hint_shows_matched_tags(self):
        """Hint content uses 'matched:' label and shows triggering tags."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Circuit Breaker", "tags": ["circuit", "resilience"]},
        ])
        try:
            result = plugin._enrich_content(
                "We need a circuit pattern for fault tolerance",
                "prompt"
            )
            assert "(matched: circuit)" in result.prompt
        finally:
            plugin.shutdown()

    def test_multiple_tags_match_same_source(self):
        """Multiple tags from the same source can match independently."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Resilience Guide", "tags": ["retry", "timeout"]},
        ])
        try:
            result = plugin._enrich_content(
                "Configure the retry and timeout policies",
                "prompt"
            )
            matched_tags = result.metadata["tag_matched_references"]["ref-1"]
            assert "retry" in matched_tags
            assert "timeout" in matched_tags
        finally:
            plugin.shutdown()

    def test_one_tag_matches_multiple_sources(self):
        """A single tag match pulls in all sources sharing that tag."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Circuit Breaker", "tags": ["resilience"]},
            {"id": "ref-2", "name": "Retry Policy", "tags": ["resilience"]},
        ])
        try:
            result = plugin._enrich_content(
                "We need resilience patterns",
                "prompt"
            )
            matched = result.metadata["tag_matched_references"]
            assert "ref-1" in matched
            assert "ref-2" in matched
        finally:
            plugin.shutdown()

    def test_no_match_returns_original_content(self):
        """Content without matching tags returns unchanged."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            original = "This content has nothing related at all"
            result = plugin._enrich_content(original, "prompt")
            assert result.prompt == original
        finally:
            plugin.shutdown()

    def test_selected_sources_excluded_from_hints(self):
        """Already-selected sources are not included in tag hints."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Java Guide", "tags": ["java"]},
        ])
        try:
            # Simulate selection
            plugin._selected_source_ids.append("ref-1")
            result = plugin._enrich_content("We use java here", "prompt")
            # ref-1 is selected, so it should not appear in hints
            assert result.metadata is None or "tag_matched_references" not in result.metadata
        finally:
            plugin.shutdown()

    def test_multi_word_tag_matches(self):
        """Multi-word tags (containing spaces) match correctly."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "CB Guide", "tags": ["circuit breaker"]},
        ])
        try:
            result = plugin._enrich_content(
                "Implement a circuit breaker for the service",
                "prompt"
            )
            assert "tag_matched_references" in result.metadata
            assert "ref-1" in result.metadata["tag_matched_references"]
        finally:
            plugin.shutdown()

    def test_dotted_tag_matches_standalone(self):
        """A tag containing a dot matches when standalone."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Spring Boot", "tags": ["spring.boot"]},
        ])
        try:
            result = plugin._enrich_content(
                "We use spring.boot for the application",
                "prompt"
            )
            assert "tag_matched_references" in result.metadata
            assert "ref-1" in result.metadata["tag_matched_references"]
        finally:
            plugin.shutdown()

    def test_dotted_tag_does_not_match_inside_longer_name(self):
        """A dotted tag should not match inside a longer dotted name."""
        plugin = _make_plugin_with_selectable_tags([
            {"id": "ref-1", "name": "Spring Boot", "tags": ["spring.boot"]},
        ])
        try:
            result = plugin._enrich_content(
                "Import org.spring.boot.autoconfigure from the classpath",
                "prompt"
            )
            assert result.metadata is None or "tag_matched_references" not in result.metadata
        finally:
            plugin.shutdown()
