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

        result, parent_map = plugin._resolve_transitive_references(["ref-1"], catalog)

        assert result == ["ref-1"]
        assert parent_map == {}

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

        result, _parent_map = plugin._resolve_transitive_references(["ref-1"], catalog)

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

        result, _parent_map = plugin._resolve_transitive_references(["ref-1"], catalog)

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

        result, _parent_map = plugin._resolve_transitive_references(["ref-1"], catalog)

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

        result, parent_map = plugin._resolve_transitive_references(["ref-1"], catalog)

        assert result == ["ref-1"]
        assert parent_map == {}

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

        result, _parent_map = plugin._resolve_transitive_references(["ref-0"], catalog)

        # Should stop at max depth, not all refs should be included
        assert len(result) <= MAX_TRANSITIVE_DEPTH + 1

    def test_empty_initial_ids_returns_empty(self):
        """Test that empty initial IDs returns empty list."""
        plugin = create_plugin()

        result, parent_map = plugin._resolve_transitive_references([], {})

        assert result == []
        assert parent_map == {}

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

        result, _parent_map = plugin._resolve_transitive_references(["initial"], catalog)

        assert result[0] == "initial"
        assert result[1] == "discovered-ref"


class TestFindReferencedPaths:
    """Tests for _find_referenced_paths — path-based transitive matching.

    When documents reference each other via relative paths (markdown links,
    ``./`` or ``../`` patterns), this method resolves those paths against
    the source's directory and matches them to catalog source resolved_paths.
    """

    def test_markdown_link_matches_sibling(self):
        """Markdown link to a sibling file matches catalog source."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry.md": {"retry-ref"}}
        content = "See [retry pattern](retry.md) for details."

        found = plugin._find_referenced_paths(content, "docs/circuit-breaker.md", path_to_ids)

        assert "retry-ref" in found

    def test_markdown_link_with_relative_parent(self):
        """Markdown link with ../ resolves to correct catalog path."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry/README.md": {"retry-ref"}}
        content = "See [retry](../retry/README.md) for details."

        found = plugin._find_referenced_paths(
            content, "docs/patterns/circuit-breaker.md", path_to_ids
        )

        assert "retry-ref" in found

    def test_markdown_link_with_dot_slash(self):
        """Markdown link with ./ resolves correctly."""
        plugin = create_plugin()

        path_to_ids = {"docs/patterns/timeout.md": {"timeout-ref"}}
        content = "Also see [timeout](./timeout.md)."

        found = plugin._find_referenced_paths(
            content, "docs/patterns/circuit-breaker.md", path_to_ids
        )

        assert "timeout-ref" in found

    def test_skips_http_urls(self):
        """HTTP/HTTPS URLs in markdown links are ignored."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry.md": {"retry-ref"}}
        content = "See [docs](https://example.com/retry.md) for more."

        found = plugin._find_referenced_paths(content, "docs/main.md", path_to_ids)

        assert len(found) == 0

    def test_skips_anchor_only_links(self):
        """Anchor-only links (#section) are ignored."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry.md": {"retry-ref"}}
        content = "See [section](#overview) for more."

        found = plugin._find_referenced_paths(content, "docs/main.md", path_to_ids)

        assert len(found) == 0

    def test_strips_anchor_from_path(self):
        """Anchor fragments are stripped before matching: path.md#sec → path.md."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry.md": {"retry-ref"}}
        content = "See [retry](retry.md#configuration) for details."

        found = plugin._find_referenced_paths(content, "docs/main.md", path_to_ids)

        assert "retry-ref" in found

    def test_bare_relative_path_with_dot_dot(self):
        """Bare ../path (not in markdown link) is extracted and matched."""
        plugin = create_plugin()

        path_to_ids = {"patterns/retry.md": {"retry-ref"}}
        content = "Refer to ../retry.md for the retry implementation."

        found = plugin._find_referenced_paths(
            content, "patterns/circuit/breaker.md", path_to_ids
        )

        assert "retry-ref" in found

    def test_bare_relative_path_with_dot_slash(self):
        """Bare ./path is extracted and matched."""
        plugin = create_plugin()

        path_to_ids = {"docs/timeout.md": {"timeout-ref"}}
        content = "Also check ./timeout.md for timeout config."

        found = plugin._find_referenced_paths(content, "docs/main.md", path_to_ids)

        assert "timeout-ref" in found

    def test_no_match_when_path_not_in_catalog(self):
        """Paths that don't resolve to any catalog source return nothing."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry.md": {"retry-ref"}}
        content = "See [notes](./notes.md) for more."

        found = plugin._find_referenced_paths(content, "docs/main.md", path_to_ids)

        assert len(found) == 0

    def test_empty_content_returns_empty(self):
        """Empty content produces no matches."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry.md": {"retry-ref"}}
        found = plugin._find_referenced_paths("", "docs/main.md", path_to_ids)

        assert len(found) == 0

    def test_no_source_path_returns_empty(self):
        """None source_resolved_path returns empty (INLINE sources)."""
        plugin = create_plugin()

        path_to_ids = {"docs/retry.md": {"retry-ref"}}
        content = "See [retry](retry.md) for details."

        found = plugin._find_referenced_paths(content, "", path_to_ids)

        assert len(found) == 0

    def test_multiple_links_match_multiple_sources(self):
        """Multiple markdown links can match different catalog sources."""
        plugin = create_plugin()

        path_to_ids = {
            "docs/retry.md": {"retry-ref"},
            "docs/timeout.md": {"timeout-ref"},
        }
        content = (
            "See [retry](retry.md) and [timeout](timeout.md) "
            "for resilience patterns."
        )

        found = plugin._find_referenced_paths(content, "docs/main.md", path_to_ids)

        assert "retry-ref" in found
        assert "timeout-ref" in found

    def test_directory_source_matches_with_trailing_slash(self):
        """Path linking to a directory matches a directory catalog source."""
        plugin = create_plugin()

        # Directory source has trailing slash in normpath
        path_to_ids = {"docs/patterns": {"patterns-ref"}}
        content = "See [all patterns](./patterns) for the full catalog."

        found = plugin._find_referenced_paths(content, "docs/main.md", path_to_ids)

        assert "patterns-ref" in found


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

    def test_transitive_via_relative_path(self):
        """Transitive detection discovers references via relative path links.

        When a selected document contains a markdown link like [text](sibling.md),
        and another catalog source has that resolved path, the linked source
        is transitively selected.
        """
        plugin = create_plugin()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create two files where file1 links to file2 via relative path
            file1 = os.path.join(temp_dir, "circuit-breaker.md")
            file2 = os.path.join(temp_dir, "retry.md")

            with open(file1, 'w') as f:
                f.write("# Circuit Breaker\n\nSee [retry pattern](./retry.md) for retry details.")
            with open(file2, 'w') as f:
                f.write("# Retry Pattern\n\nRetry content here.")

            plugin.initialize({
                "sources": [
                    {
                        "id": "cb-ref",
                        "name": "Circuit Breaker",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": file1
                    },
                    {
                        "id": "retry-ref",
                        "name": "Retry Pattern",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": file2
                    }
                ],
                "preselected": ["cb-ref"]
            })

            try:
                selected = plugin.get_selected_ids()
                assert "cb-ref" in selected
                assert "retry-ref" in selected
            finally:
                plugin.shutdown()

    def test_transitive_via_parent_relative_path(self):
        """Transitive detection resolves ../ paths between subdirectories."""
        plugin = create_plugin()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectories
            sub1 = os.path.join(temp_dir, "patterns")
            sub2 = os.path.join(temp_dir, "guides")
            os.makedirs(sub1)
            os.makedirs(sub2)

            file1 = os.path.join(sub1, "circuit-breaker.md")
            file2 = os.path.join(sub2, "retry-guide.md")

            with open(file1, 'w') as f:
                f.write("# Circuit Breaker\n\nSee [retry](../guides/retry-guide.md) for details.")
            with open(file2, 'w') as f:
                f.write("# Retry Guide\n\nRetry content here.")

            plugin.initialize({
                "sources": [
                    {
                        "id": "cb-ref",
                        "name": "Circuit Breaker",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": file1
                    },
                    {
                        "id": "retry-ref",
                        "name": "Retry Guide",
                        "description": "Test",
                        "type": "local",
                        "mode": "selectable",
                        "path": file2
                    }
                ],
                "preselected": ["cb-ref"]
            })

            try:
                selected = plugin.get_selected_ids()
                assert "cb-ref" in selected
                assert "retry-ref" in selected
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


class TestTransitiveParentMap:
    """Tests for transitive parent mapping and notification infrastructure.

    Verifies that _resolve_transitive_references() tracks which parent
    source caused each transitive discovery, and that the metadata is
    surfaced in system instructions and enrichment notifications.
    """

    def test_resolve_returns_parent_map(self):
        """_resolve_transitive_references returns a parent map alongside IDs."""
        plugin = create_plugin()

        catalog = {
            "ref-1": ReferenceSource(
                id="ref-1",
                name="Reference 1",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="This mentions ref-2 for details."
            ),
            "ref-2": ReferenceSource(
                id="ref-2",
                name="Reference 2",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="Additional content."
            )
        }

        resolved_ids, parent_map = plugin._resolve_transitive_references(
            ["ref-1"], catalog
        )

        assert "ref-1" in resolved_ids
        assert "ref-2" in resolved_ids
        # ref-2 was discovered from ref-1
        assert "ref-2" in parent_map
        assert "ref-1" in parent_map["ref-2"]
        # ref-1 is initial, not in parent map
        assert "ref-1" not in parent_map

    def test_parent_map_tracks_multiple_parents(self):
        """A source referenced by two parents has both recorded."""
        plugin = create_plugin()

        catalog = {
            "doc-a": ReferenceSource(
                id="doc-a",
                name="Doc A",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="Mentions shared-ref here."
            ),
            "doc-b": ReferenceSource(
                id="doc-b",
                name="Doc B",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="Also mentions shared-ref here."
            ),
            "shared-ref": ReferenceSource(
                id="shared-ref",
                name="Shared",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="Common content."
            )
        }

        _ids, parent_map = plugin._resolve_transitive_references(
            ["doc-a", "doc-b"], catalog
        )

        assert "shared-ref" in parent_map
        assert parent_map["shared-ref"] == {"doc-a", "doc-b"}

    def test_chain_parent_map(self):
        """Chain of transitive references tracks immediate parents only."""
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
                content="End of chain."
            )
        }

        _ids, parent_map = plugin._resolve_transitive_references(
            ["ref-1"], catalog
        )

        # ref-2 was discovered from ref-1
        assert parent_map["ref-2"] == {"ref-1"}
        # ref-3 was discovered from ref-2 (not ref-1)
        assert parent_map["ref-3"] == {"ref-2"}

    def test_empty_parent_map_when_no_transitive(self):
        """Parent map is empty when no transitive references are found."""
        plugin = create_plugin()

        catalog = {
            "ref-1": ReferenceSource(
                id="ref-1",
                name="Reference 1",
                description="Test",
                type=SourceType.INLINE,
                mode=InjectionMode.SELECTABLE,
                content="No other references here."
            )
        }

        _ids, parent_map = plugin._resolve_transitive_references(
            ["ref-1"], catalog
        )

        assert parent_map == {}


class TestTransitiveSystemInstructions:
    """Tests for transitive annotations in system instructions."""

    def test_system_instructions_annotate_transitive_sources(self):
        """Transitively selected sources are annotated in system instructions."""
        plugin = create_plugin()

        plugin.initialize({
            "sources": [
                {
                    "id": "main-doc",
                    "name": "Main Document",
                    "description": "Primary document",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Main content. See appendix-a for more."
                },
                {
                    "id": "appendix-a",
                    "name": "Appendix A",
                    "description": "Supplementary",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Appendix content."
                }
            ],
            "preselected": ["main-doc"],
            "exclude_tools": ["selectReferences", "listReferences"]
        })

        try:
            instructions = plugin.get_system_instructions()
            assert instructions is not None
            # Appendix A should be annotated as transitively included
            assert "Transitively included" in instructions
            assert "@main-doc" in instructions
            # Main Document is directly preselected — no annotation
            # Find the annotation and make sure it's associated with appendix-a
            lines = instructions.split("\n")
            found_appendix = False
            for i, line in enumerate(lines):
                if "Appendix A" in line:
                    found_appendix = True
                if found_appendix and "Transitively included" in line:
                    assert "@main-doc" in line
                    break
            else:
                pytest.fail("Transitive annotation not found after Appendix A section")
        finally:
            plugin.shutdown()

    def test_system_instructions_no_annotation_when_no_transitive(self):
        """Non-transitive sources are not annotated."""
        plugin = create_plugin()

        plugin.initialize({
            "sources": [
                {
                    "id": "doc-1",
                    "name": "Document 1",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Independent content."
                }
            ],
            "preselected": ["doc-1"],
            "exclude_tools": ["selectReferences", "listReferences"]
        })

        try:
            instructions = plugin.get_system_instructions()
            assert instructions is not None
            assert "Transitively included" not in instructions
        finally:
            plugin.shutdown()


class TestTransitiveEnrichmentNotification:
    """Tests for one-time transitive selection hint in prompt enrichment."""

    def test_transitive_hint_on_first_prompt(self):
        """First prompt enrichment includes transitive selection hint."""
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
            "preselected": ["main-doc"],
            "exclude_tools": ["selectReferences", "listReferences"]
        })

        try:
            result = plugin._enrich_content("Tell me about the project", "prompt")
            # Hint should be in the enriched content
            assert "Transitively selected references" in result.prompt
            assert "@appendix-a" in result.prompt
            assert "from @main-doc" in result.prompt
            # Metadata should contain transitive info
            assert "transitive_references" in result.metadata
            assert "appendix-a" in result.metadata["transitive_references"]
        finally:
            plugin.shutdown()

    def test_transitive_hint_fires_only_once(self):
        """Transitive hint appears only on the first prompt enrichment."""
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
            "preselected": ["main-doc"],
            "exclude_tools": ["selectReferences", "listReferences"]
        })

        try:
            # First call — should have hint
            result1 = plugin._enrich_content("First prompt", "prompt")
            assert "Transitively selected references" in result1.prompt

            # Second call — should NOT have hint
            result2 = plugin._enrich_content("Second prompt", "prompt")
            assert "Transitively selected references" not in result2.prompt
        finally:
            plugin.shutdown()

    def test_transitive_hint_skips_tool_results(self):
        """Transitive hint does not fire for tool result enrichment."""
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
            "preselected": ["main-doc"],
            "exclude_tools": ["selectReferences", "listReferences"]
        })

        try:
            # Tool result — should NOT have hint
            result1 = plugin._enrich_content("Some tool output", "tool:readFile")
            assert "Transitively selected references" not in result1.prompt

            # Next prompt — hint should still be pending and fire now
            result2 = plugin._enrich_content("First prompt", "prompt")
            assert "Transitively selected references" in result2.prompt
        finally:
            plugin.shutdown()

    def test_no_transitive_hint_when_no_transitive(self):
        """No transitive hint when no transitive references exist."""
        plugin = create_plugin()

        plugin.initialize({
            "sources": [
                {
                    "id": "doc-1",
                    "name": "Document 1",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Independent content."
                }
            ],
            "preselected": ["doc-1"],
            "exclude_tools": ["selectReferences", "listReferences"]
        })

        try:
            result = plugin._enrich_content("Tell me about the project", "prompt")
            assert "Transitively selected references" not in result.prompt
        finally:
            plugin.shutdown()

    def test_transitive_hint_metadata_structure(self):
        """Transitive metadata maps IDs to sorted parent lists."""
        plugin = create_plugin()

        plugin.initialize({
            "sources": [
                {
                    "id": "parent-a",
                    "name": "Parent A",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "See child-ref for details."
                },
                {
                    "id": "parent-b",
                    "name": "Parent B",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Also see child-ref here."
                },
                {
                    "id": "child-ref",
                    "name": "Child Reference",
                    "description": "Test",
                    "type": "inline",
                    "mode": "selectable",
                    "content": "Child content."
                }
            ],
            "preselected": ["parent-a", "parent-b"],
            "exclude_tools": ["selectReferences", "listReferences"]
        })

        try:
            result = plugin._enrich_content("Tell me about the project", "prompt")
            transitive = result.metadata["transitive_references"]
            assert "child-ref" in transitive
            # Both parents should be listed (as sorted list)
            assert sorted(transitive["child-ref"]) == ["parent-a", "parent-b"]
        finally:
            plugin.shutdown()


class TestTransitiveNotificationFormatting:
    """Tests for the registry's notification message formatting for transitive references."""

    def test_single_transitive_reference(self):
        """Single transitive reference notification."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        metadata = {
            "transitive_references": {
                "retry-ref": ["circuit-breaker-ref"]
            }
        }

        message = registry._generate_fallback_message("references", metadata)
        assert message is not None
        assert "transitively included" in message
        assert "@retry-ref" in message
        assert "@circuit-breaker-ref" in message

    def test_multiple_transitive_references(self):
        """Multiple transitive references notification."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        metadata = {
            "transitive_references": {
                "retry-ref": ["circuit-breaker-ref"],
                "timeout-ref": ["circuit-breaker-ref"]
            }
        }

        message = registry._generate_fallback_message("references", metadata)
        assert message is not None
        assert "transitively included" in message
        assert "@retry-ref" in message
        assert "@timeout-ref" in message
        assert "@circuit-breaker-ref" in message

    def test_many_transitive_references_truncated(self):
        """More than 3 transitive references are truncated with +N more."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        metadata = {
            "transitive_references": {
                "ref-a": ["parent"],
                "ref-b": ["parent"],
                "ref-c": ["parent"],
                "ref-d": ["parent"],
            }
        }

        message = registry._generate_fallback_message("references", metadata)
        assert message is not None
        assert "+1 more" in message

    def test_multiple_parents_in_notification(self):
        """Notification shows multiple parent sources."""
        from shared.plugins.registry import PluginRegistry
        registry = PluginRegistry()

        metadata = {
            "transitive_references": {
                "child-ref": ["parent-a", "parent-b"]
            }
        }

        message = registry._generate_fallback_message("references", metadata)
        assert message is not None
        assert "@parent-a" in message
        assert "@parent-b" in message
