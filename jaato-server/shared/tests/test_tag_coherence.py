"""Tests for the shared tag-coherence matching utility.

Verifies the algorithm used by both ``memory`` and ``references``
plugins for surfacing items based on tag matches against arbitrary
text.
"""

from shared.tag_coherence import (
    SEGMENT_MAX_CHARS,
    tag_coherent_in_segments,
    tag_components,
    text_segments,
)


def _segs(text):
    return text_segments(text)


class TestTagComponents:

    def test_split_on_hyphen(self):
        assert tag_components("workspace-baseline") == ["workspace", "baseline"]

    def test_split_on_underscore(self):
        assert tag_components("circuit_breaker") == ["circuit", "breaker"]

    def test_split_on_colon(self):
        assert tag_components("agent:main") == ["agent", "main"]

    def test_split_on_space(self):
        assert tag_components("circuit breaker") == ["circuit", "breaker"]

    def test_dot_kept_as_qualifier(self):
        """Dots are structural — keep tag atomic so it routes to strict
        regex matching."""
        assert tag_components("spring.boot") == ["spring.boot"]

    def test_short_components_dropped(self):
        assert tag_components("plan-v2") == ["plan"]

    def test_too_short_returns_empty(self):
        assert tag_components("gc") == []


class TestSingleComponentStrictMatching:

    def test_atomic_tag_matches_standalone_word(self):
        assert tag_coherent_in_segments("java", _segs("the Java SDK guide"))

    def test_atomic_tag_does_not_match_inside_dotted_name(self):
        assert not tag_coherent_in_segments(
            "java", _segs("Import java.util.concurrent.TimeoutException")
        )

    def test_atomic_tag_does_not_match_inside_path(self):
        assert not tag_coherent_in_segments(
            "java", _segs("file at /usr/lib/java/spec.json")
        )

    def test_atomic_tag_does_not_match_inside_file_extension(self):
        assert not tag_coherent_in_segments(
            "java", _segs("see Foo.java for details")
        )

    def test_atomic_tag_case_insensitive(self):
        assert tag_coherent_in_segments("JAVA", _segs("the java sdk"))
        assert tag_coherent_in_segments("java", _segs("the JAVA sdk"))

    def test_dotted_tag_does_not_match_inside_longer_name(self):
        """`spring.boot` is treated as atomic (1 component) and uses
        strict matching, so it doesn't match inside `org.spring.boot.x`."""
        assert not tag_coherent_in_segments(
            "spring.boot",
            _segs("Import org.spring.boot.autoconfigure from the classpath"),
        )

    def test_short_atomic_tag_matches(self):
        """Tags with no usable components fall to strict regex too."""
        assert tag_coherent_in_segments("gc", _segs("Investigate gc behaviour."))
        assert not tag_coherent_in_segments("gc", _segs("the gccompiler tool"))


class TestMultiComponentMatching:

    def test_verbatim_compound_match(self):
        assert tag_coherent_in_segments(
            "workspace-baseline", _segs("Show me the workspace-baseline.")
        )

    def test_components_co_occur_in_segment(self):
        assert tag_coherent_in_segments(
            "workspace-baseline", _segs("Show me the workspace baseline state.")
        )

    def test_components_split_across_sentences(self):
        assert not tag_coherent_in_segments(
            "workspace-baseline",
            _segs("What is the workspace? Specifically, the baseline state."),
        )

    def test_underscore_variant_matches_hyphen_tag(self):
        assert tag_coherent_in_segments(
            "circuit-breaker", _segs("Implement a circuit_breaker for the service")
        )

    def test_space_variant_matches_hyphen_tag(self):
        assert tag_coherent_in_segments(
            "circuit-breaker", _segs("Implement a circuit breaker pattern")
        )

    def test_hyphen_tag_matches_underscore_in_content(self):
        assert tag_coherent_in_segments(
            "circuit_breaker", _segs("Implement a circuit-breaker pattern")
        )

    def test_partial_components_does_not_match(self):
        assert not tag_coherent_in_segments(
            "workspace-baseline", _segs("Tell me about my workspace today.")
        )


class TestSegmentSizeCap:

    def test_long_segment_falls_back_to_verbatim_only(self):
        """Compact JSON dump (single long segment) shouldn't match
        compound tags by component coherence."""
        long_dump = "memory" + " x" * 200 + " system"  # >250 chars
        assert len(long_dump) > SEGMENT_MAX_CHARS
        assert not tag_coherent_in_segments("memory-system", _segs(long_dump))

    def test_short_segment_uses_components(self):
        assert tag_coherent_in_segments(
            "memory-system", _segs("the memory system overview")
        )
