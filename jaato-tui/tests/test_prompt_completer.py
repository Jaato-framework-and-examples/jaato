"""Tests for PercentPromptCompleter and PromptReferenceProcessor."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from file_completer import PercentPromptCompleter, PromptReferenceProcessor
from prompt_toolkit.document import Document


class TestPercentPromptCompleter:
    """Tests for PercentPromptCompleter."""

    def test_find_percent_position_at_start(self):
        """% at start of string should be found."""
        completer = PercentPromptCompleter()
        assert completer._find_percent_position("%prompt") == 0

    def test_find_percent_position_after_space(self):
        """% after space should be found."""
        completer = PercentPromptCompleter()
        assert completer._find_percent_position("Review %prompt") == 7

    def test_find_percent_position_invalid_after_alnum(self):
        """% after alphanumeric should be ignored."""
        completer = PercentPromptCompleter()
        # This looks like a format specifier or percent sign in a word
        assert completer._find_percent_position("100%done") == -1

    def test_find_percent_position_valid_after_space(self):
        """% after space with any suffix should be valid for completion."""
        completer = PercentPromptCompleter()
        # Even single char after % is valid - user might be typing prompt name
        assert completer._find_percent_position("value %d") == 6
        assert completer._find_percent_position("value %s") == 6

    def test_find_percent_position_multi_char(self):
        """Multi-char after % should be valid prompt reference."""
        completer = PercentPromptCompleter()
        # %code is not a format specifier
        assert completer._find_percent_position("use %code") == 4

    def test_completions_with_provider(self):
        """Should return completions from provider."""
        prompts = [
            {'name': 'code-review', 'description': 'Review code'},
            {'name': 'summarize', 'description': 'Summarize text'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        doc = Document(text="use %code", cursor_position=9)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        assert completions[0].text == 'code-review'
        # display_meta can be FormattedText or string - check content
        meta = completions[0].display_meta
        meta_str = meta[0][1] if hasattr(meta, '__getitem__') else str(meta)
        assert 'Review code' in meta_str

    def test_completions_without_provider(self):
        """Should return no completions without provider."""
        completer = PercentPromptCompleter()

        doc = Document(text="use %code", cursor_position=9)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 0

    def test_completions_all_prompts_on_percent(self):
        """Should show all prompts when just % is typed."""
        prompts = [
            {'name': 'code-review', 'description': 'Review code'},
            {'name': 'summarize', 'description': 'Summarize text'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        doc = Document(text="use %", cursor_position=5)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 2

    def test_completions_filter_by_prefix(self):
        """Should filter completions by typed prefix."""
        prompts = [
            {'name': 'code-review', 'description': 'Review code'},
            {'name': 'summarize', 'description': 'Summarize text'},
            {'name': 'create-test', 'description': 'Create test'},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        doc = Document(text="use %c", cursor_position=6)
        completions = list(completer.get_completions(doc, None))

        # Should match code-review and create-test
        assert len(completions) == 2
        names = [c.text for c in completions]
        assert 'code-review' in names
        assert 'create-test' in names

    def test_completions_truncate_long_description(self):
        """Should truncate descriptions longer than 50 chars."""
        prompts = [
            {'name': 'test', 'description': 'A' * 100},
        ]
        completer = PercentPromptCompleter(lambda: prompts)

        doc = Document(text="%", cursor_position=1)
        completions = list(completer.get_completions(doc, None))

        assert len(completions) == 1
        # display_meta can be FormattedText or string - check content
        meta = completions[0].display_meta
        meta_str = meta[0][1] if hasattr(meta, '__getitem__') else str(meta)
        assert len(meta_str) == 50
        assert meta_str.endswith('...')


class TestPromptReferenceProcessor:
    """Tests for PromptReferenceProcessor."""

    def test_find_references_single(self):
        """Should find single %prompt reference."""
        processor = PromptReferenceProcessor()
        refs = processor.find_references("Use %code-review")

        assert len(refs) == 1
        assert refs[0]['name'] == 'code-review'
        assert refs[0]['reference'] == '%code-review'

    def test_find_references_multiple(self):
        """Should find multiple %prompt references."""
        processor = PromptReferenceProcessor()
        refs = processor.find_references("Use %code-review and %summarize")

        assert len(refs) == 2
        names = [r['name'] for r in refs]
        assert 'code-review' in names
        assert 'summarize' in names

    def test_find_references_with_file_ref(self):
        """Should find %prompt alongside @file references."""
        processor = PromptReferenceProcessor()
        refs = processor.find_references("Review @main.py using %code-review")

        assert len(refs) == 1
        assert refs[0]['name'] == 'code-review'

    def test_find_references_pattern_matching(self):
        """Should find prompts matching the pattern (letter followed by alphanumeric)."""
        processor = PromptReferenceProcessor()
        # %d matches (single letter is valid prompt name)
        # %deploy matches
        # %123 does NOT match (starts with number)
        refs = processor.find_references("Use %deploy and %d but not %123")

        assert len(refs) == 2
        names = [r['name'] for r in refs]
        assert 'deploy' in names
        assert 'd' in names

    def test_find_references_skip_after_alnum(self):
        """Should skip % after alphanumeric characters."""
        processor = PromptReferenceProcessor()
        refs = processor.find_references("100%complete")

        assert len(refs) == 0

    def test_expand_references_without_expander(self):
        """Without expander, should return original text."""
        processor = PromptReferenceProcessor()
        text = "Use %code-review"
        result = processor.expand_references(text)

        assert result == text

    def test_expand_references_with_expander(self):
        """With expander, should expand prompt content."""
        def expander(name, params):
            return f"Expanded: {name}"

        processor = PromptReferenceProcessor(expander)
        result = processor.expand_references("Use %code-review")

        assert "code-review" in result
        assert "--- Referenced Prompts ---" in result
        assert "Expanded: code-review" in result

    def test_expand_references_removes_percent(self):
        """Should remove % from prompt names in main text."""
        def expander(name, params):
            return f"Content"

        processor = PromptReferenceProcessor(expander)
        result = processor.expand_references("Use %code-review now")

        # Main text should have prompt name without %
        assert result.startswith("Use code-review now")

    def test_expand_references_unknown_prompt(self):
        """Unknown prompts should have % stripped but no expansion."""
        def expander(name, params):
            return None  # Prompt not found

        processor = PromptReferenceProcessor(expander)
        result = processor.expand_references("Use %unknown")

        # Should strip % but not add expansion section
        assert result == "Use unknown"
        assert "--- Referenced Prompts ---" not in result

    def test_expand_references_preserves_order(self):
        """Multiple prompts should be expanded in order."""
        def expander(name, params):
            return f"Content of {name}"

        processor = PromptReferenceProcessor(expander)
        result = processor.expand_references("First %alpha then %beta")

        # Check order in expanded section
        alpha_pos = result.find("[Prompt: alpha]")
        beta_pos = result.find("[Prompt: beta]")
        assert alpha_pos < beta_pos

    def test_expand_references_expander_exception(self):
        """Expander exception should be handled gracefully."""
        def expander(name, params):
            raise RuntimeError("Oops")

        processor = PromptReferenceProcessor(expander)
        result = processor.expand_references("Use %broken")

        # Should strip % but not add expansion (exception caught)
        assert result == "Use broken"
