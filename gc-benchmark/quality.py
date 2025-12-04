"""Context quality measurement via fact retention testing."""

from typing import Dict, List

from google.genai import types

from metrics import FactRetentionResult, QualityMetrics
from scenarios import EmbeddedFact


class QualityTester:
    """Tests context quality after GC using fact retention.

    Asks the model questions about embedded facts using the post-GC
    history as context. Measures what percentage of facts are retained.
    """

    def __init__(self, generate_fn):
        """Initialize quality tester.

        Args:
            generate_fn: A callable that takes (history, prompt) and returns
                         a model response string. This allows the tester to
                         work with any LLM interface.
        """
        self._generate = generate_fn

    def test_fact_retention(
        self,
        post_gc_history: List[types.Content],
        embedded_facts: List[EmbeddedFact],
        verbose: bool = False
    ) -> QualityMetrics:
        """Test how many embedded facts can be recalled from post-GC context.

        Args:
            post_gc_history: Conversation history after GC.
            embedded_facts: Facts that were embedded in the original conversation.
            verbose: Whether to print progress.

        Returns:
            QualityMetrics with retention statistics.
        """
        if not embedded_facts:
            return QualityMetrics.empty()

        results: List[FactRetentionResult] = []

        for fact in embedded_facts:
            if verbose:
                print(f"  Testing fact: {fact.fact_id}...")

            actual_answer = self._ask_fact_question(
                post_gc_history,
                fact.verification_question
            )

            retained, confidence = self._evaluate_answer(
                fact.expected_answer,
                actual_answer
            )

            results.append(FactRetentionResult(
                fact_id=fact.fact_id,
                category=fact.category,
                turn_index=fact.turn_index,
                question=fact.verification_question,
                expected_answer=fact.expected_answer,
                actual_answer=actual_answer,
                retained=retained,
                confidence=confidence
            ))

            if verbose:
                status = "RETAINED" if retained else "LOST"
                print(f"    [{status}] Expected: {fact.expected_answer}, Got: {actual_answer}")

        return self._calculate_metrics(results, embedded_facts)

    def _ask_fact_question(
        self,
        history: List[types.Content],
        question: str
    ) -> str:
        """Ask a single fact question using the history as context.

        Args:
            history: Post-GC conversation history.
            question: The fact verification question.

        Returns:
            Model's answer string.
        """
        prompt = (
            f"Based on our conversation above, please answer this question "
            f"briefly and directly:\n\n{question}\n\n"
            f"Answer with just the relevant fact, no explanation needed."
        )

        try:
            response = self._generate(history, prompt)
            return response.strip()
        except Exception as e:
            return f"[ERROR: {str(e)}]"

    def _evaluate_answer(
        self,
        expected: str,
        actual: str
    ) -> tuple:
        """Evaluate if the actual answer matches expected.

        Uses fuzzy matching to handle minor variations:
        - Case insensitive comparison
        - Expected value contained in actual answer
        - Handles currency/number formatting variations

        Args:
            expected: Expected answer.
            actual: Actual model response.

        Returns:
            Tuple of (retained: bool, confidence: float)
        """
        if not actual or actual.startswith("[ERROR"):
            return False, 0.0

        expected_lower = expected.lower().strip()
        actual_lower = actual.lower().strip()

        # Exact match
        if expected_lower == actual_lower:
            return True, 1.0

        # Expected contained in actual (model gave more context)
        if expected_lower in actual_lower:
            return True, 0.9

        # Handle number/currency variations
        expected_normalized = self._normalize_value(expected_lower)
        actual_normalized = self._normalize_value(actual_lower)

        if expected_normalized and expected_normalized in actual_normalized:
            return True, 0.85

        # Partial match for entities (at least all words present)
        expected_words = set(expected_lower.split())
        actual_words = set(actual_lower.split())
        if expected_words and expected_words.issubset(actual_words):
            return True, 0.8

        return False, 0.0

    def _normalize_value(self, value: str) -> str:
        """Normalize a value for comparison.

        Handles currency symbols, commas, common variations.
        """
        # Remove common punctuation and formatting
        normalized = value.replace("$", "").replace(",", "").replace(".", "")
        normalized = normalized.replace("-", " ").strip()
        return normalized

    def _calculate_metrics(
        self,
        results: List[FactRetentionResult],
        embedded_facts: List[EmbeddedFact]
    ) -> QualityMetrics:
        """Calculate quality metrics from individual fact results."""
        facts_tested = len(results)
        facts_retained = sum(1 for r in results if r.retained)
        retention_rate = facts_retained / facts_tested if facts_tested > 0 else 0.0

        # By category
        retention_by_category = self._calculate_by_category(results)

        # By position (requires knowing total turns)
        retention_by_position = self._calculate_by_position(results, embedded_facts)

        return QualityMetrics(
            facts_tested=facts_tested,
            facts_retained=facts_retained,
            retention_rate=retention_rate,
            retention_by_category=retention_by_category,
            retention_by_position=retention_by_position,
            fact_results=results
        )

    def _calculate_by_category(
        self,
        results: List[FactRetentionResult]
    ) -> Dict[str, float]:
        """Calculate retention rate by fact category."""
        by_category: Dict[str, List[bool]] = {}

        for result in results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result.retained)

        return {
            category: sum(retained) / len(retained) if retained else 0.0
            for category, retained in by_category.items()
        }

    def _calculate_by_position(
        self,
        results: List[FactRetentionResult],
        embedded_facts: List[EmbeddedFact]
    ) -> Dict[str, float]:
        """Calculate retention rate by position in conversation.

        Position categories:
        - early: first third of turns
        - middle: middle third of turns
        - late: last third of turns
        """
        if not results:
            return {}

        # Find max turn index to determine boundaries
        max_turn = max(r.turn_index for r in results)
        third = max(1, max_turn // 3)

        by_position: Dict[str, List[bool]] = {
            "early": [],
            "middle": [],
            "late": []
        }

        for result in results:
            if result.turn_index < third:
                position = "early"
            elif result.turn_index < third * 2:
                position = "middle"
            else:
                position = "late"

            by_position[position].append(result.retained)

        return {
            position: sum(retained) / len(retained) if retained else 0.0
            for position, retained in by_position.items()
            if retained  # Only include positions that have facts
        }
