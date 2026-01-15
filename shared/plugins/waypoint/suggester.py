"""Description suggester for waypoints.

Uses the model to generate brief, meaningful waypoint descriptions based
on the current conversation context.
"""

from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..model_provider.types import Message


# Prompt template for generating waypoint descriptions
SUGGESTION_PROMPT = """Based on the recent conversation, suggest a brief waypoint description (3-6 words) that captures the current state or recent work.

The description should be:
- Concise (3-6 words)
- Action-oriented (e.g., "auth flow working", "before database refactor")
- Descriptive of what was accomplished or what state we're in

Examples of good descriptions:
- "auth flow working"
- "before database refactor"
- "API endpoints complete"
- "fixing test failures"
- "initial project setup"
- "user registration done"

Respond with ONLY the description text, no quotes, no explanation."""


class DescriptionSuggester:
    """Generates waypoint description suggestions using the model.

    This class provides functionality to generate contextual waypoint
    descriptions based on the current conversation state. Suggestions
    are generated asynchronously and can be used for ghost text in the
    command input.
    """

    def __init__(self):
        """Initialize the description suggester."""
        self._get_history: Optional[Callable[[], List["Message"]]] = None
        self._send_message: Optional[Callable[[str], str]] = None
        self._cached_suggestion: Optional[str] = None

    def set_callbacks(
        self,
        get_history: Callable[[], List["Message"]],
        send_message: Callable[[str], str],
    ) -> None:
        """Set callbacks for accessing session state and sending messages.

        Args:
            get_history: Returns current conversation history.
            send_message: Sends a message to the model and returns response.
        """
        self._get_history = get_history
        self._send_message = send_message

    def suggest(self) -> Optional[str]:
        """Generate a waypoint description suggestion.

        Uses the current conversation context to generate a brief,
        meaningful description.

        Returns:
            Suggested description string, or None if unable to generate.
        """
        if not self._send_message:
            return self._generate_fallback()

        try:
            # Send the suggestion prompt to the model
            response = self._send_message(SUGGESTION_PROMPT)

            if response:
                # Clean up the response
                suggestion = self._clean_suggestion(response)
                self._cached_suggestion = suggestion
                return suggestion

        except Exception:
            pass

        return self._generate_fallback()

    def suggest_async(
        self,
        callback: Callable[[Optional[str]], None],
    ) -> None:
        """Generate a suggestion asynchronously.

        The callback is called with the suggestion when ready, or None
        if generation failed.

        Args:
            callback: Function to call with the generated suggestion.
        """
        # For now, just call synchronously
        # In the future, this could use threading or async
        try:
            suggestion = self.suggest()
            callback(suggestion)
        except Exception:
            callback(None)

    def get_cached_suggestion(self) -> Optional[str]:
        """Get the most recently generated suggestion.

        Returns:
            The cached suggestion, or None if no suggestion has been generated.
        """
        return self._cached_suggestion

    def _clean_suggestion(self, response: str) -> str:
        """Clean up the model's response to get a clean description.

        Args:
            response: Raw model response.

        Returns:
            Cleaned description string.
        """
        # Remove quotes if present
        cleaned = response.strip()
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        # Remove any trailing punctuation
        cleaned = cleaned.rstrip('.')

        # Limit length
        words = cleaned.split()
        if len(words) > 8:
            cleaned = ' '.join(words[:6]) + '...'

        return cleaned

    def _generate_fallback(self) -> str:
        """Generate a fallback description when model is unavailable.

        Returns:
            A timestamp-based fallback description.
        """
        from datetime import datetime
        return datetime.now().strftime("checkpoint %H:%M")


def create_suggester() -> DescriptionSuggester:
    """Factory function to create a DescriptionSuggester instance."""
    return DescriptionSuggester()
