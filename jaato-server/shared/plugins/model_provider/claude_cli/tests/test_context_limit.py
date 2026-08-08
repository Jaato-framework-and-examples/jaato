"""Context-window resolution for the Claude CLI provider.

Part of the provider context-window standardization (resolve_context_window):
the CLI serves Claude models (200K documented window — the authoritative value
for its entire model space), overridable via the context_length knob.
"""

from shared.plugins.model_provider.claude_cli.provider import ClaudeCLIProvider


def test_default_is_claude_200k():
    """No override → the documented Claude 200K window."""
    provider = ClaudeCLIProvider()
    assert provider.get_context_limit() == 200_000


def test_override_knob_wins():
    """context_length override (e.g. GC budget-down) beats the default."""
    provider = ClaudeCLIProvider()
    provider._context_length_knob = 50_000
    assert provider.get_context_limit() == 50_000
