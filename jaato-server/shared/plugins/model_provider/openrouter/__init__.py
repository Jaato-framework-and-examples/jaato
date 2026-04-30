"""OpenRouter model provider plugin.

OpenRouter (https://openrouter.ai) is a unified gateway that exposes
hundreds of models from many vendors (OpenAI, Anthropic, Google, Meta,
Mistral, DeepSeek, ...) behind a single OpenAI-compatible API.

Authentication:
- Set ``JAATO_OPENROUTER_API_KEY`` (sk-or-...) from
  https://openrouter.ai/settings/keys
- Or run ``openrouter-auth key <api_key>`` to store it persistently.
"""

from .provider import OpenRouterProvider, create_provider

__all__ = ["OpenRouterProvider", "create_provider"]
