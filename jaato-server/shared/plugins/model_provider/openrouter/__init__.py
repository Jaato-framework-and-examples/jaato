"""OpenRouter model provider plugin (WIP).

OpenRouter (https://openrouter.ai) is a unified gateway that exposes
hundreds of models from many vendors (OpenAI, Anthropic, Google, Meta,
Mistral, DeepSeek, ...) behind a single OpenAI-compatible API.

Authentication:
- Set ``JAATO_OPENROUTER_API_KEY`` (sk-or-...) from
  https://openrouter.ai/keys
- Or run ``openrouter-auth key <api_key>`` to store it persistently.

NOTE: ``provider.py`` is not yet implemented — this package currently
ships only the supporting modules (env, errors, converters, auth).
The provider class and ``create_provider`` factory will land in a
follow-up commit, at which point the imports below will be uncommented
and the package will become discoverable by the registry.
"""

# from .provider import OpenRouterProvider, create_provider
#
# __all__ = ["OpenRouterProvider", "create_provider"]
__all__: list = []
