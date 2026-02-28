"""Ollama Model Provider Plugin.

This provider uses Ollama's Anthropic-compatible API (v0.14.0+) to run
local models like Qwen, Llama, Mistral, and others.

Benefits:
- Run models locally without API costs
- Privacy - data never leaves your machine
- Use any model Ollama supports

Requirements:
- Ollama v0.14.0+ (for Anthropic API compatibility)
- A pulled model: `ollama pull qwen3:32b`

Configuration:
    Environment variables:
        OLLAMA_HOST: Server URL (default: http://localhost:11434)
        OLLAMA_MODEL: Default model name
        OLLAMA_CONTEXT_LENGTH: Override context length

    ProviderConfig.extra:
        host: Override OLLAMA_HOST
        context_length: Override context length

Example:
    from shared.plugins.model_provider.ollama import OllamaProvider

    provider = OllamaProvider()
    provider.initialize()  # No API key needed
    provider.connect('qwen3:32b')
    response = provider.complete(messages=[...])

    # Or with custom host:
    provider.initialize(ProviderConfig(extra={'host': 'http://remote:11434'}))
"""

from .env import (
    DEFAULT_OLLAMA_HOST,
    resolve_context_length,
    resolve_host,
    resolve_model,
)
from .provider import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaProvider,
    create_provider,
)

__all__ = [
    # Main provider
    "OllamaProvider",
    "create_provider",
    # Errors
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    # Environment
    "resolve_host",
    "resolve_model",
    "resolve_context_length",
    "DEFAULT_OLLAMA_HOST",
]
