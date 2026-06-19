"""Echo Model Provider Plugin.

A deterministic, credentials-free, network-free model provider for end-to-end
tests of the agentic loop and the gossip remote-spawn path.  It returns fixed
responses (or echoes the prompt verbatim) so a test can drive the real chat
loop without a live model.

Configuration via ``ProviderConfig.extra`` (profile ``plugin_configs.echo``):
    response:   optional str returned verbatim in text mode
    tool_call:  optional dict ``{"name": str, "args": dict}`` emitted as a
                single function call on the first turn

Example:
    from shared.plugins.model_provider.echo import EchoProvider

    provider = EchoProvider()
    provider.initialize(ProviderConfig(extra={"response": "hi"}))
    provider.connect("echo")
    result = provider.complete(messages=[...])
"""

PLUGIN_KIND = "model_provider"
PLUGIN_TIER = "daemon"

from .provider import (
    ECHO_CONTEXT_LIMIT,
    ECHO_MODEL,
    EchoProvider,
    create_provider,
)

__all__ = [
    "EchoProvider",
    "create_provider",
    "ECHO_CONTEXT_LIMIT",
    "ECHO_MODEL",
    "PLUGIN_KIND",
    "PLUGIN_TIER",
]
