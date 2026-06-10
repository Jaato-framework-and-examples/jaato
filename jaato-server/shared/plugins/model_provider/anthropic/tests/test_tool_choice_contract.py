"""Contract regression: AnthropicProvider.complete() must ACCEPT the
``tool_choice`` kwarg (and no-op it) per the ModelProvider.complete
contract in base.py — providers without a force-tool-choice wire quirk
must IGNORE the kwarg, not raise.

Repro (2026-06-10, glm-5-turbo via z.ai anthropic-compatible endpoint,
session 20260610_114613): the session passes tool_choice unconditionally;
AnthropicProvider's keyword-only signature had no tool_choice param and
no **kwargs, so it raised
``TypeError: complete() got an unexpected keyword argument 'tool_choice'``
— breaking every forced-completion stage (host_validator,
build_descriptor) on the GLM-5 cross-model cascade, which routes through
this adapter (api.z.ai/api/anthropic).
"""

import inspect

from shared.plugins.model_provider.anthropic.provider import AnthropicProvider


def test_complete_accepts_tool_choice_kwarg():
    sig = inspect.signature(AnthropicProvider.complete)
    params = sig.parameters
    accepts = (
        "tool_choice" in params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    )
    assert accepts, (
        "AnthropicProvider.complete() must accept (and ignore) the "
        "tool_choice kwarg per the base.py contract; without it the "
        "session's unconditional tool_choice pass raises TypeError."
    )


def test_tool_choice_is_optional_and_defaults_none():
    sig = inspect.signature(AnthropicProvider.complete)
    if "tool_choice" in sig.parameters:
        p = sig.parameters["tool_choice"]
        assert p.default is None
