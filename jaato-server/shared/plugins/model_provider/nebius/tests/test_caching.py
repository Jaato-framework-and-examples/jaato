"""Nebius caching support: cached_tokens measurement + extra_body passthrough.

Nebius Token Factory runs vLLM backends with automatic Context Caching, and is
OpenAI-compatible.  These tests pin the two pieces that surface / reach that:

- ``cached_tokens_from`` reads the OpenAI ``usage.prompt_tokens_details
  .cached_tokens`` field into ``TokenUsage.cache_read_tokens`` so cache hit-rate
  and $ savings are measurable (the measurement half of usage-observability).
- ``extra_body`` is forwarded verbatim to ``chat.completions.create`` so
  endpoint-specific knobs the OpenAI signature doesn't name (guided decoding,
  ``cache_salt``) reach Nebius's vLLM backend.
"""

from types import SimpleNamespace as NS

from shared.plugins.model_provider.nebius.converters import (
    cached_tokens_from,
    extract_usage,
)
from shared.plugins.model_provider.nebius.provider import NebiusProvider


# --------------------------------------------------------- cached_tokens

def test_cached_tokens_from_reads_prompt_tokens_details():
    assert cached_tokens_from(NS(prompt_tokens_details=NS(cached_tokens=312000))) == 312000


def test_cached_tokens_from_none_when_absent_or_zero():
    assert cached_tokens_from(NS(prompt_tokens_details=None)) is None
    assert cached_tokens_from(NS()) is None                       # no cache hit -> no field
    assert cached_tokens_from(NS(prompt_tokens_details=NS(cached_tokens=0))) is None


def test_extract_usage_maps_cached_into_cache_read_tokens():
    resp = NS(usage=NS(prompt_tokens=660000, completion_tokens=387,
                       total_tokens=660387,
                       prompt_tokens_details=NS(cached_tokens=312000)))
    tu = extract_usage(resp)
    assert tu.prompt_tokens == 660000
    assert tu.output_tokens == 387
    assert tu.cache_read_tokens == 312000


def test_extract_usage_no_cache_field_leaves_cache_read_none():
    resp = NS(usage=NS(prompt_tokens=100, completion_tokens=10, total_tokens=110))
    assert extract_usage(resp).cache_read_tokens is None


# ------------------------------------------------------------ extra_body

def test_extra_body_forwarded_to_create_kwargs():
    p = NebiusProvider()
    p._extra_body = {"guided_json": {"type": "object"}}
    kwargs = {"tools": [object()]}
    p._apply_api_params(kwargs, None)
    assert kwargs["extra_body"] == {"guided_json": {"type": "object"}}


def test_extra_body_absent_when_not_configured():
    p = NebiusProvider()
    kwargs = {}
    p._apply_api_params(kwargs, None)
    assert "extra_body" not in kwargs
