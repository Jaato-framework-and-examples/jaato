"""Shared OpenAI-compatible provider infrastructure.

Support package (NOT a model-provider plugin — the leading underscore keeps it
out of provider-directory enumeration, and it declares no ``PLUGIN_KIND``).

Holds the OpenAI-compat machinery shared across the self-hosted / hosted
OpenAI-API providers (nim, nebius, vllm, lmstudio, tensorrt_llm, triton,
zhipuai_openai):

- ``converters``: provider-agnostic Message↔OpenAI / ToolSchema↔OpenAI
  conversion + usage extraction (hashes tool names via ``shared.tool_id_map``).
- ``_lazy``: lazy import of the ``openai`` SDK client class + module.

A forthcoming ``base.OpenAICompatProvider`` (the incremental de-dup of the
fleet's near-identical ``provider.py`` machinery) will live here too.
"""
