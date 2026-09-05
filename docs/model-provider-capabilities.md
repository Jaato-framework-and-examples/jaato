# Model-Provider Capability Matrix

<!-- AUTO-GENERATED from PROVIDER_CAPABILITIES declarations — edit the declarations, not this file. Regenerate: `python -m shared.tests.test_provider_capabilities`. -->

Each cell is a **wire-level** behavior the CI conformance guard asserts, not a label. A provider declares its row via `PROVIDER_CAPABILITIES = ProviderCapabilities(...)` in its `__init__.py`; the guard fails the build if a declared capability isn't actually delivered on the wire.

| Provider | user_message_images | tool_result_images | pdf_input | tool_choice_forwarding | thinking | prompt_caching | streaming | cancellation |
|---|---|---|---|---|---|---|---|---|
| `anthropic` | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ |
| `antigravity` | ✅ | — | — | — | ✅ | — | ✅ | ✅ |
| `chrome_ai` | — | — | — | — | — | — | ✅ | ✅ |
| `claude_cli` | — | — | — | — | ✅ | — | ✅ | — |
| `doubleword` | ✅ | ✅ | — | ✅ | ✅ | — | ✅ | ✅ |
| `github_models` | — | — | — | — | ✅ | — | ✅ | ✅ |
| `google_genai` | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | ✅ |
| `helmcode` | ✅ | ✅ | — | ✅ | ✅ | — | ✅ | ✅ |
| `lmstudio` | ✅ | ✅ | — | — | — | — | ✅ | ✅ |
| `nebius` | ✅ | ✅ | — | ✅ | ✅ | — | ✅ | ✅ |
| `nim` | ✅ | ✅ | — | — | ✅ | — | ✅ | ✅ |
| `ollama` | ✅ | ✅ | — | — | — | — | ✅ | ✅ |
| `openrouter` | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ |
| `ovhcloud` | ✅ | ✅ | — | ✅ | ✅ | — | ✅ | ✅ |
| `tensorrt_llm` | ✅ | ✅ | — | — | — | — | ✅ | ✅ |
| `triton` | ✅ | ✅ | — | — | — | — | ✅ | ✅ |
| `vllm` | ✅ | ✅ | — | ✅ | — | — | ✅ | ✅ |
| `zhipuai` | ✅ | ✅ | — | — | ✅ | — | ✅ | ✅ |
| `zhipuai_openai` | ✅ | ✅ | — | — | ✅ | — | ✅ | ✅ |

**Legend:** ✅ = implemented & verified on the wire by the conformance guard · — = not implemented (text-only / not forwarded).
