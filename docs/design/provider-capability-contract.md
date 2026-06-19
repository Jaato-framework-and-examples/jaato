# Model-Provider Capability Contract

Every model provider exposes the **same API** for the **same functionality**, and
CI enforces it — so a provider can never silently claim a capability it doesn't
deliver (the failure mode that let the multimodal wire-marshalling rot
undetected: providers *declared* `image: yes` while their converters dropped the
bytes).

This mirrors the tool-plugin tier contract (`PLUGIN_TIER` + `contract-guards`
CI) for providers.

## The three pieces

| Piece | Where | What it does |
|---|---|---|
| **Declaration** | `model_provider/<p>/__init__.py` → `PROVIDER_CAPABILITIES = ProviderCapabilities(...)` | Each provider declares, per capability, whether it implements it. The single source of truth. |
| **Doc table** | `docs/model-provider-capabilities.md` (auto-generated) | The capability matrix. Regenerate: `python -m shared.tests.test_provider_capabilities`. |
| **Guard** | `contract-guards` CI → `test_provider_capabilities.py` (structural + doc-drift) + `test_provider_capability_conformance.py` (behavioral) | Fails the PR if a provider doesn't declare, the doc drifts, or a declared capability isn't delivered on the wire. |

## The capabilities (`base.py:CAPABILITY_FIELDS`)

Each is a **concrete, wire-testable behavior**, not a label. To add a provider
that supports one, you implement the listed mechanism; if you don't, declare it
`False`.

| Capability | Contract — how a provider implements it | How the guard verifies |
|---|---|---|
| `user_message_images` | Message-→-wire converter turns a `Part.inline_data` image into the provider's image block (OpenAI `image_url` data-URL, Anthropic `image` `source.base64`, …). | Behavioral: feed an image Message through the converter; the image's base64 must appear on the wire. |
| `tool_result_images` | A tool result's image `Attachment` reaches the model. OpenAI-compat `tool` messages can't carry images, so surface them as a **follow-up `user` message** with `image_url`; Anthropic embeds an image block in `tool_result`. | Behavioral: feed a `ToolResult` with an image attachment; base64 must reach the wire. |
| `tool_choice_forwarding` | `complete(tool_choice=…)` is forwarded to the request body (e.g. `api_params.tool_choice` → wire). Accept-and-ignore counts as **`False`**. | (v1: declaration; behavioral follow-up.) |
| `thinking` | `supports_thinking()` consistent with behavior; extended-reasoning requested and/or `reasoning_content` extracted. | (v1: declaration.) |
| `prompt_caching` | Emits `cache_control` breakpoints on the wire (directly or via the cache plugin). Parsing cached-token *accounting* without emitting breakpoints is **`False`**. | (v1: declaration.) |
| `streaming` | `complete(on_chunk=…)` streams tokens. | (v1: declaration; `supports_streaming()`.) |
| `cancellation` | A `cancel_token` **actually halts** generation. Advertising `supports_stop()==True` while ignoring the token is a **lie → declare `False`**. | (v1: declaration; behavioral follow-up.) |

> Modality detection (`modalities()`) and context-window detection
> (`resolve_context_window`) remain their own richer primitives (a *set* / an
> *int*, not a bool) and are not in this bool matrix. **Cross-rule:** if a
> provider's `modalities()` can return `image` for any model, it MUST declare
> `user_message_images=True` — otherwise it's vision-declared-but-broken.

## Adding a provider

1. Implement the provider as usual.
2. Declare `PROVIDER_CAPABILITIES = ProviderCapabilities(<every field>=<bool>)` in
   its `__init__.py` (the structural guard fails the build otherwise — same
   "annotate or be excluded" gate as `PLUGIN_TIER`).
3. If it has a dict-producing message converter, add it to `_CONVERTERS` in
   `test_provider_capability_conformance.py` so its image claims are behaviorally
   verified; otherwise add it to `_CONFORMANCE_PENDING` (the coverage test
   forbids silently skipping a provider).
4. `python -m shared.tests.test_provider_capabilities` to regenerate the doc.

## Adding a capability

1. Add the field to `ProviderCapabilities` and `CAPABILITY_FIELDS` (`base.py`).
   The structural guard immediately fails every provider until each declares it.
2. Declare it on every provider per its real behavior.
3. Add a behavioral assertion to the conformance guard where testable.
4. Regenerate the doc.
