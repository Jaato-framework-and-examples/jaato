# JAATO Multi-Provider Abstraction Layer

## Executive Summary

JAATO's **multi-provider abstraction** enables a single codebase to interact with any AI model provider -- Google GenAI (Vertex AI / Gemini), Anthropic (Claude), GitHub Models, Google Antigravity, Ollama, ZhipuAI, and the Claude CLI -- through a unified interface. The abstraction is built on a **provider-agnostic type system** (`ToolSchema`, `Message`, `ProviderResponse`, etc.) and a **protocol-based plugin contract** (`ModelProviderPlugin`) that each provider implements. Providers are **discovered dynamically** via entry points or directory scanning, and sessions can **mix providers** within the same runtime, enabling cross-provider subagents.

---

## Part 1: Why a Multi-Provider Abstraction?

### The Problem

Each AI SDK has its own types, authentication flows, tool calling formats, streaming interfaces, and error handling. Building directly against any single SDK creates vendor lock-in and makes multi-model strategies impossible.

```
Without Abstraction:
┌───────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Application Code                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ google.genai.  │  │ anthropic.     │  │ azure.ai.         │  │
│  │ types.Content  │  │ types.Message  │  │ inference.models  │  │
│  │ types.FuncDecl │  │ types.ToolUse  │  │ ChatCompletions   │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
│                                                                    │
│  Problem: Different types, different APIs, different error models  │
│  Switching providers requires rewriting orchestration logic         │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘

With JAATO Abstraction:
┌───────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Application Code (uses only provider-agnostic types)              │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  ToolSchema · Message · ProviderResponse · CancelToken     │   │
│  └─────────────────────────┬──────────────────────────────────┘   │
│                             │                                      │
│                    ModelProviderPlugin (Protocol)                   │
│                             │                                      │
│           ┌─────────────────┼─────────────────┐                   │
│           ▼                 ▼                  ▼                   │
│  ┌────────────────┐ ┌──────────────┐ ┌────────────────────┐      │
│  │ Google GenAI   │ │  Anthropic   │ │  GitHub Models     │      │
│  │ Provider       │ │  Provider    │ │  Provider          │      │
│  └────────────────┘ └──────────────┘ └────────────────────┘      │
│           ▲                 ▲                  ▲                   │
│           │                 │                  │                   │
│  google-genai SDK   anthropic SDK    azure-ai-inference SDK       │
│                                                                    │
└───────────────────────────────────────────────────────────────────┘
```

### What the Abstraction Provides

| Concern | Without Abstraction | With Abstraction |
|---------|-------------------|-----------------|
| **Tool schemas** | Provider-specific formats (FunctionDeclaration, ToolUseBlock, etc.) | Single `ToolSchema` dataclass converted by each provider |
| **Messages** | `Content`, `MessageParam`, `ChatRequestMessage` | Single `Message` with `Role` and `Part` list |
| **Streaming** | SDK-specific iterators and callbacks | Unified `StreamingCallback`, `CancelToken` |
| **Token counting** | Different APIs and return formats | `TokenUsage` dataclass, `count_tokens()` method |
| **Auth** | API keys, OAuth, service accounts, ADC | `ProviderConfig` + `verify_auth()` with interactive support |
| **Error handling** | Provider-specific exceptions | `classify_error()` + `get_retry_after()` |

---

## Part 2: Provider-Agnostic Type System

All orchestration logic in JAATO operates on these internal types, defined in `shared/plugins/model_provider/types.py`:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROVIDER-AGNOSTIC TYPE SYSTEM                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   CONVERSATION TYPES                                         │    │
│  │                                                              │    │
│  │   Role (Enum)        USER | MODEL | TOOL                    │    │
│  │   Part (Dataclass)   text | function_call | function_response│    │
│  │                      | inline_data | thought                 │    │
│  │   Message            role + parts[] + message_id             │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   TOOL TYPES                                                 │    │
│  │                                                              │    │
│  │   ToolSchema         name + description + parameters (JSON   │    │
│  │                      Schema) + category + discoverability    │    │
│  │   FunctionCall       id + name + args                        │    │
│  │   ToolResult         call_id + name + result + is_error      │    │
│  │   Attachment         mime_type + data + display_name          │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   RESPONSE TYPES                                             │    │
│  │                                                              │    │
│  │   ProviderResponse   parts[] + usage + finish_reason + raw   │    │
│  │                      + structured_output + thinking          │    │
│  │   TokenUsage         prompt_tokens + output_tokens +         │    │
│  │                      total_tokens + cache_read/creation +    │    │
│  │                      reasoning_tokens                        │    │
│  │   FinishReason       STOP | MAX_TOKENS | TOOL_USE | SAFETY  │    │
│  │                      | ERROR | CANCELLED | UNKNOWN           │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   CONTROL TYPES                                              │    │
│  │                                                              │    │
│  │   CancelToken        Thread-safe cancellation signaling      │    │
│  │                      cancel() / is_cancelled / wait() /      │    │
│  │                      on_cancel() / raise_if_cancelled()      │    │
│  │   ThinkingConfig     enabled + budget (provider-agnostic)    │    │
│  │                                                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `Part`-based messages | Preserves interleaving of text and function calls in their original order |
| `message_id` on Message | Enables GC to track which messages have been collected |
| `raw` field on ProviderResponse | Allows provider-specific features without losing abstraction |
| `Attachment` type | Enables multimodal tool results (images, files) across providers |
| `CancelToken` | Thread-safe cancellation that works identically across all providers |

---

## Part 3: The ModelProviderPlugin Protocol

Each provider implements a Python `Protocol` (structural typing) with 20+ methods organized into six groups:

```
┌─────────────────────────────────────────────────────────────────────┐
│              ModelProviderPlugin Protocol                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LIFECYCLE                                                     │   │
│  │  initialize(config) → None      Set up SDK client             │   │
│  │  verify_auth(interactive) → bool  Check/trigger auth          │   │
│  │  shutdown() → None               Cleanup resources            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ CONNECTION                                                    │   │
│  │  connect(model) → None          Select model                  │   │
│  │  is_connected → bool            Connection status             │   │
│  │  model_name → str               Current model                 │   │
│  │  list_models(prefix) → [str]    Available models              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ SESSION MANAGEMENT                                            │   │
│  │  create_session(instruction, tools, history) → None           │   │
│  │  get_history() → [Message]      Provider-agnostic history     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ MESSAGING                                                     │   │
│  │  generate(prompt) → Response         One-shot generation      │   │
│  │  send_message(msg, schema) → Response  Conversational         │   │
│  │  send_message_with_parts(parts) → Response  Multimodal        │   │
│  │  send_tool_results(results) → Response  Return tool output    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ STREAMING & CANCELLATION                                      │   │
│  │  supports_streaming() → bool                                  │   │
│  │  send_message_streaming(msg, on_chunk, cancel_token,          │   │
│  │      on_usage_update, on_function_call, on_thinking)          │   │
│  │  send_tool_results_streaming(...)                             │   │
│  │  supports_stop() → bool                                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ CAPABILITIES & TOKENS                                         │   │
│  │  count_tokens(content) → int    Token counting                │   │
│  │  get_context_limit() → int      Context window size           │   │
│  │  get_token_usage() → TokenUsage Last response usage           │   │
│  │  supports_structured_output() → bool                          │   │
│  │  supports_thinking() → bool                                   │   │
│  │  set_thinking_config(config) → None                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ ERROR CLASSIFICATION                                          │   │
│  │  classify_error(exc) → {transient, rate_limit, infra}         │   │
│  │  get_retry_after(exc) → float   Extract retry-after hint      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Callback Types

The streaming interface uses six callback types, all provider-agnostic:

| Callback | Signature | Purpose |
|----------|-----------|---------|
| `OutputCallback` | `(source, text, mode)` | Real-time output routing (model/plugin/system) |
| `StreamingCallback` | `(chunk)` | Token-level text streaming |
| `ThinkingCallback` | `(thinking)` | Extended thinking content (before text) |
| `UsageUpdateCallback` | `(TokenUsage)` | Real-time token accounting |
| `GCThresholdCallback` | `(percent, threshold)` | Proactive GC notification |
| `FunctionCallDetectedCallback` | `(FunctionCall)` | Mid-stream tool call detection |

---

## Part 4: Provider Inventory

JAATO ships with seven provider implementations:

```
shared/plugins/model_provider/
├── google_genai/        # Google GenAI (Vertex AI + AI Studio)
├── anthropic/           # Anthropic Claude API
├── github_models/       # GitHub Models (azure-ai-inference SDK)
├── antigravity/         # Google Antigravity IDE backend
├── claude_cli/          # Claude Code CLI wrapper
├── ollama/              # Ollama local models
└── zhipuai/             # ZhipuAI (GLM models)
```

### Provider Comparison

| Provider | Auth Methods | Streaming | Thinking | Caching | Structured Output | Tool Calling |
|----------|-------------|-----------|----------|---------|-------------------|--------------|
| **Google GenAI** | API Key, ADC, Service Account, Impersonation | Yes | Yes (Gemini 2.0+) | No | Yes | Native |
| **Anthropic** | API Key, OAuth (PKCE), OAuth Token | Yes | Yes (Extended) | Yes (90% savings) | Yes | Native |
| **GitHub Models** | PAT, Device Code OAuth | Yes | No | No | Yes | Native |
| **Antigravity** | Google OAuth (PKCE) | Yes | Yes (Gemini 3 / Claude) | No | No | Native |
| **Claude CLI** | CLI subscription login | Yes | No | Yes (built-in) | No | Delegated or Passthrough |
| **Ollama** | None (local) | Yes | No | No | No | Native (v0.14+) |
| **ZhipuAI** | API Key | Yes | No | No | Yes | Native |

### Authentication Diversity

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AUTHENTICATION METHODS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  API Key                    Google GenAI, Anthropic, ZhipuAI        │
│  ──────────                 Simple key in environment variable       │
│                                                                      │
│  OAuth (PKCE)               Anthropic, Antigravity                  │
│  ────────────               Browser-based login, subscription       │
│                                                                      │
│  OAuth (Device Code)        GitHub Models                           │
│  ───────────────────        Device code → browser approval          │
│                                                                      │
│  Service Account            Google GenAI                             │
│  ─────────────────          JSON key file or ADC                    │
│                                                                      │
│  SA Impersonation           Google GenAI                             │
│  ──────────────────         Short-lived credentials from chain      │
│                                                                      │
│  CLI Subscription           Claude CLI                              │
│  ─────────────────          Uses `claude login` credentials         │
│                                                                      │
│  No Auth Required           Ollama                                   │
│  ──────────────────         Local server, no credentials needed     │
│                                                                      │
│  verify_auth() UNIFIES ALL: Each provider's verify_auth() handles   │
│  its specific flow, optionally triggering interactive login.         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Provider Discovery and Loading

Providers are discovered dynamically at runtime through two mechanisms:

### Discovery Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROVIDER DISCOVERY                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  discover_providers()                                                │
│       │                                                              │
│       ├──► Entry Points (setuptools)                                 │
│       │    Scans jaato.model_providers group                         │
│       │    Used when jaato is pip-installed                           │
│       │                                                              │
│       └──► Directory Scanning (development)                          │
│            Scans shared/plugins/model_provider/*/                     │
│            Looks for create_provider() or create_plugin()            │
│            Instantiates to get provider.name                          │
│       │                                                              │
│       ▼                                                              │
│  { "google_genai": <factory>,                                        │
│    "anthropic": <factory>,                                           │
│    "github_models": <factory>,                                       │
│    ... }                                                             │
│                                                                      │
│  load_provider(name, config)                                         │
│       │                                                              │
│       ├──► discover_providers()                                      │
│       ├──► factory = providers[name]                                 │
│       ├──► provider = factory()                                      │
│       └──► provider.initialize(config) if config provided            │
│       │                                                              │
│       ▼                                                              │
│  Ready ModelProviderPlugin instance                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Import Error Tracking

Failed imports are tracked rather than swallowed, enabling actionable error messages:

```
Provider 'anthropic' failed to load: No module named 'anthropic'
Hint: Run 'pip install -e jaato-sdk/. -e "jaato-server/.[all]" -e "jaato-tui/.[all]"' to install dependencies.
```

---

## Part 6: Cross-Provider Sessions

A single `JaatoRuntime` can host sessions using different providers simultaneously. This enables scenarios like using Anthropic for the main agent while spawning Google GenAI subagents for specific tasks.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROSS-PROVIDER RUNTIME                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  JaatoRuntime                                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                                                              │    │
│  │   _provider_configs: {                                       │    │
│  │     "anthropic":    ProviderConfig(...)    ◄── Default       │    │
│  │     "google_genai": ProviderConfig(...)    ◄── Registered    │    │
│  │     "ollama":       ProviderConfig(...)    ◄── Registered    │    │
│  │   }                                                          │    │
│  │                                                              │    │
│  │   Shared: registry, permissions, ledger, telemetry           │    │
│  │                                                              │    │
│  └─────────────────┬────────────────────┬───────────────────────┘    │
│                     │                    │                            │
│           create_session()        create_session()                   │
│           provider=None           provider="google_genai"            │
│                     │                    │                            │
│                     ▼                    ▼                            │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐      │
│  │  Main Session           │  │  Subagent Session            │      │
│  │  Provider: Anthropic    │  │  Provider: Google GenAI      │      │
│  │  Model: claude-sonnet   │  │  Model: gemini-2.5-flash     │      │
│  │  History: isolated      │  │  History: isolated           │      │
│  └─────────────────────────┘  └──────────────────────────────┘      │
│                                                                      │
│  Both sessions share:                                                │
│  • PluginRegistry (tools)                                            │
│  • PermissionPlugin (safety)                                         │
│  • TokenLedger (accounting)                                          │
│  • TelemetryPlugin (tracing)                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Registration Flow

```python
# Runtime starts with a default provider
runtime = JaatoRuntime(provider_name="anthropic")
runtime.connect(project_id, location)

# Register additional providers for subagents
runtime.register_provider("google_genai", ProviderConfig(
    project="my-project", location="us-central1"
))

# Create sessions with different providers
main_session = runtime.create_session(model="claude-sonnet-4-5")
sub_session = runtime.create_session(
    model="gemini-2.5-flash",
    provider_name="google_genai"
)
```

---

## Part 7: Schema Translation

Each provider translates JAATO's `ToolSchema` into its SDK-specific format during `create_session()`. The translation is bidirectional: tool schemas go provider-ward, and function calls come back as provider-agnostic `FunctionCall` objects.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SCHEMA TRANSLATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  JAATO ToolSchema                                                    │
│  ┌───────────────────────────────────┐                              │
│  │  name: "web_search"               │                              │
│  │  description: "Search the web"    │                              │
│  │  parameters: {                    │                              │
│  │    "type": "object",              │                              │
│  │    "properties": {                │                              │
│  │      "query": {"type": "string"}, │                              │
│  │      "num_results": {"type": "int"│                              │
│  │    }                              │                              │
│  │  }                                │                              │
│  └──────────────┬────────────────────┘                              │
│                  │                                                    │
│       ┌──────────┼──────────┬──────────────────┐                    │
│       ▼          ▼          ▼                   ▼                    │
│  Google      Anthropic   GitHub Models     Ollama                   │
│  FuncDecl    ToolParam   ChatCompletions   Anthropic-                │
│  (genai)     (anthropic)  FunctionDef      compatible                │
│                            (azure-ai)      format                    │
│                                                                      │
│  Model Response                                                      │
│  (provider-specific function call format)                            │
│       │          │          │                   │                    │
│       └──────────┴──────────┴───────────────────┘                   │
│                  │                                                    │
│                  ▼                                                    │
│  JAATO FunctionCall                                                  │
│  ┌───────────────────────────────────┐                              │
│  │  id: "call_abc123"                │                              │
│  │  name: "web_search"               │                              │
│  │  args: {"query": "...", ...}      │                              │
│  └───────────────────────────────────┘                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 8: Streaming and Cancellation

The streaming interface provides a consistent experience across providers regardless of their native streaming implementation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED STREAMING INTERFACE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Application                                                         │
│       │                                                              │
│       │  send_message_streaming(                                     │
│       │      message,                                                │
│       │      on_chunk = λ(text),        ◄── Token-by-token          │
│       │      cancel_token = token,      ◄── Thread-safe cancel      │
│       │      on_usage_update = λ(usage),◄── Real-time accounting    │
│       │      on_function_call = λ(fc),  ◄── Mid-stream tool calls  │
│       │      on_thinking = λ(thinking)  ◄── Thinking content        │
│       │  )                                                           │
│       │                                                              │
│       ▼                                                              │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  Provider Implementation                                   │      │
│  │                                                            │      │
│  │  Translates native stream → unified callbacks              │      │
│  │                                                            │      │
│  │  Google:   for chunk in response:                          │      │
│  │              on_chunk(chunk.text)                           │      │
│  │                                                            │      │
│  │  Anthropic: for event in stream:                           │      │
│  │               match event.type:                            │      │
│  │                 content_block_delta → on_chunk()            │      │
│  │                 thinking → on_thinking()                    │      │
│  │                 tool_use → on_function_call()               │      │
│  │                                                            │      │
│  │  All:       if cancel_token.is_cancelled:                  │      │
│  │               break  # Return partial response             │      │
│  │                                                            │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
│  CancelToken (Thread-Safe)                                           │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  Main Thread          Worker Thread                        │      │
│  │  ───────────          ─────────────                        │      │
│  │  token.cancel() ───► token.is_cancelled → break stream     │      │
│  │                  ───► token.on_cancel(callback) fires       │      │
│  │                  ───► token.wait(timeout) returns True      │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 9: Error Classification and Retry

Providers implement `classify_error()` and `get_retry_after()` to enable provider-aware retry logic:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ERROR CLASSIFICATION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Exception from SDK                                                  │
│       │                                                              │
│       ▼                                                              │
│  provider.classify_error(exc)                                        │
│       │                                                              │
│       ├──► Returns classification dict:                              │
│       │    { "transient": True,  "rate_limit": True,  "infra": False}│
│       │                                                              │
│       └──► Returns None → global fallback classification             │
│                                                                      │
│  provider.get_retry_after(exc)                                       │
│       │                                                              │
│       └──► Extracts Retry-After header → float seconds               │
│            (Provider knows its own error types)                       │
│                                                                      │
│  Error Categories:                                                   │
│  ┌─────────────────┬─────────────────────────────┬──────────────┐   │
│  │  Category       │  Examples                    │  Action      │   │
│  ├─────────────────┼─────────────────────────────┼──────────────┤   │
│  │  Rate Limit     │  429 Too Many Requests       │  Retry with  │   │
│  │                 │  ResourceExhausted           │  backoff     │   │
│  ├─────────────────┼─────────────────────────────┼──────────────┤   │
│  │  Infrastructure │  500, 503, ConnectionError   │  Retry with  │   │
│  │                 │  Timeout                     │  backoff     │   │
│  ├─────────────────┼─────────────────────────────┼──────────────┤   │
│  │  Permanent      │  400, 401, 403, InvalidModel │  Fail        │   │
│  │                 │  AuthenticationError          │  immediately │   │
│  └─────────────────┴─────────────────────────────┴──────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 10: ProviderConfig

The `ProviderConfig` dataclass serves as the universal configuration carrier:

| Field | Type | Used By |
|-------|------|---------|
| `project` | `str` | Google GenAI (GCP project ID) |
| `location` | `str` | Google GenAI (region) |
| `api_key` | `str` | Google GenAI (AI Studio), Anthropic, ZhipuAI |
| `credentials_path` | `str` | Google GenAI (service account JSON) |
| `use_vertex_ai` | `bool` | Google GenAI (Vertex AI vs AI Studio) |
| `auth_method` | `str` | Google GenAI (auto/api_key/adc/impersonation) |
| `target_service_account` | `str` | Google GenAI (impersonation chain) |
| `credentials` | `Any` | Any provider (pre-built credentials object) |
| `extra` | `Dict` | Provider-specific (caching, thinking budget, etc.) |

The `extra` dict enables provider-specific features without expanding the shared config:

| Provider | Extra Key | Purpose |
|----------|-----------|---------|
| Anthropic | `enable_caching` | Prompt caching (90% cost reduction) |
| Anthropic | `enable_thinking` | Extended thinking mode |
| Anthropic | `thinking_budget` | Max thinking tokens |
| Antigravity | `thinking_level` | Gemini 3 thinking level (minimal/low/medium/high) |
| Ollama | `host` | Ollama server URL |
| Ollama | `context_length` | Override context window size |

---

## Part 11: The Connection Flow

From application code to connected provider:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONNECTION FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. JaatoRuntime.__init__(provider_name="anthropic")                │
│     └──► Stores provider name, creates empty config registry         │
│                                                                      │
│  2. runtime.connect(project, location)                               │
│     └──► Creates ProviderConfig, registers as default                │
│     └──► Sets _connected = True                                      │
│                                                                      │
│  3. runtime.verify_auth(allow_interactive=True)                      │
│     └──► load_provider(name, config=None)  # Temporary instance     │
│     └──► provider.verify_auth()  # May open browser for OAuth        │
│                                                                      │
│  4. runtime.configure_plugins(registry, permission_plugin, ledger)   │
│     └──► Cache tool schemas and executors                            │
│     └──► Wire subagent plugin with runtime reference                 │
│                                                                      │
│  5. runtime.create_session(model="claude-sonnet-4-5")               │
│     └──► runtime.create_provider(model, provider_name)               │
│          └──► load_provider(name, config)  # Full initialization     │
│          └──► provider.connect(model)      # Select model            │
│     └──► JaatoSession(runtime, model)                                │
│     └──► session.configure(tools, system_instructions)               │
│          └──► provider.create_session(instructions, tool_schemas)    │
│                                                                      │
│  6. session.send_message("Hello")                                    │
│     └──► provider.send_message_streaming(...)                        │
│     └──► ProviderResponse returned                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 12: Related Documentation

| Document | Focus |
|----------|-------|
| [jaato_model_harness.md](jaato_model_harness.md) | How instructions, tools, and permissions wrap the provider |
| [jaato_tool_system.md](jaato_tool_system.md) | Tool schemas, discoverability, execution pipeline |
| [jaato_permission_system.md](jaato_permission_system.md) | Permission gating for tool execution |
| [jaato_subagent_architecture.md](jaato_subagent_architecture.md) | Cross-provider subagent sessions |
| [jaato_opentelemetry.md](jaato_opentelemetry.md) | Observability across providers |

---

## Part 13: Color Coding Suggestion for Infographic

- **Blue:** Provider-agnostic types (ToolSchema, Message, ProviderResponse, TokenUsage)
- **Green:** ModelProviderPlugin protocol (interface methods, callbacks)
- **Orange:** Individual provider implementations (Google, Anthropic, GitHub, etc.)
- **Purple:** JaatoRuntime (provider registry, session factory, shared resources)
- **Red:** Authentication and error classification
- **Gray:** SDK/transport layer (the underlying SDK calls each provider wraps)
- **Yellow:** Data flow arrows (schema translation, response conversion)
