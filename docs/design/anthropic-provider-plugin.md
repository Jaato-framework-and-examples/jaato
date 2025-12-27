# Anthropic Provider Plugin Design

> **Status**: Draft
> **Author**: Brainstorm session
> **Date**: 2025-12-18

## Overview

This document describes the design for adding an Anthropic (Claude) provider plugin to jaato's model provider abstraction layer. The plugin will implement the `ModelProviderPlugin` protocol, enabling Claude models as a first-class provider alongside Google GenAI and GitHub Models.

## Motivation

1. **Multi-provider flexibility**: Claude models offer different strengths (reasoning, coding, extended context)
2. **Provider redundancy**: Fallback options when one provider has issues
3. **Feature exploration**: Anthropic has unique features (extended thinking, prompt caching)
4. **SDK comparison**: Understanding different provider patterns improves the abstraction

## Design Goals

1. **Protocol compliance**: Implement full `ModelProviderPlugin` protocol
2. **Feature parity**: Support all current jaato capabilities (tools, history, multimodal)
3. **Anthropic-native features**: Expose unique capabilities where valuable
4. **Simple configuration**: API key-only auth (simpler than Google GenAI)
5. **Cost optimization**: Leverage prompt caching for efficiency

---

## Architecture

```
shared/plugins/model_provider/
├── types.py                    # Provider-agnostic types (existing)
├── base.py                     # ModelProviderPlugin protocol (existing)
├── google_genai/               # Google GenAI implementation (existing)
├── github_models/              # GitHub Models implementation (existing)
└── anthropic/                  # NEW: Anthropic implementation
    ├── __init__.py
    ├── anthropic_provider.py   # Main provider class
    └── converters.py           # Type conversions
```

### Provider Class Structure

```python
class AnthropicProvider:
    """Anthropic Claude provider implementing ModelProviderPlugin protocol."""

    # Core state
    _client: Optional[anthropic.Anthropic]
    _model: Optional[str]
    _system_instruction: Optional[str]      # Stored separately (Anthropic quirk)
    _tools: List[Dict]                       # Anthropic tool format
    _history: List[Message]                  # Manual management (like GitHub Models)
    _config: Optional[ProviderConfig]
    _enable_caching: bool                    # Prompt caching toggle
    _enable_thinking: bool                   # Extended thinking toggle
    _thinking_budget: int                    # Token budget for thinking
```

---

## Key Differences from Existing Providers

### 1. Authentication (Simpler)

| Provider | Auth Methods |
|----------|-------------|
| Google GenAI | API key, ADC, Service Account, Impersonation |
| GitHub Models | PAT with `models: read` scope |
| **Anthropic** | API key only |

```python
def initialize(self, config: ProviderConfig) -> None:
    api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY required")

    self._client = anthropic.Anthropic(api_key=api_key)
    self._enable_caching = config.extra.get("enable_caching", False)
    self._enable_thinking = config.extra.get("enable_thinking", False)
    self._thinking_budget = config.extra.get("thinking_budget", 10000)
```

### 2. Message Roles (Only 2)

| Internal Role | Google GenAI | GitHub Models | **Anthropic** |
|--------------|--------------|---------------|---------------|
| `Role.USER` | "user" | "user" | "user" |
| `Role.MODEL` | "model" | "assistant" | "assistant" |
| `Role.TOOL` | "user" (workaround) | "tool" | "user" (with tool_result blocks) |

### 3. Content Structure (Typed Blocks)

Anthropic uses typed content blocks, not simple parts:

```python
# Anthropic message format
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's the weather?"},
    ]
}

{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me check..."},
        {"type": "tool_use", "id": "toolu_abc", "name": "weather", "input": {"city": "NYC"}}
    ]
}

{
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "72°F, sunny"}
    ]
}
```

### 4. Tool Schema Format

```python
# Our ToolSchema
ToolSchema(
    name="web_search",
    description="Search the web",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }
)

# Anthropic format (note: input_schema not parameters)
{
    "name": "web_search",
    "description": "Search the web",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }
}
```

### 5. System Prompt Handling

Unlike Google GenAI where system instruction is in `GenerateContentConfig`, Anthropic has it as a **separate API parameter**:

```python
def send_message(self, message: str, response_schema: Optional[Dict] = None) -> ProviderResponse:
    response = self._client.messages.create(
        model=self._model,
        system=self._system_instruction,  # Separate parameter!
        messages=self._build_messages(message),
        tools=self._tools,
        max_tokens=8192,
    )
```

### 6. History Management

Like GitHub Models, we'll manually manage history (no SDK chat session):

```python
def _build_messages(self, new_message: str) -> List[Dict]:
    """Build full message list for API call."""
    messages = []

    # Convert history
    for msg in self._history:
        messages.append(message_to_anthropic(msg))

    # Add new user message
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": new_message}]
    })

    return messages
```

---

## Converter Functions

### converters.py

```python
from shared.plugins.model_provider.types import (
    ToolSchema, Message, Part, Role, FunctionCall, ToolResult, ProviderResponse
)

def tool_schema_to_anthropic(schema: ToolSchema) -> Dict:
    """Convert ToolSchema to Anthropic tool format."""
    return {
        "name": schema.name,
        "description": schema.description,
        "input_schema": schema.parameters  # Rename parameters → input_schema
    }

def message_to_anthropic(msg: Message) -> Dict:
    """Convert internal Message to Anthropic format."""
    content = []

    for part in msg.parts:
        if part.text:
            content.append({"type": "text", "text": part.text})

        elif part.function_call:
            content.append({
                "type": "tool_use",
                "id": part.function_call.id,
                "name": part.function_call.name,
                "input": part.function_call.args
            })

        elif part.function_response:
            content.append({
                "type": "tool_result",
                "tool_use_id": part.function_response.call_id,
                "content": json.dumps(part.function_response.result)
                    if not isinstance(part.function_response.result, str)
                    else part.function_response.result,
                "is_error": part.function_response.is_error
            })

        elif part.inline_data:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": part.inline_data.get("mime_type", "image/png"),
                    "data": part.inline_data.get("data")
                }
            })

    # Map role
    role = "assistant" if msg.role == Role.MODEL else "user"

    return {"role": role, "content": content}

def anthropic_response_to_provider_response(response) -> ProviderResponse:
    """Convert Anthropic response to ProviderResponse."""
    text_parts = []
    thinking_parts = []
    function_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "tool_use":
            function_calls.append(FunctionCall(
                id=block.id,
                name=block.name,
                args=block.input
            ))

    # Map stop_reason to FinishReason
    finish_reason_map = {
        "end_turn": FinishReason.STOP,
        "tool_use": FinishReason.TOOL_USE,
        "max_tokens": FinishReason.MAX_TOKENS,
        "stop_sequence": FinishReason.STOP,
    }

    return ProviderResponse(
        text="\n".join(text_parts) if text_parts else None,
        thinking="\n".join(thinking_parts) if thinking_parts else None,  # NEW field
        function_calls=function_calls,
        usage=TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        ),
        finish_reason=finish_reason_map.get(response.stop_reason, FinishReason.STOP),
        raw=response
    )
```

---

## Unique Features

### 1. Extended Thinking

When enabled, Claude shows its reasoning process:

```python
def send_message(self, message: str, ...) -> ProviderResponse:
    kwargs = {}

    if self._enable_thinking:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": self._thinking_budget
        }

    response = self._client.messages.create(
        model=self._model,
        system=self._system_instruction,
        messages=messages,
        tools=self._tools,
        max_tokens=16000,
        **kwargs
    )
```

**Impact on types.py:**

```python
@dataclass
class ProviderResponse:
    text: Optional[str] = None
    thinking: Optional[str] = None  # NEW: Claude's reasoning (when enabled)
    function_calls: List[FunctionCall] = field(default_factory=list)
    # ... rest unchanged
```

### 2. Prompt Caching

Mark static content for caching to reduce cost and latency:

```python
def create_session(self, system_instruction: str, tools: List[ToolSchema], ...) -> None:
    self._system_instruction = system_instruction
    self._tools = [tool_schema_to_anthropic(t) for t in tools]

    # Add cache control if enabled
    if self._enable_caching:
        # Cache the system instruction
        self._system_with_cache = [
            {
                "type": "text",
                "text": system_instruction,
                "cache_control": {"type": "ephemeral"}
            }
        ]

        # Cache tool definitions (if > 1024 tokens)
        if self._tools:
            self._tools[-1]["cache_control"] = {"type": "ephemeral"}
```

### 3. Token Counting (Real API)

Unlike the 4-chars estimate, Anthropic has a real token counting API:

```python
def count_tokens(self, content: str) -> int:
    """Count tokens using Anthropic's API (beta)."""
    result = self._client.beta.messages.count_tokens(
        model=self._model,
        messages=[{"role": "user", "content": content}],
        system=self._system_instruction or ""
    )
    return result.input_tokens
```

---

## Configuration

### Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `ANTHROPIC_API_KEY` | API authentication | Yes |

### ProviderConfig.extra Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_caching` | bool | False | Enable prompt caching |
| `enable_thinking` | bool | False | Enable extended thinking |
| `thinking_budget` | int | 10000 | Max thinking tokens |
| `cache_ttl` | str | "5m" | Cache lifetime ("5m" or "1h") |

### Example Usage

```python
from shared.plugins.model_provider.anthropic import AnthropicProvider
from shared.plugins.model_provider.base import ProviderConfig

provider = AnthropicProvider()
provider.initialize(ProviderConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    extra={
        "enable_caching": True,
        "enable_thinking": True,
        "thinking_budget": 15000
    }
))

provider.connect("claude-sonnet-4-20250514")
provider.create_session(
    system_instruction="You are a helpful assistant...",
    tools=[...],
    history=[]
)

response = provider.send_message("Explain quantum computing")
print(f"Thinking: {response.thinking}")
print(f"Response: {response.text}")
```

---

## Model Support

### Supported Models

| Model | Context | Max Output | Thinking | Notes |
|-------|---------|------------|----------|-------|
| claude-opus-4-5-20251101 | 200K | 32K | Yes | Best overall |
| claude-sonnet-4-20250514 | 200K | 64K | Yes | Balanced |
| claude-haiku-4-20250414 | 200K | 8K | No | Fast/cheap |
| claude-3-5-sonnet-20241022 | 200K | 8K | No | Legacy |

### Context Limits

```python
MODEL_CONTEXT_LIMITS = {
    "claude-opus-4-5": 200000,
    "claude-sonnet-4": 200000,
    "claude-haiku-4": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-haiku": 200000,
    "claude-3-opus": 200000,
}

def get_context_limit(self) -> int:
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if self._model and self._model.startswith(prefix):
            return limit
    return 200000  # Default
```

---

## Structured Output

Anthropic does **not** have native JSON mode like Google's `response_schema`.

### Workaround Options

1. **Prompt engineering**: Include JSON format in system prompt
2. **Tool forcing**: Define a tool with JSON schema, force its use
3. **Post-processing**: Parse JSON from markdown code blocks

### Implementation

```python
def supports_structured_output(self) -> bool:
    return False  # No native support

# Alternative: Force tool use for structured output
def send_message_structured(self, message: str, schema: Dict) -> Dict:
    """Use tool forcing to get structured output."""
    temp_tool = {
        "name": "_structured_response",
        "description": "Return the response in the required format",
        "input_schema": schema
    }

    response = self._client.messages.create(
        model=self._model,
        messages=[{"role": "user", "content": message}],
        tools=[temp_tool],
        tool_choice={"type": "tool", "name": "_structured_response"}
    )

    # Extract from tool_use block
    for block in response.content:
        if block.type == "tool_use" and block.name == "_structured_response":
            return block.input

    raise ValueError("No structured response returned")
```

---

## Multimodal Handling

### Image Input

```python
# Internal Part with inline_data
Part(inline_data={
    "mime_type": "image/png",
    "data": "base64_encoded..."
})

# Converts to Anthropic format
{
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/png",
        "data": "base64_encoded..."
    }
}
```

### Tool Results with Images

```python
# ToolResult with attachment
ToolResult(
    call_id="toolu_abc",
    name="screenshot",
    result="Screenshot captured",
    attachments=[Attachment(
        mime_type="image/png",
        data="base64..."
    )]
)

# Converts to Anthropic format
{
    "type": "tool_result",
    "tool_use_id": "toolu_abc",
    "content": [
        {"type": "text", "text": "Screenshot captured"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    ]
}
```

---

## Error Handling

```python
from anthropic import (
    APIError,
    RateLimitError,
    AuthenticationError,
    BadRequestError
)

def send_message(self, message: str, ...) -> ProviderResponse:
    try:
        response = self._client.messages.create(...)
        return anthropic_response_to_provider_response(response)

    except RateLimitError as e:
        # Let TokenLedger handle retry
        raise

    except AuthenticationError as e:
        raise ValueError(f"Invalid API key: {e}")

    except BadRequestError as e:
        # Check for context length issues
        if "context_length" in str(e).lower():
            raise ValueError(f"Context too long: {e}")
        raise

    except APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")
```

---

## Deferred Features

### Computer Use (Future)

Not included in initial implementation. Would require:
- VM/container infrastructure
- Screenshot capture tooling
- Mouse/keyboard control libraries
- Separate plugin or extension

### Batch API (Future)

For bulk processing. Not needed for interactive sessions.

---

## Implementation Checklist

### Phase 1: Core Provider
- [ ] Create `anthropic/` directory structure
- [ ] Implement `AnthropicProvider` class
- [ ] Implement basic converters
- [ ] Add to provider registry
- [ ] Basic tests

### Phase 2: Full Protocol
- [ ] `initialize()` with config validation
- [ ] `connect()` with model selection
- [ ] `create_session()` with tools/history
- [ ] `send_message()` / `send_message_with_parts()`
- [ ] `send_tool_results()`
- [ ] `get_history()` / `serialize_history()` / `deserialize_history()`
- [ ] `count_tokens()` using beta API
- [ ] `get_context_limit()`

### Phase 3: Unique Features
- [ ] Extended thinking support
- [ ] Prompt caching
- [ ] Add `thinking` field to `ProviderResponse`

### Phase 4: Polish
- [ ] Multimodal support (images)
- [ ] Tool result attachments
- [ ] Structured output workaround
- [ ] Error handling refinement
- [ ] Documentation

---

## Dependencies

```
# requirements.txt addition
anthropic>=0.40.0
```

---

## Questions for Implementation

1. **ProviderResponse.thinking**: Add as optional field, or use metadata dict?
2. **Caching granularity**: Cache tools always, or only when > N tokens?
3. **Thinking budget**: Expose as per-call parameter, or session-level only?
4. **Model validation**: Strict model name validation, or allow any string?

---

## References

- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [Tool Use Documentation](https://docs.anthropic.com/en/docs/tool-use)
- [Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
