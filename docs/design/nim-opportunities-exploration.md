# NVIDIA NIM Opportunities — Deep Exploration

This document explores concrete integration opportunities between jaato and
NVIDIA NIM, building on the existing `nim/` provider plugin. Each opportunity
is mapped to specific jaato architectural hooks and NIM APIs, with enough
detail to proceed to implementation.

---

## Table of Contents

1. [NIM Model Catalog Discovery](#1-nim-model-catalog-discovery)
2. [Self-Hosted NIM Auto-Detection & Health Monitoring](#2-self-hosted-nim-auto-detection--health-monitoring)
3. [NeMo Guardrails Integration](#3-nemo-guardrails-integration)
4. [Hybrid Provider Routing](#4-hybrid-provider-routing)
5. [Parallel Subagent Inference on NIM](#5-parallel-subagent-inference-on-nim)
6. [Cost Optimizer (Self-Hosted + Cloud)](#6-cost-optimizer-self-hosted--cloud)
7. [Offline / Air-Gapped Mode](#7-offline--air-gapped-mode)
8. [Opportunity Matrix](#8-opportunity-matrix)

---

## 1. NIM Model Catalog Discovery

**Effort: Low–Medium** | **Value: High (UX)**

### The Gap

The NIM provider's `list_models()` currently returns an empty list:

```python
# nim/provider.py:338
def list_models(self, prefix=None) -> List[str]:
    """NIM's hosted API does not provide a standard list-models endpoint
    via the OpenAI SDK, so this returns an empty list."""
    return []
```

Meanwhile, the `nim_auth` plugin hardcodes four popular models in
`get_default_models()`. Users must manually consult build.nvidia.com for
model names.

### NIM APIs Available

**Self-hosted NIM containers** expose the OpenAI-compatible `/v1/models`
endpoint:

```
GET http://<nim-host>:8000/v1/models

Response:
{
  "object": "list",
  "data": [
    {
      "id": "meta/llama-3.1-70b-instruct",
      "object": "model",
      "max_model_len": 131072,
      "permissions": [...]
    }
  ]
}
```

**Hosted NIM** (integrate.api.nvidia.com) also responds to `/v1/models`,
though the response set depends on the API key's permissions.

### Integration Design

**A. Implement `list_models()` on NIMProvider**

Use the OpenAI SDK's built-in `client.models.list()`:

```python
def list_models(self, prefix: Optional[str] = None) -> List[str]:
    if not self._client:
        return []
    try:
        models = self._client.models.list()
        names = [m.id for m in models.data]
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        return sorted(names)
    except Exception:
        return []  # Graceful fallback
```

This is zero extra dependencies — the `openai` SDK already supports it.

**B. Add `nim-models` user command**

Extend the `nim_auth` plugin (or create a small `nim_catalog` plugin) with a
user command:

```
nim-models                  # List all available models
nim-models --capabilities   # Show model metadata (context length, tool support)
nim-models --search llama   # Filter by name
```

**C. Enrich model metadata**

For self-hosted NIM, the `/v1/models` response includes `max_model_len`.
This could auto-configure `context_length` in the provider, eliminating the
need for `JAATO_NIM_CONTEXT_LENGTH`:

```python
def connect(self, model: str) -> None:
    self._model_name = model
    # Try to fetch model metadata for context length
    if self._client:
        try:
            model_info = self._client.models.retrieve(model)
            if hasattr(model_info, 'max_model_len'):
                self._context_length = model_info.max_model_len
        except Exception:
            pass  # Fall back to configured/default
```

### Hooks in jaato

- `ModelProviderPlugin.list_models()` — already in the protocol, just needs
  real implementation
- `NIMAuthPlugin.get_default_models()` — could become dynamic instead of
  hardcoded
- TUI `/model` command — already handles provider `list_models()` for
  completion

---

## 2. Self-Hosted NIM Auto-Detection & Health Monitoring

**Effort: Low–Medium** | **Value: High (DevEx)**

### The Gap

The provider's `is_self_hosted()` checks the URL pattern but doesn't probe
whether a NIM container is actually running. Users must manually start the
container and configure the URL.

### NIM Health APIs

Every NIM container exposes:

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `GET /v1/health/ready` | Model loaded, ready for inference | `200 {"message": "..."}` |
| `GET /v1/health/live` | Process alive (may still be loading) | `200` |
| `GET /v1/models` | List served models | Model list |
| `GET /v1/metrics` | Prometheus metrics | Prometheus format |

### Integration Design

**A. Health-aware connection**

Add a `probe_health()` method to NIMProvider that checks container readiness
before sending the first message:

```python
def probe_health(self) -> Dict[str, Any]:
    """Probe a self-hosted NIM container's health.

    Returns dict with 'ready', 'live', 'models', 'latency_ms' keys.
    """
    import urllib.request, json, time
    base = self._base_url.rstrip('/v1').rstrip('/')

    result = {'ready': False, 'live': False, 'models': [], 'latency_ms': None}

    for endpoint, key in [('/v1/health/live', 'live'), ('/v1/health/ready', 'ready')]:
        try:
            t0 = time.monotonic()
            req = urllib.request.Request(f"{base}{endpoint}", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                result[key] = resp.status == 200
                result['latency_ms'] = round((time.monotonic() - t0) * 1000)
        except Exception:
            pass

    if result['ready']:
        try:
            req = urllib.request.Request(f"{base}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                result['models'] = [m['id'] for m in data.get('data', [])]
        except Exception:
            pass

    return result
```

**B. Auto-detection on startup**

When `JAATO_NIM_BASE_URL` points to localhost and the user selects a NIM
model, jaato could:

1. Probe `/v1/health/ready` — if not ready, show "Waiting for NIM
   container..." with a progress indicator
2. Once ready, auto-populate the model name from `/v1/models` if user
   didn't specify one
3. Auto-set `context_length` from model metadata

**C. `nim-status` user command**

```
nim-status                  # Show NIM endpoint health
nim-status --watch          # Continuously monitor (via interactive_shell)
```

Output:
```
NVIDIA NIM Status
=================
Endpoint: http://localhost:8000/v1
  Health: Ready (latency: 12ms)
  Model:  meta/llama-3.1-70b-instruct
  Context: 131072 tokens
  GPU:    NVIDIA A100 80GB (inferred from model)
```

**D. Metrics integration**

For self-hosted containers, the `/v1/metrics` Prometheus endpoint could feed
into jaato's token accounting:

- Actual GPU memory usage
- Batch queue depth
- Tokens per second throughput

This data could inform routing decisions (see section 4).

### Hooks in jaato

- `NIMProvider.initialize()` — add optional health probe
- `NIMProvider.connect()` — auto-populate model and context from `/v1/models`
- New user command in `nim_auth` plugin
- `TokenLedger` — could ingest Prometheus metrics for richer accounting

---

## 3. NeMo Guardrails Integration

**Effort: Medium–High** | **Value: Very High (Enterprise)**

### What NeMo Guardrails Offers

NVIDIA NeMo Guardrails is available as:

1. **Python library** (`pip install nemoguardrails`) — runs in-process, open
   source
2. **Microservice** (Docker container) — production-ready, GPU-accelerated,
   OpenAI-compatible API with guardrails field in request/response

Three NIM microservices for guardrails:

| Microservice | Purpose |
|--------------|---------|
| **Content Safety** | Prevents biased or harmful outputs |
| **Topic Control** | Keeps conversations on approved topics |
| **Jailbreak Detection** | Detects prompt injection / jailbreak attempts |

### Integration Approaches

#### Approach A: In-Process Library (Lighter)

Use the `nemoguardrails` Python package directly in jaato:

```python
from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./guardrails_config")
rails = LLMRails(config)

# Wrap the model call
response = rails.generate(
    messages=[{"role": "user", "content": user_input}]
)
```

**Pros:** No external service, works offline, simple config in Colang
language.
**Cons:** Adds a heavyweight dependency, runs in the same process, limited
to what the library supports.

**jaato integration point:** This maps naturally to a **session plugin**
that intercepts `on_message_start()` and `on_message_received()`:

```python
class GuardrailsSessionPlugin:
    """NeMo Guardrails as a session lifecycle hook."""

    def on_message_start(self, message: str) -> Optional[str]:
        """Check input against guardrails before sending to model.
        Returns modified message or raises GuardrailsViolation.
        """
        result = self._rails.generate(
            messages=[{"role": "user", "content": message}]
        )
        if result.get("blocked"):
            raise GuardrailsViolation(result["reason"])
        return result.get("content", message)
```

#### Approach B: Microservice Proxy (Production)

Route through a NeMo Guardrails microservice deployed as a container. The
microservice API is OpenAI-compatible with an extra `guardrails` field:

```json
POST /v1/chat/completions
{
  "model": "meta/llama-3.1-70b-instruct",
  "messages": [...],
  "guardrails": {
    "config_id": "my_app_config"
  }
}

Response includes:
{
  "choices": [...],
  "guardrails_data": {
    "blocked": false,
    "violations": [],
    "checks_performed": ["content_safety", "topic_control"]
  }
}
```

**jaato integration point:** This maps to a **provider-level middleware**
or a new provider that wraps NIM. The provider sends requests through the
guardrails endpoint instead of directly to the model:

```python
class GuardedNIMProvider(NIMProvider):
    """NIM provider that routes through a NeMo Guardrails microservice."""

    def initialize(self, config):
        super().initialize(config)
        # Override base_url to point at guardrails service
        guardrails_url = config.extra.get("guardrails_url")
        if guardrails_url:
            self._guardrails_client = get_openai_client_class()(
                base_url=guardrails_url,
                api_key=self._api_key or "not-used",
            )
```

#### Approach C: Permission Plugin Integration (Most "jaato-native")

The existing `PermissionPlugin` already intercepts tool calls with
allow/deny decisions. Guardrails could be added as an additional policy
layer:

```python
class NIMGuardrailsPolicy:
    """Check tool calls and model outputs against NIM guardrails."""

    def check_tool_call(self, tool_name: str, args: Dict) -> PolicyMatch:
        """Ask guardrails service if this tool call is safe."""
        # Content safety check on tool arguments
        ...

    def check_output(self, text: str) -> PolicyMatch:
        """Post-process model output through content safety."""
        ...
```

### Recommended Approach

**Start with Approach B (microservice proxy)** because:

- It's a clean, external dependency (no heavyweight pip package)
- Works with any NIM model, not just the ones the library supports
- GPU-accelerated guardrails checking (important for latency)
- Enterprise customers likely already deploy guardrails as microservices
- Maps to a simple config change: set `guardrails_url` in `ProviderConfig.extra`

**Configuration:**

```python
ProviderConfig(
    api_key="nvapi-...",
    extra={
        "base_url": "https://integrate.api.nvidia.com/v1",
        "guardrails_url": "http://guardrails.internal:8000/v1",
        "guardrails_config_id": "jaato_agent",
    }
)
```

**Environment variables:**

| Variable | Purpose |
|----------|---------|
| `JAATO_NIM_GUARDRAILS_URL` | NeMo Guardrails microservice endpoint |
| `JAATO_NIM_GUARDRAILS_CONFIG` | Config ID for the guardrails deployment |

### Plugin Trait for Guardrails

Define a new plugin-level trait:

```python
# In shared/plugins/base.py
TRAIT_GUARDRAILS_PROVIDER = "guardrails_provider"
"""Plugin provides guardrails enforcement for model interactions.

Contract:
- Must expose guardrails_type property ("nim", "custom", etc.)
- Must implement check_input(messages) -> GuardrailsResult
- Must implement check_output(response) -> GuardrailsResult
"""
```

### Hooks in jaato

- `ProviderConfig.extra` — already supports arbitrary keys
- `PermissionPlugin` middleware pattern — proven interceptor architecture
- `SessionPlugin.on_message_start/on_message_received` — lifecycle hooks
- Plugin traits — declare guardrails capabilities
- System instructions — could tell the model about active guardrails

---

## 4. Hybrid Provider Routing

**Effort: High** | **Value: Very High (Cost + Performance)**

### The Opportunity

jaato already supports multi-provider sessions:

```python
# Runtime supports multiple providers
runtime.register_provider('nim', ProviderConfig(api_key="nvapi-..."))
runtime.register_provider('anthropic', ProviderConfig(api_key="sk-ant-..."))

# Sessions can target different providers
fast_session = runtime.create_session(model="llama-3.3-70b", provider_name="nim")
smart_session = runtime.create_session(model="claude-sonnet-4-5", provider_name="anthropic")
```

But there's no **automatic routing** between providers. Every session is
bound to one provider for its lifetime.

### Routing Strategies

#### Strategy A: Smart Routing (Per-Turn Provider Selection)

Route each turn to the optimal provider based on the turn's characteristics:

| Turn Characteristic | Preferred Provider | Reason |
|--------------------|--------------------|--------|
| Heavy tool calling (5+ tools) | NIM (Llama/Nemotron) | Lower latency per call, cheaper |
| Complex reasoning | Anthropic/Google | Frontier model accuracy |
| Code generation | Anthropic (Claude) | Code quality |
| Simple Q&A / classification | NIM (self-hosted) | Free, fast |
| Long context (>100k tokens) | Google (Gemini) | Largest context windows |

**Implementation sketch:**

```python
class RoutingPolicy:
    """Decides which provider handles each turn."""

    def select_provider(
        self,
        message: str,
        history: List[Message],
        pending_tool_results: List[ToolResult],
        available_providers: Dict[str, ProviderHealth],
    ) -> str:
        """Return provider name for this turn.

        Heuristics:
        - If previous turn had many tool calls → stay on fast provider
        - If message asks for analysis/reasoning → use frontier provider
        - If provider is unhealthy → failover to next
        """
        ...
```

#### Strategy B: Cascade (Try Cheap First, Escalate)

```
Turn arrives
    → Try NIM (self-hosted, fast, free)
        → If confidence is high → return response
        → If low confidence / error → escalate to cloud NIM
            → If still insufficient → escalate to Anthropic/Google
```

This requires a way to assess response quality, which could be:
- Model self-assessment (ask "are you confident?")
- Tool call pattern (if model keeps calling the same tool repeatedly →
  likely confused)
- Token count heuristic (very short response to complex question → likely
  incomplete)

#### Strategy C: Subagent Specialization

The most practical near-term approach, building on what jaato already has:

```python
# In the subagent plugin, allow provider override
class SubagentPlugin:
    def spawn_subagent(
        self,
        task: str,
        model: str = None,
        provider_name: str = None,  # ← NEW: cross-provider subagents
        tools: List[str] = None,
    ) -> str:
        session = self._runtime.create_session(
            model=model or self._default_model,
            provider_name=provider_name,  # Uses different provider
            tools=tools,
        )
        return session.send_message(task)
```

The orchestrator (on Claude/Gemini) could spawn tool-heavy subagents on
NIM for parallel execution, getting the best of both worlds.

### Architectural Requirements

1. **Provider health registry** — track which providers are available and
   their current latency/error rates
2. **Session history portability** — convert history between provider formats
   (the `Message`/`Part` types are already provider-agnostic)
3. **Routing policy configuration** — JSON/YAML config for routing rules
4. **Telemetry integration** — track which provider handled each turn for
   cost analysis

### Hooks in jaato

- `JaatoRuntime.register_provider()` — already supports multiple providers
- `JaatoRuntime.create_session(provider_name=...)` — per-session provider
  selection
- `ReliabilityPlugin` — has `_on_model_switch_suggested` callback; could be
  extended for provider-level switching
- `TokenLedger` — could track per-provider costs
- `MCPClientManager.call_tool_auto()` — pattern for routing across multiple
  backends

---

## 5. Parallel Subagent Inference on NIM

**Effort: Medium** | **Value: High (Throughput)**

### The Opportunity

jaato already supports parallel tool execution (up to 8 concurrent) and
subagent spawning. NIM containers handle concurrent requests well — they
batch internally for GPU efficiency.

Benchmarked performance for Llama 3.1 8B on 1x H100:
- **NIM ON:** 1201 tokens/s throughput, 32ms inter-token latency
- **NIM OFF:** 613 tokens/s throughput, 37ms inter-token latency

With 200 concurrent requests, NIM maintains this performance through
internal request batching and queuing.

### Integration Design

**A. NIM-Optimized Subagent Pool**

When the orchestrator needs to spawn multiple subagents for parallel tasks,
route them all to a self-hosted NIM container:

```python
# In runtime configuration
runtime.register_provider('nim_local', ProviderConfig(
    extra={
        'base_url': 'http://localhost:8000/v1',
        'context_length': 131072,
    }
))

# Spawn 8 subagents in parallel on NIM
tasks = [
    ("Search for function X", "nim_local"),
    ("Analyze file Y", "nim_local"),
    ("Check test coverage", "nim_local"),
    ...
]

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [
        pool.submit(
            runtime.create_session(model="llama-3.3-70b", provider_name="nim_local")
                   .send_message(task)
        )
        for task, provider in tasks
    ]
```

NIM's internal batching means these 8 concurrent requests are more
efficient than sequential calls — the GPU processes them as a batch.

**B. Async OpenAI Client**

The NIM provider currently uses the synchronous `OpenAI` client. For
parallel subagents, switching to `AsyncOpenAI` would enable true async
execution:

```python
from openai import AsyncOpenAI

class NIMProvider:
    async def send_message_async(self, message: str) -> ProviderResponse:
        response = await self._async_client.chat.completions.create(
            model=self._model_name,
            messages=self._build_messages(message),
        )
        return response_from_openai(response)
```

This would require an async path through jaato's session layer, which is a
larger change but would benefit all providers.

### Hooks in jaato

- `JAATO_PARALLEL_TOOLS` — existing parallel execution infrastructure
- `JaatoRuntime.create_session()` — lightweight session spawning
- Thread pool in `ai_tool_runner.py` — existing concurrent execution

---

## 6. Cost Optimizer (Self-Hosted + Cloud)

**Effort: Medium** | **Value: High (Cost Savings)**

### The Concept

Self-hosted NIM on a local GPU is effectively free per-request (only
electricity). Cloud NIM costs per-token. Anthropic and Google cost more per
token but have higher quality.

A cost optimizer routes requests to minimize cost while maintaining quality:

```
Request complexity: Low    → Self-hosted NIM (free)
Request complexity: Medium → Cloud NIM (cheap)
Request complexity: High   → Anthropic/Google (best quality)
```

### Implementation Sketch

```python
@dataclass
class CostProfile:
    provider: str
    model: str
    cost_per_1k_input: float    # USD
    cost_per_1k_output: float   # USD
    avg_latency_ms: float       # From health probes
    quality_tier: int            # 1=best, 3=basic

PROFILES = [
    CostProfile("nim_local", "llama-3.3-70b", 0.0, 0.0, 50, 3),
    CostProfile("nim", "llama-3.3-70b", 0.0003, 0.001, 200, 3),
    CostProfile("anthropic", "claude-sonnet-4-5", 0.003, 0.015, 500, 1),
]

class CostOptimizer:
    def select(self, estimated_complexity: str) -> CostProfile:
        if estimated_complexity == "low":
            return self._cheapest_available()
        elif estimated_complexity == "high":
            return self._highest_quality_available()
        else:
            return self._best_value_available()
```

### Integration with TokenLedger

The `TokenLedger` already tracks per-request token usage. Extend it with
cost tracking:

```python
@dataclass
class LedgerEntry:
    # Existing fields...
    provider: str
    model: str
    # New fields
    estimated_cost_usd: float
    routing_reason: str  # "cost_optimizer", "quality_fallback", etc.
```

### Hooks in jaato

- `TokenLedger` — extend with cost data
- `ProviderConfig.extra` — store cost profiles
- `ReliabilityPlugin` — provider availability tracking

---

## 7. Offline / Air-Gapped Mode

**Effort: Low** | **Value: High (specific use cases)**

### The Concept

NIM container + local GPU = complete jaato capability with no internet.
This is valuable for:

- Air-gapped environments (defense, healthcare)
- Poor/no internet connectivity
- Privacy-sensitive work

### What Already Works

The NIM provider already supports self-hosted containers:

```bash
export JAATO_NIM_BASE_URL=http://localhost:8000/v1
# No API key needed for self-hosted
```

### What Needs Enhancement

1. **MCP server tools need to work offline** — some MCP servers may require
   internet
2. **Plugin auto-discovery** — ensure no plugin makes network calls during
   startup
3. **Graceful degradation** — if `web_search` tool is configured but
   offline, return a clear error rather than hanging
4. **Model download management** — help users pre-pull NIM container images
   and model weights

### Implementation

Add an `--offline` flag or `JAATO_OFFLINE=true` environment variable that:

- Skips all plugins that require network access
- Disables web_search tool
- Skips auth validation (no test request to NVIDIA API)
- Shows a clear "offline mode" indicator in the TUI

### Hooks in jaato

- `PluginRegistry.discover()` — filter by network requirements
- `NIMProvider.initialize()` — skip auth validation in offline mode
- Server status bar — show offline indicator

---

## 8. Opportunity Matrix

| Opportunity | Effort | Value | Dependencies | Recommended Order |
|------------|--------|-------|--------------|-------------------|
| Model Catalog Discovery | Low | High | None | **1st** — Quick win, improves daily UX |
| Self-Hosted Health Monitoring | Low-Med | High | None | **2nd** — Natural extension of provider |
| Offline Mode | Low | High (niche) | Health monitoring | **3rd** — Small delta over health work |
| Parallel Subagent on NIM | Medium | High | None | **4th** — Leverages existing parallel infra |
| Cost Optimizer | Medium | High | Health monitoring | **5th** — Needs multi-provider tracking |
| NeMo Guardrails | Med-High | Very High | Provider working | **6th** — Enterprise feature |
| Hybrid Provider Routing | High | Very High | Cost optimizer, health | **7th** — Capstone feature |

### Recommended Starting Point

**Model Catalog Discovery** (section 1) is the quickest win:

- Zero new dependencies
- ~50 lines of code change in `NIMProvider.list_models()`
- Immediate benefit: tab-completion of NIM model names
- Foundation for auto-configuring context length

Followed by **Self-Hosted Health Monitoring** (section 2), which:

- Uses stdlib only (`urllib.request`)
- Makes self-hosted NIM a first-class experience
- Provides data for future routing/cost optimization

---

## References

- [NVIDIA NIM for Developers](https://developer.nvidia.com/nim)
- [NIM API Reference](https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html)
- [NIM Model Catalog](https://build.nvidia.com/models)
- [NeMo Guardrails GitHub](https://github.com/NVIDIA-NeMo/Guardrails)
- [NeMo Guardrails Developer Guide](https://docs.nvidia.com/nemo/guardrails/latest/index.html)
- [NeMo Guardrails Python API](https://docs.nvidia.com/nemo/guardrails/latest/run-rails/using-python-apis/overview.html)
- [NIM Guardrails Microservices Blog](https://blogs.nvidia.com/blog/nemo-guardrails-nim-microservices/)
- [NIM LLM Benchmarking](https://docs.nvidia.com/nim/benchmarking/llm/latest/parameters.html)
- [NIM Configuration Guide](https://docs.nvidia.com/nim/large-language-models/latest/configuration.html)
