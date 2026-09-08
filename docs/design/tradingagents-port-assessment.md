# Porting TradingAgents to the jaato SDK — Assessment

**Status**: assessment (no code ported yet)
**Subject**: [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
v0.4.2 (`be952b8`, Apache-2.0), a LangGraph multi-agent trading-analysis
pipeline, ported onto the jaato SDK (0.16.0) using `jaato-scaffold` as the
source of truth for what the framework offers.
**Method**: every claim about jaato below was taken from the installed
framework via `jaato-scaffold explain …`, `jaato-scaffold new … --dry-run`,
and a scaffolded-then-validated sample workspace, not from reading source.
Every claim about TradingAgents carries a `file:line` into the v0.4.2 tree.

---

## 1. Verdict

**The port is feasible and comparatively small, because TradingAgents is
not really a graph.** Its `StateGraph` encodes a fixed, fully deterministic
pipeline — four analysts in sequence, a two-party debate of `2·N` turns, a
judge, a trader, a three-party debate of `3·M` turns, a judge — and all of
the routing is Python that reads counters and string prefixes
(`graph/conditional_logic.py:52-73`). LangGraph contributes exactly five
things (§4.7): a message reducer, `ToolNode`, SQLite checkpointing,
per-node streaming, and static path-map validation. jaato replaces the
first two by construction (a stage IS a session, so there is nothing to
reduce or clear), covers the fourth with its event stream, and leaves the
third and fifth to the driver, where they are a few dozen lines.

The recommended shape is **driver-as-graph**: a Python driver (the
`cascade` archetype) owns the `AgentState` dict and opens one jaato
session per TradingAgents node, in the same order LangGraph would have
visited them. The pieces of TradingAgents worth keeping — the vendor-routed
`dataflows/` package, the Pydantic output schemas and their renderers, the
markdown decision log with its point-in-time filter, the report writer —
are all framework-free Python and carry over unchanged. The pieces that go
— `graph/`, the twelve `create_*` node closures, `llm_clients/`, the rich
CLI — are the LangChain/LangGraph binding, which is what the port is
replacing.

Rough size (§7): roughly one to two engineer-weeks to parity on the
`propagate()` path, with about 45 of the 72 upstream test files portable
as they stand.

What jaato does **not** hand you (§6): a native OpenAI / Azure / Bedrock
provider (those route through OpenRouter or an OpenAI-compatible endpoint),
pipeline checkpointing as a framework feature (the driver journals stage
payloads instead, at whatever granularity it chooses), and an embedding
memory (TradingAgents does not have one either, see §4.5).

---

## 2. What TradingAgents is (inventory)

`tradingagents/` core is ~4.4k lines; the CLI is another 1.3k.

| Layer | Files | What it does | Framework-bound? |
|---|---|---|---|
| Graph | `graph/setup.py`, `trading_graph.py` (620 lines), `conditional_logic.py`, `analyst_execution.py`, `propagation.py`, `checkpointer.py` | builds the `StateGraph`, compiles it, seeds state, checkpoints per node | **yes** — LangGraph |
| Agents | `agents/analysts/*`, `researchers/*`, `risk_mgmt/*`, `managers/*`, `trader/*` (12 factories) | each returns a closure `node(state) -> dict`; prompts are f-strings or `ChatPromptTemplate`s | **yes** — LangChain messages, `bind_tools`, `with_structured_output` |
| State | `agents/utils/agent_states.py` | `AgentState(MessagesState)` + two nested debate dicts, last-write-wins except `messages` | reducer only |
| Tools | `agents/utils/*_tools.py` (12 `@tool` functions), `dataflows/` (yfinance, Alpha Vantage, FRED, Polymarket, Reddit, StockTwits) | vendor routing with typed errors and a no-fabrication sentinel (`dataflows/interface.py:168-262`) | `@tool` decorator only; the bodies are plain functions |
| Structured output | `agents/schemas.py` (4 Pydantic models, 3 enums), `agents/utils/structured.py` | `with_structured_output` with a free-text fallback; renderers emit the markdown headers downstream greps for | thin wrapper |
| Memory | `agents/utils/memory.py` (334 lines), `graph/reflection.py` | append-only markdown decision log; realised-return reflection on the *next* run; point-in-time filter (`memory.py:70-107`) | **no** |
| Providers | `llm_clients/` (17 OpenAI-compatible specs + Anthropic, Google, Azure, Bedrock; a capabilities table) | model-gated thinking knobs, content normalisation, DeepSeek/MiniMax quirks | **yes** — LangChain chat classes |
| Reporting / CLI | `reporting.py`, `cli/` | numbered markdown report tree; a rich `Live` dashboard driven by streamed node deltas | CLI yes; `reporting.py` no |

### 2.1 The pipeline

```
START → [Market → Sentiment → News → Fundamentals]   (each: tool loop until the model stops calling tools)
      → Bull ⇄ Bear ×(2·max_debate_rounds)  → Research Manager (deep model, structured)
      → Trader (structured)
      → Aggressive → Conservative → Neutral ×(3·max_risk_discuss_rounds) → Portfolio Manager (deep model, structured)
      → END
```

Two models: `quick_think_llm` for the ten talkers, `deep_think_llm` for the
two judges (`graph/setup.py:76-92`). One reducer: `messages`. Every other
state field is last-write-wins, which is why each debate node rebuilds the
whole nested dict by hand (`researchers/bull_researcher.py:54-60`).

### 2.2 Things worth knowing before porting

- The per-analyst tool loop has **no cap** of its own; it ends when the model
  stops emitting tool calls or at the global `recursion_limit=100`
  (`graph/propagation.py:78`).
- The Sentiment Analyst never calls tools: it **pre-fetches** news,
  StockTwits and Reddit before the model runs and injects them into the
  system message (`agents/analysts/sentiment_analyst.py:70-76,130-192`).
- Debate alternation is prose-driven: `current_response.startswith("Bull")`
  against a prefix the node itself wrote (`conditional_logic.py:57`).
- The memory is **not embedding-based**. It is a markdown file with
  `<!-- ENTRY_END -->` separators, recency-selected, filtered by
  `resolved <= as_of` for backtests (`memory.py:84`). Only the Portfolio
  Manager reads it (`managers/portfolio_manager.py:36-41`).
- `backtrader` and `redis` are declared dependencies and imported nowhere.
- `dataflows/config.py` is a process-global singleton set once from
  `TradingAgentsGraph.__init__` (`trading_graph.py:102`).
- Checkpoint resume keys a thread on `sha256(ticker:date:run_signature)`,
  resumes with `invoke(None)`, and deletes rows by SQL on success
  (`graph/checkpointer.py:28-98`, `trading_graph.py:431-492`).

---

## 3. What `jaato-scaffold` established (the introspection pass)

This is the part the question was really about: how much of the port can be
decided by asking the installed framework rather than reading it. The
answer is "nearly all of the structural decisions". The verbs used, and
what each settled:

| Verb | What it answered for this port |
|---|---|
| `explain` | 18 providers, 38 plugins, 4 GC strategies, **8 archetypes** |
| `explain archetypes` / `explain archetype cascade` | there is a generated **sequential multi-session driver** whose contract is exactly the TradingAgents pipeline: a `WORKLIST` of `(profile, agent, prompt)` stages, each `await stage.complete(prompt)` returning the stage's typed payload, one `cascade_driver_id` tenanting the run |
| `explain archetype host-tools` | a client can register **host tools** (`{name, description, parameters, handler}`) that the agent calls and the client executes locally — the shape of TradingAgents' toolkit; registration must precede `create_session` |
| `explain clients` | an **`InProcessClient`** (no daemon, no socket) exists alongside IPC/WS, so the port can stay a library the way `TradingAgentsGraph.propagate()` is; `--transport in_process` on any archetype |
| `explain tiers` | `model_tiers` gives one session up to 8 named models with `enter_tier`, cross-provider — an alternative to the quick/deep split, though §4.3 argues for per-profile binding instead |
| `explain profile` | the full profile schema: `completion_payload_schema`, `spawn_payload_schema` (properties must be typed `string`), `completion_processors`, `max_turns` (most-restrictive-wins), `budget_control`, `model`/`provider`, `inherits`, per-key merge rules |
| `explain completion` | the output-side hook: `validate(payload, context)` blocks a completion and hands the model the errors; `render(payload, context)` writes a file; `max_refusals` / `on_exhausted` bound the loop; `context.tool_calls` is the ledger for cross-checking a payload against what the session actually did |
| `explain prefetch` | `{{!py:scripts/<f>.py}}` runs Python at session-prep and injects its string into the persona — the Sentiment Analyst's pre-fetch, verbatim |
| `explain plugin subagent` / `memory` / `service_connector` / `permission` | what model-driven delegation, the persistent memory store, REST discovery, and the policy knob look like, so each could be ruled in or out (§4) |
| `explain providers` | the capability matrix (tool-choice forwarding, thinking, prompt caching, streaming per provider) — and the absence of `openai`, `azure`, `bedrock` rows (§6) |
| `new profile-set --agents market_analyst,bull_researcher,research_manager,trader,portfolio_manager` | wrote a two-tier profile set (provider-agnostic `_base_<agent>.yaml` + `<set>/<agent>.yaml` binding provider and model, `temperature: 0.0` already in), then **re-validated it**: "valid by construction" |
| `new cascade --transport in_process` / `new host-tools --transport in_process` | the two generated scripts §5 is built from; both `py_compile` on emission |
| `explain archetype sweep` | an N-independent-jobs driver with an acceptance gate — the backtest-matrix shape (§4.10) |
| `validate <workspace> --set <set>` | lints against the live registry: mistyped `api_params`, unknown plugins, a spawn schema with a non-string property, an `echo` profile with no `usage` |
| `jaato-doctor --workspace … --env-file …` | 14 preflight checks; on this workspace 0 fail / 5 warn (no daemon yet, no secret given — both expected for an in-process port) |

Four incidental findings from the pass, none blocking:

- `explain provider echo` prints nothing (the test double is deliberately
  hidden from `explain providers`), so its knobs — `response`, `tool_call`,
  `retry_tool_call`, `usage` — have to be read from
  `shared/plugins/model_provider/echo/provider.py:30-60`.
- In this venv every `jaato-scaffold` invocation logs an MCP-thread traceback
  (`'types.UnionType' object has no attribute 'model_validate_json'`, from
  `shared/plugins/mcp/plugin.py:1296`). It is an `mcp` package version
  mismatch in the discovery pass and does not affect the output; a port
  that never enables the `mcp` plugin never meets it at runtime.
- `shared/plugins/subagent/README.md:347-375` documents
  `spawn_subagent(background=True)` and `get_subagent_result(...)`; neither
  exists in `plugin.py`. `explain plugin subagent` (which reads
  `get_tool_schemas`) is right and the README is stale — one more reason the
  assessment trusts the introspection over the prose.
- `examples/` holds a nine-line `hello.py` and an IPC recovery config, no
  agent example. The worked examples are the scaffold templates
  (`shared/scaffold/_client_templates.py`), the `jaato-eval/tasks/*` task
  trees (fixture + persona + prefetch + base profile + completion schema +
  graders), and `jaato-tui/.jaato.example/` for a full workspace layout.

---

## 4. Primitive-by-primitive mapping

| TradingAgents | jaato | Notes |
|---|---|---|
| `StateGraph` + `GraphSetup.setup_graph` | `run_cascade.py` driver (cascade archetype) over `InProcessClient` | §4.1 |
| `AgentState` dict | driver-owned dict; stage outputs are typed payloads | §4.2 |
| `messages` reducer, `RemoveMessage`, `Msg Clear *` nodes | **nothing** — each node is its own session | §4.2 |
| `quick_think_llm` / `deep_think_llm` | per-profile `model:`/`provider:` in a profile set | §4.3 |
| `@tool` functions + `ToolNode` + `should_continue_*` | `client_tools` host tools + the framework's own function-calling loop, bounded by `max_turns` | §4.4 |
| Sentiment pre-fetch | `{{!py:…}}` persona prefetch | §4.4 |
| prompt f-strings / `ChatPromptTemplate` | `.jaato/agents/<name>.md` personas with `agent_params` substitution | §4.2 |
| `with_structured_output` + `schemas.py` + free-text fallback | `completion_payload_schema` + `completion_processors` (`validate` / `render`) | §4.6 |
| `ConditionalLogic` debate counters | driver loops; each debater is a long-lived session and the driver relays the opponent's last turn | §4.5 |
| `TradingMemoryLog` + `Reflector` | keep the log as-is; the reflection is one `ask()` on a cheap profile; the `memory` plugin is a different (model-curated) semantics | §4.5 |
| `SignalProcessor` / `rating.py` | mostly unnecessary — `rating` is an enum in the payload; keep `is_review` as the fallback | §4.6 |
| `SqliteSaver` checkpointing | driver journals each stage's payload; resume = skip journaled stages | §4.8 |
| `llm_clients/` + `capabilities.py` | jaato providers + `plugin_configs.<provider>.api_params` | §4.3, §6 |
| `stream(stream_mode="values")` + rich CLI + `StatsCallbackHandler` | event subscriptions (`AGENT_OUTPUT`, `TOOL_CALL_*`, `TURN_COMPLETED`), the `observer` archetype, the token ledger and `budget_control` | §4.9 |
| `reporting.py` | keep; optionally a `render` processor with `output:` writes each section as it completes | §4.6 |
| `default_config.py` + `TRADINGAGENTS_*` env | profile `env:` + workspace `.env`; `jaato-scaffold validate` replaces the `_coerce` fail-fast | §4.3 |
| backtest over tickers/dates | `sweep` archetype (+ `jaato-eval` for graded matrices) | §4.10 |

### 4.1 The graph becomes the driver

The cascade archetype's generated `main()` is a `for` over a `WORKLIST`,
each stage `await stage.complete(prompt)` inside
`InProcessClient.session(profile=…, agent=…, cascade_driver_id=…)`. The
TradingAgents pipeline is that loop with three differences, all of which
are ordinary Python:

1. the analyst phase iterates over `selected_analysts` (the same list
   `build_analyst_execution_plan` validates, `graph/analyst_execution.py:56`);
2. the two debates are inner loops that hold two (or three) sessions open
   across turns rather than one session per stage;
3. the prompt for each stage is composed from the state dict — the reports,
   the debate history, `past_context` — exactly as the node closures compose
   their f-strings today.

`InProcessClient` is the right transport for parity with
`TradingAgentsGraph.propagate()`: no daemon, no socket, no cold-start; the
same script switches to `IPCClient` (daemon, warm runner pool, observers can
attach by cascade id) by changing the `_open_session` helper the generator
emits. The archetype's own recipe notes — `client_type=ClientType.API` keeps
`signal_completion`, `env_file` must be a real path, `complete()` not `ask()`
for a gated stage because `TURN_COMPLETED` fires mid-flight when the daemon
re-prompts (jaato #767) — are the traps a hand-written driver would fall
into; this is precisely why the assessment leans on the generator.

**Model-orchestrated alternative, rejected.** The `subagent` plugin lets a
root agent `spawn_subagent(profile=…, task=…)` (always asynchronous: it
returns an `agent_id` and the child runs on a thread), so one could make the
Portfolio Manager the root and have it delegate. Two facts rule it out for
this pipeline. First, a model-driven parent receives a child's result as
**injected prose** — `[SUBAGENT agent_id=… event=COMPLETED]\n<text>` via
`inject_prompt` (`shared/plugins/subagent/plugin.py:2918-2960`,
`shared/jaato_session.py:1570-1600`) — not as the typed payload; typed
payloads only cross the boundary when an SDK driver is the orchestrator.
Second, it trades a deterministic, testable pipeline for a model's
judgement about sequencing, which TradingAgents never wanted (its path maps
exist to *prevent* drift, `graph/setup.py:32-42`). It is worth revisiting
only if the product direction changes to "let the PM decide which analysts
to run".

### 4.2 State, messages, and personas

Every field in `AgentState` (`agent_states.py:47-76`) is either seed input
(`company_of_interest`, `trade_date`, `asset_type`, `instrument_context`,
`past_context`) or a node's output. In the port the driver holds the dict;
seed fields reach a stage as `agent_params` substituted into its persona,
and outputs come back as the stage's completion payload. The nested
`InvestDebateState` / `RiskDebateState` collapse to what the driver needs to
compose the next prompt: the running `history` string and the last turn per
speaker. The read-modify-write discipline that LangGraph's last-write-wins
forces on every debate node disappears.

The `messages` reducer and the four `Msg Clear` nodes
(`agents/utils/agent_utils.py:204-228`) exist because LangGraph shares one
message list across nodes and TradingAgents does not want the Market
Analyst's tool chatter in the Bull Researcher's context. In jaato every stage
is its own session with its own history, so isolation is the default and
the clearing machinery has no counterpart. (The placeholder `HumanMessage`
the clear node leaves behind — anchored to the instrument rather than
`"Continue"`, because some providers took "Continue" as the task, #888 — is
likewise unnecessary: the next stage gets a real prompt.)

Prompts move into `.jaato/agents/<name>.md`: YAML frontmatter declaring
`params` (`required`, `default`, `description`) over a markdown body with
`{{name}}` / `{{name:default}}` substitution, resolved by `resolve_agent`
(`shared/plugins/subagent/config.py:3560-3693`; an unsupplied required param
is reported in `missing_params`, never silently blanked). TradingAgents'
prompts are already parameterised by exactly the things `agent_params`
carries (`instrument_context`, `current_date`, `asset_type`, the tool
names), and the shared "collaborating with other assistants … prefix FINAL
TRANSACTION PROPOSAL" wrapper becomes a `.jaato/instructions/` base layer
(loaded alphabetically, droppable per profile with
`suppress_base_instructions` to save the ~3-5k tokens a narrow debater does
not need). The `spawn_payload_schema` on each profile documents which
params a stage requires; its properties must be typed `string` (they cross
as `key=value` tokens), which is fine — every seed field already is one.

### 4.3 Two models, seventeen providers

TradingAgents binds two chat models at construction and hands one or the
other to each factory. The direct equivalent is a **profile set**: the
scaffolded `_base_<agent>.yaml` carries the stage's determinism (plugins,
schemas, `max_turns`), and `<set>/<agent>.yaml` binds provider and model —
so `research_manager.yaml` and `portfolio_manager.yaml` name the deep model
and the other ten name the quick one. `JAATO_PROFILE_SET` switches the whole
run between, say, `openrouter_sonnet` and `nebius_deepseek` without touching
the base tier, which is the knob TradingAgents' `llm_provider` +
`deep_think_llm` + `quick_think_llm` triple was.

`model_tiers` would also work (one session, `enter_tier('planner')` for the
judge), but it asks the *model* to switch and keeps one history across the
switch, and neither is wanted here: the judge is a different node with a
different persona. Per-profile binding is the honest mapping.

The thinking knobs `TradingAgentsGraph._get_provider_kwargs` gates by model
family (`trading_graph.py:168-208`) map onto `plugin_configs.<provider>.api_params`
— `enable_thinking` / `thinking_budget` / `thinking_level` on OpenRouter and
Anthropic, `temperature: 0.0` as the determinism knob the scaffold already
writes. The model-family gating itself (drop `reasoning_effort` unless
`^(gpt-5|o[1-9])`, `openai_client.py:175-180`) has no framework counterpart
and should live in the profile set: a set is *per model*, so the knob is
simply present in the profiles where it applies.

### 4.4 Tools: the toolkit becomes host tools, `dataflows/` stays

The twelve `@tool` functions (`agents/utils/*_tools.py`) are plain functions
with a docstring and typed arguments that route through
`dataflows.interface.route_to_vendor`. The host-tools archetype's spec is
the same information — `name`, `description`, JSON-schema `parameters`, a
`handler` — so the toolkit ports as a list comprehension over those
functions, and `dataflows/` (vendor chain, typed errors, the
`NO_DATA_AVAILABLE` sentinel with its "do not estimate or fabricate"
instruction, FRED vintage pinning, the OHLCV cache and staleness guards)
ports **unchanged**. Add `auto_approve: True` to each spec: the embedded
client whitelists such tools with the permission plugin
(`jaato_embedded/client.py:707-731`), which is what a headless run needs.

Per-analyst tool sets (`trading_graph.py:210-250`) become per-stage
`client_tools` lists passed to `session(...)`. The asymmetry the inventory
found — the news analyst binds four tools while its `ToolNode` registers
five — cannot occur here, since one list serves both roles. On the wire a
host tool is a `ToolExecuteRequestEvent` answered by
`respond_to_tool_execution` (`jaato_sdk/client/ipc.py:2024-2098`); a handler
exception reaches the model as `error=`, which is how the vendor layer's
typed errors and the `NO_DATA_AVAILABLE` sentinel keep their meaning.

One property of host tools shapes the deployment choice: **they exist only
while the registering client is attached.** In-process that is always. In
daemon mode a session revived cold under a cascade id with no client
returns `DEFERRED` from `session.wake` rather than running a turn its tools
could not serve (`server/session_manager.py:7519-7538`). A daemon
deployment that wants tools to outlive the driver is the `ToolPlugin`
follow-up below.

The tool **loop** is the framework's: `JaatoSession` runs function calling
until the model returns text without calls, in parallel when the model
batches (`JAATO_PARALLEL_TOOLS`), and `max_turns` on the profile bounds it,
which is a cap TradingAgents' loop lacks. The analyst's "write the report
only when there are no tool calls" gate (`market_analyst.py:85-93`) is what
`signal_completion` makes explicit: the stage ends when the model calls it
with the report in the payload.

The Sentiment Analyst's pre-fetch (`sentiment_analyst.py:70-76`) is the
persona prefetch, one to one:

```markdown
<!-- .jaato/agents/sentiment_analyst.md -->
You are the Sentiment Analyst for {{instrument_context}} as of {{trade_date}}.
{{!py:scripts/prefetch_sentiment.py}}
```

with `render(context, args)` calling `get_news`, `fetch_stocktwits_messages`,
`fetch_reddit_posts` for `trade_date - 7d` and returning the delimited
blocks. The `explain prefetch` contract — it runs once at session-prep, a
revive restores the rendered prompt instead of re-running it, so it should
be idempotent — matches a data fetch pinned to a trade date. The script
runs in the session's process (`shared/dynamic_instructions.py`), so
in-process it imports `tradingagents.dataflows` from the driver's
environment; in daemon mode the runner's environment must have the package
installed. The mandatory form (`{{!py:` without `?`) aborts session-prep on
failure, which is the right posture for a report that must not be written
from an empty window. `jaato-eval/tasks/example-echo/.jaato/scripts/prefetch_artefact.py`
records the motivation in one line: routing a fact through a model's
discretion made it unreliable, roughly one run in four the judge simply did
not call the tool.

Two alternatives were considered and set aside for the first cut:

- **A `ToolPlugin`** in `out-of-tree-plugins/` (entry point `jaato.plugins`)
  wrapping `dataflows/`. Right for daemon deployments where other jaato
  clients should reach the same tools, and the natural second step; it
  moves the vendor config from a process global to `plugin_configs`.
- **`service_connector`** (`discover_service` / `call_service`). It wants an
  OpenAPI spec; yfinance is a library, not an API, and Alpha Vantage and
  FRED have none. Not a fit.

### 4.5 The two debates and the memory

A debate in TradingAgents is one node per side, each rebuilding a shared
`history` string and reading the opponent's `current_response`. In the port
each side is a **long-lived session**: the driver opens `bull` and `bear`
once, then alternates `await bull.ask(opponent_last)` /
`await bear.ask(opponent_last)` for `2·max_debate_rounds` turns. Each side's
own history is kept by its session, which is *better* than upstream, where
`bull_history` is a hand-concatenated string; the shared `history` the judge
reads is the driver's concatenation of the returned texts. The
`opponent_argument_or_opening` guard (`agent_utils.py:68-79`, so the first
speaker does not argue against a ghost) is a one-line conditional in the
driver. The risk debate is the same loop with three sessions and the fixed
Aggressive → Conservative → Neutral rotation. These stages are not
completion-gated — a debater's turn is its answer — so they use `ask()`,
not `complete()`, per the archetype's rule on which method fits which
session.

`send_to_sibling` (subagent plugin) would let the bull and bear address each
other without the driver relaying. It is fire-and-forget by design and the
driver would still have to count turns, so it buys nothing here.

**Memory.** `TradingMemoryLog` is 334 lines of framework-free Python with a
point-in-time filter (`memory.py:84`) that a backtest depends on. Keep it in
the driver: `store_decision` after the Portfolio Manager stage,
`_resolve_pending_entries` at the start of the next run — with the
`Reflector` (`graph/reflection.py:31-57`) becoming one `ask()` on a cheap
`reflector` profile — and `get_past_context(ticker, as_of=…)` passed as the
`past_context` agent param the PM persona reads. The `_fetch_returns`
yfinance call that leaks into `trading_graph.py:273-324` moves with it.

jaato's `memory` plugin is a **different** design: model-curated
`store_memory` / `retrieve_memories` over a JSONL store with tags and
raw→validated maturity, plus the `{{continuity_scope}}` persona pattern
(`docs/design/agent-continuity.md`). Matching is tag and keyword based, not
embedding based (no vector store anywhere in `shared/plugins/memory/`;
embeddings exist only as a protocol in `references`, with implementations
out of tree behind the `jaato.embedding` entry point). It has no "resolved
on or before this date" filter, its prompt-time hints are built from
`curated.jsonl` only — so a curator reactor is a prerequisite, not a
refinement (`shared/plugins/memory/plugin.py:246`) — and letting the PM
decide what to remember is a research change, not a port. It is the right
tool for a *later* feature — cross-run lessons the PM curates itself — and
composes with the log rather than replacing it.

### 4.6 Structured output is the completion payload

The four structured agents (Sentiment, Research Manager, Trader, Portfolio
Manager; `agents/utils/structured.py:42-89`) map onto
`completion_payload_schema`. The mechanism is worth stating precisely
because it is not a provider `response_format`: **the schema becomes the
`signal_completion` tool's own `parameters`** (`shared/lifecycle_tools.py:283-355`),
so the payload's fields are the tool call's arguments, flat at the top
level — wrapping them under a `payload` key is a documented failure mode
(`jaato_sdk/conformance/conftest.py:36-50`). Enforcement is double: the
provider constrains the tool-call shape at sampling time where it can, and
`LifecycleTools._execute_signal_completion` runs `jsonschema.validate`
server-side and returns `validation_failed` with the field path so the
model self-corrects. (A provider-level `response_schema` hook exists on the
base class but is unused across the provider tree; the tool path is the
supported one.) For models that struggle to compose a whole object at
once, the companion `prepare_completion(field_path, value)` /
`query_completion()` tools accumulate the payload field by field across
turns.

The schemas are already Pydantic (`agents/schemas.py`), so
`Model.model_json_schema()` produces them; the enum fields
(`PortfolioRating`, `TraderAction`, `SentimentBand`) become `enum`
constraints, and the nullable floats become `["number", "null"]`. For
OpenRouter set `api_params.strict_tools: true` so the schema is
grammar-enforced at sampling time; the framework's rule that authors own
strict-mode shape (`additionalProperties: false`, exhaustive `required`, no
`oneOf`) is easily met by these flat objects. Follow the payload-schema
convention (`docs/design/payload-schema-conventions.md` §3.2) and add
`errors[]` / `warnings[]` to every schema so a stage can say "I could not
answer" instead of inventing one — the sweep template records what
happens without it (a grader read a failure-to-produce as a data point).

The two pieces of `schemas.py` that are not the schema become
`completion_processors`:

- `_coerce_optional_float` (`schemas.py:33-50` — placeholder strings and
  anything ending in `%` become `None`, because a "15%" stop once became a
  $15 stop on a $600 stock, #1288) is a `validate` that returns an
  instruction ("stop_loss must be an absolute price, not a percentage") so
  the model fixes it; `max_refusals: 2`, `on_exhausted: allow`.
- the render functions (`render_pm_decision` and friends,
  `schemas.py:189-203,266`) are a `render` with `output:` pointing into the
  report tree, which makes `reporting.py`'s per-section files fall out of the
  gate rather than being assembled afterwards. `write_report_tree`
  (`reporting.py:13-101`) still produces `complete_report.md`.

`processor.validate` also receives `context.tool_calls`, which is how a
port can enforce what the Market Analyst's prompt only requests: that
`get_verified_market_snapshot` was actually called before the report was
written (`market_analyst.py:25-56`). Today that is a prompt instruction; here
it can be a gate. One caveat from the sweep template: a job run through a
*pooled* sweep has no tool-call ledger after detach, so that particular
check belongs to the cascade driver's stages, not to a backtest matrix.

The free-text fallback in `invoke_structured_or_freetext` (a thinking model
answering in prose) is the case `signal_completion`'s nudge loop already
handles: an agent that ends a turn in prose is re-prompted to complete,
within `max_turns`. `SignalProcessor` / `extract_rating`
(`agents/utils/rating.py:45-66`) becomes a fallback for the un-gated debug
path only; keep `is_review` so a `REVIEW` never silently becomes `Hold`
(#1170).

### 4.7 What LangGraph provided, and what replaces it

| LangGraph feature relied on | Where | Replacement |
|---|---|---|
| `add_messages` reducer + `RemoveMessage` | `agent_states.py:47`, `agent_utils.py:204-228` | not needed: one session per node |
| `ToolNode` | `trading_graph.py:210-250` | the session's own tool loop over `client_tools` |
| `SqliteSaver` checkpoint / `invoke(None)` resume | `graph/checkpointer.py` | driver-level stage journal (§4.8) |
| `stream(stream_mode="values")` | `cli/main.py:1144-1244` | event subscriptions / `Session.stream()` (§4.9) |
| `add_conditional_edges` + over-complete path maps | `graph/setup.py:32-42` | plain Python control flow; the drift the maps guard against cannot occur |

LangChain-core pieces (`ChatPromptTemplate`, `bind_tools`,
`with_structured_output`, `BaseCallbackHandler`, message types leaking into
`cli/main.py:948`) go with the graph.

### 4.8 Checkpoint and resume

TradingAgents checkpoints after every node and resumes a crashed run from
the last one (`trading_graph.py:431-492`, opt-in). A driver that appends
each stage's `(stage_name, payload)` to a per-run journal keyed on the same
`sha256(ticker:date:run_signature)` idea (`checkpointer.py:28-38`) and skips
journaled stages on restart gives the same behaviour in a few dozen lines.
Within a debate, journal per turn. Clear on success, as upstream does.

**The granularity is the driver's to choose, not a property of the
framework.** Upstream's analysts are checkpointed per tool round trip only
because each round trip happens to be a separate graph node (analyst →
tool node → analyst). A port can run an analyst as one session and resume
at the stage boundary, or split it into journaled sub-stages and resume
finer than upstream does: a "choose indicators" stage whose payload is the
list, a deterministic fetch (a `{{!py:…}}` prefetch or the driver itself,
no model turn), and a "write the report" stage that receives the fetched
data as prompt text. That is the Sentiment Analyst's pre-fetch pattern
(§4.4) applied to the other three analysts, and it removes the one
non-deterministic step upstream has — whether the model calls
`get_verified_market_snapshot` before writing.

Sub-stages do **not** require fresh sessions. A completion-gated session
is not one-shot: `_signal_completion_called` is reset at the start of
every turn (`shared/jaato_session.py:6110-6127`, whose docstring names the
suspend/resume shape — the agent calls `signal_completion` every turn and
the driver wakes the same session later), so one analyst session can
signal the end of each sub-stage with a typed payload the driver journals,
go quiet, and be driven again with the next prompt on the same history.
While the driver stays attached that is simply another `complete()` on the
same facade `Session`; a session that has gone cold is revived by
`session.wake` (`server/session_manager.py:7519`), which defers only when
no client is attached and host tools would have nowhere to dispatch. The
completion schema can carry a `phase` field so the driver knows which
sub-stage just ended. This keeps the tool results as retained history
rather than forwarded text, and gives resume points at every sub-stage —
finer than upstream, on one session, with nothing re-sent that the model
did not already hold.

jaato's own session persistence (`session.wake`, `SessionState` with the
profile snapshot and rendered instructions — CLAUDE.md "Session Revive")
persists *a session*, not *a pipeline*; it is what lets a half-finished
debater be woken with its history intact if the driver wants that, but the
pipeline position is the driver's to record. Two details matter for the
driver's error handling: `Session.complete(timeout=…)` stops *waiting*, not
the session (jaato #826), so a timed-out stage is still alive and wakeable;
and a wake of a cold session that needs host tools is `DEFERRED` until a
client attaches (§4.4). The `--checkpoint` / `--clear-checkpoints` flags map
to a driver argument and a journal directory wipe.

### 4.9 Progress, stats, and the CLI

The rich `Live` dashboard (`cli/main.py:265-497`) is built on per-node value
deltas and reconstructs the final state by merging chunks. In the port the
driver *has* the state, so the equivalent is a subscriber on the session's
events — `AGENT_OUTPUT` for streamed text, `TOOL_CALL_START` / `TOOL_CALL_END`
for the tool panel, `TURN_COMPLETED` / `AGENT_COMPLETED` for status
transitions — via `s.client.subscribe(EventType.…)`, which the facade
documents as additive and persistent across turns. In daemon mode the
`observer` archetype attaches to the whole cascade by id from a separate
process, and the jaato TUI can attach to any session. `StatsCallbackHandler`
(`cli/stats_handler.py`) is the token ledger: every turn's usage and
provider-reported cost are already accounted, `budget_control` caps them
per profile (min-wins under inheritance) and can brown-out the model as a
ceiling approaches, and OpenTelemetry spans carry `llm.cost.total` for
anyone with Phoenix or Langfuse.

The questionary selection flow (ticker, date, analysts, depth, provider,
models, thinking level, language) is a small typer command that writes the
seed state and picks `JAATO_PROFILE_SET`; the env-precedence rules
(`cli/main.py:974-1001`) reduce to "the `.env` and the profile `env:` map".

### 4.10 Backtests

Running the pipeline over a matrix of `(ticker, date)` is the `sweep`
archetype: N independent jobs, each `complete(prompt, timeout=…)`, a failed
job not stopping its siblings, results collected per job. Its emitted
acceptance gate (`acceptance.sh` + processor + `errors[]` in the payload) is
where "the PM produced a parseable rating and the market snapshot was
verified" becomes a graded check rather than a hope. `jaato-eval` layers a
task manifest and result store over the same primitives for the "cheaper
model on a subset of tasks — what does it cost in pass rate" question,
which is the reproducibility question TradingAgents' README raises and
declines to answer.

---

## 5. A worked slice

The pieces below were generated or validated in a scratch workspace during
this assessment; nothing here is invented syntax.

**Persona** (`resolve_agent` format; the body is the upstream Bull prompt
with its f-string slots renamed):

```markdown
---
description: Bull-side researcher in the investment debate
params:
  instrument_context: {required: true, description: "resolved ticker identity"}
  trade_date:         {required: true}
  asset_type:         {required: false, default: stock}
---
You are a Bull Analyst advocating for investing in {{instrument_context}}
({{asset_type}}) as of {{trade_date}}. Build an evidence-based case from the
analyst reports in your prompt; engage the bear's last argument directly.
Prefix your answer with "Bull Analyst:".
```

**Profile set** (generated by `new profile-set`, then edited):

```yaml
# .jaato/profiles/_base_portfolio_manager.yaml   — provider-agnostic
name: _base_portfolio_manager
description: Judge of the risk debate; emits the final rating.
plugins: []                          # tools arrive as client_tools from the driver
max_turns: 6
spawn_payload_schema:
  type: object
  additionalProperties: false
  required: [instrument_context, trade_date, asset_type]
  properties:                        # string-typed: they cross as key=value tokens
    instrument_context: {type: string}
    trade_date:         {type: string, pattern: '^\d{4}-\d{2}-\d{2}$'}
    asset_type:         {type: string, enum: [stock, crypto]}
    past_context:       {type: string}
completion_payload_schema: completion_schemas/portfolio_decision.json
completion_processors:
  - script: scripts/processors/pm_decision.py   # validate: price fields; render: 5_portfolio/decision.md
    name: pm_decision
    max_refusals: 2
    on_exhausted: allow
    output: "results/{trade_date}/5_portfolio/decision.md"   # {field}: payload, then agent_params; relative = under the workspace

# .jaato/profiles/openrouter_sonnet/portfolio_manager.yaml — the deep model
name: portfolio_manager
inherits: [_base_portfolio_manager]
plugins: []
model: anthropic/claude-opus-4.1
provider: openrouter
plugin_configs:
  openrouter:
    api_key: "${JAATO_OPENROUTER_API_KEY}"
    api_params: {temperature: 0.0, strict_tools: true, enable_thinking: true, thinking_level: high}
```

**Completion schema** (`PortfolioDecision.model_json_schema()`, strict-mode
shaped):

```json
{ "type": "object", "additionalProperties": false,
  "required": ["rating", "executive_summary", "investment_thesis", "price_target", "time_horizon", "errors", "warnings"],
  "properties": {
    "rating":            {"type": "string", "enum": ["Buy", "Overweight", "Hold", "Underweight", "Sell"]},
    "executive_summary": {"type": "string"},
    "investment_thesis": {"type": "string"},
    "price_target":      {"type": ["number", "null"]},
    "time_horizon":      {"type": ["string", "null"]},
    "errors":   {"type": "array", "items": {"type": "string"}},
    "warnings": {"type": "array", "items": {"type": "string"}} } }
```

**Host tools** from the toolkit (the `handler` is the undecorated function):

```python
from tradingagents.agents.utils import agent_utils as tk

def host_tool(t):                       # LangChain @tool -> jaato host-tool spec
    return {"name": t.name, "description": t.description,
            "parameters": t.args_schema.model_json_schema(),
            "handler": lambda args, f=t.func: f(**args), "auto_approve": True}

TOOLS = {
    "market":       [host_tool(t) for t in (tk.get_stock_data, tk.get_indicators, tk.get_verified_market_snapshot)],
    "news":         [host_tool(t) for t in (tk.get_news, tk.get_global_news, tk.get_insider_transactions,
                                            tk.get_macro_indicators, tk.get_prediction_markets)],
    "fundamentals": [host_tool(t) for t in (tk.get_fundamentals, tk.get_balance_sheet, tk.get_cashflow, tk.get_income_statement)],
}
```

**Driver** (the cascade archetype's `main()`, specialised):

```python
async def run(ticker, trade_date, cfg):
    cid = uuid.uuid4().hex
    params = seed_params(ticker, trade_date, cfg)          # instrument_context, asset_type, past_context ...
    state = {}

    for key in cfg["selected_analysts"]:                   # sequential analysts, each a gated stage
        async with _open_session(profile=f"{key}_analyst", agent=f"{key}_analyst",
                                 agent_params=params, client_tools=TOOLS.get(key, []),
                                 cascade_driver_id=cid) as s:
            state[f"{key}_report"] = (await s.complete("Write your report."))["report"]

    async with _open_session(profile="bull_researcher", agent="bull_researcher", agent_params=params, cascade_driver_id=cid) as bull, \
               _open_session(profile="bear_researcher", agent="bear_researcher", agent_params=params, cascade_driver_id=cid) as bear:
        history, last = [], None
        for turn in range(2 * cfg["max_debate_rounds"]):    # ConditionalLogic.should_continue_debate
            side = bull if turn % 2 == 0 else bear
            text = await side.ask(opening_or_rebuttal(state, last))
            history.append(text); last = text
        state["investment_debate_history"] = "\n\n".join(history)

    async with _open_session(profile="research_manager", agent="research_manager", agent_params=params, cascade_driver_id=cid) as s:
        state["investment_plan"] = await s.complete(debate_summary_prompt(state))
    # ... trader, the three-way risk loop, portfolio manager: same two shapes
    memory_log.store_decision(ticker, trade_date, state["final_trade_decision"])
    return state
```

Under the `echo` provider (`plugin_configs.echo: {response: …, tool_call: …,
usage: {prompt_tokens: 1000, output_tokens: 200}}`) the whole driver runs at
zero cost, which is how the pipeline's control flow, the journal/resume
logic and the processors get unit tests that the upstream `MagicMock`-LLM
tests (`tests/test_risk_router_path_map.py`, `tests/test_debate_opening.py`)
only approximate. Two limits of the double: it emits **one** fixed tool
call and then a fixed text, not a scripted sequence, so an analyst's
four-call loop is exercised one stage-profile at a time (a
`signal_completion` call with a canned report is the useful shape); and
without a `usage` block a turn records no spend, emits no terminal event,
and `complete()` waits forever — `validate` flags it as
`echo_reports_no_usage`. The SDK's conformance suite
(`jaato_sdk/conformance/`) builds four echo profiles — prose, terminus,
nudged, refused — one per way a session can end; a port's test matrix
should cover the same four, since every judge stage here terminates inside
a tool-use turn, which is the path a prose-only suite never exercises.

---

## 6. Gaps and risks

| # | Gap | Severity | Mitigation |
|---|---|---|---|
| 1 | **No native `openai`, `azure`, `bedrock` providers.** `explain providers` lists 18; OpenAI, xAI, DeepSeek, Qwen, GLM, MiniMax, Mistral, Kimi, Groq are reached through `openrouter`; a self-hosted or third-party OpenAI-compatible endpoint through `nim` / `vllm` with a `base_url`. Azure OpenAI and Bedrock have no route today. | medium | write the two providers (the OpenAI-compatible base class makes Azure a base-URL + header variant); Bedrock is a genuine new adapter. Not needed for the first cut. |
| 2 | **Pipeline resume is driver code, not a framework feature.** jaato persists sessions, not pipeline position; the driver journals stage payloads. Granularity is the driver's choice — stage-level, or finer than upstream by splitting an analyst into choose / fetch / write sub-stages (§4.8). | low | ~50 lines; journal per debate turn and per sub-stage. |
| 3 | **Provider quirks live in TradingAgents' client layer** (DeepSeek `reasoning_content` round-trip, MiniMax `reasoning_split`, content-block flattening). jaato's providers carry their own quirk tables; whether every TradingAgents-curated model behaves is a per-model check, not a design question. | low–medium | run the `echo`-free smoke on each model the profile set names; `quirks:` is the escape hatch. |
| 4 | **Structured-output reliability on small models.** Upstream degrades to free text; jaato re-prompts within `max_turns` and the processor's `max_refusals`. A model that never calls `signal_completion` burns its budget and returns nothing. | medium | `on_exhausted: allow` plus `is_review` fallback; pick `max_turns` per stage from a dry run; `strict_tools` where the upstream supports it. |
| 5 | **Per-stage session cost.** In-process, a session is a `JaatoSession` construction — cheap. In daemon mode each stage claims a warm pool slot (~7 s bootstrap per stage, shared across the cascade). Twelve-plus stages per run is fine; sub-second latency per node is not on offer either way. | low | in-process for the library path; daemon only when observers or multi-tenant isolation matter. |
| 6 | **Prompt-cache locality.** Each stage is a fresh session with its own system prompt, so cross-stage cache reuse is nil; within a debate each side's prefix is cached per turn. This matches upstream's economics (each node re-sends its prompt too). | none | — |
| 7 | **Global vendor config.** `dataflows/config.py` stays a process singleton; two concurrent runs with different `data_vendors` in one process are not supported, exactly as upstream. | low | the `ToolPlugin` follow-up moves it into `plugin_configs`. |
| 8 | **Licence — only for what is copied.** A reimplementation of the pipeline owes TradingAgents nothing: copyright covers expression, not the architecture. Apache-2.0 attaches only to the modules this document proposes to carry over verbatim (`dataflows/`, `schemas.py`, `memory.py`, `reporting.py`, `rating.py`) and to prompts lifted word for word, and its obligations there are attribution and notice preservation, compatible with jaato's BUSL-1.1. | note | keep licence text and notices for copied files, or reimplement them and the note disappears; the BUSL additional-use grant governs redistributing the port itself. Not legal advice. |
| 9 | **`mcp` version mismatch in the dev venv** (the traceback in §3). | none for the port | pin `mcp` when the daemon path is used. |
| 10 | **Host tools are attach-bound.** They live in the driver process; a cold session cannot use them until a client attaches (`session.wake` → `DEFERRED`). | low in-process; medium for daemon deployments | the `ToolPlugin` follow-up (Phase 3) moves `dataflows/` into the runner. |
| 11 | **`echo` cannot script a multi-turn tool loop** (one call, then text). Driver-level tests cover control flow and processors; the analyst loop itself is tested against a real model or a recorded provider trace. | low | per-stage echo profiles; `trace.provider_log` for replay fixtures. |
| 12 | **No embedding memory in the free package.** Irrelevant to parity (§4.5), but a future "similar past situations" feature needs a `jaato.embedding` provider or an external store surfaced through a prefetch or host tool. | none for parity | out of scope. |

---

## 7. Effort and phasing

| Phase | Scope | Size |
|---|---|---|
| 0 — scaffold | `new profile-set --agents <12 roles>`, personas lifted from the twelve prompts, four completion schemas from `schemas.py`, host-tool specs from the toolkit, `echo` profile set for tests; `validate` + `doctor` green | ½ day |
| 1 — parity on `propagate()` | cascade driver with the analyst loop, two debate loops, three gated judges; memory log and reflection wired; `reporting.py` kept; processors for the four structured agents; port the ~45 framework-free upstream tests and add `echo`-driven driver tests | 2–3 days |
| 2 — operability | stage journal + resume, progress subscriber (or `observer` in daemon mode), stats from the ledger, `budget_control` per profile, `sweep` for backtests | 1–2 days |
| 3 — optional | `ToolPlugin` packaging of `dataflows/` for daemon deployments; Azure / Bedrock providers if required; `memory` plugin for PM-curated lessons; TUI attach | as needed |

Roughly 1–2 engineer-weeks to a port that passes the upstream logic tests,
with the framework-bound 17 test files rewritten against the driver rather
than the graph.

---

## 8. Summary

- **Feasible, and structurally simpler than the original**: the graph is a
  fixed pipeline, so it becomes a driver over per-node sessions, and the
  message-sharing machinery LangGraph needed has no counterpart to write.
- **The valuable Python survives unchanged**: `dataflows/`, `schemas.py`,
  `memory.py`, `reporting.py`, `rating.py`, and most of the tests.
- **Every structural decision came from `jaato-scaffold`**: the cascade and
  host-tools archetypes are the driver and the toolkit; the profile set is
  the two-model split; `completion_payload_schema` + processors are the
  structured output; the persona prefetch is the sentiment pre-fetch; the
  sweep is the backtest.
- **Two real gaps**: no Azure/Bedrock providers, and pipeline resume is
  driver code. Neither blocks a first cut.
