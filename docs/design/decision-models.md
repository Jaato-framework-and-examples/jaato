# Decision Models (System One): `decisions` as an Output Modality

Status: steps 1-2 of §11 are **shipped** (the modality and `decide()` on
`openrouter`); steps 3-6 are proposed.

## 1. What a decision model is

A decision model takes a **state** (text, an email, a ticket, a JSON
record) and a map of **typed questions**, and returns **typed answers with
calibrated probabilities**, in one forward pass. It generates no text,
holds no message history and calls no tools.

Two models motivate this design, and they already share one schema:

| | Jev (TypeSafe) | Laya (ConvAI Innovations) |
|---|---|---|
| Served | `POST https://api.typesafe.ai/v1/systemone`; through OpenRouter at `POST https://openrouter.ai/api/alpha/decisions` | open weights on Hugging Face (ModernBERT-large / mmBERT-base + a two-layer decision head) |
| Question types | `noul` (yes/no probability), `choice` (one of up to 255 options), `score` (an ordered rubric of 2-10 levels) | the same three, under the same names |
| Context | 32K tokens of state | 512 (English) / 1,024 (multilingual) tokens |
| Latency | 70-500 ms end to end | ~33 ms per forward pass |
| Price (OpenRouter, Jev 1.13) | $0.042 / 1M input tokens, output free | self-hosted |

### 1.1 The wire, as documented (TypeSafe API reference, Jev 1.13)

Request:

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent":   {"type": "noul", "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive",
                                 "false": "No urgency expressed"}},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds",
                                 "technical": "Bugs, outages, integrations",
                                 "sales": null}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}
```

- `state` is a string, an object or an array.
- `instructions` and every `criteria` value may be a string, an object or an array, so a question can carry structured reference data.
- Question ids are the caller's own. They are not sent to the model and come back as the keys of `answers`.

Response:

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent":   {"type": "noul", "noul": 0.95},
    "department":  {"type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                    "confidence": 0.81},
    "frustration": {"type": "score", "score": 1.05,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                    "confidence": 0.92}
  },
  "usage": {"input_tokens": 296, "output_tokens": 20}
}
```

Fields to note:

- A `noul` answer carries no `confidence`; its probability is the answer.
- A `score` answer is probability-weighted, so it can land between levels.
- `confidence` is derived from the distribution and is separate from it: two options at 0.5 each give a confident distribution but low confidence.

Errors: `401` bad key, `422` malformed question (the body names the
field), `429` rate limit, `529` overloaded. Both `429` and `529` mean
retry with backoff.

### 1.2 OpenRouter already calls it a modality

`GET /api/v1/models/typesafe/jev-1.13/endpoints` (fetched 2026-10-06):

```json
"architecture": {"modality": "text->decisions",
                 "input_modalities": ["text"],
                 "output_modalities": ["decisions"]},
"context_length": 32000,
"supported_parameters": []
```

Two details matter for detection:

- **`typesafe/jev-1.13` is not in the `GET /api/v1/models` listing.** Only `typesafe/jev-router` is. The per-model `/endpoints` document is where the modality is reported, so a lookup by listing alone will not find it.
- **`typesafe/jev-router` is a different thing**: a chat-shaped router (`…->text`, chat parameters, `pricing: -1`) that uses Jev internally to pick a downstream model. It works through the existing `openrouter` chat path today and is out of scope here.

## 2. The proposal in one paragraph

`decisions` becomes an **output modality**, declared beside `text` and
`audio`. It is detected from the catalog where the catalog reports it, and
asserted through the existing `output_modalities` knob where it does not.
A provider that serves such a model gains a **separate call**,
`decide(state, questions) -> DecisionResult`, sent to the decision endpoint
instead of chat completions. `complete()` on a decisions-only model refuses
by name. The session's agent loop never runs on a decision model. Callers
are a `decide` tool, a permission gate, completion gates and routing,
each of which hands the model a narrow, typed question and acts on the
calibrated answer.

## 3. Why a separate call, not a rerouted `complete()`

The modality does choose the endpoint, as the `openai` provider's
`api: chat | responses` selector does. The difference is that `chat` and
`responses` answer the same question (*what is the next assistant
turn?*), and a decision endpoint answers a different one. Hiding it
behind `complete()` would break three things:

1. **The session loop cannot run on it.** `JaatoSession` expects a `ProviderResponse` of parts and function calls and loops until the model stops calling tools. A decision model never calls a tool, writes no text and never calls `signal_completion`, so the completion nudge, `NudgeExhausted` and the #687 truncation guard would all see empty turns.
2. **The input is a different shape.** The model needs a state and a question schema, not a message history. Laya's window is 512 tokens, so a history could not be sent anyway. GC, the instruction budget, prompt caching and reasoning replay have nothing to act on.
3. **Every consumer of `ProviderResponse` would need to understand a second shape.** That is two meanings in one return type, the "one check, one door" failure this tree has paid for before.

So the rule is: **the modality decides which methods a provider serves for
a model, not which URL `complete()` posts to.**

## 4. The modality

### 4.1 Vocabulary

Add `MODALITY_DECISIONS = "decisions"` beside the existing modality
constants, spelled as OpenRouter spells it. It is **output only**: no input
gate reads it, and `modalities()` (input) never returns it.

### 4.2 The text floor needs one exception

`ModalityCapabilityMixin.output_modalities()` always adds `text` ("a model
that speaks still writes"). That is wrong for a decision model, which
writes nothing. The rule becomes:

- a resolved set that contains `decisions` and no other generative modality is returned **as is**, without `text`;
- every other set keeps the text floor, unchanged.

`is_decisions_only(model)` is the predicate both the `complete()` refusal
and the tier check read, so the two cannot disagree.

### 4.3 Resolution order (OpenRouter)

Catalog first, then the knob, then the text floor, mirroring
`modalities()` for input:

1. `architecture.output_modalities` from the per-model `/endpoints` document (§1.2), cached per model as the listing is.
2. `framework_overrides.output_modalities` (existing knob).
3. `{text}`.

The comment on `_apply_media_output_knobs` that says the catalog reports
nothing about output becomes false and is corrected in the same change.

### 4.4 Capability declaration

`ProviderCapabilities.decisions: bool = False` says the adapter **delivers**
`decide()` on the wire. It plays the same role `output_media` plays for
audio. The behavioural half of the capability guard drives `decide()`
against a stand-in server for every provider that sets it.

## 5. The types

Placed in `jaato_sdk` so clients and plugins can build questions without
importing the server:

```python
@dataclass(frozen=True)
class DecisionQuestion:
    type: Literal["noul", "choice", "score"]
    instructions: Union[str, dict, list]
    criteria: Union[None, dict, list] = None
    # noul:   {"true": ..., "false": ...} or None
    # choice: {option: description | None}, 1..255 options
    # score:  [level, ...], 2..10 levels

@dataclass(frozen=True)
class DecisionAnswer:
    type: str
    value: Union[float, str]              # noul probability | choice option | score
    probabilities: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None    # None for noul: absent is not zero
    legend: Optional[Dict[str, str]] = None

@dataclass(frozen=True)
class DecisionResult:
    model: str                            # the model that answered (e.g. "jev-1.13.0")
    answers: Dict[str, DecisionAnswer]
    usage: TokenUsage                     # reported=False when the wire sent none (#688)
```

Rules:

- **Validated before sending.** Option counts, level counts, a `criteria` shape that matches `type`, unique ids. A malformed question is a local error naming the question, not a `422` round trip.
- **An answer missing from the response is an error naming the id**, never a default. A decision that silently became "no" or the first option is the worst failure this feature could have.
- **`confidence` absent stays absent.** A `noul` carries none, and a reader must not see `0.0`.
- **No JSON-Schema translation layer in v1.** The vendor shape is already typed and small. Whether `completion_payload_schema` authors should write decisions in JSON Schema (an `enum` for `choice`) is an open question (§10).

## 6. The provider surface

```python
class DecisionCapable(Protocol):
    def decide(
        self,
        state: Union[str, dict, list],
        questions: Mapping[str, DecisionQuestion],
        *,
        model: Optional[str] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> DecisionResult: ...
```

- **Not streamed.** One response, one parse.
- **Cancellable.** The token closes the HTTP request, as the chat path does.
- **Bounded.** The provider's existing connect and request deadlines apply (`framework_overrides.connect_timeout` / `request_timeout` on OpenRouter). The stream-idle deadline does not.
- **Retries** go through the existing `with_retry` with the vendor's classification: `429` / `529` are transient, `401` / `422` are not.
- **Sized against the window.** The state and questions are estimated against `context_length` before sending, and an oversize state is refused locally. Nothing is ever truncated silently.
- **`complete()` on a decisions-only model** raises `DecisionModelOnlyError` naming the model and pointing at `decide`.

### 6.1 Adapters

| Adapter | Wire | Notes |
|---|---|---|
| `openrouter` | `POST /api/alpha/decisions` | Reuses the provider's key resolution, attribution headers (`HTTP-Referer`, `X-OpenRouter-Title`), deadlines and `cost` parsing. **`/api/alpha/` is not a stable route**, so the parser is defensive and the guard fixtures are the vendor's documented examples |
| `typesafe` (optional) | `POST https://api.typesafe.ai/v1/systemone` | The same body. Only worth adding if a deployment needs the vendor directly (its own rate limits, no gateway) |
| `laya` | local forward pass, or an HTTP endpoint serving it | See §9 for the in-process cost |

`decide()` on a chat model with structured outputs (a JSON-schema
`response_format` plus logprobs) is a possible fallback for providers with
no decision model, but it is **not** in v1. Its confidences would not be
calibrated in the sense these models are trained for, and putting the two
behind one type would let a reader mistake one for the other. If it is
added, `DecisionResult` carries a `calibrated: bool`.

## 7. Binding: outside the tier ladder

A decision model must never become the session's active model.
`enter_tier` into one would strand the agent (§3.1). So:

- a tier whose resolved model is decisions-only is **refused at startup** by the tier check (`supports_output_modality` already runs there), with a message pointing at the binding below;
- decision models are bound in their own profile block:

```yaml
decision_models:
  triage:                       # a name consumers refer to
    provider: openrouter
    model: typesafe/jev-1.13
  local:
    provider: laya
    model: convaiinnovations/laya
```

Each entry is a (provider, model) pair, like a tier: `model` is required,
and `provider` defaults to the session's own. The session builds an entry's
provider on first use (`JaatoSession.decision_provider(name)`), as an
instance of its own that the chat provider never shares, and caches it for
the session. A consumer names an entry, never a model directly, so swapping
Jev for Laya is a profile change.

Inheritance: child replaces, by entry name. Two parents binding one name to
different models, with no child entry to settle it, is a conflict.

`jaato-scaffold validate` reports:

| Code | Severity | When |
|---|---|---|
| `decision_model_provider_unsupported` | error | the entry's provider does not declare `capabilities.decisions`, so it has no `decide()` |
| `decision_model_provider_unknown` | error | the entry's provider is not installed |
| `decision_model_as_tier` | error | a `model_tiers` entry binds a model the profile declares under `decision_models:` |

Whether the catalog reports an entry's model as `decisions` is not checked
statically: `validate` makes no network requests. A tier bound to a
decisions-only model the profile does not declare is still refused, at
session start. A consumer naming a missing entry is checked once the
consumers exist (steps 4 and 5); at runtime it raises
`UnknownDecisionModelError`, which lists the bound names.

## 8. Consumers, in build order

### 8.1 The `decide` tool

A tool plugin, `decide(model, state, questions)`, where `model` names a
`decision_models` entry. The agent sends a ticket or a row together with
its questions and gets typed answers back, which makes map-reduce over a
dataset cheap. Its schema exposes the three question types and enumerates
the configured entry names. It is not auto-approved: it spends money,
however little, and sends workspace content to a third party. It is
deferred, in its own `decision` tool category (typed judgements from a
decision model), so `list_tools` groups it apart from web fetching.

This needs nothing from the framework beyond §4-§7, and it proves the
contract end to end.

Shipped as the `decision` plugin (`shared/plugins/decision/`, runner tier,
deferred). What the call looks like:

```
decide(model="triage", state="<ticket text or JSON>", questions=[
  {"id": "urgent", "type": "noul",   "instructions": "Is this urgent?",
   "true_means": "needs a reply today"},
  {"id": "team",   "type": "choice", "instructions": "Which team?",
   "options": [{"option": "shipping", "description": "deliveries"},
               {"option": "billing"}]},
  {"id": "anger",  "type": "score",  "instructions": "How frustrated?",
   "levels": ["calm", "annoyed", "furious"]}])
-> {"model": "triage", "answered_by": "...", "answers": {...}, "usage": {...}}
```

| Rule | Why |
|---|---|
| criteria are typed fields per question type (`options`, `levels`, `true_means` / `false_means`), not the wire's free-form `criteria` | a schema property with no type is rejected by some providers, and typed fields are what a model fills in correctly |
| `questions` is an array of objects with an `id`, not an object keyed by id | no `additionalProperties` in the schema, which some providers reject |
| hidden from a session whose profile binds no entry; `model` is an `enum` of the calling session's entry names | read per session at call time (`is_tool_visible`, `narrow_tool_schema`), because the plugin instance is shared with in-process subagents |
| a refusal is `(False, {error})` | an unknown entry, a malformed question, a provider error: the reliability plugin and the trace see a failure (#1053) |
| a `decision` record in the session's token ledger | the call's spend is invisible to the chat accounting |
| at most 32 questions a call | one call is one decision about one state |

Not covered: a decision call's spend does not feed `budget_control` or the
consumption report; the ledger record is the only account of it.
`jaato-scaffold validate` warns `decision_tool_without_models` when a
profile enables `decision` and binds no entry.

### 8.2 The permission gate

The motivating fit. TypeSafe's own playground example is *"A noul question
returns the probability that a condition holds. Gate an agent tool call on
it"*, using a proposed `delete_rows` call as the state. The gate is built
into the permission plugin and configured beside its policy:

```yaml
plugin_configs:
  permission:
    decision_gate:
      model: triage                 # a decision_models: entry of the session
      tools: [cli_based_tool, writeNewFile]   # or "default": every tool
      question: "Is this action safe to run without a human approving it first?"
      allow_above: 0.9
      deny_below: 0.1
```

- `noul >= allow_above` → ALLOW
- `noul <= deny_below` → DENY, with a comment naming the probability so the model can read it
- otherwise → **ASK**: a person decides, which is the existing escalation path

`question` is optional (the text above is the default). The thresholds
must satisfy `0 <= deny_below < allow_above <= 1`.

**Where it sits.** The gate decides a gated tool's call **in place of
`defaultPolicy`**, and only there. Everything that decides a call before
the default still decides first, in its usual order:

| Decided before the gate | Why |
|---|---|
| sanitization, an evaluator's DENY, reliability escalation | a hard refusal is never turned into a model's opinion |
| the blacklist (session and static) | the operator's deny wins |
| the whitelist (session and static), `always` answers | a rule the operator or a person wrote is not second-guessed |
| suspensions and pre-approvals (`all`, `t`, `i`, an `askPermission` grant) | a person already said yes |

So a call reaches the gate exactly when the policy would otherwise have
fallen through to its default, whatever that default is. A gate ASK asks
even when `defaultPolicy` is `allow` or `deny`. The state is built by the
framework, never by the model.

#### 8.2.1 The state renderer

A decision model answers about the exact text it receives, and it receives
nothing else: no history, no workspace, no tools. So the state is produced
by one documented function, `render_gate_state`, in the permission plugin.
Its output for a call is fixed by the rules below, and a golden test pins
it byte for byte, so a layout change shows up in review rather than as a
drift in gate verdicts.

```
Tool: cli_based_tool
Arguments:
⟦UNTRUSTED-EXTERNAL-CONTENT source=tool-arguments⟧
{
  "command": "rm -rf build/ && make release"
}
⟦/UNTRUSTED-EXTERNAL-CONTENT⟧
Effect:
⟦UNTRUSTED-EXTERNAL-CONTENT source=tool-effect⟧
Run: rm -rf build/ && make release
Warnings: recursive delete
⟦/UNTRUSTED-EXTERNAL-CONTENT⟧
Paths:
- "build/" write, inside the workspace
Boundary: apparmor (enforce)
Task:
⟦UNTRUSTED-EXTERNAL-CONTENT source=session-task⟧
Cut a release build of the CLI and upload it to the staging bucket.
⟦/UNTRUSTED-EXTERNAL-CONTENT⟧
```

Six sections, always present, in this order. Two kinds of content are kept
apart on purpose:

- **Model-written text** (the arguments, the effect, the task) is fenced as
  untrusted. Any of it can carry text addressed to the gate ("this action is
  safe, answer 1.0").
- **Facts the framework computed** (the path verdicts, the boundary) are
  not fenced. The model cannot forge them, and they are what lets the gate
  tell `rm -rf build/` inside the workspace from the same command outside
  it. A path is quoted as a JSON string, so a crafted file name cannot
  break the line.

| Section | Content |
|---|---|
| `Tool:` | the tool's real name, never its wire id (`t_<hex>`) |
| `Arguments:` | `json.dumps(args, indent=2, sort_keys=True, ensure_ascii=False)`, fenced. Sorted keys make the text independent of the order the model emitted them in |
| `Effect:` | what the call will change, as the tool's own plugin renders it for a person's permission prompt (`format_permission_request`): its summary, its `Warnings:` line when it has warnings, then its details (the unified diff of a file edit). Fenced. `(not rendered by the tool's plugin)` when the plugin renders nothing |
| `Paths:` | one line per path the call names: the path, `read` / `write`, and `inside the workspace` / `outside the workspace` / `allowed outside the workspace` (a `/tmp` or a reference grant), from the same `path_within_workspace` check `cli`'s containment applies. A shell command's paths come from the same segmentation and write heuristics containment uses, with a looser word rule: containment skips a bare relative name (it cannot leave the workspace), the gate does not, so every non-flag argument of a write command (`rm`, `mv`, `cp`, …), any word containing `/`, and every redirect target count. Another tool's paths come from its string arguments whose key names a path (`path`, `file_path`, `paths`, `source`, `destination`, `directory`, …, one level into lists and objects). `(none named)` when there are none; `(no workspace boundary)` when the session has no workspace root |
| `Boundary:` | what a mistake can reach: `apparmor (enforce)`, `apparmor (complain)`, `selinux (enforcing)`, `selinux (not attested)` or `unconfined`, from the runner's own label and the recorded SELinux boundary, read the way `get_environment(aspect="runtime")` reads them |
| `Task:` | the session's first user message, fenced. For a subagent that is the task it was spawned with. `(none)` before the first message |

Rules that apply to every section:

| Rule | Why |
|---|---|
| **A string value over 2,000 characters is cut** to its first 1,500 and last 400 characters, joined by `… [N characters omitted] …`. The task and the effect are cut the same way | a whole file passed to `writeNewFile` would otherwise fill the window; the head and tail are where a command's intent usually is |
| **Later messages, earlier tool results and the agent's own reasoning are not included** | they grow without bound, they are the main injection route, and a state whose size depends on the conversation makes the thresholds drift |
| **Secrets are redacted before rendering** (`secret_redaction`, #1215) | the state leaves the host; a token the model put in a `curl` argument must not go with it |
| **No timestamps, ids, or the workspace's own path** | anything that varies per call without changing the decision adds noise to the verdict. A path inside the workspace is shown relative to it |

**Fitting the window.** The rendered state plus the question must fit the
model's context window (64,000 tokens for `typesafe/jev-1.13`, from the
live catalog), at the ~4 characters a token `decide()` itself assumes. If
it does not after the per-value cuts, the longest remaining string is cut
further, repeatedly, down to a floor of 200 characters per value. If it
still does not fit, the gate does not call the model and **asks**, with
the reason in the DECISION trace line (`decision_gate: state over
window`). `decide()`'s own pre-send window check stays as the backstop.

**What the DECISION line records** is `method=decision_gate`, the entry
name, the probability and the SHA-256 of the rendered state, not the
state itself, so an auditor can match a verdict to a re-rendered call
without the trace holding the arguments twice.

Moderation builds its state the same way, with its own renderer
(`render_moderation_state`): the content verbatim, wrapped as untrusted,
under a `Checkpoint:` line and, for tool results, a `Tool:` line. See
[Content Moderation](content-moderation.md) §4.1.

#### 8.2.2 Two gaps in the evaluator seam

- **`PolicyDecision` has no ASK.** `FALLBACK` defers to the policy, which only asks when `defaultPolicy` is `ask`. The vocabulary gains `ASK`: a person decides whatever the default says. An evaluator's ASK is not lifted by a whitelist entry, a suspension, the trusted bridge or an `askPermission` grant (the reliability re-confirmation's posture); a blacklist still denies.
- **Evaluators are scripts with no provider handle.** `EvalContext` gains an optional `decide(name, state, questions)` callable bound to the calling session's `decision_models:`, so a hand-written evaluator script can use a configured entry too. `None` when the session binds none.

**Failure asks.** A gate whose call fails (timeout, `529`, an unknown
entry, a provider without `decide()`) **asks**; it never allows on an
error. A gate whose session binds no decision models, or no session at
all, asks too. A malformed `decision_gate` block is reported by
`jaato-scaffold validate` and logged at ERROR, and the plugin then runs
without a gate (the default decides, as before).

### 8.3 Later

- **Completion processors.** A scored acceptance gate, via the same `decide` callable on the processor context.
- **Tier routing.** A `choice` over the declared tiers before a turn, so `enter_tier` happens without the planner spending a turn deciding to switch.
- **Reactors / webhook triage.** One call per inbound event to route it to a session.
- **References.** Ranking under the `max_transitive_references` cut.

## 9. Costs and limits

- **Laya in-process means torch in the runner.** That brings back the #1565 problem: a slot that loaded a model is retired rather than returned to the pool. The `laya` adapter should prefer an HTTP endpoint, and an in-process mode should declare the slot-retirement reason the same way `references` does.
- **Calibration is the vendor's claim, not ours.** jaato forwards the probabilities and confidence it is given. The thresholds in §8.2 are examples, not recommendations; a gate has no default thresholds, so `validate` refuses a `decision_gate` that declares none.
- **Data leaves the host.** A decision state is workspace content sent to TypeSafe (directly or via OpenRouter). `routing.data_collection: deny` and `zdr` apply on OpenRouter as they do for chat.
- **Usage accounting.** Output tokens are free on Jev but still reported. Decision calls are recorded in the ledger as their own record type (`decision`), not as a turn, and count toward `budget_control.limits.usd` / `tokens` but never `turns`.

## 10. Open questions

1. Should questions also be authorable in JSON Schema (`enum` for `choice`, a bounded `integer` for `score`), so a `completion_payload_schema` author and a decision author share one vocabulary?
2. `typesafe/jev-router` is chat-shaped and works today. Do we want it documented as a routing option (like `openrouter/auto`), or left alone?
3. Does the permission gate run before or after the whitelist/blacklist? Proposed: after the blacklist (a deny stays a deny) and before the default policy.
4. Should the `decide` tool batch: let the model send many states with one question set and get a list back, which is the map-reduce case?

## 11. Rollout

1. The modality vocabulary, the text-floor exception and catalog detection on `openrouter` (§4). Small, and testable without a key. **Shipped**: `MODALITY_DECISIONS`, `normalise_output_modalities` / `is_decisions_only_set` in `model_provider/base.py`, `ModalityCapabilityMixin.is_decisions_only`, and `OpenRouterProvider.output_modalities` (listing, then the per-model endpoints document, then the knob). The listing turned out to report `output_modalities` for every model, so the audio models resolve without the knob too. Guard: `jaato_server/shared/tests/test_decisions_is_an_output_modality.py`, four reversions.
2. The types and `decide()` on `openrouter` against `/api/alpha/decisions`, with the vendor's examples as fixtures and a stand-in server in the capability guard (§5-§6). **Shipped**:
   - The wire contract is `jaato_sdk/plugins/model_provider/decisions.py`, with no I/O. It holds the types, `build_decision_request` (validates locally, so a malformed question never costs a `422`) and `parse_decision_response`. A question left unanswered, an answer of another type, a choice outside the options, or a probability outside [0, 1] each raise `DecisionResponseError` naming the question. `DecisionResult.raw` keeps the body as received.
   - `ProviderCapabilities.decisions` is a new column. Only `openrouter` declares it.
   - `OpenRouterProvider.decide()` posts to the endpoint derived from `base_url` (`…/api/v1` becomes `…/api/alpha/decisions`); `framework_overrides.decisions_url` overrides it. The request carries the chat key and attribution headers (`_attribution_headers`, one definition for both wires) and runs under the connect and request deadlines, on a worker thread so a cancel token stops the wait and closes the client.
   - Errors and retries: `429` / `5xx` are retried by `with_retry`. `401`, `404` and `422` are not; a `422` is `DecisionRequestRejectedError` carrying the body.
   - A request estimated over the model's window is refused before sending.
   - `connect()` records whether the model is decisions-only, and `complete()` then raises `DecisionModelOnlyError`.
   - Guards: `jaato_server/shared/tests/test_decide_on_openrouter.py` (five reversions) uses the vendor's documented request and response as fixtures. The stand-in is `decision_standin.py`. In `test_provider_capability_conformance.py`, a declaring provider must answer the stand-in and an undeclared one may not expose `decide`.
   - `examples/provider_smoke_decisions.py` runs the same checks against the stand-in. With `--live-openrouter` it sends one request to `typesafe/jev-1.13` and prints the raw response beside the parse.
   - Verified against the live endpoint on 2026-10-09: the response matched the documented shape, with integer probabilities, a `cost` in usage (kept as `cost_usd`) and `id` / `provider` keys the parser ignores. That response is a fixture (`LIVE_RESPONSE`).
3. `decision_models:` and the tier refusal (§7). **Shipped**:
   - `shared/decision_models.py` is the one definition of an entry: `parse_decision_models` (refuses a malformed entry by name, at most 16 entries), `DecisionModelBinding`, `merge_decision_models`, and the two runtime errors `UnknownDecisionModelError` / `DecisionModelUnavailableError`.
   - `SubagentProfile.decision_models` is read by every profile loader (workspace and premium tiers, inline specs, `SubagentConfig`, `validate_profile`), kept in the session snapshot, merged across `inherits:`, and carried on `SessionInitEnvelope.decision_models` (additive, no version bump; an older runner ignores it).
   - The runner bootstrap and both in-process subagent spawn paths pass it to `create_session(decision_models=)`. `JaatoSession.decision_provider(name)` builds the entry's provider through `runtime.create_provider` with the session's `plugin_configs`, refuses one without `decide()`, and caches it.
   - `JaatoSession._refuse_decisions_only_tiers` runs where the provider is first created and refuses a same-provider tier whose model `is_decisions_only`, naming `decision_models:`. A lookup that fails is not evidence and refuses nothing. A cross-provider tier still fails at its first turn with `DecisionModelOnlyError`.
   - Not covered: the embedded client and the isolated sub-runner do not carry the block (neither carries `model_tiers` either).
   - Guard: `jaato_server/shared/tests/test_decision_models_are_bound_by_name.py`, seven reversions.
4. The `decide` tool (§8.1). **Shipped**: the `decision` plugin. Guard: `jaato_server/shared/tests/test_the_decide_tool.py`, six reversions, driving the real plugin, session and `OpenRouterProvider` against the decision stand-in.
5. The permission gate, with `PolicyDecision.ASK` (§8.2). **Shipped**:
   - `permission/decision_gate.py` is the gate's whole definition: `parse_decision_gate` (no default thresholds; a malformed block is refused by name), `render_gate_state` (the six sections, pinned byte for byte, and the doc's example is asserted to be its output), `fit_gate_state`, `evaluate_gate` (asks on every failure) and a `decision` ledger record with `via: decision_gate`.
   - The plugin runs the gate only where the policy matched its default, after evaluators, the blacklist, the whitelist and suspensions. One `_ForcedAsk` applies the gray zone, an evaluator's ASK and the reliability re-confirmation to the first check and to the re-check under the policy lock. Before it, the re-check undid the reliability re-confirmation on a whitelisted tool and recursed without end.
   - `PolicyDecision.ASK`, `EvalContext.decide`, the DECISION line's `gate_model` / `gate_probability` / `gate_state_sha256`, the config schema entry (`explain plugin permission`), and `validate`'s `decision_gate_invalid` / `decision_gate_unknown_model`.
   - Guard: `jaato_server/shared/tests/test_the_permission_decision_gate.py`, eight reversions, against the real plugin, session and `OpenRouterProvider` with the decision stand-in.
6. The `laya` adapter.
