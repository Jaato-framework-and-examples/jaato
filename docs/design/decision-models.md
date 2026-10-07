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
are a `decide` tool, a permission evaluator, completion gates and routing,
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

Each entry is a (provider, model) pair, like a tier, built lazily on first
use and cached per session. A consumer names an entry, never a model
directly, so swapping Jev for Laya is a profile change.

Inheritance: child replaces, by entry name. `jaato-scaffold validate`
reports an entry whose model the catalog does not report as `decisions`
(`decision_model_not_decisions`, warn, since the knob may assert it), an
entry on a provider that does not declare `capabilities.decisions`
(error), and a consumer naming an entry that does not exist (error).

## 8. Consumers, in build order

### 8.1 The `decide` tool

A tool plugin, `decide(model, state, questions)`, where `model` names a
`decision_models` entry. The agent sends a ticket or a row together with
its questions and gets typed answers back, which makes map-reduce over a
dataset cheap. Its schema exposes the three question types and enumerates
the configured entry names. It is not auto-approved: it spends money,
however little, and sends workspace content to a third party.

This needs nothing from the framework beyond §4-§7, and it proves the
contract end to end.

### 8.2 The permission evaluator

The motivating fit. TypeSafe's own playground example is *"A noul question
returns the probability that a condition holds. Gate an agent tool call on
it"*, using a proposed `delete_rows` call as the state. Map it onto the
existing evaluator seam (`permission/evaluator.py`):

```yaml
plugin_configs:
  permission:
    decision_gate:
      model: triage
      tools: [cli_based_tool, writeNewFile]   # or "default"
      question: "Is this action safe to run without a human approving it first?"
      allow_above: 0.9
      deny_below: 0.1
```

- `noul >= allow_above` → ALLOW
- `noul <= deny_below` → DENY, with a comment naming the probability so the model can read it
- otherwise → **ASK**: a person decides, which is the existing escalation path

The state is the tool name, its arguments and the session's task, built by
the framework, never by the model.

Two gaps in today's seam:

- **`PolicyDecision` has no ASK.** `FALLBACK` defers to the policy, which only asks when `defaultPolicy` is `ask`. The gray zone must ask whatever the default says, so the evaluator vocabulary gains `ASK`.
- **Evaluators are scripts with no provider handle.** The gate is built into the permission plugin instead (it owns the call, the thresholds and the DECISION trace line), and `EvalContext` gains an optional `decide` callable so a hand-written evaluator script can use a configured entry too.

Every gate decision is traced on the existing `[PERMISSION] DECISION` line
with `method=decision_gate`, the model, and the probability, so an
auditor can see that a machine allowed something and with what
confidence. A gate whose call fails (timeout, `529`) **asks**; it never
allows on an error.

### 8.3 Later

- **Completion processors.** A scored acceptance gate, via the same `decide` callable on the processor context.
- **Tier routing.** A `choice` over the declared tiers before a turn, so `enter_tier` happens without the planner spending a turn deciding to switch.
- **Reactors / webhook triage.** One call per inbound event to route it to a session.
- **References.** Ranking under the `max_transitive_references` cut.

## 9. Costs and limits

- **Laya in-process means torch in the runner.** That brings back the #1565 problem: a slot that loaded a model is retired rather than returned to the pool. The `laya` adapter should prefer an HTTP endpoint, and an in-process mode should declare the slot-retirement reason the same way `references` does.
- **Calibration is the vendor's claim, not ours.** jaato forwards the probabilities and confidence it is given. Threshold defaults in §8.2 are examples, not recommendations, and `validate` warns when a gate declares none.
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
   - Not yet verified against the live endpoint.
3. `decision_models:` and the tier refusal (§7).
4. The `decide` tool (§8.1).
5. The permission gate, with `PolicyDecision.ASK` (§8.2).
6. The `laya` adapter.
