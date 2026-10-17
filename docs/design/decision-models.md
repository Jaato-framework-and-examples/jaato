# Decision Models (System One): `decisions` as an Output Modality

Status: steps 1-8 of §11 are **shipped**: the modality, `decide()` on
`openrouter`, `decision_models:`, the `decide` tool, the permission gate,
the `laya` adapter, the `typesafe` adapter (§6.2), and images and the `box`
question on `typesafe` (§12.2-§12.4). §8.3 lists the consumers still to
come. Steps 9 (images in the `decide` tool) and 10 (images through
OpenRouter) are not built.

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
| `typesafe` (§6.2) | `POST https://api.typesafe.ai/v1/systemone` | The vendor's own API, the same body. The route TypeSafe documents images and `box` on, so the first adapter to declare `decision_images` (§12.6). Also serves text-only decisions to a deployment with a TypeSafe key and no OpenRouter account |
| `laya` | `POST {host}/v1/systemone` on `laya-serve` (`mode: http`, the default), or `laya.load(...).predict()` in the runner (`mode: local`) | `laya-serve` speaks Jev's request and response, so the HTTP transport is shared with `openrouter`. See §9 for the in-process cost |

### 6.2 The `typesafe` adapter

`model_provider/typesafe/`, a decisions-only provider like `laya`
(`chat=False`, so `jaato-scaffold new` refuses it as `--provider` and
`validate` reports `decision_only_provider` where a profile or tier names
it). It is bound in `decision_models:`, never in a tier.

| | Value (TypeSafe quick start, read 2026-10-16) |
|---|---|
| endpoint | `POST https://api.typesafe.ai/v1/systemone` |
| auth | `Authorization: Bearer <key>`; keys from `console.typesafe.ai/keys` |
| model ids | `jev-latest` (the alias), `jev-1.13.0`, `jev-1.14.0` / `jev-preview` (§12.1). Passed through as written; no table of ids |
| transport | the shared `model_provider/_decisions_http.py` (`openrouter` and `laya` already bind it) |

**The key: the profile knob first, environment variables as fallback.**
The same order every provider here uses (#863's posture: a session's own
configuration outranks the daemon's environment):

1. `plugin_configs.typesafe.api_key` (may carry a `pass://` / `vault://`
   URI, resolved daemon-side like any other knob, #1605);
2. `JAATO_TYPESAFE_API_KEY`;
3. `TYPESAFE_API_KEY`, the vendor's own variable (its SDKs read it), so a
   host already set up for `typesafe-sdk` works with no extra config;
4. the stored `typesafe_auth.json` written by `typesafe-auth key <key>`
   (validated with one `noul` request, stored like the other `*_auth.json`
   files), so an IPC deployment can configure a key from the prompt.

`PROVIDER_AUTH_RESOLUTION` declares the four in that order, and
`test_a_named_auth_command_exists.py` then requires the `typesafe-auth`
command to exist. A key from 2-4 is announced the way the other providers'
`auth_info` names its source, so an operator can tell a profile key from
an inherited one.

Other knobs, under `plugin_configs.typesafe`, each with any env fallback
beneath it:

| Knob | Env fallback | Meaning |
|---|---|---|
| `base_url` | `JAATO_TYPESAFE_BASE_URL` | `https://api.typesafe.ai` by default; a proxy or a regional endpoint |
| `context_length` | `JAATO_TYPESAFE_CONTEXT_LENGTH` | the window the pre-send estimate checks against. TypeSafe publishes 32K tokens of state for 1.13 and no catalog, so with neither set the window is not checked (as `laya` does for an unknown model), never guessed |
| `framework_overrides.connect_timeout`, `framework_overrides.request_timeout` | none | the deadlines `decide()` runs under (§6); 10 s and 60 s, `0` = none |

No `danger_zone` knob: the images header is sent when, and only when, the
request carries an image or a `box` question (§12.2), so it cannot be left
on by a profile that forgot it, nor off on a request that needs it.

Errors and retries follow §6: `429` / `529` / `5xx` and an unreachable host
are retried by `with_retry`; `400` (an unreferenced asset), `401`, `404`
and `422` are not, and a `400` or `422` carries the body.

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
| `decision_only_provider` | error | the profile's `provider:` or a tier's `provider:` declares `capabilities.chat=False` (`laya`): it serves `decide()` only, so the session fails on its first turn |

A provider that has no chat wire at all says so statically, as
`ProviderCapabilities.chat=False`; `jaato-scaffold new profile-set` and
`new client` refuse it as `--provider` before writing anything, and the
conformance guard holds the flag to `complete()` raising
`DecisionModelOnlyError`.

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
5. Should jaato prepare images for the vendor (convert GIF/WebP to PNG, apply EXIF orientation, downscale past 1568 px) when Pillow is installed, instead of refusing them (§12.3)? Two outcomes for one image depending on an optional extra is what argues against it.
6. Should images go multipart instead of base64 (a third smaller, so more fits under the 32 MiB body cap)? v1 sends base64 in JSON because the transport is shared (§12.3).

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
6. The `laya` adapter. **Shipped**: `model_provider/laya/`.
   - `laya-serve` answers `POST /v1/systemone` with Jev's shape, so `mode: http` (the default) posts there through `model_provider/_decisions_http.py`, the transport `openrouter/decisions.py` now also binds to its own error types. `host` defaults to `http://localhost:8000` (where `laya-serve` binds); the bearer token is `plugin_configs.laya.api_key`, then `JAATO_LAYA_API_KEY`, then `LAYA_API_KEY` (the server's own variable). `401` / `404` / `422` are not retried, `429` / `5xx` / an unreachable server are.
   - `mode: local` loads the `laya` package in the runner (`laya.load(model, subfolder=...)`, once per provider), by name, so `validate` does not report the package missing for an HTTP-mode profile. Before the load it notes a process retirement in `shared/process_retirement.py`, which the runner's `session.end` sweep (`slot_retire_reasons`) reports beside the plugins' reasons, so the slot is closed rather than pooled (#1565). A provider is not a registry plugin, which is why it could not answer `slot_retire_reason()` itself.
   - The window: `context_length`, then `JAATO_LAYA_CONTEXT_LENGTH`, then the model card's table (512 English, 1,024 multilingual and typed-decisions). An unknown model's window is not checked. A request estimated over it is refused before sending.
   - Laya truncates a state over its window to one window, and says so in `usage` (`truncated`, `state_tokens`, `state_tokens_dropped`, `truncated_questions`; read from the `laya` 0.4.6 source, `Agent.system_one` and `serve.py`). Such an answer is refused with the same `DecisionQuestionError` (`refuse_truncated`), naming the question when one question's head had no room. A body without those keys (`LAYA_JEV_STRICT`) is not evidence and passes. The pre-send estimate is ~4 characters a token and can be wrong in either direction; this check is the measured one.
   - `model: convaiinnovations/laya` sent to `laya-serve` means "let the server's router choose" (English or multilingual by language); `convaiinnovations/laya-multilingual` and `-typed-decisions` pin a checkpoint. The window table is keyed the same way, so the bundle id is checked against 512.
   - `complete()` and history serialisation raise `DecisionModelOnlyError`; the output set is `{decisions}`.
   - Not run against a live `laya-serve` or the real package (torch and the weights were not installed here); the wire was read from the package source. `examples/provider_smoke_decisions.py --live-laya HOST` is that check.
   - Guard: `jaato_server/shared/tests/test_decide_on_laya.py`, seven reversions, against the decision stand-in (which now also serves `/v1/systemone`) and a stand-in `laya` module.
7. **The `typesafe` adapter, text only** (§6.2). **Shipped**: `model_provider/typesafe/` and the `typesafe_auth` plugin.
   - The key resolves knob first (`plugin_configs.typesafe.api_key` or `ProviderConfig.api_key`), then `JAATO_TYPESAFE_API_KEY`, `TYPESAFE_API_KEY`, then `typesafe_auth.json`; `get_auth_info()` and a `401` name the source. With none, `initialize()` raises `CredentialsNotFoundError` listing all four.
   - `base_url` and `context_length` read the knob, then `JAATO_TYPESAFE_BASE_URL` / `JAATO_TYPESAFE_CONTEXT_LENGTH`. With no window configured the size check is skipped.
   - `typesafe-auth key` validates with one `noul` question to `/v1/systemone` (TypeSafe has no cheaper authenticated endpoint) and stores the key at mode 600.
   - The transport is `_decisions_http.py`; `429` / `5xx` are retried, `401` / `404` / `4xx` are not. No images header is sent (§12).
   - `examples/provider_smoke_decisions.py --typesafe` (stand-in) and `--live-typesafe` (the real API, key from the environment or the store).
   - Not run against the live API here (no key). Guard: `jaato_server/shared/tests/test_decide_on_typesafe.py`, six reversions, against the decision stand-in.
8. **The contract** (§12.2-§12.4). **Shipped**:
   - `jaato_sdk/plugins/model_provider/decision_images.py` reads a still PNG (`IHDR`, refusing an `acTL` before `IDAT`) or JPEG (the first `SOFn`) header with the standard library and holds the vendor's limits (`image_limit_problem`).
   - In `decisions.py`: `DecisionImage`, `DecisionBox` (with `to_pixels`), `box` in `QUESTION_TYPES`, `DecisionImagesUnsupportedError`, and `build_decision_request(..., images=)`. It mints `jaato-image-<uuid4>` per image, substitutes `{{image:NAME}}` in the state, instructions and criteria values, and builds `assets`. It refuses, naming the question where one is to blame: a token naming no image, an image no token names, a token in an object key, a duplicate name, more than 64 images, a mime the bytes contradict, a format other than PNG/JPEG, a size limit, a request over 32 MiB, images over 32 MiB counted per use, `box` with criteria or with no image. With no images the body is byte-identical to before. `needs_image_header(body)` says when the vendor header is due (the doc proposed a flag returned by `build_decision_request`; a separate predicate keeps its return type).
   - `parse_decision_response` reads `box`, refusing an edge outside [0, 1] or `left >= right` / `top >= bottom` with `DecisionResponseError` naming the question.
   - `ProviderCapabilities.decision_images`, a new column every provider declares. `typesafe` declares it, sends `assets`, adds `TypeSafe-Danger-Zone-Images: true` when `needs_image_header` says so, and reports `image` in `modalities()`. `openrouter` and `laya` declare it `False` and refuse `images=` before sending.
   - The stand-in answers `box` with TypeSafe's documented example and answers `400`, as TypeSafe does, to a request with images or `box` and no header, or with an asset referenced nowhere. `png_bytes` / `jpeg_bytes` build fixtures.
   - `examples/provider_smoke_decisions.py --typesafe --image PATH` (stand-in) and `--live-typesafe --image PATH` (one more request, to `jev-1.14.0` unless `--image-model` names another) ask a `noul` and a `box` question and print the box in pixels.
   - Not run against the live API here. Guards: `jaato_server/shared/tests/test_decision_images_and_box.py`, seven reversions; in `test_provider_capability_conformance.py`, a provider declaring `decision_images` must put the image in `assets` and read the `box`, and a decision provider that does not must refuse `images=` with nothing sent.
9. **The `decide` tool's `images`** (§12.5). **Shipped**:
   - `shared/plugins/decision/images.py` resolves each entry of `images` (`{name, path}` or `{name, attachment_id}`, exactly one). A `path` is joined onto the workspace root, checked with `check_path_with_jaato_containment` and read through `path_safety` (no check-then-open race); `.env` at any depth and `.jaato/*_auth.json` are refused, and the format is read from the bytes. An `attachment_id` (`att_<16 hex>`, the ingest digest `media_identity` mints) is looked up in the session's history, user-message parts and tool-result attachments alike, newest first; an id that an eviction marker names is refused as evicted. Every refusal names the image.
   - `JaatoSession.decision_image_model_names()` lists the entries whose provider declares `decision_images`, read from the provider package's `PROVIDER_CAPABILITIES` (`model_provider/_decision_capability.py`) so no provider is built to answer. `narrow_tool_schema` adds `images` and the `box` question type only when it is non-empty, and the description names those entries. The package is found where `load_provider` finds the provider: the built-in package first (it wins a name clash, as in `discover_providers`), else the `jaato.model_providers` entry point of that name, whose module or nearest package must declare `PROVIDER_CAPABILITIES`.
   - A call with `images` on an entry not in that list is refused before any file is read.
   - A `box` answer's `value` is the four edge fractions; beside it, `image` and `box_px` (`DecisionBox.to_pixels` on that image's own dimensions). The image is the one the question's instructions name with `{{image:NAME}}` when they name exactly one, else the call's only image; otherwise `box_px` is omitted rather than computed against a guess.
   - The ledger's `decision` record gains `images`: name, source, ref, sha256, bytes, format, width, height. Never the bytes.
   - Not checked here: whether a client host tool's screenshot result reaches history as an attachment (the computer-use controller's case). Guard: `jaato_server/shared/tests/test_the_decide_tool_takes_images.py`, seven reversions, against the real plugin, session and `TypeSafeProvider` with the decision stand-in.
10. **OpenRouter** (§12.6). Shipped. OpenRouter's Multimodal Decisions guide (2026-10) settled the open questions: an image is an `image_url` part with a base64 `data:` URL inside the text (`state`, and for Jev `instructions` and `criteria`), never `assets` (undocumented) nor a top-level `images` (a `400`); `box` is answered by `typesafe/jev-1.14` and `~typesafe/jev-latest`, and Jev 1.13 answers `400`.
   - One implementation for both routes: `build_decision_request(..., image_layout=)` takes `assets` (TypeSafe's API, unchanged) or `parts` (OpenRouter), both from the one substitution walk, so every check in step 8 holds for both. With `parts` a string holding `{{image:NAME}}` becomes its non-empty text pieces and an `image_url` part per token; inside a list the pieces are spliced in, so a state array stays flat. The request-size check measures the body as built (the bytes are already in it).
   - `openrouter` declares `decision_images` and sends images to TypeSafe Jev only (`openrouter/decisions.takes_images`): OpenRouter's other image decision models (Clef, Mercury Decide, Liquid d1, ...) have their own limits, and the local checks are Jev's. Another model is refused before sending with `DecisionImagesUnsupportedError` naming it. A `box` question for a model that does not answer it (`answers_box`: Jev 1.14 and later, or a version-less alias) is a local `DecisionQuestionError` naming the question. No TypeSafe header is sent: OpenRouter documents none.
   - The capability is per model: a package may define `decision_images_for_model(model)` beside `PROVIDER_CAPABILITIES`, and `_decision_capability.model_declares_decision_images(provider, model)` asks it once `decision_images` is declared. `openrouter` defines it as `takes_images`, so `JaatoSession.decision_image_model_names()` offers `images` only to an entry bound to a Jev model. A rule that raises answers no.
   - The window check counts each `image_url` part as its marker, not its base64.
   - Verified against the live endpoint on 2026-10-17 with `examples/provider_smoke_decisions.py --live-openrouter --image PATH`, on an 800x600 PNG holding one red disc at x 500-700, y 150-350. `typesafe/jev-1.14` answered the `noul` "is there a person" with 0.02 and the `box` with top 0.25, bottom 0.59, left 0.61, right 0.88 (pixels 150 / 354 / 488 / 704) at confidence 0.91, against the true 0.25 / 0.583 / 0.625 / 0.875. The same day, text questions (`noul`, `choice`, `score`) parsed unchanged on four non-Jev decision models: `inception/mercury-decide:free`, `microsoft/microsoft-decision-1`, `perplexity/pplx-decider-flash-v1.2` and `togethercomputer/tev1-4b-experimental`. `respan/span-01-lite:free` accepts `noul` questions only (OpenRouter answers `400` naming the question), and its catalog entry reports a context length of 0. Two fixes followed: the provider reads the catalog with `?output_modalities=all` (the default listing omits every decision model, so each cost a per-model `/endpoints` request), and a decisions-only model connects with no window, its pre-send size check then skipped as on the `typesafe` provider; a chat model with no window is still refused (`test_openrouter_lists_decision_models.py`, two reversions). Guard: `jaato_server/shared/tests/test_decide_on_openrouter_takes_images.py`, five reversions; the decision stand-in holds an OpenRouter request to the guide's rules.

## 12. Images and `box` (Jev 1.14)

Steps 8 (the contract, on `typesafe`), 9 (the `decide` tool) and 10 (OpenRouter, Jev only) are shipped.

### 12.1 What TypeSafe shipped

From the Jev 1.14 post (`typesafe.ai/blog/jev-1-14`) and the two docs pages
it links (`docs.typesafe.ai/go/images`, `docs.typesafe.ai/go/box`), read on
2026-10-16. Every example is TypeSafe's own endpoint,
`POST https://api.typesafe.ai/v1/systemone`; neither page mentions
OpenRouter.

**Images.** A request gains a top-level `assets` map. Each key is a
reference string the caller chooses; the image is used by writing that
string wherever the image belongs:

```json
{
  "model": "jev-latest",
  "state": {"door": "6f1c…-uuid"},
  "assets": {
    "6f1c…-uuid": {"type": "image", "format": "jpg", "data_base64": "<base64>"}
  },
  "questions": {
    "person_present": {"type": "noul", "instructions": "Is there a person in `door`?"}
  }
}
```

| Rule | Value |
|---|---|
| header | `TypeSafe-Danger-Zone-Images: true` on any request carrying an image or a `box` question |
| asset fields | `type` (optional, `"image"`), `format` (`png` / `jpg` / `jpeg`, lowercase, required with `data_base64`), `data_base64` (standard alphabet, padded, no whitespace, no `data:` prefix), or `part` (a multipart file part; wins over `data_base64`) |
| where a reference may appear | a value in `state`, mid-sentence in `instructions`, an option's value, `description` or `examples` in `criteria`. Object keys are never replaced |
| matching | plain text. An asset referenced nowhere is a `400`. UUIDs are recommended so a reference cannot match by accident |
| reference size | ≤ 1,024 bytes each, ≤ 65,536 bytes for all of them |
| formats | still PNG or JPEG. Animated PNG is rejected; GIF, WebP, HEIC must be converted. Baseline 8-bit JPEG preferred |
| dimensions | each side 280-1,568 px; 78,400-2,458,624 pixels total |
| bytes | 16 MiB per image (decoded); 32 MiB request body; 32 MiB of images after processing; 32 MiB expanded (every use of an image counts) |
| out of limits | rejected, never resized |
| orientation | EXIF is stripped and not applied; pixels must already be upright |

Jev 1.13 (`jev-latest` until 1.14 takes the alias, "in about a week")
accepts images too. The post says multimodal answers are weaker than
text-only ones on every model and recommends breaking a question into
simple steps.

**`box`.** A fourth question type, Jev 1.14 only (`jev-preview` /
`jev-1.14.0`). It takes `instructions` and no `criteria`, needs an image in
the request, and answers

```json
{"type": "box",
 "box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84},
 "confidence": 0.9}
```

Edges are fractions of the image from its top-left corner (`left` /
`right` of the width, `top` / `bottom` of the height), `left < right`,
`top < bottom`. There is always a box: there is no "not found", so absence
is a separate `noul`. One question finds one target. `confidence` is 0-1
and the docs do not call it a probability.

**Stability.** The docs call Images and `box` "Dreamer-ready", not
production-ready: the request format, the answers and the limits may change
between releases. Long context beyond the production default needs a
second header (`TypeSafe-Danger-Zone-Long-Context: true`); out of scope
here.

### 12.2 The contract

`decide()` gains one keyword:

```python
@dataclass(frozen=True)
class DecisionImage:
    mime_type: str        # image/png | image/jpeg
    data: bytes           # decoded
    name: str             # the caller's name for it, used in {{image:NAME}}

def decide(self, state, questions, *, images: Sequence[DecisionImage] = (),
           model=None, cancel_token=None) -> DecisionResult: ...
```

The caller never writes a vendor reference. It writes `{{image:NAME}}`
where an image belongs, in `state` (as a whole value, or inside a string),
in `instructions`, or in a `criteria` value, `description` or `examples`.
`build_decision_request` mints one reference per image
(`jaato-image-<uuid4>`), substitutes it, and builds `assets`.

| Rule | Why |
|---|---|
| jaato mints the references, never the caller | the vendor matches references as plain text, so a name like `door` would also replace the word *door* in every instruction. A token with its own syntax cannot collide, and a UUID cannot either |
| a `{{image:NAME}}` naming no image, an image no token names, and a token in an object key are refused before sending, naming the question | the first two are the vendor's `400`; the third is never replaced and would reach the model as literal text |
| a request with images is refused with `DecisionImagesUnsupportedError` by an adapter that does not declare `ProviderCapabilities.decision_images` | the alternative is the vendor silently answering about a reference string (§12.6 is exactly that risk for OpenRouter) |
| text-only requests are byte-identical to today, header included | `images=()` builds no `assets` and sends no danger-zone header |

The header is added by the adapter, not the contract: it is TypeSafe's, and
`laya-serve` or a future vendor may want another or none.
`needs_image_header(body)` answers whether the request carries an image or
a `box` question.

### 12.3 Checked before sending

Every limit in §12.1 that can be measured from the bytes is checked
locally, so an oversize image is a `DecisionQuestionError` naming the image
and the limit rather than a vendor round trip:

- **Format from the bytes**, not the declared mime: PNG and JPEG signatures. A mime that disagrees with the bytes is refused (#829's rule: never relabel). An animated PNG (an `acTL` chunk before `IDAT`) is refused.
- **Dimensions from the header**: PNG `IHDR`; JPEG the first `SOFn` marker. Stdlib only, no Pillow (an optional extra, `[diagrams]`). An image whose header cannot be read is refused, not sent on trust.
- **Sizes**: 16 MiB per image, and the base64 request estimate against 32 MiB.
- **References**: the minted ones are fixed length, so the 1,024 / 65,536-byte limits are checked by counting images (at most 64 per request, 65,536 / 1,024).

What jaato cannot check is refused nowhere and stated: EXIF orientation
(the caller must send upright pixels), whether the content is what the
question assumes, and the "after processing" image total. jaato never
converts, rotates or resizes an image in v1 (§10 question 5).

Images go base64 in the JSON body. Multipart is a third smaller but needs a
second transport beside `_decisions_http.py` (§10 question 6).

### 12.4 `box`

`QUESTION_TYPES` gains `"box"`. `DecisionQuestion(type="box",
instructions=..., criteria=None)`; any `criteria` is refused, and a `box`
question in a request with no image is refused before sending.

```python
@dataclass(frozen=True)
class DecisionBox:
    top: float
    bottom: float
    left: float
    right: float          # fractions, 0..1, from the top-left corner
```

`DecisionAnswer.value` is a `DecisionBox` for a `box` answer;
`probabilities` is `None`; `confidence` is the vendor's. The parser refuses
an edge outside [0, 1], `left >= right` or `top >= bottom`, naming the
question (`DecisionResponseError`), as it refuses an out-of-range score.

Which models answer `box` is **not** declared in jaato. The catalog does
not say, and a hardcoded per-model table would be stale the day `jev-latest`
moves. A `box` question to Jev 1.13 is the vendor's `422`, which is already
`DecisionRequestRejectedError`, carries the body and is not retried.

### 12.5 The `decide` tool

```
decide(model="triage",
       images=[{"name": "door", "path": "captures/door.jpg"},
               {"name": "shot", "attachment_id": "att_3f2a9c1b7d4e5a60"}],
       state={"photo": "{{image:door}}"},
       questions=[{"id": "person", "type": "noul",
                   "instructions": "Is there a person in `photo`?"},
                  {"id": "where",  "type": "box",
                   "instructions": "the person in `photo`"}])
```

| Rule | Why |
|---|---|
| an image is named by a workspace `path` or an `attachment_id`, never by bytes in the arguments | bytes in a tool call would enter history and be replayed on every request (#850). A path or an id costs a few tokens |
| `path` goes through `check_path_with_jaato_containment`, and `.env` / `.jaato/*_auth.json` are refused | the `readFile` rule; the tool sends the file to a third party |
| `attachment_id` resolves against the session's own history (the ingest id `media_identity` mints, #850) | the image the user sent, or one a tool produced, without a copy on disk. An evicted image (only its marker left) is refused by name |
| the `images` parameter exists only when the session has an entry whose provider declares `decision_images` | `narrow_tool_schema`, per session, as the `model` enum already is |
| a `box` answer also carries pixel coordinates (`box_px`) computed from the dimensions §12.3 read | the model reasons in pixels when it crops |
| the tool description says `box` always returns a box, and to pair it with a `noul` | the one property of `box` that a model will otherwise get wrong |
| the ledger's `decision` record carries each image's digest, size and dimensions, never its bytes | it says what was sent without becoming a copy of it |

Not here: the permission gate. Its state is a rendered tool call (§8.2.1),
which has no image to send.

### 12.6 Which route carries images

| Route | Images and `box` | Why |
|---|---|---|
| `typesafe` (§6.2) | **yes**, from step 8 | the route TypeSafe documents them on; every example in §12.1 is this endpoint |
| `openrouter` | **yes, for TypeSafe Jev**, from step 10; `box` on Jev 1.14 and later | OpenRouter's Multimodal Decisions guide: `image_url` parts in the text, not `assets`. Other image decision models there are refused (their limits are not the ones checked) |
| `laya` | no | text only |

The questions step 10 was waiting on, as the guide answers them (2026-10):

| Question | Answer |
|---|---|
| does `POST /api/alpha/decisions` accept `assets`? | not documented; an image is an `image_url` part with a base64 `data:` URL, read by Jev at any depth of `state`, `instructions` and `criteria`. A top-level `images` is a `400`; remote URLs are not fetched |
| does OpenRouter need TypeSafe's header? | it documents none, and none is sent |
| is `box` reachable? | on `typesafe/jev-1.14` and `~typesafe/jev-latest`; Jev 1.13 answers `400`, and no other decision model answers it |
| limits | Jev's own (§12.3): PNG or JPEG, 256 to 1,568 px a side, 16 MiB a file, 32 MiB a request. jaato keeps TypeSafe's 280 px minimum, the stricter of the two figures |

A profile wanting images and an OpenRouter-billed text path binds two
entries in `decision_models:` (one per provider); nothing picks a route
for it.

### 12.7 Costs and limits

- **Weaker answers.** TypeSafe says multimodal decisions are less accurate than text ones; the calibration caveat in §9 applies more strongly. The `decide` tool's description says to ask simple questions, one target per `box`.
- **Bytes leave the host.** An image is workspace content sent to TypeSafe (and, through OpenRouter, to OpenRouter as well). Same rule as §9; `routing.zdr` applies on OpenRouter.
- **The format may change.** "Dreamer-ready" means the parser must stay strict and the fixtures dated: an answer in a shape it does not know is an error naming the question, never a guess.
- **No preprocessing.** A screenshot over 1,568 px, a WebP, a phone photo with EXIF rotation: each is refused or answered wrong. The refusal names the limit so the caller can fix it; the rotation cannot be detected (§12.3).

