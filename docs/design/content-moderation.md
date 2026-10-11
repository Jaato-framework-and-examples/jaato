# Content Moderation: Classifier Checkpoints on Input, Tool Results and Output

**Status:** proposed. Nothing here is built.
**Related:** [Decision Models](decision-models.md) (one backend), [EU AI Act](eu-ai-act.md) (the audit and incident contracts this reuses).

## 1. What this is

A session can be given a **moderation policy**: one or more classifiers that
judge content at three checkpoints and decide what happens to it.

| Checkpoint | Content judged | Typical categories |
|---|---|---|
| `input` | what a person sends: the prompt, a `session.wake` payload, an injected message, a clarification answer | self-harm, abuse, off-topic, PII |
| `tool_results` | what a tool returns before the model reads it: web pages, MCP results, files, subagent reports | prompt injection, harmful content relayed from the web |
| `output` | what the model writes before (or while) a person reads it | harassment, regulated advice, leaked secrets |

The checkpoints need different policies, so each is configured separately.
A banking assistant and a coding agent need different categories, so the
policy is per profile.

The framework provides the checkpoints, the policy, the actions and the
evidence. Which classifier judges is configuration: a decision model
(TypeSafe Jev, ConvAI Laya), Llama Guard served by any chat provider,
Azure AI Content Safety, or a deployment's own model.

## 2. What exists today, and what does not

Measured against the tree on 2026-10-11:

| Need | Today | Gap |
|---|---|---|
| A hook on user input | prompt enrichment (`JaatoSession._enrich_and_clean_prompt` → `registry.enrich_prompt`). `PromptEnrichmentResult` is `{prompt, metadata}` | **it can rewrite, never refuse** |
| A hook on tool results | tool-result enrichment, then untrusted-content fencing (`TRAIT_UNTRUSTED_CONTENT`) | **it can rewrite, never refuse**, and a streaming tool's `ToolOutputEvent` chunks reach clients before it runs |
| A hook on model output | none. Text reaches clients from four `streaming_callback` closures in `jaato_session.py` and the batched path. Completion processors gate only `signal_completion`; the formatter pipeline only renders | **no hook at all** |
| Stopping a turn mid-stream | `CancelToken`, `request_stop()` | exists |
| Telling a client a turn ended blocked | nothing. `TurnCompletedEvent` has no refusal | **missing event** |
| Removing text a client already drew | nothing | **missing event** |
| Calling a classifier | `decide()` on `openrouter` (decision models, step 2) | only one backend, and the wrong shape for Llama Guard / Azure |
| Recording a verdict | the token ledger, chained under `record_keeping.integrity` | no record type |
| Raising an incident | `jaato_server.shared.incidents.raise_incident` | exists |
| Validation for high-risk profiles | `HIGH_RISK_ESCALATED_CODES` in `scaffold/validate.py` | no moderation codes |

So the classifier has nowhere to plug in today. Most of this design is the
three hooks and the client contract; the classifier call is the small part.

## 3. How a hosted model service does it (and why we copy the shape)

Anthropic's published approach (Constitutional Classifiers, 2025; the
Messages API's `stop_reason: "refusal"`) is the reference shape:

- responses **stream**; nothing is held back until the end;
- an output classifier scores the stream **as it grows** and can stop
  generation part-way;
- a stopped response ends with a distinct stop reason, so a client can tell
  a blocked answer from a finished one; text already streamed has been
  delivered and the client decides what to do with it;
- input is classified too.

The cost is that a few tokens may be seen before the cut. The alternative,
buffering the whole response, costs a full response of latency on every
turn. Both are legitimate, so both are offered (§5), and the profile
chooses.

## 4. Classifier backends

### 4.1 The interface

A backend takes content and the categories a policy asks about, and returns
a verdict per category. It is not `decide()`: Llama Guard writes
`unsafe\nS3` and Azure returns per-category severities, and neither is a
decision model. `decide()` is one adapter behind it.

```python
@dataclass(frozen=True)
class ModerationRequest:
    checkpoint: str                    # input | tool_results | output
    content: str
    categories: Tuple[str, ...]        # the backend's own names
    context: Mapping[str, str]         # tool name, agent id; never the model's say-so

@dataclass(frozen=True)
class CategoryVerdict:
    category: str
    score: float                       # 0..1, higher = more likely in the category
    detail: Optional[str] = None       # the backend's label, e.g. "S3"

@dataclass(frozen=True)
class ModerationVerdict:
    verdicts: Mapping[str, CategoryVerdict]
    backend: str                       # adapter name
    model: str                         # the version the backend reported
    usage: TokenUsage                  # reported=False when the backend says nothing
    latency_ms: float

class ModerationBackend(Protocol):
    name: str
    def categories(self) -> FrozenSet[str]: ...        # what it can judge
    def accepts(self, mime_type: str) -> bool: ...     # text/plain at least
    def classify(self, request: ModerationRequest,
                 *, cancel_token: Optional[CancelToken] = None) -> ModerationVerdict: ...
```

Rules every adapter holds to, each because the alternative fails open:

- **A category the backend does not report is an error, never a zero.** A
  verdict that silently became "not harmful" is the worst failure this
  feature can have (the rule `parse_decision_response` already follows).
- **`categories()` is declared, not discovered**, so `validate` can refuse a
  policy naming a category the backend cannot judge.
- **The backend never sees the model's own framing of the content.** The
  framework builds the request with `render_moderation_state`: the content
  verbatim, wrapped as untrusted, under a `Checkpoint:` line and, for tool
  results, a `Tool:` line, with the same secret redaction, value cuts and
  window fitting as the permission gate's renderer
  ([Decision Models](decision-models.md) §8.2.1). On output it cannot
  fit, the check fails as `on_unavailable` says, never by passing.

### 4.2 Adapters

| Adapter | Calls | Category vocabulary | Score |
|---|---|---|---|
| `decisions` | `decide()` on a `decision_models` entry: one `noul` question per category, with the category's instruction text from the policy | the policy's own (any name, each with an instruction) | the `noul` probability |
| `llama_guard` | `complete()` on any chat provider with the Llama Guard prompt template | `S1`..`S14` (MLCommons hazard taxonomy), aliased to readable names | 1.0 for a listed category, else 0.0; the first-token logprob where the provider returns logprobs |
| `azure_content_safety` | `POST {endpoint}/contentsafety/text:analyze` | `Hate`, `SelfHarm`, `Sexual`, `Violence` (+ Prompt Shields for injection) | severity 0..7 divided by 7 |
| `local` | a model loaded in-process (out of tree) | declared by the plugin | declared |

Adapters register as plugins (`PLUGIN_KIND = "moderation"`), so a deployment's
own classifier is an out-of-tree distribution behind the same entry-point
trust gate as any other plugin. The in-tree default is `decisions`, because
it needs no model served locally and its categories are whatever the policy
writes.

### 4.3 Where the call runs

| Checkpoint | Process | Why |
|---|---|---|
| `input` | **daemon**, in `SessionManager` before the request reaches the runner | covers IPC, WS and every verb (`send_message`, `session.wake`, `inject_prompt`, `session.message`) at one point; a blocked message never reaches a runner |
| `tool_results` | runner | the results only exist there |
| `output` | runner | the stream only exists there |

A remote backend costs the same in either process. A `local` backend in the
runner loads a model into a pool slot, which then must be retired after use
(#1565). The adapter declares it through `slot_retire_reason()` like
`references` does, so a slot is never returned to the pool holding a model.

## 5. The output checkpoint and streaming

### 5.1 Two delivery modes

| `delivery` | Client sees | Latency added | Can a blocked word be seen? |
|---|---|---|---|
| `stream` (default) | text as it is generated | ≈0 per chunk; one check per window | yes, up to one window |
| `buffer` | nothing until the check passes, then the whole response | one classifier call after the full response | no |

In `stream` mode the hook accumulates text and classifies at each window
boundary (`check_every: sentence` or a token count) and once on the final
text. A block:

1. trips the session's cancel token (the provider aborts the request);
2. emits `OutputWithheldEvent` (§7) naming the turn's output to withdraw;
3. ends the turn with `TurnCompletedEvent.stop_reason = "moderated"` (a new field, §7);
4. records what the model wrote, not what the client kept (§8).

In `buffer` mode the streaming callbacks still run but send nothing to
clients; the hook releases the buffer as one `AgentOutputEvent` after the
final check, or emits `OutputWithheldEvent` and no text.

### 5.2 One door for model text

There are four `streaming_callback` closures in `jaato_session.py`
(`_run_chat_loop`, `_do_send_tool_results`, `_check_and_handle_mid_turn_prompt`
and `_complete_parts_turn`) plus the batched path, which emits assembled
text. The output hook must see every
one of them, so model text gets **one emit function** (`_emit_model_text`)
and an AST guard fails the build on an `on_output("model", ...)` call made
outside it. The same rule the history invariant (#674) and the agent-status
door use: a seam covering four of five paths is a seam that leaks.

### 5.3 What the model sees after a block

The withheld text is **kept in history**, replaced for the model by a
`⟦JAATO⟧` note naming the category: "Your previous response was withheld
by the content policy (category: harassment)." Deleting it would leave a
turn the model has no record of; keeping it verbatim would let the next
turn continue from it. A tool call in the withheld response is answered
with a refusal so pairing holds (#674).

### 5.4 Window size and backend speed

A remote classifier at every sentence is a round trip per sentence. The
policy can split the work: a fast classifier on the windows
(`classifier: fast`, a local Llama Guard) and a slower, more accurate one
on the whole turn (`final_classifier: jev`). Window checks run on a worker
and never block the stream; if a window's verdict arrives after the next
window has been sent, the block still withdraws everything since the
turn began.

## 6. The profile block

### 6.1 Shape

```yaml
moderation:
  classifiers:                    # named backends, like decision_models:
    fast:  {backend: llama_guard, provider: vllm, model: meta-llama/Llama-Guard-4-12B}
    jev:   {backend: decisions, model: triage}        # a decision_models entry
    azure: {backend: azure_content_safety, endpoint: https://x.cognitiveservices.azure.com,
            key_env: AZURE_CS_KEY}

  input:
    classifier: jev
    categories:
      self_harm: {threshold: 0.5, action: block,
                  instruction: "Does the message express intent to self-harm?"}
      pii:       {threshold: 0.7, action: redact}
      off_topic: {threshold: 0.8, action: flag,
                  instruction: "Is this unrelated to retail banking?"}

  tool_results:
    classifier: azure
    categories:
      prompt_injection: {threshold: 0.6, action: block}

  output:
    classifier: fast
    final_classifier: jev         # optional second pass on the whole turn
    delivery: stream              # stream | buffer
    check_every: sentence         # sentence | <N> (tokens); stream only
    categories:
      harassment:     {threshold: 0.5, action: block}
      medical_advice: {threshold: 0.7, action: ask,
                       instruction: "Does this give a specific medical diagnosis or dosage?"}

  on_unavailable: block           # block (default) | allow
  on_unanswerable_ask: block      # block (default) | allow | flag
  timeout_seconds: 5
```

`instruction` is required for a `decisions` classifier (it is the `noul`
question) and ignored by backends with a fixed taxonomy.

### 6.2 Actions

| Action | `input` | `tool_results` | `output` |
|---|---|---|---|
| `block` | refused with `ModerationRefusedEvent`; the turn does not start | the result is replaced by a refusal the model reads | §5.1 |
| `ask` | a permission-style prompt to an attached person; their answer decides | same | `buffer`: same; `stream`: treated as `block` (the text is already out) |
| `redact` | spans the backend located are replaced before the model sees them | same | `buffer` only; under `stream` it is `flag` |
| `flag` | recorded and passed through | same | same |

`redact` needs spans. Backends that report none (`decisions`, Llama Guard)
cannot redact; `validate` reports a `redact` action on such a backend as an
error. PII redaction is better done by pseudonymization (Presidio, premium),
which this design does not replace (§9.3).

### 6.3 Inheritance

Most-restrictive-wins, the rule `runtime_limits.max_parallel_tools` and
`budget_control.limits` use, per field:

| Field | Rule |
|---|---|
| `categories.<c>.threshold` | minimum across layers |
| `categories.<c>.action` | strongest: `block > ask > redact > flag` |
| a category | union: a category any layer declares stays declared |
| `delivery` | `buffer` beats `stream` |
| `check_every` | the smaller window |
| `on_unavailable`, `on_unanswerable_ask` | `block` beats `allow` |
| `classifiers`, `classifier` | child replaces, by name |

A child can tighten a parent's policy and never loosen it.

### 6.4 Failure

- **`on_unavailable`**: the backend timed out, answered `5xx`, or returned a
  verdict missing a category. `block` (default) treats the content as
  flagged by every category whose action is `block`. `allow` passes it, is
  announced at WARNING once per session, and is an error under
  `risk_class: high`, the treatment `scrub_secret_env: none` gets.
- **`on_unanswerable_ask`**: `ask` with nobody attached (a headless `api`
  client, a cascade stage, a woken session with no client). `block` is the
  default because an unanswered prompt would hang the turn.
- Retries go through `with_retry` within `timeout_seconds` in total; a
  check never waits longer than that.

### 6.5 Why a top-level block

Not `plugin_configs.moderation`, for the reason `budget_control` is
top-level: it decides what a session may do rather than configuring one
plugin, it needs the inheritance rules above (which `plugin_configs` does
not apply), and it is what an auditor reads. It travels on
`SessionInitEnvelope` (a version bump) so pool slots and cold spawns get
the same policy, and the daemon-side input checkpoint reads the same
resolved block.

## 7. The client contract (protocol bump)

| Event | When | Fields |
|---|---|---|
| `ModerationRefusedEvent` | input blocked, before any turn | `request_id`, `checkpoint`, `categories`, `message` (a sentence for the person, never the scores) |
| `OutputWithheldEvent` | output blocked | `agent_id`, `turn_index`, `categories`, `replacement` (text to draw instead, may be empty) |
| `TurnCompletedEvent.stop_reason` | every turn | a new optional field: `"moderated"` when a block ended the turn, absent otherwise (today the event carries no stop reason) |

A client that draws streamed text must, on `OutputWithheldEvent`, remove or
collapse everything it drew for that turn and draw `replacement`. The TUI
and web client get this in the same change; the TS and Python SDKs expose
both events, and `Session.ask` raises `ContentWithheld` (a `SessionError`
subclass carrying the categories) the way #1007's `SessionEnded` does.

**An older client does not know the event and leaves the text on screen.**
That is not acceptable for a block, so a session whose policy has
`output.delivery: stream` with any `block` action refuses an attaching
client below the new protocol version, by name. `buffer` mode needs no
refusal: nothing is sent before the check.

## 8. Evidence

Each verdict writes a `moderation` ledger record, chained like every other
record under `record_keeping.integrity`:

```json
{"type": "moderation", "checkpoint": "output", "agent_id": "main",
 "turn_index": 4, "backend": "decisions", "model": "typesafe/jev-1.13-20260917",
 "verdicts": {"harassment": 0.91, "medical_advice": 0.02},
 "action": "block", "category": "harassment", "content_digest": "sha256:...",
 "latency_ms": 212, "user_id": "app:alice"}
```

- The content is recorded as a **digest**, never the text: the ledger is an
  audit store, and a record of blocked harmful text would be a copy of it.
- A `block` raises an incident (`raise_incident`, kind `content_blocked`),
  so `jaato-doctor --incidents` lists it with the Article 73 windows.
- A verdict that failed (`on_unavailable`) is recorded as `action:
  unavailable` with the error class: a check that did not run is evidence
  too.
- The classifier's cost goes into `budget_control.limits.usd` / `tokens`,
  never `turns`, as decision calls do.
- `jaato-scaffold explain moderation [<profile>]` renders the resolved
  policy, the backends' declared categories and what each action does,
  read from the code that enforces it.

## 9. Interactions

### 9.1 Untrusted-content fencing

Fencing tells the model that tool output is data; the `tool_results`
checkpoint decides whether the model sees it at all. They compose: a result
that passes moderation is still fenced.

### 9.2 The permission gate

The decision-model permission gate (decision-models §8.2) judges an
**action** the model wants to take; moderation judges **content**. Separate
configuration, shared backends: both can name the same `decision_models`
entry.

### 9.3 Pseudonymization

When the input classifier is remote, pseudonymization runs **first**, so
the moderation call does not send the PII to a third party. Order:
pseudonymize → moderate → model. `validate` warns when a profile enables a
remote input classifier and no pseudonymization, under `risk_class: high`.

### 9.4 Media

Every backend above is text-only. An audio or image part is either judged
by a backend whose `accepts()` takes its mime, or judged through a
transcript where one exists (outbound audio has one, #869; inbound audio
does not). Otherwise it **passes unjudged, and that is recorded**
(`action: unjudged`, with the mime), never silently. A profile may set
`unjudged_media: block` to refuse media its classifiers cannot read.

### 9.5 Subagents

A subagent inherits its parent's policy unless its own profile declares
one, and then the merge of §6.3 applies, so a child cannot run with a
looser policy than the parent that spawned it.

## 10. Validation

| Code | Severity | When |
|---|---|---|
| `moderation_category_unknown` | error | a category the backend does not declare |
| `moderation_classifier_missing` | error | a checkpoint names a classifier not in `classifiers` |
| `moderation_redact_unsupported` | error | `redact` on a backend that reports no spans |
| `moderation_instruction_missing` | error | a `decisions` category with no `instruction` |
| `moderation_fails_open` | warn; error under high | `on_unavailable: allow` |
| `moderation_stream_unchecked_final` | warn; error under high | `delivery: stream` with no `final_classifier` |
| `moderation_absent` | added to `HIGH_RISK_ESCALATED_CODES` | a `risk_class: high` profile with no `moderation` block |

## 11. Costs and limits

- **Stream mode leaks up to one window.** Stated, and the reason `buffer`
  exists.
- **Classifiers are wrong sometimes.** Thresholds are the deployment's
  call; jaato forwards scores and does not calibrate them. `flag` is the
  way to measure a policy before enforcing it.
- **Latency.** Input and tool-result checks are on the critical path of
  every turn and every tool call. A remote classifier adds a round trip to
  each; a profile that moderates tool results should prefer a local
  backend.
- **The framework cannot judge what it never sees.** Content a tool writes
  to disk and a person opens outside jaato is out of scope.

## 12. Rollout

1. **The output door.** `_emit_model_text` and its AST guard, with no
   behaviour change. Safe on its own and required by everything after.
2. **The interface and the `decisions` adapter**, the profile block, the
   envelope field, `validate` and `explain moderation`. Policies parse and
   validate; nothing enforces yet.
3. **The input checkpoint** (daemon side), `ModerationRefusedEvent`, the
   ledger record and the incident. The protocol bump lands here.
4. **The output checkpoint**, both delivery modes, `OutputWithheldEvent`,
   `stop_reason: "moderated"`, the old-client refusal, TUI and web client.
5. **The tool-results checkpoint**, including streaming tool output.
6. **`llama_guard` and `azure_content_safety` adapters.**

Steps 3 to 5 each ship with a guard that drives a real session against a
stand-in classifier, and with reversions for the fail-closed rules.

## 13. Open questions

1. Should `ask` on output under `stream` hold the stream while the person
   decides (turning it into `buffer` for that turn), instead of becoming
   `block`?
2. Does the input checkpoint judge the raw message or the prompt after
   enrichment (references expanded, memories injected)? Proposed: the raw
   message, since enrichment content is ours.
3. Should a client be able to *raise* the policy (opt into `buffer` for its
   own screen) through `PresentationContext`, never lower it?
4. Is a per-category `message` (what the person is told) needed, or is one
   policy-wide message enough?
