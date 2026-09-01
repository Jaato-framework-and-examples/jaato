# Rewind-with-Hint — Reactive Recovery from Truncated Tool Calls

## Problem

When an LLM tries to emit a tool call whose arguments are very large (typically
a `writeNewFile(content=...)` with thousands of lines inlined), the model can
hit its `max_tokens` cap mid-serialization of the arguments JSON. The function
name survives in the emitted output, but the `arguments` object ends up
truncated — most commonly as an empty `{}`. The tool executor then rejects the
call with `"path is required"` / `"No code provided"` / etc.

Observed in session `20260420_223740`: five consecutive failures across three
turns, alternating between `writeNewFile({})` and `notebook_execute({})`. The
model's narration before each call was coherent and correct ("I am about to
generate a large reference document (~1000+ lines). I'll write it directly to
the file to keep the conversation clean.") — only the serialized arguments were
lost.

Current workaround: bake guidance into the system prompt telling the agent to
chunk large writes. This pays a context-budget cost on every session for a
pathology that fires rarely, and the guidance is generic — it does not name the
specific tool the agent was about to misuse.

## Proposal

Detect the pathology at the session layer and recover reactively:

1. **Detect** — after the model response arrives and is appended to history,
   inspect it for the failure signature (empty/missing required args on a tool
   call, typically combined with `finish_reason == MAX_TOKENS`).
2. **Rewind** — rewrite the last assistant message to keep the narration text
   parts and drop the truncated `tool_use` part. The model's stated intent
   becomes the anchor for the corrective hint.
3. **Inject hint** — append a synthetic user-role message that references the
   preserved narration, names the specific tool, and gives a concrete chunking
   recipe.
4. **Retry** — re-enter the chat loop. The model now sees its own reasoning
   followed by targeted guidance, rather than a disorienting memory wipe.
5. **Bound** — cap the number of rewinds per logical operation (1–2) so a
   persistently-failing model does not loop. After the cap, surface the
   failure normally.

## Why reactive, not pre-emptive

We cannot know in advance how large a payload the model will try to feed a
tool. Pre-loading every session with defensive instructions bloats context
permanently for a pathology that fires rarely. Just-in-time hints only pay
their cost when actually needed and can be targeted to the specific tool and
parameter involved.

A predictive variant (run a cheap classifier on the model's pre-tool narration
to anticipate a large call before generation) is possible but adds latency and
can misfire. The reactive variant is deterministic and bounded — ship it
first, add prediction later if the pattern proves out.

## Why keep the narration

Turning the rewind into a *mid-thought course correction* rather than a memory
wipe produces better follow-through on the retry. The model's stated intent
("I'm about to write a ~1000-line document") becomes the anchor that the hint
attaches to, so the injected message reads as "yes, and here's the right way
to do that" rather than an unexplained reset.

Concretely, the rewind target is the `tool_use` block plus any post-narration
text emitted alongside it, not the full assistant turn.

## Integration points

File paths are under `/home/user/jaato/`.

### Hook location

`jaato-server/shared/jaato_session.py:3962` — right after
`_add_model_response_to_history()` appends the `ProviderResponse` and before
`_execute_tools_and_continue()` iterates tool_use blocks.

```python
# pseudocode — slots between lines 3962 and the tool loop
rewind_reason = self._rewind_detector.inspect(response)
if rewind_reason and self._rewind_budget.allow():
    narration, bad_call = _split_narration_from_tool_use(response)
    self._history.pop_last()                              # drops full assistant msg
    self._history.append(_narration_only_message(narration))
    self._history.append(_oracle_hint(bad_call.name, rewind_reason))
    turn_span.set_attribute("jaato.rewind.reason", rewind_reason)
    turn_span.set_attribute("jaato.rewind.tool", bad_call.name)
    continue  # re-enter chat loop at line 3927
```

### Component map

| Component | Location | Notes |
|-----------|----------|-------|
| **Detector** | new module, e.g. `shared/rewind.py` | Reads `response.finish_reason` (`FinishReason.MAX_TOKENS`) and scans `response.parts[*].function_call.args` for `{}` or missing required keys per the tool's `ToolSchema`. Returns a reason string or `None`. |
| **Rewinder** | uses `SessionHistory.pop_last()` at `shared/session_history.py:85` | Precedent exists — GC already manipulates history (`_remove_tool_results_from_history` at `jaato_session.py:5274`). Need a variant that rewrites the last assistant message to keep text parts and drop tool_use parts. |
| **Hint injector** | `SessionHistory.append(Message.from_text(Role.USER, ...))` | Synthetic user turn referencing the preserved narration and the specific tool. |
| **Budget** | per-session counter on `JaatoSession` | Cap at 1–2 rewinds per logical operation; reset on successful tool_result. |
| **Telemetry** | `turn_span` in the chat loop | Tool span is too narrow — the tool never ran. Set `jaato.rewind.reason` and `jaato.rewind.tool` on the turn span since we are pre-empting execution. |

### Model provider contract

From `jaato-sdk/jaato_sdk/plugins/model_provider/types.py`:

- `ProviderResponse.finish_reason: FinishReason` — enum with `MAX_TOKENS`,
  `TOOL_USE`, `STOP`, etc. The primary detection signal.
- `ProviderResponse.parts: List[Part]` — ordered; `Part.function_call:
  Optional[FunctionCall]` carries the tool call.
- `FunctionCall.args: Dict[str, Any]` — inspect for `{}` or missing required
  keys relative to the tool's declared schema.

#### The precondition providers must not destroy (#745)

Detection needs `MAX_TOKENS` **and** function calls on the *same*
`ProviderResponse`. That combination was unreachable on every streamed turn
until 2026-08-31, because each streaming provider ended its turn with an
unconditional override:

```python
if function_calls and not was_cancelled:
    finish_reason = FinishReason.TOOL_USE
```

The line has a legitimate job — several upstreams report `stop`, or report
nothing, on a turn that did emit tool calls — but running it unconditionally
also rewrote the reason on a turn the provider had correctly read as
truncated. So the detector's precondition could never be satisfied, and this
whole design was dead code for streaming: the truncated turn presented
downstream as a turn that wanted a tool executed, and the session executed or
failed on the fragment and went round again.

The contract now lives in one place —
`jaato_sdk.plugins.model_provider.types.resolve_tool_use_finish`. `TOOL_USE`
is a **fallback, not an override**: it fills in an unreported or merely-`stop`
finish and never displaces a reason in `TERMINAL_FINISH_REASONS`
(`MAX_TOKENS`, `SAFETY`, `ERROR`, `CANCELLED`). Every streaming provider calls
it instead of assigning, and
`shared/tests/test_truncation_is_not_reported_as_tool_use.py` fails if one
goes back to assigning.

#### The finish reason a stream never reported (#687)

The sibling of the above, from the other direction. Every streaming
accumulator starts at `FinishReason.UNKNOWN` and overwrites it only when the
wire delivers a terminal event — Anthropic's `message_stop` /
`message_delta.stop_reason`, an OpenAI-compatible chunk carrying
`choice.finish_reason`, a Google candidate carrying `finish_reason`, the
Claude CLI's `ResultMessage`. When the stream simply stops (a proxy drops the
connection, a gateway times out, an upstream 5xx lands mid-body, TLS resets)
none of those arrive, the iterator ends quietly, and the accumulator still
holds `UNKNOWN` plus whatever text got through.

`UNKNOWN` was grouped with the two *success* outcomes at every consumer, so
that half-finished turn was accepted as a completed one: silent truncation for
the user, a tool call severed mid-serialisation handed downstream as a
request, and — because no exception was ever raised — no retry for the one
failure that most deserves one.

Two names now carry the distinction, and they are opposites:

- **`UNKNOWN`** — the turn ended, and the upstream's word for why was not one
  we map. Still a success.
- **`INCOMPLETE`** — the upstream never said the turn ended at all. Terminal
  (it is in `TERMINAL_FINISH_REASONS`, so `resolve_tool_use_finish` will not
  relabel its fragments `TOOL_USE`), maps to `TurnOutcome.ERROR`, and in no
  consumer's continue-set.

`jaato_sdk.plugins.model_provider.types.require_terminated_stream` is the
single implementation. Every streaming accumulator ends with it; providers
track whether a terminal event was **seen**, separately from what it said, so
that an unmapped label does not read as an interruption and an interruption
does not read as an unmapped label. When the event never came it drops the
accumulated function calls, marks the response `INCOMPLETE`, and raises
`StreamInterruptedError` — which `shared/retry_utils.py` classifies as
transient, so `with_retry` retries it for every provider at once (each
provider's own `classify_error` returns `None` for types it does not know).

`shared/tests/test_a_dead_stream_is_not_a_finished_turn.py` fails if a
provider stops proving its stream terminated, or stops recording one of the
wire signals it reads as terminal.

## Hint message template

The injected user-role message should:

- Reference the model's own narration by quoting or paraphrasing it.
- Name the specific tool that was about to be called.
- State the detection reason ("the completion budget truncated your arguments
  before the `content` field could be written").
- Offer a concrete recipe (write a skeleton first, then append sections;
  break into chunks; use a CLI redirection for very large files).

Example:

> Before that tool call lands: you said you were about to write a ~1000-line
> reference document using `writeNewFile`. The oracle predicts the `content`
> argument will be truncated by the completion budget — the previous attempt
> emitted empty arguments for the same reason. Please write this in sections
> instead: start with a skeleton `writeNewFile(path, content=<outline>)`, then
> append each section with a follow-up call.

## Design questions to resolve before coding

### 1. Cache invalidation cost

The rewind edits the last assistant message. For providers using prompt
caching (e.g. `cache_control` in `shared/plugins/model_provider/anthropic/`),
this invalidates the cache breakpoint at the edited turn — the next
`provider.complete()` call re-reads the full prefix once. Acceptable for a
rare event, but worth confirming that the cache breakpoint placement does not
anchor downstream of the edit in a way that also invalidates older turns.

### 2. Detector scope

v1 should be conservative to avoid false positives:

- **Fire only on** `finish_reason == MAX_TOKENS` AND empty/missing-required
  args.
- **Do not fire on** normal tool calls that happen to have small arg objects
  — e.g. `shell_list()` legitimately takes `{}`.

Broader heuristics (truncated but non-empty JSON strings, repeated
`(tool, empty_args)` cycles across turns) can be added behind flags once v1
proves clean.

### 3. What counts as a "logical operation" for budget reset

Simplest: reset the rewind counter on any successful `tool_result` that is
not itself a rewind retry. This prevents unrelated follow-on failures later
in the session from being starved of rewinds.

## Telemetry

New attributes on the `jaato.turn` span:

- `jaato.rewind.reason: str` — `"max_tokens_empty_args"`,
  `"max_tokens_missing_required"`, etc.
- `jaato.rewind.tool: str` — name of the tool that was about to be called.
- `jaato.rewind.count: int` — cumulative rewinds in this session.
- `jaato.rewind.succeeded: bool` — set on the *next* turn's span if the
  retry produced a non-empty tool call for the same tool.

These let us measure how often the pathology fires and whether the hint
actually changes behavior on retry — otherwise we are flying blind on whether
the mechanism earns its keep.

## Out of scope

- Predictive/pre-emptive detection via narration classification.
- Provider-side prevention (e.g. raising `max_tokens` dynamically when a
  large tool call is anticipated).
- Multi-turn rewinds (rewinding more than the most recent assistant message).
- Structural changes to `SessionHistory` beyond what `pop_last()` already
  supports.

## Related work

- `docs/jaato_gc_system.md` — GC plugins already manipulate history; the
  rewrite-last-assistant-message pattern is novel but adjacent.
- `docs/opentelemetry-design.md` — turn/tool span hierarchy that the new
  attributes plug into.
- `docs/reliability-plugin-design.md` — retry/backoff machinery that could
  eventually wrap the rewind budget.
