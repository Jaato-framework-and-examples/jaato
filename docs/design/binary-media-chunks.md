# Binary Media Chunks — modality direction, and getting bytes to a client

**Status:** design note. §5.5 (the direction-qualified tier key) is
**implemented** — `shared/model_tiers.py` parses `{kind: direction}` into
`TierEntry.inbound_modalities` / `.outbound_modalities`, the content gate and
startup check ask directional questions, and `jaato-scaffold validate` warns
that outbound roles are inert. Everything else here is still a proposal. Written to give two parallel
workstreams — model-tier modality roles, and OpenAI audio streaming output —
one shared chunk primitive instead of two.

**Scope.** How binary content (audio, images, PDFs, video) moves through the
framework, in *both* directions, and which of the existing mechanisms each
half should reuse. It does **not** specify the OpenAI wire format; that
belongs to the audio-streaming workstream.

---

## 1. Three directions, not two

The framework's modality vocabulary is currently written as if there were one
direction. There are three distinct paths, with very different maturity:

| # | Path | Meaning | State |
|---|------|---------|-------|
| 1 | **inbound** | content → the model (a screenshot the model looks at) | **works** |
| 2 | **outbound** | the model → content (the model emits speech) | **absent end to end** |
| 3 | **tool → client** | a tool produces bytes a *person* consumes; the model may never see them | **data exists, delivery missing** |

Path 3 is the one most easily missed, and it is the cheapest. It needs none of
path 2's machinery: no provider changes, no response parsing, no adapter work.
A tool already returns `ToolResult.attachments=[Attachment(mime_type, data)]`.

Worth stating plainly: **today path 3's bytes are destroyed.** The modality
content gate (`jaato_session._gate_one_tool_result`, `:7845`) strips attachments the
active model can't consume and keeps only `kept`; the withheld ones are
dropped. The gate is a *filter* where it should be a *router* — content the
model cannot consume is exactly the content a client might want.

---

## 2. What already exists (verified inventory)

Read this before proposing anything new; more is built than it first appears.

### 2.1 Tool result streaming — `shared/plugins/streaming/protocol.py`

A complete chunk protocol, already solving ordering and identity:

| Piece | Carries |
|-------|---------|
| `StreamChunk` | `content: str`, `chunk_type`, **`sequence`**, `timestamp`, `metadata` |
| `StreamHandle` | `stream_id`, `plugin_name`, `tool_name`, `initial_chunks`, `status` |
| `StreamState` | accumulated `chunks`, a **`chunks_delivered` cursor**, `status`, `final_result` |
| `StreamStatus` | `starting / streaming / paused / completed / failed / dismissed` |
| `StreamingCapable` | `supports_streaming()`, `execute_streaming(..., on_chunk)` |

Sequence numbers, stream ids, per-chunk type hints and a lifecycle all exist.
An audio stream is structurally the same object.

> `StreamStatus.PAUSED` is **declared but never set** — it appears only in
> `StreamState.is_active()`. It is a hook to build flow control on, not
> working flow control.

### 2.2 Three subscription surfaces, all already built

Subscription is a solved problem here, at every level. Nothing in this design
needs a new one — this is the single most important thing to know before
proposing machinery.

**1. The SDK client subscribes** — `jaato_sdk/client/ipc.py:572-604`, backed
by `client/_handler_registry.py` and mirrored on `IPCRecoveryClient`
(`client/recovery.py:1087`):

```python
client.subscribe(EventType.TOOL_OUTPUT, on_chunk)   # typed
client.subscribe_once(EventType.SESSION_TERMINATED, on_done)
client.subscribe_all(firehose)                      # catchall
client.subscribe_many({...})                        # atomic multi-register
```

Each returns an idempotent `Unsubscribe`. Sync handlers run inline, async
handlers are scheduled fire-and-forget. Dispatch snapshots each bucket first,
so subscribe/unsubscribe during dispatch takes effect on the next event. **Any
client — a TUI, an SDK orchestrator, a cascade driver — subscribes this way.**

**2. The model subscribes** — `shared/event_bus_tools.py` registers
`subscribeToEvents` / `getEvents` / `listSubscriptions` / `unsubscribe` as core
tools, delivering matched events into the conversation via `inject_prompt()`.
So an *agent* can wait on events too, which is how a cascade parent watches a
child.

**3. Plugins and reactor rules subscribe** — `shared/event_bus.py`, the
per-runtime in-process `EventBus`: `subscribe(subscriber_name, filter,
callback)` / `publish()`.

`server.emit()` is the single fan-out point feeding all three. `_SERVER_TO_BUS`
(`server/core.py:129`) decides which protocol events are *also* republished on
the in-process bus; events absent from that map still reach clients, they are
just not visible to surface 3.

**Consequence for this design.** A client that wants audio already has its
subscription: `client.subscribe(EventType.TOOL_OUTPUT, handler)`. The only
thing missing is that the event carries no bytes (§3.1). Widen the payload and
the whole subscription path — client, agent and plugin — lights up at once,
with no new API on any of the three surfaces.

### 2.3 The per-chunk client event

`ToolOutputEvent` (`events.py:690`) already exists and is already emitted
(`server/core.py:3640`), already bus-mapped (`EventType.TOOL_OUTPUT →
BusEventType.TOOL_OUTPUT`), and already correlates by `call_id`:

```python
class ToolOutputEvent(Event):
    """Live output chunk from a running tool (tail -f style)."""
    agent_id: str = ""
    call_id: str = ""
    chunk: str = ""      # text only
```

### 2.4 Client capability declaration

`PresentationContext` (`jaato_sdk/events.py:2243`) already flows
client → `ClientConfigRequest.presentation` → `SessionManager` → `JaatoSession`,
and already declares renderability: `supports_images`, `supports_tables`,
`supports_mermaid`, `supports_expandable_content`, `client_type`.

### 2.5 Inbound binary

`Part.inline_data {mime_type, data}` is marshalled to the wire by anthropic,
google_genai, antigravity, openrouter and nebius. Capability columns
`user_message_images`, `tool_result_images`, `pdf_input`, `audio_input`
(`model_provider/base.py`).

`audio_input` is the newest and the one this document's own subject created
the need for. Outbound audio landed first (§4 below); the inbound direction
had no wire format at all — `input_audio`, OpenAI's content-block form for
audio input, appeared nowhere in the tree, so a model whose catalog entry
declared `audio` as an input modality was handed nothing and the framework
could speak but not be spoken to (#830). The dispatch lives with its siblings
in `model_provider/_attachments.py`, opted into per wire
(`audio_as_input_audio=True` for `openrouter`), and refuses any container
outside the wire's closed `format` vocabulary rather than renaming it to one
inside — the #829 lesson applied before it could be relearned. Google's
`inline_data` path always carried audio; what blocked Gemini was its own
`MODEL_INPUT_MODALITIES` table omitting `audio`, so the gate refused content
the wire underneath would have delivered.

---

## 3. What is actually missing

Precisely four things. Note how small this list is relative to §2.

1. **Every chunk payload is `str`.**
   - `StreamChunk.content: str`
   - `ToolOutputEvent.chunk: str`
   - `StreamingCallback = Callable[[str], None]` (`model_provider/base.py:54`) —
     the model-streaming callback, blocking path 2 entirely.

2. **The tool-stream chunk collapses to text at the client boundary.**
   `jaato_session._execute_streaming_tool` (defined at `:6951`; the callback
   at `:6982`):

   ```python
   def on_chunk(chunk: StreamChunk) -> None:
       if on_output:
           on_output("streaming", f"<hidden>[{base_name}] {chunk.content}</hidden>", "append")
   ```

   `sequence`, `chunk_type` and `metadata` are discarded here. The structure
   of §2.1 never reaches a client.

3. **Audience is hardcoded.** That `<hidden>` wrapper encodes one fixed
   policy: *for the model, hidden from the user*. Media inverts it — *for the
   user, and possibly withheld from the model*. Audience must become data.

4. **No backpressure.** `_event_queues[client_id] = asyncio.Queue()`
   (`server/ipc.py:484`) is unbounded, so the `QueueFull` branch at `:922` is
   unreachable. A slow consumer grows the queue without bound: for text a
   cosmetic lag, for audio unbounded memory and monotonically increasing drift.

**Gotcha.** At the `ToolOutputEvent` emit site (`core.py:3630`) the chunk is
run through `agent_pipeline.process_chunk()` — a *text formatter* pipeline.
Binary must bypass it. A formatter that reflows text will corrupt bytes.

---

## 4. Is the bus fast enough for audio?

Yes; the transport is not the constraint. PCM16 24 kHz mono is ~48 KB/s. At
100 ms chunks that is **10 events/sec of ~4.8 KB (~6.4 KB base64)**. The bus
already sustains higher event rates streaming text tokens via
`AgentOutputEvent`.

Costs to accept, none fatal:

- **base64 in a UTF-8 JSON frame**: +33% and an encode/decode per chunk.
- **10 MB frame cap**: irrelevant for chunks; a whole-blob event must chunk
  above ~7.5 MB pre-encoding.
- **Jitter**: belongs in the client (a playback buffer), consistent with the
  framework's pipeline-emits-data / client-decides-presentation split.

The real constraints are §3.4 (backpressure) and §3.1 (payload type).

---

## 5. Design

### 5.1 One chunk primitive, made binary-capable

Widen `StreamChunk` rather than introduce a parallel media type. Keep
`content: str` as-is so every existing producer is untouched; add an optional
binary sibling:

```python
@dataclass
class StreamChunk:
    content: str = ""                              # unchanged
    inline_data: Optional[Dict[str, Any]] = None   # {"mime_type": str, "data": bytes}
    chunk_type: str = "result"
    sequence: int = 0
    timestamp: datetime = ...
    metadata: Optional[Dict[str, Any]] = None
    audience: Audience = Audience.MODEL            # see 5.2
```

`inline_data` deliberately mirrors `Part.inline_data` so the same
`{mime_type, data}` shape is used on every path — inbound parts, tool result
attachments, and chunks.

### 5.2 Audience becomes data, not a hardcoded wrapper

```python
class Audience(str, Enum):
    MODEL  = "model"    # today's <hidden> behaviour — into the conversation
    CLIENT = "client"   # to subscribed clients only; never enters history
    BOTH   = "both"
```

Default `MODEL` preserves current behaviour exactly. Audio from a TTS tool is
`CLIENT`. A screenshot for a vision tier is `BOTH`.

This is also where the content gate stops destroying bytes: when a modality is
withheld from the model, re-route that attachment as a `CLIENT` chunk instead
of dropping it, and keep the existing self-correcting note for the model.

**Cascades.** `Audience` is about *this* session's model, not about who may
observe. A parent agent watching a child subscribes through surface 2
(`subscribeToEvents` → `inject_prompt`), and a cascade driver through surface 1
— both see a `CLIENT` chunk, because audience selects whether the bytes enter
*this* conversation's history, not whether the event is published. That keeps
one rule doing both jobs: a parent that wants to look at a child's screenshot
must itself be in a tier declaring `image: inbound`, and if it is not, the same
gate that protected the child protects the parent, with the same actionable
note. No separate cascade path is needed.

### 5.3 The client-facing event

Widen `ToolOutputEvent` rather than add a rival event — it already carries
`call_id` correlation, is already bus-mapped, and clients already handle it:

```python
class ToolOutputEvent(Event):
    agent_id: str = ""
    call_id: str = ""
    chunk: str = ""                                # unchanged, text
    stream_id: str = ""                            # NEW — correlate a media stream
    sequence: Optional[int] = None                 # NEW — ordering, from StreamChunk
    mime_type: Optional[str] = None                # NEW — tags the payload
    data_b64: Optional[str] = None                 # NEW — binary payload
    final: bool = False                            # NEW — last chunk of this stream
```

Rules:

- `mime_type`/`data_b64` set ⇒ **bypass the formatter pipeline** (§3 gotcha).
- `sequence` is the `StreamChunk.sequence` already assigned — pass it through
  rather than inventing a second counter.
- A whole-blob delivery (a tool returning a finished WAV) is just a
  single-chunk stream with `sequence=0, final=True`.

### 5.4 Two capability axes, kept apart

The easiest mistake here is conflating what the *model* can do with what the
*client* can do. They have different owners and different lifetimes.

| Axis | Declares | Where | Consumer |
|------|----------|-------|----------|
| **Model** | what a tier can accept / emit | `model_tiers.<tier>.modalities` | content gate, startup capability check |
| **Client** | what the viewer can render / play | `PresentationContext` | media-chunk routing |

`PresentationContext` gains a renderable-media declaration alongside its
existing `supports_images` / `supports_mermaid` — the natural spelling is a
set of playable/renderable mime types or modality tokens, so a TUI declares
none, a web client declares image+audio, a voice client declares audio.

Routing then reads: *model can't consume it* → withhold from the model (as
today) → *some client declares it* → emit as a `CLIENT`-audience chunk.

### 5.5 Direction vocabulary

The tier key takes a direction per modality, with the list form as sugar:

```yaml
model_tiers:
  speaker:
    model: gpt-4o-audio-preview
    modalities:
      audio: bidirectional     # inbound | outbound | bidirectional
  looker:
    model: google/gemini-3-pro
    modalities: [image]        # ≡ {image: inbound}
```

- **`bidirectional`**, not `both` — `both` says nothing about *what* it is
  both of, and does not parallel `inbound`/`outbound` grammatically. Not
  `duplex`: that connotes *simultaneity*, and a tier declares capability, not
  concurrency. Avoid `on`/`off`/`yes`/`no` anywhere in this enum (YAML 1.1
  parses them as booleans).
- Stored as two sets on `TierEntry` (`inbound_modalities` /
  `outbound_modalities`), not a `{kind: direction}` map: consumers ask
  directional questions, and a map would make every one of them filter.
- Backward compatible: today's `modalities: [image]` and the implicit
  `vision` ⇒ `{image: inbound}` are unchanged.

**What `outbound` does before delivery exists.** Not "reject" (profiles could
not be written ahead, and enabling it later becomes a behaviour change) and
not "accept and ignore" (a silent no-op). Instead: parse it fully; verify it
against `supports_output_modality()` *when the provider implements that*, skip
otherwise (no false failures); and have `jaato-scaffold validate` emit a
**warning** — declared, but no adapter delivers model-generated media yet.
When delivery lands, delete the warning; every profile already parses.

### 5.6 Naming collision to avoid

OpenAI's Chat Completions request field is **also** called `modalities`, and
it means **output** — `modalities: ["text","audio"]` with `audio: {voice,
format}`. Both keys end up in profiles:

```yaml
plugin_configs:
  <provider>:
    api_params:
      modalities: [text, audio]   # OUTPUT — what the model should EMIT
model_tiers:
  speaker:
    modalities: {audio: outbound} # tier ROLE, direction-qualified
```

The map-with-directions shape in §5.5 is what keeps these visually distinct.

Note also that `_openai_compat/base.py:108` **allowlists** api_params —
`modalities` and `audio` are currently dropped with a warning and must be
added to `_FORWARDED_API_PARAMS`. That set is shared by nebius, ovhcloud,
doubleword, nim, lmstudio, vllm, tensorrt_llm, triton and zhipuai_openai.

Finally: whatever queries *output* capability must **not** be called
`modalities()`. That name is framework-wide for input
(`base.py:299`; `supports_modality` = "accepts `kind` as **input**").
Use `output_modalities()` / `supports_output_modality()`.

---

## 6. Seams

The two workstreams meet at `StreamChunk` and `ToolOutputEvent` and nowhere
else. Neither needs to wait on the other.

**Declaration half** (tier modality roles):
direction-qualified `modalities`; `tiers_for_modality(kind, direction)`;
startup check against output capability when available; `validate` / `explain`;
and parsing the currently-**discarded** right-hand side of
`architecture.modality` (`nebius/provider.py:248`, `ovhcloud:299`,
`doubleword:353` all do `split("->", 1)[0]`) into `output_modalities()` — the
output-capability source already being fetched and thrown away.

**Delivery half** (media chunks):
`StreamChunk.inline_data` + `audience`; the `ToolOutputEvent` fields in §5.3;
formatter bypass; the content-gate re-route in §5.2; a bounded queue with an
explicit drop policy (§3.4); `PresentationContext` renderable media.

**Model-outbound half** (audio streaming output), depends on delivery:
widening `StreamingCallback` beyond `Callable[[str], None]`; response-media
parsing in the adapters (note `content_block_to_part` at
`anthropic/converters.py:363` *already* builds `Part(inline_data=…)` from an
image block — it is wired to history rehydration at `:387`, never to the
response path); an outbound media column in `ProviderCapabilities`; and the
`_FORWARDED_API_PARAMS` additions.

---

## 7. Open questions

1. **Drop policy** when a bounded queue fills for a `CLIENT` media stream —
   drop oldest (audio prefers recency) or oldest-text/newest-media? Needs to
   be per-audience, not global.
2. **Does a `CLIENT` chunk enter history at all?** Proposed: no. It never
   reached the model, so replaying it on revive would be a lie. But the
   *fact* that media was produced probably should be recorded.
3. **Whole-blob vs chunked threshold** for tool-produced media — is a
   single-chunk stream always right, or is there a size above which a tool
   should be required to chunk?
4. **Should the startup capability check cover cross-provider tiers?**
   Today it skips them so validating one doesn't eagerly construct that
   provider on turn 1, and there is no lazy check on entry. The gate now
   reports the misconfiguration as a profile error instead of looping, so
   this is a diagnosis-latency question rather than a correctness one —
   but it is the only role declaration nothing verifies before content
   arrives.
5. **Should `PresentationContext` renderability gate production**, or only
   delivery? Generating TTS audio no client can play is waste, but the
   capability is known only per-connected-client and a session may have
   several.

---

## 8. Ears and voice in the same turn (#837)

The two halves above were built independently and, once both landed, could
not be used together. A user message carrying an attachment leaves
`JaatoSession.send_message` for `send_message_with_parts`, and the loop it
lands in — `_run_chat_loop_with_parts` — called the **batched**
`provider.complete()` unconditionally. `_use_streaming` was never consulted
there; the `streaming=False` telemetry label was accurate, not a mislabel.

Measured on one daemon, one model (`openai/gpt-audio-mini` on openrouter),
one `modalities: {audio: bidirectional}` tier:

| Request | Result |
|---|---|
| prompt, **no** attachment | works — 30 media chunks, a spoken answer |
| prompt + `audio/wav` attachment | `400 {'message': 'Audio output requires stream: true'}` |

The upstream is right to refuse: OpenAI emits audio **only** while streaming
(which is why `STREAM_AUDIO_MIME` spells out the pcm16 parameters), so a
batched request that also asks for audio output cannot be satisfied. Each
half validated cleanly on its own — `jaato-scaffold validate` reported no
findings and the session started — and the first turn that used both
directions failed.

It went unnoticed because the parts path predates media output and was built
for **images**, where a batched vision turn is perfectly reasonable. Audio
input is the first attachment kind whose *reply* may itself be audio, which
is what makes the two paths mutually exclusive.

The decision now lives in one place, `JaatoSession._resolve_use_streaming`,
which both chat loops call; the parts loop dispatches both of its provider
calls through `_complete_parts_turn`, so the two sites cannot drift apart
again. On the streaming branch a `MediaDelta` goes to
`_deliver_model_media` and text goes to `on_output` chunk by chunk — so the
assembled-response emission that followed each provider call is suppressed
(via `_emit_batched_response_text`), because doing both renders every answer
twice. A provider reporting `supports_streaming() == False` still gets the
batched call, and still emits its text: the vision turns this path was built
for are unchanged.

Guarded by `shared/tests/test_an_attachment_does_not_silence_the_model.py`,
which asserts on the dispatch and on audio reaching a subscribed client, and
declares both reversions to the meta-suite.

---

## 9. A message whose payload is not text (#838)

With #837 fixed, the voice turn still did nothing — one step earlier, and in
the failure mode that is hardest to read.

The daemon decides whether a `SendMessageRequest` becomes a model turn in
`SessionManager.handle_request`, and that decision consulted the message
**text** and nothing else:

```python
if not (message_text and message_text.strip()):
    self._emit_to_client(client_id, TurnCompletedEvent())
    return                                   # no model turn, ever
```

`event.attachments` sat on the same object, read twenty lines later to be
handed to `server.send_message` — on the path this branch had already
returned from. Measured, one daemon, one session profile, the same 88 KB
`audio/wav` attachment, only the text differing:

| `text` | Events observed | Reached the provider? |
|---|---|---|
| `"Answer what you hear."` | provider `400`, surfaced as `AgentError` | yes |
| `""` | one `TURN_COMPLETED`, nothing else | **no** |

The first row is the diagnostic one: the same attachment *with* text got far
enough to be refused by the upstream, so the empty-text form was dropped
**before** the wire rather than failing at it.

**Why this shape matters.** For an image, blank text is unusual — there is
normally a question about the picture. For **audio it is the normal case**:
the attachment *is* the message. A voice turn is "here is what I said", and
supplying text alongside asks a second question the persona then has to
choose between. So `session.complete("", attachments=[utterance])`, the
natural voice request, was exactly the one that silently did nothing.

**Every layer below already handled it**, which is why nothing else had to
change: `JaatoSession._parts_from_user_message` says so in its own docstring
("an empty `message` (image-only turn) yields parts with no text"), the
runner RPC accepts `""` as a valid `str` prompt, and the standalone-WS
handler dispatches an attachment-only send with **no emptiness check at
all**. That asymmetry — the same message working over WS-standalone and
dropped over IPC/SDK — is what identifies this one site as the defect rather
than the policy.

**And the failure mode was the bad kind.** Not an exception, not a refusal
naming a reason — a completed turn. A caller reads an empty payload and
cannot tell "the model had nothing to say" from "nothing was ever asked"; it
cost a debugging round precisely because the first symptom (`payload=None`)
looked like a model that declined to answer. So the branch that remains,
`SessionManager._close_contentless_message`, distinguishes its two arrivals:
a solely-`%name --help` message closes quietly, because the help *was* the
answer, while a request that arrives with no text and no attachments is
refused by name (`ErrorEvent(error_type="EmptyMessageError")`) before its
turn is closed. Both still emit the synthetic `TurnCompletedEvent` that
keeps a client's stall detector from killing the session.

Guarded by `server/tests/test_an_attachment_is_content.py`, which
declares both reversions to the meta-suite.

## 10. What a model was given is replayed to whatever model comes next (#847)

A session that has **heard** audio could not switch to a tier whose model
has no audio input. The caller's utterance stays in history — it must; the
audio tier needs it on the next turn — and history is replayed on *every*
later request, including the ones made while a text tier is active. The
upstream refused those:

```
ModelNotFoundError: Model not found: openai/gpt-4o-mini
404 {'error': {'message': 'No endpoints found that support input audio',
     'metadata': {'failed_routing_step': 'Filter by Input Audio Support'}}}
```

Not a missing model. The model exists; the *request* carried input audio.

**The gate existed and covered one direction.**
`_gate_tool_results_for_active_modalities` withholds attachments the active
model cannot consume and appends a note — for **tool results only**.
`ensure_spoken_part` substitutes a transcript for the model's own *outbound*
speech. Neither says anything about audio the model was *given*, so
`duet`-style profiles (audio outbound only, model media never entering
history) were fine and the failure appeared only once a session accepted
inbound audio, which §*The ears* made possible.

`JaatoSession._gate_history_for_active_modalities` is the missing half, and
it sits where `docs/design/multimodal-model-support.md` always said the gate
belonged: *the session's send path, right before history→provider
conversion*, where the active provider and the outgoing `Part`s are both in
scope. Every `provider.complete()` call site now reads its message list from
`_history_for_provider()` rather than `SessionHistory.messages`.

**Per-request, never destructive.** This is the sharp requirement, and the
reason the fix is not "strip audio from history". The filter returns a
*copy*; the stored history keeps the bytes, so `enter_tier("voice")` finds
them again. A fix that mutated history would repair the planner by
permanently deafening the session — the text tier's first turn would cost
the audio tier every turn after it. `message_id` survives the copy, because
GC's history-budget sync keys on it, and when nothing is withheld the caller
gets the stored `Message` objects themselves, so a text-only session
allocates nothing.

**A note, not a silent drop**, for the same reason it is one on the tool
path: a planner handed a user turn whose audio quietly vanished answers as
though the caller said nothing, and a contentless turn is indistinguishable
from a caller who was silent. `_build_withheld_attachment_note` produces it,
now taking a `retry_action` so the history gate does not tell the agent to
"re-run this tool" — nothing needs re-running, the bytes are still in
history and the voice tier's next request carries them again.

**It covers tool results in history too**, which have the same problem one
step removed: `_gate_one_tool_result` runs when a result is produced,
against the model active *then*, so an image a vision tier legitimately kept
is still in history when the agent switches to a text tier.

**The narrower fix was available and is the wrong layer.**
`openrouter/converters.py` hardcodes `audio_as_input_audio=True` at two
sites, applying OpenRouter's `ProviderCapabilities.audio_input` — true of
*some* of its models — to every model on the provider; asking
`self.modalities()` instead would have fixed this 404 in two lines. It would
also have dropped the content silently, and only for `openrouter`, while the
same latent hardcode (a wire capability standing in for a model check) lives
in every OpenAI-shaped converter: `_openai_compat` emits `image_url` for an
`inline_data` image whatever the model declares. Gating above the converter
answers all of them with the framework's one answer to "can this model
consume this", `provider.supports_modality()` — the same answer the
tool-result gate and the startup tier check already read.

**Consequence worth stating.** That uniformity is a behaviour change for the
providers that inherit the text-only floor from `ModalityCapabilityMixin`
(nim, vllm, lmstudio, tensorrt_llm, triton, zhipuai_openai, …): a
user-message image now meets the same withhold their *tool-result* images
have always met. The two paths contradicting each other was the older bug;
the note names it out loud rather than letting bytes reach a model the
framework has already declared cannot see them. A provider serving a vision
model with no `modalities` knob to assert it is a separate gap, in that
provider.

**The span follows the request, not the store.**
`_record_input_messages_telemetry` built its OpenInference
`llm.input_messages.*` from `_history.messages`, which was the same list as
the request's until this gate existed. It now reads `_history_for_provider()`
for the same reason the request does: a span showing a text tier receiving
audio it was specifically not sent is the one reading that makes this 404
look impossible. Every call site already sits inside the `llm_span` wrapping
`complete()` and after the turn's history append, so it resolves against the
same active model the request will use.

Guarded by `shared/tests/test_history_modality_gate.py`, which checks the
stored history in every case — a test asserting only "the text tier sent no
audio" passes for the destructive fix too.

## 11. How long anyone sees it: consumed-media eviction (#850)

§10 fixed *which* model sees an utterance. This is about how long **anyone**
sees it.

Media had a lifecycle in exactly one direction. Outbound is right: model
media is `CLIENT`-audience so it never enters history at all, and
`ensure_spoken_part` leaves the *transcript* where the audio would have gone
— words kept, bytes discarded. Inbound got neither treatment. An utterance
stayed in history verbatim and rode every subsequent request, forever.

Measured on a five-question helpdesk call (`heard N bytes` per turn):

```
603244 + 888044 + 417644 + 680044 + 243244  =  ~2.8 MB of audio
```

By the last question the request carried all of it (~3.8 MB base64), growing
every turn. The run ended with `runner RPC closed before id=83 responded` at
9.6/11 GiB of host memory — causation unproven (dmesg was unreadable), the
growth itself measured, and a cost/latency problem regardless.

### 11.1 GC could not see any of it

`grep -rn "inline_data" shared/plugins/gc_*/` returned nothing.
`estimate_message_tokens` walked `text`, `function_call` and
`function_response` and never looked at `inline_data`, so a 600 KB utterance
was sized at the one-token floor: a threshold set at 60% could not fire on
the payload it exists to bound, and eviction ordering could not prefer a turn
it believed was empty. `_update_conversation_budget` had the same hole, so
the InstructionBudget denominator omitted it too.

Two fixes, deliberately different in kind:

| | Unit | Where | Why this unit |
|---|---|---|---|
| `estimate_media_tokens` | tokens | `estimate_message_tokens`, `_update_conversation_budget` | so a media-carrying turn is *sized*, and eviction ordering can prefer it |
| `context_usage["media_bytes"]` | **bytes** | `media_pressure_reason`, consulted by all four strategies | so a session can trigger collection on payload while far under its token threshold |

The byte channel exists because the thing that killed the session was request
**size**, and an operator bounding it thinks in megabytes. Laundering bytes
through a token estimate to compare against `threshold_percent` would hide
the quantity that actually matters behind a guess. `MEDIA_BYTES_PER_TOKEN` is
an order-of-magnitude anchor per top-level mime (audio ≈ 4.8 kB/token, image
and PDF ≈ 1.5 kB/token, anything unclassified charged at its base64 wire cost
of 3 B/token — the harshest rate, because a payload nobody can name is the
one whose real cost is least knowable). Wrong-but-visible beats invisible;
the rate shifts *when* collection fires, never whether the bytes are seen.

### 11.2 Purging cannot simply delete

In a domain with call-retention duties, "the agent handled a claim from audio
nobody can produce" is exactly the audit objection. Whatever replaces the
bytes has to carry a reference that can be cross-referenced with an archived
recording.

Nothing supported that: a user-message attachment was normalised to
`{mime_type, data, display_name}` — **no id** — while outbound media had
carried `stream_id`/`sequence` since #824. So the identifier had to be minted
at **ingest**, before anything was in a position to purge.

`jaato_sdk/media_identity.py` mints it as a digest of the payload
(`att_<16 hex>` of SHA-256) rather than a uuid, because the cross-reference
has to work from the *archive* side: someone holding a `.wav` and a
transcript naming an id can recompute the id from the recording itself,
so the link survives the loss of every intermediate record. Two
byte-identical attachments collide, and that is correct — they are the same
recording. Minting is idempotent and side-effect free, which is what lets the
SDK client mint it (the sender is the side that archives the file, and needs
to know what to file it under) while `_parts_from_user_message` back-fills
the same value for clients that send none; an id the caller supplied under a
scheme of its own is never overwritten.

Note the separation the fix rests on: purging from **model context** is not
discarding the recording. Archiving is a file on disk; keeping the bytes in a
prompt serves nobody.

### 11.3 Shape A, with shape B left open

The issue names two shapes. **A** — evict the bytes, leave a marker carrying
the id, duration and mime — is what ships: no transcript source needed, no
new dependency. **B** — substitute a transcript, mirroring
`ensure_spoken_part` — is higher fidelity and symmetric with outbound, and
needs a transcript that does not exist for inbound audio on chat-completions
today (the audio model consumes the bytes directly and emits no input
transcription). They compose: B where a transcript is available, A as the
floor. Both need the ingest id.

```
[Media evicted after the turn that consumed it — question.wav, audio/wav,
 12.6s, 588.0 KB, ref att_9f2c1ab73e0d4455. The recording is not discarded
 by this; cite the ref to locate the archived original.]
```

### 11.4 Destructive, and at the START of a turn

`evict_consumed_media` (`plugins/gc/utils.py`) is the opposite of
`_gate_history_for_active_modalities` in the one way that matters: the gate
filters a per-request **copy** so a later `enter_tier` back to a voice model
can still hear, while eviction rewrites the **stored** history and the model
genuinely cannot re-listen. That is the accepted trade — re-hearing a
recording yields the understanding the conversation already records in words.

`JaatoSession._evict_consumed_media` runs at the **start** of a turn, from
both chat loops, immediately before the new user message is appended.
Everything in history at that moment belongs to a turn that has completed,
which is what makes "consumed" true without having to reason about it; and a
turn that failed before the model ever saw its audio keeps the bytes, so the
caller can simply send again. Running it in the previous turn's `finally`
would look equivalent and would purge on exactly the path where the audio was
never used. Both loops call it because a text turn following a voice turn
must stop carrying the audio too, or the growth resumes whenever the caller
types instead of speaking.

Bookkeeping mirrors `_dedup_history_for_gc`, the other history-rewriting
maintenance pass: the rewrite preserves `message_id`, so the per-message
token cache is invalidated for the touched messages before the budget
re-sync, or it reads back the pre-eviction size.

Audio only, by default (`GCConfig.media_evict_mime_prefixes`). An image is
routinely re-examined across turns ("what does the third column say?") and a
PDF is a document the conversation keeps referring back to; a recording
re-heard says what it said the first time.

### 11.5 Knobs

| Surface | Key | Default |
|---|---|---|
| profile `gc:` | `evict_consumed_media` | `true` |
| profile `gc:` | `media_evict_mime_prefixes` | `["audio/"]` |
| profile `gc:` | `media_bytes_threshold` | 8 MiB |
| `.jaato/gc.json` | same three keys | same |
| env | `JAATO_GC_MEDIA_BYTES` | 8 MiB (`0` disables) |

The profile and `gc.json` layers pass a media key **only when it is set**,
because these defaults live on `GCConfig` — one of them behind the env var —
and spelling them at the loader would make every profile carrying a `gc:`
block silently outrank `JAATO_GC_MEDIA_BYTES`.

Guarded by `shared/tests/test_heard_audio_does_not_accumulate.py`, whose
final assertion is the issue's own acceptance criterion: request payload
across six voice turns must not grow with the turn count.

## 12. The words reach the client on the final chunk (#869)

§11 settled what remains when the bytes go: the *words*. For inbound audio
that is a marker naming an id; for outbound, `ensure_spoken_part` leaves the
model's transcript in history where the audio would have been. But history
was the **only** place it went. Every spoken turn produced a transcript
inside the provider, and no client could obtain it.

### 12.1 Where it went

`emit_audio_delta` took the transcript off `delta.audio` and appended it to a
caller-owned list, `transcript_sink`, emitting no chunk for it — correctly,
since an empty `MediaDelta` would hand a client zero bytes to play. At the
end of the stream the joined sink became a text Part on the
`ProviderResponse` (`ensure_spoken_part`). That is *after* streaming has
finished, past every point that emits to a client: nothing produced it as
`AGENT_OUTPUT`, and `ToolOutputEvent.chunk` — sitting empty on every media
event — never received it either.

Measured in one session:

| Turn | `ask()` returned |
|---|---|
| model speaks only | `''` — 14 media chunks, 5.45 s of audio, one `AGENT_OUTPUT` with empty text |
| model writes *and* speaks | the written text |

A voice client could therefore record how long the agent spoke but not what
it said; the alternative was re-transcribing audio it already held through a
second model, paying again for words the first model had produced and thrown
away.

### 12.2 Ride the final chunk

The `pending` one-slot buffer (#828, `is_end_of_audio`)
already holds the last chunk back so it can be marked `final`, and the
transcript is complete at exactly that moment — every transcript delta
measured precedes the end-of-audio marker. So the chunk released as `final`
carries `"".join(transcript_sink)` in `MediaDelta.transcript`, which
`_deliver_model_media` already forwarded into `ToolOutputEvent.chunk`. One
existing event gains text; no new event, no new subscription, and the
downstream plumbing needed no change at all. The end-of-stream flush
(`flush_audio_stream`) takes the same arguments, so a provider that sends no
marker delivers the words too.

Intermediate chunks stay wordless. A client reads the utterance exactly
once, off the same event that tells it playback is over.

### 12.3 The same rule as history

`ensure_spoken_part` appends only when the model wrote no text of its own,
because a turn that both wrote and spoke already has its words. The wire
carries the constraint over, through one shared predicate —
`model_wrote_text(parts, accumulated_text)` — so what a client reads off the
wire is what history records and a call log never gets the same answer
twice: the written text arrived as `AGENT_OUTPUT`, and the final media chunk
then carries no transcript.

The predicate is a **callable, read at the marker**, not a flag read at the
start of the stream: the marker lands mid-stream, and a written answer at
that moment still sits in the loop's not-yet-flushed accumulator (text is
flushed into a Part only at a tool-call boundary or at the end of the
stream), which is why it is asked about both.

Acceptance, as the issue stated it, is guarded by
`TestTheWordsRideTheFinalChunk` in `test_media_output_contract.py`: a client
subscribed to model media obtains the spoken words without a second model
call; a turn that speaks and writes does not deliver the same words twice;
and a transcript-only delta still produces no playable chunk.

## 13. Driving a session that has already ended (#845)

Sections 8–12 made a session that *hears* and *speaks* possible. They all
concern one live turn. This section is about the turn after the session
stopped.

There are exactly two ways to drive an **existing** session, and both were
text-only:

```python
# server/command_router.py  _handle_session_wake
#   payload {session_id, text, source?, event_id?}   — no attachments
# jaato_sdk/client/ipc.py   inject_prompt(self, text, source_type=..., ...)
# jaato_sdk/events.py       InjectPromptRequest.text: str = ""
```

`attachments` existed only on `send_message` (and the `ask` / `complete` /
`stream` wrappers over it) — the **live-session** path.

### 13.1 Why that closes the door on a voice agent

A completion-gated session is *designed* to end: `signal_completion` makes it
quiescent, emits `SESSION_TERMINATED` and releases the runner. The documented
way back in is `session.wake`, which cold-revives from disk. That works
perfectly for a text agent, whose next input is a sentence. For a voice agent
the next input is a spoken utterance, and there was no field to put it in.

Measured while building a push-to-talk loop against a
`modalities: {audio: bidirectional}` tier:

| Attempt | Result |
|---|---|
| reuse the live session object after completion | `400 … tool_call_ids did not have response messages` on a real second question; a **hang** when the same audio was sent twice |
| `session.wake` with the utterance | not expressible |
| `inject_prompt` with the utterance | not expressible; and into a completed session it returns `terminated`, a documented non-delivery |

The workaround — drop the completion contract, keep one live session, call
`ask()` per turn — is right for a *conversation* and does nothing for the
shapes wake exists for: a suspended agent revived hours later, a cascade arm
resumed on a schedule, anything cold-started from disk.

### 13.2 Bytes cannot be wrapped, so the boundary is stated beside them

A wake payload is untrusted by construction — whoever holds the session id
drives it, which may be a webhook, a cron, or a public PR comment — so
`wake_session` has always passed its `text` through `wrap_untrusted_content`.
An attachment has nowhere to put a marker: an audio part is `inline_data` on
the wire and there is nothing in it to defang. Inheriting the text-only wrap
"by accident" would have left the model weighing a **spoken** instruction
differently from the identical **typed** one, which is the asymmetry the
boundary exists to remove.

`_wrap_wake_content` answers it explicitly: each attachment is named *inside*
the wrapper — mime, display name, and the ingest id §11.2 mints — with a line
saying the media delivered alongside is data to interpret and never
instructions. Metadata only; the payload stays an `inline_data` part, because
base64 in the prompt would both double the cost of the bytes and bypass the
provider's own media handling. A wake carrying attachments and **no text** is
normal (for a spoken message the attachment *is* the message, §9), and
produces a wrapper carrying only the manifest — never an empty one, so the
model is never handed unexplained media.

### 13.3 An attachment-bearing inject is idle-only

Wake always drives a fresh turn, so it can carry anything a turn can carry.
`inject_prompt` has two outcomes and only one of them can:

| Outcome | What happens to the message | Can it carry bytes? |
|---------|-----------------------------|---------------------|
| `needs_turn` → **drive** | becomes a `SendMessageRequest` — the same path a client send takes | yes |
| **queue** behind a running turn | folded in as TEXT: appended to the last tool result's `model_suffix` (`_send_tool_results_and_continue`) or replayed as `Message.from_text` (`_handle_pending_mid_turn_prompt`) | no |

Neither queued shape has anywhere to put an `inline_data` part. So
`deliver_prompt_to_session` sets `require_idle` whenever attachments are
present, whether or not the caller asked: a busy target answers `BUSY` with
**nothing enqueued**, a retry-safe refusal, instead of accepting the message
and discarding the payload that *was* the message. Silently stripping it
would reproduce §9's failure one layer up — a turn reported as delivered
that the model has nothing to answer.

A text-only inject is bit-identical to before; the idle-only rule is what
attachments cost, and it is paid only by callers who send them.

### 13.4 An old daemon is refused, not degraded

`attachments` is an additive optional field, and additive optional fields are
normally safe to send blind — an older peer ignores them and the call degrades
to what it always did. That reasoning holds for a `request_id` (protocol 1.3)
and fails here: the degraded call is a turn driven **without** the audio, and
for a blank-text utterance that is an empty turn reported as a success. The
SDK therefore raises when the daemon predates protocol 1.5
(`IPCClient.MIN_ATTACHMENT_RESUME_PROTOCOL`), rather than letting the payload
vanish between two versions that both claim compatibility.

### 13.5 What did NOT change

The queue does not learn to carry media. `QueuedMessage` is still text, and
mid-turn piggyback is still a string appended to a tool result — teaching
half the drain sites to carry parts would make delivery a lottery on which
site collected the message. The honest shape is the one above: the drive path
carries bytes, the queue path refuses them by name.

The **HTTP wake ingress** (`server/wake_ingress.py`) stays text-only too. Its
canonical signed body is `{wake_ref, text, source, event_id, ts}` and the
relay signs the RAW bytes of it; putting a base64 payload inside that
envelope changes a verified security contract (and the size of every signed
request) rather than adding a field. A relay that needs to deliver media
should be given its own body shape and its own decision about what is
signed — a separate change, deliberately not made in passing here.


## 14. Answering a question with media (#989)

§13 made an *existing* session drivable with bytes. This is the other
direction the agent itself opens: `request_clarification` asks a question and
blocks the turn until someone answers it, and the answer was text-only. A
Telegram user asked "what's your name?" and replying with a voice note had
nowhere to go — submitting the audio as an ordinary turn instead is stashed
silently behind the pending clarification (there is no path that folds a
mid-turn attachment into a running turn), so the question stays unanswered
and the recording is delivered later as a surprise turn.

### 14.1 A clarification answer is a tool result

The decisive fact, and the one the issue as filed missed.
`_execute_clarification` returns a `ToolResult`, so the natural home for the
bytes is `ToolResult.attachments`:

| what it buys | where it already exists |
|---|---|
| the wire | `google_genai/converters.py`, `anthropic/converters.py` and `_attachments.tool_result_followup_message` (which emits `input_audio` under `audio_as_input_audio`) all marshal `ToolResult.attachments` today |
| the modality gate | `_gate_one_tool_result` already runs on exactly this shape, withholds what the active model cannot consume, routes the refused bytes to the CLIENT, and leaves the model a note |
| the lifecycle | once GC can see a tool result's attachments (§14.5), `evict_consumed_media`'s per-mime default is already the right policy |

The alternative — ferrying them in as a user turn through
`_parts_from_user_message` — puts a message in history the user never sent,
leaves the tool result claiming it carried nothing, and *worsens* the
question↔answer adjacency rather than improving it: on every OpenAI-shaped
wire the tool-result path already synthesises a follow-up `role: "user"`
message, so it is the same wire shape with the framework-level bookkeeping
done by hand.

### 14.2 Attachments are orthogonal to the answer's TYPE

The issue proposed "free_text answers become text and/or attachments; choice
answers stay ordinal (attachments N/A)". N/A is wrong, and the case that
shows it is ordinary:

```
How should we design this?
  1. You attach a screenshot of the design
  2. We discuss the design
```

Choosing (1) *and attaching the screenshot* is `selected_choices=[1]` **plus**
an image. The ordinal says which branch; the attachment says what content.
So `attachments` is a field of `Answer`, beside `selected_choices` /
`free_text` / `skipped` — never a free-text sub-case, which would foreclose
that permanently.

The question-side half follows the same split: in that example option 1 wants
a file and option 2 does not, so a question-level flag is too coarse.
`Choice.expects_attachment` is the declaration, and `QuestionType` stays three
values. It is advisory — nothing refuses an answer that ignores it, and
nothing refuses an attachment on a choice that never declared it. What it buys
is a client that can render an attach control in the right place instead of
the agent writing *"attach a screenshot"* as prose nothing downstream can act
on.

### 14.3 What is refused at submit, and what is deliberately not

`JaatoServer.respond_to_clarification_batch` validates the whole map before
resolving anything, and on **any** problem emits an `ErrorEvent` and leaves
the clarification OPEN — resolving it with the media dropped would report an
answer the user never gave, and for a voice-only answer an empty one (§9).

| refusal | why the daemon is the right place |
|---|---|
| an index that names no question | the relay records the batch's own question count when it emits it |
| a payload that does not decode | the alternative is an attachment that is silently empty, which is worse than an absent one |
| the batch over a **6 MiB** cap | load-bearing, not hygiene — see below |
| attachments with no relay waiting | the daemon-local `QueueChannel` carries answer STRINGS through a queue and has nowhere to put bytes |

The cap is the frame budget. The answers cross daemon→runner as the
*response* to `client.request_clarification`, under the 10 MB
`MAX_MESSAGE_SIZE`, and an oversized response is **not written at all**:
`RunnerRPCClient._write_frame_json` raises `FrameTooLargeError`, which is
logged and swallowed, so the runner's call never resolves and the turn hangs
behind a clarification nobody can answer. A 120 s utterance is 3.84 MB raw /
~5.12 MB serialised: one fits, two do not. What the right multi-question voice
policy is — per-question delivery, a per-answer cap, chunking — is still open;
failing loudly at submit is what keeps that question open instead of answering
it with a hang.

**Not** refused at submit: whether the active model can consume a mime. The
daemon does not hold the runner's provider, and the tier can change between
the question and the answer (§10), so a submit-time check would be a guess
wrong in both directions — refusing content the model could consume, or
passing content it cannot. `_gate_one_tool_result` answers it against the
model the bytes actually reach. A note the user never sees is the acknowledged
cost, and it is partly paid back by the same gate routing the withheld bytes
to the client as `CLIENT` media.

### 14.4 Which of §13's rules transfer

**Rule 1 (the untrusted boundary stated beside the bytes) transfers as
LABELLING, not as a trust boundary.** A wake payload is untrusted because it
arrives from a webhook or a cron. A clarification answer is a direct reply to
a question *this agent just asked*, and today's **text** answer is inserted
into the tool result verbatim with no wrapper. Defanging only the audio would
weigh a spoken answer differently from the identical typed one — the exact
asymmetry `_wrap_wake_content` exists to avoid, inverted. So each attachment
is NAMED beside the answer it belongs to (mime, display name, ingest id,
duration; never the payload), which is also what makes a multi-question batch
attributable at all: the wire's follow-up message names files in one lead line
with no structural link to a question.

**Rule 2 (an attachment-bearing inject is idle-only) cannot transfer** — a
clarification answer arrives mid-turn by construction. Its *reason* was
mechanical (the queue folds a message in as text and has nowhere to put an
`inline_data` part) and does not apply: this path resolves a future whose
result becomes a tool result, which does have somewhere to put bytes.

**Rule 3 (refuse an old daemon) transfers verbatim.** An older daemon ignores
`answer_attachments` and answers the clarification with the media gone — and
for a voice-only answer `_parse_answer` reads the empty response as
`free_text=""`, so a blank answer is reported as a successful one. Protocol
**1.6**, and both SDKs raise below it
(`MIN_CLARIFICATION_ATTACHMENT_PROTOCOL`).

And §9's rule holds inside the answer: an attachment IS content, so a
blank-text answer carrying bytes is un-skipped
(`_parse_answer_with_attachments`). Without that, the normal voice case is
reported to the model as "the user declined to answer" with its audio beside
the claim.

### 14.5 The prerequisite: GC could not see a tool result's bytes

`message_media_bytes`, `estimate_media_tokens` and the eviction walk all read
`part.inline_data` and nothing else; `function_response` appeared in
`gc/utils.py` only for message *grouping*. So a `ToolResult.attachments`
payload was sized at the one-token floor, absent from
`context_usage["media_bytes"]`, and immune to `_evict_consumed_media` — §11's
hole, one part-shape over. Pre-existing for the `_multimodal` image tools; a
clarification answered by voice is where it recurs, once per question, every
turn.

`part_media_views` is now the single answer to "what binary payload does this
part carry", rendering a tool result's attachments into the same
`{mime_type, data, display_name}` dict an `inline_data` part already is, so
the rate table, the byte count, the mime match and the marker text stay one
implementation. The id is **minted from the payload** rather than read off a
field: `Attachment` has none, and the id is a digest of the bytes, so
recomputing it yields exactly what ingest would have.

Eviction cannot swap the part for a text one — the model answered from the
result's `result`, and the ledger, completion processors and enrichment all
read it — so the attachment is dropped and the marker appended to
`model_suffix`, the existing model-facing-only channel.

**No new knob.** `media_evict_mime_prefixes` already defaults to `("audio/",)`
on §11's reasoning that an image is routinely re-examined across turns and a
recording is not. That is exactly the right split here: the screenshot
attached to a design question survives the turns that discuss it; the voice
note answering "what's your name" does not.

### 14.6 Scoped out

- **Subagents.** `ParentBridgedChannel` parses the parent's answer out of an
  injected TEXT prompt and has no representation for bytes at all. A subagent's
  `Answer.attachments` is therefore always empty, stated on the class rather
  than left to look supported.
- **Advertising the accepted mimes at ask time.** Putting
  `accepted_attachment_mimes` on `ClarificationBatchEvent` — so a client knows
  before the user records whether to offer the microphone — needs the
  clarification plugin to hold a session and read the active provider's
  vocabulary. Worth doing; an unpopulated field would only lie by omission
  (empty meaning "unknown", not "nothing accepted"), so it is absent rather
  than stubbed.
- **The TS client's normalisation.** `respondToClarificationBatch` gained the
  parameter and the protocol refusal, but that SDK reads no files and encodes
  nothing (exactly as `sendMessage` and `wakeSession` do not): a TS caller
  passes canonical wire dicts.
