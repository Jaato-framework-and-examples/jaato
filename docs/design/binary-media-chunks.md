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
