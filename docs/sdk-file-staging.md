# SDK file staging — wire protocol

WS clients (telegram, future mobile/embedded) need to put files into a
server-provisioned workspace they cannot reach via the filesystem. The
`StageFilesRequest` event is the canonical SDK primitive for this.

## Why a multi-frame protocol

The obvious encoding — base64-in-JSON — inflates payloads by 33% and
forces the entire file through the JSON parser. Both costs scale badly
once files exceed a megabyte or two. WebSocket frames support raw
binary natively; the protocol below uses a single TEXT frame for
metadata followed by N raw BINARY frames for the payloads.

Frame ordering is preserved per-connection by the WebSocket protocol,
and the server's per-connection receive loop (`async for message in
websocket: await _handle_message(...)`) awaits each handler to
completion before pulling the next frame. The staging handler reads its
binary frames inline via explicit `await ws.recv()` calls — they cannot
interleave with other event types on the same connection.

## Wire sequence

```
client                                                      server
  │                                                            │
  │  TEXT  StageFilesRequest                                   │
  │ ─────────────────────────────────────────────────────────► │
  │  {                                                         │
  │    "type": "workspace.files.stage_request",                │
  │    "workspace_id": "ws_a1b2c3d4",  /* "" → current */      │
  │    "files": [                                              │
  │      {"name": "input.pdf",  "size": 184320,                │
  │       "content_type": "application/pdf"},                  │
  │      {"name": "sub/notes.md", "size": 4096}                │
  │    ]                                                       │
  │  }                                                         │
  │                                                            │
  │  BINARY  <184320 bytes of input.pdf>                       │
  │ ─────────────────────────────────────────────────────────► │
  │                                                            │
  │  BINARY  <4096 bytes of sub/notes.md>                      │
  │ ─────────────────────────────────────────────────────────► │
  │                                                            │
  │                    TEXT  StageFilesEvent                   │
  │ ◄───────────────────────────────────────────────────────── │
  │                    {                                       │
  │                      "type": "workspace.files.staged",     │
  │                      "workspace_id": "ws_a1b2c3d4",        │
  │                      "staged": ["input.pdf", "sub/notes.md"], │
  │                      "failed": []                          │
  │                    }                                       │
```

The number of binary frames must equal `len(files)`. They must arrive
in the same order as the `files` array. Each frame's byte length must
exactly equal the corresponding `size` field.

## Validation and error categories

Up-front (before any binary frame is read):

- `workspace_not_found` — `workspace_id` doesn't match the client's
  current workspace, or the client has none. The server still drains
  the binary frames the client is about to send so the connection
  stays aligned.
- `size_limit_total` — sum of declared `size` values exceeds the total
  payload cap (default 50 MB).

Per-file (some failures still allow processing of subsequent files):

- `unsafe_path` — `name` is empty, absolute, or contains `..`. Frame
  is drained; remaining files continue.
- `size_limit_per_file` — declared `size` exceeds the per-file cap
  (default 10 MB). Frame is drained; remaining files continue.
- `size_mismatch` — the binary frame's actual length doesn't equal the
  declared `size`. File is failed; remaining files continue.
- `io_error` — write failed (disk full, permission denied, AppArmor
  refusal). File is failed; remaining files continue.

Fatal (aborts the remainder of the stream):

- TEXT frame where BINARY was expected — protocol violation. Recorded
  as `size_mismatch` with `"protocol violation"` in the message.
  Subsequent files in the request are silently skipped because we
  cannot trust the stream alignment.

## Caps

| Limit | Default | Where enforced |
|-------|---------|----------------|
| Per-file size | 10 MB | `DEFAULT_STAGE_PER_FILE_LIMIT` in `server/websocket.py` |
| Total payload size | 50 MB | `DEFAULT_STAGE_TOTAL_LIMIT` in `server/websocket.py` |

These will become per-deployment configurable when the next consumer
needs different values.

## Workspace targeting

`workspace_id` identifies the target workspace. Two valid values:

- `""` (empty) — target the connection's currently-attached workspace.
  This is the common case for telegram-style clients that auto-
  provision their workspace via `session.new`.
- `"<basename>"` — target a specific workspace by its directory
  basename. The server only allows this when the basename matches the
  client's currently-attached workspace; cross-client staging is not
  permitted without per-user auth (see
  `project_backlog_ws_per_user_auth.md`).

Clients learn the basename from `WorkspaceCreatedEvent` and from the
workspace path in `SessionInfoEvent`.

## Relation to legacy `staged_files` on `session.new`

The premium `<jaato-task>` web component uses an inline base64
`staged_files` field on the `session.new` envelope. That mechanism is
preserved for back-compat — it bundles workspace creation and file
staging into a single request, which suits the auto-provision flow
where the workspace doesn't exist before `session.new` runs.

The two paths share the same write code (`_write_staged_payload` in
`server/websocket.py`). New SDK-built clients should prefer
`StageFilesRequest` because:

- It avoids base64 inflation.
- It can stage files into an *already-existing* workspace mid-session.
- It has its own response event with explicit per-file results.

## Client integration: jaato-client-telegram

The Telegram bot (`jaato-client-telegram`) uses a single `WSTransport`
connection managed by `SessionPool`. Each Telegram user gets an
isolated session on the server. To adopt `StageFilesRequest`, the
following changes are needed.

### 1. Track workspace_id per session

Currently `SessionPool` stores `SessionMetadata(session_id, …)` but
does not capture the workspace path or basename from the
`SessionInfoEvent`. The event carries `workspace_path` which encodes
the workspace basename (e.g. `/data/workspaces/ws_a1b2c3d4` → basename
`ws_a1b2c3d4`).

**Change:** Extend `SessionMetadata` with a `workspace_id: str` field
(default `""`). Populate it in the `SessionInfoEvent` handler (both
`WSTransport._receiver_loop` and `SessionPool.on_session_info_event`).

### 2. Add `stage_files()` to WSTransport

The transport's `send()` method only handles TEXT frames via
`serialize_event()`. Staging requires sending the TEXT request followed
by N raw BINARY frames. Add a dedicated method:

```python
async def stage_files(
    self,
    workspace_id: str,
    specs: list[StagedFileSpec],
    payloads: list[bytes],
) -> StageFilesEvent:
    """Send StageFilesRequest + binary frames, await StageFilesEvent."""
    if not self._ws or not self._connected:
        raise RuntimeError("Not connected")
    if len(specs) != len(payloads):
        raise ValueError("specs and payloads must have the same length")

    request = StageFilesRequest(
        workspace_id=workspace_id,
        files=specs,
    )
    await self._ws.send(serialize_event(request))

    for data in payloads:
        await self._ws.send(data)

    # The receiver loop will route the StageFilesEvent to the calling
    # session's queue. Block until it arrives.
    future = asyncio.get_running_loop().create_future()
    self._stage_files_future = future
    try:
        return await asyncio.wait_for(future, timeout=120.0)
    finally:
        self._stage_files_future = None
```

The `_receiver_loop` must also be updated to recognise
`StageFilesEvent` and resolve this future (same pattern as the
existing `_session_future` for `SessionInfoEvent`).

**Imports to add** in `transport.py`:
- `StageFilesRequest`, `StageFilesEvent`, `StagedFileSpec` from
  `jaato_sdk.events`.

### 3. Add `stage_files()` to SessionPool

Expose a convenience method that resolves the workspace_id from the
session's metadata:

```python
async def stage_files(
    self,
    chat_id: int,
    files: list[tuple[str, bytes, str | None]],
) -> StageFilesEvent:
    """Stage files into the session's workspace.

    Args:
        chat_id: Telegram chat_id.
        files: List of (name, raw_bytes, content_type_or_None).

    Returns:
        StageFilesEvent from the server.
    """
    meta = self._sessions.get(chat_id)
    if not meta:
        raise RuntimeError(f"No session for chat_id {chat_id}")
    specs = [
        StagedFileSpec(name=name, size=len(data), content_type=ct)
        for name, data, ct in files
    ]
    payloads = [data for _, data, _ in files]
    return await self._transport.stage_files(
        workspace_id=meta.workspace_id,
        specs=specs,
        payloads=payloads,
    )
```

### 4. Wire into file-handling flows

The Telegram bot already has a `FileHandler` that downloads documents
and photos sent by users. After download, instead of saving locally,
the handler can call `session_pool.stage_files()` to push the file
into the agent's workspace. The agent then sees it at the declared
workspace-relative path.

### 5. Error handling

After receiving the `StageFilesEvent`, check `failed[]` before
proceeding. Each entry has `{"name", "category", "error"}`. Surface
`"unsafe_path"` and `"size_limit_*"` as user-facing messages;
`"io_error"` as a transient retry; `"workspace_not_found"` as a
session reset.

### 6. File size guard on the Telegram side

Telegram's Bot API limits downloads to 20 MB. The server caps are
10 MB per file and 50 MB total. Add a pre-send check:

```python
STAGE_PER_FILE_LIMIT = 10 * 1024 * 1024   # 10 MB
STAGE_TOTAL_LIMIT   = 50 * 1024 * 1024   # 50 MB
```

Reject files that exceed these limits before sending the request,
giving the user an immediate error message rather than waiting for
a server round-trip.

### Files to modify

| File | Change |
|------|--------|
| `transport.py` | Add `stage_files()` method; handle `StageFilesEvent` in receiver loop; add imports |
| `session_pool.py` | Add `workspace_id` to `SessionMetadata`; add `stage_files()` convenience method; extract workspace_id from `SessionInfoEvent` |
| `bot.py` | No changes needed (wiring is in session_pool/transport) |
| `handlers/` | Call `pool.stage_files()` from file-handling handlers after user sends a document |

## Future work

Already on the design backlog (see `project_backlog_sdk_file_staging`):

- Operator-configurable size caps (`stage.per_file_limit`,
  `stage.total_limit` in deployment config).
- `WorkspaceCreateRequest.initial_files` for clients that manage
  workspace lifecycle explicitly. Defer until such a client exists.
- Streaming/chunked uploads above the 50 MB cap. Cross that bridge
  when an HTTP companion endpoint becomes the better fit than WS.
