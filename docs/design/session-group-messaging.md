# Session Group Messaging — assessment and design

**Status**: assessment + proposed design (no code in this change)
**Date**: 2026-09-19
**Scope**: any session in a *group* can message any other session in the
same group; a target that is idle, detached, unloaded, or whose runner
slot is gone is revived to process the message; the payload carries text,
file references, and attachments (text or binary).

The framework already holds most of the pieces. This document inventories
them against the requirement, names the blocks that are missing, and
proposes the smallest design that closes the gaps without building a
second copy of anything that exists.

---

## 1. The requirement, restated as invariants

1. **Group** — a session belongs to a group by one of two facts it already
   carries: the authenticated user that created it (`created_by`), or the
   cascade it was stamped with (`cascade_driver_id`). Both are persisted
   (record 2.9 / 2.10) and restored on load.
2. **Any-to-any** — a member may address any other member. No parent
   authority, no driver relay.
3. **Wake on delivery** — a target that cannot act *now* is made able to:
   idle → a turn is driven; detached → driven headless; unloaded → revived
   from disk, runner respawned (pool slot or cold spawn), then driven.
4. **Payload** — text, file references, text attachments, binary
   attachments, in one envelope.

---

## 2. What exists today

### 2.1 Grouping facts

| Fact | Where it lives | Persisted | Used for |
|---|---|---|---|
| `created_by` | `Session.created_by`, `SessionState.created_by` | yes (2.9) | `session.list` / `session.attach` visibility (`CommandRouter._sessions_visible_to`), ledger, telemetry |
| `cascade_driver_id` | `Session.cascade_driver_id`, `SessionState.cascade_driver_id`, `RunnerIdentity.cascade` | yes (2.10, and in the workspace index's `identity` section) | sibling addressing, cascade observers (`_cascade_clients`), budget pool, slot affinity |
| `sibling_name` | `Session.sibling_name`, `SessionState.sibling_name` | yes | the address `send_to_sibling` resolves; unique **within a cid** only |
| parent → child (subagent) | in-process, runtime-level; isolated sub-runners carry `parent_session_id` | no session record of its own for in-process children | `send_to_subagent`, `share_context` |

Neither `created_by` nor `cascade_driver_id` is currently a *messaging*
boundary in the general sense. Only the cid is, and only for loaded peers.

### 2.2 Delivery primitives

| Primitive | Tier | Target state | Payload | Wakes cold? | Sender identity |
|---|---|---|---|---|---|
| `send_to_sibling` → `SessionManager.deliver_sibling_message` | model tool (daemon-forwarded) | loaded only | text ≤ 8 KiB | **no** — `sibling_cold` by design (sibling design §11 Q2) | stamped by daemon from its table |
| `session.send` → `send_to_named_session` | client verb | loaded only | text | no — `sibling_cold` | operator (transport-authenticated) |
| `session.wake` → `wake_session` | client verb / HTTP ingress | any: revives cold via `SessionWorkspaceIndex` → `resume_session` | text + inline attachments | **yes** | none — any authenticated caller, any session id |
| `InjectPromptRequest` → `deliver_prompt_to_session` | client verb | loaded only | text + inline attachments (idle-only when attachments present) | no | caller-supplied `source_id` |
| `send_to_subagent` | model tool, in-process | own children | text | n/a | parent |

All of the loaded-target paths converge on **one decision** —
`shared.message_delivery` + `JaatoSession.offer_message`: the target
session answers atomically whether it is mid-turn (`queued`, drained at the
turn boundary by `_drain_child_messages`) or idle (`needs_turn`, so the
daemon drives a turn with `send_message_to_session`). The receipt
vocabulary (`accepted` / `queued` / `busy` / `terminated` / `no_session` /
`unreachable` / `not_confirmed`) is already the right one for a group
message and should be reused verbatim.

### 2.3 Revive and runner acquisition

`wake_session` on a cold id: `SessionWorkspaceIndex.resolve(id)` (daemon-
owned, durable under `~/.jaato/`, refuses ambiguous ids) → `resume_session`
→ `_load_session` (record, history, profile snapshot, rendered persona,
`created_by`, cid, sibling name all restored) → the load path re-provisions
a runner through the same `spawn_session_runner` a fresh session uses — a
pool slot when one fits the boundary key (#1033/#1100), otherwise a cold
spawn. So "no runner pool any more" is already covered: the pool is an
optimisation, not a precondition. `resume_session` then re-applies the
headless/API presentation so the permission layer runs policy-only rather
than blocking on an ASK nobody can answer.

Two lifetime facts matter for a woken, clientless target:

- it is **never** stopped by the orphan bound (`_ever_attached` is false for
  a cold revive — deliberate, #812), and
- once idle it is unloaded after `unload_grace_seconds` (60 s default,
  #1106), returning the slot.

That is exactly the shape a group message wants: wake, process, go back
to sleep.

### 2.4 The deferred-turn gate

`wake_session` does **not** drive a cold-revived session when the caller
passed a `cascade_driver_id` and no client is attached: it parks the wake
in `_pending_wakes`, emits `SessionWokenEvent` to the cid's cascade
observers, and waits for a client to attach (the transport drives the
pending wake after flushing the client's host tools). The reason is real —
host (client-provided) tools have no client to execute on — but the gate
is keyed on *whether the caller supplied a cid*, not on whether the target
actually declares client tools. For agent-to-agent traffic nobody will
attach, so deferral means the message is never processed.

Two latent defects in that mechanism, independent of this design:

- `_pending_wakes` is keyed by `session_id` with **one slot**: a second
  wake arriving before re-attach silently overwrites the first.
- `_pending_wakes` is **in-memory**: a daemon restart drops every deferred
  wake, while the binding that invited it survives on disk.

### 2.5 Payload carriage

- **Text** — every path.
- **Inline attachments** — `{mime_type, data: base64, display_name,
  attachment_id}` (`IPCClient._normalize_attachments`, #845). Reach the
  model as `inline_data` parts **only on the drive branch**; the queued
  branch folds a message into the running turn as text and has nowhere to
  put bytes, so an attachment-bearing delivery to a busy target is refused
  `busy` with nothing enqueued.
- **File references** — no such shape exists on any delivery verb. The
  nearest thing is `StageFilesRequest` (WS only, multi-frame binary,
  writes into the *connection's* selected workspace, caps 10 MB / 50 MB)
  and the legacy inline `staged_files` on `session.new`. Both put bytes on
  disk; neither tells a session about them.
- **Untrusted boundary** — `wrap_untrusted_content` on every inbound
  peer/wake body; `_wrap_wake_content` names each attachment *inside* the
  wrapper. The sibling path additionally refuses the
  `<permission_response>` / `<clarification_response>` grammar.

### 2.6 Cross-workspace reality

A cascade shares one workspace; a user's sessions do not. `created_by`
groups therefore span workspaces, and:

- persisted-session lookup is per workspace (`_get_persisted_sessions(
  workspace_path)`); the daemon-wide `list_sessions` unions the workspaces
  it knows plus the index's, which is enough to *find* a member but there
  is no owner → sessions index;
- a confined runner cannot read another workspace's files (AppArmor is
  keyed on `workspace_root`), so a path in workspace A is not a usable
  reference for a session in workspace B;
- `sibling_name` uniqueness is enforced per cid, so two of a user's
  sessions in different cascades (or none) may share a name.

---

## 3. Gap list

| # | Missing block | Why the existing piece does not cover it |
|---|---|---|
| G1 | **A group predicate** — `same_group(a, b)` over `{cid, created_by}` | sibling messaging is cid-only; `wake_session` checks nothing; `_sessions_visible_to` is a client-facing rule, not a session-to-session one |
| G2 | **Member resolution across workspaces** — by id or by name, live ∪ cold | `_resolve_sibling` is scoped to the cascade's one workspace; no owner index |
| G3 | **Wake on delivery for a peer** | `deliver_sibling_message` refuses cold on purpose; `wake_session` wakes but is unauthenticated per-session and defers when a cid is involved |
| G4 | **A durable per-session inbox** | runner-side queues die with the session; `_pending_wakes` is in-memory and single-slot; a busy target with an attachment is refused rather than held |
| G5 | **A payload envelope** with file references and a rule for bytes that cannot ride the queue | attachments only on the drive branch; no file-reference shape at all |
| G6 | **Cross-workspace file delivery** | a reference is only meaningful inside the target's confinement; nothing copies/stages *on behalf of a session* |
| G7 | **Model-facing surface** — `send_to_session` / `list_group_sessions` | `send_to_sibling` is the right shape but cid-scoped and cold-refusing |
| G8 | **Observability** — receipts, trace lines, a listing field for pending inbox | exists per path (`SIBLING_DELIVERY`, `DELIVERY_*`), not for a group verb |

---

## 4. Proposed design

### 4.1 Group membership (G1)

One module, `server/session_groups.py`, stdlib-only, with one predicate:

```
group_keys(session) -> frozenset[str]
    {"cid:" + cid}          if cascade_driver_id
    {"user:" + created_by}  if created_by
same_group(a, b) := group_keys(a) & group_keys(b) != ∅
```

Rules, each attached to a way it goes wrong:

- **`None` never matches `None`.** Two anonymous IPC sessions must not form
  a daemon-wide group. Same posture as refusing `created_by=""` at the
  ticket door (#1074): absence of identity is not an identity.
- **`created_by` is already app-qualified** (`app:user`), so a user-group
  never crosses an application boundary on a shared WS daemon.
- **Membership is computed from the target's persisted record, never from
  the payload.** The sender is read from the daemon's own table (the
  `deliver_sibling_message` rule); the target is read from the loaded
  `Session` or, cold, from its record via the workspace index.
- **A group is not declared, it is derived** — from the two facts the
  requirement names. An explicit `group_id` is a later addition if a
  deployment wants sessions of different users or cascades to talk; it
  would be a third key in the same set, not a new mechanism.

### 4.2 Addressing and resolution (G2)

Address by **session id first**, name second:

- `session_id` is always accepted. The index resolves its workspace; the
  record supplies cid / owner for the membership check.
- `sibling_name` is accepted **within a cid group only**, where uniqueness
  is already enforced. Within a user group a name is advisory (it may
  collide across cascades), so a name-addressed send that resolves to more
  than one member is refused `ambiguous` with the candidates' ids — never
  delivered to the first match, which is the silent-misdelivery shape the
  sibling design refused.

Resolution needs a daemon-owned **owner → session ids** view. The cheapest
honest source is the `SessionWorkspaceIndex`, which already carries a
per-id `identity` section with the cascade: add `created_by` and
`sibling_name` to what `_save_session` records there. That keeps one index
rather than a second file that can disagree with it, and a record written
by an older daemon simply has no owner (so it joins no user group — the
safe direction).

`list_group_sessions(viewer)` = live ∪ cold rows for every key in the
viewer's group set, each row `{session_id, sibling_name, group_keys,
status: active|idle|cold, workspace_path, profile_name, description}`.
`description` stays untrusted (it is the peer's own text). No self row.

### 4.3 Delivery with wake (G3)

One daemon-side method composes what exists:

```
deliver_group_message(sender_id, target, envelope) -> receipt

  1. resolve target (4.2); refuse unless same_group(sender, target)
  2. grammar / size / cap checks (reuse the sibling ones)
  3. wrap body as untrusted content, source = "peer:<sender address>"
  4. if target loaded:
         status = deliver_prompt_to_session(target, wrapped,
                    source_type=SIBLING, attachments=inline_parts)
     else:
         resume_session(target, workspace from index)   # respawns runner
         status = send_message_to_session(target, wrapped, attachments)
         receipt.woken = True
  5. any non-DELIVERED status with a retry-safe cause -> spool (4.4)
```

Decisions inside it:

- **No deferred-turn gate for a peer sender.** A peer is not a client and
  none will attach. The woken target runs headless (policy-only
  permissions, no host tools), which is the state `resume_session` already
  produces. If the target's record shows it *had* client tools, the
  receipt says so (`headless: true, client_tools_unavailable: [...]`) and
  delivery still proceeds; refusing would make the message undeliverable
  by construction.
- **Tier is `SIBLING`** (idle-only). A peer coordinates, it does not
  steer; keeping it out of the high-priority tiers is what makes a
  ping-pong bounded by turn boundaries.
- **Cost lands on the target**, and through `_accumulate_cascade_budget`
  on the shared pool for cid groups. A cold target whose profile declares
  no `budget_control` is woken with a WARNING naming the profile — #812
  and #947 are exactly a clientless session with no ceiling, and a wake
  verb is how one gets created.
- **`event_id` dedup** is reused from `wake_session` (the `_wake_seen_
  event_ids` claim/release protocol), so an at-least-once sender can
  retry a `not_confirmed` receipt safely.
- **Caps** generalise from per-cid to per-group-key: message size 8 KiB
  of *text* (attachments are bounded separately, 4.5), pending cap per
  target, exchange cap per group. `budget_control` remains the real
  terminator.

`send_to_sibling` / `session.send` are left exactly as they are: they
promise not to wake, and callers rely on that.

### 4.4 A durable inbox (G4)

Every message that is *accepted for delivery* but cannot be handed to a
running turn right now is written to the target's inbox **before** the
receipt is returned:

```
<workspace>/.jaato/sessions/<id>.inbox/<message_id>.json   # envelope
<workspace>/.jaato/sessions/<id>.inbox/<message_id>/       # spooled bytes
```

Under the target's own `.jaato/sessions/` because that is the directory a
revive already reads, it is workspace-state (never committed — the
gitignore block already excludes `sessions/`), and it lands inside the
target's confinement so the target's own file tools can read what was
spooled for it.

States and drains:

| When | Who drains |
|---|---|
| target loaded and idle | delivered immediately (drive); nothing spooled |
| target busy, text-only | `queued` runner-side as today **and** spooled as `pending`; the turn-end drain marks it `delivered`. The spool copy is what survives an unload between queue and drain — the documented loss today |
| target busy, with bytes | spooled `pending`; the end-of-turn hook drives the next turn with it, so bytes ride the drive branch. No more `busy` refusal for attachments |
| target cold, revive fails transiently | spooled `pending`; the lifetime watchdog sweep retries revive+drive with backoff and gives up at a bounded age (the wake-binding TTL is the precedent) |
| target revived by anything (attach, wake, resume) | `_load_session` drains `pending` on the first drive |

One inbox, many messages — which is also the fix for `_pending_wakes`'
single slot: a deferred wake becomes an inbox entry with `defer_until_
client: true`, and `drive_pending_wake` becomes "drain the inbox". The
in-memory dict goes away rather than growing a sibling.

`message_id` is minted daemon-side (uuid); the sender's `event_id` is the
idempotency key. A redelivered `event_id` finds its inbox entry and
answers `duplicate`.

### 4.5 The envelope, and where bytes go (G5, G6)

```json
{
  "message_id": "…",             "event_id": "…",
  "from": {"session_id": "…", "sibling_name": "…", "group_key": "cid:…"},
  "to":   {"session_id": "…"},
  "ts":   1758000000.0,          "reply_to": null,
  "text": "…",
  "attachments": [ {"mime_type": "audio/wav", "data": "<b64>",
                    "display_name": "note.wav", "attachment_id": "sha256:…"} ],
  "text_attachments": [ {"display_name": "diff.patch", "mime_type": "text/x-diff",
                         "text": "…"} ],
  "file_refs": [ {"path": "reports/q3.md", "workspace": "/srv/ws/a",
                  "sha256": "…", "size": 18321, "mime_type": "text/markdown"} ]
}
```

Four payload kinds, one rule each:

| Kind | On the drive branch | On the queue / spool branch |
|---|---|---|
| `text` | the untrusted-wrapped body | same |
| `text_attachments` | inlined under a fenced block inside the wrapper, up to a cap (32 KiB total); beyond it, spooled and delivered as a `file_ref` | same |
| `attachments` (binary) | `inline_data` parts (existing #845 path), manifest inside the wrapper | spooled to `<inbox>/<message_id>/<display_name>`, delivered as a `file_ref`, and re-offered as inline parts when the next drive happens |
| `file_refs` | see below | same |

**A file reference is a claim the daemon verifies, then re-issues in the
target's terms.** The sender names `(workspace, path)`; the daemon checks
the sender's session actually lives in that workspace (a session may not
reference files it could not itself read), that the path is inside it, and
that it exists. Then:

- **same workspace** (every cid group; some user groups) → delivered as
  the relative path plus digest and size, in a `[files referenced by this
  message]` manifest inside the untrusted wrapper. The target reads it
  with its own tools under its own permission policy. Nothing is copied.
- **different workspace** → the daemon **copies** the file into the
  target's inbox directory (bounded by the staging caps, 10 MB per file /
  50 MB per message, `_materialize_staged_files` is the write path to
  reuse, digest re-verified after copy) and delivers a `file_ref` under
  `.jaato/sessions/<id>.inbox/<message_id>/…`. A file over the cap is
  refused by name in the receipt (`file_too_large`), never silently
  dropped from a message reported as delivered.

Copy rather than symlink or bind-mount: a link out of the workspace is
exactly what `_resolve_under_root` and the AppArmor profile refuse, and
what the notebook containment treats as an escape.

**The boundary is stated beside every kind.** The wrapper carries the
attachment manifest (#845's rule), the file manifest, and the sender
address, so a spooled PDF and a typed sentence are weighed the same way.

### 4.6 Surfaces (G7, G8)

| Surface | Shape |
|---|---|
| model tool `send_to_session` | `{target: <id or name>, text, file_refs?, attachments?, reply_to?}` → receipt. `TRAIT_UNTRUSTED_CONTENT` **not** set (the receipt is framework text); permission-gated naming the target, as `send_to_sibling` is. Visible only when the session is in at least one group (`is_tool_visible`, the telepathy precedent) |
| model tool `list_group_sessions` | the roster of 4.2; auto-approved (read-only); rows carry untrusted descriptions |
| client verb `session.message` (protocol 1.18) | the same envelope from an SDK/operator, answered by one typed `SessionMessageResultEvent` carrying the receipt — a request/result pair with `request_id`, not a `SystemMessageEvent` string. Missing-verb rule: the SDK refuses below 1.18, because an older daemon ignores the command and "delivered" would describe nothing |
| receipt | `{status, message_id, target_session_id, woken: bool, spooled: bool, headless: bool, files: [{name, disposition: referenced|copied|refused, reason?}]}` with `status` from `shared.message_delivery` plus `spooled` (accepted for later delivery) and `ambiguous` |
| trace | one `GROUP_DELIVERY:` line per attempt in the application trace, beside `SIBLING_DELIVERY` / `DELIVERY_*`, naming sender, target, group key, branch (drive/queue/spool/wake), bytes and file dispositions |
| listing | `session.list` rows gain `inbox_pending: <n>` — the #1138 shape: a fact a client attached elsewhere cannot otherwise learn |
| events | `SessionWokenEvent` is reused for a peer-triggered wake, routed to cid observers as today **and** to the owner's connections for user groups (a `broadcast_event` filtered by `created_by`), so a UI can show "session B woke to handle a message from A" |

### 4.7 Authorization, in one place

| Check | Where | Answer when it fails |
|---|---|---|
| sender identity | daemon table, never the payload | n/a — cannot be forged |
| same group | `same_group` on both records | `refused: not in a common group` — the same wording whether the target exists or not, so the verb is not an existence oracle across groups |
| grammar | `_sibling_grammar_violation` | `refused` |
| file reference within the sender's own workspace | `_resolve_under_root` shape | `refused: file_ref outside sender workspace` |
| caps | per group key | `refused` with the cap named |
| target terminal | `_terminal_reason` | `terminated` — never woken; a session that ended on an error or an exhausted budget is not brought back by a peer |

`session.wake` itself stays as it is (any authenticated caller, any id):
its callers are ingress shims and operators, and adding a membership check
there is a behaviour change for them. `session.message` is where a *peer*
speaks, and it is checked.

---

## 5. Rollout

| Phase | Delivers | New code, roughly |
|---|---|---|
| 1 | `session_groups.py`; index carries owner + name; `deliver_group_message` over the existing drive/queue/wake primitives (text + inline attachments, wake on cold, no spool); `send_to_session` + `list_group_sessions`; `session.message` + result event; `GROUP_DELIVERY` trace | one module, one method on `SessionManager`, two executors in the subagent plugin, one router handler, one SDK method per SDK |
| 2 | the inbox: spool on busy/failed-revive, drain on load and turn end, watchdog retry; `_pending_wakes` folded into it; `inbox_pending` on the listing | inbox module + three drain hooks |
| 3 | `file_refs` and `text_attachments`; cross-workspace copy through the staging write path; receipt file dispositions | envelope validation + copy helper |

Phase 1 is what the requirement asks for and is almost entirely
composition. Phase 2 is what makes "it will be processed" a guarantee
rather than a receipt. Phase 3 is the payload width.

---

## 6. Open decisions

1. **Wake unconditionally, or per profile?** Proposed: unconditional for
   `session.message`, with the no-ceiling WARNING. Alternative: a profile
   key `peer_wake: false` for a session that must only be woken by an
   operator. Cheap to add; not needed for the first cut.
2. **Should a user group cross workspaces at all?** The requirement
   implies yes. Cost: every cross-workspace file reference is a copy, and
   a session in workspace A learns that B exists. If a deployment wants
   isolation between a user's own workspaces, the group predicate is the
   one place to narrow it (`user:` key + workspace).
3. **Headless drive when the target had client tools.** Proposed: deliver
   anyway, say so in the receipt. Alternative: spool with
   `defer_until_client` and notify the owner, as the cid path does today.
   Both are one branch in step 4 of 4.3; the receipt vocabulary covers
   either.
4. **Reply channel.** Kept fire-and-forget with `reply_to` for
   correlation. A blocking `ask_session` would reintroduce the deadlock
   the sibling design refused; if it is ever wanted it should be a
   clarification-shaped request with a timeout, not a variant of this
   verb.
5. **In-process subagents.** They have no session record and are reached
   only through their parent. Out of scope here: a group is a set of
   daemon sessions. An isolated sub-runner *does* have a record and joins
   its parent's groups by inheriting `created_by` (already true) and cid.

---

## 7. What this does not change

- `send_to_sibling`, `session.send`, `session.wake`, `inject_prompt` keep
  their contracts, including "cold peers are not woken" on the first two.
- The queue-or-drive decision stays in `shared.message_delivery` and
  `offer_message`; the new method calls it, it does not copy it.
- Untrusted-content marking, the sibling grammar refusal, daemon-stamped
  sender identity, and the `budget_control` terminator all apply unchanged.
- Workspace resolution for a cold target stays server-owned (the index),
  never caller-supplied.
