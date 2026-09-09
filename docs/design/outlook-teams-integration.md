# Outlook and Teams — integration brainstorm

> Status: **brainstorm**, not a plan of record. Nothing here is committed to a
> release. The purpose is to name the *seams the tree already has*, the gaps
> that are real, and the order in which the work stops being speculative.

## 0. The one distinction that organises everything

"Integrate Outlook and Teams" is three unrelated projects wearing one name.
They touch different parts of the framework, have different security stories,
and can be shipped in any order:

| # | Direction | The agent is… | jaato seam |
|---|-----------|---------------|-----------|
| **A** | **Outbound** — read mail, send mail, find a meeting slot, post to a channel | an HTTP **caller** | tool plugin / `service_connector` / `mcp` |
| **B** | **Inbound** — a mail arrives, someone @mentions the bot | an **event consumer** | `webhook` plugin + `session.wake` / `inject_prompt` |
| **C** | **Surface** — Teams *is* the UI, the way the TUI is | a **conversant** | a jaato **client** (WS), `ClientType.CHAT`, permission/clarification channels |

Most "add Teams support" tickets mean A, discover they wanted B, and only
notice C exists once someone asks where the approval button lives. C is the
one that is distinctive to this framework and the one nobody else's SDK makes
easy — see §5.

Everything on both sides talks to **Microsoft Graph** (`https://graph.microsoft.com/v1.0`).
Outlook mail, Outlook calendar, Teams chats and Teams channel messages are all
Graph resources; there is no separate Outlook API worth using in 2026. The one
exception is C, where a Teams *bot* is an Azure Bot Service registration
speaking the Bot Framework activity protocol, not Graph.

---

## 1. Direction A — the agent as caller

### A0. Do nothing: point `.mcp.json` at a Graph MCP server

```json
{"mcpServers": {"m365": {"type": "stdio", "command": "npx",
                          "args": ["-y", "<some-graph-mcp-server>"]}}}
```

Zero framework code. This is the correct **first** move — it answers "is a
mail-reading agent actually useful to us" for the price of an afternoon,
before anyone argues about plugin naming.

What it costs: auth lives inside somebody else's process, so the token is
outside jaato's credential story entirely; `scrub_secret_env` (#863) applies to
the MCP stdio spawn, which means the server's *own* `env` grant in `.mcp.json`
is the only supported way to hand it a secret — a profile's `env:` map is
explicitly **not** a grant. Tool naming is theirs, so the model sees whatever
verbs the server chose, hashed to `t_<8 hex>` on the wire (#873). And per-tool
permission gating works, but the granularity is the server author's.

### A1. Almost nothing: `service_connector` + Graph's OpenAPI

`service_connector` already does discovery → endpoint listing → validated
call, with `oauth2_client` (client-credentials) auth reading its secret from
session-scoped env. **App-only Graph access works today with zero framework
code**:

```yaml
plugins: [service_connector]
plugin_configs:
  service_connector: {}
env:
  GRAPH_CLIENT_ID: "..."
  GRAPH_CLIENT_SECRET: "pass://work/graph/client-secret"
```

```
configure_service_auth(service="graph", type="oauth2_client",
  token_url="https://login.microsoftonline.com/${TENANT}/oauth2/v2.0/token",
  client_id_env="GRAPH_CLIENT_ID", client_secret_env="GRAPH_CLIENT_SECRET",
  scope="https://graph.microsoft.com/.default")
call_service(service="graph", method="GET", path="/users/{id}/messages")
```

Two real frictions:

- **The spec is enormous.** The published `graph-v1.0` OpenAPI description
  (microsoft/msgraph-metadata) is tens of megabytes and thousands of
  endpoints. `discover_service` will parse it; `list_endpoints` will drown the
  model. The fix is not framework work — it is checking a **trimmed** spec
  (mail + calendar + chats + teams, a few dozen operations) into
  `.jaato/services/`, which the schema store already supports.
- **Delegated auth is missing.** `AuthManager` supports client-credentials but
  not authorization-code + refresh, and *most* useful Outlook work is
  delegated ("my mail", "my calendar"). App-only `Mail.Read` is a tenant-wide
  grant — see §6. This is the gap that motivates A2.

`call_service` already carries `TRAIT_GREPPABLE_CONTENT`, so `result_grep` can
shrink a 400 KB HTML mail body before it reaches the context. That matters more
here than anywhere else in the tree.

### A2. A first-class `m365` tool plugin

The argument for it is the same argument that made `web_search` a plugin rather
than "just use `cli` and `curl`": the model should not have to know that "find
me a slot next Tuesday with Anna" is `POST /me/findMeetingTimes` with a
`meetingTimeSuggestion` body, nor that pagination is `@odata.nextLink`, nor
that a message body arrives as HTML that must be reduced to text before it is
worth a token.

Sketch of the tool surface (all `discoverability="discoverable"`):

| Tool | Notes |
|------|-------|
| `mail_search` | KQL / `$search` + `$filter`, returns id + from + subject + snippet, **never** full bodies |
| `mail_read` | one message, body reduced to text, attachments listed not inlined |
| `mail_send` / `mail_reply` | write — always permission-gated |
| `mail_attachment_fetch` | returns bytes as an `Attachment`; a PDF goes straight to a `pdf_input` model (#830 machinery) |
| `calendar_list` / `calendar_find_time` / `calendar_create_event` | `findMeetingTimes` is the whole reason to wrap this |
| `teams_list_channels` / `teams_post_message` / `teams_read_thread` | Graph `chatMessage` |

**One plugin, not three.** `parse_plugin_entry` already gives per-session
least privilege at *tool* granularity, so a scheduler profile writes

```yaml
plugins: [m365([calendar_list, calendar_find_time, calendar_create_event])]
```

and never sees `mail_send` in its wire body or grammar surface at all. Three
plugins (`outlook_mail`, `outlook_calendar`, `teams`) would buy nothing the
allow-list does not already buy, and would triple the auth wiring.

### A3. The auth plugin both A1 and A2 need

`github_auth` is the exact precedent: device-code OAuth, token persisted to
`~/.jaato/`, a `*-auth login/logout/status` client command. An
`microsoft_auth` (or `entra_auth`) plugin doing the same for Graph would serve
`service_connector` *and* `m365` *and* — if anyone wants it — a future
Graph-backed model provider.

**Do not add `msal`.** `azure_openai` already pulls `azure-identity` for
`auth: aad`, and `DeviceCodeCredential` / `InteractiveBrowserCredential` /
`ManagedIdentityCredential` cover every deployment shape this needs, with token
caching and refresh already solved. The `[azure-openai]` extra becomes a
shared `[microsoft]` extra.

The auth plugin should hold **both** postures explicitly, because they are not
interchangeable:

- **delegated** (device code / interactive) — acts as a person, scopes are
  consent-gated, `/me` works, and the blast radius is that one mailbox;
- **app-only** (client credentials / managed identity) — acts as the tenant,
  no `/me`, and `Mail.Read` means *every mailbox in the organisation* unless an
  Application Access Policy narrows it. §6.

---

## 2. Direction B — the agent as event consumer

Graph delivers events as **change notifications**: you `POST /subscriptions`
naming a resource, a `notificationUrl`, a `clientState` and an
`expirationDateTime`; Graph then POSTs batches of notifications to that URL.
The `webhook` plugin is the right home, and it needs three specific things it
does not have.

### B1. The validation handshake (blocking)

When a subscription is created (and on renewal), Graph immediately POSTs to
`notificationUrl` with a `?validationToken=<opaque>` query parameter and
expects the token echoed back as the **body**, `Content-Type: text/plain`,
status 200, within seconds. `http_server.py`'s `do_POST` unconditionally
answers `{"status": "accepted"}` as JSON, so **subscription creation fails
today**. This is a small, contained route option:

```json
"routes": {"graph": {"path": "/webhook/graph",
                     "validation_token_param": "validationToken"}}
```

### B2. `clientState` is not an HMAC header

`routes.verify_signature` only knows `hmac-sha256` over the body with the
signature in a *header*, and `parse_webhook_request` fails closed when a route
declares half of that pair. Graph's basic (non-encrypted) notifications carry
no signature at all — the shared secret is `clientState`, echoed **inside each
notification object in the JSON body**.

Two honest options, and they should both exist:

- a `body_field` verification mode (`clientState` compared with
  `hmac.compare_digest` per notification, mismatches dropped and logged);
- lean on `allowed_ips` + TLS, which the plugin already has, since Microsoft
  publishes the Graph notification egress ranges.

Note what neither gives you: `clientState` is a bearer secret in a request
body, so it authenticates the *sender*, not the *payload*. Treat the
notification as a **hint that something changed**, then re-read the resource
from Graph over an authenticated call. That is Microsoft's own advice and it
happens to be the shape that survives a forged POST.

### B3. Subscriptions expire, and quickly

Mail and calendar subscriptions live on the order of days; **Teams message
subscriptions are on the order of an hour**. Something must renew them or the
integration silently goes deaf — the worst failure mode available, because
nothing errors.

Ownership is a genuine design question:

- a renewal thread inside the plugin (like the `interactive_shell` reaper) —
  simple, but the plugin now holds a Graph credential and does I/O the model
  did not ask for;
- a long-running agent session with the `background` plugin that renews on a
  timer — no framework code, fully visible in the ledger, but it burns a
  session and dies when the session does;
- renewal as a `TRAIT_SLOT_SCOPED` concern — wrong, slots are per-cascade.

Leaning toward the first, with the credential coming from the auth plugin of
§A3 rather than the webhook plugin's own config, and a **loud** WARNING (plus a
`webhook_status` field) when a subscription is within one renewal window of
expiry.

### B4. What the event *does*

The event has to become a turn. Both existing verbs work, and #845 already
widened them to carry bytes:

```
notification → (re-read resource) → session.wake(text=..., attachments=[...])
```

An Outlook mail is the canonical case: subject and sender as text, the PDF the
model actually needs as an attachment, `_wrap_wake_content` stating the
untrusted boundary *beside* the bytes. Note the #845 rule that bites here — an
attachment-bearing **inject** is idle-only, so a busy target answers `BUSY`
with nothing enqueued. A mail-driven agent wants a queue in front of it, and
the framework does not supply one; that is the integration's job, not the
framework's.

---

## 3. Direction C — Teams as a jaato client

This is the interesting one. The server already speaks WebSocket with bearer
auth and `set_client_user()`; a Teams bot is then a **client**, exactly like
`jaato-tui/rich_client.py`, and everything the client protocol already carries
lights up:

| jaato concept | Teams rendering |
|---------------|-----------------|
| `PresentationContext(client_type=CHAT)` | already a first-class `ClientType`; `supports_expandable_content=True` and let the client collapse overflow |
| `CommunicationStyle.CONVERSATIONAL` | short frequent messages — literally what the enum was written for |
| `PermissionRequestedEvent` | **Adaptive Card** with Approve / Deny / Always buttons |
| `ChannelResponse.approver` (#859) | the Entra identity of whoever clicked — already plumbed to `PermissionResolvedEvent.approver` and the ledger `permission-check` record |
| clarification `qa_pairs` | Adaptive Card `Input.ChoiceSet`; the plugin emits structure precisely so the client picks the presentation |
| `PlanUpdatedEvent` / todo | a card edited in place |
| `ToolOutputEvent` media chunks | audio/images posted as attachments; `renderable_media` declares what Teams can play |

The approver story is the strongest argument for doing C at all. #859 built the
plumbing for "who approved this" and left it to be filled in by an external
approval system; a Teams Adaptive Card is that system, with SSO-verified
identity, in the place where the humans already are. An audit trail that reads
"`cli: rm -rf build/` approved by dani@… from the #eng-agents channel" is
worth more than the tool call.

**Approval without the whole client.** The cheap half of C is already
implemented: `WebhookChannel` posts permission requests to an HTTP endpoint and
reads back `{decision, reason, approver}`. A ~200-line Teams bot that turns
that POST into a card and the button-press into the response gives you
Teams-mediated approvals with **no framework change at all**. If only one thing
on this page gets built, this is the one with the best ratio.

Open questions for the full client: streaming (Teams bots can update a message
in place, but the cadence limits need measuring before promising token-by-token),
threading (one Teams thread ↔ one `session_id` is the obvious mapping and
survives `session.wake`), and whether the bot process holds one WS connection
multiplexing all users or one per conversation — the latter is cleaner for
`created_by` (#859) and the per-session `runtime_limits`, and the WS auth
contract already supports a token per connection.

---

## 4. What this composes into

The interesting product is not any single direction — it is B → A → C in one
session:

> A mail lands in a shared mailbox (**B**). The agent reads the thread and the
> attached PDF (**A**), drafts a reply, and asks in the Teams channel whether to
> send it (**C**). Someone taps Approve; the card records who. The agent sends
> the mail (**A**) and posts the thread link back.

Every arrow in that paragraph is a seam that exists. What is missing is the
validation handshake, delegated auth, and the card renderer.

---

## 5. Staging

| Phase | Work | Ships without |
|-------|------|---------------|
| **0** | `.mcp.json` Graph MCP server, or `service_connector` + trimmed spec + client credentials | any framework change |
| **1** | Teams approval bot behind `WebhookChannel` | any framework change |
| **2** | `microsoft_auth` plugin (device code + refresh via `azure-identity`) | — |
| **3** | `m365` tool plugin (mail / calendar / teams), tool-allow-listable | — |
| **4** | `webhook`: `validation_token_param`, `clientState` body verification, subscription lifecycle | — |
| **5** | Teams client: WS + Adaptive Cards for permission, clarification, plan | — |

Phases 0 and 1 are deliberately first *because they require no framework
change*: they are how we find out whether phases 2–5 are worth their
maintenance cost, and they are individually useful if the answer is no.

---

## 6. Cross-cutting concerns

**Least privilege is a Graph problem before it is a jaato problem.**
App-only `Mail.Read` reads every mailbox in the tenant. The mitigation is an
**Application Access Policy** scoping the app registration to a mail-enabled
security group — an Exchange admin step, out-of-band, and it must be in the
deployment doc, not discovered later. Delegated auth avoids the question
entirely and should be the default posture for anything acting on behalf of a
person.

**Mail is the canonical untrusted content.** A mail body is attacker-authored
text arriving inside the agent's context — indirect prompt injection with a
delivery guarantee. Two things follow: the framework's security boundary layer
must stay on (`suppress_base_instructions` keeps `security` unless *named*, and
an m365 profile must never name it), and no `m365` write tool should be
auto-approved on a mail-triggered session. "Summarise this thread" is a read;
"reply to it" is a decision.

**Secrets.** Graph tokens are `*_TOKEN` / `*_SECRET` shaped, so the default
`scrub_secret_env` set (#863) already keeps them out of model-driven `cli`,
`interactive_shell` and `mcp` subprocesses. Do not exempt them. Client secrets
belong in a profile's `env:` as `pass://` / `vault://`, which stays unresolved
on disk.

**Throttling.** Graph answers 429 with `Retry-After` and means it. Whatever
does the calling must honour the header rather than applying the framework's
generic backoff, and a fan-out of eight parallel `mail_read` calls is exactly
the shape that trips it — `runtime_limits.max_parallel_tools: 2` (#862) is the
existing knob and an m365-heavy profile should set it.

**Result size.** Mail bodies and Teams threads are the largest tool results in
any of this. `TRAIT_GREPPABLE_CONTENT` + `result_grep` for the structured path,
HTML→text reduction inside the plugin, and never returning a body from a
*search* tool.

**Sovereign clouds.** Graph has national-cloud endpoints
(`graph.microsoft.us`, `microsoftgraph.chinacloudapi.cn`, …) with different
login authorities. A `base_url` / `authority` knob costs nothing to add up
front and is very annoying to retrofit — the same lesson `minimax`'s `.cn`
platform taught.

## 7. Verify before building

Facts this document leans on that are worth re-checking against current Graph
documentation before any of it is scheduled, because they move:

- exact maximum subscription lifetimes per resource type (mail/calendar vs
  Teams `chatMessage`), and whether encrypted-resource-data subscriptions
  extend them;
- whether the validation handshake still requires a plain-text echo and its
  current deadline;
- current Teams bot message-update / streaming cadence limits, which decide
  whether C can stream at all;
- whether the published Graph OpenAPI description is still the practical source
  for a trimmed `service_connector` spec.
