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
| **A** | **Outbound** — read mail, send mail, find a meeting slot, post to a channel | an HTTP **caller** | `mcp` / `service_connector` / an **out-of-tree** tool plugin |
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

### A2. An **out-of-tree** `m365` plugin

The tool-plugin idea is right; putting it in `shared/plugins/` is not. It
should ship as its own distribution, `jaato-m365`, declaring
`[project.entry-points."jaato.plugins"]` — the path #684 built the trust policy
for, and which today only `jaato-premium` (`profile_tools`, `session_ops`,
`auto_steering`) exercises.

Five reasons, in decreasing order of how much they'd survive an argument:

1. **Nothing in it is framework machinery.** It is a vendor connector. The
   line already exists in the tree: `web_search` and `web_fetch` are generic
   capabilities, and there is not one per-vendor plugin in `shared/plugins/`.
   `m365` would be the first, and the precedent is worse than the feature.
2. **Cadence.** Graph changes on Microsoft's schedule. In-tree, a Graph
   deprecation becomes a jaato-server release; out-of-tree it is a patch
   release of a package that only M365 shops install.
3. **Dependency weight.** `azure-identity` (and whatever HTTP stack) becomes a
   `jaato-server[m365]` extra that every contributor's resolver has to think
   about, to serve a minority of deployments. Out-of-tree it is `pip install
   jaato-m365` and it is their problem.
4. **Tenant policy is not framework opinion.** Which mailboxes, which scopes,
   which national cloud, whether app-only is even permitted — all deployment
   facts. A plugin whose correct configuration differs per *organisation*
   wants to be versioned by that organisation.
5. **It is the proof the extension point works.** The entry-point path has one
   consumer and it is not public. Building `m365` against it as a worked
   example is how the gaps get found — and it already found two (§A4).

#### The contract, precisely

Verified against `registry.py` / `entry_point_trust.py` at the current tree,
because most of this is discoverable only by reading them:

```toml
# pyproject.toml
[project]
name = "jaato-m365"
dependencies = ["jaato-sdk", "azure-identity", "httpx"]

[project.entry-points."jaato.plugins"]
m365 = "jaato_m365.plugin:create_plugin"
```

```
src/jaato_m365/
  __init__.py     # PLUGIN_KIND, PLUGIN_TIER, create_plugin re-export
  plugin.py       # M365Plugin — the ToolPlugin protocol
  auth.py         # azure-identity credential chain + token cache
  graph.py        # thin Graph client: paging, $select, 429/Retry-After
  reduce.py       # HTML → text, body truncation
  tools/          # mail.py, calendar.py, teams.py
```

| Fact | Consequence for the author |
|------|---------------------------|
| the entry-point **group** decides the kind (`jaato.plugins` → `"tool"`) | `PLUGIN_KIND` is only needed by the directory scan; declare it anyway for symmetry |
| `create_plugin = ep.load()`, then `plugin = create_plugin()` | **the factory takes no arguments** — config arrives at `initialize(config)`, never at construction |
| the protocol lives in **`jaato_sdk.plugins.base`**, `ToolSchema` in `jaato_sdk.plugins.model_provider.types` | the hard dependency is **jaato-sdk**, not jaato-server. An out-of-tree plugin never imports the server |
| `_protocol_gap` warns by name on a missing method (since #171) | implement all nine: `name`, `get_tool_schemas`, `get_executors`, `initialize`, `shutdown`, `reset_for_next_session`, `get_system_instructions`, `get_auto_approved_tools`, `get_user_commands` |
| the trust gate runs **twice** — on `ep.name` before `ep.load()`, and on `plugin.name` after | keep them equal; a mismatch is treated as an attempt to slip a reserved name past the pre-load check |
| `m365` is not a built-in module name | decision is `external`: no warning, no `JAATO_PLUGIN_ALLOW_SHADOW` needed. Naming it `mcp` or `web_fetch` would be refused, and `permission` / `cli` / `file_edit` / `sandbox_manager` / `interactive_shell` are refused *even with* the opt-in |
| `_augment_plugin_config` `setdefault`s `workspace_path`, `config_root`, `session_id`, `agent_name` | the token cache belongs under the injected `config_root`, not a hardcoded `~/.jaato` |
| discovery registers; only `expose_all(requested_plugins=…)` initialises | installing the distribution does **not** activate it — a profile must still say `plugins: [m365]`. Worth knowing before an auditor asks |

#### The trap worth the whole exercise

`PLUGIN_TIER` is not optional, and getting it wrong fails in the worst
available way.

```python
# src/jaato_m365/__init__.py
PLUGIN_KIND = "tool"
PLUGIN_TIER = "runner"        # ← omit this and the plugin disappears
```

`_lookup_module_tier` reads `PLUGIN_TIER` from the factory's module *or its
parent package*, and `_tier_filter_matches` returns **False** for an
unannotated plugin whenever a filter is set:

```python
if tier_filter is None:
    return True          # unannotated passes
if plugin_tier is None:
    return False         # filter set, no annotation → skipped
```

No filter is set in an unfiltered/in-process context — which is exactly what a
plugin author's dev loop looks like. A filter *is* set on the daemon+runner
path, which is production. So an unannotated `jaato-m365` works perfectly on
the author's machine and is silently absent in the deployment, with the skip
recorded only in a debug `_trace`. In-tree this is caught by
`test_plugin_tier_partition`; **out-of-tree nothing catches it**, which is why
an example that documents it is worth more than one that merely works.

#### Auth: simpler out-of-tree, not harder

The in-tree sketch wanted a separate `microsoft_auth` plugin mirroring
`github_auth`. Out-of-tree that split stops paying for itself:
`TRAIT_AUTH_PROVIDER` is for *model-provider* auth (it demands a
`provider_name`), so it does not apply here, and `get_user_commands()` is on
the same `ToolPlugin` protocol the connector already implements. One
distribution, one plugin, one command surface:

```
m365-auth login    # DeviceCodeCredential → token cache under config_root
m365-auth status   # tenant, identity, scopes, expiry
m365-auth logout
```

`azure-identity` supplies the whole chain — `DeviceCodeCredential` for
delegated, `ClientSecretCredential` / `ManagedIdentityCredential` for app-only
— with refresh already solved, and `azure_openai`'s `auth: aad` proves the
dependency is already acceptable in this ecosystem. Do not add `msal`.

Per the `get_auto_approved_tools()` docstring, user commands must be listed
there or they raise permission prompts for something the human invoked
directly. **Nothing else goes in that list.** A read tool here is not
harmless: `mail_search` is the injection vector, not the write path, and
`m365` is the one plugin where "read-only, therefore auto-approve" is wrong.

#### Tool surface

All `discoverability="discoverable"`, so an unused `m365` costs no context.

| Tool | Notes |
|------|-------|
| `mail_search` | `$search` / `$filter`; returns id + from + subject + snippet, **never** bodies |
| `mail_read` | one message, HTML reduced to text, attachments listed not inlined; `TRAIT_GREPPABLE_CONTENT` |
| `mail_send` / `mail_reply` | write — permission-gated, never whitelisted by default |
| `mail_attachment_fetch` | returns an `Attachment`; a PDF goes straight to a `pdf_input` model |
| `calendar_list` / `calendar_find_time` / `calendar_create_event` | `findMeetingTimes` is the whole reason to wrap this |
| `teams_list_channels` / `teams_read_thread` / `teams_post_message` | `TRAIT_GREPPABLE_CONTENT` on the read |

**One plugin, not three.** `parse_plugin_entry` already allow-lists at *tool*
granularity, so least privilege is a profile concern:

```yaml
plugins: [m365([calendar_list, calendar_find_time, calendar_create_event])]
```

— and `mail_send` is absent from that session's wire body and grammar surface
entirely. Splitting into `outlook_mail` / `outlook_calendar` / `teams` would
buy nothing the allow-list does not already buy, and would triple the auth
wiring.

Implement the optional `get_config_schema()` too: it costs ten lines and makes
`jaato-scaffold validate` able to check an `m365` profile block it has never
heard of.

#### Operator posture

```bash
pip install jaato-m365
export JAATO_PLUGIN_ENTRY_POINT_ALLOWLIST=jaato-m365   # optional: pin the set
```

`jaato-scaffold plugins` will then show `m365 <- jaato-m365 (jaato_m365.plugin)`
via `PluginOrigin`, so the provenance is visible without reading logs — which
is the answer to "how do we know what third-party code is in the agent".

### A3. The two upstream gaps this example exposes

Both are small, and neither blocks `jaato-m365`; they are what the exercise is
*for*.

1. **An unannotated out-of-tree plugin is silently skipped.** The `PLUGIN_TIER`
   contract is enforced only by an in-tree test. Either the registry should
   log a WARNING when it skips an *entry-point* plugin for a missing
   annotation (it currently `_trace`s at debug), or `_gate_entry_point` should
   name it — the same reasoning that promoted the protocol-gap check from
   `_trace` to `logger.warning` at #171. An external author has no other
   feedback channel.
2. **`get_session_env` is not on the SDK.** It lives in
   `shared/session_context.py`, so the one thing every credential-bearing
   connector plugin needs is reachable only by importing jaato-server. The
   workaround is a soft import:

   ```python
   try:
       from shared.session_context import get_session_env
   except ImportError:                      # running outside the daemon
       def get_session_env(key, default=None):
           return os.environ.get(key, default)
   ```

   which is correct — `get_session_env` falls back to `os.environ` anyway — but
   every out-of-tree connector will write it, and writing it wrong means
   concurrent sessions clobber each other's tokens. A three-line re-export in
   `jaato_sdk` would close it.

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
§A2 rather than the webhook plugin's own config, and a **loud** WARNING (plus a
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
| **2** | `jaato-m365` **out-of-tree** distribution: entry point, `PLUGIN_TIER`, `m365-auth` device code | any framework change |
| **3** | `jaato-m365` tool surface (mail / calendar / teams), tool-allow-listable | any framework change |
| **4** | `webhook`: `validation_token_param`, `clientState` body verification, subscription lifecycle | — |
| **5** | Teams client: WS + Adaptive Cards for permission, clarification, plan | — |

Moving the plugin out of tree changes the shape of this table: **the first
four phases now ship without touching jaato at all**, and only 4 and 5 — the
webhook handshake and the Teams client, both genuinely framework machinery —
need a server release. That is the test a connector should pass. If building
`jaato-m365` required a framework change, the framework would be missing an
extension point; it isn't, and the exercise is what proves it.

The two micro-gaps in §A3 (a WARNING on the `PLUGIN_TIER` skip, an SDK
re-export of `get_session_env`) sit outside the phases: neither blocks
anything, both are worth upstreaming once the example has demonstrated it
needed them.

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
