# The audit record

What a jaato deployment records, where it lands, and how long it is kept.

Regulation (EU) 2024/1689 asks three things of a high-risk AI system's
logs, and this page is the answer to the middle one:

| Article | Asks for | Answered by |
|---|---|---|
| 12(1) | "automatic recording of events (logs) over the lifetime of the system" | the five stores below, which record without being asked |
| **13(3)(f)** | the instructions for use to describe "the mechanisms included within the AI system that allows deployers to properly collect, store and interpret the logs" | **this page, and `jaato-scaffold explain audit`** |
| 19(1), 26(6) | the provider and the deployer each to keep those logs for at least six months | the `record_keeping:` profile block |

> **Read `jaato-scaffold explain audit` first.** This page is prose; that
> command renders the same schema **computed from the installed tree**, and
> `explain audit <profile>` names the concrete files *that* profile writes
> to. Where the two disagree, the command is right — it reads
> `jaato_sdk.audit.AUDIT_SCHEMA`, and a guard fails the build when a writer
> stops writing a field the schema promises.

## It is not a sixth store

Five things in this framework already record. The gap #1119 closes was not
that nothing was written down; it was that **nothing said what was
guaranteed**, so a deployer answering 13(3)(f) would have had to
reverse-engineer a description from five formats.

`jaato_sdk.audit` is a *contract over the stores that exist*:
`AUDIT_SCHEMA` enumerates each event, the fields it carries, and the store
it lands in. Nothing new writes.

| Store | Where | Governed by |
|---|---|---|
| ledger | `trace.ledger` (env `LEDGER_PATH`) | `retention_days` |
| application trace | `trace.session_log` (env `JAATO_TRACE_LOG`) | `retention_days` |
| provider trace | `trace.provider_log` (env `JAATO_PROVIDER_TRACE`) | `retention_days` |
| session record | `<workspace>/.jaato/sessions/<id>/` | `conversation_retention_days` |

The last row is the one that is **not** an audit log. It is the
conversation — personal data somebody may ask to have erased — and it is
governed by its own clock for exactly that reason.

A path written as a **relative** value is resolved per session against the
workspace, so each session gets its own file; an **absolute** one is a
single file shared by every session using that profile.

## Three rules a reader has to know

**Absent is not zero.** A dimension nothing reported is *omitted* from a
record, never written as `null` or `0`. A provider that reported no cache
must not read as a cache that never hit. `explain audit` marks such fields
with `?`.

**`user_id` and `approver` absent means nobody was asked.** On a
`permission-check` row and on a `DECISION` line, those two fields are
present only when a human or an external approval system answered. A
policy decision — a whitelist hit, a default deny — carries neither, and
that distinction is the point: "nobody was asked" has to stay
distinguishable from "somebody answered".

**The `DECISION` line is machine-readable by construction.** Scalars first,
free-text `reason` last, so the line parses.
`shared.plugins.permission.plugin.parse_decision_trace` reads it back. It
is the one artefact every deployment gets: the ledger's row needs a ledger
configured, and `PermissionResolvedEvent` is opt-in.

## Keeping it: `record_keeping:`

```yaml
record_keeping:
  retention_days: 180              # the audit stores, minimum
  conversation_retention_days: 30  # the session record may go sooner
  integrity: none                  # none | sha256-chain
```

**Declared, never defaulted.** Unlike `max_orphan_seconds` — which has a
framework default precisely because the session that needs it is the one
whose profile declared nothing — this block changes **what delete means**,
and a default would change that for every existing deployment on upgrade.
A profile with no block deletes exactly as it always has.

**`0` means "keep until something deletes it"**, the `0`-disables spelling
`max_session_seconds` uses.

**Inheritance is most-restrictive-wins**, and "most restrictive" is spelled
two ways here. The two day counts are *minimums*, so restrictive means
**longer** — the maximum across declaring layers — and `0` cannot win it: a
child may keep longer than its base, never shorter, and may not disable a
retention an ancestor set. `integrity` is ranked, so restrictive means
further along `none → sha256-chain`.

### What it changes about deleting

| Verb | With no block | With `retention_days` declared and unelapsed |
|---|---|---|
| `session.delete` | removes the conversation; the trace and ledger files live outside the session directory and were never touched | unchanged |
| `workspace.delete` | removes the whole tree, logs included | **refused**, naming the files held, the minimum, and when the last of them expires |

`workspace.delete` refuses rather than preserving the files as orphans or
overriding with a warning. Preserving would leave files in a directory an
operator asked to be gone — a surprise discovered later. Overriding makes
the policy something any delete silently defeats, which is not a policy. A
refusal is visible, is recoverable (wait, or drop the block), and cannot
silently destroy a record somebody declared had to be kept. The remedy the
message names is `session.delete`, which is the verb for "remove the
conversation, keep the record".

### Letting go

A retention policy that only ever keeps is a one-way ratchet, which is its
own problem under GDPR storage limitation. The daemon's lifetime watchdog
(#812) runs a **retention pass** on its own much coarser clock — hourly,
because the shortest retention anybody writes is a day — removing audit
files past their minimum.

Four properties, each attached to a way a deleting sweep goes wrong:

- **it acts only on a declared policy** — a workspace whose profiles
  declare no `record_keeping:` is never touched;
- **only positive evidence expires a record** — a path whose age cannot be
  read is kept;
- **a loaded session's workspace is skipped** — its logs are open;
- **it names what it kept, not only what it removed** — a pass that only
  logs deletions cannot answer an operator asking why a record is still
  there.

## What `validate` says

| Finding | Severity | Fires when |
|---|---|---|
| `high_risk_without_retention` | error | `risk_class: high` and no `record_keeping:` — the record is written and nothing keeps it |
| `retention_below_article_19` | warn | `retention_days` under 180. Art. 19(1) says "unless otherwise provided", so a shorter period may be lawful — hence a warning |
| `record_keeping_inert` | warn | a block declared over stores this profile never writes to |
| `high_risk_without_record_keeping` | error | `risk_class: high` and no `trace.session_log` — nothing is *written* at all |

The last two pair the way `budget_control_absent` and
`budget_limits_without_abort` do: one says you declared nothing, the other
says what you declared cannot act, and neither is useful without the other.

## What this does not do

- **It does not sign anything.** `integrity: sha256-chain` proves a file
  was not edited in place; it does not prove who wrote it, because a writer
  holding the file can re-chain from any point.
- **It does not decide your retention.** Six months is the floor Article
  19(1) names; what is "appropriate to the intended purpose" is the
  provider's determination, which is why the validator warns rather than
  refuses below it.
- **It does not record what a plugin records.** An out-of-tree plugin
  writing its own log is outside this contract and outside the retention
  pass.
