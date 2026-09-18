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

It removes both things the block governs, on their two clocks: audit files
past `retention_days`, and session records past
`conversation_retention_days`.

Six properties, each attached to a way a deleting sweep goes wrong:

- **it acts only on a declared policy** — a workspace whose profiles
  declare no `record_keeping:` is never touched;
- **only positive evidence expires a record** — a path whose age cannot be
  read is kept;
- **a loaded session's workspace is skipped** — its logs are open — and a
  loaded session's own record is never removed whatever its age;
- **each profile's files are judged under that profile's own clock.**
  Pooling them under the strictest declaration in the workspace let one
  profile's 30-day retention unlink a sibling's record whose own profile
  said `retention_days: 0`. A policy that silently governs another
  profile's files is not a policy;
- **a trace path is expanded, not taken literally.** The provider channel
  splits per agent — `prov.jsonl` becomes `prov_subagent_1.jsonl`, and
  `prov{agent_suffix}.jsonl` says so explicitly — so the siblings are
  globbed. Judging the literal string left them accumulating forever while
  the pass reported nothing kept and nothing removed;
- **it names what it kept, not only what it removed** — a pass that only
  logs deletions cannot answer an operator asking why a record is still
  there.

The two clocks resolve differently across profiles, because they govern
different objects. `retention_days` governs the files a profile **names**,
so each profile's declaration reaches only its own. A session record is
named by no profile — one directory per session, whichever profile ran it —
so the workspace has **one** conversation clock, and the only safe reading
of several declarations is the longest.

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

## Tamper evidence: `integrity: sha256-chain`

Article 73(6) asks that, after a serious incident, the logs used in the
investigation not have been altered. With `integrity: sha256-chain`, each
record carries `prev_digest` — the SHA-256 of the previous record's
canonical bytes — and `digest`, its own. The first record of a FILE chains
to the literal `genesis`, so "the start of the chain" and "somebody deleted
the field" are different states on disk.

```bash
jaato-doctor --audit-verify .jaato/logs/ledger.jsonl
```

It needs nothing but the file and the standard library, reports the first
record whose link broke and names the line, and **never fails the run** —
`jaato-doctor` is documented as usable as a CI gate, and a broken chain is
a finding for a person to act on rather than a build error.

**What it proves and what it does not.** It proves the file was not edited
in place after the fact. It does **not** prove who wrote it: a writer
holding the file can re-chain from any point, and nothing here stops them.
That distinction is exactly what an investigator needs, so it is printed in
the verifier's own output rather than left to be inferred from the absence
of a signature. Signing — a key the daemon holds — is the next step and is
deliberately not built.

**An unchained file reports as unchained, not as intact.** The question is
whether this file was tampered with; for a file carrying no digests the
true answer is that it is evidence of nothing either way, and answering
"fine" would be answering a different question.

**Both write paths chain identically.** `TokenLedger` appends per record
and `write_ledger` flushes whatever the append path did not, so one file
can be written by both. If only one chained, the file would break in the
middle — which reads exactly like tampering.

**The chain belongs to the FILE, not to the process writing it.** Every
chained append takes an exclusive lock on the target, reads the last
record's digest back off disk, and links to that. So a daemon restart, a
second session sharing an absolute `trace.ledger`, and two processes
appending concurrently all continue the one chain — rather than each
beginning a rival one and leaving a `genesis` link in the middle of a file
that nobody touched. Holding the pointer in memory alone was exactly that
defect, and its symptom was the mechanism accusing its own normal
deployment.

A file that already holds UNCHAINED records and then gets chained appends
is announced once at WARNING: the older half verifies as *carries no chain
fields*, which is evidence of nothing either way rather than of tampering,
and a reader should know which half is which.

**Stated cost.** A chained file cannot be pruned from the front: removing a
line breaks every link after it. Retention therefore rotates whole
**files** — a new file per period, each starting at `genesis` — rather than
deleting lines from one.

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
