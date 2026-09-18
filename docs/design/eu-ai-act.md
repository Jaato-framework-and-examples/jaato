# The EU AI Act — what it asks of a jaato application, and what the framework should build

## Summary

Regulation (EU) 2024/1689 (the AI Act) regulates **AI systems** and
**general-purpose AI models**, and it addresses its obligations to the
**provider** who places a system on the market and the **deployer** who runs it.
jaato is neither of those things on its own: it is a framework, and a jaato
*application* — a profile, a persona, a tool surface and a model binding,
served as sessions — is the AI system. So the Act does not ask jaato to be
compliant; it asks the people shipping on jaato to be, and the question this
document answers is which of their obligations the framework can turn into a
mechanism, and which it should leave alone.

Three findings shape everything below.

1. **The open-source exemptions do not apply.** Both distributions are
   `license = "BUSL-1.1"` (`jaato-server/pyproject.toml`), and the Business
   Source License is source-available, not free and open-source. Article 2(12)
   (systems released under free and open-source licences) and the carve-out in
   Article 25(4) (third parties making components available under such a
   licence) are therefore out of reach. The practical consequence is Article
   25(4)'s first sentence: a component supplier to a high-risk system must, *by
   written agreement*, provide "the necessary information, capabilities,
   technical access and other assistance" the provider needs — which means the
   framework needs a documented, versioned account of what it does and does not
   guarantee, not a README.

2. **Article 50 has applied since 2 August 2026; the high-risk chapter has
   not.** The Digital Omnibus on AI (Regulation (EU) 2026/1744, OJ 24 July
   2026, in force 27 July 2026) deferred Annex III high-risk obligations to
   **2 December 2027** and Annex I ones to **2 August 2028**, and left Article
   50 on schedule with one grace period: systems already on the market before
   2 August 2026 have until **2 December 2026** to carry the machine-readable
   marking of Article 50(2). So the two obligations that bind a jaato
   application *today* are the ones about talking to people — say that you are
   an AI, mark what you generate — and jaato has no mechanism for either.

3. **Most of the high-risk requirements are already half-built, as engineering.**
   Logging (Article 12), human oversight (Article 14), robustness and
   cybersecurity (Article 15) describe things the framework has been building
   for its own reasons: trace lines, the permission gate, budget ladders, the
   orphan watchdog, AppArmor, secret scrubbing, the untrusted-content boundary.
   What is missing is the *shape* the Act wants — one audit record with a
   stated schema and a retention policy instead of five stores with none, and
   documentation that is generated from the running framework rather than
   written about it.

The recommendation is a short list of mechanisms, ordered by when they bind
(§5). The first three are small and are due now; the rest are what lets a
deployer adopt a jaato application into a high-risk process in 2027 without
re-deriving the framework's behaviour from its source.

## Status & verification disclaimer

The Act was read from the Official Journal text at EUR-Lex
(CELEX:32024R1689, English), fetched 2026-09-18; article quotations below are
from that text. The Omnibus amendments are taken from secondary sources (law
firm alerts and the Cloud Security Alliance's research note), because the
consolidated text was not retrievable here; the dates are consistent across
four sources but should be checked against the OJ before anything is
scheduled on them. Framework claims were verified against commit `982a0b3`
(server 0.16.0, SDK 0.23.0) on 2026-09-18, and each names the file it was
read from. Nothing here is legal advice; it is an engineering reading of a
legal text, written to decide what to build.

## 1. Where jaato sits in the Act's vocabulary

| Act term (Article 3) | In a jaato deployment |
|---|---|
| **AI system** (3(1)): "a machine-based system that is designed to operate with varying levels of autonomy … infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments" | A session: a profile bound to a provider and model, running a persona over a tool surface. An agent that executes `cli`, writes files, calls services and speaks is the *high-autonomy* end of that definition, not a borderline case. |
| **General-purpose AI model** (3(63)) | The upstream model — Anthropic's, OpenAI's, a Gemini, a Kimi. jaato trains nothing and hosts no weights; the provider plugins are wires to somebody else's model. Chapter V (Articles 51–56) is those vendors' problem. |
| **Downstream provider** (3(68)): "a provider of an AI system … which integrates an AI model … provided by another entity" | Whoever ships a jaato application. Article 53(1)(b) obliges the model vendor to give them Annex XII information; jaato's job is to record *which* model served *which* turn so that information can be attached to the right output (it does: the consumption aspect keys spend by `(provider, model, tier)` binding). |
| **Provider** (3(3)) | The organisation that develops the application and puts it into service under its own name — including a team using jaato internally, since "putting into service" covers "own use". |
| **Deployer** (3(4)) | The organisation using it under its authority. Often the same organisation as the provider; the Act still separates the two sets of duties. |
| **Third party supplying tools, services, components** (25(4)) | jaato itself, and every plugin distribution. |
| **Intended purpose** (3(12)) | Nothing in a profile today. The Act's whole risk classification hangs on it, and the profile is where every other property of an application is declared — see §4.1. |

Two things follow from the table. The framework cannot classify an
application's risk, because risk is a function of intended purpose and the
framework does not know it; and the framework *is* the place where nearly
every technical requirement the Act attaches to a high-risk system would be
implemented, because a jaato application has almost no code of its own.
That combination is what makes a `regulatory:` declaration in the profile
the load-bearing mechanism: it is the one fact only the author knows, and
everything else the Act asks for can be derived from it plus what the
framework already introspects.

## 2. What applies, and since when

| Chapter / Article | Who | Applies from | Reaches a jaato application when |
|---|---|---|---|
| Article 5 — prohibited practices | everyone | 2 Feb 2025 (Omnibus adds non-consensual intimate imagery / CSAM generation, technical safeguards by 2 Dec 2026) | a persona is written to do one of them; see §4.10 |
| Article 4 — AI literacy | providers and deployers | 2 Feb 2025 (softened by the Omnibus to "support the development of" literacy) | always; it is organisational |
| Chapter V — GPAI models | model providers | 2 Aug 2025 | never directly |
| **Article 50 — transparency** | providers (50(1), 50(2)) and deployers (50(3), 50(4)) | **2 Aug 2026**; 50(2) marking on pre-existing systems by 2 Dec 2026 | **any application that talks to people or generates content — i.e. nearly all of them** |
| Chapter III — high-risk (Articles 6–27, 72–73) | providers and deployers of Annex III systems | **2 Dec 2027** (Annex III), **2 Aug 2028** (Annex I) | the application's intended purpose is in Annex III: recruitment, worker management, credit scoring, education access, essential-service eligibility, emergency triage, law enforcement, migration, justice |
| Article 86 — right to explanation | deployers of Annex III systems | with Chapter III | same |
| Article 99 — penalties | — | 2 Aug 2025 (Omnibus: SME/small mid-cap relief extended) | — |

Two readings of that table that matter for scope:

- **A general-purpose assistant is not high-risk by being capable.** Annex III
  lists *intended purposes*, and Article 6(3) exempts systems that perform "a
  narrow procedural task", "improve the result of a previously completed human
  activity", or "a preparatory task to an assessment". A coding agent, a
  documentation writer, a helpdesk bot answering product questions: limited
  risk, so Article 50 and nothing from Chapter III. A jaato application
  screening CVs or scoring loan applications: high-risk, whatever it was built
  from. Article 25(1)(c) is the trap for the framework's users — modifying the
  intended purpose of a general-purpose system so that it becomes high-risk
  makes *you* its provider.
- **Article 50 is the one that already binds, and it binds broadly.** 50(1)
  covers any system "intended to interact directly with natural persons" —
  chat bots, voice agents, the web client. 50(2) covers any system "generating
  synthetic audio, image, video or text content" — every jaato session
  generates text, and the voice tiers generate audio. The exemptions are
  narrow: "obvious from the point of view of a natural person who is
  reasonably well-informed" for 50(1), and "assistive function for standard
  editing" for 50(2).

## 3. What the framework already has, read against the Act

The Act's technical requirements for high-risk systems are Articles 9–15.
Reading each against the tree, most of what it asks for exists in some form,
and the gap is almost always the same: the fact is *produced* but not
*collected*, or it is enforced but not *declared*.

### Article 12 — record-keeping ("automatic recording of events over the lifetime of the system")

| What exists | Where | What the Act would read it as |
|---|---|---|
| provider trace (`JAATO_PROVIDER_TRACE` / `trace.provider_log`) — every request and response, tool ids resolved (#873) | `jaato_sdk/trace.py`, per provider | the model-facing half of the log |
| application trace (`JAATO_TRACE_LOG` / `trace.session_log`) — `[PERMISSION] … DECISION` lines (#951, machine-readable, `asked=`, `policy=`, `user_id=`/`approver=`), `[TOOL_RUNNER] resolve/permission/result`, `BUDGET CEILING` / `BUDGET RUNG` (#955), `HISTORY_INVARIANT`, `ENRICH` | `shared/plugins/permission/plugin.py`, `shared/ai_tool_runner.py`, `shared/jaato_session.py` | events "relevant for identifying situations that may result in … a risk" (12(2)(a)) — the decision record |
| token ledger — `response` rows (tokens, `user_id`) and `permission-check` rows (tool, args, verdict, method, caller, approver) | `shared/token_accounting.py`, written from `jaato_session.py:11851` and `ai_tool_runner.py:638` | the attributable account of what ran and who allowed it |
| session record (2.x) — history, `profile_snapshot`, `rendered_instructions`, `agent_params`, `created_by`, `runner_identity`, `sandbox_mode`, `turn_count` | `<workspace>/.jaato/sessions/`, `shared/session_persistence.py` | "the period of each use" (12(3)(a)), and the exact configuration that produced every output |
| telemetry — OpenInference spans, `redact_content` default true, cost with provenance | `shared/plugins/telemetry/` | post-market monitoring feed (12(2)(b), 72) |
| per-session logs | `.jaato/logs/` | operational |

Three gaps, in decreasing order of surprise:

1. **The ledger never reaches disk on the daemon path.** `TokenLedger.write_ledger()`
   is the only method that writes `LEDGER_PATH`, and it has no caller outside
   tests (measured: `grep -rn "write_ledger(" --include=*.py .` returns the
   definition and its own docstring). `JaatoServer` constructs a ledger
   (`server/core.py:693`) and both writers append to it, so the `permission-check`
   rows with approver identity that #859 and #951 added are held in memory and
   lost with the process. The consumption aspect reads them live, which is why
   nobody noticed. That is a defect independent of the Act, and it is the Act's
   one hard requirement in this area: 12(1) says *automatic* recording, and
   19(1) / 26(6) say the logs are *kept*.
2. **No retention policy exists anywhere.** `grep -rn -i retention` across the
   session plugin and the session manager returns nothing. Articles 19(1) and
   26(6) require logs kept "for a period appropriate to the intended purpose …
   of at least six months"; Article 2(7) makes GDPR's data-minimisation apply to
   the same records. Both `session.delete` and `workspace.delete` remove the
   session record outright, so today a deployer cannot delete a conversation
   and keep its log, which is the combination those two articles together
   require.
3. **Five stores, no schema, no integrity.** The facts are spread across the
   two traces, the ledger, the record and telemetry, in three formats, with no
   statement of which events are guaranteed to appear where. #507's
   tamper-evidence half is still open. Article 12 does not require integrity,
   but Article 73(6) forbids "altering the AI system concerned in a way which
   may affect any subsequent evaluation of the causes of the incident", and a
   log that cannot show it was not altered is weak evidence in either
   direction.

### Article 14 — human oversight

14(4) is a checklist, and the framework answers most of it already:

| 14(4) | Mechanism in the tree |
|---|---|
| (a) understand capacities and limitations, monitor operation, detect anomalies | `jaato-scaffold explain` (profile / plugins / provider / runtime / completion / env), `PROVIDER_NOTES`, `validate`; the trace and telemetry for monitoring; the `reliability` plugin's pattern detector and circuit breaker for anomalies |
| (b) remain aware of automation bias | nothing — see below |
| (c) correctly interpret the output | reasoning replay / thinking surfaced to the client; `PresentationContext`; `ClarificationBatchEvent` when the agent is unsure |
| (d) decide not to use, disregard, override or reverse the output | the permission gate (`defaultPolicy: ask`, whitelist/blacklist, `askPermission`, external approval channels, approver identity #859, `emit_decision_events`); `file_edit` per-session backups and `rewind` for the reversible subset of tool effects |
| (e) intervene or interrupt through a "stop" button, halting in a safe state | `jaato-server --stop` (saves every session, then shuts each down), `session.stop <id>` (#812), `client.stop()` / `request_stop()` and the cancel token, `budget_control` `abort` rungs, the orphan watchdog (`max_orphan_seconds`), `permissions suspend`, completion gates (`completion_processors`) that refuse an unacceptable result |

The gaps are narrower than the coverage but real:

- **The stop button exists twice, and neither is documented as one.**
  `jaato-server --stop` halts the whole deployment from the host shell with
  no client and no session id: `SessionManager.shutdown()` saves every
  loaded session to disk *before* shutting its server down, and the runner
  template is asked to stop politely before SIGTERM. `session.stop <id>`
  does the same for one session by the same cancellation path. That is
  14(4)(e)'s "halt in a safe state" and 26(5)'s "suspend the use of that
  system"; what is missing is a surface that names them as such, which
  §4.5 puts on `explain` and `jaato-doctor`.
- **Reversal is partial by nature.** `file_edit` backs up, `rewind` restores
  history; a `cli` command, a `call_service` POST, a Telegram reply cannot be
  reversed by the framework. The honest mechanism is the *record* of effects,
  not an undo — which is the same audit log as §4.4.
- **Automation bias has no mechanism, and probably should not.** It is a
  property of the person and the process; the Act puts it on the provider's
  instructions for use and the deployer's training (26(2)). What the framework
  can do is not make it worse: the reasoning block collapsed-by-default in the
  TUI and the uncertainty a clarification carries are the right shape.

Worth naming, because it is the one place the framework already implements
a 14(3)(a) measure "built into the system by the provider": a profile with
`plugin_configs.permission.policy.defaultPolicy: ask` *is* a human-in-the-loop
design, and #957 made it hold per subagent. It is not documented as an
oversight measure anywhere a deployer would look.

### Article 15 — accuracy, robustness, cybersecurity

| 15 | Mechanism |
|---|---|
| (1),(3) accuracy, declared with metrics | `jaato-eval` — a benchmark harness over profiles with `script` and `processor` graders that its own status report calls trustworthy, and a `judge` grader it does not (`jaato-eval/STATUS.md`). Nothing carries a result into anything a deployer reads. |
| (4) resilience to errors, faults, inconsistencies; fail-safe | retry with backoff, request and stream deadlines (#732), the `reliability` plugin, `history_invariant` repair (#674), budget ceilings observed mid-turn (#955), the completion-nudge bound (#919/#934), `runtime_limits` (cgroups, timeouts, output caps, #735) |
| (4) feedback loops in systems that continue to learn | the `memory` plugin and the continuity pattern **are** a post-deployment learning loop: model-written memories are re-injected into later sessions. The raw-then-curated lifecycle exists (`docs/design/agent-continuity.md`) and `allowed_scopes` gates the write side; nothing records *which session and model* wrote a memory, and curation is a pattern, not a knob. |
| (5) resilience against third parties altering use, outputs or performance; "inputs designed to cause the AI model to make a mistake"; confidentiality attacks | the untrusted-content boundary (`TRAIT_UNTRUSTED_CONTENT`) and untrusted-schema sanitisation (`TRAIT_UNTRUSTED_SCHEMA`) for indirect prompt injection; AppArmor per-session profiles with per-thread verification (#1023, #1014, #1033, #1100); cgroups; `scrub_secret_env` on by default (#863) with the `/proc` deny backstop (#712); path containment on `cli`, `interactive_shell` (#722) and the notebook (#710); entry-point plugin trust (#684); WS bearer auth, app credentials and tickets (#1074); IPC peer entitlement; workspace-name containment; credential-file locking (#683); the secret-safe `repr` (#721) |

Two things to say plainly in the instructions for use, because the tree
already says them internally: the prompt-injection boundary is
defence-in-depth and not a boundary (a model can be talked past a marker),
and the notebook's audit-hook tier is not a kernel boundary either
(`kernel_sandbox`'s own docstring). 15(5) asks for measures "appropriate to
the relevant circumstances and the risks", and the AppArmor tier is the
one answer this tree has that is a boundary in the strict sense — which is
why `JAATO_REQUIRE_APPARMOR=1` exists and why a high-risk deployment should
set it.

### Article 13 and Annex IV — transparency to deployers, technical documentation

Annex IV wants a system description (components, versions, how it interacts
with other software, the interface given to the deployer), the human
oversight measures, the cybersecurity measures, the logging arrangements,
validation and testing, and a change history. The framework can *generate*
about half of that from a profile, because half of it is exactly what
`jaato-scaffold explain` already computes — and computing it is what stops it
drifting, the argument `explain` was built on. The other half (intended
purpose, risk management, accuracy against declared metrics, foreseeable
misuse) only the provider can write; the mechanism is a skeleton with those
sections marked as theirs.

### Articles 9, 17, 72, 73 — risk management, quality management, post-market monitoring, serious incidents

These are organisational systems the provider must *have*; a framework
cannot have them on anyone's behalf. What it can supply is the signal:

- **Incidents.** `SessionTerminatedEvent(reason="error" | "budget_exhausted")`,
  `AgentErrorEvent`, `ErrorEvent`, `BudgetRungFiredEvent`, the reliability
  plugin's circuit-breaker trips, `NudgeExhausted`, a completion gate's
  `on_exhausted: fail`, a permission DENY on a tool the profile expected to
  run — each is a candidate entry in an incident register, and today each is
  an event or a trace line with no register to land in. Article 73's clocks
  are 15 days, 2 days for a widespread infringement, 10 days for a death, all
  counted from *becoming aware* — so the register's job is to make awareness
  a queryable fact rather than a log grep.
- **Post-market monitoring** (72(2): "actively and systematically collect,
  document and analyse relevant data … on the performance of high-risk AI
  systems throughout their lifetime"). Telemetry plus a scheduled
  `jaato-eval` regression run against the deployed profile is the technical
  half of a monitoring plan; the eval harness's "eval environments" design
  (`docs/design/eval-environments-layer.md`) is the right place for it.
- **Do not alter the system while investigating** (73(6)). The persisted
  profile snapshot and rendered persona (#787) are the freeze; the missing
  piece is an export that is demonstrably the record as it was.

### Article 50 — the two that bind now

**50(1)** — "AI systems intended to interact directly with natural persons
are designed and developed in such a way that the natural persons concerned
are informed that they are interacting with an AI system", "at the latest at
the time of the first interaction" (50(5)). The framework knows what kind of
surface it is talking to (`ClientType`: `terminal`, `web`, `chat`, `api`)
and carries a framework instruction layer with three named pieces
(`disk`, `constants`, `security`). None of the constants say anything about
being an AI; a persona can be written to claim to be a person and nothing
warns; there is no first-interaction announcement a chat or voice client can
render; and `suppress_base_instructions: true` drops `constants` wholesale,
so any disclosure instruction placed there would vanish with the rest.

**50(2)** — outputs "marked in a machine-readable format and detectable as
artificially generated", "effective, interoperable, robust and reliable as
far as this is technically feasible … taking into account … the generally
acknowledged state of the art". A speaking tier delivers headerless pcm16
(`STREAM_AUDIO_MIME`, `_media_deltas.py`) in `ToolOutputEvent` chunks;
`StreamChunk` and `MediaDelta` carry no provenance field; the containerising
into a file that a person hears happens in the *client* (a bot, the web
page), which is where a C2PA manifest or an audio watermark would be
attached — and the client is not told, in any typed way, that the bytes are
model-generated. `grep -rn -i "watermark|c2pa"` across the tree returns
nothing. For text, the AI Office's code of practice on marking is still being
drawn up (50(7)); text has no interoperable marking standard today, and the
obligation is bounded by "technically feasible" — so text is a documentation
question until a standard exists, and audio and images are an engineering one
now.

### Article 4 — AI literacy

Organisational, and softened by the Omnibus. The framework's contribution is
that its documentation is generated from itself: `explain`, `validate`,
`jaato-doctor`, the `jaato-sdk` skill. Nothing to build; one thing to write
(§4.7).

### Article 2(7), 26(9) — GDPR runs alongside

Session records and memories carry personal data whenever a person talks to
the agent. Telemetry withholds content by default (#858), secrets are
scrubbed from subprocesses, and `SessionHistory` exposes a write-side
transformer seam whose canonical consumer is pseudonymisation (jaato-premium's
`[pseudonymization]` extra). The tension the Act adds is that 19(1)/26(6)'s
"at least six months" and GDPR's minimisation apply to the same log — which
the retention mechanism in §4.4 has to resolve by keeping the *audit* record
and letting the *conversation* be deleted, or by pseudonymising the record
through that seam.

## 4. The mechanisms

Each is stated as a contract the way the rest of this tree states them:
what it does, where it lives, what it deliberately does not do.

### 4.1 A `regulatory:` block in the profile — the one fact only the author knows

```yaml
# .jaato/profiles/screener.yaml
regulatory:
  intended_purpose: >
    Pre-screens inbound job applications against the role's stated
    requirements and drafts a shortlist for a recruiter's review.
  risk_class: high            # minimal | limited | high
  annex_iii: 4a               # optional; the Annex III point when high
  provider:
    name: Acme Talent GmbH
    contact: compliance@acme.example
  interacts_with_persons: true
```

- **`validate` escalates with the declared class.** Today
  `budget_control_absent`, `budget_limits_without_abort`,
  `secret_scrub_disabled`, `permission` findings and `missing_description`
  are warnings, for the reason the whole family is — an error would fail every
  existing workspace. Under `risk_class: high` the same findings become
  **errors**, plus new ones: no `permission` policy declared
  (`high_risk_without_oversight_policy`), no trace or audit log configured
  (`high_risk_without_record_keeping`), `suppress_base_instructions` naming
  the disclosure piece, `require_confinement` unset on `interactive_shell`.
  Nothing changes for a profile that declares no class: absent is `minimal`
  for validation purposes and `unknown` for documentation purposes, and the
  dossier says so.
- **`explain profile` renders it, the dossier (§4.6) is built from it, the
  disclosure (§4.2) reads `provider.name` and `interacts_with_persons`.**
- **Inheritance is child-replaces for the scalars and most-restrictive-wins
  for `risk_class`** — a child may not declare itself `minimal` under a base
  that says `high`, the `max_parallel_tools` shape.
- **Deliberately not done:** inferring the class. A framework that guesses an
  application is not high-risk has made a legal determination on the author's
  behalf and made it silently; the Act (6(4)) requires the provider to
  *document* that assessment, and a declared key is the documentation.

### 4.2 Disclosure of AI interaction (Article 50(1))

Three touches, one fact:

1. **A fourth named instruction piece, `disclosure`,** beside `disk`,
   `constants`, `security` in `suppress_base_instructions`: *this assistant is
   an AI system; it must not claim to be a human being, and when asked whether
   it is an AI it says so.* The blanket `true` keeps it, exactly as it keeps
   `security`; dropping it needs `{disclosure: true}` by name and is announced
   at WARNING, because it is a legal posture change and the framework's
   existing rule is that weakened postures announce themselves
   (`scrub_secret_env: none`, `--ws-unsafe-no-auth`, `allow_inline`).
2. **A first-interaction announcement** the framework emits, not the model:
   `SessionInfo` / the first `AgentOutputEvent(source="system")` on a session
   whose profile declares `interacts_with_persons: true`, carrying a short
   text (`disclosure.text`, default rendered from `regulatory.provider.name`)
   that a chat client renders as a message and a speaking tier renders as
   the first utterance through `ensure_spoken_part`. A `PresentationContext`
   flag (`client_discloses_ai: true`) lets a client that already shows a
   badge suppress the message — the Act's "unless this is obvious" clause,
   asserted by the party that can see the screen.
3. **A `validate` finding,** `disclosure_absent` (warn; error under
   `risk_class: high`), for a profile that binds a persona and declares
   neither `interacts_with_persons` nor a client-side disclosure.

Not done: detecting a persona that lies about being human. The constant makes
the model refuse; a `validate` grep for "you are a human" is a lint that
would be defeated by the first synonym and would certify what it did not
find.

### 4.3 Provenance on generated output (Article 50(2))

The framework's boundary is the event protocol, and the event protocol is
where a client learns what it is about to show a person. So:

1. **A `generated_by` stamp on every model-generated payload that crosses the
   wire**: `MediaDelta` → `StreamChunk` → `ToolOutputEvent` for audio and
   images (`{"kind": "ai", "provider": ..., "model": ..., "session_id": ...,
   "ingest_id": ...}`), and the same shape on tool-result `Attachment`s the
   model produced (the `_multimodal` image tools). Additive fields, protocol
   minor bump, absent for a chunk a tool merely relayed (a fetched image is
   not AI-generated because an agent fetched it). `AgentOutputEvent` text is
   already attributed by `source`; the stamp adds the binding. This is the
   *machine-readable* half, at the layer jaato owns.
2. **A marker hook for the client-facing half.** A plugin trait,
   `TRAIT_OUTPUT_MARKER`, invoked on `CLIENT`-audience media before delivery
   and on files the agent stages, with one in-tree implementation that is
   stdlib-feasible — a C2PA-style manifest sidecar for image and PDF
   attachments naming the stamp above — and audio watermarking left to an
   out-of-tree plugin, because there is no dependency-free implementation
   and the state of the art moves faster than a release. The trait is the
   contract; which algorithm satisfies "state of the art" is the provider's
   choice, and the dossier records which marker ran.
3. **Documentation, not code, for text**, until the Article 50(7) code of
   practice names a standard. The instructions for use say so, and say that
   50(4)'s deployer-side disclosure ("text which is published with the
   purpose of informing the public") is a publishing decision the framework
   cannot see.

The 2 December 2026 grace period is for systems already on the market; a
jaato application put into service after 2 August 2026 has no grace at all,
which is why the stamp is in the "now" tier of §5 even though the hook can
follow.

### 4.4 One audit record, with a retention policy (Articles 12, 19, 26(6))

Not a sixth store. A **contract over the stores that exist**:

- **Fix the ledger first.** `write_ledger()` gets a caller — appended per
  record rather than flushed at exit, so a process that dies mid-turn has
  written what it recorded — or the ledger is folded into the application
  trace, which already reaches disk and already carries the `DECISION` line
  the ledger's `permission-check` row duplicates. Either way, a `permission-check`
  row with an `approver` has to exist somewhere after the daemon exits.
- **A declared event schema**, `docs/audit-log.md` plus a typed
  `AuditRecord` in `jaato_sdk`, enumerating what is guaranteed to be recorded
  per session and where: lifecycle (start, end, reason, client identity,
  framework version, profile snapshot digest, model bindings), each model
  round trip (binding, tokens, finish reason, cost source), each tool call
  (name, argument digest, verdict, method, `asked`, approver, `ok`,
  duration), each budget rung, tier switch and completion-gate verdict, each
  incident (§4.7). The `DECISION` line's `key=value` grammar (#968) is the
  precedent — a trace line that is a contract. `jaato-scaffold explain
  audit` renders the schema and the paths the profile writes to, so a
  deployer's "mechanisms … to properly collect, store and interpret the logs"
  (13(3)(f)) is a computed page.
- **A `record_keeping:` profile block**, the `runtime_limits` shape:

  ```yaml
  record_keeping:
    retention_days: 180        # minimum kept; 0 = keep until deleted
    conversation_retention_days: 30   # the history may go sooner than the log
    integrity: sha256-chain    # none | sha256-chain
  ```

  Inheritance most-restrictive-wins for the minimums (a child may keep
  longer, never shorter). `session.delete` and `workspace.delete` honour it
  by deleting the conversation and keeping the audit record until its
  minimum has elapsed — the shape that satisfies 19(1) and GDPR at once —
  and the daemon's lifetime sweep (#812) gains a retention pass.
- **`integrity: sha256-chain`** links each record to the previous one's
  digest, the cheapest form of #507's tamper evidence. It proves that the
  file was not edited in place; it does not prove who wrote it. Stated cost:
  a chained file cannot be pruned from the front, so retention rotates whole
  segments.

### 4.5 A stop button — already there; make the tools say so

An earlier draft proposed a daemon-level `halt` verb: stop every loaded
session, keep the daemon up, refuse new turns. It is not needed, and the
first reader of the draft said so. `jaato-server --stop` already cancels
everything, persists every record and returns the pool, from the host
shell, with no client or session id — which is where an oversight person
stands — and `session.stop <id>` does it for one session by the same
cancellation path. "Halt in a safe state" (14(4)(e)) is *cancelled at the
next check point and persisted*, and both verbs do exactly that. What the
draft's verb would have added — a daemon that stays up refusing turns — is
an operational nicety nothing in the Act asks for, at the price of a new
lifecycle state and a protocol bump.

What is missing is that no surface a deployer reads *names* them as the
oversight measures, and the framework's rule for facts about itself is that
they are computed, not written down (`explain` exists so that a documented
order that disagrees with the loaded one cannot happen). So the deliverable
is two small additions to the surfaces that already exist:

- **`jaato-scaffold explain oversight`** — a topic rendered from the tree, in
  the Act's own 14(4) vocabulary: the two stop verbs and what each persists
  before it cancels; the permission policy in force for a named profile
  (`defaultPolicy`, whitelist, approval channel) as the 14(4)(d) measure; the
  budget ladder's `abort` rung and the orphan watchdog as the in-built
  constraints "that cannot be overridden by the system itself"; and what is
  reversible (`file_edit` backups, `rewind`) against what is not. Read from
  the same helpers the daemon reads, so it cannot drift from them.
- **`jaato-doctor`** — on a running deployment, one preflight line naming the
  daemon's pidfile and the exact `--stop` invocation for it, and how many
  sessions are loaded (`session.orphans` already answers the second half).
  A person who has to stop a system they did not start should not have to
  find the command by reading this document.

The dossier (§4.6) then quotes the `explain oversight` output rather than
restating it, which is the one way to keep the instructions for use and the
running framework saying the same thing.

### 4.6 A generated technical dossier (Articles 11, 13, Annex IV, 25(4))

`jaato-scaffold new dossier --profile <name>` writes an Annex IV skeleton
computed from the installed framework and the resolved profile: system
description (plugins with provenance, providers and models per tier, tool
surface and permission policy, runtime limits, scrub posture, confinement
tier and whether it is enforced, `record_keeping`, disclosure), versions of
every `jaato-*` distribution, the human oversight measures in force (in the
Act's own 14(4) vocabulary), the cybersecurity measures, the logging
arrangements, the last `jaato-eval` results if a results file is named — and
`TODO` sections the framework cannot fill: intended purpose (pre-filled from
`regulatory:`), risk management, accuracy metrics and thresholds, foreseeable
misuse, post-market monitoring plan, change history. Everything computed is
labelled with the commit and date it was computed at, the disclaimer shape
this document uses.

The same generator, run with `--component`, produces the **Article 25(4)
information pack** for jaato as a supplier: what the framework guarantees
(the contracts this tree already has — `PROVIDER_CAPABILITIES`, the
history invariant, the permission single-exit, the confinement verifier),
what it does not (the prompt-injection boundary, in-process notebook memory
reach, `finalize` being advice), and the versioned surface a written
agreement can point to. That pack is the deliverable the BUSL licence makes
necessary, and it is worth producing whether or not any customer is
high-risk, because it is also the honest instructions for use.

### 4.7 An incident register (Articles 72, 73, 26(5))

A typed `IncidentEvent` in the SDK and an `incident` record in the audit
log, raised by the framework at the sites that already know: a session
terminated with `error` or `budget_exhausted`, a circuit breaker opening, a
completion gate exhausting, a confinement verification refusing a bootstrap,
a permission DENY under `defaultPolicy: deny` on a whitelisted-by-intent
tool, `NudgeExhausted`. Each carries severity, session, binding and a
one-line cause. `jaato-doctor incidents --since 15d` lists them with the
Article 73 clocks beside them; the register is a query over the audit log,
not a second store. Whether an entry *is* a "serious incident" (3(49)) is a
human determination, and the tool says so in its output rather than
classifying.

### 4.8 Memory provenance and a curation gate (Article 15(4))

Every memory the model writes records its author binding and session id
(the `generated_by` stamp of §4.3, applied to storage), and the raw→curated
promotion becomes a profile knob (`plugin_configs.memory.require_curation:
true`) under which uncurated memories are stored but never re-injected. The
continuity pattern already describes the curator; the knob is what makes
"this deployment's learning loop is reviewed" a statement `validate` can
check and the dossier can print.

### 4.9 Eval results into the dossier (Article 15(3), 9(6)–(8))

`jaato-eval` writes JSONL per arm. A small adapter renders the metrics of a
named results file into the dossier's accuracy section, with the harness's
own caveat carried verbatim (the `judge` grader is not yet a trustworthy
instrument). Nothing else: choosing the metrics and thresholds "appropriate
to the intended purpose" is the provider's call.

### 4.10 What is deliberately not proposed

- **Classifying risk automatically** — §4.1's argument.
- **A prohibited-practice lint on personas.** Article 5's practices are
  about effect and intent (manipulation causing significant harm, social
  scoring, workplace emotion inference); none is detectable from a prompt.
  The Omnibus's new prohibition on generating intimate imagery is the one
  with a technical safeguard, and it binds providers of *image generation*
  systems; jaato's image tools are consumers of an upstream model whose
  vendor carries that safeguard.
- **Conformity assessment, CE marking, registration (Articles 43–49).**
  Organisational, and the Annex III procedure is internal control (Annex
  VI) — the dossier is its input, the rest is paperwork.
- **A FRIA questionnaire (Article 27).** The AI Office is to publish the
  template; a framework version would be superseded on publication.
- **GPAI obligations.** jaato provides no model.
- **Audio watermarking in the core.** No dependency-free implementation, and
  "state of the art" is a moving target; the trait is the contract.

## 5. Order of work

| # | Mechanism | Article | Binds | Size | Depends on |
|---|---|---|---|---|---|
| 1 | `regulatory:` profile block + `validate` escalation | 6(4), 25(1)(c) | now (documentation of the assessment) | S | — |
| 2 | `disclosure` instruction piece + first-interaction announcement | 50(1), 50(5) | **now** | S | 1 |
| 3 | `generated_by` stamp on outbound media and attachments | 50(2) | **now** (2 Dec 2026 for pre-existing systems) | S | — |
| 4 | Ledger reaches disk | 12(1), 19(1) | now (a defect regardless) | S | — |
| 5 | `explain oversight` topic + `jaato-doctor` stop line | 13(3)(d), 14(4) | now (small, computed) | S | — |
| 6 | Audit-record contract, `record_keeping:` retention, `explain audit` | 12, 13(3)(f), 19, 26(6) | 2 Dec 2027 | M | 4 |
| 7 | Dossier generator + the 25(4) component pack | 11, 13, 25(4), Annex IV | 2 Dec 2027 (the pack: now, for any commercial supply) | M | 1, 5, 6 |
| 8 | Incident register | 72, 73 | 2 Dec 2027 | M | 6 |
| 9 | Output-marker trait + C2PA sidecar for files | 50(2) | now, feasibility-bounded | M | 3 |
| 10 | Memory provenance + curation knob | 15(4) | 2 Dec 2027 | M | 3 |
| 11 | Eval results → dossier | 15(3), 9(8) | 2 Dec 2027 | S | 7 |
| 12 | `sha256-chain` integrity | 73(6), #507 | 2 Dec 2027 | S | 6 |

Items 1–5 are small, independent of each other, and two of them are overdue
for any jaato application already talking to people in the EU. Items 6–8
are the ones that let a deployer adopt a jaato application into an Annex III
process without re-deriving what the framework does from its source, and
they are the same items that make a production deployment debuggable, which
is why several of them were already half-built. The remainder are
refinements with a clear owner.

## 6. Sources

- Regulation (EU) 2024/1689, Official Journal L, 12 July 2024 — EUR-Lex
  CELEX:32024R1689 (the text read for this document).
- Regulation (EU) 2026/1744 (the Digital Omnibus on AI), OJ 24 July 2026,
  in force 27 July 2026 — dates and changes per the Cloud Security Alliance
  research note "EU AI Act's High-Risk Deadline: Deferred, Not Cancelled",
  Gibson Dunn "EU AI Act Omnibus Agreement — Postponed High-Risk Deadlines
  and Other Key Changes", Morgan Lewis "EU Approves Delays and Other
  Amendments to Certain EU AI Act Obligations" (June 2026), and Travers
  Smith "EU agrees to delay key AI Act compliance deadlines" (May 2026).
