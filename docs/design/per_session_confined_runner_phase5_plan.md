# Phase 5 Implementation Plan — Production Hardening

**Status:** Draft, awaiting peer review.

**Parent design:** `per_session_confined_runner.md` (Phase 1 design).
**Predecessor:** Phase 4 (merged 2026-05-11, PR #58, merge commit
`d8333af8`).
**Authoritative source for §4.3 hardening surfaces:**
`phase4_implementation_audits.md` §4.3.9 (already enumerates the 10
items with full design context).  This plan **references that
ledger by section/item number** rather than duplicating its
content.

Phase 4 delivered the §3.11 isolated-subagent opt-in, the J.A/J.B
closures (§4.1 / §4.2), the description-callback bridge (§4.4), and
the multi-turn UX investigation (§4.7).  Performance baseline (§4.5)
and idle-runner-shutdown knob (§4.6) were re-deferred to Phase 5 to
absorb the §4.3 scope expansion.

Phase 5 picks up:

- The **§4.3 hardening surfaces** consolidated in
  `phase4_implementation_audits.md` §4.3.9 (10 items — see ledger
  for the per-item rationale, today's-posture, and acceptance
  criteria).
- The **Phase 4 deferrals** from
  `per_session_confined_runner_phase4_plan.md` §3.5 / §3.6 (§4.5
  performance baseline, §4.6 idle-runner-shutdown).
- The **security-hardening backlog** that accumulated across Phase
  3/4 diagnostic sessions but lives outside the §4.3.9 ledger —
  AppArmor child-subprofile escape vector, per-session egress
  allowlist (memory: `project_backlog_apparmor_child_subprofile`,
  `project_backlog_per_session_egress_allowlist`).
- The **resilience-tier carryover** items the parent design
  flagged as Phase 5 candidates (memory:
  `project_backlog_token_expiry_detection`,
  `project_backlog_gc_exception_recovery`,
  `project_backlog_session_idle_timeout`,
  `project_backlog_subagent_pool_limits`,
  `project_backlog_provider_circuit_breaker`).

Phase 5 is **NOT** Phase 6 (cleanup + cross-platform).  Daemon-side
AppArmor profile, removal of vestigial machinery, and Windows/macOS
support remain Phase 6 scope per the parent design.

---

## 1. Goal + scope

**Goal.** After Phase 5 lands:

- The 10 §4.3.9 hardening items are either shipped or explicitly
  re-deferred with a concrete reason recorded against the ledger.
- Performance baseline is recorded as a CI regression ratchet.
- Idle-runner-shutdown knob is wired and operator-documented.
- The AppArmor escape-vector class is closed (child-subprofile +
  egress allowlist).
- Resilience-tier carryover items have a triaged disposition.
- The audit-discipline pattern (audit-before-code, scope
  re-decision on uncertainty, manual real-host verification,
  audible-failure on opt-in) from Phase 3 / 4 closure recaps is
  applied to every Phase 5 commit.

**Out of scope.**  Phase 6 items (daemon AppArmor profile,
vestigial-machinery removal, cross-platform runner).  New
architectural primitives — Phase 5 is hardening, not expansion;
any "new ability" surfacing during work goes through a scope-
decision audit per Phase 4 §4.3.0 precedent.

---

## 2. Theme structure + critical-path constraint

The work clusters into **four themes by shared blast radius**.
Audit-discipline §4 (each architectural decision needs its own
audit commit) applies **per theme, not per task** — inside a theme,
tasks share machinery and audit context.

### Theme A — Resource-isolation defaults (security)

Three items from `phase4_implementation_audits.md` §4.3.9, plus one
surfaced during §5.1 work:

| Plan | Audit ledger item |
|------|-------------------|
| §5.1 | §4.3.9 item 1 — Default `RuntimeLimits` for isolated subagents (shipped) |
| §5.2 | §4.3.9 item 2 — Nested cgroup structure (parent-bounded sub) — **deferred to Phase 6; visibility-only instrumentation shipped** (`phase5_5_2_nested_cgroup_deferral_audit.md`) |
| §5.3 | §4.3.9 item 4 — Cgroup-leak audit at session shutdown (shipped) |
| §5.1b | §4.3.9 item 11 — Mainline `RuntimeLimits` app-layer passthrough to runner (shipped) |

**Shared audit context:** all four touch the runtime-limits wiring
path.  §5.1 / §5.2 / §5.3 share the
`SessionManager._spawn_isolated_runner` helper chain.  §5.1b sits on
the parallel `runner_spawn.spawn_session_runner` mainline flow but
consumes the same `RuntimeLimits` dataclass + `RunnerSpawner.spawn`
env-passthrough machinery, so the Theme A audit covers it under the
same rules table.

### Theme B — Runtime observability + crash detection (operability)

Two items from §4.3.9 plus the two Phase 4 deferrals:

| Plan | Source |
|------|--------|
| §5.4 | §4.3.9 item 3 — Sub-runner crash detection (EOF, optional heartbeat) |
| §5.5 | §4.3.9 item 9 — Cross-runner event ordering tags |
| §5.6 | Phase 4 plan §3.5 — Performance baseline (carryover) |
| §5.7 | Phase 4 plan §3.6 — Idle-runner-shutdown knob (carryover) |

**Shared audit context:** §5.4 / §5.5 share the cross-runner
forwarding path.  §5.6 / §5.7 share the runner-lifecycle
measurement points.  Two audit-commits — one per pair.

### Theme C — Confused-deputy surface tightening (security)

Two items from §4.3.9 plus two memory-backlog items:

| Plan | Source |
|------|--------|
| §5.8  | §4.3.9 item 10 — `profile_payload` typed model / allow-list |
| §5.9  | §4.3.9 item 5 — Supervisor-declared sub-profile tightening flags |
| §5.10 | memory `project_backlog_apparmor_child_subprofile` — AppArmor escape-vector closure (in-process denylist landed Phase 4; subprocess-side needs child subprofile) |
| §5.11 | memory `project_backlog_per_session_egress_allowlist` — Per-session network-egress allowlist (closes the host-filter hole AppArmor can't reach) |

**Shared audit context:** all four close confused-deputy surfaces
where an "isolated" sub-runner can still reach out beyond its
intended bounds.  Single audit-commit covers the threat model;
§5.9 + §5.10 may compose (supervisor flags applied at the child-
subprofile level).

**§5.11 prerequisite:** a 2-hour spike per memory
`project_backlog_mitmproxy_spike_for_egress` to pick mitmproxy
vs custom-proxy implementation shape.

### Theme D — Isolated-handle lifecycle parity (operability)

Three items from §4.3.9:

| Plan | Audit ledger item |
|------|-------------------|
| §5.12 | §4.3.9 item 6 — `subagent_send` / `cancel_subagent` / `close_subagent` rewire for isolated handles |
| §5.13 | §4.3.9 item 7 — Sub-profile `add_reference_fragment` opt-in |
| §5.14 | §4.3.9 item 8 — Filename collision on rapid sequential spawn (monotonic suffix) |

**Shared audit context:** all three are user-visible operability
gaps in the isolated-runner UX.  Single audit-commit covers the
lifecycle parity table; implementations are independent.

### Resilience-tier carryover (separate from confined-runner themes)

Five items each tracked in their own memory backlog:

| Plan | Memory backlog |
|------|----------------|
| §5.15 | `project_backlog_token_expiry_detection` |
| §5.16 | `project_backlog_gc_exception_recovery` |
| §5.17 | `project_backlog_session_idle_timeout` |
| §5.18 | `project_backlog_subagent_pool_limits` |
| §5.19 | `project_backlog_provider_circuit_breaker` |

These are individually small (each backlog has its own design
sketch); each gets its own commit chain.  No cross-coupling with
themes A–D.

---

## 3. Sequencing recommendation

Themes A–D can run in parallel (independent blast radii).  Within
each theme, items can ship in any order — they share machinery, so
a single audit-commit upfront then code commits can land in any
sequence.

**Suggested ship order** if work is sequential:

1. Theme C first — security closes the most-load-bearing risk
   classes (§5.10 in particular closes a verified escape vector).
2. Theme A second — depends on the threat model Theme C
   establishes.
3. Theme B third — observability is more useful when the
   security primitives are stable.
4. Theme D fourth — operability polish.
5. Resilience-tier items can interleave anywhere; they're
   independent.

**§5.6 perf baseline** can run in parallel to all of the above —
its CI ratchet output is needed before any of the others claim
"no perf regression".  Recommend landing §5.6 audit + initial
baseline AT THE START of Phase 5 so subsequent commits ratchet
against it.

---

## 4. Audit-discipline reminders (carried from Phase 3 / 4 closure)

Per Phase 4 closure recap, the discipline pattern Phase 5
preserves:

1. **Audit before code at every architectural decision point** —
   each theme's first commit is an audit doc that consumes the
   relevant §4.3.9 item(s) by reference.
2. **Scope re-decision on uncertainty** — Phase 4 §4.3.0 precedent.
3. **Manual real-host verification** for security-sensitive
   primitives — Phase 4 §4.3.9 established the playbook pattern;
   Phase 5 should add one per security-sensitive ship (§5.1, §5.10,
   §5.11 specifically).
4. **Audible failure** on explicit opt-in features — Phase 4 §4.3
   peer-review fix (silent-isolation-downgrade → audible error
   envelope) is the canonical example.
5. **Differential test posture validation** — every PR's CI must
   show net-zero new regressions against the merge-base.
6. **Test-pinning the contract, not the bug** — Phase 4 §4.3
   peer-review caught a test that pinned the security-violating
   fallback.  Reviewers flag tests whose assertions describe
   undesirable behavior.
7. **Phase 6+ carryover recorded in each commit's audit section**
   mirroring §4.3.9's consolidated list.

---

## 5. Acceptance gate

Phase 5 closes when:

- The 10 items in `phase4_implementation_audits.md` §4.3.9 are
  each marked **shipped** (with commit reference) or
  **re-deferred** (with concrete reason) in the audit ledger.
- Performance baseline (§5.6) is a live CI ratchet — regressions
  block merge.
- AppArmor child-subprofile (§5.10) closes the escape-vector
  class; verified on a cgroup-v2 + AppArmor-enabled real host
  per the manual-verification playbook pattern.
- Per-session egress allowlist (§5.11) has a shipped
  implementation OR a re-defer-to-Phase-6 audit explaining why.
- Resilience-tier carryover (§5.15–§5.19) has a triaged
  disposition per item recorded in the ledger.

---

## 6. Out of scope (clarifying boundaries)

- **Daemon-side AppArmor profile** — Phase 6 (parent §6.3).
- **Vestigial-machinery removal** (legacy in-process tool
  execution paths now unused post-§7c) — Phase 6.
- **Cross-platform runner** (Windows / macOS) — Phase 6.
- **New architectural primitives** — Phase 5 is hardening.  Items
  that would expand the surface need their own scope-decision
  audit per Phase 4 §4.3.0 precedent.
- **TS SDK refresh** — separate sub-track tracked in memory
  backlog `project_backlog_sdk_feature_parity`.

---

## 7. Phase 6 preview (for context, not commitments)

Recorded so Phase 5 audits can flag Phase-6-belonging items
explicitly:

- Daemon AppArmor profile (parent §6.3).
- Removal of in-process tool-execution paths (parent §6.1).
- Cross-platform runner via platform-specific sandbox primitives.
- Multi-tenant acceptance gate (parent §6.4) — per-operator
  policy store, capability-based fragment loading.
- **Nested cgroup layout** (§4.3.9 item 2 — deferred from Phase 5
  §5.2).  Today's sibling structure means sub cgroups don't
  compose under parent's bounds.  True nesting (`/main/`
  sub-cgroup + `subtree_control`) requires parent-runner PID
  migration + dual code paths through `CgroupsManager`; ship
  when a multi-tenant deployment files a request for parent-
  bounded composition.  Phase 5 §5.2 shipped visibility-only
  instrumentation so operators can see where nesting WOULD
  have mattered.
- **`max_output_bytes` / `max_output_chars` naming consistency.**
  `RuntimeLimits.max_output_bytes` (the profile field) maps to
  `RunnerSpawner.spawn(max_output_chars=...)` → env var
  `JAATO_RUNNER_MAX_OUTPUT_CHARS` → cli plugin's `_max_output_chars`
  attribute.  For ASCII the two are interchangeable; for multi-byte
  content they're not.  Surfaced during Phase 5 §5.1b peer review;
  preserve mapping today, rename to one or the other in Phase 6
  cleanup.

Phase 5 audits SHOULD flag any item drifting toward this list and
re-decide rather than absorb.
