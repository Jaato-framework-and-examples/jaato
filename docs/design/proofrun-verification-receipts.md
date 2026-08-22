# Verification Receipts — ProofRun Integration Feasibility

**Status:** feasibility assessment / design proposal — nothing implemented.
**Subject:** [yebiguo/ProofRun](https://github.com/yebiguo/ProofRun) (Go, MIT, v0.3, pre-1.0).
**Verdict:** adopt the *idea*, decline the *dependency*. Run a zero-code spike
first (§6), and only then build a native plugin (§7).

---

## 1. What ProofRun is

A ~66-commit Go CLI that answers one question: *did this check actually run
against the code that exists right now?*

```bash
proofrun run test -- pytest     # runs pytest for real, binds exit code to code state
proofrun status                 # test  PASS  (exit 0, 1841ms)
# ... agent edits one byte anywhere in the repo ...
proofrun status                 # test  STALE (last run: pass — code changed since)
```

Mechanism, in full:

| Piece | Detail |
|---|---|
| **Execution** | `exec` of an **argv array** — never through a shell. Records exit code + duration. Nothing parses test output; exit code is the entire verdict. |
| **Fingerprint** | `git HEAD` + `sha256(git diff HEAD ‖ contents of untracked non-ignored files)`. |
| **Storage** | `.proofrun/receipt.json`, schema `proofrun/v2`, one entry per check. |
| **Integrity** | HMAC-SHA256 per entry, key at `.proofrun/secret`, generated on first use, excluded via `.git/info/exclude`. |
| **States** | `PASS` / `FAIL` (stored, from exit code) and `STALE` / `NOT RUN` (**computed at read time**, never stored). |
| **Gate** | `status --strict` exits non-zero if a `required: true` check isn't PASS. A GitHub Action re-runs everything from scratch, trusting no checked-in receipt. |

Config is a flat `.proofrun.yml`:

```yaml
checks:
  test:  { command: [pytest],          required: true }
  build: { command: [npm, run, build], required: true }
  lint:  { command: [ruff, check, .],  required: false }
```

Explicit non-goals (their `AGENTS.md`): no LLM calls anywhere, no output
parsing, no code-quality judgement, no auto-fix, no network.

---

## 2. The gap it targets in jaato

We already own most of the machinery and none of the idea. Concretely:

| We have | Where | What it does **not** do |
|---|---|---|
| `completion_processors` validators with `context.tool_calls` — a per-call ledger (`name`/`args`/`result`/`success`/`call_id`/`turn_index`/`enrichment_metadata`) that a kb `validate()` can cross-check against a completion payload, blocking completion via `on_error: fail_completion` | `shared/completion_processors.py`, `shared/dynamic_instructions.py:179` | In-memory, **per session**, and **has no code-state binding**. A turn that ran pytest and a later turn that rewrote the file under test are indistinguishable to it. |
| `reliability` prerequisite policies (`prerequisite_tool` → `gated_tools`, `lookback_turns`, graduated nudges) | `shared/plugins/reliability/`, `.jaato/reliability-policies.json` | Enforces **ordering**, not **currency**. "You called the test tool at some point" stays true forever. |
| `waypoint` (diverged-from backups) and `artifact_tracker` | `shared/plugins/waypoint/`, `.../artifact_tracker/` | Conversation-scoped bookkeeping for restore/reporting — not a verdict-invalidating fingerprint. |
| `cli` plugin — `shlex.split` to argv, returns `{stdout, stderr, returncode}` | `shared/plugins/cli/plugin.py:1427` | Has the raw material for a receipt (argv + exit code) and discards it after the turn. Note it does **not** currently measure wall-clock duration (`estimate_duration` is a *hint* for the UI, not a measurement). |

**The missing primitive is automatic invalidation.** Nothing in the stack makes
a "tests pass" claim decay when a file changes three turns later. That is
exactly the cascade failure mode: stage N verifies green, stage N+1 edits code,
the final completion payload still asserts green — and every check we have
(`tool_calls` ledger, prerequisite policy, schema validation) agrees with it,
because each one only ever asked *did the call happen*, never *does it still
count*.

---

## 3. Why not vendor the binary

### 3.1 The signing is close to worthless in our threat model

ProofRun's own README is candid: tamper-evident, not tamper-proof;
machine-local, not portable evidence; no rollback/replay defense. The agent
that runs `proofrun run` holds workspace write access, therefore holds
`.proofrun/secret`. Our adversary is a **confabulating LLM**, not an HMAC
forger. Against that adversary the fingerprint does ~all the work and the
signature does ~none.

Worse, it's a weak substitute for a primitive we *already have and they don't*:
the **daemon/runner tier split**. `PLUGIN_TIER` is real and enforced —
`model_provider`, `gc`, `cache`, `background`, and the `*_auth` plugins run
daemon-side; `cli`, `file_edit`, `reliability`, `subagent` run runner-side. Put
receipt *writing* in the daemon tier and, under the confined-runner path, the
agent cannot forge a receipt because the kernel says so. That is a real
guarantee. An HMAC key sitting in a directory the agent can read is a gesture
toward one.

### 3.2 The fingerprint is far too coarse for an agentic loop

Repo-wide `git diff HEAD` + untracked, with no per-check path scoping in v0.3.
The agent touches `README.md` → the pytest receipt goes STALE. On our workloads
— where the agent edits continuously by construction — that is a permanent
STALE and therefore no signal at all. This is the single biggest design
mismatch, and it is not a configuration problem; scoping doesn't exist.

### 3.3 Operational cost, for a fingerprint function

- **New compiled non-Python runtime.** Per-platform release download +
  checksum verification, in a tree with zero precedent for shipping a binary.
- **MSYS2/Windows.** A Go binary at the argv/path boundary is precisely the
  class of problem `docs/path-boundary-pattern.md` exists to contain.
- **Requires git.** Our sessions run in arbitrary workspaces — non-git dirs,
  sandboxed and confined runners, ephemeral cascade scratch dirs.
- **No Python client.** You shell out and parse `report --json` regardless, so
  the purchase is: a fingerprint function, a JSON schema, and a GitHub Action.
- **Maturity.** Pre-1.0, single author, 14 stars. The receipt schema already
  broke once (no migration for pre-v0.3 receipts — they silently read as
  `NOT RUN`). Their v0.4 roadmap items (structured runner output) and their
  documented known limitation (`.proofrun.yml` can be weakened by the same PR
  it gates) are things we would want settled on day one.

### 3.4 What *is* worth taking

The insight, stated once and clearly: **bind the verdict to a fingerprint of
the code, recompute on read, and let staleness be automatic rather than
remembered.** Plus two design choices worth copying verbatim:

1. **Argv-exact, never a shell string.** A check declared as
   `pytest -k "foo bar"` must not be satisfiable by something that merely
   flattens to similar text. Their pre-release case study is exactly this bug:
   a misquoted argument made a check run zero tests and still report PASS.
2. **`STALE` and `NOT RUN` are computed, never stored.** There is no writable
   field an agent can set to make a check look fresh. The only way to write
   `pass` is to actually exit 0.

---

## 4. Recommendation

| | Score | Reasoning |
|---|---|---|
| **The idea** (fingerprint-bound verdicts, automatic STALE) | **8/10** | Closes a real hole in the cascade completion path that nothing else we have addresses. |
| **The dependency** (the Go binary) | **3/10** | Buys ~40 lines of Python we can write, at the cost of a compiled runtime, a git requirement, and an over-coarse fingerprint we'd have to work around anyway. |

Adopt the idea. Credit ProofRun in the plugin README. Skip the binary.

---

## 5. Threat model — say this out loud

The receipt system defends against **an agent that is wrong, not an agent that
is malicious.**

In scope:
- The model asserts green from inference rather than observation.
- The model genuinely ran the check, then edited code and never re-ran.
- A cascade stage inherits a sibling stage's stale verdict.
- A model hand-writes a plausible completion payload with no supporting call.

Out of scope (and we should not pretend otherwise):
- A model that deliberately runs `pytest --co -q` and records it as `test`.
  Mitigation is argv-exactness against a declared command, which is a
  *correctness* control, not a security one.
- Anything the runner can write, the runner can forge. The **only** real
  integrity boundary is tier placement + AppArmor (§7.4), not a checksum.

---

## 6. Step 1 — zero-code spike (do this first)

ProofRun is a CLI. We can evaluate the entire idea without writing a line of
framework code, using two extension points we already ship:

1. Add `.proofrun.yml` at repo root declaring `test` and `lint`.
2. Let the agent invoke `proofrun run` / `proofrun status` through the existing
   `cli` plugin — no new tool, no permission model change.
3. Add **one** `completion_processors` entry pointing at a kb script under
   `.jaato/scripts/processors/`:

```yaml
# .jaato/profiles/<profile>.yml
completion_processors:
  - script: processors/verify_receipts.py
    on_error: fail_completion
    description: Reject a completion claiming green when proofrun says otherwise.
```

```python
# .jaato/scripts/processors/verify_receipts.py
import json, subprocess

def validate(payload: dict, context) -> list[str]:
    """Block completion when the payload asserts green but the receipt disagrees.

    Shells `proofrun report --json` — ProofRun's *trusted view*, which drops
    entries whose signature doesn't verify. Never reads receipt.json directly:
    the raw file is untrusted storage, and trusting `status == "pass"` off disk
    reintroduces exactly the false-PASS this whole mechanism exists to close.
    """
    if not payload.get("tests_pass"):
        return []
    proc = subprocess.run(
        ["proofrun", "report", "--json"],
        cwd=context.workspace_path, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return [f"proofrun report failed (exit {proc.returncode}): {proc.stderr.strip()}"]
    checks = json.loads(proc.stdout).get("checks", {})
    return [
        f"payload claims tests_pass but check '{name}' is {c.get('status', 'NOT RUN').upper()}"
        for name, c in checks.items()
        if c.get("required") and c.get("status") != "pass"
    ]
```

**Cost:** an afternoon. **What it buys:** empirical evidence on the one
question that decides everything downstream — *does the STALE signal fire often
enough to be useful, and rarely enough to be trusted?* If §3.2 is right, this
spike goes permanently STALE within a few turns and we have learned that for
free. Do not skip to §7 without running this.

---

## 7. Step 2 — native `verification` plugin (only if the spike earns it)

A Python plugin under `jaato-server/shared/plugins/verification/`, no Go
dependency. Five pieces, in dependency order.

### 7.1 Fingerprint (~40 lines)

```python
def fingerprint(workspace: Path, paths: list[str] | None = None) -> Fingerprint:
    """HEAD + sha256 over the scoped working-tree delta.

    paths=None reproduces ProofRun's repo-wide behaviour. A non-empty list
    scopes both the diff and the untracked-file sweep to those pathspecs —
    the fix for §3.2.
    """
```

`git rev-parse HEAD`, `git diff HEAD -- <pathspec>`,
`git ls-files --others --exclude-standard -- <pathspec>`, hash the lot.
**Non-git workspaces:** no silent fallback — the plugin reports
`unavailable: not a git repository` and every check reads `NOT RUN`. Fail
loud, per house rule.

### 7.2 Path scoping — the reason to build rather than adopt

```yaml
# .jaato/verification.yml
checks:
  test:
    command: [pytest, jaato-server/shared/tests/]
    paths:   [jaato-server/**, jaato-sdk/**]     # docs edits don't invalidate
    required: true
  lint:
    command: [ruff, check, .]
    paths:   ["**/*.py"]
    required: false
```

This is the whole ballgame. Un-scoped, the mechanism cries wolf until it's
ignored; scoped, a STALE on `test` means *code the tests cover has changed*,
which is a claim worth blocking a completion over.

### 7.3 Automatic recording via a tool trait

The agent must not have to *remember* to use a special tool — an agent that
forgets to record is the same agent that forgets to re-run.

Add `TRAIT_VERIFICATION_CHECK` alongside the existing constants in
`jaato-sdk/jaato_sdk/plugins/model_provider/types.py` (`TRAIT_FILE_WRITER`,
`TRAIT_GREPPABLE_CONTENT`, `TRAIT_FRAMEWORK_LEVEL`, `TRAIT_REPLAY_SAFE`,
`TRAIT_UNTRUSTED_CONTENT`). Session already queries
`registry.get_tool_traits(tool_name)` to pick an enrichment strategy; extend
that path so any `cli` invocation whose argv **matches a declared check
element-for-element** records a receipt automatically.

Two prerequisites, both small and both real work:
- `cli` must return a **measured** duration. Today `RunResult` carries
  `stdout`/`stderr`/`returncode`/`truncated`/`timed_out`/`cancelled` and no
  timing; `estimate_duration()` is a pattern-matched hint, not a measurement.
- Comparison must be on the argv list, after `shlex.split`, never on the
  reconstructed string (§3.4). Commands that route through
  `requires_shell()` are **not eligible** to satisfy a check — a pipeline has
  no single exit code worth binding.

### 7.4 Receipts written daemon-side

Store at `.jaato/receipts.json`, written by a **daemon-tier** component. The
runner-tier agent produces the exit code; the daemon binds it to a fingerprint
and persists it. Under the confined-runner path the agent then cannot write a
receipt at all — kernel-enforced, which is strictly stronger than ProofRun's
local HMAC, and it needs no key management.

This is the one place our architecture makes the guarantee *better* rather than
merely equivalent, and it is the main engineering argument for building over
adopting. It is also the piece with real design work in it: `cli` is
`PLUGIN_TIER = "runner"` today, so the recording hop crosses the tier boundary
and needs a narrow, well-specified RPC — not a general "write arbitrary
receipt" call, or we have rebuilt the forgeable version with extra steps.

Signing is then **optional and off by default**: on the confined path it is
redundant; on an unconfined single-user dev host it is theatre. If we add it
later, it should be for the portable-evidence case (CI), where it needs real
keys, not a machine-local secret.

### 7.5 Turn-level enrichment — cheapest, highest leverage

An `EnrichmentPlugin` (`PLUGIN_KIND = "enrichment"`,
`subscribes_to_prompt_enrichment()` + `enrich_prompt()`) that injects current
status into every turn:

```
Verification status:
  test   STALE  — jaato-server/shared/plugins/cli/plugin.py changed since last run
  lint   PASS
```

The model watches its own claim decay in real time, in-context, before it
drafts a completion payload. Costs a few dozen tokens per turn and needs
nothing from §7.3 or §7.4 to be useful — it can ship on top of the §6 spike.

### 7.6 Graduated enforcement via `reliability`

Rather than a bespoke blocker, express "signalled completion with a required
check STALE" as a `PrerequisitePolicy` in `.jaato/reliability-policies.json`
with the existing `minor`/`moderate`/`severe` nudge ladder — `direct` nudge
first, `interrupt` on repeat. Reuses `NudgeInjector` wholesale and keeps every
enforcement knob in the one file operators already tune.

---

## 8. Suggested sequencing

| Phase | Scope | Gate to proceed |
|---|---|---|
| **0** | §6 spike: `.proofrun.yml` + one completion processor | STALE fires on real edits without going permanently STALE |
| **1** | §7.1 fingerprint + §7.2 scoping + §7.5 enrichment, receipts runner-side, no trait | Enrichment visibly changes model behaviour |
| **2** | §7.3 trait-based auto-record + measured duration in `cli` | Recording works without the agent cooperating |
| **3** | §7.4 daemon-tier writes + §7.6 reliability policy | — |

Phase 1 is independently useful and ships without touching `cli`, the tier
boundary, or the trait system. Phases 2–3 are where the cost is, and neither is
worth paying before phase 1 has proven the signal.

## 9. Open questions

1. **Cascade semantics.** Are receipts per-workspace or per-cascade-run?
   Subagents share a `JaatoRuntime` but hold isolated `JaatoSession`s — a
   disk receipt is the natural cross-session evidence bus, but two stages
   editing concurrently in one workspace will invalidate each other's
   verdicts. Possibly wants per-stage worktrees, which is a much larger change.
2. **Does the completion payload schema get a reserved key?** e.g. a
   framework-validated `verification: {test: pass}` block, versus leaving kb
   authors to name their own field and wire their own processor. See
   `docs/design/payload-schema-conventions.md`.
3. **Untracked-file noise.** `.jaato/logs/`, `cascade_state/`, and pytest
   caches all land in the untracked sweep. Scoping (§7.2) mostly handles it;
   an explicit ignore list may still be needed.
4. **CI story.** ProofRun's GitHub Action is the part local receipts can't
   replace — a third party should trust an independent re-run, never a
   checked-in artifact. Our equivalent is just "run the checks in CI", which
   we already do. There may be nothing to build here at all.

---

## 10. Credit

The fingerprint-binding idea, the four-state model, the computed-not-stored
treatment of `STALE`/`NOT RUN`, and argv-exact command matching are all
ProofRun's (MIT). If we build §7, its README should say so.
