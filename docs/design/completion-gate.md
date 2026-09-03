# The Self-Bounding Completion Gate

`completion_processors` is the framework's fix-until-it-passes loop: profile-declared
Python whose `validate(payload, context)` runs when the agent calls `signal_completion`,
after the payload has passed `completion_payload_schema`. A non-empty error return
blocks the completion and hands the agent every string as a `validation_failed` result,
so it fixes the underlying problem and signals again — inside its own `max_turns`.

It is the output-side twin of the `{{!py:...}}` prefetch hook, and it was almost
undocumented. `jaato-scaffold explain` had a `prefetch` scope and, for this,
nothing: `grep -rn completion_processor jaato-server/shared/scaffold/` returned zero
files. So every author of a self-correcting agent rediscovered the shape, and the
failure modes are not the ones you would guess.

This document records what a working processor had to get right, each rule attached to
the incident that produced it, and which of those rules are now the framework's job
rather than the author's.

**Start here instead of reading this file end to end:**

```
jaato-scaffold explain completion                       # the contract
jaato-scaffold new processor --name <n> --workspace .   # a working one
```

The explain scope is computed from the framework (the dataclass fields, the parser's
own vocabularies, `ProcessorResult.__annotations__`), so it cannot drift from the code
the way this prose can.

---

## 1. The retry loop is the framework's — do not build a second one

The first attempt at self-correction (#731) added a grader-feedback retry loop to the
eval runner: a whole extra session round-trip, a second `--max-attempts` knob beside
`max_turns`, and it worked only for `jaato_eval` callers. That PR was closed on finding
`completion_processors`, which is the same loop with no new framework code, no second
budget, and it works for any driver.

**`max_turns` IS the retry budget.** There is no second attempts knob and there should
not be one.

## 2. The loop does not terminate on its own

Nothing bounded it. The processor refuses, the agent re-claims completion, the processor
refuses again. Observed 2026-09-01: **seven refusals in 156 seconds**, some nine seconds
apart, every one reporting the same two errors, with no work in between. The arm ended
BLOCKED having spent its budget on the loop, where the run before it had reached a
graded verdict.

Nothing upstream catches this. `MAX_COMPLETION_NUDGES` bounds the *opposite* direction
(an agent that stops **without** signalling), and was itself unbounded until #767.

**Now:** a processor entry declares its own ceiling.

```yaml
completion_processors:
  - script: scripts/processors/acceptance.py
    name: acceptance
    max_refusals: 3
    on_exhausted: allow      # allow | fail
```

`max_refusals` is the number of times *this gate* may block. Absent, the behaviour is
the old one — unbounded — because the bound must be opt-in: a profile that says nothing
must not silently acquire a ceiling that lets an unfinished completion through.

## 3. The counter has a declared home now

Authors used to keep it in a module-level global:

```python
_MAX_REFUSALS = 3
_refusals = 0

def validate(payload, context):
    global _refusals
    ...
```

That worked, on an undocumented guarantee: `LifecycleTools` loads processors once per
session and caches them, so module state survives across `signal_completion` calls. If
the framework had ever reloaded per call, every such processor would have gone on
working while its ceiling silently stopped existing (#765).

**Now:** the counter lives on the framework's per-session `LoadedProcessor`, the caching
is a stated contract with a guard behind it
(`test_completion_processor_refusal_budget.py::test_processors_are_loaded_once_per_session`),
and a generated processor carries no counter of its own.

## 4. On exhaustion, `allow` — usually

At the ceiling, `on_exhausted: allow` downgrades the processor's errors to warnings and
lets the unfinished completion stand. The checks still fail, and whatever grades the run
says so a moment later with the same script. That is deliberate: **a FAIL verdict
carries information and a BLOCKED arm carries none.**

For a caller where an unfinished completion is worse than none — one that writes to a
shared store, say — `on_exhausted: fail` keeps blocking. Both are real choices; the
point is that it is a choice, and the profile is where it gets made.

## 5. Never accept on a broken gate

With `--all`-style semantics, a checking script prints one `<check>: <reason>` line per
failure and nothing on success — so stdout **is** the error list. An empty stdout with a
non-zero exit means the checking script itself broke, and returning "no errors" there
waves the completion through on a gate that is not running.

This is the most repeated defect class in this codebase: **an error path returning the
same value as success.** Three instances in one session — a grader crashing on a missing
import and being read as a pass, a `git diff` failing and being read as "nothing
changed", and this.

**Now the framework holds the same line.** A load failure, a raise, a malformed return
and a failed write all block, spend no refusal, and are never waved through by
exhaustion. A `validate` that falls off the end returning `None` is a malformed return,
not a pass — it used to be indistinguishable from the internal "already reported"
sentinel.

## 6. Separate a wrong answer from an environment fault

A missing `acceptance.sh`, an absent `issue_id`, a checks timeout: none is something the
agent's fix can address. A retryable message about an unfixable fault burns the whole
budget without ever producing a verdict.

**Now:** `ProcessorResult` has a fourth channel.

| channel | blocks completion? | spends a refusal? |
|---|---|---|
| `errors` | yes, per `on_error` | yes — one per **call**, not per message |
| `faults` | once per session | never |
| `warnings` | never | never |
| `incomplete` | never (gates `is_complete` on a `phase: completeness` processor) | never |

A fault blocks exactly one round-trip — the one the agent needs to record it in its
payload — and is advisory afterwards. Blocking repeatedly on a condition no retry can
clear is the non-terminating loop of rule 2 wearing a different hat.

## 7. Write the strings as instructions for the retry

The return value is read by a model that is about to try again, not by a human reading a
log. Name the failure and what to do about it. The framework appends the attempts
remaining, so an author does not have to:

```
Your fix does not pass this task's acceptance checks yet. Fix the underlying cause of
each failure below in repo/, commit again, then call signal_completion again. Do not
report success while any of these still fails.
[acceptance] You have 2 further attempt(s) at this gate before this completion is
accepted as it stands and processed unfinished. Re-sending the same claim without
changing anything spends one.
```

## 8. One script for the in-session gate and the post-hoc graders

Run the same checks the graders run, selected by the same parameter. Two scripts cannot
end up grading different things; one script can only be right or wrong. A two-tier split
(generic checks plus a per-case script) keeps the harness case-agnostic while the gate
stays specific.

---

## Why a generator and not only documentation

A hand-written processor got rules 2, 5, 6 and 7 wrong on its first pass. The generated
half of that same harness got them right, because a template encodes a contract where a
docstring merely asserts one.

`jaato-scaffold new processor --name <n>` emits a module with the four-channel return,
the broken-gate discrimination, the fault split, and no refusal counter of its own —
then loads it through the framework's own `load_processors` and drives it through
`invoke_processors`, so a generated processor that would not load, or that would wave a
completion through, fails at scaffold time rather than in a run.

## Why this file is not the guard

Scaffold templates rot fast, and there was a live example: an archetype doc asserted
that a client "waits on the FIRST of `{TURN_COMPLETED, SESSION_TERMINATED}`" after #767
had changed that, on a branch touching zero scaffold files. Prose cannot notice.

So the claims above are enforced by tests that read the framework:

| guard | what it would catch |
|---|---|
| `shared/tests/test_completion_processor_refusal_budget.py` | the ceiling not bounding, a broken gate being waved through, a fault spending a refusal, the load-once caching going away |
| `shared/tests/test_scaffold_completion_contract.py` | `explain completion` drifting from `CompletionProcessor`, `ProcessorResult` or the parser's vocabularies; the generator regressing to a hand-rolled counter; the generated processor not actually terminating |
| `shared/tests/test_scaffold_archetype_docs.py` | the `processor` archetype's declared output drifting from what `new` writes |

Each of the first two also declares a `REVERSIONS` entry, so
`test_every_guard_detects_its_own_reversion` puts the defect back and checks the guard
goes red.
