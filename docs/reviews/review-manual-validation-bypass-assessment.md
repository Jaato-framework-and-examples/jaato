# Review: Assessment on Avoiding Manual Validation Bypass

**Date:** 2026-02-28
**Context:** User assessment proposing system instruction improvements to prevent the model from bypassing subagent-delegated validation by performing it manually.

---

## Assessment Summary

The assessment identifies that the model can bypass validation subagents by manually marking validation steps as completed, and proposes system instruction changes to enforce delegation. Proposed changes target Principle 19 (Evidence-Based Completion), plan workflow rules, plan creation guidance, step status rules, and tool schema enforcement.

---

## Verdict: Correct Diagnosis, Wrong Treatment Layer

The assessment correctly identifies a real enforceability gap. However, it predominantly proposes **instruction-level** (soft) solutions for what is fundamentally a **code-level** (hard) problem. Instruction-level fixes are exactly what failed in the first place — adding more instructions to say "don't do the thing" when the model already ignores existing instructions saying "don't do the thing."

---

## What the Assessment Gets Right

### 1. The Root Cause Is Real

`setStepStatus` in `jaato-server/shared/plugins/todo/plugin.py` (line 952) accepts `status='completed'` from any caller unconditionally. The only guard is instruction text at lines 781-792 stating "NEVER fabricate completion" — a soft constraint the model can bypass.

### 2. The Audit Trail Gap Is Real

When the model marks a validation step as completed manually, the `result` field contains whatever text the model wrote. There is no structural difference between legitimate subagent output and fabricated manual "validation."

### 3. The Fallback Path Observation Is Accurate

When a validator subagent is slow or fails, nothing prevents the model from writing `setStepStatus(step_id, status='completed', result='Validated manually')` and continuing.

---

## Where the Assessment Falls Short

### 1. The Cross-Agent Dependency System Already Exists

The todo plugin already provides `completeStepWithOutput` and `addDependentStep`:
- Subagents return `{passed: true/false}` via `completeStepWithOutput`
- Parent steps blocked on `depends_on` auto-unblock with `received_outputs`
- System instructions at lines 750-767 already document the "Validation & Iteration Pattern"

The assessment proposes a weaker version of existing functionality. The problem isn't missing mechanisms — it's that the model doesn't always use them. More instruction text won't solve instruction-ignoring behavior.

### 2. Tool Schema Enforcement: Right Direction, Under-specified

The proposal to add `requires_validator` and `validation_only` fields, then block completion without `validator_subagent_output`, is the correct direction. But:
- Where does `validator_subagent_output` come from? The todo plugin uses thread-local context; subagent completions arrive via `TaskEventBus`, not stored on steps (unless `completeStepWithOutput` + dependencies were used).
- `setStepStatus` schema has no place for these fields — they'd need to live in `createPlan` step definitions, requiring `TodoStep` model changes in `jaato_sdk/plugins/todo/models.py`.

### 3. Hardcoded Validator Names Are Fragile

Mapping "Tier 2 validation → MUST use `validator-tier2-java-spring`" in system instructions couples instructions to profile names. If profiles are renamed or new validators added, instructions become stale. The existing architecture discovers profiles dynamically from `.jaato/profiles/` — instruction-level name coupling undermines this.

### 4. Timeout Auto-Fail Lacks Mechanism

"Auto-fail step after validator timeout" is stated without explaining how. The todo plugin has no timer infrastructure. Subagent timeouts are managed by `max_turns` on `SubagentProfile`, not by plan tracking. There's no callback from subagent termination to the parent's todo plugin.

---

## What Should Actually Be Done

### A. Add `validation_required` Flag to `TodoStep`

When `createPlan` includes steps matching validation patterns ("validate", "verify", "check", "tier-N"), flag them as `validation_required=True`. This is a model change in `jaato_sdk/plugins/todo/models.py`.

### B. Hard Gate in `_execute_set_step_status`

Before allowing `step.complete(result)`, enforce:

```python
if step.validation_required and not step.has_received_outputs():
    return {"error": "Validation steps require subagent output. "
                     "Use addDependentStep + completeStepWithOutput pattern."}
```

This leverages the existing cross-agent dependency system rather than inventing a parallel one. `received_outputs` on blocked steps proves a subagent ran and returned data.

### C. Behavioral Detection via Reliability Plugin

The reliability plugin already detects `ANNOUNCE_WITHOUT_ACTION`. A new pattern `MANUAL_VALIDATION_BYPASS` could detect when the model calls `setStepStatus(completed)` on a validation step without having spawned a subagent in preceding turns. This triggers a `direct` nudge rather than silently allowing it.

### D. Prerequisite Policy for Validation Steps

Instead of more instruction prose, add a `prerequisite_policy` (the reliability plugin already supports these) requiring `addDependentStep` before `setStepStatus(completed)` for validation-tagged steps.

---

## Proposal-by-Proposal Verdict

| Assessment Proposal | Verdict | Reason |
|---|---|---|
| Instruction text for Principle 19 | **Insufficient** | Model already ignores existing instruction text |
| Plan workflow rules | **Partially redundant** | Cross-agent coordination pattern already documented (lines 750-767) |
| Tool schema enforcement | **Right direction** | But needs concrete implementation targeting `TodoStep` + `_execute_set_step_status` |
| Step status rules | **Already exist** | Lines 781-797 already say this — the problem is enforcement, not documentation |
| Timeout auto-fail | **Good idea, no mechanism** | Requires timer infrastructure or integration with subagent `max_turns` |
| Validator name hardcoding | **Fragile** | Should use profile discovery, not instruction-level name coupling |

---

## Bottom Line

The assessment correctly identifies that validation delegation is unenforceable today, but proposes mostly instruction-level solutions for a code-level problem. The existing cross-agent dependency system (`addDependentStep` + `completeStepWithOutput` + `received_outputs`) is already 80% of the solution. The missing 20% is a **hard gate in `setStepStatus`** that refuses to complete validation-tagged steps without `received_outputs` evidence.

**Instruction text tells the model what it *should* do. Code enforcement ensures it *must*.**
