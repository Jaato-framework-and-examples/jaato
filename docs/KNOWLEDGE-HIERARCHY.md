# Knowledge Hierarchy Design

This document explains the rationale behind how knowledge assets (ADRs, ERIs, Modules, Skills) are used by different agent profiles.

## The Hierarchy

```
ADR (Architecture Decision Record)
 │   "Why" - Strategic decisions, constraints, rationale
 │
 └──► ERI (Enterprise Reference Implementation)
       │   "What/How" - Concrete patterns that embody ADR decisions
       │
       └──► Module (Reusable Templates)
             │   "Building blocks" - Parameterized code templates
             │
             └──► Skill (Executable Workflow)
                   "Recipe" - Step-by-step execution with module composition
```

## Key Insight: ADRs Are Redundant for Execution

When an agent executes a skill, it's essentially **following a recipe**:

1. Read the Skill specification (workflow)
2. Apply Module templates (code generation)
3. Reference ERI for edge cases (patterns)

The agent doesn't need to know *why* pagination defaults to 20 items per page (ADR decision). It just needs to know *that* it should use 20 items per page (ERI pattern, Module template).

**The ADR's "truth" is already baked into the ERI.**

A well-written ERI embodies all the relevant ADR decisions in its concrete implementation patterns. The rationale has been "compiled down" into the reference code.

## When ADRs Add Value

ADRs become valuable when the agent needs to:

| Scenario | Why ADRs Help |
|----------|---------------|
| **Validate compliance** | Need to know constraints to check against |
| **Handle edge cases** | ERI might not cover; ADR principles help extrapolate |
| **Resolve conflicts** | When patterns seem contradictory, ADR context decides |
| **Make architectural choices** | Planning/design tasks need full context |

## Profile Configuration Guidelines

### Skill Profiles (Execution)
```json
{
  "preselected": [
    "enablement-knowledge-base",
    "eri-code-XXX",           // Reference implementation
    "skill-XXX"               // Workflow specification
  ]
}
```
- **No ADRs needed** - ERIs contain the embodied decisions
- Keeps context lean for execution focus
- Agent follows patterns, doesn't need to understand why

### Validator Profiles (Compliance Checking)
```json
{
  "preselected": [
    "enablement-knowledge-base",
    "adr-XXX",                // Constraints to validate against
    "adr-YYY",
    "skill-XXX",              // Expected patterns
    "skill-YYY"
  ]
}
```
- **ADRs required** - Need constraint boundaries for validation
- "Is this value within the allowed range per ADR-001?"
- "Does this implementation follow the resilience requirements from ADR-004?"

## The Principle

> **Skills follow patterns. Validators check constraints.**

- **Execution** = Apply the recipe (Skill + Module + ERI)
- **Validation** = Verify against the rules (ADR constraints)

## Example: Circuit Breaker

**For skill-001 (execution):**
```
Agent reads:
├── eri-code-008 (circuit breaker patterns)
├── skill-001 (workflow: analyze → add annotation → configure)
└── mod-001 (templates with {{placeholders}})

Agent generates code following the patterns.
```

**For validator-tier3 (compliance):**
```
Agent reads:
├── adr-004 (resilience constraints: "failureRateThreshold must be 40-60%")
├── skill-001 (expected structure)
└── Generated code

Agent checks: "Is failureRateThreshold=50 within ADR-004 bounds?" ✓
```

## Benefits of This Separation

1. **Leaner execution context** - Skills load only what's needed to generate code
2. **Explicit validation rules** - ADRs serve as the "source of truth" for compliance
3. **Clear responsibility** - Skills generate, validators verify
4. **Reduced redundancy** - Don't load ADRs twice (once in ERI, once directly)

## Summary

| Profile Type | Loads ADRs? | Rationale |
|--------------|-------------|-----------|
| Skill (execution) | No | ERI embodies ADR decisions |
| Validator (compliance) | Yes | Need constraints to check against |
| Planner (architecture) | Yes | Need full context for decisions |
