# Enablement 2.0: Technical Architecture Guide

**Version:** 1.2
**Date:** 2025-12-12
**Audience:** Software Architects, Tech Leads, Senior Engineers
**Classification:** Internal Technical

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Architecture](#2-solution-architecture)
3. [Knowledge Base Model](#3-knowledge-base-model)
4. [Asset Types Deep Dive](#4-asset-types-deep-dive)
5. [Validation System](#5-validation-system)
6. [Traceability System](#6-traceability-system)
7. [Platform Architecture](#7-platform-architecture)
8. [Roles and Processes](#8-roles-and-processes)
9. [Integration Points](#9-integration-points)
10. [Examples and Walkthroughs](#10-examples-and-walkthroughs)

---

## 1. Problem Statement

### 1.1 Current State Challenges

The software development lifecycle (SDLC) faces several critical challenges:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      IDENTIFIED PROBLEMS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ❌ FRAGMENTED KNOWLEDGE                                            │
│     • Architectural decisions in the minds of few people            │
│     • Outdated or non-existent documentation                        │
│     • Patterns reinvented in every project                          │
│                                                                      │
│  ❌ LOW STANDARDS ADOPTION                                          │
│     • 30-40% adoption of corporate frameworks                       │
│     • Each team implements their own way                            │
│     • Inconsistency between projects                                │
│                                                                      │
│  ❌ SLOW ONBOARDING                                                 │
│     • 3-6 months for full productivity                              │
│     • Tribal knowledge difficult to transfer                        │
│     • Steep learning curve                                          │
│                                                                      │
│  ❌ REACTIVE GOVERNANCE                                             │
│     • Manual validation prone to errors                             │
│     • Compliance verified late in the cycle                         │
│     • Limited traceability                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Impact Quantification

| Problem | Estimated Annual Impact |
|---------|-------------------------|
| Time lost on decisions already made | ~$1.5M |
| Non-standard code requiring refactoring | ~$2M |
| Defects from inconsistency | ~$1M |
| Extended onboarding | ~$500K |
| **Total** | **~$5M** |

### 1.3 Root Cause Analysis

```
                    ┌──────────────────────┐
                    │   ROOT CAUSE:        │
                    │   Non-Codified       │
                    │   Knowledge          │
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │ No single     │  │ No automatic  │  │ No way to     │
    │ source of     │  │ way to apply  │  │ verify        │
    │ truth         │  │ it            │  │ compliance    │
    └───────────────┘  └───────────────┘  └───────────────┘
```

---

## 2. Solution Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENABLEMENT 2.0 PLATFORM                              │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         KNOWLEDGE BASE                                  │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │ │
│  │  │  ADRs   │  │  ERIs   │  │ Modules │  │ Skills  │  │ Validators  │  │ │
│  │  │Strategic│  │Tactical │  │Template │  │Execution│  │  Quality    │  │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      AI ORCHESTRATION LAYER                             │ │
│  │                                                                          │ │
│  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │ │
│  │   │    Intent    │──▶│    Skill     │──▶│    Skill     │               │ │
│  │   │   Parser     │   │  Discovery   │   │  Execution   │               │ │
│  │   └──────────────┘   └──────────────┘   └──────────────┘               │ │
│  │                                                │                        │ │
│  │                                                ▼                        │ │
│  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │ │
│  │   │ Traceability │◀──│  Validation  │◀──│   Module     │               │ │
│  │   │  Generator   │   │ Orchestrator │   │  Composer    │               │ │
│  │   └──────────────┘   └──────────────┘   └──────────────┘               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           OUTPUTS                                       │ │
│  │                                                                          │ │
│  │   📁 Code Projects    📄 Documents    📊 Reports    ✅ Compliance       │ │
│  │   (.enablement/       (HLD, LLD)      (Quality,     (Audit trails)      │ │
│  │    manifest.json)                      Security)                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Principles

1. **Knowledge as Code (KaC)**
   - All architectural knowledge versioned in Git
   - Standardized and machine-readable structure
   - Evolves over time

2. **Meta-Model / Instances Separation**
   - `model/` = How to create things (specifications)
   - `knowledge/` = The things created (instances)

3. **Validation as a First-Class Citizen**
   - Every output is automatically validated
   - Validators are reusable assets
   - Verifiable and auditable compliance

4. **End-to-End Traceability**
   - Every decision documented
   - Every output has known origin
   - Guaranteed reproducibility

---

## 3. Knowledge Base Model

### 3.1 Directory Structure

```
knowledge/
│
├── model/                              # META-LEVEL (Specifications)
│   ├── ENABLEMENT-MODEL-v2.0.md       # Master document
│   └── standards/
│       ├── ASSET-STANDARDS-v1.4.md    # Asset structure
│       ├── authoring/                  # Creation guides
│       │   ├── ADR.md
│       │   ├── ERI.md
│       │   ├── MODULE.md
│       │   ├── SKILL.md               # ⚠️ CRITICAL
│       │   ├── VALIDATOR.md
│       │   ├── CAPABILITY.md
│       │   └── PATTERN.md
│       ├── validation/README.md        # Validation system
│       └── traceability/               # Traceability system
│           ├── BASE-MODEL.md
│           └── profiles/
│
├── ADRs/                               # INSTANCES - Decisions
│   └── adr-XXX-{topic}/
│
├── ERIs/                               # INSTANCES - Implementations
│   └── eri-{domain}-XXX-{pattern}-{framework}-{library}/
│
├── validators/                         # INSTANCES - Validators
│   ├── tier-1-universal/
│   ├── tier-2-technology/
│   └── tier-3-modules/
│
├── capabilities/                       # INSTANCES - Capabilities
│
├── patterns/                           # INSTANCES - Patterns
│
└── skills/                             # INSTANCES - Skills
    ├── modules/
    └── skill-{domain}-{NNN}-{type}-{target}/
```

### 3.2 Asset Relationships

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ASSET RELATIONSHIP MODEL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ADR ─────────────────────────────────────────────────────────────────     │
│    │ "What and Why" (Framework-agnostic)                                     │
│    │                                                                         │
│    │ implements (1:N)                                                        │
│    ▼                                                                         │
│   ERI ─────────────────────────────────────────────────────────────────     │
│    │ "How" for specific technology                                           │
│    │                                                                         │
│    │ abstracts_to (1:N)                                                      │
│    ▼                                                                         │
│   Module ──────────────────────────────────────────────────────────────     │
│    │ Reusable templates + Tier 3 validation                                  │
│    │                                                                         │
│    │ used_by (N:N)                                                           │
│    ▼                                                                         │
│   Skill ───────────────────────────────────────────────────────────────     │
│    │ Executable capability                                                   │
│    │                                                                         │
│    │ orchestrates (N:N)                                                      │
│    ▼                                                                         │
│   Validator ───────────────────────────────────────────────────────────     │
│      Quality assurance                                                       │
│                                                                              │
│   ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│   Capability ◀────────────── groups ──────────────▶ Feature                 │
│                                  │                                           │
│                                  ▼                                           │
│                              Component                                       │
│                                  │                                           │
│                                  ▼                                           │
│                               Module                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Naming Conventions

| Asset | Pattern | Example |
|-------|---------|---------|
| ADR | `adr-XXX-{topic}` | `adr-004-resilience-patterns` |
| ERI | `eri-{domain}-XXX-{pattern}-{framework}-{library}` | `eri-code-001-hexagonal-light-java-spring` |
| Module | `mod-XXX-{pattern}-{framework}-{library}` | `mod-code-001-circuit-breaker-java-resilience4j` |
| Skill | `skill-{domain}-{NNN}-{type}-{target}-{framework}-{library}` | `skill-code-020-generate-microservice-java-spring` |
| Validator | `val-{tier}-{category}-{name}` | `val-tier2-code-projects-java-spring` |

---

## 4. Asset Types Deep Dive

### 4.1 ADR (Architectural Decision Record)

**Purpose:** Document strategic framework-agnostic decisions.

```markdown
# ADR-XXX: {Title}

## Status
{Draft|Proposed|Accepted|Deprecated|Superseded}

## Context
[The problem and forces at play]

## Decision
[The decision made - prescriptive]

## Rationale
[Why this decision was made]

## Consequences
[Positive, negative, neutral]

## Implementation
[How it's implemented - references to ERIs]
```

**Ownership:** Software Architect
**Review:** Architecture Review Board

### 4.2 ERI (Enterprise Reference Implementation)

**Purpose:** Complete and compilable implementation of an ADR for a specific technology.

```markdown
# ERI-{DOMAIN}-XXX: {Title}

## Technology Stack
| Component | Technology | Version |
|-----------|------------|---------|

## Project Structure
[Directory layout]

## Code Reference
[Complete, compilable code examples]

## Configuration
[Complete configuration files]

## Compliance Checklist
[What implementations MUST satisfy]

## Annex: Implementation Constraints (MANDATORY)
[Machine-readable YAML with eri_constraints]
```

**Key Innovation:** Every ERI MUST include a machine-readable annex (`eri_constraints`) that defines:
- `structural_constraints` - Code organization rules
- `configuration_constraints` - Configuration requirements
- `dependency_constraints` - Required/optional dependencies
- `testing_constraints` - Testing requirements

This annex serves as the **source of truth** for MODULE validators and enables AI-powered automation.

**Ownership:** Tech Lead / Senior Engineer
**Review:** Architecture Team

### 4.3 Module

**Purpose:** Parameterized templates derived from ERIs + Tier 3 validation.

```
modules/mod-XXX-{pattern}/
├── MODULE.md           # Complete documentation
├── OVERVIEW.md         # Quick reference
├── templates/          # Handlebars/FreeMarker templates
│   └── *.hbs
└── validation/         # Tier 3 validation
    └── *-check.sh
```

**Key Innovation:** Each module includes its own validation that verifies ERI constraints are met.

### 4.4 Skill

**Purpose:** Executable capability that orchestrates modules and validators.

```
skills/skill-{domain}-{NNN}-{type}-{target}/
├── SKILL.md            # Complete specification
├── OVERVIEW.md         # Quick reference
├── README.md           # External documentation
├── prompts/            # ⚠️ CRITICAL - Prompt engineering
│   ├── system.md       # Role, context, constraints
│   ├── user.md         # Request template
│   └── examples/       # Few-shot examples
└── validation/
    └── validate.sh     # Orchestrates Tier 1, 2, 3
```

**Prompt Derivation:** Prompts are derived from the knowledge base:

```
ADR Constraints    ──▶  prompts/system.md (MUST/MUST NOT)
ERI Patterns       ──▶  prompts/system.md (Context)
Module Templates   ──▶  prompts/system.md (Available tools)
Examples           ──▶  prompts/examples/ (Few-shot)
```

### 4.5 Validator

**Purpose:** Reusable validation components organized by artifact type.

```
validators/
├── tier-1-universal/           # ALWAYS executed
│   ├── project-structure/
│   └── naming-conventions/
├── tier-2-technology/         # CONDITIONAL by type
│   ├── code-projects/
│   │   └── java-spring/
│   ├── deployments/
│   │   └── docker/
│   ├── documents/
│   └── reports/
└── tier-3-modules/           # Embedded in modules
```

**Cross-Domain Usage:** Validators are organized by *what they validate*, not by *who uses them*. This allows the same `java-spring` validator to be used by CODE and QA skills.

---

## 5. Validation System

### 5.1 Domain-Based Validation

The validation strategy **differs based on skill domain**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VALIDATION ORCHESTRATION BY DOMAIN                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CODE DOMAIN                          │  DESIGN / QA / GOV DOMAINS          │
│  ─────────────────                    │  ──────────────────────────          │
│  validate.sh ORCHESTRATES:            │  validate.sh INVOKES:               │
│                                        │                                      │
│  ✅ Tier-1 Universal (traceability)   │  ✅ Tier-1 Universal (traceability) │
│  ✅ Tier-1 Code (structure, naming)   │  ✅ Embedded (skill-specific)        │
│  ✅ Tier-2 (tech stack)               │                                      │
│  ✅ Tier-3 (modules)                  │  ❌ Tier-1 Code (not applicable)    │
│                                        │  ❌ Tier-2 (not applicable)         │
│                                        │  ❌ Tier-3 (not applicable)         │
│                                        │                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Rationale:** Code artifacts have predictable, standardized structures that benefit from shared validators. Documents and reports have skill-specific formats requiring embedded validation.

### 5.2 Tier Definitions

| Tier | Location | Applies To | Execution |
|------|----------|------------|-----------|
| **1 Universal** | `tier-1-universal/traceability/` | All domains | ALWAYS |
| **1 Code** | `tier-1-universal/code-projects/` | CODE only | ALWAYS for CODE |
| **2 Artifacts** | `tier-2-technology/` | CODE only | Conditional |
| **3 Modules** | `modules/{mod}/validation/` | CODE only | Conditional |
| **Embedded** | `skills/{skill}/validation/` | DESIGN/QA/GOV | ALWAYS for non-CODE |
| **4 Runtime** | CI/CD | All | Future |

### 5.3 Validation Script Standard

```bash
#!/bin/bash
# {name}-check.sh

TARGET_DIR="${1:-.}"
ERRORS=0

# Output functions
pass() { echo -e "✅ PASS: $1"; }
fail() { echo -e "❌ FAIL: $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "⚠️  WARN: $1"; }
skip() { echo -e "⏭️  SKIP: $1"; }

# Check implementation
if [ condition ]; then
    pass "description"
else
    fail "description"
fi

exit $ERRORS
```

---

## 6. Traceability System

### 6.1 BASE-MODEL

Common fields required by ALL skills:

```json
{
  "generation": {
    "id": "uuid",
    "timestamp": "ISO-8601",
    "duration_seconds": 45
  },
  "skill": {
    "id": "skill-code-020-...",
    "version": "1.0.0",
    "domain": "code"
  },
  "orchestrator": {
    "model": "claude-sonnet-4",
    "knowledge_base_version": "5.0"
  },
  "request": {
    "raw": "original user request",
    "parsed_intent": "structured interpretation"
  },
  "decisions": [
    {
      "decision": "what was decided",
      "reason": "why",
      "adr_reference": "adr-XXX"
    }
  ],
  "modules_used": ["mod-code-001", "mod-code-015"],
  "validators_executed": [
    {
      "validator": "val-tier1-...",
      "result": "PASS",
      "checks": 5
    }
  ],
  "status": {
    "overall": "SUCCESS|PARTIAL|FAILED",
    "errors": 0,
    "warnings": 2
  }
}
```

### 6.2 Output-Type Profiles

| Profile | Used By | Additional Fields |
|---------|---------|-------------------|
| `code-project` | skill-code-*-generate-* | artifacts_generated, dependencies_added |
| `code-transformation` | skill-code-*-add/remove-* | artifacts_modified, rollback_info |
| `document` | skill-design-*, skill-gov-* | document_type, diagrams_included |
| `report` | skill-qa-* | findings[], scores, recommendations |

---

## 7. Platform Architecture

### 7.1 Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENABLEMENT 2.0 PLATFORM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        USER INTERFACES                               │   │
│   │                                                                       │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│   │   │  CLI     │  │  IDE     │  │  Portal  │  │  Engineering     │   │   │
│   │   │(AI-chat) │  │Extension │  │  (Web)   │  │  Portal Plugin   │   │   │
│   │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │   │
│   │        │             │             │                  │             │   │
│   └────────┼─────────────┼─────────────┼──────────────────┼─────────────┘   │
│            │             │             │                  │                  │
│            └─────────────┴─────────────┴──────────────────┘                  │
│                                    │                                         │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      ORCHESTRATION LAYER                             │   │
│   │                                                                       │   │
│   │   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐      │   │
│   │   │  Intent Parser │   │ Skill Discovery│   │ Skill Executor │      │   │
│   │   │                │──▶│                │──▶│                │      │   │
│   │   │  NLP + Context │   │ Capability     │   │ Multi-step     │      │   │
│   │   │  Understanding │   │ Matching       │   │ Orchestration  │      │   │
│   │   └────────────────┘   └────────────────┘   └────────────────┘      │   │
│   │                                                      │               │   │
│   └──────────────────────────────────────────────────────┼───────────────┘   │
│                                                          │                   │
│                                                          ▼                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      KNOWLEDGE BASE                                  │   │
│   │                                                                       │   │
│   │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│   │   │  ADRs   │  │  ERIs   │  │ Modules │  │ Skills  │  │Validators│  │   │
│   │   └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  │   │
│   │                                                                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      INTEGRATION LAYER                               │   │
│   │                                                                       │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│   │   │   Git    │  │  CI/CD   │  │  Artifact│  │  Engineering     │   │   │
│   │   │          │  │(Jenkins) │  │  Repo    │  │  Portal APIs     │   │   │
│   │   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│   │                                                                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Orchestration Flow

```
User                       Platform                      Knowledge Base
   │                           │                              │
   │  "Create microservice     │                              │
   │   for customer mgmt"      │                              │
   │ ─────────────────────────▶│                              │
   │                           │                              │
   │                           │  1. Parse intent             │
   │                           │  ─────────────────────────▶  │
   │                           │                              │
   │                           │  2. Match capabilities       │
   │                           │  ◀─────────────────────────  │
   │                           │     [skill-code-020-...]     │
   │                           │                              │
   │                           │  3. Load skill + dependencies│
   │                           │  ─────────────────────────▶  │
   │                           │                              │
   │                           │  4. Get ADR constraints      │
   │                           │  ◀─────────────────────────  │
   │                           │     [adr-004, adr-009]       │
   │                           │                              │
   │                           │  5. Get modules              │
   │                           │  ◀─────────────────────────  │
   │                           │     [mod-code-001, mod-code-015]       │
   │                           │                              │
   │                           │  6. Execute generation       │
   │                           │  (with AI + templates)       │
   │                           │                              │
   │                           │  7. Run validators           │
   │                           │  ─────────────────────────▶  │
   │                           │  ◀─────────────────────────  │
   │                           │     [✅ 47/47 checks]        │
   │                           │                              │
   │                           │  8. Generate traceability    │
   │                           │                              │
   │  Output + Manifest        │                              │
   │ ◀─────────────────────────│                              │
   │                           │                              │
```

> **Implementation Details:** The orchestration layer is formally specified in:
> - `knowledge/orchestration/discovery-rules.md` - How prompts map to skills
> - `knowledge/orchestration/execution-framework.md` - Generic execution flow
> - `knowledge/orchestration/prompt-template.md` - Standard input format
> - Each Skill's `EXECUTION-FLOW.md` - Skill-specific deterministic steps

### 7.3 MCP Integration (Model Context Protocol)

For integration with Claude and other LLMs:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MCP SERVER: Enablement                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TOOLS:                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  list_capabilities()     → List available capabilities        │   │
│  │  get_skill(id)           → Get skill spec                     │   │
│  │  execute_skill(id, args) → Execute skill                      │   │
│  │  validate_output(path)   → Validate an output                 │   │
│  │  get_adr(id)             → Get ADR                            │   │
│  │  get_eri(id)             → Get ERI                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  RESOURCES:                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  knowledge://adrs/{id}                                        │   │
│  │  knowledge://eris/{id}                                        │   │
│  │  knowledge://skills/{id}                                      │   │
│  │  knowledge://capabilities/{id}                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Roles and Processes

### 8.1 Role Matrix

| Role | Creates | Consumes | Reviews |
|------|---------|----------|---------|
| **Software Architect** | ADRs, Patterns | Skills (DESIGN) | ERIs, Skills |
| **Tech Lead** | ERIs, Modules | Skills (CODE) | Modules |
| **Senior Engineer** | Modules, Skills | Skills (CODE) | Skills |
| **Developer** | - | Skills (CODE) | - |
| **Solution Architect** | - | Skills (DESIGN) | ADRs |
| **QA Engineer** | - | Skills (QA) | Reports |
| **C4E Team** | All | All | All |

### 8.2 Process: Creating a New ADR

```
┌─────────────────────────────────────────────────────────────────────┐
│              PROCESS: New Architectural Decision                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. IDENTIFICATION                                                   │
│     Software Architect identifies need for standardization           │
│     ↓                                                                │
│  2. DRAFT                                                            │
│     Architect creates ADR draft using:                               │
│     - CLI/Chat with AI (dialogue-based)                              │
│     - Template from authoring/ADR.md                                 │
│     ↓                                                                │
│  3. REVIEW                                                           │
│     Architecture Review Board reviews                                │
│     ↓                                                                │
│  4. ACCEPTANCE                                                       │
│     ADR marked as "Accepted"                                         │
│     ↓                                                                │
│  5. IMPLEMENTATION                                                   │
│     Tech Lead creates ERIs for each technology                       │
│     ↓                                                                │
│  6. PROPAGATION                                                      │
│     Skills updated to use new constraints                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Process: Developer Using Platform

```
┌─────────────────────────────────────────────────────────────────────┐
│              PROCESS: Developer Creates Microservice                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. REQUEST                                                          │
│     Developer: "I need a customer microservice with circuit          │
│                breaker and REST API"                                 │
│     ↓                                                                │
│  2. INTENT PARSING                                                   │
│     Platform interprets:                                             │
│     - Type: generate-microservice                                    │
│     - Features: [circuit-breaker, rest-api]                          │
│     - Domain: customer                                               │
│     ↓                                                                │
│  3. SKILL SELECTION                                                  │
│     skill-code-020-generate-microservice-java-spring                 │
│     + mod-code-001-circuit-breaker-java-resilience4j                      │
│     ↓                                                                │
│  4. EXECUTION                                                        │
│     - Load ADR constraints (adr-004, adr-009)                        │
│     - Generate code using templates                                  │
│     - Apply AI for domain-specific logic                             │
│     ↓                                                                │
│  5. VALIDATION                                                       │
│     - Tier 1: ✅ Structure OK                                        │
│     - Tier 2: ✅ Compiles, Tests pass                                │
│     - Tier 3: ✅ Circuit breaker correct                             │
│     ↓                                                                │
│  6. OUTPUT                                                           │
│     - customer-service/ (complete project)                           │
│     - .enablement/manifest.json (traceability)                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Integration Points

### 9.1 Git Integration

```yaml
# .github/workflows/enablement-validation.yml
name: Enablement Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Enablement Validators
        run: |
          .enablement/validation/validate-all.sh
      - name: Upload Validation Report
        uses: actions/upload-artifact@v3
        with:
          name: validation-report
          path: .enablement/validation/report.md
```

### 9.2 Engineering Portal Integration

```
Engineering Portal
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ENABLEMENT PORTAL PLUGIN                           │
│                                                                      │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│   │  Capability  │   │   Project    │   │  Governance  │           │
│   │  Catalog     │   │  Generator   │   │  Dashboard   │           │
│   └──────────────┘   └──────────────┘   └──────────────┘           │
│                                                                      │
│   - Browse available skills                                          │
│   - Generate projects via UI                                         │
│   - View compliance metrics                                          │
│   - Track adoption KPIs                                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Examples and Walkthroughs

### 10.1 Example: Generate Customer Microservice

**Input:**
```json
{
  "serviceName": "CustomerService",
  "packageName": "com.company.customer",
  "features": ["circuit-breaker", "rest-api"],
  "domain": {
    "entities": ["Customer"],
    "operations": ["create", "read", "update", "delete"]
  }
}
```

**Skill Executed:** `skill-code-020-generate-microservice-java-spring`

**Output Structure:**
```
customer-service/
├── .enablement/
│   ├── manifest.json         # Traceability
│   └── validation/
│       └── report.md         # Validation results
├── src/
│   ├── main/
│   │   ├── java/com/company/customer/
│   │   │   ├── domain/       # Pure domain logic
│   │   │   ├── application/  # Use cases
│   │   │   └── infrastructure/ # Adapters
│   │   └── resources/
│   │       └── application.yml
│   └── test/
├── pom.xml
├── Dockerfile
└── README.md
```

**Validation Results:**
```
═══════════════════════════════════════════════════
TIER 1: GENERIC
═══════════════════════════════════════════════════
✅ PASS: src/main/java exists
✅ PASS: src/test/java exists
✅ PASS: Naming conventions correct

═══════════════════════════════════════════════════
TIER 2: ARTIFACTS
═══════════════════════════════════════════════════
✅ PASS: Project compiles
✅ PASS: Tests pass (5/5)
✅ PASS: Actuator configured
✅ PASS: application.yml valid
✅ PASS: Dockerfile valid

═══════════════════════════════════════════════════
TIER 3: MODULE
═══════════════════════════════════════════════════
✅ PASS: Hexagonal structure correct
✅ PASS: Circuit breaker configured
✅ PASS: Fallback methods present

═══════════════════════════════════════════════════
TOTAL: 11/11 checks passed
═══════════════════════════════════════════════════
```

---

## Appendix: Document References

| Document | Purpose | Location |
|----------|---------|----------|
| ENABLEMENT-MODEL-v2.0.md | Master model | `model/` |
| ASSET-STANDARDS-v1.4.md | Asset structure | `model/standards/` |
| authoring/SKILL.md | How to create skills | `model/standards/authoring/` |
| validators/README.md | Validation system | `knowledge/validators/` |
| traceability/BASE-MODEL.md | Traceability fields | `model/standards/traceability/` |

---

*Enablement 2.0 Technical Architecture Guide v1.0*
