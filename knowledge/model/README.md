# Enablement 2.0 Model

> Meta-model definition: how the system works, standards, and authoring guides

## Purpose

This directory contains the **system definition** for Enablement 2.0:
- Master model document (v2.0)
- Agent context specifications (Consumer and Author roles)
- Asset standards and determinism rules
- Authoring guides
- Domain definitions with capabilities
- Executive and technical overviews

## Model v2.0 - Capability-Based Architecture

Model v2.0 introduces a **capability-based organization** where:
- **Domains** define areas of the SDLC (CODE, DESIGN, QA, GOVERNANCE)
- **Capabilities** represent what can be done within a domain
- **Features** are specific implementations of capabilities
- **Skills** implement one or more capabilities
- **Modules** provide templates that implement specific features

```
Domain (CODE)
  └── Capability (resilience)
        └── Feature (circuit-breaker)
              ├── Implemented by: skill-040-add-resilience
              └── Template in: mod-code-001-circuit-breaker
```

## Two Interaction Roles

The platform supports two distinct interaction roles:

| Role | Purpose | Prompt Document | Users |
|------|---------|-----------------|-------|
| **CONSUMER** | Use skills to produce SDLC outputs | `CONSUMER-PROMPT.md` | Developers, Solution Architects, Engineering Portal |
| **AUTHOR** | Create/evolve model and knowledge assets | `AUTHOR-PROMPT.md` | C4E Team (platform owners) |

### Consumer Role
Executes existing skills to generate code, designs, reports, etc. The agent follows discovery → skill selection → variant resolution → execution flows.

### Author Role
Creates new knowledge assets (ADRs, ERIs, Modules, Skills, Flows, etc.) following authoring guides. Must always consult `model/standards/authoring/` before creating any asset. Must ensure coherence between ERIs and Modules.

## Structure

```
model/
├── README.md                          # This file
│
├── ENABLEMENT-MODEL-v2.0.md           # ⭐ Master document (current)
├── CONSUMER-PROMPT.md                 # Consumer agent system prompt (v1.4)
├── AUTHOR-PROMPT.md                   # Author/C4E system prompt
├── ENABLEMENT-EXECUTIVE-BRIEF.md      # Executive summary
├── ENABLEMENT-TECHNICAL-GUIDE.md      # Technical architecture
│
├── standards/
│   ├── ASSET-STANDARDS-v1.4.md        # Structure and naming
│   ├── DETERMINISM-RULES.md           # ⭐ Code generation patterns
│   ├── authoring/                     # How to CREATE assets
│   │   ├── README.md                  # Authoring overview
│   │   ├── ADR.md                     # ADR authoring guide
│   │   ├── ERI.md                     # ERI authoring guide (v1.2)
│   │   ├── MODULE.md                  # Module authoring guide (v1.8)
│   │   ├── SKILL.md                   # Skill authoring guide (v3.0) ⭐
│   │   ├── FLOW.md                    # Flow authoring guide (v1.1)
│   │   ├── CAPABILITY.md              # Capability authoring guide
│   │   └── VALIDATOR.md               # Validator authoring guide
│   ├── validation/                    # Validation standards
│   └── traceability/                  # Traceability standards
│
└── domains/                           # Domain definitions
    ├── README.md                      # Domains overview
    ├── code/
    │   ├── DOMAIN.md                  # CODE domain specification
    │   ├── capabilities/              # Domain capabilities ⭐
    │   │   ├── resilience.md
    │   │   ├── persistence.md
    │   │   ├── api-exposure.md
    │   │   ├── api-integration.md
    │   │   └── ...
    │   └── module-structure.md
    ├── design/
    │   └── DOMAIN.md                  # DESIGN domain (planned)
    ├── qa/
    │   └── DOMAIN.md                  # QA domain (planned)
    └── governance/
        └── DOMAIN.md                  # GOVERNANCE domain (planned)
```

## Key Documents

| Document | Audience | Purpose |
|----------|----------|---------|
| **ENABLEMENT-MODEL-v2.0.md** | All | Complete system specification |
| **CONSUMER-PROMPT.md** | Consumer agents | System prompt for skill execution |
| **AUTHOR-PROMPT.md** | C4E Team | System prompt for authoring sessions |
| **DETERMINISM-RULES.md** | All | Mandatory patterns for code generation |
| **ENABLEMENT-EXECUTIVE-BRIEF.md** | Leadership | Business value, ROI |
| **ENABLEMENT-TECHNICAL-GUIDE.md** | Architects | Technical architecture |
| **standards/ASSET-STANDARDS-v1.4.md** | All | Naming, structure |
| **standards/authoring/*.md** | Authors | How to create assets |

## What's New in Model v2.0 (2025-01-15)

| Change | Description |
|--------|-------------|
| **Capability-Based Architecture** | Skills now implement domain capabilities, not just operations |
| **Skill Types** | CREATION, TRANSFORMATION, ANALYSIS, VALIDATION |
| **implements section** | Skills and Modules declare which capability.feature they implement |
| **extends support** | Skills can extend other skills for composition |
| **Multi-capability skills** | Single skill can implement multiple features |
| **Module alignment** | Modules declare `implements.capability` and `implements.feature` |

### Key Concepts in v2.0

**Capability Hierarchy:**
```
Domain → Capability → Feature → Skill/Module
```

**Skill Declaration:**
```yaml
implements:
  capability: resilience
  features: [circuit-breaker, retry, timeout, rate-limiter]
```

**Module Declaration:**
```yaml
implements:
  capability: resilience
  feature: circuit-breaker
```

### Previous Versions

| Version | Key Changes |
|---------|-------------|
| **v1.7** | Coherence and Determinism. Asset derivation chain formalized. |
| **v1.6** | Interpretive Discovery. Holistic Execution. Multi-domain support. |
| **v1.5** | Module system. Variant resolution. |

## Reading Order

### For Executives (15 min)
1. `ENABLEMENT-EXECUTIVE-BRIEF.md`

### For Architects (1-2 hours)
1. `ENABLEMENT-MODEL-v2.0.md`
2. `ENABLEMENT-TECHNICAL-GUIDE.md`
3. `standards/ASSET-STANDARDS-v1.4.md`
4. `standards/DETERMINISM-RULES.md`

### For Developers / Consumers (2-4 hours)
1. `ENABLEMENT-MODEL-v2.0.md` (sections 1-5, 8-11)
2. `CONSUMER-PROMPT.md` (understand how agents work)
3. `standards/ASSET-STANDARDS-v1.4.md`

### For C4E / Authors (2-4 hours)
1. `ENABLEMENT-MODEL-v2.0.md` (full document, especially §10 Coherence)
2. `AUTHOR-PROMPT.md` (load at start of every authoring session)
3. `standards/authoring/README.md`
4. `standards/authoring/{asset-type}.md` (as needed)
5. `standards/DETERMINISM-RULES.md`

## Related Directories

| Directory | Content |
|-----------|---------|
| `/knowledge/` | ADRs, ERIs (pure knowledge) |
| `/modules/` | Executable skills |
| `/modules/` | Reusable templates |
| `/runtime/` | Discovery, flows, validators |

## Versioning

Documents include version in filename:
- `ENABLEMENT-MODEL-v2.0.md` = Version 2.0 (current)
- `ASSET-STANDARDS-v1.4.md` = Version 1.4

---

**Last Updated:** 2025-01-15
**Version:** 6.0
