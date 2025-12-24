# Architecture Decision Records (ADRs)

**Version:** 1.0  
**Last Updated:** 2025-11-28

---

## Overview

ADRs document significant architectural decisions made for the organization. They provide context, rationale, and consequences for each decision.

---

## Naming Convention

```
adr-XXX-{topic}/
└── ADR.md
```

| Component | Description |
|-----------|-------------|
| `XXX` | Sequential number (001-999) |
| `{topic}` | Kebab-case topic name |

---

## Current ADRs

| ADR | Title | Status | Domain |
|-----|-------|--------|--------|
| [ADR-001](./adr-001-api-design-standards/) | API Design Standards | Accepted | API |
| [ADR-004](./adr-004-resilience-patterns/) | Resilience Patterns | Accepted | Resilience |
| [ADR-009](./adr-009-service-architecture-patterns/) | Service Architecture Patterns (Hexagonal Light) | Accepted | Architecture |
| [ADR-011](./adr-011-persistence-patterns/) | Persistence Patterns | Accepted | Persistence |

---

## ADR Structure

Each ADR follows this structure:

1. **Status** - Proposed, Accepted, Deprecated, Superseded
2. **Context** - The issue motivating this decision
3. **Decision** - The change we're making
4. **Consequences** - What becomes easier or harder
5. **Compliance** - MUST/SHOULD/MAY requirements

---

## Relationships

```
ADR (Decision)
  ↓ implements
ERI (Reference Implementation)
  ↓ encapsulates
Module (Reusable Template)
  ↓ orchestrates
Skill (Executable Generation)
```

### ADR to ERI Mapping

| ADR | ERIs |
|-----|------|
| ADR-001 | (API standards referenced in all ERIs) |
| ADR-004 | ERI-CODE-008, ERI-CODE-009, ERI-CODE-010, ERI-CODE-011 |
| ADR-009 | ERI-CODE-001 |
| ADR-011 | ERI-CODE-012, ERI-CODE-013, ERI-CODE-014, ERI-CODE-015 (planned) |

---

## Related

- `ERIs/` - Reference implementations of ADR patterns
- `capabilities/` - Feature definitions based on ADRs
- `model/standards/authoring/ADR.md` - How to create ADRs

---

**Last Updated:** 2025-11-28
